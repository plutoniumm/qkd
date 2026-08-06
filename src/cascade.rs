use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::std::check_err;
use crate::threefry::Stream;

// The counted Cascade simulation behind `qkd.reconcile.run_cascade`. Brassard & Salvail,
// EUROCRYPT '93, in the parameter sets of Martinez-Mateo, Pacher, Peev, Ciurana & Martin,
// "Demystifying the information reconciliation protocol Cascade", QIC 15, 453 (2015),
// arXiv:1407.3257, Table 1.
//
// Alice's string is all-zero and Bob's is the error pattern. Returns a DISCLOSED-PARITY COUNT and
// a frame error count: not an `eps_cor`, not a bound. Counting is Martinez-Mateo Sec. III.A.
//
// Ties in the heap key (length, block id) are EQUAL keys, so `BinaryHeap<Reverse<_>>` pops the
// sequence Python's `heapq` does, value for value.
//
// RNG: `crate::threefry`, NOT Python's Mersenne Twister: only sample statistics compare across
// the port. A frame is a pure function of (seed, frame index).

/// Separate keys: the pass count moves no error bit.
const ERRORS: u32 = 40;

const PERMUTE: u32 = 41;

/// Bit indices and arena offsets cross as `u32`.
const FRAME_MAX: usize = 1 << 22;

/// Ceiling on frame x passes, one `u32` per bit per pass in the arena and in `slots`.
const CELLS_MAX: usize = 1 << 24;

/// Empty chain cell, and the slot of a pass not yet run.
const NONE: u32 = u32::MAX;

/// A window into the arena and the parity Alice and Bob disagree by.
#[derive(Clone, Copy)]
struct Block {
    off: u32,
    len: u32,
    bit: u8,
}

/// `slots` (initial blocks) and `head`/`link` (chained subblocks) index bit to block; flips commute.
struct Frame {
    err: Vec<u8>,
    arena: Vec<u32>,
    held: Vec<Block>,
    slots: Vec<u32>,
    head: Vec<u32>,
    link: Vec<(u32, u32)>,
    odd: BinaryHeap<Reverse<(u32, u32)>>,
    passes: usize,
    leak: u64,
    reuse: bool,
}

impl Frame {
    fn new(err: Vec<u8>, passes: usize, reuse: bool) -> Self {
        let frame = err.len();

        Self {
            err,
            arena: Vec::with_capacity(frame * passes),
            held: Vec::new(),
            slots: vec![NONE; frame * passes],
            head: vec![NONE; frame],
            link: Vec::new(),
            odd: BinaryHeap::new(),
            passes,
            leak: 0,
            reuse,
        }
    }

    fn parity(&self, off: u32, len: u32) -> u8 {
        let mut bit = 0u8;
        for &i in &self.arena[off as usize..(off + len) as usize] {
            bit ^= self.err[i as usize];
        }

        bit
    }

    fn hold(&mut self, off: u32, len: u32, bit: u8) -> u32 {
        let bid = self.held.len() as u32;
        self.held.push(Block { off, len, bit });
        if bit != 0 {
            self.odd.push(Reverse((len, bid)));
        }

        bid
    }

    fn announce(&mut self, off: u32, len: u32, bit: u8, step: usize) {
        let bid = self.hold(off, len, bit);
        for k in off..off + len {
            let i = self.arena[k as usize] as usize;
            self.slots[i * self.passes + step] = bid;
        }
    }

    /// Chained: how many a bit collects depends on how often its blocks turned out odd.
    fn subblock(&mut self, off: u32, len: u32, bit: u8) {
        let bid = self.hold(off, len, bit);
        for k in off..off + len {
            let i = self.arena[k as usize] as usize;
            let cell = self.link.len() as u32;
            self.link.push((bid, self.head[i]));
            self.head[i] = cell;
        }
    }

    fn flip(&mut self, bid: u32) {
        self.held[bid as usize].bit ^= 1;
        let block = self.held[bid as usize];
        if block.bit != 0 {
            self.odd.push(Reverse((block.len, bid)));
        }
    }

    /// One disclosed parity per step. `seed.bit` is the WHOLE block's parity, the right sibling's
    /// complement at every depth.
    fn correct(&mut self, bid: u32) {
        let seed = self.held[bid as usize];
        let (mut off, mut len) = (seed.off, seed.len);
        while len > 1 {
            let cut = len / 2;
            let bit = self.parity(off, cut);
            self.leak += 1;
            if self.reuse {
                self.subblock(off, cut, bit);
                self.subblock(off + cut, len - cut, bit ^ seed.bit);
            }

            if bit != 0 {
                len = cut;
            } else {
                off += cut;
                len -= cut;
            }
        }

        let spot = self.arena[off as usize] as usize;
        self.err[spot] ^= 1;
        for p in 0..self.passes {
            let bid = self.slots[spot * self.passes + p];
            if bid != NONE {
                self.flip(bid);
            }
        }

        let mut cell = self.head[spot];
        while cell != NONE {
            let (bid, next) = self.link[cell as usize];
            self.flip(bid);
            cell = next;
        }
    }

    fn drain(&mut self) {
        while let Some(Reverse((len, bid))) = self.odd.pop() {
            let block = self.held[bid as usize];
            // `len == block.len` is a tautology (lengths never change after `hold`) kept with
            // the port; the live test is `bit`.
            if block.bit != 0 && len == block.len {
                self.correct(bid);
            }
        }
    }

    /// The last block of every pass after the first costs nothing.
    fn pass(&mut self, order: &[u32], size: usize, step: usize) {
        let base = self.arena.len() as u32;
        self.arena.extend_from_slice(order);
        let mut count = 0u64;
        let mut i = 0usize;
        while i < order.len() {
            let len = size.min(order.len() - i);
            let off = base + i as u32;
            let bit = self.parity(off, len as u32);
            self.announce(off, len as u32, bit, step);
            count += 1;
            i += len;
        }

        self.leak += count.saturating_sub(u64::from(step > 0));
        self.drain();
    }

    fn residue(&self) -> u64 {
        self.err.iter().map(|&e| u64::from(e)).sum()
    }
}

/// Fisher-Yates, one Threefry block per four swaps, `w * (i + 1) >> 32` picking the partner.
/// One word per step: a rejection sampler's variable count would break the pinned index map.
fn shuffle(order: &mut [u32], mix: &Stream, base: u64) {
    let mut w = [0u32; 4];
    for (k, i) in (1..order.len()).rev().enumerate() {
        if k % 4 == 0 {
            w = mix.block(base + (k / 4) as u64, 0);
        }

        let j = (u64::from(w[k % 4]) * (i as u64 + 1)) >> 32;
        order.swap(i, j as usize);
    }
}

/// `(disclosed parities, residual errors)` of frame `no`, a pure function of `(seed, no)`.
fn one_frame(qber: f64, frame: usize, sizes: &[usize], reuse: bool, seed: u64, no: u64) -> (u64, u64) {
    let per = (frame as u64).div_ceil(4);
    let bits = Stream::new(seed, ERRORS);
    let mut err = vec![0u8; frame];
    for (c, cell) in err.chunks_mut(4).enumerate() {
        let u = bits.uniforms(no * per + c as u64);
        for (j, e) in cell.iter_mut().enumerate() {
            *e = u8::from(u[j] < qber);
        }
    }

    let mix = Stream::new(seed, PERMUTE);
    let mut order: Vec<u32> = (0..frame as u32).collect();
    let mut f = Frame::new(err, sizes.len(), reuse);
    for (step, &size) in sizes.iter().enumerate() {
        if step > 0 {
            shuffle(&mut order, &mix, (no * sizes.len() as u64 + step as u64) * per);
        }

        f.pass(&order, size, step);
    }

    (f.leak, f.residue())
}

fn check_shape(frame: usize, sizes: &[usize]) -> PyResult<()> {
    if frame == 0 || frame > FRAME_MAX {
        return Err(PyValueError::new_err(format!(
            "frame must be in 1..={FRAME_MAX} bits, got {frame}"
        )));
    }

    if sizes.is_empty() {
        return Err(PyValueError::new_err(
            "sizes must hold at least one pass, one block size per pass",
        ));
    }

    if frame * sizes.len() > CELLS_MAX {
        return Err(PyValueError::new_err(format!(
            "{frame} bits over {} passes is {} block-index cells, above the \
             {CELLS_MAX} this holds in one arena",
            sizes.len(),
            frame * sizes.len()
        )));
    }

    if sizes.iter().any(|&k| k == 0) {
        return Err(PyValueError::new_err(
            "block sizes must be >= 1: the k_i of Martinez-Mateo Table 1",
        ));
    }

    Ok(())
}

/// `(mean disclosed parities per frame, frames left with a residual error)` over `frames` frames
/// of `frame` bits at `qber`; `sizes` one block size per pass, `reuse` whether the bisection's
/// subblocks are announced too. A SAMPLE MEAN, carrying sampling error; the second return is a
/// count, not a probability.
#[pyfunction]
#[pyo3(signature = (qber, frame, sizes, reuse, seed, frames))]
pub(crate) fn cascade_run(
    py: Python<'_>,
    qber: f64,
    frame: usize,
    sizes: Vec<usize>,
    reuse: bool,
    seed: u64,
    frames: usize,
) -> PyResult<(f64, u64)> {
    check_err("qber", qber)?;
    check_shape(frame, &sizes)?;
    if frames == 0 {
        return Err(PyValueError::new_err("frames must be >= 1"));
    }

    let (leak, failed) = py.detach(|| {
        (0..frames as u64)
            .into_par_iter()
            .map(|no| {
                let (leak, residue) = one_frame(qber, frame, &sizes, reuse, seed, no);

                (leak, u64::from(residue > 0))
            })
            .reduce(|| (0u64, 0u64), |a, b| (a.0 + b.0, a.1 + b.1))
    });

    Ok((leak as f64 / frames as f64, failed))
}

/// `(disclosed parities, residual errors)` of one frame on SUPPLIED draws: `err` Bob's error
/// pattern, `order` one permutation of `0..len(err)` per pass, end to end. The arbiter against
/// test/reconcile.py's reference, not an API.
#[pyfunction]
#[pyo3(signature = (err, order, sizes, reuse))]
pub(crate) fn cascade_replay(err: Vec<u8>, order: Vec<u32>, sizes: Vec<usize>, reuse: bool) -> PyResult<(u64, u64)> {
    let frame = err.len();
    check_shape(frame, &sizes)?;
    if err.iter().any(|&b| b > 1) {
        return Err(PyValueError::new_err("err is a bit pattern: every entry is 0 or 1"));
    }

    if order.len() != frame * sizes.len() {
        return Err(PyValueError::new_err(format!(
            "order carries {} entries; one permutation of {frame} per pass over \
             {} passes is {}",
            order.len(),
            sizes.len(),
            frame * sizes.len()
        )));
    }

    let mut seen = vec![false; frame];
    for (step, chunk) in order.chunks(frame).enumerate() {
        for &v in chunk {
            let i = v as usize;
            if i >= frame || seen[i] {
                return Err(PyValueError::new_err(format!(
                    "pass {step} of order is not a permutation of 0..{frame}"
                )));
            }

            seen[i] = true;
        }

        for &v in chunk {
            seen[v as usize] = false;
        }
    }

    let mut f = Frame::new(err, sizes.len(), reuse);
    for (step, &size) in sizes.iter().enumerate() {
        f.pass(&order[step * frame..(step + 1) * frame], size, step);
    }

    Ok((f.leak, f.residue()))
}
