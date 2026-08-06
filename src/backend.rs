use std::fmt;
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::std::{check_nonneg, check_pos, check_unit, ramp, wrap};
use crate::threefry::{Stream, ALICE, CHANNEL, DETECT, WALK_LO, WALK_TX};

/// Dispatch asks for a precision, not a device. WGSL has no `f64`, so the GPU
/// path is permanently f32; covariance, symplectic maps and Holevo stay f64/CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    F32,
    F64,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        })
    }
}

impl Precision {
    /// Inverse of `Display`.
    pub fn parse(name: &str) -> PyResult<Self> {
        match name {
            "f32" => Ok(Precision::F32),
            "f64" => Ok(Precision::F64),
            other => Err(PyValueError::new_err(format!(
                "unknown precision {other:?}, expected \"f32\" or \"f64\""
            ))),
        }
    }
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    fn precision(&self) -> Precision;

    fn device(&self) -> String;

    fn supports(&self, want: Precision) -> bool {
        match (self.precision(), want) {
            (Precision::F64, _) => true,
            (Precision::F32, Precision::F32) => true,
            (Precision::F32, Precision::F64) => false,
        }
    }
}

/// f64 rayon backend: the arbiter every GPU kernel is checked against.
pub struct Cpu {
    threads: usize,
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            threads: rayon::current_num_threads(),
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn precision(&self) -> Precision {
        Precision::F64
    }

    fn device(&self) -> String {
        format!("rayon, {} threads", self.threads)
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{Backend, Precision};

    /// wgpu-backed f32 device, resolved at runtime (Metal / Vulkan / DX12).
    pub struct Gpu {
        adapter: String,
        backend: String,
    }

    impl Gpu {
        /// `None` when no adapter answers (manylinux): a fallback, not an error.
        pub fn probe() -> Option<Self> {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .ok()?;
            let info = adapter.get_info();

            Some(Self {
                adapter: info.name,
                backend: format!("{:?}", info.backend),
            })
        }
    }

    impl Backend for Gpu {
        fn name(&self) -> &'static str {
            "gpu"
        }

        fn precision(&self) -> Precision {
            Precision::F32
        }

        fn device(&self) -> String {
            format!("{} ({})", self.adapter, self.backend)
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::Gpu;

/// GPU when compiled in *and* an adapter answers, else CPU. Never fails.
pub fn resolve() -> &'static dyn Backend {
    // Probed once: enumerating adapters costs ~7 us per call.
    static RESOLVED: OnceLock<Box<dyn Backend>> = OnceLock::new();

    RESOLVED
        .get_or_init(|| {
            #[cfg(feature = "gpu")]
            if let Some(g) = Gpu::probe() {
                return Box::new(g) as Box<dyn Backend>;
            }

            Box::new(Cpu::new())
        })
        .as_ref()
}

// Oracle-phase branch of `pipeline::run_symbols`; the f32 kernels mirror `cpu_raw`.

/// Symbols per DSP block = per GPU thread, so the within-block scan is free.
const WALK_SIZE: u32 = 256;

/// Blocks per rayon task, `pipeline::CHUNK`'s 4096 symbols. Fixed: fold order ignores threads.
const TASK: usize = 16;

/// Sufficient statistics of one symbol run, pooled over both quadratures.
#[derive(Clone, Copy)]
struct Raw {
    s_aa: f64,
    s_ai: f64,
    s_yy: f64,
    sig: f64,
    noise: f64,
}

impl Raw {
    fn zero() -> Self {
        Self {
            s_aa: 0.0,
            s_ai: 0.0,
            s_yy: 0.0,
            sig: 0.0,
            noise: 0.0,
        }
    }

    fn merge(a: Self, b: Self) -> Self {
        Self {
            s_aa: a.s_aa + b.s_aa,
            s_ai: a.s_ai + b.s_ai,
            s_yy: a.s_yy + b.s_yy,
            sig: a.sig + b.sig,
            noise: a.noise + b.noise,
        }
    }
}

/// Everything the per-symbol map needs, in shot-noise units, precomputed once.
struct Sym {
    seed: u64,
    bl: usize,
    nb: usize,
    ampl: f64,
    root_t: f64,
    ch_sd: f64,
    gain: f64,
    det_sd: f64,
    pilot_amp: f64,
    pilot_sd: f64,
    frac: f64,
    sd_tx: f64,
    sd_lo: f64,
    eta: f64,
    vel: f64,
}

impl Sym {
    /// `run_symbols`'s parameters less the CFO, which cancels in the oracle branch.
    fn new(
        n: u64,
        seed: u64,
        va: f64,
        t: f64,
        xi: f64,
        eta: f64,
        vel: f64,
        lw_alice: f64,
        lw_lo: f64,
        rate: f64,
        pilot_db: f64,
        pilot_frac: f64,
    ) -> PyResult<Self> {
        check_nonneg("va", va)?;
        check_unit("t", t)?;
        check_nonneg("xi", xi)?;
        check_unit("eta", eta)?;
        check_nonneg("vel", vel)?;
        check_nonneg("lw_alice", lw_alice)?;
        check_nonneg("lw_lo", lw_lo)?;
        check_pos("symbol_rate", rate)?;
        if !pilot_db.is_finite() || !pilot_frac.is_finite() {
            return Err(PyValueError::new_err(
                "pilot_db and pilot_frac must be finite",
            ));
        }

        let bl = WALK_SIZE as usize;
        let nb = (n / WALK_SIZE as u64) as usize;
        if nb < 2 {
            return Err(PyValueError::new_err(format!(
                "n must cover at least two blocks of {bl} symbols"
            )));
        }
        if (nb as u64) * (bl as u64) > u32::MAX as u64 {
            return Err(PyValueError::new_err(
                "n must stay below 2^32: the kernel indexes symbols with a u32",
            ));
        }

        // Wiener variance 2*pi*linewidth/symbol_rate per walk; phase carries 2*pi*(dnu_tx + dnu_lo)*T_s.
        let step = std::f64::consts::TAU / rate;

        Ok(Self {
            seed,
            bl,
            nb,
            ampl: va.sqrt(),
            root_t: t.sqrt(),
            ch_sd: (1.0 + t * xi).sqrt(),
            gain: (eta / 2.0).sqrt(),
            det_sd: ((1.0 - eta) / 2.0 + 0.5 + vel).sqrt(),
            pilot_amp: (eta / 2.0).sqrt() * t.sqrt() * (10f64.powf(pilot_db / 10.0) * va).sqrt(),
            // gain^2 ch_sd^2 + det_sd^2, spelled from the inputs: squaring the fields' roots
            // reassociates and moves the last bits of every pilot statistic.
            pilot_sd: ((eta / 2.0) * (1.0 + t * xi) + (1.0 - eta) / 2.0 + 0.5 + vel).sqrt(),
            frac: pilot_frac,
            sd_tx: (step * lw_alice).sqrt(),
            sd_lo: (step * lw_lo).sqrt(),
            eta,
            vel,
        })
    }

    fn used(&self) -> usize {
        self.nb * self.bl
    }

    /// The five counter-RNG streams, in the stage order the kernels index by.
    fn keys(&self) -> [Stream; 5] {
        [
            Stream::new(self.seed, ALICE),
            Stream::new(self.seed, CHANNEL),
            Stream::new(self.seed, DETECT),
            Stream::new(self.seed, WALK_TX),
            Stream::new(self.seed, WALK_LO),
        ]
    }
}

/// Per-block sums of the phase-walk increments.
///
/// `normals(k)[0]` wastes nothing: LLVM drops the unused Box-Muller half. Do not re-propose a
/// `normal()` helper. `block_raw` redraws these; storing them costs 8 B/symbol and the WGSL mirror.
fn walk_sums(p: &Sym, keys: &[Stream; 5]) -> Vec<f64> {
    (0..p.nb)
        .into_par_iter()
        .with_min_len(8)
        .map(|b| {
            let base = b * p.bl;
            let mut s = 0.0;
            for j in 0..p.bl {
                let k = (base + j) as u64;
                s += p.sd_tx * keys[3].normals(k)[0] - p.sd_lo * keys[4].normals(k)[0];
            }

            s
        })
        .collect()
}

/// Exclusive wrapped scan of the block sums, sequential f64: f32 never sees a magnitude past pi.
fn walk_carry(sums: &[f64]) -> Vec<f64> {
    let mut carry = vec![0.0; sums.len()];
    let mut acc = 0.0;
    for (slot, s) in carry.iter_mut().zip(sums.iter()) {
        *slot = acc;
        acc = wrap(acc + s);
    }

    carry
}

/// One block of the fused map in f64: symbols.wgsl's line-for-line reference.
fn block_raw(p: &Sym, b: usize, carry: f64, keys: &[Stream; 5]) -> Raw {
    let base = b * p.bl;
    let mut run = carry;
    let mut acc = Raw::zero();
    let mut p_re = 0.0;
    let mut p_im = 0.0;
    let mut n_re = 0.0;
    let mut n_im = 0.0;
    for j in 0..p.bl {
        let k = base + j;
        let idx = k as u64;

        run = wrap(run + p.sd_tx * keys[3].normals(idx)[0] - p.sd_lo * keys[4].normals(idx)[0]);
        let (ts, tc) = run.sin_cos();

        let ga = keys[0].normals(idx);
        let gc = keys[1].normals(idx);
        let gd = keys[2].normals(idx);
        let xa = p.ampl * ga[0];
        let pa = p.ampl * ga[1];

        let sx = p.root_t * (xa * tc - pa * ts) + p.ch_sd * gc[0];
        let sp = p.root_t * (xa * ts + pa * tc) + p.ch_sd * gc[1];
        let yx = p.gain * sx + p.det_sd * gd[0];
        let yp = p.gain * sp + p.det_sd * gd[1];

        let ix = yx * tc + yp * ts;
        let ip = -yx * ts + yp * tc;
        acc.s_aa += xa * xa + pa * pa;
        acc.s_ai += xa * ix + pa * ip;
        acc.s_yy += yx * yx + yp * yp;

        let carrier = ramp(p.frac, k);
        let (cs, cc) = carrier.sin_cos();
        let (ps, pc) = (carrier + run).sin_cos();
        let px = p.pilot_amp * pc + p.pilot_sd * gd[2];
        let pp = p.pilot_amp * ps + p.pilot_sd * gd[3];
        let dx = px * cc + pp * cs;
        let dp = -px * cs + pp * cc;
        p_re += dx;
        p_im += dp;

        let (ns, nc) = (std::f64::consts::TAU * (j as f64) / (p.bl as f64)).sin_cos();
        n_re += dx * nc + dp * ns;
        n_im += -dx * ns + dp * nc;
    }

    acc.sig = p_re * p_re + p_im * p_im;
    acc.noise = n_re * n_re + n_im * n_im;

    acc
}

/// The f64 arbiter. Fold serially in index order, never rayon's `reduce`: a thread-dependent tree
/// diverged by 2.2e-15, which only `bench/repro.py` catches.
fn cpu_raw(p: &Sym) -> Raw {
    let keys = p.keys();
    let carry = walk_carry(&walk_sums(p, &keys));
    let parts: Vec<Raw> = carry
        .par_chunks(TASK)
        .enumerate()
        .map(|(g, cs)| {
            let mut acc = Raw::zero();
            for (i, c) in cs.iter().enumerate() {
                acc = Raw::merge(acc, block_raw(p, g * TASK + i, *c, &keys));
            }

            acc
        })
        .collect();

    parts.iter().fold(Raw::zero(), |a, b| Raw::merge(a, *b))
}

/// One run's estimator outputs and cost. Downstream of these stays f64 on CPU.
#[pyclass]
pub(crate) struct SymStats {
    t_hat: f64,
    t_chan: f64,
    sigma2: f64,
    xi: f64,
    pilot_snr: f64,
    n_used: u64,
    groups: u64,
    walk_ms: f64,
    fused_ms: f64,
    host_ms: f64,
}

#[pymethods]
impl SymStats {
    /// Regression slope sum(a.y)/sum(a.a) under the oracle phase: sqrt(eta*T/2).
    #[getter]
    fn t_hat(&self) -> f64 {
        self.t_hat
    }

    /// Channel transmittance recovered from the slope: 2*t_hat^2/eta.
    #[getter]
    fn t_chan(&self) -> f64 {
        self.t_chan
    }

    /// Per-quadrature residual variance of y - t_hat*a, in SNU.
    #[getter]
    fn sigma2(&self) -> f64 {
        self.sigma2
    }

    /// Excess noise referred to the channel INPUT (SNU); error in sigma2 amplifies by 1/(T*xi).
    #[getter]
    fn xi(&self) -> f64 {
        self.xi
    }

    /// Pilot signal-to-noise ratio, linear, against the adjacent empty DFT bin.
    #[getter]
    fn pilot_snr(&self) -> f64 {
        self.pilot_snr
    }

    /// Symbols actually processed: n truncated to a whole number of blocks.
    #[getter]
    fn n_used(&self) -> u64 {
        self.n_used
    }

    /// Partial sums finished in f64: one per workgroup, or per group of blocks.
    #[getter]
    fn groups(&self) -> u64 {
        self.groups
    }

    /// GPU ms in the first pass (walk, or store in the split arm). Zero on CPU.
    #[getter]
    fn walk_ms(&self) -> f64 {
        self.walk_ms
    }

    /// GPU ms in the second pass (fused, or the split variant's stats kernel).
    #[getter]
    fn fused_ms(&self) -> f64 {
        self.fused_ms
    }

    /// Wall clock for the call: creation, both submits, both readbacks, finish.
    #[getter]
    fn host_ms(&self) -> f64 {
        self.host_ms
    }
}

/// Sufficient statistics to estimator outputs, f64 whatever produced the sums.
fn finish(p: &Sym, r: Raw, groups: usize, walk_ms: f64, fused_ms: f64, host_ms: f64) -> SymStats {
    let pairs = 2.0 * p.used() as f64;
    let t_hat = if r.s_aa > 0.0 { r.s_ai / r.s_aa } else { 0.0 };
    let sigma2 = (r.s_yy - t_hat * t_hat * r.s_aa) / pairs;
    let xi = if t_hat > 0.0 {
        (sigma2 - 1.0 - p.vel) / (t_hat * t_hat)
    } else {
        0.0
    };
    let snr = if r.noise > 0.0 {
        ((r.sig - r.noise) / r.noise).max(0.0)
    } else {
        0.0
    };

    SymStats {
        t_hat,
        t_chan: 2.0 * t_hat * t_hat / p.eta,
        sigma2,
        xi,
        pilot_snr: snr,
        n_used: p.used() as u64,
        groups: groups as u64,
        walk_ms,
        fused_ms,
        host_ms,
    }
}

#[cfg(feature = "gpu")]
mod kernels {
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::time::Instant;

    use super::{walk_carry, Raw, Sym, WALK_SIZE};

    /// Pass-two width; must match `@workgroup_size` in symbols/stats.wgsl.
    const GROUP: usize = 256;

    /// `frac` mod 1, fixed-point: exact u32 stepping, as f32 loses it at 1e7.
    fn ramp_fixed(frac: f64) -> (u32, u32) {
        let f = frac - frac.floor();
        let scaled = f * 4_294_967_296.0;
        let hi = scaled.floor();
        let lo = ((scaled - hi) * 4_294_967_296.0).floor();

        (hi as u32, lo as u32)
    }

    /// Concatenated ahead of every kernel: WGSL has no include directive.
    const PRELUDE: &str = include_str!("threefry.wgsl");

    const WORDS_SRC: &str = include_str!("words.wgsl");

    const WALK_SRC: &str = include_str!("walk.wgsl");

    const SYMBOLS_SRC: &str = include_str!("symbols.wgsl");

    const STORE_SRC: &str = include_str!("store.wgsl");

    const STATS_SRC: &str = include_str!("stats.wgsl");

    /// Threads per workgroup in words.wgsl, walk.wgsl and store.wgsl.
    const SMALL: u32 = 64;

    pub(crate) struct Ctx {
        device: wgpu::Device,
        queue: wgpu::Queue,
        words: wgpu::ComputePipeline,
        walk: wgpu::ComputePipeline,
        symbols: wgpu::ComputePipeline,
        store: wgpu::ComputePipeline,
        stats: wgpu::ComputePipeline,
        scratch: Mutex<Pool>,
        adapter: String,
        timing: bool,
        period: f32,
    }

    fn put_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn put_f32(v: &mut Vec<u8>, x: f32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn as_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn as_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn as_u64(bytes: &[u8]) -> Vec<u64> {
        bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    /// A mapped readback wants an 8-byte-aligned size, a copy a 4-byte one.
    fn padded(bytes: usize) -> u64 {
        bytes.next_multiple_of(8).max(8) as u64
    }

    fn pipeline(
        device: &wgpu::Device,
        name: &str,
        body: &str,
    ) -> Result<wgpu::ComputePipeline, String> {
        // Scoped so a WGSL parse error degrades to "no GPU path", not an abort.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(format!("{PRELUDE}\n{body}").into()),
        });
        let built = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(format!("{name}.wgsl: {err}"));
        }

        Ok(built)
    }

    fn build() -> Result<Ctx, String> {
        let instance = wgpu::Instance::default();
        let mut opts = wgpu::RequestAdapterOptions::default();
        opts.power_preference = wgpu::PowerPreference::HighPerformance;
        let adapter = pollster::block_on(instance.request_adapter(&opts))
            .map_err(|e| format!("no wgpu adapter: {e}"))?;
        let info = adapter.get_info();

        // Without timestamps the compute path still works, minus per-kernel time.
        let timing = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut spec = wgpu::DeviceDescriptor::default();
        spec.label = Some("qkd");
        spec.required_limits = adapter.limits();
        if timing {
            spec.required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&spec))
            .map_err(|e| format!("no wgpu device: {e}"))?;

        let words = pipeline(&device, "words", WORDS_SRC)?;
        let walk = pipeline(&device, "walk", WALK_SRC)?;
        let symbols = pipeline(&device, "symbols", SYMBOLS_SRC)?;
        let store = pipeline(&device, "store", STORE_SRC)?;
        let stats = pipeline(&device, "stats", STATS_SRC)?;
        let period = queue.get_timestamp_period();

        Ok(Ctx {
            device,
            queue,
            words,
            walk,
            symbols,
            store,
            stats,
            scratch: Mutex::new(Pool::default()),
            adapter: format!("{} ({:?})", info.name, info.backend),
            timing,
            period,
        })
    }

    pub(crate) fn ctx() -> Result<&'static Ctx, String> {
        static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

        match CTX.get_or_init(build) {
            Ok(c) => Ok(c),
            Err(e) => Err(e.to_string()),
        }
    }

    pub(crate) fn describe() -> String {
        match ctx() {
            Ok(c) => {
                let clock = if c.timing {
                    format!("timestamps at {:.3} ns/tick", c.period)
                } else {
                    "no timestamp query".to_string()
                };

                format!("{}, {clock}", c.adapter)
            }
            Err(e) => e,
        }
    }

    impl Ctx {
        fn storage(&self, label: &str, size: u64, src: bool) -> wgpu::Buffer {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if src {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }

            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        }

        fn staging(&self, label: &str, size: u64) -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }

        fn resolve(&self, label: &str, size: u64) -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        }

        fn bind(&self, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
            let layout = pipe.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();

            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            })
        }

        /// Map `len` bytes, not the whole pooled power-of-two buffer.
        fn read(&self, buf: &wgpu::Buffer, len: u64) -> Result<Vec<u8>, String> {
            let (tx, rx) = std::sync::mpsc::channel();
            buf.slice(0..len).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r.is_ok());
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("device poll: {e}"))?;
            match rx.recv() {
                Ok(true) => {}
                Ok(false) => return Err("buffer map failed".to_string()),
                Err(e) => return Err(format!("map callback lost: {e}")),
            }

            // Unmap either way: one left mapped fails every later call.
            let got = buf.slice(0..len).get_mapped_range().map(|view| view.to_vec());
            buf.unmap();

            got.map_err(|e| format!("mapped range: {e}"))
        }

        fn groups(&self, threads: usize, per: u32) -> Result<u32, String> {
            let n = threads.div_ceil(per as usize);
            let cap = self.device.limits().max_compute_workgroups_per_dimension as usize;
            if n > cap {
                return Err(format!("{n} workgroups exceeds the device limit of {cap}"));
            }

            Ok(n as u32)
        }

        /// Serialised. A poisoned lock drops the scratch, keeps the device.
        fn pool(&self) -> MutexGuard<'_, Pool> {
            match self.scratch.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    let mut guard = poisoned.into_inner();
                    guard.clear();

                    guard
                }
            }
        }
    }

    /// Power-of-two rounding; the 2x slack is 1.5 MB at n = 1e8.
    fn room(n: usize) -> usize {
        match n.checked_next_power_of_two() {
            Some(v) => v.max(64),
            None => n,
        }
    }

    struct Ticks {
        set: wgpu::QuerySet,
        raw: wgpu::Buffer,
    }

    /// Rebuilt whole when outgrown, so no bind group names a replaced buffer.
    struct Grid {
        cap: usize,
        cfg: wgpu::Buffer,
        sums: wgpu::Buffer,
        carry: wgpu::Buffer,
        parts: wgpu::Buffer,
        back: wgpu::Buffer,
        pback: wgpu::Buffer,
        theta: Option<wgpu::Buffer>,
        ticks: Option<Ticks>,
        first: wgpu::BindGroup,
        second: wgpu::BindGroup,
    }

    impl Grid {
        fn new(c: &Ctx, cap: usize, split: bool) -> Result<Self, String> {
            let plen = padded(cap.div_ceil(GROUP) * 5 * 4);
            let blen = padded(cap * 4);
            let cfg = c.storage("sym.cfg", 256, false);
            let sums = c.storage("sym.sums", blen, true);
            let carry = c.storage("sym.carry", blen, false);
            let parts = c.storage("sym.parts", plen, true);
            let back = c.staging("sym.back", blen);

            // 32 bytes past the partials for the four timestamps.
            let pback = c.staging("sym.parts.back", plen + 32);
            let theta = if split {
                let want = padded(cap * WALK_SIZE as usize * 4);
                let roof = c.device.limits().max_storage_buffer_binding_size as u64;
                if want > roof {
                    return Err(format!(
                        "the split variant wants a {want}-byte phase array, \
                         past the {roof}-byte binding cap"
                    ));
                }

                Some(c.storage("sym.theta", want, false))
            } else {
                None
            };
            let ticks = if c.timing {
                Some(Ticks {
                    set: c.device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("sym.ticks"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 4,
                    }),
                    raw: c.resolve("sym.ticks.raw", 256),
                })
            } else {
                None
            };
            let (first, second) = match theta.as_ref() {
                Some(t) => (
                    c.bind(&c.store, &[&cfg, &sums, t]),
                    c.bind(&c.stats, &[&cfg, &carry, &parts, t]),
                ),
                None => (
                    c.bind(&c.walk, &[&cfg, &sums]),
                    c.bind(&c.symbols, &[&cfg, &carry, &parts]),
                ),
            };

            Ok(Self {
                cap,
                cfg,
                sums,
                carry,
                parts,
                back,
                pback,
                theta,
                ticks,
                first,
                second,
            })
        }
    }

    struct Draw {
        cap: usize,
        cfg: wgpu::Buffer,
        out: wgpu::Buffer,
        back: wgpu::Buffer,
        bind: wgpu::BindGroup,
    }

    impl Draw {
        fn new(c: &Ctx, cap: usize) -> Self {
            let len = padded(cap * 16);
            let cfg = c.storage("words.cfg", 64, false);
            let out = c.storage("words.out", len, true);
            let back = c.staging("words.back", len);
            let bind = c.bind(&c.words, &[&cfg, &out]);

            // The empty dispatch needs cfg.n zero; write it, do not inherit it.
            c.queue.write_buffer(&cfg, 0, &[0u8; 36]);

            Self {
                cap,
                cfg,
                out,
                back,
                bind,
            }
        }
    }

    /// Grown, never shrunk: crossover 2850 -> 2275 symbols. Mutexed: one staging buffer, two runs.
    #[derive(Default)]
    struct Pool {
        fused: Option<Grid>,
        split: Option<Grid>,
        words: Option<Draw>,
        idle: Option<Draw>,
    }

    impl Pool {
        fn clear(&mut self) {
            self.fused = None;
            self.split = None;
            self.words = None;
            self.idle = None;
        }

        fn grid(&mut self, c: &Ctx, nb: usize, split: bool) -> Result<&Grid, String> {
            let slot = if split {
                &mut self.split
            } else {
                &mut self.fused
            };
            let big = match slot {
                Some(g) => g.cap < nb,
                None => true,
            };
            if big {
                // Drop before allocating: the split phase array is 4 B/symbol.
                *slot = None;
                *slot = Some(Grid::new(c, room(nb), split)?);
            }

            match slot {
                Some(g) => Ok(g),
                None => Err("grid pool did not fill".to_string()),
            }
        }

        fn draw(&mut self, c: &Ctx, n: usize) -> Result<&Draw, String> {
            let big = match &self.words {
                Some(d) => d.cap < n,
                None => true,
            };
            if big {
                self.words = None;
                self.words = Some(Draw::new(c, room(n)));
            }

            match &self.words {
                Some(d) => Ok(d),
                None => Err("words pool did not fill".to_string()),
            }
        }

        fn idle(&mut self, c: &Ctx) -> Result<&Draw, String> {
            if self.idle.is_none() {
                self.idle = Some(Draw::new(c, 1));
            }

            match &self.idle {
                Some(d) => Ok(d),
                None => Err("idle pool did not fill".to_string()),
            }
        }
    }

    pub(crate) fn words(key: [u32; 5], base: u64, n: usize, draw: u32) -> Result<Vec<u32>, String> {
        let c = ctx()?;
        let bytes = n * 16;

        // Against what the pool allocates (a power of two), not what was asked.
        let want = room(n) * 16;
        let cap = c.device.limits().max_storage_buffer_binding_size as usize;
        if bytes == 0 || want > cap {
            return Err(format!("{want} bytes is outside the storage cap of {cap}"));
        }

        let mut cfg = Vec::new();
        for k in key {
            put_u32(&mut cfg, k);
        }
        put_u32(&mut cfg, base as u32);
        put_u32(&mut cfg, (base >> 32) as u32);
        put_u32(&mut cfg, draw);
        put_u32(&mut cfg, n as u32);

        let span = padded(bytes);
        let mut pool = c.pool();
        let d = pool.draw(c, n)?;
        c.queue.write_buffer(&d.cfg, 0, &cfg);
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&c.words);
            pass.set_bind_group(0, &d.bind, &[]);
            pass.dispatch_workgroups(c.groups(n, SMALL)?, 1, 1);
        }
        enc.copy_buffer_to_buffer(&d.out, 0, &d.back, 0, span);
        c.queue.submit(Some(enc.finish()));

        let data = c.read(&d.back, span)?;
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(format!("words dispatch: {err}"));
        }

        let mut got = as_u32(&data);
        got.truncate(n * 4);

        Ok(got)
    }

    /// Pack the plan into `SymCfg`'s layout: 25 key words, four u32, nine f32.
    fn pack(p: &Sym) -> Vec<u8> {
        let mut cfg = Vec::new();
        for s in p.keys() {
            for k in s.key() {
                put_u32(&mut cfg, k);
            }
        }

        let (hi, lo) = ramp_fixed(p.frac);
        put_u32(&mut cfg, p.bl as u32);
        put_u32(&mut cfg, p.nb as u32);
        put_u32(&mut cfg, hi);
        put_u32(&mut cfg, lo);
        for x in [
            p.ampl,
            p.root_t,
            p.ch_sd,
            p.gain,
            p.det_sd,
            p.pilot_amp,
            p.pilot_sd,
            p.sd_tx,
            p.sd_lo,
        ] {
            put_f32(&mut cfg, x as f32);
        }

        cfg
    }

    /// `split` runs store/stats.wgsl for walk/symbols: benchmark 3's arm, not a shipping path.
    pub(crate) fn run(p: &Sym, split: bool) -> Result<(Raw, usize, f64, f64), String> {
        let c = ctx()?;
        let nb = p.nb;
        let cfg = pack(p);
        let ngroups = c.groups(nb, GROUP as u32)?;
        let plen = padded(ngroups as usize * 5 * 4);
        let blen = padded(nb * 4);

        let mut pool = c.pool();
        let g = pool.grid(c, nb, split)?;
        let (one, two) = if split {
            (&c.store, &c.stats)
        } else {
            (&c.walk, &c.symbols)
        };

        // The one buffer the block count does not size; short = stale phases.
        if let Some(t) = g.theta.as_ref() {
            let want = (nb as u64) * (p.bl as u64) * 4;
            if t.size() < want {
                return Err(format!(
                    "phase array is {} bytes, {want} needed",
                    t.size()
                ));
            }
        }

        c.queue.write_buffer(&g.cfg, 0, &cfg);
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("walk"),
                timestamp_writes: g.ticks.as_ref().map(|t| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &t.set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(one);
            pass.set_bind_group(0, &g.first, &[]);
            pass.dispatch_workgroups(c.groups(nb, SMALL)?, 1, 1);
        }
        enc.copy_buffer_to_buffer(&g.sums, 0, &g.back, 0, blen);
        c.queue.submit(Some(enc.finish()));

        // The scan stays f64, sequential, on the host: a monolithic f32 scan
        // errs 4.0e-2 rad, the size of the residual phase noise being measured.
        let mut chunk = as_f32(&c.read(&g.back, blen)?);
        chunk.truncate(nb);
        let wide: Vec<f64> = chunk.iter().map(|x| *x as f64).collect();
        let carried: Vec<u8> = walk_carry(&wide)
            .iter()
            .flat_map(|x| (*x as f32).to_le_bytes())
            .collect();
        c.queue.write_buffer(&g.carry, 0, &carried);

        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("symbols"),
                timestamp_writes: g.ticks.as_ref().map(|t| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &t.set,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
            });
            pass.set_pipeline(two);
            pass.set_bind_group(0, &g.second, &[]);
            pass.dispatch_workgroups(ngroups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&g.parts, 0, &g.pback, 0, plen);
        let tail = if let Some(t) = g.ticks.as_ref() {
            enc.resolve_query_set(&t.set, 0..4, &t.raw, 0);
            enc.copy_buffer_to_buffer(&t.raw, 0, &g.pback, plen, 32);
            32
        } else {
            0
        };
        c.queue.submit(Some(enc.finish()));

        let home = c.read(&g.pback, plen + tail)?;
        let flat = as_f32(&home);
        let mut span = (0.0, 0.0);
        if tail > 0 {
            let ticks = as_u64(&home[plen as usize..]);
            if ticks.len() >= 4 {
                let scale = c.period as f64 * 1e-6;
                span = (
                    ticks[1].saturating_sub(ticks[0]) as f64 * scale,
                    ticks[3].saturating_sub(ticks[2]) as f64 * scale,
                );
            }
        }
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(format!("symbols dispatch: {err}"));
        }

        // f64 finish: a flat f32 accumulator is 27% wrong at n = 1e8, linear in n (dev/notes.md).
        let mut acc = Raw::zero();
        for g in 0..ngroups as usize {
            let at = g * 5;
            if at + 4 >= flat.len() {
                break;
            }

            acc = Raw::merge(
                acc,
                Raw {
                    s_aa: flat[at] as f64,
                    s_ai: flat[at + 1] as f64,
                    s_yy: flat[at + 2] as f64,
                    sig: flat[at + 3] as f64,
                    noise: flat[at + 4] as f64,
                },
            );
        }

        Ok((acc, ngroups as usize, span.0, span.1))
    }

    /// The dispatch latency floor; its buffers are pooled before the clock runs.
    pub(crate) fn empty() -> Result<f64, String> {
        let c = ctx()?;
        let mut pool = c.pool();
        let d = pool.idle(c)?;
        let start = Instant::now();
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&c.words);
            pass.set_bind_group(0, &d.bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&d.out, 0, &d.back, 0, 8);
        c.queue.submit(Some(enc.finish()));
        c.read(&d.back, 8)?;

        Ok(start.elapsed().as_secs_f64() * 1e3)
    }
}

fn no_gpu(why: String) -> PyErr {
    PyRuntimeError::new_err(format!("no gpu compute path: {why}"))
}

#[cfg(feature = "gpu")]
fn ready() -> bool {
    kernels::ctx().is_ok()
}

#[cfg(not(feature = "gpu"))]
fn ready() -> bool {
    false
}

#[cfg(feature = "gpu")]
fn probe() -> String {
    kernels::describe()
}

#[cfg(not(feature = "gpu"))]
fn probe() -> String {
    "built without the gpu feature".to_string()
}

#[cfg(feature = "gpu")]
fn run_words(key: [u32; 5], base: u64, n: u64, draw: u32) -> PyResult<Vec<u32>> {
    kernels::words(key, base, n as usize, draw).map_err(no_gpu)
}

#[cfg(not(feature = "gpu"))]
fn run_words(_key: [u32; 5], _base: u64, _n: u64, _draw: u32) -> PyResult<Vec<u32>> {
    Err(no_gpu("built without the gpu feature".to_string()))
}

#[cfg(feature = "gpu")]
fn run_gpu(p: &Sym, split: bool) -> PyResult<SymStats> {
    let start = std::time::Instant::now();
    let (raw, groups, walk_ms, fused_ms) = kernels::run(p, split).map_err(no_gpu)?;
    let ms = start.elapsed().as_secs_f64() * 1e3;

    Ok(finish(p, raw, groups, walk_ms, fused_ms, ms))
}

#[cfg(not(feature = "gpu"))]
fn run_gpu(_p: &Sym, _split: bool) -> PyResult<SymStats> {
    Err(no_gpu("built without the gpu feature".to_string()))
}

#[cfg(feature = "gpu")]
fn run_empty() -> PyResult<f64> {
    kernels::empty().map_err(no_gpu)
}

#[cfg(not(feature = "gpu"))]
fn run_empty() -> PyResult<f64> {
    Err(no_gpu("built without the gpu feature".to_string()))
}

/// Adapter answered, device opened, kernels compiled. `False` is ordinary.
#[pyfunction]
pub(crate) fn gpu_ready() -> bool {
    ready()
}

/// The adapter and its timing, or why there is no GPU path. Not dispatch logic.
#[pyfunction]
pub(crate) fn gpu_probe() -> String {
    probe()
}

/// `n` raw Threefry-4x32-20 blocks from the CPU reference, from counter `base`.
#[pyfunction]
pub(crate) fn cpu_words<'py>(
    py: Python<'py>,
    seed: u64,
    stage: u32,
    base: u64,
    n: u64,
    draw: u32,
) -> Bound<'py, PyArray1<u32>> {
    let out = py.detach(|| {
        let s = Stream::new(seed, stage);
        let mut out = Vec::with_capacity(n as usize * 4);
        for i in 0..n {
            out.extend_from_slice(&s.block(base.wrapping_add(i), draw));
        }

        out
    });

    out.into_pyarray(py)
}

/// The same blocks out of the WGSL kernel; bit-identical to `cpu_words`.
#[pyfunction]
pub(crate) fn gpu_words<'py>(
    py: Python<'py>,
    seed: u64,
    stage: u32,
    base: u64,
    n: u64,
    draw: u32,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out = py.detach(|| run_words(Stream::new(seed, stage).key(), base, n, draw))?;

    Ok(out.into_pyarray(py))
}

/// Wall-clock milliseconds for one empty dispatch and its readback.
#[pyfunction]
pub(crate) fn gpu_empty() -> PyResult<f64> {
    run_empty()
}

/// The f64 arbiter: the per-symbol map reduced to sufficient statistics.
#[pyfunction]
pub(crate) fn cpu_symbols(
    py: Python<'_>,
    n: u64,
    seed: u64,
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    lw_alice: f64,
    lw_lo: f64,
    symbol_rate: f64,
    pilot_db: f64,
    pilot_frac: f64,
) -> PyResult<SymStats> {
    let p = Sym::new(
        n, seed, va, t, xi, eta, vel, lw_alice, lw_lo, symbol_rate, pilot_db, pilot_frac,
    )?;
    let (raw, ms) = py.detach(|| {
        let start = std::time::Instant::now();
        let raw = cpu_raw(&p);

        (raw, start.elapsed().as_secs_f64() * 1e3)
    });

    Ok(finish(&p, raw, p.nb.div_ceil(TASK), 0.0, 0.0, ms))
}

/// The same map on the GPU in f32. `RuntimeError` when there is no compute path.
#[pyfunction]
pub(crate) fn gpu_symbols(
    py: Python<'_>,
    n: u64,
    seed: u64,
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    lw_alice: f64,
    lw_lo: f64,
    symbol_rate: f64,
    pilot_db: f64,
    pilot_frac: f64,
) -> PyResult<SymStats> {
    let p = Sym::new(
        n, seed, va, t, xi, eta, vel, lw_alice, lw_lo, symbol_rate, pilot_db, pilot_frac,
    )?;

    py.detach(|| run_gpu(&p, false))
}

/// The same map with the phase stored between two kernels rather than re-derived:
/// benchmark 3's arm, not a shipping path. `RuntimeError` past the binding cap.
#[pyfunction]
pub(crate) fn split_symbols(
    py: Python<'_>,
    n: u64,
    seed: u64,
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    lw_alice: f64,
    lw_lo: f64,
    symbol_rate: f64,
    pilot_db: f64,
    pilot_frac: f64,
) -> PyResult<SymStats> {
    let p = Sym::new(
        n, seed, va, t, xi, eta, vel, lw_alice, lw_lo, symbol_rate, pilot_db, pilot_frac,
    )?;

    py.detach(|| run_gpu(&p, true))
}
