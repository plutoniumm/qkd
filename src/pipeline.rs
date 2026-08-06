use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::std::{check_nonneg, check_pos, check_unit, ramp, wrap};
use crate::threefry::{Stream, ALICE, CHANNEL, DETECT, WALK_LO, WALK_TX};

// Modulation -> sampling fade -> thermal-loss channel -> phase walk -> heterodyne -> pilot DSP ->
// parameter estimation. CPU, f64. SNU: vacuum quadrature variance 1 (gaussian.rs uses 1/2), xi at
// the CHANNEL INPUT. Estimator: Leverrier/Grosshans/Grangier PRA 81 062343 (2010) Eq. (20),
// xi_hat = (sigma2_hat - 1 - v_el)/t_hat^2 with t_hat -> sqrt(eta*T/2).
//
// `run_symbols`: local oscillator, two walks, pilot recovery. `run_tlo`: transmitted oscillator,
// one laser, stationary residual, no estimator. At zero delay and linewidths the oracle branches
// are bit-equal.
//
// `jitter` is in the loop, not on the budget: a budget row charges the noise half alone.

/// Walk and reduction grain. Never derive it from the thread count: partials fold in index order.
const CHUNK: usize = 4096;

/// Coarse-acquisition sub-bin resolution, steps of 2*pi/(CFO_GRID*block) rad/sample.
const CFO_GRID: usize = 4;

/// Coarse-acquisition preamble, symbols: ten sigma inside half a grid step at a 0 dB pilot.
const ACQUIRE: usize = 1 << 18;

/// Shared-DETECT buffer cap, f64 slots, 512 MB / ~33.5 M symbols. Both sides are bit-equal.
const CACHE_MAX: usize = (512 << 20) / 8;

/// DSP blocks per reduction group, ~CHUNK symbols. A function of the plan alone.
fn group(bl: usize) -> usize {
    CHUNK.div_ceil(bl).max(1)
}

/// Per-block sufficient statistics, SHIFTED about r = y - t0*a, t0 = sqrt(eta*T/2). Unshifted
/// costs 2.7 decimal digits at V_A = 1e3.
#[derive(Clone, Copy, Default)]
struct Acc {
    s_aa: f64,
    s_ar: f64,
    s_rr: f64,
    s_ai: f64,
    s_ii: f64,
    phi2: f64,
    /// Sum of 1 - cos(theta - theta_hat), the LOST correlation: the cosine quantises v_err at 2e-16.
    vers: f64,
}

impl Acc {
    fn merge(a: Self, b: Self) -> Self {
        Self {
            s_aa: a.s_aa + b.s_aa,
            s_ar: a.s_ar + b.s_ar,
            s_rr: a.s_rr + b.s_rr,
            s_ai: a.s_ai + b.s_ai,
            s_ii: a.s_ii + b.s_ii,
            phi2: a.phi2 + b.phi2,
            vers: a.vers + b.vers,
        }
    }
}

/// Per-symbol constants, all in SNU.
struct Plan {
    ampl: f64,
    root_t: f64,
    ch_sd: f64,
    gain: f64,
    det_sd: f64,
    pilot_amp: f64,
    pilot_sd: f64,
    frac: f64,
    block: usize,
    /// jitter^2/4, the overlap being exp(-fade*g^2). ZERO switches `overlap` off.
    fade: f64,
    /// One period of the acquired coarse ramp; EMPTY at the zero grid point, the exact identity.
    ramp: Vec<f64>,
}

impl Plan {
    /// Pilot fields off and `ramp` empty; `run_symbols` fills them after acquisition.
    fn new(va: f64, t: f64, xi: f64, eta: f64, vel: f64, block: usize, fade: f64) -> Self {
        Self {
            ampl: va.sqrt(),
            root_t: t.sqrt(),
            ch_sd: (1.0 + t * xi).sqrt(),
            gain: (eta / 2.0).sqrt(),
            det_sd: ((1.0 - eta) / 2.0 + 0.5 + vel).sqrt(),
            pilot_amp: 0.0,
            pilot_sd: 0.0,
            frac: 0.0,
            block,
            fade,
            ramp: Vec::new(),
        }
    }

    /// Matched-filter amplitude overlap exp(-delta^2/(4 w^2)), delta/w = jitter*g. Its first two
    /// moments must stay `qkd.impairments.jitter_fading`'s pair: at u = jitter^2,
    /// E[overlap] = (1 + u/2)^(-1/2) = <sqrt(eta)> and E[overlap^2] = (1 + u)^(-1/2) = <eta>.
    ///
    /// Off is exactly 1.0 and draws nothing; on, it reads the THIRD normal of ALICE's block `k`.
    fn overlap(&self, alice: &Stream, k: usize) -> f64 {
        if self.fade == 0.0 {
            return 1.0;
        }

        let g = alice.normals(k as u64)[2];

        (-self.fade * g * g).exp()
    }

    /// Acquired coarse phase at offset `j` of block `b`; exact, not an approximation.
    fn deramp(&self, b: usize, j: usize) -> f64 {
        if self.ramp.is_empty() {
            0.0
        } else {
            self.ramp[(b % CFO_GRID) * self.block + j]
        }
    }
}

/// One link run. `xi_hat` from the DSP frames, `xi_ideal` the same estimator on the oracle phase.
#[pyclass]
pub(crate) struct SimOut {
    v_err: f64,
    v_wrap: f64,
    t_hat: f64,
    t_chan: f64,
    sigma2_hat: f64,
    xi_hat: f64,
    t_ideal: f64,
    xi_ideal: f64,
    pilot_snr: f64,
    cfo: f64,
    n_used: u64,
    frames_x: Vec<f64>,
    frames_p: Vec<f64>,
}

#[pymethods]
impl SimOut {
    /// Residual phase-error variance, rad^2, as -2*ln(E[cos(theta-theta_hat)]). Unbounded above.
    /// The budget takes THIS, not `v_wrap`.
    #[getter]
    fn v_err(&self) -> f64 {
        self.v_err
    }

    /// Wrapped second moment E[wrap(theta - theta_hat)^2], rad^2. NOT the budget input: saturates
    /// at pi^2/3 = 3.2899, agrees with `v_err` only below ~0.6 rad^2.
    #[getter]
    fn v_wrap(&self) -> f64 {
        self.v_wrap
    }

    /// Regression slope sum(a.y)/sum(a.a), estimating sqrt(eta*T/2).
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
    fn sigma2_hat(&self) -> f64 {
        self.sigma2_hat
    }

    /// Input-referred excess noise implied by the DSP-recovered frames (SNU).
    #[getter]
    fn xi_hat(&self) -> f64 {
        self.xi_hat
    }

    #[getter]
    fn t_ideal(&self) -> f64 {
        self.t_ideal
    }

    /// Input-referred excess noise under the oracle phase (SNU), fade included. The fade sits in
    /// both branches: `t_hat / t_ideal` is a phase factor alone.
    #[getter]
    fn xi_ideal(&self) -> f64 {
        self.xi_ideal
    }

    /// Pilot SNR, linear, against the adjacent empty DFT bin. Observable, not oracle.
    #[getter]
    fn pilot_snr(&self) -> f64 {
        self.pilot_snr
    }

    /// Signed CFO recovered from the pilot, Hz, floored by the laser walk's own drift.
    #[getter]
    fn cfo(&self) -> f64 {
        self.cfo
    }

    /// Symbols actually processed: n truncated to a whole number of blocks.
    #[getter]
    fn n_used(&self) -> u64 {
        self.n_used
    }

    /// DSP-recovered x quadratures (SNU), empty unless keep_frames. A fresh copy, never a view.
    #[getter]
    fn frames_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.frames_x.to_pyarray(py)
    }

    /// DSP-recovered p quadratures (SNU), as `frames_x`.
    #[getter]
    fn frames_p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.frames_p.to_pyarray(py)
    }
}

/// Wrapped theta = theta_tx - theta_lo of two Wiener walks (Alice's laser, Bob's LO).
pub(crate) fn walk(used: usize, seed: u64, sd_tx: f64, sd_lo: f64) -> Vec<f64> {
    let tx = Stream::new(seed, WALK_TX);
    let lo = Stream::new(seed, WALK_LO);

    // Skipping a zero-weight draw is safe: normals are indexed, not stateful.
    let inc = |k: u64| {
        let a = if sd_tx == 0.0 {
            0.0
        } else {
            sd_tx * tx.normals(k)[0]
        };

        if sd_lo == 0.0 {
            a
        } else {
            a - sd_lo * lo.normals(k)[0]
        }
    };

    // Storing the increments for pass 2 is cheaper than regenerating them.
    let nch = used.div_ceil(CHUNK);
    let mut theta = vec![0.0; used];
    let sums: Vec<f64> = theta
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(c, out)| {
            let base = c * CHUNK;
            let mut s = 0.0;
            for (j, slot) in out.iter_mut().enumerate() {
                let v = inc((base + j) as u64);
                *slot = v;
                s += v;
            }

            s
        })
        .collect();

    let mut carry = vec![0.0; nch];
    let mut acc = 0.0;
    for c in 0..nch {
        carry[c] = acc;
        acc = wrap(acc + sums[c]);
    }

    theta
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, out)| {
            let mut run = carry[c];
            for slot in out.iter_mut() {
                run = wrap(run + *slot);
                *slot = run;
            }
        });

    theta
}

/// Common-mode residual phase of a TRANSMITTED oscillator: the laser's Wiener increment across the
/// multiplexing delay, zero mean, variance `sd*sd`, independent per symbol. NOT `walk`.
fn common(used: usize, seed: u64, sd: f64) -> Vec<f64> {
    let tx = Stream::new(seed, WALK_TX);
    let mut theta = vec![0.0; used];
    theta
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, out)| {
            let base = c * CHUNK;
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = sd * tx.normals((base + j) as u64)[0];
            }
        });

    theta
}

/// Pilot sample at `k` demodulated by the transmitted tone: complex baseband
/// carrying theta[k] plus detection noise, the tone's frequency cancelled.
fn pilot_at(k: usize, theta: &[f64], plan: &Plan, det: &Stream) -> (f64, f64, f64, f64) {
    let g = det.normals(k as u64);
    let carrier = ramp(plan.frac, k);
    let (ps, pc) = (carrier + theta[k]).sin_cos();
    let px = plan.pilot_amp * pc + plan.pilot_sd * g[2];
    let pp = plan.pilot_amp * ps + plan.pilot_sd * g[3];
    let (cs, cc) = carrier.sin_cos();

    (px * cc + pp * cs, -px * cs + pp * cc, g[0], g[1])
}

/// Coarse CFO acquisition: arg(sum_k z_{k+1} conj(z_k)) over the preamble, rad
/// per sample. Unambiguous across |f| < R/2; the walk moves only the modulus.
fn cfo_coarse(theta: &[f64], plan: &Plan, det: &Stream, used: usize) -> f64 {
    let span = used.min(ACQUIRE);
    let nch = (span - 1).div_ceil(CHUNK);
    let parts: Vec<(f64, f64)> = (0..nch)
        .into_par_iter()
        .map(|c| {
            let lo = c * CHUNK;
            let hi = ((c + 1) * CHUNK).min(span - 1);
            let (mut px, mut pp, _, _) = pilot_at(lo, theta, plan, det);
            let (mut re, mut im) = (0.0, 0.0);
            for k in lo..hi {
                let (nx, np, _, _) = pilot_at(k + 1, theta, plan, det);
                re += nx * px + np * pp;
                im += np * px - nx * pp;
                px = nx;
                pp = np;
            }

            (re, im)
        })
        .collect();

    let (mut re, mut im) = (0.0, 0.0);
    for p in &parts {
        re += p.0;
        im += p.1;
    }

    im.atan2(re)
}

/// One period of the snapped coarse ramp from `w` rad/sample, empty at the zero
/// grid point. The snap makes the phase at k an exact rational multiple of 2pi.
fn ramp_table(w: f64, bl: usize) -> (i64, Vec<f64>) {
    let cells = CFO_GRID * bl;
    let m = (w * (cells as f64) / std::f64::consts::TAU).round() as i64;
    if m == 0 {
        return (0, Vec::new());
    }

    let table = (0..cells)
        .map(|i| {
            let slot = (m * i as i64).rem_euclid(cells as i64);

            std::f64::consts::TAU * (slot as f64) / (cells as f64)
        })
        .collect();

    (m, table)
}

/// One block's contribution to `pilot_bins`. `cache` receives the first two
/// normals of the block's DETECT draws for `block_stats` to read back.
fn one_bin(
    b: usize,
    theta: &[f64],
    plan: &Plan,
    det: &Stream,
    tw: &[(f64, f64)],
    mut cache: Option<&mut [f64]>,
) -> [f64; 4] {
    let bl = plan.block;
    let mut acc = [0.0f64; 4];
    for j in 0..bl {
        let k = b * bl + j;

        let (rx, rp, g0, g1) = pilot_at(k, theta, plan, det);
        if let Some(cc) = cache.as_mut() {
            cc[2 * j] = g0;
            cc[2 * j + 1] = g1;
        }

        let (dx, dp) = if plan.ramp.is_empty() {
            (rx, rp)
        } else {
            let (qs, qc) = plan.deramp(b, j).sin_cos();

            (rx * qc + rp * qs, -rx * qs + rp * qc)
        };
        acc[0] += dx;
        acc[1] += dp;

        // Adjacent bin: no tone, the noise floor.
        let (ns, nc) = tw[j];
        acc[2] += dx * nc + dp * ns;
        acc[3] += -dx * ns + dp * nc;
    }

    acc
}

/// Per-block single-bin DFT of the pilot plus the adjacent empty bin (the
/// measured noise reference). Coarse ramp off first, so the tone is coherent.
fn pilot_bins(
    theta: &[f64],
    plan: &Plan,
    det: &Stream,
    nb: usize,
    cache: &mut [f64],
) -> Vec<[f64; 4]> {
    let bl = plan.block;
    let tw: Vec<(f64, f64)> = (0..bl)
        .map(|j| (std::f64::consts::TAU * (j as f64) / (bl as f64)).sin_cos())
        .collect();

    if cache.is_empty() {
        (0..nb)
            .into_par_iter()
            .with_min_len(8)
            .map(|b| one_bin(b, theta, plan, det, &tw, None))
            .collect()
    } else {
        cache
            .par_chunks_mut(2 * bl)
            .enumerate()
            .with_min_len(8)
            .map(|(b, cc)| one_bin(b, theta, plan, det, &tw, Some(cc)))
            .collect()
    }
}

fn unwrap_arg(bins: &[[f64; 4]]) -> Vec<f64> {
    let mut psi = Vec::with_capacity(bins.len());
    let mut prev = 0.0;
    let mut turns = 0.0;
    for (i, b) in bins.iter().enumerate() {
        let raw = b[1].atan2(b[0]);
        if i > 0 {
            turns += wrap(raw - prev) - (raw - prev);
        }
        prev = raw;
        psi.push(raw + turns);
    }

    psi
}

/// Fine half of the CFO estimate: least-squares slope of the unwrapped RESIDUAL
/// block phases, rad per block. Reported only -- interpolation is affine-
/// equivariant, so this ramp is already inside theta_hat. Do not reapply it.
fn cfo_slope(psi: &[f64]) -> f64 {
    let n = psi.len();
    if n < 2 {
        return 0.0;
    }

    let mid = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, p) in psi.iter().enumerate() {
        let d = i as f64 - mid;
        num += d * p;
        den += d * d;
    }

    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Phase at offset `j` in block `b`, interpolated between the two nearest block
/// centres and held flat past the ends.
fn interp(psi: &[f64], b: usize, j: usize, bl: usize) -> f64 {
    let d = j as f64 - (bl as f64 - 1.0) / 2.0;
    let step = if d < 0.0 {
        if b == 0 {
            0.0
        } else {
            psi[b] - psi[b - 1]
        }
    } else if b + 1 == psi.len() {
        0.0
    } else {
        psi[b + 1] - psi[b]
    };

    psi[b] + step * d / bl as f64
}

/// One block: regenerate Alice's symbols and the noise from the counter RNG,
/// apply channel and detector, derotate by both the DSP and the oracle phase.
fn block_stats(
    b: usize,
    plan: &Plan,
    theta: &[f64],
    psi: &[f64],
    keys: &[Stream; 3],
    cache: &[f64],
    mut frames: Option<(&mut [f64], &mut [f64])>,
) -> Acc {
    let bl = plan.block;
    let mut acc = Acc::default();
    for j in 0..bl {
        let k = b * bl + j;
        let ga = keys[0].normals(k as u64);
        let gc = keys[1].normals(k as u64);
        let gd = if cache.is_empty() {
            let g = keys[2].normals(k as u64);

            [g[0], g[1]]
        } else {
            [cache[2 * k], cache[2 * k + 1]]
        };
        let xa = plan.ampl * ga[0];
        let pa = plan.ampl * ga[1];

        // The SIGNAL arm alone fades, as `impairments.timing`: not the added noise, not the pilot.
        let root_t = plan.root_t * plan.overlap(&keys[0], k);

        // xi is at the channel input, hence the T on the added noise.
        let (ts, tc) = theta[k].sin_cos();
        let sx = root_t * (xa * tc - pa * ts) + plan.ch_sd * gc[0];
        let sp = root_t * (xa * ts + pa * tc) + plan.ch_sd * gc[1];

        let yx = plan.gain * sx + plan.det_sd * gd[0];
        let yp = plan.gain * sp + plan.det_sd * gd[1];

        // psi was formed on the deramped pilot: coarse + residual recombine
        // here and nowhere else.
        let est = plan.deramp(b, j) + interp(psi, b, j, bl);
        let (es, ec) = est.sin_cos();
        let dx = yx * ec + yp * es;
        let dp = -yx * es + yp * ec;
        let ix = yx * tc + yp * ts;
        let ip = -yx * ts + yp * tc;

        // Residuals against the nominal slope, one pair per branch: the shift breaks symmetry.
        let t0 = plan.gain * plan.root_t;
        let rx = dx - t0 * xa;
        let rp = dp - t0 * pa;
        let qx = ix - t0 * xa;
        let qp = ip - t0 * pa;

        // Two non-interchangeable summaries (SimOut::v_err). The versine is 2*pi-periodic.
        let phi = wrap(theta[k] - est);
        let half = (phi / 2.0).sin();
        acc.s_aa += xa * xa + pa * pa;
        acc.s_ar += xa * rx + pa * rp;
        acc.s_rr += rx * rx + rp * rp;
        acc.s_ai += xa * qx + pa * qp;
        acc.s_ii += qx * qx + qp * qp;
        acc.phi2 += phi * phi;
        acc.vers += 2.0 * half * half;

        if let Some((fx, fp)) = frames.as_mut() {
            fx[j] = dx;
            fp[j] = dp;
        }
    }

    acc
}

/// The `grp` blocks from `g * grp`, folded on one thread in block order.
fn group_stats(
    g: usize,
    grp: usize,
    nb: usize,
    plan: &Plan,
    theta: &[f64],
    psi: &[f64],
    keys: &[Stream; 3],
    cache: &[f64],
    frames: Option<(&mut [f64], &mut [f64])>,
) -> Acc {
    let bl = plan.block;
    let lo = g * grp;
    let mut acc = Acc::default();
    match frames {
        Some((fx, fp)) => {
            for (i, (cx, cp)) in fx.chunks_mut(bl).zip(fp.chunks_mut(bl)).enumerate() {
                let one = block_stats(lo + i, plan, theta, psi, keys, cache, Some((cx, cp)));
                acc = Acc::merge(acc, one);
            }
        }
        None => {
            for b in lo..((lo + grp).min(nb)) {
                acc = Acc::merge(acc, block_stats(b, plan, theta, psi, keys, cache, None));
            }
        }
    }

    acc
}

/// Left to right on one thread. Never rayon's `reduce`: f64 addition is not associative. Only
/// `bench/repro.py` catches that regression.
fn fold_acc(parts: &[Acc]) -> Acc {
    parts.iter().fold(Acc::default(), |acc, p| Acc::merge(acc, *p))
}

/// Group partials over `nb` blocks, folded. ONE grouping whether or not frames are kept.
fn gather(
    plan: &Plan,
    theta: &[f64],
    psi: &[f64],
    keys: &[Stream; 3],
    cache: &[f64],
    nb: usize,
    grp: usize,
    keep_frames: bool,
) -> (Acc, Vec<f64>, Vec<f64>) {
    let used = nb * plan.block;
    let span = plan.block * grp;
    let mut frames_x = vec![0.0; if keep_frames { used } else { 0 }];
    let mut frames_p = vec![0.0; if keep_frames { used } else { 0 }];
    let parts: Vec<Acc> = if keep_frames {
        frames_x
            .par_chunks_mut(span)
            .zip(frames_p.par_chunks_mut(span))
            .enumerate()
            .map(|(g, (fx, fp))| group_stats(g, grp, nb, plan, theta, psi, keys, cache, Some((fx, fp))))
            .collect()
    } else {
        (0..nb.div_ceil(grp))
            .into_par_iter()
            .map(|g| group_stats(g, grp, nb, plan, theta, psi, keys, cache, None))
            .collect()
    };

    (fold_acc(&parts), frames_x, frames_p)
}

/// `Acc`'s shift undone. At va = 0 only sigma2_hat means anything. ONE estimator for both entry
/// points.
fn readout(
    stats: &Acc,
    plan: &Plan,
    used: usize,
    eta: f64,
    vel: f64,
    pilot_snr: f64,
    cfo: f64,
    frames_x: Vec<f64>,
    frames_p: Vec<f64>,
) -> SimOut {
    let pairs = 2.0 * used as f64;
    let t0 = plan.gain * plan.root_t;
    let refit = |num: f64, sq: f64| {
        if stats.s_aa > 0.0 {
            (t0 + num / stats.s_aa, (sq - num * num / stats.s_aa) / pairs)
        } else {
            (0.0, sq / pairs)
        }
    };
    let (t_hat, sigma2) = refit(stats.s_ar, stats.s_rr);
    let (t_ideal, s2_ideal) = refit(stats.s_ai, stats.s_ii);
    let solve = |sl: f64, s2: f64| {
        if sl > 0.0 {
            (s2 - 1.0 - vel) / (sl * sl)
        } else {
            0.0
        }
    };

    // 1 - kappa, kappa = E[cos(theta - theta_hat)], through ln_1p. Unweighted where the slope
    // weights by |a|^2: 5e-4 of sampling error apart at 1e6 symbols.
    let lost = stats.vers / used as f64;

    SimOut {
        v_err: if lost < 1.0 {
            -2.0 * (-lost).ln_1p()
        } else {
            f64::INFINITY
        },
        v_wrap: stats.phi2 / used as f64,
        t_hat,
        t_chan: 2.0 * t_hat * t_hat / eta,
        sigma2_hat: sigma2,
        xi_hat: solve(t_hat, sigma2),
        t_ideal,
        xi_ideal: solve(t_ideal, s2_ideal),
        pilot_snr,
        cfo,
        n_used: used as u64,
        frames_x,
        frames_p,
    }
}

/// `Plan::fade`. A finite jitter can square to inf, and exp(-inf * 0) is NaN.
fn fade_of(jitter: f64) -> PyResult<f64> {
    let fade = 0.25 * jitter * jitter;
    if fade.is_finite() {
        Ok(fade)
    } else {
        Err(PyValueError::new_err(format!(
            "jitter must square to a finite width ratio, got {jitter}"
        )))
    }
}

/// Simulate one link run over `n` symbols, truncated to whole DSP blocks.
/// `va`, `xi`, `vel` in SNU with `xi` at the CHANNEL INPUT; linewidths,
/// `symbol_rate`, signed `cfo` in Hz; `pilot_db` against the per-quadrature
/// modulation variance, `pilot_frac` a fraction of the rate.
/// `jitter` is the rms sampling-instant error as a FRACTION of the pulse width, j/w.
#[pyfunction]
#[pyo3(signature = (
    n,
    seed,
    va,
    t,
    xi,
    eta,
    vel,
    lw_alice,
    lw_lo,
    cfo,
    symbol_rate,
    pilot_db,
    pilot_frac,
    block,
    keep_frames,
    jitter = 0.0,
))]
pub(crate) fn run_symbols(
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
    cfo: f64,
    symbol_rate: f64,
    pilot_db: f64,
    pilot_frac: f64,
    block: u64,
    keep_frames: bool,
    jitter: f64,
) -> PyResult<SimOut> {
    check_nonneg("va", va)?;
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;
    check_unit("eta", eta)?;
    check_nonneg("vel", vel)?;
    check_nonneg("lw_alice", lw_alice)?;
    check_nonneg("lw_lo", lw_lo)?;
    check_nonneg("jitter", jitter)?;
    check_pos("symbol_rate", symbol_rate)?;
    let fade = fade_of(jitter)?;
    if !pilot_db.is_finite() || !pilot_frac.is_finite() {
        return Err(PyValueError::new_err("pilot_db and pilot_frac must be finite"));
    }

    // Never clamp cfo in front of this: f + R and f are the same sequence.
    if !cfo.is_finite() || 2.0 * cfo.abs() >= symbol_rate {
        return Err(PyValueError::new_err(format!(
            "cfo must satisfy 2*|cfo| < symbol_rate, got {cfo} at {symbol_rate}: one sample per \
             symbol aliases an offset at or past Nyquist; lower cfo or raise symbol_rate"
        )));
    }
    if block < 2 {
        return Err(PyValueError::new_err(format!("block must be >= 2, got {block}")));
    }

    let bl = block as usize;
    let nb = (n / block) as usize;
    if nb < 2 {
        return Err(PyValueError::new_err(
            "n must cover at least two DSP blocks".to_string(),
        ));
    }

    let used = nb * bl;

    // GIL released for the whole kernel: no Python object is touched below.
    Ok(py.detach(|| {
        let mut plan = Plan::new(va, t, xi, eta, vel, bl, fade);
        plan.pilot_amp = (eta / 2.0).sqrt() * t.sqrt() * (10f64.powf(pilot_db / 10.0) * va).sqrt();
        plan.pilot_sd = ((eta / 2.0) * (1.0 + t * xi) + (1.0 - eta) / 2.0 + 0.5 + vel).sqrt();
        plan.frac = pilot_frac;

        // Wiener increment variance 2*pi*linewidth/symbol_rate per walk.
        let step = std::f64::consts::TAU / symbol_rate;
        let mut theta = walk(used, seed, (step * lw_alice).sqrt(), (step * lw_lo).sqrt());

        // One LO serves pilot and symbols: the offset rides on theta.
        if cfo != 0.0 {
            let cycles = cfo / symbol_rate;
            theta
                .par_iter_mut()
                .enumerate()
                .for_each(|(k, th)| *th = wrap(*th + ramp(cycles, k)));
        }

        // Acquire and snap first: psi tracks only what the coarse ramp left.
        let det = Stream::new(seed, DETECT);
        let (grid, table) = ramp_table(cfo_coarse(&theta, &plan, &det, used), bl);
        plan.ramp = table;
        let coarse = std::f64::consts::TAU * (grid as f64) / ((CFO_GRID * bl) as f64);

        // `pilot_at` reads g[2], g[3] and `block_stats` g[0], g[1] of the same
        // DETECT block; sharing costs 16 B/symbol and measures 0.86-0.91.
        let share = 2 * used <= CACHE_MAX;
        let mut cache = vec![0.0; if share { 2 * used } else { 0 }];
        let bins = pilot_bins(&theta, &plan, &det, nb, &mut cache);
        let psi = unwrap_arg(&bins);
        let slope = cfo_slope(&psi);

        let keys = [
            Stream::new(seed, ALICE),
            Stream::new(seed, CHANNEL),
            det,
        ];
        let (stats, frames_x, frames_p) = gather(&plan, &theta, &psi, &keys, &cache, nb, group(bl), keep_frames);

        // Pilot bin power above the adjacent empty bin, in units of that bin.
        let mut sig = 0.0;
        let mut noise = 0.0;
        for b in &bins {
            sig += b[0] * b[0] + b[1] * b[1];
            noise += b[2] * b[2] + b[3] * b[3];
        }
        let snr = if noise > 0.0 {
            ((sig - noise) / noise).max(0.0)
        } else {
            0.0
        };

        // Two terms so an unacquired coarse half contributes a literal 0.0.
        let offset = coarse * symbol_rate / std::f64::consts::TAU
            + slope * symbol_rate / (std::f64::consts::TAU * bl as f64);

        readout(&stats, &plan, used, eta, vel, snr, offset, frames_x, frames_p)
    }))
}

/// Simulate one TRANSMITTED-oscillator link run over `n` symbols, truncated to whole CHUNKs. Units
/// as `run_symbols`. `linewidth` is ALICE's laser, Hz; `delay` the multiplexing delay, s.
///
/// Residual phase variance is `tlo_phase(linewidth, delay)`, called, not restated. No pilot, no
/// CFO: `t_hat`/`xi_hat` are the UNDEROTATED data and carry the phase cost, `t_ideal`/`xi_ideal`
/// do not. `pilot_snr` is NaN, `cfo` exactly 0.0.
///
/// NOT A SECURITY MODEL: the unit is the one the ARRIVING oscillator sets; see src/tlo.rs.
#[pyfunction]
#[pyo3(signature = (
    n,
    seed,
    va,
    t,
    xi,
    eta,
    vel,
    linewidth,
    delay,
    symbol_rate,
    keep_frames,
    jitter = 0.0,
))]
pub(crate) fn run_tlo(
    py: Python<'_>,
    n: u64,
    seed: u64,
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    linewidth: f64,
    delay: f64,
    symbol_rate: f64,
    keep_frames: bool,
    jitter: f64,
) -> PyResult<SimOut> {
    check_nonneg("va", va)?;
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;
    check_unit("eta", eta)?;
    check_nonneg("vel", vel)?;
    check_nonneg("jitter", jitter)?;
    check_pos("symbol_rate", symbol_rate)?;

    // Validates `linewidth` and `delay`.
    let v_delay = crate::tlo::tlo_phase(linewidth, delay)?;
    let fade = fade_of(jitter)?;

    // Increments over [k*Ts, k*Ts + delay] are disjoint exactly while delay <= Ts.
    let period = 1.0 / symbol_rate;
    if delay > period {
        return Err(PyValueError::new_err(format!(
            "the multiplexing delay is {delay} s against a symbol period of {period} s: residual \
             phases are independent only while the delay fits inside one symbol period; shorten \
             the delay or lower symbol_rate"
        )));
    }

    let bl = CHUNK;
    let nb = (n / bl as u64) as usize;
    if nb < 1 {
        return Err(PyValueError::new_err(format!(
            "n must cover at least one {bl}-symbol group, got {n}"
        )));
    }

    let used = nb * bl;

    // GIL released for the whole kernel: no Python object is touched below.
    Ok(py.detach(|| {
        let plan = Plan::new(va, t, xi, eta, vel, bl, fade);
        let theta = common(used, seed, v_delay.sqrt());

        // Zeros: the recovered branch derotates by nothing, which IS this receiver.
        let psi = vec![0.0; nb];
        let keys = [
            Stream::new(seed, ALICE),
            Stream::new(seed, CHANNEL),
            Stream::new(seed, DETECT),
        ];

        // One CHUNK per group.
        let (stats, frames_x, frames_p) = gather(&plan, &theta, &psi, &keys, &[], nb, 1, keep_frames);

        // NaN, not 0.0: a zero SNR reads as a tone sent and lost.
        readout(&stats, &plan, used, eta, vel, f64::NAN, 0.0, frames_x, frames_p)
    }))
}
