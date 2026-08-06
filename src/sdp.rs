use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::std::{
    check_eps, check_err, check_fec, check_gate, check_nonneg, check_pos, check_prob, check_unit,
    h2, sqrt0,
};

// Semidefinite solver for ONE shape: `min c . v` s.t. `F_k(v) = F_k^0 + sum_i v_i F_k^i >= 0`,
// every block dense, real symmetric and small. Not the relative-entropy key rate, which is
// `dmcs::dm_secure` on `herm.rs`; the two stay apart.
//
// Log-barrier path-following on the DUAL: every iterate is strictly feasible (a step whose
// Cholesky fails is rejected), so `c . v` at any iterate is a bound by weak duality and an early
// stop is LOOSER, never wrong. An unverified primal optimum fails in the insecure direction.
//
// Problem: Seksaria & Prabhakar, "Short reach by theorem: the certifiable key rate of COW QKD"
// (2026), Sec. VI D, tightening Gao et al., Opt. Express 30, 23783 (2022), arXiv:2107.09329.
//
// No `q.Link` reaches it: `q.IntensityKeying` is THREE-sequence and COW' adds the empty |0>|0>,
// whose gains move the certified phase error 0.274 -> 0.852 off the dark floor at 1e-2.

/// Newton steps per barrier round before the round is abandoned.
const NEWTON_MAX: usize = 120;

/// Squared Newton decrement at which a barrier subproblem counts as centred.
const CENTRED: f64 = 1e-11;

/// Barrier parameter growth per outer iteration.
const T_GROW: f64 = 15.0;

/// Backtracking floor: below this the step is shorter than the f64 resolution of `v`.
const STEP_MIN: f64 = 1e-13;

/// Armijo constant for the barrier line search.
const ARMIJO: f64 = 0.25;

/// Retries after a Newton breakdown; each takes the square root of the growth.
const RETRY_MAX: usize = 6;

/// Ridges tried in order on a Hessian whose Cholesky fails, relative to its unit Jacobi diagonal.
const RIDGES: [f64; 4] = [0.0, 1e-13, 1e-11, 1e-9];

// `chol`, `chol_solve` duplicate `herm::real_chol`, `real_solve` on purpose; no shared invariant.

/// Lower Cholesky factor, row-major, or `None` if not positive definite: the feasibility test.
fn chol(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];

            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }

            if i == j {
                if !(s > 0.0) {
                    return None;
                }

                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    Some(l)
}

/// log det A from its Cholesky factor.
fn chol_logdet(l: &[f64], n: usize) -> f64 {
    let mut s = 0.0;

    for i in 0..n {
        s += l[i * n + i].ln();
    }

    2.0 * s
}

/// A^-1 from its Cholesky factor: invert the triangle, then form Linv^T Linv.
fn chol_inv(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0f64; n * n];

    for i in 0..n {
        inv[i * n + i] = 1.0 / l[i * n + i];

        for j in 0..i {
            let mut s = 0.0;

            for k in j..i {
                s += l[i * n + k] * inv[k * n + j];
            }

            inv[i * n + j] = -s / l[i * n + i];
        }
    }

    let mut out = vec![0.0f64; n * n];

    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;

            for k in i.max(j)..n {
                s += inv[k * n + i] * inv[k * n + j];
            }

            out[i * n + j] = s;
        }
    }

    out
}

/// Solve A x = b from A's Cholesky factor.
fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; n];

    for i in 0..n {
        let mut s = b[i];

        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }

        y[i] = s / l[i * n + i];
    }

    let mut x = vec![0.0f64; n];

    for step in 0..n {
        let i = n - 1 - step;
        let mut s = y[i];

        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }

        x[i] = s / l[i * n + i];
    }

    x
}

/// Smallest eigenvalue of a symmetric matrix by cyclic Jacobi, to certify the returned dual point.
/// The 80 and 1e-32 stay bare: named, they would read as shared with `herm::SWEEP_MAX`, `OFF_TOL`.
fn eig_min(a: &[f64], n: usize) -> f64 {
    let mut m = a.to_vec();
    let mut scale = 0.0f64;

    for x in m.iter() {
        scale = scale.max(x.abs());
    }

    if scale == 0.0 {
        return 0.0;
    }

    for _ in 0..80 {
        let mut off = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }

        if off <= 1e-32 * scale * scale {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];

                if apq == 0.0 {
                    continue;
                }

                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let tan = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let cos = 1.0 / (tan * tan + 1.0).sqrt();
                let sin = tan * cos;

                for k in 0..n {
                    let akp = m[k * n + p];
                    let akq = m[k * n + q];
                    m[k * n + p] = cos * akp - sin * akq;
                    m[k * n + q] = sin * akp + cos * akq;
                }

                for k in 0..n {
                    let apk = m[p * n + k];
                    let aqk = m[q * n + k];
                    m[p * n + k] = cos * apk - sin * aqk;
                    m[q * n + k] = sin * apk + cos * aqk;
                }
            }
        }
    }

    let mut lo = m[0];

    for i in 1..n {
        lo = lo.min(m[i * n + i]);
    }

    lo
}

/// One linear matrix inequality: `f` holds `1 + n` matrices of side `m`, row-major and
/// concatenated, the constant term first.
struct Block {
    m: usize,
    f: Vec<f64>,
}

impl Block {
    fn term(&self, i: usize) -> &[f64] {
        let sq = self.m * self.m;

        &self.f[i * sq..(i + 1) * sq]
    }

    /// F(v), the slack matrix at `v`.
    fn slack(&self, v: &[f64]) -> Vec<f64> {
        let sq = self.m * self.m;
        let mut s = self.term(0).to_vec();

        for (i, &vi) in v.iter().enumerate() {
            if vi == 0.0 {
                continue;
            }

            let f = self.term(i + 1);

            for k in 0..sq {
                s[k] += vi * f[k];
            }
        }

        s
    }
}

/// `min c . v` subject to every block's `F_k(v) >= 0`.
struct Lmi {
    n: usize,
    c: Vec<f64>,
    blocks: Vec<Block>,
}

/// `v` is dual feasible whatever `status` says, so `c . v` bounds the primal optimum
/// either way.
struct Report {
    v: Vec<f64>,
    gap: f64,
    iters: usize,
    status: &'static str,
}

impl Lmi {
    /// Barrier value `t (c . v) - sum_k logdet F_k(v)`, or `None` at an infeasible point.
    fn barrier(&self, v: &[f64], t: f64) -> Option<f64> {
        let mut acc = 0.0;

        for (a, b) in self.c.iter().zip(v.iter()) {
            acc += a * b;
        }

        acc *= t;

        for blk in self.blocks.iter() {
            let s = blk.slack(v);
            let l = chol(&s, blk.m)?;
            acc -= chol_logdet(&l, blk.m);
        }

        if acc.is_finite() {
            Some(acc)
        } else {
            None
        }
    }

    /// Newton direction and squared decrement, or `None` at an unusable point or Hessian.
    fn newton(&self, v: &[f64], t: f64) -> Option<(Vec<f64>, f64)> {
        let n = self.n;
        let mut grad = vec![0.0f64; n];
        let mut hess = vec![0.0f64; n * n];

        for (gi, ci) in grad.iter_mut().zip(self.c.iter()) {
            *gi = t * ci;
        }

        for blk in self.blocks.iter() {
            let m = blk.m;
            let sq = m * m;
            let s = blk.slack(v);
            let l = chol(&s, m)?;
            let si = chol_inv(&l, m);

            // sandwich[a] = S^-1 F^a S^-1, so H_ab = <sandwich[a], F^b>.
            let mut sand = vec![0.0f64; n * sq];

            for a in 0..n {
                let f = blk.term(a + 1);
                let mut tr = 0.0;

                for i in 0..m {
                    for j in 0..m {
                        tr += si[i * m + j] * f[j * m + i];
                    }
                }

                grad[a] -= tr;

                let mut tmp = vec![0.0f64; sq];

                for i in 0..m {
                    for j in 0..m {
                        let mut acc = 0.0;

                        for k in 0..m {
                            acc += si[i * m + k] * f[k * m + j];
                        }

                        tmp[i * m + j] = acc;
                    }
                }

                for i in 0..m {
                    for j in 0..m {
                        let mut acc = 0.0;

                        for k in 0..m {
                            acc += tmp[i * m + k] * si[k * m + j];
                        }

                        sand[a * sq + i * m + j] = acc;
                    }
                }
            }

            for a in 0..n {
                for b in a..n {
                    let fb = blk.term(b + 1);
                    let mut acc = 0.0;

                    for k in 0..sq {
                        acc += sand[a * sq + k] * fb[k];
                    }

                    hess[a * n + b] += acc;
                }
            }
        }

        for a in 0..n {
            for b in a..n {
                let x = hess[a * n + b];
                hess[b * n + a] = x;
            }
        }

        // Jacobi preconditioning: the COW' dual's y- and Z-blocks differ by eight decades.
        let mut d = vec![0.0f64; n];

        for a in 0..n {
            let diag = hess[a * n + a];

            if !(diag > 0.0) {
                return None;
            }

            d[a] = 1.0 / diag.sqrt();
        }

        let mut scaled = vec![0.0f64; n * n];

        for a in 0..n {
            for b in 0..n {
                scaled[a * n + b] = hess[a * n + b] * d[a] * d[b];
            }
        }

        let mut rhs = vec![0.0f64; n];

        for a in 0..n {
            rhs[a] = -grad[a] * d[a];
        }

        for ridge in RIDGES.iter() {
            let mut trial = scaled.to_vec();

            for a in 0..n {
                trial[a * n + a] += ridge;
            }

            if let Some(l) = chol(&trial, n) {
                let w = chol_solve(&l, n, &rhs);
                let mut dir = vec![0.0f64; n];
                let mut lam2 = 0.0;

                for a in 0..n {
                    dir[a] = w[a] * d[a];
                    lam2 -= grad[a] * dir[a];
                }

                if !dir.iter().all(|x| x.is_finite()) {
                    return None;
                }

                return Some((dir, lam2));
            }
        }

        None
    }

    /// Backtracking line search that never leaves the strictly feasible region.
    fn advance(&self, v: &[f64], dir: &[f64], t: f64, lam2: f64) -> Option<Vec<f64>> {
        let base = self.barrier(v, t)?;
        let mut step = 1.0f64;

        while step > STEP_MIN {
            let mut trial = vec![0.0f64; self.n];

            for a in 0..self.n {
                trial[a] = v[a] + step * dir[a];
            }

            if let Some(val) = self.barrier(&trial, t) {
                if val <= base - ARMIJO * step * lam2 {
                    return Some(trial);
                }
            }

            step *= 0.5;
        }

        None
    }

    /// Path-following from a strictly feasible `start` until the central-path gap `nu/t`
    /// falls to `gaptol`, `nu` the sum of block sides. Returns the last completed round's point.
    fn run(&self, start: &[f64], gaptol: f64) -> Report {
        let nu: f64 = self.blocks.iter().map(|b| b.m as f64).sum();
        let mut v = start.to_vec();
        let mut best = start.to_vec();
        let mut grow = T_GROW;
        let mut held = 1.0f64;
        let mut t = 1.0f64;
        let mut iters = 0usize;
        let mut tries = 0usize;
        let mut status = "converged";

        loop {
            let mut broke = false;

            for _ in 0..NEWTON_MAX {
                iters += 1;

                match self.newton(&v, t) {
                    None => {
                        broke = true;
                        break;
                    }
                    Some((dir, lam2)) => {
                        if lam2 <= CENTRED {
                            break;
                        }

                        match self.advance(&v, &dir, t, lam2) {
                            None => {
                                broke = true;
                                break;
                            }
                            Some(next) => v = next,
                        }
                    }
                }
            }

            // Breakdown: retry from the last centred point at the square root of the growth.
            if broke {
                if tries >= RETRY_MAX || grow <= 1.0 + 1e-3 {
                    status = "breakdown";
                    break;
                }

                tries += 1;
                grow = grow.sqrt();
                v.copy_from_slice(&best);
                t = held * grow;
                continue;
            }

            best.copy_from_slice(&v);
            held = t;

            if nu / t <= gaptol {
                break;
            }

            t *= grow;

            if !t.is_finite() {
                status = "exhausted";
                break;
            }
        }

        Report {
            v: best,
            gap: nu / held,
            iters,
            status,
        }
    }
}

// The COW' phase-error problem, Seksaria & Prabhakar Sec. VI D, over the sequences |0>|0>,
// |a>|a>, |0>|a> = |0_z>, |a>|0> = |1_z>. Gam^i_jk = <s_j| E_Mi |s_k> is a SUB-POVM whose
// achievable set is EXACTLY Gam^0, Gam^1, G - Gam^0 - Gam^1 >= 0, the gains fixing the diagonals.
// Gao's Eq. (10): E_p = 1/2 + (Gam^1_23 - Gam^0_23) / C, C the four Z-row gains. Complex phases
// drop without loss. The dual, W the symmetric 1/2 on entry (2,3), each feasible point >= E_p:
//
//      min  Tr(Z G) + y0 . Q0 + y1 . Q1
//      s.t. Z >= 0,   Z + diag(y0) + W >= 0,   Z + diag(y1) - W >= 0

/// Index pairs of the ten independent entries of a symmetric 4x4, diagonal first.
const ZIDX: [(usize, usize); 10] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
];

/// Scalar variables of the COW' dual: ten for Z, four for y0, four for y1.
const NVAR: usize = 18;

/// Repair margin in largest-block-entry units: Cholesky accepts a matrix a few epsilons negative.
const REPAIR: f64 = 64.0 * f64::EPSILON;

/// Gram matrix of the four COW' sequences, row-major 4x4, order `|0>|0>, |a>|a>, |0>|a>, |a>|0>`.
#[pyfunction]
pub(crate) fn sdp_gram(mu: f64) -> PyResult<Vec<f64>> {
    check_pos("mu", mu)?;

    Ok(gram(mu).to_vec())
}

fn gram(mu: f64) -> [f64; 16] {
    let c = (-0.5 * mu).exp();
    let c2 = (-mu).exp();

    [
        1.0, c2, c, c, c2, 1.0, c, c, c, c, 1.0, c2, c, c, c2, 1.0,
    ]
}

/// `(constructive, destructive)` port amplitudes of each sequence in pulse-amplitude units, slot
/// `n` at `(E_n +/- E_{n-1})/2`. Not `/sqrt(2)`: that is 3 dB (~15 km) more monitor light.
fn ports(mu: f64) -> ([f64; 4], [f64; 4]) {
    let a = mu.sqrt();

    ([0.0, a, 0.5 * a, 0.5 * a], [0.0, 0.0, -0.5 * a, 0.5 * a])
}

/// Honest monitoring gains `(q0, q1)` at `D_M0` (constructive) and `D_M1` (destructive), order
/// `|0>|0>, |a>|a>, |0>|a>, |a>|0>`. `eta` EXCLUDES Bob's splitter, so the monitor sees
/// `eta * (1 - t_b)`; `dark` is per gate.
#[pyfunction]
pub(crate) fn sdp_gains(
    mu: f64,
    t_b: f64,
    eta: f64,
    dark: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    check_pos("mu", mu)?;
    check_prob("t_b", t_b)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    let (q0, q1) = honest(mu, t_b, eta, dark);

    Ok((q0.to_vec(), q1.to_vec()))
}

/// `(Gam^0, Gam^1)` in full: `G_jk [1 - (1 - dark) exp(-eta_m v_j v_k)]` behind pure loss, with the
/// UNattenuated `G` (the loss cancels against the attenuated overlap).
fn coherences(mu: f64, t_b: f64, eta: f64, dark: f64) -> ([f64; 16], [f64; 16]) {
    let g = gram(mu);
    let em = eta * (1.0 - t_b);
    let (vp, vm) = ports(mu);
    let mut out = ([0.0f64; 16], [0.0f64; 16]);

    for j in 0..4 {
        for k in 0..4 {
            let idx = j * 4 + k;
            out.0[idx] = g[idx] * (1.0 - (1.0 - dark) * (-em * vp[j] * vp[k]).exp());
            out.1[idx] = g[idx] * (1.0 - (1.0 - dark) * (-em * vm[j] * vm[k]).exp());
        }
    }

    out
}

fn honest(mu: f64, t_b: f64, eta: f64, dark: f64) -> ([f64; 4], [f64; 4]) {
    let (a, b) = coherences(mu, t_b, eta, dark);
    let mut q0 = [0.0f64; 4];
    let mut q1 = [0.0f64; 4];

    for j in 0..4 {
        q0[j] = a[j * 4 + j];
        q1[j] = b[j * 4 + j];
    }

    (q0, q1)
}

/// Phase error rate of the honest channel: a point INSIDE the constraint set, so a floor under
/// every certified value. Raw, uncapped.
#[pyfunction]
pub(crate) fn sdp_honest(mu: f64, t_b: f64, eta: f64, dark: f64) -> PyResult<f64> {
    check_pos("mu", mu)?;
    check_prob("t_b", t_b)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    let (a, b) = coherences(mu, t_b, eta, dark);
    let (q0, q1) = honest(mu, t_b, eta, dark);
    let den = q0[2] + q1[2] + q0[3] + q1[3];

    if den <= 0.0 {
        return Err(PyValueError::new_err(
            "the monitoring line never clicked on either bit sequence, so the phase error \
             rate is a ratio 0/0 and is undefined",
        ));
    }

    Ok(0.5 + (b[11] - a[11]) / den)
}

/// `(gain, qber)` of the COW' data line per sequence, light bin through `eta * t_b`, a double click
/// a random bit. Dark counts are the whole honest error.
#[pyfunction]
pub(crate) fn sdp_data(mu: f64, t_b: f64, eta: f64, dark: f64) -> PyResult<(f64, f64)> {
    check_pos("mu", mu)?;
    check_prob("t_b", t_b)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    let hit = 1.0 - (1.0 - dark) * (-eta * t_b * mu).exp();
    let right = hit * (1.0 - dark) + 0.5 * hit * dark;
    let wrong = (1.0 - hit) * dark + 0.5 * hit * dark;
    let gain = right + wrong;

    if gain <= 0.0 {
        return Ok((0.0, 0.5));
    }

    Ok((gain, (wrong / gain).min(0.5)))
}

/// Gao's analytic phase-error bound, Opt. Express 30, 23783 (2022) Eqs. (7) and (12), Eq. (10) of
/// the review. Its one unmeasurable term, bounded by 1, is the looseness the SDP removes.
/// Raw: >= 1/2 aborts.
#[pyfunction]
pub(crate) fn sdp_analytic(mu: f64, q0: Vec<f64>, q1: Vec<f64>) -> PyResult<f64> {
    check_pos("mu", mu)?;
    check_record(&q0, &q1)?;

    let np = 2.0 * (1.0 + (-mu).exp());
    let nm = 2.0 * (1.0 - (-mu).exp());
    let up = (0.5 * mu).exp() * sqrt0(q1[1]) + (-0.5 * mu).exp() * sqrt0(q1[0]);
    let hi = (up * up
        + nm * (0.25 * nm * mu.exp() + mu.exp() * sqrt0(q1[1]) + sqrt0(q1[0])))
        / np;
    let lo0 = (0.5 * mu).exp() * sqrt0(q0[1]) - (-0.5 * mu).exp() * sqrt0(q0[0]);
    let lo =
        (lo0 * lo0 - nm * (mu.exp() * sqrt0(q0[1]) + sqrt0(q0[0]))) / np;

    // Bounds on a click probability, so clipping to [0, 1] is valid.
    let hi = hi.min(1.0);
    let lo = lo.max(0.0);
    let den = 2.0 * (q0[2] + q1[2] + q0[3] + q1[3]);

    if den <= 0.0 {
        return Err(PyValueError::new_err(
            "the monitoring line never clicked on either bit sequence, so the phase error \
             rate is a ratio 0/0 and is undefined",
        ));
    }

    Ok((np * hi + 2.0 * (q0[2] + q0[3]) - np * lo) / den)
}

/// Shared validation of an observed monitoring record.
fn check_record(q0: &[f64], q1: &[f64]) -> PyResult<()> {
    if q0.len() != 4 || q1.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "q0 and q1 must each hold the four sequence gains |0>|0>, |a>|a>, |0>|a>, \
             |a>|0>, got {} and {}",
            q0.len(),
            q1.len()
        )));
    }

    for j in 0..4 {
        check_prob("q0", q0[j])?;
        check_prob("q1", q1[j])?;

        if q0[j] + q1[j] > 1.0 {
            return Err(PyValueError::new_err(format!(
                "sequence {j} clicks at both monitoring detectors with total probability \
                 {} > 1, which no sub-POVM can produce",
                q0[j] + q1[j]
            )));
        }
    }

    Ok(())
}

// Composable finite key against COHERENT attacks: Li, Cao, Xie, Yin & Chen, Phys. Rev. Research 6,
// 013022 (2024), arXiv:2309.16136. Chain `sdp_kato` (x6) -> `sdp_interval` -> `sdp_sample` ->
// `sdp_length`, on the ANALYTIC estimator, not the certified one (see `sdp_finite`).
//
// Printing errors in arXiv:2309.16136v2, corrected here: Eqs. (16)-(17) divide by the undefined
// `n_w`, where a gain needs sequences SENT, `N_w = N*p_d`; Eq. (21) writes `N * E_p` for `n_z`.

/// Li's per-estimation failure probability `eps_sec / 10`, all terms of
/// `eps_sec = 2*eps + eps_0 + 6*eps_1 + eps_2` equal. The share, NOT the total; `sdp_length`'s `5`
/// is the same split.
#[pyfunction]
pub(crate) fn sdp_eps(eps_sec: f64) -> PyResult<f64> {
    check_eps("eps_sec", eps_sec)?;

    Ok(eps_sec / 10.0)
}

/// Kato's deviation in COUNTS at its optimised `(a, b)`: `[b + a*(2*count/k - 1)]*sqrt(k)` over `k`
/// rounds. `upper = true` bounds the expected value at `count + this`, `false` at `count - this`.
///
/// Kato, "Concentration inequality using unconfirmed knowledge", arXiv:2002.04357 (2020).
/// Minimisers: Curras-Lorenzo, Navarrete, Azuma, Kato, Curty & Razavi, npj Quantum Inf. 7, 22
/// (2021), as Li, Cao, Xie, Yin & Chen, Phys. Rev. Research 6, 013022 (2024), Eqs. (24)-(28).
///
/// The two directions are NOT one function of a sign: `(1 +/- 4a/(3*sqrt(k)))` changes `a` and `b`.
#[pyfunction]
pub(crate) fn sdp_kato(k: f64, count: f64, eps: f64, upper: bool) -> PyResult<f64> {
    check_pos("k", k)?;
    check_nonneg("count", count)?;
    check_eps("eps", eps)?;

    if count > k {
        return Err(PyValueError::new_err(format!(
            "count must not exceed k (got {count} > {k}): Kato's variables are Bernoulli, one \
             per round, so k rounds cannot produce more than k clicks"
        )));
    }

    let ln_e = eps.ln();
    let side = k.sqrt();
    let spread = 9.0 * count * (k - count) - 2.0 * k * ln_e;
    let root = (-k * k * ln_e * spread).sqrt();
    let mixed = 72.0 * side * count * (k - count) * ln_e - 16.0 * k * side * ln_e * ln_e;
    let tail = 9.0 * (2.0f64).sqrt() * (k - 2.0 * count) * root;
    let den = 4.0 * (9.0 * k - 8.0 * ln_e) * spread;

    if !(den.is_finite() && den != 0.0) {
        return Err(PyValueError::new_err(
            "Kato's optimiser has no finite solution at this (k, count, eps)",
        ));
    }

    let a = if upper {
        3.0 * (mixed + tail) / den
    } else {
        -3.0 * (mixed - tail) / den
    };
    let cross = if upper { 24.0 * a * side } else { -24.0 * a * side };
    let inner = 18.0 * a * a * k - (16.0 * a * a + cross + 9.0 * k) * ln_e;

    if !(inner >= 0.0) {
        return Err(PyValueError::new_err(format!(
            "Kato's b has no real value at k = {k}, count = {count}, eps = {eps}, so no \
             deviation is certified here"
        )));
    }

    let b = inner.sqrt() / (3.0 * (2.0 * k).sqrt());

    if !(b >= a.abs()) {
        return Err(PyValueError::new_err(format!(
            "Kato's inequality requires b >= |a| and the optimiser returned b = {b}, a = {a}: \
             the bound it would give is outside the theorem"
        )));
    }

    let dev = (b + a * (2.0 * count / k - 1.0)) * side;

    if !dev.is_finite() {
        return Err(PyValueError::new_err(
            "the Kato deviation is not finite at this (k, count, eps)",
        ));
    }

    Ok(dev)
}

/// Validates an interval record: four upper-bounded gains per detector, two lower bounds at `D_M0`.
fn check_interval(q0: &[f64], q1: &[f64], lo: &[f64]) -> PyResult<()> {
    if q0.len() != 4 || q1.len() != 4 || lo.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "q0 and q1 must each hold the four sequence gains |0>|0>, |a>|a>, |0>|a>, |a>|0> \
             and lo the two lower bounds |0>|0>, |a>|a> at D_M0, got {}, {} and {}",
            q0.len(),
            q1.len(),
            lo.len()
        )));
    }

    for j in 0..4 {
        check_nonneg("q0", q0[j])?;
        check_nonneg("q1", q1[j])?;
    }

    // The bit entries are observed, not bounded, so the sub-POVM condition binds on them.
    for j in 2..4 {
        check_prob("q0", q0[j])?;
        check_prob("q1", q1[j])?;

        if q0[j] + q1[j] > 1.0 {
            return Err(PyValueError::new_err(format!(
                "sequence {j} clicks at both monitoring detectors with total probability {} > 1, \
                 which no sub-POVM can produce",
                q0[j] + q1[j]
            )));
        }
    }

    for j in 0..2 {
        check_nonneg("lo", lo[j])?;

        if lo[j] > q0[j] {
            return Err(PyValueError::new_err(format!(
                "the lower bound on decoy gain {j} at D_M0 exceeds its upper bound ({} > {}): \
                 the two came from different records",
                lo[j], q0[j]
            )));
        }
    }

    Ok(())
}

/// `sdp_analytic` on Kato bounds: an EXPECTED rate, which `sdp_sample` makes observed. `q0`, `q1`
/// as in `sdp_analytic`, DECOY entries upper-bounded and BIT entries observed; `lo` the two decoy
/// lower bounds at `D_M0`.
/// Li, Cao, Xie, Yin & Chen, Phys. Rev. Research 6, 013022 (2024), Eqs. (18)-(20).
///
/// Eq. (19) EXPANDS the square `sdp_analytic` groups: leading terms take LOWER bounds, the cross
/// term UPPER, so a grouped `(sqrt(lo_aa) - sqrt(lo_00))^2` is unproved. Raw, uncapped.
#[pyfunction]
pub(crate) fn sdp_interval(
    mu: f64,
    q0: Vec<f64>,
    q1: Vec<f64>,
    lo: Vec<f64>,
) -> PyResult<f64> {
    check_pos("mu", mu)?;
    check_interval(&q0, &q1, &lo)?;

    // A Kato upper bound can pass 1; a click probability cannot.
    let m1_bright = q1[1].min(1.0);
    let m1_empty = q1[0].min(1.0);
    let m0_bright = q0[1].min(1.0);
    let m0_empty = q0[0].min(1.0);
    let low_bright = lo[1].min(1.0);
    let low_empty = lo[0].min(1.0);
    let np = 2.0 * (1.0 + (-mu).exp());
    let nm = 2.0 * (1.0 - (-mu).exp());
    let up = (0.5 * mu).exp() * sqrt0(m1_bright) + (-0.5 * mu).exp() * sqrt0(m1_empty);
    let hi = (up * up
        + nm * (0.25 * nm * mu.exp() + mu.exp() * sqrt0(m1_bright) + sqrt0(m1_empty)))
        / np;
    let bulk = mu.exp() * low_bright + (-mu).exp() * low_empty
        - 2.0 * sqrt0(m0_empty * m0_bright);
    let low =
        bulk / np - (nm / np) * (mu.exp() * sqrt0(m0_bright) + sqrt0(m0_empty));
    let hi = hi.min(1.0);
    let low = low.max(0.0);
    let den = 2.0 * (q0[2] + q1[2] + q0[3] + q1[3]);

    if den <= 0.0 {
        return Err(PyValueError::new_err(
            "the monitoring line never clicked on either bit sequence, so the phase error \
             rate is a ratio 0/0 and is undefined",
        ));
    }

    Ok((np * hi + 2.0 * (q0[2] + q0[3]) - np * low) / den)
}

/// Li Eq. (21), expected to observed: `expect + sqrt(ln(1/eps) / (2*n_z))` over `n_z` sifted
/// rounds. Kato at `a = 0` (Azuma's width), as Curras-Lorenzo, Navarrete, Azuma, Kato, Curty &
/// Razavi, npj Quantum Inf. 7, 22 (2021): the optimised `a` needs the count being bounded.
/// Raw, uncapped.
#[pyfunction]
pub(crate) fn sdp_sample(expect: f64, n_z: f64, eps: f64) -> PyResult<f64> {
    check_nonneg("expect", expect)?;
    check_pos("n_z", n_z)?;
    check_eps("eps", eps)?;

    Ok(expect + (0.5 * (1.0 / eps).ln() / n_z).sqrt())
}

/// Composable finite-key LENGTH in bits for COW', Li, Cao, Xie, Yin & Chen, Phys. Rev.
/// Research 6, 013022 (2024), Eq. (23):
/// `l = floor( n_z*[1 - h2(phase)] - f_ec*n_z*h2(e_z) - log2(2/eps_cor) - 2*log2(5/eps_sec) )`.
/// A LENGTH over the whole block, not bits per pulse. Clamped at zero.
///
/// `phase` is `sdp_sample`'s OBSERVED bound, not `sdp_interval`'s expected one. The `5` is
/// `eps_0 = eps_sec/10` inside `2*log2(1/(2*eps_0))`, not a fifth epsilon. `eps_sec` is the TOTAL
/// secrecy parameter; `eps_cor` is additive.
#[pyfunction]
pub(crate) fn sdp_length(
    n_z: f64,
    e_z: f64,
    phase: f64,
    f_ec: f64,
    eps_sec: f64,
    eps_cor: f64,
) -> PyResult<f64> {
    check_nonneg("n_z", n_z)?;
    check_err("e_z", e_z)?;
    check_err("phase", phase)?;
    check_fec("f_ec", f_ec)?;
    check_eps("eps_sec", eps_sec)?;
    check_eps("eps_cor", eps_cor)?;

    let leak = f_ec * n_z * h2(e_z);
    let budget = (2.0 / eps_cor).log2() + 2.0 * (5.0 / eps_sec).log2();
    let len = n_z * (1.0 - h2(phase)) - leak - budget;

    Ok(len.max(0.0).floor())
}

/// The CERTIFIED phase error carried into a finite key: refused. `rounds` is checked first.
#[pyfunction]
pub(crate) fn sdp_finite(rounds: f64) -> PyResult<f64> {
    check_pos("rounds", rounds)?;

    Err(PyNotImplementedError::new_err(
        "no finite key on the certified COW' phase error; use the analytic chain sdp_kato -> \
         sdp_interval -> sdp_sample -> sdp_length, which q.security.keylength reaches as \
         \"cow-vacuum\". Missing: (1) THE CONSTRAINT SET IS THE WRONG SHAPE: sdp_phase reads \
         the eight gains as equalities, finite statistics give a box of six Kato intervals and \
         four bit gains, and no monotonicity makes a corner of the box a bound. \
         (2) THE EPSILONS DO NOT COMPOSE: Li's eps_sec = 2*eps + eps_0 + 6*eps_1 + eps_2 \
         counts six bounded gains, a box programme reads ten, and no term charges the \
         solver's failure to converge. (3) THE CLAIM IS NOT IN PRINT: Seksaria & Prabhakar, \
         \"Short reach by theorem: the certifiable key rate of COW QKD\" (2026), Sec. VI D, \
         states the finite-key extension as a conditional",
    ))
}

/// Assemble the dual LMI and its strictly feasible start.
fn cow_problem(mu: f64, q0: &[f64], q1: &[f64]) -> (Lmi, Vec<f64>) {
    let g = gram(mu);
    let mut w = [0.0f64; 16];
    w[2 * 4 + 3] = 0.5;
    w[3 * 4 + 2] = 0.5;

    let mut blocks = Vec::with_capacity(3);

    for which in 0..3usize {
        let mut f = vec![0.0f64; (NVAR + 1) * 16];

        if which == 1 {
            f[..16].copy_from_slice(&w);
        }

        if which == 2 {
            for k in 0..16 {
                f[k] = -w[k];
            }
        }

        for (i, &(a, b)) in ZIDX.iter().enumerate() {
            let base = (i + 1) * 16;
            f[base + a * 4 + b] = 1.0;
            f[base + b * 4 + a] = 1.0;
        }

        if which > 0 {
            let off = 10 + 4 * (which - 1);

            for j in 0..4 {
                f[(off + j + 1) * 16 + j * 4 + j] = 1.0;
            }
        }

        blocks.push(Block { m: 4, f });
    }

    let mut c = vec![0.0f64; NVAR];

    for (i, &(a, b)) in ZIDX.iter().enumerate() {
        c[i] = if a == b { g[a * 4 + b] } else { 2.0 * g[a * 4 + b] };
    }

    for j in 0..4 {
        c[10 + j] = q0[j];
        c[14 + j] = q1[j];
    }

    // Z = I, y = 0: the identity dominates W, whose spectral norm is 1/2.
    let mut start = vec![0.0f64; NVAR];

    for j in 0..4 {
        start[j] = 1.0;
    }

    (
        Lmi {
            n: NVAR,
            c,
            blocks,
        },
        start,
    )
}

/// Worst-case phase error over every measurement consistent with the eight monitoring gains:
/// `(bound, gap, iterations)`. `q0`, `q1` as in `sdp_analytic`.
///
/// `bound` IS PROVED: the objective at a verified strictly dual-feasible point, converged or not.
/// `gap` is the last round's `nu/t`, a diagnostic. Raw, uncapped: >= 1/2 aborts.
#[pyfunction]
#[pyo3(signature = (mu, q0, q1, reltol=1e-9))]
pub(crate) fn sdp_phase(
    mu: f64,
    q0: Vec<f64>,
    q1: Vec<f64>,
    reltol: f64,
) -> PyResult<(f64, f64, usize)> {
    check_pos("mu", mu)?;
    check_pos("reltol", reltol)?;
    check_record(&q0, &q1)?;

    for j in 0..4 {
        if q0[j] <= 0.0 || q1[j] <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "every monitoring gain must be > 0; sequence {j} reads {} and {}: a zero \
                 gain leaves the constraint set no interior for the path. Use sdp_analytic, \
                 which needs none",
                q0[j], q1[j]
            )));
        }
    }

    let den = q0[2] + q1[2] + q0[3] + q1[3];
    let (lmi, start) = cow_problem(mu, &q0, &q1);
    let mut cmax = 0.0f64;

    for x in lmi.c.iter() {
        cmax = cmax.max(x.abs());
    }

    if cmax <= 0.0 {
        return Err(PyValueError::new_err(
            "the dual objective is identically zero, which no Gram matrix produces",
        ));
    }

    let scaled = Lmi {
        n: lmi.n,
        c: lmi.c.iter().map(|x| x / cmax).collect(),
        blocks: lmi.blocks,
    };
    let rep = scaled.run(&start, reltol * den / cmax);
    let gap = rep.gap * cmax;

    // Every block carries +Z, so one shift of Z lifts all three.
    let mut lowest = f64::INFINITY;
    let mut biggest = 0.0f64;

    for blk in scaled.blocks.iter() {
        let s = blk.slack(&rep.v);

        for x in s.iter() {
            biggest = biggest.max(x.abs());
        }

        lowest = lowest.min(eig_min(&s, blk.m));
    }

    if !lowest.is_finite() {
        return Err(PyValueError::new_err(format!(
            "the dual point could not be certified after {} iterations: a slack block has \
             no finite spectrum",
            rep.iters
        )));
    }

    let shift = (-lowest).max(0.0) + REPAIR * biggest.max(1.0);
    let mut value = 0.0;

    for (a, b) in lmi.c.iter().zip(rep.v.iter()) {
        value += a * b;
    }

    // Tr(shift I G) = 4 * shift for a unit-diagonal Gram.
    value += 4.0 * shift;

    let ceiling = (q0[2] * q0[3]).sqrt() + (q1[2] * q1[3]).sqrt();
    let slack = gap.abs() + 1e-9 * ceiling;

    if value < -ceiling - slack {
        return Err(PyValueError::new_err(format!(
            "no measurement reproduces this record: the certified dual value {value} sits \
             below the Cauchy-Schwarz floor {} that every feasible point respects, which \
             proves the constraint set is empty",
            -ceiling
        )));
    }

    if !value.is_finite() || value > ceiling + slack.max(1e-6 * ceiling) {
        return Err(PyValueError::new_err(format!(
            "the interior-point path did not resolve this record: it stopped after {} \
             iterations with status \"{}\" at a dual value {value} above the \
             Cauchy-Schwarz ceiling {ceiling}, so the bound certifies nothing the \
             two-by-two minors did not already",
            rep.iters, rep.status
        )));
    }

    // The closed form bounds the same worst case: a certified value above it is an unconverged run.
    let plain = sdp_analytic(mu, q0.to_vec(), q1.to_vec())?;
    let seen = 0.5 + value / den;

    if seen > plain + 1e-9 * plain.abs().max(1.0) {
        return Err(PyValueError::new_err(format!(
            "the interior-point path did not resolve this record: it stopped after {} \
             iterations with status \"{}\" at a certified {seen}, above the {plain} the \
             Cauchy-Schwarz closed form already gives. The value is a valid bound and a \
             useless one; nothing tighter than the closed form was proved here",
            rep.iters, rep.status
        )));
    }

    if !(gap <= reltol * den) {
        return Err(PyValueError::new_err(format!(
            "the interior-point path stopped after {} iterations with status \"{}\" at a \
             central-path gap of {} in phase-error units, above the requested {}. The \
             bound it reached is still valid; ask for a larger reltol to accept it",
            rep.iters,
            rep.status,
            gap / den,
            reltol
        )));
    }

    Ok((0.5 + value / den, gap / den, rep.iters))
}

/// The bare solver, for tests: `min c . v` s.t. `F_k(v) = F_k^0 + sum_i v_i F_k^i >= 0`. `dims` are
/// the block sides; `mats` concatenates each block's `1 + len(c)` row-major matrices, constant
/// first. `start` must be strictly feasible. `value` is an upper bound on the minimum, and on the
/// primal maximum of a dual.
#[pyfunction]
#[pyo3(signature = (c, dims, mats, start, gaptol=1e-9))]
pub(crate) fn sdp_lmi(
    c: Vec<f64>,
    dims: Vec<usize>,
    mats: Vec<f64>,
    start: Vec<f64>,
    gaptol: f64,
) -> PyResult<(f64, f64, usize, Vec<f64>)> {
    let n = c.len();

    if n == 0 || dims.is_empty() {
        return Err(PyValueError::new_err(
            "an LMI needs at least one variable and one block",
        ));
    }

    if start.len() != n {
        return Err(PyValueError::new_err(format!(
            "start must carry one value per variable: got {} for {n}",
            start.len()
        )));
    }

    check_pos("gaptol", gaptol)?;

    let want: usize = dims.iter().map(|m| (n + 1) * m * m).sum();

    if mats.len() != want {
        return Err(PyValueError::new_err(format!(
            "mats must hold {want} entries -- {} matrices per block, flattened row-major \
             -- got {}",
            n + 1,
            mats.len()
        )));
    }

    let mut blocks = Vec::with_capacity(dims.len());
    let mut at = 0usize;

    for &m in dims.iter() {
        if m == 0 {
            return Err(PyValueError::new_err("a block of side 0 has no interior"));
        }

        let take = (n + 1) * m * m;
        blocks.push(Block {
            m,
            f: mats[at..at + take].to_vec(),
        });
        at += take;
    }

    let lmi = Lmi { n, c, blocks };

    for blk in lmi.blocks.iter() {
        let s = blk.slack(&start);

        if chol(&s, blk.m).is_none() {
            return Err(PyValueError::new_err(
                "start is not strictly feasible: some F_k(start) is not positive definite, \
                 and the barrier is undefined there",
            ));
        }
    }

    let rep = lmi.run(&start, gaptol);
    let mut lowest = f64::INFINITY;

    for blk in lmi.blocks.iter() {
        lowest = lowest.min(eig_min(&blk.slack(&rep.v), blk.m));
    }

    if lowest < 0.0 {
        return Err(PyValueError::new_err(format!(
            "the returned point is {} short of feasibility after {} iterations with status \
             \"{}\", so its objective certifies nothing",
            -lowest, rep.iters, rep.status
        )));
    }

    let mut value = 0.0;

    for (a, b) in lmi.c.iter().zip(rep.v.iter()) {
        value += a * b;
    }

    if !value.is_finite() {
        return Err(PyValueError::new_err(
            "the objective is not finite at the returned point",
        ));
    }

    Ok((value, rep.gap, rep.iters, rep.v))
}
