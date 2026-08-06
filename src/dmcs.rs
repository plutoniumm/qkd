use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::herm::{eig, funm, inner, read_herm, Block, Certificate, Cone, Cx, Term};
use crate::numeric::{bisect, kl_bits};
use crate::std::{check_nonneg, check_pos, check_prob, check_unit, g_ent, h2, ln_fact, sqrt0};

// Discrete-modulation CV-QKD: finite coherent-state constellation, heterodyne, reverse
// reconciliation, asymptotic, collective attacks, ideal receiver.
// SNU (vacuum 1), xi at the CHANNEL INPUT as in keyrate.rs; the hbar = 1 layer's numbers double.
// Z is undetermined by the data and bounded from below; that bound is the analysis.
// Denys, Brown & Leverrier, Quantum 5, 540 (2021), arXiv:2103.13945.

/// Below three states tau's covariance stops being isotropic, which every formula here assumes.
const M_MIN: usize = 3;

/// Cap keeping the ratios lambda_k/lambda_{k+1} representable.
const M_MAX: usize = 64;

/// Amplitude ceiling; alpha^2 is the mean photon number and sets the Poisson sum length below.
const ALPHA_MAX: f64 = 100.0;

/// Amplitude floor: below it the ratios lambda_k/lambda_{k+1} leave f64 range at M_MAX.
const ALPHA_MIN: f64 = 1e-6;

/// Bona fide slack: a cancellation-heavy discriminant puts a pure input a few ulp under 1.
const PHYS_TOL: f64 = 1e-9;

/// Nodes per quadrature above `GH_ALPHA`; 4.5e-9 against 192 nodes at alpha = 2. Past ~192 the
/// `gauher` seed for the second root sends Newton into the wrong basin, and a sum-of-weights check
/// misses it (exact to 1e-15 through n = 198 with a node wrong in the seventh digit): check per
/// root. Townsend-Trogdon-Olver is the replacement.
const GH_NODES: usize = 64;

/// Nodes at or below `GH_ALPHA`; the requirement tracks alpha, not m.
const GH_FEW: usize = 16;

/// Keeps the pinned anchors (m = 8, alpha = 1) on the 64-node branch.
const GH_ALPHA: f64 = 0.4;

/// Largest order `gh_rule` may be asked for, and the largest verified: at 371 the weight
/// w = 2/pp^2 overflows to zero.
const GH_MAX: usize = 370;

const _: () = assert!(GH_NODES <= GH_MAX && GH_NODES % 2 == 0);

const _: () = assert!(GH_FEW <= GH_MAX && GH_FEW % 2 == 0);

fn check_size(m: usize) -> PyResult<()> {
    if (M_MIN..=M_MAX).contains(&m) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "m must be in [{M_MIN}, {M_MAX}] states, got {m}"
        )))
    }
}

/// An alphabet entropy in bits; inf is a continuous modulation.
fn check_bits(bits: f64) -> PyResult<()> {
    if !bits.is_nan() && bits > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "bits must be > 0 (use inf for a continuous modulation), got {bits}"
        )))
    }
}

fn check_alpha(alpha: f64) -> PyResult<()> {
    if alpha.is_finite() && (ALPHA_MIN..=ALPHA_MAX).contains(&alpha) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "alpha must be in [{ALPHA_MIN}, {ALPHA_MAX}], got {alpha}"
        )))
    }
}

/// Natural LOGS of tau's eigenvalues: the k-th is the Poisson(alpha^2) mass in residue class k
/// mod M. Not values or a DFT: the small ones underflow f64, their ratios do not.
fn log_spectrum(m: usize, alpha: f64) -> Vec<f64> {
    let mu = alpha * alpha;

    // Out to where the factorial buries the tail, never fewer than one turn around the ring.
    let top = (mu + 12.0 * mu.sqrt() + 3.0 * (m as f64) + 80.0).ceil() as usize;
    let ln_mu = mu.ln();

    // ln(mu^n / n!) by recurrence.
    let mut terms = vec![0.0_f64; top + 1];
    for n in 1..=top {
        terms[n] = terms[n - 1] + ln_mu - (n as f64).ln();
    }

    // Log-sum-exp per residue class, shifted by its largest term.
    let mut peak = vec![f64::NEG_INFINITY; m];
    for (n, t) in terms.iter().enumerate() {
        let k = n % m;
        if *t > peak[k] {
            peak[k] = *t;
        }
    }

    let mut acc = vec![0.0_f64; m];
    for (n, t) in terms.iter().enumerate() {
        acc[n % m] += (t - peak[n % m]).exp();
    }

    let mut out: Vec<f64> = (0..m).map(|k| peak[k] + acc[k].ln()).collect();

    let hi = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total = hi + out.iter().map(|v| (v - hi).exp()).sum::<f64>().ln();
    for v in out.iter_mut() {
        *v -= total;
    }

    out
}

/// `(p_n, p_{n-1})` at `z` by the normalised Hermite recurrence; the unnormalised one overflows.
fn gh_poly(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = std::f64::consts::PI.powf(-0.25);
    let mut p2 = 0.0_f64;
    for j in 1..=n {
        let p3 = p2;
        p2 = p1;
        p1 = z * (2.0 / j as f64).sqrt() * p2 - ((j - 1) as f64 / j as f64).sqrt() * p3;
    }

    (p1, p2)
}

/// Half a Sturm bound on consecutive zeros of H_n (>= pi/sqrt(2n+1) apart); the bracket's step.
fn gh_step(n: usize) -> f64 {
    0.5 * std::f64::consts::PI / (2.0 * n as f64 + 1.0).sqrt()
}

/// Bracket around the largest zero of H_n strictly below `above`. H_n carries (-1)^i between
/// roots i and i-1, so walking down in `gh_step` increments to the first sign change brackets it.
fn gh_bracket(n: usize, i: usize, above: f64) -> (f64, f64) {
    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
    let step = gh_step(n);
    let mut hi = above - step;
    let mut lo = hi - step;
    while lo > 0.0 && gh_poly(n, lo).0 * sign > 0.0 {
        hi = lo;
        lo -= step;
    }

    (lo.max(0.0), hi)
}

/// That root, by bisection.
fn gh_bisect(n: usize, i: usize, lo: f64, hi: f64) -> f64 {
    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
    let (mut a, mut b) = (lo, hi);
    for _ in 0..200 {
        let mid = 0.5 * (a + b);
        if mid <= a || mid >= b {
            break;
        }

        if gh_poly(n, mid).0 * sign > 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }

    0.5 * (a + b)
}

/// Nodes and weights of the `n`-point Gauss-Hermite rule for `integral exp(-x^2) f(x) dx`, roots
/// largest first, by Newton from Numerical Recipes' `gauher` guesses (2nd ed., Sec. 4.5).
/// `n` must be even. Newton is bit-identical below 188 and collapses by 200; `gh_bracket` gates it.
fn gh_rule(n: usize) -> (Vec<f64>, Vec<f64>) {
    let order = n as f64;
    let mut nodes = vec![0.0_f64; n];
    let mut wts = vec![0.0_f64; n];
    let mut z = 0.0_f64;
    let mut slope = 0.0_f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * order + 1.0).sqrt() - 1.85575 * (2.0 * order + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * order.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * nodes[0],
            3 => 1.91 * z - 0.91 * nodes[1],
            _ => 2.0 * z - nodes[i - 2],
        };

        let mut done = false;
        for _ in 0..12 {
            let (p1, p2) = gh_poly(n, z);
            slope = (2.0 * order).sqrt() * p2;
            let step = p1 / slope;
            z -= step;
            if step.abs() <= 3e-14 {
                done = true;
                break;
            }
        }

        // Search capped by the last root found, or for the first by |x| < sqrt(2n + 1).
        let above = if i == 0 {
            (2.0 * order + 1.0).sqrt()
        } else {
            nodes[i - 1]
        };
        let (lo, hi) = gh_bracket(n, i, above);
        if !done || !(z > lo && z < above) {
            z = gh_bisect(n, i, lo, hi);
            slope = (2.0 * order).sqrt() * gh_poly(n, z).1;
        }

        nodes[i] = z;
        nodes[n - 1 - i] = -z;
        wts[i] = 2.0 / (slope * slope);
        wts[n - 1 - i] = wts[i];
    }

    (nodes, wts)
}

/// The rule `mutual` runs at this amplitude, built once per branch; the gate is alpha alone.
fn gauss_hermite(alpha: f64) -> &'static (Vec<f64>, Vec<f64>) {
    static FEW: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    static FULL: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();

    if alpha <= GH_ALPHA {
        FEW.get_or_init(|| gh_rule(GH_FEW))
    } else {
        FULL.get_or_init(|| gh_rule(GH_NODES))
    }
}

/// Gaussian-channel mutual information at the same SNR, log2(1 + T*V_A/(2 + T*xi)) -- the value
/// the source uses for every modulation. The 2 is the mode's vacuum plus Bob's balanced splitter.
fn gauss_info(va: f64, t: f64, xi: f64) -> f64 {
    (1.0 + t * va / (2.0 + t * xi)).log2()
}

/// I(X;Y) in bits/symbol for M-PSK, Bob heterodyning: Y = sqrt(T)*s_k + N(0, 2 + T*xi) per
/// quadrature, s_k = 2*Re(alpha_k), so I = log2(m) - E_n[ log2 sum_j exp(-(|d_j + n|^2 - |n|^2)
/// / (2*sigma^2)) ], d_j the offset to symbol j. One sent symbol suffices: ring, isotropic.
fn mutual(m: usize, alpha: f64, t: f64, xi: f64) -> f64 {
    let s2 = 2.0 + t * xi;
    let step = 2.0 * std::f64::consts::PI / (m as f64);
    let reach = 2.0 * alpha * t.sqrt();

    // Offsets from the sent symbol (index 0, on the real axis).
    let mut off = vec![(0.0_f64, 0.0_f64); m];
    let mut base = vec![0.0_f64; m];
    let mut amp = vec![0.0_f64; m];
    for (j, o) in off.iter_mut().enumerate() {
        let (s, c) = (step * (j as f64)).sin_cos();
        *o = (reach * (1.0 - c), -reach * s);
        base[j] = -(o.0 * o.0 + o.1 * o.1) / (2.0 * s2);
        amp[j] = base[j].exp();
    }

    let (nodes, wts) = gauss_hermite(alpha);
    let n = nodes.len();
    let half = n / 2;
    let spread = (2.0 * s2).sqrt();

    // One node's log factor for one quadrature offset, recomputed in the fallback rather than
    // tabulated: the same operations in the same order make the two paths bit-identical.
    let lg = |o: f64, node: f64| -o * spread * node / s2;

    // Per-quadrature exponentials, indexed [quadrature][symbol][node].
    let mut fact = vec![0.0_f64; 2 * m * n];
    for j in 0..m {
        for (i, &node) in nodes.iter().enumerate() {
            fact[j * n + i] = lg(off[j].0, node).exp();
            fact[m * n + j * n + i] = lg(off[j].1, node).exp();
        }
    }

    let mut acc = 0.0;
    for (a, &wa) in wts.iter().enumerate() {
        for b in 0..half {
            let mut sum = 0.0;
            for j in 0..m {
                sum += amp[j] * fact[j * n + a] * fact[m * n + j * n + b];
            }

            let bits = if sum.is_finite() && sum > 0.0 {
                sum.log2()
            } else {
                let mut hi = f64::NEG_INFINITY;
                for j in 0..m {
                    let e = base[j] + lg(off[j].0, nodes[a]) + lg(off[j].1, nodes[b]);
                    if e > hi {
                        hi = e;
                    }
                }

                let mut tail = 0.0;
                for j in 0..m {
                    tail += (base[j] + lg(off[j].0, nodes[a]) + lg(off[j].1, nodes[b]) - hi).exp();
                }

                (hi + tail.ln()) * std::f64::consts::LOG2_E
            };

            acc += 2.0 * wa * wts[b] * bits;
        }
    }

    (m as f64).log2() - acc / std::f64::consts::PI
}

struct Moments {
    va: f64,
    z_lin: f64,
    w: f64,
    z_gauss: f64,
}

/// The three constellation numbers plus the Gaussian value they are measured against, over
/// q_k = sqrt(lambda_k/lambda_{k+1}) drawn with probability lambda_k. The prefactors differ:
/// `z_lin` = va*E[q] = 2*tr(tau^{1/2} a tau^{1/2} a^dag), and `w` = alpha^2*Var[q] = (va/2)*Var[q],
/// Eq. (13)'s constellation-averaged variance of tau^{1/2} a tau^{-1/2}, zero for a Gaussian.
/// `dm_rate` pairs them as Z* = sqrt(T)*(z_lin - sqrt(2*xi*w)), arXiv:2103.13945v3 Eq. (25).
fn moments(m: usize, alpha: f64) -> PyResult<Moments> {
    let ln_lam = log_spectrum(m, alpha);
    let lam: Vec<f64> = ln_lam.iter().map(|v| v.exp()).collect();
    let ratio: Vec<f64> = (0..m)
        .map(|k| (0.5 * (ln_lam[k] - ln_lam[(k + 1) % m])).exp())
        .collect();

    let mean = (0..m).map(|k| lam[k] * ratio[k]).sum::<f64>();
    let var = (0..m)
        .map(|k| lam[k] * (ratio[k] - mean) * (ratio[k] - mean))
        .sum::<f64>();

    let va = 2.0 * alpha * alpha;
    let out = Moments {
        va,
        z_lin: va * mean,
        w: alpha * alpha * var,
        z_gauss: (va * va + 2.0 * va).sqrt(),
    };

    if !out.z_lin.is_finite() || !out.w.is_finite() {
        return Err(PyValueError::new_err(format!(
            "constellation moments overflowed at m = {m}, alpha = {alpha}: \
             the eigenvalue ratios are outside f64 range. Raise alpha or lower m"
        )));
    }

    Ok(out)
}

struct DmPoint {
    i_ab: f64,
    chi_be: f64,
    key: f64,
}

/// One point of the asymptotic reverse-reconciliation rate, K = beta*I_AB - chi_BE, on the matrix
/// keyrate.rs builds but with `z` free instead of sqrt(T*(V^2 - 1)). Eve purifies AB; the ideal
/// receiver collapses Lodewyck's nu3, nu4, nu5 to one eigenvalue. `i_ab` is alphabet-dependent.
fn dm_core(va: f64, t: f64, xi: f64, z: f64, beta: f64, i_ab: f64) -> PyResult<DmPoint> {
    let a = va + 1.0;
    let b = 1.0 + t * (va + xi);
    let ab = a * b - z * z;
    let delta = a * a + b * b - 2.0 * z * z;
    let disc = sqrt0(delta * delta - 4.0 * ab * ab);
    let nu1 = sqrt0((delta + disc) / 2.0);

    // The SMALL root from nu1*nu2 = |ab|, never from (delta - disc)/2: those agree to every
    // digit once Bob's variance dominates Alice's, and the subtraction then returns nu2 = 0.
    let nu2 = if nu1 > 0.0 { ab.abs() / nu1 } else { 0.0 };
    let nu3 = a - z * z / (b + 1.0);

    // An arithmetic blow-up fails SAFE: the input is absurd, not malformed.
    if !nu1.is_finite() || !nu2.is_finite() || !nu3.is_finite() {
        return Ok(DmPoint {
            i_ab: 0.0,
            chi_be: f64::INFINITY,
            key: f64::NEG_INFINITY,
        });
    }

    // nu2 below 1 lands in the G(x <= 0) branch and understates Eve's information.
    if nu2 < 1.0 - PHYS_TOL || nu3 < 1.0 - PHYS_TOL {
        return Err(PyValueError::new_err(format!(
            "z = {z} makes the covariance matrix unphysical at va = {va}, t = {t}, \
             xi = {xi} (symplectic eigenvalues {nu2}, {nu3} below the vacuum)"
        )));
    }

    let chi_be = g_ent((nu1 - 1.0) / 2.0) + g_ent((nu2 - 1.0) / 2.0) - g_ent((nu3 - 1.0) / 2.0);

    // Fail SAFE on overflow, as in keyrate.rs: the arithmetic returns chi_be = 0 at absurd inputs.
    if !chi_be.is_finite() || !i_ab.is_finite() {
        return Ok(DmPoint {
            i_ab: if i_ab.is_finite() { i_ab } else { 0.0 },
            chi_be: f64::INFINITY,
            key: f64::NEG_INFINITY,
        });
    }

    Ok(DmPoint {
        i_ab,
        chi_be,
        key: beta * i_ab - chi_be,
    })
}

/// The `m` M-PSK amplitudes as `(re, im)`, each with probability 1/m, the first on the real axis.
/// Photon amplitudes, not quadratures: the mean quadratures are (2*Re alpha, 2*Im alpha) in SNU.
#[pyfunction]
pub(crate) fn dm_states(m: usize, alpha: f64) -> PyResult<Vec<(f64, f64)>> {
    check_size(m)?;
    check_alpha(alpha)?;

    let step = 2.0 * std::f64::consts::PI / (m as f64);

    Ok((0..m)
        .map(|k| {
            let (s, c) = (step * (k as f64)).sin_cos();

            (alpha * c, alpha * s)
        })
        .collect())
}

/// `(va, z_lin, w, z_gauss)` at the CHANNEL INPUT plane; `va` = 2*alpha^2, Alice's marginal
/// (1 + va)*I_2. z_gauss - z_lin is the penalty at zero excess noise, `w` scales xi in `dm_rate`.
#[pyfunction]
pub(crate) fn dm_moments(m: usize, alpha: f64) -> PyResult<(f64, f64, f64, f64)> {
    check_size(m)?;
    check_alpha(alpha)?;

    let mo = moments(m, alpha)?;

    Ok((mo.va, mo.z_lin, mo.w, mo.z_gauss))
}

/// Asymptotic reverse reconciliation `(i_ab, chi_be, key)` in bits/symbol with `z` given, at the
/// CHANNEL OUTPUT plane and already carrying sqrt(T). `key` is NOT clamped; `bits` is Alice's
/// alphabet entropy, capping the Gaussian I_AB.
#[pyfunction]
pub(crate) fn dm_holevo(
    va: f64,
    t: f64,
    xi: f64,
    z: f64,
    bits: f64,
    beta: f64,
) -> PyResult<(f64, f64, f64)> {
    check_pos("va", va)?;
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;
    check_nonneg("z", z)?;
    check_bits(bits)?;
    check_prob("beta", beta)?;

    let p = dm_core(va, t, xi, z, beta, gauss_info(va, t, xi).min(bits))?;

    Ok((p.i_ab, p.chi_be, p.key))
}

/// `(exact, gauss)` in bits/symbol: I(X;Y) for the m-state constellation, and the log2(1 + SNR)
/// the literature substitutes. The gap is under a percent where these run, 60% at 8-PSK alpha = 1.
/// `t`/`xi` have the receiver folded in.
#[pyfunction]
pub(crate) fn dm_info(m: usize, alpha: f64, t: f64, xi: f64) -> PyResult<(f64, f64)> {
    check_size(m)?;
    check_alpha(alpha)?;
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;

    Ok((
        mutual(m, alpha, t, xi),
        gauss_info(2.0 * alpha * alpha, t, xi),
    ))
}

/// Asymptotic M-PSK key rate `(i_ab, chi_be, key, z_star)` in bits/symbol; `z_star` =
/// sqrt(T)*(z_lin - sqrt(2*xi*w)) clamped at zero (chi_BE goes as z^2), `i_ab` capped at log2(m).
/// UNTRUSTED ONLY: eta, vel enter as T -> eta*T, xi -> xi + 2*vel/(eta*T) and nowhere else.
#[pyfunction]
pub(crate) fn dm_rate(
    m: usize,
    alpha: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    beta: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_size(m)?;
    check_alpha(alpha)?;
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;
    check_unit("eta", eta)?;
    check_nonneg("vel", vel)?;
    check_prob("beta", beta)?;

    let t_eff = eta * t;
    let xi_eff = xi + 2.0 * vel / t_eff;
    let mo = moments(m, alpha)?;
    let z_star = (t_eff.sqrt() * (mo.z_lin - (2.0 * xi_eff * mo.w).sqrt())).max(0.0);
    let i_ab = mutual(m, alpha, t_eff, xi_eff).min((m as f64).log2());
    let p = dm_core(mo.va, t_eff, xi_eff, z_star, beta, i_ab)?;

    Ok((p.i_ab, p.chi_be, p.key, z_star))
}

// WINICK-LUTKENHAUS-COLES FOR DISCRETE MODULATION
//
// Above is the ANALYTIC bound (Denys, Brown & Leverrier; the route of Ghorai, Grangier, Diamanti &
// Leverrier, PRX 9, 021059 (2019), arXiv:1902.01317). Below is the numerical proof of Lin,
// Upadhyaya & Lutkenhaus, PRX 9, 041064 (2019), arXiv:1905.10896 -- protocol 2, heterodyne,
// reverse reconciliation -- on the two-step method of Winick, Lutkenhaus & Coles, Quantum 2, 77
// (2018), arXiv:1710.05511.
//
// Step 1 (Frank-Wolfe) is an UPPER bound, `upper`, never the key. Step 2 (WLC Theorem 3) is the
// proof, `bound`; every `herm::Cone` iterate is strictly feasible, so an early stop is looser.
//
// Rigorous GIVEN LUL19 Sec. III B's photon-number cutoff, an assumption in their own words. The
// dimension reduction removing it -- Upadhyaya, van Himbeeck, Lin & Lutkenhaus, PRX Quantum 2,
// 020325 (2021), arXiv:2101.05799 -- is NOT implemented; `dm_secure` returns its cost as `viol`.
//
// LUL19 Sec. III C's q = (a^dag + a)/sqrt(2), p = i(a^dag - a)/sqrt(2) are qkd's, so every
// VARIANCE here is half the SNU number `dm_rate` and `keyrate.rs` carry: (1 + eta*xi)/2 against
// LUL19 Sec. IV A's 1 + T*xi. `xi`, a variance RATIO less one, is the same number in both.

/// Largest photon-number cutoff accepted. Cost is cubic in `m * (nc + 1)` in the solver and in
/// `m^2 * (nc + 1)` in the perturbation bookkeeping.
const NC_MAX: usize = 30;

/// Largest constellation the numerical proof accepts. The matrices are `m * (nc + 1)` on a
/// side and the Kraus image is `m` times that, so eight is already 704 in the entropy terms.
const SDP_M_MAX: usize = 8;

/// Floor added to the diagonal, relative to the trace, when a state's Cholesky fails.
const JITTER: f64 = 1e-13;

/// Golden-section evaluations in the Frank-Wolfe line search; twelve reach 3e-3 of the segment.
const SEARCH_STEPS: usize = 12;

/// Relative improvement in `f_eps` below which Frank-Wolfe stops early.
const FW_TOL: f64 = 1e-10;

/// Shortest step the relaxed walk's line search will consider, as a base-ten logarithm.
/// Below 1e-9 of the segment the iterate does not move in f64.
const LOG_STEP: f64 = -9.0;

/// Margin added to the repaired dual point, in units of the largest slack entry: a Cholesky
/// accepts a few epsilons of negativity. `sdp.rs::REPAIR` is the same charge for the real-symmetric solver.
const REPAIR: f64 = 64.0 * f64::EPSILON;

// Shared by `certify` and `dm_relaxed`; the checks that raise them differ on purpose.

const NO_START: &str = "the starting state has no Cholesky factor even after a diagonal floor, so the \
                        relative entropy cannot be evaluated at it";

const NO_GRADIENT: &str = "the Frank-Wolfe iterate left the positive cone, so no gradient exists there \
                           and step 2 has nothing to linearise about";

const NOT_FINITE: &str = "the certified bound is not finite, which no feasible dual point produces";

/// No key map can hand out more than `log2(m)` bits per surviving round.
fn over_cap(bound: f64, pass: f64, m: usize) -> PyErr {
    PyValueError::new_err(format!(
        "the certified bound {bound} exceeds the {} bits the key map can carry over a \
         sifting probability of {pass}, which no relative entropy produces",
        pass * (m as f64).log2()
    ))
}

/// erfc(x) for x >= 0: positive-term series below 3, Lentz continued fraction above.
fn erfc(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    if x < 3.0 {
        // erf(x) = (2/sqrt(pi)) e^{-x^2} sum_n 2^n x^{2n+1} / (1*3*...*(2n+1))
        let mut term = x;
        let mut sum = x;

        for n in 0..200 {
            term *= 2.0 * x * x / (2.0 * (n as f64) + 3.0);
            sum += term;

            if term <= 1e-18 * sum {
                break;
            }
        }

        let erf = 2.0 * (-x * x).exp() * sum / std::f64::consts::PI.sqrt();

        return (1.0 - erf).max(0.0);
    }

    // erfc(x) = e^{-x^2}/(x sqrt(pi)) * 1/(1 + (1/2)/(x^2 + (2/2)/(1 + ...))) by Lentz.
    let mut f = 1e-300f64;
    let mut c = f;
    let mut d = 0.0f64;

    for i in 0..300 {
        let (a, b) = if i == 0 {
            (1.0, x * x + 0.5)
        } else {
            (-(i as f64) * ((i as f64) - 0.5), x * x + 0.5 + 2.0 * (i as f64))
        };

        d = b + a * d;

        if d.abs() < 1e-300 {
            d = 1e-300;
        }

        c = b + a / c;

        if c.abs() < 1e-300 {
            c = 1e-300;
        }

        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-17 {
            break;
        }
    }

    x * (-x * x).exp() * f / std::f64::consts::PI.sqrt()
}

/// `Gamma(k/2 + 1, x)` for `k = 0 ..= top`, the upper incomplete gamma. Seeded at orders 1 and
/// 3/2 and raised by `Gamma(s + 1, x) = s Gamma(s, x) + x^s e^{-x}`, stable upward.
fn gamma_ladder(top: usize, x: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; top + 1];
    let ex = (-x).exp();
    out[0] = ex;

    if top >= 1 {
        // Gamma(1/2, x) = sqrt(pi) erfc(sqrt(x)); Gamma(3/2, x) = (1/2)Gamma(1/2, x) + sqrt(x) e^{-x}.
        let half = std::f64::consts::PI.sqrt() * erfc(x.sqrt());
        out[1] = 0.5 * half + x.sqrt() * ex;
    }

    for k in 2..=top {
        let s = 0.5 * (k as f64);
        out[k] = s * out[k - 2] + x.powf(s) * ex;
    }

    out
}

/// The region operator `R_z` on Bob's truncated Fock space, LUL19 Sec. III D generalised to an
/// `m`-sector key map. In polar coordinates the integral factorises: the radial part is an
/// upper incomplete gamma, the angular part over a sector of half-width `w` centred on
/// `2 pi z / m` is `e^{i (n - n') 2 pi z / m} * 2 sin((n - n') w) / (n - n')`. `cut` is LUL19's
/// `Delta_a` and `phase` their `Delta_p`; at both zero and one sector this is the identity.
fn region(m: usize, nc: usize, z: usize, cut: f64, phase: f64) -> Cx {
    let nb = nc + 1;
    let mut out = Cx::zeros(nb);
    let lf = ln_fact(nb);
    let gl = gamma_ladder(2 * nc, cut * cut);
    let w = std::f64::consts::PI / (m as f64) - phase;
    let centre = 2.0 * std::f64::consts::PI * (z as f64) / (m as f64);

    for n in 0..nb {
        for j in 0..nb {
            let k = (n as isize) - (j as isize);
            let ang = if k == 0 {
                2.0 * w
            } else {
                2.0 * ((k as f64) * w).sin() / (k as f64)
            };
            let mag = gl[n + j] * (-0.5 * (lf[n] + lf[j])).exp() / (2.0 * std::f64::consts::PI);
            let (s, c) = ((k as f64) * centre).sin_cos();
            out.set(n, j, mag * ang * c, mag * ang * s);
        }
    }

    out.symmetrise();

    out
}

/// `I_A tensor xb` as a dense `m * nb` matrix.
fn lift(m: usize, xb: &Cx) -> Cx {
    let nb = xb.n;
    let mut out = Cx::zeros(m * nb);

    for x in 0..m {
        for i in 0..nb {
            for j in 0..nb {
                let (r, s) = xb.get(i, j);
                out.set(x * nb + i, x * nb + j, r, s);
            }
        }
    }

    out
}

/// Positive square root of a Hermitian positive-semidefinite matrix, negative round-off
/// clamped to zero.
fn psd_root(a: &Cx) -> Cx {
    let (vals, vecs) = eig(a);

    funm(&vals, &vecs, |x| if x > 0.0 { x.sqrt() } else { 0.0 })
}

/// Cholesky with a diagonal jitter retry; the Frank-Wolfe iterate can sit on the cone boundary.
fn floor_chol(a: &Cx) -> Option<Cx> {
    if let Some(l) = a.chol() {
        return Some(l);
    }

    let base = a.trace().abs().max(1.0);

    for step in 1..8 {
        let mut trial = a.copy();
        let bump = JITTER * base * (100.0f64).powi(step);

        for i in 0..trial.n {
            trial.add(i, i, bump, 0.0);
        }

        if let Some(l) = trial.chol() {
            return Some(l);
        }
    }

    None
}

/// The whole postprocessing side of the problem: the region operators, their roots, and the
/// perturbed objective `f_eps` with its gradient.
struct Wlc {
    m: usize,
    d: usize,
    dp: f64,
    eps: f64,
    root: Vec<Cx>,
    total: Cx,
}

/// `f_eps(rho)`, `Tr G(rho)`, and the gradient when it was asked for.
struct Point {
    value: f64,
    pass: f64,
    grad: Cx,
}

impl Wlc {
    fn new(m: usize, nc: usize, rs: &[Cx], eps: f64) -> Wlc {
        let nb = nc + 1;
        let d = m * nb;
        let mut sum = Cx::zeros(nb);
        let mut root = Vec::with_capacity(m);

        for rz in rs.iter() {
            for k in 0..nb * nb {
                sum.re[k] += rz.re[k];
                sum.im[k] += rz.im[k];
            }

            root.push(lift(m, &psd_root(rz)));
        }

        Wlc {
            m,
            d,
            dp: ((m * d) as f64),
            eps,
            root,
            total: lift(m, &sum),
        }
    }

    /// `f_eps` and, when `want`, `grad f_eps`. Nothing of side `m * d` is formed: with
    /// `rho = L L^dag`, `G(rho)`'s nonzero spectrum is that of `L^dag (I tensor R) L`, and the 0/0
    /// of a literal `K^dag log(G(rho)) K` is removed analytically as `g(lambda)/lambda`.
    fn eval(&self, rho: &Cx, want: bool) -> Option<Point> {
        let ln2 = std::f64::consts::LN_2;
        let l = floor_chol(rho)?;
        let mut x = l.dag().mul(&self.total).mul(&l);
        x.symmetrise();

        let (lam, wv) = eig(&x);
        let pass: f64 = lam.iter().map(|v| v.max(0.0)).sum();
        let c = self.eps * pass / self.dp;

        if !(c > 0.0) {
            return None;
        }

        let rest = self.dp - (self.d as f64);
        let mut value = rest * c * c.log2();
        let mut trace_sig = rest * c.log2();

        for v in lam.iter() {
            let s = (1.0 - self.eps) * v.max(0.0) + c;
            value += s * s.log2();
            trace_sig += s.log2();
        }

        let mut trace_zed = 0.0;
        let mut logs = Vec::with_capacity(self.m);

        // `Cx::mul` skips zeros of its LEFT operand only, so block-diagonal `root[z]` goes left
        // twice: `X R = (R X^dag)^dag`, the outer dagger absorbed by `symmetrise`.
        for z in 0..self.m {
            let mut bz = self.root[z].mul(&self.root[z].mul(rho).dag());
            bz.symmetrise();

            let (mu, mv) = eig(&bz);

            for v in mu.iter() {
                let t = (1.0 - self.eps) * v.max(0.0) + c;
                value -= t * t.log2();
                trace_zed += t.log2();
            }

            if want {
                logs.push(funm(&mu, &mv, |v| {
                    ((1.0 - self.eps) * v.max(0.0) + c).log2()
                }));
            }
        }

        if !want {
            return Some(Point {
                value,
                pass,
                grad: Cx::zeros(0),
            });
        }

        // K^dag log(sigma_eps) K, without ever forming sigma_eps.
        let y = self.total.mul(&l).mul(&wv);
        let mut ratio = vec![0.0f64; self.d];

        for (i, v) in lam.iter().enumerate() {
            let lo = v.max(0.0);
            let u = (1.0 - self.eps) * lo / c;
            ratio[i] = if u < 1e-8 {
                (1.0 - self.eps) / (c * ln2)
            } else {
                u.ln_1p() / (lo * ln2)
            };
        }

        let mut scaled = Cx::zeros(self.d);

        for j in 0..self.d {
            for i in 0..self.d {
                let (r, s) = y.get(i, j);
                scaled.set(i, j, r * ratio[j], s * ratio[j]);
            }
        }

        let mut grad = scaled.mul(&y.dag());
        grad.symmetrise();

        let shift = c.log2();

        for k in 0..self.d * self.d {
            grad.re[k] += shift * self.total.re[k];
            grad.im[k] += shift * self.total.im[k];
        }

        for (z, lz) in logs.iter().enumerate() {
            let back = self.root[z].mul(&self.root[z].mul(lz).dag());

            for k in 0..self.d * self.d {
                grad.re[k] -= back.re[k];
                grad.im[k] -= back.im[k];
            }
        }

        // The depolarising leg of G_eps: its adjoint is (1 - eps) X + (eps/d') Tr(X) I, and
        // K^dag I K is the same (I tensor R) the first term already carries.
        let pull = self.eps * (trace_sig - trace_zed) / self.dp;

        for k in 0..self.d * self.d {
            grad.re[k] = (1.0 - self.eps) * grad.re[k] + pull * self.total.re[k];
            grad.im[k] = (1.0 - self.eps) * grad.im[k] + pull * self.total.im[k];
        }

        grad.symmetrise();

        Some(Point { value, pass, grad })
    }

    /// `zeta_eps = 2 eps (d' - 1) log(d' / [eps (d' - 1)])`, the continuity price of the
    /// epsilon-perturbation named in Winick-Lutkenhaus-Coles Theorem 2, in bits (log2: conservative).
    fn zeta(&self) -> f64 {
        let n = self.dp - 1.0;

        2.0 * self.eps * n * (self.dp / (self.eps * n)).log2()
    }
}

/// The four constraint observables on Bob's truncated Fock space, LUL19 Sec. III C:
/// `q = (a^dag + a)/sqrt(2)`, `p = i(a^dag - a)/sqrt(2)`, `n = a^dag a` and
/// `d = a^2 + (a^dag)^2`. All four are banded, keeping the Newton Hessian free.
fn quadratures(nb: usize) -> Vec<Cx> {
    let mut q = Cx::zeros(nb);
    let mut p = Cx::zeros(nb);
    let mut num = Cx::zeros(nb);
    let mut dd = Cx::zeros(nb);
    let root = std::f64::consts::SQRT_2;

    for n in 0..nb {
        num.set(n, n, n as f64, 0.0);

        if n + 1 < nb {
            let w = ((n + 1) as f64).sqrt() / root;
            q.set(n, n + 1, w, 0.0);
            q.set(n + 1, n, w, 0.0);
            p.set(n, n + 1, 0.0, -w);
            p.set(n + 1, n, 0.0, w);
        }

        if n + 2 < nb {
            let w = (((n + 1) * (n + 2)) as f64).sqrt();
            dd.set(n, n + 2, w, 0.0);
            dd.set(n + 2, n, w, 0.0);
        }
    }

    vec![q, p, num, dd]
}

// TRUSTED DETECTOR NOISE
//
// Lin & Lutkenhaus, Phys. Rev. Applied 14, 064030 (2020), arXiv:2006.06166. The calibrated
// receiver goes in Bob's POVM, not the channel. LL20 Sec. III B's noisy heterodyne element:
//
//     G_y = (1 / (eta_d pi)) D(y / sqrt(eta_d)) rho_th(nbar) D^dag(y / sqrt(eta_d)),
//
// nbar = (1 - eta_d + v_el) / eta_d. Region operators and observables both change; their Fock
// elements carry an associated Laguerre polynomial (LL20 App. B 1, after Mollow & Glauber),
// expanded here into UPPER INCOMPLETE gammas so the radial postselection Delta_a stays closed-form
// where LL20 App. B 1 subtracts a numerical integral.
//
// LL20 Sec. IV B's basis {q, p, n + d/2 + 1, n - d/2 + 1}: same span, same key rate;
// `noisy_moments` returns it in the old order.

/// LL20 Sec. III A's noisy heterodyne receiver, held by BOB: quantum efficiency `eta` and
/// electronic noise `vel` in shot-noise units, both calibrated. `nbar` is the mean photon number
/// of the thermal state every POVM element is a displaced copy of.
#[derive(Clone, Copy)]
struct Detector {
    eta: f64,
    vel: f64,
    nbar: f64,
}

impl Detector {
    fn new(eta: f64, vel: f64) -> Detector {
        Detector {
            eta,
            vel,
            nbar: (1.0 - eta + vel) / eta,
        }
    }
}

/// `ln` of the one radial integral every trusted operator reduces to,
///
///     Integral_{cut}^{inf} r^e exp(-r^2 / [eta_d (1 + nbar)])
///                          L_m^{(d)}(-r^2 / [eta_d nbar (1 + nbar)]) dr,
///
/// stripped of the `[eta_d (1 + nbar)]^{(e + 1) / 2} / 2` that `weight` puts back. Term by term
/// the Laguerre polynomial makes it `Sum_k binom(m + d, m - k) nbar^{m - k} / k!
/// * Gamma((e + 1) / 2 + k, x)`, every summand positive, so the underflowing `nbar^m` never forms.
/// `gl` is `gamma_ladder(3 nc + 2, x)`, read at `e - 1 + 2k`; `x` carries the postselection radius.
fn ln_radial(m: usize, d: usize, e: usize, nbar: f64, gl: &[f64], lf: &[f64]) -> f64 {
    let ln_nb = nbar.ln();
    let mut terms = Vec::with_capacity(m + 1);
    let mut peak = f64::NEG_INFINITY;

    for k in 0..=m {
        let g = gl[e - 1 + 2 * k];

        if !(g > 0.0) {
            continue;
        }

        let t = lf[m + d] - lf[m - k] - lf[d + k] - lf[k] + ((m - k) as f64) * ln_nb + g.ln();

        if t > peak {
            peak = t;
        }

        terms.push(t);
    }

    if !peak.is_finite() {
        return f64::NEG_INFINITY;
    }

    peak + terms.iter().map(|t| (t - peak).exp()).sum::<f64>().ln()
}

/// One matrix element's magnitude for `n >= m`: LL20 App. B 1's `C_{m,n}` prefactor times the
/// radial integral, with `eta_d` cancelling between the two wherever `e = d + 1`. `e` is the
/// power of `r` the operator's own weight contributes -- `d + 1` for a region operator, `d + 2`
/// for a first moment, `d + 3` for a second. The lower triangle is the conjugate.
fn weight(m: usize, n: usize, e: usize, det: &Detector, gl: &[f64], lf: &[f64]) -> f64 {
    let d = n - m;
    let rad = ln_radial(m, d, e, det.nbar, gl, lf);

    if !rad.is_finite() {
        return 0.0;
    }

    let ex = e as f64;
    let pre = 0.5 * (lf[m] - lf[n])
        + 0.5 * (ex - (d as f64) - 1.0) * det.eta.ln()
        + (0.5 * (ex + 1.0) - (n as f64) - 1.0) * (1.0 + det.nbar).ln();

    (pre + rad - (2.0 * std::f64::consts::PI).ln()).exp()
}

/// LL20's region operator `R_z = Integral_{A_z} G_y d^2 y` for the noisy receiver -- `region`'s
/// trusted counterpart, same sectors, `cut` and `phase`. The angular factor is `region`'s; the
/// detector enters through the radial one alone.
fn noisy_region(m: usize, nc: usize, z: usize, cut: f64, phase: f64, det: &Detector) -> Cx {
    let nb = nc + 1;
    let mut out = Cx::zeros(nb);
    let lf = ln_fact(2 * nc + 2);
    let gl = gamma_ladder(3 * nc + 2, cut * cut / (det.eta * (1.0 + det.nbar)));
    let w = std::f64::consts::PI / (m as f64) - phase;
    let centre = 2.0 * std::f64::consts::PI * (z as f64) / (m as f64);

    for n in 0..nb {
        for j in n..nb {
            let k = j - n;
            let ang = if k == 0 {
                2.0 * w
            } else {
                2.0 * ((k as f64) * w).sin() / (k as f64)
            };
            let mag = weight(n, j, k + 1, det, &gl, &lf) * ang;
            let (s, c) = ((k as f64) * centre).sin_cos();

            out.set(n, j, mag * c, -mag * s);
            out.set(j, n, mag * c, mag * s);
        }
    }

    out.symmetrise();

    out
}

/// LL20 Sec. IV B's four constraint observables for the noisy receiver, in the order
/// `quadratures` returns `q`, `p`, `n`, `d`: `F_Q`, `F_P`, `S_Q`, `S_P`, each
/// `Integral f(y, y*) G_y d^2 y` over the WHOLE outcome plane rather than one sector. At an
/// ideal receiver they are `q`, `p`, `q^2 + 1/2` and `p^2 + 1/2`.
fn noisy_moments(nc: usize, det: &Detector) -> Vec<Cx> {
    let nb = nc + 1;
    let lf = ln_fact(2 * nc + 2);
    let gl = gamma_ladder(3 * nc + 2, 0.0);
    let first = std::f64::consts::SQRT_2 * std::f64::consts::PI;
    let mut fq = Cx::zeros(nb);
    let mut fp = Cx::zeros(nb);
    let mut sq = Cx::zeros(nb);
    let mut sp = Cx::zeros(nb);

    for n in 0..nb {
        // f = 2 Re(y)^2 and 2 Im(y)^2 share their angle-independent half, hence one diagonal.
        let dia = 2.0 * std::f64::consts::PI * weight(n, n, 3, det, &gl, &lf);

        sq.set(n, n, dia, 0.0);
        sp.set(n, n, dia, 0.0);

        if n + 1 < nb {
            // (y + y*)/sqrt(2) and i(y* - y)/sqrt(2) are one harmonic, so |m - n| = 1 alone.
            let v = first * weight(n, n + 1, 3, det, &gl, &lf);

            fq.set(n, n + 1, v, 0.0);
            fq.set(n + 1, n, v, 0.0);
            fp.set(n, n + 1, 0.0, -v);
            fp.set(n + 1, n, 0.0, v);
        }

        if n + 2 < nb {
            // The second harmonic of those squares, equal and OPPOSITE between q and p.
            let v = std::f64::consts::PI * weight(n, n + 2, 5, det, &gl, &lf);

            sq.set(n, n + 2, v, 0.0);
            sq.set(n + 2, n, v, 0.0);
            sp.set(n, n + 2, -v, 0.0);
            sp.set(n + 2, n, -v, 0.0);
        }
    }

    vec![fq, fp, sq, sp]
}

/// The `m` region operators, ideal or trusted according to `det`.
fn regions(m: usize, nc: usize, cut: f64, phase: f64, det: Option<Detector>) -> Vec<Cx> {
    (0..m)
        .map(|z| match det {
            Some(d) => noisy_region(m, nc, z, cut, phase, &d),
            None => region(m, nc, z, cut, phase),
        })
        .collect()
}

/// The honest bipartite state after a phase-invariant Gaussian channel, truncated at `nc`: a
/// pure-loss stage then a phase-insensitive amplifier, `E = Amp_G . Att_{eta/G}` with
/// `G = 1 + eta xi / 2`, reproducing LUL19 Sec. IV A's displaced thermal state of variance
/// `(1 + eta xi)/2`, in closed form in the photon-number basis.
fn honest(m: usize, alpha: f64, eta: f64, xi: f64, nc: usize) -> Cx {
    let nb = nc + 1;
    let d = m * nb;
    let mut out = Cx::zeros(d);
    let lf = ln_fact(nb + 1);
    let gain = 1.0 + 0.5 * eta * xi;
    let att = eta / gain;
    let kappa = 1.0 - 1.0 / gain;
    let amp = (att / gain).sqrt() * alpha;
    let step = 2.0 * std::f64::consts::PI / (m as f64);

    // exp(-kappa(|a|^2 + |b|^2)/2 - (|c|^2 + |d|^2)/2), all four moduli equal on a ring.
    let bulk = (-kappa * att * alpha * alpha - amp * amp).exp() / gain;

    for x in 0..m {
        for y in 0..m {
            let phase = step * ((x as f64) - (y as f64));
            let (s, c) = phase.sin_cos();

            // The attenuator's scalar, exp(-(1 - att) alpha^2 (1 - e^{i(theta_x - theta_y)})).
            let lead = (-(1.0 - att) * alpha * alpha * (1.0 - c)).exp() / (m as f64);
            let (ls, lc) = ((1.0 - att) * alpha * alpha * s).sin_cos();

            for n in 0..nb {
                for j in 0..nb {
                    let mut sr = 0.0;
                    let mut si = 0.0;

                    for k in 0..=n.min(j) {
                        if k > 0 && !(kappa > 0.0) {
                            break;
                        }

                        if (n > k || j > k) && !(amp > 0.0) {
                            continue;
                        }

                        let mut lg = 0.5 * (lf[n] + lf[j]) - lf[k] - lf[n - k] - lf[j - k];

                        if k > 0 {
                            lg += (k as f64) * kappa.ln();
                        }

                        if n + j > 2 * k {
                            lg += ((n + j - 2 * k) as f64) * amp.ln();
                        }

                        let w = lg.exp();
                        let ang = step * (((n - k) as f64) * (x as f64) - ((j - k) as f64) * (y as f64));
                        let (ta, tc) = ang.sin_cos();
                        sr += w * tc;
                        si += w * ta;
                    }

                    let gr = bulk * (sr * lc - si * ls);
                    let gi = bulk * (sr * ls + si * lc);
                    out.set(x * nb + n, y * nb + j, lead * gr, lead * gi);
                }
            }
        }
    }

    out.symmetrise();

    out
}

/// The full numerical problem: the postprocessing maps, the sparse constraint operators and
/// the observations they are set equal to.
struct Problem {
    wlc: Wlc,
    terms: Vec<Term>,
    gam: Vec<f64>,
    flat: Vec<usize>,
    rs: Vec<Cx>,
}

/// `Tr(rho Gamma_i)` against `gamma_i`, worst case over the constraints.
fn violation(pr: &Problem, rho: &Cx) -> f64 {
    let mut worst = 0.0f64;

    for (t, g) in pr.terms.iter().zip(pr.gam.iter()) {
        worst = worst.max((t.against(rho) - g).abs());
    }

    worst
}

/// `det` is the whole difference between the two entry points: `None` hands the receiver to Eve
/// and `eta` is then the transmittance of channel and receiver together, `Some` keeps it in
/// Bob's lab and `eta` is the CHANNEL alone.
fn assemble(
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
    cut: f64,
    phase: f64,
    eps: f64,
    det: Option<Detector>,
) -> Problem {
    let nb = nc + 1;
    let rs = regions(m, nc, cut, phase, det);
    let wlc = Wlc::new(m, nc, &rs, eps);
    let ops = match det {
        Some(d) => noisy_moments(nc, &d),
        None => quadratures(nb),
    };
    let step = 2.0 * std::f64::consts::PI / (m as f64);
    let px = 1.0 / (m as f64);
    let mut terms = Vec::new();
    let mut gam = Vec::new();
    let mut flat = Vec::new();

    // Alice's marginal, fixed by the source-replacement overlaps and untouchable by Eve.
    for x in 0..m {
        let mut e = Vec::with_capacity(nb);

        for n in 0..nb {
            e.push((x * nb + n, x * nb + n, 1.0, 0.0));
        }

        flat.push(terms.len());
        terms.push(Term { e });
        gam.push(px);
    }

    for x in 0..m {
        for y in (x + 1)..m {
            let ang = step * ((x as f64) - (y as f64));
            let (s, c) = ang.sin_cos();
            let mag = px * (-alpha * alpha * (1.0 - c)).exp();
            let (os, oc) = (alpha * alpha * s).sin_cos();
            let mut re = Vec::with_capacity(2 * nb);
            let mut im = Vec::with_capacity(2 * nb);

            for n in 0..nb {
                re.push((x * nb + n, y * nb + n, 1.0, 0.0));
                re.push((y * nb + n, x * nb + n, 1.0, 0.0));
                im.push((x * nb + n, y * nb + n, 0.0, 1.0));
                im.push((y * nb + n, x * nb + n, 0.0, -1.0));
            }

            terms.push(Term { e: re });
            gam.push(2.0 * mag * oc);
            terms.push(Term { e: im });
            gam.push(2.0 * mag * os);
        }
    }

    // Ideal receiver: LUL19 Sec. IV E, displaced thermal at sqrt(eta) alpha_x, variance
    // (1 + eta xi)/2. Noisy: LL20 Sec. V B; detector loss multiplies the mean, v_el ADDS to the
    // second moments.
    for x in 0..m {
        let theta = step * (x as f64);
        let (s, c) = theta.sin_cos();
        let obs = match det {
            Some(d) => {
                let tot = d.eta * eta;
                let base = 1.0 + 0.5 * tot * xi + d.vel;

                [
                    (2.0 * tot).sqrt() * alpha * c,
                    (2.0 * tot).sqrt() * alpha * s,
                    2.0 * tot * alpha * alpha * c * c + base,
                    2.0 * tot * alpha * alpha * s * s + base,
                ]
            }
            None => [
                (2.0 * eta).sqrt() * alpha * c,
                (2.0 * eta).sqrt() * alpha * s,
                eta * alpha * alpha + 0.5 * eta * xi,
                2.0 * eta * alpha * alpha * (2.0 * theta).cos(),
            ],
        };

        for (which, op) in ops.iter().enumerate() {
            let mut e = Vec::new();

            for i in 0..nb {
                for j in 0..nb {
                    let (r, m2) = op.get(i, j);

                    if r != 0.0 || m2 != 0.0 {
                        e.push((x * nb + i, x * nb + j, r, m2));
                    }
                }
            }

            terms.push(Term { e });
            gam.push(px * obs[which]);
        }
    }

    Problem {
        wlc,
        terms,
        gam,
        flat,
        rs,
    }
}

/// Gershgorin's lower bound on the lowest eigenvalue of a Hermitian matrix.
fn gershgorin(a: &Cx) -> f64 {
    let n = a.n;
    let mut lo = f64::INFINITY;

    for i in 0..n {
        let mut band = 0.0;

        for j in 0..n {
            if i != j {
                let (r, s) = a.get(i, j);
                band += (r * r + s * s).sqrt();
            }
        }

        lo = lo.min(a.re[i * n + i] - band);
    }

    lo
}

/// A strictly dual-feasible start: shift `W` by `s I`, which the `m` marginal-diagonal terms
/// span exactly.
fn dual_start(pr: &Problem, w: &Cx) -> Vec<f64> {
    let shift = (-gershgorin(w)).max(0.0) + 1e-8 * (w.peak() + 1.0);
    let mut y = vec![0.0f64; pr.terms.len()];

    for &i in pr.flat.iter() {
        y[i] = -shift;
    }

    y
}

/// `W - sum_i y_i Gamma_i`, the dual slack whose positivity is the whole certificate.
fn cone_slack(pr: &Problem, w: &Cx, y: &[f64]) -> Cx {
    let mut s = w.copy();

    for (t, &yi) in pr.terms.iter().zip(y.iter()) {
        if yi == 0.0 {
            continue;
        }

        for &(i, j, gr, gm) in t.e.iter() {
            s.add(i, j, -yi * gr, -yi * gm);
        }
    }

    s.symmetrise();

    s
}

/// One linear semidefinite subproblem: `min Tr(rho W)` over the feasible set, solved on its
/// dual so that both a certified value and a primal point come back.
fn subproblem(pr: &Problem, w: &Cx, gaptol: f64) -> Certificate {
    let cone = Cone::one(
        w.copy(),
        pr.terms.iter().map(|t| Term { e: t.e.to_vec() }).collect(),
        pr.gam.to_vec(),
    );
    let start = dual_start(pr, w);

    cone.run(&start, gaptol)
}

/// `rho + lam * dir`, symmetrised: the point both line searches and both walks step to.
fn advance(rho: &Cx, dir: &Cx, lam: f64) -> Cx {
    let mut trial = rho.copy();

    for k in 0..trial.re.len() {
        trial.re[k] += lam * dir.re[k];
        trial.im[k] += lam * dir.im[k];
    }

    trial.symmetrise();

    trial
}

/// `f_eps` at `rho + lam * dir`, or `+inf` where the point leaves the cone.
fn segment(wlc: &Wlc, rho: &Cx, dir: &Cx, lam: f64) -> f64 {
    match wlc.eval(&advance(rho, dir, lam), false) {
        Some(p) => p.value,
        None => f64::INFINITY,
    }
}

/// Step 1: Frank-Wolfe, Winick-Lutkenhaus-Coles Algorithm 1. Returns the iterate and the
/// value of `f_eps` there -- an UPPER bound on the key rate and not a proof of anything.
fn frank_wolfe(pr: &Problem, start: &Cx, steps: usize, gaptol: f64) -> PyResult<(Cx, f64, usize)> {
    let mut rho = start.copy();
    let mut best = match pr.wlc.eval(&rho, false) {
        Some(p) => p.value,
        None => return Err(PyValueError::new_err(NO_START)),
    };
    let mut taken = 0usize;

    for _ in 0..steps {
        let here = match pr.wlc.eval(&rho, true) {
            Some(p) => p,
            None => break,
        };
        let mut cert = subproblem(pr, &here.grad, gaptol);

        if cert.status == "uncertified" {
            break;
        }

        // The descent direction, sigma - rho, and the golden-section search along it.
        let mut dir = cert.rho.swap_remove(0);

        for k in 0..dir.re.len() {
            dir.re[k] -= rho.re[k];
            dir.im[k] -= rho.im[k];
        }

        let phi = 0.5 * (5.0f64.sqrt() - 1.0);
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        let mut xa = hi - phi * (hi - lo);
        let mut xb = lo + phi * (hi - lo);
        let mut fa = segment(&pr.wlc, &rho, &dir, xa);
        let mut fb = segment(&pr.wlc, &rho, &dir, xb);

        for _ in 0..SEARCH_STEPS {
            if fa < fb {
                hi = xb;
                xb = xa;
                fb = fa;
                xa = hi - phi * (hi - lo);
                fa = segment(&pr.wlc, &rho, &dir, xa);
            } else {
                lo = xa;
                xa = xb;
                fa = fb;
                xb = lo + phi * (hi - lo);
                fb = segment(&pr.wlc, &rho, &dir, xb);
            }
        }

        let trial = advance(&rho, &dir, 0.5 * (lo + hi));
        let got = match pr.wlc.eval(&trial, false) {
            Some(p) => p.value,
            None => break,
        };

        taken += 1;
        let gained = best - got;
        rho = trial;
        best = got;

        if gained <= FW_TOL * best.abs().max(1.0) {
            break;
        }
    }

    Ok((rho, best, taken))
}

/// The honest error-correction cost: `(p_pass, delta_ec, H(Z), H(Z|X))`, LUL19 Sec. IV D:
/// `delta_EC = (1 - beta) H(Z) + beta H(Z|X)` on the POSTSELECTED distribution.
fn cost(m: usize, nc: usize, rho: &Cx, regions: &[Cx], beta: f64) -> (f64, f64, f64, f64) {
    let nb = nc + 1;
    let mut joint = vec![0.0f64; m * m];
    let mut pass = 0.0;

    for x in 0..m {
        for (z, rz) in regions.iter().enumerate() {
            let mut acc = 0.0;

            for i in 0..nb {
                for j in 0..nb {
                    let (rr, ri) = rz.get(i, j);
                    let (br, bi) = rho.get(x * nb + j, x * nb + i);
                    acc += rr * br - ri * bi;
                }
            }

            joint[x * m + z] = acc.max(0.0);
            pass += acc.max(0.0);
        }
    }

    if pass <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut hz = 0.0;
    let mut hzx = 0.0;

    for z in 0..m {
        let mut pz = 0.0;

        for x in 0..m {
            pz += joint[x * m + z] / pass;
        }

        if pz > 0.0 {
            hz -= pz * pz.log2();
        }
    }

    for x in 0..m {
        let mut px = 0.0;

        for z in 0..m {
            px += joint[x * m + z] / pass;
        }

        if px <= 0.0 {
            continue;
        }

        for z in 0..m {
            let p = joint[x * m + z] / pass;

            if p > 0.0 {
                hzx -= p * (p / px).log2();
            }
        }
    }

    (pass, (1.0 - beta) * hz + beta * hzx, hz, hzx)
}

fn check_cutoff(nc: usize) -> PyResult<()> {
    if nc >= 2 && nc <= NC_MAX {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "nc must be in [2, {NC_MAX}] photons, got {nc}"
        )))
    }
}

fn check_sdp_m(m: usize) -> PyResult<()> {
    if (2..=SDP_M_MAX).contains(&m) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "the numerical proof accepts m in [2, {SDP_M_MAX}] states, got {m}: the entropy \
             terms live on m^2 (nc + 1) dimensions and the cost is cubic there"
        )))
    }
}

fn check_steps(steps: usize) -> PyResult<()> {
    if steps == 0 {
        return Err(PyValueError::new_err(
            "steps must be at least 1: with no Frank-Wolfe iteration the linearisation runs at \
             the honest state, which is a valid bound and a useless one",
        ));
    }

    Ok(())
}

fn check_sector(m: usize, z: usize, phase: f64) -> PyResult<()> {
    if z >= m {
        return Err(PyValueError::new_err(format!(
            "z must name one of the {m} key symbols, got {z}"
        )));
    }

    if phase >= std::f64::consts::PI / (m as f64) {
        return Err(PyValueError::new_err(format!(
            "phase must be below pi/m = {}, got {phase}: a wider guard band closes the sector \
             and no outcome maps to a key symbol",
            std::f64::consts::PI / (m as f64)
        )));
    }

    Ok(())
}

/// `eps` must sit inside Winick-Lutkenhaus-Coles Theorem 2's domain, `0 < eps <= 1/[e(d'-1)]`.
/// Not `std::check_eps`, which is the open unit interval.
fn check_eps(eps: f64, dp: f64) -> PyResult<()> {
    let top = 1.0 / (std::f64::consts::E * (dp - 1.0));

    if eps.is_finite() && eps > 0.0 && eps <= top {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "eps must be in (0, {top}] for a Kraus image of dimension {dp}, got {eps}: outside \
             that range Winick-Lutkenhaus-Coles Theorem 2 does not apply and the continuity \
             price zeta is not a bound"
        )))
    }
}

/// The region operator `R_z` on Bob's truncated Fock space as flat row-major
/// `(real, imaginary)` planes of side `nc + 1`.
///
/// `cut` and `phase` are LUL19's postselection parameters `Delta_a` and `Delta_p`. With both
/// zero the `m` operators sum to the identity.
#[pyfunction]
#[pyo3(signature = (m, nc, z, cut=0.0, phase=0.0))]
pub(crate) fn dm_region<'py>(
    py: Python<'py>,
    m: usize,
    nc: usize,
    z: usize,
    cut: f64,
    phase: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    check_sdp_m(m)?;
    check_cutoff(nc)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;
    check_sector(m, z, phase)?;

    let r = region(m, nc, z, cut, phase);

    Ok((r.re.into_pyarray(py), r.im.into_pyarray(py)))
}

/// The honest bipartite state `rho_AB` after the simulated channel, truncated at `nc` photons,
/// as flat row-major `(real, imaginary)` planes of side `m (nc + 1)`.
///
/// Alice's index runs slowest. `xi` is at the CHANNEL INPUT, the same number `dm_rate` takes. The
/// trace falls short of one by exactly the weight the photon-number cutoff drops.
#[pyfunction]
#[pyo3(signature = (m, alpha, eta, xi, nc))]
pub(crate) fn dm_honest<'py>(
    py: Python<'py>,
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    check_sdp_m(m)?;
    check_alpha(alpha)?;
    check_unit("eta", eta)?;
    check_nonneg("xi", xi)?;
    check_cutoff(nc)?;

    let rho = honest(m, alpha, eta, xi, nc);

    Ok((rho.re.into_pyarray(py), rho.im.into_pyarray(py)))
}

/// The perturbed objective and its gradient at a given state:
/// `(f_eps, p_pass, grad_real, grad_imag)`.
///
/// `f_eps` is `D(G_eps(rho) || Z[G_eps(rho)])` in bits, with `G_eps` the postprocessing map
/// followed by the depolarising perturbation Winick-Lutkenhaus-Coles introduce ahead of their
/// Theorem 2. The gradient is the one their gradient lemma states,
/// `G_eps^dag(log G_eps(rho)) - G_eps^dag(log Z[G_eps(rho)])`.
#[pyfunction]
#[pyo3(signature = (m, nc, rho_re, rho_im, cut=0.0, phase=0.0, eps=1e-10))]
pub(crate) fn dm_relent<'py>(
    py: Python<'py>,
    m: usize,
    nc: usize,
    rho_re: Vec<f64>,
    rho_im: Vec<f64>,
    cut: f64,
    phase: f64,
    eps: f64,
) -> PyResult<(f64, f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    check_sdp_m(m)?;
    check_cutoff(nc)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;

    let rho = read_herm("rho", &rho_re, &rho_im, 1e-9)?;

    if rho.n != m * (nc + 1) {
        return Err(PyValueError::new_err(format!(
            "rho must have side m (nc + 1) = {}, got {}",
            m * (nc + 1),
            rho.n
        )));
    }

    let wlc = Wlc::new(m, nc, &regions(m, nc, cut, phase, None), eps);
    check_eps(eps, wlc.dp)?;

    match wlc.eval(&rho, true) {
        None => Err(PyValueError::new_err(
            "rho has no Cholesky factor even after a diagonal floor, so it is not a state and \
             the relative entropy is undefined there",
        )),
        Some(p) => Ok((
            p.value,
            p.pass,
            p.grad.re.into_pyarray(py),
            p.grad.im.into_pyarray(py),
        )),
    }
}

/// The honest error-correction cost of the key map: `(p_pass, delta_ec, h_z, h_zx)` in bits.
///
/// LUL19 Sec. IV D: `delta_EC = (1 - beta) H(Z) + beta H(Z|X)`, both entropies taken on the
/// distribution RENORMALISED by the postselection probability, with `beta` the reconciliation
/// efficiency. `p_pass` is the probability a round survives the key map at all.
#[pyfunction]
#[pyo3(signature = (m, alpha, eta, xi, nc, cut=0.0, phase=0.0, beta=0.95))]
pub(crate) fn dm_cost(
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
    cut: f64,
    phase: f64,
    beta: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_sdp_m(m)?;
    check_alpha(alpha)?;
    check_unit("eta", eta)?;
    check_nonneg("xi", xi)?;
    check_cutoff(nc)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;
    check_prob("beta", beta)?;

    let rho = honest(m, alpha, eta, xi, nc);
    let rs = regions(m, nc, cut, phase, None);

    Ok(cost(m, nc, &rho, &rs, beta))
}

/// The certified asymptotic key rate by the two-step method of Winick, Lutkenhaus & Coles:
/// `(key, bound, upper, p_pass, delta_ec, viol, zeta, steps, status)`, rates in bits per pulse.
///
/// `bound` is step 2, the PROOF: Theorem 3 at the step-1 state,
///
///     bound = f_eps(rho) - Tr(rho grad f_eps(rho)) + gamma . y - viol sum_i |y_i| - zeta,
///
/// with `y`'s slack verified positive SEMIDEFINITE by its spectrum, not merely by a Cholesky.
/// `key = bound - p_pass delta_ec`, NOT clamped. `upper` is step 1's Frank-Wolfe value,
/// `f_eps(rho) - p_pass delta_ec`: an UPPER bound that PROVES NOTHING.
///
/// `viol` is Theorem 3's `epsilon'`, the price the photon-number cutoff charges; `zeta` the
/// continuity price of `eps`; `status` the interior-point path's.
///
/// UNDER THE PHOTON-NUMBER CUTOFF, LUL19 Sec. III B's working assumption; its removal (Upadhyaya,
/// van Himbeeck, Lin & Lutkenhaus, PRX Quantum 2, 020325 (2021)) is not implemented. Raise `nc`
/// until `key` stops moving, but do not call it cutoff-free.
#[pyfunction]
#[pyo3(signature = (m, alpha, eta, xi, nc, cut=0.0, phase=0.0, beta=0.95, eps=1e-10, steps=15, gaptol=1e-8))]
pub(crate) fn dm_secure(
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
    cut: f64,
    phase: f64,
    beta: f64,
    eps: f64,
    steps: usize,
    gaptol: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, usize, String)> {
    check_sdp_m(m)?;
    check_alpha(alpha)?;
    check_unit("eta", eta)?;
    check_nonneg("xi", xi)?;
    check_cutoff(nc)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;
    check_prob("beta", beta)?;
    check_pos("gaptol", gaptol)?;

    check_steps(steps)?;

    let pr = assemble(m, alpha, eta, xi, nc, cut, phase, eps, None);
    check_eps(eps, pr.wlc.dp)?;

    let start = honest(m, alpha, eta, xi, nc);

    certify(&pr, &start, m, nc, xi, beta, steps, gaptol)
}

/// Both steps and every refusal of `dm_secure` and `dm_trusted`; `nc` and `xi` only feed messages.
fn certify(
    pr: &Problem,
    start: &Cx,
    m: usize,
    nc: usize,
    xi: f64,
    beta: f64,
    steps: usize,
    gaptol: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, usize, String)> {
    let (rho, upper, taken) = frank_wolfe(pr, start, steps, gaptol)?;
    let point = match pr.wlc.eval(&rho, true) {
        Some(p) => p,
        None => return Err(PyValueError::new_err(NO_GRADIENT)),
    };
    let cert = subproblem(pr, &point.grad, gaptol);

    if cert.status == "uncertified" {
        return Err(PyValueError::new_err(format!(
            "the dual point could not be certified after {} interior-point iterations: its \
             slack lost positive definiteness, so its objective bounds nothing",
            cert.iters
        )));
    }

    let viol = violation(pr, &rho);
    let zeta = pr.wlc.zeta();

    // Certify by spectrum, not by the Cholesky that accepted the point. The `m` marginal-diagonal
    // terms sum to I with observations summing to one: a common shift lifts the slack by `bump I`
    // at dual cost `bump`.
    let mut y = cert.y.to_vec();
    let block = cone_slack(pr, &point.grad, &y);
    let (vals, _vecs) = eig(&block);
    let lowest = vals.iter().copied().fold(f64::INFINITY, f64::min);

    if !lowest.is_finite() {
        return Err(PyValueError::new_err(
            "the dual slack has no finite spectrum, so the point certifies nothing",
        ));
    }

    let bump = (-lowest).max(0.0) + REPAIR * block.peak().max(1.0);
    let mut weight = 0.0;

    for &i in pr.flat.iter() {
        y[i] -= bump;
        weight += pr.gam[i];
    }

    let dual = cert.dual - bump * weight;
    let mass: f64 = y.iter().map(|v| v.abs()).sum();
    let bound = point.value - inner(&rho, &point.grad) + dual - viol * mass - zeta;
    let (pass, dec, _, _) = cost(m, nc, start, &pr.rs, beta);

    if !bound.is_finite() || !point.value.is_finite() {
        return Err(PyValueError::new_err(NOT_FINITE));
    }

    let slack = 1e-6 * point.value.abs().max(1.0) + cert.gap.abs();

    if bound > point.value + slack {
        return Err(PyValueError::new_err(format!(
            "the certified bound {bound} exceeds the step-1 value {} by more than the \
             interior-point gap {slack}: a broken gradient or an infeasible constraint set, most \
             likely because nc = {nc} is too small to reproduce the observations. Raise nc",
            point.value
        )));
    }

    // PURE LOSS bites here: at xi = 0 rho_AB loses rank (LUL19 Sec. IV B).
    let floor = -(pass * (m as f64).log2() + 1.0);

    if bound < floor {
        return Err(PyValueError::new_err(format!(
            "the certified bound {bound} fell below {floor}, looser than the zero every \
             relative entropy satisfies: the linearisation collapsed, almost certainly because \
             xi = {xi} leaves rho_AB near rank deficient (nearly pure loss, as LUL19 report). \
             Nothing was proved; raise xi off zero or lower the photon-number cutoff"
        )));
    }

    if bound > pass * (m as f64).log2() + slack {
        return Err(over_cap(bound, pass, m));
    }

    Ok((
        bound - pass * dec,
        bound,
        upper - pass * dec,
        pass,
        dec,
        viol,
        zeta,
        taken,
        cert.status.to_string(),
    ))
}

/// LL20 Sec. III A's detector model needs `eta_d` strictly inside `(0, 1)`; at `eta_d = 1` the
/// Laguerre argument `-r^2 / [eta_d nbar (1 + nbar)]` also loses its scale.
fn check_detector(eta_d: f64, v_el: f64) -> PyResult<()> {
    check_nonneg("v_el", v_el)?;

    if eta_d > 0.0 && eta_d < 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "eta_d must be in (0, 1), got {eta_d}: Lin & Lutkenhaus's detector is a beamsplitter \
             of transmittance eta_d, and a lossless one has no port for the electronic noise. \
             For an ideal receiver use dm_secure, not this at v_el = 0"
        )))
    }
}

/// The trusted-detector region operator `R_z` as flat row-major `(real, imaginary)` planes of
/// side `nc + 1` -- `dm_region`'s counterpart for a receiver whose loss and electronic noise are
/// calibrated instead of handed to Eve.
///
/// Exposed for arbitration, not as an API. Without postselection the `m` operators sum to the
/// identity and each carries exactly `1/m` on its diagonal.
#[pyfunction]
#[pyo3(signature = (m, nc, z, eta_d, v_el, cut=0.0, phase=0.0))]
pub(crate) fn dm_noisy<'py>(
    py: Python<'py>,
    m: usize,
    nc: usize,
    z: usize,
    eta_d: f64,
    v_el: f64,
    cut: f64,
    phase: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    check_sdp_m(m)?;
    check_cutoff(nc)?;
    check_detector(eta_d, v_el)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;
    check_sector(m, z, phase)?;

    let r = noisy_region(m, nc, z, cut, phase, &Detector::new(eta_d, v_el));

    Ok((r.re.into_pyarray(py), r.im.into_pyarray(py)))
}

/// One trusted-detector observable as flat row-major `(real, imaginary)` planes of side
/// `nc + 1`; `which` selects `F_Q`, `F_P`, `S_Q`, `S_P` in that order.
///
/// Slots match `quadratures`' `q`, `p`, `n`, `d`, but the last two are LL20 Sec. IV B's basis:
/// `S_Q = n + d/2 + 1`, `S_P = n - d/2 + 1` at an ideal receiver. Exposed for arbitration, not as an API.
#[pyfunction]
#[pyo3(signature = (nc, which, eta_d, v_el))]
pub(crate) fn dm_observe<'py>(
    py: Python<'py>,
    nc: usize,
    which: usize,
    eta_d: f64,
    v_el: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    check_cutoff(nc)?;
    check_detector(eta_d, v_el)?;

    if which >= 4 {
        return Err(PyValueError::new_err(format!(
            "which must name one of F_Q, F_P, S_Q, S_P as 0..3, got {which}"
        )));
    }

    let mut ops = noisy_moments(nc, &Detector::new(eta_d, v_el));
    let o = ops.swap_remove(which);

    Ok((o.re.into_pyarray(py), o.im.into_pyarray(py)))
}

/// The certified asymptotic key rate with the detector TRUSTED: `dm_secure`'s `(key, bound,
/// upper, p_pass, delta_ec, viol, zeta, steps, status)` with the same meanings, `bound` the step-2
/// proof and `upper` the step-1 value that proves nothing.
///
/// Lin & Lutkenhaus, Phys. Rev. Applied 14, 064030 (2020), arXiv:2006.06166. `eta` is the CHANNEL
/// alone and `eta_d`, `v_el` Bob's calibrated receiver: folding the receiver into `eta` is the
/// referring-plane error and double-counts the loss.
///
/// Untrusted, `v_el` refers back as `xi -> xi + 2 v_el / (eta_d eta)` and grows as `1/eta`;
/// trusted, it stays additive in Bob's second moments.
///
/// UNDER THE SAME PHOTON-NUMBER CUTOFF (LUL19 Sec. III B; LL20 Sec. VIII). Trusting the detector
/// buys distance, not rigour.
#[pyfunction]
#[pyo3(signature = (m, alpha, eta, xi, nc, eta_d, v_el, cut=0.0, phase=0.0, beta=0.95, eps=1e-10, steps=15, gaptol=1e-8))]
pub(crate) fn dm_trusted(
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
    eta_d: f64,
    v_el: f64,
    cut: f64,
    phase: f64,
    beta: f64,
    eps: f64,
    steps: usize,
    gaptol: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, usize, String)> {
    check_sdp_m(m)?;
    check_alpha(alpha)?;
    check_unit("eta", eta)?;
    check_nonneg("xi", xi)?;
    check_cutoff(nc)?;
    check_detector(eta_d, v_el)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;
    check_prob("beta", beta)?;
    check_pos("gaptol", gaptol)?;

    check_steps(steps)?;

    let pr = assemble(m, alpha, eta, xi, nc, cut, phase, eps, Some(Detector::new(eta_d, v_el)));
    check_eps(eps, pr.wlc.dp)?;

    // CHANNEL output only: Bob's loss and noise are already in the POVM.
    let start = honest(m, alpha, eta, xi, nc);

    certify(&pr, &start, m, nc, xi, beta, steps, gaptol)
}

// KANITSCHAR EQ. (21): THE RELAXED FEASIBLE SET
//
// Kanitschar, George, Lin, Upadhyaya & Lutkenhaus, PRX Quantum 4, 040306 (2023),
// arXiv:2301.08686.
//
// Theorem and equation numbers here are arXiv v2's; PRX Quantum renumbers v2's Theorems 3, 4, 6
// as 2, 3, 4. Do NOT renumber: refusal literals carry the v2 numbers. APS likewise
// renumbers Lucamarini's Eq. (45) as (C16).
//
// `dm_secure`/`dm_trusted` minimise over Eq. (19), equalities Tr(Gamma_j rho) = gamma_j on a
// NORMALISED state; Eq. (20) widens each to half-width mu_j and Eq. (21) relaxes again for the
// dimension reduction. Eq. (19)'s set is CONTAINED in Eq. (21)'s, so its minimum is the LARGER
// and overstates the key length in Theorem 6 Eq. (9).
//
// Read Eq. (21) off Appendix E, not the body: the body's four rows per operator demand
// 2 mu_j <= w ||Gamma_j||, false at the paper's own operating point, so the printed set is empty.
// Assembled here: `Tr(Gamma_j rhobar) <= mu_j + gamma_j`, `Tr(-Gamma_j rhobar) <= mu_j - gamma_j
// + w ||Gamma_j||`.
//
// Constraint operators are displaced number operators `|j><j| tensor n_beta_j` and its square,
// beta_j = sqrt(eta) alpha_j, not `assemble`'s quadratures. Under the honest channel both
// observations are the thermal moments nbar = eta xi / 2 and 2 nbar^2 + nbar for every j.
//
// The objective runs on SUBNORMALISED states; `Wlc::eval` holds since `Tr Z[A] = Tr A`.
//
// NO TRUSTED BRANCH: Kanitschar App. B's rescaling coefficients `A`..`E`, an asymptotic expansion
// in `M / (sqrt(eta_d) c)`, are unchecked, and `[n^2]'` reads the fourth radial moment of LL20's
// `G_y` where `noisy_moments` reaches the second.

/// `n_beta = (a^dag - beta*)(a - beta)` and its square on `nb` Fock levels.
///
/// The square is taken on `nb + 2` levels and cut: `P n_beta^2 P`, NOT `(P n_beta P)^2`. The
/// dimension-reduction shift `w ||Gamma||_inf` is written on the first.
fn displaced(nb: usize, br: f64, bi: f64) -> (Cx, Cx) {
    let big = nb + 2;
    let mut n1 = Cx::zeros(big);
    let mag = br * br + bi * bi;

    for k in 0..big {
        n1.set(k, k, (k as f64) + mag, 0.0);

        if k + 1 < big {
            let root = ((k + 1) as f64).sqrt();
            // -beta* a on the upper diagonal, -beta a^dag on the lower one.
            n1.set(k, k + 1, -br * root, bi * root);
            n1.set(k + 1, k, -br * root, -bi * root);
        }
    }

    let sq = n1.mul(&n1);
    let mut out1 = Cx::zeros(nb);
    let mut out2 = Cx::zeros(nb);

    for i in 0..nb {
        for j in 0..nb {
            let (r, m) = n1.get(i, j);
            out1.set(i, j, r, m);
            let (r2, m2) = sq.get(i, j);
            out2.set(i, j, r2, m2);
        }
    }

    out1.symmetrise();
    out2.symmetrise();

    (out1, out2)
}

/// Alice's marginal `rho_A`, fixed by the source-replacement overlaps and untouchable by Eve:
/// `sqrt(p_i p_j) <Psi_j|Psi_i>` over the `m` ring states of amplitude `alpha`.
fn marginal(m: usize, alpha: f64) -> Cx {
    let mut out = Cx::zeros(m);
    let step = 2.0 * std::f64::consts::PI / (m as f64);
    let px = 1.0 / (m as f64);

    for x in 0..m {
        for y in 0..m {
            let ang = step * ((x as f64) - (y as f64));
            let (s, c) = ang.sin_cos();
            let mag = px * (-alpha * alpha * (1.0 - c)).exp();
            let (os, oc) = (alpha * alpha * s).sin_cos();
            out.set(x, y, mag * oc, mag * os);
        }
    }

    out.symmetrise();

    out
}

/// The `m * m` real coordinates of a Hermitian `m x m` matrix, each a coordinate list with
/// BOTH triangles listed: `m` diagonal projectors, then a real and an imaginary direction per
/// off-diagonal pair.
fn hermbasis(m: usize) -> Vec<Vec<(usize, usize, f64, f64)>> {
    let mut out = Vec::with_capacity(m * m);

    for i in 0..m {
        out.push(vec![(i, i, 1.0, 0.0)]);
    }

    for i in 0..m {
        for j in (i + 1)..m {
            out.push(vec![(i, j, 1.0, 0.0), (j, i, 1.0, 0.0)]);
            out.push(vec![(i, j, 0.0, 1.0), (j, i, 0.0, -1.0)]);
        }
    }

    out
}

/// Kanitschar Eq. (21) assembled: the postprocessing maps, the `2m` interval constraints and
/// the relaxation of Alice's marginal.
struct Relaxed {
    wlc: Wlc,
    rs: Vec<Cx>,
    m: usize,
    d: usize,
    ops: Vec<Term>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    basis: Vec<Vec<(usize, usize, f64, f64)>>,
    lifted: Vec<Vec<(usize, usize, f64, f64)>>,
    ra: Vec<f64>,
    rho_a: Cx,
    w: f64,
}

impl Relaxed {
    /// Four offsets into `y`, which runs upper multipliers, lower multipliers, the three
    /// scalars `c`, `e`, `s`, then `tau` and `Theta` in the Hermitian basis: where the
    /// LOWERS start, where `c` sits, where `tau` starts, and where `Theta` starts. `s` is
    /// the entry one below `tau`.
    fn vars(&self) -> (usize, usize, usize, usize) {
        let j = self.ops.len();
        let mm = self.basis.len();

        (j, 2 * j, 2 * j + 3, 2 * j + 3 + mm)
    }
}

/// The relaxed problem at one operating point. `mu_n`/`mu_n2` are Theorem 4's half-widths for
/// the two observable kinds and `x_n`/`x_n2` their clipped infinity norms, both the caller's.
fn relax(
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
    w: f64,
    mu: (f64, f64),
    x: (f64, f64),
    cut: f64,
    phase: f64,
    eps: f64,
) -> Relaxed {
    let nb = nc + 1;
    let d = m * nb;
    let rs = regions(m, nc, cut, phase, None);
    let wlc = Wlc::new(m, nc, &rs, eps);
    let step = 2.0 * std::f64::consts::PI / (m as f64);
    let px = 1.0 / (m as f64);
    let nbar = 0.5 * eta * xi;
    let obs = [nbar, 2.0 * nbar * nbar + nbar];
    let mut ops = Vec::with_capacity(2 * m);
    let mut lo = Vec::with_capacity(2 * m);
    let mut hi = Vec::with_capacity(2 * m);

    for j in 0..m {
        let theta = step * (j as f64);
        let (si, co) = theta.sin_cos();
        let root = eta.sqrt() * alpha;
        let (n1, n2) = displaced(nb, root * co, root * si);

        for (which, op) in [n1, n2].iter().enumerate() {
            let mut e = Vec::new();

            for a in 0..nb {
                for b in 0..nb {
                    let (r, i) = op.get(a, b);

                    if r != 0.0 || i != 0.0 {
                        e.push((j * nb + a, j * nb + b, r, i));
                    }
                }
            }

            let seen = px * obs[which];
            let half = if which == 0 { mu.0 } else { mu.1 };
            let norm = if which == 0 { x.0 } else { x.1 };
            ops.push(Term { e });
            hi.push(seen + half);
            lo.push(seen - half - w * norm);
        }
    }

    let basis = hermbasis(m);
    let ra_matrix = marginal(m, alpha);
    let ra: Vec<f64> = basis
        .iter()
        .map(|b| Term { e: b.to_vec() }.against(&ra_matrix))
        .collect();
    let lifted: Vec<Vec<(usize, usize, f64, f64)>> = basis
        .iter()
        .map(|b| {
            let mut e = Vec::with_capacity(b.len() * nb);

            for &(i, j, r, im) in b.iter() {
                for k in 0..nb {
                    e.push((i * nb + k, j * nb + k, r, im));
                }
            }

            e
        })
        .collect();

    Relaxed {
        wlc,
        rs,
        m,
        d,
        ops,
        lo,
        hi,
        basis,
        lifted,
        ra,
        rho_a: ra_matrix,
        w,
    }
}

/// The dual cone of the linearised Eq. (21) problem at gradient `g`, in six blocks: the
/// entropy block of side `m (nc + 1)`, four blocks of side `m` carrying `tau >= 0`,
/// `s I - tau >= 0`, `Theta >= 0` and `s I - Theta >= 0`, and one DIAGONAL block holding the
/// sign conditions on the interval and trace multipliers.
///
/// Derived here, not transcribed: the Lagrangian dual of `min <G, sigma>` over `sigma, P, N >= 0`
/// with `l_j <= Tr(Gamma_j sigma) <= h_j`, `1 - w <= Tr sigma <= 1`, `Tr P + Tr N <= 2 sqrt(w)`,
/// `P >= Tr_B[sigma] - rho_A` and `N >= -(Tr_B[sigma] - rho_A)`.
fn relaxed_cone(rl: &Relaxed, g: &Cx) -> Cone {
    let (nlow, nc_var, ntau, ntheta) = rl.vars();
    let mm = rl.basis.len();
    let nvar = ntheta + mm;
    let signs = 2 * nlow + 3;
    let sizes = [rl.d, rl.m, rl.m, rl.m, rl.m, signs];
    let mut blocks: Vec<Block> = sizes
        .iter()
        .enumerate()
        .map(|(b, &n)| Block {
            w: if b == 0 { g.copy() } else { Cx::zeros(n) },
            g: (0..nvar).map(|_| Term { e: Vec::new() }).collect(),
        })
        .collect();
    let mut gam = vec![0.0f64; nvar];

    // S_0 = G + sum_j (a_j - b_j) Gamma_j + (c - e) I + (tau - Theta) tensor I_B, written as
    // W - sum_i y_i Gamma_i^(0), so every coefficient enters negated.
    for (j, op) in rl.ops.iter().enumerate() {
        for &(a, b, r, i) in op.e.iter() {
            blocks[0].g[j].e.push((a, b, -r, -i));
            blocks[0].g[nlow + j].e.push((a, b, r, i));
        }

        gam[j] = -rl.hi[j];
        gam[nlow + j] = rl.lo[j];
        blocks[5].g[j].e.push((j, j, -1.0, 0.0));
        blocks[5].g[nlow + j].e.push((nlow + j, nlow + j, -1.0, 0.0));
    }

    for k in 0..rl.d {
        blocks[0].g[nc_var].e.push((k, k, -1.0, 0.0));
        blocks[0].g[nc_var + 1].e.push((k, k, 1.0, 0.0));
    }

    gam[nc_var] = -1.0;
    gam[nc_var + 1] = 1.0 - rl.w;
    gam[ntau - 1] = -2.0 * rl.w.sqrt();
    blocks[5].g[nc_var].e.push((nc_var, nc_var, -1.0, 0.0));
    blocks[5].g[nc_var + 1].e.push((nc_var + 1, nc_var + 1, -1.0, 0.0));
    blocks[5].g[ntau - 1].e.push((ntau - 1, ntau - 1, -1.0, 0.0));

    for k in 0..rl.m {
        blocks[2].g[ntau - 1].e.push((k, k, -1.0, 0.0));
        blocks[4].g[ntau - 1].e.push((k, k, -1.0, 0.0));
    }

    for k in 0..mm {
        let (t, h) = (ntau + k, ntheta + k);
        gam[t] = -rl.ra[k];
        gam[h] = rl.ra[k];

        for &(a, b, r, i) in rl.lifted[k].iter() {
            blocks[0].g[t].e.push((a, b, -r, -i));
            blocks[0].g[h].e.push((a, b, r, i));
        }

        for &(a, b, r, i) in rl.basis[k].iter() {
            blocks[1].g[t].e.push((a, b, -r, -i));
            blocks[2].g[t].e.push((a, b, r, i));
            blocks[3].g[h].e.push((a, b, -r, -i));
            blocks[4].g[h].e.push((a, b, r, i));
        }
    }

    Cone { b: blocks, gam }
}

/// A strictly feasible dual start: `tau = Theta = delta I` cancels in the entropy block,
/// `s = 2 delta` opens the two difference blocks, and `c - e` shifts `G` positive by a
/// Gershgorin bound on its lowest eigenvalue.
fn relaxed_start(rl: &Relaxed, g: &Cx) -> Vec<f64> {
    let (nlow, nc_var, ntau, ntheta) = rl.vars();
    let mm = rl.basis.len();
    let shift = (-gershgorin(g)).max(0.0) + 1e-8 * (g.peak() + 1.0);
    let seed = 1e-6 * (g.peak() + 1.0);
    let mut y = vec![0.0f64; ntheta + mm];

    for j in 0..nlow {
        y[j] = seed;
        y[nlow + j] = seed;
    }

    y[nc_var] = shift + seed;
    y[nc_var + 1] = seed;
    y[ntau - 1] = 2.0 * seed;

    for k in 0..rl.m {
        y[ntau + k] = seed;
        y[ntheta + k] = seed;
    }

    y
}

/// The largest amount by which a recovered primal point leaves Eq. (21), over the interval
/// rows, the trace window, the trace-norm budget and the two marginal inequalities.
///
/// REPORTED, NOT CHARGED, unlike `certify`: WLC's `epsilon'` enlarges an EQUALITY set a truncated
/// state cannot meet exactly, while Eq. (21)'s inequalities admit the honest truncated state with
/// room. A solver diagnostic.
fn relaxed_violation(rl: &Relaxed, blocks: &[Cx]) -> f64 {
    let sigma = &blocks[0];
    let mut worst = 0.0f64;

    for (j, op) in rl.ops.iter().enumerate() {
        let seen = op.against(sigma);
        worst = worst.max(seen - rl.hi[j]).max(rl.lo[j] - seen);
    }

    let tr = sigma.trace();
    worst = worst.max(tr - 1.0).max(1.0 - rl.w - tr);
    worst = worst.max(blocks[2].trace() + blocks[4].trace() - 2.0 * rl.w.sqrt());

    // P - (Tr_B[sigma] - rho_A) and N + (Tr_B[sigma] - rho_A), read through their lowest
    // eigenvalues: a negative one is the margin by which the marginal relaxation is missed.
    let ra = &rl.rho_a;
    let mut gapp = Cx::zeros(rl.m);
    let mut gapn = Cx::zeros(rl.m);
    let nb = rl.d / rl.m;

    for a in 0..rl.m {
        for b in 0..rl.m {
            let mut re = 0.0;
            let mut im = 0.0;

            for k in 0..nb {
                let (r, i) = sigma.get(a * nb + k, b * nb + k);
                re += r;
                im += i;
            }

            let (pr, pi) = blocks[2].get(a, b);
            let (nr, ni) = blocks[4].get(a, b);
            gapp.set(a, b, pr - re + ra.re[a * rl.m + b], pi - im + ra.im[a * rl.m + b]);
            gapn.set(a, b, nr + re - ra.re[a * rl.m + b], ni + im - ra.im[a * rl.m + b]);
        }
    }

    for block in [gapp, gapn].iter() {
        let (vals, _v) = eig(block);

        for v in vals.iter() {
            worst = worst.max(-v);
        }
    }

    worst.max(0.0)
}

/// Step 1 over the relaxed set: Frank-Wolfe, with the linear subproblem taken on the
/// six-block dual so that both a certified value and a primal point come back.
fn relaxed_walk(rl: &Relaxed, start: &Cx, steps: usize, gaptol: f64) -> PyResult<(Cx, f64, usize)> {
    let mut rho = start.copy();
    let mut best = match rl.wlc.eval(&rho, false) {
        Some(p) => p.value,
        None => return Err(PyValueError::new_err(NO_START)),
    };
    let mut taken = 0usize;

    for _ in 0..steps {
        let here = match rl.wlc.eval(&rho, true) {
            Some(p) => p,
            None => break,
        };
        let cone = relaxed_cone(rl, &here.grad);
        let mut cert = cone.run(&relaxed_start(rl, &here.grad), gaptol);

        if cert.status == "uncertified" {
            break;
        }

        let mut dir = cert.rho.swap_remove(0);

        for k in 0..dir.re.len() {
            dir.re[k] -= rho.re[k];
            dir.im[k] -= rho.im[k];
        }

        // Golden section on log10(lambda), NOT lambda: the useful step is a few parts in a
        // million, and a linear bracket (0.618^12 = 5e-3) leaves the step-2 gap three orders
        // wider than `frank_wolfe` reaches.
        let phi = 0.5 * (5.0f64.sqrt() - 1.0);
        let mut lo = LOG_STEP;
        let mut hi = 0.0f64;
        let mut xa = hi - phi * (hi - lo);
        let mut xb = lo + phi * (hi - lo);
        let mut fa = segment(&rl.wlc, &rho, &dir, exp10(xa));
        let mut fb = segment(&rl.wlc, &rho, &dir, exp10(xb));

        for _ in 0..SEARCH_STEPS {
            if fa < fb {
                hi = xb;
                xb = xa;
                fb = fa;
                xa = hi - phi * (hi - lo);
                fa = segment(&rl.wlc, &rho, &dir, exp10(xa));
            } else {
                lo = xa;
                xa = xb;
                fa = fb;
                xb = lo + phi * (hi - lo);
                fb = segment(&rl.wlc, &rho, &dir, exp10(xb));
            }
        }

        let trial = advance(&rho, &dir, exp10(0.5 * (lo + hi)));
        let got = match rl.wlc.eval(&trial, false) {
            Some(p) => p.value,
            None => break,
        };

        let gained = best - got;

        // A rise means the bracket missed, not that the walk arrived: do not take it.
        if gained <= FW_TOL * best.abs().max(1.0) {
            break;
        }

        taken += 1;
        rho = trial;
        best = got;
    }

    Ok((rho, best, taken))
}

/// `10^u`, the line search's step from its logarithmic bracket.
fn exp10(u: f64) -> f64 {
    (u * std::f64::consts::LN_10).exp()
}

/// Repair a dual point until EVERY block's slack is positive semidefinite by its spectrum,
/// and return the objective there. Three moves, each exact in what it costs:
///
///   `tau += d I`, `Theta += d I`, `s += d`   leaves the entropy block and both difference
///                                            blocks alone and costs `2 sqrt(w) d`,
///   `s += d`                                 opens the two difference blocks, same cost,
///   `c += d`                                 lifts the entropy block by `d I` and costs `d`.
fn relaxed_repair(rl: &Relaxed, cone: &Cone, y: &mut [f64]) -> PyResult<f64> {
    let (_nlow, nc_var, ntau, ntheta) = rl.vars();

    for round in 0..3 {
        let slacks = cone.slack(y);
        let pick: &[usize] = match round {
            0 => &[1, 3],
            1 => &[2, 4],
            _ => &[0],
        };
        let mut worst = 0.0f64;
        let mut peak = 1.0f64;

        for &b in pick.iter() {
            let (vals, _v) = eig(&slacks[b]);
            peak = peak.max(slacks[b].peak());

            for v in vals.iter() {
                if !v.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "block {b} of the dual slack has no finite spectrum, so the point \
                         certifies nothing"
                    )));
                }

                worst = worst.min(*v);
            }
        }

        let bump = (-worst).max(0.0) + REPAIR * peak;

        match round {
            0 => {
                for k in 0..rl.m {
                    y[ntau + k] += bump;
                    y[ntheta + k] += bump;
                }

                y[ntau - 1] += bump;
            }
            1 => y[ntau - 1] += bump,
            _ => y[nc_var] += bump,
        }
    }

    for (b, s) in cone.slack(y).iter().enumerate() {
        let (vals, _v) = eig(s);

        for v in vals.iter() {
            if !(*v > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "block {b} of the repaired dual slack still carries the eigenvalue {v}, so \
                     it is not in the cone and its objective bounds nothing"
                )));
            }
        }
    }

    let mut dual = 0.0;

    for (a, b) in cone.gam.iter().zip(y.iter()) {
        dual += a * b;
    }

    Ok(dual)
}

/// The certified minimum of `H(X|E')` over Kanitschar Eq. (21)'s relaxed feasible set
/// `S^(E&A)`, the input of their Theorem 6 Eq. (9) that `dm_secure` may not stand in for:
/// `(entropy, bound, upper, p_pass, delta_ec, viol, zeta, steps, status)`.
///
/// `entropy` is `bound / p_pass`, bits per KEPT round, the plane `dm_length` takes. `bound` is
/// the minimum in bits per pulse and is the PROOF, WLC Theorem 3 at the step-1 state,
///
///     bound = f_eps(rho) - Tr(rho grad f_eps(rho)) + dual - zeta,
///
/// with `dual` at a six-block point whose every slack is verified positive definite by its
/// spectrum. `viol` is REPORTED, NOT CHARGED (see `relaxed_violation`). `upper` is step 1's
/// Frank-Wolfe value and PROVES NOTHING.
///
/// `w` IS DECLARED, NOT DERIVED: Kanitschar Sec. VI B choose it freely, paying in `eps_ET`, by
/// `w = max(w_exp, w_eps, w_min)`; `dm_weight` is the first term and `dm_wchoice` the second.
/// `mu_n`/`mu_n2` are Theorem 4's acceptance half-widths on `Tr(Gamma_j rho)`, already carrying
/// Alice's `1/m`; `x_n`/`x_n2` the CLIPPED infinity norms from `dm_clip`. All four are the
/// caller's: nothing here clips a detector.
///
/// UNTRUSTED, IDEAL RECEIVER ONLY: `eta` is channel and receiver together, `xi` at the channel
/// input, as in `dm_secure`. `dm_trusted` is the trusted ASYMPTOTIC bound, a different quantity.
#[pyfunction]
#[pyo3(signature = (m, alpha, eta, xi, nc, w, mu_n, mu_n2, x_n, x_n2, cut=0.0, phase=0.0, beta=0.95, eps=1e-10, steps=15, gaptol=1e-8))]
pub(crate) fn dm_relaxed(
    m: usize,
    alpha: f64,
    eta: f64,
    xi: f64,
    nc: usize,
    w: f64,
    mu_n: f64,
    mu_n2: f64,
    x_n: f64,
    x_n2: f64,
    cut: f64,
    phase: f64,
    beta: f64,
    eps: f64,
    steps: usize,
    gaptol: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, usize, String)> {
    check_sdp_m(m)?;
    check_alpha(alpha)?;
    check_unit("eta", eta)?;
    check_nonneg("xi", xi)?;
    check_cutoff(nc)?;
    check_nonneg("mu_n", mu_n)?;
    check_nonneg("mu_n2", mu_n2)?;
    check_pos("x_n", x_n)?;
    check_pos("x_n2", x_n2)?;
    check_nonneg("cut", cut)?;
    check_nonneg("phase", phase)?;
    check_prob("beta", beta)?;
    check_pos("gaptol", gaptol)?;

    if !(w > 0.0 && w < 1.0) {
        return Err(PyValueError::new_err(format!(
            "w must be in (0, 1), got {w}: at w = 0 the relaxed set pins the trace and Alice's \
             marginal and has no interior for the barrier. For the equality problem use \
             dm_secure, not this at zero weight"
        )));
    }

    check_steps(steps)?;

    let rl = relax(m, alpha, eta, xi, nc, w, (mu_n, mu_n2), (x_n, x_n2), cut, phase, eps);
    check_eps(eps, rl.wlc.dp)?;

    let start = honest(m, alpha, eta, xi, nc);
    let (rho, upper, taken) = relaxed_walk(&rl, &start, steps, gaptol)?;
    let point = match rl.wlc.eval(&rho, true) {
        Some(p) => p,
        None => return Err(PyValueError::new_err(NO_GRADIENT)),
    };
    let cone = relaxed_cone(&rl, &point.grad);
    let cert = cone.run(&relaxed_start(&rl, &point.grad), gaptol);

    if cert.status == "uncertified" {
        return Err(PyValueError::new_err(format!(
            "the dual point could not be certified after {} interior-point iterations: some \
             block's slack lost positive definiteness, so its objective bounds nothing",
            cert.iters
        )));
    }

    let viol = relaxed_violation(&rl, &cert.rho);
    let zeta = rl.wlc.zeta();
    let mut y = cert.y.to_vec();
    let dual = relaxed_repair(&rl, &cone, &mut y)?;
    let bound = point.value - inner(&rho, &point.grad) + dual - zeta;
    let (pass, dec, _, _) = cost(m, nc, &start, &rl.rs, beta);

    if !bound.is_finite() || !point.value.is_finite() || !(pass > 0.0) {
        return Err(PyValueError::new_err(NOT_FINITE));
    }

    let slack = 1e-6 * point.value.abs().max(1.0) + cert.gap.abs();

    if bound > point.value + slack {
        return Err(PyValueError::new_err(format!(
            "the certified bound {bound} exceeds the step-1 value {} by more than the \
             interior-point gap {slack}: the relaxed set is empty, as Kanitschar Eq. (21) is \
             when read as PRINTED (2 mu_j <= w ||Gamma_j||_inf). Widen mu, raise w, or raise \
             the photon-number cutoff nc = {nc}",
            point.value
        )));
    }

    let floor = -(pass * (m as f64).log2() + 1.0);

    if bound < floor {
        return Err(PyValueError::new_err(format!(
            "the certified bound {bound} fell below {floor}, looser than the zero every \
             relative entropy satisfies; nothing was proved. Either w = {w} leaves the barrier \
             too thin a shell around the normalised states (the equality problem is dm_secure), \
             or xi = {xi} leaves rho_AB near rank deficient (nearly pure loss, as LUL19 report). \
             Raise w, raise xi off zero, or lower the photon-number cutoff"
        )));
    }

    if bound > pass * (m as f64).log2() + slack {
        return Err(over_cap(bound, pass, m));
    }

    Ok((
        bound / pass,
        bound,
        upper,
        pass,
        dec,
        viol,
        zeta,
        taken,
        cert.status.to_string(),
    ))
}

/// The displaced number operator `n_beta` and its square on `nc + 1` Fock levels, as two
/// pairs of flat row-major `(real, imaginary)` planes: `(n_re, n_im, sq_re, sq_im)`.
///
/// `n_beta = (a^dag - beta*)(a - beta)`, Kanitschar Eq. (21)'s constraint operator where
/// `dm_secure` uses `{q, p, n, d}`. The square is `P n_beta^2 P`, NOT `(P n_beta P)^2`. Exposed
/// for arbitration, not as an API.
#[pyfunction]
#[pyo3(signature = (nc, beta_re, beta_im))]
pub(crate) fn dm_number<'py>(
    py: Python<'py>,
    nc: usize,
    beta_re: f64,
    beta_im: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    check_cutoff(nc)?;

    if !beta_re.is_finite() || !beta_im.is_finite() {
        return Err(PyValueError::new_err(format!(
            "beta must be finite, got {beta_re} + {beta_im}i"
        )));
    }

    let (one, two) = displaced(nc + 1, beta_re, beta_im);

    Ok((
        one.re.into_pyarray(py),
        one.im.into_pyarray(py),
        two.re.into_pyarray(py),
        two.im.into_pyarray(py),
    ))
}

/// The weight `w_exp` the honest channel leaves OUTSIDE the photon-number cutoff,
/// `1 - Tr rhobar`, which is the first of the three terms in Kanitschar Sec. VI B's
/// `w = max(w_exp, w_eps, w_min)`.
///
/// Exact, not a bound: a declared `w` below it is a claim the honest state itself fails.
#[pyfunction]
pub(crate) fn dm_weight(m: usize, alpha: f64, eta: f64, xi: f64, nc: usize) -> PyResult<f64> {
    check_sdp_m(m)?;
    check_alpha(alpha)?;
    check_unit("eta", eta)?;
    check_nonneg("xi", xi)?;
    check_cutoff(nc)?;

    Ok((1.0 - honest(m, alpha, eta, xi, nc).trace()).max(0.0))
}

/// The clipped infinity norms `(x_n, x_n2)` of the displaced number operator and its square
/// over a soft detection limit `m_soft`, for a displacement of modulus `beta`.
///
/// Kanitschar Eq. (25) clips a heterodyne outcome componentwise into `[-M, M]^2`, and their
/// Appendix B writes the two observables anti-normally ordered as `f_n = |zeta|^2 - 1` and
/// `f_n2 = |zeta|^4 - 3 |zeta|^2 + 1`. With `u := |zeta - beta|^2` running over `[0, U]` and
/// `U = (M + |beta|)^2 + M^2` at the far corner, `x_n = max(U - 1, 1)` and
/// `x_n2 = max(U^2 - 3U + 1, 5/4)`, the second minimum being `|f_n2|` at its interior
/// stationary point `u = 3/2`.
///
/// AT `beta = 0` THIS IS `2M^2 - 1` AND `4M^4 - 6M^2 + 1`, NOT Kanitschar Sec. V F's printed
/// `M^2 - 1/2` and `M^4 - M^2/2` -- 49 and 2351 against 24.5 and 612.5 at their `M = 5`; their
/// Appendix B states `g_n = 2M^2 - 1`. The printed constants narrow acceptance and RAISE the key.
#[pyfunction]
#[pyo3(signature = (m_soft, beta=0.0))]
pub(crate) fn dm_clip(m_soft: f64, beta: f64) -> PyResult<(f64, f64)> {
    check_pos("m_soft", m_soft)?;
    check_nonneg("beta", beta)?;

    let far = (m_soft + beta) * (m_soft + beta) + m_soft * m_soft;

    Ok((
        (far - 1.0).max(1.0),
        (far * far - 3.0 * far + 1.0).max(1.25),
    ))
}

/// Kanitschar Eq. (4)'s dimension-reduction charge, in bits per pulse.
fn price(w: f64, alphabet: f64) -> f64 {
    let root = sqrt0(w);
    let share = root / (1.0 + root);

    root * alphabet.log2() + (1.0 + root) * h2(share)
}

/// `Gamma(nc + 1, 0) / Gamma(nc + 1, beta)`, the ratio `r >= 1` of Kanitschar Theorem 3.
fn ratio(nc: usize, beta: f64) -> PyResult<f64> {
    let lf = ln_fact(nc);
    let gl = gamma_ladder(2 * nc, beta);
    let low = gl[2 * nc];

    if low <= 0.0 || !low.is_finite() {
        return Err(PyValueError::new_err(format!(
            "beta_test = {beta} leaves Gamma({}, beta_test) outside f64 range at nc = {nc}, so \
             Kanitschar Theorem 3's r is unrepresentable. Move beta_test down toward the cutoff",
            nc + 1
        )));
    }

    Ok(lf[nc].exp() / low)
}

/// The dimension-reduction charge `Delta(w)` in bits per pulse: `sqrt(w) log2(alphabet) +
/// (1 + sqrt(w)) h(sqrt(w) / (1 + sqrt(w)))`, Kanitschar, George, Lin, Upadhyaya &
/// Lutkenhaus, PRX Quantum 4, 040306 (2023), arXiv:2301.08686, Eq. (4), carrying the tighter
/// correction term of Upadhyaya, van Himbeeck & Lutkenhaus, arXiv:2210.14296 over the
/// dimension reduction of Upadhyaya, van Himbeeck, Lin & Lutkenhaus, PRX Quantum 2, 020325
/// (2021), arXiv:2101.05799.
///
/// Never droppable: over a four-symbol key map it is 0.101 bit per pulse at `w = 1e-4` and still
/// 1.34e-2 bit at `w = 1e-6`. `alphabet` is `|Z|`, the key-map size, not the number of signal states.
#[pyfunction]
pub(crate) fn dm_price(w: f64, alphabet: usize) -> PyResult<f64> {
    check_prob("w", w)?;
    check_alphabet(alphabet)?;

    Ok(price(w, alphabet as f64))
}

fn check_alphabet(alphabet: usize) -> PyResult<()> {
    if alphabet >= 2 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "alphabet must be at least 2 key symbols, got {alphabet}: a one-symbol key map \
             carries no key and log2|Z| would report the charge as free"
        )))
    }
}

/// Kanitschar Theorem 3, the noise-robust energy test: `(log2 eps_ET, r, D)` for `k_t` rounds
/// sacrificed to a heterodyne test, at most `l_t` of them OUTSIDE the disc `beta_test`, claiming a
/// weight `w` outside the photon cutoff `nc`.
///
/// Eq. (5): `eps_ET = (l_T + 1) 2^(-k_T D(P_lT || Q_(w/r)))`, `P_j = (1 - j/k_T, j/k_T)`,
/// `Q_y = (1 - y, y)`, `r = Gamma(nc + 1, 0) / Gamma(nc + 1, beta_test)`. The first return is the
/// base-two LOGARITHM: at their `l_T / k_T = 1e-8` and tens of billions of rounds the epsilon
/// underflows f64, and 0.0 reads as unconditional security.
///
/// READ THE EVENT OFF THE PROOF, NOT THE STATEMENT: Appendix A counts rounds OUTSIDE the disc
/// (`V_1` of Eq. (A2)); Theorem 3 as printed counts the inside, a test that passes exactly the
/// states it exists to reject. `beta_test` is a PHOTON NUMBER, `|alpha|^2 >= beta_test`, the
/// reading under which Eq. (A3) gives `r`; Theorem 3's quadrature range `M` and Appendix A's
/// "radius" cannot both be it.
///
/// Does not run the test: rigorous GIVEN it ran and passed on `k_t` random rounds of the block.
#[pyfunction]
pub(crate) fn dm_energy(
    nc: usize,
    beta_test: f64,
    k_t: f64,
    l_t: f64,
    w: f64,
) -> PyResult<(f64, f64, f64)> {
    check_cutoff(nc)?;
    check_pos("beta_test", beta_test)?;
    check_pos("k_t", k_t)?;
    check_nonneg("l_t", l_t)?;
    check_unit("w", w)?;

    if l_t >= k_t {
        return Err(PyValueError::new_err(format!(
            "l_t must be below k_t (got {l_t} >= {k_t}): a test that tolerates every round \
             failing accepts every state and bounds nothing"
        )));
    }

    let r = ratio(nc, beta_test)?;
    let seen = l_t / k_t;
    let claim = w / r;

    if seen >= claim {
        return Err(PyValueError::new_err(format!(
            "Kanitschar Theorem 3 holds only while l_t / k_t < w / r, and here {seen} >= \
             {claim} at r = {r}. Raise w, lower l_t, or move beta_test toward the cutoff to bring \
             r nearer 1"
        )));
    }

    if claim >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "w / r = {claim} must be below 1, where D(P_lT || Q_(w/r)) diverges. Lower w"
        )));
    }

    // In bits because Theorem 3 exponentiates the divergence base two.
    let div = kl_bits(seen, claim);

    Ok(((l_t + 1.0).log2() - k_t * div, r, div))
}

/// The smallest weight `w` compatible with a target energy-test failure probability, by
/// inverting Kanitschar Eq. (5): their Sec. VI B's `w_eps`, the alternative to `w_exp` when a
/// caller fixes `eps_ET` first. `target` is `log2 eps_ET` and is NEGATIVE. Returns the middle
/// term of `w = max(w_exp, w_eps, w_min)` alone.
#[pyfunction]
pub(crate) fn dm_wchoice(
    nc: usize,
    beta_test: f64,
    k_t: f64,
    l_t: f64,
    target: f64,
) -> PyResult<(f64, f64)> {
    check_cutoff(nc)?;
    check_pos("beta_test", beta_test)?;
    check_pos("k_t", k_t)?;
    check_nonneg("l_t", l_t)?;

    if !target.is_finite() || target >= 0.0 {
        return Err(PyValueError::new_err(format!(
            "target must be a finite NEGATIVE log2 eps_ET, got {target}: eps_ET is a \
             probability and a non-negative logarithm is not one"
        )));
    }

    if l_t >= k_t {
        return Err(PyValueError::new_err(format!(
            "l_t must be below k_t (got {l_t} >= {k_t}): a test that tolerates every round \
             failing accepts every state and bounds nothing"
        )));
    }

    let r = ratio(nc, beta_test)?;
    let seen = l_t / k_t;
    let want = ((l_t + 1.0).log2() - target) / k_t;
    let (_, hi) = bisect(seen, 1.0, 160, |claim| kl_bits(seen, claim) < want);
    let w = hi * r;

    if w > 1.0 {
        return Err(PyValueError::new_err(format!(
            "no weight in [0, 1] reaches log2 eps_ET = {target} on {k_t} test rounds at r = \
             {r}: the inversion wants w = {w}. Sacrifice more rounds or accept a larger eps_ET"
        )));
    }

    Ok((w, r))
}

/// Kanitschar Theorem 4's acceptance half-width `mu_X` for one observable: `sqrt(2 x^2 / m
/// ln(2 / eps_AT))`, or `sqrt(x^2 / (2 m) ln(2 / eps_AT))` when `psd` says the operator is
/// positive semidefinite and its range is `[0, x]` rather than `[-x, x]`.
///
/// `x` IS `||X||_inf` AND IS THE CALLER'S: Hoeffding needs a bounded observable and none of
/// `assemble`'s (q, p, n, d, F_Q, F_P, S_Q, S_P) is. Kanitschar clip the detector at a soft limit
/// M (Sec. IV B, Eq. (25)); nothing here clips. `tests` counts rounds on this observable alone.
#[pyfunction]
#[pyo3(signature = (x, tests, eps_at, psd=false))]
pub(crate) fn dm_accept(x: f64, tests: f64, eps_at: f64, psd: bool) -> PyResult<f64> {
    check_pos("x", x)?;
    check_pos("tests", tests)?;
    crate::std::check_eps("eps_at", eps_at)?;

    let span = (2.0 / eps_at).ln();
    let var = if psd {
        x * x / (2.0 * tests)
    } else {
        2.0 * x * x / tests
    };

    Ok((var * span).sqrt())
}

/// Kanitschar Theorem 6's composed security parameter, `eps_EC + max(eps_PA / 2 + epsbar,
/// eps_ET + eps_AT)`.
///
/// Neither a sum (`bb84_length`) nor a plain maximum (`rrdps_finite`). At the paper's
/// demonstration values -- `eps_EC = eps_PA = (1/5) 1e-10`, `epsbar = eps_AT = (7/10) 1e-10`,
/// `eps_ET = (1/10) 1e-10` -- the two branches tie at 8e-11 and the total is exactly 1e-10.
#[pyfunction]
pub(crate) fn dm_secpar(
    eps_ec: f64,
    eps_pa: f64,
    eps_bar: f64,
    eps_et: f64,
    eps_at: f64,
) -> PyResult<f64> {
    crate::std::check_eps("eps_ec", eps_ec)?;
    crate::std::check_eps("eps_pa", eps_pa)?;
    crate::std::check_eps("eps_bar", eps_bar)?;
    crate::std::check_eps("eps_et", eps_et)?;
    crate::std::check_eps("eps_at", eps_at)?;

    Ok(eps_ec + (0.5 * eps_pa + eps_bar).max(eps_et + eps_at))
}

/// Kanitschar Theorem 6 Eq. (9) assembled: `(length, raw, aep, price)` in BITS over the whole
/// block, where `length` is `raw` clamped at zero and floored, `aep` is the equipartition
/// correction `delta(epsbar) = 2 log2(alphabet + 3) sqrt(log2(2 / epsbar) / n_key)` and
/// `price` is `Delta(w)`. `leak` is the WHOLE error-correction cost in bits over the block,
/// verification included.
///
/// `entropy` is `min_{rho in S^(E&A)} H(X|E')` in bits per kept round AND THE CALLER'S, as
/// `dm_relaxed` returns it. `source` has no default: `dm_secure`/`dm_trusted` minimise over
/// Eq. (19)'s set, contained in Eq. (21)'s, and would overstate the key.
#[pyfunction]
#[pyo3(signature = (n_total, n_key, entropy, alphabet, w, eps_bar, eps_pa, leak, source))]
pub(crate) fn dm_length(
    n_total: f64,
    n_key: f64,
    entropy: f64,
    alphabet: usize,
    w: f64,
    eps_bar: f64,
    eps_pa: f64,
    leak: f64,
    source: &str,
) -> PyResult<(f64, f64, f64, f64)> {
    check_pos("n_total", n_total)?;
    check_pos("n_key", n_key)?;
    check_nonneg("entropy", entropy)?;
    check_alphabet(alphabet)?;
    check_prob("w", w)?;
    crate::std::check_eps("eps_bar", eps_bar)?;
    crate::std::check_eps("eps_pa", eps_pa)?;
    check_nonneg("leak", leak)?;

    if n_key > n_total {
        return Err(PyValueError::new_err(format!(
            "n_key must not exceed n_total (got {n_key} > {n_total}): the energy test and the \
             acceptance test are paid for out of the same block the key is drawn from"
        )));
    }

    let cap = (alphabet as f64).log2();

    if entropy > cap {
        return Err(PyValueError::new_err(format!(
            "entropy = {entropy} bits exceeds log2|Z| = {cap} for a {alphabet}-symbol key map. \
             Pass H(X|E') in bits per kept round, not per symbol pair or in nats"
        )));
    }

    match source {
        "relaxed" => (),
        "certified" | "asymptotic" | "dm_secure" | "dm_trusted" => {
            return Err(PyNotImplementedError::new_err(format!(
                "source={source:?} is refused: dm_secure and dm_trusted minimise over \
                 Kanitschar Eq. (19)'s equalities on a normalised state, a set CONTAINED in the \
                 relaxed set S^(E&A) of their Eq. (21), so their minimum is the LARGER and \
                 overstates the key length in Eq. (9). Their observables also differ \
                 ({{q, p, n, d}} against {{n_beta_j, n^2_beta_j}}). Take the entropy from \
                 dm_relaxed and pass source=\"relaxed\""
            )));
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown source {other:?}, expected \"relaxed\" for a minimum taken over \
                 Kanitschar Eq. (21), or \"certified\", \"asymptotic\", \"dm_secure\", \
                 \"dm_trusted\" to read why the other set is refused"
            )));
        }
    }

    let aep = 2.0 * ((alphabet as f64) + 3.0).log2() * ((2.0 / eps_bar).log2() / n_key).sqrt();
    let cost = price(w, alphabet as f64);
    let raw = n_key * (entropy - aep - cost) - leak - 2.0 * (1.0 / eps_pa).log2();

    Ok((raw.max(0.0).floor(), raw, aep, cost))
}

/// Finite-key discrete modulation, REFUSED with one message per bound: `certified = false` for
/// the ANALYTIC `dm_rate`, `true` for the CERTIFIED `dm_secure` / `dm_trusted`. `n_total` is
/// checked first.
#[pyfunction]
pub(crate) fn dm_finite(n_total: f64, certified: bool) -> PyResult<f64> {
    check_pos("n_total", n_total)?;

    if !certified {
        return Err(PyNotImplementedError::new_err(
            "finite-key discrete modulation is refused on the ANALYTIC bound: the blocker is the \
             RECEIVER. Lupo & Ouyang, PRX Quantum 3, 010341 (2022), arXiv:2108.00428, prove it on \
             a heterodyne with outcomes confined to q, p in [-R, R] and digitised into d bins per \
             quadrature (their Sec. III): their Eq. (49) reads d, Eqs. (64) and (65) read R, and \
             Eq. (66) reads P_0(R), the COUNTED out-of-range fraction that also fixes the \
             continuity penalty of their Eqs. (17) and (21). Neither reaches a rate here: \
             q.Link refuses q.Heterodyne.clip by name and a q.ADC is refused by name, both \
             charged as budget noise rows instead, and dm_info integrates an unbounded outcome, \
             so no parameter-estimation epsilon exists (Lupo & Ouyang: Leverrier's GAUSSIAN \
             symmetry route does not transfer to discrete modulation). Their gamma_B, gamma_AB \
             come from the programmes of their Eqs. (40) and (47), feeding Eq. (48); dm_rate \
             bounds the same correlation by Denys, Brown & Leverrier, Quantum 5, 540 (2021), \
             arXiv:2103.13945v3 Eq. (25), on an ideal receiver, and neither paper composes the \
             two. For the certified bound's different blocker call dm_finite with \
             certified = true. Use dm_rate and call it asymptotic",
        ));
    }

    Err(PyNotImplementedError::new_err(
        "finite-key discrete modulation is refused on the CERTIFIED bound, though all four pieces \
         of the length ship. Kanitschar, George, Lin, Upadhyaya & Lutkenhaus, PRX Quantum 4, \
         040306 (2023), arXiv:2301.08686, prove composable finite-size security against i.i.d. \
         COLLECTIVE attacks on the problem dm_secure solves, over the receiver dm_trusted \
         carries. Their Theorem 6 Eq. (9) is dm_length, and its epsilon eps_EC + max(eps_PA/2 + \
         epsbar, eps_ET + eps_AT) is dm_secpar. \
         (1) Delta(w), their Eq. (4), is dm_price: over a four-symbol key map 0.101 bit per \
         pulse at w = 1e-4 and still 1.34e-2 bit at w = 1e-6, so dropping it is wrong, not loose. \
         (2) The energy test, their Theorem 3, is dm_energy, and dm_wchoice its inverse. \
         (3) The Hoeffding acceptance test, their Theorem 4, is dm_accept, with ||X||_inf the \
         caller's: q, p, n, d and F_Q, F_P, S_Q, S_P are unbounded. \
         (4) The relaxed set of their Eq. (21) -- 1 - w <= Tr rho <= 1, Tr P + Tr N <= 2 sqrt(w) \
         -- is dm_relaxed; read it off their Appendix E, since the body's is empty. \
         WHAT REFUSES IS EVERY REMAINING PIECE BEING A PROTOCOL STEP: no q.Link sacrifices k_T \
         rounds to an energy test, splits a block, or clips a detector at a soft limit M. \
         Discharge them by hand: dm_weight and dm_wchoice for w, dm_clip for the norms, \
         dm_accept for the widths, dm_relaxed for the entropy, dm_secpar for the epsilon and \
         dm_length for the bits. Coherent attacks (Bauml, Pascual-Garcia, Wright, Fawzi & Acin, \
         Quantum 8, 1418 (2024), arXiv:2303.09255) need the same binned receiver and cutoff; the \
         general-attack proofs needing none of this are BINARY modulation (Matsuura, Maeda, \
         Sasaki & Koashi, Nat. Commun. 12, 252 (2021), arXiv:2006.04661; Matsuura, Yamano, \
         Kuramochi, Sasaki & Koashi, Quantum 7, 1095 (2023), arXiv:2301.03171), and check_size \
         refuses m = 2. Otherwise use dm_secure or dm_trusted and call it asymptotic",
    ))
}
