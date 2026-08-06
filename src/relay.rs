use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::gaussian::GaussianState;
use crate::std::{
    check_eps, check_finite, check_gate, check_nonneg, check_pos, check_prob, check_unit, click_p,
    sqrt0,
};

// Untrusted-relay topologies: two senders, one Eve-owned measurement node; no `trusted` variant.
// SNU (vacuum 1), xpxp; gaussian.rs is hbar = 1, so states halve in and the 4x4 doubles out.
//
// Asymptotic layer: Pirandola, Ottaviani, Spedalieri, Weedbrook, Braunstein, Lloyd, Gehring,
// Jacobsen & Andersen, Nature Photonics 9, 397 (2015), arXiv:1312.4104, for the asymmetric relay,
// the post-relay covariance and the large-modulation bound; Ottaviani, Spedalieri, Braunstein &
// Pirandola, "Continuous-variable quantum cryptography with an untrusted relay: Detailed security
// analysis of the symmetric configuration", Phys. Rev. A 91, 022320 (2015), arXiv:1506.05430, for
// the symmetric configuration and the two-mode correlated attack (g, g').
//
// Finite size: Papanastasiou, Ottaviani & Pirandola, "Finite size analysis of measurement device
// independent quantum cryptography with continuous variables", Phys. Rev. A 96, 042332 (2017),
// arXiv:1707.04599, hereafter POP17. Zhang, Zhang, Zhao, Wang, Yu & Guo, Phys. Rev. A 96, 042334
// (2017), arXiv:1707.05931, is an independent finite-size CV-MDI analysis published beside it and
// is NOT what runs here.
//
// THE THREE RELAY NUMBERS ARE THREE QUANTITIES AND NONE BOUNDS ANOTHER. `cvmdi_rate` is the
// large-modulation security bound at the Shannon limit; `beta` never reaches it. `cvmdi_point` is
// POP17's K^inf, one named Eve at finite `va` and the configured `beta`, never a security claim.
// `cvmdi_finite` is POP17 Eq. (23), `cvmdi_point`'s arithmetic at the pessimistic end of four
// confidence intervals, a bound only against TWO-MODE GAUSSIAN attacks.

/// Relative arm gap below which the symmetric branch runs: the asymmetric logs cancel as arms meet.
const SEAM: f64 = 1e-9;

/// von Neumann entropy (bits) at symplectic eigenvalue x, 0 for x <= 1. Grouped as b*log2(1 + 1/b)
/// + log2((x+1)/2) with b = (x-1)/2: the plain difference cancels nine digits at large x.
fn h_ent(x: f64) -> f64 {
    if !(x > 1.0) {
        return 0.0;
    }

    let b = (x - 1.0) / 2.0;

    b * (2.0 / (x - 1.0)).ln_1p() / std::f64::consts::LN_2 + ((x + 1.0) / 2.0).log2()
}

/// Modulation-variance ceiling: the conditioning differences scale as va^2 and spend the mantissa.
const VA_MAX: f64 = 1e5;

fn check_va(x: f64) -> PyResult<()> {
    check_pos("va", x)?;
    if x <= VA_MAX {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "va must be <= {VA_MAX:e}, got {x}: the conditional blocks cancel \
             catastrophically and the rate exceeds its asymptotic bound. Use \
             cvmdi_rate for the large-modulation limit"
        )))
    }
}

/// Environment variance omega of a thermal-loss arm, omega = 1 + tau*xi/(1 - tau), with xi
/// referred to the CHANNEL INPUT. tau = 1 has no environment mode, so it needs xi = 0.
#[pyfunction]
pub(crate) fn relay_omega(tau: f64, xi: f64) -> PyResult<f64> {
    check_unit("tau", tau)?;
    check_nonneg("xi", xi)?;

    if tau >= 1.0 {
        if xi > 0.0 {
            return Err(PyValueError::new_err(format!(
                "a lossless arm (tau = 1) has no environment to carry xi = {xi}"
            )));
        }

        return Ok(1.0);
    }

    Ok(1.0 + tau * xi / (1.0 - tau))
}

/// Equivalent noise of two PURE-LOSS arms, `2*(tau_a + tau_b)/(tau_a*tau_b)`. A reference point,
/// not a lower bound: correlated environments push chi below it. The lower bound is `cvmdi_least`.
#[pyfunction]
pub(crate) fn cvmdi_floor(ta: f64, tb: f64) -> PyResult<f64> {
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;

    Ok(2.0 * (ta + tb) / (ta * tb))
}

/// Smallest equivalent noise two arms can carry, `(tau_a + tau_b)^2/(tau_a*tau_b)`, where the
/// relay noise term lambda vanishes. Exactly 4 for equal arms, the symmetric rate's pole.
#[pyfunction]
pub(crate) fn cvmdi_least(ta: f64, tb: f64) -> PyResult<f64> {
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;

    Ok(chi_least(ta, tb))
}

fn chi_least(ta: f64, tb: f64) -> f64 {
    (ta + tb) * (ta + tb) / (ta * tb)
}

/// Equivalent noise chi of the two arms; `g`/`gp` = 0 is two independent entangling cloners, and
/// `chi - cvmdi_floor` may be negative for a correlated attack. chi is a geometric mean over both
/// quadratures: a named attack goes through `cvmdi_point`.
#[pyfunction]
pub(crate) fn cvmdi_noise(ta: f64, tb: f64, wa: f64, wb: f64, g: f64, gp: f64) -> PyResult<f64> {
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_omega("omega_a", wa)?;
    check_omega("omega_b", wb)?;
    check_corr(wa, wb, g, gp)?;

    let (lam, lamp) = lambdas(ta, tb, wa, wb, g, gp);
    let sum = ta + tb;

    Ok(sum / (ta * tb) * sqrt0((sum + lam) * (sum + lamp)))
}

fn check_omega(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && x >= 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} is an environment variance and must be >= 1 (vacuum), got {x}"
        )))
    }
}

/// Eve's environment pair is itself a state: |g|, |gp| < sqrt(omega_a*omega_b)
/// and its least symplectic eigenvalue at or above the vacuum.
fn check_corr(wa: f64, wb: f64, g: f64, gp: f64) -> PyResult<()> {
    check_finite("g", g)?;
    check_finite("gp", gp)?;

    let cap = (wa * wb).sqrt();
    if g.abs() >= cap || gp.abs() >= cap {
        return Err(PyValueError::new_err(format!(
            "|g|, |gp| must be < sqrt(omega_a*omega_b) = {cap}, got {g} and {gp}"
        )));
    }

    let det = (wa * wb - g * g) * (wa * wb - gp * gp);
    let delta = wa * wa + wb * wb + 2.0 * g * gp;
    let nu2 = (delta - sqrt0(delta * delta - 4.0 * det)) / 2.0;
    if nu2 < 1.0 - 1e-9 {
        return Err(PyValueError::new_err(format!(
            "g = {g}, gp = {gp} violate the uncertainty principle for Eve's \
             environment pair: least symplectic eigenvalue {} < 1",
            sqrt0(nu2)
        )));
    }

    Ok(())
}

/// Per-quadrature noise combinations (lambda, lambda') = (kappa - u*g, kappa + u*gp).
fn lambdas(ta: f64, tb: f64, wa: f64, wb: f64, g: f64, gp: f64) -> (f64, f64) {
    let kappa = (1.0 - ta) * wa + (1.0 - tb) * wb;
    let u = 2.0 * ((1.0 - ta) * (1.0 - tb)).sqrt();

    (kappa - u * g, kappa + u * gp)
}

/// Asymptotic CV relay rate, bits per relay use, Alice encoding; not symmetric in the arms.
/// A security bound in the large-V_A limit at the Shannon limit; `beta` never reaches it.
/// `cvmdi_point` is the per-attack number, and neither of the two bounds the other.
/// Negatives are returned as they come; the Python layer clamps.
#[pyfunction]
pub(crate) fn cvmdi_rate(ta: f64, tb: f64, chi: f64) -> PyResult<f64> {
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_pos("chi", chi)?;

    let sum = ta + tb;
    let gap = (ta - tb).abs();

    // The symmetric log has a pole at chi = 4 and chi_least is exactly 4 for equal arms, so one
    // bound closes both domains; the max covers near-equal arms where chi_least dips below 4.
    let least = if gap <= SEAM * sum {
        chi_least(ta, tb).max(4.0)
    } else {
        chi_least(ta, tb)
    };
    if !(chi > least) {
        return Err(PyValueError::new_err(format!(
            "chi = {chi} is at or below {least}, where these arms' noise term \
             vanishes: no relay covariance exists there"
        )));
    }

    if gap <= SEAM * sum {
        return Ok(h_ent(chi / 2.0 - 1.0)
            + (16.0 / (std::f64::consts::E * std::f64::consts::E * chi * (chi - 4.0))).log2());
    }

    Ok(h_ent(ta * chi / sum - 1.0) - h_ent((ta * tb * chi - sum * sum) / (gap * sum))
        + (2.0 * sum / (std::f64::consts::E * gap * chi)).log2())
}

/// Post-relay covariance of the senders' retained modes (Alice's, Bob's): 4x4, xpxp, row-major,
/// SHOT-NOISE units (vacuum 1), `va` per quadrature. Halved into the `hbar = 1` Gaussian layer and
/// doubled back out.
///
/// ATTACK-SIDE, NOT BOUND-SIDE: finite `va`, ONE named Eve, never a security claim. Two attacks can
/// share `chi`, and so `cvmdi_rate`, TO THE LAST BIT while their `cvmdi_point` rates differ.
#[pyfunction]
pub(crate) fn cvmdi_cov(
    va: f64,
    ta: f64,
    tb: f64,
    wa: f64,
    wb: f64,
    g: f64,
    gp: f64,
) -> PyResult<Vec<f64>> {
    check_va(va)?;
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_omega("omega_a", wa)?;
    check_omega("omega_b", wb)?;
    check_corr(wa, wb, g, gp)?;

    let mu = va + 1.0;
    let s = sqrt0(mu * mu - 1.0);
    let u = ((1.0 - ta) * (1.0 - tb)).sqrt();

    // Modes: a (Alice keeps), A (Alice sends), B (Bob sends), b (Bob keeps).
    // Shot-noise units, halved below for the Gaussian layer.
    let mut cov = vec![0.0; 64];
    let diag = [
        mu,
        mu,
        ta * mu + (1.0 - ta) * wa,
        ta * mu + (1.0 - ta) * wa,
        tb * mu + (1.0 - tb) * wb,
        tb * mu + (1.0 - tb) * wb,
        mu,
        mu,
    ];
    for (i, d) in diag.iter().enumerate() {
        cov[i * 8 + i] = *d;
    }

    // Two-mode squeezing correlates x with x and p with -p.
    let pairs = [
        (0, 2, ta.sqrt() * s),
        (1, 3, -ta.sqrt() * s),
        (4, 6, tb.sqrt() * s),
        (5, 7, -tb.sqrt() * s),
        (2, 4, u * g),
        (3, 5, u * gp),
    ];
    for (i, j, v) in pairs {
        cov[i * 8 + j] = v;
        cov[j * 8 + i] = v;
    }

    for x in cov.iter_mut() {
        *x *= 0.5;
    }

    let st = GaussianState::from_moments(vec![0.0; 8], cov)?;

    // Sum port measures p, difference port q; the conditional covariance ignores both outcomes.
    let st = st.bs(1, 2, 0.5)?;
    let st = st.condition(1, std::f64::consts::FRAC_PI_2, 0.0)?;
    let st = st.condition(1, 0.0, 0.0)?;

    // Not re-admitted through `from_moments`: `bs` and `condition` carry the bona fide condition.
    Ok(st.cov().into_iter().map(|x| 2.0 * x).collect())
}

/// Determinant of the 2x2 block of the 4x4 row-major `v` at rows `r`, cols `c`.
fn det2(v: &[f64], r: usize, c: usize) -> f64 {
    v[r * 4 + c] * v[(r + 1) * 4 + c + 1] - v[r * 4 + c + 1] * v[(r + 1) * 4 + c]
}

/// Symplectic spectrum of a two-mode covariance in shot-noise units.
fn spectrum(v: &[f64]) -> (f64, f64) {
    let delta = det2(v, 0, 0) + det2(v, 2, 2) + 2.0 * det2(v, 0, 2);
    let mut full = 0.0;

    // 4x4 determinant by cofactor expansion along the first row.
    for c in 0..4 {
        let mut minor = [0.0; 9];
        let mut k = 0;
        for i in 1..4 {
            for j in 0..4 {
                if j != c {
                    minor[k] = v[i * 4 + j];
                    k += 1;
                }
            }
        }
        let d3 = minor[0] * (minor[4] * minor[8] - minor[5] * minor[7])
            - minor[1] * (minor[3] * minor[8] - minor[5] * minor[6])
            + minor[2] * (minor[3] * minor[7] - minor[4] * minor[6]);
        full += if c % 2 == 0 { v[c] * d3 } else { -v[c] * d3 };
    }

    let disc = sqrt0(delta * delta - 4.0 * full);

    (sqrt0((delta + disc) / 2.0), sqrt0((delta - disc) / 2.0))
}

/// CV relay rate at FINITE modulation: `(i_ab, chi_e, key)`, `key = beta*i_ab - chi_e`.
/// Both senders heterodyne; Eve's bound is the Holevo quantity on Alice's variable.
/// Per-attack at finite `va` and the configured `beta`, never a security bound: helping
/// correlations put it above `cvmdi_rate`. Not clamped.
#[pyfunction]
pub(crate) fn cvmdi_point(
    va: f64,
    ta: f64,
    tb: f64,
    wa: f64,
    wb: f64,
    g: f64,
    gp: f64,
    beta: f64,
) -> PyResult<(f64, f64, f64)> {
    check_beta(beta)?;
    let v = cvmdi_cov(va, ta, tb, wa, wb, g, gp)?;

    rate_core(&v, beta)
}

fn check_beta(beta: f64) -> PyResult<()> {
    if !(beta.is_finite() && (0.0..=1.0).contains(&beta)) {
        return Err(PyValueError::new_err(format!(
            "beta must be in [0, 1], got {beta}"
        )));
    }

    Ok(())
}

/// `(i_ab, chi_e, key)` from a post-relay 4x4 covariance in SNU.
fn rate_core(v: &[f64], beta: f64) -> PyResult<(f64, f64, f64)> {
    let (n1, n2) = spectrum(v);

    // Bob's mode after Alice heterodynes: B - C^T (A + I)^-1 C, the I being her splitter's vacuum.
    let ai = [v[0] + 1.0, v[1], v[4], v[5] + 1.0];
    let dai = ai[0] * ai[3] - ai[1] * ai[2];
    if !(dai > 0.0) {
        return Err(PyValueError::new_err(
            "Alice's heterodyne covariance is singular: no rate is defined".to_string(),
        ));
    }

    let inv = [ai[3] / dai, -ai[1] / dai, -ai[2] / dai, ai[0] / dai];
    let mut bc = [0.0; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0;
            for k in 0..2 {
                for l in 0..2 {
                    acc += v[k * 4 + 2 + i] * inv[k * 2 + l] * v[l * 4 + 2 + j];
                }
            }
            bc[i * 2 + j] = v[(2 + i) * 4 + 2 + j] - acc;
        }
    }

    let db = (v[10] + 1.0) * (v[15] + 1.0) - v[11] * v[14];
    let dbc = (bc[0] + 1.0) * (bc[3] + 1.0) - bc[1] * bc[2];
    if !(db > 0.0 && dbc > 0.0) {
        return Err(PyValueError::new_err(
            "Bob's heterodyne covariance is singular: no rate is defined".to_string(),
        ));
    }

    let i_ab = 0.5 * (db / dbc).log2();
    let chi_e = h_ent(n1) + h_ent(n2) - h_ent(sqrt0(bc[0] * bc[3] - bc[1] * bc[2]));

    Ok((i_ab, chi_e, beta * i_ab - chi_e))
}

// Finite size, POP17. The parties estimate the relay-output noise pair of its Eq. (10),
// `V_{q,N} = 1 + V_q` and `V_{p,N} = 1 + V_p`; by its Eq. (24) the post-relay covariance depends on
// Eve's four environment parameters ONLY through that pair.

/// Refuses a (v_q, v_p) no two-mode Gaussian environment produces: `2*(1 + v) > ta + tb` per
/// quadrature. STRICTER than `cvmdi_least` on the geometric mean: it refuses some `chi` that
/// `cvmdi_rate` admits.
fn check_reach(ta: f64, tb: f64, vq: f64, vp: f64) -> PyResult<()> {
    let sum = ta + tb;
    for (name, v) in [("v_q", vq), ("v_p", vp)] {
        check_finite(name, v)?;
        if !(2.0 * (1.0 + v) > sum) {
            return Err(PyValueError::new_err(format!(
                "{name} = {v} is at or below {}, where these arms' noise term vanishes: no \
                 two-mode Gaussian environment drives 2*(1 + {name}) to tau_a + tau_b = {sum} \
                 (POP17 Eq. (10)). This is cvmdi_least's edge per quadrature; re-estimate {name} \
                 or the arm transmissivities",
                0.5 * sum - 1.0
            )));
        }
    }

    Ok(())
}

/// Per-quadrature excess noise at the relay's two outputs, `(v_q, v_p)` in SNU, from the arms and
/// Eve's environment. POP17 Eqs. (11)-(13): `v_q = k - g*u`, `v_p = k + g'*u` with
/// `k = [(1 - tau_b)*(omega_b - 1) + (1 - tau_a)*(omega_a - 1)]/2` and
/// `u = sqrt((1 - tau_a)*(1 - tau_b))`.
///
/// `chi = 2*(tau_a + tau_b)/(tau_a*tau_b) * sqrt((1 + v_q)*(1 + v_p))`. A helping correlation
/// makes one NEGATIVE: the attack, not an error.
#[pyfunction]
pub(crate) fn cvmdi_excess(
    ta: f64,
    tb: f64,
    wa: f64,
    wb: f64,
    g: f64,
    gp: f64,
) -> PyResult<(f64, f64)> {
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_omega("omega_a", wa)?;
    check_omega("omega_b", wb)?;
    check_corr(wa, wb, g, gp)?;

    let (lam, lamp) = lambdas(ta, tb, wa, wb, g, gp);
    let base = 2.0 - ta - tb;

    Ok((0.5 * (lam - base), 0.5 * (lamp - base)))
}

/// Post-relay covariance in the ESTIMATED coordinates, POP17 Eqs. (24)-(26), 4x4 row-major xpxp
/// in SNU; arguments already checked and `ta`, `tb` already clamped into [0, 1].
fn gauss_cov(va: f64, ta: f64, tb: f64, vq: f64, vp: f64) -> Vec<f64> {
    let phi = (ta + tb) * va + 2.0 + 2.0 * vq;
    let phip = (ta + tb) * va + 2.0 + 2.0 * vp;
    let mu = va + 1.0;
    let sq = va * (va + 2.0);
    let off = (ta * tb).sqrt();

    vec![
        mu - sq * ta / phi,
        0.0,
        sq * off / phi,
        0.0,
        0.0,
        mu - sq * ta / phip,
        0.0,
        -sq * off / phip,
        sq * off / phi,
        0.0,
        mu - sq * tb / phi,
        0.0,
        0.0,
        -sq * off / phip,
        0.0,
        mu - sq * tb / phip,
    ]
}

/// `(i_ab, chi_e, key)` at finite modulation from the arms and the two ESTIMATED relay-output
/// excess noises, POP17's `K^inf` of Eqs. (3), (6), (7) over the covariances of its Eqs. (24)-(26).
/// `cvmdi_point` in the measured coordinates (`cvmdi_excess` converts). Not clamped.
///
/// STILL AN ATTACK-SIDE NUMBER at one (v_q, v_p), never a security claim.
#[pyfunction]
pub(crate) fn cvmdi_gauss(
    va: f64,
    ta: f64,
    tb: f64,
    vq: f64,
    vp: f64,
    beta: f64,
) -> PyResult<(f64, f64, f64)> {
    check_va(va)?;
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_beta(beta)?;
    check_reach(ta, tb, vq, vp)?;

    rate_core(&gauss_cov(va, ta, tb, vq, vp), beta)
}

/// Standard deviations of POP17's four estimators, arguments already checked.
fn widths(m: f64, va: f64, ta: f64, tb: f64, vq: f64, vp: f64) -> (f64, f64, f64, f64) {
    let (vqn, vpn) = (1.0 + vq, 1.0 + vp);
    let arm = |near: f64, far: f64, vn: f64| {
        let mix = near + 0.5 * far;

        8.0 * near / m * mix * (1.0 + vn / (mix * va))
    };
    let opt = |q: f64, p: f64| sqrt0(q * p / (q + p));
    let root = (2.0 / m).sqrt();

    (
        opt(arm(ta, tb, vqn), arm(ta, tb, vpn)),
        opt(arm(tb, ta, vqn), arm(tb, ta, vpn)),
        root * vqn,
        root * vpn,
    )
}

/// Standard deviations of the four channel estimators from `m` estimation ROUNDS,
/// `(sigma_a, sigma_b, s_q, s_p)`. POP17 Eqs. (14)-(16) for the transmissivities and Eqs. (18),
/// (20) for the noises:
///
/// `Var(tau_A) = Var_q*Var_p/(Var_q + Var_p)`, `Var_x = 8*tau_A/m*(tau_A + tau_B/2)*
/// [1 + V_{x,N}/((tau_A + tau_B/2)*V_M)]`, and `s_x = sqrt(2/m)*V_{x,N}` with `V_{x,N} = 1 + v_x`.
///
/// `m` counts ROUNDS, not quadratures: no factor of two. POP17 print only tau_A; `sigma_b` swaps
/// A <-> B in BOTH places. The transmissivity estimator is only asymptotically unbiased (bias 1/m,
/// width 1/sqrt(m)); POP17 take `m > 1e5`.
#[pyfunction]
pub(crate) fn cvmdi_width(
    m: f64,
    va: f64,
    ta: f64,
    tb: f64,
    vq: f64,
    vp: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_pos("m", m)?;
    check_va(va)?;
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_reach(ta, tb, vq, vp)?;

    Ok(widths(m, va, ta, tb, vq, vp))
}

/// Finite-size CV-MDI key rate, bits per relay use, POP17 Eq. (23):
///
/// `K = (n/N)*(K^inf(beta, va, tau_a^low, tau_b^low, v_q^up, v_p^up) - Delta(n))`
///
/// with `m = pe_fraction*N` estimation rounds, `n = N - m` key rounds, and the pessimistic values
/// of Eqs. (21)-(22) taken `conf` standard deviations out. Returns
/// `(key, tau_a_low, tau_b_low, v_q_up, v_p_up, delta, n)`. NOT clamped.
///
/// `conf` IS NOT AN EPSILON: POP17's coefficient in standard deviations (they run 6.5 beside a
/// quoted 1e-10), four intervals at one coefficient, no union bound. `eps_smooth` and `eps_pa`
/// enter `Delta(n)` alone; no correctness term, no composed security parameter.
///
/// `Delta(n)` IS NOT POP17'S OWN EXPRESSION. They print only `Delta(n) ~ sqrt(log2(2/eps_pa)/n)`
/// and refer to Leverrier, Grosshans & Grangier,
/// "Finite-size analysis of continuous-variable quantum key distribution", Phys. Rev. A 81, 062343
/// (2010); this runs `keyrate::cv_finite`'s two-term penalty at `dim H_X = 2`.
///
/// All of `K^inf`, `I_AB` included, is taken at the pessimistic point: POP17's instantiation, NOT
/// `cv_finite`'s, which moves only the Holevo bound.
#[pyfunction]
pub(crate) fn cvmdi_finite(
    va: f64,
    ta: f64,
    tb: f64,
    vq: f64,
    vp: f64,
    beta: f64,
    n_total: f64,
    pe_fraction: f64,
    conf: f64,
    eps_smooth: f64,
    eps_pa: f64,
    attack: &str,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
    check_va(va)?;
    check_unit("tau_a", ta)?;
    check_unit("tau_b", tb)?;
    check_beta(beta)?;
    check_reach(ta, tb, vq, vp)?;
    check_pos("n_total", n_total)?;
    check_eps("pe_fraction", pe_fraction)?;
    check_pos("conf", conf)?;
    check_eps("eps_smooth", eps_smooth)?;
    check_eps("eps_pa", eps_pa)?;
    check_class(attack)?;

    let m = pe_fraction * n_total;
    let n_key = n_total - m;
    if !(m > 0.0 && n_key > 0.0) {
        return Err(PyValueError::new_err(format!(
            "pe_fraction = {pe_fraction} of n_total = {n_total} leaves {m} rounds for estimation \
             and {n_key} for the key: POP17's split needs both sides positive; raise n_total"
        )));
    }

    let (sa, sb, sq, sp) = widths(m, va, ta, tb, vq, vp);

    // Only the transmissivity input is clamped, at 0, the pessimistic end; never the key.
    let ta_low = (ta - conf * sa).clamp(0.0, ta);
    let tb_low = (tb - conf * sb).clamp(0.0, tb);
    let (vq_up, vp_up) = (vq + conf * sq, vp + conf * sp);
    let delta = 7.0 * ((2.0 / eps_smooth).log2() / n_key).sqrt()
        + (2.0 / n_key) * (1.0 / eps_pa).log2();

    let v = gauss_cov(va, ta_low, tb_low, vq_up, vp_up);
    let (_, _, worst) = rate_core(&v, beta)?;
    let key = n_key / n_total * (worst - delta);

    Ok((key, ta_low, tb_low, vq_up, vp_up, delta, n_key))
}

/// The attack class `cvmdi_finite` is stated against; the two refusals are different failures.
fn check_class(attack: &str) -> PyResult<()> {
    match attack {
        "gaussian" => Ok(()),
        "collective" => Err(PyNotImplementedError::new_err(
            "attack=\"collective\" is refused: this is Papanastasiou, Ottaviani & Pirandola's \
             TWO-MODE GAUSSIAN result, and the gap to a general collective attack sits in the \
             ESTIMATORS rather than in the rate: their Appendix C widths, Eqs. (14) to (20), \
             assume jointly NORMAL samples, and Gaussian extremality prices the rate, not an \
             estimator. The proof without that assumption is Lupo, Ottaviani, Papanastasiou & \
             Pirandola, Phys. Rev. A 97, 052327 (2018), arXiv:1704.07924, and does not ship. \
             Pass attack=\"gaussian\"",
        )),
        "coherent" => Err(PyNotImplementedError::new_err(
            "attack=\"coherent\" is refused: Papanastasiou, Ottaviani & Pirandola state \"The \
             validity of the described analysis is in fact restricted to the case of Gaussian \
             attacks\". The coherent-attack proof is Lupo, Ottaviani, Papanastasiou & Pirandola, \
             Phys. Rev. A 97, 052327 (2018), arXiv:1704.07924, a different calculation needing \
             (1) the ENERGY TEST and ACTIVE SYMMETRIZATION of Leverrier, Phys. Rev. Lett. 118, \
             200501 (2017), neither of which qkd runs; (2) the composable epsilon of Leverrier, \
             Phys. Rev. Lett. 114, 070501 (2015); (3) its own confidence intervals in place of \
             POP17's Eqs. (14) to (22). Pass attack=\"gaussian\" and call the number a \
             Gaussian-attack bound",
        )),
        other => Err(PyValueError::new_err(format!(
            "unknown attack class {other:?}, expected \"gaussian\", \"collective\" or \
             \"coherent\""
        ))),
    }
}

// `relay_ports`, `relay_counts`, `relay_error` are the phase-RESOLVED substrate under
// `mode_overlap`: no key rate reads `relay_error`, and MDI-BB84 would price one imperfection twice.

/// Intensity overlap |<psi_a|psi_b>|^2 of two Gaussian temporal modes of intensity FWHM `fwhm_a`,
/// `fwhm_b` separated by `offset`, in the caller's time unit. A first-order fringe contrast is its
/// SQUARE ROOT; a Hong-Ou-Mandel visibility is proportional to it.
#[pyfunction]
pub(crate) fn mode_overlap(fwhm_a: f64, fwhm_b: f64, offset: f64) -> PyResult<f64> {
    check_pos("fwhm_a", fwhm_a)?;
    check_pos("fwhm_b", fwhm_b)?;
    check_finite("offset", offset)?;

    // Intensity FWHM to standard deviation.
    let scale = 2.0 * (2.0 * std::f64::consts::LN_2).sqrt();
    let sa = fwhm_a / scale;
    let sb = fwhm_b / scale;
    let sq = sa * sa + sb * sb;

    Ok(2.0 * sa * sb / sq * (-offset * offset / (2.0 * sq)).exp())
}

/// Hong-Ou-Mandel dip visibility of two INDEPENDENT weak coherent sources. Ceiling 1/2 -- a source
/// property, not an alignment one -- so thresholds for it live on a [0, 1/2] scale.
#[pyfunction]
pub(crate) fn hom_visibility(overlap: f64, mu_a: f64, mu_b: f64) -> PyResult<f64> {
    check_prob("overlap", overlap)?;
    check_pos("mu_a", mu_a)?;
    check_pos("mu_b", mu_b)?;

    let sum = mu_a + mu_b;

    Ok(2.0 * mu_a * mu_b * overlap / (sum * sum))
}

/// Mean photon numbers at the relay beamsplitter's two output ports for arriving `mu_a`, `mu_b`,
/// relative phase `delta` and first-order fringe `contrast` -- the AMPLITUDE overlap, sqrt of
/// `mode_overlap`. The ports sum to `mu_a + mu_b` whatever the mode match.
#[pyfunction]
pub(crate) fn relay_ports(mu_a: f64, mu_b: f64, delta: f64, contrast: f64) -> PyResult<(f64, f64)> {
    check_pos("mu_a", mu_a)?;
    check_pos("mu_b", mu_b)?;
    check_finite("delta", delta)?;
    check_prob("contrast", contrast)?;

    let flux = (mu_a + mu_b) / 2.0;
    let cross = (mu_a * mu_b).sqrt() * delta.cos() * contrast;

    Ok((flux + cross, flux - cross))
}

/// Relay threshold-detector outcomes `(single, double)`: exactly one detector fired, and both did.
/// `single` is the announceable event, `double` the coincidence a rate formula discards.
#[pyfunction]
pub(crate) fn relay_counts(mu_c: f64, mu_d: f64, eta: f64, dark: f64) -> PyResult<(f64, f64)> {
    check_nonneg("mu_c", mu_c)?;
    check_nonneg("mu_d", mu_d)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    let pc = click_p(eta, mu_c, dark);
    let pd = click_p(eta, mu_d, dark);

    Ok((pc * (1.0 - pd) + pd * (1.0 - pc), pc * pd))
}

/// Bit error rate of a relay announcement: the chance the detector that fired is the one the
/// senders' phases said should stay dark. (1 - contrast)/2 in the weak-flux limit, plus dark.
#[pyfunction]
pub(crate) fn relay_error(
    mu_a: f64,
    mu_b: f64,
    contrast: f64,
    eta: f64,
    dark: f64,
) -> PyResult<f64> {
    let (mu_c, mu_d) = relay_ports(mu_a, mu_b, 0.0, contrast)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    let pc = click_p(eta, mu_c, dark);
    let pd = click_p(eta, mu_d, dark);
    let right = pc * (1.0 - pd);
    let wrong = pd * (1.0 - pc);
    let single = right + wrong;

    // Saturated detectors announce nothing distinguishable.
    if !(single > 0.0) {
        return Ok(0.5);
    }

    Ok(wrong / single)
}
