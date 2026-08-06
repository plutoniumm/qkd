use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::decoy::hoeff;
use crate::numeric::bisect;
use crate::std::{
    check_eps, check_err, check_fec, check_nonneg, check_pos, check_prob, check_unit, h2, sqrt0, E_CAP,
};

// SARG04: the four BB84 states, sifted on announced non-orthogonal PAIRS. Scarani, Acin, Ribordy
// & Gisin, Phys. Rev. Lett. 92, 057901 (2004). The quantum layer is BB84's verbatim; sifting is
// 1/4 and RISES with misalignment. The unconditional rate is Fung, Tamaki & Lo, Phys. Rev. A 73,
// 012337 (2006), arXiv:quant-ph/0510025, Eq. (39), on the Bell-diagonal relations their Theorems
// 1 and 2 (Eqs. (9) and (10)) prove, after Tamaki & Lo, Phys. Rev. A 73, 010302(R) (2006); the
// detector model, yields and gains are their Eqs. (40)-(43). `sarg_ceiling` is a different
// quantity from a different paper.
//
// Past `E1_DEAD = 1/3` and `E2_DEAD = 1/6`, `1 - H(Z1|X1)` and `1 - h2(e_p2)` turn back UPWARD.
// Neither cutoff refuses: each term credits zero there; `sarg_domain` shows the crossing.
//
// `decoy_bounds`'s `(y1_lo, e1_hi)` and `decoy_eta` transfer from `decoy.rs`; `decoy_gain` and
// `decoy_ideal` do NOT (BB84 yield law): use `sarg_yield` and `sarg_gain`.
//
// Three intensities give `min Y2 = 0`, so the shipped rate sits below Eq. (39)'s. A fourth buys it:
// Yin, Fu, Mao & Chen, Sci. Rep. 6, 29482 (2016), arXiv:1607.02366, Eqs. (30) and (31), asymptotic.
//
// `sarg_length` is Nian, Nie, Zhang & Lu, Commun. Theor. Phys. 76, 065101 (2024), Eq. (5): Rusca,
// Boaron, Grunenfelder, Martin & Zbinden, Appl. Phys. Lett. 112, 171104 (2018), arXiv:1801.03443,
// Eq. (1) with `1 - h2(phi)` replaced by `1 - H(Z1|X1)`, so `sarg_counts` and `sarg_errors` are
// Rusca's one-decoy estimators, Eqs. (A14), (A16), (A17), (A19) and (A22). Nian has NO arXiv
// version and was read as the publisher PDF. NOT implemented: Wang, Corrigan & Lutkenhaus,
// arXiv:2603.22448, Eq. (3), numerical, no decoy layer.
//
// DEPARTS FROM NIAN, safe direction: Nian Eq. (6) subtracts `s0_hi` where Rusca Eq. (A17) subtracts
// `s0_hi / tau_0`; the printed key is 7.3% to 7.9% longer at zero distance, 5.0 km to 6.0 km
// further on Table 1 (`test_departure_costs`). `sarg_counts` runs Rusca's.

/// Where `sarg_phase` reaches 1/2: Eq. (10) puts the root of `e_p2(e) = 1/2` at `w = 1/3 - e = 1/6`.
const E2_DEAD: f64 = 1.0 / 6.0;

/// Where `H(Z1|X1)` reaches one bit and `(3/2) e1` reaches 1/2.
const E1_DEAD: f64 = 1.0 / 3.0;

/// Nian Eq. (13), `eps_sec = 2*(2*a1 + a2 + a3) + V + 4*eps_1 + 5*eps_2` at one common value:
/// 8 + 1 + 4 + 5 = 18. NOT 21 (Lim Eq. (B4), `discrete.rs`) and NOT 19 (Rusca Eq. (A24)).
const EPS_TERMS: f64 = 18.0;

/// Four-outcome Shannon entropy, bits; non-positive weights contribute 0, not NaN.
fn h4(p: [f64; 4]) -> f64 {
    p.iter()
        .map(|&x| if x > 0.0 { -x * x.log2() } else { 0.0 })
        .sum()
}

/// Root of a DECREASING `f` on `[lo, hi]`.
fn root(lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let (a, b) = bisect(lo, hi, 100, |x| f(x) > 0.0);

    0.5 * (a + b)
}

/// Bell-diagonal weights `(p_I, p_X, p_Y, p_Z)` at bit error `e1`, Fung, Tamaki & Lo Eq. (9),
/// `e1/2 <= a <= e1`. Worst case `a = (3/2) e1^2` clamped, not an endpoint: their `a = e1/2`
/// UNDERSTATES the cost above `e1 = 1/3`.
fn bell(e1: f64) -> [f64; 4] {
    let a = (1.5 * e1 * e1).max(0.5 * e1).min(e1);
    let (px, py, pz) = (e1 - a, a, 1.5 * e1 - a);

    [1.0 - px - py - pz, px, py, pz]
}

/// `sarg_phase` unguarded, for `sarg_limit`'s bisection.
fn phase2(e2: f64) -> f64 {
    if e2 >= E2_DEAD {
        return E_CAP;
    }

    let w = 1.0 / 3.0 - e2;

    0.5 + sqrt0(1.5 * (1.0 - 9.0 * w * w)) / 6.0 - 0.75 * std::f64::consts::SQRT_2 * w
}

/// `H(Z1|X1)` at Eq. (9)'s worst case, uncapped, for `sarg_limit`; `charged` is what the rate pays.
fn cost1(e1: f64) -> f64 {
    h4(bell(e1)) - h2(e1)
}

/// `cost1` below `E1_DEAD`, one full bit at or above it, so the term buys exactly zero there.
fn charged(e1: f64) -> f64 {
    if e1 >= E1_DEAD {
        return 1.0;
    }

    cost1(e1)
}

/// Probability a fired detector gives a CONCLUSIVE outcome, Fung, Tamaki & Lo Eq. (40)'s bracket
/// `e_det/2 + 1/4`, `e_det = sin^2(theta)`. Misalignment RAISES it.
#[pyfunction]
pub(crate) fn sarg_sift(e_det: f64) -> PyResult<f64> {
    check_err("e_det", e_det)?;

    Ok(0.5 * e_det + 0.25)
}

/// `(Y_n, e_n)` of an `n`-photon emission, Fung, Tamaki & Lo Eqs. (40) and (41), `eta_n = 1 -
/// (1 - eta)^n`. `dark` is the RECEIVER's per-pulse background (`decoy_gain`'s `y0`), NOT per
/// detector per gate. Zero yield returns `(0, 1/2)`. INFINITE-decoy limit.
#[pyfunction]
pub(crate) fn sarg_yield(eta: f64, dark: f64, e_det: f64, n: u32) -> PyResult<(f64, f64)> {
    check_prob("eta", eta)?;
    check_prob("dark", dark)?;
    check_err("e_det", e_det)?;

    let seen = 1.0 - (1.0 - eta).powi(n as i32);
    let y = seen * (0.5 * e_det + 0.25) + (1.0 - seen) * 0.5 * dark;
    if y <= 0.0 {
        return Ok((0.0, E_CAP));
    }

    let err = seen * 0.5 * e_det + (1.0 - seen) * 0.25 * dark;

    Ok((y.min(1.0), (err / y).min(E_CAP)))
}

/// `(Q_mu, E_mu)` of a phase-randomised coherent state, Fung, Tamaki & Lo Eqs. (42) and (43). Gain
/// is CONCLUSIVE probability per emitted pulse, sifting inside. Spelled in the equations' term
/// order, not as a sum over `sarg_yield`: they differ in the last bit, each anchored to its equation.
#[pyfunction]
pub(crate) fn sarg_gain(mu: f64, eta: f64, dark: f64, e_det: f64) -> PyResult<(f64, f64)> {
    check_pos("mu", mu)?;
    check_prob("eta", eta)?;
    check_prob("dark", dark)?;
    check_err("e_det", e_det)?;

    let vac = (-eta * mu).exp();
    let gain = 0.5 * dark * vac + (0.5 * e_det + 0.25) * (1.0 - vac);
    if gain <= 0.0 {
        return Ok((0.0, E_CAP));
    }

    let err = 0.25 * dark * vac + 0.5 * e_det * (1.0 - vac);

    Ok((gain.min(1.0), (err / gain).min(E_CAP)))
}

/// `(H(Z1|X1), e_p1)` at single-photon bit error `e1`: cost `H4(bell) - h2(e1)` and phase error
/// `(3/2) e1`. `e_p1` does NOT enter the rate: `h2(1.5 e1)` over-charges. At `E1_DEAD` and above,
/// one full bit and `E_CAP`; not refused.
#[pyfunction]
pub(crate) fn sarg_cost(e1: f64) -> PyResult<(f64, f64)> {
    check_err("e1", e1)?;

    Ok((charged(e1), (1.5 * e1).min(E_CAP)))
}

/// Two-photon phase error, Fung, Tamaki & Lo Eq. (10) at worst case `a = 0`, closed form
/// `1/2 + sqrt(1.5 * (1 - 9w^2)) / 6 - (3*sqrt2/4) * w`, `w = 1/3 - e2`; `sin^2(pi/8)` at `e2 = 0`.
/// DEPARTS from Eq. (10): returns 1/2 at and above `E2_DEAD = 1/6`, where Eq. (10) keeps rising.
#[pyfunction]
pub(crate) fn sarg_phase(e2: f64) -> PyResult<f64> {
    check_err("e2", e2)?;

    Ok(phase2(e2))
}

/// `(live1, live2, e1_dead, e2_dead)`: whether each of Eq. (39)'s terms still certifies (strict
/// `<`), and the cutoffs. `sarg_rate` drops a dead term silently.
#[pyfunction]
pub(crate) fn sarg_domain(e1: f64, e2: f64) -> PyResult<(bool, bool, f64, f64)> {
    check_err("e1", e1)?;
    check_err("e2", e2)?;

    Ok((e1 < E1_DEAD, e2 < E2_DEAD, E1_DEAD, E2_DEAD))
}

/// Asymptotic decoy-state SARG04 rate, bits per EMITTED PULSE, Fung, Tamaki & Lo Eq. (39):
/// `R = -Q_mu * f_ec * h2(E_mu) + Q1 * [1 - H(Z1|X1)] + Q2 * [1 - h2(e_p2)]`. No `q_sift`. Clamped
/// at zero as a whole, NOT per term; the dead cutoffs drop a POSITIVE term.
#[pyfunction]
pub(crate) fn sarg_rate(
    q_mu: f64,
    e_mu: f64,
    q1: f64,
    e1: f64,
    q2: f64,
    e2: f64,
    f_ec: f64,
) -> PyResult<f64> {
    check_prob("q_mu", q_mu)?;
    check_err("e_mu", e_mu)?;
    check_prob("q1", q1)?;
    check_err("e1", e1)?;
    check_prob("q2", q2)?;
    check_err("e2", e2)?;
    check_fec("f_ec", f_ec)?;

    if q1 + q2 > q_mu {
        return Err(PyValueError::new_err(format!(
            "q1 + q2 must not exceed q_mu, got {} > {q_mu}: both are part of the conclusive gain",
            q1 + q2
        )));
    }

    let single = q1 * (1.0 - charged(e1));
    let double = q2 * (1.0 - h2(phase2(e2)));
    let rate = single + double - q_mu * f_ec * h2(e_mu);

    Ok(rate.max(0.0))
}

/// Tolerable bit error of a `photons`-photon signal: root of `1 - H(X) - H(Z|X)`, Fung, Tamaki &
/// Lo Eq. (5). 0.096892 and 0.027101 against Tamaki & Lo, Phys. Rev. A 73, 010302(R) (2006)'s
/// 9.68% and 2.71%. Branciard, Gisin, Kraus & Scarani, Phys. Rev. A 72, 032301 (2005)'s 10.95%
/// needs bit-flip preprocessing, not implemented. Bracket stops at the dead cutoff: the objective
/// turns back upward near `e = 0.385`.
#[pyfunction]
pub(crate) fn sarg_limit(photons: u32) -> PyResult<f64> {
    match photons {
        1 => Ok(root(0.0, E1_DEAD, |e| 1.0 - h2(e) - cost1(e))),
        2 => Ok(root(0.0, E2_DEAD, |e| 1.0 - h2(e) - h2(phase2(e)))),
        other => Err(PyValueError::new_err(format!(
            "photons must be 1 or 2, got {other}: SARG04 certifies one- and two-photon signals only"
        ))),
    }
}

/// Photon-number-splitting CEILING on SARG04 against INCOHERENT attacks, no decoy states, bits
/// per pulse: Branciard, Gisin, Kraus & Scarani, Phys. Rev. A 72, 032301 (2005), Eq. (106),
/// `R_sk = (eta/4) * (1 - I_S(1)) * (mu * t - mu^3 / 12)`, `1 - I_S(1) = h2(1/2 + 1/(2 sqrt2)) =
/// 0.60088` their Eq. (73), optimum their Eq. (107) at `mu_opt = 2 sqrt(t)`. BB84 scales as `t^2`
/// under the same class (Niederberger, Scarani & Gisin, Phys. Rev. A 71, 042316 (2005), Eqs. (29)
/// and (30)).
///
/// NOT A SECURITY BOUND and not `sarg_rate`'s quantity: an UPPER bound over a restricted attack
/// class, no `1 - h2(Q)`, no `n >= 4`, no dark counts; vanishes at `mu = 2 sqrt(3t)`. Koashi,
/// arXiv:quant-ph/0507154, Eq. (11) proves the `O(eta^{3/2})` scaling as a lower bound. Clamped at
/// zero.
#[pyfunction]
pub(crate) fn sarg_ceiling(mu: f64, t: f64, eta: f64) -> PyResult<f64> {
    check_pos("mu", mu)?;
    check_unit("t", t)?;
    check_unit("eta", eta)?;

    let p_store = 0.5 + 0.5 * (0.5f64).sqrt();
    let left = h2(p_store);

    Ok((0.25 * eta * left * (mu * t - mu * mu * mu / 12.0)).max(0.0))
}

/// `eps_sec / 18`, the per-bound failure probability `sarg_counts` and `sarg_errors` take.
#[pyfunction]
pub(crate) fn sarg_eps(eps_sec: f64) -> PyResult<f64> {
    check_eps("eps_sec", eps_sec)?;

    Ok(eps_sec / EPS_TERMS)
}

/// `tau_n = sum_k p_k e^{-k} k^n / n!` over Alice's WHOLE two-intensity set.
fn tau(n: u32, mu: f64, nu: f64, p_mu: f64, p_nu: f64) -> f64 {
    let fact = (1..=n).map(f64::from).product::<f64>();
    let term = |p: f64, k: f64| p * (-k).exp() * k.powi(n as i32) / fact;

    term(p_mu, mu) + term(p_nu, nu)
}

/// Hoeffding half-width `sqrt((total/2) * ln(1/eps))`, Rusca Eq. (3); `total` sums BOTH intensities.
fn dev(total: f64, eps: f64) -> f64 {
    (0.5 * total * (1.0 / eps).ln()).sqrt()
}

fn check_pair(mu: f64, nu: f64, p_mu: f64, p_nu: f64) -> PyResult<()> {
    check_pos("mu", mu)?;
    check_pos("nu", nu)?;
    check_unit("p_mu", p_mu)?;
    check_unit("p_nu", p_nu)?;
    if nu >= mu {
        return Err(PyValueError::new_err(format!(
            "the one-decoy intensities must satisfy nu < mu, got mu = {mu} and nu = {nu}"
        )));
    }

    if p_mu + p_nu > 1.0 {
        return Err(PyValueError::new_err(format!(
            "p_mu + p_nu must not exceed 1, got {}: tau_0 and tau_1 sum over Alice's WHOLE \
             intensity set",
            p_mu + p_nu
        )));
    }

    Ok(())
}

/// `(s0_lo, s1_lo, s0_hi)`: vacuum FLOOR, single-photon floor, vacuum CEILING, as detection
/// COUNTS in one basis from a ONE-DECOY record. Rusca, Boaron, Grunenfelder, Martin & Zbinden,
/// Appl. Phys. Lett. 112, 171104 (2018), arXiv:1801.03443, Eqs. (A19), (A17) and (A14)/(A16),
/// Nian's Eqs. (8), (6) and (7).
///
/// `probs = (p_mu, p_nu)`, `counts = (n_mu, n_nu)`, `errors = (m_mu, m_nu)`, `eps` from
/// `sarg_eps`; errors build the ceiling, Rusca Eq. (A12). `s0_hi` is the max over Rusca's two
/// branches AND both intensities: neither paper fixes Eq. (A16)'s `k`. Floors clamped at zero; the
/// ceiling is not.
#[pyfunction]
pub(crate) fn sarg_counts(
    mu: f64,
    nu: f64,
    probs: (f64, f64),
    counts: (f64, f64),
    errors: (f64, f64),
    eps: f64,
) -> PyResult<(f64, f64, f64)> {
    let (p_mu, p_nu) = probs;
    let (n_mu, n_nu) = counts;
    let (m_mu, m_nu) = errors;
    check_pair(mu, nu, p_mu, p_nu)?;
    check_eps("eps", eps)?;
    for (name, x) in [("n_mu", n_mu), ("n_nu", n_nu), ("m_mu", m_mu), ("m_nu", m_nu)] {
        check_nonneg(name, x)?;
    }

    if m_mu > n_mu || m_nu > n_nu {
        return Err(PyValueError::new_err(format!(
            "the error counts must not exceed the detections they are drawn from, got \
             ({m_mu}, {m_nu}) errors against ({n_mu}, {n_nu}) detections"
        )));
    }

    let seen = n_mu + n_nu;
    let wrong = m_mu + m_nu;
    let t0 = tau(0, mu, nu, p_mu, p_nu);
    let t1 = tau(1, mu, nu, p_mu, p_nu);
    let n_hi = hoeff(n_mu, p_mu, mu, seen, eps).0;
    let n_lo = hoeff(n_nu, p_nu, nu, seen, eps).1;

    let s0_lo = (t0 * (mu * n_lo - nu * n_hi) / (mu - nu)).max(0.0);

    let (d_seen, d_wrong) = (dev(seen, eps), dev(wrong, eps));
    let branch = |m_k: f64, p_k: f64, k: f64| 2.0 * (t0 * k.exp() / p_k * (m_k + d_wrong) + d_seen);
    let s0_hi = (2.0 * (wrong + d_seen))
        .max(branch(m_mu, p_mu, mu))
        .max(branch(m_nu, p_nu, nu));

    let share = (mu * mu - nu * nu) / (mu * mu);
    let bracket = n_lo - nu * nu / (mu * mu) * n_hi - share * s0_hi / t0;
    let s1_lo = (t1 * mu / (nu * (mu - nu)) * bracket).max(0.0);

    Ok((s0_lo, s1_lo, s0_hi))
}

/// `v1_hi`: upper bound on single-photon bit-ERROR detections from a ONE-DECOY record. Rusca Eq.
/// (A22), Lim Eq. (4) at two intensities, Nian Eq. (10). Clamped below at zero. Nian's `v_{X,1}`
/// subscript is a slip: it is the Z-basis (same-basis) count.
#[pyfunction]
pub(crate) fn sarg_errors(mu: f64, nu: f64, probs: (f64, f64), errors: (f64, f64), eps: f64) -> PyResult<f64> {
    let (p_mu, p_nu) = probs;
    let (m_mu, m_nu) = errors;
    check_pair(mu, nu, p_mu, p_nu)?;
    check_eps("eps", eps)?;
    for (name, x) in [("m_mu", m_mu), ("m_nu", m_nu)] {
        check_nonneg(name, x)?;
    }

    let wrong = m_mu + m_nu;
    let hi = hoeff(m_mu, p_mu, mu, wrong, eps).0;
    let lo = hoeff(m_nu, p_nu, nu, wrong, eps).1;

    Ok((tau(1, mu, nu, p_mu, p_nu) * (hi - lo) / (mu - nu)).max(0.0))
}

/// Finite-key LENGTH in bits for one-decoy SARG04, Nian, Nie, Zhang & Lu, Commun. Theor. Phys.
/// 76, 065101 (2024), Eq. (5): `l = floor( s0 + s1*[1 - H(Z1|X1)] - f_ec*n_key*h2(e_key) -
/// 6*log2(18/eps_sec) - log2(2/eps_cor) )`. `e1 = min(v1/s1, 1/2)` is formed HERE; `n_key` is the
/// CONCLUSIVE count. NO TWO-PHOTON TERM: a positive term dropped. Clamped at zero.
#[pyfunction]
pub(crate) fn sarg_length(
    s0: f64,
    s1: f64,
    v1: f64,
    n_key: f64,
    e_key: f64,
    f_ec: f64,
    eps_sec: f64,
    eps_cor: f64,
) -> PyResult<f64> {
    check_nonneg("s0", s0)?;
    check_nonneg("s1", s1)?;
    check_nonneg("v1", v1)?;
    check_nonneg("n_key", n_key)?;
    check_err("e_key", e_key)?;
    check_fec("f_ec", f_ec)?;
    check_eps("eps_sec", eps_sec)?;
    check_eps("eps_cor", eps_cor)?;

    if s0 + s1 > n_key {
        return Err(PyValueError::new_err(format!(
            "s0 + s1 must not exceed n_key, got {} > {n_key}: both are part of the conclusive count",
            s0 + s1
        )));
    }

    let e1 = if s1 > 0.0 { (v1 / s1).min(E_CAP) } else { E_CAP };
    let leak = f_ec * n_key * h2(e_key);
    let budget = 6.0 * (EPS_TERMS / eps_sec).log2() + (2.0 / eps_cor).log2();
    let len = s0 + s1 * (1.0 - charged(e1)) - leak - budget;

    Ok(len.max(0.0).floor())
}

/// Refused: `sarg_length` needs the counted record. `n_total` is checked first.
#[pyfunction]
pub(crate) fn sarg_finite(n_total: f64) -> PyResult<f64> {
    check_pos("n_total", n_total)?;

    Err(PyNotImplementedError::new_err(
        "sarg_finite cannot state a key length from a block size: call sarg_eps, sarg_counts, \
         sarg_errors and sarg_length (Nian, Nie, Zhang & Lu, Commun. Theor. Phys. 76, 065101 \
         (2024), Eq. (5), on Rusca, Boaron, Grunenfelder, Martin & Zbinden, Appl. Phys. Lett. 112, \
         171104 (2018), arXiv:1801.03443), or sarg_rate for the asymptotic bound (Fung, Tamaki & \
         Lo, Phys. Rev. A 73, 012337 (2006), Eq. (39)). RETIRED REASON, wrong about the \
         literature: \"no published analysis states a key length\". What stands: (1) Eq. (39) \
         pays privacy amplification on the TWO-photon gain Q2 and no counted bound on s2 or v2 \
         exists (Yin, Fu, Mao & Chen, Sci. Rep. 6, 29482 (2016), Eqs. (30) and (31), need n + 2 \
         intensities for Y_n), so sarg_length DROPS that term; (2) there is no complementary-basis \
         sample, and none is needed: e_p <= (3/2) e_b derives the phase error. NOT implemented: \
         Wang, Corrigan & Lutkenhaus, arXiv:2603.22448, Eq. (3)",
    ))
}
