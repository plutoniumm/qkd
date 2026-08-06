use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::std::{check_eps, check_nonneg, check_pos, check_prob, E0, E_CAP};

// Decoy-state bounds on Y1, e1 and Q1 for a phase-randomised weak-coherent source; per pulse,
// dimensionless. Ma, Qi, Zhao & Lo, "Practical Decoy State for Quantum Key Distribution",
// PRA 72, 012326 (2005), quant-ph/0503005.
//
// Assumed, unchecked, and each erring OPTIMISTIC (y1 up, e1 down): Alice emits, per intensity k,
// exactly the Poisson mixture sum_n e^-mu_k mu_k^n/n! |n><n|. That needs
// 1. continuous phase randomisation -- N discrete phases give N intensity-dependent pseudo-Fock
//    sectors: Cao, Zhang, Lo & Ma, New J. Phys. 17, 053014 (2015), arXiv:1410.3217;
// 2. exact intensities, emitted not nominal -- Wang, Peng, Zhang, Yang & Pan, Phys. Rev. A 77,
//    042311 (2008), arXiv:0802.3177, re-derives from RANGES;
// 3. no pulse-to-pulse correlation -- Zapatero, Navarrete, Tamaki & Curty, Quantum 5, 602 (2021),
//    arXiv:2105.11165; Sixto, Zapatero & Curty, Phys. Rev. Applied 18, 044069 (2022),
//    arXiv:2206.06700.
// `source=None` asserts all three; `check_source` refuses the other models by name.

/// `(y1_lo, e1_hi, q1_lo)` from signal `mu` and decoys `nu1 > nu2 >= 0`, `nu1 + nu2 < mu`.
/// Vacuum+weak: `nu2 = 0`, `q_nu2` = the measured vacuum gain (it *is* Y0), `e_nu2 = 0.5`.
#[pyfunction]
#[pyo3(signature = (mu, nu1, nu2, q_mu, e_mu, q_nu1, e_nu1, q_nu2, e_nu2, source=None))]
pub(crate) fn decoy_bounds(
    mu: f64,
    nu1: f64,
    nu2: f64,
    q_mu: f64,
    e_mu: f64,
    q_nu1: f64,
    e_nu1: f64,
    q_nu2: f64,
    e_nu2: f64,
    source: Option<&str>,
) -> PyResult<(f64, f64, f64)> {
    // First: a bad gain must not mask the model refusal.
    check_source(source)?;
    check_pos("mu", mu)?;
    check_pos("nu1", nu1)?;
    check_nonneg("nu2", nu2)?;
    check_prob("q_mu", q_mu)?;
    check_prob("e_mu", e_mu)?;
    check_prob("q_nu1", q_nu1)?;
    check_prob("e_nu1", e_nu1)?;
    check_prob("q_nu2", q_nu2)?;
    check_prob("e_nu2", e_nu2)?;

    check_order(nu1, nu2, mu)?;

    let (xm, x1, x2) = (q_mu * mu.exp(), q_nu1 * nu1.exp(), q_nu2 * nu2.exp());

    // Exactly the measured vacuum gain at nu2 = 0.
    let y0_lo = ((nu1 * x2 - nu2 * x1) / (nu1 - nu2)).max(0.0);

    let den = (nu1 - nu2) * (mu - nu1 - nu2);
    let two = (nu1 * nu1 - nu2 * nu2) / (mu * mu);
    let bracket = x1 - x2 - two * (xm - y0_lo);

    // A negative yield would flip e1_hi's sign.
    let y1_lo = (mu * bracket / den).max(0.0);
    let q1_lo = y1_lo * mu * (-mu).exp();

    if y1_lo <= 0.0 {
        return Ok((0.0, E_CAP, 0.0));
    }

    let err = (e_nu1 * x1 - e_nu2 * x2) / ((nu1 - nu2) * y1_lo);

    // Upper bound: capped, never lowered. Infeasible (negative or NaN) reads E_CAP, not 0.
    let e1_hi = if err.is_finite() && err >= 0.0 {
        err.min(E_CAP)
    } else {
        E_CAP
    };

    Ok((y1_lo, e1_hi, q1_lo))
}

/// `(y1, e1)` in the infinite-decoy limit, the ceiling on every finite-decoy bound.
/// `- y0 * eta` is the double-count correction of Ma Eq. (7).
#[pyfunction]
pub(crate) fn decoy_ideal(eta: f64, y0: f64, e_det: f64) -> PyResult<(f64, f64)> {
    check_prob("eta", eta)?;
    check_prob("y0", y0)?;
    check_prob("e_det", e_det)?;

    let y1 = y0 + eta - y0 * eta;

    if y1 <= 0.0 {
        return Ok((0.0, E_CAP));
    }

    Ok((y1, (E0 * y0 + e_det * eta) / y1))
}

/// `(q_mu, e_mu)` of a phase-randomised coherent state of intensity `mu`: Ma Eq. (7) summed
/// exactly, with the cross term Eq. (10) drops. `y0 + 1 - exp(-eta*mu)` puts `decoy_bounds`
/// 0.74% above its own ceiling; do not revert to it.
#[pyfunction]
pub(crate) fn decoy_gain(mu: f64, eta: f64, y0: f64, e_det: f64) -> PyResult<(f64, f64)> {
    check_pos("mu", mu)?;
    check_prob("eta", eta)?;
    check_prob("y0", y0)?;
    check_prob("e_det", e_det)?;

    let hit = 1.0 - (-eta * mu).exp();

    // Composed, not added: y0 + hit can exceed 1.
    let q_mu = y0 + hit - y0 * hit;

    if q_mu <= 0.0 {
        return Ok((0.0, E_CAP));
    }

    Ok((q_mu.min(1.0), (E0 * y0 + e_det * hit) / q_mu))
}

/// `tau_n`, the probability of an `n`-photon emission over the intensity set at its probabilities.
/// Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A 89, 022307 (2014), defined ahead of their
/// Eq. (2).
fn tau(n: u32, mu: [f64; 3], p: [f64; 3]) -> f64 {
    let fact = (1..=n).map(f64::from).product::<f64>();

    (0..3)
        .map(|i| p[i] * (-mu[i]).exp() * mu[i].powi(n as i32) / fact)
        .sum()
}

/// Hoeffding-corrected per-intensity count, Lim's supplementary Eqs. (A1)-(A2):
/// `n_k^{+/-} = (e^k / p_k) * [n_k +/- sqrt((n_tot/2) * ln(1/eps))]`, `eps = eps_sec/21` making
/// the radicand his printed `ln(21/eps_sec)`.
///
/// `n_tot` is the total over the three intensities in THAT basis -- not the block size, not the
/// line's own count. `sarg.rs` reads the same map as Rusca's one-decoy Eq. (3).
pub(crate) fn hoeff(n_k: f64, p_k: f64, mu_k: f64, n_tot: f64, eps: f64) -> (f64, f64) {
    let d = (0.5 * n_tot * (1.0 / eps).ln()).sqrt();
    let scale = mu_k.exp() / p_k;

    (scale * (n_k + d), scale * (n_k - d))
}

/// `(s0_lo, s1_lo)`: lower bounds on the vacuum and single-photon detection COUNTS in one basis,
/// from that basis's three per-intensity counts. Lim Eqs. (2) and (3), the paper's
/// `mu1 > mu2 > mu3` as `(mu, nu1, nu2)`; `eps` is per bound, `eps_sec/21` at the caller.
/// Clamped at zero: unclamped, a downstream length would rise on worse data.
#[pyfunction]
#[pyo3(signature = (mu, nu1, nu2, probs, counts, eps, source=None))]
pub(crate) fn decoy_counts(
    mu: f64,
    nu1: f64,
    nu2: f64,
    probs: (f64, f64, f64),
    counts: (f64, f64, f64),
    eps: f64,
    source: Option<&str>,
) -> PyResult<(f64, f64)> {
    check_source(source)?;
    check_pos("mu", mu)?;
    check_pos("nu1", nu1)?;
    check_nonneg("nu2", nu2)?;
    check_eps("eps", eps)?;

    let ints = [mu, nu1, nu2];
    let p = [probs.0, probs.1, probs.2];
    let n = [counts.0, counts.1, counts.2];
    for (name, x) in [("p_mu", p[0]), ("p_nu1", p[1]), ("p_nu2", p[2])] {
        check_pos(name, x)?;
    }

    for (name, x) in [("n_mu", n[0]), ("n_nu1", n[1]), ("n_nu2", n[2])] {
        check_nonneg(name, x)?;
    }

    check_order(nu1, nu2, mu)?;

    let total = n[0] + n[1] + n[2];

    let dev = [
        hoeff(n[0], p[0], ints[0], total, eps),
        hoeff(n[1], p[1], ints[1], total, eps),
        hoeff(n[2], p[2], ints[2], total, eps),
    ];
    let hi = |i: usize| dev[i].0;
    let lo = |i: usize| dev[i].1;

    let t0 = tau(0, ints, p);
    let t1 = tau(1, ints, p);

    // Eq. (2); at nu2 = 0 it is tau_0 * n_nu2^-, the measured vacuum count.
    let s0 = (t0 * (nu1 * lo(2) - nu2 * hi(1)) / (nu1 - nu2)).max(0.0);

    // Eq. (3); den > 0 by `check_order`.
    let den = mu * (nu1 - nu2) - nu1 * nu1 + nu2 * nu2;
    let bracket = lo(1) - hi(2) - (nu1 * nu1 - nu2 * nu2) / (mu * mu) * (hi(0) - s0 / t0);
    let s1 = (t1 * mu * bracket / den).max(0.0);

    Ok((s0, s1))
}

/// `v1_hi`: upper bound on the single-photon detections that were bit ERRORS, from one basis's
/// three per-intensity error counts. Lim Eq. (4), `v1 <= tau_1 * (m_nu1^+ - m_nu2^-) / (nu1 - nu2)`.
/// Clamped at zero from below only.
#[pyfunction]
#[pyo3(signature = (mu, nu1, nu2, probs, errors, eps, source=None))]
pub(crate) fn decoy_errors(
    mu: f64,
    nu1: f64,
    nu2: f64,
    probs: (f64, f64, f64),
    errors: (f64, f64, f64),
    eps: f64,
    source: Option<&str>,
) -> PyResult<f64> {
    check_source(source)?;
    check_pos("mu", mu)?;
    check_pos("nu1", nu1)?;
    check_nonneg("nu2", nu2)?;
    check_eps("eps", eps)?;

    let ints = [mu, nu1, nu2];
    let p = [probs.0, probs.1, probs.2];
    let m = [errors.0, errors.1, errors.2];
    for (name, x) in [("p_mu", p[0]), ("p_nu1", p[1]), ("p_nu2", p[2])] {
        check_pos(name, x)?;
    }

    for (name, x) in [("m_mu", m[0]), ("m_nu1", m[1]), ("m_nu2", m[2])] {
        check_nonneg(name, x)?;
    }

    check_order(nu1, nu2, mu)?;

    let total = m[0] + m[1] + m[2];
    let hi = hoeff(m[1], p[1], nu1, total, eps).0;
    let lo = hoeff(m[2], p[2], nu2, total, eps).1;

    Ok((tau(1, ints, p) * (hi - lo) / (nu1 - nu2)).max(0.0))
}

/// `nu2 < nu1` and `nu1 + nu2 < mu`. Shared with `discrete::run_basis`.
pub(crate) fn check_order(nu1: f64, nu2: f64, mu: f64) -> PyResult<()> {
    if nu2 >= nu1 {
        return Err(PyValueError::new_err(format!(
            "decoy intensities must satisfy nu2 < nu1, got nu1 = {nu1}, nu2 = {nu2}"
        )));
    }

    if nu1 + nu2 >= mu {
        return Err(PyValueError::new_err(format!(
            "decoys must be weaker than the signal: nu1 + nu2 < mu, got {nu1} + {nu2} >= {mu}"
        )));
    }

    Ok(())
}

/// `q1 > q_mu` is an inverted decoy bound, not something to rescale. Shared with `bb84_rate` and
/// `sixstate_rate`.
pub(crate) fn check_gains(q1: f64, q_mu: f64) -> PyResult<()> {
    if q1 > q_mu {
        return Err(PyValueError::new_err(format!(
            "q1 must not exceed q_mu (got {q1} > {q_mu}): the single-photon gain is part of the total gain"
        )));
    }

    Ok(())
}

/// `None` and `"ideal"` assert the module comment's Poisson mixture; the other three refuse.
fn check_source(source: Option<&str>) -> PyResult<()> {
    match source {
        None | Some("ideal") => Ok(()),
        Some("discrete-phase") => Err(PyNotImplementedError::new_err(
            "source=\"discrete-phase\" is refused: N discrete phases split the Poisson mixture \
             into N intensity-dependent pseudo-Fock sectors, so Y_n is not common across \
             intensities. Not implemented: Cao, Zhang, Lo & Ma, New J. Phys. 17, 053014 (2015), \
             arXiv:1410.3217. Direction: INSECURE -- y1 overstated, e1 understated",
        )),
        Some("intensity-error") => Err(PyNotImplementedError::new_err(
            "source=\"intensity-error\" is refused: mu, nu1 and nu2 are read as the emitted \
             intensities, and averaged over a fluctuation the per-setting weights are not Poisson \
             at any mean. Not implemented: Wang, Peng, Zhang, Yang & Pan, Phys. Rev. A 77, 042311 \
             (2008), arXiv:0802.3177, which bounds the key from the parameter RANGE. Direction: \
             two-sided in the deviation, one-sided in Eve's favour once she can tell which line a \
             pulse came from (source=\"correlated\")",
        )),
        Some("correlated") => Err(PyNotImplementedError::new_err(
            "source=\"correlated\" is refused: modulator memory makes round i's state depend on \
             the setting history, leaving no per-intensity state to invert, and no component \
             carries the correlation range or maximum relative intensity deviation a proof needs. \
             Not implemented: Zapatero, Navarrete, Tamaki & Curty, Quantum 5, 602 (2021), \
             arXiv:2105.11165; Sixto, Zapatero & Curty, Phys. Rev. Applied 18, 044069 (2022), \
             arXiv:2206.06700. Direction: INSECURE -- y1 overstated, e1 understated",
        )),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown source {other:?}, expected \"ideal\", \"discrete-phase\", \
             \"intensity-error\" or \"correlated\""
        ))),
    }
}

/// Alice-to-Bob transmittance: `length` km of fibre at `alpha` dB/km into `eta_bob`.
#[pyfunction]
pub(crate) fn decoy_eta(alpha: f64, length: f64, eta_bob: f64) -> PyResult<f64> {
    check_nonneg("alpha", alpha)?;
    check_nonneg("length", length)?;
    check_prob("eta_bob", eta_bob)?;

    Ok(eta_bob * 10f64.powf(-alpha * length / 10.0))
}
