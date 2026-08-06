use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// House helpers, one spelling each. `crate::std` shadows the `std` crate in lib.rs ONLY: write
// `::std::` there.

/// Error rate of a background click. NOT `E_CAP`: same value, a physical rate not a cap.
pub(crate) const E0: f64 = 0.5;

/// Cap on every error rate a bound returns: above 1/2, 1 - h2(e) turns back upward. Also the
/// degenerate-yield sentinel.
pub(crate) const E_CAP: f64 = 0.5;

/// Tsirelson's bound, 2*sqrt(2).
pub(crate) const TSIRELSON: f64 = 2.0 * std::f64::consts::SQRT_2;

pub(crate) fn check_finite(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite, got {x}")))
    }
}

pub(crate) fn check_pos(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && x > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be > 0, got {x}")))
    }
}

pub(crate) fn check_nonneg(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be >= 0, got {x}")))
    }
}

/// `(0, 1]`: a transmittance or an efficiency.
pub(crate) fn check_unit(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && x > 0.0 && x <= 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in (0, 1], got {x}")))
    }
}

/// `[0, 1]`: a probability or a contrast.
pub(crate) fn check_prob(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && (0.0..=1.0).contains(&x) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in [0, 1], got {x}")))
    }
}

/// `[0, 1)`: a per-gate probability; 1 fires whatever the light did.
pub(crate) fn check_gate(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && (0.0..1.0).contains(&x) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in [0, 1), got {x}")))
    }
}

/// `[0, 1/2]`: an error rate; see `E_CAP`.
pub(crate) fn check_err(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && (0.0..=0.5).contains(&x) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in [0, 1/2], got {x}")))
    }
}

/// `f_ec >= 1`, the error-correction INEFFICIENCY multiplying h2(e); 1 is the Shannon limit.
/// NOT the reconciliation efficiency beta <= 1, its reciprocal.
pub(crate) fn check_fec(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && x >= 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be >= 1, got {x}: an error-correction inefficiency multiplying h2(e), \
             1 being the Shannon limit. The reconciliation efficiency beta in [0, 1] multiplying \
             I_AB is its reciprocal"
        )))
    }
}

/// `(0, 1)`: a failure probability, or the overlap of two distinct non-orthogonal states.
pub(crate) fn check_eps(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && x > 0.0 && x < 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in (0, 1), got {x}")))
    }
}

/// sqrt clamped at zero, NaN included. `keyrate::sqrt0` passes NaN through on purpose.
pub(crate) fn sqrt0(x: f64) -> f64 {
    if x > 0.0 {
        x.sqrt()
    } else {
        0.0
    }
}

/// Wrap a phase into [-pi, pi]. Wrap every partial sum of a walk as it is formed.
pub(crate) fn wrap(x: f64) -> f64 {
    x - std::f64::consts::TAU * (x / std::f64::consts::TAU).round()
}

/// Pilot tone phase at symbol `k`, reduced modulo one cycle before the trig call.
pub(crate) fn ramp(frac: f64, k: usize) -> f64 {
    let z = frac * (k as f64);

    std::f64::consts::TAU * (z - z.floor())
}

/// Binary entropy h2(e) in bits; 0 outside (0, 1).
pub(crate) fn h2(e: f64) -> f64 {
    if e <= 0.0 || e >= 1.0 {
        return 0.0;
    }

    -e * e.log2() - (1.0 - e) * (1.0 - e).log2()
}

/// G(x) = (x+1) log2(x+1) - x log2 x, the thermal entropy at mean photon number x, taken at
/// (nu - 1)/2 for a symplectic eigenvalue nu; G(x <= 0) = 0. Two groupings: the literal form
/// returns exactly 0 past x ~ 9e15, the rearrangement cancels below 1.
pub(crate) fn g_ent(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    if x > 1.0 {
        return x.log2() + (x + 1.0) * (1.0 / x).ln_1p() / std::f64::consts::LN_2;
    }

    (x + 1.0) * (x + 1.0).log2() - x * x.log2()
}

/// ln(k!) for k in 0..=n.
pub(crate) fn ln_fact(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n + 1];
    for k in 1..=n {
        out[k] = out[k - 1] + (k as f64).ln();
    }

    out
}

/// Threshold click probability 1 - (1 - dark)*exp(-eta*mu).
pub(crate) fn click_p(eta: f64, mu: f64, dark: f64) -> f64 {
    1.0 - (1.0 - dark) * (-eta * mu).exp()
}
