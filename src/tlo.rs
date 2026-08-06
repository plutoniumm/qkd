use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::gaussian::GaussianState;
use crate::std::{check_nonneg, check_pos, check_prob, check_unit, sqrt0};

// Transmitted-local-oscillator (TLO) CV-QKD: the oscillator crosses the signal's fibre.
//
// Every xi is CHANNEL INPUT referred (Bob sees eta*T*xi), except `tlo_shot`'s `n_bob`, at DETECTOR
// INPUT before eta. ASSUMED unit N'_0 and ACTUAL unit N_0 never interchange; `ratio` is always
// assumed/actual, 1.0 no attack. SNU vacuum is 1, gaussian.rs's is 1/2: `tlo_cov` halves once.
//
// Calibration attack: Jouguet, Kunz-Jacques & Diamanti, PRA 87, 062313 (2013), arXiv:1304.7024.
// LO fluctuation: Ma, Sun, Jiang & Liang, PRA 88, 022339 (2013), arXiv:1303.6043. Real-time
// shot-noise measurement: Kunz-Jacques & Jouguet, PRA 91, 022307 (2015), arXiv:1406.7554.
// Wavelength attack: Huang, Kunz-Jacques, Jouguet, Weedbrook, Yin, Wang, Chen, Guo & Han, PRA 89,
// 032304 (2014), arXiv:1402.6921.
//
// No key rate: no proof reads a shot-noise unit an adversary can move. Huang et al. forge the null
// `tlo_monitor` tests against; Trisetyarso, Yulianti & Surendro, arXiv:2607.24855 (2026), a
// preprint with no journal reference, Theorem 1, give an untrusted unit zero forgery detectability.
// Carry a hand-assembled budget as q.Channel(T=..., xi=..., ref='input').

/// `(n_bob, t_lo, unit, vel)`: LO photon number at DETECTOR INPUT, the LO arm's whole
/// transmittance, the operating shot-noise unit as a multiple of the characterisation one, and
/// the electronic noise referred to it. `n_lo` is per pulse at ALICE's output, `t` the fibre span
/// the signal also crosses, `mux` the LO arm's extra transmittance, `n_ref` and `vel_ref` (SNU)
/// the characterisation point. Detector efficiency cancels out of `unit`.
#[pyfunction]
pub(crate) fn tlo_shot(
    n_lo: f64,
    t: f64,
    mux: f64,
    n_ref: f64,
    vel_ref: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_pos("n_lo", n_lo)?;
    check_unit("t", t)?;
    check_unit("mux", mux)?;
    check_pos("n_ref", n_ref)?;
    check_nonneg("vel_ref", vel_ref)?;

    let t_lo = t * mux;
    let n_bob = n_lo * t_lo;
    let unit = n_bob / n_ref;
    if !(unit.is_finite() && unit > 0.0) {
        return Err(PyValueError::new_err(format!(
            "the operating shot-noise unit is {unit} of the calibrated one ({n_bob} photons \
             arriving against {n_ref} characterised): nothing to normalise a quadrature by"
        )));
    }

    Ok((n_bob, t_lo, unit, vel_ref / unit))
}

/// The calibration line Z(P) = slope*P + floor, read at the monitored LO power `p_mon` and the
/// power `p_eff` actually mixing at the sampling instant (equal when honest). `floor` is
/// electronic and does NOT move with the LO.
///
/// `(n0_cal, n0_real, ratio, vel_cal, observed)`: `ratio` = assumed/actual = p_mon/p_eff;
/// `vel_cal` in the assumed unit; `observed` the zero-signal variance a monitor reads, 1 + vel_cal
/// honest and 1/ratio + vel_cal not.
#[pyfunction]
pub(crate) fn tlo_calib(
    p_mon: f64,
    p_eff: f64,
    slope: f64,
    floor: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_pos("p_mon", p_mon)?;
    check_pos("p_eff", p_eff)?;
    check_pos("slope", slope)?;
    check_nonneg("floor", floor)?;

    let n0_cal = slope * p_mon;
    let n0_real = slope * p_eff;
    if !(n0_cal > 0.0 && n0_real > 0.0 && n0_cal.is_finite() && n0_real.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "the calibration line gives shot-noise units {n0_cal} (monitored) and {n0_real} \
             (sampled); a unit that is zero, infinite or negative normalises nothing"
        )));
    }

    let ratio = p_mon / p_eff;
    let vel_cal = floor / n0_cal;

    Ok((n0_cal, n0_real, ratio, vel_cal, n0_real / n0_cal + vel_cal))
}

/// `(t_est, xi_est, xi_flat, gap, loss_db)`, what Alice and Bob estimate with the unit off by
/// `ratio` (assumed/actual). `t`, `xi`, `eta` are TRUE, `xi` CHANNEL INPUT referred in the ACTUAL
/// unit. `xi_flat = xi_est/ratio` is the same quantity on the UNBIASED slope t_hat^2 = eta*T,
/// Jouguet arXiv:1304.7024 Eq. (8); `gap = xi - xi_est` is the input-referred excess noise hidden,
/// `loss_db = 10*log10(ratio)` the apparent extra loss. A negative `xi_est` returns unclamped.
#[pyfunction]
pub(crate) fn tlo_estimate(
    t: f64,
    xi: f64,
    eta: f64,
    ratio: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;
    check_unit("eta", eta)?;
    check_pos("ratio", ratio)?;

    let t_est = t / ratio;
    let xi_est = xi + (1.0 - ratio) / (eta * t);
    let xi_flat = xi_est / ratio;

    Ok((t_est, xi_est, xi_flat, xi - xi_est, 10.0 * ratio.log10()))
}

/// Absolute slack at the zero crossing of xi + (1 - ratio)/(eta*T): gaussian.rs's BONA_TOL.
const XI_TOL: f64 = 1e-9;

/// The two-mode covariance Bob reconstructs from the biased estimate: gamma_AB = [[V I2, C
/// sigma_z], [C sigma_z, B I2]], V = va + 1, C = sqrt(T*va*(va + 2)), B = T*V + 1 - T + T*xi, at
/// the ESTIMATED `(t_est, xi_est)`. `va` in SNU; returns INTERNAL units (hbar = 1, vacuum 1/2,
/// xpxp).
#[pyfunction]
pub(crate) fn tlo_cov(va: f64, t: f64, xi: f64, eta: f64, ratio: f64) -> PyResult<GaussianState> {
    check_pos("va", va)?;

    let (t_est, xi_est, _flat, _gap, _db) = tlo_estimate(t, xi, eta, ratio)?;
    if xi_est < -XI_TOL {
        return Err(PyValueError::new_err(format!(
            "estimated excess noise {xi_est} SNU at shot-noise ratio {ratio} is negative, and no \
             thermal-loss channel is quieter than the vacuum: the unit is misnormalised, read \
             tlo_estimate instead"
        )));
    }

    let v = va + 1.0;
    let c = sqrt0(t_est * va * (va + 2.0));
    let b = t_est * v + 1.0 - t_est + t_est * xi_est.max(0.0);

    // SNU vacuum 1 -> internal 1/2, once. xpxp: sigma_z signs the p-p corner alone.
    let (hv, hc, hb) = (0.5 * v, 0.5 * c, 0.5 * b);
    let cov = vec![
        hv, 0.0, hc, 0.0, //
        0.0, hv, 0.0, -hc, //
        hc, 0.0, hb, 0.0, //
        0.0, -hc, 0.0, hb,
    ];

    GaussianState::from_moments(vec![0.0; 4], cov)
}

/// Real-time shot-noise monitoring from `n` vacuum samples, Kunz-Jacques & Jouguet, PRA 91,
/// 022307 (2015), arXiv:1406.7554. `z` in standard deviations, `vel` in the assumed unit.
///
/// `(sigma, gap, detected, residual)`: `sigma = (1 + vel)*sqrt(2/(n - 1))` at the null;
/// `gap = 1 - 1/ratio`; `detected` is |gap| > z*sigma two-sided; `residual` the CHANNEL INPUT
/// referred excess noise hideable under the threshold, independent of `ratio`.
///
/// `residual` is a LOWER bound on what can be hidden, never a margin: it prices a uniform rescale
/// only, and the wavelength attack forges the null itself.
#[pyfunction]
pub(crate) fn tlo_monitor(
    n: u64,
    z: f64,
    ratio: f64,
    vel: f64,
    t: f64,
    eta: f64,
) -> PyResult<(f64, f64, bool, f64)> {
    check_pos("z", z)?;
    check_pos("ratio", ratio)?;
    check_nonneg("vel", vel)?;
    check_unit("t", t)?;
    check_unit("eta", eta)?;
    if n < 2 {
        return Err(PyValueError::new_err(format!(
            "n must be >= 2, got {n}: one sample has no variance to estimate"
        )));
    }

    let sigma = (1.0 + vel) * (2.0 / (n - 1) as f64).sqrt();
    let bar = z * sigma;
    if bar >= 1.0 {
        let need = 1.0 + 2.0 * z * z * (1.0 + vel) * (1.0 + vel);

        return Err(PyValueError::new_err(format!(
            "{n} vacuum samples resolve the shot-noise unit only to {bar} of itself at {z} \
             sigma, the whole unit: the monitor certifies nothing. n must exceed {need}"
        )));
    }

    let gap = 1.0 - 1.0 / ratio;
    let residual = (1.0 / (1.0 - bar) - 1.0) / (eta * t);

    Ok((sigma, gap, gap.abs() > bar, residual))
}

/// A detector's linear half-range `alpha`, quoted against the ASSUMED unit (q.Saturation.alpha),
/// as the same clipping point against the ACTUAL unit, alpha*sqrt(ratio): clipping is at a fixed
/// photocurrent. For the honest distance effect pass `ratio = n_ref/n_bob` from `tlo_shot`.
#[pyfunction]
pub(crate) fn tlo_range(alpha: f64, ratio: f64) -> PyResult<f64> {
    check_pos("alpha", alpha)?;
    check_pos("ratio", ratio)?;

    Ok(alpha * ratio.sqrt())
}

/// Residual phase-error variance of a co-propagating reference, rad^2: 2*pi*linewidth*delay,
/// the Wiener variance of a Lorentzian line of FWHM `linewidth` Hz over `delay` s. Feed it to
/// q.budget.phase as `v_err`; the xi it produces is channel-input referred.
///
/// Not the LLO phase budget: only the multiplexing delay survives. Independent across symbols only
/// while the delay fits inside one symbol period. Polarisation-mode dispersion is unmodelled.
#[pyfunction]
pub(crate) fn tlo_phase(linewidth: f64, delay: f64) -> PyResult<f64> {
    check_nonneg("linewidth", linewidth)?;
    check_nonneg("delay", delay)?;

    Ok(std::f64::consts::TAU * linewidth * delay)
}

/// Quadrature displacement a leaking oscillator puts on the signal mode, SNU amplitude
/// 2*sqrt(leak*n_bob). `n_bob` is `tlo_shot`'s first return, `leak` the fraction reaching the
/// signal mode. A displacement, not a variance: the leak's excess noise is unmodelled.
#[pyfunction]
pub(crate) fn tlo_leak(n_bob: f64, leak: f64) -> PyResult<f64> {
    check_pos("n_bob", n_bob)?;
    check_prob("leak", leak)?;

    Ok(2.0 * sqrt0(leak * n_bob))
}
