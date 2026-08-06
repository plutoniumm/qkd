use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::pairs::{pair_chsh, pair_holevo, pair_rate, DI_ETA_PARTIAL, DI_ETA_SINGLET};
use crate::std::{check_err, check_fec, check_finite, check_pos, check_prob, h2, TSIRELSON};

// E91: Ekert, Phys. Rev. Lett. 67, 661 (1991). Key priced by the CHSH VIOLATION, not the error
// rate. Asymptotic, per pump pulse, in `pair_rate`'s units.
//
// E91 and BBM92 (Bennett, Brassard & Mermin, Phys. Rev. Lett. 68, 557 (1992)) are not two security
// levels: with characterised analysers E91's rate IS `pair_rate`, and the CHSH price is strictly
// worse wherever a state assumption holds -- zero-key error rate 11.003% against 7.149%.
//
// Nothing here is device independent: `ekert_rate` takes S as an argument, never derives one.
//
// Entropy accumulation on E91 estimates a PHASE ERROR from trusted measurements, never a CHSH
// value: Dupuis, Fawzi & Renner, "Entropy Accumulation", Commun. Math. Phys. 379, 867 (2020),
// arXiv:1607.01796, Sec. 5, and Metger, Fawzi, Sutter & Renner, "Generalised Entropy
// Accumulation", Commun. Math. Phys. 405, 261 (2024), arXiv:2203.04989, Sec. 5.2, both on the
// uncertainty relation of Berta, Christandl, Colbeck, Renes & Renner. DFR's Theorem 5.1 at
// `mu -> 0` is `pair_rate` at `theta_EC = f_ec h2(e)`. CHSH enters only device-independently:
// Arnon-Friedman, Dupuis, Fawzi, Renner & Vidick, "Practical device-independent quantum
// cryptography via entropy accumulation", Nat. Commun. 9, 459 (2018).
//
// A LOCAL violation read as ALICE'S MEASUREMENT OVERLAP, not a price on Eve: Lim, Portmann,
// Tomamichel, Renner & Gisin, "Device-Independent Quantum Key Distribution with Local Bell Test",
// Phys. Rev. X 3, 031006 (2013), arXiv:1208.0023, Lemma 6, on Tomamichel & Haenggi, "The link
// between entropic uncertainty and nonlocality", J. Phys. A: Math. Theor. 46, 055301 (2013); the
// length is Tomamichel, Lim, Gisin & Renner, Nature Communications 3, 634 (2012). Their CHSH test
// runs inside Alice's laboratory and their Lemma 7 charges the post-selection.

/// |S|, at most Tsirelson's bound. Ekert's Eq. (3) S is negative at his own settings.
fn chsh_mag(s: f64) -> PyResult<f64> {
    check_finite("s", s)?;

    let mag = s.abs();
    if mag > TSIRELSON + 1e-12 {
        return Err(PyValueError::new_err(format!(
            "|s| must not exceed Tsirelson's bound {TSIRELSON}, got {s}"
        )));
    }

    Ok(mag)
}

/// Ekert's analyser orientations, radians from the vertical x axis: `(a1, a2, a3, b1, b2, b3)` =
/// `(0, pi/4, pi/2, pi/4, pi/2, 3*pi/4)`, the paragraph before his Eq. (1). `(a2, b1)` and
/// `(a3, b2)` are the KEY pairs at `E = -1`; `(a1, a3)` against `(b1, b3)` is the CHSH quadruple
/// of his Eq. (3). Uniform choice leaves `sift = 2/9`.
#[pyfunction]
pub(crate) fn ekert_settings() -> (f64, f64, f64, f64, f64, f64) {
    let quarter = std::f64::consts::FRAC_PI_4;

    (
        0.0,
        quarter,
        2.0 * quarter,
        quarter,
        2.0 * quarter,
        3.0 * quarter,
    )
}

/// Ekert Eq. (2), `E(a, b) = -a.b`, at singlet visibility `v`: `E = -v*cos(a - b)`, angles in
/// radians; his Eq. (1)'s `P++ + P-- - P+- - P-+`. A FORWARD MODEL, never a bound.
#[pyfunction]
pub(crate) fn ekert_correlate(a: f64, b: f64, v: f64) -> PyResult<f64> {
    check_finite("a", a)?;
    check_finite("b", b)?;
    check_prob("v", v)?;

    Ok(-v * (a - b).cos())
}

/// Ekert Eq. (3), `S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3)`, on the same model. SIGNED:
/// Ekert Eq. (4) reads `-2*sqrt(2)` where Acin et al. read `+2*sqrt(2)`. His Eq. (7)'s `sqrt(2)`
/// bounds his intercept-and-resend measure, NOT local models; the rate here is written against 2.
/// A `"modelled"` `s`.
#[pyfunction]
pub(crate) fn ekert_chsh(a1: f64, a3: f64, b1: f64, b3: f64, v: f64) -> PyResult<f64> {
    let s = ekert_correlate(a1, b1, v)? - ekert_correlate(a1, b3, v)?
        + ekert_correlate(a3, b1, v)?
        + ekert_correlate(a3, b3, v)?;

    Ok(s)
}

/// Bit error rate of a Bell-diagonal state at CHSH magnitude `|s|`, `Q = (1 - |s|/(2*sqrt(2)))/2`,
/// Acin, Brunner, Gisin, Massar, Pironio & Scarani, Phys. Rev. Lett. 98, 230501 (2007). MODEL
/// DEPENDENT. `s = 2` gives 14.645%, `s = 2*sqrt(2)` gives 0.
#[pyfunction]
pub(crate) fn ekert_qber(s: f64) -> PyResult<f64> {
    let mag = chsh_mag(s)?;

    Ok((0.5 * (1.0 - mag / TSIRELSON)).clamp(0.0, 0.5))
}

/// `sift * gain * (1 - chi(mag) - f_ec*h2(e_bit))`, clamped at 0; shared by bound and point.
fn chsh_key(gain: f64, e_bit: f64, mag: f64, f_ec: f64, sift: f64) -> PyResult<f64> {
    let key = sift * gain * (1.0 - pair_holevo(mag)? - f_ec * h2(e_bit));

    Ok(key.max(0.0))
}

/// Asymptotic E91 key rate, bits per pump pulse: `sift * gain * (1 - chi(|s|) - f_ec*h2(e_bit))`,
/// `chi` the collective-attack Holevo bound of `pair_holevo`. Pironio, Acin, Brunner, Gisin,
/// Massar & Scarani, New J. Phys. 11, 045021 (2009), Eqs. (11) and (12); at `f_ec = 1`,
/// `gain = sift = 1` this IS their Eq. (12), `r >= 1 - h(Q) - h((1 + sqrt((S/2)^2 - 1))/2)`,
/// Acin et al. 2007's Eqs. (2) and (3). Positionally `pair_rate` with `s` where `e_phase` stands.
///
/// `source` has no default: `"measured"` is a collective-attack bound; `"modelled"` is refused
/// (`ekert_point`); `"device-independent"` is refused permanently.
#[pyfunction]
pub(crate) fn ekert_rate(
    gain: f64,
    e_bit: f64,
    s: f64,
    f_ec: f64,
    sift: f64,
    source: &str,
) -> PyResult<f64> {
    check_prob("gain", gain)?;
    check_err("e_bit", e_bit)?;
    check_fec("f_ec", f_ec)?;
    check_prob("sift", sift)?;

    let mag = chsh_mag(s)?;
    match source {
        "measured" => (),
        "modelled" => {
            return Err(PyValueError::new_err(
                "source=\"modelled\" is refused: an s from a state model is a point, not a bound, \
                 and pricing Eve by it is circular. Use ekert_point for the point, or pair_rate \
                 for the bound a characterised-device assumption buys",
            ))
        }
        "device-independent" => {
            return Err(PyNotImplementedError::new_err(format!(
                "source=\"device-independent\" is refused: qkd ships no device-independent bound, \
                 by decision. Missing: the NPA hierarchy, a semidefinite programme over moment \
                 matrices of non-commuting operators that none of the THREE solvers here \
                 (sdp_phase, dm_secure, lp_dual) solves; entropy accumulation for coherent \
                 attacks; and an s free of the detection loophole, where discarding no-click \
                 rounds IS fair sampling. That needs detection efficiency {DI_ETA_SINGLET} for \
                 the maximally entangled state with a missing click read as -1 (Pironio, Acin, \
                 Brunner, Gisin, Massar & Scarani, New J. Phys. 11, 045021 (2009), Fig. 3), \
                 {DI_ETA_PARTIAL} for a partially entangled two-qubit state under plain CHSH \
                 (Woodhead, Acin & Pironio, Quantum 5, 443 (2021), table 3); q.ClickDetector's \
                 default is 0.20. Pass source=\"measured\" for the collective-attack rate"
            )))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown source {other:?}, expected \"measured\", \"modelled\" or \
                 \"device-independent\""
            )))
        }
    }

    chsh_key(gain, e_bit, mag, f_ec, sift)
}

/// `(s, key_chsh, key_symmetry)` under the depolarised-singlet model at bit error `e_bit`:
/// `s = pair_chsh(1 - 2*e_bit)`, `key_chsh` prices Eve by it, `key_symmetry` is `pair_rate` at
/// `e_phase = e_bit` (BBM92). NEVER A SECURITY CLAIM: `s` is MODELLED. `key_chsh <= key_symmetry`,
/// equal only at `e_bit = 0`.
#[pyfunction]
pub(crate) fn ekert_point(
    gain: f64,
    e_bit: f64,
    f_ec: f64,
    sift: f64,
) -> PyResult<(f64, f64, f64)> {
    check_prob("gain", gain)?;
    check_err("e_bit", e_bit)?;
    check_fec("f_ec", f_ec)?;
    check_prob("sift", sift)?;

    let s = pair_chsh(1.0 - 2.0 * e_bit)?;
    let chsh = chsh_key(gain, e_bit, s, f_ec, sift)?;
    let symmetry = pair_rate(gain, e_bit, e_bit, f_ec, sift)?;

    Ok((s, chsh, symmetry))
}

/// Bit error rate at which the CHSH-priced rate vanishes on the depolarised-singlet model:
/// 7.149% at `f_ec = 1`, the 7.1% Acin et al. 2007 quote for the device-independent scenario,
/// against the 11.003% of their Eq. (12) discussion, `pair_rate`'s `1 - 2*h2(Q)` zero.
#[pyfunction]
pub(crate) fn ekert_threshold(f_ec: f64) -> PyResult<f64> {
    check_fec("f_ec", f_ec)?;

    let margin = |q: f64| -> PyResult<f64> {
        let s = pair_chsh(1.0 - 2.0 * q)?;

        Ok(1.0 - pair_holevo(s)? - f_ec * h2(q))
    };

    let mut lo = 0.0;
    let mut hi = ekert_qber(2.0)?;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if margin(mid)? > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(0.5 * (lo + hi))
}

/// `(c_test, c_key)`, the overlap of Alice's two measurements a CHSH magnitude certifies. Lim,
/// Portmann, Tomamichel, Renner & Gisin, Phys. Rev. X 3, 031006 (2013), arXiv:1208.0023: Lemma 6,
/// on Tomamichel & Haenggi, J. Phys. A: Math. Theor. 46, 055301 (2013), `c_test <= 1/2 +
/// (s/8)*sqrt(8 - s^2)`, 1 at the local bound and 1/2 at Tsirelson's; Lemma 7, `c_key <= 1/2 +
/// (c_test - 1/2)/eta`, the post-selection charge at surviving key-basis fraction `eta`.
///
/// NOT A PRICE ON EVE (that is `pair_holevo`): ALICE'S PREPARATION QUALITY, the `c` of
/// `eur_quality` and the `cbar` of `pair_length`. `eta` has no default: their Eq. (2)'s visibility
/// 0.999 at a 1% error rate needs `eta` above 0.1134. A `c_key >= 1` returns raw. `s` must be
/// LOCAL, inside Alice's laboratory, not `ekert_rate`'s two-arm `s`.
#[pyfunction]
pub(crate) fn ekert_overlap(s: f64, eta: f64) -> PyResult<(f64, f64)> {
    let mag = chsh_mag(s)?;
    check_prob("eta", eta)?;
    if eta <= 0.0 {
        return Err(PyValueError::new_err(
            "eta must be > 0: Lemma 7 divides by the surviving key-basis fraction",
        ));
    }

    if mag < 2.0 {
        return Err(PyValueError::new_err(format!(
            "|s| must be at least the local bound 2, got {s}: Lemma 6's Jensen step needs a \
             monotonic decreasing right hand side, true only on Lim et al.'s Eq. (1) domain \
             2 <= S <= 2*sqrt(2)"
        )));
    }

    let test = 0.5 + 0.125 * mag * (8.0 - mag * mag).max(0.0).sqrt();

    Ok((test, 0.5 + (test - 0.5) / eta))
}

/// Finite-key E91: refused. `n_total` is checked and plays no part.
#[pyfunction]
pub(crate) fn ekert_finite(n_total: f64) -> PyResult<f64> {
    check_pos("n_total", n_total)?;

    Err(PyNotImplementedError::new_err(
        "finite-key E91 is refused. (1) chi(|s|) is Acin, Brunner, Gisin, Massar, Pironio & \
         Scarani, Phys. Rev. Lett. 98, 230501 (2007), an ASYMPTOTIC per-round Holevo quantity \
         against collective attacks; entropy accumulation on E91 -- Dupuis, Fawzi & Renner, \
         Commun. Math. Phys. 379, 867 (2020), arXiv:1607.01796, Sec. 5; Metger, Fawzi, Sutter & \
         Renner, Commun. Math. Phys. 405, 261 (2024), arXiv:2203.04989, Sec. 5.2 -- estimates a \
         phase error and NEITHER READS A CHSH VALUE; one does only device-independently, \
         Arnon-Friedman, Dupuis, Fawzi, Renner & Vidick, Nat. Commun. 9, 459 (2018). (2) s \
         ARRIVES AS AN ARGUMENT with no round count, so no confidence interval can be taken on \
         it; Lim, Portmann, Tomamichel, Renner & Gisin, Phys. Rev. X 3, 031006 (2013), \
         arXiv:1208.0023, Lemma 8, test a tolerated S_tol instead. An epsilon on gain and e_bit \
         alone is the composition b92_finite refuses. (3) A local violation certifies Alice's \
         measurement overlap (ekert_overlap), worth at most the one bit a mutually unbiased pair \
         gives outright: 7.149% against 11.003%. pair_finite is that length at deterministic \
         detection. (4) The post-selection must be charged: Lim et al. Lemma 7 leaves no key below \
         eta = 0.1134 at visibility 0.999 and a 1% error rate; an uncharged coincidence-counted s \
         IS fair sampling. Use ekert_rate with source=\"measured\" and call it asymptotic",
    ))
}
