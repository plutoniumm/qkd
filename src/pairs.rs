use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::numeric::{golden, PHI};
use crate::std::{
    check_eps, check_err, check_fec, check_gate, check_nonneg, check_pos, check_prob, h2, E0,
    TSIRELSON,
};

// BBM92 (Bennett, Brassard & Mermin, Phys. Rev. Lett. 68, 557 (1992)) from a photon-pair source,
// read through the parametric down-conversion model of Ma, Fung & Lo, Phys. Rev. A 76, 012307
// (2007), quant-ph/0703122: TWO independent two-mode squeezers of parameter lambda = sinh^2(chi)
// (`q.gaussian.Epr(r)`'s sinh^2(r)), so pair number is negative-binomial with r = 2, mean 2*lambda.
//
// Not device independent: see `pair_holevo`. The observables are FORWARD MODELS, not bounds.
//
// Finite key: Tomamichel & Leverrier, "A largely self-contained and complete security proof for
// quantum key distribution", Quantum 1, 14 (2017), arXiv:1506.08458, Part I, at DETERMINISTIC
// DETECTION with the block length a protocol parameter. `pair_finite` refuses coincidence counting.
//
// `pair_length` is per SIFTED ROUND, `pair_rate` per PUMP PULSE; neither converts into the other.

/// One device-independent efficiency requirement under two source models, NOT two values of one
/// number: a message quoting one names its model. `DI_ETA_PARTIAL`, 86.5479%, plain CHSH with no
/// noise preprocessing, a partially entangled two-qubit state: Woodhead, Acin & Pironio,
/// "Device-independent quantum key distribution with asymmetric CHSH inequalities", Quantum 5, 443
/// (2021), Table 3 (82.5742% with maximal noise preprocessing; Table 4 at a 0.5% channel error,
/// 87.6469%). `DI_ETA_SINGLET`, 92.4%, the MAXIMALLY entangled state at Q = 0, S = 2*sqrt(2), a
/// missing click read as -1: Pironio, Acin, Brunner, Gisin, Massar & Scarani, New J. Phys. 11,
/// 045021 (2009), Fig. 3 caption (Woodhead's Table 2 reads 90.7768%, not binning Bob's
/// nondetection). Neither gates anything.
pub(crate) const DI_ETA_PARTIAL: f64 = 0.865;
pub(crate) const DI_ETA_SINGLET: f64 = 0.924;

/// `(1 + eta_a*lam)`, `(1 + eta_b*lam)` and the joint-silence factor `(1 + eta_a*lam + eta_b*lam
/// - eta_a*eta_b*lam)`.
fn factors(lam: f64, eta_a: f64, eta_b: f64) -> (f64, f64, f64) {
    (
        1.0 + eta_a * lam,
        1.0 + eta_b * lam,
        1.0 + lam * (eta_a + eta_b - eta_a * eta_b),
    )
}

fn check_pair(lam: f64, eta_a: f64, eta_b: f64, y0_a: f64, y0_b: f64) -> PyResult<()> {
    check_nonneg("lam", lam)?;
    check_prob("eta_a", eta_a)?;
    check_prob("eta_b", eta_b)?;
    check_gate("y0_a", y0_a)?;
    check_gate("y0_b", y0_b)?;

    Ok(())
}

/// Probability that ONE pump pulse produces a coincidence, Ma, Fung & Lo Eq. (9). `lam` is pairs
/// per pulse per mode pair (mean pair number `2*lam`); `eta_a`, `eta_b` per-arm end-to-end
/// efficiencies; `y0_a`, `y0_b` per-arm RECEIVER background probabilities, not per-detector.
#[pyfunction]
pub(crate) fn pair_gain(lam: f64, eta_a: f64, eta_b: f64, y0_a: f64, y0_b: f64) -> PyResult<f64> {
    check_pair(lam, eta_a, eta_b, y0_a, y0_b)?;

    let (a, b, c) = factors(lam, eta_a, eta_b);
    let gain = 1.0 - (1.0 - y0_a) / (a * a) - (1.0 - y0_b) / (b * b)
        + (1.0 - y0_a) * (1.0 - y0_b) / (c * c);

    Ok(gain.clamp(0.0, 1.0))
}

/// The CORRELATED coincidences, one emitted pair detected at both ends: Ma, Fung & Lo Eq. (10)'s
/// second term without its `2*(e0 - e_d)` prefactor, `2*(1/c - 1/(a*b))`. Written with NO
/// background factor as published: exact at `y0 = 0`, 0.008% high at `y0 = 1e-3`, 5.2% high at
/// `y0 = 0.05`. The anchors are on the published form.
#[pyfunction]
pub(crate) fn pair_correlated(lam: f64, eta_a: f64, eta_b: f64) -> PyResult<f64> {
    check_nonneg("lam", lam)?;
    check_prob("eta_a", eta_a)?;
    check_prob("eta_b", eta_b)?;

    let (a, b, c) = factors(lam, eta_a, eta_b);

    Ok((2.0 * eta_a * eta_b * lam * (1.0 + lam) / (a * b * c)).max(0.0))
}

/// Ma, Fung & Lo Eq. (10): correlated coincidences err at `e_d`, every other at `e0 = 1/2`, so
/// `E*Q = e0*Q - (e0 - e_d)*correlated`. Returns 1/2 where the gain vanishes.
#[pyfunction]
pub(crate) fn pair_error(
    lam: f64,
    eta_a: f64,
    eta_b: f64,
    y0_a: f64,
    y0_b: f64,
    e_d: f64,
) -> PyResult<f64> {
    check_pair(lam, eta_a, eta_b, y0_a, y0_b)?;
    check_err("e_d", e_d)?;

    let gain = pair_gain(lam, eta_a, eta_b, y0_a, y0_b)?;
    if gain <= 0.0 {
        return Ok(E0);
    }

    let corr = pair_correlated(lam, eta_a, eta_b)?;

    Ok(((E0 * gain - (E0 - e_d) * corr) / gain).clamp(0.0, E0))
}

/// Asymptotic BBM92 rate, bits per pump pulse: `sift * gain * (1 - h2(e_phase) - f_ec*h2(e_bit))`,
/// Ma, Fung & Lo Eq. (5) (Shor-Preskill through Koashi-Preskill); basis symmetry gives `e_phase =
/// e_bit = E_lambda` asymptotically. Clamped at 0. Same shape as `cow_rate`, kept separate: two
/// security arguments.
#[pyfunction]
pub(crate) fn pair_rate(
    gain: f64,
    e_bit: f64,
    e_phase: f64,
    f_ec: f64,
    sift: f64,
) -> PyResult<f64> {
    check_prob("gain", gain)?;
    check_err("e_bit", e_bit)?;
    check_err("e_phase", e_phase)?;
    check_fec("f_ec", f_ec)?;
    check_prob("sift", sift)?;

    let rate = sift * gain * (1.0 - h2(e_phase) - f_ec * h2(e_bit));

    Ok(rate.max(0.0))
}

/// `(gain, e_lambda, rate)` at one brightness.
fn pair_point(
    lam: f64,
    eta_a: f64,
    eta_b: f64,
    y0_a: f64,
    y0_b: f64,
    e_d: f64,
    f_ec: f64,
    sift: f64,
) -> PyResult<(f64, f64, f64)> {
    let gain = pair_gain(lam, eta_a, eta_b, y0_a, y0_b)?;
    let err = pair_error(lam, eta_a, eta_b, y0_a, y0_b, e_d)?;
    let rate = pair_rate(gain, err, err, f_ec, sift)?;

    Ok((gain, err, rate))
}

/// `(lam, gain, e_lambda, rate)` at the brightness maximising the rate over `[1e-9, 10]`, a scan
/// then golden section in `ln(lam)` (its own loop: the objective is fallible). All-zero rates
/// return the scan's first point.
#[pyfunction]
pub(crate) fn pair_optimum(
    eta_a: f64,
    eta_b: f64,
    y0_a: f64,
    y0_b: f64,
    e_d: f64,
    f_ec: f64,
    sift: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_pair(1.0, eta_a, eta_b, y0_a, y0_b)?;
    check_err("e_d", e_d)?;
    check_fec("f_ec", f_ec)?;
    check_prob("sift", sift)?;

    let lo = (1e-9_f64).ln();
    let hi = (10.0_f64).ln();
    let steps = 200;
    let mut best = (lo, 0.0);
    for i in 0..=steps {
        let u = lo + (hi - lo) * (i as f64) / (steps as f64);
        let r = pair_point(u.exp(), eta_a, eta_b, y0_a, y0_b, e_d, f_ec, sift)?.2;
        if r > best.1 {
            best = (u, r);
        }
    }

    let width = (hi - lo) / (steps as f64);
    let (mut left, mut right) = (best.0 - width, best.0 + width);
    for _ in 0..80 {
        let m1 = right - PHI * (right - left);
        let m2 = left + PHI * (right - left);
        let r1 = pair_point(m1.exp(), eta_a, eta_b, y0_a, y0_b, e_d, f_ec, sift)?.2;
        let r2 = pair_point(m2.exp(), eta_a, eta_b, y0_a, y0_b, e_d, f_ec, sift)?.2;
        if r1 < r2 {
            left = m1;
        } else {
            right = m2;
        }
    }

    let lam = (0.5 * (left + right)).exp();
    let (gain, err, rate) = pair_point(lam, eta_a, eta_b, y0_a, y0_b, e_d, f_ec, sift)?;
    if rate <= best.1 {
        let (g0, e0v, r0) = pair_point(best.0.exp(), eta_a, eta_b, y0_a, y0_b, e_d, f_ec, sift)?;

        return Ok((best.0.exp(), g0, e0v, r0));
    }

    Ok((lam, gain, err, rate))
}

/// Eve's Holevo information against a CHSH value, `h2((1 + sqrt(S^2/4 - 1))/2)` bits per key
/// round: Acin, Brunner, Gisin, Massar, Pironio & Scarani, Phys. Rev. Lett. 98, 230501 (2007),
/// tight against COLLECTIVE attacks. `s = 2` returns 1, `s = 2*sqrt(2)` returns 0. NOT A KEY
/// RATE: discarding no-click rounds IS fair sampling, and `s` must be MEASURED.
#[pyfunction]
pub(crate) fn pair_holevo(s: f64) -> PyResult<f64> {
    check_nonneg("s", s)?;

    if s > TSIRELSON + 1e-12 {
        return Err(PyValueError::new_err(format!(
            "s must not exceed Tsirelson's bound {TSIRELSON}, got {s}. A device-independent claim \
             also needs detection efficiency {DI_ETA_PARTIAL} (partially entangled, plain CHSH) \
             or {DI_ETA_SINGLET} (maximally entangled, missing click read as -1); \
             `q.ClickDetector`'s default is 0.20"
        )));
    }

    if s <= 2.0 {
        return Ok(1.0);
    }

    let root = (0.25 * s * s - 1.0).max(0.0).sqrt();

    Ok(h2(0.5 * (1.0 + root)).clamp(0.0, 1.0))
}

/// `S = 2*sqrt(2)*v` for a singlet of visibility `v` at the optimal settings. A model-dependent
/// DIAGNOSTIC; feeding it to `pair_holevo` and calling the result device independent is circular.
#[pyfunction]
pub(crate) fn pair_chsh(v: f64) -> PyResult<f64> {
    check_prob("v", v)?;

    Ok(TSIRELSON * v)
}

/// Tomamichel & Leverrier's `eps_pe`, Theorem 3 Eq. (58): `2 exp(-(m - k) k^2 nu^2 / (m (k + 1)))`.
fn pe_term(m: f64, k: f64, nu: f64) -> f64 {
    let n = m - k;

    2.0 * (-(n * k * k * nu * nu) / (m * (k + 1.0))).exp()
}

/// `cbar < 1` is Tomamichel & Leverrier's own necessary condition for secrecy; refused, not clamped.
fn check_cbar(cbar: f64) -> PyResult<()> {
    if !(cbar.is_finite() && cbar > 0.0 && cbar < 1.0) {
        return Err(PyValueError::new_err(format!(
            "cbar must be in (0, 1), got {cbar}: the overlap of ALICE'S two measurements, 1/2 for \
             mutually unbiased qubit measurements"
        )));
    }

    Ok(())
}

fn check_split(m: f64, k: f64) -> PyResult<()> {
    check_pos("m", m)?;
    check_pos("k", k)?;
    if k >= m {
        return Err(PyValueError::new_err(format!(
            "k must be < m, got {k} >= {m}: k rounds go to parameter estimation and m - k carry \
             the key"
        )));
    }

    Ok(())
}

/// `nu` strictly inside Theorem 3's interval; `delta + nu` is the error rate the key term pays.
fn check_nu(nu: f64, delta: f64) -> PyResult<()> {
    let hi = 0.5 - delta;
    if !(nu.is_finite() && nu > 0.0 && nu < hi) {
        return Err(PyValueError::new_err(format!(
            "nu must be in (0, 1/2 - delta) = (0, {hi}), got {nu}: the tradeoff parameter Theorem \
             3 takes an infimum over"
        )));
    }

    Ok(())
}

/// `(eps_pe, eps_pa, eps_ec, total)` of one finite BBM92 run at key length `ell`: Tomamichel &
/// Leverrier, Quantum 1, 14 (2017), arXiv:1506.08458, Theorem 3 Eq. (58) for the first two,
/// Theorem 2 Eq. (56) for `eps_ec = 2^-t`; Lemma 1 puts the diamond distance at `eps_ec + eps_pa`
/// and Theorem 3 the secrecy half at `eps_pe + eps_pa`, so the sum is what meets a target. `m` is
/// the SIFTED block with `k` on estimation, `delta` the tolerated error, `leak` the syndrome
/// length, `t` the hash length. NOT THREE EPSILONS OF ONE KIND: `eps_ec` is CORRECTNESS, the
/// other two SECRECY.
#[pyfunction]
pub(crate) fn pair_secpar(
    m: f64,
    k: f64,
    nu: f64,
    delta: f64,
    cbar: f64,
    leak: f64,
    t: f64,
    ell: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_split(m, k)?;
    check_cbar(cbar)?;
    check_err("delta", delta)?;
    check_nu(nu, delta)?;
    check_nonneg("leak", leak)?;
    check_pos("t", t)?;
    check_nonneg("ell", ell)?;

    let n = m - k;
    let exponent = -n * ((1.0 / cbar).log2() - h2(delta + nu)) + leak + t + ell;
    let pa = 0.5 * (2.0_f64).powf(0.5 * exponent);
    let pe = pe_term(m, k, nu);
    let ec = (2.0_f64).powf(-t);

    Ok((pe, pa, ec, pe + pa + ec))
}

/// Finite-key length in BITS, Tomamichel & Leverrier Theorem 3 Eq. (58) solved for `ell`: `floor(
/// (m - k)*(log2(1/cbar) - h2(delta + nu)) - leak - t + 2*log2(2*eps_pa) )`, clamped at zero.
/// `eps_pa` is the PRIVACY-AMPLIFICATION half alone. PER SIFTED ROUND, NEVER PER PUMP PULSE.
#[pyfunction]
pub(crate) fn pair_length(
    m: f64,
    k: f64,
    nu: f64,
    delta: f64,
    cbar: f64,
    leak: f64,
    t: f64,
    eps_pa: f64,
) -> PyResult<f64> {
    check_split(m, k)?;
    check_cbar(cbar)?;
    check_err("delta", delta)?;
    check_nu(nu, delta)?;
    check_nonneg("leak", leak)?;
    check_pos("t", t)?;
    check_eps("eps_pa", eps_pa)?;
    if eps_pa >= 0.5 {
        return Err(PyValueError::new_err(format!(
            "eps_pa must be < 1/2, got {eps_pa}: Eq. (58)'s 2*log2(2*eps_pa) is a CHARGE only \
             while 2*eps_pa < 1"
        )));
    }

    let n = m - k;
    let ell = n * ((1.0 / cbar).log2() - h2(delta + nu)) - leak - t + 2.0 * (2.0 * eps_pa).log2();

    Ok(ell.max(0.0).floor())
}

/// `ell` at one `(k, nu)`, hash length free, `leak = f_ec*(m - k)*h2(delta)`: Eq. (58) is
/// stationary in `t` at `2^-t = (eps - eps_pe)/3`, collapsing the two hash terms to `3*log2(eps -
/// eps_pe) + log2(16/27)`. Negative infinity where eps_pe alone spends the budget.
fn tl_smooth(m: f64, delta: f64, f_ec: f64, cbar: f64, eps: f64, k: f64, nu: f64) -> f64 {
    let n = m - k;
    let budget = eps - pe_term(m, k, nu);
    if budget <= 0.0 {
        return f64::NEG_INFINITY;
    }

    n * ((1.0 / cbar).log2() - h2(delta + nu) - f_ec * h2(delta))
        + 3.0 * budget.log2()
        + (16.0_f64 / 27.0).log2()
}

/// `(ell, t, eps_pa)` at one `(k, nu)`, `t` the next whole bit above the stationary point. `ell`
/// unclamped.
fn tl_exact(
    m: f64,
    delta: f64,
    f_ec: f64,
    cbar: f64,
    eps: f64,
    k: f64,
    nu: f64,
) -> (f64, f64, f64) {
    let n = m - k;
    let budget = eps - pe_term(m, k, nu);
    if budget <= 0.0 {
        return (f64::NEG_INFINITY, 0.0, 0.0);
    }

    let t = (3.0 / budget).log2().ceil().max(1.0);
    let eps_pa = budget - (2.0_f64).powf(-t);
    if eps_pa <= 0.0 {
        return (f64::NEG_INFINITY, t, 0.0);
    }

    let ell = n * ((1.0 / cbar).log2() - h2(delta + nu) - f_ec * h2(delta)) - t
        + 2.0 * (2.0 * eps_pa).log2();

    (ell, t, eps_pa)
}

/// The `nu` maximising `tl_smooth` at fixed `k`; None where no `nu` leaves a positive budget.
fn best_nu(m: f64, delta: f64, f_ec: f64, cbar: f64, eps: f64, k: f64) -> Option<f64> {
    let hi = 0.5 - delta;
    let steps = 200;
    let mut top = f64::NEG_INFINITY;
    let mut peak = 0.0;
    for i in 1..steps {
        let nu = hi * (i as f64) / (steps as f64);
        let v = tl_smooth(m, delta, f_ec, cbar, eps, k, nu);
        if v > top {
            top = v;
            peak = nu;
        }
    }

    if !top.is_finite() {
        return None;
    }

    let width = hi / (steps as f64);
    let left = (peak - width).max(1e-12);
    let right = (peak + width).min(hi - 1e-12);

    Some(golden(left, right, 80, |x| {
        tl_smooth(m, delta, f_ec, cbar, eps, k, x)
    }))
}

fn at_k(m: f64, delta: f64, f_ec: f64, cbar: f64, eps: f64, k: f64) -> f64 {
    match best_nu(m, delta, f_ec, cbar, eps, k) {
        Some(nu) => tl_smooth(m, delta, f_ec, cbar, eps, k, nu),
        None => f64::NEG_INFINITY,
    }
}

/// `(k, nu)` maximising `tl_smooth`, `k` in `ln(k)`: the optimum tracks a power of `m`.
fn tl_search(m: f64, delta: f64, f_ec: f64, cbar: f64, eps: f64) -> Option<(f64, f64)> {
    let hi = (m - 1.0).ln();
    if !(hi > 0.0) {
        return None;
    }

    let steps = 160;
    let mut top = f64::NEG_INFINITY;
    let mut spot = 0.0;
    for i in 0..=steps {
        let u = hi * (i as f64) / (steps as f64);
        let v = at_k(m, delta, f_ec, cbar, eps, u.exp());
        if v > top {
            top = v;
            spot = u;
        }
    }

    if !top.is_finite() {
        return None;
    }

    let width = hi / (steps as f64);
    let left = (spot - width).max(0.0);
    let right = (spot + width).min(hi);
    let k = golden(left, right, 60, |x| at_k(m, delta, f_ec, cbar, eps, x.exp())).exp();
    let nu = best_nu(m, delta, f_ec, cbar, eps, k)?;

    Some((k, nu))
}

/// `(length, rate, k, nu, t)` over a SIFTED block of `m` rounds, the rate per sifted round. `k`,
/// `nu`, `t` are Tomamichel & Leverrier's free parameters, optimised as their Figure 7 does, with
/// `leak = f_ec*(m - k)*h2(delta)` (their `1.1*(m - k)*h(delta)` is `f_ec = 1.1`). `eps` is the
/// WHOLE security parameter, correctness included. `detection="coincidence"` is refused.
#[pyfunction]
pub(crate) fn pair_finite(
    m: f64,
    delta: f64,
    f_ec: f64,
    cbar: f64,
    eps: f64,
    detection: &str,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_pos("m", m)?;
    check_err("delta", delta)?;
    check_fec("f_ec", f_ec)?;
    check_cbar(cbar)?;
    check_eps("eps", eps)?;
    if eps >= 0.5 {
        return Err(PyValueError::new_err(format!(
            "eps must be < 1/2, got {eps}: Eq. (58)'s 2*log2(2*eps_pa) is a CHARGE only while \
             2*eps_pa < 1"
        )));
    }

    match detection {
        "deterministic" => (),
        "coincidence" => {
            return Err(PyNotImplementedError::new_err(
                "detection=\"coincidence\" is refused: qkd ships no finite-key BBM92 length for a \
                 coincidence-counted pair link, and three pieces are missing. The bound here is \
                 Tomamichel & Leverrier, Quantum 1, 14 (2017), arXiv:1506.08458, whose Part I \
                 protocol assumes DETERMINISTIC DETECTION at both receivers (their Section 3.1: \
                 both devices always output 0 or 1), where pair_gain is a probability below 1. \
                 (1) THE SPLIT PART II SUPPLIES IS ONE-SIDED. Bob's half transfers: Eq. (108) asks \
                 that the inconclusive POVM element coincide in both bases, Lemma 16 splits his \
                 ternary measurement into a conclusive/inconclusive decision followed by Part \
                 I's ideal one, and Part I bounds no complementarity for Bob. Alice's does not: \
                 Part II replaces her MEASUREMENT with a PREPARATION and reads the bound off that. \
                 Kawakami, Takasugi & Azuma, \"Security of passive entanglement-based key \
                 distribution protocols\", arXiv:2607.10659 (2026), name the same obstruction: \
                 every party in BBM92 is a receiver. The two-sided case is settled ASYMPTOTICALLY \
                 only: Koashi, Adachi, Yamamoto & Imoto, \"Security of entanglement-based quantum \
                 key distribution with practical detectors\", arXiv:0804.0891 (2008, no journal \
                 version), prove BBM92 secure with threshold detectors at BOTH ends by DISCARDING \
                 double clicks, not the convention pair_error is written in, with no block \
                 length. A proof would state Eq. (108) at ALICE'S receiver, a Lemma 16 for it, and \
                 Lemma 17 on the coincidence set -- their Eq. (139) invariance is read off Bob's \
                 Omega alone, and a pair link sifts on the intersection of two. (2) cbar IS WHAT \
                 THE LENGTH IS WRITTEN ON, AND THE SQUASHING MODEL IS NOT THE BLOCKER. Part II \
                 takes cbar from Alice's PREPARATION, Corollary 15 on Eqs. (104) and (106), which \
                 a pair link cannot supply; the analyser bound is published -- Gittsovich, \
                 Beaudry, Narasimhachar, Romero Alvarez, Moroder & Lutkenhaus, Phys. Rev. A 89, \
                 012325 (2014), Theorem 10 (active, double clicks assigned at random), Theorem 13 \
                 (biased active) and Theorem 14 (passive, 50/50) each give a QUBIT target whose \
                 vacuum flag structure (Section IV A) leaves the inconclusive element a \
                 basis-independent vacuum projector, Eq. (108) exactly. Two steps are unwritten: \
                 Eq. (21)'s cbar is the complementarity of the FILTERED operators of Eq. (134), \
                 not the raw ternary ones, and nobody states it is 1/2 for a squashed analyser; \
                 and those squashing models stop at a passive UNBIASED analyser -- Kawakami et \
                 al. close the biased passive case in the ASYMPTOTIC regime only and name \
                 finite-key security as the next step. (3) m IS A PROTOCOL PARAMETER THERE AND A \
                 COINCIDENCE COUNT HERE. Their Section 7.3 sifting map, Eq. (118), tolerates a \
                 random conclusive set (Alice takes the first m indices of Omega where the bases \
                 agree and ABORTS otherwise), but Omega is BOB'S alone; pair_gain counts a \
                 coincidence of two background clicks and pair_error charges it at e0 = 1/2, so \
                 the count depends on the state Eve prepared at BOTH ends. Their one per-pulse \
                 conversion, Eqs. (121) and (122), ell/M = (m/M)*(ell/m) with m/M about eta/2, is \
                 an expectation holding asymptotically, and they point at Pfister, Lutkenhaus, \
                 Wehner & Coles, New J. Phys. 18, 053001 (2016) for schemes that do not fix the \
                 block in advance, where sifting attacks live. Pass detection=\"deterministic\" \
                 for the length Part I proves, or pair_rate for the asymptotic coincidence-counted \
                 rate",
            ))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown detection {other:?}, expected \"deterministic\" or \"coincidence\""
            )))
        }
    }

    let (kr, nu) = match tl_search(m, delta, f_ec, cbar, eps) {
        Some(found) => found,
        None => {
            return Err(PyValueError::new_err(format!(
                "no split of a block of {m} leaves Theorem 3's budget positive at eps = {eps}: \
                 eps_pe stays above eps for every k and nu. The block is too short for the \
                 security parameter; lengthen it or raise eps"
            )))
        }
    };

    // Rounding k can overspend eps_pe on a short block; the unrounded split is reported there.
    let ki = kr.round().clamp(1.0, m - 1.0);
    let (k, got) = match tl_exact(m, delta, f_ec, cbar, eps, ki, nu) {
        whole if whole.0.is_finite() => (ki, whole),
        _ => (kr, tl_exact(m, delta, f_ec, cbar, eps, kr, nu)),
    };
    let ell = got.0.max(0.0).floor();

    Ok((ell, ell / m, k, nu, got.1))
}
