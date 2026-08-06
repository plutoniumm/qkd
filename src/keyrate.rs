use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::std::{check_eps, check_fec, check_nonneg, check_pos, check_prob, check_unit, g_ent, h2};

// Key rates for Gaussian-modulated CV-QKD, plus the DPS individual-attack bound. SNU: vacuum 1,
// V_SNU = 2*V_internal. xi at the CHANNEL INPUT. Lodewyck, PRA 76, 042305 (2007): homodyne,
// trusted. Fossier, J. Phys. B 42, 114014 (2009): heterodyne. Weedbrook, RMP 84, 621 (2012) and
// Laudenbach, arXiv:1703.09278 Eqs. (7.14)-(7.15): cancellation-free eigenvalues -- chapter
// numbering; the Adv. Quantum Technol. 1, 1800011 (2018) version renumbers and was not opened, so
// quote no flat equation number for it. Laudenbach & Pacher, Adv.
// Quantum Technol. 2, 1900055 (2019): untrusted. Mountogiannakis, PRResearch 4, 013099 (2022):
// finite size -- Eqs. (3)-(4) the homodyne input-output model and its noise variance, (39) and
// (44)-(54) parameter estimation through to the worst-case pair, (92) the composable rate, whose
// Delta_AEP term is (93); do not cite (93) as the rate. Leverrier, PRA 81, 062343 (2010): z_pe.
// Waks, PRA 73, 012344 (2006): DPS. Leverrier, PRL 118, 200501 (2017): the Gaussian de Finetti
// reduction, in cv_general's block below.

/// Slack on nu - 1 >= 0; below -PHYS_TOL the point fails safe.
const PHYS_TOL: f64 = 1e-6;

/// sqrt clamped at zero, NaN NOT clamped: an intermediate that left f64 range must not be zeroed
/// into a vacuum. `std::sqrt0` is the other convention.
fn sqrt0(x: f64) -> f64 {
    if x > 0.0 {
        x.sqrt()
    } else if x.is_nan() {
        x
    } else {
        0.0
    }
}

struct CvPoint {
    i_ab: f64,
    chi_be: f64,
    key: f64,
}

fn check_point(va: f64, t: f64, xi: f64, eta: f64, vel: f64, beta: f64) -> PyResult<()> {
    check_pos("va", va)?;
    check_unit("t", t)?;
    check_nonneg("xi", xi)?;
    check_unit("eta", eta)?;
    check_nonneg("vel", vel)?;
    check_prob("beta", beta)
}

/// Bob's detector at the CHANNEL OUTPUT plane, SNU. Heterodyne: v_el twice (two receivers), plus
/// the balanced splitter's vacuum unit.
fn chi_det(eta: f64, vel: f64, hom: bool) -> f64 {
    if hom {
        ((1.0 - eta) + vel) / eta
    } else {
        (1.0 + (1.0 - eta) + 2.0 * vel) / eta
    }
}

/// K = beta*I_AB - chi_BE against Gaussian collective attacks (Lodewyck Eqs. (19)-(24), Fossier
/// for heterodyne). trusted: eta, v_el inside Bob's lab, in the conditional step only. untrusted:
/// T -> eta*T, xi -> xi + mu_det*v_el/(eta*T) at eta = 1, v_el = 0, mu_det = 1 hom / 2 het.
fn cv_core(va: f64, t: f64, xi: f64, eta: f64, vel: f64, beta: f64, hom: bool, trusted: bool) -> CvPoint {
    let (t, xi, eta, vel) = if trusted {
        (t, xi, eta, vel)
    } else {
        let mu_det = if hom { 1.0 } else { 2.0 };

        (eta * t, xi + mu_det * vel / (eta * t), 1.0, 0.0)
    };

    // Channel-input referred.
    let v = va + 1.0;
    let chi_line = 1.0 / t - 1.0 + xi;
    let chi_det = chi_det(eta, vel, hom);
    let chi_tot = chi_line + chi_det / t;

    // Heterodyne reads both quadratures: the full log2, not half.
    let snr = (v + chi_tot) / (1.0 + chi_tot);
    let i_ab = if hom { 0.5 * snr.log2() } else { snr.log2() };

    // Symplectic eigenvalues of gamma_AB (Lodewyck Eq. 20; Laudenbach Eqs. (7.14)-(7.15)) in
    // loss = 1 - T and input-referred u: A = w^2 + 2*sb, A^2 - 4B = w^2*(w^2 + 4*sb) exactly.
    // Do not simplify back to V and chi_line: the cancellation keeps chi_BE accurate.
    let loss = 1.0 - t;
    let u = loss + t * xi;
    let sb = v * u + t;
    let beta_1 = (v - 1.0) * loss + v * t * xi;
    let w = (v - 1.0) * loss - t * xi;
    let aw = w.abs();
    let span = sqrt0(w * w + 4.0 * sb);

    // nu - 1, not nu: G((nu-1)/2) has infinite slope at the vacuum. span - 2 = (w^2 +
    // 4*beta_1)/(span + 2) is exactly 0 at T = 1, xi = 0.
    let n1 = 0.5 * ((w * w + 4.0 * beta_1) / (span + 2.0) + aw);
    let nu1 = 1.0 + n1;

    // Small root from nu1*nu2 = sqrt(B), never from (A - sqrt(A^2 - 4B))/2, which returns
    // exactly 0 past A^2 ~ 4.5e15 * B.
    let n2 = if nu1 > 0.0 { (beta_1 - n1) / nu1 } else { 0.0 };

    // Conditional eigenvalues (Lodewyck Eq. (24), Fossier for heterodyne), q = T*(V + chi_tot).
    // C^2 - 4D cancelled by hand so the discriminant vanishes identically at T = 1, xi = 0.
    let q = t * v + u + chi_det;
    let (n3, rd_1) = if hom {
        let p = sb * (v + sb * chi_det);
        let pq = sqrt0(p) * sqrt0(q);
        let x_2q = chi_det * (w * w + 2.0 * beta_1) + w + v * beta_1;
        let rd_1 = (u * (v * v - 1.0) + chi_det * beta_1 * (2.0 + beta_1)) / (pq + q);

        let bulk = x_2q + 2.0 * q + 2.0 * pq;
        let sum = sqrt0(bulk / q);
        let core = v * beta_1 + w - chi_det * w * w;
        let disc = core * core + 4.0 * chi_det * sb * w * ((v + chi_det) * w + beta_1);
        let diff = sqrt0(disc / (q * bulk));

        (0.5 * ((x_2q + 2.0 * q * rd_1) / (q * (sum + 2.0)) + diff), rd_1)
    } else {
        let gap = w + chi_det * beta_1;
        let delta = w * chi_det + beta_1;
        let bulk = sqrt0(delta * delta + 4.0 * (v + sb * chi_det) * q);
        let step = (delta * delta + 4.0 * q * gap) / (bulk + 2.0 * q);

        (0.5 * (step + delta.abs()) / q, gap / q)
    };
    let nu3 = 1.0 + n3;

    // nu3*nu4 = sqrt(D) exactly, the same stable product as nu1*nu2.
    let n4 = if nu3 > 0.0 { (rd_1 - n3) / nu3 } else { 0.0 };

    // S(E) = S(AB), S(E|m_B) = S(AFG|m_B); the fifth eigenvalue is exactly 1.
    let chi_be = g_ent(n1 / 2.0) + g_ent(n2 / 2.0) - g_ent(n3 / 2.0) - g_ent(n4 / 2.0);

    // Test the EIGENVALUES: w^2 leaves f64 near V_A ~ 1e154. Fail safe to chi_be = inf, not 0.
    let broken = ![n1, n2, n3, n4]
        .iter()
        .all(|v| v.is_finite() && *v >= -PHYS_TOL);
    if broken || !chi_be.is_finite() || !i_ab.is_finite() {
        return CvPoint {
            i_ab: if i_ab.is_finite() { i_ab } else { 0.0 },
            chi_be: f64::INFINITY,
            key: f64::NEG_INFINITY,
        };
    }

    // Eigenvalues vouched for, so chi_be < 0 is ~1e-7 round-off.
    let chi_be = chi_be.max(0.0);

    CvPoint {
        i_ab,
        chi_be,
        key: beta * i_ab - chi_be,
    }
}

/// Inverse standard-normal CDF at 1 - eps/2, Acklam's rational approximation, relative error
/// ~1.15e-9; the tail branch takes eps/2 directly, so eps down to ~1e-300 never hits the 1 -
/// eps/2 == 1.0 rounding wall.
fn z_quantile(eps: f64) -> f64 {
    // ca/cb central (|p - 1/2| <= 0.47575), cc/cd tail.
    let ca = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let cb = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let cc = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let cd = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let tail = eps / 2.0;
    if tail < 0.02425 {
        // Lower-tail formula at `tail` yields -z.
        let q = sqrt0(-2.0 * tail.ln());
        let num = ((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5];
        let den = (((cd[0] * q + cd[1]) * q + cd[2]) * q + cd[3]) * q + 1.0;

        -(num / den)
    } else {
        let q = 0.5 - tail;
        let r = q * q;
        let num = ((((ca[0] * r + ca[1]) * r + ca[2]) * r + ca[3]) * r + ca[4]) * r + ca[5];
        let den = ((((cb[0] * r + cb[1]) * r + cb[2]) * r + cb[3]) * r + cb[4]) * r + 1.0;

        num * q / den
    }
}

/// `(i_ab, chi_be, key)` bits/symbol against Gaussian collective attacks. `key` is NOT clamped.
#[pyfunction]
pub(crate) fn cv_rate(
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    beta: f64,
    hom: bool,
    trusted: bool,
) -> PyResult<(f64, f64, f64)> {
    check_point(va, t, xi, eta, vel, beta)?;

    let p = cv_core(va, t, xi, eta, vel, beta, hom, trusted);

    Ok((p.i_ab, p.chi_be, p.key))
}

/// Worst-case channel at confidence 1 - eps: `(t_lo, xi_hi)`, T_min = t_lo^2, `(0, +inf)` at
/// t_lo <= 0; `m` scalar pairs (two per heterodyne symbol). `sigma2_hat` is CHANNEL OUTPUT, SNU,
/// carrying the vacuum and chi_det; `xi_hi` is CHANNEL INPUT. Mountogiannakis Eqs. (39),
/// (44)-(54), the last being the worst-case pair `T_m = T_hat - w*sigma_T`, `Xi_m = Xi_hat +
/// w*sigma_Xi`.
#[pyfunction]
pub(crate) fn cv_bounds(t_hat: f64, sigma2_hat: f64, m: f64, va: f64, eps: f64) -> PyResult<(f64, f64)> {
    check_nonneg("t_hat", t_hat)?;
    check_pos("m", m)?;
    check_pos("va", va)?;
    check_eps("eps", eps)?;
    if !(sigma2_hat.is_finite() && sigma2_hat >= 1.0) {
        return Err(PyValueError::new_err(format!(
            "sigma2_hat must be >= 1, got {sigma2_hat}: the channel-output variance in SNU \
             includes the vacuum; pass SNU (twice the Gaussian layer's hbar = 1 variances)"
        )));
    }

    let z = z_quantile(eps);
    let t_lo = t_hat - z * (sigma2_hat / (m * va)).sqrt();
    if t_lo <= 0.0 {
        return Ok((0.0, f64::INFINITY));
    }

    let s2_max = sigma2_hat * (1.0 + z * std::f64::consts::SQRT_2 / m.sqrt());

    Ok((t_lo, (s2_max - 1.0) / (t_lo * t_lo)))
}

/// `cv_finite` without its clamp, for `cv_general`.
fn finite_core(
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    beta: f64,
    hom: bool,
    trusted: bool,
    n_total: f64,
    pe_fraction: f64,
    eps_pe: f64,
    eps_smooth: f64,
    eps_pa: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    check_point(va, t, xi, eta, vel, beta)?;
    check_pos("n_total", n_total)?;
    check_eps("pe_fraction", pe_fraction)?;
    check_eps("eps_pe", eps_pe)?;
    check_eps("eps_smooth", eps_smooth)?;
    check_eps("eps_pa", eps_pa)?;

    let point = cv_core(va, t, xi, eta, vel, beta, hom, trusted);
    let m = pe_fraction * n_total;
    let n_key = n_total - m;

    let det = chi_det(eta, vel, hom);
    // The variance Bob's sample has, not the detector-free 1 + T*xi.
    let (t_lo, xi_hi) = cv_bounds(t.sqrt(), 1.0 + det + t * xi, m, va, eps_pe)?;
    let t_min = t_lo * t_lo;

    // Smooth-min-entropy penalty, arXiv:2103.16589 Eq. (4); dim H_X = 2 gives the 7.
    let delta = 7.0 * ((2.0 / eps_smooth).log2() / n_key).sqrt()
        + (2.0 / n_key) * (1.0 / eps_pa).log2();

    if t_min <= 0.0 {
        return Ok((point.i_ab, f64::INFINITY, 0.0, 0.0, f64::INFINITY, delta));
    }

    // Detector share referred back out: cv_core wants a channel-only pair.
    let xi_max = (xi_hi - det / t_min).max(0.0);
    let worst = cv_core(va, t_min, xi_max, eta, vel, beta, hom, trusted);
    let key = (n_key / n_total) * (beta * point.i_ab - worst.chi_be - delta);

    Ok((point.i_ab, worst.chi_be, key, t_min, xi_max, delta))
}

/// Finite-size rate, m = pe_fraction * n_total disclosed symbols bounding the channel via
/// `cv_bounds`, chi_BE at the worst (T_min, xi_max) except with probability eps_pe. Returns
/// `(i_ab, chi_worst, key, t_min, xi_max, delta)`; `key` IS clamped at 0. COLLECTIVE ATTACKS
/// ONLY; `cv_general` is a separate entry point and no number returned here passes through it.
#[pyfunction]
pub(crate) fn cv_finite(
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    beta: f64,
    hom: bool,
    trusted: bool,
    n_total: f64,
    pe_fraction: f64,
    eps_pe: f64,
    eps_smooth: f64,
    eps_pa: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let (i_ab, chi, key, t_min, xi_max, delta) = finite_core(
        va,
        t,
        xi,
        eta,
        vel,
        beta,
        hom,
        trusted,
        n_total,
        pe_fraction,
        eps_pe,
        eps_smooth,
        eps_pa,
    )?;

    Ok((i_ab, chi, key.max(0.0), t_min, xi_max, delta))
}

// The Gaussian de Finetti reduction, `cv_general`: the lift of the COLLECTIVE-ONLY `cv_finite` to
// GENERAL attacks, a separate entry point nothing routes through. Leverrier, "Security of
// continuous-variable quantum key distribution via a Gaussian de Finetti reduction", Phys. Rev.
// Lett. 118, 200501 (2017), arXiv:1701.03393, Eq. (5): if E_0 is covariant under U(n) on its n
// modes and eps-secure against GAUSSIAN COLLECTIVE attacks, then E = R . E_0 . T -- an energy test
// T in front, one extra privacy-amplification step R behind -- is eps'-secure against GENERAL
// attacks, with
//
//      eps' = K^4 eps / 50,      R shortening the key by ceil(2 log2 C(K+4, 4)) bits,
//      K = max{1, n(d_A + d_B) (1 + 2 sqrt(ln(8/eps)/2n) + ln(8/eps)/n)
//                             (1 - 2 sqrt(ln(8/eps)/2k))^-1}.
//
// K^4/50 multiplies a FAILURE PROBABILITY, NOT a key rate: target eps' runs the collective proof
// at eps = 50 eps'/K^4, and the toll is a separate field. EVERY ROUNDING GOES TOWARD THE LARGER K:
// `energy_cutoff` and `hash_toll` take ceilings, `mode_energy` overstates Bob's honest energy.
//
// Eq. (5) is an IMPLICATION and this computes its arithmetic; three hypotheses are not discharged:
//   The U(n) covariance has to be ENFORCED by random conjugate linear-optical networks, which
//   Leverrier calls "computationally costly" and believes can be bypassed -- a belief, not a
//   proof. `pipeline.rs` does not.
//   The energy test is a protocol step with an abort on k modes ON TOP of the n keyed from.
//   Nothing in `qkd/` runs one; `k_test` buys the analysis and no hardware.
//   `cv_finite`'s eps is not shown to be the eps Eq. (5) consumes: Leverrier's is the composable
//   diamond-distance parameter of Leverrier, Phys. Rev. Lett. 114, 070501 (2015), arXiv:1408.5689,
//   while `cv_finite` is Devetak-Winter with a smooth-min-entropy correction and no correctness
//   term. `eps_target` is split three ways on `qkd/security.py`'s composition rule, not a theorem.

/// Honest mean photon number per mode, vacuum-1 SNU: Alice's kept mode (entanglement-based
/// picture), Bob's channel output with `chi_det` referred out. Heterodyne only. Bob's counts
/// detector noise as signal energy, OVERSTATING d_B (the safe direction).
fn mode_energy(va: f64, t: f64, xi: f64, eta: f64, vel: f64) -> (f64, f64) {
    (0.5 * va, 0.5 * (t * (va + xi) + chi_det(eta, vel, false)))
}

/// K of Eq. (5), CEILED; `+inf` when `k` is too small for the last factor's denominator.
fn energy_cutoff(n: f64, k: f64, d_a: f64, d_b: f64, eps: f64) -> f64 {
    let lg = (8.0 / eps).ln();
    let up = 1.0 + 2.0 * (lg / (2.0 * n)).sqrt() + lg / n;
    let dn = 1.0 - 2.0 * (lg / (2.0 * k)).sqrt();
    if !(dn > 0.0) {
        return f64::INFINITY;
    }

    (n * (d_a + d_b) * up / dn).max(1.0).ceil()
}

/// ceil(2 log2 C(K+4, 4)), the bits step R spends; in logs, C(K+4, 4) leaving u64.
fn hash_toll(cutoff: f64) -> f64 {
    let lg: f64 = [1.0, 2.0, 3.0, 4.0].iter().map(|j| (cutoff + j).log2()).sum();

    (2.0 * (lg - 24.0_f64.log2())).ceil()
}

/// `(eps, K)` with K^4 eps/50 <= `target` RECHECKED at the returned pair; `None` when k is too
/// small for a finite K or K^4 leaves f64.
fn collective_eps(n: f64, k: f64, d_a: f64, d_b: f64, target: f64) -> Option<(f64, f64)> {
    let mut eps = target;

    // Descends to the fixed point from above, so this loop alone returns an eps too large.
    for _ in 0..64 {
        let cut = energy_cutoff(n, k, d_a, d_b, eps);
        let next = 50.0 * target / cut.powi(4);
        if !(next.is_finite() && next > 0.0) {
            return None;
        }

        let settled = (next - eps).abs() <= 1e-14 * eps;
        eps = next;
        if settled {
            break;
        }
    }

    // K recomputed at the eps about to be returned, eps shrunk until K^4 eps/50 <= target holds.
    for _ in 0..16 {
        if !(eps.is_finite() && eps > 0.0) {
            return None;
        }

        let cut = energy_cutoff(n, k, d_a, d_b, eps);
        let lifted = cut.powi(4) / 50.0 * eps;
        if !(lifted.is_finite() && lifted > 0.0) {
            return None;
        }

        if lifted <= target {
            return Some((eps, cut));
        }

        eps *= target / lifted;
    }

    None
}

/// Key LENGTH in bits against GENERAL attacks, Leverrier, PRL 118, 200501 (2017), Eq. (5) on
/// `cv_finite`'s arithmetic: `(length, raw, cutoff, eps_coll, toll)`, `length` clamped at 0 and
/// `raw` not, `cutoff` = K, `eps_coll` the collective budget of the whole three-term epsilon,
/// `toll` the bits step R spends. `k_test` is ON TOP of `n_total`; `d_a`, `d_b` are the test's
/// thresholds in mean photons per mode and must clear the honest values. Heterodyne only. The
/// block above says what it does not certify.
#[pyfunction]
pub(crate) fn cv_general(
    va: f64,
    t: f64,
    xi: f64,
    eta: f64,
    vel: f64,
    beta: f64,
    hom: bool,
    trusted: bool,
    n_total: f64,
    pe_fraction: f64,
    eps_target: f64,
    k_test: f64,
    d_a: f64,
    d_b: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_point(va, t, xi, eta, vel, beta)?;
    check_pos("n_total", n_total)?;
    check_eps("pe_fraction", pe_fraction)?;
    check_eps("eps_target", eps_target)?;
    check_pos("k_test", k_test)?;
    check_pos("d_a", d_a)?;
    check_pos("d_b", d_b)?;

    if hom {
        return Err(PyNotImplementedError::new_err(
            "hom must be false: Eq. (5) needs a protocol covariant under U(n), and Leverrier names \
             the heterodyne no-switching protocol (Weedbrook, Lance, Bowen, Symul, Ralph & Lam, \
             PRL 93, 170504 (2004)); for homodyne use cv_finite's collective-attack rate",
        ));
    }

    let floor = 2.0 * (8.0 / eps_target).ln();
    if k_test <= floor {
        return Err(PyValueError::new_err(format!(
            "k_test must exceed 2*ln(8/eps_target) = {floor}, got {k_test}: below it no finite \
             photon cutoff K exists. Necessary, not sufficient: the reduction's eps sits tens of \
             orders below eps_target, so raise k_test further if certification fails"
        )));
    }

    let (honest_a, honest_b) = mode_energy(va, t, xi, eta, vel);
    for (name, given, honest) in [("d_a", d_a, honest_a), ("d_b", d_b, honest_b)] {
        if given < honest {
            return Err(PyValueError::new_err(format!(
                "{name} must be at least the honest {honest} mean photons per mode, got {given}: \
                 a threshold below the state's own energy understates K, the insecure direction"
            )));
        }
    }

    let (eps_coll, cutoff) = collective_eps(n_total, k_test, d_a, d_b, eps_target).ok_or_else(|| {
        PyValueError::new_err(format!(
            "no collective budget certifies K^4*eps/50 <= {eps_target} at n_total = {n_total}, \
             k_test = {k_test}, d_a + d_b = {}: raise k_test or eps_target (K^4 leaves f64 past \
             K ~ 1.15e77)",
            d_a + d_b
        ))
    })?;

    let share = eps_coll / 3.0;
    if !(share > 0.0 && share < 1.0) {
        return Err(PyValueError::new_err(format!(
            "the reduction's collective budget {eps_coll} leaves (0, 3) when split three ways: \
             eps_target is too large for a cutoff this small"
        )));
    }

    let (_i_ab, _chi, key, _t_min, _xi_max, _delta) = finite_core(
        va,
        t,
        xi,
        eta,
        vel,
        beta,
        hom,
        trusted,
        n_total,
        pe_fraction,
        share,
        share,
        share,
    )?;

    let toll = hash_toll(cutoff);
    let raw = n_total * key - toll;

    Ok((raw.max(0.0), raw, cutoff, eps_coll, toll))
}

/// DPS rate against general individual attacks, Waks PRA 73, 012344 (2006) Eq. (37): R =
/// p_click*[-(1 - 2*mu)*log2(P_c0) - f_ec*h2(e)], P_c0 = 1 - e^2 - (1 - 6e)^2/2 from Eq. (34)
/// (the bound, not the rate), (1 - 2*mu) the splitting concession. Eq. (37) as printed has the
/// privacy-amplification sign inverted (novel.md E1); the shipped form is corrected. Bits/pulse,
/// clamped at 0; 0 outside the domain (e >= 6/38, mu >= 1/2).
#[pyfunction]
pub(crate) fn dps_rate(p_click: f64, qber: f64, mu: f64, f_ec: f64) -> PyResult<f64> {
    check_prob("p_click", p_click)?;
    check_prob("qber", qber)?;
    check_fec("f_ec", f_ec)?;

    // At mu = 0 the (1 - 2*mu) factor pays key from an empty channel.
    if !(mu.is_finite() && mu > 0.0) {
        return Err(PyValueError::new_err(format!(
            "mu must be > 0, got {mu}: a pulse train with no photons yields no key"
        )));
    }

    if qber >= 6.0 / 38.0 || mu >= 0.5 {
        return Ok(0.0);
    }

    let pc0 = 1.0 - qber * qber - (1.0 - 6.0 * qber) * (1.0 - 6.0 * qber) / 2.0;
    let rate = p_click * (-(1.0 - 2.0 * mu) * pc0.log2() - f_ec * h2(qber));

    Ok(rate.max(0.0))
}

/// Two-sided z-score Phi^-1(1 - eps/2), Leverrier, Grosshans & Grangier, PRA 81, 062343 (2010)
/// convention: eps = 1e-10 is the literature's "6.5 sigma" (6.466951).
#[pyfunction]
pub(crate) fn z_pe(eps: f64) -> PyResult<f64> {
    check_eps("eps", eps)?;

    Ok(z_quantile(eps))
}
