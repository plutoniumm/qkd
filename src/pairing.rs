use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::std::{
    check_eps, check_err, check_fec, check_nonneg, check_pos, check_prob, check_unit, h2, E_CAP,
};

// Mode-pairing (asynchronous) MDI-QKD: two weak-coherent senders into one untrusted
// single-photon-interference station, the two clicks of a key bit chosen AFTER the announcement.
// Everything here is per ROUND (per emitted pulse pair) and dimensionless.
//
// Zeng, Zhou, Wu & Ma, "Mode-pairing quantum key distribution", Nat. Commun. 13, 3903 (2022),
// arXiv:2201.04300; every equation number is that arXiv version's -- Eq. (4) the pairing rate,
// Eq. (7) the key rate, Eqs. (90)-(104) the forward model of its "Simulation formulas" appendix.
// Phase drift behind the pairing window: Xie, Lu, Weng, Cao et al., "Breaking the rate-loss bound
// of quantum key distribution with asynchronous two-photon interference", arXiv:2112.11635,
// Sec. III, the prose after its Eqs. (3) and (4); its Eq. (5) is a decoy yield bound, unrelated.
// The asymptotic rate is Zeng's, the finite-size LENGTH Xie's, and their event taxonomies do not
// mix (finite-key header below).
//
// SYMMETRIC STATION ONLY: Zeng Eq. (92) assumes eta_a = eta_b. The asymmetric variant is Lu,
// Wang, Li & Cao, arXiv:2401.01727, not implemented.
//
// ROUND ORDER IS LOAD-BEARING: `pairing_pairs` and `pairing_drift` are not functions of a
// per-round marginal, so the family is not permutation invariant.
//
// `pairing_pairs` is r_p, pairs per ROUND, carrying the transmittance scaling; `pairing_sift` is
// r_s, the fraction usable as Z-pairs, ~1/8 at every distance; they multiply in `pairing_rate`.
// q11 IS A FRACTION, NOT A GAIN: `mdi::mdi_gain`'s Poisson prefactor is already in
// `pairing_single`. THE PHASE ERROR IS NOT COMPUTED HERE: Zeng Eq. (104) takes e^X_(1,1) from Ma &
// Razavi's Eqs. (21) and (23) (their appendices continue the main-text numbering), which is
// `mdi::mdi_yield`'s second return at `eta_a = eta_b = eta_s`, passed into `pairing_rate`.
//
// Zeng's Eq. (122) writes the PLOB bound as `-log2(1 - eta)`, 11% wrong by 160 dB and exactly
// 0.0 by 176 dB. Use `-log1p(-eta)/ln 2`.

/// Past this `f64 -> u64` stops being exact; refused, not converted.
const SPAN_MAX: f64 = 1e15;

/// Two rounds per pair, so 1/2 is a combinatorial ceiling, not a tolerance.
const PAIRS_MAX: f64 = 0.5;

/// One detector's per-gate dark probability, capped at 1/2: the click law reads the pair through
/// `1 - 2*dark`.
fn check_dark(name: &str, x: f64) -> PyResult<()> {
    if x.is_finite() && (0.0..=0.5).contains(&x) {
        return Ok(());
    }

    Err(PyValueError::new_err(format!(
        "{name} must be in [0, 1/2], got {x}: one detector's per-gate dark probability, and \
         2*{name} is the station's background announcement floor"
    )))
}

/// The maximal pairing interval `l` of both papers, in rounds.
fn check_span(span: u64) -> PyResult<()> {
    if span >= 1 {
        return Ok(());
    }

    Err(PyValueError::new_err(
        "span must be at least 1 round: the maximal pairing interval",
    ))
}

/// Zeng Eq. (92), the announcement probability at intensity pattern `slots = z^a + z^b` in
/// {0, 1, 2}: `1 - (1 - 2*dark)*exp(-eta_s*mu*slots)` regrouped; the literal form loses 8 of 16
/// digits on the vacuum row at a 1e-8 dark rate.
fn click_set(eta_s: f64, mu: f64, dark: f64, slots: f64) -> f64 {
    2.0 * dark - (1.0 - 2.0 * dark) * (-eta_s * mu * slots).exp_m1()
}

/// `1 - (1 - eta)^photons` as `h <- h + eta*(1 - h)`: the literal difference keeps twelve digits
/// of sixteen at the 5e-5 single-arm transmittance of 500 km.
fn hit_num(eta: f64, photons: u32) -> f64 {
    let mut hit = 0.0;

    for _ in 0..photons {
        hit += eta * (1.0 - hit);
    }

    hit
}

/// Zeng Eq. (94), the announcement probability in photon number: `1 - (1 - 2*dark)*(1 -
/// eta_s)^photons`.
fn click_num(eta_s: f64, dark: f64, photons: u32) -> f64 {
    2.0 * dark + (1.0 - 2.0 * dark) * hit_num(eta_s, photons)
}

/// `click_set` at the three patterns a round can carry, `(empty, one, both)`.
fn click_row(eta_s: f64, mu: f64, dark: f64) -> (f64, f64, f64) {
    (
        click_set(eta_s, mu, dark, 0.0),
        click_set(eta_s, mu, dark, 1.0),
        click_set(eta_s, mu, dark, 2.0),
    )
}

/// Zeng Eq. (93), the row averaged over its four equally likely patterns.
fn click_mean(row: (f64, f64, f64)) -> f64 {
    (0.25 * (row.0 + 2.0 * row.1 + row.2)).min(1.0)
}

/// `(effective, erroneous)`, the four-term sums of Zeng's Eqs. (98) and (102) over the patterns
/// `[z_i, z_j]` with `z_i xor z_j = 11`. Their ratio IS the Z-basis error rate, `r_s` and `p^2`
/// cancelling.
fn pair_sums(row: (f64, f64, f64)) -> (f64, f64) {
    let (empty, one, both) = row;

    (2.0 * (empty * both + one * one), 2.0 * empty * both)
}

/// Average announcement probability, Zeng Eqs. (92) and (93). `eta_s` is the single-arm
/// transmittance to the station WITH detector efficiency inside, `dark` one detector's per-gate
/// probability. The 1/4 is Zeng's asymptotic setting: vacuum or `mu` at 1/2 each, no decoy.
#[pyfunction]
pub(crate) fn pairing_click(eta_s: f64, mu: f64, dark: f64) -> PyResult<f64> {
    check_unit("eta_s", eta_s)?;
    check_pos("mu", mu)?;
    check_dark("dark", dark)?;

    Ok(click_mean(click_row(eta_s, mu, dark)))
}

/// `r_p`, expected pairs per ROUND for Zeng's Algorithm 1, Eq. (4) restated as Eq. (91):
/// `r_p(p, l) = [1/(p*(1 - (1-p)^l)) + 1/p]^(-1)`, `p` from `pairing_click`, `span` = `l`.
/// `span -> inf` gives `p/2 = O(sqrt(eta))`; `span = 1` gives `p^2/(1 + p) = O(eta)`, time-bin
/// MDI-QKD's scaling. `1 - (1-p)^l` is `-expm1(l*ln1p(-p))`: powering and subtracting sheds
/// about `log10(1/(l*p))` digits.
#[pyfunction]
pub(crate) fn pairing_pairs(p: f64, span: u64) -> PyResult<f64> {
    check_unit("p", p)?;
    check_span(span)?;

    let gap = -((span as f64) * (-p).ln_1p()).exp_m1();

    if !(gap > 0.0) {
        return Ok(0.0);
    }

    Ok((1.0 / (p * gap) + 1.0 / p).recip().min(PAIRS_MAX))
}

/// `(r_s, E^Z)`: the fraction of clicked pairs surviving as Z-pairs (opposite intensity patterns
/// on BOTH arms, `z_i xor z_j = 11`, four of sixteen, `r_s -> 1/8` in the weak-signal limit) and
/// their bit error rate. Zeng Eqs. (98) and (102). `E^Z` has no misalignment term and is exactly
/// 0.0 at `dark = 0`; misalignment enters through `mdi_yield`'s phase error.
#[pyfunction]
pub(crate) fn pairing_sift(eta_s: f64, mu: f64, dark: f64) -> PyResult<(f64, f64)> {
    check_unit("eta_s", eta_s)?;
    check_pos("mu", mu)?;
    check_dark("dark", dark)?;

    let row = click_row(eta_s, mu, dark);
    let (effective, wrong) = pair_sums(row);
    let p = click_mean(row);

    if !(effective > 0.0) || !(p > 0.0) {
        return Ok((0.0, E_CAP));
    }

    Ok((
        (effective / (16.0 * p * p)).min(1.0),
        (wrong / effective).min(E_CAP),
    ))
}

/// `q11`, the fraction of sifted Z-pairs in which each party's lit slot held exactly one photon,
/// Zeng Eq. (103): `(mu*exp(-mu))^2 * sum_n / sum_eff`, `sum_n` the four-pattern sum re-read in
/// photon number. A conditional probability, NOT a gain: `exp(-2*mu)` in the weak-signal limit,
/// 0.3679 at Zeng's large-`span` optimum `mu = 1/2`. Capped at 1 for round-off.
#[pyfunction]
pub(crate) fn pairing_single(eta_s: f64, mu: f64, dark: f64) -> PyResult<f64> {
    check_unit("eta_s", eta_s)?;
    check_pos("mu", mu)?;
    check_dark("dark", dark)?;

    let (effective, _) = pair_sums(click_row(eta_s, mu, dark));

    if !(effective > 0.0) {
        return Ok(0.0);
    }

    let empty = click_num(eta_s, dark, 0);
    let one = click_num(eta_s, dark, 1);
    let both = click_num(eta_s, dark, 2);
    let counted = 2.0 * (empty * both + one * one);
    let poisson = mu * (-mu).exp();

    Ok((poisson * poisson * counted / effective).min(1.0))
}

/// `floor(clock * coherence)` rounds, Zeng's own estimate ("`l` can be estimated by multiplying
/// the laser coherence time by the system repetition rate"): his 5 us at 625 MHz gives 3125,
/// against the `l = 3000 ~ 4000` he quotes. `clock` in Hz, `coherence` in s. A window shorter
/// than one round is REFUSED, not floored to 0.
#[pyfunction]
pub(crate) fn pairing_span(clock: f64, coherence: f64) -> PyResult<u64> {
    check_pos("clock", clock)?;
    check_pos("coherence", coherence)?;

    let rounds = clock * coherence;

    if rounds < 1.0 {
        return Err(PyValueError::new_err(format!(
            "clock*coherence = {rounds} rounds: the phase reference does not survive one round, \
             so no two rounds can be paired"
        )));
    }

    if rounds > SPAN_MAX {
        return Err(PyValueError::new_err(format!(
            "clock*coherence = {rounds} rounds exceeds {SPAN_MAX}, a coherence time nobody meant \
             to type"
        )));
    }

    Ok(rounds.floor() as u64)
}

/// `(phase, e_x)`: mean phase excursion across a `span`-round window at `clock` Hz, and the
/// interference error it leaves, Xie Sec. III. `drift` is the differential drift rate in rad/s:
/// the measured fibre figure with lasers locked (8 rad/ms at 402 km), `2*pi*dv` free-running.
/// The half is Xie's: `phase = drift*span/(2*clock)`, `e = (1 - V)/2` at
/// `V = (1 - 2*misalign)*cos(phase)`; neither paper prints that composition. The cosine is taken
/// at `min(phase, pi)`, else a LONGER window reports a SMALLER error; `phase` is unclamped.
#[pyfunction]
pub(crate) fn pairing_drift(
    clock: f64,
    span: u64,
    drift: f64,
    misalign: f64,
) -> PyResult<(f64, f64)> {
    check_pos("clock", clock)?;
    check_span(span)?;
    check_nonneg("drift", drift)?;
    check_err("misalign", misalign)?;

    let phase = 0.5 * drift * (span as f64) / clock;
    let vis = (1.0 - 2.0 * misalign) * phase.min(std::f64::consts::PI).cos();

    Ok((phase, ((1.0 - vis) / 2.0).min(E_CAP)))
}

/// Asymptotic rate, bits per ROUND, Zeng Eq. (7) restated as Eq. (90): `R = r_p * r_s *
/// {q11*[1 - h2(e_x)] - f_ec*h2(e_z)}`; `e_x` is Ma & Razavi via `mdi_yield`, NOT computed here.
/// Per round, comparable with `mdi_rate` per pulse pair. Do NOT halve it: `pairs` already charges
/// both rounds (Zeng's Eq. (105) halves the time-bin MDI rate instead). Clamped at zero.
#[pyfunction]
pub(crate) fn pairing_rate(
    pairs: f64,
    sift: f64,
    q11: f64,
    e_x: f64,
    e_z: f64,
    f_ec: f64,
) -> PyResult<f64> {
    check_prob("pairs", pairs)?;
    check_prob("sift", sift)?;
    check_prob("q11", q11)?;
    check_err("e_x", e_x)?;
    check_err("e_z", e_z)?;
    check_fec("f_ec", f_ec)?;

    if pairs > PAIRS_MAX {
        return Err(PyValueError::new_err(format!(
            "pairs must be at most {PAIRS_MAX}, got {pairs}: two rounds per pair. Pass \
             `pairing_pairs(p, span)`'s result, not the announcement probability p"
        )));
    }

    let rate = pairs * sift * (q11 * (1.0 - h2(e_x)) - f_ec * h2(e_z));

    Ok(rate.max(0.0))
}

/// One party's PAIR intensity ladder, the six values of `mu_i + mu_j` over `{0, nu, mu}` per
/// round with their probabilities, ascending: Zeng's Eq. (72) marginal, `probs = (s_0, s_nu,
/// s_mu)`. `decoy_bounds` does NOT apply: `Pr(k|mu)` here is Poisson in the SUM of two slots
/// (Zeng Eq. (73)). `mdi_y11` does apply to a Z-pair grid (Zeng Eq. (89); ladder `[mu, nu, 0]`);
/// the X-pairs need all six rungs. Coinciding rungs (`mu = 2*nu`) are not refused.
#[pyfunction]
pub(crate) fn pairing_ladder(
    nu: f64,
    mu: f64,
    probs: (f64, f64, f64),
) -> PyResult<Vec<(f64, f64)>> {
    check_pos("nu", nu)?;
    check_pos("mu", mu)?;
    check_shares(probs)?;

    if nu >= mu {
        return Err(PyValueError::new_err(format!(
            "the decoy must be weaker than the signal: nu < mu, got nu = {nu}, mu = {mu}"
        )));
    }

    let (empty, weak, strong) = probs;
    let mut rungs = vec![
        (0.0, empty * empty),
        (nu, 2.0 * empty * weak),
        (mu, 2.0 * empty * strong),
        (2.0 * nu, weak * weak),
        (nu + mu, 2.0 * weak * strong),
        (2.0 * mu, strong * strong),
    ];
    rungs.sort_by(|a, b| a.0.total_cmp(&b.0));

    Ok(rungs)
}

/// `(z, x, zero, dropped)`, how one party's pair of rounds is labelled at `(s_0, s_nu, s_mu)`:
/// Zeng's basis-assignment table, Methods, "Mode-pairing scheme with decoy states". `Z` when
/// exactly one slot is empty, `X` when both are lit at the SAME intensity, `'0'` when both are
/// empty, dropped otherwise. Sum to 1.
#[pyfunction]
pub(crate) fn pairing_bases(probs: (f64, f64, f64)) -> PyResult<(f64, f64, f64, f64)> {
    check_shares(probs)?;

    let (empty, weak, strong) = probs;

    Ok((
        2.0 * empty * (weak + strong),
        weak * weak + strong * strong,
        empty * empty,
        2.0 * weak * strong,
    ))
}

fn check_shares(probs: (f64, f64, f64)) -> PyResult<()> {
    let (empty, weak, strong) = probs;
    for (name, x) in [("s_0", empty), ("s_nu", weak), ("s_mu", strong)] {
        check_prob(name, x)?;
    }

    let total = empty + weak + strong;

    if (total - 1.0).abs() <= 1e-9 {
        return Ok(());
    }

    Err(PyValueError::new_err(format!(
        "the intensity probabilities must sum to 1, got {empty} + {weak} + \
         {strong} = {total}"
    )))
}

// Finite key: Xie, Lu, Weng, Cao, Jia, Bao, Wang, Fu, Yin & Chen, "Breaking the Rate-Loss Bound
// of Quantum Key Distribution with Asynchronous Two-Photon Interference", PRX Quantum 3, 020315
// (2022), arXiv:2112.11635, its key-length equation in the Postprocessing step and its
// "Statistical fluctuation analysis" appendix.
//
// TWO CITATION TRAPS. The mode-pairing paper is arXiv:2201.04300, NOT 2201.04956, which resolves
// to an astrobiology paper. Its authors are Zeng, Zhou, WU & Ma; "Zeng, Zhou, Yin & Zhang" merges
// them with the asynchronous-MDI team (Xie, Lu, Weng, Cao, Jia, Bao, Wang, Fu, Yin & Chen).
//
// `EPS_TERMS` is Xie's count, `eps_sec = 2(eps' + eps_hat + 2 eps_e) + eps_beta + eps_0 + eps_1 +
// eps_PA`, the Chernoff bound used 14 times symmetric and 13 asymmetric, so one common `eps`
// gives 24 and 23. The same paper's simulation-parameter table prints `eps = 36/23 x 10^-10`,
// 1.57e-10, which reproduces NEITHER of its own two bounds under its own composition; no numeric
// anchor is taken from it.
//
// TWO LINKS OF XIE'S CHAIN ARE NOT HERE, and no linear programme retires either: `s_11^x` lacks
// the phase slice `m` (`pairing_click` is phase-averaged); `s_11^z` lacks Xie's DECLARED vacuum
// (`pairing_bases` has no setting for it). On `pairing_yield` the closed form is already the
// optimum: `lp_dual` reaches it from below, relative 1e-14 at `(nu, mu) = (0.05, 0.4)`.

/// `eps_sec = 24 eps`, the SYMMETRIC case; asymmetric is 23.
const EPS_TERMS: f64 = 24.0;

/// `beta = ln(1/eps)`.
fn beta(eps: f64) -> f64 {
    (1.0 / eps).ln()
}

/// `eps_sec/24`, the per-bound failure probability; the "one number, unsplit" convention of
/// `q.FiniteSize` and `q.KeyBlock`.
#[pyfunction]
pub(crate) fn pairing_eps(eps_sec: f64) -> PyResult<f64> {
    check_eps("eps_sec", eps_sec)?;

    Ok(eps_sec / EPS_TERMS)
}

/// `(lo, hi)` an OBSERVED count may take about a known mean, Xie's Chernoff pair `phi^L(x*) = x*
/// - sqrt(2 beta x*)`, `phi^U(x*) = x* + beta/2 + sqrt(2 beta x* + beta^2/4)`. The direction a
/// SIMULATION runs; `pairing_expect` is NOT its inverse, the widths differing at order `beta`.
/// Lower bound clamped at zero.
#[pyfunction]
pub(crate) fn pairing_observe(mean: f64, eps: f64) -> PyResult<(f64, f64)> {
    check_nonneg("mean", mean)?;
    check_eps("eps", eps)?;

    let b = beta(eps);
    let lo = mean - (2.0 * b * mean).sqrt();
    let hi = mean + 0.5 * b + (2.0 * b * mean + 0.25 * b * b).sqrt();

    Ok((lo.max(0.0), hi))
}

/// `(lo, hi)` the EXPECTED value may take given an observed count, Xie's variant Chernoff pair
/// `x*_lo = max(x - beta/2 - sqrt(2 beta x + beta^2/4), 0)`, `x*_hi = x + beta + sqrt(2 beta x +
/// beta^2)`. The direction a real run needs; every estimator below is fed from it.
#[pyfunction]
pub(crate) fn pairing_expect(x: f64, eps: f64) -> PyResult<(f64, f64)> {
    check_nonneg("x", x)?;
    check_eps("eps", eps)?;

    let b = beta(eps);
    let lo = x - 0.5 * b - (2.0 * b * x + 0.25 * b * b).sqrt();
    let hi = x + b + (2.0 * b * x + b * b).sqrt();

    Ok((lo.max(0.0), hi))
}

/// Lower bound on one arm's single-photon yield against the other arm's vacuum, Xie's decoy
/// equations for `y_01` and `y_10`: `y >= (mu/(mu*nu - nu^2)) * [ e^nu q_nu - (nu^2/mu^2) e^mu
/// q_mu - ((mu^2 - nu^2)/mu^2) q_0 ]`, `q_nu` a LOWER bound on the per-round gain at (vacuum,
/// decoy), `q_mu` an UPPER bound at (vacuum, signal), `q_0` an UPPER bound at (vacuum, vacuum),
/// dimensionless as `decoy::decoy_bounds` takes them. The one one-dimensional decoy step: a
/// Z-pair has one lit slot per party. The X-pairs need `pairing_ladder`'s six rungs and are not
/// estimated here. Clamped into `[0, 1]`.
#[pyfunction]
pub(crate) fn pairing_yield(
    nu: f64,
    mu: f64,
    q_nu: f64,
    q_mu: f64,
    q_vac: f64,
) -> PyResult<f64> {
    check_pos("nu", nu)?;
    check_pos("mu", mu)?;
    check_prob("q_nu", q_nu)?;
    check_prob("q_mu", q_mu)?;
    check_prob("q_vac", q_vac)?;

    if nu >= mu {
        return Err(PyValueError::new_err(format!(
            "the decoy must be weaker than the signal: nu < mu, got nu = {nu}, mu = {mu}"
        )));
    }

    let ratio = nu * nu / (mu * mu);
    let bracket = nu.exp() * q_nu - ratio * mu.exp() * q_mu - (1.0 - ratio) * q_vac;

    Ok((mu * bracket / (mu * nu - nu * nu)).clamp(0.0, 1.0))
}

/// Xie's sampling-without-replacement penalty `gamma^U(n, k, lambda, eps)`, by which the
/// UNSAMPLED string's error rate may exceed the sampled one: `[ (1-2L)AG/(n+k) + sqrt(A^2
/// G^2/(n+k)^2 + 4 L(1-L) G) ] / [ 2 + 2 A^2 G/(n+k)^2 ]`, `A = max(n, k)`, `G = ((n+k)/(nk))
/// ln((n+k)/(2 pi n k L(1-L) eps^2))`, `L = lambda`; `n` unsampled, `k` sampled. A third form
/// beside Lim's `gamma` (`discrete::bb84_phase`) and Curty's `Upsilon` (`mdi_phase`), NOT
/// interchangeable. The `lambda -> 0` limit does not vanish, unlike Lim's: it tends to
/// `(n+k)/(2 max(n,k))`, so `lam <= 0` returns `E_CAP`, as does `G <= 0`.
#[pyfunction]
pub(crate) fn pairing_gamma(n: f64, k: f64, lam: f64, eps: f64) -> PyResult<f64> {
    check_nonneg("n", n)?;
    check_nonneg("k", k)?;
    check_prob("lam", lam)?;
    check_eps("eps", eps)?;

    if !(n > 0.0) || !(k > 0.0) || !(lam > 0.0) || lam >= 1.0 {
        return Ok(E_CAP);
    }

    let span = n + k;
    let big = n.max(k);
    let mass = lam * (1.0 - lam);
    let g = (span / (n * k)) * (span / (2.0 * std::f64::consts::PI * n * k * mass * eps * eps)).ln();

    if !(g > 0.0) {
        return Ok(E_CAP);
    }

    let shape = big * g / span;
    let num = (1.0 - 2.0 * lam) * shape + (shape * shape + 4.0 * mass * g).sqrt();
    let den = 2.0 + 2.0 * big * shape / span;

    Ok((num / den).clamp(0.0, E_CAP))
}

/// Upper bound on the key basis's single-photon-pair PHASE error, Xie's `phi_11^z <= e_11^x +
/// gamma^U(s_11^z, s_11^x, e_11^x, eps)`: `s_key`, `s_test` the certified single-photon-pair
/// counts in Z and X, `e_test` the X-basis bit error. Capped at 1/2.
#[pyfunction]
pub(crate) fn pairing_phase(
    s_key: f64,
    s_test: f64,
    e_test: f64,
    eps: f64,
) -> PyResult<f64> {
    check_nonneg("s_key", s_key)?;
    check_nonneg("s_test", s_test)?;
    check_err("e_test", e_test)?;
    check_eps("eps", eps)?;

    let width = pairing_gamma(s_key, s_test, e_test, eps)?;

    Ok((e_test + width).min(E_CAP))
}

/// Finite-key length in BITS, Xie's key-length equation: `l = s_0 + s_11[1 - H2(phi_11)] -
/// lambda_EC - log2(2/eps_cor) - 2 log2(2/(eps' eps_hat)) - 2 log2(1/(2 eps_PA))`, `lambda_EC =
/// n_z f H2(E_z)`, `eps' = eps_hat = eps_PA = eps_sec/24`. `s0` is the certified count of Z-pairs
/// with both of Alice's matched slots vacuum, `s11` the single-photon-pair count, `phi` from
/// `pairing_phase`, `n_z` the sifted Z-PAIR count; divide by the round count `N` for
/// `pairing_rate`'s unit. Xie prints `log2(2/eps_cor)` where Curty's MDI-BB84 length prints
/// `log2(8/eps_cor)`, each kept as printed. Clamped at zero and floored.
#[pyfunction]
pub(crate) fn pairing_length(
    s0: f64,
    s11: f64,
    phi: f64,
    n_z: f64,
    e_z: f64,
    f_ec: f64,
    eps_sec: f64,
    eps_cor: f64,
) -> PyResult<f64> {
    check_nonneg("s0", s0)?;
    check_nonneg("s11", s11)?;
    check_err("phi", phi)?;
    check_nonneg("n_z", n_z)?;
    check_err("e_z", e_z)?;
    check_fec("f_ec", f_ec)?;
    check_eps("eps_sec", eps_sec)?;
    check_eps("eps_cor", eps_cor)?;

    if s0 + s11 > n_z {
        return Err(PyValueError::new_err(format!(
            "s0 + s11 must not exceed n_z, got {} > {n_z}: the vacuum and single-photon pairs \
             are part of the sifted Z-pairs",
            s0 + s11
        )));
    }

    let eps = eps_sec / EPS_TERMS;
    let leak = f_ec * n_z * h2(e_z);
    let budget = (2.0 / eps_cor).log2() + 2.0 * (2.0 / (eps * eps)).log2()
        + 2.0 * (1.0 / (2.0 * eps)).log2();
    let len = s0 + s11 * (1.0 - h2(phi)) - leak - budget;

    Ok(len.max(0.0).floor())
}
