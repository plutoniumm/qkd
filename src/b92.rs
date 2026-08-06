use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::numeric::{bisect, golden};
use crate::std::{
    check_eps, check_err, check_fec, check_gate, check_pos, check_prob, check_unit, click_p,
    h2, sqrt0, E_CAP,
};

// B92, Bennett, Phys. Rev. Lett. 68, 3121 (1992). Threshold-detector family: click.rs's
// conventions hold, |alpha|^2 a photon number and `dark` PER GATE.
//
// PLAIN B92 DERIVES `e_ph`: Tamaki & Lutkenhaus, Phys. Rev. A 69, 032316 (2004), quant-ph/0308048,
// Sec. III, extending the loss-free Tamaki, Koashi & Imoto, "Unconditionally Secure Key
// Distribution Based on Two Nonorthogonal States", Phys. Rev. Lett. 90, 167904 (2003),
// quant-ph/0212162. Single-photon source.
//
// STRONG-REFERENCE B92 SUPPLIES `e_ph` ON ONE RECEIVER AND DERIVES IT ON ANOTHER. Supplied:
// Koashi, Phys. Rev. Lett. 93, 120501 (2004), quant-ph/0403131, `b92_point(reference=true)`; his
// Eqs. (2)-(3) have no closed form. Derived: Tamaki, Lutkenhaus, Koashi & Batuwantudawe, Phys.
// Rev. A 80, 032302 (2009), read as quant-ph/0607082 V2 (V1 estimated from the FAILED pairs, its
// note [15]), on a photon-number-resolving receiver, `b92_strong`. Do not carry a number between
// the two receivers.
//
// USD breaks plain B92 at loss L = |<phi_0|phi_1>| on a NOISELESS channel only; noise moves it IN.
// Koashi's O(t) scaling is asymptotic in low transmittance.
//
// THE ONE SEAM: `b92_point`'s plain branch prices a coherent-state receiver with Tamaki &
// Lutkenhaus's single-photon proof -- a defensible number, not a proof; they name coherent-state
// B92 an open problem. `b92_plain` has no seam.

/// `b92_optimum`'s `mu` scan.
const SCAN_LO: f64 = 1e-6;

const SCAN_HI: f64 = 1e2;

const SCAN_N: usize = 400;

/// `b92_phase`'s split scan, refined by golden section in the winning cell.
const SPLIT_N: usize = 256;

/// `b92_limit`'s overlap scan and bisection depth.
const OVER_N: usize = 160;

const LIMIT_N: usize = 40;

const LIMIT_HI: f64 = 0.25;

/// The zero-bit-error case is a TANGENCY; a strict test loses the exact answer on a last bit.
const SLACK: f64 = 1e-12;

/// Strong-reference solve depths, and the Poisson term below which the window sum stops moving.
const SRP_N: usize = 200;

const SRP_TINY: f64 = 1e-300;

/// `0.5*ln(2*pi)`.
const LN_ROOT_TAU: f64 = 0.918_938_533_204_672_74;

/// Above this the Stirling series is at f64 round-off; smaller arguments are shifted up.
const STIRLING: f64 = 12.0;

/// `(hit, miss)` photon numbers on the displaced signal; at `V = 1`, `4*t*mu` and `0`, Koashi's
/// `|<-beta|gamma>|^2 = exp(-4*t*mu)` as a click.
fn ports(mu: f64, t: f64, vis: f64) -> (f64, f64) {
    let flux = 2.0 * t * mu;

    (flux * (1.0 + vis), flux * (1.0 - vis))
}

fn detect(mu: f64, t: f64, eta: f64, vis: f64, dark: f64) -> (f64, f64) {
    let (hit, miss) = ports(mu, t, vis);
    let p_hit = click_p(eta, hit, dark);
    let p_miss = click_p(eta, miss, dark);
    let total = p_hit + p_miss;
    if total <= 0.0 {
        return (0.0, 0.0);
    }

    (0.5 * total, (p_miss / total).min(0.5))
}

/// Overlap on Eve's beam-split share: the fraction unambiguous discrimination leaves her ignorant of.
fn opaque(mu: f64, t: f64) -> f64 {
    (-2.0 * (1.0 - t) * mu).exp()
}

fn ceiling(mu: f64, t: f64, eta: f64, vis: f64, dark: f64) -> f64 {
    detect(mu, t, eta, vis, dark).0.min(opaque(mu, t))
}

fn bracket(e_bit: f64, e_ph: f64, f_ec: f64) -> f64 {
    1.0 - h2(e_ph) - f_ec * h2(e_bit)
}

/// Largest `x` one split `L_0` admits, or `-1` when it cannot have produced the data: the upper
/// root of Tamaki & Lutkenhaus's `f(x) >= g`, squared twice into a quadratic. A negative
/// discriminant is CLAMPED, not rejected: the zero-bit-error tangency, and overshoot is safe.
fn xcap(l0: f64, c: f64, loss: f64, delta: f64, g: f64) -> f64 {
    let span = 1.0 - loss;
    let l1 = loss - l0;
    let a = (delta + l1 / c).abs();
    let b = (c - delta - l0 / c).abs();
    let top = span - b;
    let w = span * span - g * g;
    if top < a - SLACK || w <= 0.0 {
        return -1.0;
    }

    let k = g * g - span * span - a * a + b * b;
    let u = (-span * k + g * sqrt0(k * k - 4.0 * a * a * w)) / (2.0 * w);
    if u < a - SLACK {
        return -1.0;
    }

    u.max(a).min(top) + l1
}

/// Tamaki-Lutkenhaus phase error `n_ph_bar/n_fil`. The split runs over `[alpha^2 L, beta^2 L]`,
/// NOT `[0, L]`: a wider scan inflates the maximum. Do not relax `lo`/`hi`.
fn phase(c: f64, loss: f64, gain: f64, e_bit: f64) -> f64 {
    let pair = (1.0 - c * c) / 2.0;
    let delta = (gain - pair) / c;
    let g = gain * (1.0 - 2.0 * e_bit) / (sqrt0(1.0 - c * c) / 2.0);
    if !g.is_finite() || g <= 0.0 {
        return 0.5;
    }

    let lo = 0.5 * (1.0 - c) * loss;
    let hi = 0.5 * (1.0 + c) * loss;
    let mut best = xcap(lo, c, loss, delta, g);
    if hi > lo {
        let step = (hi - lo) / (SPLIT_N as f64);
        let mut at = 0usize;
        for i in 1..=SPLIT_N {
            let v = xcap(lo + step * (i as f64), c, loss, delta, g);
            if v > best {
                best = v;
                at = i;
            }
        }

        let left = (lo + step * (at as f64) - step).max(lo);
        let right = (lo + step * (at as f64) + step).min(hi);
        let crest = golden(left, right, 60, |x| xcap(x, c, loss, delta, g));
        let v = xcap(crest, c, loss, delta, g);
        if v > best {
            best = v;
        }
    }

    if best < 0.0 {
        return 0.5;
    }

    ((best + c * delta) / (2.0 * gain)).clamp(0.0, 0.5)
}

/// `(gain, e_bit)` at `gamma = beta` over Tamaki & Lutkenhaus's example channel `rho -> L|V><V| +
/// (1 - L)[(1 - p) rho + (p/3) sum_i sigma_i rho sigma_i]`, Bloch vector shrunk by `1 - 4p/3`.
fn depolarise(c: f64, loss: f64, depol: f64) -> (f64, f64) {
    let lam = 1.0 - 4.0 * depol / 3.0;
    let flip = 0.25 * (1.0 - lam);
    let total = 0.5 * (1.0 - c * c) * lam + 2.0 * flip;
    if total <= 0.0 {
        return (0.0, 0.0);
    }

    ((1.0 - loss) * total, flip / total)
}

/// The single-photon gain Tamaki & Lutkenhaus's channel attaches to (overlap, loss, bit error).
fn regain(c: f64, loss: f64, e_bit: f64) -> f64 {
    let den = 1.0 - 2.0 * e_bit * c * c;
    if den <= 0.0 {
        return 0.0;
    }

    let w = 2.0 * e_bit * (1.0 - c * c) / den;

    0.5 * (1.0 - loss) * ((1.0 - c * c) + w * c * c)
}

/// `(gain, e_bit, e_ph, rate)` over the depolarising-with-loss channel.
fn tamaki(c: f64, loss: f64, depol: f64, f_ec: f64) -> (f64, f64, f64, f64) {
    let (gain, e_bit) = depolarise(c, loss, depol);
    if gain <= 0.0 {
        return (0.0, 0.0, 0.5, 0.0);
    }

    let e_ph = phase(c, loss, gain, e_bit);

    (gain, e_bit, e_ph, (gain * bracket(e_bit, e_ph, f_ec)).max(0.0))
}

/// `(rate, overlap)` maximised over the overlap.
fn sweep(loss: f64, depol: f64, f_ec: f64) -> (f64, f64) {
    let mut best = (0.0f64, 0.0f64);
    for i in 1..OVER_N {
        let c = (i as f64) / (OVER_N as f64);
        let rate = tamaki(c, loss, depol, f_ec).3;
        if rate > best.0 {
            best = (rate, c);
        }
    }

    best
}

/// `ln(Gamma(z))`, `z > 0`, Stirling after shifting past `STIRLING`.
fn ln_gamma(z0: f64) -> f64 {
    let mut z = z0;
    let mut shift = 0.0;
    while z < STIRLING {
        shift += z.ln();
        z += 1.0;
    }

    let inv = 1.0 / z;
    let sq = inv * inv;
    let tail = inv * (1.0 / 12.0 - sq * (1.0 / 360.0 - sq * (1.0 / 1260.0 - sq / 1680.0)));

    (z - 0.5) * z.ln() - z + LN_ROOT_TAU + tail - shift
}

/// `sum_{nu=lo}^{hi} exp(-x) x^nu/nu!`, anchored at the mode CLAMPED INTO THE WINDOW. `nu` reaches
/// 1e9: do not use `std::ln_fact`.
fn pois_span(x: f64, lo: u64, hi: u64) -> f64 {
    if hi < lo || x <= 0.0 {
        return 0.0;
    }

    let mid = x.round().max(lo as f64).min(hi as f64);
    let k0 = mid as u64;
    let peak = -x + (k0 as f64) * x.ln() - ln_gamma(k0 as f64 + 1.0);
    if peak < -700.0 {
        return 0.0;
    }

    let mut total = 1.0f64;
    let mut term = 1.0f64;
    let mut k = k0;
    while k > lo && term > SRP_TINY {
        term *= (k as f64) / x;
        total += term;
        k -= 1;
    }

    term = 1.0;
    k = k0;
    while k < hi && term > SRP_TINY {
        term *= x / ((k + 1) as f64);
        total += term;
        k += 1;
    }

    (total * peak.exp()).clamp(0.0, 1.0)
}

/// `(alpha_nu^2, beta_nu^2, G_nu)` of Tamaki, Lutkenhaus, Koashi & Batuwantudawe's Eq. (7), `R`
/// the reflectivity of BS1.
fn srp_qubit(nu: f64, r: f64) -> (f64, f64, f64) {
    let nr = nu * r;
    let gain = ((nu - 1.0) * (-r).ln_1p()).exp() * (1.0 + nr);

    (nr / (1.0 + nr), 1.0 / (1.0 + nr), gain)
}

/// `C_nu` of Eq. (B11), the inverse of Eq. (B9) written out; checked symbolically against (B9)
/// under `alpha^2 + beta^2 = 1`.
fn srp_cone(nu: f64, r: f64) -> [[f64; 4]; 4] {
    let (a2, b2, g) = srp_qubit(nu, r);
    let d = b2 - a2;

    [
        [b2 * b2 / d, -b2, -a2 / (g * d), -1.0 / g],
        [-a2 * a2 / d, -a2, a2 / (g * d), 1.0 / g],
        [a2 * b2 / d, b2, -b2 / (g * d), 1.0 / g],
        [-a2 * b2 / d, a2, b2 / (g * d), -1.0 / g],
    ]
}

/// `C'` of Eq. (B12), the entrywise maximum of `C_nu` over the photon range. Two endpoints are
/// the whole maximum only under `srp_span`'s conditions; a missed maximum fails insecure.
fn srp_wall(lo: f64, hi: f64, r: f64) -> [[f64; 4]; 4] {
    let a = srp_cone(lo, r);
    let b = srp_cone(hi, r);
    let mut out = a;
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j].max(b[i][j]);
        }
    }

    out
}

/// `(nu_i, nu_f, share)`: D1's acceptance window `[nu_i, nu_f - 1]`, the total-photon range
/// `lambda = [nu_i, nu_f]`, and the SRP's Poisson weight inside the acceptance window.
fn srp_span(kappa: f64, mu: f64, t: f64, eta: f64, width: f64) -> PyResult<(u64, u64, f64)> {
    let r = kappa / mu;
    let seen = t * eta * mu;
    let half = width * seen.sqrt();
    let lo = (seen - half).round().max(1.0);
    let hi = (seen + half).round();
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo || hi > (u64::MAX as f64) {
        return Err(PyValueError::new_err(format!(
            "no photon-number window at kappa = {kappa}, mu = {mu}, t = {t}, eta = {eta}, \
             width = {width}: the strong reference arrives with mean {seen:e} photons and the \
             window [{lo:e}, {hi:e}] is empty or unrepresentable"
        )));
    }

    // Without both, Appendix B's C_nu entries change the sign of their derivative.
    if hi * r >= 1.0 || hi > -1.0 / (2.0 * (-r).ln_1p()) {
        return Err(PyValueError::new_err(format!(
            "the strong-reference window reaches nu_f = {hi:e} at reflectivity R = {r:e}, past \
             Appendix B's conditions nu_f*R < 1 and nu_f <= -1/(2 ln(1 - R)) of Tamaki, \
             Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302 (2009), where the \
             entrywise maximum of Eq. (B12) leaves the endpoints. Raise mu against kappa"
        )));
    }

    Ok((lo as u64, hi as u64, pois_span(t * eta * (mu - kappa), lo as u64, hi as u64 - 1)))
}

/// Eq. (17)'s five rates, `(fil_all, fil_win, vac, bit_win, bit_all)`; see `b92_monitor`.
fn srp_watch(kappa: f64, t: f64, eta: f64, noise: f64, share: f64) -> [f64; 5] {
    let seen = t * eta * kappa;
    let clean = (-2.0 * seen).exp();
    let fil_all = clean * 2.0 * seen * (1.0 - noise) + (-seen).exp() * noise;
    let bit_all = 0.5 * (-seen).exp() * noise;

    [fil_all, fil_all * share, clean * (1.0 - noise) * share, bit_all * share, bit_all]
}

/// The largest `Lambda_ph,lambda` Eq. (15) admits, or `None`. `g` is CONCAVE in the unknown: the
/// admissible set is one interval. The paper's square roots hold only inside the non-negativity
/// domain, solved first.
fn srp_solve(wall: [[f64; 4]; 4], watch: [f64; 5], top: f64, tilt: f64, pre: f64) -> Option<f64> {
    let upper = [1.0, tilt, watch[0]];
    let lower = [top, tilt - 1.0 + top, watch[1]];
    let mut base = [0.0f64; 4];
    let mut slope = [0.0f64; 4];
    for i in 0..4 {
        for j in 0..3 {
            base[i] += wall[i][j] * if wall[i][j] >= 0.0 { upper[j] } else { lower[j] };
        }

        slope[i] = wall[i][3];
    }

    let mut lo = 0.0f64;
    let mut hi = watch[0];
    for i in 0..4 {
        if slope[i] > 0.0 {
            lo = lo.max(-base[i] / slope[i]);
        } else if slope[i] < 0.0 {
            hi = hi.min(-base[i] / slope[i]);
        }
    }

    if !(lo.is_finite() && hi.is_finite()) || hi < lo {
        return None;
    }

    let reach = |x: f64| {
        let u: Vec<f64> = (0..4).map(|i| (base[i] + slope[i] * x).max(0.0)).collect();

        pre * (sqrt0(u[0] * u[3]) + sqrt0(u[1] * u[2]))
    };
    let want = watch[1] - 2.0 * watch[4];
    if reach(hi) >= want {
        return Some(hi);
    }

    let crest = golden(lo, hi, SRP_N, &reach);
    if reach(crest) < want {
        return None;
    }

    Some(bisect(crest, hi, SRP_N, |x| reach(x) >= want).0)
}

/// `(p_fil, e_bit, e_ph, rate)`.
fn srp_key(
    kappa: f64,
    mu: f64,
    t: f64,
    eta: f64,
    noise: f64,
    width: f64,
    f_ec: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let (lo, hi, share) = srp_span(kappa, mu, t, eta, width)?;
    let watch = srp_watch(kappa, t, eta, noise, share);
    if watch[1] <= 0.0 {
        return Ok((0.0, 0.0, E_CAP, 0.0));
    }

    let r = kappa / mu;
    let wall = srp_wall(lo as f64, hi as f64, r);
    let (a2, b2, gain) = srp_qubit(hi as f64, r);
    let tilt = 0.5 * (1.0 - (-2.0 * kappa).exp());
    let top = watch[2] + watch[1];
    let solved = srp_solve(wall, watch, top, tilt, 2.0 * gain * (a2 * b2).sqrt());
    let e_ph = match solved {
        Some(v) => (v / watch[1]).clamp(0.0, E_CAP),
        None => E_CAP,
    };
    let e_bit = (watch[3] / watch[1]).min(E_CAP);

    Ok((watch[1], e_bit, e_ph, (watch[1] * bracket(e_bit, e_ph, f_ec)).max(0.0)))
}

fn check_optics(mu: f64, t: f64, eta: f64, vis: f64, dark: f64) -> PyResult<()> {
    check_pos("mu", mu)?;
    check_unit("t", t)?;
    check_unit("eta", eta)?;
    check_prob("visibility", vis)?;
    check_gate("dark", dark)?;

    Ok(())
}

/// `|<alpha|-alpha>| = exp(-2*mu)`, `mu = |alpha|^2`: Koashi's `<-beta|beta> = exp(-2*|beta|^2)`,
/// Tamaki & Lutkenhaus's `<phi_0|phi_1> = beta^2 - alpha^2`.
#[pyfunction]
pub(crate) fn b92_overlap(mu: f64) -> PyResult<f64> {
    check_pos("mu", mu)?;

    Ok((-2.0 * mu).exp())
}

/// Optimal unambiguous-discrimination success `1 - c` for two equiprobable pure states of overlap
/// `c`: Ivanovic, Phys. Lett. A 123, 257 (1987); Dieks, Phys. Lett. A 126, 303 (1988); Peres,
/// Phys. Lett. A 128, 19 (1988); the equiprobable optimum Jaeger & Shimony, Phys. Lett. A 197, 83
/// (1995). EVE's number, not Bob's: plain B92 dies at loss `L = c` on a NOISELESS channel.
#[pyfunction]
pub(crate) fn b92_usd(overlap: f64) -> PyResult<f64> {
    check_prob("overlap", overlap)?;

    Ok(1.0 - overlap)
}

/// `(p_fil, e_bit)` per emitted pulse, DERIVED: Koashi's POVM `F_0 = (1 - |-beta><-beta|)/2`,
/// `F_1 = (1 - |beta><beta|)/2`, a displacement by a locked local oscillator and one threshold
/// gate, the `/2` Bob's random hypothesis. Never feed this to `b92_phase`: its single-photon gain
/// `(1 - c^2)/2 * (1 - L)` is smaller.
#[pyfunction]
pub(crate) fn b92_detect(
    mu: f64,
    t: f64,
    eta: f64,
    visibility: f64,
    dark: f64,
) -> PyResult<(f64, f64)> {
    check_optics(mu, t, eta, visibility, dark)?;

    Ok(detect(mu, t, eta, visibility, dark))
}

/// Upper bound on any COHERENT-source B92 rate over pure loss, bits per pulse: `min(p_fil,
/// exp(-2*(1 - t)*mu))`, Eve keeping the `(1 - t)` share and replacing the fibre with a lossless
/// one. One attack's ceiling, not the tightest; it says nothing about the SINGLE-PHOTON protocol.
/// Below the `b92_optimum` crossing `p_fil` is the smaller arm and `rate <= ceiling` tests the
/// bracket, not this bound.
#[pyfunction]
pub(crate) fn b92_ceiling(
    mu: f64,
    t: f64,
    eta: f64,
    visibility: f64,
    dark: f64,
) -> PyResult<f64> {
    check_optics(mu, t, eta, visibility, dark)?;

    Ok(ceiling(mu, t, eta, visibility, dark))
}

/// Tamaki-Lutkenhaus upper bound on PLAIN B92's phase error `n_ph_bar/n_fil`, DERIVED, clamped
/// into `[0, 1/2]`: Phys. Rev. A 69, 032316 (2004), Sec. III, maximised over the split of `L`.
/// DO NOT UNCLAMP: `e_ph = 0.53` at `e_bit = 0` leaves a POSITIVE rate outside the domain
/// `2 n_ph_bar <= n_fil`. At no bit errors `alpha^2/(beta^2 - alpha^2) * L/(1 - L)`, `1/2` at
/// `L = overlap`. `gain` is `n_fil/N` per EMITTED pulse, `loss` the fraction reaching no
/// decision, `overlap` Alice's `beta^2 - alpha^2`, a property of her PREPARATION: loss does not
/// widen it.
#[pyfunction]
pub(crate) fn b92_phase(overlap: f64, loss: f64, gain: f64, e_bit: f64) -> PyResult<f64> {
    check_eps("overlap", overlap)?;
    check_gate("loss", loss)?;
    check_unit("gain", gain)?;
    check_err("e_bit", e_bit)?;

    Ok(phase(overlap, loss, gain, e_bit))
}

/// `(gain, e_bit)` of Tamaki & Lutkenhaus's example channel `rho -> L|V><V| + (1 - L)[(1 - p) rho
/// + (p/3) sum_i sigma_i rho sigma_i]` at `gamma = beta`. `depol` is their `p`, a BB84 bit error
/// rate as `QBER = 2p/3`; B92's own is smaller.
#[pyfunction]
pub(crate) fn b92_channel(overlap: f64, loss: f64, depol: f64) -> PyResult<(f64, f64)> {
    check_eps("overlap", overlap)?;
    check_gate("loss", loss)?;
    check_prob("depol", depol)?;

    Ok(depolarise(overlap, loss, depol))
}

/// `(gain, e_bit, e_ph, rate)` bits per emitted pulse over the depolarising channel with loss,
/// nothing dialled: the SEAM-FREE single-photon branch. At `overlap = 0.68125` the rate crosses
/// zero at `loss = 0.68125, 0.61884, 0.52015, 0.42221` for `depol = 0, 0.001, 0.005, 0.01`.
#[pyfunction]
pub(crate) fn b92_plain(
    overlap: f64,
    loss: f64,
    depol: f64,
    f_ec: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_eps("overlap", overlap)?;
    check_gate("loss", loss)?;
    check_prob("depol", depol)?;
    check_fec("f_ec", f_ec)?;

    Ok(tamaki(overlap, loss, depol, f_ec))
}

/// `(depol, overlap)`: the largest depolarising rate plain B92 distils at one loss, and the
/// overlap attaining it. The first is Tamaki & Lutkenhaus's Fig. 2 number at `f_ec = 1`: `p ~
/// 0.034` at `L = 0`, `0.023` at `0.2`, `0.012` at `0.5`. The second is the AMPLITUDE overlap
/// (0.68125, 0.75, 0.85625); Fig. 2(b) plots its SQUARE (0.46410, 0.56250, 0.73316).
#[pyfunction]
pub(crate) fn b92_limit(loss: f64, f_ec: f64) -> PyResult<(f64, f64)> {
    check_gate("loss", loss)?;
    check_fec("f_ec", f_ec)?;

    let (lo, _) = bisect(0.0, LIMIT_HI, LIMIT_N, |depol| sweep(loss, depol, f_ec).0 > 0.0);

    Ok((lo, sweep(loss, lo, f_ec).1))
}

/// `p_fil * (1 - h2(e_ph) - f_ec*h2(e_bit))` bits per emitted pulse, clamped at zero. ONE
/// expression for Koashi, Phys. Rev. Lett. 93, 120501 (2004); Tamaki & Lutkenhaus, Phys. Rev. A
/// 69, 032316 (2004) at `gamma = beta`; and Tamaki, Koashi & Imoto, Phys. Rev. Lett. 90, 167904
/// (2003). `f_ec = 1` recovers their Shannon-limit charge. Domain `e_ph <= 1/2`.
#[pyfunction]
pub(crate) fn b92_rate(p_fil: f64, e_bit: f64, e_ph: f64, f_ec: f64) -> PyResult<f64> {
    check_prob("p_fil", p_fil)?;
    check_err("e_bit", e_bit)?;
    check_err("e_ph", e_ph)?;
    check_fec("f_ec", f_ec)?;

    Ok((p_fil * bracket(e_bit, e_ph, f_ec)).max(0.0))
}

/// The root of `f_ec*h2(e_bit) = 1 - h2(e_ph)` below 1/2; `0.5` when the phase term costs
/// nothing, `0` when everything. At `e_ph = e_bit`, `f_ec = 1` it is Shor-Preskill's 11.00%.
/// Published depolarising thresholds are `b92_limit`'s.
#[pyfunction]
pub(crate) fn b92_tolerance(e_ph: f64, f_ec: f64) -> PyResult<f64> {
    check_err("e_ph", e_ph)?;
    check_fec("f_ec", f_ec)?;

    let want = (1.0 - h2(e_ph)) / f_ec;
    if want <= 0.0 {
        return Ok(0.0);
    }

    if want >= 1.0 {
        return Ok(0.5);
    }

    let (lo, hi) = bisect(0.0, 0.5, 200, |e| h2(e) < want);

    Ok(0.5 * (lo + hi))
}

/// `(p_fil, e_bit, ceiling, rate)` bits per emitted pulse on the COHERENT hardware, rate =
/// `min(p_fil * bracket, b92_ceiling)` clamped at zero. `reference = true` is Koashi's variant
/// with `e_ph` SUPPLIED; `false` is plain B92 with `e_ph` DERIVED by `b92_phase` at
/// `overlap = exp(-2*mu)`, `loss = 1 - eta*t`, the supplied `e_ph` kept as a FLOOR. See the module
/// comment's seam.
#[pyfunction]
pub(crate) fn b92_point(
    mu: f64,
    t: f64,
    eta: f64,
    visibility: f64,
    dark: f64,
    e_ph: f64,
    f_ec: f64,
    reference: bool,
) -> PyResult<(f64, f64, f64, f64)> {
    check_optics(mu, t, eta, visibility, dark)?;
    check_err("e_ph", e_ph)?;
    check_fec("f_ec", f_ec)?;
    let (p_fil, e_bit) = detect(mu, t, eta, visibility, dark);
    let limit = ceiling(mu, t, eta, visibility, dark);
    let mut used = e_ph;
    if !reference {
        let over = (-2.0 * mu).exp();
        let loss = 1.0 - eta * t;
        let gain = regain(over, loss, e_bit);
        if over > 0.0 && over < 1.0 && loss < 1.0 && gain > 0.0 {
            used = used.max(phase(over, loss, gain, e_bit));
        } else {
            used = 0.5;
        }
    }

    let rate = (p_fil * bracket(e_bit, used, f_ec)).min(limit).max(0.0);

    Ok((p_fil, e_bit, limit, rate))
}

/// `(mu, overlap, ceiling)` maximising `b92_ceiling`, not `b92_rate`; the single-photon optimum
/// overlap is `b92_limit`'s second return. A lossless channel is REFUSED.
#[pyfunction]
pub(crate) fn b92_optimum(
    t: f64,
    eta: f64,
    visibility: f64,
    dark: f64,
) -> PyResult<(f64, f64, f64)> {
    check_optics(1.0, t, eta, visibility, dark)?;

    let lo = SCAN_LO.ln();
    let hi = SCAN_HI.ln();
    let step = |i: usize| lo + (hi - lo) * (i as f64) / (SCAN_N as f64);
    let mut best = (0usize, ceiling(lo.exp(), t, eta, visibility, dark));
    for i in 1..=SCAN_N {
        let c = ceiling(step(i).exp(), t, eta, visibility, dark);
        if c > best.1 {
            best = (i, c);
        }
    }

    if best.0 == 0 || best.0 == SCAN_N {
        return Err(PyValueError::new_err(format!(
            "no interior optimum at t = {t}: the ceiling is monotone over [{SCAN_LO:e}, \
             {SCAN_HI:e}], so Bob's conclusive rate and Eve's unambiguous-discrimination share \
             never cross; at t = 1 the beam-splitting attack takes nothing"
        )));
    }

    let width = (hi - lo) / (SCAN_N as f64);
    let peak = golden(step(best.0) - width, step(best.0) + width, 80, |x| {
        ceiling(x.exp(), t, eta, visibility, dark)
    });
    let mu = peak.exp();

    Ok((mu, (-2.0 * mu).exp(), ceiling(mu, t, eta, visibility, dark)))
}

fn check_strong(kappa: f64, mu: f64, t: f64, eta: f64, noise: f64, width: f64) -> PyResult<()> {
    check_pos("kappa", kappa)?;
    check_pos("mu", mu)?;
    check_unit("t", t)?;
    check_unit("eta", eta)?;
    check_prob("noise", noise)?;
    check_pos("width", width)?;
    if mu <= kappa {
        return Err(PyValueError::new_err(format!(
            "the strong reference carries mu = {mu} against a signal of kappa = {kappa}, so BS1's \
             reflectivity R = kappa/mu is not below 1: pass mu >> kappa"
        )));
    }

    Ok(())
}

/// D1's window `(nu_i, nu_f, share)`: acceptance `[nu_i, nu_f - 1]`, total-photon range `lambda =
/// [nu_i, nu_f]`, and the reference's Poisson weight inside the acceptance range. Tamaki,
/// Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302 (2009), quant-ph/0607082, Eqs.
/// (1), (2) and (13); `width` is their `a` in units of `sqrt(t*eta*mu)`, endpoints ROUNDED.
#[pyfunction]
pub(crate) fn b92_window(
    kappa: f64,
    mu: f64,
    t: f64,
    eta: f64,
    width: f64,
) -> PyResult<(u64, u64, f64)> {
    check_strong(kappa, mu, t, eta, 0.0, width)?;

    srp_span(kappa, mu, t, eta, width)
}

/// `(fil_all, fil_win, vac, bit_win, bit_all)` per emitted pulse, Eq. (17) of Phys. Rev. A 80,
/// 032302 (2009) at `eta -> t*eta`: all conclusive events, those with D1 in the window, those with
/// D2 and D3 silent and D1 in the window, and the errors among each. Eq. (15) reads the
/// un-windowed `bit_all`; do not substitute `bit_win`. `noise` is Eq. (17)'s `p`, NOT a per-gate
/// dark count: a single photon in place of the signal, clicking at D2 or D3 at random.
#[pyfunction]
pub(crate) fn b92_monitor(
    kappa: f64,
    mu: f64,
    t: f64,
    eta: f64,
    noise: f64,
    width: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_strong(kappa, mu, t, eta, noise, width)?;
    let (_, _, share) = srp_span(kappa, mu, t, eta, width)?;
    let w = srp_watch(kappa, t, eta, noise, share);

    Ok((w[0], w[1], w[2], w[3], w[4]))
}

/// Strong-reference phase error `Lambda_ph,lambda'/Lambda_fil,lambda'`, DERIVED, clamped into
/// `[0, 1/2]`: Phys. Rev. A 80, 032302 (2009), Eq. (15), the largest `Lambda_ph,lambda` with
/// `Lambda_fil,lambda' - 2 Lambda_bit,all <= 2 G_nuf alpha_nuf beta_nuf g(C'_+ Z_U - C'_- Z_L)`,
/// `g((a,b,c,d)) = sqrt(a*d) + sqrt(b*c)`, `Z_U`, `Z_L` the entrywise bounds of Eqs. (C2)-(C8),
/// `C'` the entrywise maximum of Eq. (B11); `Lambda_ph,lambda' <= Lambda_ph,lambda` makes it a
/// bound on the windowed rate. `1/2` when no value satisfies Eq. (15).
#[pyfunction]
pub(crate) fn b92_derived(
    kappa: f64,
    mu: f64,
    t: f64,
    eta: f64,
    noise: f64,
    width: f64,
) -> PyResult<f64> {
    check_strong(kappa, mu, t, eta, noise, width)?;

    Ok(srp_key(kappa, mu, t, eta, noise, width, 1.0)?.2)
}

/// `(p_fil, e_bit, e_ph, rate)` bits per emitted pulse with the phase error DERIVED: Tamaki,
/// Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302 (2009), quant-ph/0607082, Eqs.
/// (17), (15) and (16), `f_ec = 1` recovering their Shannon-limit charge; `p_fil` is
/// `Lambda_fil,lambda'`, already the protocol's sifting. LINEAR in transmission. At their hardware
/// (`noise = 1.7e-6`, `eta = 0.045`, `kappa = 10^-0.92`, `width = 3.2`, 0.21 dB/km) the rate
/// crosses zero at 55, 100 and 122 km for `mu = 1e5`, `10^6.59`, `1e10`, and no `mu` carries it
/// past about 124 km.
#[pyfunction]
pub(crate) fn b92_strong(
    kappa: f64,
    mu: f64,
    t: f64,
    eta: f64,
    noise: f64,
    width: f64,
    f_ec: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_strong(kappa, mu, t, eta, noise, width)?;
    check_fec("f_ec", f_ec)?;

    srp_key(kappa, mu, t, eta, noise, width, f_ec)
}

/// Azuma slack `(code, test)` by which a monitored rate may sit from the running sum of its
/// conditional probabilities, at `eps` PER RATE over `n_total` pairs labelled test with
/// probability `test`: Tamaki, Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302
/// (2009), quant-ph/0607082, Eq. (A3) at bounded difference 1, rescaled by Eqs. (A4) and (A5) to
/// `2*exp(-N(1 - t)^2 e^2/2)` and `2*exp(-N t^2 e^2/2)`, giving `sqrt(2 ln(2/eps)/N)` over `1 - t`
/// and over `t`; Eq. (C7) is the second applied to Alice's x-basis weight. NOT COMPOSED and NOT a
/// licence to widen the observed rates and re-run `b92_strong`: Appendix C relaxes the ASSEMBLED
/// Eq. (C1), never its inputs.
#[pyfunction]
pub(crate) fn b92_azuma(n_total: f64, test: f64, eps: f64) -> PyResult<(f64, f64)> {
    check_pos("n_total", n_total)?;
    check_eps("test", test)?;
    check_eps("eps", eps)?;
    let root = (2.0 * (2.0 / eps).ln() / n_total).sqrt();

    Ok((root / (1.0 - test), root / test))
}

/// REFUSED on both branches, differently per `reference`; `n_total` is checked first. The
/// EXTENDED B92 the plain refusal names, Lucamarini, Di Giuseppe & Tamaki, Phys. Rev. A 80, 032327
/// (2009), is arXiv:0907.0493.
#[pyfunction]
pub(crate) fn b92_finite(n_total: f64, reference: bool) -> PyResult<f64> {
    check_pos("n_total", n_total)?;

    if reference {
        return Err(PyNotImplementedError::new_err(
            "finite-key strong-reference B92 is refused on BOTH receivers. On \
             b92_point(reference=true) -- Koashi, Phys. Rev. Lett. 93, 120501 (2004) -- the \
             question is malformed rather than unimplemented: e_ph is SUPPLIED, so the dominant \
             failure term sits outside the epsilon, the composition cow_rate refuses. On \
             b92_strong -- Tamaki, Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302 \
             (2009), quant-ph/0607082 -- four pieces of a key LENGTH are missing. (1) The \
             finite-N form of Eq. (C1) is 'a slightly relaxed version' never written down; \
             b92_azuma is its Azuma width, not it. (2) Eq. (16) is an asymptotic RATE cited to \
             Shor & Preskill, Phys. Rev. Lett. 85, 441 (2000); nothing turns a bounded \
             phase-error count into a smooth min-entropy. (3) No correctness term and no \
             composition of the five failure probabilities into one eps_sec. (4) Eq. (16) never \
             pays for the test fraction t every deviation bound reads. Use b92_strong and call \
             it asymptotic",
        ));
    }

    Err(PyNotImplementedError::new_err(
        "finite-key plain B92 is refused: no published analysis covers the protocol as shipped, \
         and THREE cover a plain B92 that is not this one. Tamaki & Lutkenhaus, Phys. Rev. A 69, \
         032316 (2004) is asymptotic; a finite-key form of its IMPLICIT phase-error bound needs a \
         joint confidence region on the conclusive, error and no-decision counts through the \
         maximisation over the lost-round split, then a smooth-min-entropy accounting. Sasaki, \
         Matsumoto & Uyematsu, Proc. 2015 IEEE ISIT, pp. 696-699, arXiv:1504.05628: COLLECTIVE \
         attacks on a qubit POVM whose Eqs. (4)-(7) SUM TO THE IDENTITY, no loss. Mafu, Garapo & \
         Petruccione, Phys. Rev. A 88, 062306 (2013): a PREPROCESSING STEP the shipped protocol \
         lacks; NO arXiv version, abstract only read. Bunandar, Govia, Krovi & Englund, npj \
         Quantum Information 6, 104 (2020), arXiv:1911.07860: a depolarising channel, no loss. \
         Amer & Krawec, arXiv:2001.05940, and Krawec, arXiv:2510.11488, prove the EXTENDED B92 of \
         Lucamarini, Di Giuseppe & Tamaki, Phys. Rev. A 80, 032327 (2009), with two test states \
         neither variant here has. Use b92_plain and call it asymptotic",
    ))
}
