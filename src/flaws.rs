use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::std::{
    check_eps, check_err, check_fec, check_finite, check_gate, check_nonneg, check_pos, check_unit, h2, sqrt0, wrap,
    E0, E_CAP,
};

// Loss-tolerant QKD with state-preparation flaws: Tamaki, Curty, Kato, Lo & Azuma, "Loss-tolerant
// quantum cryptography with imperfect sources", Phys. Rev. A 90, 052314 (2014), arXiv:1312.3514,
// with the device model and the four-state form of Pereira, Curty & Tamaki, "Quantum key
// distribution with flawed and leaky sources", npj Quantum Information 5, 62 (2019),
// arXiv:1902.02126, doi 10.1038/s41534-019-0180-9. Alice's modulator applies `phi + delta*phi/pi`
// where she intends `phi`, Pereira Eq. (2); her Eq. (3) prints three states, the fourth is ours.
// Tamaki App. C's `+sin(delta/2)` against Pereira Eq. (3)'s `-sin(delta/2)` is the Bloch x sign only.
//
// `flaw_tolerant` reads e_phase exactly, `flaw_standard` bounds it via a quantum coin: different
// quantities, never subtract.
//
// ASSUMES a single-photon qubit source: no side channel, Trojan-horse leakage or mode dependency.
//
// `tilt` HAS NO DEFAULT: at `tilt = 0` Bob's modulator cancels the flaw and `flaw_phase` reads
// flaw-free. Both papers write at `tilt = -delta`: dark-free e_phase `sin^2(delta/2)`, Pereira (E2).
//
// UNRECONCILED: Tamaki App. C's `delta -> 3*delta/2` lands at `sin^2(3*delta/8)`, Pereira Eq. (E2)
// at `sin^2(delta/2)`; `test_tamaki_ratio` pins the 4/3.
//
// Does NOT compose with `qkd.attacks`: both analyses need a basis-independent M_f, which efficiency
// mismatch breaks. Mismatch composition, arXiv:2412.09684 (2024), is not implemented.
//
// Amplitude flaws leave the X-Z great circle (`basis3` has no `r_y` column) and fail SILENTLY: the
// inversion can read LOW, the insecure direction.
//
// Finite key: Mizutani, Curty, Lim, Imoto & Tamaki, "Finite-key security analysis of quantum key
// distribution with imperfect light sources", New J. Phys. 17, 093011 (2015), arXiv:1504.08151.
// Sec. V's `dtheta_A = xi*theta_A/pi`, `dtheta_B = -dtheta_A` are `delta` and `tilt = -delta`.
// Eq. (43) length, Eq. (36) phase-error COUNT, Eq. (37) `basis3`'s inverse, Lemma 4 `flaw_azuma`.
// No random-sampling correction: the inversion reads the key's own detections, no test subsample.
//
// Not implemented: the Hoeffding intervals of Xu, Wei, Sajeed, Kaiser, Sun, Tang, Qian, Makarov &
// Lo, "Experimental quantum key distribution with source flaws", Phys. Rev. A 92, 032305 (2015),
// arXiv:1408.3667, Eq. (8), collective attacks only; and Curras-Lorenzo, Navarrete, Pereira &
// Tamaki, "Finite-key analysis of loss-tolerant quantum key distribution based on random sampling
// theory", Phys. Rev. A 104, 012406 (2021), arXiv:2101.12603, which replaces Azuma, not tightens it.
//
// Mizutani Sec. IV's decoy layer is NOT implemented: single photon, so `m0 = 0` and each `Decoy_1`
// is its observed count. No Figs. 2-7 curve is reproduced.

/// Alice's modulation flaw in radians, `[0, pi)`: at `pi` the three-state matrix is singular.
fn check_flaw(x: f64) -> PyResult<()> {
    if x.is_finite() && (0.0..std::f64::consts::PI).contains(&x) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "delta must be in [0, pi), got {x}: a modulation deviation in radians; \
             at pi Alice's two Z-basis states coincide"
        )))
    }
}

/// Alice's four Bloch polar angles, measured from +z in the X-Z plane.
fn angles(delta: f64) -> [f64; 4] {
    let pi = std::f64::consts::PI;

    [
        0.0,
        pi + delta,
        pi / 2.0 + delta / 2.0,
        3.0 * pi / 2.0 + 3.0 * delta / 2.0,
    ]
}

/// The two virtual X-basis states as `(prior, angle)`; see `flaw_virtual`.
fn virtuals(delta: f64) -> [(f64, f64); 2] {
    let pi = std::f64::consts::PI;
    let s = (delta / 2.0).sin();

    [
        ((1.0 - s) / 2.0, pi / 2.0 + delta / 2.0),
        ((1.0 + s) / 2.0, 3.0 * pi / 2.0 + delta / 2.0),
    ]
}

/// Bob's `(Z, X)` projector angles; his X carries his own modulator's `delta/2`. `tilt` is the
/// rigid frame misalignment, not a flaw of either modulator.
fn projectors(delta: f64, tilt: f64) -> [f64; 2] {
    [tilt, tilt + std::f64::consts::PI / 2.0 + delta / 2.0]
}

/// `(p_0, p_1)`: probability Bob's detector 0 or 1 fires and he reads that bit, single photon at
/// Bloch angle `sig` against a projector at `bob`. Pereira arXiv:1902.02126 Appendix E, Eq. (E2);
/// double clicks resolved by a coin. The pair sums to `2*(1 - eta/2)*dark + eta/2` at any angle:
/// the basis-independent efficiency both analyses assume.
fn readout(sig: f64, bob: f64, eta: f64, dark: f64) -> (f64, f64) {
    let p0 = (1.0 + (sig - bob).cos()) / 2.0;
    let p1 = 1.0 - p0;
    let floor = (1.0 - eta / 2.0) * dark;
    let seen = eta / 2.0 * (1.0 - dark / 2.0);
    let both = eta / 4.0 * dark;

    (floor + seen * p0 + both * p1, floor + seen * p1 + both * p0)
}

/// 3x3 Gaussian elimination with partial pivoting; `None` when the Bloch vectors are collinear.
fn solve3(m: [[f64; 3]; 3], y: [f64; 3]) -> Option<[f64; 3]> {
    let mut a = [
        [m[0][0], m[0][1], m[0][2], y[0]],
        [m[1][0], m[1][1], m[1][2], y[1]],
        [m[2][0], m[2][1], m[2][2], y[2]],
    ];

    for c in 0..3 {
        let mut best = c;
        for r in (c + 1)..3 {
            if a[r][c].abs() > a[best][c].abs() {
                best = r;
            }
        }
        a.swap(c, best);

        let pivot = a[c][c];
        if !(pivot.abs() > 0.0) {
            return None;
        }

        for r in 0..3 {
            if r == c {
                continue;
            }

            let f = a[r][c] / pivot;
            for k in c..4 {
                a[r][k] -= f * a[c][k];
            }
        }
    }

    Some([a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]])
}

/// Rows `(1, sin theta, cos theta)` for 0Z, 1Z and 0X.
fn basis3(delta: f64) -> [[f64; 3]; 3] {
    let a = angles(delta);
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        *row = [1.0, a[i].sin(), a[i].cos()];
    }

    m
}

/// Bit error rate of a two-state `(prior, angle)` ensemble read at `bob`.
fn error_rate(pairs: [(f64, f64); 2], bob: f64, eta: f64, dark: f64) -> f64 {
    let mut wrong = 0.0;
    let mut total = 0.0;
    for (j, (prior, sig)) in pairs.iter().enumerate() {
        let (p0, p1) = readout(*sig, bob, eta, dark);
        wrong += prior * if j == 0 { p1 } else { p0 };
        total += prior * (p0 + p1);
    }

    if !(total > 0.0) {
        return E0;
    }

    (wrong / total).min(E_CAP)
}

/// Alice's four Bloch polar angles in radians, `(0Z, 1Z, 0X, 1X)`, from +z in the X-Z plane. Not
/// a rectangle. Pereira, Curty & Tamaki, arXiv:1902.02126 Eqs. (2) and (3), plus the fourth state.
#[pyfunction]
pub(crate) fn flaw_angles(delta: f64) -> PyResult<(f64, f64, f64, f64)> {
    check_flaw(delta)?;

    let a = angles(delta);

    Ok((a[0], a[1], a[2], a[3]))
}

/// The two virtual X-basis states as `(p_0, theta_0, p_1, theta_1)`: never sent; the phase error
/// rate is their bit error rate. Antipodal, priors `[1 -/+ sin(delta/2)]/2`. Tamaki Appendix C
/// places the pair at `delta/4`: the header's 4/3.
#[pyfunction]
pub(crate) fn flaw_virtual(delta: f64) -> PyResult<(f64, f64, f64, f64)> {
    check_flaw(delta)?;

    let v = virtuals(delta);

    Ok((v[0].0, v[0].1, v[1].0, v[1].1))
}

/// Tamaki's triangle condition as a determinant, `2*cos(delta/2) + sin(delta)`: 2.0 ideal,
/// `3*sqrt(3)/2` at `pi/3`, `-> 0` as `delta -> pi`. A diagnostic, not a rate.
#[pyfunction]
pub(crate) fn flaw_triangle(delta: f64) -> PyResult<f64> {
    check_flaw(delta)?;

    let m = basis3(delta);
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    Ok(det.abs())
}

/// Fidelity `Tr|sqrt(rho_Z) sqrt(rho_X)|` of Alice's two basis states, single photon:
/// `sqrt(1 - [sin^2(delta/2) + sin^3(delta/2)]/2)`. The standard analysis's only flaw input.
#[pyfunction]
pub(crate) fn flaw_fidelity(delta: f64) -> PyResult<f64> {
    check_flaw(delta)?;

    let a = angles(delta);
    let mean = |i: usize, j: usize| {
        (
            (a[i].sin() + a[j].sin()) / 2.0,
            (a[i].cos() + a[j].cos()) / 2.0,
        )
    };
    let (zx, zz) = mean(0, 1);
    let (xx, xz) = mean(2, 3);

    let dot = zx * xx + zz * xz;
    let nz = zx * zx + zz * zz;
    let nx = xx * xx + xz * xz;
    let sq = (1.0 + dot + sqrt0((1.0 - nz) * (1.0 - nx))) / 2.0;

    Ok(sqrt0(sq).min(1.0))
}

/// `(y1, e_bit)`: single-photon detection probability and Z-basis bit error rate, overall
/// transmittance `eta`, per-gate `dark`, Bob's frame at `tilt` radians. Pereira arXiv:1902.02126
/// Appendix E: `y1` is Sec. IV A's `4*(1 - eta/2)*dark + eta` without `P_ZA P_ZB` (taken as
/// `q_sift`), `e_bit` her Eq. (33); both reproduced at `tilt = -delta` only. `e_bit` capped at 1/2
/// (her form passes it above `delta ~ 1.75`).
///
/// `y1` is independent of `delta` and `tilt`. It EXCEEDS 1 at `eta = 1`: Pereira's first-order sum
/// reads `1 + 2*dark` where Tamaki App. C reads `1 + 2.5*dark - dark^2`. Quoted, not repaired.
#[pyfunction]
pub(crate) fn flaw_channel(delta: f64, eta: f64, dark: f64, tilt: f64) -> PyResult<(f64, f64)> {
    check_flaw(delta)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_finite("tilt", tilt)?;

    let a = angles(delta);
    let bob = projectors(delta, wrap(tilt));
    let pairs = [(0.5, a[0]), (0.5, a[1])];

    let (u0, u1) = readout(a[0], bob[0], eta, dark);
    let (v0, v1) = readout(a[1], bob[0], eta, dark);

    Ok((u0 + u1 + v0 + v1, error_rate(pairs, bob[0], eta, dark)))
}

/// Phase error rate under the LOSS-TOLERANT analysis, Tamaki, Curty, Kato, Lo & Azuma, Phys.
/// Rev. A 90, 052314 (2014), Eqs. (1)-(4). EXACT, not a bound; inverts the rejected X-basis data,
/// no loss term. All flaw dependence is through `tilt`. Capped at 1/2; collinear states refused.
#[pyfunction]
pub(crate) fn flaw_phase(delta: f64, eta: f64, dark: f64, tilt: f64) -> PyResult<f64> {
    check_flaw(delta)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_finite("tilt", tilt)?;

    let a = angles(delta);
    let bob = projectors(delta, wrap(tilt));
    let seen: Vec<(f64, f64)> = (0..3).map(|i| readout(a[i], bob[1], eta, dark)).collect();
    let m = basis3(delta);

    let mut rates = [[0.0f64; 3]; 2];
    for s in 0..2 {
        let col = [
            if s == 0 { seen[0].0 } else { seen[0].1 },
            if s == 0 { seen[1].0 } else { seen[1].1 },
            if s == 0 { seen[2].0 } else { seen[2].1 },
        ];
        match solve3(m, col) {
            Some(q) => rates[s] = q,
            None => return Err(collinear(delta)),
        }
    }

    let mut wrong = 0.0;
    let mut total = 0.0;
    for (j, (prior, sig)) in virtuals(delta).iter().enumerate() {
        let v = [1.0, sig.sin(), sig.cos()];
        let read = |s: usize| v[0] * rates[s][0] + v[1] * rates[s][1] + v[2] * rates[s][2];
        wrong += prior * if j == 0 { read(1) } else { read(0) };
        total += prior * (read(0) + read(1));
    }

    if !(total > 0.0) {
        return Ok(E_CAP);
    }

    Ok((wrong / total).min(E_CAP))
}

/// `flaw_phase` by measuring the virtual states directly. Not a protocol step: the arbiter.
#[pyfunction]
pub(crate) fn flaw_direct(delta: f64, eta: f64, dark: f64, tilt: f64) -> PyResult<f64> {
    check_flaw(delta)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_finite("tilt", tilt)?;

    let bob = projectors(delta, wrap(tilt));

    Ok(error_rate(virtuals(delta), bob[1], eta, dark))
}

/// Quantum coin imbalance under the STANDARD analysis, `(1 - F)/(2*y1)` capped at 1/2, `F` from
/// `flaw_fidelity`. Gottesman, Lo, Lutkenhaus & Preskill, QIC 5, 325 (2004),
/// arXiv:quant-ph/0212066, through Lo & Preskill, QIC 7, 431 (2007), arXiv:quant-ph/0610203; the
/// form used here is Tamaki, Lo, Fung & Qi, "Phase encoding schemes for measurement device
/// independent quantum key distribution and basis-dependent flaw", Phys. Rev. A 85, 042307 (2012),
/// arXiv:1111.3413, Eq. (7), with the one-fidelity `Delta_ini` (their Eq. (9) is two-fidelity).
/// Eve amplifies the imbalance by the loss: the loss-tolerant analysis never divides by `y1`.
#[pyfunction]
pub(crate) fn flaw_coin(delta: f64, y1: f64) -> PyResult<f64> {
    check_flaw(delta)?;

    if !(y1.is_finite() && y1 > 0.0) {
        return Err(PyValueError::new_err(format!(
            "y1 must be > 0, got {y1}: the single-photon detection probability \
             the coin imbalance divides by"
        )));
    }

    let ini = (1.0 - flaw_fidelity(delta)?) / 2.0;

    Ok((ini / y1).min(E_CAP))
}

/// Phase error rate under the STANDARD analysis:
/// `e_bit + 4d(1-d)(1 - 2 e_bit) + 4(1 - 2d) sqrt(d(1-d) e_bit(1 - e_bit))`, `d = coin`. Lo &
/// Preskill, QIC 7, 431 (2007), as printed by Pereira, Curty & Tamaki, arXiv:1902.02126 Eq. (31)
/// and (D6). An UPPER BOUND on what `flaw_phase` computes exactly. Capped at 1/2; `coin = 0`
/// returns `e_bit` bit for bit, where the two analyses coincide.
#[pyfunction]
pub(crate) fn flaw_gllp(e_bit: f64, coin: f64) -> PyResult<f64> {
    if !(e_bit.is_finite() && (0.0..=E_CAP).contains(&e_bit)) {
        return Err(PyValueError::new_err(format!(
            "e_bit must be in [0, 1/2], got {e_bit}"
        )));
    }

    if !(coin.is_finite() && (0.0..=E_CAP).contains(&coin)) {
        return Err(PyValueError::new_err(format!(
            "coin must be in [0, 1/2], got {coin}: a quantum coin imbalance"
        )));
    }

    if coin <= 0.0 {
        return Ok(e_bit);
    }

    let mix = coin * (1.0 - coin);
    let bound = e_bit
        + 4.0 * mix * (1.0 - 2.0 * e_bit)
        + 4.0 * (1.0 - 2.0 * coin) * sqrt0(mix * e_bit * (1.0 - e_bit));

    Ok(bound.min(E_CAP))
}

/// `(rate, y1, e_bit, e_phase)`: asymptotic key rate under the LOSS-TOLERANT analysis, bits per
/// emitted single photon, `q_sift * y1 * [1 - h2(e_phase) - f_ec*h2(e_bit)]`, Pereira
/// arXiv:1902.02126 Eq. (4) with her `P_ZA P_ZB` as `q_sift`. Slot 4 is a rate, `flaw_standard`'s
/// a coin: never line the tuples up. Clamped at zero.
#[pyfunction]
pub(crate) fn flaw_tolerant(
    delta: f64,
    eta: f64,
    dark: f64,
    tilt: f64,
    f_ec: f64,
    q_sift: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_fec("f_ec", f_ec)?;
    check_unit("q_sift", q_sift)?;

    let (y1, e_bit) = flaw_channel(delta, eta, dark, tilt)?;
    let e_phase = flaw_phase(delta, eta, dark, tilt)?;
    let rate = q_sift * y1 * (1.0 - h2(e_phase) - f_ec * h2(e_bit));

    Ok((rate.max(0.0), y1, e_bit, e_phase))
}

/// `(rate, y1, e_bit, coin)`: asymptotic key rate under the STANDARD analysis, bits per emitted
/// single photon, `q_sift * y1 * [1 - h2(flaw_gllp(e_bit, coin)) - f_ec*h2(e_bit)]`. Slot 4 is a
/// coin, NOT a phase error: `flaw_gllp` converts it. Clamped at zero.
#[pyfunction]
pub(crate) fn flaw_standard(
    delta: f64,
    eta: f64,
    dark: f64,
    tilt: f64,
    f_ec: f64,
    q_sift: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    check_fec("f_ec", f_ec)?;
    check_unit("q_sift", q_sift)?;

    let (y1, e_bit) = flaw_channel(delta, eta, dark, tilt)?;
    let coin = flaw_coin(delta, y1)?;
    let bound = flaw_gllp(e_bit, coin)?;
    let rate = q_sift * y1 * (1.0 - h2(bound) - f_ec * h2(e_bit));

    Ok((rate.max(0.0), y1, e_bit, coin))
}

/// Azuma applications in Mizutani Eq. (36): (three counts + trailing Delta) per virtual state.
const AZUMA_TERMS: f64 = 8.0;

/// `(a, b)` with `|phi> = a|0z> + b|1z>` at Bloch polar angle `theta`, Mizutani Eq. (34) at `i = 1`.
fn amps(theta: f64) -> (f64, f64) {
    ((theta / 2.0).cos(), (theta / 2.0).sin())
}

/// `(c, x, z)`: `|phi^0z><phi^1z| + h.c.` in `sigma_I, sigma_X, sigma_Z`, Mizutani's `C_{t,l}`.
/// `c` is the overlap of his Eqs. (36) and (82); `(1 +/- c)/2` are `virtuals`'s priors.
fn cross(delta: f64) -> (f64, f64, f64) {
    let a = angles(delta);
    let (a0, b0) = amps(a[0]);
    let (a1, b1) = amps(a[1]);

    (a0 * a1 + b0 * b1, a0 * b1 + b0 * a1, a0 * a1 - b0 * b1)
}

/// `A^-1` of Mizutani Eq. (37); see `flaw_inverse`. `A` is `basis3` halved: twice the plain
/// inverse. `None` when collinear.
fn inverse(delta: f64) -> Option<[f64; 9]> {
    let a = angles(delta);
    let rx = [a[0].sin(), a[1].sin(), a[2].sin()];
    let rz = [a[0].cos(), a[1].cos(), a[2].cos()];
    let det = rx[1] * (rz[2] - rz[0]) + rx[2] * (rz[0] - rz[1]) + rx[0] * (rz[1] - rz[2]);

    if !(det.abs() > 0.0) {
        return None;
    }

    let k = 2.0 / det;

    Some([
        k * (rx[1] * rz[2] - rx[2] * rz[1]),
        k * (rx[2] * rz[0] - rx[0] * rz[2]),
        k * (rx[0] * rz[1] - rx[1] * rz[0]),
        k * (rz[1] - rz[2]),
        k * (rz[2] - rz[0]),
        k * (rz[0] - rz[1]),
        k * (rx[2] - rx[1]),
        k * (rx[0] - rx[2]),
        k * (rx[1] - rx[0]),
    ])
}

/// Shared by the asymptotic and finite estimators.
fn collinear(delta: f64) -> PyErr {
    PyValueError::new_err(format!(
        "the three Bloch vectors at delta = {delta} are collinear, so the \
         transmission rates of Id, sigma_x and sigma_z are not determined \
         and the loss-tolerant phase error does not exist"
    ))
}

/// Azuma deviation `g_A(n, eps) = sqrt(2 n ln(1/eps))` in COUNTS, Mizutani, Curty, Lim, Imoto &
/// Tamaki, New J. Phys. 17, 093011 (2015), arXiv:1504.08151, Lemma 4 and Eq. (50); `eps`
/// one-sided. A martingale bound, so it holds against coherent attacks where no Chernoff or
/// Hoeffding interval does; scales as `sqrt(n)`, not `sqrt(n p)`.
#[pyfunction]
pub(crate) fn flaw_azuma(n: f64, eps: f64) -> PyResult<f64> {
    check_nonneg("n", n)?;
    check_eps("eps", eps)?;

    Ok(sqrt0(2.0 * n * (1.0 / eps).ln()))
}

/// `eps_ph / 8`: the per-bound failure probability `flaw_errors` runs each Azuma application at.
#[pyfunction]
pub(crate) fn flaw_eps(eps_ph: f64) -> PyResult<f64> {
    check_eps("eps_ph", eps_ph)?;

    Ok(eps_ph / AZUMA_TERMS)
}

/// Mizutani Eq. (37)'s `A^-1`, row-major: rows `(sigma_I, sigma_X, sigma_Z)`, columns
/// `(0Z, 1Z, 0X)`; maps state transmission rates to Pauli transmission rates.
///
/// THE (2, 2) ENTRY IS NOT WHAT EQ. (37) PRINTS: e-print and journal both repeat
/// `2(r_x^0z - r_x^0x)`; the cofactor is `r_x^1z - r_x^0z`. `test_printed_entry` pins it.
#[pyfunction]
pub(crate) fn flaw_inverse(delta: f64) -> PyResult<Vec<f64>> {
    check_flaw(delta)?;

    match inverse(delta) {
        Some(m) => Ok(m.to_vec()),
        None => Err(collinear(delta)),
    }
}

/// `N_ph`: upper bound on the phase-error COUNT among single-photon Z-basis detections, Mizutani
/// Eq. (36); divide by `m1` for a rate. `counts` is `Lambda_{Omega,s}` ordered `(3, 0), (4, 0),
/// (5, 0), (3, 1), (4, 1), (5, 1)`, `Omega = 3, 4, 5` for 0Z, 1Z, 0X, `s` Bob's X outcome; `pz` the
/// shared Z-basis probability; `n1` the single-photon detection count, his Eq. (84).
///
/// Eq. (36) is LINEAR, so the worst case over Eqs. (97)-(99)'s box is centre + `|coef| * width`.
/// At zero width it is `m1 * flaw_phase` exactly (`test_region_collapse`).
#[pyfunction]
pub(crate) fn flaw_errors(delta: f64, pz: f64, counts: Vec<f64>, n1: f64, eps_ph: f64) -> PyResult<f64> {
    check_flaw(delta)?;
    check_eps("pz", pz)?;
    check_pos("n1", n1)?;
    check_eps("eps_ph", eps_ph)?;

    if counts.len() != 6 {
        return Err(PyValueError::new_err(format!(
            "counts must hold 6 single-photon detection counts, got {}: Mizutani's \
             Lambda_(Omega,s) in the order (3,0), (4,0), (5,0), (3,1), (4,1), (5,1)",
            counts.len()
        )));
    }

    for c in counts.iter() {
        check_nonneg("counts", *c)?;
    }

    let inv = match inverse(delta) {
        Some(m) => m,
        None => return Err(collinear(delta)),
    };

    let (ci, cx, cz) = cross(delta);
    let coef = [
        ci * inv[0] + cx * inv[3] + cz * inv[6],
        ci * inv[1] + cx * inv[4] + cz * inv[7],
        ci * inv[2] + cx * inv[5] + cz * inv[8],
    ];

    let px = 1.0 - pz;
    let share = [pz * px / 2.0, pz * px / 2.0, px * px];
    let weight = pz * pz / 4.0;
    let width = flaw_azuma(n1, eps_ph / AZUMA_TERMS)?;

    let mut total = 0.0;
    for s in 0..2 {
        let sign = if s == 0 { 1.0 } else { -1.0 };
        let k = [
            weight * (1.0 + sign * coef[0]),
            weight * (1.0 + sign * coef[1]),
            weight * sign * coef[2],
        ];
        // Eq. (100): virtual state `s` reads Bob's OPPOSITE outcome, the counts included.
        let seen = (s ^ 1) * 3;
        for l in 0..3 {
            total += (k[l] * counts[seen + l] + k[l].abs() * width) / share[l];
        }
        total += width;
    }

    Ok(total.max(0.0))
}

/// Finite key LENGTH in bits, Mizutani Eq. (43):
/// `l = floor( m0 + m1*[1 - h2(e_ph)] - log2(2/(eps_s^2 - eps_est)) - leak_ec - log2(2/eps_c) )`.
/// `m0`, `m1` vacuum and single-photon sifted counts, `e_ph` = `flaw_errors / m1`.
///
/// `eps_est` IS EQ. (43)'s `eta`, a FAILURE PROBABILITY, not a transmittance; `eps_s^2 <= eps_est`
/// is refused, not clamped. `m0` at full weight. Clamped at zero and floored.
#[pyfunction]
pub(crate) fn flaw_length(
    m0: f64,
    m1: f64,
    e_ph: f64,
    leak_ec: f64,
    eps_s: f64,
    eps_c: f64,
    eps_est: f64,
) -> PyResult<f64> {
    check_nonneg("m0", m0)?;
    check_nonneg("m1", m1)?;
    check_err("e_ph", e_ph)?;
    check_nonneg("leak_ec", leak_ec)?;
    check_eps("eps_s", eps_s)?;
    check_eps("eps_c", eps_c)?;
    check_eps("eps_est", eps_est)?;

    if eps_est >= eps_s * eps_s {
        return Err(PyValueError::new_err(format!(
            "eps_est must be below eps_s^2 (got {eps_est} against {}): Mizutani \
             arXiv:1504.08151 Eq. (43) charges log2(2/(eps_s^2 - eps_est)); lower \
             eps_est or raise eps_s",
            eps_s * eps_s
        )));
    }

    let budget = (2.0 / (eps_s * eps_s - eps_est)).log2() + (2.0 / eps_c).log2();
    let len = m0 + m1 * (1.0 - h2(e_ph)) - leak_ec - budget;

    Ok(len.max(0.0).floor())
}

/// `(length, n1, m1, e_bit, e_phase)`: finite key length in BITS over `n_emit` emitted single
/// photons, Mizutani Eq. (43) over Eq. (36) on `flaw_channel`, sifting `pz^2`. Slot 1 is a length
/// over the block, not a rate; `length / n_emit` approaches `flaw_tolerant` at `q_sift = pz^2` from
/// below. `eps_ph` must sit below `eps_s^2`. `m0 = 0` is the single-photon source, exact.
#[pyfunction]
pub(crate) fn flaw_finite(
    delta: f64,
    eta: f64,
    dark: f64,
    tilt: f64,
    pz: f64,
    n_emit: f64,
    f_ec: f64,
    eps_s: f64,
    eps_c: f64,
    eps_ph: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_eps("pz", pz)?;
    check_pos("n_emit", n_emit)?;
    check_fec("f_ec", f_ec)?;

    let (y1, e_bit) = flaw_channel(delta, eta, dark, tilt)?;
    let n1 = n_emit * y1;
    let m1 = n1 * pz * pz;

    if !(m1 > 0.0) {
        return Err(PyValueError::new_err(
            "the block detects nothing (m1 = 0): raise n_emit, the transmittance or pz",
        ));
    }

    let a = angles(delta);
    let bob = projectors(delta, wrap(tilt));
    let px = 1.0 - pz;
    let share = [pz * px / 2.0, pz * px / 2.0, px * px];

    let mut counts = vec![0.0f64; 6];
    for l in 0..3 {
        let (p0, p1) = readout(a[l], bob[1], eta, dark);
        let seen = p0 + p1;
        if !(seen > 0.0) {
            return Err(PyValueError::new_err(
                "Bob's X-basis analyser never fires on Alice's states: no rejected \
                 data to invert",
            ));
        }

        counts[l] = n1 * share[l] * p0 / seen;
        counts[3 + l] = n1 * share[l] * p1 / seen;
    }

    let n_ph = flaw_errors(delta, pz, counts, n1, eps_ph)?;
    let e_phase = (n_ph / m1).min(E_CAP);
    let leak = f_ec * m1 * h2(e_bit);
    let length = flaw_length(0.0, m1, e_phase, leak, eps_s, eps_c, eps_ph)?;

    Ok((length, n1, m1, e_bit, e_phase))
}
