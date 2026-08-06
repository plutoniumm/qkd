use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::std::{
    check_eps, check_err, check_fec, check_gate, check_nonneg, check_pos, check_prob,
    check_unit, h2, E0, E_CAP,
};

// Measurement-device-independent BB84. Everything here is per PULSE PAIR and dimensionless --
// intensities are mean photon numbers, gains are announcement probabilities, error rates are
// fractions.
//
// Forward model: Ma & Razavi, Phys. Rev. A 86, 062319 (2012), arXiv:1204.4856.
// Decoy inversion: Xu, Curty, Qi & Lo, New J. Phys. 15, 113007 (2013), arXiv:1305.6965.
// Protocol: Lo, Curty & Qi, Phys. Rev. Lett. 108, 130503 (2012), arXiv:1109.1473.
//
// `mdi_yield` is a FORWARD model of Y11 and `mdi_y11` a security BOUND on it; composing them the
// wrong way round is the error that makes key rates look great.

/// Interference argument past which the Bessel series is refused. A domain guard, not a
/// numerical limit: the series is safe well beyond it.
const I0_MAX: f64 = 100.0;

/// Relative size at which the series stops; every term is positive, so the last bounds the error.
const I0_TOL: f64 = 1e-18;

/// `I0(2*sqrt(z)) - 1 = sum_{k>=1} z^k/(k!)^2`, the leading 1 never formed: subtracting it
/// leaves four digits at `z = 1e-12`.
fn bessel_excess(z: f64) -> f64 {
    let mut term = 1.0;
    let mut total = 0.0;
    let mut k = 1.0;

    while term > I0_TOL * (total + 1.0) {
        term *= z / (k * k);
        total += term;
        k += 1.0;
    }

    total
}

/// `I0(2x) - 4*I0(x) + 3 = sum_{k>=2} x^(2k) * (1 - 4^(1-k)) / (k!)^2`, taking `x2 = x^2` -- the
/// whole bracket of Eq. (35). Three Bessel evaluations round away the diagonal QBER's 1/4 limit.
fn bessel_dip(x2: f64) -> f64 {
    let mut term = x2 * x2 / 4.0;
    let mut weight = 0.75;
    let mut total = term * weight;
    let mut k = 2.0;

    while term * weight > I0_TOL * (total + 1.0) {
        k += 1.0;
        term *= x2 / (k * k);
        weight = 1.0 - 4.0f64.powf(1.0 - k);
        total += term * weight;
    }

    total
}

/// Domain guard shared by both gains.
fn check_bessel(x: f64) -> PyResult<()> {
    if x.is_finite() && x.abs() <= I0_MAX {
        return Ok(());
    }

    Err(PyValueError::new_err(format!(
        "the interference argument sqrt(eta_a*mu_a*eta_b*mu_b)/2 must be finite and \
         at most {I0_MAX}, got {x}"
    )))
}

/// Single-photon-pair yield and error rate of the relay's Bell measurement: Ma & Razavi
/// Eqs. (21) and (23). A FORWARD model, never a bound -- `mdi_y11` is the bound.
///
/// `dark` is PER DETECTOR PER GATE, roughly half the total background a paper quotes as Y0, the
/// relay reading four gates.
#[pyfunction]
pub(crate) fn mdi_yield(eta_a: f64, eta_b: f64, dark: f64, e_d: f64) -> PyResult<(f64, f64)> {
    check_unit("eta_a", eta_a)?;
    check_unit("eta_b", eta_b)?;
    check_gate("dark", dark)?;
    check_err("e_d", e_d)?;

    let live = (1.0 - dark) * (1.0 - dark);
    let core = eta_a * eta_b / 2.0;
    let one = (2.0 * eta_a + 2.0 * eta_b - 3.0 * eta_a * eta_b) * dark;
    let two = 4.0 * (1.0 - eta_a) * (1.0 - eta_b) * dark * dark;
    let total = core + one + two;
    let y11 = live * total;

    if !(total > 0.0) {
        return Ok((0.0, E_CAP));
    }

    // Misalignment weights the signal term; background terms land at e0 = 1/2.
    Ok((y11, ((e_d * core + E0 * (one + two)) / total).min(E_CAP)))
}

/// `(Q, E)` per PULSE PAIR in the RECTILINEAR (key) basis, Ma & Razavi Eqs. (51)-(54); `dark` per
/// gate. Its polarisations are separated, not interfered: interference enters only via dark counts.
#[pyfunction]
pub(crate) fn mdi_rect(
    mu_a: f64,
    mu_b: f64,
    eta_a: f64,
    eta_b: f64,
    dark: f64,
    e_d: f64,
) -> PyResult<(f64, f64)> {
    let (arrive, x) = arrivals(mu_a, mu_b, eta_a, eta_b)?;
    check_gate("dark", dark)?;
    check_err("e_d", e_d)?;
    check_bessel(x)?;

    let live = 1.0 - dark;
    let bulk = (-arrive / 2.0).exp();
    // 1 - (1 - dark)*exp(-s) through expm1, cancellation-free at dark = 0.
    let hit_a = -(-eta_a * mu_a / 2.0).exp_m1() + dark * (-eta_a * mu_a / 2.0).exp();
    let hit_b = -(-eta_b * mu_b / 2.0).exp_m1() + dark * (-eta_b * mu_b / 2.0).exp();

    // Eq. (52): one photon each side, landing in orthogonal ports.
    let correct = 2.0 * live * live * bulk * hit_a * hit_b;
    // Eq. (53): a dark count completing a pattern, bracket regrouped as `hit_a` is.
    let inner = bessel_excess(x * x) - (-arrive / 2.0).exp_m1() + dark * bulk;
    let wrong = 2.0 * dark * live * live * bulk * inner;
    let gain = correct + wrong;

    if !(gain > 0.0) {
        return Ok((0.0, E_CAP));
    }

    // Eq. (54): the dark branch already fired the WRONG detectors, so it weighs (1 - e_d), not e0.
    Ok((
        gain.min(1.0),
        ((e_d * correct + (1.0 - e_d) * wrong) / gain).min(E_CAP),
    ))
}

/// `(Q, E)` per PULSE PAIR in the DIAGONAL (test) basis, Ma & Razavi Eqs. (35) and (39). `E` tends
/// to `1/4 + e_d/2`, not `e_d`, at weak signal: a ~25% diagonal QBER is the protocol working.
#[pyfunction]
pub(crate) fn mdi_diag(
    mu_a: f64,
    mu_b: f64,
    eta_a: f64,
    eta_b: f64,
    dark: f64,
    e_d: f64,
) -> PyResult<(f64, f64)> {
    let (arrive, x) = arrivals(mu_a, mu_b, eta_a, eta_b)?;
    check_gate("dark", dark)?;
    check_err("e_d", e_d)?;
    check_bessel(x)?;

    let y = (1.0 - dark) * (-arrive / 4.0).exp();
    let yy = y * y;
    // 1 - y, formed the same cancellation-free way as `mdi_rect`'s hit terms.
    let gap = -(-arrive / 4.0).exp_m1() + dark * (-arrive / 4.0).exp();
    let close = bessel_excess(x * x / 4.0);
    let far = bessel_excess(x * x);

    // Eq. (35) regrouped as 2*(1-y)^2 + 4*(1-y)*(I0(x) - 1) + [I0(2x) - 4*I0(x) + 3]: the literal
    // bracket is rounding noise below mu*eta ~ 1e-5.
    let gain = 2.0 * yy * (2.0 * gap * gap + 4.0 * gap * close + bessel_dip(x * x));

    if !(gain > 0.0) {
        return Ok((0.0, E_CAP));
    }

    // Eq. (39): e0 everywhere, less the interference the aligned single-photon pairs recover.
    let errs = E0 * gain - 2.0 * (E0 - e_d) * yy * far;

    Ok((gain.min(1.0), (errs / gain).clamp(0.0, E_CAP)))
}

/// Ma & Razavi Eq. (30): the total mean photon number reaching the relay and the interference
/// argument `x = sqrt(eta_a*mu_a*eta_b*mu_b)/2` both gains are written in.
fn arrivals(mu_a: f64, mu_b: f64, eta_a: f64, eta_b: f64) -> PyResult<(f64, f64)> {
    check_nonneg("mu_a", mu_a)?;
    check_nonneg("mu_b", mu_b)?;
    check_unit("eta_a", eta_a)?;
    check_unit("eta_b", eta_b)?;

    let a = eta_a * mu_a;
    let b = eta_b * mu_b;

    Ok((a + b, (a * b).sqrt() / 2.0))
}

/// Lower bound on the single-photon-pair yield `Y11` from a THREE-INTENSITY grid of observed
/// gains -- Xu, Curty, Qi & Lo, New J. Phys. 15, 113007 (2013), Table 2. A SECURITY BOUND,
/// inferred from observations alone; `mdi_yield` is the forward model it must sit under.
///
/// `a` and `b` are each `[signal, decoy, vacuum]` mean photon numbers, strictly decreasing,
/// for Alice and Bob. `q` is the 3x3 grid of measured gains in the SAME basis, row-major with
/// Alice's intensity as the row: `q[3*i + j]` is the gain at `(a[i], b[j])`.
///
/// Reads seven cells, not `(signal_a, decoy_b)` or `(decoy_a, signal_b)`; `mdi_program` reads nine.
#[pyfunction]
pub(crate) fn mdi_y11(a: Vec<f64>, b: Vec<f64>, q: Vec<f64>) -> PyResult<f64> {
    check_set("a", &a)?;
    check_set("b", &b)?;

    if q.len() != 9 {
        return Err(PyValueError::new_err(format!(
            "q must be the 3x3 gain grid, row-major with Alice's intensity as the row, \
             got {} values",
            q.len()
        )));
    }

    for (k, g) in q.iter().enumerate() {
        check_prob(&format!("q[{k}]"), *g)?;
    }

    // Table 2's case split: eliminate in the party with the shallower intensity ladder.
    let ratio_a = (a[0] + a[2]) / (a[1] + a[2]);
    let ratio_b = (b[0] + b[2]) / (b[1] + b[2]);
    if ratio_a <= ratio_b {
        return Ok(y11_case(&a, &b, &q));
    }

    let mut swapped = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            swapped[3 * i + j] = q[3 * j + i];
        }
    }

    Ok(y11_case(&b, &a, &swapped))
}

/// One branch of Xu et al. Table 2, eliminating in `a`'s photon-number index. Clamped at zero: a
/// negative result would flip the sign of `mdi_e11`, which divides by it.
fn y11_case(a: &[f64], b: &[f64], q: &[f64]) -> f64 {
    let (mu_a, nu_a, om_a) = (a[0], a[1], a[2]);
    let (mu_b, nu_b, om_b) = (b[0], b[1], b[2]);

    // The four-cell double difference kills both zero-photon rows.
    let block = |ia: usize, ib: usize, xa: f64, xb: f64| {
        q[3 * ia + ib] * (xa + xb).exp() + q[3 * 2 + 2] * (om_a + om_b).exp()
            - q[3 * ia + 2] * (xa + om_b).exp()
            - q[3 * 2 + ib] * (om_a + xb).exp()
    };
    let m_decoy = block(1, 1, nu_a, nu_b);
    let m_signal = block(0, 0, mu_a, mu_b);

    let num = (mu_a * mu_a - om_a * om_a) * (mu_b - om_b) * m_decoy
        - (nu_a * nu_a - om_a * om_a) * (nu_b - om_b) * m_signal;
    let den =
        (mu_a - om_a) * (mu_b - om_b) * (nu_a - om_a) * (nu_b - om_b) * (mu_a - nu_a);

    (num / den).max(0.0)
}

/// Upper bound on the single-photon-pair error rate `e11` in the basis the four supplied
/// products were measured in -- Xu, Curty, Qi & Lo, Table 2. A SECURITY BOUND.
///
/// The four arguments `qe_*` are the products `Q*E` at the decoy and vacuum settings:
/// `qe_dd` at `(decoy_a, decoy_b)`, `qe_vv` at `(vacuum_a, vacuum_b)`, `qe_dv` at
/// `(decoy_a, vacuum_b)` and `qe_vd` at `(vacuum_a, decoy_b)`. `y11` is the lower bound on
/// the single-photon-pair yield IN THE SAME BASIS, from `mdi_y11` fed that basis's gains.
///
/// Infeasible data reads as `E_CAP`, never zero, which would claim error-free single photons.
#[pyfunction]
pub(crate) fn mdi_e11(
    nu_a: f64,
    om_a: f64,
    nu_b: f64,
    om_b: f64,
    qe_dd: f64,
    qe_vv: f64,
    qe_dv: f64,
    qe_vd: f64,
    y11: f64,
) -> PyResult<f64> {
    for (name, x) in [("nu_a", nu_a), ("om_a", om_a), ("nu_b", nu_b), ("om_b", om_b)] {
        check_nonneg(name, x)?;
    }

    for (name, x) in [
        ("qe_dd", qe_dd),
        ("qe_vv", qe_vv),
        ("qe_dv", qe_dv),
        ("qe_vd", qe_vd),
    ] {
        check_prob(name, x)?;
    }

    check_prob("y11", y11)?;

    if nu_a <= om_a || nu_b <= om_b {
        return Err(PyValueError::new_err(format!(
            "the decoy must be stronger than the vacuum setting on both arms, got \
             nu_a = {nu_a}, om_a = {om_a}, nu_b = {nu_b}, om_b = {om_b}"
        )));
    }

    if !(y11 > 0.0) {
        return Ok(E_CAP);
    }

    let num = qe_dd * (nu_a + nu_b).exp() + qe_vv * (om_a + om_b).exp()
        - qe_dv * (nu_a + om_b).exp()
        - qe_vd * (om_a + nu_b).exp();
    let err = num / ((nu_a - om_a) * (nu_b - om_b) * y11);

    if err.is_finite() && err >= 0.0 {
        return Ok(err.min(E_CAP));
    }

    Ok(E_CAP)
}

/// The three intensities of one party, `[signal, decoy, vacuum]`, strictly decreasing.
fn check_set(name: &str, s: &[f64]) -> PyResult<()> {
    if s.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must be [signal, decoy, vacuum] mean photon numbers, got {} values",
            s.len()
        )));
    }

    for (k, x) in s.iter().enumerate() {
        check_nonneg(&format!("{name}[{k}]"), *x)?;
    }

    if !(s[0] > s[1] && s[1] > s[2]) {
        return Err(PyValueError::new_err(format!(
            "{name} must decrease strictly: signal > decoy > vacuum, got {s:?}"
        )));
    }

    Ok(())
}

/// Single-photon-pair GAIN per pulse pair from its yield: `mu_a*mu_b*exp(-mu_a - mu_b)*yield`.
#[pyfunction]
pub(crate) fn mdi_gain(yield_11: f64, mu_a: f64, mu_b: f64) -> PyResult<f64> {
    check_prob("yield_11", yield_11)?;
    check_nonneg("mu_a", mu_a)?;
    check_nonneg("mu_b", mu_b)?;

    Ok(mu_a * mu_b * (-mu_a - mu_b).exp() * yield_11)
}

// Nine-cell decoy programme: `mdi_program`, `mdi_disturb` read Table 2's data as a LINEAR
// PROGRAMME over the truncated yield matrix on `src/lp.rs`. Observations only, like `mdi_y11`.
//
// The truncation is the whole cost: at Lo-Curty-Qi hardware (mu = 0.55, nu = 0.005) cut = 5 is
// LOOSER than the closed form past 150 km, cut = 8 at 200 km; cut = 10 beats it to 230 km.
// Against a TRUE vacuum setting Table 2 is near exact already and `mdi_disturb` refuses most data.

/// Photon-number cutoff the programme is written over unless the caller says otherwise.
const CUT_DEFAULT: usize = 10;

/// Largest cutoff: `(cut+1)(cut+2)/2` grid rows plus at most eighteen stay inside `lp::ROW_MAX`.
const CUT_CAP: usize = 16;

/// Round-off charge on the certificate, in `f64::EPSILON` times the row's own scale:
/// `sum_i |y_i| * (sum_j |A_ij| + |b_i|)` epsilons off a lower bound, onto an upper one.
const SLIP: f64 = 4.0;

/// Poisson weights `e^{-mu} mu^n / n!`, `n = 0..=cut`, by recurrence: a vacuum gives exactly
/// `[1, 0, 0, ...]` with no `0^0`.
fn weights(mu: f64, cut: usize) -> Vec<f64> {
    let mut out = vec![0.0; cut + 1];
    let mut w = (-mu).exp();
    out[0] = w;

    for n in 1..=cut {
        w *= mu / (n as f64);
        out[n] = w;
    }

    out
}

/// `(c, A, b, start, scale)` of the standard-form programme, `A` row-major.
type Cards = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Assemble `min sign * X_11` subject to `A v = b`, `v >= 0`, over the truncated grid
/// `n + m <= cut`, together with a dual start strictly feasible for every input.
///
/// Variables in order: `X_nm` on the grid; one box slack per grid point; and, per cell with a
/// non-empty Poisson tail, the pair `(s, r)` holding it. Rows in order: the box `X_nm + u_nm = 1`;
/// per cell the gain row `sum_G p_n(a) p_m(b) X_nm + s = Q`; and `s + r = tail`. Those two rows
/// read as `sum_G p_n p_m X_nm <= Q` and `>= Q - tail`.
///
/// **Do not split the pair into two rows.** Identical coefficients leave the dual an exactly null
/// direction, the path runs out to `|y| ~ 1e19`, and the round-off in `A v = b` is amplified past
/// the objective: measured returning a certified 8.38 for a yield whose true value is 1.05e-2.
///
/// Dual start, data-free: `-delta` on box rows, `-1` on gain and tail rows, `z > 0` for either sign
/// at `delta = 1 + max(0, -sign)`.
fn deck(a: &[f64], b: &[f64], obs: &[f64], cut: usize, sign: f64) -> Cards {
    let pts: Vec<(usize, usize)> = (0..=cut)
        .flat_map(|n| (0..=(cut - n)).map(move |m| (n, m)))
        .collect();
    let g = pts.len();
    let pa: Vec<Vec<f64>> = a.iter().map(|x| weights(*x, cut)).collect();
    let pb: Vec<Vec<f64>> = b.iter().map(|x| weights(*x, cut)).collect();
    let mut cell = vec![0.0; 9 * g];
    let mut tail = vec![0.0; 9];

    for i in 0..3 {
        for j in 0..3 {
            let k = 3 * i + j;
            let mut kept = 0.0;

            for (t, (n, m)) in pts.iter().enumerate() {
                let w = pa[i][*n] * pb[j][*m];
                cell[k * g + t] = w;
                kept += w;
            }

            // The observed gain caps the dropped tail too: six decades under 1 at a long span.
            tail[k] = (1.0 - kept).max(0.0).min(obs[k].max(0.0));
        }
    }

    let extra = tail.iter().filter(|t| **t > 0.0).count();
    let n = 2 * g + 2 * extra;
    let m = g + 9 + extra;
    let delta = 1.0 + (-sign).max(0.0);
    let mut cvec = vec![0.0; n];
    let mut amat = vec![0.0; m * n];
    let mut bvec = vec![0.0; m];
    let mut start = vec![0.0; m];
    // `n` outermost: `(1, 1)` is one past the `n = 0` row of `cut + 1` points.
    cvec[cut + 2] = sign;

    for t in 0..g {
        amat[t * n + t] = 1.0;
        amat[t * n + g + t] = 1.0;
        bvec[t] = 1.0;
        start[t] = -delta;
    }

    let mut col = 2 * g;
    let mut row = g;

    for k in 0..9 {
        for t in 0..g {
            amat[row * n + t] = cell[k * g + t];
        }

        bvec[row] = obs[k];
        start[row] = -1.0;

        if tail[k] > 0.0 {
            amat[row * n + col] = 1.0;
            amat[(row + 1) * n + col] = 1.0;
            amat[(row + 1) * n + col + 1] = 1.0;
            bvec[row + 1] = tail[k];
            start[row + 1] = -1.0;
            col += 2;
            row += 2;
        } else {
            row += 1;
        }
    }

    let mut scale = vec![0.0; m];

    for i in 0..m {
        let mut acc = bvec[i].abs();

        for j in 0..n {
            acc += amat[i * n + j].abs();
        }

        scale[i] = acc;
    }

    (cvec, amat, bvec, start, scale)
}

/// Run one programme: `(value, gap, iters)`, `value` pushed DOWN by `SLIP`, pessimistic for a
/// MINIMUM. A maximum negated in and out is pushed UP.
fn certify(cards: Cards, gaptol: f64) -> PyResult<(f64, f64, usize)> {
    let (cvec, amat, bvec, start, scale) = cards;
    let (value, gap, iters, y) = crate::lp::lp_dual(cvec, amat, bvec, start, gaptol)?;
    let mut slip = 0.0;

    for (yi, si) in y.iter().zip(scale.iter()) {
        slip += yi.abs() * si;
    }

    Ok((value - SLIP * f64::EPSILON * slip, gap, iters))
}

/// Validates both nine-cell programmes' grid arguments.
fn check_deck(
    name: &str,
    a: &[f64],
    b: &[f64],
    obs: &[f64],
    cut: usize,
    reltol: f64,
) -> PyResult<()> {
    check_set("a", a)?;
    check_set("b", b)?;

    if obs.len() != 9 {
        return Err(PyValueError::new_err(format!(
            "{name} must be the 3x3 grid over joint intensity settings, row-major with \
             Alice's intensity as the row, got {} values",
            obs.len()
        )));
    }

    for (k, x) in obs.iter().enumerate() {
        check_prob(&format!("{name}[{k}]"), *x)?;

        if !(*x > 0.0) {
            return Err(PyValueError::new_err(format!(
                "{name}[{k}] must be strictly positive, got {x}: an empty cell leaves the \
                 programme no strictly feasible point. Use mdi_y11 or mdi_e11, which accept \
                 an empty cell"
            )));
        }
    }

    if cut < 2 || cut > CUT_CAP {
        return Err(PyValueError::new_err(format!(
            "cut is the photon-number truncation n + m <= cut and must lie between 2 and \
             {CUT_CAP}, got {cut}"
        )));
    }

    check_pos("reltol", reltol)?;

    Ok(())
}

/// Lower bound on `Y11` from ALL NINE cells, by linear programme. A SECURITY BOUND, tighter than
/// `mdi_y11`; `mdi_yield` is the forward model both sit under. Arguments as `mdi_y11`; `cut`
/// truncates at `n + m <= cut`, `reltol` is the gap as a fraction of the closed form.
///
/// Returns `(bound, plain, gap, iters)`, `plain` being `mdi_y11`, which `bound` must beat or this
/// raises. `gap` is a diagnostic. No case split: the assembly is symmetric in `a`, `b`.
#[pyfunction]
#[pyo3(signature = (a, b, gains, cut=CUT_DEFAULT, reltol=1e-6))]
pub(crate) fn mdi_program(
    a: Vec<f64>,
    b: Vec<f64>,
    gains: Vec<f64>,
    cut: usize,
    reltol: f64,
) -> PyResult<(f64, f64, f64, usize)> {
    check_deck("gains", &a, &b, &gains, cut, reltol)?;

    let plain = mdi_y11(a.to_vec(), b.to_vec(), gains.to_vec())?;
    let floor = plain.max(f64::MIN_POSITIVE);
    let (bound, gap, iters) = certify(deck(&a, &b, &gains, cut, 1.0), reltol * floor)?;

    if !(bound <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "the certified {bound} is not a yield: it left [0, 1] after {iters} \
             iterations and proves nothing. Use mdi_y11"
        )));
    }

    if !(bound > plain) {
        return Err(PyValueError::new_err(format!(
            "the certified {bound} for cut = {cut} is at or below the {plain} mdi_y11 \
             gives on the same cells: nothing tighter than the closed form was proved. \
             Raise cut, or use mdi_y11"
        )));
    }

    Ok((bound, plain, gap, iters))
}

/// Upper bound on `e11` from ALL NINE cells of the `Q * E` grid, by linear programme over
/// `e_nm * Y_nm`. A SECURITY BOUND, tighter than `mdi_e11`. `y11` is a LOWER bound IN THE SAME
/// BASIS, `mdi_program`'s to compare like with like.
///
/// **The sign is arranged here, not in the solver:** `lp.rs` bounds a minimum from below, so the
/// maximum of `e11 * Y11` is the negated minimum of its negative. The reverse is insecure.
///
/// Returns `(bound, plain, gap, iters)`, `plain` being `mdi_e11` at the same `y11`, which `bound`
/// must beat. A numerator <= 0 is refused, never clamped.
#[pyfunction]
#[pyo3(signature = (a, b, errors, y11, cut=CUT_DEFAULT, reltol=1e-6))]
pub(crate) fn mdi_disturb(
    a: Vec<f64>,
    b: Vec<f64>,
    errors: Vec<f64>,
    y11: f64,
    cut: usize,
    reltol: f64,
) -> PyResult<(f64, f64, f64, usize)> {
    check_deck("errors", &a, &b, &errors, cut, reltol)?;
    check_prob("y11", y11)?;

    if !(y11 > 0.0) {
        return Err(PyValueError::new_err(format!(
            "y11 must be strictly positive, got {y11}: the error rate is a share of that \
             yield. Pass mdi_program's bound"
        )));
    }

    let plain = mdi_e11(
        a[1],
        a[2],
        b[1],
        b[2],
        errors[4],
        errors[8],
        errors[5],
        errors[7],
        y11,
    )?;
    let floor = (plain * y11).max(f64::MIN_POSITIVE);
    let (value, gap, iters) = certify(deck(&a, &b, &errors, cut, -1.0), reltol * floor)?;
    let top = -value;

    if !(top > 0.0) {
        return Err(PyValueError::new_err(format!(
            "the certified {top} is not an error count: the largest e11*Y11 came out <= 0 \
             after {iters} iterations, which would claim error-free single photons. Use \
             mdi_e11"
        )));
    }

    let bound = (top / y11).min(E_CAP);

    if !(bound < plain) {
        return Err(PyValueError::new_err(format!(
            "the certified {bound} for cut = {cut} is at or above the {plain} mdi_e11 \
             gives at the same y11: nothing tighter than the closed form was proved. Raise \
             cut; a vacuum setting of exactly zero makes mdi_e11's \
             four-cell elimination exact, no cut helps there, and mdi_e11 is the bound"
        )));
    }

    Ok((bound, plain, gap, iters))
}

/// Asymptotic MDI-BB84 key rate, bits per PULSE PAIR: Xu, Curty, Qi & Lo Eq. (1),
/// `q11*[1 - h2(e11_x)] - q_z*f_ec*h2(e_z)`.
///
/// `q11` is the key-basis single-photon-pair GAIN (`mdi_gain`); `e11_x` the TEST-basis
/// single-photon-pair error rate, standing in for the phase error; `q_z`, `e_z` the key basis's
/// gain and QBER at signal intensities.
///
/// **No sifting prefactor:** `q_z` counts key-basis rounds only, in the biased-basis limit.
/// `discrete::bb84_rate`'s `q_sift` would charge sifting twice.
///
/// Ma & Razavi Eq. (7), the ideal single-photon-source rate `Y11*[1 - h2(e11) - f*h2(e11)]`, is
/// this expression at `q11 = q_z = Y11` and `e11_x = e_z = e11`.
#[pyfunction]
pub(crate) fn mdi_rate(q11: f64, e11_x: f64, q_z: f64, e_z: f64, f_ec: f64) -> PyResult<f64> {
    check_prob("q11", q11)?;
    check_err("e11_x", e11_x)?;
    check_prob("q_z", q_z)?;
    check_err("e_z", e_z)?;
    check_fec("f_ec", f_ec)?;

    if q11 > q_z {
        return Err(PyValueError::new_err(format!(
            "q11 must not exceed q_z (got {q11} > {q_z}): the single-photon-pair gain is \
             part of the key basis's total gain"
        )));
    }

    let rate = q11 * (1.0 - h2(e11_x)) - q_z * f_ec * h2(e_z);

    Ok(rate.max(0.0))
}

// Finite key. Curty, Xu, Cui, Lim, Tamaki & Lo, "Finite-key analysis for
// measurement-device-independent quantum key distribution", Nature
// Communications 5, 3732 (2014), arXiv:1307.1081.
//
// The decoy table is TWO-dimensional: a marginal fed to `decoy.rs`'s one-axis inversion drops the
// cross terms and RAISES the bound. Every count is PER BELL STATE k.

/// Curty's epsilon budget at three intensities per side, collapsed to one common epsilon:
/// `eps_{k,sec} = 2(eps' + 2 eps_e + eps_hat) + eps_b + eps_0 + eps_1 + eps_PA`, at
/// `gamma_{a,b} = 3 eps` per grid cell with `eps_0 <= 11 eps`, `eps_1 <= 29 eps` and
/// `eps_e <= 55 eps`, so `2(1 + 110 + 1) + 1 + 11 + 29 + 1 = 266`.
const EPS_TERMS: f64 = 266.0;

/// Claim 3's `test_1`, `(2/eps)^(1/mu_L) <= exp[(3/(4 sqrt 2))^2]`, in logarithms:
/// `ln(2/eps) <= (9/32) mu_L`.
const T1_SLOPE: f64 = 9.0 / 32.0;

/// Claim 3's `test_2`, `(1/eps)^(1/mu_L) < exp(1/3)`, in logarithms.
const T2_SLOPE: f64 = 1.0 / 3.0;

/// The `(hi, lo)` index pairs of a `[signal, decoy, vacuum]` ladder: every `a_0 > a_1` Curty's
/// vectors `v = [a_0, a_1, b_0, b_1]` may take.
const RUNGS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// The `(v, v')` rung pairs sharing one intensity with `v`'s other strictly larger. All give
/// `sum(v) > sum(v')`, so `c_11 > 0` with no run-time test. Same values as `RUNGS`, not it: entries
/// index `RUNGS`, not the ladder.
const NESTED: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// `g(x, y) = sqrt(2 x ln(1/y))` of Curty's Claim 1, taken in `beta = ln(1/y)` so a caller
/// composing `y = eps^4/16` never forms the ratio. Zero at a vanishing count, not NaN.
fn dev(x: f64, beta: f64) -> f64 {
    if x > 0.0 && beta > 0.0 {
        return (2.0 * x * beta).sqrt();
    }

    0.0
}

/// `(delta_lo, delta_hi, branch)`: Curty's Claim 3 multiplicative Chernoff deviation of an
/// observed count `x` over `n` Bernoulli trials at failure probability `eps`. The OBSERVED value
/// carries the bound, not the mean. The Hoeffding bound `mu_L = x - sqrt((n/2) ln(1/eps))` picks:
///
/// 1. `test_1 && test_2`  -- both tails multiplicative, `g(x, eps^4/16)` and `g(x, eps^(3/2))`
/// 2. `test_1 && test_3`  -- upper tail on the weaker `exp(-mu eps^2/4)` form, `g(x, eps^2)`
/// 3. `test_1` alone      -- upper tail falls back to Hoeffding
/// 4. `test_2` alone      -- lower tail falls back to Hoeffding
/// 5. `test_3` alone      -- lower tail Hoeffding, upper tail `g(x, eps^2)`
/// 6. none                -- both tails Hoeffding
///
/// A branch above 1 means too few counts for the multiplicative form: ~105 at `eps = 1e-10/266`.
///
/// `mu_L <= 0` is branch 6: a negative exponent makes `(2/eps)^(1/mu_L)` SMALL, a spurious pass.
///
/// Claim 3 item 6 composes as `2 eps`; `EPS_TERMS` charges `3 eps` throughout, loose by one term.
///
/// Curty prints `log` in the Hoeffding fallbacks where his derivation gives `ln`; `ln` runs here,
/// tighter by `sqrt(log2 / ln) = 1.2011`. See `dev/novel.md` E13c.
#[pyfunction]
pub(crate) fn mdi_deviate(x: f64, n: f64, eps: f64) -> PyResult<(f64, f64, u32)> {
    check_nonneg("x", x)?;
    check_nonneg("n", n)?;
    check_eps("eps", eps)?;

    if x > n {
        return Err(PyValueError::new_err(format!(
            "x must not exceed n (got {x} > {n}): x is the observed number of successes \
             among the n Bernoulli trials the Chernoff bound is taken over"
        )));
    }

    let beta = (1.0 / eps).ln();
    let flat = (0.5 * n * beta).sqrt();
    let mu_lo = x - flat;
    let cut = (2.0 * ::std::f64::consts::E - 1.0) / 2.0;
    let t1 = mu_lo > 0.0 && (2.0 / eps).ln() <= T1_SLOPE * mu_lo;
    let t2 = mu_lo > 0.0 && beta < T2_SLOPE * mu_lo;
    let t3 = mu_lo > 0.0 && beta < cut * cut * mu_lo;

    // The lower tail is the same multiplicative form whenever test_1 holds.
    let lo = if t1 {
        dev(x, 16f64.ln() + 4.0 * beta)
    } else {
        flat
    };

    let (hi, upper) = if t2 {
        (dev(x, 1.5 * beta), 0u32)
    } else if t3 {
        (dev(x, 2.0 * beta), 1u32)
    } else {
        (flat, 2u32)
    };

    let branch = match (t1, upper) {
        (true, 0) => 1,
        (true, 1) => 2,
        (true, _) => 3,
        (false, 0) => 4,
        (false, 1) => 5,
        (false, _) => 6,
    };

    Ok((lo, hi, branch))
}

/// The three-by-three grid of announcement counts, one cell per joint intensity setting.
fn check_grid(name: &str, g: &[f64]) -> PyResult<()> {
    if g.len() != 9 {
        return Err(PyValueError::new_err(format!(
            "{name} must be the 3x3 grid over joint intensity settings, row-major with \
             Alice's intensity as the row, got {} values",
            g.len()
        )));
    }

    for (k, x) in g.iter().enumerate() {
        check_nonneg(&format!("{name}[{k}]"), *x)?;
    }

    Ok(())
}

/// The three per-cell quantities Curty's elimination is written in: the rescaled count
/// `|Z~^{a,b}| = e^{a+b} |Z^{a,b}| / p_{a,b}` and the two rescaled deviations
/// `Gamma = e^{a+b} Delta / p_{a,b}`, `Gamma_hat = e^{a+b} Delta_hat / p_{a,b}`.
///
/// Claim 3's `n` is the grid TOTAL; the deviation reads the cell's own count.
fn rescale(
    a: &[f64],
    b: &[f64],
    p: &[f64],
    g: &[f64],
    eps: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let total: f64 = g.iter().sum();
    let mut zt = vec![0.0; 9];
    let mut lo = vec![0.0; 9];
    let mut hi = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let k = 3 * i + j;
            let scale = (a[i] + b[j]).exp() / p[k];
            let (d_lo, d_hi, _) = mdi_deviate(g[k], total, eps)?;
            zt[k] = scale * g[k];
            lo[k] = scale * d_lo;
            hi[k] = scale * d_hi;
        }
    }

    Ok((zt, lo, hi))
}

/// The nine joint intensity probabilities `p_{a,b,basis}`, each positive and summing to at
/// most 1 -- the remainder is the other basis.
fn check_share(name: &str, p: &[f64]) -> PyResult<()> {
    if p.len() != 9 {
        return Err(PyValueError::new_err(format!(
            "{name} must be the 3x3 grid of joint intensity probabilities, got {} values",
            p.len()
        )));
    }

    for (k, x) in p.iter().enumerate() {
        check_pos(&format!("{name}[{k}]"), *x)?;
    }

    let total: f64 = p.iter().sum();

    if total <= 1.0 + 1e-9 {
        return Ok(());
    }

    Err(PyValueError::new_err(format!(
        "{name} must sum to at most 1, got {total}: it is the probability of one basis's \
         joint intensity settings, and the complement is the other basis"
    )))
}

/// `tau_11 = sum_{a,b} e^{-(a+b)} a b p_{a,b}`, Curty's `tau_nm` at `n = m = 1`: the probability
/// Alice and Bob each emitted exactly one photon in this basis.
fn tau11(a: &[f64], b: &[f64], p: &[f64]) -> f64 {
    let mut out = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            out += (-(a[i] + b[j])).exp() * a[i] * b[j] * p[3 * i + j];
        }
    }

    out
}

/// Curty's four-cell double difference `G_{k,v} = |Z~^{a0,b0}| + |Z~^{a1,b1}| - |Z~^{a0,b1}|
/// - |Z~^{a1,b0}|`, which kills every `S_{k,0m}` and `S_{k,n0}` term outright.
fn corner(x: &[f64], i0: usize, i1: usize, j0: usize, j1: usize) -> f64 {
    x[3 * i0 + j0] + x[3 * i1 + j1] - x[3 * i0 + j1] - x[3 * i1 + j0]
}

/// The largest that same double difference's fluctuation can be, given each cell's
/// `delta ~ [-Gamma, Gamma_hat]`: cells entering with a MINUS sign contribute `+Gamma`, cells
/// entering with a plus sign `+Gamma_hat`.
///
/// **Departs from Curty's printed `Gamma_{k,v,v'}`** (`Gamma_hat` on the primed block, `Gamma` on
/// the unprimed), which is not the maximum: the lower deviation is ~1.65x larger at
/// `beta = ln(266/1e-10)`, and too small a term raises `S_11` and the key.
fn spread(lo: &[f64], hi: &[f64], i0: usize, i1: usize, j0: usize, j1: usize) -> f64 {
    hi[3 * i0 + j0] + hi[3 * i1 + j1] + lo[3 * i0 + j1] + lo[3 * i1 + j0]
}

/// Lower bound on `S_{k,11}`, the announcements of Bell state `k` in which Alice and Bob EACH
/// emitted exactly one photon, in the basis the supplied grid was measured in. Curty's
/// `S_{k,11} >= max{s_{k,V}, s_{k,V'}, 0}` over both of his cases.
///
/// `a` and `b` are `[signal, decoy, vacuum]` mean photon numbers per party, strictly decreasing;
/// `probs` and `counts` are the 3x3 joint-setting probability and announcement grids, row-major
/// with Alice's intensity as the row; `eps` is `mdi_eps(eps_sec)`.
///
/// A SECURITY BOUND over counts, the two-dimensional `decoy::decoy_counts`. Clamped at zero.
#[pyfunction]
pub(crate) fn mdi_pairs(
    a: Vec<f64>,
    b: Vec<f64>,
    probs: Vec<f64>,
    counts: Vec<f64>,
    eps: f64,
) -> PyResult<f64> {
    check_pairs(&a, &b, &probs, &counts, eps)?;

    pairs_core(&a, &b, &probs, &counts, eps)
}

/// The four grid arguments every pair bound shares, validated once.
fn check_pairs(a: &[f64], b: &[f64], p: &[f64], g: &[f64], eps: f64) -> PyResult<()> {
    check_set("a", a)?;
    check_set("b", b)?;
    check_share("probs", p)?;
    check_grid("counts", g)?;
    check_eps("eps", eps)?;

    Ok(())
}

/// `mdi_pairs` over borrowed slices, so `mdi_single` reaches it without copying its grids.
fn pairs_core(a: &[f64], b: &[f64], probs: &[f64], counts: &[f64], eps: f64) -> PyResult<f64> {
    let (zt, lo, hi) = rescale(a, b, probs, counts, eps)?;
    let tau = tau11(a, b, probs);
    let mut best = 0.0f64;

    for &(ra, ra2) in NESTED.iter() {
        for &(rb, rb2) in NESTED.iter() {
            let (i0, i1) = RUNGS[ra];
            let (j0, j1) = RUNGS[ra2];
            let (k0, k1) = RUNGS[rb];
            let (l0, l1) = RUNGS[rb2];
            let (a0, a1) = (a[i0], a[i1]);
            let (a2, a3) = (a[j0], a[j1]);
            let (b0, b1) = (b[k0], b[k1]);
            let (b2, b3) = (b[l0], b[l1]);

            // Case 1 weights the b-side squares, Case 2 the a-side; the ratio test picks
            // whichever elimination cancels the (1,2) or (2,1) term.
            let one = (a0 + a1) / (a2 + a3) > (b0 + b1) / (b2 + b3);
            let (wa, wb) = if one {
                ((b0 * b0 - b1 * b1) * (a0 - a1), (b2 * b2 - b3 * b3) * (a2 - a3))
            } else {
                ((a0 * a0 - a1 * a1) * (b0 - b1), (a2 * a2 - a3 * a3) * (b2 - b3))
            };

            let j = wa * corner(&zt, j0, j1, l0, l1) - wb * corner(&zt, i0, i1, k0, k1);
            let width = wa * spread(&lo, &hi, j0, j1, l0, l1)
                + wb * spread(&hi, &lo, i0, i1, k0, k1);
            let gap = if one {
                b0 + b1 - b2 - b3
            } else {
                a0 + a1 - a2 - a3
            };
            let c11 = (a0 - a1) * (a2 - a3) * (b0 - b1) * (b2 - b3) * gap;

            if c11 > 0.0 {
                best = best.max(tau * (j - width) / c11);
            }
        }
    }

    Ok(best.max(0.0))
}

/// Upper bound on `E_{k,11}`, the number of announcements of Bell state `k` in which both
/// parties emitted one photon AND their sifted bits differ, in the basis the supplied grid was
/// measured in. Curty's Eq. `E_{k,11} <= min_v (F_{k,v} - Gamma_{k,v}) tau_11 /
/// ((a_0 - a_1)(b_0 - b_1))`.
///
/// `errors` is the 3x3 grid of ERROR counts `|E_k^{a,b}|`. `F_{k,v}`'s remaining terms are
/// NON-NEGATIVE: no second elimination, no case split. Clamped at zero from below only.
#[pyfunction]
pub(crate) fn mdi_faults(
    a: Vec<f64>,
    b: Vec<f64>,
    probs: Vec<f64>,
    errors: Vec<f64>,
    eps: f64,
) -> PyResult<f64> {
    check_set("a", &a)?;
    check_set("b", &b)?;
    check_share("probs", &probs)?;
    check_grid("errors", &errors)?;
    check_eps("eps", eps)?;

    let (et, lo, hi) = rescale(&a, &b, &probs, &errors, eps)?;
    let tau = tau11(&a, &b, &probs);
    let mut best = f64::INFINITY;

    for &(i0, i1) in RUNGS.iter() {
        for &(j0, j1) in RUNGS.iter() {
            // Curty's Gamma_{k,v} is NEGATIVE and subtracted, so the width is added.
            let width = lo[3 * i0 + j0] + lo[3 * i1 + j1] + hi[3 * i0 + j1] + hi[3 * i1 + j0];
            let den = (a[i0] - a[i1]) * (b[j0] - b[j1]);
            let got = tau * (corner(&et, i0, i1, j0, j1) + width) / den;

            best = best.min(got);
        }
    }

    if !best.is_finite() {
        return Ok(0.0);
    }

    Ok(best.max(0.0))
}

/// Serfling's inequality for random sampling WITHOUT replacement, Curty's `Lambda(x, y, z) =
/// sqrt((x - y + 1) ln(1/z) / (2 x y))` applied as `n = floor(y m / x - y Lambda(x, y, z))`.
/// `m` is the certified count in the population of size `x`, `y` the size of the sample drawn
/// from it. Clamped at zero and floored: a key length is whole bits.
fn draw(m: f64, x: f64, y: f64, eps: f64) -> f64 {
    if !(x > 0.0) || !(y > 0.0) {
        return 0.0;
    }

    let width = ((x - y + 1.0) * (1.0 / eps).ln() / (2.0 * x * y)).sqrt();

    (y * m / x - y * width).max(0.0).floor()
}

/// Lower bound on `n_{k,0}`, the bits of the code string `Z_k` where ALICE emitted vacuum:
/// Curty's `T_{k,0m}` bound over the `b_signal` column, converted to `m_{k,0}`, then Serfling onto
/// the `n_key` code bits. `n_key` is `n_k`, the bits Alice draws from the signal-setting
/// announcements to form the code string; the rest go to the error-rate test.
///
/// Reads the `b_signal` COLUMN only, so it charges `3 gamma` where `mdi_pairs` charges `9 gamma`.
/// `n_{k,0}` enters the key length at FULL weight: a vacuum emission tells Eve nothing.
#[pyfunction]
pub(crate) fn mdi_vacuum(
    a: Vec<f64>,
    b: Vec<f64>,
    probs: Vec<f64>,
    counts: Vec<f64>,
    n_key: f64,
    eps: f64,
) -> PyResult<f64> {
    // `check_pairs` respelled with `n_key` between the grid and `eps`: the order is which refusal
    // a doubly-bad input names.
    check_set("a", &a)?;
    check_set("b", &b)?;
    check_share("probs", &probs)?;
    check_grid("counts", &counts)?;
    check_nonneg("n_key", n_key)?;
    check_eps("eps", eps)?;

    if n_key > counts[0] {
        return Err(PyValueError::new_err(format!(
            "n_key must not exceed the signal-setting count counts[0] (got {n_key} > {}): \
             the code string is drawn from those announcements",
            counts[0]
        )));
    }

    let (zt, lo, hi) = rescale(&a, &b, &probs, &counts, eps)?;
    let mut t0 = 0.0f64;

    for &(i0, i1) in RUNGS.iter() {
        let (a0, a1) = (a[i0], a[i1]);
        let l = a0 * zt[3 * i1] - a1 * zt[3 * i0];

        t0 = t0.max((l - a0 * hi[3 * i1] - a1 * lo[3 * i0]) / (a0 - a1));
    }

    let raw = (probs[0] * (-(a[0] + b[0])).exp() * t0).max(0.0);
    let m0 = (raw - dev(raw, (1.0 / eps).ln())).max(0.0);

    Ok(draw(m0, counts[0], n_key, eps))
}

/// Lower bound on `n_{k,1}`, the number of bits of the code string `Z_k` where Alice AND Bob
/// each emitted a single photon. `mdi_pairs` over the key basis, converted through Curty's
/// `m_{k,1} = p_{a_s,b_s|11} S_{k,11} - Delta_1`, then Serfling onto the `n_key` code bits.
///
/// `tau_11` cancels between `S_{k,11}` and `p_{a_s,b_s|11}`.
#[pyfunction]
pub(crate) fn mdi_single(
    a: Vec<f64>,
    b: Vec<f64>,
    probs: Vec<f64>,
    counts: Vec<f64>,
    n_key: f64,
    eps: f64,
) -> PyResult<f64> {
    check_nonneg("n_key", n_key)?;
    check_grid("counts", &counts)?;

    if n_key > counts[0] {
        return Err(PyValueError::new_err(format!(
            "n_key must not exceed the signal-setting count counts[0] (got {n_key} > {}): \
             the code string is drawn from those announcements",
            counts[0]
        )));
    }

    check_pairs(&a, &b, &probs, &counts, eps)?;

    let s11 = pairs_core(&a, &b, &probs, &counts, eps)?;
    let tau = tau11(&a, &b, &probs);

    if !(tau > 0.0) {
        return Ok(0.0);
    }

    let share = (-(a[0] + b[0])).exp() * a[0] * b[0] * probs[0] / tau;
    let raw = share * s11;
    let m1 = (raw - dev(raw, (1.0 / eps).ln())).max(0.0);

    Ok(draw(m1, counts[0], n_key, eps))
}

/// Upper bound on the single-photon-pair PHASE ERROR RATE of the code string, Curty's Eq.
/// for `e_{k,1}` with his `Upsilon(x, y, z) = sqrt((x + 1) ln(1/z) / (2 y (x + y)))`.
///
/// `n1` is `n_{k,1}` from `mdi_single` (key basis), `nbar` is `n_bar_{k,1}` from `mdi_pairs` run
/// on the TEST basis grid, and `efault` is `e_bar_{k,1}` from `mdi_faults`.
///
/// **Curty's expression is a COUNT, his key length writes `h(e_{k,1})`**: this returns the RATIO
/// `e_{k,1}/n_{k,1}` that `mdi_length` consumes.
///
/// Capped at 1/2, not Curty's 1: past 1/2 `1 - h2(e)` rises again. A vanishing sample returns 1/2.
#[pyfunction]
pub(crate) fn mdi_phase(n1: f64, nbar: f64, efault: f64, eps: f64) -> PyResult<f64> {
    check_nonneg("n1", n1)?;
    check_nonneg("nbar", nbar)?;
    check_nonneg("efault", efault)?;
    check_eps("eps", eps)?;

    if !(n1 > 0.0) || !(nbar > 0.0) {
        return Ok(E_CAP);
    }

    let width = ((n1 + 1.0) * (1.0 / eps).ln() / (2.0 * nbar * (n1 + nbar))).sqrt();
    let count = (n1 * (efault / nbar) + (n1 + nbar) * width).ceil().min(n1);

    Ok((count / n1).clamp(0.0, E_CAP))
}

/// Per-bound failure probability `eps_sec / EPS_TERMS` from the PER BELL STATE secrecy parameter,
/// every bound at an equal share.
#[pyfunction]
pub(crate) fn mdi_eps(eps_sec: f64) -> PyResult<f64> {
    check_eps("eps_sec", eps_sec)?;

    Ok(eps_sec / EPS_TERMS)
}

/// Finite-key length in BITS for ONE announced Bell state `k`, Curty's Eq. for `l_k`:
/// `l_k <= n_{k,0} + n_{k,1}[1 - h(e_{k,1})] - leak_EC - log2(8/eps_cor)
/// - 2 log2(2/(eps' eps_hat)) - 2 log2(1/(2 eps_PA))`.
///
/// `n0`, `n1` are `mdi_vacuum` and `mdi_single`; `phi` is `mdi_phase`; `n_key` is the code
/// string length `n_k` and `e_key` its measured error rate. `eps_sec` is the composed
/// PER-STATE secrecy parameter, and `eps' = eps_hat = eps_PA = eps_sec/266` inside.
///
/// **The total key sums over announced Bell states**, `l = sum_k l_k` at
/// `eps_sec = sum_k eps_{k,sec}`: a linear-optics relay announces two, so run twice at `eps/2`.
///
/// Curty prints `leak_EC = n_key * f_ec * h2(e_key)` with its `n_k`; the factor
/// `discrete::bb84_length` restores is Lim's alone. Clamped at zero and floored.
#[pyfunction]
pub(crate) fn mdi_length(
    n0: f64,
    n1: f64,
    phi: f64,
    n_key: f64,
    e_key: f64,
    f_ec: f64,
    eps_sec: f64,
    eps_cor: f64,
) -> PyResult<f64> {
    check_nonneg("n0", n0)?;
    check_nonneg("n1", n1)?;
    check_err("phi", phi)?;
    check_nonneg("n_key", n_key)?;
    check_err("e_key", e_key)?;
    check_fec("f_ec", f_ec)?;
    check_eps("eps_sec", eps_sec)?;
    check_eps("eps_cor", eps_cor)?;

    if n0 + n1 > n_key {
        return Err(PyValueError::new_err(format!(
            "n0 + n1 must not exceed n_key (got {} > {n_key}): the vacuum and \
             single-photon-pair bits are part of the code string they were estimated from",
            n0 + n1
        )));
    }

    let eps = eps_sec / EPS_TERMS;
    let leak = f_ec * n_key * h2(e_key);
    let budget = (8.0 / eps_cor).log2() + 2.0 * (2.0 / (eps * eps)).log2()
        + 2.0 * (1.0 / (2.0 * eps)).log2();
    let len = n0 + n1 * (1.0 - h2(phi)) - leak - budget;

    Ok(len.max(0.0).floor())
}
