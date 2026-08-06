use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::lp::{polytope_cap, simplex_cap, Facets, ROW_MAX};
use crate::numeric::{bisect, kl_bits};
use crate::std::{
    check_eps, check_err, check_fec, check_gate, check_nonneg, check_pos, check_prob,
    check_unit, h2,
};

// Round-robin differential phase shift: Sasaki, Yamamoto & Koashi, Nature 509, 475 (2014).
// Privacy amplification reads the packet length L and the photon number only, no observable.
// A packet yields at most one bit, so every rate here is divided by L.
//
//   `rrdps_phase` + h2   Takesue, Sasaki, Tamaki & Koashi, Nature Photonics 9, 827 (2015),
//                        arXiv:1505.07914, Eqs. (3) and (4).
//   `rrdps_tag`          Zhang, Yuan, Cao & Ma, New J. Phys. 19, 033013 (2017),
//                        arXiv:1505.02481, Eq. (4.4); the same expression is Yin, Wang, Chen,
//                        Han, Wang, Guo & Han, Nature Communications 9, 457 (2018),
//                        arXiv:1702.01260, Methods.
//   `rrdps_gllp`         Zhang, Yuan, Cao & Ma Eq. (4.5).
//
// Published (4.4)/(4.5) are arXiv v1's (12)/(13), and v1's (13) is a DIFFERENT expression.
// Zhang et al. call (4.4) "the original SYK analysis" and (4.5) GLLP tagging: cite the equation.
// `rrdps_gllp <= rrdps_tag <= h2(rrdps_phase)` at every argument.
//
// `rrdps_collective` is Yin et al. Eq. (1),
//
//     I_AE <= max { sum_{k=1}^{nu} phi((L-k) x_k, k x_{k+1}) / (L-1) },
//
// over `x_1..x_{nu+1}` on the simplex, with `L >= nu + 1`.
//
// `rrdps_entropy` is Matsuura, Sasaki & Koashi, Phys. Rev. A 99, 042303 (2019), arXiv:1812.10916,
// Eq. (60): `H(X|Y)`, not `h2` of an error rate. Certified here; the paper's SLSQP reports no bracket.
// Eq. (60)'s `P(M, U, X=1) = c(M, U) P(M, U)` is eliminated as a change of variable: 2L coordinates.
// Their finite-size Eq. (58) is NOT implemented; `rrdps_finite` is Takesue's chain.

/// Longest packet any entry point will accept; `rrdps_optimum`'s sweep is quadratic in it.
const L_MAX: u64 = 4096;

/// Bisection steps: 2^-100 of a unit bracket is below f64 resolution.
const STEPS: usize = 100;

/// Ascent steps before giving up. Worst case 25,770 at `l = 4096, nu = 2047`; `nu <= 20` closes inside 250.
const ASCENT_MAX: usize = 60000;

/// Relative width of `[witness, cap]` at which the ascent stops; tighter than any `reltol` default.
const ASCENT_TOL: f64 = 1e-11;

/// Trailing ascent iterates kept as cutting planes.
const CUT_KEEP: usize = 4;

/// Central-path gap the linear programme is asked for, in bits per sifted bit.
const LP_GAPTOL: f64 = 1e-9;

/// Total cap under which Eq. (60)'s top source rows merge into one; moves the bound UP by < 1e-13 bits.
const TAIL_FLOOR: f64 = 1e-15;

/// Longest packet `rrdps_entropy` assembles: a source row per photon count, a simplex row and a row
/// per cut, against `lp::ROW_MAX`.
const POLYTOPE_MAX: u64 = (ROW_MAX - CUT_KEEP - 2) as u64;

fn check_train(l: u64) -> PyResult<()> {
    if l < 2 {
        return Err(PyValueError::new_err(format!(
            "l must be at least 2 pulses, got {l}: Bob reads the phase difference \
             of a PAIR drawn from the packet, and a packet of one pulse holds none"
        )));
    }

    if l > L_MAX {
        return Err(PyValueError::new_err(format!(
            "l must not exceed {L_MAX} pulses, got {l}: the variable-delay \
             interferometer has to switch among l - 1 delays per packet"
        )));
    }

    Ok(())
}

/// Poisson CDF at `k`, mean `m`. Past `m ~ 745` `exp(-m)` underflows and the tail reads 1: no key.
fn poisson_cdf(m: f64, k: u64) -> f64 {
    let mut term = (-m).exp();
    let mut sum = term;
    for i in 1..=k {
        term *= m / (i as f64);
        sum += term;
    }

    sum.min(1.0)
}

/// SYK's `min(n/(L-1), 1/2)`. Uncapped, `h2` turns back down and a known key reads as free.
fn phase_n(l: u64, n: u64) -> f64 {
    ((n as f64) / ((l - 1) as f64)).min(0.5)
}

/// Yin et al. Eq. (1)'s kernel `(x + y) h2(x / (x + y))`, jointly concave; zero arguments give 0.
fn phi(x: f64, y: f64) -> f64 {
    let mut out = 0.0;
    if x > 0.0 {
        out -= x * x.log2();
    }
    if y > 0.0 {
        out -= y * y.log2();
    }
    if x + y > 0.0 {
        out += (x + y) * (x + y).log2();
    }

    out
}

/// Leaked bits per sifted bit `h2(min(nu_th/(L-1), 1/2))`, at most `nu_th` photons per packet:
/// Sasaki, Yamamoto & Koashi, Nature 509, 475 (2014); printed as Zhang, Yuan, Cao & Ma,
/// New J. Phys. 19, 033013 (2017) Eq. (3.1). Exactly 1 bit once `nu_th >= (l - 1)/2`.
#[pyfunction]
pub(crate) fn rrdps_leak(l: u64, nu_th: u64) -> PyResult<f64> {
    check_train(l)?;

    Ok(h2(phase_n(l, nu_th)))
}

/// Probability a packet holds more than `nu_th` photons, Takesue et al. Eq. (2). `mu` is per PULSE;
/// the Poisson mean is `l * mu`.
#[pyfunction]
pub(crate) fn rrdps_src(l: u64, mu: f64, nu_th: u64) -> PyResult<f64> {
    check_train(l)?;
    check_nonneg("mu", mu)?;

    Ok((1.0 - poisson_cdf((l as f64) * mu, nu_th)).clamp(0.0, 1.0))
}

/// Phase error rate of the sifted key, Takesue et al. Eq. (3). `q` is the sifted rate per EMITTED
/// packet, Takesue's `Q = N/N_em`. Capped at 1/2, and 1/2 when `q <= e_src`.
#[pyfunction]
pub(crate) fn rrdps_phase(q: f64, l: u64, mu: f64, nu_th: u64) -> PyResult<f64> {
    check_prob("q", q)?;
    let src = rrdps_src(l, mu, nu_th)?;
    if !(q > 0.0) || src >= q {
        return Ok(0.5);
    }

    // Mix the RAW nu_th/(L-1), then cap: mixing the capped `phase_n` under-reads the result.
    let tagged = src / q;
    let share = ((nu_th as f64) / ((l - 1) as f64)).min(1.0);

    Ok((tagged + (1.0 - tagged) * share).min(0.5))
}

/// Privacy amplification per sifted bit, `(e_src + (q - e_src) * leak) / q`: Zhang, Yuan, Cao & Ma,
/// New J. Phys. 19, 033013 (2017) Eq. (4.4); Yin et al., Nature Communications 9, 457 (2018), Methods.
/// `leak` is `rrdps_leak` or `rrdps_collective`. 1 when `e_src >= q`.
#[pyfunction]
pub(crate) fn rrdps_tag(q: f64, e_src: f64, leak: f64) -> PyResult<f64> {
    check_prob("q", q)?;
    check_prob("e_src", e_src)?;
    check_prob("leak", leak)?;

    if !(q > 0.0) || e_src >= q {
        return Ok(1.0);
    }

    Ok(((e_src + (q - e_src) * leak) / q).min(1.0))
}

/// Privacy amplification per sifted bit charged per photon number, Zhang, Yuan, Cao & Ma Eq. (4.5),
/// Poisson weights at packet mean `l * mu`.
#[pyfunction]
pub(crate) fn rrdps_gllp(q: f64, l: u64, mu: f64, nu_th: u64) -> PyResult<f64> {
    check_prob("q", q)?;
    check_nonneg("mu", mu)?;
    check_train(l)?;

    let src = rrdps_src(l, mu, nu_th)?;
    if !(q > 0.0) || src >= q {
        return Ok(1.0);
    }

    // Stop past BOTH `nu_th` and `(l-1)/2`: `(l-1)/2` alone double-charges counts at or below `nu_th`.
    let stop = (l - 1).div_ceil(2).max(nu_th + 1);
    let m = (l as f64) * mu;
    let mut term = (-m).exp();
    let mut cdf = term;
    let mut acc = 0.0;
    for n in 1..stop {
        term *= m / (n as f64);
        cdf += term;
        if n > nu_th {
            acc += term * h2(phase_n(l, n));
        }
    }

    acc += (1.0 - cdf).max(0.0);

    Ok(((acc + (q - src) * h2(phase_n(l, nu_th))) / q).min(1.0))
}

/// Asymptotic key rate in bits per PULSE, Takesue et al. Eq. (4) divided by `l`; multiply by `l` for
/// Takesue's per-packet `G/N_em`. `cost` is `h2(rrdps_phase)`, `rrdps_tag` or `rrdps_gllp`. Clamped at zero.
#[pyfunction]
pub(crate) fn rrdps_rate(q: f64, e_bit: f64, cost: f64, f_ec: f64, l: u64) -> PyResult<f64> {
    check_prob("q", q)?;
    check_err("e_bit", e_bit)?;
    check_prob("cost", cost)?;
    check_fec("f_ec", f_ec)?;
    check_train(l)?;

    let rate = q * (1.0 - f_ec * h2(e_bit) - cost) / (l as f64);

    Ok(rate.max(0.0))
}

/// Yin et al. Eq. (1) as `joint_slope` edges, 0-indexed: `(k-1, l-k, k, k)` per photon number `k`.
fn simplex_edges(l: u64, nu: u64) -> Vec<(usize, f64, usize, f64)> {
    (1..=nu)
        .map(|k| ((k - 1) as usize, (l - k) as f64, k as usize, k as f64))
        .collect()
}

/// One multiplicative ascent step `x <- x * g / (x . g)`; `false` when there is nowhere to move.
/// Divide by `x . g`, not `f(x)`: equal by Euler's identity, but only `x . g` keeps `sum x = 1`.
/// The 1e-300 floor keeps `x` off the boundary `joint_slope` refuses.
fn simplex_step(x: &mut [f64], g: &[f64]) -> bool {
    let mut total = 0.0;

    for (xi, gi) in x.iter().zip(g.iter()) {
        total += xi * gi;
    }

    if !(total > 0.0) || !total.is_finite() {
        return false;
    }

    let mut mass = 0.0;

    for (xi, gi) in x.iter_mut().zip(g.iter()) {
        *xi = (*xi * gi / total).max(1e-300);
        mass += *xi;
    }

    if !(mass > 0.0) || !mass.is_finite() {
        return false;
    }

    for xi in x.iter_mut() {
        *xi /= mass;
    }

    true
}

fn boundary(l: u64, nu: u64) -> PyErr {
    PyValueError::new_err(format!(
        "the ascent for l = {l}, nu = {nu} reached a packet-weight vector with a vanishing \
         coordinate, where Eq. (1)'s gradient is infinite. Use rrdps_leak"
    ))
}

/// `rrdps_simplex` without argument checks.
fn simplex_bound(l: u64, nu: u64, reltol: f64) -> PyResult<(f64, f64, f64, usize, usize)> {
    let dim = (nu + 1) as usize;
    let edges = simplex_edges(l, nu);
    let mut x = vec![1.0 / (dim as f64); dim];
    let mut g = vec![0.0f64; dim];
    let mut witness = f64::NEG_INFINITY;
    let mut steps = 0usize;
    let mut closed = false;

    for _ in 0..ASCENT_MAX {
        let val = joint_slope(l, &edges, &x, &mut g).ok_or_else(|| boundary(l, nu))?;
        let mut top = f64::NEG_INFINITY;

        for gi in g.iter() {
            top = top.max(*gi);
        }

        witness = witness.max(val);

        if top - val <= ASCENT_TOL * val.abs().max(1.0) {
            closed = true;
            break;
        }

        if !simplex_step(&mut x, &g) {
            break;
        }

        steps += 1;
    }

    if !closed {
        return Err(PyValueError::new_err(format!(
            "the ascent for l = {l}, nu = {nu} did not close its bracket in {ASCENT_MAX} \
             steps, so nothing was certified. Use rrdps_leak"
        )));
    }

    let mut cuts = Vec::with_capacity(CUT_KEEP * dim);
    let mut levels = Vec::with_capacity(CUT_KEEP);
    let mut tangent = f64::INFINITY;

    for _ in 0..CUT_KEEP {
        let val = joint_slope(l, &edges, &x, &mut g).ok_or_else(|| boundary(l, nu))?;
        let mut top = f64::NEG_INFINITY;
        let mut dot = 0.0;

        for (xi, gi) in x.iter().zip(g.iter()) {
            top = top.max(*gi);
            dot += xi * gi;
        }

        // Form the intercept: Euler's identity zeroes it only while the objective is homogeneous.
        let level = val - dot;
        witness = witness.max(val);
        tangent = tangent.min(top + level);
        cuts.extend_from_slice(&g);
        levels.push(level);

        if !simplex_step(&mut x, &g) {
            break;
        }

        steps += 1;
    }

    let (raw, gap, iters, status) = simplex_cap(dim, &cuts, &levels, LP_GAPTOL)?;
    let slack = gap.abs() + 1e-9 * tangent.abs().max(1.0);

    // Same planes as `tangent`: a certified value above it is an unresolved path.
    if !(raw <= tangent + slack) {
        return Err(PyValueError::new_err(format!(
            "the interior-point path did not resolve l = {l}, nu = {nu}: it stopped after \
             {iters} iterations with status \"{status}\" at a certified {raw}, above the \
             {tangent} the tangent cap on the same cuts already gives. The value is a \
             valid bound and a useless one; nothing tighter than one plane was proved here"
        )));
    }

    let bound = raw.min(tangent).clamp(0.0, 1.0);
    let plain = h2(phase_n(l, nu));

    if bound > plain + 1e-9 * plain.max(1.0) {
        return Err(PyValueError::new_err(format!(
            "the certified {bound} for l = {l}, nu = {nu} sits above the {plain} \
             rrdps_leak gives on the same quantity. Use rrdps_leak"
        )));
    }

    if !(bound - witness <= reltol * witness.abs().max(1.0)) {
        return Err(PyValueError::new_err(format!(
            "the bracket for l = {l}, nu = {nu} closed only to {} bits per sifted bit, \
             above reltol = {reltol} (witness {witness}, bound {bound}). The bound is valid; \
             pass a larger reltol to accept it",
            bound - witness
        )));
    }

    Ok((bound, witness, tangent, steps, iters))
}

/// Yin et al. Eq. (1) as `(bound, witness, tangent, steps, iters)`, `l` pulses carrying `nu` photons.
///
/// `bound` is certified and is what `rrdps_collective` returns. `witness` is the objective at the
/// ascent's last point: a LOWER bracket that PROVES NOTHING. `tangent` is the one-plane cap without
/// the solver. `steps` counts ascent iterations, `iters` barrier Newton steps.
///
/// Cut placement is heuristic: a poor ascent widens `bound - witness`, never breaks `bound`.
/// `nu <= 20` takes under a millisecond; `l = 4096, nu = 4095` takes 2 s over 25,640 ascent steps.
#[pyfunction]
#[pyo3(signature = (l, nu, reltol=1e-9))]
pub(crate) fn rrdps_simplex(
    l: u64,
    nu: u64,
    reltol: f64,
) -> PyResult<(f64, f64, f64, usize, usize)> {
    check_train(l)?;
    check_pos("reltol", reltol)?;
    check_photons(l, nu)?;

    if nu == 0 {
        return Err(PyValueError::new_err(
            "nu = 0 leaves Eq. (1) an empty sum with no simplex to search. \
             rrdps_collective(l, 0) returns its value, 0",
        ));
    }

    simplex_bound(l, nu, reltol)
}

fn check_photons(l: u64, n: u64) -> PyResult<()> {
    if n + 1 > l {
        return Err(PyValueError::new_err(format!(
            "a packet of {l} pulses cannot be read at n = {n} photons: Yin et al.'s \
             Eq. (1) runs over the (n+1)-simplex and needs L >= n + 1. Use rrdps_leak, \
             which holds at every photon number and is exactly 1 here"
        )));
    }

    Ok(())
}

/// Eve's information per sifted bit against COLLECTIVE attacks, `l` pulses carrying `n` photons:
/// Yin, Wang, Chen, Han, Wang, Guo & Han, Nature Communications 9, 457 (2018), Eq. (1) without
/// Eq. (2)'s error-rate constraint, so it reads no observable.
///
/// `n = 1` is exact by bisection; `n >= 2` is certified through `rrdps_simplex`.
/// Strictly below `rrdps_leak(l, n)` (Yin et al.'s Corollary at `n = 1`); both are 1 at `l = 2`.
#[pyfunction]
pub(crate) fn rrdps_collective(l: u64, n: u64) -> PyResult<f64> {
    check_train(l)?;
    check_photons(l, n)?;

    if n == 0 {
        return Ok(0.0);
    }

    if n > 1 {
        return Ok(simplex_bound(l, n, 1e-9)?.0);
    }

    let a = (l - 1) as f64;

    // d/dx of phi((l-1)x, 1-x): positive at 0, negative at 1.
    let slope = |x: f64| {
        -a * (a * x).log2() + (1.0 - x).log2() + (a - 1.0) * (1.0 + (a - 1.0) * x).log2()
    };
    let (lo, hi) = bisect(0.0, 1.0, STEPS, |x| slope(x) > 0.0);
    let x = 0.5 * (lo + hi);

    Ok((phi(a * x, 1.0 - x) / a).clamp(0.0, 1.0))
}

/// Eq. (60)'s coordinate for `(m, u)`, contiguous per count: `(0, 0)`, `(1, 0), (1, 1)` .. `(l, 1)`.
/// `(0, 1)` and `(l, 0)` are absent: `c(M, U)` leaves `[0, 1]` there.
fn seat(l: u64, m: u64, u: u64) -> usize {
    if m == 0 {
        0
    } else if m == l {
        (2 * l - 1) as usize
    } else {
        (2 * m - 1 + u) as usize
    }
}

/// First coordinate of photon count `m` and how many the count holds.
fn perch(l: u64, m: u64) -> (usize, usize) {
    if m == 0 || m == l {
        (seat(l, m, 1), 1)
    } else {
        (seat(l, m, 0), 2)
    }
}

/// Eq. (60)'s `H(X|Y)` as `joint_slope` edges, one per `y = (a, b)`. Each coordinate's weights sum
/// to `L - 1`: the objective is homogeneous of degree one.
fn joint_edges(l: u64) -> Vec<(usize, f64, usize, f64)> {
    let mut out = Vec::with_capacity(2 * (l - 1) as usize);

    for a in 0..(l - 1) {
        for b in 0..2u64 {
            out.push((
                seat(l, a + b, b),
                (l - 1 - a) as f64,
                seat(l, a + 2 - b, 1 - b),
                (a + 1) as f64,
            ));
        }
    }

    out
}

/// Binomial `b_{l,p}` masses, in logs so the mode survives an underflowing tail.
fn binomial(l: u64, p: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; (l + 1) as usize];

    if !(p > 0.0) {
        out[0] = 1.0;

        return out;
    }

    if !(p < 1.0) {
        out[l as usize] = 1.0;

        return out;
    }

    let lp = p.ln();
    let lq = (1.0 - p).ln();
    let mut lnc = 0.0f64;

    for m in 0..=l {
        if m > 0 {
            lnc += (((l - m + 1) as f64) / (m as f64)).ln();
        }

        out[m as usize] = (lnc + (m as f64) * lp + ((l - m) as f64) * lq).exp();
    }

    out
}

/// Eq. (60)'s source rows `P(M) <= b_{l,p}(M)/q` as `(first, len, cap)` bands covering every
/// coordinate once. Rows exist for `M >= l p` only; a cap `>= 1` is `INFINITY`, no row.
/// The `TAIL_FLOOR` merge and the underflow floor both widen the polytope: the leakage only rises.
fn joint_bands(l: u64, p_src: f64, q: f64) -> Vec<(usize, usize, f64)> {
    let mass = binomial(l, p_src);
    let mut caps = vec![f64::INFINITY; (l + 1) as usize];

    for m in 0..=l {
        let cap = mass[m as usize] / q;

        if (m as f64) >= (l as f64) * p_src && cap < 1.0 {
            caps[m as usize] = cap.max(f64::MIN_POSITIVE);
        }
    }

    // Count 0 never carries a row, so the walk stops and one band stays uncapped.
    let mut cut = l + 1;
    let mut tail = 0.0;

    for m in (0..=l).rev() {
        let cap = caps[m as usize];

        if !cap.is_finite() || tail + cap > TAIL_FLOOR {
            break;
        }

        tail += cap;
        cut = m;
    }

    let mut out = Vec::new();

    for m in 0..cut {
        let (at, len) = perch(l, m);
        out.push((at, len, caps[m as usize]));
    }

    if cut <= l {
        let (at, _) = perch(l, cut);
        out.push((at, 2 * (l as usize) - at, tail));
    }

    out
}

/// `sum_edges phi(wu x_i, wv x_j) / (l-1)` with its gradient in `g`; `None` at a vanishing coordinate.
/// Never truncate there: a finite slope for an infinite one understates Eve's information.
fn joint_slope(
    l: u64,
    edges: &[(usize, f64, usize, f64)],
    x: &[f64],
    g: &mut [f64],
) -> Option<f64> {
    let span = (l - 1) as f64;
    let mut acc = 0.0;

    for gi in g.iter_mut() {
        *gi = 0.0;
    }

    for (i, wu, j, wv) in edges.iter() {
        let u = wu * x[*i];
        let v = wv * x[*j];

        if !(u > 0.0) || !(v > 0.0) {
            return None;
        }

        let s = u + v;
        acc += phi(u, v);
        g[*i] += wu * (s / u).log2();
        g[*j] += wv * (s / v).log2();
    }

    for gi in g.iter_mut() {
        *gi /= span;
    }

    let out = acc / span;

    if out.is_finite() && g.iter().all(|gi| gi.is_finite()) {
        Some(out)
    } else {
        None
    }
}

/// Scale positive weights onto Eq. (60)'s polytope, pinning each band over its cap and spreading
/// the rest. `false` means the polytope is empty, not that a step failed.
fn joint_fill(bands: &[(usize, usize, f64)], x: &mut [f64]) -> bool {
    let mut pinned = vec![false; bands.len()];
    let mut mass = 0.0;

    for xi in x.iter_mut() {
        *xi = xi.max(f64::MIN_POSITIVE);
        mass += *xi;
    }

    if !(mass > 0.0) || !mass.is_finite() {
        return false;
    }

    for xi in x.iter_mut() {
        *xi /= mass;
    }

    for _ in 0..=bands.len() {
        let mut fixed = 0.0;
        let mut loose = 0.0;
        let mut found = false;

        for (b, (at, len, cap)) in bands.iter().enumerate() {
            let mut sum = 0.0;

            for xi in x[*at..at + len].iter() {
                sum += *xi;
            }

            if !pinned[b] && sum > *cap {
                pinned[b] = true;
                found = true;
            }

            if pinned[b] {
                fixed += cap;
            } else {
                loose += sum;
            }
        }

        if !found {
            return true;
        }

        let rest = 1.0 - fixed;

        if !(rest > 0.0) || !(loose > 0.0) {
            return false;
        }

        for (b, (at, len, cap)) in bands.iter().enumerate() {
            let mut sum = 0.0;

            for xi in x[*at..at + len].iter() {
                sum += *xi;
            }

            let scale = if pinned[b] { cap / sum } else { rest / loose };

            for xi in x[*at..at + len].iter_mut() {
                *xi *= scale;
            }
        }
    }

    true
}

/// Max of the plane `level + g . x` over Eq. (60)'s polytope, by its dual
/// `min_lam lam + sum_B cap_B max(0, W_B - lam)`, `W_B` the band's largest gradient entry.
fn joint_tangent(bands: &[(usize, usize, f64)], g: &[f64], level: f64) -> Option<f64> {
    let mut roof = f64::NEG_INFINITY;
    let mut held: Vec<(f64, f64)> = Vec::new();

    for (at, len, cap) in bands.iter() {
        let mut top = f64::NEG_INFINITY;

        for gi in g[*at..at + len].iter() {
            top = top.max(*gi);
        }

        if cap.is_finite() {
            held.push((top, *cap));
        } else {
            roof = roof.max(top);
        }
    }

    if !roof.is_finite() {
        return None;
    }

    held.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut best = f64::INFINITY;
    let mut share = 0.0;
    let mut moment = 0.0;

    // A tie contributes `cap * (W - lam) = 0`: no second pass over equal entries.
    for (top, cap) in held.iter() {
        if !(*top > roof) {
            break;
        }

        best = best.min(top + moment - share * top);
        share += cap;
        moment += cap * top;
    }

    best = best.min(roof + moment - share * roof);

    if best.is_finite() {
        Some(best + level)
    } else {
        None
    }
}

/// `rrdps_tag` at its best threshold on Eq. (60)'s binomial source: the closed form a certified
/// `rrdps_entropy` must beat.
fn joint_plain(l: u64, p_src: f64, q: f64) -> PyResult<f64> {
    let mass = binomial(l, p_src);
    let mut tail = 1.0f64;
    let mut best = 1.0f64;

    for nu in 0..=l {
        tail = (tail - mass[nu as usize]).clamp(0.0, 1.0);
        best = best.min(rrdps_tag(q, tail, h2(phase_n(l, nu)))?);
    }

    Ok(best)
}

fn cornered(l: u64) -> PyErr {
    PyValueError::new_err(format!(
        "the ascent for l = {l} reached a weight vector with a vanishing coordinate, where \
         Eq. (60)'s gradient is infinite. Use rrdps_tag"
    ))
}

/// `rrdps_polytope` without argument checks.
fn joint_bound(
    l: u64,
    p_src: f64,
    q: f64,
    reltol: f64,
) -> PyResult<(f64, f64, f64, f64, usize, usize)> {
    // `p_src = 0` pins everything on count 0, where `c(0, 0) = 0`; the ascent cannot run it.
    if !(p_src > 0.0) {
        return Ok((0.0, 0.0, 0.0, joint_plain(l, p_src, q)?, 0, 0));
    }

    let dim = 2 * (l as usize);
    let edges = joint_edges(l);
    let bands = joint_bands(l, p_src, q);
    let mut x = vec![1.0f64; dim];
    let mut g = vec![0.0f64; dim];

    if !joint_fill(&bands, &mut x) {
        return Err(PyValueError::new_err(format!(
            "the source rows for l = {l}, p_src = {p_src}, q = {q} leave Eq. (60) nothing \
             to normalise: the caps that bind take up the whole distribution and the counts \
             carrying no row are left none of it"
        )));
    }

    let mut witness = f64::NEG_INFINITY;
    let mut tangent = f64::INFINITY;
    let mut steps = 0usize;
    let mut closed = false;

    for _ in 0..ASCENT_MAX {
        let val = joint_slope(l, &edges, &x, &mut g).ok_or_else(|| cornered(l))?;
        let mut dot = 0.0;

        for (xi, gi) in x.iter().zip(g.iter()) {
            dot += xi * gi;
        }

        witness = witness.max(val);
        tangent = tangent.min(joint_tangent(&bands, &g, val - dot).ok_or_else(|| cornered(l))?);

        if tangent - witness <= ASCENT_TOL * witness.abs().max(1.0) {
            closed = true;
            break;
        }

        for (xi, gi) in x.iter_mut().zip(g.iter()) {
            *xi *= gi;
        }

        if !joint_fill(&bands, &mut x) {
            break;
        }

        steps += 1;
    }

    if !closed {
        return Err(PyValueError::new_err(format!(
            "the ascent for l = {l}, p_src = {p_src}, q = {q} did not close its bracket in \
             {ASCENT_MAX} steps, so nothing was certified. Use rrdps_tag"
        )));
    }

    let mut rows = Vec::new();
    let mut rhs = Vec::new();

    for (at, len, cap) in bands.iter() {
        if !cap.is_finite() {
            continue;
        }

        let mut row = vec![0.0f64; dim];

        for entry in row[*at..at + len].iter_mut() {
            *entry = 1.0;
        }

        rows.extend_from_slice(&row);
        rhs.push(*cap);
    }

    let mut cuts = Vec::with_capacity(CUT_KEEP * dim);
    let mut levels = Vec::with_capacity(CUT_KEEP);

    for _ in 0..CUT_KEEP {
        let val = joint_slope(l, &edges, &x, &mut g).ok_or_else(|| cornered(l))?;
        let mut dot = 0.0;

        for (xi, gi) in x.iter().zip(g.iter()) {
            dot += xi * gi;
        }

        let level = val - dot;
        witness = witness.max(val);
        tangent = tangent.min(joint_tangent(&bands, &g, level).ok_or_else(|| cornered(l))?);
        cuts.extend_from_slice(&g);
        levels.push(level);

        for (xi, gi) in x.iter_mut().zip(g.iter()) {
            *xi *= gi;
        }

        if !joint_fill(&bands, &mut x) {
            break;
        }

        steps += 1;
    }

    let facets = Facets {
        eq: &[],
        eqb: &[],
        le: &rows,
        leb: &rhs,
    };
    let (raw, gap, iters, status) = polytope_cap(dim, &cuts, &levels, &facets, LP_GAPTOL)?;
    let slack = gap.abs() + 1e-9 * tangent.abs().max(1.0);

    if !(raw <= tangent + slack) {
        return Err(PyValueError::new_err(format!(
            "the interior-point path did not resolve l = {l}, p_src = {p_src}, q = {q}: it \
             stopped after {iters} iterations with status \"{status}\" at a certified \
             {raw}, above the {tangent} the tangent cap on the same cuts already gives. The \
             value is a valid bound and a useless one; nothing tighter than one plane was \
             proved here"
        )));
    }

    let bound = raw.min(tangent).clamp(0.0, 1.0);

    // A bound under `witness` fails INSECURE, seen at `p_src` < ~1e-30; `reltol` is vacuous there.
    if bound < witness - 1e-9 * witness.abs() {
        return Err(PyValueError::new_err(format!(
            "the certificate for l = {l}, p_src = {p_src}, q = {q} came back at {bound}, \
             below the {witness} Eq. (60) attains, so it bounds nothing. Use rrdps_tag"
        )));
    }

    let plain = joint_plain(l, p_src, q)?;

    if bound > plain + 1e-9 * plain.max(1.0) {
        return Err(PyValueError::new_err(format!(
            "the certified {bound} for l = {l}, p_src = {p_src}, q = {q} sits above the \
             {plain} rrdps_tag gives on the same quantity. Use rrdps_tag"
        )));
    }

    if !(bound - witness <= reltol * witness.abs().max(1.0)) {
        return Err(PyValueError::new_err(format!(
            "the bracket for l = {l}, p_src = {p_src}, q = {q} closed only to {} bits per \
             sifted bit, above reltol = {reltol} (witness {witness}, bound {bound}). The bound \
             is valid; pass a larger reltol to accept it",
            bound - witness
        )));
    }

    Ok((bound, witness, tangent, plain, steps, iters))
}

/// Matsuura's Eq. (60) as `(bound, witness, tangent, plain, steps, iters)`: `l` pulses, odd photon
/// number with probability `p_src` per pulse, sifted rate `q` per emitted packet.
///
/// Fields as `rrdps_simplex`'s; `bound` is what `rrdps_entropy` returns, `plain` the tagging closed
/// form it must beat.
#[pyfunction]
#[pyo3(signature = (l, p_src, q, reltol=1e-9))]
pub(crate) fn rrdps_polytope(
    l: u64,
    p_src: f64,
    q: f64,
    reltol: f64,
) -> PyResult<(f64, f64, f64, f64, usize, usize)> {
    check_train(l)?;
    check_prob("p_src", p_src)?;
    check_unit("q", q)?;
    check_pos("reltol", reltol)?;
    check_packet(l)?;

    joint_bound(l, p_src, q, reltol)
}

fn check_packet(l: u64) -> PyResult<()> {
    if l > POLYTOPE_MAX {
        return Err(PyValueError::new_err(format!(
            "l must not exceed {POLYTOPE_MAX} pulses for the refined bound, got {l}: its \
             programme carries one source row per photon count against {ROW_MAX} rows. \
             Use rrdps_leak, rrdps_collective or rrdps_tag, which hold at every length"
        )));
    }

    Ok(())
}

/// Privacy amplification per sifted bit, Matsuura, Sasaki & Koashi, Phys. Rev. A 99, 042303
/// (2019), arXiv:1812.10916, Eq. (60): the largest `H(X|Y)` consistent with the source and `q`.
/// A `cost` for `rrdps_rate`; `rrdps_polytope` shows the working.
///
/// Not comparable with `rrdps_collective`, which fixes the photon number; this maximises over it.
#[pyfunction]
pub(crate) fn rrdps_entropy(l: u64, p_src: f64, q: f64) -> PyResult<f64> {
    Ok(rrdps_polytope(l, p_src, q, 1e-9)?.0)
}

/// Largest bit error rate leaving key at `cost`: the root of `f_ec h2(e) = 1 - cost` on `[0, 1/2]`.
/// 1/2 when `1 - cost >= f_ec`, 0 when `cost >= 1`.
#[pyfunction]
pub(crate) fn rrdps_tolerance(cost: f64, f_ec: f64) -> PyResult<f64> {
    check_prob("cost", cost)?;
    check_fec("f_ec", f_ec)?;

    let want = (1.0 - cost) / f_ec;
    if want <= 0.0 {
        return Ok(0.0);
    }

    if want >= 1.0 {
        return Ok(0.5);
    }

    let (lo, hi) = bisect(0.0, 0.5, STEPS, |e| h2(e) < want);

    Ok(0.5 * (lo + hi))
}

/// `(q, e_bit)`, sifted rate per emitted packet and bit error rate, from the receiver model of
/// Yin et al., Nature Communications 9, 457 (2018), Methods. `mu` is photons per pulse at Alice,
/// `eta` modulator-to-detector transmittance, `dark` per pulse per detector.
///
/// `e_bit` reduces to `(eta mu e_mis + dark) / (eta mu + 2 dark)` at every `l`; the sum stays so a
/// test of that cancellation reads shipped arithmetic.
#[pyfunction]
pub(crate) fn rrdps_counts(
    l: u64,
    mu: f64,
    eta: f64,
    dark: f64,
    e_mis: f64,
) -> PyResult<(f64, f64)> {
    check_train(l)?;
    check_pos("mu", mu)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_err("e_mis", e_mis)?;

    let flux = eta * mu;
    let mut gain = 0.0;
    let mut errs = 0.0;
    for r in 1..l {
        let k = (l - r) as f64;
        let live = (1.0 - dark).powi((2 * (l - r) - 1) as i32);
        let seen = (-k * flux).exp();
        let w = live * seen * k;
        gain += w * (flux + 2.0 * dark);
        errs += w * (flux * e_mis + dark);
    }

    // Output invariant, unreachable (measured max 0.48); an over-one `q` would reach `rrdps_rate`.
    let span = (l - 1) as f64;
    let q = gain / span;
    if !(q.is_finite() && (0.0..=1.0).contains(&q)) {
        return Err(PyValueError::new_err(format!(
            "the sifted rate came out {q}, which is not a probability: this receiver \
             model linearises the dark-count term, so a packet that long at that dark \
             rate has left the model's domain. Shorten l or lower dark"
        )));
    }

    Ok((q, if gain > 0.0 { errs / gain } else { 0.0 }))
}

/// `(l, nu_th, rate)` maximising the key rate in bits per pulse over `l` in `2..=l_max` and `nu_th`
/// in `1..=(l-1)/2`, at `rrdps_tag`'s cost; `(0, 0, 0.0)` when none yields key. `mu` is not swept.
#[pyfunction]
pub(crate) fn rrdps_optimum(
    l_max: u64,
    mu: f64,
    eta: f64,
    dark: f64,
    e_mis: f64,
    f_ec: f64,
) -> PyResult<(u64, u64, f64)> {
    check_train(l_max)?;
    check_pos("mu", mu)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_err("e_mis", e_mis)?;
    check_fec("f_ec", f_ec)?;

    let mut best = (0u64, 0u64, 0.0f64);
    for l in 2..=l_max {
        let (q, e_bit) = rrdps_counts(l, mu, eta, dark, e_mis)?;
        let leak = 1.0 - f_ec * h2(e_bit);
        if !(q > 0.0) || leak <= 0.0 {
            continue;
        }

        // Poisson mass walked forward with nu_th: linear in the threshold.
        let m = (l as f64) * mu;
        let mut term = (-m).exp();
        let mut cdf = term;
        for nu in 1..=(l - 1).div_ceil(2) {
            term *= m / (nu as f64);
            cdf += term;
            let src = (1.0 - cdf).clamp(0.0, 1.0);
            if src >= q {
                continue;
            }

            let cost = (src + (q - src) * h2(phase_n(l, nu))) / q;
            let rate = q * (leak - cost) / (l as f64);
            if rate > best.2 {
                best = (l, nu, rate);
            }
        }
    }

    Ok(best)
}

// Finite key: Takesue, Sasaki, Tamaki & Koashi, "Experimental quantum key distribution without
// monitoring signal disturbance", Nature Photonics 9, 827 (2015), arXiv:1505.07914,
// DOI 10.1038/nphoton.2015.173, Methods, "Finite-key security analysis".
//
// NEAR MISS: arXiv:1505.07884 is Wang, Yin, Chen, He, Song, Li, Zhang, Zhou, Guo & Han, Nature
// Photonics 9, 832 (2015), DOI 10.1038/nphoton.2015.209, the OTHER round-robin experiment under a
// near-identical title. CHECK THE AUTHORS. Nature 509, 475 has no preprint.
//
//   eps_2  over-threshold packets in Alice's source, binomial in N_em at `rrdps_src`.
//   eps_3  at most nu_th/(l-1) phase errors per sifted bit, WHATEVER Eve did.
//   eps_1  double clicks: Bob's photon number, the one observable that enters.
//
// `e_bit` enters only through the error-correction leakage, with no confidence interval.
// The security parameter is a MAX over two branches, not a sum: never divide `d` by a term count.
// The eps_1 branch is RECEIVER-SPECIFIC: SYK's number-resolving receiver needs none, and
// `rrdps_counts`'s variable-delay receiver has no double-click law here.

/// Least double-click probability of a multiphoton packet on Takesue's passive four-interferometer receiver.
const P_TWO: f64 = 0.125;

/// Bisection steps for the two tail inversions.
const TAIL_STEPS: usize = 200;

/// Smallest `k` with the upper binomial tail `f(k; n, p) <= eps`, by the Chernoff-Kullback-Leibler
/// bound: ceilinged, never below `n p` (valid above the mean only), `n` when nothing passes.
/// A tail of a KNOWN binomial, not a confidence interval on an unknown rate.
#[pyfunction]
pub(crate) fn rrdps_above(n: f64, p: f64, eps: f64) -> PyResult<f64> {
    check_nonneg("n", n)?;
    check_prob("p", p)?;
    check_eps("eps", eps)?;

    let want = (1.0 / eps).log2();

    if !(n > 0.0) {
        return Ok(0.0);
    }

    // p = 0: the divergence is infinite for k >= 1 and the bisection returns 1.
    if p >= 1.0 || n * (1.0 / p).log2() < want {
        return Ok(n);
    }

    let (_, hi) = bisect(p, 1.0, TAIL_STEPS, |k| n * kl_bits(k, p) < want);

    Ok((n * hi).ceil().min(n))
}

/// Takesue's `N_bar_mB`: least `m` with the lower binomial tail `g(n_double; m, p_two) <= eps`, by the
/// same Kullback-Leibler bound. Bounds the POPULATION at an OBSERVED count, the reverse of `rrdps_above`.
/// A constant at `n_double = 0`, independent of block size.
#[pyfunction]
pub(crate) fn rrdps_tagged(n_double: f64, p_two: f64, eps: f64) -> PyResult<f64> {
    check_nonneg("n_double", n_double)?;
    check_eps("eps", eps)?;

    if !(p_two > 0.0) || p_two >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "p_two must be in (0, 1), got {p_two}: it is the least probability that a packet \
             carrying two or more photons produces a double click, and Takesue's passive \
             four-interferometer receiver gives {P_TWO}"
        )));
    }

    let want = (1.0 / eps).log2();
    let floor = n_double / p_two;
    let mut hi = floor.max(want / (1.0 / (1.0 - p_two)).log2()).max(1.0);

    while hi * kl_bits(n_double / hi, p_two) < want {
        hi *= 2.0;
    }

    let (_, hi) = bisect(floor, hi, TAIL_STEPS, |m| {
        m <= floor || m * kl_bits(n_double / m, p_two) < want
    });

    Ok(hi.ceil())
}

/// Takesue's `max(eps_1, eta_z + sqrt2 sqrt(eps_2 + eps_3 + eta_x))`, `eta_x = 2^-s_x`,
/// `eta_z = 2^-s_z`: half the trace distance between real and ideal key states.
#[pyfunction]
pub(crate) fn rrdps_secpar(
    eps1: f64,
    eps2: f64,
    eps3: f64,
    s_x: f64,
    s_z: f64,
) -> PyResult<f64> {
    check_eps("eps1", eps1)?;
    check_eps("eps2", eps2)?;
    check_eps("eps3", eps3)?;
    check_nonneg("s_x", s_x)?;
    check_nonneg("s_z", s_z)?;

    let keep = 2f64.powf(-s_z) + std::f64::consts::SQRT_2 * (eps2 + eps3 + 2f64.powf(-s_x)).sqrt();

    Ok(eps1.max(keep))
}

/// `(eps1, eps2, eps3, s_x, s_z)` at target `d`, Takesue's own assignment: `eps_1 = d`, `eta_z = d/2`,
/// `eps_2 = eps_3 = eta_x = d^2/24`, both branches landing on `d`. At `d = 2^-50`, his Methods' values.
#[pyfunction]
pub(crate) fn rrdps_split(d: f64) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_eps("d", d)?;

    let small = d * d / 24.0;

    Ok((d, small, small, (1.0 / small).log2(), (2.0 / d).log2()))
}

/// Finite-key length in BITS, Takesue's `G_f`. `tagged` is `rrdps_tagged`; `faults` is
/// `rrdps_above(n_em, e_src, eps_2)`; `phase` is `rrdps_above(n_sift - tagged - faults, nu_th/(l-1),
/// eps_3)`; `s_x`, `s_z` are hash lengths in bits.
///
/// Privacy amplification is charged on `n_sift - tagged`, error correction on all of `n_sift`:
/// swapping them raises the length. The phase-error fraction is capped at 1/2.
#[pyfunction]
pub(crate) fn rrdps_length(
    n_sift: f64,
    tagged: f64,
    faults: f64,
    phase: f64,
    e_bit: f64,
    f_ec: f64,
    s_x: f64,
    s_z: f64,
) -> PyResult<f64> {
    check_nonneg("n_sift", n_sift)?;
    check_nonneg("tagged", tagged)?;
    check_nonneg("faults", faults)?;
    check_nonneg("phase", phase)?;
    check_err("e_bit", e_bit)?;
    check_fec("f_ec", f_ec)?;
    check_nonneg("s_x", s_x)?;
    check_nonneg("s_z", s_z)?;

    let kept = n_sift - tagged;

    if !(kept > 0.0) {
        return Ok(0.0);
    }

    let share = ((faults + phase) / kept).min(0.5);
    let len = kept * (1.0 - h2(share)) - f_ec * n_sift * h2(e_bit) - s_x - s_z;

    Ok(len.max(0.0).floor())
}

/// Takesue's chain, `(length, secpar, tagged, faults, phase)`, for `n_em` emitted packets yielding
/// `n_sift` sifted bits and `n_double` double clicks, at target security parameter `d`.
///
/// `faults` is a tail over `n_em`, `phase` over `n_sift - tagged - faults`; over all of `n_sift` it
/// comes out too small. `p_two = 0.125` is Takesue's passive splitter.
#[pyfunction]
pub(crate) fn rrdps_finite(
    n_em: f64,
    n_sift: f64,
    n_double: f64,
    l: u64,
    mu: f64,
    nu_th: u64,
    e_bit: f64,
    f_ec: f64,
    d: f64,
    p_two: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_nonneg("n_em", n_em)?;
    check_nonneg("n_sift", n_sift)?;
    check_nonneg("n_double", n_double)?;
    check_err("e_bit", e_bit)?;
    check_fec("f_ec", f_ec)?;
    check_eps("d", d)?;

    if n_sift > n_em {
        return Err(PyValueError::new_err(format!(
            "n_sift must not exceed n_em (got {n_sift} > {n_em}): a packet yields at most \
             one sifted bit"
        )));
    }

    let (eps1, eps2, eps3, s_x, s_z) = rrdps_split(d)?;
    let src = rrdps_src(l, mu, nu_th)?;
    let tagged = rrdps_tagged(n_double, p_two, eps1)?;
    let faults = rrdps_above(n_em, src, eps2)?;
    let pool = (n_sift - tagged - faults).max(0.0);
    let phase = rrdps_above(pool, ((nu_th as f64) / ((l - 1) as f64)).min(1.0), eps3)?;
    let len = rrdps_length(n_sift, tagged, faults, phase, e_bit, f_ec, s_x, s_z)?;
    let secpar = rrdps_secpar(eps1, eps2, eps3, s_x, s_z)?;

    Ok((len, secpar, tagged, faults, phase))
}
