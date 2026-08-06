use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::click::gauss_mass;
use crate::fock::MAX_D;
use crate::std::{check_gate, check_nonneg, check_pos, check_unit, ln_fact};

// Photon-number-resolving detection. Two devices, one returned object: a `Povm` carrying
// {Pi_0 .. Pi_m}, diagonal in the Fock basis and complete on the truncated space. Closed
// operators, never samples.
//
// 1. MULTIPLEXED THRESHOLD ARRAY, DERIVED. N identical threshold elements of efficiency `eta` and
// per-gate dark probability `dark` behind a balanced N-port, spatial or temporal. The number j of
// bins holding a DETECTED photon is a Markov chain in the photon index, P(j -> j + 1) =
// eta * (N - j) / N; dark counts then fire each silent bin independently, k = j + Bin(N - j, dark).
// The printed inclusion-exclusion form for p(k | n) CANCELS CATASTROPHICALLY past N ~ 20. N = 1 is
// the threshold detector of src/click.rs, folding against a Poisson to `crate::std::click_p`.
//
// UNIFORM ELEMENTS ONLY: the recursion is the bins' exchangeability. Temporal multiplexing (Fitch,
// Jacobs, Pittman & Franson, Phys. Rev. A 68, 043814 (2003), quant-ph/0305193) is covered only with
// the loop's per-bin loss folded into one common eta.
//
// 2. TRUE PNR WITH FINITE RESOLUTION, a transition-edge-sensor-shaped device:
//
//   m ~ Binomial(n, eta) + Poisson(background)      detected number
//   h ~ Normal(mu(m), sigma)                        pulse height, one photon = 1
//   report k, where h falls in the bin around mu(k)
//
// mu(m) = m at sat = 0, else sat * (1 - exp(-m / sat)): resolution is finite AND number-dependent.
// Bin edges are the midpoints between neighbouring mu. THE READOUT MODEL IS PHENOMENOLOGICAL, not
// fitted to a device; sigma = 0 is the identity. The top outcome ABSORBS ("d-1 or more"), keeping
// completeness exact under a background.
//
// 3. DETECTOR DECOY. Moroder, Curty & Lutkenhaus, "Detector decoy quantum key distribution", New J.
// Phys. 11, 045008 (2009), arXiv:0811.0027 (one version only, so the equation numbers here are
// unambiguous). A variable attenuator `atten` before ONE threshold detector (`eta_det`, `dark`);
// their Eq. (17): p_vac(atten) = (1 - dark) sum_n (1 - atten*eta_det)^n p_n. `pnr_decoy` is their
// Proposition 2.1 ("Finite settings"), Eqs. (5)-(7).
//
// NOT `decoy.rs`: detector decoy's kernel is geometric in n and its p_n arbitrary. Only a signal
// already Poissonian at Bob reduces to it, with intensities t_i * mu.

/// Elements in a multiplexed array.
const MAX_N: usize = 4096;

/// Diagonal entries a single POVM may carry, outcomes x cutoff.
const MAX_CELLS: usize = 2_097_152;

/// Poisson mass allowed past the cutoff before `Povm::coherent` refuses.
const TAIL_TOL: f64 = 1e-15;

/// Slack on an input distribution's total, so a rounded histogram is accepted.
const SUM_TOL: f64 = 1e-9;

/// N(mu, sd) cumulative at `x`, clamped into [0, 1].
fn norm_cdf(x: f64, mu: f64, sd: f64) -> f64 {
    let d = x - mu;
    let half = if d >= 0.0 {
        gauss_mass(0.0, d, sd)
    } else {
        -gauss_mass(d, 0.0, sd)
    };

    (0.5 + half).clamp(0.0, 1.0)
}

/// Binomial pmf over 0..=n at probability `eta`. eta = 1 is the delta at n; the
/// log form would give 0 * ln(0) = NaN there.
fn binom_pmf(n: usize, eta: f64, lnf: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n + 1];
    if eta >= 1.0 {
        out[n] = 1.0;

        return out;
    }

    let (le, lq) = (eta.ln(), (1.0 - eta).ln());
    for (i, o) in out.iter_mut().enumerate() {
        *o = (lnf[n] - lnf[i] - lnf[n - i] + (i as f64) * le + ((n - i) as f64) * lq).exp();
    }

    out
}

/// Poisson pmf over 0..len-1 at mean `mean`; mean = 0 is the delta at 0.
fn pois_pmf(mean: f64, len: usize, lnf: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; len];
    if mean <= 0.0 {
        out[0] = 1.0;

        return out;
    }

    let lm = mean.ln();
    for (j, o) in out.iter_mut().enumerate() {
        *o = (-mean + (j as f64) * lm - lnf[j]).exp();
    }

    out
}

/// Readout signal for `m` detected photons, in units of one photon's step.
fn readout(m: usize, sat: f64) -> f64 {
    let x = m as f64;
    if sat <= 0.0 {
        return x;
    }

    sat * (1.0 - (-x / sat).exp())
}

/// Confusion matrix C[k * d + m] = P(report k | m detected), d = top + 1. The
/// cumulative is forced monotone in k, so each column telescopes to exactly 1.
fn confuse(top: usize, sigma: f64, sat: f64) -> Vec<f64> {
    let d = top + 1;
    let mut out = vec![0.0; d * d];
    if sigma <= 0.0 {
        for m in 0..d {
            out[m * d + m] = 1.0;
        }

        return out;
    }

    let mu: Vec<f64> = (0..d).map(|m| readout(m, sat)).collect();
    let edge: Vec<f64> = (0..top).map(|k| 0.5 * (mu[k] + mu[k + 1])).collect();
    for m in 0..d {
        let mut prev = 0.0;
        for k in 0..d {
            let cum = if k + 1 == d {
                1.0
            } else {
                norm_cdf(edge[k], mu[m], sigma)
            }
            .max(prev);
            out[k * d + m] = cum - prev;
            prev = cum;
        }
    }

    out
}

/// P(m detected | n incident) over m = 0..d-1, the top index absorbing, so the
/// row sums to 1 whatever the background puts above the cutoff.
fn detected(n: usize, d: usize, eta: f64, bg: f64, lnf: &[f64]) -> Vec<f64> {
    let bin = binom_pmf(n, eta, lnf);
    let mut out = vec![0.0; d];
    let top = d - 1;
    if bg <= 0.0 {
        for (i, b) in bin.iter().enumerate().take(top) {
            out[i] = *b;
        }
    } else {
        let pois = pois_pmf(bg, d, lnf);
        for (i, b) in bin.iter().enumerate().take(top) {
            for (j, p) in pois.iter().enumerate().take(top - i) {
                out[i + j] += b * p;
            }
        }
    }

    let head: f64 = out.iter().take(top).sum();
    out[top] = (1.0 - head).max(0.0);

    out
}

/// Every entry a probability and the total at most one; both distribution entry points run it.
fn check_probs(probs: &[f64]) -> PyResult<()> {
    let mut total = 0.0;
    for (n, p) in probs.iter().enumerate() {
        if !(p.is_finite() && *p >= 0.0) {
            return Err(PyValueError::new_err(format!(
                "probs[{n}] is {p}, which is not a probability"
            )));
        }

        total += p;
    }

    if total > 1.0 + SUM_TOL {
        return Err(PyValueError::new_err(format!(
            "probs sums to {total}, which is more than one photon-number \
             distribution's worth"
        )));
    }

    Ok(())
}

/// Transpose an [n * outs + k] buffer into the [k * cutoff + n] layout a POVM
/// diagonal is read in.
fn transpose(buf: &[f64], cutoff: usize, outs: usize) -> Vec<f64> {
    let mut out = vec![0.0; outs * cutoff];
    for n in 0..cutoff {
        for k in 0..outs {
            out[k * cutoff + n] = buf[n * outs + k];
        }
    }

    out
}

/// A photon-number-resolving measurement as its POVM. Every element is diagonal
/// in the Fock basis, so the whole measurement is one (outcomes x cutoff) table
/// of probabilities and `element` inflates one row into a square matrix.
#[pyclass]
pub(crate) struct Povm {
    diag: Vec<f64>,
    cutoff: usize,
    outs: usize,
    label: String,
}

#[pymethods]
impl Povm {
    /// Number of outcomes, i.e. the highest reportable count plus one.
    #[getter]
    fn outcomes(&self) -> usize {
        self.outs
    }

    /// Fock levels the elements are defined on: levels 0..cutoff-1.
    #[getter]
    fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// How the measurement was built.
    #[getter]
    fn label(&self) -> String {
        self.label.to_string()
    }

    /// The whole POVM as an (outcomes, cutoff) array: row k is <n|Pi_k|n>.
    fn diagonal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let a = ArrayView2::from_shape((self.outs, self.cutoff), &self.diag)
            .map_err(|e| PyValueError::new_err(format!("diagonal is not rectangular: {e}")))?;

        Ok(a.to_pyarray(py))
    }

    /// One element Pi_k as a (cutoff, cutoff) matrix. Diagonal by construction,
    /// so its eigenvalues are its diagonal and positivity is readable off it.
    fn element<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if k >= self.outs {
            return Err(PyValueError::new_err(format!(
                "outcome {k} is past the {} this POVM carries",
                self.outs
            )));
        }

        let mut m = Array2::<f64>::zeros((self.cutoff, self.cutoff));
        for n in 0..self.cutoff {
            m[[n, n]] = self.diag[k * self.cutoff + n];
        }

        Ok(m.into_pyarray(py))
    }

    /// max_n |sum_k <n|Pi_k|n> - 1|: how far sum_k Pi_k is from the identity.
    fn defect(&self) -> f64 {
        let mut worst: f64 = 0.0;
        for n in 0..self.cutoff {
            let mut s = 0.0;
            for k in 0..self.outs {
                s += self.diag[k * self.cutoff + n];
            }

            worst = worst.max((s - 1.0).abs());
        }

        worst
    }

    /// Smallest diagonal entry across every element: the positivity witness.
    fn floor(&self) -> f64 {
        self.diag.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Outcome distribution for a state whose photon-number distribution is
    /// `probs`, which must be given on exactly this cutoff.
    fn fold<'py>(
        &self,
        py: Python<'py>,
        probs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if probs.len() != self.cutoff {
            return Err(PyValueError::new_err(format!(
                "probs carries {} levels but this POVM is defined on {}: a photon-number \
                 distribution must be given on the same truncated space",
                probs.len(),
                self.cutoff
            )));
        }

        check_probs(&probs)?;

        Ok(self.apply(&probs).into_pyarray(py))
    }

    /// Outcome distribution for a coherent (or phase-randomised coherent -- the
    /// elements are diagonal, so the two give the same counts) state of mean
    /// photon number `mu`. Refuses when the cutoff would drop more than
    /// TAIL_TOL of the input.
    fn coherent<'py>(&self, py: Python<'py>, mu: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_nonneg("mu", mu)?;
        let lnf = ln_fact(self.cutoff);
        let probs = pois_pmf(mu, self.cutoff, &lnf);
        let tail = 1.0 - probs.iter().sum::<f64>();
        if tail > TAIL_TOL {
            return Err(PyValueError::new_err(format!(
                "a coherent state of mu = {mu} puts {tail:e} of its photon-number mass above \
                 level {}, past this POVM's cutoff of {}: raise the cutoff",
                self.cutoff - 1,
                self.cutoff
            )));
        }

        Ok(self.apply(&probs).into_pyarray(py))
    }
}

impl Povm {
    /// sum_n probs[n] <n|Pi_k|n>, the fold both public entry points perform.
    fn apply(&self, probs: &[f64]) -> Vec<f64> {
        (0..self.outs)
            .map(|k| {
                let row = &self.diag[k * self.cutoff..(k + 1) * self.cutoff];

                row.iter().zip(probs.iter()).map(|(a, b)| a * b).sum()
            })
            .collect()
    }
}

/// Shared cutoff and cell guards; `outs` is the outcome count the caller will
/// build.
fn check_shape(cutoff: usize, outs: usize) -> PyResult<()> {
    if !(1..=MAX_D).contains(&cutoff) {
        return Err(PyValueError::new_err(format!(
            "cutoff must be in 1..={MAX_D}, got {cutoff}"
        )));
    }

    if outs * cutoff > MAX_CELLS {
        return Err(PyValueError::new_err(format!(
            "{outs} outcomes on a cutoff of {cutoff} is {} diagonal entries, \
             past the {MAX_CELLS} a single POVM carries",
            outs * cutoff
        )));
    }

    Ok(())
}

/// The array POVM's diagonal in [k * cutoff + n] layout.
fn array_diag(cutoff: usize, elems: usize, eta: f64, dark: f64) -> Vec<f64> {
    let outs = elems + 1;
    let jmax = elems.min(cutoff - 1);
    let width = elems as f64;

    // One row of `stride` per Fock level; j <= photon index < cutoff: the jmax cap drops no mass.
    let stride = jmax + 1;
    let mut fire = vec![0.0; cutoff * stride];
    fire[0] = 1.0;
    for n in 1..cutoff {
        let (done, rest) = fire.split_at_mut(n * stride);
        let src = &done[(n - 1) * stride..];
        for (j, p) in src.iter().enumerate() {
            let up = eta * ((elems - j) as f64) / width;
            rest[j] += p * (1.0 - up);
            if j < jmax {
                rest[j + 1] += p * up;
            }
        }
    }

    let mut buf = vec![0.0; cutoff * outs];
    if dark <= 0.0 {
        for (n, out) in buf.chunks_mut(outs).enumerate() {
            out[..stride].copy_from_slice(&fire[n * stride..(n + 1) * stride]);
        }

        return transpose(&buf, cutoff, outs);
    }

    let lnf = ln_fact(elems);
    let (ld, lq) = (dark.ln(), (1.0 - dark).ln());
    buf.par_chunks_mut(outs).enumerate().for_each(|(n, out)| {
        for (k, o) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for j in 0..=jmax.min(k) {
                let free = elems - j;
                let hits = k - j;
                if hits > free {
                    continue;
                }

                let w = lnf[free] - lnf[hits] - lnf[free - hits]
                    + (hits as f64) * ld
                    + ((elems - k) as f64) * lq;
                s += fire[n * stride + j] * w.exp();
            }

            *o = s;
        }
    });

    transpose(&buf, cutoff, outs)
}

/// A multiplexed threshold array as a POVM on levels 0..`cutoff`-1: `elements`
/// identical threshold detectors behind a balanced splitter, each of efficiency
/// `eta` and per-gate dark probability `dark`. Outcomes are 0..`elements`, the
/// number of elements that fired. `elements` = 1 is the threshold detector of
/// `click.rs`.
#[pyfunction]
pub(crate) fn pnr_array(
    py: Python<'_>,
    cutoff: usize,
    elements: usize,
    eta: f64,
    dark: f64,
) -> PyResult<Povm> {
    if !(1..=MAX_N).contains(&elements) {
        return Err(PyValueError::new_err(format!(
            "elements must be in 1..={MAX_N}, got {elements}: an array of no \
             elements measures nothing"
        )));
    }

    check_shape(cutoff, elements + 1)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    let diag = py.detach(|| array_diag(cutoff, elements, eta, dark));

    Ok(Povm {
        diag,
        cutoff,
        outs: elements + 1,
        label: format!("array(elements={elements}, eta={eta}, dark={dark})"),
    })
}

/// The TES POVM's diagonal in [k * cutoff + n] layout.
fn tes_diag(cutoff: usize, eta: f64, bg: f64, sigma: f64, sat: f64) -> Vec<f64> {
    let c = confuse(cutoff - 1, sigma, sat);
    let lnf = ln_fact(cutoff);
    let mut buf = vec![0.0; cutoff * cutoff];
    buf.par_chunks_mut(cutoff).enumerate().for_each(|(n, out)| {
        let b = detected(n, cutoff, eta, bg, &lnf);
        for (k, o) in out.iter_mut().enumerate() {
            let row = &c[k * cutoff..(k + 1) * cutoff];

            *o = row.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        }
    });

    transpose(&buf, cutoff, cutoff)
}

/// A number-resolving detector with finite resolution, as a POVM on levels
/// 0..`cutoff`-1 with the same number of outcomes. `eta` thins the incident
/// number, `background` is the mean spurious count per gate (a COUNT, not the
/// per-gate probability `pnr_array` takes), `sigma` is the readout width in
/// units of one photon's step and `sat` the saturation scale (0 = linear
/// response). The top outcome absorbs: it reads "cutoff-1 or more".
#[pyfunction]
#[pyo3(signature = (cutoff, eta, background, sigma, sat = 0.0))]
pub(crate) fn pnr_tes(
    py: Python<'_>,
    cutoff: usize,
    eta: f64,
    background: f64,
    sigma: f64,
    sat: f64,
) -> PyResult<Povm> {
    check_shape(cutoff, cutoff)?;
    check_unit("eta", eta)?;
    check_nonneg("background", background)?;
    check_nonneg("sigma", sigma)?;
    check_nonneg("sat", sat)?;

    let diag = py.detach(|| tes_diag(cutoff, eta, background, sigma, sat));

    Ok(Povm {
        diag,
        cutoff,
        outs: cutoff,
        label: format!("tes(eta={eta}, background={background}, sigma={sigma}, sat={sat})"),
    })
}

/// The readout confusion matrix alone, as a (`top`+1, `top`+1) array with
/// C[k, m] = P(report k | m detected). Columns sum to 1; the top row absorbs.
#[pyfunction]
#[pyo3(signature = (top, sigma, sat = 0.0))]
pub(crate) fn pnr_confuse(
    py: Python<'_>,
    top: usize,
    sigma: f64,
    sat: f64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    if top >= MAX_D {
        return Err(PyValueError::new_err(format!(
            "top must be below {MAX_D}, got {top}"
        )));
    }

    check_nonneg("sigma", sigma)?;
    check_nonneg("sat", sat)?;

    let d = top + 1;
    let a = Array2::from_shape_vec((d, d), confuse(top, sigma, sat))
        .map_err(|e| PyValueError::new_err(format!("confusion is not square: {e}")))?;

    Ok(a.into_pyarray(py))
}

/// Outcome distribution of a multiplexed array for a coherent state of mean
/// photon number `mu`, in closed form: a balanced splitter leaves each bin
/// Poisson(mu / elements) and independent, so each fires with probability
/// 1 - (1-dark) exp(-eta*mu/elements) and the count is binomial. Carries no Fock
/// truncation, so it is the arbiter `Povm::coherent` is checked against.
#[pyfunction]
pub(crate) fn pnr_coherent(
    py: Python<'_>,
    elements: usize,
    eta: f64,
    dark: f64,
    mu: f64,
) -> PyResult<Bound<'_, PyArray1<f64>>> {
    if !(1..=MAX_N).contains(&elements) {
        return Err(PyValueError::new_err(format!(
            "elements must be in 1..={MAX_N}, got {elements}"
        )));
    }

    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_nonneg("mu", mu)?;

    let c = 1.0 - (1.0 - dark) * (-eta * mu / (elements as f64)).exp();
    let lnf = ln_fact(elements);

    Ok(binom_pmf(elements, c, &lnf).into_pyarray(py))
}

/// Probability that `n` photons land in `elements` distinct bins,
/// prod_{j<n} (1 - j/elements): the array's fidelity to true PNR at unit
/// efficiency. 0 once n exceeds the element count, by the pigeonhole.
#[pyfunction]
pub(crate) fn pnr_distinct(elements: usize, n: usize) -> PyResult<f64> {
    if !(1..=MAX_N).contains(&elements) {
        return Err(PyValueError::new_err(format!(
            "elements must be in 1..={MAX_N}, got {elements}"
        )));
    }

    let width = elements as f64;

    Ok((0..n)
        .map(|j| 1.0 - (j as f64) / width)
        .map(|x| x.max(0.0))
        .product())
}

/// No-click probability behind an attenuator, Moroder Eq. (17):
/// `(1 - dark) sum_n (1 - atten*eta)^n probs[n]`: what the receiver observes at
/// one attenuator setting.
#[pyfunction]
pub(crate) fn pnr_noclick(eta: f64, dark: f64, atten: f64, probs: Vec<f64>) -> PyResult<f64> {
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_unit("atten", atten)?;
    if probs.is_empty() {
        return Err(PyValueError::new_err(
            "probs is empty: a photon-number distribution needs at least the \
             vacuum term"
                .to_string(),
        ));
    }

    check_probs(&probs)?;

    let c = 1.0 - atten * eta;
    let sum: f64 = probs
        .iter()
        .enumerate()
        .map(|(n, p)| p * c.powi(n as i32))
        .sum();

    Ok((1.0 - dark) * sum)
}

/// Detector decoy: `(x0, l1, u1, l2, u2)` from three attenuator settings.
/// Moroder, Curty & Lutkenhaus, New J. Phys. 11, 045008 (2009), arXiv:0811.0027,
/// Proposition 2.1, Eqs. (5)-(7). `attens` are the three transmittances in the
/// order (a0, a1, a2) and `vacs` the observed no-click probabilities at them;
/// `cap` is the C of the proposition, the bound on sum_n x_n, which is 1 for a
/// photon-number distribution. `x0` is exact, the others are two-sided bounds
/// on p_1 and p_2.
///
/// REFUSES c_0 = 1 - a_0*eta != 0, i.e. anything short of unit attenuation AND unit detector
/// efficiency: Moroder's footnote to Eq. (17) gives eta_det < 1 to the channel, not to this.
#[pyfunction]
pub(crate) fn pnr_decoy(
    eta: f64,
    dark: f64,
    attens: (f64, f64, f64),
    vacs: (f64, f64, f64),
    cap: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    check_pos("cap", cap)?;
    let a = [attens.0, attens.1, attens.2];
    let v = [vacs.0, vacs.1, vacs.2];
    for (name, x) in [("atten0", a[0]), ("atten1", a[1]), ("atten2", a[2])] {
        check_unit(name, x)?;
    }

    for (name, x) in [("vac0", v[0]), ("vac1", v[1]), ("vac2", v[2])] {
        check_nonneg(name, x)?;
    }

    let c1 = 1.0 - a[1] * eta;
    let c2 = 1.0 - a[2] * eta;
    let c0 = 1.0 - a[0] * eta;
    if c0 != 0.0 {
        return Err(PyValueError::new_err(format!(
            "the first setting reaches c = 1 - atten0*eta = {c0}, not 0: Proposition 2.1 \
             reads x_0 off f(0), which needs atten0 = 1 and eta = 1 (got {eta}). Hand the \
             detector efficiency to the channel"
        )));
    }

    if !(0.0 < c1 && c1 < c2 && c2 < 1.0) {
        return Err(PyValueError::new_err(format!(
            "the two probing settings must satisfy 0 < c1 < c2 < 1, got c1 = {c1} and \
             c2 = {c2}: Proposition 2.1 converges along c1 = delta, c2 = sqrt(delta)"
        )));
    }

    let f: Vec<f64> = v.iter().map(|x| x / (1.0 - dark)).collect();
    let x0 = f[0];

    // Eq. (5). u1 comes from f(c1) >= f(0) + c1*x1; l1 additionally spends the
    // whole remaining mass at n = 2, which is the worst case for x1.
    let u1 = (f[1] - x0) / c1;
    let l1 = (f[1] - x0 * (1.0 - c1 * c1) - c1 * c1 * cap) / (c1 - c1 * c1);

    // Eqs. (6) and (7). Each spends the OTHER side's Eq. (5) bound, so l2 rides
    // u1 and u2 rides l1; feeding a bound its own side would not close.
    let l2 =
        (f[2] - x0 * (1.0 - c2 * c2 * c2) - u1 * (c2 - c2 * c2 * c2) - c2 * c2 * c2 * cap)
            / (c2 * c2 - c2 * c2 * c2);
    let u2 = (f[2] - x0 - c2 * l1) / (c2 * c2);

    Ok((
        x0,
        l1.max(0.0),
        u1.min(cap),
        l2.max(0.0),
        u2.min(cap),
    ))
}
