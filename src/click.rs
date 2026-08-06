use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

use crate::pipeline::walk;
use crate::std::{
    check_eps, check_err, check_fec, check_finite, check_gate, check_nonneg, check_pos, check_prob,
    check_unit, click_p, h2, wrap, E_CAP,
};
use crate::threefry::Stream;

// Threshold detection on a distributed-phase-reference pulse train; QBER is an OUTPUT.
// |alpha|^2 is photons: (dx^2 + dp^2)/2 in gaussian.rs, (x^2 + p^2)/4 in SNU.
// Slot k's ports carry (alpha_k +- alpha_{k-d})/2, not /sqrt(2): the stencil overlaps at every k.
// V scales the interference term alone: weak-flux error (1 - V)/2.
// Two detectors, own eta and dark, same Threefry draws: an equal pair is one detector bit for bit.
// The arrival-time response is ONE shared Smear, convolved before either threshold is read.
//
// Detector memory, three models, two run:
// (1) Hold-off: ceil(dead_time*symbol_rate) WHOLE gates, then full efficiency; the ceil reads gain
//     pessimistic, QBER untouched. `recovery` refuses a free-running ramp.
// (2) The afterpulse coin ignores the hold-off: QBER PESSIMISTIC, gain optimistic.
//     `after_shape` DISCARDS carriers released during the hold-off rather than deferring them.
//     No analytic shape ships: Ziarkash, Joshi, Stipcevic & Ursin, Sci. Rep. 8, 5076 (2018).
// (3) No SNSPD physics; eta = 0.74 gets InGaAs semantics and reads OPTIMISTIC on three counts:
//     step recovery (Kerman, Dauler, Keicher, Yang, Berggren, Gol'tsman & Voronov, Appl. Phys.
//     Lett. 88, 111116 (2006): 8.5 ns to 90% efficiency), no polarisation (up to 2x in count
//     rate, same paper), and latching is a stop, not a smaller eta.

// DPS FINITE-KEY layer at the foot of this file: a different protocol from the train above.
// Never compose them. Mizutani, Takeuchi & Tamaki, "Finite-key security analysis of
// differential-phase-shift quantum key distribution", Phys. Rev. Research 5, 023132 (2023),
// arXiv:2301.09844 (MTT23) -- Corollary 1's Eq. (24) on Theorem 3's Eq. (42), by Koashi's
// complementarity, over three-pulse blocks on two PHOTON-NUMBER-RESOLVING detectors.
//
// Asymptotic proof: Mizutani, Sasaki, Takeuchi, Tamaki & Koashi, "Quantum key distribution with
// simply characterized light sources", npj Quantum Information 5, 87 (2019). Kato, "Concentration
// inequality using unconfirmed knowledge", arXiv:2002.04357v2 (2020), via Eqs. (48) and (49).
// Not to be merged with the 2017 complementarity proof, a fourth author list: Mizutani, Sasaki,
// Kato, Takeuchi & Tamaki, Quantum Sci. Technol. 3, 014003 (2017).

/// Threefry stage ids for the click family, clear of the CV pipeline's 1..5.
const KEYING: u32 = 20;

const GATE: u32 = 21;

/// Squashing coin: own stream, indexed by slot, so it is position-pure.
const SQUASH: u32 = 22;

/// Parallel grain. Work handout only: no summation tree here to reshape.
const GRAIN: usize = 4096;

/// FWHM to standard deviation, 2*sqrt(2*ln 2); the API takes FWHM.
const FWHM_SD: f64 = 2.354_820_045_030_949_3;

/// Gaussian truncation in standard deviations: 1.1e-19 of the mass lies past it on each side.
const CORE_CUT: f64 = 9.0;

/// Response mass allowed outside the modelled span; it would be bin leak.
const SPAN_TOL: f64 = 1e-12;

/// Slots either side the response may reach before refusal (~9 periods).
const SPAN_MAX: usize = 256;

/// Longest afterpulse curve, in gates: 65.5 us at 1 GHz, past any InGaAs detrapping tail.
const AFTER_MAX: usize = 65536;

/// Eight-point Gauss-Legendre on [-1, 1], panels <= 1 sigma: exact to rounding on a Gaussian.
/// Not a rational erf: its ~1e-7 ABSOLUTE floor reads as bin leak here and photon-number confusion
/// in `pnr.rs`.
const GL_X: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329_0,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329_0,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];

const GL_W: [f64; 8] = [
    0.101_228_536_290_376_3,
    0.222_381_034_453_374_5,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362_0,
    0.362_683_783_378_362_0,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

/// Mass of N(0, sd) on [a, b]; exactly 0.0 past CORE_CUT. `pnr.rs` reads it over its readout bins.
pub(crate) fn gauss_mass(a: f64, b: f64, sd: f64) -> f64 {
    let lo = a.max(-CORE_CUT * sd);
    let hi = b.min(CORE_CUT * sd);
    if hi <= lo {
        return 0.0;
    }

    let panels = ((hi - lo) / sd).ceil().max(1.0);
    let step = (hi - lo) / panels;
    let mut sum = 0.0;
    let mut p = 0.0;
    while p < panels {
        let mid = lo + step * (p + 0.5);
        for (x, w) in GL_X.iter().zip(GL_W.iter()) {
            let t = (mid + 0.5 * step * x) / sd;
            sum += w * (-0.5 * t * t).exp();
        }

        p += 1.0;
    }

    sum * 0.5 * step / (sd * std::f64::consts::TAU.sqrt())
}

/// Tail mass on [a, b]. Carriers diffuse LATE, so the clip at zero is physics.
fn tail_mass(a: f64, b: f64, tau: f64) -> f64 {
    let lo = a.max(0.0);
    let hi = b.max(0.0);
    if hi <= lo {
        return 0.0;
    }

    (-lo / tau).exp() - (-hi / tau).exp()
}

/// How much of one slot's light each nearby slot's window collects.
struct Smear {
    w: Vec<f64>,
    span: usize,
    accept: f64,
    own: f64,
}

impl Smear {
    /// Detections in no window: a pure loss. Clamped; tiling windows overshoot.
    fn loss(&self) -> f64 {
        (1.0 - self.accept).max(0.0)
    }

    /// ACCEPTED detections in a neighbour's window; half wrong, QBER = leak/2.
    fn leak(&self) -> f64 {
        (self.accept - self.own) / self.accept
    }
}

/// Response kernel; `None` when there is no response and the slot arithmetic
/// must stay bit-for-bit. `window` 0 = contiguous bins, no gate loss.
fn build_smear(
    period: f64,
    jitter: f64,
    window: f64,
    frac: f64,
    tau: f64,
) -> PyResult<Option<Smear>> {
    if jitter == 0.0 && frac == 0.0 {
        return Ok(None);
    }

    let sd = jitter / FWHM_SD;
    let half = 0.5 * if window > 0.0 { window } else { period };
    let core = if frac < 1.0 { CORE_CUT * sd } else { 0.0 };
    let drag = if frac > 0.0 {
        tau * (frac / SPAN_TOL).ln().max(0.0)
    } else {
        0.0
    };
    let reach = ((core.max(drag) + half) / period).ceil();
    if !(reach.is_finite() && reach <= SPAN_MAX as f64) {
        return Err(PyValueError::new_err(format!(
            "a timing response of {jitter:e} s FWHM with a {frac} tail at \
             {tau:e} s reaches past {SPAN_MAX} symbol periods of {period:e} s: \
             at that point the detector cannot tell one slot from any other"
        )));
    }

    let span = reach.max(0.0) as usize;
    let mut w = vec![0.0; 2 * span + 1];
    for (i, out) in w.iter_mut().enumerate() {
        let mid = (i as f64 - span as f64) * period;
        let (a, b) = (mid - half, mid + half);
        let hit = f64::from(a <= 0.0 && 0.0 < b);
        let core = if sd > 0.0 { gauss_mass(a, b, sd) } else { hit };
        let drag = if tau > 0.0 { tail_mass(a, b, tau) } else { hit };
        *out = (1.0 - frac) * core + frac * drag;
    }

    let accept: f64 = w.iter().sum();
    if !(accept > 0.0 && accept <= 1.0 + 1e-9) {
        return Err(PyValueError::new_err(format!(
            "the timing response collects {accept} of the light, which is not a \
             probability: check jitter, window, tail_frac and tail_time"
        )));
    }

    let own = w[span];

    Ok(Some(Smear {
        w,
        span,
        accept,
        own,
    }))
}

/// Also rejects a window wider than the period, and a tail with no constant.
fn check_timing(rate: f64, jitter: f64, window: f64, frac: f64, tau: f64) -> PyResult<()> {
    check_nonneg("jitter", jitter)?;
    check_nonneg("window", window)?;
    check_prob("tail_frac", frac)?;
    check_nonneg("tail_time", tau)?;

    if window * rate > 1.0 {
        return Err(PyValueError::new_err(format!(
            "window {window:e} s exceeds the symbol period {:e} s: neighbouring \
             acceptance windows would overlap and count one detection twice",
            1.0 / rate
        )));
    }

    if frac > 0.0 && tau <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "tail_frac {frac} puts weight in a diffusion tail while tail_time \
             is {tau}: give the tail a time constant or set tail_frac to 0"
        )));
    }

    Ok(())
}

/// One run's report; the counts are exact integers and the ratios derived.
#[pyclass]
pub(crate) struct ClickOut {
    n_slots: u64,
    clicks: u64,
    clicks_d0: u64,
    clicks_d1: u64,
    sifted: u64,
    doubles: u64,
    errors: u64,
    errors_d0: u64,
    errors_d1: u64,
    qber: f64,
    sift_rate: f64,
    click_rate: f64,
    visibility: f64,
    v_phase: f64,
    mu_bob: f64,
    window_loss: f64,
    bin_leak: f64,
    record_d0: Vec<u8>,
    record_d1: Vec<u8>,
    record_bit: Vec<u8>,
}

#[pymethods]
impl ClickOut {
    /// Detection slots evaluated: pulses less the delay.
    #[getter]
    fn n_slots(&self) -> u64 {
        self.n_slots
    }

    /// Clicks recorded across both detectors, after dead time and afterpulsing.
    #[getter]
    fn clicks(&self) -> u64 {
        self.clicks
    }

    /// Clicks on the detector that reads a zero differential phase.
    #[getter]
    fn clicks_d0(&self) -> u64 {
        self.clicks_d0
    }

    /// Clicks on the detector that reads a pi differential phase.
    #[getter]
    fn clicks_d1(&self) -> u64 {
        self.clicks_d1
    }

    /// Slots with at least one click: the sifted key, doubles included.
    #[getter]
    fn sifted(&self) -> u64 {
        self.sifted
    }

    /// Double clicks. Counted here AND in `sifted`: key riding the coin.
    #[getter]
    fn doubles(&self) -> u64 {
        self.doubles
    }

    /// Sifted slots whose bit disagreed with Alice's keyed differential phase.
    #[getter]
    fn errors(&self) -> u64 {
        self.errors
    }

    /// Errors the zero-phase detector made ALONE; its single count is `clicks_d0 - doubles`.
    #[getter]
    fn errors_d0(&self) -> u64 {
        self.errors_d0
    }

    /// As `errors_d0`, for the pi-phase detector. `errors - errors_d0 - errors_d1` is the squashing
    /// coin's.
    #[getter]
    fn errors_d1(&self) -> u64 {
        self.errors_d1
    }

    /// Quantum bit error rate, errors / sifted; 0 when nothing was sifted.
    #[getter]
    fn qber(&self) -> f64 {
        self.qber
    }

    /// Sifted slots per detection slot.
    #[getter]
    fn sift_rate(&self) -> f64 {
        self.sift_rate
    }

    /// Clicks per detection slot, both detectors pooled.
    #[getter]
    fn click_rate(&self) -> f64 {
        self.click_rate
    }

    /// (I_max - I_min)/(I_max + I_min) from the port intensities, dark counts
    /// EXCLUDED: qber = (1 - visibility)/2 only at a negligible dark floor.
    #[getter]
    fn visibility(&self) -> f64 {
        self.visibility
    }

    /// Mean squared differential phase E[(theta_k - theta_{k-d})^2] in rad^2, expected
    /// 2*pi*linewidth*d/symbol_rate. Error map at unit visibility: e = (1 - exp(-v_phase/2))/2,
    /// limit 1/2; Chen et al.'s sigma_phi = 0.13 rad reads 0.42% against their 0.5%.
    ///
    /// Not another geometry's: M-slice post-compensated encoding gives (1 - (M/pi)*sin(pi/M))/2,
    /// 4.984% at M = 4. Neither form crosses geometries.
    #[getter]
    fn v_phase(&self) -> f64 {
        self.v_phase
    }

    /// Mean photons per pulse at Bob's interferometer, t * mu, BEFORE the timing response.
    #[getter]
    fn mu_bob(&self) -> f64 {
        self.mu_bob
    }

    /// Detections into no acceptance window: a loss, i.e. a smaller eta.
    #[getter]
    fn window_loss(&self) -> f64 {
        self.window_loss
    }

    /// ACCEPTED detections in a neighbour's window, half errors; not foldable into eta. 0 without
    /// jitter; under the default tiling (window = 0) a Gaussian core DOES leak.
    #[getter]
    fn bin_leak(&self) -> f64 {
        self.bin_leak
    }

    /// Per-slot clicks on the zero-phase detector, one byte per gate, empty unless keep_record.
    /// A fresh copy per read.
    #[getter]
    fn record_d0<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.record_d0)
    }

    /// Per-slot clicks on the pi-phase detector, as `record_d0`.
    #[getter]
    fn record_d1<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.record_d1)
    }

    /// Alice's keyed differential bit per slot, as `record_d0`.
    #[getter]
    fn record_bit<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.record_bit)
    }
}

/// Threshold click probability for coherent `alpha = re + i*im`:
/// `1 - (1 - dark)*exp(-eta*|alpha|^2)`; `dark` PER GATE, `|alpha|^2` photons.
#[pyfunction]
pub(crate) fn click_prob(re: f64, im: f64, eta: f64, dark: f64) -> PyResult<f64> {
    check_finite("re", re)?;
    check_finite("im", im)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    Ok(click_p(eta, re * re + im * im, dark))
}

/// Delay-interferometer port intensities in photons, `(constructive,
/// destructive)`, summing to half the flux at any `vis`. Raises on overflow.
#[pyfunction]
pub(crate) fn interfere(
    re_now: f64,
    im_now: f64,
    re_old: f64,
    im_old: f64,
    vis: f64,
) -> PyResult<(f64, f64)> {
    check_finite("re_now", re_now)?;
    check_finite("im_now", im_now)?;
    check_finite("re_old", re_old)?;
    check_finite("im_old", im_old)?;
    check_prob("vis", vis)?;

    // Halve BEFORE squaring: squaring first overflows from |alpha| ~ 1e154.
    let (an, bn) = (re_now * 0.5, im_now * 0.5);
    let (ao, bo) = (re_old * 0.5, im_old * 0.5);
    let flux = an * an + bn * bn + ao * ao + bo * bo;

    // |cross| <= flux by AM-GM, so the flux is the only term that can overflow.
    let cross = 2.0 * vis * (an * ao + bn * bo);
    if !flux.is_finite() {
        return Err(PyValueError::new_err(format!(
            "the flux of ({re_now:e}, {im_now:e}) and ({re_old:e}, {im_old:e}) overflows \
             f64: |alpha|^2 has no representation at these amplitudes, and the \
             destructive port would come back inf - inf = NaN"
        )));
    }

    Ok((flux + cross, flux - cross))
}

/// The two jitter contributions as `(window_loss, bin_leak)`, no pulse train.
/// `jitter` is the Gaussian core FWHM in s, `window` the acceptance width in s
/// within one period (0 = tiled), `tail_frac`/`tail_time` the tail's weight/s.
#[pyfunction]
pub(crate) fn jitter_split(
    symbol_rate: f64,
    jitter: f64,
    window: f64,
    tail_frac: f64,
    tail_time: f64,
) -> PyResult<(f64, f64)> {
    check_pos("symbol_rate", symbol_rate)?;
    check_timing(symbol_rate, jitter, window, tail_frac, tail_time)?;
    let sm = build_smear(1.0 / symbol_rate, jitter, window, tail_frac, tail_time)?;

    Ok(sm.as_ref().map_or((0.0, 0.0), |s| (s.loss(), s.leak())))
}

// BACKGROUND flux COMPOSES with the dark floor in `bg_floor`, never adds; the result is
// q.ClickDetector(dark=...). E0 = 1/2 already charges its errors: no separate QBER term.
//
// Three factors, three owners: `pol` the SOURCE, `split` the receiver optics, `gate` the detector.
//
// Raman: Kumar, Qin & Alleaume, "Coexistence of continuous variable QKD with
// intense DWDM classical channels", New J. Phys. 17, 043027 (2015),
// arXiv:1412.1403, Eq. (6) is photons per DETECTION MODE. No local oscillator filters a threshold
// detector, so M = passband*gate*pol, the OPTICAL time-bandwidth product. Reproduces Eraerds,
// Walenta, Legre, Gisin & Zbinden, "Quantum key distribution and 1 Gbit/s data encryption over
// a single fibre", New J. Phys. 12, 063027 (2010), arXiv:0912.1798, Eq. (11) EXACTLY at pol = 2,
// split = 1, P_ram from their Eqs. (2)-(3); Eq. (6)'s 1/2 is the local oscillator's selectivity.
// test_raman_matches_power pins it.
//
// Rayleigh: elastic and in band, no passband. split = 1/2 across two detectors: Mandil, Qian & Lo,
// "Long-fiber Sagnac interferometers for twin field quantum key distribution networks",
// arXiv:2407.08009.
//
// The click law is POISSON; Eraerds' is linear (their note under Eq. (15)) and reads HIGH,
// x >= 1 - exp(-x), understating a key rate.
//
// `bg_thermal` is the ARBITER of the Poisson step, on the Raman route ALONE: Rayleigh backscatter
// is speckled and its mode count depends on a source linewidth nothing here states.

/// `M = passband*gate*pol`, the modes one gate integrates over.
fn bg_count(passband: f64, gate: f64, pol: u32) -> PyResult<f64> {
    let modes = match pol {
        1 | 2 => f64::from(pol),
        other => {
            return Err(PyValueError::new_err(format!(
                "pol must be 1 or 2, got {other}: pass 2 for an unpolarised background (Kumar, \
                 Qin & Alleaume Eq. (6)) or 1 for an occupancy stated on one polarisation; the \
                 share reaching one detector is `split`"
            )))
        }
    };

    Ok(passband * gate * modes)
}

/// Refuses a saturated gate under the name that produced it, not downstream as `dark`.
fn bg_valid(p: f64, what: &str) -> PyResult<f64> {
    if p >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "{what} fires the gate with probability 1 and carries no signal: lower the flux, \
             the passband or the gate"
        )));
    }

    Ok(p)
}

/// `(flux, modes)`: mean spontaneous-Raman photons in ONE detector's gate, and M.
///
/// `occupancy` photons per detection mode (`impairments.raman_photons`), `passband` the OPTICAL
/// filter FWHM in Hz (`qkd.budget.raman_width` inverts dlambda = lambda^2 B/c), `gate` in s, `pol`
/// the source's polarisation modes, `split` the share reaching this detector. `passband` admits no
/// zero.
#[pyfunction]
pub(crate) fn bg_raman(
    occupancy: f64,
    passband: f64,
    gate: f64,
    pol: u32,
    split: f64,
) -> PyResult<(f64, f64)> {
    check_nonneg("occupancy", occupancy)?;
    check_pos("passband", passband)?;
    check_pos("gate", gate)?;
    check_prob("split", split)?;
    let modes = bg_count(passband, gate, pol)?;
    let flux = occupancy * modes * split;
    if !flux.is_finite() {
        return Err(PyValueError::new_err(format!(
            "an occupancy of {occupancy:e} over {modes:e} modes overflows f64: lower the \
             occupancy, passband or gate"
        )));
    }

    Ok((flux, modes))
}

/// Mean Rayleigh-backscattered photons standing in ONE detector's gate.
///
/// `photons` the mean over a symbol `period` (`impairments.rayleigh_photons`, whose 1/2 is the
/// local oscillator's selectivity, not `split`), `gate` in s, `split` the share reaching this
/// detector: 1/2 for the two-detector receivers here.
#[pyfunction]
pub(crate) fn bg_rayleigh(photons: f64, period: f64, gate: f64, split: f64) -> PyResult<f64> {
    check_nonneg("photons", photons)?;
    check_pos("period", period)?;
    check_pos("gate", gate)?;
    check_prob("split", split)?;
    if gate > period {
        return Err(PyValueError::new_err(format!(
            "gate {gate} must not exceed the symbol period {period}: a wider gate counts the \
             neighbouring slot's photons twice; shorten the gate"
        )));
    }

    Ok(photons * (gate / period) * split)
}

/// Per-gate click probability of the dark floor and every background flux:
/// `1 - (1 - dark)*exp(-eta*sum(flux))`. Hand it to `q.ClickDetector(dark=...)`.
#[pyfunction]
pub(crate) fn bg_floor(dark: f64, flux: Vec<f64>, eta: f64) -> PyResult<f64> {
    check_gate("dark", dark)?;
    check_unit("eta", eta)?;
    let mut total = 0.0;
    for (i, f) in flux.iter().enumerate() {
        check_nonneg(&format!("flux[{i}]"), *f)?;
        total += f;
    }

    check_finite("the summed flux", total)?;

    bg_valid(click_p(eta, total, dark), &format!("a flux of {total:e} photons at eta = {eta}"))
}

/// M-mode chaotic-light per-gate click probability `1 - (1 - dark)*(1 + eta*split*occupancy)^-M`:
/// the arbiter of `bg_floor`'s Poisson step, NOT a second answer. Reads occupancy and M APART.
/// Arguments as `bg_raman`, plus `eta` and per-gate `dark`.
#[pyfunction]
pub(crate) fn bg_thermal(
    occupancy: f64,
    passband: f64,
    gate: f64,
    pol: u32,
    split: f64,
    eta: f64,
    dark: f64,
) -> PyResult<f64> {
    check_nonneg("occupancy", occupancy)?;
    check_pos("passband", passband)?;
    check_pos("gate", gate)?;
    check_prob("split", split)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;
    let modes = bg_count(passband, gate, pol)?;

    bg_valid(
        1.0 - (1.0 - dark) * (1.0 + eta * split * occupancy).powf(-modes),
        &format!("an occupancy of {occupancy:e} over {modes:e} chaotic modes at eta = {eta}"),
    )
}

/// One random phase bit per pulse, pure in the index: chunking cannot move it.
fn keying(used: usize, seed: u64, cz: usize) -> Vec<u8> {
    let key = Stream::new(seed, KEYING);
    let mut bits = vec![0u8; used];
    bits.par_chunks_mut(cz).enumerate().for_each(|(c, out)| {
        let base = c * cz;
        for (j, b) in out.iter_mut().enumerate() {
            *b = u8::from(key.uniforms((base + j) as u64)[0] < 0.5);
        }
    });

    bits
}

/// SIGNED fringe per slot, sign(bit) * V * cos(theta_k - theta_{k-d}), so slot
/// k's ports carry (mu/2)(1 +- fringe_k). Zero outside the span, so the jitter
/// convolution reads the ends as dark. `mono`: theta is +0.0 and empty.
fn fringes(
    used: usize,
    d: usize,
    cz: usize,
    theta: &[f64],
    bits: &[u8],
    vis: f64,
    mono: bool,
) -> Vec<f64> {
    let mut fr = vec![0.0; used];
    fr.par_chunks_mut(cz).enumerate().for_each(|(c, out)| {
        let base = c * cz;
        if mono {
            for (j, f) in out.iter_mut().enumerate() {
                let k = base + j;
                if k < d {
                    continue;
                }

                *f = (1.0 - 2.0 * f64::from(bits[k] ^ bits[k - d])) * vis;
            }

            return;
        }

        for (j, f) in out.iter_mut().enumerate() {
            let k = base + j;
            if k < d {
                continue;
            }

            let sign = 1.0 - 2.0 * f64::from(bits[k] ^ bits[k - d]);
            *f = sign * (vis * wrap(theta[k] - theta[k - d]).cos());
        }
    });

    fr
}

/// Per-slot byte: bit 0 d0's raw click, bit 1 d1's, bit 2 Alice's differential
/// bit, bits 3 and 4 the afterpulse coins. Second return is the accepted
/// weight, empty without jitter and NOT constant near the train's ends.
fn raw_slots(
    used: usize,
    d: usize,
    cz: usize,
    seed: u64,
    fr: &[f64],
    bits: &[u8],
    p: &Optics,
    sm: Option<&Smear>,
    mono: bool,
) -> (Vec<u8>, Vec<f64>) {
    let gate = Stream::new(seed, GATE);

    // Both conditions load-bearing: lw == 0 alone gives 2.1x-4.7x QBER error.
    let half = p.mu / 2.0;
    let flat = if mono && sm.is_none() {
        let thr = |j: usize, v: f64| click_p(p.eta[j], half * (1.0 + v), p.dark[j]);

        Some(([thr(0, p.vis), thr(0, -p.vis)], [thr(1, p.vis), thr(1, -p.vis)]))
    } else {
        None
    };
    let one = |k: usize| -> (u8, f64) {
        if k < d {
            return (0, 0.0);
        }

        let bit = bits[k] ^ bits[k - d];
        // d0 reads the port slot k's bit made bright, d1 the other; own eta and dark, same u.
        let (got, c0, c1) = match flat {
            Some((d0, d1)) => {
                if bit == 0 {
                    (1.0, d0[0], d1[1])
                } else {
                    (1.0, d0[1], d1[0])
                }
            }
            None => {
                let (got, band) = collect(k, d, used, fr, sm);
                let ip = half * (got + band);
                let im = half * (got - band);
                let c0 = click_p(p.eta[0], ip, p.dark[0]);

                (got, c0, click_p(p.eta[1], im, p.dark[1]))
            }
        };
        let u = gate.uniforms(k as u64);
        let mut byte = bit << 2;
        if u[0] < c0 {
            byte |= 1;
        }
        if u[1] < c1 {
            byte |= 2;
        }
        if u[2] < p.after[0] {
            byte |= 8;
        }
        if u[3] < p.after[1] {
            byte |= 16;
        }

        (byte, got)
    };
    let mut slots = vec![0u8; used];
    let mut kept = vec![0.0; if sm.is_some() { used } else { 0 }];
    if sm.is_some() {
        slots
            .par_chunks_mut(cz)
            .zip(kept.par_chunks_mut(cz))
            .enumerate()
            .for_each(|(c, (out, acc))| {
                let base = c * cz;
                for (j, (s, a)) in out.iter_mut().zip(acc.iter_mut()).enumerate() {
                    let got = one(base + j);
                    *s = got.0;
                    *a = got.1;
                }
            });
    } else {
        slots.par_chunks_mut(cz).enumerate().for_each(|(c, out)| {
            let base = c * cz;
            for (j, s) in out.iter_mut().enumerate() {
                *s = one(base + j).0;
            }
        });
    }

    (slots, kept)
}

/// What slot k's window collects: `(sum_m w_m, sum_m w_m fringe_{k-m})`, giving
/// ports (mu/2)(weight +- fringe). Without a response, (1, fringe_k).
#[inline]
fn collect(k: usize, d: usize, used: usize, fr: &[f64], sm: Option<&Smear>) -> (f64, f64) {
    let s = match sm {
        None => return (1.0, fr[k]),
        Some(s) => s,
    };
    let mut got = 0.0;
    let mut band = 0.0;
    for (i, w) in s.w.iter().enumerate() {
        if k + s.span < i {
            continue;
        }

        let src = k + s.span - i;
        if src < d || src >= used {
            continue;
        }

        got += w;
        band += w * fr[src];
    }

    (got, band)
}

/// One slot's constants; `mu` in photons, `dark`/`after` per gate. Arrays are BY DETECTOR:
/// [0] zero-phase port, [1] pi-phase.
struct Optics {
    mu: f64,
    eta: [f64; 2],
    dark: [f64; 2],
    vis: f64,
    after: [f64; 2],
}

/// Entry `m`: probability one click makes an afterpulse `m + 1` gates later, GIVEN the detector is
/// live then. Each entry per gate, the sum below 1.
fn check_after(name: &str, shape: Option<&[f64]>) -> PyResult<()> {
    let s = match shape {
        None => return Ok(()),
        Some(s) => s,
    };

    if s.is_empty() || s.len() > AFTER_MAX {
        return Err(PyValueError::new_err(format!(
            "{name} has {} entries, expected 1 to {AFTER_MAX}, one per GATE since the \
             click; a single entry is the flat coin",
            s.len()
        )));
    }

    let mut total = 0.0;
    for (m, p) in s.iter().enumerate() {
        check_gate(&format!("{name}[{m}]"), *p)?;
        total += p;
    }

    if total >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "{name} sums to {total} expected afterpulses per click: at 1 or more the \
             detector free-runs on its own carriers; scale the curve below 1"
        )));
    }

    Ok(())
}

/// One DPS-style run of `n` phase-keyed pulses. `mu` photons/pulse, `linewidth`
/// and `symbol_rate` Hz, `dead_time` s, `delay` in periods, `dark`/`afterpulse`
/// per gate, `visibility` the pre-walk contrast, `chunk` the grain (0 =
/// default), `jitter` through `tail_time` as `jitter_split`.
///
/// `eta`, `dark`, `dead_time` and `afterpulse` are the ZERO-PHASE detector's, and the pi-phase
/// one's unless `eta_d1`/`dark_d1`/`dead_d1`/`after_d1` are set: the only place a
/// detection-efficiency mismatch is hardware rather than a priced attack.
///
/// `after_shape` is a MEASURED curve in place of the flat coin, as `check_after`. Given for either
/// detector it runs for both, falling back to `after_shape` then `[afterpulse]`: `after_shape =
/// [afterpulse]` at `dead_time = 0` is the default run bit for bit. `recovery` > 0 is refused.
#[pyfunction]
#[pyo3(signature = (
    n,
    seed,
    mu,
    t,
    eta,
    dark,
    visibility,
    delay,
    linewidth,
    symbol_rate,
    dead_time,
    afterpulse,
    chunk,
    keep_record,
    jitter = 0.0,
    window = 0.0,
    tail_frac = 0.0,
    tail_time = 0.0,
    eta_d1 = None,
    dark_d1 = None,
    dead_d1 = None,
    after_d1 = None,
    after_shape = None,
    after_shape_d1 = None,
    recovery = None,
))]
pub(crate) fn run_clicks(
    py: Python<'_>,
    n: u64,
    seed: u64,
    mu: f64,
    t: f64,
    eta: f64,
    dark: f64,
    visibility: f64,
    delay: u64,
    linewidth: f64,
    symbol_rate: f64,
    dead_time: f64,
    afterpulse: f64,
    chunk: u64,
    keep_record: bool,
    jitter: f64,
    window: f64,
    tail_frac: f64,
    tail_time: f64,
    eta_d1: Option<f64>,
    dark_d1: Option<f64>,
    dead_d1: Option<f64>,
    after_d1: Option<f64>,
    after_shape: Option<Vec<f64>>,
    after_shape_d1: Option<Vec<f64>>,
    recovery: Option<f64>,
) -> PyResult<ClickOut> {
    check_pos("mu", mu)?;
    check_unit("t", t)?;
    check_unit("eta", eta)?;
    check_gate("dark", dark)?;

    // Detector 1 falls back to detector 0's numbers.
    let etas = [eta, eta_d1.unwrap_or(eta)];
    let darks = [dark, dark_d1.unwrap_or(dark)];
    let deads = [dead_time, dead_d1.unwrap_or(dead_time)];
    let afters = [afterpulse, after_d1.unwrap_or(afterpulse)];
    check_unit("eta_d1", etas[1])?;
    check_gate("dark_d1", darks[1])?;
    check_gate("afterpulse", afterpulse)?;
    check_gate("after_d1", afters[1])?;
    check_after("after_shape", after_shape.as_deref())?;
    check_after("after_shape_d1", after_shape_d1.as_deref())?;
    check_prob("visibility", visibility)?;
    check_nonneg("linewidth", linewidth)?;
    check_pos("symbol_rate", symbol_rate)?;
    check_nonneg("dead_time", dead_time)?;

    // AFTER dead_time's check: an unset dead_d1 inherits it and would refuse under the wrong name.
    check_nonneg("dead_d1", deads[1])?;
    check_timing(symbol_rate, jitter, window, tail_frac, tail_time)?;
    if recovery.is_some_and(|r| r > 0.0) {
        return Err(PyNotImplementedError::new_err(
            "recovery is refused: a partial-efficiency ramp is a FREE-RUNNING detector's reset, \
             and four pieces are missing. (1) Its shape is a per-device measured efficiency \
             versus bias current curve -- Kerman, Dauler, Keicher, Yang, Berggren, Gol'tsman & \
             Voronov, Appl. Phys. Lett. 88, 111116 (2006): I/Ic = 0.78 reads 6e-2 of the \
             plateau -- and no exponential stands in for it. (2) Its time axis is each device's \
             measured kinetic inductance. (3) click_p carries ONE efficiency and ONE dark floor \
             and the Smear is built once; a recovering junction moves all three in no measured \
             ratio. (4) Below the return current the device LATCHES, a stop rather than a \
             smaller eta. For the gated detector q.ClickDetector documents, pass recovery=None \
             or 0.0; assemble a free-running receiver by hand",
        ))
    }

    if delay == 0 {
        return Err(PyValueError::new_err(
            "delay must be >= 1 symbol period (1 is standard DPS)".to_string(),
        ));
    }

    let d = delay as usize;
    let used = n as usize;
    if used < d + 2 {
        return Err(PyValueError::new_err(format!(
            "n must exceed delay by at least two pulses, got n={n} delay={delay}"
        )));
    }

    let cz = if chunk == 0 { GRAIN } else { chunk as usize }.min(used);
    let p = Optics {
        mu: t * mu,
        eta: etas,
        dark: darks,
        vis: visibility,
        after: afters,
    };

    // Both detectors run a curve or neither: partner's curve, then the length-1 flat coin.
    let shapes: [Vec<f64>; 2] = if after_shape.is_none() && after_shape_d1.is_none() {
        [Vec::new(), Vec::new()]
    } else {
        let s0 = match after_shape.as_deref() {
            Some(v) => v.to_vec(),
            None => vec![afters[0]],
        };
        let s1 = match after_shape_d1.as_deref().or(after_shape.as_deref()) {
            Some(v) => v.to_vec(),
            None => vec![afters[1]],
        };

        [s0, s1]
    };

    let sm = build_smear(1.0 / symbol_rate, jitter, window, tail_frac, tail_time)?;

    // GIL released for the kernel; `build_smear` is the one fallible step.
    Ok(py.detach(|| {
        // One walk, not a tx/LO pair: DPS is self-referencing. `mono` skips it bit-identically.
        let step = std::f64::consts::TAU / symbol_rate;
        let mono = linewidth == 0.0;
        let theta = if mono {
            Vec::new()
        } else {
            walk(used, seed, (step * linewidth).sqrt(), 0.0)
        };
        let bits = keying(used, seed, cz);
        let fr = fringes(used, d, cz, &theta, &bits, p.vis, mono);
        let (slots, kept) = raw_slots(used, d, cz, seed, &fr, &bits, &p, sm.as_ref(), mono);
        let own = sm.as_ref().map_or(1.0, |s| s.own);

        let squash = Stream::new(seed, SQUASH);
        let coin = Stream::new(seed, GATE);
        let gates = |x: f64| (x * symbol_rate).ceil().min(used as f64) as usize;
        let dead = [gates(deads[0]), gates(deads[1])];
        let curved = !shapes[0].is_empty() || !shapes[1].is_empty();
        let wheel = shapes[0].len().max(shapes[1].len()) + 1;

        // Survival product per FUTURE gate, one wheel per detector: a click scales k+1+m by 1 - p[m].
        let mut wheels = [vec![1.0; wheel], vec![1.0; wheel]];
        let span = used - d;
        let mut rec = if keep_record {
            (
                Vec::with_capacity(span),
                Vec::with_capacity(span),
                Vec::with_capacity(span),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Serial: slot k depends on every earlier slot. The curve's coin re-derives u[2]/u[3] of
        // raw_slots' Threefry block, consuming no new counter.
        let mut live = [0usize; 2];
        let mut armed = [false; 2];
        let mut counts = [0u64; 2];
        let mut wrongs = [0u64; 2];
        let mut sifted = 0u64;
        let mut errors = 0u64;
        let mut doubles = 0u64;
        let mut i_max = 0.0;
        let mut i_min = 0.0;
        let mut phi2 = 0.0;
        for k in d..used {
            let s = slots[k];
            let bit = (s >> 2) & 1;
            let mut fired = [false; 2];

            // Read the wheel BEFORE the hold-off test: a carrier released while held off is lost.
            let mut pend = [0.0; 2];
            if curved {
                let at = k % wheel;
                for (j, w) in wheels.iter_mut().enumerate() {
                    pend[j] = 1.0 - w[at];
                    w[at] = 1.0;
                }
            }

            let u = if curved { coin.uniforms(k as u64) } else { [0.0; 4] };
            for j in 0..2 {
                if k < live[j] {
                    continue;
                }

                let raw = (s & (1u8 << j)) != 0;
                let after = if curved {
                    u[2 + j] < pend[j]
                } else {
                    armed[j] && (s & (8u8 << j)) != 0
                };
                armed[j] = false;
                if raw || after {
                    fired[j] = true;
                    live[j] = k + 1 + dead[j];
                    armed[j] = true;
                    counts[j] += 1;
                    for (m, q) in shapes[j].iter().enumerate() {
                        wheels[j][(k + 1 + m) % wheel] *= 1.0 - q;
                    }
                }
            }

            match (fired[0], fired[1]) {
                (true, false) => {
                    sifted += 1;
                    wrongs[0] += u64::from(bit != 0);
                    errors += u64::from(bit != 0);
                }
                (false, true) => {
                    sifted += 1;
                    wrongs[1] += u64::from(bit != 1);
                    errors += u64::from(bit != 1);
                }
                // Squashing, not optional: a double is KEPT with a random bit, or blinding hides.
                (true, true) => {
                    doubles += 1;
                    sifted += 1;
                    errors += u64::from(u8::from(squash.uniforms(k as u64)[0] < 0.5) != bit);
                }
                (false, false) => {}
            }

            // Leaked light is bit-uncorrelated: measured V drops to (own/accept)*V.
            let dphi = if mono {
                0.0
            } else {
                wrap(theta[k] - theta[k - d])
            };
            let fringe = fr[k] * (1.0 - 2.0 * f64::from(bit));
            let got = if kept.is_empty() { 1.0 } else { kept[k] };
            i_max += p.mu * (got + own * fringe) / 2.0;
            i_min += p.mu * (got - own * fringe) / 2.0;
            phi2 += dphi * dphi;

            if keep_record {
                rec.0.push(u8::from(fired[0]));
                rec.1.push(u8::from(fired[1]));
                rec.2.push(bit);
            }
        }

        let slots_f = span as f64;
        let total = counts[0] + counts[1];
        let flux = i_max + i_min;

        ClickOut {
            n_slots: span as u64,
            clicks: total,
            clicks_d0: counts[0],
            clicks_d1: counts[1],
            sifted,
            doubles,
            errors,
            errors_d0: wrongs[0],
            errors_d1: wrongs[1],
            qber: if sifted > 0 {
                errors as f64 / sifted as f64
            } else {
                0.0
            },
            sift_rate: sifted as f64 / slots_f,
            click_rate: total as f64 / slots_f,
            visibility: if flux > 0.0 {
                (i_max - i_min) / flux
            } else {
                0.0
            },
            v_phase: phi2 / slots_f,
            mu_bob: p.mu,
            window_loss: sm.as_ref().map_or(0.0, Smear::loss),
            bin_leak: sm.as_ref().map_or(0.0, Smear::leak),
            record_d0: rec.0,
            record_d1: rec.1,
            record_bit: rec.2,
        }
    }))
}

/// Terms the Poisson tail sum takes before giving up.
const TAIL_TERMS: usize = 4096;

/// lambda := 3 + sqrt(5), MTT23 Eq. (44).
fn dps_lam() -> f64 {
    3.0 + 5.0_f64.sqrt()
}

/// Delta(x, y) := sqrt(2*x*n_det*ln(1/y)), MTT23 Eq. (43): Azuma's deviation
/// over `n_det` detected rounds at bounded difference sqrt(x).
fn dps_delta(x: f64, y: f64, n_det: f64) -> f64 {
    (2.0 * x * n_det * (1.0 / y).ln()).sqrt()
}

/// Gamma_n of MTT23 Eq. (45), the Chernoff slack above t*q_n*n_em on the number
/// of emitted blocks carrying n or more photons.
fn dps_gamma(q: f64, t: f64, n_em: f64, eps2: f64) -> f64 {
    let le = eps2.ln();

    0.5 * (-le + (le * le - 8.0 * t * q * n_em * le).sqrt())
}

/// sum_{nu >= a} e^{-l} l^nu / nu!, by the head complement where the head is
/// small and by direct summation where it is not: at l ~ 1e-4 the complement
/// 1 - e^{-l}(1 + l) cancels eight digits away, and q_2 IS that difference.
fn poisson_tail(a: usize, l: f64) -> f64 {
    let mut term = (-l).exp();
    let mut head = 0.0;
    for nu in 0..a {
        if nu > 0 {
            term *= l / nu as f64;
        }
        head += term;
    }

    if head < 0.5 {
        return 1.0 - head;
    }

    let mut sum = 0.0;
    let mut cur = term;
    for nu in a..(a + TAIL_TERMS) {
        cur *= l / nu as f64;
        sum += cur;
        if (nu as f64) > l && cur <= 1e-19 * sum {
            break;
        }
    }

    sum
}

/// `(q1, q2, q3)` of MTT23 Eq. (3) for the coherent source of their Eq. (51): a
/// block of three pulses each of mean photon number `mu`, so
/// q_a = sum_{nu >= a} e^{-3mu}(3mu)^nu/nu! is the probability that one block
/// carries a or more photons in all optical modes.
#[pyfunction]
pub(crate) fn dps_source(mu: f64) -> PyResult<(f64, f64, f64)> {
    check_nonneg("mu", mu)?;
    let l = 3.0 * mu;

    Ok((poisson_tail(1, l), poisson_tail(2, l), poisson_tail(3, l)))
}

/// `(n_det, n_code, n_samp)` under MTT23 Eq. (52), a FORWARD model of their
/// Sec. V simulation rather than a receiver: n_det = n_em*2*eta*mu*exp(-2*eta*mu)
/// is the Poisson probability that EXACTLY ONE photon lands across the block's
/// two interior time slots, whose mean is 2*eta*mu. No dark floor, no
/// misalignment and no jitter -- it is the paper's own idealisation.
#[pyfunction]
pub(crate) fn dps_counts(n_em: f64, eta: f64, mu: f64, t: f64) -> PyResult<(f64, f64, f64)> {
    check_pos("n_em", n_em)?;
    check_unit("eta", eta)?;
    check_nonneg("mu", mu)?;
    check_eps("t", t)?;
    let flux = 2.0 * eta * mu;
    let n_det = n_em * flux * (-flux).exp();

    Ok((n_det, t * n_det, (1.0 - t) * n_det))
}

/// `(a*, b*)` of MTT23 Eqs. (48) and (49): the pair minimising Kato's deviation
/// term [b + a*(2m/n - 1)]*sqrt(n) at failure probability `eps` over `n`
/// detected rounds, given the prediction `m` of the three-photon count.
///
/// `m` must be strictly below n/2, which is the paper's own condition on the
/// prediction; a* is Eq. (93)'s clamp of Eq. (91) at -sqrt(n)/2.
#[pyfunction]
pub(crate) fn dps_kato(n: f64, m: f64, eps: f64) -> PyResult<(f64, f64)> {
    check_pos("n", n)?;
    check_nonneg("m", m)?;
    check_eps("eps", eps)?;
    if m >= 0.5 * n {
        return Err(PyValueError::new_err(format!(
            "m must be < n/2, got m={m} n={n}: Kato's prediction sits strictly below half the \
             trials; clamp it at floor((n - 1)/2) as N_3* does"
        )));
    }

    let le = eps.ln();
    let rn = n.sqrt();
    let disc = 9.0 * m * (n - m) - 2.0 * n * le;
    let num = 216.0 * rn * m * (n - m) * le - 48.0 * n * rn * le * le
        + 27.0 * 2.0_f64.sqrt() * (n - 2.0 * m) * (-n * n * le * disc).sqrt();
    let den = 4.0 * (9.0 * n - 8.0 * le) * disc;
    let a = (num / den).max(-0.5 * rn);
    let b = (18.0 * a * a * n - (16.0 * a * a + 24.0 * a * rn + 9.0 * n) * le).sqrt()
        / (3.0 * (2.0 * n).sqrt());

    Ok((a, b))
}

/// MTT23 Theorem 3, Eq. (42): the upper bound N_ph^U on the number of phase
/// errors among the code rounds, in rounds. `kato = false` takes their Eq. (53)
/// instead, which bounds the same three-photon sum S_3 by Azuma rather than
/// Kato and is looser at every argument.
///
/// `q1`, `q2` and `q3` are Eq. (3)'s source bounds, `t` Bob's code-round
/// probability, and `e_bit` the error rate over the SAMPLE rounds alone; N_det
/// is `n_code + n_samp` rather than an argument of its own. The bound holds
/// except with probability 3*eps1 + 3*eps2. It is written on the
/// photon-number-resolving receiver `dps_finite` names.
#[pyfunction]
pub(crate) fn dps_phase(
    n_em: f64,
    n_code: f64,
    n_samp: f64,
    e_bit: f64,
    q1: f64,
    q2: f64,
    q3: f64,
    t: f64,
    eps1: f64,
    eps2: f64,
    kato: bool,
) -> PyResult<f64> {
    check_pos("n_em", n_em)?;
    check_nonneg("n_code", n_code)?;
    check_nonneg("n_samp", n_samp)?;
    check_err("e_bit", e_bit)?;
    check_prob("q1", q1)?;
    check_prob("q2", q2)?;
    check_prob("q3", q3)?;
    check_eps("t", t)?;
    check_eps("eps1", eps1)?;
    check_eps("eps2", eps2)?;
    if !(q1 >= q2 && q2 >= q3) {
        return Err(PyValueError::new_err(format!(
            "q1 >= q2 >= q3 is required, got q1={q1} q2={q2} q3={q3}: Eq. (3)'s q_n is the \
             probability of n OR MORE photons in a block; take all three from dps_source"
        )));
    }

    let n_det = n_code + n_samp;
    if !(n_det > 0.0) {
        return Err(PyValueError::new_err(
            "n_code + n_samp must be > 0: no detected round is no statistics, not a zero phase \
             error"
                .to_string(),
        ));
    }

    let lam = dps_lam();
    let m1 = t * q1 * n_em + dps_gamma(q1, t, n_em, eps2);
    let m2 = t * q2 * n_em + dps_gamma(q2, t, n_em, eps2);
    let m3 = t * q3 * n_em + dps_gamma(q3, t, n_em, eps2);
    let inner = if kato {
        let n3 = m3.min(((n_det - 1.0) / 2.0).floor()).max(0.0);
        let (a, b) = dps_kato(n_det, n3, eps1)?;

        m3 * (1.0 + 2.0 * a / n_det.sqrt()) + (b - a) * n_det.sqrt()
    } else {
        m3 + dps_delta(1.0, eps1, n_det)
    };

    // D of Eq. (47). Eq. (53) prints an undefined F: the Azuma branch changes only S_3, so it is D.
    let d = (lam / (1.0 - t) + 1.0).max(1.0 / t + lam + 1.0);
    let bits = lam * t * e_bit * n_samp / (1.0 - t);
    let root = lam * ((m1 + dps_delta(1.0, eps1, n_det)) * inner).sqrt();

    Ok(bits + m2 + root + t * dps_delta(d * d, eps1, n_det))
}

/// MTT23 Corollary 1, Eq. (24): the secret key LENGTH in bits over the whole
/// run, `n_code*[1 - h(N_ph^U/n_code)] - zeta - n_ec - zeta_ec`, with n_ec =
/// `f_ec*n_code*h2(e_bit)` and N_ph^U from `dps_phase`. Clamped at zero and
/// floored, as a length. Composes as
/// eps_sec = 2^-zeta_ec + sqrt(2)*sqrt(3*eps1 + 3*eps2 + 2^-zeta).
///
/// `detector` has NO default: `"pnr"` runs, `"threshold"` is refused.
#[pyfunction]
pub(crate) fn dps_finite(
    n_em: f64,
    n_code: f64,
    n_samp: f64,
    e_bit: f64,
    q1: f64,
    q2: f64,
    q3: f64,
    t: f64,
    eps1: f64,
    eps2: f64,
    zeta: f64,
    zeta_ec: f64,
    f_ec: f64,
    kato: bool,
    detector: &str,
) -> PyResult<f64> {
    match detector {
        "pnr" => (),
        "threshold" => {
            return Err(PyNotImplementedError::new_err(
                "detector=\"threshold\" is refused: the finite-key DPS bound here is Mizutani, \
                 Takeuchi & Tamaki, \"Finite-key security analysis of \
                 differential-phase-shift quantum key distribution\", Phys. Rev. Research 5, \
                 023132 (2023), arXiv:2301.09844, written for a PHOTON-NUMBER-RESOLVING receiver. \
                 Four pieces are missing. (1) A round is detected only when EXACTLY ONE photon \
                 arrives across the block's first and second slots (assumption (B2), step (P1)b); \
                 a threshold pair keeps the two-photon rounds Theorem 3 prices through q_2 and q_3, \
                 moving the bound in the INSECURE direction. src/pnr.rs builds the three-outcome \
                 POVM, but nothing folds it into a resolved count here. (2) THE PROTOCOL IS \
                 BLOCK-WISE, three-pulse blocks under (A1); run_clicks runs the continuous train \
                 of Inoue, Waks & Yamamoto, Phys. Rev. Lett. 89, 037902 (2002), with no block \
                 boundary to read n_code and n_samp from. (3) BOB'S CODE/SAMPLE COIN, step (P2) \
                 at probability t, does not exist here: ClickOut carries one QBER and no t. (4) \
                 (B2) WANTS ONE SYMMETRIC PAIR, and run_clicks accepts eta_d1, dark_d1, dead_d1 \
                 and after_d1. For the threshold train use dps_rate: Waks, Takesue & Yamamoto, \
                 Phys. Rev. A 73, 012344 (2006), asymptotic, individual attacks. Endo, Sasaki, \
                 Takeoka, Fujiwara, Koashi & Sasaki, \"Line-of-sight quantum key distribution \
                 with differential phase shift keying\", New J. Phys. 24, 025008 (2022) is over \
                 threshold detectors but free-space line-of-sight, excluded. Otherwise pass \
                 detector=\"pnr\" with counts from a receiver resolving 0, 1 and 2+ photons; \
                 dps_counts is the paper's Sec. V forward model for one",
            ))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "detector must be \"pnr\" or \"threshold\", got {other:?}"
            )))
        }
    }

    check_fec("f_ec", f_ec)?;
    check_pos("zeta", zeta)?;
    check_pos("zeta_ec", zeta_ec)?;
    if !(n_code > 0.0) {
        return Err(PyValueError::new_err(format!(
            "n_code must be > 0, got {n_code}: the phase error rate is N_ph^U/n_code; keep at \
             least one code round"
        )));
    }

    let u = dps_phase(n_em, n_code, n_samp, e_bit, q1, q2, q3, t, eps1, eps2, kato)?;

    // MTT23 Eq. (8) reads h(x) = 1 past 1/2, where crate::std::h2 returns 0.
    let e_ph = (u / n_code).min(E_CAP);
    let leak = f_ec * n_code * h2(e_bit);
    let len = n_code * (1.0 - h2(e_ph)) - zeta - leak - zeta_ec;

    Ok(len.max(0.0).floor())
}
