use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::decoy::{check_gains, check_order};
use crate::std::{
    check_eps, check_err, check_fec, check_nonneg, check_pos, check_prob, check_unit, h2, sqrt0,
};
use crate::threefry::Stream;

// Key rates for the discrete click protocols: BB84 with weak coherent pulses, and coherent-one-way.
// Bits per emitted pulse, clamped at zero. q1 and e1 arrive from the decoy layer. Each optional
// layer (`cow_*` extinction, `run_basis`, `pol_*`) reduces exactly to the closed form at ideal
// hardware. Threefry stage ids 30 and 31 are the symbol-level run's, clear of the CV pipeline's
// 1..5 and the click module's 20..22.

/// Asymptotic BB84 key rate for a weak-coherent source, bits per pulse:
/// `q_sift * (q1*[1 - h2(e1)] - q_mu*f_ec*h2(e_mu))`. `q1`, `e1` are the decoy layer's worst-case
/// bounds; `q_sift` is 1/2 uniform, near 1 biased.
#[pyfunction]
pub(crate) fn bb84_rate(
    q_sift: f64,
    q_mu: f64,
    e_mu: f64,
    q1: f64,
    e1: f64,
    f_ec: f64,
) -> PyResult<f64> {
    check_unit("q_sift", q_sift)?;
    check_prob("q_mu", q_mu)?;
    check_err("e_mu", e_mu)?;
    check_prob("q1", q1)?;
    check_err("e1", e1)?;
    check_fec("f_ec", f_ec)?;
    check_gains(q1, q_mu)?;

    let rate = q_sift * (q1 * (1.0 - h2(e1)) - q_mu * f_ec * h2(e_mu));

    Ok(rate.max(0.0))
}

/// Lim's supplementary Eq. (B4) at ONE common epsilon, 8 + 1 + 10 + 2 = 21: the 21 in
/// `log2(21/eps_sec)` of his Eq. (1), SQUARED in the random-sampling term of his Eq. (5).
/// `eps_cor` is NOT one of the 21: it reaches `bb84_length` through `log2(2/eps_cor)` alone.
const EPS_TERMS: f64 = 21.0;

/// The per-bound failure probability every estimator in this family is taken at, `eps_sec / 21`.
/// `bb84_length` takes the composed `eps_sec` and divides internally.
#[pyfunction]
pub(crate) fn bb84_eps(eps_sec: f64) -> PyResult<f64> {
    check_eps("eps_sec", eps_sec)?;

    Ok(eps_sec / EPS_TERMS)
}

/// Phase-error rate in the KEY basis, Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A 89, 022307
/// (2014), Eq. (5), random sampling WITHOUT replacement. Capped at 1/2: past it a worse channel
/// reads as a longer key. A vanishing test sample returns 1/2.
#[pyfunction]
pub(crate) fn bb84_phase(v1: f64, s1_test: f64, s1_key: f64, eps_sec: f64) -> PyResult<f64> {
    check_nonneg("v1", v1)?;
    check_nonneg("s1_test", s1_test)?;
    check_nonneg("s1_key", s1_key)?;
    check_eps("eps_sec", eps_sec)?;

    if !(s1_test > 0.0) || !(s1_key > 0.0) {
        return Ok(0.5);
    }

    let b = v1 / s1_test;
    if !(b > 0.0) {
        // No certified test-basis error: the sampling term has no width.
        return Ok(0.0);
    }

    if b >= 1.0 {
        return Ok(0.5);
    }

    let (c, d) = (s1_test, s1_key);
    let mass = (c + d) * (1.0 - b) * b;
    let inner = (c + d) / (c * d * (1.0 - b) * b) * (EPS_TERMS * EPS_TERMS) / (eps_sec * eps_sec);
    let gamma = sqrt0(mass / (c * d * std::f64::consts::LN_2) * inner.log2());

    Ok((b + gamma).min(0.5))
}

/// Finite-key length in BITS for decoy-state BB84 with weak coherent pulses, Lim Eq. (1). Lim's X
/// is the KEY basis and Z the test basis; the letters swap between papers, so arguments say
/// key/test.
///
/// `s0`, `s1` are the key basis's vacuum and single-photon detection COUNTS from `decoy_counts`,
/// `phi` the `bb84_phase` bound, `n_key` the key basis's sifted count, `e_key` its error rate.
/// `s0` enters at FULL weight with no entropy factor. Clamped at zero. The `6` in
/// `6*log2(21/eps_sec)` is 2*3 terms of Lim's supplementary Eq. (B3), not six epsilons.
/// `eps_sec` is a fixed input where Lim sets the fixed point `kappa*l`, `kappa = 1e-15`.
///
/// TWO PRINTING ERRORS IN THE SOURCE, neither an erratum: `lambda_EC` is `n_key * f_ec *
/// h2(e_key)`, a bit COUNT, not the printed "simple function f_EC h(e_obs)" (about 2x over the
/// published figure); and `eta_sys`, not the printed `eta_ch`, reproduces the figure.
///
/// The published bound is not rigorous: Tupkary et al., arXiv:2502.10340v3, find its phase-error
/// estimate "goes through a Taylor approximation step. Thus the estimate is not a true bound".
/// No fix is implemented; the caveat travels with any length this returns.
#[pyfunction]
pub(crate) fn bb84_length(
    s0: f64,
    s1: f64,
    phi: f64,
    n_key: f64,
    e_key: f64,
    f_ec: f64,
    eps_sec: f64,
    eps_cor: f64,
) -> PyResult<f64> {
    check_nonneg("s0", s0)?;
    check_nonneg("s1", s1)?;
    check_err("phi", phi)?;
    check_nonneg("n_key", n_key)?;
    check_err("e_key", e_key)?;
    check_fec("f_ec", f_ec)?;
    check_eps("eps_sec", eps_sec)?;
    check_eps("eps_cor", eps_cor)?;

    if s0 + s1 > n_key {
        return Err(PyValueError::new_err(format!(
            "s0 + s1 must not exceed n_key (got {} > {n_key}): the vacuum and single-photon \
             detections are part of the sifted count they were estimated from",
            s0 + s1
        )));
    }

    let leak = f_ec * n_key * h2(e_key);
    let budget = 6.0 * (EPS_TERMS / eps_sec).log2() + (2.0 / eps_cor).log2();
    let len = s0 + s1 * (1.0 - h2(phi)) - leak - budget;

    Ok(len.max(0.0).floor())
}

/// Asymptotic coherent-one-way key rate, bits per pulse:
/// `q_z * (1 - h2(e_phase) - f_ec*h2(e_bit))`, `q_z` the data-line gain. `e_phase` is an input
/// UPPER BOUND, not a monitoring-line visibility: the zero-error attack keeps both honest.
#[pyfunction]
pub(crate) fn cow_rate(q_z: f64, e_bit: f64, e_phase: f64, f_ec: f64) -> PyResult<f64> {
    check_prob("q_z", q_z)?;
    check_err("e_bit", e_bit)?;
    check_err("e_phase", e_phase)?;
    check_fec("f_ec", f_ec)?;

    let rate = q_z * (1.0 - h2(e_phase) - f_ec * h2(e_bit));

    Ok(rate.max(0.0))
}

/// Finite-key COW, the THREE-sequence protocol `cow_rate` prices. REFUSED; `n_total` is checked
/// first.
#[pyfunction]
pub(crate) fn cow_finite(n_total: f64) -> PyResult<f64> {
    check_pos("n_total", n_total)?;

    Err(PyNotImplementedError::new_err(
        "finite-key COW is refused, and the question is malformed rather than unimplemented. \
         (1) THE EPSILON WOULD COVER EVERYTHING BUT THE THING IT IS \
         FOR: the SUPPLIED e_phase sits outside the budget. Korzh et al., Nat. Photon. 9, 163 \
         (2015) published that composition, and Gonzalez-Payo, Trenyi, Wang & Curty, Phys. Rev. \
         Lett. 125, 260510 (2020) declared it insecure. (2) THERE IS NOTHING TO ESTIMATE: the \
         zero-error attack of Trenyi & Curty, New J. Phys. 23, 093005 (2021), holds the \
         visibility at 1 and the QBER at 0 through a total break. (3) THE ROUTE IS A PROTOCOL \
         CHANGE, NOT A POST-PROCESSING ONE: COW' appends the vacuum sequence |0>|0>, making \
         e_phase derived (Gao, Xie, Gu, Liu, Weng, Li, Yin & Chen, Opt. Express 30, 23783 \
         (2022), arXiv:2107.09329). Use sdp_interval, sdp_sample and sdp_length (Li, Cao, Xie, \
         Yin & Chen, Phys. Rev. Research 6, 013022 (2024), arXiv:2309.16136); their numbers are \
         not comparable with cow_rate's",
    ))
}

/// Upper bound on any COW key rate: `(1 - f_dec) * (1 - exp(-t*mu))`, the probability that Alice
/// sends a bit signal at `mu` and Bob clicks through `t`. A ceiling to assert against, not a rate;
/// the sequential-attack bound sharpens it and is not reproduced here.
#[pyfunction]
pub(crate) fn cow_ceiling(f_dec: f64, t: f64, mu: f64) -> PyResult<f64> {
    check_prob("f_dec", f_dec)?;
    check_unit("t", t)?;
    check_pos("mu", mu)?;

    Ok((1.0 - f_dec) * (1.0 - (-t * mu).exp()))
}

/// Monitoring-line visibility `(p_m0 - p_m1) / (p_m0 + p_m1)`, in `[-1, 1]`, NOT clamped at 0. A
/// diagnostic, not an input to `cow_rate`: visibility 1 is consistent with a full zero-error attack.
#[pyfunction]
pub(crate) fn cow_visibility(p_m0: f64, p_m1: f64) -> PyResult<f64> {
    check_nonneg("p_m0", p_m0)?;
    check_nonneg("p_m1", p_m1)?;

    let total = p_m0 + p_m1;
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "monitoring line never clicked (p_m0 + p_m1 = 0): visibility is undefined",
        ));
    }

    Ok((p_m0 - p_m1) / total)
}

/// Residual mean photon number in the nominally-empty slot: `mu * 10^(-ext_db/10)`, `ext_db > 0`
/// the modulator's on/off contrast in dB. `mu` stays the ON slot: the pair's flux is not held fixed.
#[pyfunction]
pub(crate) fn cow_residual(mu: f64, ext_db: f64) -> PyResult<f64> {
    check_pos("mu", mu)?;
    check_pos("ext_db", ext_db)?;

    Ok(mu * 10f64.powf(-ext_db / 10.0))
}

/// Data-line gain and bit error rate once the empty slot carries `mu_res` photons, correcting
/// `q_sig`, `e_sig`. `p_res` has no dark term: `q_sig` already carries it. Doubles get a random
/// bit; `e_bit` caps at 1/2.
#[pyfunction]
pub(crate) fn cow_data(q_sig: f64, e_sig: f64, mu_res: f64, t: f64) -> PyResult<(f64, f64)> {
    check_prob("q_sig", q_sig)?;
    check_err("e_sig", e_sig)?;
    check_nonneg("mu_res", mu_res)?;
    check_unit("t", t)?;

    let p_res = 1.0 - (-t * mu_res).exp();
    let gain = q_sig + p_res - q_sig * p_res;
    if gain <= 0.0 {
        return Ok((0.0, 0.0));
    }

    let errs =
        e_sig * q_sig * (1.0 - p_res) + (1.0 - q_sig) * p_res + 0.5 * q_sig * p_res;

    Ok((gain, (errs / gain).min(0.5)))
}

/// `(covered, certified)`: `covered = (1 + f_dec) / (1 + f_dec + (1 - f_dec)/r)`, `r` the linear
/// extinction ratio, is the share of emitted photons the coherence check constrains, 1 at perfect
/// extinction. The fringe does not move; `certified = contrast * covered` never enters `cow_rate`.
#[pyfunction]
pub(crate) fn cow_monitor(contrast: f64, ext_db: f64, f_dec: f64) -> PyResult<(f64, f64)> {
    check_prob("contrast", contrast)?;
    check_pos("ext_db", ext_db)?;
    check_prob("f_dec", f_dec)?;

    let leak = 10f64.powf(-ext_db / 10.0);
    let covered = (1.0 + f_dec) / (1.0 + f_dec + (1.0 - f_dec) * leak);

    Ok((covered, contrast * covered))
}

/// Polarisation contrast `1 - 2e` after a Jones rotation of rms angle `sigma` (rad) and a mean
/// differential group delay `dgd` (s) against a pulse of rms width `width` (s). JONES, not
/// Poincare: theta here is 2*theta there, and only the LINEAR axes rotate. `width = 0` needs
/// `dgd = 0`.
#[pyfunction]
pub(crate) fn pol_contrast(sigma: f64, dgd: f64, width: f64) -> PyResult<f64> {
    check_nonneg("sigma", sigma)?;
    check_nonneg("dgd", dgd)?;
    check_nonneg("width", width)?;

    if width <= 0.0 {
        if dgd > 0.0 {
            return Err(PyValueError::new_err(format!(
                "a differential group delay of {dgd} s needs a pulse width to be \
                 compared against: pass width > 0 or dgd = 0"
            )));
        }

        return Ok((-2.0 * sigma * sigma).exp());
    }

    let spread = dgd / width;

    Ok((-2.0 * sigma * sigma - spread * spread / 8.0).exp())
}

/// Time-averaged contrast of a FREE-RUNNING frame diffusing at `rate` rad/sqrt(s), last aligned
/// `interval` s ago: `(1 - exp(-2 rate^2 interval)) / (2 rate^2 interval)`, exactly 1 at 0.
/// An actively tracked receiver is `pol_contrast` at its residual misalignment instead.
#[pyfunction]
pub(crate) fn pol_walk(rate: f64, interval: f64) -> PyResult<f64> {
    check_nonneg("rate", rate)?;
    check_nonneg("interval", interval)?;

    let spread = 2.0 * rate * rate * interval;
    if spread <= 0.0 {
        return Ok(1.0);
    }

    Ok((1.0 - (-spread).exp()) / spread)
}

/// Wrong-detector probability from static optical misalignment and polarisation contrast:
/// `(1 - (1 - 2*misalign) * contrast) / 2`. Contrasts multiply, error rates do not; a perfect
/// contrast returns `misalign` bit for bit rather than through the arithmetic.
#[pyfunction]
pub(crate) fn pol_error(misalign: f64, contrast: f64) -> PyResult<f64> {
    check_err("misalign", misalign)?;
    check_prob("contrast", contrast)?;

    if contrast >= 1.0 {
        return Ok(misalign);
    }

    Ok(0.5 * (1.0 - (1.0 - 2.0 * misalign) * contrast))
}

/// Alice's per-pulse draws: intensity, basis match, bit, photon-on-detector.
const SOURCE: u32 = 30;

/// Bob's per-pulse draws: misalignment coin, one dark coin per detector, double-click squash coin.
/// Unequal dark rates move a threshold, never which coin is drawn.
const READOUT: u32 = 31;

/// Parallel grain default; no chunk size or thread count moves a count.
const GRAIN: usize = 4096;

/// Per-pulse probability that BOTH parties picked the KEY basis, `q^2`, from the sifting factor
/// `sift = q^2 + (1 - q)^2` at bias `q >= 1/2`. A `sift` below 1/2 is unreachable and clamps.
#[pyfunction]
pub(crate) fn key_share(sift: f64) -> PyResult<f64> {
    check_prob("sift", sift)?;

    let q = 0.5 * (1.0 + sqrt0(2.0 * sift - 1.0));

    Ok((q * q).min(sift))
}

/// `key_share` for a THREE-basis run: the inverse of `sixstate_sift`'s `m = b^2 + (1 - b)^2/2`,
/// key share `b^2`, each monitor basis `(1 - b)^2/4`. `m < 1/3` is unreachable and clamps to
/// `b = 1/3`.
fn three_share(sift: f64) -> PyResult<f64> {
    check_prob("sift", sift)?;

    let bias = (1.0 + sqrt0(6.0 * sift - 2.0)) / 3.0;

    Ok((bias * bias).min(sift))
}

/// Running counts of one basis-keyed run, per intensity. `key`/`key_err` are the KEY-basis subset
/// of `sifted`/`errors`, `mon`/`mon_err` the FIRST monitor basis's; the second is the difference.
#[derive(Clone, Default)]
struct Tally {
    sent: [u64; 3],
    clicks: [u64; 3],
    sifted: [u64; 3],
    errors: [u64; 3],
    key: [u64; 3],
    key_err: [u64; 3],
    mon: [u64; 3],
    mon_err: [u64; 3],
    det: [u64; 2],
    doubles: u64,
}

impl Tally {
    fn merge(mut self, other: Self) -> Self {
        for i in 0..3 {
            self.sent[i] += other.sent[i];
            self.clicks[i] += other.clicks[i];
            self.sifted[i] += other.sifted[i];
            self.errors[i] += other.errors[i];
            self.key[i] += other.key[i];
            self.key_err[i] += other.key_err[i];
            self.mon[i] += other.mon[i];
            self.mon_err[i] += other.mon_err[i];
        }
        for j in 0..2 {
            self.det[j] += other.det[j];
        }
        self.doubles += other.doubles;

        self
    }
}

/// Everything one basis-keyed run reports; every count is an exact integer.
#[pyclass]
pub(crate) struct BasisOut {
    n_pulses: u64,
    sent: [u64; 3],
    clicks: [u64; 3],
    sifted: [u64; 3],
    errors: [u64; 3],
    key: [u64; 3],
    key_err: [u64; 3],
    mon: [u64; 3],
    mon_err: [u64; 3],
    det: [u64; 2],
    doubles: u64,
    gain: [f64; 3],
    qber: [f64; 3],
    sift_rate: f64,
    key_share: f64,
    bases: u8,
}

#[pymethods]
impl BasisOut {
    /// Pulses emitted, across all three intensities.
    #[getter]
    fn n_pulses(&self) -> u64 {
        self.n_pulses
    }

    /// Pulses emitted at each intensity, in the order (signal, decoy, weak).
    #[getter]
    fn sent(&self) -> Vec<u64> {
        self.sent.to_vec()
    }

    /// Per intensity, pulses with at least one detector firing: the gain's numerator, unsifted.
    #[getter]
    fn clicks(&self) -> Vec<u64> {
        self.clicks.to_vec()
    }

    /// Per intensity, clicks surviving basis reconciliation; doubles included, with the squash bit.
    #[getter]
    fn sifted(&self) -> Vec<u64> {
        self.sifted.to_vec()
    }

    /// Sifted bits at each intensity that disagreed with Alice's.
    #[getter]
    fn errors(&self) -> Vec<u64> {
        self.errors.to_vec()
    }

    /// Per intensity, the KEY-basis subset of `sifted`. Key, X and Y partition `sifted` exactly.
    #[getter]
    fn key_sifted(&self) -> Vec<u64> {
        self.key.to_vec()
    }

    /// Per intensity, the KEY-basis subset of `errors`.
    #[getter]
    fn key_errors(&self) -> Vec<u64> {
        self.key_err.to_vec()
    }

    /// Per intensity, `sifted - key_sifted`: the TEST-basis counts the phase-error bound reads.
    #[getter]
    fn test_sifted(&self) -> Vec<u64> {
        let mut out = self.sifted;
        for i in 0..3 {
            out[i] -= self.key[i];
        }

        out.to_vec()
    }

    /// Per intensity, `errors - key_errors`.
    #[getter]
    fn test_errors(&self) -> Vec<u64> {
        let mut out = self.errors;
        for i in 0..3 {
            out[i] -= self.key_err[i];
        }

        out.to_vec()
    }

    /// Per intensity, the FIRST monitor basis's share of `sifted`: the whole test basis in a
    /// two-basis run, where this equals `test_sifted`; the X basis in a three-basis one.
    #[getter]
    fn x_sifted(&self) -> Vec<u64> {
        self.mon.to_vec()
    }

    /// Per intensity, the first monitor basis's errors: `x_sifted`'s numerator for `e_x`.
    #[getter]
    fn x_errors(&self) -> Vec<u64> {
        self.mon_err.to_vec()
    }

    /// Per intensity, `sifted - key_sifted - x_sifted`: the SECOND monitor basis. Zero throughout
    /// a two-basis run, not a missing count.
    #[getter]
    fn y_sifted(&self) -> Vec<u64> {
        let mut out = self.sifted;
        for i in 0..3 {
            out[i] -= self.key[i] + self.mon[i];
        }

        out.to_vec()
    }

    /// Per intensity, `errors - key_errors - x_errors`.
    #[getter]
    fn y_errors(&self) -> Vec<u64> {
        let mut out = self.errors;
        for i in 0..3 {
            out[i] -= self.key_err[i] + self.mon_err[i];
        }

        out.to_vec()
    }

    /// How many mutually unbiased bases the run sifted into, 2 or 3.
    #[getter]
    fn bases(&self) -> u8 {
        self.bases
    }

    /// Per-pulse probability that both parties picked the key basis, derived from `sift`:
    /// `key_share`'s inversion at `bases = 2`, `three_share`'s at `bases = 3`, and they differ.
    #[getter]
    fn key_share(&self) -> f64 {
        self.key_share
    }

    /// Sifted slots where both detectors fired, pooled; also counted inside `sifted`.
    #[getter]
    fn doubles(&self) -> u64 {
        self.doubles
    }

    /// Pulses in which detector 0 fired, pooled over intensities, sifted or not. A pulse firing
    /// both counts in each: the pair sums to the clicks plus every double, not to the clicks.
    #[getter]
    fn clicks_d0(&self) -> u64 {
        self.det[0]
    }

    /// Detector 1's count, as `clicks_d0`.
    #[getter]
    fn clicks_d1(&self) -> u64 {
        self.det[1]
    }

    /// Gain per intensity, clicks / sent: the measured `Q_mu` the decoy bounds invert; 0 if unsent.
    #[getter]
    fn gain(&self) -> Vec<f64> {
        self.gain.to_vec()
    }

    /// Error rate per intensity, errors / sifted: the measured `E_mu`.
    #[getter]
    fn qber(&self) -> Vec<f64> {
        self.qber.to_vec()
    }

    /// Sifted bits per click, pooled: an estimator of the sifting factor's Bernoulli thinning.
    #[getter]
    fn sift_rate(&self) -> f64 {
        self.sift_rate
    }
}

/// Simulate one basis-keyed run of `n` phase-randomised weak coherent pulses. `chunk` sets the
/// parallel grain only (0 takes the default). Counted, not integrated: against `decoy_gain` the
/// simulated error rate is lower by about `dark * (1 - exp(-t*eta*mu))`, exact at `dark = 0`.
/// Dead time and afterpulsing are absent.
///
/// `eta` and `dark` are detector 0's, and detector 1's too unless `eta_d1`/`dark_d1` say
/// otherwise; an equal pair is the same run bit for bit. `bases` is 2 (BB84) or 3 (six-state);
/// `misalign_x`/`misalign_y` are the MONITOR bases' rates, `misalign` the key basis's.
///
/// ONE FLIP RATE IN EVERY BASIS IS THE DEPOLARISING CHANNEL: left unset, the monitor rates are the
/// key basis's and `sixstate_bell` reads the counters' SAMPLING NOISE as a Bell-diagonal spectrum
/// until its triangle inequality fails. Pass asymmetric rates. At `bases = 3` the three are held
/// to `sixstate_bell`'s inequalities.
#[pyfunction]
#[pyo3(signature = (
    n,
    seed,
    intensities,
    probs,
    t,
    eta,
    dark,
    misalign,
    sift,
    chunk,
    eta_d1 = None,
    dark_d1 = None,
    bases = 2,
    misalign_x = None,
    misalign_y = None,
))]
pub(crate) fn run_basis(
    py: Python<'_>,
    n: u64,
    seed: u64,
    intensities: (f64, f64, f64),
    probs: (f64, f64, f64),
    t: f64,
    eta: f64,
    dark: f64,
    misalign: f64,
    sift: f64,
    chunk: u64,
    eta_d1: Option<f64>,
    dark_d1: Option<f64>,
    bases: u8,
    misalign_x: Option<f64>,
    misalign_y: Option<f64>,
) -> PyResult<BasisOut> {
    let mu = [intensities.0, intensities.1, intensities.2];
    let p = [probs.0, probs.1, probs.2];
    check_pos("mu", mu[0])?;
    check_pos("nu1", mu[1])?;
    check_nonneg("nu2", mu[2])?;
    check_unit("t", t)?;
    check_unit("eta", eta)?;
    check_prob("dark", dark)?;

    let etas = [eta, eta_d1.unwrap_or(eta)];
    let darks = [dark, dark_d1.unwrap_or(dark)];
    check_unit("eta_d1", etas[1])?;
    check_prob("dark_d1", darks[1])?;

    // Slot 0 the key basis, 1 and 2 the monitor bases, 3 unmatched: its flip reaches only Bob's
    // detector tallies.
    let miss = [
        misalign,
        misalign_x.unwrap_or(misalign),
        misalign_y.unwrap_or(misalign),
        misalign,
    ];
    check_err("misalign", misalign)?;
    check_err("misalign_x", miss[1])?;
    check_err("misalign_y", miss[2])?;
    check_unit("sift", sift)?;
    if bases != 2 && bases != 3 {
        return Err(PyValueError::new_err(format!(
            "bases must be 2 (one conjugate pair) or 3 (the three mutually unbiased \
             bases), got {bases}"
        )));
    }

    if bases == 2 && misalign_y.is_some() {
        return Err(PyValueError::new_err(
            "misalign_y is the second monitor basis's rate and a two-basis run has none: \
             pass bases = 3, or leave it unset",
        ));
    }

    // `sixstate_bell`'s triangle inequalities apply to the RUN.
    if bases == 3 {
        for (a, b, c, pair) in [
            (miss[1], miss[2], miss[0], "misalign_x + misalign_y"),
            (miss[2], miss[0], miss[1], "misalign_y + misalign"),
            (miss[0], miss[1], miss[2], "misalign + misalign_x"),
        ] {
            if a + b < c {
                return Err(PyValueError::new_err(format!(
                    "{pair} must be at least the third rate, got {a} + {b} < {c}: no Pauli \
                     channel has those three error rates; pass rates obeying the triangle \
                     inequality"
                )));
            }
        }
    }

    for (name, x) in [("p_mu", p[0]), ("p_nu1", p[1]), ("p_nu2", p[2])] {
        check_prob(name, x)?;
    }

    // `decoy_bounds`'s check and message.
    check_order(mu[1], mu[2], mu[0])?;

    let total = p[0] + p[1] + p[2];
    if (total - 1.0).abs() > 1e-9 {
        return Err(PyValueError::new_err(format!(
            "intensity probabilities must sum to 1, got {total}"
        )));
    }

    if n == 0 {
        return Err(PyValueError::new_err("n must be at least one pulse"));
    }

    let used = n as usize;
    let grain = if chunk == 0 { GRAIN } else { chunk as usize }.min(used);
    let src = Stream::new(seed, SOURCE);
    let rd = Stream::new(seed, READOUT);

    // Per detector. One expression per entry: an equal pair is the same f64, not a near one.
    let hit = [
        [1.0 - (-t * etas[0] * mu[0]).exp(), 1.0 - (-t * etas[1] * mu[0]).exp()],
        [1.0 - (-t * etas[0] * mu[1]).exp(), 1.0 - (-t * etas[1] * mu[1]).exp()],
        [1.0 - (-t * etas[0] * mu[2]).exp(), 1.0 - (-t * etas[1] * mu[2]).exp()],
    ];
    let edge = [p[0], p[0] + p[1]];

    // The monitor bases split the non-key share evenly; a two-basis run leaves slot 2 empty.
    let share = if bases == 3 {
        three_share(sift)?
    } else {
        key_share(sift)?
    };
    let mid = if bases == 3 {
        share + 0.5 * (sift - share)
    } else {
        sift
    };

    let counts = py.detach(|| {
        (0..used)
            .into_par_iter()
            .with_min_len(grain)
            .fold(Tally::default, |mut acc, k| {
                let u = src.uniforms(k as u64);
                let v = rd.uniforms(k as u64);
                let idx = if u[0] < edge[0] {
                    0
                } else if u[0] < edge[1] {
                    1
                } else {
                    2
                };
                acc.sent[idx] += 1;

                let bit = usize::from(u[2] < 0.5);
                let mut fired = [v[1] < darks[0], v[2] < darks[1]];

                // The SAME draw decides basis and sifting; read before the sifting return, the
                // misalignment rate being per basis.
                let slot = if u[1] < share {
                    0
                } else if u[1] < mid {
                    1
                } else if u[1] < sift {
                    2
                } else {
                    3
                };

                // Alice's bit's detector, or its partner when misaligned, at THAT detector's
                // efficiency.
                let lands = bit ^ usize::from(v[0] < miss[slot]);
                if u[3] < hit[idx][lands] {
                    fired[lands] = true;
                }

                if !(fired[0] || fired[1]) {
                    return acc;
                }

                acc.clicks[idx] += 1;
                acc.det[0] += u64::from(fired[0]);
                acc.det[1] += u64::from(fired[1]);
                if u[1] >= sift {
                    return acc;
                }

                acc.sifted[idx] += 1;
                let read = if fired[0] && fired[1] {
                    acc.doubles += 1;

                    usize::from(v[3] < 0.5)
                } else {
                    usize::from(fired[1])
                };
                let wrong = u64::from(read != bit);
                acc.errors[idx] += wrong;

                // Slot 2 keeps no counter: it is the remainder.
                if slot == 0 {
                    acc.key[idx] += 1;
                    acc.key_err[idx] += wrong;
                } else if slot == 1 {
                    acc.mon[idx] += 1;
                    acc.mon_err[idx] += wrong;
                }

                acc
            })
            .reduce(Tally::default, Tally::merge)
    });

    let mut gain = [0.0f64; 3];
    let mut qber = [0.0f64; 3];
    for i in 0..3 {
        if counts.sent[i] > 0 {
            gain[i] = counts.clicks[i] as f64 / counts.sent[i] as f64;
        }

        if counts.sifted[i] > 0 {
            qber[i] = counts.errors[i] as f64 / counts.sifted[i] as f64;
        }
    }

    let clicked: u64 = counts.clicks.iter().sum();
    let kept: u64 = counts.sifted.iter().sum();

    Ok(BasisOut {
        n_pulses: n,
        sent: counts.sent,
        clicks: counts.clicks,
        sifted: counts.sifted,
        errors: counts.errors,
        key: counts.key,
        key_err: counts.key_err,
        mon: counts.mon,
        mon_err: counts.mon_err,
        det: counts.det,
        doubles: counts.doubles,
        gain,
        qber,
        sift_rate: if clicked > 0 {
            kept as f64 / clicked as f64
        } else {
            0.0
        },
        key_share: share,
        bases,
    })
}
