use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::postselect::N_MAX;
use crate::std::{check_eps, check_err, check_finite, check_nonneg, check_pos, check_prob, h2};

// The entropic uncertainty relation and the composable key length it carries. NOT WIRED:
// `eur_family` refuses every protocol this tree ships by name.
//
// THREE STATEMENTS THAT MUST NOT STAND IN FOR EACH OTHER.
//
//   (1) MAASSEN & UFFINK, "Generalized entropic uncertainty relations", Phys. Rev. Lett. 60,
//       1103 (1988), no preprint (1988 predates arXiv; `dev/novel.md` W3 and W7):
//
//           H(X) + H(Z)  >=  q,     q := log2(1/c),   c := max_{x,z} |<x|z>|^2,
//
//       memoryless adversary. `eur_quality`, `eur_conjugate` and `eur_misalign` compute `q`; no
//       rate here reads this form.
//
//   (2) BERTA, CHRISTANDL, COLBECK, RENES & RENNER, "The uncertainty principle in the presence
//       of quantum memory", Nature Physics 6, 659 (2010), arXiv:0909.0950, Eq. (2):
//
//           H(X|B) + H(Z|B)  >=  q + H(A|B),
//
//       and the two-adversary form a key rate reads, `H(X|B) + H(Z|C) >= q`, conjectured by
//       RENES & BOILEAU, "Conjectured strong complementary information tradeoff", Phys. Rev.
//       Lett. 103, 020402 (2009), arXiv:0806.3984. `eur_memory` is the first, `eur_asymptotic`
//       the second through Devetak-Winter -- COLLECTIVE attacks only, in Berta et al.'s words:
//       "The argument given here applies only to collective attacks but can be extended to
//       arbitrary attacks using the post-selection technique". QUOTE THE WHOLE SENTENCE: its
//       route is `postselect.rs`, blocked by `ps_family`.
//
//   (3) TOMAMICHEL & RENNER, "The uncertainty relation for smooth entropies", Phys. Rev. Lett.
//       106, 110506 (2011), arXiv:1009.2015, Theorem 1:
//
//           H_min^eps(X|B) + H_max^eps(Z|C)  >=  q,
//
//       coherent attacks with no de Finetti, postselection or equipartition step. `eur_smooth`.
//
// The key length is TOMAMICHEL, LIM, GISIN & RENNER, "Tight finite-key analysis for quantum
// cryptography", Nature Communications 3, 634 (2012), arXiv:1103.4130, Eq. (2):
//
//      l  <=  n (q - h2(Q_tol + mu)) - leak_EC - log2(2 / (eps_s^2 eps_c)),
//      mu := sqrt( ((n + k) / (n k)) ((k + 1) / k) ln(4 / eps_s) ),
//
// chaining (3) for `H_min`, Serfling on a random subset of one string for `H_max`, and the quantum
// leftover hash lemma -- Tomamichel, Schaffner, Smith & Renner, "Leftover hashing against quantum
// side information", IEEE Trans. Inf. Theory 57, 5524 (2011), arXiv:1002.2436. `eur_length` is
// the formula, `eur_secret` the formula behind the family gate.
//
// The CV overlap: FURRER, BERTA, TOMAMICHEL, SCHOLZ & CHRISTANDL, "Position-momentum uncertainty
// relations in the presence of quantum memory", J. Math. Phys. 55, 122205 (2014),
// arXiv:1308.4527, `c(dq, dp) = (dq dp / 2pi) S_0^(1)(1, dq dp / 4)^2`, `S_0^(1)` the 0th radial
// prolate spheroidal wave function of the first kind. `eur_binned` ships it at `S = 1`, a bound;
// a quadrature of the sinc kernel certifies neither side, and a `q` too large is insecure. TWO
// PRINTINGS DISAGREE: Sec. II's Eq. (10), `(dq dp)/2 * S^2`, is a factor of pi out and Eq. (9)
// drops the square; TAKE Sec. V C's Eq. (113), `(1/2pi) dq dp * S^2`, the only printing in v1.
// Eq. (10)'s larger `c` errs safe by `log2(pi) = 1.65` bits. Convention `[Q, P] = i`, `hbar = 1`,
// vacuum variance 1/2 -- this tree's, NOT the CV literature's `hbar = 2`.
//
// Requirements, which `eur_family`'s refusals name by number:
//   C1  TWO POVMs ON ONE SYSTEM A with a state-independent overlap.
//   C2  A CERTIFIED PREPARATION QUALITY `c = max_{x,z} ||sqrt(M_x) sqrt(N_z)||_inf^2`, a property
//       of ALICE'S SOURCE: TLGR's Supplementary Note 1 admits a qubit source or one preparing by
//       measuring half of an entangled pair, nothing else. BOB IS EXEMPT -- ONE-SIDED device
//       independence.
//   C3  THE COMPLEMENTARY OUTCOME SAMPLED FROM THE SAME STRING, `H_max^eps(Z|Z')` from a random
//       subset of ONE `n + k` string by Serfling.
//   C4  A MEMORYLESS RECEIVER -- Tomamichel & Renner, "no assumption about Bob's measurement
//       device is required except that it is memoryless".

/// A mutually unbiased PAIR exists in every dimension, so this bounds the argument, not the
/// physics; a COMPLETE set of d + 1 needs a prime power and is not claimed.
const D_MAX: u32 = 1 << 20;

/// `2 pi` in `hbar = 1`, at and above which the closed-form quality is 0.
const AREA_MAX: f64 = std::f64::consts::TAU;

/// Not `postselect::check_block`, which also demands integrality; both refusal literals are pinned.
fn check_block(name: &str, n: f64) -> PyResult<()> {
    if n.is_finite() && n >= 1.0 && n <= N_MAX {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a count in [1, {N_MAX:e}], got {n}"
        )))
    }
}

/// `q = -log2 c` bits, `c = max_{x,z} ||sqrt(M_x) sqrt(N_z)||_inf^2` (TLGR Sec. I.B). PASS AN
/// UPPER BOUND ON `c`: `q` enters every consumer with a PLUS sign.
#[pyfunction]
pub(crate) fn eur_quality(c: f64) -> PyResult<f64> {
    check_prob("c", c)?;

    if c <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "c must be > 0, got {c}: two POVMs resolving the same identity overlap at least 1/d"
        )));
    }

    let q = -c.log2();
    if !q.is_finite() || q < 0.0 {
        return Err(PyValueError::new_err(format!(
            "q came out {q} at c = {c}, which is not a non-negative finite quality"
        )));
    }

    Ok(q)
}

/// Two mutually unbiased bases in dimension `d`: `c = 1/d` exactly, `q = log2 d`.
#[pyfunction]
pub(crate) fn eur_conjugate(d: u32) -> PyResult<f64> {
    if !(2..=D_MAX).contains(&d) {
        return Err(PyValueError::new_err(format!(
            "d must be a dimension in [2, {D_MAX}], got {d}: one basis has no complementary \
             partner"
        )));
    }

    eur_quality(1.0 / (d as f64))
}

/// Two rank-1 qubit bases whose Bloch axes subtend `theta` rad: `c = (1 + |cos theta|) / 2`,
/// `q = 1` at `pi/2` and 0 at 0. PASS THE ANGLE FURTHEST FROM pi/2 the source's specification
/// allows; `q.impairments.Modulator`'s angle enters here and nowhere else in this file.
#[pyfunction]
pub(crate) fn eur_misalign(theta: f64) -> PyResult<f64> {
    check_finite("theta", theta)?;

    eur_quality(0.5 * (1.0 + theta.cos().abs()))
}

/// Berta et al. Eq. (2), right-hand side `q + H(A|B)` in bits. `h_ab` is a conditional entropy,
/// negative exactly when A and B are entangled (`-log2 d` at a maximally entangled pair, cancelling
/// `q`). Returned raw, negatives included.
#[pyfunction]
pub(crate) fn eur_memory(q: f64, h_ab: f64) -> PyResult<f64> {
    check_nonneg("q", q)?;
    check_finite("h_ab", h_ab)?;

    Ok(q + h_ab)
}

/// Asymptotic rate against COLLECTIVE attacks, bits per sifted signal: `K >= q - h2(e_bit) -
/// h2(e_phase)`, Berta et al.'s `K >= log2(1/c) - H(R|R') - H(S|S')`. The coherent-attack
/// statement is `eur_length`, a LENGTH, and this rate does not convert into it. At `q = 1`,
/// `e_bit = e_phase = e` it is `1 - 2 h2(e)`, zero at 11.0028%. Returned raw, negatives included.
#[pyfunction]
pub(crate) fn eur_asymptotic(q: f64, e_bit: f64, e_phase: f64) -> PyResult<f64> {
    check_nonneg("q", q)?;
    check_err("e_bit", e_bit)?;
    check_err("e_phase", e_phase)?;

    Ok(q - h2(e_bit) - h2(e_phase))
}

/// Tomamichel & Renner Theorem 1 over `n` rounds: `H_min^eps(X|E) >= n q - H_max^eps(Z|Z')` bits.
/// `h_max` MUST BE AN UPPER BOUND on the smooth max-entropy: a min-entropy in its place reports
/// an Eve more ignorant than the record forces. Returned raw, negatives included.
#[pyfunction]
pub(crate) fn eur_smooth(n: f64, q: f64, h_max: f64) -> PyResult<f64> {
    check_block("n", n)?;
    check_nonneg("q", q)?;
    check_nonneg("h_max", h_max)?;

    let total = n * q;
    let bound = total - h_max;

    if !(bound <= total) {
        return Err(PyValueError::new_err(format!(
            "the min-entropy bound {bound} is not at or below n q = {total}: the max-entropy is \
             SPENT"
        )));
    }

    Ok(bound)
}

/// TLGR Eq. (2)'s `mu = sqrt( ((n + k) / (n k)) ((k + 1) / k) ln(4 / eps_s) )`, added to the
/// tolerated error before `h2`. SERFLING, NOT HOEFFDING: `k` positions WITHOUT replacement from
/// `n + k`, whence `(n + k)/(n k)`; the with-replacement `sqrt(ln(1/eps) / 2k)` is smaller, the
/// insecure direction. NATURAL logarithm, as TLGR print it.
#[pyfunction]
pub(crate) fn eur_serfling(n: f64, k: f64, eps_s: f64) -> PyResult<f64> {
    check_block("n", n)?;
    check_block("k", k)?;
    check_eps("eps_s", eps_s)?;

    let share = (n + k) / (n * k);
    let bias = (k + 1.0) / k;
    let mu = (share * bias * (4.0 / eps_s).ln()).sqrt();

    if !mu.is_finite() || mu <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "mu came out {mu} at n = {n}, k = {k}, eps_s = {eps_s}, which is not a positive \
             finite half-width"
        )));
    }

    Ok(mu)
}

/// `log2 gamma(t)` bits, the smooth max-entropy price of an average bin distance `t` between
/// Alice's and Bob's discretised quadratures: Furrer, Franz, Berta, Leverrier, Scholz, Tomamichel
/// & Werner, Phys. Rev. Lett. 109, 100502 (2012) + Erratum Phys. Rev. Lett. 112, 019902 (2014),
/// arXiv:1112.2179, Appendix, `gamma(t) = (t + sqrt(1 + t^2)) (t / (sqrt(1 + t^2) - 1))^t`. The
/// printed grouping divides by zero below t ~ 1e-8, so the conjugate form runs: `asinh(t)/ln2 +
/// t log2((1 + sqrt(1 + t^2)) / t)`. The CV counterpart of `h2(Q_tol + mu)`.
#[pyfunction]
pub(crate) fn eur_ball(t: f64) -> PyResult<f64> {
    check_nonneg("t", t)?;

    if t == 0.0 {
        return Ok(0.0);
    }

    let root = (1.0 + t * t).sqrt();
    let cost = t.asinh() / std::f64::consts::LN_2 + t * ((1.0 + root) / t).log2();

    if !cost.is_finite() || cost < 0.0 {
        return Err(PyValueError::new_err(format!(
            "log2 gamma({t}) came out {cost}, which is not a non-negative finite cost"
        )));
    }

    Ok(cost)
}

/// Binned homodyne, `q >= log2(2 pi / (dq dp))` bits at bin widths `dq`, `dp` in `hbar = 1`. A
/// THEOREM, not the small-bin approximation the papers call it: the overlap is the largest
/// eigenvalue of the trace-class `Q[dq] P[dp] Q[dq]`, whose trace is `dq dp / 2pi`, so this `q`
/// is at or below the exact `-log2 c(dq, dp)` -- short by 1.0e-10 bits at `dq = dp = 0.01` and
/// 9.99e-3 at 1.0. Refuses `dq dp >= 2 pi` rather than returning 0, which may be subtracted from.
#[pyfunction]
pub(crate) fn eur_binned(dq: f64, dp: f64) -> PyResult<f64> {
    check_pos("dq", dq)?;
    check_pos("dp", dp)?;

    let area = dq * dp;
    if !(area < AREA_MAX) {
        return Err(PyValueError::new_err(format!(
            "dq dp = {area} is at or above 2 pi = {AREA_MAX}, where the closed-form quality is at \
             or below zero and certifies nothing; the exact constant there (0.357 bits at \
             dq = dp = 2.5) needs the radial prolate spheroidal wave function, not implemented"
        )));
    }

    eur_quality(area / AREA_MAX)
}

/// TLGR Eq. (2), the `eps_s`-secret key LENGTH in bits: `l = n (q - h2(Q_tol + mu)) - leak_EC -
/// log2(2 / (eps_s^2 eps_c))`, `mu` from `eur_serfling`; composable parameter `eps_c + eps_s`,
/// against COHERENT attacks. Over `n` it is bits per raw-key round, not per signal (TLGR divide by
/// `M(n, k)`). `h2` saturates at 1 once `Q_tol + mu` reaches 1/2. Returned raw, negatives included.
#[pyfunction]
pub(crate) fn eur_length(
    n: f64,
    k: f64,
    q: f64,
    q_tol: f64,
    leak: f64,
    eps_s: f64,
    eps_c: f64,
) -> PyResult<f64> {
    check_nonneg("q", q)?;
    check_err("q_tol", q_tol)?;
    check_nonneg("leak", leak)?;
    check_eps("eps_c", eps_c)?;

    let mu = eur_serfling(n, k, eps_s)?;
    let seen = q_tol + mu;
    let cost = if seen >= 0.5 { 1.0 } else { h2(seen) };

    let total = n * q;
    let slack = (2.0 / (eps_s * eps_s * eps_c)).log2();
    let length = total - n * cost - leak - slack;

    if !length.is_finite() || !(length <= total) {
        return Err(PyValueError::new_err(format!(
            "the key length {length} is not at or below n q = {total}: every correction here is \
             SUBTRACTED"
        )));
    }

    Ok(length)
}

/// The gate. `"qubit"` (TLGR's own source in two conjugate bases, `q = 1`) returns a number and is
/// not a path here; every shipped family is refused naming the C1 to C4 it fails.
#[pyfunction]
pub(crate) fn eur_family(name: &str) -> PyResult<f64> {
    match name {
        "qubit" => eur_conjugate(2),

        "bb84" => Err(PyNotImplementedError::new_err(
            "bb84: C2 fails, and C3 with it. qkd's BB84 is weak-coherent with a decoy layer, and \
             TLGR admit a qubit source or one preparing by measuring half of an entangled pair, \
             printing numerics for \"a perfect single-photon source, i.e. q = 1\". C3: the test \
             statistic is a sifted QBER from decoy gains, not a Serfling subset of one Z-string. \
             Use `bb84_length`, Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A 89, 022307 (2014), \
             already valid against general attacks",
        )),

        "sixstate" => Err(PyNotImplementedError::new_err(
            "sixstate: C2 fails on the weak-coherent source, as for bb84. C1 holds: the relation \
             reads exactly TWO POVMs, and Wang, Yin, Liu, Wang, Chen, Guo & Han, Phys. Rev. \
             Research 3, 023019 (2021), arXiv:2008.03510, Eq. (A.19) reads all three bases through \
             a CONJUGATE STRING, M0 = Y or M1 = Z at position i by whether the key basis erred \
             there, at overlap 1/2^n. For a single-photon source use `sixstate_finite`, on Scarani \
             & Renner, Phys. Rev. Lett. 100, 200501 (2008), arXiv:0708.0709",
        )),

        "sarg" => Err(PyNotImplementedError::new_err(
            "sarg: C1 fails: the announcement is a PAIR, so the sifted quantity is not a second \
             POVM on Alice's system. `sarg_rate` pays privacy amplification on Q_1 AND Q_2 -- \
             Fung, Tamaki & Lo, Phys. Rev. A 73, 012337 (2006), Eq. (39)",
        )),

        "b92" => Err(PyNotImplementedError::new_err(
            "b92: C1 fails. Two non-orthogonal states and one measurement: no second basis, no \
             overlap c, no q. `b92_phase` derives the phase error from the loss and the state \
             overlap by discrimination",
        )),

        "cow" | "dps" => Err(PyNotImplementedError::new_err(
            "cow/dps: C4 fails first and C1 second: a delay interferometer reads CONSECUTIVE \
             pulses, so the receiver is not memoryless, and the visibility is not a POVM on the \
             signal system. The route is a PROTOCOL CHANGE and both ship. COW' with the vacuum \
             decoy sequence (Gao, Xie, Gu, Liu, Weng, Li, Yin & Chen, Opt. Express 30, 23783 \
             (2022), arXiv:2107.09329; Li, Cao, Xie, Yin & Chen, Phys. Rev. Research 6, 013022 \
             (2024), arXiv:2309.16136) is `sdp_length`; DPS (Mizutani, Takeuchi & Tamaki, Phys. \
             Rev. Research 5, 023132 (2023), arXiv:2301.09844, Sec. IV.1) is `dps_finite`. \
             `cow_rate` reads the THREE-sequence protocol, where e_phase stays supplied",
        )),

        "rrdps" => Err(PyNotImplementedError::new_err(
            "rrdps: C3 fails structurally, and C4 with it: RRDPS monitors no disturbance, so there \
             is no complementary-basis sample, and the receiver compares two pulses chosen after \
             the fact. Use `rrdps_length`, a finite key on a different argument",
        )),

        "mdi" | "cvmdi" => Err(PyNotImplementedError::new_err(
            "mdi/cvmdi: C1 fails across parties: the key statistic is a COINCIDENCE at an \
             untrusted relay, not two POVMs on one system; C2 fails at both weak-coherent senders. \
             Use `mdi_length`, Curty, Xu, Cui, Lim, Tamaki & Lo, Nature Communications 5, 3732 \
             (2014), arXiv:1307.1081, already against general attacks; `cvmdi_rate` is asymptotic",
        )),

        "pairing" => Err(PyNotImplementedError::new_err(
            "pairing: C3 and C4 fail together: rounds pair AFTER the announcement, so there is no \
             fixed n-round string for Serfling and the receiver reads across rounds -- the \
             property that fails ps_family(\"pairing\")",
        )),

        "bbm92" | "e91" => Err(PyNotImplementedError::new_err(
            "bbm92/e91: C1, C2 and C4 hold, TLGR's entanglement-based source at q = 1. Missing: \
             `q.PairLink` ships no key length, and post-selecting on coincidence at threshold \
             detectors IS the fair-sampling assumption. The relation is ONE-SIDED device \
             independent, no step toward what `ekert.rs` rules out",
        )),

        "gaussian" | "cv" => Err(PyNotImplementedError::new_err(
            "gaussian/cv: C2 and C3 fail on the PROTOCOL SHAPE. Furrer, Franz, Berta, Leverrier, \
             Scholz, Tomamichel & Werner, Phys. Rev. Lett. 109, 100502 (2012) + Erratum Phys. Rev. \
             Lett. 112, 019902 (2014), arXiv:1112.2179, need a TRUSTED EPR SOURCE in Alice's lab, \
             BINNED HOMODYNE on both sides and DIRECT reconciliation: \"Placing the trusted source \
             in Alice's lab also implies that the analysis is not compatible with reverse \
             reconciliation\". qkd's Gaussian link is coherent-state and reverse-reconciled, and \
             `cv_finite` is Leverrier, Grosshans & Grangier, Phys. Rev. A 81, 062343 (2010), \
             collective-only. Their epsilon term is O(log 1/(eps_s eps_c)), with no constant to \
             transcribe",
        )),

        "dmcs" => Err(PyNotImplementedError::new_err(
            "dmcs: C1 fails. M-PSK has one heterodyne receiver and no complementary basis; \
             `dm_secure` bounds a relative entropy over region operators on a different cone \
             (`herm.rs`), and the two proofs share no q",
        )),

        "squeezed" => Err(PyNotImplementedError::new_err(
            "squeezed: not a protocol qkd ships, and q is not a constant for it -- a trusted EPR \
             source read by binned homodyne has a quality set by the bin widths: call \
             eur_binned(dq, dp)",
        )),

        _ => Err(PyValueError::new_err(format!(
            "unknown family {name:?}. \"qubit\" (TLGR's source in two conjugate bases) returns a \
             quality and is not a path here; every shipped family is refused by name: bb84, \
             sixstate, sarg, b92, cow, dps, rrdps, mdi, cvmdi, pairing, bbm92, e91, gaussian, cv, \
             dmcs; \"squeezed\" points at eur_binned"
        ))),
    }
}

/// TLGR Eq. (2) end to end, bits. The family name is mandatory and `eur_family` refuses every
/// shipped protocol; the bare arithmetic is `eur_length`.
#[pyfunction]
pub(crate) fn eur_secret(
    family: &str,
    n: f64,
    k: f64,
    q_tol: f64,
    leak: f64,
    eps_s: f64,
    eps_c: f64,
) -> PyResult<f64> {
    let q = eur_family(family)?;

    eur_length(n, k, q, q_tol, leak, eps_s, eps_c)
}
