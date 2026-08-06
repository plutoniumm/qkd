use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::decoy::check_gains;
use crate::std::{check_eps, check_err, check_fec, check_pos, check_prob, check_unit, h2};

// Six-state QKD: BB84 over the three mutually unbiased qubit bases, Bruss, Phys. Rev. Lett. 81,
// 3018 (1998). Tomographically complete: the one-way tolerable QBER rises from 11.0% to 12.6%.
// Formulas: Scarani, Bechmann-Pasquinucci, Cerf, Dusek, Lutkenhaus & Peev, Rev. Mod. Phys. 81,
// 1301 (2009), Appendix A, Eqs. (A1)-(A6); the same appendix derives BB84's Eq. (A9)
// `I_E = h2(e_x)`: `bb84_rate` and `sixstate_rate` differ in exactly one factor and nowhere else.
// Bell-diagonal parametrisation: Renner, Gisin & Kraus, Phys. Rev. A 72, 012332 (2005), Eq. (10);
// unconditional security: Lo, Quant. Inf. Comput. 1, 81 (2001).
//
// No second decoy layer: `decoy.rs`'s Y1 and e1 are basis-blind, so a decoy six-state rate is
// `decoy_bounds` feeding `sixstate_rate`; `sixstate_finite` refuses a decoy source.
//
// `sixstate_rate` gets one error rate from `decoy_bounds`, hence the depolarising point;
// `sixstate_length` reads ONE monitor rate, the AVERAGE of the two (`sixstate_pooled`).

/// At `bias = 1/3` all three bases are equally likely and the sifting factor is Lo's 1/3.
const UNIFORM: f64 = 1.0 / 3.0;

/// Shannon entropy of a Bell-diagonal spectrum, bits; zero weights contribute 0, not NaN.
fn shannon(l: [f64; 4]) -> f64 {
    l.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| -x * x.log2())
        .sum()
}

/// The three triangle inequalities under which the Bell weights are non-negative. Refused, not
/// clamped.
fn check_bell(e_x: f64, e_y: f64, e_z: f64) -> PyResult<()> {
    for (a, b, c, pair) in [
        (e_x, e_y, e_z, "e_x + e_y"),
        (e_y, e_z, e_x, "e_y + e_z"),
        (e_z, e_x, e_y, "e_z + e_x"),
    ] {
        if a + b < c {
            return Err(PyValueError::new_err(format!(
                "the three error rates must satisfy {pair} >= the third, got {a} + {b} < {c}: \
                 no Bell-diagonal state has those marginals, one of its weights being negative"
            )));
        }
    }

    Ok(())
}

/// The Bell weights `(l1, l2, l3, l4)` of Rev. Mod. Phys. 81, 1301 (2009), Eq. (A1), coefficients
/// of `|Phi+>`, `|Phi->`, `|Psi+>`, `|Psi->`, from inverting Eqs. (A2) and (A3): `l1 = 1 - (e_x +
/// e_y + e_z)/2`, `l2 = (e_x + e_y - e_z)/2`, `l3 = (e_y + e_z - e_x)/2`, `l4 = (e_z + e_x -
/// e_y)/2`. At the depolarising point `l1 = 1 - 3Q/2`, `l2 = l3 = l4 = Q/2`, UNNUMBERED prose in
/// Renner, Gisin & Kraus Sec. V.A; their Eq. (10) is the generic four-weight form.
#[pyfunction]
pub(crate) fn sixstate_bell(e_x: f64, e_y: f64, e_z: f64) -> PyResult<(f64, f64, f64, f64)> {
    check_err("e_x", e_x)?;
    check_err("e_y", e_y)?;
    check_err("e_z", e_z)?;
    check_bell(e_x, e_y, e_z)?;

    let total = e_x + e_y + e_z;

    Ok((
        1.0 - 0.5 * total,
        0.5 * (e_x + e_y - e_z),
        0.5 * (e_y + e_z - e_x),
        0.5 * (e_z + e_x - e_y),
    ))
}

/// Eve's Holevo information, bits per sifted single-photon signal, Rev. Mod. Phys. 81, 1301
/// (2009), Eq. (A4): `I_E = H(lambda) - h2(e_z)`. The key basis is the THIRD slot. Eq. (A4)
/// rather than the equivalent Eq. (A5), which divides by `e_z`.
#[pyfunction]
pub(crate) fn sixstate_eve(e_x: f64, e_y: f64, e_z: f64) -> PyResult<f64> {
    let (l1, l2, l3, l4) = sixstate_bell(e_x, e_y, e_z)?;

    Ok(shannon([l1, l2, l3, l4]) - h2(e_z))
}

/// `I_E` at the depolarising point, Rev. Mod. Phys. 81, 1301 (2009), Eq. (A6): `Q + (1 -
/// Q)*h2((1 - 3Q/2)/(1 - Q))`. The case a decoy layer reaches, `decoy_bounds` returning one `e1`.
#[pyfunction]
pub(crate) fn sixstate_holevo(qber: f64) -> PyResult<f64> {
    check_err("qber", qber)?;

    Ok(qber + (1.0 - qber) * h2((1.0 - 1.5 * qber) / (1.0 - qber)))
}

/// Secret fraction per sifted signal for a SINGLE-QUBIT source against collective attacks, `r =
/// 1 - f_ec*h2(e_z) - I_E`, Rev. Mod. Phys. 81, 1301 (2009), the sentence after Eq. (A6). Clamped
/// at zero. At `f_ec = 1` the crossing is 12.6193% against the paper's 12.61%. `sixstate_rate` is
/// the weak-coherent one.
#[pyfunction]
pub(crate) fn sixstate_secret(e_x: f64, e_y: f64, e_z: f64, f_ec: f64) -> PyResult<f64> {
    check_fec("f_ec", f_ec)?;
    let eve = sixstate_eve(e_x, e_y, e_z)?;

    Ok((1.0 - f_ec * h2(e_z) - eve).max(0.0))
}

/// Asymptotic weak-coherent rate, bits per emitted pulse: `q_sift * (q1*[1 - I_E(e1)] -
/// q_mu*f_ec*h2(e_mu))`, Eq. (A6)'s `I_E` in place of BB84's `h2(e1)` of Eq. (A9); every other
/// term is `bb84_rate`'s verbatim. `q_sift` is `sixstate_sift`'s first return, not 1/2.
#[pyfunction]
pub(crate) fn sixstate_rate(
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

    let eve = sixstate_holevo(e1)?;
    let rate = q_sift * (q1 * (1.0 - eve) - q_mu * f_ec * h2(e_mu));

    Ok(rate.max(0.0))
}

/// `(matched, key)` per emitted pulse, key basis at probability `bias` and each monitor basis at
/// `(1 - bias)/2`, both parties independently: `matched = bias^2 + (1 - bias)^2/2`, `key =
/// bias^2`. Lo, Quant. Inf. Comput. 1, 81 (2001): at `bias = 1/3` `matched` is the factor for
/// `sixstate_rate`; biased, `key` is, the monitor bases being spent on estimation.
#[pyfunction]
pub(crate) fn sixstate_sift(bias: f64) -> PyResult<(f64, f64)> {
    check_prob("bias", bias)?;

    if bias < UNIFORM {
        return Err(PyValueError::new_err(format!(
            "bias must be >= 1/3, got {bias}: below that the key basis is chosen less often \
             than each monitor basis"
        )));
    }

    if bias >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "bias must be < 1, got {bias}: at 1 the monitor bases are never measured and e_x, \
             e_y never estimated"
        )));
    }

    let off = 0.5 * (1.0 - bias);

    Ok((bias * bias + 2.0 * off * off, bias * bias))
}

/// Scarani & Renner instantiate Lemma 3 at `d = 2` for both estimates.
const OUTCOMES: f64 = 2.0;

/// Leftover-hash slack, smoothing, parameter estimation: one number in three slots, as
/// `q.FiniteSize` on the Gaussian path.
const SLOTS: f64 = 3.0;

/// Lemma 3 applications the estimation slot is spread across by union bound: one on the `n`
/// key-basis bits, one on the `m_test` monitor-basis bits.
const ESTIMATES: f64 = 2.0;

/// `(slot, each)`: the estimation slot and the per-estimate share inside it.
fn budget(eps: f64, eps_ec: f64) -> (f64, f64) {
    let slot = (eps - eps_ec) / SLOTS;

    (slot, slot / ESTIMATES)
}

fn check_budget(eps: f64, eps_ec: f64) -> PyResult<()> {
    if eps_ec >= eps {
        return Err(PyValueError::new_err(format!(
            "eps_ec must be < eps, got {eps_ec} >= {eps}: eps_ec is nested inside eps, and \
             Scarani & Renner's constraint eps - eps_ec > eps_bar > eps_pe >= 0 has nothing left \
             to split"
        )));
    }

    Ok(())
}

/// Lemma 3's width.
fn width(m: f64, eps: f64) -> f64 {
    ((2.0 * (1.0 / eps).ln() + OUTCOMES * (m + 1.0).ln()) / m).sqrt()
}

/// `H(X|E)` from a key-basis and a common monitor-basis error rate.
fn hxe(e_key: f64, e_test: f64) -> f64 {
    let ratio = (1.0 - e_test - 0.5 * e_key) / (1.0 - e_key);

    (1.0 - e_key) * (1.0 - h2(ratio))
}

/// Scarani & Renner, Phys. Rev. Lett. 100, 200501 (2008), arXiv:0708.0709, Lemma 3: `xi = sqrt(
/// (2*ln(1/eps) + d*ln(m + 1)) / m )`, `d = 2` as in both their instantiations; the state lies
/// within `xi` of the observed statistics except with probability `eps`. NOT A HOEFFDING WIDTH
/// (`decoy::hoeff` bounds a COUNT): a RELATIVE FREQUENCY paying a method-of-types `d*ln(m + 1)`
/// (Cover & Thomas, "Elements of Information Theory" (Wiley, 1991), Theorem 12.2.1 and Lemma
/// 12.6.1).
#[pyfunction]
pub(crate) fn sixstate_width(m: f64, eps: f64) -> PyResult<f64> {
    check_pos("m", m)?;
    check_eps("eps", eps)?;

    Ok(width(m, eps))
}

/// `H(X|E)` bits, Scarani & Renner's "Six-states" paragraph: `(1 - e_key) * [1 - h2( (1 - e_test -
/// e_key/2) / (1 - e_key) )]`. `2*e_test < e_key` is REFUSED: `l2 < 0` there, yet the expression
/// returns numbers. The BARE quantity; `sixstate_length` minimises it over the compatible set.
/// `e_test` IS THE AVERAGE OF THE TWO MONITOR RATES: the smaller of two unequal rates reports a
/// longer key than the data license. Use `sixstate_pooled`.
#[pyfunction]
pub(crate) fn sixstate_bound(e_key: f64, e_test: f64) -> PyResult<f64> {
    check_err("e_key", e_key)?;
    check_err("e_test", e_test)?;
    check_bell(e_test, e_test, e_key)?;

    Ok(hxe(e_key, e_test))
}

/// `(hxe, e_avg)` from the key-basis rate and the two monitor rates measured SEPARATELY. The
/// entropy reads the monitor bases ONLY THROUGH THEIR SUM (regroup Rev. Mod. Phys. 81, 1301
/// (2009), Eq. (A4)), so `hxe(e_key, (e_x + e_y)/2)` is a lower bound for EVERY state carrying
/// that sum, tight at `e_x = e_y` -- Scarani & Renner's "for e1 = e2, a case that minimizes it"
/// ahead of their Eq. (7). In the sum
/// variable: Rong Wang, Zhen-Qiang Yin, Hang Liu, Shuang Wang, Wei Chen, Guang-Can Guo & Zheng-Fu
/// Han, "Tight finite-key analysis for generalized high-dimensional quantum key distribution",
/// Phys. Rev. Research 3, 023019 (2021), arXiv:2008.03510, Eq. (A.36), `xi_{0|0} = (1 - (Q_x + Q_y
/// + Q_z)/2)/(1 - Q_x)`, `Q_x` their KEY basis. That number is the PREPRINT's: APS returns 403.
#[pyfunction]
pub(crate) fn sixstate_pooled(e_key: f64, e_x: f64, e_y: f64) -> PyResult<(f64, f64)> {
    check_err("e_key", e_key)?;
    check_err("e_x", e_x)?;
    check_err("e_y", e_y)?;
    check_bell(e_x, e_y, e_key)?;

    let avg = 0.5 * (e_x + e_y);

    Ok((hxe(e_key, avg), avg))
}

/// `(slot, each)`: `slot = (eps - eps_ec)/3`, Scarani & Renner's three gaps set equal, and `each
/// = slot/2`, the union bound over two estimates. Not `mdi_eps`'s scalar shape: Scarani & Renner
/// nest `eps_ec` INSIDE `eps`.
#[pyfunction]
pub(crate) fn sixstate_eps(eps: f64, eps_ec: f64) -> PyResult<(f64, f64)> {
    check_eps("eps", eps)?;
    check_eps("eps_ec", eps_ec)?;
    check_budget(eps, eps_ec)?;

    Ok(budget(eps, eps_ec))
}

/// Finite-key length in BITS for a SINGLE-QUBIT run, Scarani & Renner, Phys. Rev. Lett. 100,
/// 200501 (2008), Lemmas 1 to 3: `l = floor( n*H_xi(X|E) - leak_EC - 7*sqrt(n*log2(2/(eps_bar -
/// eps_pe))) - 2*log2(1/(2*(eps - eps_bar - eps_ec))) )`, `leak_EC = f_ec*n*h2(e_key)` at the
/// OBSERVED `e_key` (their `leak_EC/n = 1.2*h(Q)`), `H_xi` the minimum over the compatible set,
/// the three gaps of `eps - eps_ec > eps_bar > eps_pe >= 0` set equal. `m_test` is ONE monitor
/// basis's sample. Returns `(length, hxe, e_key_used, e_test_used, delta)`, `delta` the two
/// subtracted corrections in bits; clamped at zero and floored.
///
/// COLLECTIVE ATTACKS: Eq. (5) assumes `rho_{A^N B^N} = sigma^{tensor N}`, `sigma` two-qubit.
///
/// The `7` is transcribed and does not reproduce: Renner, arXiv:quant-ph/0512258, Corollary 3.3.7
/// gives 5 at d = 2. 7 subtracts more; do not "fix" it to 5.
///
/// Two departures from the printed instantiation, both in the secure direction, so figures differ
/// from the paper's: (1) `H(X|E)` RISES with `e_key`, so the minimum is at `e_key - xi(n)`, not the
/// "Six-states" paragraph's `e_key + xi(n)`; (2) Lemma 3 is applied twice, each width at
/// `eps_pe/2`.
#[pyfunction]
pub(crate) fn sixstate_length(
    n: f64,
    m_test: f64,
    e_key: f64,
    e_test: f64,
    f_ec: f64,
    eps: f64,
    eps_ec: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    check_pos("n", n)?;
    check_pos("m_test", m_test)?;
    check_err("e_key", e_key)?;
    check_err("e_test", e_test)?;
    check_fec("f_ec", f_ec)?;
    check_eps("eps", eps)?;
    check_eps("eps_ec", eps_ec)?;
    check_bell(e_test, e_test, e_key)?;

    check_budget(eps, eps_ec)?;

    let (slot, each) = budget(eps, eps_ec);
    let (dk, dt) = (width(n, each), width(m_test, each));

    // The corner that binds is (e_key - dk, e_test + dt); the other three are scanned anyway.
    let mut used = ((e_key - dk).max(0.0), (e_test + dt).min(0.5));
    let mut best = hxe(used.0, used.1);
    for a in [(e_key - dk).max(0.0), (e_key + dk).min(0.5)] {
        for b in [(e_test - dt).max(0.0), (e_test + dt).min(0.5)] {
            if a > 2.0 * b {
                continue;
            }

            let got = hxe(a, b);
            if got < best {
                best = got;
                used = (a, b);
            }
        }
    }

    let leak = f_ec * n * h2(e_key);
    let delta = 7.0 * (n * (2.0 / slot).log2()).sqrt() + 2.0 * (1.0 / (2.0 * slot)).log2();
    let len = (n * best - leak - delta).max(0.0).floor();

    Ok((len, best, used.0, used.1, delta))
}

/// `(length, rate, n, m_test)` for a block of `n_total` emitted signals, the rate per EMITTED
/// signal (divide by `sixstate_sift(bias).1` for `sixstate_secret`'s per-sifted fraction). Sifting
/// is Scarani & Renner's "Six-states": `n = n_total*bias^2`, `m_test = n_total*((1 - bias)/2)^2`
/// per monitor basis. `source = "decoy"` is REFUSED.
#[pyfunction]
pub(crate) fn sixstate_finite(
    n_total: f64,
    bias: f64,
    e_key: f64,
    e_test: f64,
    f_ec: f64,
    eps: f64,
    eps_ec: f64,
    source: &str,
) -> PyResult<(f64, f64, f64, f64)> {
    check_pos("n_total", n_total)?;
    let (_, key) = sixstate_sift(bias)?;
    match source {
        "single" => (),
        "decoy" => {
            return Err(PyNotImplementedError::new_err(
                "source=\"decoy\" is refused: qkd ships no finite-key six-state bound for a \
                 weak-coherent source. Pass source=\"single\" for the single-qubit length, or \
                 sixstate_rate for the asymptotic weak-coherent rate. The photon-number inversion \
                 is basis-blind and carries over, and THE STATISTICAL TRANSFER IS NOT WHAT FAILS \
                 EITHER, though this refusal once said so; nor is the constraint set. (A) ONLY \
                 THE SUM OF THE MONITOR RATES IS NEEDED: regrouping Rev. Mod. Phys. 81, 1301 \
                 (2009), Eq. (A4), l1 moves with the sum and l3/e_z with the difference, and h2 \
                 peaks at 1/2, so one pooled number is the worst case -- Scarani & Renner's \"for \
                 e1 = e2, a case that minimizes it\". In the key basis's variables: Rong Wang, \
                 Zhen-Qiang Yin, Hang Liu, Shuang Wang, Wei Chen, Guang-Can Guo & Zheng-Fu Han, \
                 \"Tight finite-key analysis for generalized high-dimensional quantum key \
                 distribution\", Phys. Rev. Research 3, 023019 (2021), arXiv:2008.03510, Eq. \
                 (A.36), xi_{0|0} = (1 - (Q_x + Q_y + Q_z)/2)/(1 - Q_x); A-numbers are the \
                 preprint's, whose appendix continues the main text's count, so quote the \
                 equation, not the label. sixstate_pooled ships it. (B) THE POOLED SAMPLE IS A \
                 CLASSICAL POPULATION. Tupkary, Nahar, Sinha & Lutkenhaus, \"Phase error rate \
                 estimation in QKD with imperfect detectors\", Quantum 9, 1937 (2025), \
                 arXiv:2408.17349: Remark 5's \"since such a joint distribution does not exist\" \
                 scopes their proof, in which \"all the random variables whose joint distribution \
                 is used in our arguments can indeed exist at the same time\"; Remark 20, \"we \
                 are only interested in making statements on the marginal probability \
                 distribution\"; footnote 5, \"the standard serfling argument is directly \
                 applicable\". (C) POOLING BEFORE THE DECOY INVERSION IS IN PRINT: Feng-Yu Lu, \
                 Zhen-Qiang Yin, Guan-Jie Fan-Yuan, Rong Wang, Hang Liu, Shuang Wang, Wei Chen, \
                 De-Yong He, Wei Huang, Bing-Jie Xu, Guang-Can Guo & Zheng-Fu Han, \"Efficient \
                 decoy states for the reference-frame-independent measurement-device-independent \
                 quantum key distribution\", Phys. Rev. A 101, 052318 (2020), arXiv:2002.03672, \
                 Relation 3, \"Any combinations of XX, XY, YX, YY basis can be regarded as an \
                 entirety\"; Zhiyu Tian, Ziran Xie, Rong Wang, Chunmei Zhang & Shihai Sun, \
                 \"Experimental demonstration of improved reference-frame-independent quantum key \
                 distribution over 175km\", Opt. Express 32, 22460 (2024), arXiv:2403.10294, \
                 prepare-and-measure with decoys and a composable finite key. Both are RFI. ALL \
                 THREE ROUTES FAIL ON THE SOURCE, NONE ON THE THIRD BASIS. (1) THE PHASE-ERROR \
                 TRANSFER: Lim, Curty, Walenta, Xu & Zbinden, \"Concise security bounds for \
                 practical decoy-state quantum key distribution\", Phys. Rev. A 89, 022307 \
                 (2014), arXiv:1311.7129, Eq. (5), takes its gamma from Fung, Ma & Chau, \
                 \"Practical issues in quantum-key-distribution post-processing\", Phys. Rev. A \
                 81, 012318 (2010), arXiv:0910.0312, scoped to \"the BB84 protocol with a single \
                 or entangled photon source\". (2) THE SMOOTH ENTROPIC UNCERTAINTY RELATION: \
                 src/eur.rs's condition C1 is TWO POVMs on one system, and arXiv:2008.03510 Eq. \
                 (A.19) reads all three bases through it with \"M0 = Y and M1 = Z\" chosen per \
                 round -- \"exactly the six-state protocol\", proved \"against general attacks\". \
                 C2, the source, admits a qubit or entangled source only. (3) THE COMPATIBLE-SET \
                 MINIMISATION, implemented here: Scarani & Renner, Phys. Rev. Lett. 100, 200501 \
                 (2008), arXiv:0708.0709. Lemma 3 wants \"m samples of sigma according to a POVM \
                 with d outcomes\"; decoy hands over bounds on counts, Lim Eqs. (2) to (4). They \
                 apply it \"when implemented with single qubits\", and Lemma 2 is \"a Corollary \
                 3.3.7 of [13]\", Renner's quant-ph/0512258, an i.i.d. product-state bound lifted \
                 to general attacks \"only thanks to specific symmetries\". SO THE ONE THING \
                 MISSING IS THE TAGGED SINGLE-PHOTON LAYER: a single-photon PER-ROUND REGISTER \
                 fixed before the entropy is taken. arXiv:2008.03510 says only that the \
                 finite-key case \"can intuitively be solved by using decoy states\"; Tupkary et \
                 al. Remark 21: \"one must first fix the number of pulses corresponding to these \
                 events in order to have the registers be well-defined\", and \"This subtlety is \
                 missing in\" Lim et al. No solver retires it: src/lp.rs fits the inversion and \
                 sixstate_eve is concave (test_eve_concave). NEAREST SYSTEMS, NONE THIS ONE. Liu, \
                 Luo, Luo, Li, Zhang & Wei, \"Reference-frame-independent quantum key \
                 distribution over 250 km of optical fiber\", Phys. Rev. Applied 22, 064018 \
                 (2024), arXiv:2405.16558 (arXiv title misspelt), prepare \"quantum states with \
                 intensity k in {mu, nu} in three orthogonal bases alpha in {Z_A, X_A, Y_A}\" \
                 with a finite key, but price Eve by the correlator of Laing, Scarani, Rarity & \
                 O'Brien, \"Reference frame independent quantum key distribution\", Phys. Rev. A \
                 82, 012304 (2010), arXiv:1003.1050, whose rate \"coincides with the rate of the \
                 six-state protocol for white noise\" at one channel, asymptotically, on qubits. \
                 Abruzzo, Mertz, Kampermann & Bruss, \"Finite-key analysis of the six-state \
                 protocol with photon-number-resolution detectors\", Proc. SPIE 8189, 818917 \
                 (2011), arXiv:1111.2798, measure the photon number rather than invert decoys. \
                 Kamin & Lutkenhaus, \"Improved Decoy-state and Flag-state Squashing Methods\", \
                 Phys. Rev. Research 6, 043223 (2024), arXiv:2405.05069, Sec. VIII \"Example 2: \
                 Biased passive WCP 6-state\", stop at \"We only consider the asymptotic regime, \
                 but our methods can be easily extended to the finite-size regime\". THE \
                 SQUASHING MODEL IS NOT THE BLOCKER: Beaudry, Moroder & Lutkenhaus, Phys. Rev. \
                 Lett. 101, 093601 (2008), arXiv:0804.3082, find none for six-state; Gittsovich, \
                 Beaudry, Narasimhachar, Romero Alvarez, Moroder & Lutkenhaus, Phys. Rev. A 89, \
                 012325 (2014), arXiv:1310.5059, Theorem 12, give one once post-processing flips \
                 single-click bit values with probability 1/6; Kato & Tamaki, \"Security of \
                 six-state quantum key distribution protocol with threshold detectors\", Sci. \
                 Rep. 6, 30044 (2016), arXiv:1008.4663, need none but \"take the asymptotic limit \
                 such that the number of the pulses is infinite\" on \"an attenuated laser source \
                 by GLLP idea\" (12.611% against 12.61931% here). Kamin, Tupkary & Lutkenhaus, \
                 \"Improved finite-size effects in QKD protocols with applications to decoy-state \
                 QKD\", arXiv:2502.05382, published as \"Improved finite-size effects in quantum \
                 key distribution with applications to decoy-state protocols\", Phys. Rev. \
                 Research 8, 013332 (2026), reach a finite key on a three-basis receiver where \
                 Alice still prepares FOUR states. Ze-Hao Wang, Zhen-Qiang Yin, Shuang Wang, Rong \
                 Wang, Feng-Yu Lu, Wei Chen, De-Yong He, Guang-Can Guo & Zheng-Fu Han, \"Tight \
                 finite-key analysis for mode-pairing quantum key distribution\", \
                 arXiv:2302.13481, propose \"a six-state MP-QKD protocol\" whose Eq. (2) takes \
                 \"the upper bound of the sum of the single-photon bit error rate\" on a \
                 decoy-tagged count (sixstate_bound at half the sum, test_wang_form), inside a \
                 relay; the author order is the preprint's, and the Commun. Phys. 6, 265 (2023) \
                 version permutes positions two to four. TWO PAPERS WERE NOT READ: Comfort Sekga \
                 & Mhlambululi Mafu, \"Security of quantum-key-distribution protocol by using the \
                 post-selection technique\", Physics Open 7, 100075 (2021), \
                 doi:10.1016/j.physo.2021.100075, a six-state SARG04 variant with a multi-photon \
                 source at finite resources, full text unreachable; Costantino Agnesi, Paolo \
                 Villoresi & Giuseppe Vallone, \"Improved Bounds for Practical \
                 Reference-Frame-Independent Quantum Key Distribution\", 2026 International \
                 Conference on Quantum Communications, Networking, and Computing (QCNC), pp. \
                 845-851, doi:10.1109/qcnc69040.2026.00139, a one-decoy finite-key RFI analysis \
                 of which ONLY ITS CROSSREF RECORD WAS OPENED. Either could move this refusal",
            ))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown source {other:?}, expected \"single\" or \"decoy\""
            )))
        }
    }

    let off = 0.5 * (1.0 - bias);
    let n = n_total * key;
    let m_test = n_total * off * off;
    let (len, ..) = sixstate_length(n, m_test, e_key, e_test, f_ec, eps, eps_ec)?;

    Ok((len, len / n_total, n, m_test))
}
