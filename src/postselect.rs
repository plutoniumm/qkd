use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::std::{check_eps, check_nonneg};

// The postselection technique: the lift from an IID-COLLECTIVE proof to COHERENT attacks for
// permutation-invariant protocols on FINITE-dimensional systems. NOT WIRED: `ps_family` refuses
// every protocol this tree ships by name.
//
// Christandl, Koenig & Renner, "Post-selection technique for quantum channels with applications to
// quantum cryptography", Phys. Rev. Lett. 102, 020504 (2009), arXiv:0809.3019, Theorem 1: if for
// every permutation pi there is a CPTP map K_pi with `Delta . pi = K_pi . Delta`, then
//
//      ||Delta||_diamond  <=  g_{n,d} ||(Delta (x) id)(tau_{H^n R})||_1,
//      g_{n,d} = C(n + d^2 - 1, n) <= (n + 1)^(d^2 - 1),      d = dim H,
//
// `tau` the de Finetti state `int sigma^{(x)n} d(sigma)`. Their QKD application sets
// `H := H_A (x) H_B`, so `d^2 = d_A^2 d_B^2`, the `x` every function here takes.
//
// TWO COSTS, TWO QUANTITIES, returned by `ps_lift` as two fields: the factor multiplies a FAILURE
// PROBABILITY UPWARD and does NOT scale a key rate; the key LENGTH separately loses `2 log2 g`.
//
// The version implemented is Nahar, Tupkary, Zhao, Lutkenhaus & Tan, "Postselection technique for
// optical Quantum Key Distribution with improved de Finetti reductions", PRX Quantum 5, 040315
// (2024), arXiv:2403.11851, which fixes a flaw in CKR09 whose omission errs INSECURE. Corollary
// 3.1: given an IID proof of the shape of their Eqs. (10) and (11) -- an acceptance test failing
// with probability eps_AT, a leftover-hash bound with hashing parameter eps_PA and smoothing
// parameter eps_bar -- the protocol hashing to `l' = l - 2 log2 g_{n,x}` is
//
//      g_{n,x} ( eps_PA + 2 eps_bar + 2 sqrt(2 eps_AT) )   -secret against coherent attacks,
//
// `x = d_A^2 d_B^2`, `g_{n,x} = C(n + x - 1, x - 1) <= (e (n + x - 1)/(x - 1))^(x - 1)` (their
// Sec. V.1.1, Item 2). CKR09's `g_{n,x} max{eps_PA + 2 eps_bar, eps_AT}` is strictly smaller at
// every admissible allocation and does not ship.
//
// The budget side is in LOGARITHMS because the epsilon the lift demands is usually not a
// representable f64, and 0.0 as a security parameter reads as unconditional security: `ps_budget`
// returns `log2`, `ps_thirds` allocates in `log2`, `ps_width` takes `log2 eps_AT`, and
// `ps_epsilon`, the one function forming a probability, refuses at or above 1.
//
// Requirements, which `ps_family`'s refusals name by number:
//   C1  PERMUTATION INVARIANCE of `E - E_ideal`, NTZLT Definition 5. Both papers note it can be
//       ENFORCED by a public random permutation of the rounds, ~`n log n` public bits.
//   C2  FINITE DIMENSION, CKR09 Theorem 1 being for "subsystems H of finite dimension". Every
//       optical protocol fails it on BOTH sides; repairing it needs NTZLT's tagged-state source map
//       (Sec. IV.3) and weight-preserving flag-state squasher (Sec. IV.2.1, "the existing
//       flag-state squasher cannot be used"). qkd implements NEITHER.
//   C3  A FIXED MARGINAL `sigma_A^{(x)n}` from the source description (NTZLT Sec. IV.1), which
//       extends the technique to prepare-and-measure; CKR09 is entanglement-based only.
//   C4  AN IID PROOF OF THE SHAPE OF NTZLT's Eqs. (10) and (11). A proof already against general
//       attacks has no such decomposition, and lifting its epsilon bounds nothing.

/// A bound on work, not physics: the exact binomial costs O(x) logarithms.
const X_MAX: u128 = 1 << 20;

/// A squashed optical receiver in the literature reaches tens.
const D_MAX: u32 = 1024;

/// Inside the range f64 counts integers exactly; `eur.rs` reads it.
pub(crate) const N_MAX: f64 = 1e15;

fn check_dim(name: &str, d: u32) -> PyResult<()> {
    if d >= 1 && d <= D_MAX {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a dimension in [1, {D_MAX}], got {d}"
        )))
    }
}

/// Zero allowed; capped at `D_MAX` so the products in `ps_tagged` and `ps_optical` cannot overflow.
fn check_count(name: &str, v: u32) -> PyResult<()> {
    if v <= D_MAX {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a count in [0, {D_MAX}], got {v}"
        )))
    }
}

/// Integral, unlike `eur::check_block`; both refusal literals are pinned.
fn check_block(n: f64) -> PyResult<()> {
    if n.is_finite() && n >= 1.0 && n <= N_MAX && n.fract() == 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "n must be a whole number of signals in [1, {N_MAX:e}], got {n}"
        )))
    }
}

/// `x = 1` is refused: `g = 1` makes the lift a no-op that still reads as a security statement.
fn check_x(x: u128) -> PyResult<u64> {
    if x >= 2 && x <= X_MAX {
        Ok(x as u64)
    } else {
        Err(PyValueError::new_err(format!(
            "x must be a reduction dimension in [2, {X_MAX}], got {x}: at x = 1, g = 1 makes the \
             lift a no-op"
        )))
    }
}

fn check_log2(name: &str, v: f64) -> PyResult<()> {
    if v.is_finite() && v < 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a finite negative log2 of a probability, got {v}"
        )))
    }
}

/// `x = d_A^2 d_B^2`, NTZLT Corollary 3.1 (CKR09's `d = dim(H_A (x) H_B)` squared, the same
/// integer). `d_B` is ONE ROUND of Bob's system after squashing; no value describes an unsquashed
/// threshold detector.
#[pyfunction]
pub(crate) fn ps_dim(d_a: u32, d_b: u32) -> PyResult<u64> {
    check_dim("d_a", d_a)?;
    check_dim("d_b", d_b)?;

    let x = (d_a as u128) * (d_a as u128) * (d_b as u128) * (d_b as u128);

    check_x(x)
}

/// A two-mode threshold receiver behind the weight-preserving flag-state squasher, NTZLT Eq. (37):
/// `x = d_A^2 (sum_{i=0}^{N} (i+1)^2 + n_meas)`, `cutoff` = `N`, `flags` = `n_meas` (their
/// footnote 5: fewer than Bob's POVM elements when coarse-grained). Their Sec. V.2 setup
/// `(2, 1, 8)` is `4 * (1 + 4 + 8) = 52`.
#[pyfunction]
pub(crate) fn ps_squash(d_a: u32, cutoff: u32, flags: u32) -> PyResult<u64> {
    check_dim("d_a", d_a)?;
    check_count("cutoff", cutoff)?;
    check_count("flags", flags)?;

    let mut blocks: u128 = flags as u128;
    for i in 0..=(cutoff as u128) {
        blocks += (i + 1) * (i + 1);
    }

    let x = (d_a as u128) * (d_a as u128) * blocks;

    check_x(x)
}

/// A decoy-state source reduced to tagged states, NTZLT Eq. (38): `x = n_int^2 (N_ph + 2) d_A^2
/// d_B^2`, `d_a` WITHOUT the shield system. NTZLT Sec. IV.4: a higher `N_ph` helps the IID decoy
/// analysis and hurts the lift.
#[pyfunction]
pub(crate) fn ps_tagged(n_int: u32, n_ph: u32, d_a: u32, d_b: u32) -> PyResult<u64> {
    check_dim("n_int", n_int)?;
    check_count("n_ph", n_ph)?;
    check_dim("d_a", d_a)?;
    check_dim("d_b", d_b)?;

    let x = (n_int as u128)
        * (n_int as u128)
        * (n_ph as u128 + 2)
        * (d_a as u128)
        * (d_a as u128)
        * (d_b as u128)
        * (d_b as u128);

    check_x(x)
}

/// Both reductions, NTZLT Eq. (39): `x = n_int^2 (N_ph + 2) d_A^2 (sum_{i=0}^{N} (i+1)^2 +
/// n_meas)` -- what a decoy-state BB84 would be assembled at, so `ps_family("bb84")`'s refusal can
/// be priced.
#[pyfunction]
pub(crate) fn ps_optical(
    n_int: u32,
    n_ph: u32,
    d_a: u32,
    cutoff: u32,
    flags: u32,
) -> PyResult<u64> {
    check_dim("n_int", n_int)?;
    check_count("n_ph", n_ph)?;

    let squashed = ps_squash(d_a, cutoff, flags)? as u128;
    let x = (n_int as u128) * (n_int as u128) * (n_ph as u128 + 2) * squashed;

    check_x(x)
}

/// `log2 g_{n,x}`, `g = C(n + x - 1, x - 1) = dim Sym^n(C^x)`, as `sum_{k=1}^{x-1} ln(n + k) -
/// ln((x-1)!)`. Not `std::ln_fact`: only the last entry is read and `x` reaches 2^20. A logarithm
/// because `g` overflows f64 near `x = 40` at a 1e9 block.
#[pyfunction]
pub(crate) fn ps_cost(n: f64, x: u64) -> PyResult<f64> {
    check_block(n)?;
    let x = check_x(x as u128)?;

    let k = (x - 1) as usize;
    let mut acc = 0.0;
    let mut lf = 0.0;
    for j in 1..=k {
        acc += (n + j as f64).ln();
        lf += (j as f64).ln();
    }

    let cost = (acc - lf) / std::f64::consts::LN_2;
    if !cost.is_finite() || cost <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "log2 g_{{{n},{x}}} came out {cost}, which is not a positive finite cost"
        )));
    }

    Ok(cost)
}

/// `log2` of NTZLT Sec. V.1.1 Item 2, `g_{n,x} <= (e (n + x - 1) / (x - 1))^(x - 1)`. At or above
/// `ps_cost` always (`test_cap_above`), so substituting it is safe; nothing here substitutes the
/// other way.
#[pyfunction]
pub(crate) fn ps_cap(n: f64, x: u64) -> PyResult<f64> {
    check_block(n)?;
    let x = check_x(x as u128)?;

    let k = (x - 1) as f64;
    let cap = k * (std::f64::consts::E * (n + k) / k).log2();
    if !cap.is_finite() || cap <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "the closed-form cap came out {cap}, which is not a positive finite cost"
        )));
    }

    Ok(cap)
}

/// NTZLT Corollary 3.1's input, `eps_PA + 2 eps_bar + 2 sqrt(2 eps_AT)`: hashing and smoothing
/// parameters of the leftover-hash bound (their Eq. 11) and the acceptance-test failure (their
/// Eq. 10). The square-root term is NTZLT's correction: CKR09's `max{eps_PA + 2 eps_bar, eps_AT}`
/// is smaller wherever eps_AT is not negligible.
#[pyfunction]
pub(crate) fn ps_secrecy(eps_pa: f64, eps_bar: f64, eps_at: f64) -> PyResult<f64> {
    check_eps("eps_pa", eps_pa)?;
    check_eps("eps_bar", eps_bar)?;
    check_eps("eps_at", eps_at)?;

    let total = eps_pa + 2.0 * eps_bar + 2.0 * (2.0 * eps_at).sqrt();
    if !(total < 1.0) {
        return Err(PyValueError::new_err(format!(
            "the IID secrecy parameter eps_PA + 2 eps_bar + 2 sqrt(2 eps_AT) composed to {total}, \
             not below 1; the sqrt term alone is {} at eps_at = {eps_at}",
            2.0 * (2.0 * eps_at).sqrt()
        )));
    }

    Ok(total)
}

/// `g_{n,x}` times the IID secrecy parameter, NTZLT Corollary 3.1: UPWARD, always. At optical
/// dimensions no positive f64 `eps_iid` clears the bar (`x = 2340`, `n = 1e9` needs `log2 eps_iid
/// <= -47153`; f64 stops at -1074), so refusing every input is a fact about the lift; read the
/// requirement off `ps_budget`.
#[pyfunction]
pub(crate) fn ps_epsilon(eps_iid: f64, n: f64, x: u64) -> PyResult<f64> {
    check_eps("eps_iid", eps_iid)?;

    let cost = ps_cost(n, x)?;
    let lifted = 2f64.powf(eps_iid.log2() + cost);

    if !(lifted < 1.0) || !lifted.is_finite() {
        return Err(PyValueError::new_err(format!(
            "the lift of eps_iid = {eps_iid:e} at n = {n:e}, x = {x} does not land below 1: \
             log2 g = {cost:.3}, so log2 eps_iid <= {:.3} is needed, which f64 holds no positive \
             number for below 2^-1074; the requirement is not representable -- read it off \
             ps_budget, which returns log2",
            -cost
        )));
    }

    if !(lifted > eps_iid) {
        return Err(PyValueError::new_err(format!(
            "the lifted secrecy parameter {lifted:e} is not above the IID one {eps_iid:e}: the \
             factor multiplies a failure probability UPWARD"
        )));
    }

    Ok(lifted)
}

/// `log2(eps_target) - log2 g_{n,x}`, the `log2` IID parameter the lift needs to land at
/// `eps_target`; CKR09's `eps_bar := eps (n+1)^-(d^2-1)`. A LOGARITHM: at NTZLT's three-state
/// point the answer is 2^-2021, and 0.0 reads as unconditional security.
#[pyfunction]
pub(crate) fn ps_budget(eps_target: f64, n: f64, x: u64) -> PyResult<f64> {
    check_eps("eps_target", eps_target)?;

    let cost = ps_cost(n, x)?;
    let budget = eps_target.log2() - cost;

    if !(budget < eps_target.log2()) {
        return Err(PyValueError::new_err(format!(
            "the IID budget log2 = {budget} is not below the target log2 = {}: the lift makes \
             the IID requirement STRICTER",
            eps_target.log2()
        )));
    }

    Ok(budget)
}

/// Equal thirds of `L = log2(eps_PA + 2 eps_bar + 2 sqrt(2 eps_AT))`: `(log2 eps_PA, log2 eps_bar,
/// log2 eps_AT)` = `(L - log2 3, L - log2 6, 2L - log2 72)`. The SQUARE in the last is why the
/// acceptance test, not the hash, is what the lift costs. A choice made here: NTZLT Sec. V.1.1
/// leaves the allocation to the caller.
#[pyfunction]
pub(crate) fn ps_thirds(budget: f64) -> PyResult<(f64, f64, f64)> {
    check_log2("budget", budget)?;

    let pa = budget - 3f64.log2();
    let bar = budget - 6f64.log2();
    let at = 2.0 * budget - 72f64.log2();

    Ok((pa, bar, at))
}

/// `l' = l - 2 log2 g_{n,x}` bits, NTZLT Corollary 3.1 and CKR09's `2 log2 dim N <= 2(d^2 - 1)
/// log2(n+1)`. Returned raw, negatives included, as `cvmdi_rate` does.
#[pyfunction]
pub(crate) fn ps_length(l: f64, n: f64, x: u64) -> PyResult<f64> {
    check_nonneg("l", l)?;

    let cost = ps_cost(n, x)?;
    let short = l - 2.0 * cost;

    if !(short < l) {
        return Err(PyValueError::new_err(format!(
            "the shortened length {short} is not below the input length {l}: the lift SPENDS key"
        )));
    }

    Ok(short)
}

/// Acceptance-test half-width, NTZLT Lemma 9 (their Eq. 43): `mu = sqrt(ln(2 |Sigma| / eps_AT) /
/// (2m))`, `m` test rounds over `|Sigma|` outcomes, `eps` = `log2 eps_AT`. The lift is paid here:
/// once `mu` reaches the QBER the key is gone with no negative length to show for it. The inner
/// logarithm is NATURAL (NTZLT print bare `log`); `log2` would inflate `mu` by 1.201.
#[pyfunction]
pub(crate) fn ps_width(m: f64, outcomes: f64, eps: f64) -> PyResult<f64> {
    check_block(m)?;
    check_log2("eps", eps)?;

    if !(outcomes >= 1.0) || !outcomes.is_finite() {
        return Err(PyValueError::new_err(format!(
            "outcomes must be at least one POVM element, got {outcomes}"
        )));
    }

    let inner = (2.0 * outcomes).ln() - eps * std::f64::consts::LN_2;

    Ok((inner / (2.0 * m)).sqrt())
}

/// The gate the rest of this file sits behind. Two names return a number and neither is a path
/// here: `"qubit"` is `ps_dim(2, 2) = 16` (CKR09 keeps `d` general and prints neither the number
/// nor the word); `"three-state"` is NTZLT Sec. V.2's worked optical setup, `x = 52`. Every
/// shipped family is refused, naming which of C1 to C4 it fails.
#[pyfunction]
pub(crate) fn ps_family(name: &str) -> PyResult<u64> {
    match name {
        "qubit" => ps_dim(2, 2),
        "three-state" => ps_squash(2, 1, 8),

        "bb84" => Err(PyNotImplementedError::new_err(
            "bb84: C2 fails twice and C4 once. A phase-randomised weak-coherent source and a \
             threshold receiver are both infinite-dimensional, and neither NTZLT reduction ships \
             (Secs. IV.3 and IV.2.1). Nothing to lift: `bb84_length` implements Lim, \
             Curty, Walenta, Xu & Zbinden, Phys. Rev. A 89, 022307 (2014), 'valid against \
             general attacks', whose epsilons are not (eps_PA, eps_bar, eps_AT). Price the \
             dimension with ps_optical",
        )),

        "sixstate" => Err(PyNotImplementedError::new_err(
            "sixstate: C2 holds only on the branch with nothing to lift. \
             `sixstate_finite(source=\"decoy\")` refuses; `source=\"single\"` is a qubit \
             protocol, x = 16, but implements Scarani & Renner, Phys. Rev. Lett. 100, 200501 \
             (2008), arXiv:0708.0709, whose abstract says that while 'our proof relies on the \
             assumption of collective attacks, unconditional security follows immediately for \
             standard protocols like Bennett-Brassard 1984 and six-states', so C4 fails: no \
             (eps_PA, eps_bar, eps_AT) decomposition for Corollary 3.1 to consume",
        )),

        "sarg" => Err(PyNotImplementedError::new_err(
            "sarg: C4 fails, and no longer vacuously -- `sarg_length` implements Nian, Nie, \
             Zhang & Lu, Commun. Theor. Phys. 76, 065101 (2024), Eq. (5), stated under \
             composable security against general attacks, and its (eps_sec, eps_cor) is not the \
             (eps_PA, eps_bar, eps_AT) triple. C2 fails as bb84's does. Wang, Corrigan & \
             Lutkenhaus, arXiv:2603.22448, Eq. (3), is SARG04 lifted by this technique, a \
             NUMERICAL bound on a different cone",
        )),

        "b92" => Err(PyNotImplementedError::new_err(
            "b92: C4 fails vacuously -- `b92_finite` refuses per variant, so there is no \
             finite-key length to shorten",
        )),

        "mdi" => Err(PyNotImplementedError::new_err(
            "mdi: C4 fails. `mdi_length` implements Curty, Xu, Cui, Lim, Tamaki & Lo, Nature \
             Communications 5, 3732 (2014), arXiv:1307.1081, 'a rigorous security proof against \
             general attacks in the finite-key regime': nothing to lift. C2 fails too: two \
             weak-coherent senders and threshold detectors at the relay, none reduced",
        )),

        "rrdps" => Err(PyNotImplementedError::new_err(
            "rrdps: C2 fails. A weak-coherent train read on threshold detectors, neither side \
             finite-dimensional as shipped; the exchangeable unit is the PACKET, which C1 \
             accommodates, but no source map or squasher here bounds its dimension",
        )),

        "pairing" => Err(PyNotImplementedError::new_err(
            "pairing: C1 fails, the only family here that fails on permutation invariance rather \
             than dimension: rounds pair AFTER the announcement within a maximal interval, so a \
             random permutation does not commute with the map, and pairs are not fixed in advance \
             for C1 to hold per pair",
        )),

        "bbm92" | "e91" => Err(PyNotImplementedError::new_err(
            "bbm92/e91: C4 fails vacuously -- `q.PairLink` is asymptotic and ships no key \
             length. C2 is not established: coincidence counting on threshold detectors leaves \
             d_B = 2 an encoding, not a certified single-round dimension. Lifting the IID \
             assumption is no step toward the device-independent claim `ekert.rs` rules out",
        )),

        "cow" => Err(PyNotImplementedError::new_err(
            "cow: C4 fails vacuously on the three-sequence protocol -- `cow_rate` takes its \
             phase error as a supplied upper bound, so there is no acceptance test with an \
             eps_AT. The FOUR-sequence variant is a different family: `sdp_length` ships Li, \
             Cao, Xie, Yin & Chen, Phys. Rev. Research 6, 013022 (2024), arXiv:2309.16136, \
             Eq. (23), already against coherent attacks by an entropic uncertainty relation",
        )),

        "dps" => Err(PyNotImplementedError::new_err(
            "dps: C4 fails, and no longer vacuously -- `dps_finite` implements Mizutani, \
             Takeuchi & Tamaki, Phys. Rev. Research 5, 023132 (2023), arXiv:2301.09844, \
             Corollary 1 Eq. (24), already against GENERAL attacks by Koashi's complementarity, \
             with no (eps_PA, eps_bar, eps_AT) triple. C2 fails as well: that length is written \
             on two photon-number-resolving detectors, not the threshold receiver this tree \
             runs, and `dps_rate`, the bound on a threshold train, takes its phase error as a \
             supplied upper bound",
        )),

        "gaussian" | "cv" => Err(PyNotImplementedError::new_err(
            "gaussian/cv: C2 fails outright. `cv_finite`, the one collective-only finite-key \
             path here -- Leverrier, Grosshans & Grangier, Phys. Rev. A 81, 062343 (2010) -- \
             reads infinite-dimensional, continuous-outcome modes with no squashing map. The \
             continuous-variable route is the Gaussian de Finetti reduction: \
             `keyrate::cv_general` ships its arithmetic, not its symmetrisation or energy test",
        )),

        "dmcs" => Err(PyNotImplementedError::new_err(
            "dmcs: C4 fails -- `dm_secure` is asymptotic. C2 fails in the way that matters: the \
             photon-number cutoff `nc` is a working assumption in LUL19's own words, not a \
             squashing map, so nc + 1 is not a certified d_B, and passing it to ps_dim would \
             understate g -- the one misuse here that is insecure rather than wrong",
        )),

        "cvmdi" => Err(PyNotImplementedError::new_err(
            "cvmdi: C2 fails -- continuous-variable modes at both senders and a Bell detector \
             that is not a threshold measurement. C4 fails too: `cvmdi_rate` is asymptotic",
        )),

        _ => Err(PyValueError::new_err(format!(
            "unknown family {name:?}. \"qubit\" (CKR09's entanglement-based pair) and \
             \"three-state\" (NTZLT Sec. V.2) return a dimension and neither is a path here; \
             every shipped family is refused by name: bb84, sixstate, sarg, b92, mdi, rrdps, \
             pairing, bbm92, e91, cow, dps, gaussian, cv, dmcs, cvmdi"
        ))),
    }
}

/// NTZLT Corollary 3.1 end to end: `(l', eps_coherent)`, TWO FIELDS FOR TWO QUANTITIES. The
/// family name is mandatory and `ps_family` refuses every shipped protocol; the bare arithmetic
/// is `ps_cost`, `ps_epsilon` and `ps_length`.
#[pyfunction]
pub(crate) fn ps_lift(
    family: &str,
    l: f64,
    eps_pa: f64,
    eps_bar: f64,
    eps_at: f64,
    n: f64,
) -> PyResult<(f64, f64)> {
    let x = ps_family(family)?;
    let iid = ps_secrecy(eps_pa, eps_bar, eps_at)?;
    let lifted = ps_epsilon(iid, n, x)?;
    let short = ps_length(l, n, x)?;

    Ok((short, lifted))
}
