from . import _core

# THE FINITE-KEY DIFFERENTIAL-PHASE-SHIFT LAYER, AND A DIFFERENT PROTOCOL FROM
# THE ONE q.Link RUNS. Mizutani, Takeuchi & Tamaki, "Finite-key security
# analysis of differential-phase-shift quantum key distribution", Phys. Rev.
# Research 5, 023132 (2023), arXiv:2301.09844, Corollary 1's Eq. (24) on
# Theorem 3's Eq. (42), by Koashi's complementarity rather than by a smooth
# min-entropy. Alice sends independent THREE-PULSE BLOCKS, at most one bit per
# block from its two interior slots; Bob reads them with TWO PHOTON-NUMBER-
# RESOLVING DETECTORS and sorts each detected round into a code or a sample
# round by an announced coin.
#
# q.Link with q.DifferentialPhase runs the ORIGINAL CONTINUOUS TRAIN of Inoue,
# Waks & Yamamoto, Phys. Rev. Lett. 89, 037902 (2002), over threshold detectors:
# no block boundary in the record and no code/sample coin. Its asymptotic rate
# is _core.dps_rate, Waks, Takesue & Yamamoto, Phys. Rev. A 73, 012344 (2006),
# individual attacks only and a bound on a COLLISION PROBABILITY rather than on
# a phase error -- so it has no finite-key sibling for this length to be.
#
# Substituting that link's click counts for N_det here keeps every two-photon
# round a resolving receiver discards -- exactly the rounds Theorem 3 prices
# through q_2 and q_3 -- so the bound moves in the INSECURE direction.
#
# The epsilon shape is qkd.security family 'dps', which composes Eq. (50)
# rather than summing the four terms; qkd.security.keylength refuses it and
# names this module, a ledger stating no receiver.


def source(mu):
    """
    `(q1, q2, q3)` of MTT23 Eq. (3) for the coherent source of their Eq. (51):
    the probability that one three-pulse block of mean photon number `mu` per
    pulse carries a or more photons in all optical modes, for a = 1, 2, 3.
    """

    return _core.dps_source(mu)


def counts(n_em, eta, mu, t):
    """
    `(n_det, n_code, n_samp)` under MTT23 Eq. (52). A FORWARD MODEL of their
    Sec. V simulation rather than a receiver: no dark floor, no misalignment and
    no jitter, and `n_det` counts blocks where EXACTLY ONE photon landed across
    the two interior slots. `t` is Bob's code-round probability.
    """

    return _core.dps_counts(n_em, eta, mu, t)


def optimum(n, m, eps):
    """
    `(a*, b*)` of MTT23 Eqs. (48) and (49): the pair minimising Kato's deviation
    term [b + a*(2m/n - 1)]*sqrt(n) at failure probability `eps` over `n`
    detected rounds, given the prediction `m` of the three-photon count. `m`
    must sit strictly below n/2, the paper's own condition.
    """

    return _core.dps_kato(n, m, eps)


def phase(n_em, n_code, n_samp, e_bit, q1, q2, q3, t, eps1, eps2, kato):
    """
    MTT23 Theorem 3, Eq. (42): the upper bound N_ph^U on the number of phase
    errors among the code rounds, IN ROUNDS rather than as a rate. `e_bit` is
    the error rate over the SAMPLE rounds alone, and N_det is `n_code + n_samp`
    rather than an argument. `kato` false takes their Eq. (53), which bounds the
    same three-photon sum by Azuma and is looser at every argument.

    Holds except with probability 3*eps1 + 3*eps2, and is written on the
    photon-number-resolving receiver `finite` names.
    """

    return _core.dps_phase(n_em, n_code, n_samp, e_bit, q1, q2, q3, t, eps1, eps2, kato)


def finite(
    n_em,
    n_code,
    n_samp,
    e_bit,
    q1,
    q2,
    q3,
    t,
    eps1,
    eps2,
    zeta,
    zeta_ec,
    f_ec,
    kato,
    *,
    detector,
):
    """
    MTT23 Corollary 1, Eq. (24): the secret key LENGTH in bits over the whole
    run, `n_code*[1 - h(N_ph^U/n_code)] - zeta - n_ec - zeta_ec`. `zeta` and
    `zeta_ec` are hash lengths in BITS, and the composed parameter is
    qkd.security.compose('dps', ...).secrecy.

    `detector` is KEYWORD-ONLY AND HAS NO DEFAULT. "pnr" runs; "threshold" is
    refused by the engine, which enumerates the four things a click record
    cannot supply.
    """

    return _core.dps_finite(
        n_em,
        n_code,
        n_samp,
        e_bit,
        q1,
        q2,
        q3,
        t,
        eps1,
        eps2,
        zeta,
        zeta_ec,
        f_ec,
        kato,
        detector,
    )
