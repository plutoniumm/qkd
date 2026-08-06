import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, load

from kit.anchors import ALPHA, E_DET, ETA_BOB, F_REC, Y_DARK
from kit.checks import Guarded
from kit.forms import bisect, h2, poisson
from qkd import _core

# GYS row: ALPHA, ETA_BOB, E_DET and Y_DARK are Fung, Tamaki & Lo's Table I verbatim, and
# F_REC is their f(E_mu) = 1.22. Fung, Tamaki & Lo, Phys. Rev. A 73, 012337 (2006),
# arXiv:quant-ph/0510025, Sec. V is the whole of the decoy comparison below.
#
# The photon-number-splitting exam is a DIFFERENT setting and a different security model:
# 0.25 dB/km and a detector efficiency of 0.1, the pair Branciard, Gisin, Kraus & Scarani,
# Phys. Rev. A 72, 032301 (2005) and Niederberger, Scarani & Gisin, Phys. Rev. A 71, 042316
# (2005) both draw their figures on. Neither closed form carries a dark count.
A_PNS = 0.25
ETA_PNS = 0.1

# Shor & Preskill's one-way BB84 threshold, the root of 1 = 2 h2(e), as Fung, Tamaki & Lo quote it.
E_BB84 = 0.110

# Nian, Nie, Zhang & Lu, Commun. Theor. Phys. 76, 065101 (2024), table 1, with its security parameters.
# NOT the GYS row: no counted-exam number compares with SargDecoy's.
NIAN_DARK = 6e-7
NIAN_EDET = 5e-3
NIAN_ETA = 0.1
NIAN_ALPHA = 0.2
NIAN_FEC = 1.16
NIAN_ESEC = 1e-9
NIAN_ECOR = 1e-15

# (mu, nu, p_mu) the counted exams read the chain at, not an optimiser's argument.
FIXED = (0.46, 0.23, 0.9)

# (mu, nu/mu, p_mu) grid the block ladder optimises over; the paper prints only a figure.
GRID = tuple(
    (round(0.32 + 0.03 * i, 2), share, p_mu) for i in range(7) for share in (0.3, 0.4, 0.5) for p_mu in (0.7, 0.9, 0.95)
)


def bell_cost(e1, a):
    """
    H(Z1|X1) at p_Y = a inside Fung, Tamaki & Lo Eq. (9)'s range e1/2 <= a <= e1.
    """
    px, py, pz = e1 - a, a, 1.5 * e1 - a
    parts = (1.0 - px - py - pz, px, py, pz)
    joint = sum(-w * math.log2(w) for w in parts if w > 0.0)

    return joint - h2(e1)


def worst_cost(e1, steps=4000):
    """
    The largest H(Z1|X1) over Fung, Tamaki & Lo Eq. (9)'s range e1/2 <= p_Y <= e1, by search.
    """
    lo, hi = 0.5 * e1, e1
    if hi <= lo:
        return bell_cost(e1, lo)

    return max(bell_cost(e1, lo + (hi - lo) * i / steps) for i in range(steps + 1))


def g_two(x):
    """
    Fung, Tamaki & Lo Eq. (10)'s g(x).
    """

    return (3.0 - 2.0 * x + math.sqrt(6.0 - 6.0 * math.sqrt(2.0) * x + 4.0 * x * x)) / 6.0


def eq_thirtyfour(e2):
    """
    Fung, Tamaki & Lo Eq. (34) at a = 0, Eq. (10)'s printed minimum, unclamped.
    """
    gap = 1.0 - 3.0 * e2

    return 0.5 - gap / (2.0 * math.sqrt(2.0)) + math.sqrt((1.0 - gap * gap) / 24.0)


def bell_max(e1):
    """
    H(Z1|X1) at the analytic maximiser of Eq. (9)'s range.
    """

    return bell_cost(e1, min(max(1.5 * e1 * e1, 0.5 * e1), e1))


def uncut_point(dist, mu):
    """
    Eq. (39) at ``dist`` km with neither cutoff: Eq. (9)'s maximiser and Eq. (34) unclamped.
    """
    q_mu, e_mu, q1, e1, q2, e2 = sarg_gains(dist, mu)
    single = q1 * (1.0 - bell_max(e1))
    double = q2 * (1.0 - h2(eq_thirtyfour(e2)))

    return single + double - q_mu * F_REC * h2(e_mu)


def sarg_gains(dist, mu):
    """
    Infinite-decoy SARG04 (Q_mu, E_mu, Q1, e1, Q2, e2) on the GYS channel at ``dist`` km.
    """
    eta = _core.decoy_eta(ALPHA, dist, ETA_BOB)
    q_mu, e_mu = _core.sarg_gain(mu, eta, Y_DARK, E_DET)
    y1, e1 = _core.sarg_yield(eta, Y_DARK, E_DET, 1)
    y2, e2 = _core.sarg_yield(eta, Y_DARK, E_DET, 2)
    weight = math.exp(-mu)

    return q_mu, e_mu, y1 * weight * mu, e1, y2 * weight * mu * mu / 2.0, e2


def sarg_point(dist, mu, two=True):
    """
    SARG04 rate at ``dist`` km and intensity ``mu``; ``two=False`` drops the two-photon term.
    """
    q_mu, e_mu, q1, e1, q2, e2 = sarg_gains(dist, mu)

    return _core.sarg_rate(q_mu, e_mu, q1, e1, q2 if two else 0.0, e2, F_REC)


def bb84_gains(dist, mu):
    """
    Fung, Tamaki & Lo's BB84 comparator, Eqs. (20) and (21); the yields carry the 1/2 sifting,
    so q_sift = 1.
    """
    eta = _core.decoy_eta(ALPHA, dist, ETA_BOB)
    vac = math.exp(-eta * mu)
    q_mu = 0.5 * (1.0 - vac + vac * Y_DARK)
    e_mu = (0.5 * E_DET * (1.0 - vac) + 0.25 * Y_DARK * vac) / q_mu
    y1 = 0.5 * (eta + (1.0 - eta) * Y_DARK)
    e1 = (0.5 * eta * E_DET + 0.25 * (1.0 - eta) * Y_DARK) / y1

    return q_mu, e_mu, y1 * math.exp(-mu) * mu, e1


def bb84_point(dist, mu):
    """
    BB84 rate at ``dist`` km and intensity ``mu``, Fung, Tamaki & Lo Eq. (12).
    """
    q_mu, e_mu, q1, e1 = bb84_gains(dist, mu)

    return _core.bb84_rate(1.0, q_mu, e_mu, q1, e1, F_REC)


def sarg_best(dist, two=True):
    """
    (rate, mu) at the optimal intensity.
    """

    return max((sarg_point(dist, i * 2e-3, two), i * 2e-3) for i in range(1, 401))


def bb84_best(dist):
    return max((bb84_point(dist, i * 2e-3), i * 2e-3) for i in range(1, 401))


def reach(photons, target):
    """
    GYS fibre length in km where the certified ``photons``-photon bit error rate reaches
    ``target``.
    """

    return bisect(
        lambda d: _core.sarg_yield(_core.decoy_eta(ALPHA, d, ETA_BOB), Y_DARK, E_DET, photons)[1] - target,
        1.0,
        400.0,
    )


def max_dist(fn):
    """
    The distance past which no intensity yields key, in km.
    """

    return bisect(lambda d: fn(d)[0], 1.0, 260.0)


def pns_sarg(t, mu):
    return _core.sarg_ceiling(mu, t, ETA_PNS)


def pns_bb84(t, mu):
    """
    Niederberger, Scarani & Gisin Eq. (28) at unit visibility and a Poissonian source: S =
    (eta/2)(mu t - mu^2/2), optimum Eq. (30) (eta/4) t^2 at mu = t.
    """

    return max(0.0, 0.5 * mu * ETA_PNS * (t - 0.5 * mu))


def pns_best(fn, t):
    """
    (rate, mu) at the optimal intensity; the grid reaches mu = 4, the short-distance optimum
    sitting above 1.
    """

    return max((fn(t, i * 5e-4), i * 5e-4) for i in range(1, 8001))


def log_slope(fn, hi, lo):
    """
    Log-log slope of an optimised ceiling against transmittance.
    """

    return math.log(pns_best(fn, hi)[0] / pns_best(fn, lo)[0]) / math.log(hi / lo)


def nian_eta(dist):
    """
    Alice-to-Bob transmittance at ``dist`` km on Nian, Nie, Zhang & Lu's table 1 fibre.
    """

    return _core.decoy_eta(NIAN_ALPHA, dist, NIAN_ETA)


def nian_taus(mu, nu, p_mu):
    """
    (tau_0, tau_1) over Alice's two intensities, to rebuild Rusca's bracket outside the engine.
    """
    p_nu = 1.0 - p_mu
    zero = p_mu * math.exp(-mu) + p_nu * math.exp(-nu)
    one = p_mu * math.exp(-mu) * mu + p_nu * math.exp(-nu) * nu

    return zero, one


def nian_record(dist, mu, nu, p_mu, n_z):
    """
    (counts, errors, gain, e_key) for ``n_z`` conclusive SARG04 detections, Nian Eqs. (14) and
    (15) with the basis and set probabilities cancelled.
    """
    eta = nian_eta(dist)
    q_mu, e_mu = _core.sarg_gain(mu, eta, NIAN_DARK, NIAN_EDET)
    q_nu, e_nu = _core.sarg_gain(nu, eta, NIAN_DARK, NIAN_EDET)
    gain = p_mu * q_mu + (1.0 - p_mu) * q_nu
    counts = (n_z * p_mu * q_mu / gain, n_z * (1.0 - p_mu) * q_nu / gain)
    errors = (counts[0] * e_mu, counts[1] * e_nu)

    return counts, errors, gain, (errors[0] + errors[1]) / n_z


def nian_chain(dist, mu, nu, p_mu, n_z):
    """
    (bits, s0, s1, v1, s0_hi, gain) from the shipped one-decoy chain.
    """
    counts, errors, gain, e_key = nian_record(dist, mu, nu, p_mu, n_z)
    eps = _core.sarg_eps(NIAN_ESEC)
    probs = (p_mu, 1.0 - p_mu)
    s0, s1, s0_hi = _core.sarg_counts(mu, nu, probs, counts, errors, eps)
    v1 = _core.sarg_errors(mu, nu, probs, errors, eps)
    bits = _core.sarg_length(s0, s1, v1, n_z, e_key, NIAN_FEC, NIAN_ESEC, NIAN_ECOR)

    return bits, s0, s1, v1, s0_hi, gain


def nian_bits(dist, mu, nu, p_mu, n_z):
    """
    (length in bits, conclusive gain per emitted pulse) from the shipped chain.
    """
    chain = nian_chain(dist, mu, nu, p_mu, n_z)

    return chain[0], chain[5]


def model_counts(dist, mu, nu, p_mu, n_z):
    """
    The model's vacuum, single-photon and single-photon-error counts in ``n_z`` detections: the
    truth the estimators bound.
    """
    eta = nian_eta(dist)
    gain = nian_record(dist, mu, nu, p_mu, n_z)[2]
    zero, one = nian_taus(mu, nu, p_mu)
    y0 = _core.sarg_yield(eta, NIAN_DARK, NIAN_EDET, 0)[0]
    y1, e1 = _core.sarg_yield(eta, NIAN_DARK, NIAN_EDET, 1)

    return n_z * zero * y0 / gain, n_z * one * y1 / gain, n_z * one * y1 * e1 / gain


def printed_single(dist, mu, nu, p_mu, n_z):
    """
    Nian Eq. (6) as printed: the vacuum ceiling subtracted WITHOUT Rusca Eq. (A17)'s 1/tau_0.
    """
    counts = nian_record(dist, mu, nu, p_mu, n_z)[0]
    s0_hi = nian_chain(dist, mu, nu, p_mu, n_z)[4]
    one = nian_taus(mu, nu, p_mu)[1]
    spread = math.sqrt(0.5 * (counts[0] + counts[1]) * math.log(1.0 / _core.sarg_eps(NIAN_ESEC)))
    hi = math.exp(mu) / p_mu * (counts[0] + spread)
    lo = math.exp(nu) / (1.0 - p_mu) * (counts[1] - spread)
    share = (mu * mu - nu * nu) / (mu * mu)
    bracket = lo - nu * nu / (mu * mu) * hi - share * s0_hi

    return max(0.0, one * mu / (nu * (mu - nu)) * bracket)


def printed_bits(dist, mu, nu, p_mu, n_z):
    """
    The key length Nian Eq. (6) as printed would report.
    """
    counts, errors, gain, e_key = nian_record(dist, mu, nu, p_mu, n_z)
    eps = _core.sarg_eps(NIAN_ESEC)
    probs = (p_mu, 1.0 - p_mu)
    s0 = _core.sarg_counts(mu, nu, probs, counts, errors, eps)[0]
    s1 = min(printed_single(dist, mu, nu, p_mu, n_z), n_z - s0)
    v1 = _core.sarg_errors(mu, nu, probs, errors, eps)

    return _core.sarg_length(s0, s1, v1, n_z, e_key, NIAN_FEC, NIAN_ESEC, NIAN_ECOR), gain


def nian_best(dist, n_z, form=nian_bits):
    """
    (rate per emitted pulse, (mu, nu, p_mu)) at the best point of GRID.
    """
    top, arg = 0.0, None
    for mu, share, p_mu in GRID:
        nu = round(mu * share, 4)
        bits, gain = form(dist, mu, nu, p_mu, n_z)
        rate = bits / n_z * gain
        if rate > top:
            top, arg = rate, (mu, nu, p_mu)

    return top, arg


def nian_reach(n_z, form=nian_bits):
    """
    The fibre length past which no point of GRID yields key at that block size.
    """

    return bisect(lambda d: nian_best(d, n_z, form)[0], 1.0, 260.0)


def nian_asym(dist, mu, two=True):
    """
    (rate per emitted pulse, conclusive gain) of infinite-decoy SARG04 on the table 1 channel.
    """
    eta = nian_eta(dist)
    q_mu, e_mu = _core.sarg_gain(mu, eta, NIAN_DARK, NIAN_EDET)
    y1, e1 = _core.sarg_yield(eta, NIAN_DARK, NIAN_EDET, 1)
    y2, e2 = _core.sarg_yield(eta, NIAN_DARK, NIAN_EDET, 2)
    weight = math.exp(-mu)
    q2 = y2 * weight * mu * mu / 2.0 if two else 0.0

    return _core.sarg_rate(q_mu, e_mu, y1 * weight * mu, e1, q2, e2 if two else 0.0, NIAN_FEC), q_mu


def asym_best(dist, two=True):
    """
    The infinite-decoy rate at the best of 400 intensities on the table 1 channel.
    """

    return max(nian_asym(dist, i * 4e-3, two)[0] for i in range(1, 401))


class SargSifting(Guarded):
    """
    Alice announces a non-orthogonal PAIR and Bob keeps the rounds that conclusively exclude a
    member -- Fung, Tamaki & Lo, Phys. Rev. A 73, 012337 (2006) Eqs. (40)-(43), after Scarani,
    Acin, Ribordy & Gisin, Phys. Rev. Lett. 92, 057901 (2004).
    """

    def test_ideal_quarter(self):
        """
        `sarg_sift` is 1/4 at aligned detectors, against BB84's 1/2, and RISES with misalignment
        to 0.2665 at the GYS e_det.
        """
        got = _core.sarg_sift(0.0)

        self.assertEqual(got, 0.25, msg="sift != 1/4")

        rising = [_core.sarg_sift(e) for e in (0.0, 0.01, 0.033, 0.1, 0.5)]

        self.assertMonotone(rising, rising=True, msg="sift not rising with e_det")
        self.assertClose(_core.sarg_sift(E_DET), 0.2665, atol=1e-15, msg="GYS conclusive fraction")

    def test_yield_is_half_bb84(self):
        """
        The n-photon yield is half BB84's at aligned detectors with no dark counts, and the GYS
        bit error rate 1.876 times BB84's at n = 1 and 2.
        """
        for n in (1, 2, 3, 4):
            seen = 1.0 - (1.0 - 0.1) ** n
            got = _core.sarg_yield(0.1, 0.0, 0.0, n)[0]
            self.assertClose(got, 0.25 * seen, atol=1e-15, msg=f"Y_{n} != half BB84")

        eta = _core.decoy_eta(ALPHA, 25.0, ETA_BOB)
        for n in (1, 2):
            seen = 1.0 - (1.0 - eta) ** n
            y_bb = 0.5 * (seen + (1.0 - seen) * Y_DARK)
            e_bb = (0.5 * seen * E_DET + 0.25 * (1.0 - seen) * Y_DARK) / y_bb
            got = _core.sarg_yield(eta, Y_DARK, E_DET, n)[1]
            self.assertClose(got / e_bb, 1.876, atol=1e-3, msg=f"e_{n} against BB84's")

    def test_gain_is_poisson_sum(self):
        """
        `sarg_gain`'s gain and QBER equal the Poisson sums of the per-photon-number yields to
        1e-15.
        """
        eta = _core.decoy_eta(ALPHA, 25.0, ETA_BOB)
        mu = 0.3
        total, errors = 0.0, 0.0
        for n in range(0, 60):
            weight = math.exp(-mu) * mu**n / math.factorial(n)
            y_n, e_n = _core.sarg_yield(eta, Y_DARK, E_DET, n)
            total += weight * y_n
            errors += weight * y_n * e_n

        q_mu, e_mu = _core.sarg_gain(mu, eta, Y_DARK, E_DET)

        self.assertClose(total, q_mu, atol=1e-15, msg="gain != Poisson sum")
        self.assertClose(errors / total, e_mu, atol=1e-15, msg="QBER != Poisson sum")

    def test_vacuum_yield(self):
        """
        Y_0 is half the background and e_0 is 1/2, and a vanishing yield returns e = 1/2 rather
        than dividing by zero.
        """
        y_0, e_0 = _core.sarg_yield(0.1, 1.7e-6, 0.033, 0)

        self.assertClose(y_0, 0.85e-6, atol=1e-18, msg="Y_0 != half the background")
        self.assertEqual(e_0, 0.5, msg="e_0 != 1/2")

        got = _core.sarg_yield(0.0, 0.0, 0.0, 2)

        self.assertEqual(got, (0.0, 0.5), msg="(Y_2, e_2) != (0, 1/2)")


class SargBounds(Guarded):
    """
    Bit and phase errors are CORRELATED in SARG04: Fung, Tamaki & Lo Eqs. (9) and (10), after
    Tamaki & Lo, Phys. Rev. A 73, 010302(R) (2006).
    """

    def test_phase_is_half_again(self):
        """
        The one-photon phase error rate is 1.5 e1 at the worst case p_Y = e1/2, Tamaki and Lo's
        p_ph = 1.5 p_bit, and the charged H(Z1|X1) sits strictly below its h2.
        """
        for e1 in (0.0, 0.01, 0.05, 0.0968, 0.2):
            got = _core.sarg_cost(e1)[1]
            self.assertClose(got, 1.5 * e1, atol=1e-15, msg=f"e_p1 at e1 = {e1}")

        for e1 in (0.01, 0.05, 0.0968, 0.2):
            cost, e_p1 = _core.sarg_cost(e1)
            self.assertLess(cost, h2(e_p1), msg=f"cost >= h2(e_p) at e1 = {e1}")

    def test_worst_case_weight(self):
        """
        `sarg_cost` equals the searched worst H(Z1|X1) over $e_1/2 \\le p_Y \\le e_1$, at p_Y =
        e1/2, below e1 = 1/3 and charges a full bit above.
        """
        for e1 in (0.01, 0.05, 0.0968):
            grid = [bell_cost(e1, 0.5 * e1 + i * 0.5 * e1 / 200.0) for i in range(201)]
            self.assertClose(max(grid), _core.sarg_cost(e1)[0], atol=1e-14, msg=f"worst p_Y at {e1}")

        for e1 in (0.01, 0.05, 0.0968, 0.2, 0.3):
            self.assertClose(
                _core.sarg_cost(e1)[0],
                worst_cost(e1),
                atol=1e-12,
                msg=f"cost != worst case at e1 = {e1}",
            )

        for e1 in (0.34, 0.4, 0.45, 0.5):
            self.assertGreater(
                _core.sarg_cost(e1)[0],
                worst_cost(e1),
                msg=f"cost <= worst case at e1 = {e1}",
            )

    def test_maximiser_moves(self):
        """
        Above e1 = 1/3 the independence point p_Y = 1.5 e1^2 enters Eq. (9)'s range and
        maximises: 0.9097 against 0.8262 at p_Y = e1/2, at e1 = 0.45.
        """
        peak = 1.5 * 0.45 * 0.45

        self.assertClose(
            bell_cost(0.45, 0.225),
            0.82622114,
            atol=1e-8,
            msg="p_Y = e1/2 at e1 = 0.45",
        )
        self.assertClose(
            bell_cost(0.45, peak),
            0.90973612,
            atol=1e-8,
            msg="p_Y = 1.5 e1^2 at e1 = 0.45",
        )
        self.assertClose(
            worst_cost(0.45),
            bell_cost(0.45, peak),
            atol=1e-9,
            msg="worst case != the independence point",
        )

        for e1 in (0.34, 0.4, 0.45, 0.5):
            self.assertGreater(
                worst_cost(e1),
                bell_cost(e1, 0.5 * e1),
                msg=f"p_Y = e1/2 still worst at e1 = {e1}",
            )

    def test_no_key_above_a_third(self):
        """
        No single-photon bit error rate at or above 1/3 certifies key at any f_ec, H(Z1|X1)
        being one bit and the phase error 1/2 there, while 0.33 still certifies.
        """
        self.assertEqual(
            _core.sarg_rate(0.01, 0.001, 0.005, 0.45, 0.0, 0.0, 1.0),
            0.0,
            msg="key at e1 = 0.45",
        )

        for e1 in (1.0 / 3.0, 0.34, 0.36, 0.4, 0.45, 0.49, 0.5):
            self.assertEqual(
                _core.sarg_rate(1.0, 0.0, 1.0, e1, 0.0, 0.0, 1.0),
                0.0,
                msg=f"key at f_ec = 1, e1 = {e1}",
            )
        self.assertClose(_core.sarg_cost(1.0 / 3.0)[0], 1.0, atol=1e-15, msg="one bit at e1 = 1/3")
        self.assertClose(_core.sarg_cost(1.0 / 3.0)[1], 0.5, atol=1e-15, msg="e_p1 = 1/2 at e1 = 1/3")
        self.assertGreater(
            _core.sarg_rate(1.0, 0.0, 1.0, 0.33, 0.0, 0.0, 1.0),
            0.0,
            msg="no key at e1 = 0.33",
        )
        self.assertEqual(
            _core.sarg_rate(1.0, 0.0, 1.0, 1.0 / 3.0, 0.0, 0.0, 1.0),
            0.0,
            msg="key at e1 = 1/3",
        )

    def test_cutoff_leaves_no_trace(self):
        """
        Past either cutoff `sarg_cost` returns one bit and `sarg_phase` 1/2, the same constants
        at the edge and far past it, so the value does not say how far past a channel is.
        """
        for e1 in (1.0 / 3.0, 0.36, 0.45, 0.5):
            self.assertEqual(_core.sarg_cost(e1)[0], 1.0, msg=f"one bit charged at e1 = {e1}")
            self.assertEqual(_core.sarg_cost(e1)[1], 0.5, msg=f"e_p1 pinned at 1/2 for e1 = {e1}")

        for e2 in (1.0 / 6.0, 0.2, 0.35, 0.5):
            self.assertEqual(_core.sarg_phase(e2), 0.5, msg=f"e_p2 pinned at 1/2 for e2 = {e2}")
        self.assertGreater(
            eq_thirtyfour(0.35),
            _core.sarg_phase(0.35),
            msg="Eq. (10) not above sarg_phase at 0.35",
        )

    def test_domain_sees_crossing(self):
        """
        `sarg_domain`'s live1 and live2 track `sarg_cost` below 1 and `sarg_phase` below 1/2,
        flipping exactly at the returned 1/3 and 1/6, with e2 = 0 keeping live2 true and e1 or
        e2 outside [0, 1/2] refused.
        """
        for i in range(51):
            e = i / 100
            live1, live2, e1_dead, e2_dead = _core.sarg_domain(e, e)
            self.assertEqual((e1_dead, e2_dead), (1.0 / 3.0, 1.0 / 6.0), msg="cutoffs != (1/3, 1/6)")
            self.assertEqual(live1, _core.sarg_cost(e)[0] < 1.0, msg=f"live1 != cost < 1 at {e}")
            self.assertEqual(live2, _core.sarg_phase(e) < 0.5, msg=f"live2 != phase < 1/2 at {e}")

        for dead, slot in ((1.0 / 3.0, 0), (1.0 / 6.0, 1)):
            args = [0.0, 0.0]
            args[slot] = dead - 1e-12
            self.assertTrue(_core.sarg_domain(*args)[slot], msg=f"dead just under {dead}")
            args[slot] = dead
            self.assertFalse(_core.sarg_domain(*args)[slot], msg=f"dead exactly at {dead}")
        for e1 in (0.0, 0.3, 0.5):
            self.assertTrue(_core.sarg_domain(e1, 0.0)[1], msg=f"live2 false at e2 = 0, e1 = {e1}")

        for bad in (-0.01, 0.51):
            self.assertBad("e1 must be in [0, 1/2]", _core.sarg_domain, (bad, 0.0), msg=f"e1 = {bad}")
            self.assertBad("e2 must be in [0, 1/2]", _core.sarg_domain, (0.0, bad), msg=f"e2 = {bad}")

    def test_bracket_ends_at_a_third(self):
        """
        1 - h2(e) - H(Z1|X1) turns upward past e1 = 0.385, so the one-photon bisection brackets
        on (0, 1/3], with `sarg_limit(1)` inside it.
        """
        curve = [1.0 - h2(e) - worst_cost(e) for e in (0.38, 0.4, 0.45, 0.5)]

        self.assertMonotone(curve, rising=True, msg="1 - h2 - cost not rising past 0.385")
        self.assertLess(_core.sarg_limit(1), 1.0 / 3.0, msg="sarg_limit(1) >= 1/3")

    def test_phase_minimises_g(self):
        """
        `sarg_phase` agrees to 1e-7 with a direct minimisation of x e2 + g(x) over a wide grid
        of x.
        """
        for e2 in (0.005, 0.0271, 0.05, 0.1, 0.16):
            grid = min(x * 1e-3 * e2 + g_two(x * 1e-3) for x in range(-200000, 200001))
            self.assertClose(_core.sarg_phase(e2), grid, atol=1e-7, msg=f"min at {e2}")

    def test_phase_is_equation_34(self):
        """
        `sarg_phase` equals Fung, Tamaki & Lo Eq. (34) at a = 0 to 1e-15 across its domain,
        reaching their $\\sin^2(\\pi/8)$ floor at zero bit error.
        """
        self.assertClose(
            _core.sarg_phase(0.0),
            math.sin(math.pi / 8.0) ** 2,
            atol=1e-15,
            msg="two-photon phase floor",
        )

        for e2 in (0.0, 0.01, 0.0271, 0.05, 0.1, 0.16):
            self.assertClose(
                _core.sarg_phase(e2),
                eq_thirtyfour(e2),
                atol=1e-15,
                msg=f"Eq. (34) at e2 = {e2}",
            )

    def test_phase_departs_past_a_sixth(self):
        """
        `sarg_phase` reaches 1/2 at e2 = 1/6 and is capped there, departing from Eq. (10)'s
        0.6093 at 0.25 and 0.6677 at 0.30, which unclamped would credit 0.0347 bit per
        two-photon detection at 0.25.
        """
        self.assertClose(_core.sarg_phase(1.0 / 6.0), 0.5, atol=1e-15, msg="sarg_phase(1/6) != 1/2")
        self.assertLess(_core.sarg_phase(0.16), 0.5, msg="sarg_phase(0.16) >= 1/2")
        self.assertEqual(_core.sarg_phase(0.4), 0.5, msg="sarg_phase(0.4) != 1/2")

        for e2, want in ((0.25, 0.60925401), (0.3, 0.66774562)):
            self.assertClose(eq_thirtyfour(e2), want, atol=1e-8, msg=f"Eq. (10) at e2 = {e2}")
            self.assertEqual(_core.sarg_phase(e2), 0.5, msg=f"sarg_phase != 1/2 at e2 = {e2}")
            self.assertGreater(
                1.0 - h2(eq_thirtyfour(e2)),
                0.0,
                msg=f"unclamped Eq. (10) credits nothing at {e2}",
            )
        self.assertClose(
            1.0 - h2(eq_thirtyfour(0.25)),
            0.0347207,
            atol=1e-7,
            msg="declined key != 0.0347",
        )
        self.assertEqual(
            1.0 - h2(_core.sarg_phase(0.25)),
            0.0,
            msg="clamped bound credits key",
        )

    def test_tolerable_limits(self):
        """
        The one- and two-photon tolerable bit error rates are 9.689% and 2.710%, against the
        9.68% and 2.71% Tamaki and Lo publish, both below BB84's 11.0%.
        """
        one = _core.sarg_limit(1)
        two = _core.sarg_limit(2)

        self.assertClose(one, 0.0968, atol=1.5e-4, msg="published 9.68% single-photon")
        self.assertClose(one, 0.09689249, atol=1e-8, msg="computed single-photon limit")
        self.assertClose(two, 0.0271, atol=1e-5, msg="published 2.71% two-photon")
        self.assertClose(two, 0.02710093, atol=1e-8, msg="computed two-photon limit")
        self.assertLess(one, E_BB84, msg="one-photon limit above BB84's")
        self.assertLess(two, one, msg="two-photon limit above one-photon")

    def test_limit_is_a_root(self):
        """
        The one-way distillation rate 1 - H(X) - H(Z|X) is zero at `sarg_limit(1)`, positive a
        tenth below it and negative a tenth above.
        """
        lim = _core.sarg_limit(1)

        self.assertClose(
            1.0 - h2(lim) - _core.sarg_cost(lim)[0],
            0.0,
            atol=1e-12,
            msg="root at 9.69%",
        )
        self.assertGreater(
            1.0 - h2(0.9 * lim) - _core.sarg_cost(0.9 * lim)[0],
            0.0,
            msg="below the root",
        )
        self.assertLess(
            1.0 - h2(1.1 * lim) - _core.sarg_cost(1.1 * lim)[0],
            0.0,
            msg="above the root",
        )


class SargDecoy(Guarded):
    """
    Infinite-decoy SARG04 against BB84 on the GYS parameters, Fung, Tamaki & Lo, Phys. Rev. A
    73, 012337 (2006), Sec. V and Fig. 1; with decoys BB84 leads at every distance, the known
    crossover belonging to SargSplitting.
    """

    def test_gys_secure_distance(self):
        """
        Fung, Tamaki and Lo's 97.2 km maximum secure distance for decoy SARG04 at the optimal
        intensity, of which the two-photon term buys 2.72 km.
        """
        both = max_dist(sarg_best)
        one = max_dist(lambda d: sarg_best(d, two=False))

        self.assertClose(both, 97.2, atol=0.15, msg="published SARG04 cutoff 97.2 km")
        self.assertGreater(both, one, msg=f"reach {both}, one-photon {one}")
        self.assertClose(both - one, 2.72, atol=0.1, msg="two-photon reach in km")

    def test_two_photon_share(self):
        """
        The two-photon term is 13.7% of the optimised rate at 0 km and 20.1% at 75 km, their
        "small contribution at all distances".
        """
        for dist, want in ((0.0, 0.1371), (75.0, 0.2005)):
            both = sarg_best(dist)[0]
            one = sarg_best(dist, two=False)[0]
            self.assertClose(
                (both - one) / both,
                want,
                atol=2e-3,
                msg=f"two-photon share at {dist} km",
            )

    def test_optimal_intensity(self):
        """
        The optimal mean photon number over 0-75 km is 0.22 for SARG04 and 0.485 for BB84, the
        two curves of their Fig. 3.
        """
        for dist in (0.0, 25.0, 50.0, 75.0):
            self.assertClose(sarg_best(dist)[1], 0.22, atol=0.01, msg=f"SARG04 mu at {dist} km")
            self.assertClose(bb84_best(dist)[1], 0.485, atol=0.01, msg=f"BB84 mu at {dist} km")

    def test_bb84_ahead_everywhere(self):
        """
        With decoys BB84 beats SARG04 at every distance to 90 km, 10.11 times at 0 km, and
        reaches over 40 km further.
        """
        for dist in (0.0, 25.0, 50.0, 75.0, 90.0):
            self.assertGreater(bb84_best(dist)[0], sarg_best(dist)[0], msg=f"BB84 ahead at {dist} km")
        self.assertClose(bb84_best(0.0)[0] / sarg_best(0.0)[0], 10.11, atol=0.05, msg="0 km ratio")
        self.assertGreater(max_dist(bb84_best), max_dist(sarg_best) + 40.0, msg="reach gap")

    def test_distance_upper_bounds(self):
        """
        GYS distances where the certified error rates reach their ceilings: e1 = 1/3 at 207.68
        km, e2 = 1/6 at 187.99 km, and the two-photon ceiling at 201.4266 km from the printed
        0.2265 against 201.4348 km from Theorem 2's $(3 - \\sqrt{2})/7$, 8 m apart.
        """
        one = reach(1, 1.0 / 3.0)
        two = reach(2, 0.2265)
        exact = reach(2, (3.0 - math.sqrt(2.0)) / 7.0)
        dead = reach(2, 1.0 / 6.0)

        self.assertClose(one, 207.6802, atol=1e-3, msg="e1 = 1/3 distance")
        self.assertClose(two, 201.4266, atol=1e-3, msg="rounded e2 = 0.2265 distance")
        self.assertClose(exact, 201.4348, atol=1e-3, msg="exact e2 = (3 - sqrt2)/7 distance")
        self.assertClose(dead, 187.9910, atol=1e-3, msg="e2 = 1/6 distance")
        self.assertLess(dead, two, msg="e2 = 1/6 not before the ceiling")
        self.assertLess(dead, one, msg="e2 = 1/6 not before e1 = 1/3")

    def test_cutoff_never_moves_gys(self):
        """
        Neither cutoff moves the GYS rate: the two- and one-photon terms die 90.69 km and 110.38
        km past where the optimised rate is already zero, and Eq. (39) read literally stays
        negative to 300 km.
        """
        edge = max_dist(sarg_best)

        self.assertClose(
            reach(2, 1.0 / 6.0) - edge,
            90.69,
            atol=0.02,
            msg="two-photon crossing past death",
        )
        self.assertClose(
            reach(1, 1.0 / 3.0) - edge,
            110.38,
            atol=0.02,
            msg="one-photon crossing past death",
        )

        for dist in (200.0, 250.0, 300.0):
            loose = max(uncut_point(dist, i * 5e-3) for i in range(1, 161))
            self.assertLess(loose, 0.0, msg=f"uncut Eq. (39) {loose} at {dist} km")
            self.assertEqual(
                sarg_best(dist)[0],
                0.0,
                msg=f"rate != 0 at {dist} km",
            )

    def test_two_photon_uncertified(self):
        """
        A three-intensity record admits Y2 = 0: zeroing Y2 and raising Y1 by $\\mu\\nu
        Y_2/(2(\\mu + \\nu))$ and Y3 by $3 Y_2/(\\mu + \\nu)$ reproduces all three GYS gains to
        4.4e-19 with every yield in [0, 1].
        """
        mu, nu = 0.22, 0.1
        for dist in (0.0, 25.0, 50.0, 75.0):
            eta = _core.decoy_eta(ALPHA, dist, ETA_BOB)
            true = [_core.sarg_yield(eta, Y_DARK, E_DET, n)[0] for n in range(61)]
            spare = list(true)
            spare[1] += mu * nu * true[2] / (2.0 * (mu + nu))
            spare[3] += 3.0 * true[2] / (mu + nu)
            spare[2] = 0.0
            self.assertTrue(
                all(0.0 <= y <= 1.0 for y in spare),
                msg=f"witness yield outside [0, 1] at {dist} km",
            )
            self.assertGreater(true[2], 0.0, msg=f"true Y2 <= 0 at {dist} km")

            for level in (mu, nu, 0.0):
                want = sum(poisson(level, n) * true[n] for n in range(61))
                got = sum(poisson(level, n) * spare[n] for n in range(61))
                self.assertClose(got, want, atol=1e-18, msg=f"gain at mu = {level}, {dist} km")

    def test_decoy_layer_transfers(self):
        """
        The Ma-Qi-Zhao-Lo inversion in decoy_bounds is protocol-blind: fed SARG04 conclusive
        gains, it brackets the infinite-decoy Y1, e1 and Q1 from the correct sides.
        """
        eta = _core.decoy_eta(ALPHA, 25.0, ETA_BOB)
        mu, nu1 = 0.3, 0.1
        q_mu, e_mu = _core.sarg_gain(mu, eta, Y_DARK, E_DET)
        q_nu, e_nu = _core.sarg_gain(nu1, eta, Y_DARK, E_DET)
        y1_lo, e1_hi, q1_lo = _core.decoy_bounds(mu, nu1, 0.0, q_mu, e_mu, q_nu, e_nu, 0.5 * Y_DARK, 0.5)
        y1, e1 = _core.sarg_yield(eta, Y_DARK, E_DET, 1)

        self.assertLess(y1_lo, y1, msg="y1_lo above the true Y1")
        self.assertGreater(e1_hi, e1, msg="e1_hi below the true e1")
        self.assertLess(q1_lo, y1 * mu * math.exp(-mu), msg="q1_lo above the true Q1")

    def test_rate_falls_with_distance(self):
        """
        The optimised rate falls with fibre length and is exactly 0.0, never negative, past the
        maximum distance.
        """
        rates = [sarg_best(d)[0] for d in (0.0, 20.0, 40.0, 60.0, 80.0, 95.0)]

        self.assertMonotone(rates, rising=False, msg=f"rates: {rates}")

        for dist in (100.0, 150.0, 250.0):
            self.assertEqual(sarg_best(dist)[0], 0.0, msg=f"rate != 0 at {dist} km")

    def test_order_of_transmittance(self):
        """
        Both decoy rates are O(eta), but over 25-75 km SARG04's log-log slope against
        transmittance is 1.166 against BB84's 1.018.
        """
        span = math.log(10 ** (ALPHA * 50.0 / 10.0))
        near = math.log(sarg_best(25.0)[0] / sarg_best(75.0)[0]) / span
        far = math.log(bb84_best(25.0)[0] / bb84_best(75.0)[0]) / span

        self.assertClose(near, 1.166, atol=0.01, msg="SARG04 log-log slope")
        self.assertClose(far, 1.018, atol=0.01, msg="BB84 log-log slope")


class SargSplitting(Guarded):
    """
    Branciard, Gisin, Kraus & Scarani, Phys. Rev. A 72, 032301 (2005) Eq. (106) against
    Niederberger, Scarani & Gisin, Phys. Rev. A 71, 042316 (2005) Eq. (28): incoherent attacks,
    no decoys, unit visibility, CEILINGS rather than security bounds.
    """

    def test_optimal_intensity(self):
        """
        The ceiling peaks at $\\mu = 2\\sqrt{t}$, Branciard Eq. (107), where BB84's peaks at
        $\\mu = t$, Niederberger Eq. (29), equalling Branciard's $(\\eta/3) h_2(p_1) t^{3/2}$
        with $h_2(p_1) = 0.60088$.
        """
        left = h2(0.5 + 0.5 / math.sqrt(2.0))

        self.assertClose(left, 0.60087604, atol=1e-8, msg="1 - I_S(1)")

        for t in (0.5, 0.1, 0.01):
            want = ETA_PNS / 3.0 * left * t**1.5
            peak = pns_best(pns_sarg, t)
            self.assertClose(peak[1], 2.0 * math.sqrt(t), atol=1e-3, msg=f"mu at t = {t}")
            self.assertClose(pns_best(pns_bb84, t)[1], t, atol=1e-3, msg=f"BB84 mu at {t}")
            self.assertClose(peak[0], want, atol=1e-6 * want, msg=f"optimum at {t}")

    def test_scaling_exponents(self):
        """
        The optimised ceilings scale as $t^{3/2}$ for SARG04 and $t^2$ for BB84, the advantage
        widening as the square root of the loss: 2.5335 at t = 0.1, 25.335 at t = 0.001.
        """
        self.assertClose(
            log_slope(pns_sarg, 0.1, 1e-3),
            1.5,
            atol=1e-4,
            msg="SARG04 slope != 1.5",
        )
        self.assertClose(log_slope(pns_bb84, 0.1, 1e-3), 2.0, atol=1e-4, msg="BB84 slope != 2")

        for trans, want in ((0.1, 2.5335), (1e-3, 25.335)):
            ratio = pns_best(pns_sarg, trans)[0] / pns_best(pns_bb84, trans)[0]
            self.assertClose(ratio, want, atol=1e-3 * want, msg=f"advantage at t = {trans}")

    def test_crossover_exists(self):
        """
        The two ceilings cross once, at 7.702 km of 0.25 dB/km fibre, BB84 above before it and
        SARG04 after.
        """
        cross = bisect(
            lambda d: pns_best(pns_sarg, 10 ** (-A_PNS * d / 10.0))[0]
            - pns_best(pns_bb84, 10 ** (-A_PNS * d / 10.0))[0],
            0.0,
            40.0,
        )

        self.assertClose(cross, 7.702, atol=0.01, msg="crossover distance in km")

        for dist in (0.0, 2.0, 5.0, 7.0):
            trans = 10 ** (-A_PNS * dist / 10.0)
            self.assertLess(
                pns_best(pns_sarg, trans)[0],
                pns_best(pns_bb84, trans)[0],
                msg=f"BB84 not ahead at {dist} km",
            )

        for dist in (10.0, 20.0, 50.0):
            trans = 10 ** (-A_PNS * dist / 10.0)
            self.assertGreater(
                pns_best(pns_sarg, trans)[0],
                pns_best(pns_bb84, trans)[0],
                msg=f"SARG04 not ahead at {dist} km",
            )

    def test_ceiling_root(self):
        """
        `sarg_ceiling` vanishes at $\\mu = 2\\sqrt{3t}$ and clamps to zero past it.
        """
        for t in (0.5, 0.1, 0.01):
            edge = 2.0 * math.sqrt(3.0 * t)
            self.assertClose(
                _core.sarg_ceiling(edge, t, ETA_PNS),
                0.0,
                atol=1e-16,
                msg=f"root at {t}",
            )
            self.assertGreater(
                _core.sarg_ceiling(0.999 * edge, t, ETA_PNS),
                0.0,
                msg=f"below root at {t}",
            )
            self.assertEqual(_core.sarg_ceiling(1.5 * edge, t, ETA_PNS), 0.0, msg=f"clamped past {t}")


class SargCounted(Guarded):
    """
    The one-decoy estimators, Rusca, Boaron, Grunenfelder, Martin & Zbinden, Appl. Phys. Lett.
    112, 171104 (2018), arXiv:1801.03443, Eqs. (A14) to (A19) and (A22), as Nian, Nie, Zhang &
    Lu, Commun. Theor. Phys. 76, 065101 (2024) call them for SARG04, against the counts the
    channel model put in the block.
    """

    def test_epsilon_terms(self):
        """
        `sarg_eps` divides eps_sec by 18, Nian Eq. (13) at one common value, not by Lim's 21 or
        Rusca's one-decoy 19.
        """
        self.assertEqual(_core.sarg_eps(NIAN_ESEC), NIAN_ESEC / 18.0, msg="eps_sec/18")
        self.assertEqual(
            2.0 * (2.0 + 1.0 + 1.0) + 1.0 + 4.0 + 5.0,
            18.0,
            msg="Eq. (13) counts 18 terms at one common value",
        )
        self.assertGreater(
            _core.sarg_eps(NIAN_ESEC),
            _core.bb84_eps(NIAN_ESEC),
            msg="sarg_eps not above bb84_eps",
        )

    def test_bounds_bracket(self):
        """
        At a 1e10 block s0, s1 and v1 bracket the model's counts from the right sides, s1 at
        0.79 to 0.85 of the true single-photon count and v1 at 1.73 to 1.91 times the true error
        count.
        """
        mu, nu, p_mu = FIXED
        floors = (0.84628, 0.83752, 0.83273, 0.79387)
        roofs = (1.86109, 1.91021, 1.89049, 1.73327)
        for dist, floor, roof in zip((0.0, 50.0, 100.0, 150.0), floors, roofs):
            s0, s1, v1 = nian_chain(dist, mu, nu, p_mu, 1e10)[1:4]
            zero, one, wrong = model_counts(dist, mu, nu, p_mu, 1e10)
            self.assertLessEqual(s0, zero, msg=f"s0 above the model count at {dist} km")
            self.assertLess(s1, one, msg=f"s1 above the model count at {dist} km")
            self.assertGreater(v1, wrong, msg=f"v1 below the model count at {dist} km")
            self.assertClose(s1 / one, floor, atol=1e-5, msg=f"s1 share at {dist} km")
            self.assertClose(v1 / wrong, roof, atol=1e-5, msg=f"v1 excess at {dist} km")

    def test_vacuum_ceiling(self):
        """
        The vacuum ceiling is the max over Rusca Eqs. (A14) and (A16) at both intensities; at 50
        km the signal's Eq. (A16) wins by 7.9%, 123.18 times the vacuum count it bounds.
        """
        mu, nu, p_mu = FIXED
        counts, errors = nian_record(50.0, mu, nu, p_mu, 1e10)[:2]
        s0_hi = nian_chain(50.0, mu, nu, p_mu, 1e10)[4]
        eps = _core.sarg_eps(NIAN_ESEC)
        zero = nian_taus(mu, nu, p_mu)[0]
        seen = math.sqrt(0.5 * (counts[0] + counts[1]) * math.log(1.0 / eps))
        wrong = math.sqrt(0.5 * (errors[0] + errors[1]) * math.log(1.0 / eps))
        plain = 2.0 * (errors[0] + errors[1] + seen)
        signal = 2.0 * (zero * math.exp(mu) / p_mu * (errors[0] + wrong) + seen)
        decoy = 2.0 * (zero * math.exp(nu) / (1.0 - p_mu) * (errors[1] + wrong) + seen)

        self.assertClose(s0_hi, max(plain, signal, decoy), atol=1.0, msg="Eq. (7) is a max")
        self.assertClose(signal / plain, 1.079175, atol=1e-5, msg="Eq. (A16) at mu beats Eq. (A14)")
        self.assertLess(decoy, plain, msg="decoy Eq. (A16) not the loosest")
        self.assertClose(
            s0_hi / model_counts(50.0, mu, nu, p_mu, 1e10)[0],
            123.1776,
            atol=1e-3,
            msg="ceiling/vacuum != 123.1776",
        )

    def test_floors_never_meet(self):
        """
        The vacuum floor is zero to 200 km and positive only past 219.76 km, where the
        single-photon floor has died, so every certified bit comes from s1.
        """
        mu, nu, p_mu = FIXED
        for dist in (0.0, 50.0, 100.0, 150.0, 200.0):
            s0, s1 = nian_chain(dist, mu, nu, p_mu, 1e10)[1:3]
            self.assertEqual(s0, 0.0, msg=f"s0 != 0 at {dist} km")
            self.assertGreater(s1, 0.0, msg=f"s1 <= 0 at {dist} km")
            self.assertGreater(
                model_counts(dist, mu, nu, p_mu, 1e10)[0],
                0.0,
                msg=f"model vacuum count <= 0 at {dist} km",
            )

        dead = bisect(lambda d: nian_chain(d, mu, nu, p_mu, 1e10)[2], 150.0, 300.0)

        self.assertClose(dead, 219.7610, atol=1e-3, msg="s1 root != 219.761 km")
        self.assertGreater(
            nian_chain(220.0, mu, nu, p_mu, 1e10)[1],
            0.0,
            msg="s0 <= 0 at 220 km",
        )

    def test_tau_departure(self):
        """
        Nian Eq. (6) as printed omits Rusca Eq. (A17)'s 1/tau_0 and returns a single-photon
        floor larger by that factor's closed-form cost, at tau_0 = 0.6476.
        """
        mu, nu, p_mu = FIXED
        zero, one = nian_taus(mu, nu, p_mu)

        self.assertClose(zero, 0.6476086, atol=1e-6, msg="tau_0 at the fixed configuration")
        share = (mu * mu - nu * nu) / (mu * mu)
        for dist in (0.0, 50.0, 100.0, 150.0):
            chain = nian_chain(dist, mu, nu, p_mu, 1e10)
            printed = printed_single(dist, mu, nu, p_mu, 1e10)
            want = one * mu / (nu * (mu - nu)) * share * chain[4] * (1.0 / zero - 1.0)
            self.assertGreater(printed, chain[2], msg=f"printed s1 not above shipped at {dist} km")
            self.assertClose(printed - chain[2], want, atol=1.0, msg=f"gap at {dist} km")


class SargLength(Guarded):
    """
    The finite key length in bits, Nian, Nie, Zhang & Lu, Commun. Theor. Phys. 76, 065101
    (2024), Eq. (5), on their table 1 channel; the paper prints figures, so what reproduces is
    the SHAPE of figure 2(a) on a stated grid.
    """

    def test_block_ladder(self):
        """
        The maximum secure distance rises with the block from 165.25 km at 1e6 to 172.02 km at
        1e10, the first two decades buying 5.47 km and the last two 1.31 km, the shape of their
        figure 2(a).
        """
        ladder = [nian_reach(10.0**k) for k in (6, 7, 8, 9, 10)]
        want = (165.2476, 169.1589, 170.7127, 171.5900, 172.0218)
        for got, target, k in zip(ladder, want, (6, 7, 8, 9, 10)):
            self.assertClose(got, target, atol=1e-3, msg=f"reach at a 1e{k} block")

        self.assertMonotone(ladder, rising=True, msg=f"reach: {ladder}")
        self.assertClose(ladder[2] - ladder[0], 5.4651, atol=1e-3, msg="1e6 to 1e8 buys 5.47 km")
        self.assertClose(ladder[4] - ladder[2], 1.3091, atol=1e-3, msg="1e8 to 1e10 buys 1.31 km")

    def test_under_the_ceiling(self):
        """
        Every finite block sits under the infinite-decoy reach of 187.72 km on the single-photon
        term, 191.996 km with the two-photon term Eq. (5) drops, the 1e10 block conceding 15.69
        km.
        """
        one = bisect(lambda d: asym_best(d, two=False), 1.0, 260.0)
        both = bisect(lambda d: asym_best(d, two=True), 1.0, 260.0)

        self.assertClose(one, 187.7161, atol=1e-3, msg="asymptotic one-photon reach")
        self.assertClose(both, 191.9958, atol=1e-3, msg="asymptotic one- and two-photon reach")
        for k in (6, 8, 10):
            self.assertLess(nian_reach(10.0**k), one, msg=f"1e{k} reach above the asymptote")

        self.assertClose(one - nian_reach(1e10), 15.6943, atol=2e-3, msg="concession != 15.6943 km")

    def test_rate_converges(self):
        """
        At 50 km bits per conclusive detection rise from 0.1897 at a 1e6 block to 0.3487 at
        1e12, short of the infinite-decoy single-photon 0.4816: the residue is the one-decoy
        inversion's price, not finite size.
        """
        mu, nu, p_mu = FIXED
        want = (0.189726, 0.297447, 0.332523, 0.343711, 0.347258, 0.348735)
        rates = [nian_chain(50.0, mu, nu, p_mu, 10.0**k)[0] / 10.0**k for k in (6, 7, 8, 9, 10, 12)]
        for got, target, k in zip(rates, want, (6, 7, 8, 9, 10, 12)):
            self.assertClose(got, target, atol=1e-6, msg=f"bits per detection at a 1e{k} block")

        self.assertMonotone(rates, rising=True, msg=f"rates: {rates}")
        ceiling = nian_asym(50.0, mu, two=False)

        self.assertClose(ceiling[0] / ceiling[1], 0.481570, atol=1e-6, msg="infinite-decoy ceiling")
        self.assertLess(rates[-1], ceiling[0] / ceiling[1], msg="1e12 block at or above the ceiling")

    def test_two_photon_forgone(self):
        """
        Eq. (5) and `sarg_length` carry no two-photon term, forgoing 13.76% of the asymptotic
        rate at 50 km for want of a counted lower bound on s2.
        """
        mu = FIXED[0]
        one = nian_asym(50.0, mu, two=False)[0]
        both = nian_asym(50.0, mu, two=True)[0]

        self.assertClose((both - one) / both, 0.137612, atol=1e-6, msg="the two-photon share")
        self.assertGreater(both, one, msg="two-photon term not positive")

    def test_domain_binds(self):
        """
        v1/s1 reaches the 1/3 cutoff, certified 20.67% at 200 km against the model's 6.20%, only
        at 206.06 km, 35.7 km past this configuration's last key at 170.41 km.
        """
        mu, nu, p_mu = FIXED
        s1, v1 = nian_chain(200.0, mu, nu, p_mu, 1e10)[2:4]
        one, wrong = model_counts(200.0, mu, nu, p_mu, 1e10)[1:3]

        self.assertClose(v1 / s1, 0.206724, atol=1e-5, msg="the certified single-photon QBER")
        self.assertClose(wrong / one, 0.061947, atol=1e-5, msg="model single-photon QBER")
        cross = bisect(
            lambda d: (1.0 / 3.0) - nian_chain(d, mu, nu, p_mu, 1e10)[3] / nian_chain(d, mu, nu, p_mu, 1e10)[2],
            1.0,
            215.0,
        )
        dead = bisect(lambda d: nian_chain(d, mu, nu, p_mu, 1e10)[0], 100.0, 300.0)

        self.assertClose(cross, 206.0628, atol=1e-3, msg="1/3 crossing != 206.0628 km")
        self.assertClose(dead, 170.4058, atol=1e-3, msg="length root != 170.4058 km")
        self.assertEqual(
            nian_chain(207.0, mu, nu, p_mu, 1e10)[0],
            0.0,
            msg="length != 0 at 207 km",
        )

    def test_departure_costs(self):
        """
        Nian Eq. (6) as printed reports 7.3% to 7.9% more key at zero distance and 5.0 to 6.0 km
        more reach than Rusca Eq. (A17), after intensity optimisation.
        """
        for k, ratio, extra in ((6, 1.07901, 4.9916), (8, 1.07323, 5.6946), (10, 1.07517, 6.0052)):
            block = 10.0**k
            shipped = nian_best(0.0, block)[0]
            printed = nian_best(0.0, block, form=printed_bits)[0]
            self.assertGreater(printed, shipped, msg=f"printed rate not above shipped at 1e{k}")
            self.assertClose(printed / shipped, ratio, atol=1e-4, msg=f"rate ratio at 1e{k}")
            self.assertClose(
                nian_reach(block, form=printed_bits) - nian_reach(block),
                extra,
                atol=2e-3,
                msg=f"extra reach at 1e{k}",
            )

    def test_route_named(self):
        """
        `sarg_finite` refuses a block size, routing to `sarg_length`, naming both papers it
        stands on and keeping its retired reason.
        """
        for needle in (
            "sarg_length",
            "Commun. Theor. Phys. 76, 065101 (2024)",
            "Appl. Phys. Lett. 112, 171104 (2018)",
            "no published analysis states a key length",
            "arXiv:2603.22448",
            "Sci. Rep. 6, 29482 (2016), Eqs. (30) and (31)",
        ):
            self.assertFails(
                NotImplementedError,
                needle,
                _core.sarg_finite,
                1e10,
                msg=f"refusal missing {needle}",
            )


class SargGuards(Guarded):
    """
    Argument domains: every rate and bound refuses its own bad slot with the parameter named.
    """

    def test_rate_slots(self):
        """
        `sarg_rate` refuses gains outside [0, 1], error rates outside [0, 1/2], a sub-Shannon
        f_ec, and q1 + q2 above q_mu, naming the slot.
        """
        ok = (4e-3, 0.01, 1.6e-3, 0.01, 4e-4, 0.01, 1.22)
        cases = (
            (0, "q_mu", (-1e-3, 1.5)),
            (1, "e_mu", (-0.01, 0.6)),
            (2, "q1", (-1e-3, 1.5)),
            (3, "e1", (-0.01, 0.6)),
            (4, "q2", (-1e-3, 1.5)),
            (5, "e2", (-0.01, 0.6)),
            (6, "f_ec", (0.0, 0.95)),
        )

        self.assertGreater(_core.sarg_rate(*ok), 0.0, msg="baseline must be valid")
        self.assertSlots(_core.sarg_rate, ok, cases, msg="sarg_rate")
        self.assertBad(
            "q1 + q2 must not exceed q_mu",
            _core.sarg_rate,
            (1e-3, 0.01, 8e-4, 0.01, 4e-4, 0.01, 1.22),
            msg="q1 + q2 > q_mu unguarded",
        )

    def test_photon_number(self):
        """
        `sarg_limit` takes 1 or 2 photons and refuses 0, 3 and 7.
        """
        for bad in (0, 3, 7):
            self.assertBad(
                "photons must be 1 or 2",
                _core.sarg_limit,
                (bad,),
                msg=f"sarg_limit took {bad}",
            )

    def test_yield_slots(self):
        """
        `sarg_yield` refuses eta or dark outside [0, 1] and e_det above 1/2, and `sarg_ceiling`
        a non-positive mu and a t or eta outside (0, 1].
        """
        self.assertSlots(
            _core.sarg_yield,
            (0.1, 1.7e-6, 0.033, 1),
            (
                (0, "eta", (-0.1, 1.5)),
                (1, "dark", (-1e-6, 1.5)),
                (2, "e_det", (-0.01, 0.6)),
            ),
            msg="sarg_yield",
        )

        ok = (0.6, 0.1, 0.1)

        self.assertGreater(_core.sarg_ceiling(*ok), 0.0, msg="baseline must be valid")
        self.assertSlots(
            _core.sarg_ceiling,
            ok,
            (
                (0, "mu", (0.0, -0.1)),
                (1, "t", (0.0, 1.5)),
                (2, "eta", (0.0, 1.5)),
            ),
            msg="sarg_ceiling",
        )

    def test_counted_slots(self):
        """
        The one-decoy estimators refuse an inverted intensity pair, intensity probabilities
        summing above 1, errors above detections, and eps outside (0, 1).
        """
        counts, errors = (8e5, 2e5), (8e3, 2e3)
        ok = (0.46, 0.23, (0.9, 0.1), counts, errors, 5e-11)

        self.assertGreater(_core.sarg_counts(*ok)[1], 0.0, msg="baseline must be valid")
        for bad, needle in (((0.23, 0.46), "nu < mu"), ((0.23, 0.23), "nu < mu")):
            self.assertBad(
                needle,
                _core.sarg_counts,
                (bad[0], bad[1], (0.9, 0.1), counts, errors, 5e-11),
                msg=f"mu = {bad[0]}, nu = {bad[1]} unguarded",
            )

        self.assertBad(
            "p_mu + p_nu must not exceed 1",
            _core.sarg_counts,
            (0.46, 0.23, (0.9, 0.2), counts, errors, 5e-11),
            msg="p_mu + p_nu > 1 unguarded",
        )
        self.assertBad(
            "must not exceed the detections",
            _core.sarg_counts,
            (0.46, 0.23, (0.9, 0.1), counts, (9e5, 2e3), 5e-11),
            msg="errors > detections unguarded",
        )
        self.assertSlots(
            _core.sarg_errors,
            (0.46, 0.23, (0.9, 0.1), errors, 5e-11),
            (
                (0, "mu", (0.0, -0.1)),
                (1, "nu", (0.0, -0.1)),
                (4, "eps", (0.0, 1.0)),
            ),
            msg="sarg_errors",
        )

    def test_length_slots(self):
        """
        `sarg_length` refuses s0 + s1 above n_key and every epsilon outside (0, 1); v1 is an
        uncapped count, a ratio above 1/2 buying zero.
        """
        ok = (1e4, 6e5, 1.2e4, 1e6, 0.02, 1.16, 1e-9, 1e-15)

        self.assertGreater(_core.sarg_length(*ok), 0.0, msg="baseline must be valid")
        self.assertBad(
            "s0 + s1 must not exceed n_key",
            _core.sarg_length,
            (1e4, 6e5, 1.2e4, 5e5, 0.02, 1.16, 1e-9, 1e-15),
            msg="s0 + s1 > n_key unguarded",
        )
        self.assertSlots(
            _core.sarg_length,
            ok,
            (
                (2, "v1", (-1.0,)),
                (4, "e_key", (-0.01, 0.6)),
                (5, "f_ec", (0.0, 0.95)),
                (6, "eps_sec", (0.0, 1.0)),
                (7, "eps_cor", (0.0, 1.0)),
            ),
            msg="sarg_length",
        )
        self.assertEqual(
            _core.sarg_length(1e4, 6e5, 9e5, 1e6, 0.02, 1.16, 1e-9, 1e-15),
            0.0,
            msg="v1 above s1 did not buy zero",
        )

    def test_error_domains(self):
        """
        `sarg_sift`, `sarg_cost` and `sarg_phase` refuse an error rate outside [0, 1/2], past
        which the entropy factors turn upward and a worse channel reads as more key.
        """
        for fn, name in (
            (_core.sarg_sift, "e_det"),
            (_core.sarg_cost, "e1"),
            (_core.sarg_phase, "e2"),
        ):
            for bad in (-0.01, 0.51, 1.0):
                self.assertBad(
                    f"{name} must be in [0, 1/2]",
                    fn,
                    (bad,),
                    msg=f"{name} = {bad} unguarded",
                )


if __name__ == "__main__":
    rc = Exam(
        "SargSifting",
        "Tier A: SARG04's conclusive-exclusion sifting, a quarter where BB84 keeps a half",
        "sarg_sifting.md",
    ).run(load(SargSifting))
    rc |= Exam(
        "SargBounds",
        "Correlated bit and phase errors, and the tolerable QBERs they fix",
        "sarg_bounds.md",
    ).run(load(SargBounds))
    rc |= Exam(
        "SargDecoy",
        "Infinite-decoy SARG04 against BB84 on the GYS parameters",
        "sarg_decoy.md",
    ).run(load(SargDecoy))
    rc |= Exam(
        "SargSplitting",
        "Photon-number splitting: t^1.5 against BB84's t^2, and where they cross",
        "sarg_splitting.md",
    ).run(load(SargSplitting))
    rc |= Exam(
        "SargCounted",
        "One-decoy counts, Rusca et al., Appl. Phys. Lett. 112, 171104 (2018)",
        "sarg_counted.md",
    ).run(load(SargCounted))
    rc |= Exam(
        "SargLength",
        "The finite key length, Nian et al., Commun. Theor. Phys. 76, 065101 (2024)",
        "sarg_length.md",
    ).run(load(SargLength))
    rc |= Exam(
        "SargGuards",
        "Argument domains of the SARG04 engine",
        "sarg_guards.md",
    ).run(load(SargGuards))
    sys.exit(rc)
