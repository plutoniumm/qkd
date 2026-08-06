import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import ALPHA, ETA_BOB, E_DET, E_VAC, F_REC, Y_DARK
from kit.checks import Guarded
from kit.forms import bisect, h2
from qkd import _core

# The one-way collective-attack thresholds of Scarani, Bechmann-Pasquinucci, Cerf, Dusek,
# Lutkenhaus & Peev, Rev. Mod. Phys. 81, 1301 (2009), Appendix A, at the Shannon limit
# f_ec = 1, computed here to f64: 12.6193% after Eq. (A6) and 11.0028% after Eq. (A9). The
# review prints "Q ~ 12.61%" and "Q ~ 11%"; 12.6193% truncates to 12.61% and rounds to
# 12.62%, so the printed figure is the truncation of the computed one and not a four-digit
# agreement with it. Renner, Gisin & Kraus, Phys. Rev. A 72, 012332 (2005), Sec. V.1 quotes
# the same pair as 0.126 and 0.110.
SIX_ZERO = 0.12619308327682110
BB84_ZERO = 0.11002786443835955

# Lo, Quant. Inf. Comput. 1, 81 (2001) proves 12.6% by one-way CSS hashing and lifts it to
# 12.7% with a degenerate DiVincenzo-Shor-Smolin code. Only the first is implemented: the
# degenerate-code gain is not a Devetak-Winter quantity and no engine here produces it.
LO_HASHING = 0.126

# Kato & Tamaki, "Security of six-state quantum key distribution protocol with threshold
# detectors", Sci. Rep. 6, 30044 (2016), arXiv:1008.4663, print the qubit-based threshold to
# five figures where the review prints two. 12.619% is both the rounding and the truncation
# of the computed 12.61931%, so this is a five-figure agreement with print; SIX_ZERO's
# truncation note above is about the review alone. Their own threshold-detector 12.611% is a
# DIFFERENT quantity: the same protocol on real detectors, asymptotic, single-photon, and
# 8e-5 below the qubit one.
KATO_QUBIT = 0.12619
KATO_CLICK = 0.12611

# The uniform three-basis sifting factor, Lo's "two-thirds of the times they disagree".
UNIFORM = 1.0 / 3.0

# One argument list for every literature exam: a lost needle then fails on the message.
DECOY = (1e12, UNIFORM, 0.02, 0.02, 1.0, 1e-9, 1e-12, "decoy")

# The GYS hardware of test/discrete.py unchanged; only privacy amplification differs.
MU_GRID = 500
MU_STEP = 2e-3

# Every distance below is a zero of the grid-optimised rate: refining MU_STEP to 1e-5 moves
# the reaches by 1.0e-6 and 7.7e-6 km and the crossover by 1.5e-4 km, all inside DIST_BAND.
DIST_BAND = 1e-3

# The two maximum secure distances on GYS, each floored to 1e-4 km so the pin itself carries
# key. REACH_GAP moves by under 7e-6 km with MU_STEP, so it is pinned ten times tighter.
SIX_REACH = 146.1986
BB84_REACH = 142.2085
REACH_GAP = 3.9901

# A deterministic lattice over the three error rates, filtered to the Bell-compatible region.
LATTICE = [(x / 14.0, y / 14.0, z / 14.0) for x in range(8) for y in range(8) for z in range(8)]
SIMPLEX = [t for t in LATTICE if t[0] + t[1] >= t[2] and t[1] + t[2] >= t[0] and t[2] + t[0] >= t[1]]

# One setting shared by every shape measurement: the widths sixstate_length spends.
EPS = 1e-9
EPS_EC = 1e-12

# (n, m_test, e_key, e_test), ending on the e_key = 2*e_test edge where the Bell region cuts
# the box rather than containing it.
BOXES = (
    (1e6, 1e5, 0.03, 0.03),
    (1e12, 1e11, 0.02, 0.02),
    (1e8, 1e4, 0.05, 0.04),
    (1e9, 1e9, 0.0, 0.02),
    (1e7, 1e6, 0.08, 0.04),
    (1e10, 1e5, 0.11, 0.09),
)


def hxe(e_key, e_test):
    """
    The residual entropy sixstate_bound returns, mirrored so the four-corner scan is
    arbitrated against a grid and not against itself.
    """
    ratio = (1.0 - e_test - 0.5 * e_key) / (1.0 - e_key)

    return (1.0 - e_key) * (1.0 - h2(ratio))


def allowed(rates):
    """
    Whether a triple of error rates lies in the Bell-compatible region.
    """
    e_x, e_y, e_z = rates

    return e_x + e_y >= e_z and e_y + e_z >= e_x and e_z + e_x >= e_y


def wang(e_key, e_sum):
    """
    The bracket of Wang, Yin, Wang, Wang, Lu, Chen, He, Guo & Han, "Tight
    finite-key analysis for mode-pairing quantum key distribution",
    arXiv:2302.13481, Eq. (2), in the SUM of the two monitor rates as they write it.
    """
    ratio = (1.0 - 0.5 * (e_key + e_sum)) / (1.0 - e_key)

    return (1.0 - e_key) * (1.0 - h2(ratio))


def regroup(e_x, e_y, e_z):
    """
    Eve's information regrouped so the first entropy reads only e_x + e_y and the
    second only e_y - e_x.
    """
    weight = 1.0 - 0.5 * (e_x + e_y + e_z)
    swing = 0.5 * (e_y + e_z - e_x)

    return (1.0 - e_z) * h2(weight / (1.0 - e_z)) + e_z * h2(swing / e_z)


def splits(e_sum, e_z, steps=120):
    """
    Every Bell-compatible way of splitting a fixed monitor sum into its two rates.
    """
    out = []
    for i in range(steps + 1):
        e_x = e_sum * i / steps
        e_y = e_sum - e_x
        if max(e_x, e_y) <= 0.5 and allowed((e_x, e_y, e_z)):
            out.append((e_x, e_y))

    return out


def sweep(e_key, e_test, dk, dt, steps=200):
    """
    The least residual entropy on a dense grid over the compatible box.
    """
    lo_k, hi_k = max(e_key - dk, 0.0), min(e_key + dk, 0.5)
    lo_t, hi_t = max(e_test - dt, 0.0), min(e_test + dt, 0.5)
    floor = None
    for i in range(steps + 1):
        a = lo_k + (hi_k - lo_k) * i / steps
        for j in range(steps + 1):
            b = lo_t + (hi_t - lo_t) * j / steps
            if a > 2.0 * b:
                continue

            got = hxe(a, b)
            if floor is None or got < floor:
                floor = got

    return floor


def eve_a5(e_x, e_y, e_z):
    """
    Eve's information as Rev. Mod. Phys. 81, 1301 (2009), Eq. (A5) prints it, division
    by e_z included.
    """
    swing = 0.5 * (1.0 + (e_x - e_y) / e_z)
    rest = (1.0 - 0.5 * (e_x + e_y + e_z)) / (1.0 - e_z)

    return e_z * h2(swing) + (1.0 - e_z) * h2(rest)


def six_frac(qber, f_ec=1.0):
    """
    Six-state secret fraction at a depolarising QBER, all three bases equal.
    """

    return _core.sixstate_secret(qber, qber, qber, f_ec)


def bb84_frac(qber, f_ec=1.0):
    """
    BB84's secret fraction on the same footing, 1 - 2*h2(Q), through bb84_rate at unit
    gains.
    """

    return _core.bb84_rate(1.0, 1.0, qber, 1.0, qber, f_ec)


def gys(dist, mu):
    """
    Infinite-decoy GYS observables at ``dist`` km: (gain, qber, q1, e1).
    """
    eta = _core.decoy_eta(ALPHA, dist, ETA_BOB)
    y1 = Y_DARK + eta - Y_DARK * eta
    gain, qber = _core.decoy_gain(mu, eta, Y_DARK, E_DET)

    return gain, qber, y1 * mu * math.exp(-mu), (E_VAC * Y_DARK + E_DET * eta) / y1


def six_gys(dist, mu, sift=UNIFORM):
    gain, qber, q1, e1 = gys(dist, mu)

    return _core.sixstate_rate(sift, gain, qber, q1, e1, F_REC)


def bb84_gys(dist, mu, sift=0.5):
    gain, qber, q1, e1 = gys(dist, mu)

    return _core.bb84_rate(sift, gain, qber, q1, e1, F_REC)


def best(fn, dist, sift=None):
    """
    The rate at ``dist`` km with the signal intensity optimised over a fixed grid.
    """
    grid = range(1, MU_GRID + 1)
    if sift is None:
        return max(fn(dist, k * MU_STEP) for k in grid)

    return max(fn(dist, k * MU_STEP, sift) for k in grid)


def reach(fn, sift=None):
    """
    The distance in km at which the intensity-optimised rate reaches zero.
    """

    return bisect(lambda d: best(fn, d, sift), 0.0, 400.0)


class Thresholds(Question):
    """
    The zero crossings and the gap between them.
    """

    def test_six_crossing(self):
        """
        The six-state fraction crosses zero at 12.6193% and BB84's at 11.003%, a gap of
        1.6 points and a ratio of 1.146919.
        """
        zero = bisect(six_frac, 0.05, 0.2)
        flat = bisect(bb84_frac, 0.05, 0.2)

        self.assertClose(zero, SIX_ZERO, atol=1e-15, msg=f"six-state threshold {zero}")
        self.assertClose(zero, LO_HASHING, atol=2e-4, msg=f"{zero} is not Lo's 12.6%")
        self.assertClose(flat, BB84_ZERO, atol=1e-15, msg=f"BB84 threshold {flat}")
        self.assertGreater(SIX_ZERO - flat, 0.016, msg=f"gap {SIX_ZERO - flat}")
        self.assertLess(SIX_ZERO - flat, 0.017, msg=f"gap {SIX_ZERO - flat}")
        self.assertClose(SIX_ZERO / BB84_ZERO, 1.146919, atol=1e-6, msg=f"ratio {SIX_ZERO / BB84_ZERO}")

    def test_gap_positive(self):
        """
        At every equal QBER below its own threshold the six-state fraction is the larger.
        """
        for k in range(1, 126):
            qber = k * 1e-3
            six = six_frac(qber)

            self.assertGreater(six, bb84_frac(qber), msg=f"BB84 wins at Q = {qber}")

    def test_noiseless_limit(self):
        """
        Both families give one full bit at Q = 0, where Eve's information is zero, and the
        six-state fraction falls monotonically in Q.
        """

        self.assertClose(six_frac(0.0), 1.0, atol=0.0, msg=f"six-state at Q=0 is {six_frac(0.0)}")
        self.assertClose(bb84_frac(0.0), 1.0, atol=0.0, msg=f"BB84 at Q=0 is {bb84_frac(0.0)}")
        self.assertClose(_core.sixstate_holevo(0.0), 0.0, atol=0.0, msg="Eve knows something at Q=0")

        seq = [six_frac(k * 5e-3) for k in range(0, 25)]

        self.assertMonotone(seq, rising=False, msg="fraction is not monotone in Q")

    def test_kato_printed(self):
        """
        The computed crossing agrees to five figures with Kato and Tamaki's qubit
        threshold and sits 8e-5 above their threshold-detector figure.
        """

        self.assertClose(SIX_ZERO, KATO_QUBIT, atol=5e-6, msg=f"{SIX_ZERO} is not Kato's 12.619%")
        self.assertGreater(SIX_ZERO, KATO_CLICK, msg=f"qubit {SIX_ZERO} below click {KATO_CLICK}")
        self.assertClose(KATO_QUBIT - KATO_CLICK, 8e-5, atol=1e-6, msg=f"printed gap {KATO_QUBIT - KATO_CLICK}")

    def test_leak_moves(self):
        """
        Charging F_REC leakage rather than the Shannon limit moves the crossing to
        10.7322%.
        """
        real = bisect(lambda x: six_frac(x, F_REC), 0.02, 0.2)

        self.assertLess(real, SIX_ZERO, msg=f"crossing {real} not below {SIX_ZERO}")
        self.assertClose(real, 0.107322, atol=1e-6, msg=f"f_ec = 1.22 crossing {real}")


class Forms(Question):
    """
    The three printed forms of Eve's information, checked against each other.
    """

    def test_a4_matches(self):
        """
        Eq. (A4) and Eq. (A5) agree to 1e-14 where both are defined, Eq. (A5) dividing by
        e_z where a noiseless key basis is a number here.
        """
        worst = 0.0
        for e_x in [k * 0.02 for k in range(1, 26)]:
            for e_y in [k * 0.05 for k in range(1, 11)]:
                for e_z in [k * 0.05 for k in range(1, 11)]:
                    if not allowed((e_x, e_y, e_z)):
                        continue

                    got = _core.sixstate_eve(e_x, e_y, e_z)
                    worst = max(worst, abs(got - eve_a5(e_x, e_y, e_z)))

        self.assertLess(worst, 1e-14, msg=f"A4 and A5 differ by {worst:.2e}")

        got = _core.sixstate_eve(0.1, 0.1, 0.0)

        self.assertClose(got, h2(0.1), atol=1e-15, msg="e_z = 0 is not h2(e_x)")

        with self.assertRaises(ZeroDivisionError):
            eve_a5(0.1, 0.1, 0.0)

    def test_a6_matches(self):
        """
        Eq. (A6) reproduces the three-basis form at equal error rates to 1e-15, on Renner,
        Gisin and Kraus Eq. (10) weights 1 - 3Q/2 and Q/2.
        """
        for k in range(0, 51):
            qber = k * 0.01
            got = _core.sixstate_holevo(qber)

            self.assertClose(
                got,
                _core.sixstate_eve(qber, qber, qber),
                atol=1e-15,
                msg=f"A6 != A4 at Q = {qber}",
            )

            l1, l2, l3, l4 = _core.sixstate_bell(qber, qber, qber)

            self.assertClose(l1, 1.0 - 1.5 * qber, atol=1e-15, msg="l1 != 1 - 3Q/2")
            self.assertClose(max(l2, l3, l4), 0.5 * qber, atol=1e-15, msg="l != Q/2")
            self.assertClose(min(l2, l3, l4), 0.5 * qber, atol=1e-15, msg="l != Q/2")

    def test_bell_inverts(self):
        """
        The weights reproduce through Eqs. (A2) and (A3) the three error rates they were
        read from, stay non-negative and sum to one.
        """
        for e_x, e_y, e_z in [(0.05, 0.05, 0.05), (0.1, 0.04, 0.07), (0.3, 0.2, 0.5)]:
            l1, l2, l3, l4 = _core.sixstate_bell(e_x, e_y, e_z)

            self.assertClose(l1 + l2 + l3 + l4, 1.0, atol=1e-15, msg="weights lost norm")
            self.assertClose(l3 + l4, e_z, atol=1e-15, msg="A2: e_z != l3 + l4")
            self.assertClose(l2 + l4, e_x, atol=1e-15, msg="A3: e_x != l2 + l4")
            self.assertClose(l2 + l3, e_y, atol=1e-15, msg="A3: e_y != l2 + l3")
            self.assertGreaterEqual(min(l1, l2, l3, l4), 0.0, msg="a weight went negative")

    def test_eve_below(self):
        """
        Eve learns strictly less than BB84's h2(Q) between the ends, both saturating at 1
        at Q = 1/2.
        """
        for k in range(1, 50):
            qber = k * 0.01

            self.assertLess(
                _core.sixstate_holevo(qber),
                h2(qber),
                msg=f"A6 is not below A9 at Q = {qber}",
            )

        self.assertClose(_core.sixstate_holevo(0.5), 1.0, atol=1e-15, msg="A6 != 1 at 1/2")
        self.assertClose(h2(0.5), 1.0, atol=0.0, msg="A9 != 1 at 1/2")

    def test_key_basis(self):
        """
        Permuting the key basis with the first moves `sixstate_eve` by exactly the
        difference of the two subtracted h2(e_z).
        """
        keyed = _core.sixstate_eve(0.10, 0.04, 0.07)
        other = _core.sixstate_eve(0.07, 0.04, 0.10)

        self.assertGreater(abs(keyed - other), 1e-3, msg=f"permutation moved {abs(keyed - other)}")
        self.assertClose(
            keyed - other,
            h2(0.10) - h2(0.07),
            atol=1e-15,
            msg=f"permutation moved {keyed - other}, not h2(0.10) - h2(0.07)",
        )

    def test_rate_reduces(self):
        """
        `sixstate_rate` collapses onto the single-qubit secret fraction at unit gains, and
        on error-free single photons onto the plain GLLP difference.
        """
        for k in range(0, 13):
            qber = k * 0.01
            got = _core.sixstate_rate(1.0, 1.0, qber, 1.0, qber, 1.0)

            self.assertClose(got, six_frac(qber), atol=1e-15, msg="rate != fraction")

        got = _core.sixstate_rate(0.5, 0.2, 0.03, 0.1, 0.0, 1.16)
        want = 0.5 * (0.1 - 0.2 * 1.16 * h2(0.03))

        self.assertClose(got, want, atol=1e-15, msg="GLLP shape changed")


class Decoy(Question):
    """
    `decoy.rs` is basis-blind, so the six-state engine adds no decoy arithmetic.
    """

    def test_gys_point(self):
        """
        GYS at 25 km gives 5.77121e-4 through the Y1 and e1 bounds BB84 consumes, and
        `decoy_bounds`' vacuum-plus-weak inversion feeds `sixstate_rate` at a cost in rate.
        """
        gain, qber, q1, e1 = gys(25.0, 0.48)
        got = _core.sixstate_rate(UNIFORM, gain, qber, q1, e1, F_REC)

        self.assertGreater(got, 0.0, msg=f"GYS 25 km rate {got}")
        self.assertFinite(got, msg="GYS 25 km rate is not finite")
        self.assertClose(got, 5.77121e-4, atol=1e-9, msg=f"GYS 25 km rate {got}")

        eta = _core.decoy_eta(ALPHA, 25.0, ETA_BOB)
        obs = [_core.decoy_gain(x, eta, Y_DARK, E_DET) for x in (0.48, 0.1)]
        vac = _core.decoy_gain(1e-12, eta, Y_DARK, E_DET)
        y1, e1, q1 = _core.decoy_bounds(0.48, 0.1, 0.0, obs[0][0], obs[0][1], obs[1][0], obs[1][1], vac[0], E_VAC)
        held = _core.sixstate_rate(UNIFORM, obs[0][0], obs[0][1], q1, e1, F_REC)

        self.assertGreater(y1, 0.0, msg=f"y1 = {y1}")
        self.assertGreater(held, 0.0, msg=f"rate through the decoy bounds {held}")
        self.assertLess(held, got, msg=f"bounded {held} not below ideal {got}")

    def test_gain_ordering(self):
        """
        At an equal sifting factor the six-state rate beats BB84's on GYS by 1.14856x at 0
        km and 2.9694x at 140 km, the ratio growing with loss.
        """
        near = best(six_gys, 0.0, 0.5) / best(bb84_gys, 0.0)
        far = best(six_gys, 140.0, 0.5) / best(bb84_gys, 140.0)

        self.assertClose(near, 1.14856, atol=1e-5, msg=f"0 km ratio {near}")
        self.assertClose(far, 2.9694, atol=1e-4, msg=f"140 km ratio {far}")
        self.assertMonotone(
            [best(six_gys, d * 20.0, 0.5) / best(bb84_gys, d * 20.0) for d in range(0, 8)],
            msg="the ratio does not grow with loss",
        )

    def test_reach_longer(self):
        """
        On GYS the six-state reach is 146.1986 km against BB84's 142.2085 km, a gap of
        3.9901 km, each pin carrying key and reading exactly zero a metre out.
        """
        six = reach(six_gys)
        bb84 = reach(bb84_gys)

        self.assertClose(six, SIX_REACH, atol=DIST_BAND, msg=f"six-state reach {six}")
        self.assertClose(bb84, BB84_REACH, atol=DIST_BAND, msg=f"BB84 reach {bb84}")
        self.assertClose(six - bb84, REACH_GAP, atol=1e-4, msg=f"reach gap {six - bb84}")
        self.assertGreater(six - bb84, 3.9, msg=f"reach gap {six - bb84}")

        for fn, km in ((six_gys, SIX_REACH), (bb84_gys, BB84_REACH)):
            self.assertGreater(best(fn, km), 0.0, msg=f"no key at {km} km")
            self.assertClose(
                best(fn, km + DIST_BAND),
                0.0,
                atol=0.0,
                msg=f"key survives a metre past {km} km",
            )

        self.assertClose(
            reach(six_gys, 0.5),
            reach(six_gys, UNIFORM),
            atol=1e-9,
            msg="q_sift moved the reach",
        )

    def test_sift_cost(self):
        """
        Uniform three-basis sifting costs more than the tighter bound buys until 132.578
        km.
        """
        cross = bisect(lambda d: best(bb84_gys, d) - best(six_gys, d), 100.0, 145.0)

        self.assertClose(cross, 132.578, atol=DIST_BAND, msg=f"crossover at {cross}")
        self.assertLess(
            best(six_gys, 50.0),
            best(bb84_gys, 50.0),
            msg=f"six-state {best(six_gys, 50.0)} not below BB84 at 50 km",
        )
        self.assertGreater(
            best(six_gys, 140.0),
            best(bb84_gys, 140.0),
            msg=f"six-state {best(six_gys, 140.0)} not above BB84 at 140 km",
        )


class Sifting(Question):
    """
    Three bases sift harder than two, and biasing the choice buys it back.
    """

    def test_uniform_third(self):
        """
        Equally likely bases match one time in three and one ninth in a nominated basis,
        and under a bias the key share is Lo's (1 - 2*epsilon)^2.
        """
        matched, key = _core.sixstate_sift(UNIFORM)

        self.assertClose(matched, UNIFORM, atol=1e-15, msg=f"matched share {matched}")
        self.assertClose(key, 1.0 / 9.0, atol=1e-15, msg=f"key share {key}")

        for bias in (0.5, 0.7, 0.9, 0.99):
            matched, key = _core.sixstate_sift(bias)
            eps = 0.5 * (1.0 - bias)

            self.assertClose(key, (1.0 - 2.0 * eps) ** 2, atol=1e-15, msg=f"key share {key} at {bias}")
            self.assertGreaterEqual(matched, key, msg=f"key exceeds matched at {bias}")

        top = _core.sixstate_sift(1.0 - 1e-9)

        self.assertClose(top[1], 1.0, atol=1e-8, msg=f"key share at bias 1-1e-9 is {top[1]}")

    def test_both_rise(self):
        """
        Both the matched share and the key share grow with the bias.
        """
        seq = [_core.sixstate_sift(UNIFORM + k * 0.05) for k in range(0, 13)]

        self.assertMonotone([s[0] for s in seq], msg="matched share is not monotone")
        self.assertMonotone([s[1] for s in seq], msg="key share is not monotone")


class Budget(Question):
    """
    The finite-key epsilon budget, read off the engine that spends it.
    """

    def test_slot_recomposes(self):
        """
        Three slots plus the nested error-correction term recompose the declared
        deviation, and the per-estimate share is half a slot by the union bound.
        """
        for eps, eps_ec in ((1e-3, 1e-4), (1e-5, 1e-10), (1e-9, 1e-12)):
            slot, _ = _core.sixstate_eps(eps, eps_ec)

            self.assertClose(3.0 * slot + eps_ec, eps, msg=f"eps {eps} with eps_ec {eps_ec}")

        slot, share = _core.sixstate_eps(1e-5, 1e-10)

        self.assertClose(2.0 * share, slot, msg=f"share {share} against slot {slot}")
        self.assertGreater(slot, share, msg="the union bound must cost something")

    def test_length_spends(self):
        """
        The slot rebuilds the two subtracted correction terms and the share the two
        Lemma 3 widths, which taken at the slot would report a longer key.
        """
        n, m_test = 1e8, 2e7
        eps, eps_ec = 1e-5, 1e-10
        slot, share = _core.sixstate_eps(eps, eps_ec)
        out = _core.sixstate_length(n, m_test, 0.03, 0.03, 1.0, eps, eps_ec)
        delta = 7.0 * math.sqrt(n * math.log2(2.0 / slot))
        delta += 2.0 * math.log2(1.0 / (2.0 * slot))

        self.assertClose(out[4], delta, msg=f"delta {out[4]} against {delta}")
        self.assertClose(
            out[2],
            0.03 - _core.sixstate_width(n, share),
            msg="the key-basis edge is the width at the share, not at the slot",
        )
        self.assertClose(
            out[3],
            0.03 + _core.sixstate_width(m_test, share),
            msg="the monitor edge is the width at the share, not at the slot",
        )

        wide = _core.sixstate_width(1e8, share)
        narrow = _core.sixstate_width(1e8, slot)

        self.assertGreater(wide, narrow, msg=f"{wide} must exceed {narrow}")


class Objective(Question):
    """
    The shape of the function the finite-key length optimises.
    """

    def test_eve_concave(self):
        """
        Eve's information is concave over the Bell-compatible region and strictly so
        somewhere on it, the shape src/lp.rs takes tangent planes for.
        """
        strict = 0
        for a in SIMPLEX:
            for b in SIMPLEX:
                mid = tuple(0.5 * (p + q) for p, q in zip(a, b))
                if not allowed(mid):
                    continue

                chord = 0.5 * (_core.sixstate_eve(*a) + _core.sixstate_eve(*b))
                arc = _core.sixstate_eve(*mid)

                self.assertGreater(arc - chord, -1e-12, msg=f"concavity fails between {a} and {b}")
                strict += arc - chord > 1e-9

        self.assertGreater(strict, 0, msg="an affine function would satisfy concavity vacuously")

    def test_corner_exact(self):
        """
        The four-corner scan returns the residual entropy's true minimum on the compatible
        box to 1e-12 against a dense grid, at a Bell-compatible corner.
        """
        _slot, share = _core.sixstate_eps(EPS, EPS_EC)
        for n, m_test, e_key, e_test in BOXES:
            out = _core.sixstate_length(n, m_test, e_key, e_test, 1.0, EPS, EPS_EC)
            floor = sweep(e_key, e_test, _core.sixstate_width(n, share), _core.sixstate_width(m_test, share))

            self.assertClose(out[1], floor, atol=1e-12, msg=f"corner {out[1]} against grid {floor} at n {n}")
            self.assertGreater(2.0 * out[3] - out[2], -1e-15, msg=f"the corner used at n {n} is not Bell-compatible")


class Pooled(Question):
    """
    The entropy reads the two monitor bases only through their sum.
    """

    def test_sum_only(self):
        """
        Over 12245 Bell-compatible splits none lowers the residual entropy below its value
        at the equal split, so a bound on the sum alone is a bound.
        """
        seen = 0
        for e_z in (0.0, 0.01, 0.03, 0.05, 0.08, 0.11, 0.125):
            for step in range(1, 100):
                e_sum = step / 100.0
                pair = splits(e_sum, e_z)
                if not pair:
                    continue

                even = _core.sixstate_bound(e_z, 0.5 * e_sum)
                for e_x, e_y in pair:
                    got = 1.0 - _core.sixstate_eve(e_x, e_y, e_z)

                    self.assertGreater(got - even, -1e-12, msg=f"split {e_x},{e_y} at e_z {e_z} beats the pooled bound")
                    seen += 1

        self.assertEqual(seen, 12245, msg=f"walked {seen} splits")

    def test_regroup_isolates(self):
        """
        Eve's information regroups into a term reading only the monitor sum and one only
        their difference, which peaks at the equal split.
        """
        for e_x, e_y, e_z in SIMPLEX:
            if e_z <= 0.0:
                continue

            self.assertClose(
                _core.sixstate_eve(e_x, e_y, e_z),
                regroup(e_x, e_y, e_z),
                atol=1e-12,
                msg=f"the regrouping fails at {e_x},{e_y},{e_z}",
            )

        held = regroup(0.04, 0.06, 0.05)

        self.assertClose(held, regroup(0.06, 0.04, 0.05), atol=1e-12, msg="the difference term is not symmetric")
        self.assertLess(held, regroup(0.05, 0.05, 0.05), msg="the equal split does not maximise Eve's information")

    def test_pooled_averages(self):
        """
        `sixstate_pooled` reads two monitor rates at their average and returns that average
        beside the entropy.
        """
        for e_x, e_y, e_z in SIMPLEX:
            got, avg = _core.sixstate_pooled(e_z, e_x, e_y)

            self.assertClose(avg, 0.5 * (e_x + e_y), atol=0.0, msg=f"the average moved at {e_x},{e_y}")
            self.assertClose(got, hxe(e_z, avg), atol=1e-15, msg=f"the pooled entropy moved at {e_x},{e_y}")
            self.assertClose(got, _core.sixstate_bound(e_z, avg), atol=0.0, msg="pooled and bound disagree")

    def test_picking_overclaims(self):
        """
        Handing `sixstate_bound` the smaller of two unequal monitor rates over-claims by
        0.121743 bit at 0.03 and 0.07, the larger under-claiming.
        """
        got, avg = _core.sixstate_pooled(0.05, 0.03, 0.07)

        self.assertClose(avg, 0.05, atol=1e-15, msg="the average of 0.03 and 0.07 is not 0.05")
        wrong = _core.sixstate_bound(0.05, 0.03)

        self.assertGreater(wrong, got, msg="picking the smaller rate did not over-claim")
        self.assertClose(wrong - got, 0.121743, atol=1e-6, msg="the size of the over-claim moved")
        self.assertLess(_core.sixstate_bound(0.05, 0.07), got, msg="picking the larger rate did not under-claim")

    def test_wang_form(self):
        """
        `sixstate_bound` at half a summed monitor rate is the bracket of arXiv:2302.13481
        Eq. (2) to 1e-14 across the compatible region.
        """
        worst = 0.0
        for e_key in [i / 250.0 for i in range(0, 32)]:
            for step in range(0, 125):
                e_sum = step / 250.0
                if e_sum < e_key or 0.5 * e_sum > 0.5:
                    continue

                worst = max(worst, abs(_core.sixstate_bound(e_key, 0.5 * e_sum) - wang(e_key, e_sum)))

        self.assertLess(worst, 1e-14, msg=f"Eq. (2) and sixstate_bound differ by {worst}")


class Guards(Guarded):
    """
    Out-of-domain six-state inputs raise rather than returning a wrong number.
    """

    def test_bell_domain(self):
        """
        Each error rate is refused outside [0, 1/2] one slot at a time, and a triple no
        Bell-diagonal state can produce is refused rather than clamped.
        """
        bad = [-1e-9, 0.5 + 1e-9, 1.0, float("nan"), float("inf")]

        self.assertSlots(
            _core.sixstate_bell,
            (0.05, 0.05, 0.05),
            [
                (0, "e_x", bad),
                (1, "e_y", bad),
                (2, "e_z", bad),
            ],
            msg="sixstate_bell",
        )
        self.assertBad(
            "Bell-diagonal",
            _core.sixstate_bell,
            (0.5, 0.01, 0.01),
            msg="e_y + e_z < e_x",
        )
        self.assertBad(
            "Bell-diagonal",
            _core.sixstate_eve,
            (0.01, 0.01, 0.5),
            msg="e_x + e_y < e_z",
        )
        self.assertBad(
            "negative",
            _core.sixstate_secret,
            (0.01, 0.5, 0.01, 1.0),
            msg="e_z + e_x < e_y",
        )

    def test_pooled_domain(self):
        """
        `sixstate_pooled` checks the triple it was handed, so a pair no Bell-diagonal state
        carries is refused even where its average would pass.
        """
        bad = [-1e-9, 0.5 + 1e-9, 1.0, float("nan"), float("inf")]

        self.assertSlots(
            _core.sixstate_pooled,
            (0.05, 0.05, 0.05),
            [
                (0, "e_key", bad),
                (1, "e_x", bad),
                (2, "e_y", bad),
            ],
            msg="sixstate_pooled",
        )
        self.assertBad(
            "Bell-diagonal",
            _core.sixstate_pooled,
            (0.05, 0.02, 0.08),
            msg="e_z + e_x < e_y at an average that would pass",
        )
        self.assertClose(
            _core.sixstate_pooled(0.05, 0.05, 0.05)[0],
            _core.sixstate_bound(0.05, 0.05),
            atol=0.0,
            msg="the average that would pass returns the same number",
        )

    def test_holevo_domain(self):
        """
        `sixstate_holevo` refuses anything outside [0, 1/2], including a NaN.
        """
        for bad in (-1e-9, 0.5 + 1e-9, float("nan")):
            self.assertBad("qber", _core.sixstate_holevo, (bad,), msg=f"qber = {bad}")

    def test_rate_domain(self):
        """
        `sixstate_rate` refuses a vanishing sifting factor, a gain or error rate out of
        range, a below-Shannon f_ec, and an inverted decoy bound q1 > q_mu.
        """

        self.assertSlots(
            _core.sixstate_rate,
            (0.5, 0.2, 0.03, 0.1, 0.02, 1.16),
            [
                (0, "q_sift", [0.0, -0.1, 1.5]),
                (1, "q_mu", [-1e-9, 1.0 + 1e-9]),
                (2, "e_mu", [0.5 + 1e-9, float("nan")]),
                (4, "e1", [0.5 + 1e-9, -1e-9]),
                (5, "f_ec", [0.999, 0.0, float("nan")]),
            ],
            msg="sixstate_rate",
        )
        self.assertBad(
            "q1 must not exceed q_mu",
            _core.sixstate_rate,
            (0.5, 0.1, 0.03, 0.2, 0.02, 1.16),
            msg="q1 > q_mu",
        )

    def test_sift_domain(self):
        """
        The basis bias is refused below 1/3 and at 1.
        """

        self.assertBad(
            "bias must be >= 1/3",
            _core.sixstate_sift,
            (0.3,),
            msg="bias below uniform",
        )
        self.assertBad(
            "bias must be < 1",
            _core.sixstate_sift,
            (1.0,),
            msg="no monitor bases",
        )
        self.assertBad("bias", _core.sixstate_sift, (-0.1,), msg="negative bias")

    def test_budget_domain(self):
        """
        `sixstate_eps` refuses either epsilon outside (0, 1) and a correction term at or
        above the deviation it nests inside.
        """
        # Epsilons live in the OPEN (0, 1); 1/2 is not a boundary here.
        bad = [0.0, 1.0, -1e-12, float("nan"), float("inf")]

        self.assertSlots(
            _core.sixstate_eps,
            (1e-5, 1e-10),
            [
                (0, "eps", bad),
                (1, "eps_ec", bad),
            ],
            msg="sixstate_eps",
        )
        self.assertBad(
            "eps_ec must be < eps",
            _core.sixstate_eps,
            (1e-10, 1e-5),
            msg="the correction term outgrows the deviation",
        )

    def test_secret_leak(self):
        """
        `sixstate_secret` refuses an f_ec below the Shannon limit.
        """

        self.assertBad(
            "f_ec",
            _core.sixstate_secret,
            (0.05, 0.05, 0.05, 0.5),
            msg="f_ec below 1",
        )


class Literature(Guarded):
    """
    The finite-key six-state literature is written for an ideal source and the
    weak-coherent half stops at the asymptotic limit.
    """

    def test_decoy_names(self):
        """
        The weak-coherent refusal names the photon-number-resolving finite-key analysis,
        the asymptotic weak-coherent one, the four-state protocol behind a three-basis
        receiver, and the squashing model it prices rather than misses.
        """
        for needle in (
            "STATISTICAL TRANSFER",
            "arXiv:1111.2798",
            "Phys. Rev. Research 6, 043223",
            "Example 2: Biased passive WCP 6-state",
            "can be easily extended to the finite-size regime",
            "arXiv:2502.05382",
            "Alice still prepares FOUR states",
            "SQUASHING MODEL IS NOT THE BLOCKER",
            "Phys. Rev. Lett. 101, 093601",
            "Phys. Rev. A 89, 012325",
            "flips single-click bit values with probability 1/6",
        ):
            self.assertBad(
                needle,
                _core.sixstate_finite,
                DECOY,
                exc=NotImplementedError,
                msg=f"refusal is missing {needle!r}",
            )

    def test_route_sources(self):
        """
        The refusal sources the three routes to a finite key: the phase-error transfer, the
        smooth entropic uncertainty relation over two POVMs, and the compatible-set
        minimisation implemented here.
        """
        for needle in (
            "arXiv:1311.7129",
            "arXiv:0910.0312",
            "the BB84 protocol with a single or entangled photon source",
            "TWO POVMs on one system",
            "arXiv:0708.0709",
            "m samples of sigma according to a POVM with d outcomes",
            "when implemented with single qubits",
            "arXiv:2408.17349",
            "Quantum 9, 1937",
            "since such a joint distribution does not exist",
        ):
            self.assertBad(
                needle,
                _core.sixstate_finite,
                DECOY,
                exc=NotImplementedError,
                msg=f"refusal is missing {needle!r}",
            )

    def test_near_sources(self):
        """
        The refusal names the one published system carrying three bases, decoy intensities
        and a finite block together, and the threshold-detector proof with the two sentences
        that keep it asymptotic and single-photon.
        """
        for needle in (
            "arXiv:2405.16558",
            "Phys. Rev. Applied 22, 064018",
            "three orthogonal bases alpha in {Z_A, X_A, Y_A}",
            "arXiv:1003.1050",
            "coincides with the rate of the six-state protocol for white noise",
            "arXiv:1008.4663",
            "Sci. Rep. 6, 30044",
            "take the asymptotic limit such that the number of the pulses is infinite",
            "an attenuated laser source by GLLP idea",
            "arXiv:0804.3082",
            "arXiv:1310.5059",
            "arXiv:2302.13481",
            "a six-state MP-QKD protocol",
        ):
            self.assertBad(
                needle,
                _core.sixstate_finite,
                DECOY,
                exc=NotImplementedError,
                msg=f"refusal is missing {needle!r}",
            )

    def test_retired_transfer(self):
        """
        The refusal records that it once blamed the statistical transfer, and carries the
        two passages of the cited paper that go the other way.
        """
        for needle in (
            "THE STATISTICAL TRANSFER IS NOT WHAT FAILS EITHER",
            "can indeed exist at the same time",
            "on the marginal probability distribution",
            "the standard serfling argument is directly applicable",
            "for e1 = e2, a case that minimizes it",
            "sixstate_pooled",
            "Relation 3",
            "can be regarded as an entirety",
            "arXiv:2002.03672",
            "arXiv:2403.10294",
        ):
            self.assertBad(
                needle,
                _core.sixstate_finite,
                DECOY,
                exc=NotImplementedError,
                msg=f"refusal is missing {needle!r}",
            )

    def test_missing_register(self):
        """
        The refusal names a single-photon per-round register as the one thing missing, and
        the three-basis analysis that declines the weak-coherent source.
        """
        for needle in (
            "SO THE ONE THING MISSING IS THE TAGGED SINGLE-PHOTON LAYER",
            "one must first fix the number of pulses",
            "This subtlety is missing in",
            "arXiv:2008.03510",
            "Phys. Rev. Research 3, 023019",
            "exactly the six-state protocol",
            "against general attacks",
            "can intuitively be solved by using decoy states",
            "M0 = Y and M1 = Z",
            "a Corollary 3.3.7 of [13]",
            "only thanks to specific symmetries",
            "the upper bound of the sum of the single-photon bit error rate",
            "Phys. Rev. Research 8, 013332",
            "TWO PAPERS WERE NOT READ",
            "Physics Open 7, 100075",
            "ONLY ITS CROSSREF RECORD WAS OPENED",
        ):
            self.assertBad(
                needle,
                _core.sixstate_finite,
                DECOY,
                exc=NotImplementedError,
                msg=f"refusal is missing {needle!r}",
            )

    def test_single_runs(self):
        """
        The block size the decoy source is refused at returns a length for a single-photon
        source, so the refusal is on the source.
        """
        length, rate, n, m_test = _core.sixstate_finite(1e12, 1.0 / 3.0, 0.02, 0.02, 1.0, 1e-9, 1e-12, "single")

        self.assertGreater(length, 0.0, msg=f"length {length}")
        self.assertClose(rate, length / 1e12, msg=f"rate {rate} is not length per signal")
        self.assertClose(n, 1e12 / 9.0, atol=1.0, msg=f"key-basis rounds {n}")
        self.assertClose(m_test, 1e12 / 9.0, atol=1.0, msg=f"monitor rounds {m_test}")


if __name__ == "__main__":
    rc = Exam(
        "SixStateThresholds",
        "The 12.6% zero crossing, against BB84's 11.0% on the same arithmetic",
        "sixstate_thresholds.md",
    ).run(load(Thresholds))
    rc |= Exam(
        "SixStateForms",
        "Eqs. (A4), (A5) and (A6) of Rev. Mod. Phys. 81, 1301 checked against each other",
        "sixstate_forms.md",
    ).run(load(Forms))
    rc |= Exam(
        "SixStateDecoy",
        "Composition with the basis-blind decoy layer on the GYS parameter set",
        "sixstate_decoy.md",
    ).run(load(Decoy))
    rc |= Exam(
        "SixStateSifting",
        "Three-basis sifting, uniform and biased",
        "sixstate_sifting.md",
    ).run(load(Sifting))
    rc |= Exam(
        "SixStateBudget",
        "The finite-key epsilon budget, its three slots and the union bound inside one",
        "sixstate_budget.md",
    ).run(load(Budget))
    rc |= Exam(
        "SixStateObjective",
        "The shape of the function the finite-key length optimises",
        "sixstate_objective.md",
    ).run(load(Objective))
    rc |= Exam(
        "SixStatePooled",
        "The two monitor bases enter only through their sum, which is what the decoy refusal turns on",
        "sixstate_pooled.md",
    ).run(load(Pooled))
    rc |= Exam(
        "SixStateGuards",
        "Out-of-domain six-state inputs raise rather than returning a wrong number",
        "sixstate_guards.md",
    ).run(load(Guards))
    rc |= Exam(
        "SixStateLiterature",
        "The decoy refusal names published work, and each paper it names is pinned",
        "sixstate_literature.md",
    ).run(load(Literature))
    sys.exit(rc)
