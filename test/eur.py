import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.forms import h2
from qkd import _core

# The entropic uncertainty relation and the composable key length it carries. Three statements
# that never stand in for each other:
#
# Maassen & Uffink, Phys. Rev. Lett. 60, 1103 (1988), state H(X) + H(Z) >= q for a memoryless
# adversary. Berta, Christandl, Colbeck, Renes & Renner, Nature Physics 6, 659 (2010),
# arXiv:0909.0950, add a quantum memory and pay for it with H(A|B), which is negative on an
# entangled pair. Tomamichel & Renner, Phys. Rev. Lett. 106, 110506 (2011), arXiv:1009.2015,
# restate it for smooth entropies, the version that reaches coherent attacks with no de Finetti
# reduction and no postselection technique. Tomamichel, Lim, Gisin & Renner, Nature
# Communications 3, 634 (2012), arXiv:1103.4130, turn that into a key length with Serfling and the
# quantum leftover hash lemma.
#
# q enters with a plus sign and must be a lower bound on -log2 c; H_max enters with a minus sign
# and must be an upper bound.
#
# Nothing in `qkd/` reaches this module, and every family it ships is refused by name.

# Every protocol name `eur_family` refuses, with the substring its message must carry.
REFUSED = [
    ("bb84", "perfect single-photon source"),
    ("sixstate", "reads exactly TWO POVMs"),
    ("sarg", "announcement is a PAIR"),
    ("b92", "one measurement"),
    ("cow", "PROTOCOL CHANGE"),
    ("dps", "PROTOCOL CHANGE"),
    ("rrdps", "monitors no disturbance"),
    ("mdi", "COINCIDENCE at an untrusted relay"),
    ("cvmdi", "COINCIDENCE at an untrusted relay"),
    ("pairing", "AFTER the announcement"),
    ("bbm92", "fair-sampling assumption"),
    ("e91", "fair-sampling assumption"),
    ("gaussian", "reverse reconciliation"),
    ("cv", "reverse reconciliation"),
    ("dmcs", "one heterodyne receiver"),
    ("squeezed", "eur_binned"),
]

# Bin areas for the tightness table; the overlap at each is recomputed by `slepian`, not quoted.
AREAS = [0.01, 0.1, 0.5, 1.0, 2.0, 2.5]

# A megabit block, a tenth on parameter estimation, 1% tolerated error, TLGR's own 1.1 x h2 leak.
POINT = (1e6, 1e5, 0.01, 1.1e6 * 0.0807, 1e-10, 1e-10)


def slepian(area, nodes=400):
    """
    Largest eigenvalue of the sinc kernel sin(c(u-v))/(pi(u-v)) on [-1, 1] at c = area/4: the
    exact position-momentum bin overlap.
    """
    c = 0.25 * area
    u, w = np.polynomial.legendre.leggauss(nodes)
    gap = u[:, None] - u[None, :]
    near = np.abs(gap) < 1e-13
    safe = np.where(near, 1.0, gap)
    kern = np.where(near, c / math.pi, np.sin(c * safe) / (math.pi * safe))
    root = np.sqrt(w)

    return float(np.linalg.eigvalsh(root[:, None] * kern * root[None, :])[-1])


def gamma(t):
    """
    Furrer et al.'s gamma(t) exactly as printed, which divides by zero below t ~ 1e-8.
    """
    root = math.sqrt(1.0 + t * t)

    return (t + root) * (t / (root - 1.0)) ** t


def serfling(n, k, eps):
    """
    TLGR Eq. (2)'s statistical correction, transcribed from the paper.
    """
    return math.sqrt((n + k) / (n * k) * (k + 1.0) / k * math.log(4.0 / eps))


class EurRelations(Question):
    """
    The three published relations, as their papers print them.
    """

    def test_maassen_pair(self):
        """
        Two mutually unbiased bases in dimension d give overlap 1/d and quality log2 d.
        """
        for d in (2, 3, 4, 8, 16, 1024):
            self.assertClose(
                _core.eur_conjugate(d),
                math.log2(d),
                atol=1e-14,
                msg=f"q = log2 d at d = {d}",
            )
            self.assertClose(
                _core.eur_quality(1.0 / d),
                _core.eur_conjugate(d),
                atol=1e-14,
                msg=f"eur_quality agrees at c = 1/{d}",
            )

    def test_qubit_conjugate(self):
        """
        BB84's conjugate qubit bases give q = 1 bit, TLGR's numeric setting.
        """
        self.assertClose(_core.eur_conjugate(2), 1.0, atol=0.0, msg="q = 1 for a qubit pair")
        self.assertClose(_core.eur_quality(0.5), 1.0, atol=0.0, msg="c = 1/2 is the same number")
        self.assertClose(
            _core.eur_misalign(0.5 * math.pi),
            1.0,
            atol=1e-15,
            msg="Bloch axes at pi/2 are the conjugate pair",
        )
        self.assertClose(
            _core.eur_family("qubit"),
            1.0,
            atol=0.0,
            msg="the gate agrees",
        )

    def test_misalign_curve(self):
        """
        A misaligned qubit pair has overlap (1 + |cos theta|)/2, falling to q = 0 at theta = 0.
        """
        self.assertClose(_core.eur_misalign(0.0), 0.0, atol=0.0, msg="one basis has no partner")

        for theta in (0.05, 0.3, 0.8, 1.2, 1.5, 1.5707963267948966):
            want = -math.log2(0.5 * (1.0 + abs(math.cos(theta))))

            self.assertClose(
                _core.eur_misalign(theta),
                want,
                atol=1e-14,
                msg=f"c = (1 + |cos theta|)/2 at theta = {theta}",
            )

        walk = [_core.eur_misalign(0.1 * j) for j in range(16)]

        self.assertMonotone(walk, rising=True, strict=True, msg="quality rises toward pi/2")

    def test_memory_trivial(self):
        """
        Berta et al.'s H(A|B) term cancels q exactly on a maximally entangled pair.
        """
        for d in (2, 4, 8, 64):
            self.assertClose(
                _core.eur_memory(math.log2(d), -math.log2(d)),
                0.0,
                atol=1e-13,
                msg=f"the bound collapses to 0 at d = {d}",
            )

        self.assertGreaterEqual(
            _core.eur_memory(1.0, 0.0),
            _core.eur_conjugate(2),
            msg="a separable pair recovers Maassen and Uffink",
        )
        self.assertLess(
            _core.eur_memory(1.0, -1.0 + 1e-9),
            1e-8,
            msg="an almost maximally entangled pair leaves nothing",
        )

    def test_devetak_threshold(self):
        """
        The collective-attack rate is 1 - 2 h2(e) at q = 1, crossing zero at 11.0028%.
        """
        for e in (0.0, 0.01, 0.05, 0.11, 0.2, 0.5):
            self.assertClose(
                _core.eur_asymptotic(1.0, e, e),
                1.0 - 2.0 * h2(e),
                atol=1e-13,
                msg=f"1 - 2 h2(e) at e = {e}",
            )

        edge = 0.11002786443836032

        self.assertClose(
            _core.eur_asymptotic(1.0, edge, edge),
            0.0,
            atol=1e-13,
            msg="the zero of the one-way threshold",
        )
        self.assertLess(
            _core.eur_asymptotic(1.0, edge + 1e-4, edge + 1e-4),
            0.0,
            msg="above it the bound is negative and returned raw",
        )

    def test_smooth_chain(self):
        """
        The smooth relation is n q minus the max-entropy, reducing to n q at zero disturbance.
        """
        for n in (1.0, 100.0, 1e6):
            for h_max in (0.0, 1.0, 0.3 * n):
                self.assertClose(
                    _core.eur_smooth(n, 1.0, h_max),
                    n - h_max,
                    atol=1e-9 * max(1.0, n),
                    msg=f"n q - H_max at n = {n:g}, H_max = {h_max:g}",
                )

        self.assertClose(
            _core.eur_smooth(1.0, 1.0, 0.0),
            _core.eur_conjugate(2),
            atol=0.0,
            msg="one perfectly correlated round is the memoryless relation",
        )


class EurBinned(Question):
    """
    The position-momentum overlap, and the side the shipped closed form errs on.
    """

    def test_binned_conservative(self):
        """
        The closed-form quality never exceeds the exact one built from the Slepian eigenvalue.
        """
        for area in AREAS:
            exact = -math.log2(slepian(area))

            self.assertLessEqual(
                _core.eur_binned(math.sqrt(area), math.sqrt(area)),
                exact,
                msg=f"log2(2 pi / area) is at or below -log2 c at area = {area}",
            )

    def test_binned_tightness(self):
        """
        The shortfall against the exact quality is 1.0e-10 bits at delta = 0.01 and 9.99e-3 at
        1.
        """
        table = {
            0.01: 1.0e-10,
            0.1: 1.002e-06,
            1.0: 9.990e-03,
        }

        for delta, want in table.items():
            area = delta * delta
            gap = -math.log2(slepian(area)) - _core.eur_binned(delta, delta)

            self.assertClose(
                gap,
                want,
                atol=0.005 * want + 1e-12,
                msg=f"the closed form is short by {want:g} bits at delta = {delta}",
            )

    def test_binned_product(self):
        """
        The quality depends on the product of the two bin widths and nothing else.
        """
        want = _core.eur_binned(0.1, 1.0)

        for dq, dp in ((1.0, 0.1), (0.01, 10.0), (0.5, 0.2), (0.25, 0.4)):
            self.assertClose(
                _core.eur_binned(dq, dp),
                want,
                atol=1e-13,
                msg=f"dilation invariance at ({dq}, {dp})",
            )

    def test_differential_limit(self):
        """
        Unit bin area returns log2(2 pi), the differential relation at hbar = 1.
        """
        self.assertClose(
            _core.eur_binned(1.0, 1.0),
            math.log2(2.0 * math.pi),
            atol=1e-15,
            msg="h(Q|B) + h(P|C) >= log2(2 pi)",
        )
        self.assertClose(
            _core.eur_binned(0.01, 0.01) - _core.eur_binned(1.0, 1.0),
            -2.0 * math.log2(0.01),
            atol=1e-12,
            msg="the rest is the discretisation offset -2 log2 delta",
        )

    def test_area_ceiling(self):
        """
        A bin area at or above 2 pi is refused rather than returning a quality of zero.
        """
        area = 2.0 * math.pi
        edge = math.sqrt(area)

        self.assertFails(
            ValueError,
            "at or above 2 pi",
            lambda: _core.eur_binned(2.0, math.pi),
            msg="the closed form certifies nothing at exactly 2 pi",
        )
        self.assertFails(
            ValueError,
            "at or above 2 pi",
            lambda: _core.eur_binned(3.0, 3.0),
            msg="nor above it",
        )
        self.assertGreater(
            _core.eur_binned(edge - 1e-6, edge - 1e-6),
            0.0,
            msg="just below it the quality is positive",
        )
        self.assertGreater(
            -math.log2(slepian(area)),
            0.3,
            msg="the exact quality is still 0.36 bits there, so turning back is conservative",
        )

    def test_ball_grouping(self):
        """
        log2 gamma matches the printed grouping where it is evaluable and returns a number where
        it is not.
        """
        for t in (1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
            self.assertClose(
                _core.eur_ball(t),
                math.log2(gamma(t)),
                atol=1e-11,
                msg=f"log2 gamma(t) at t = {t}",
            )

        self.assertFails(
            ZeroDivisionError,
            "division by zero",
            lambda: gamma(1e-12),
            msg="the printed grouping cannot be evaluated there",
        )
        self.assertClose(
            _core.eur_ball(1e-12),
            4.2e-11,
            atol=1e-12,
            msg="the conjugate grouping returns a number",
        )
        self.assertClose(_core.eur_ball(0.0), 0.0, atol=0.0, msg="a perfect string costs nothing")

    def test_ball_monotone(self):
        """
        The max-entropy price rises with the average bin distance.
        """
        walk = [_core.eur_ball(0.25 * j) for j in range(12)]

        self.assertMonotone(walk, rising=True, strict=True, msg="log2 gamma rises with t")


class EurLength(Question):
    """
    The finite-key length, its corrections and its asymptotic limit.
    """

    def test_serfling_form(self):
        """
        The statistical correction reproduces TLGR Eq. (2) term for term.
        """
        for n in (1e4, 1e6, 1e9):
            for k in (1e3, 1e5):
                for eps in (1e-6, 1e-10):
                    self.assertClose(
                        _core.eur_serfling(n, k, eps),
                        serfling(n, k, eps),
                        atol=1e-14,
                        msg=f"mu at n = {n:g}, k = {k:g}, eps = {eps:g}",
                    )

    def test_serfling_above(self):
        """
        Sampling without replacement is wider than the with-replacement Hoeffding half-width.
        """
        for n in (1e5, 1e7, 1e9):
            for k in (1e3, 1e5):
                loose = math.sqrt(math.log(1.0 / 1e-10) / (2.0 * k))

                self.assertGreater(
                    _core.eur_serfling(n, k, 1e-10),
                    loose,
                    msg=f"Serfling exceeds Hoeffding at n = {n:g}, k = {k:g}",
                )

    def test_serfling_falls(self):
        """
        The correction falls as the sample grows and rises as the failure probability shrinks.
        """
        walk = [_core.eur_serfling(1e9, 10.0**j, 1e-10) for j in range(3, 9)]

        self.assertMonotone(walk, rising=False, strict=True, msg="mu falls with k")

        tight = [_core.eur_serfling(1e9, 1e5, 10.0**-j) for j in range(4, 14)]

        self.assertMonotone(tight, rising=True, strict=True, msg="mu rises as eps shrinks")

    def test_length_form(self):
        """
        The key length reproduces TLGR Eq. (2) with every correction subtracted.
        """
        n, k, q_tol, leak, eps_s, eps_c = POINT
        mu = serfling(n, k, eps_s)
        want = n * (1.0 - h2(q_tol + mu)) - leak - math.log2(2.0 / (eps_s * eps_s * eps_c))

        self.assertClose(
            _core.eur_length(n, k, 1.0, q_tol, leak, eps_s, eps_c),
            want,
            atol=1e-6,
            msg="TLGR Eq. (2)",
        )
        self.assertClose(
            _core.eur_secret("qubit", n, k, q_tol, leak, eps_s, eps_c),
            want,
            atol=1e-6,
            msg="the gated form is the same arithmetic",
        )

    def test_length_below(self):
        """
        Every correction shortens the key, so the length stays under n q.
        """
        n, k, q_tol, leak, eps_s, eps_c = POINT

        self.assertLess(
            _core.eur_length(n, k, 1.0, q_tol, leak, eps_s, eps_c),
            n,
            msg="the length sits below n q",
        )

        for extra in (0.0, 1e4, 1e5):
            longer = _core.eur_length(n, k, 1.0, q_tol, leak, eps_s, eps_c)
            shorter = _core.eur_length(n, k, 1.0, q_tol, leak + extra, eps_s, eps_c)

            self.assertLessEqual(shorter, longer, msg=f"more leakage is never more key ({extra:g})")

    def test_entropy_saturates(self):
        """
        A tolerated error past 1/2 saturates at one bit rather than letting h2 turn back.
        """
        n, k = 1e6, 1e5
        mu = serfling(n, k, 1e-10)
        walk = [_core.eur_length(n, k, 1.0, 0.4 + 0.02 * j, 0.0, 1e-10, 1e-10) for j in range(6)]

        self.assertMonotone(walk, rising=False, strict=False, msg="a worse channel is never more key")
        self.assertGreater(0.5 - mu, 0.4, msg="the sweep straddles the turn-back at 1/2")
        self.assertClose(
            _core.eur_length(n, k, 1.0, 0.5, 0.0, 1e-10, 1e-10),
            -math.log2(2.0 / (1e-10 * 1e-10 * 1e-10)),
            atol=1e-6,
            msg="at saturation the whole n q is spent",
        )

    def test_asymptotic_limit(self):
        """
        With Shannon-limit leakage the length per round approaches the collective-attack rate.
        """
        rate = _core.eur_asymptotic(1.0, 0.01, 0.01)

        for n in (1e8, 1e10, 1e12):
            k = math.sqrt(n)
            leak = n * h2(0.01)
            got = _core.eur_length(n, k, 1.0, 0.01, leak, 1e-10, 1e-10) / n

            self.assertLess(got, rate, msg=f"finite size costs at n = {n:g}")

        far = _core.eur_length(1e14, 1e10, 1.0, 0.01, 1e14 * h2(0.01), 1e-10, 1e-10) / 1e14

        self.assertClose(far, rate, atol=1e-3, msg="and vanishes as the block grows")


class EurFamilies(Question):
    """
    Every family this tree ships, and the condition each one fails.
    """

    def test_families_refused(self):
        """
        Each shipped family is refused by name with the condition it fails.
        """
        for name, needle in REFUSED:
            self.assertFails(
                NotImplementedError,
                needle,
                lambda n=name: _core.eur_family(n),
                msg=f"eur_family refuses {name}",
            )

    def test_reference_only(self):
        """
        The one name that returns a quality, qubit, is not a protocol this tree runs.
        """
        self.assertClose(_core.eur_family("qubit"), 1.0, atol=0.0, msg="TLGR's own setting")

        self.assertFails(
            ValueError,
            "unknown family",
            lambda: _core.eur_family("bb84-wcp"),
            msg="an unlisted name",
        )
        self.assertFails(
            ValueError,
            "qubit",
            lambda: _core.eur_family(""),
            msg="the refusal lists the one name that works",
        )

    def test_secret_gated(self):
        """
        The one function returning a key length refuses every shipped family by name.
        """
        n, k, q_tol, leak, eps_s, eps_c = POINT

        for name, _ in REFUSED:
            self.assertFails(
                NotImplementedError,
                name.split("/")[0][:3],
                lambda x=name: _core.eur_secret(x, n, k, q_tol, leak, eps_s, eps_c),
                msg=f"eur_secret refuses {name}",
            )

        self.assertGreater(
            _core.eur_secret("qubit", n, k, q_tol, leak, eps_s, eps_c),
            0.0,
            msg="the reference protocol does return a length",
        )

    def test_not_wired(self):
        """
        No module under qkd/ names any entry point of this engine.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(os.path.dirname(here), "qkd")
        hits = []

        for name in sorted(os.listdir(root)):
            if not name.endswith(".py"):
                continue

            with open(os.path.join(root, name), encoding="utf-8") as fh:
                if re.search(r"\beur_\w+", fh.read()):
                    hits.append(name)

        self.assertEqual(hits, [], msg="the engine has no Python surface")

    def test_names_exported(self):
        """
        All twelve entry points are reachable on _core and none is in qkd.__all__.
        """
        import qkd

        names = sorted(n for n in dir(_core) if n.startswith("eur_"))

        self.assertEqual(len(names), 12, msg="twelve entry points")

        for name in names:
            self.assertNotIn(name, qkd.__all__, msg=f"{name} is not a public surface")


class EurGuards(Guarded):
    """
    The direction each quantity moves in, and the arguments refused.
    """

    def test_quality_domain(self):
        """
        An overlap outside (0, 1] is refused, each end for its own reason.
        """
        self.assertFails(
            ValueError,
            "must be in [0, 1]",
            lambda: _core.eur_quality(1.5),
            msg="above one is not an overlap",
        )
        self.assertFails(
            ValueError,
            "overlap at least",
            lambda: _core.eur_quality(0.0),
            msg="zero would return an infinite quality",
        )
        self.assertFails(
            ValueError,
            "must be in [0, 1]",
            lambda: _core.eur_quality(-0.1),
            msg="negative is not an overlap",
        )
        self.assertClose(_core.eur_quality(1.0), 0.0, atol=0.0, msg="c = 1 is the vacuous end")

    def test_quality_direction(self):
        """
        A smaller overlap returns a larger quality, the direction a key length reads.
        """
        walk = [_core.eur_quality(1.0 / 2.0**j) for j in range(1, 12)]

        self.assertMonotone(walk, rising=True, strict=True, msg="q rises as c falls")

        for c in (0.05, 0.2, 0.5, 0.9):
            self.assertClose(
                _core.eur_quality(c),
                -math.log2(c),
                atol=1e-14,
                msg=f"q = -log2 c at c = {c}",
            )

    def test_smooth_subtracts(self):
        """
        The max-entropy is subtracted, so more of it is never more min-entropy.
        """
        walk = [_core.eur_smooth(1e6, 1.0, 1e5 * j) for j in range(10)]

        self.assertMonotone(walk, rising=False, strict=True, msg="H_min falls as H_max rises")
        self.assertLess(
            _core.eur_smooth(1e6, 1.0, 2e6),
            0.0,
            msg="a block whose correlation certifies nothing returns a negative bound",
        )

    def test_length_direction(self):
        """
        A tighter security parameter and a noisier channel both shorten the key.
        """
        n, k, q_tol, leak, _, eps_c = POINT
        tight = [_core.eur_length(n, k, 1.0, q_tol, leak, 10.0**-j, eps_c) for j in range(4, 14)]

        self.assertMonotone(tight, rising=False, strict=True, msg="a smaller eps_s costs key")

        noisy = [_core.eur_length(n, k, 1.0, 0.01 * j, leak, 1e-10, eps_c) for j in range(1, 20)]

        self.assertMonotone(noisy, rising=False, strict=False, msg="a worse channel costs key")

    def test_slots_refused(self):
        """
        Each entry point refuses the arguments outside its domain, one slot at a time.
        """
        self.assertSlots(
            _core.eur_serfling,
            (1e6, 1e5, 1e-10),
            [
                (0, "count in", [0.0, -1.0, 1e16, float("nan")]),
                (1, "count in", [0.0, -1.0, float("inf")]),
                (2, "must be in (0, 1)", [0.0, 1.0, -1e-9]),
            ],
            msg="eur_serfling",
        )
        self.assertSlots(
            _core.eur_length,
            (1e6, 1e5, 1.0, 0.01, 0.0, 1e-10, 1e-10),
            [
                (2, "must be >= 0", [-1e-9, float("nan")]),
                (3, "must be in [0, 1/2]", [0.6, 1.0, -1e-9]),
                (4, "must be >= 0", [-1.0, float("inf")]),
                (6, "must be in (0, 1)", [0.0, 1.0]),
            ],
            msg="eur_length",
        )
        self.assertSlots(
            _core.eur_binned,
            (0.1, 0.1),
            [
                (0, "must be > 0", [0.0, -0.1, float("nan")]),
                (1, "must be > 0", [0.0, -0.1, float("inf")]),
            ],
            msg="eur_binned",
        )
        self.assertSlots(
            _core.eur_asymptotic,
            (1.0, 0.05, 0.05),
            [
                (0, "must be >= 0", [-1e-9, float("nan")]),
                (1, "must be in [0, 1/2]", [0.51, 1.0]),
                (2, "must be in [0, 1/2]", [0.51, -1e-9]),
            ],
            msg="eur_asymptotic",
        )
        self.assertSlots(
            _core.eur_ball,
            (0.5,),
            [(0, "must be >= 0", [-1e-9, float("inf")])],
            msg="eur_ball",
        )

    def test_dimension_refused(self):
        """
        A one-dimensional system has no complementary partner and is refused by name.
        """
        self.assertFails(
            ValueError,
            "no complementary partner",
            lambda: _core.eur_conjugate(1),
            msg="d = 1 is vacuous",
        )
        self.assertFails(
            ValueError,
            "no complementary partner",
            lambda: _core.eur_conjugate(0),
            msg="d = 0 is not a dimension",
        )
        self.assertFails(
            OverflowError,
            "",
            lambda: _core.eur_conjugate(-2),
            msg="a negative dimension does not reach the engine",
        )


if __name__ == "__main__":
    rc = Exam(
        "EurRelations",
        "The entropic uncertainty relation as Maassen-Uffink, Berta and Tomamichel print it",
        "eur_relations.md",
    ).run(load(EurRelations))
    rc |= Exam(
        "EurBinned",
        "The position-momentum overlap, and the side the shipped closed form errs on",
        "eur_binned.md",
    ).run(load(EurBinned))
    rc |= Exam(
        "EurLength",
        "The composable finite-key length, its corrections and its asymptotic limit",
        "eur_length.md",
    ).run(load(EurLength))
    rc |= Exam(
        "EurFamilies",
        "Every family this tree ships, and the condition each one fails",
        "eur_families.md",
    ).run(load(EurFamilies))
    rc |= Exam(
        "EurGuards",
        "The direction each quantity moves in, and the arguments refused",
        "eur_guards.md",
    ).run(load(EurGuards))
    sys.exit(rc)
