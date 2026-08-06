import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.procs import ROOT
from qkd import _core

# The postselection technique: the lift from an IID-collective proof to a coherent one.
#
# CHRISTANDL, KOENIG & RENNER, Phys. Rev. Lett. 102, 020504 (2009), arXiv:0809.3019,
# Theorem 1, with g_{n,d} = C(n + d^2 - 1, n) <= (n+1)^(d^2-1) for d = dim H, applied to QKD
# with H = H_A (x) H_B in their Eq. (4).
#
# NAHAR, TUPKARY, ZHAO, LUTKENHAUS & TAN, PRX Quantum 5, 040315 (2024), arXiv:2403.11851,
# are the version `src/postselect.rs` implements. Skipping their fix moves the security
# parameter INSECURE: CKR09 compose as max{eps_PA + 2 eps_bar, eps_AT}, NTZLT as
# eps_PA + 2 eps_bar + 2 sqrt(2 eps_AT).
#
# The factor transforms a FAILURE PROBABILITY upward and a key LENGTH downward by 2 log2 g
# bits. Never a rate.
#
# Every family this tree ships is refused by name; that is the module's result.

# NTZLT Sec. V.2's optical setup, Eq. (37) and Corollary 3.3's block form both giving 52.
THREE = (2, 1, 8)

# A decoy-state BB84 SHAPE in NTZLT Eq. (39)'s arguments. The receiver half is theirs; the
# three intensities and the three-photon cutoff are NOT from a paper.
DECOY = (3, 3, 2, 1, 8)

# Every name `ps_family` refuses, with the substring its message must carry.
REFUSED = [
    ("bb84", "general attacks"),
    ("sixstate", "unconditional security follows immediately"),
    ("sarg", "no longer vacuously"),
    ("b92", "b92_finite"),
    ("mdi", "against general attacks in the finite-key regime"),
    ("rrdps", "C2 fails"),
    ("pairing", "C1 fails"),
    ("bbm92", "asymptotic"),
    ("e91", "asymptotic"),
    ("cow", "supplied upper bound"),
    ("dps", "no longer vacuously"),
    ("gaussian", "collective-only"),
    ("cv", "collective-only"),
    ("dmcs", "working assumption"),
    ("cvmdi", "continuous-variable"),
]

# The names that DO return a dimension. Neither is a path in this tree.
ALLOWED = {
    "qubit": 16,
    "three-state": 52,
}


def exact_log2(k):
    """
    log2 of an arbitrary-precision integer, without overflowing float on the way.
    """
    shift = k.bit_length() - 53

    if shift <= 0:
        return math.log2(k)

    return shift + math.log2(k >> shift)


def reference(n, x):
    """
    log2 C(n + x - 1, x - 1) from Python's exact integer binomial.
    """
    return exact_log2(math.comb(int(n) + x - 1, x - 1))


class PostselectTheorem(Question):
    """
    The theorem as CKR09 and NTZLT print it.
    """

    def test_binomial_exact(self):
        """
        ps_cost reproduces log2 of the exact integer binomial C(n + x - 1, x - 1).
        """
        for n in (1.0, 10.0, 1000.0, 1e6, 1e9):
            for x in (2, 3, 16, 52, 231):
                got = _core.ps_cost(n, x)
                self.assertClose(
                    got,
                    reference(n, x),
                    atol=1e-9 * max(1.0, abs(got)),
                    msg=f"log2 g at n={n:g}, x={x}",
                )

    def test_ckr_power(self):
        """
        The exact cost sits under CKR09's printed relaxation g_{n,d} <= (n+1)^(d^2 - 1).
        """
        for n in (1e3, 1e6, 1e9, 1e13):
            for x in (16, 52, 231):
                loose = (x - 1) * math.log2(n + 1.0)
                self.assertLessEqual(
                    _core.ps_cost(n, x),
                    loose,
                    msg=f"above CKR09 relaxation at n={n:g}, x={x}",
                )

    def test_cap_above(self):
        """
        NTZLT's closed form (e(n + x - 1)/(x - 1))^(x - 1) is at or above the exact cost.
        """
        for n in (1.0, 1e3, 1e9, 1e13):
            for x in (2, 16, 52, 2340):
                self.assertGreaterEqual(
                    _core.ps_cap(n, x),
                    _core.ps_cost(n, x),
                    msg=f"cap below cost at n={n:g}, x={x}",
                )

    def test_qubit_dim(self):
        """
        A qubit pair reduces at x = 16, by NTZLT's d_A^2 d_B^2 and by CKR09's d^2 alike.
        """
        both = (2 * 2) ** 2
        self.assertEqual(_core.ps_dim(2, 2), 16, msg="ps_dim(2, 2)")
        self.assertEqual(both, 16, msg="CKR09 d^2")
        self.assertEqual(_core.ps_family("qubit"), 16, msg="ps_family")

    def test_three_state(self):
        """
        NTZLT Sec. V.2's optical setup reduces at x = 52, by Eq. (37) and by the block form.
        """
        blocks = sum((i + 1) ** 2 for i in range(2)) + 8
        printed = 4**2 + 9 * 2**2
        self.assertEqual(_core.ps_squash(*THREE), 52, msg="NTZLT Eq. (37)")
        self.assertEqual(4 * blocks, 52, msg="Eq. (37) recomputed")
        self.assertEqual(printed, 52, msg="Sec. V.2 block form")
        self.assertEqual(_core.ps_family("three-state"), 52, msg="ps_family")

    def test_secrecy_form(self):
        """
        The composition is NTZLT's sum with the square root, strictly above CKR09's max form.
        """
        pa, bar, at = 1e-12, 1e-12, 1e-20

        got = _core.ps_secrecy(pa, bar, at)
        self.assertClose(
            got,
            pa + 2.0 * bar + 2.0 * math.sqrt(2.0 * at),
            atol=1e-24,
            msg="NTZLT Corollary 3.1",
        )

        for one in (1e-30, 1e-12, 1e-4, 0.01):
            for two in (1e-30, 1e-12, 1e-4, 0.01):
                sums = _core.ps_secrecy(one, one, two)
                self.assertGreater(
                    sums,
                    max(one + 2.0 * one, two),
                    msg=f"not above max form at eps_PA={one:g}, eps_AT={two:g}",
                )

    def test_optical_shape(self):
        """
        Eq. (39) is Eq. (38)'s source factor times Eq. (37)'s receiver factor.
        """
        n_int, n_ph, d_a, cut, flags = DECOY

        source = n_int * n_int * (n_ph + 2)
        self.assertEqual(
            _core.ps_optical(*DECOY),
            source * _core.ps_squash(d_a, cut, flags),
            msg="Eq. (39) = Eq. (38) * Eq. (37)",
        )
        self.assertEqual(
            _core.ps_tagged(n_int, n_ph, d_a, 2),
            source * _core.ps_dim(d_a, 2),
            msg="Eq. (38) alone",
        )
        self.assertEqual(_core.ps_optical(*DECOY), 2340, msg="ps_optical(DECOY)")


class PostselectCost(Question):
    """
    The lift's cost in key bits, security parameter and acceptance-test width.
    """

    def test_qubit_cost(self):
        """
        A qubit pair over a 1e9 block: log2 g = 408.2102, a key cost of exactly 2 log2 g,
        and an IID budget of 2^-441.4294.
        """
        cost = _core.ps_cost(1e9, 16)
        self.assertClose(cost, 408.2102, atol=1e-3, msg="log2 g at x = 16, n = 1e9")
        self.assertClose(
            1e6 - _core.ps_length(1e6, 1e9, 16),
            2.0 * cost,
            atol=1e-9,
            msg="key cost != 2 log2 g",
        )

        budget = _core.ps_budget(1e-10, 1e9, 16)
        self.assertClose(budget, -441.4294, atol=1e-3, msg="log2 IID budget")
        self.assertGreater(2.0**budget, 0.0, msg="2^budget underflowed at x = 16")

    def test_three_state(self):
        """
        At NTZLT's x = 52, n = 1.08e13 the cost is 1988.2204 and the IID budget underflows
        f64, so ps_epsilon refuses even 5e-324.
        """
        cost = _core.ps_cost(1.08e13, 52)
        self.assertClose(cost, 1988.2204, atol=1e-3, msg="log2 g at x = 52, n = 1.08e13")

        budget = _core.ps_budget(1e-10, 1.08e13, 52)
        self.assertClose(budget, -2021.4404, atol=1e-3, msg="log2 IID budget")
        self.assertEqual(2.0**budget, 0.0, msg="2^budget did not underflow at x = 52")
        self.assertFails(
            ValueError,
            "not representable",
            lambda: _core.ps_epsilon(5e-324, 1.08e13, 52),
            msg="ps_epsilon at 5e-324",
        )

    def test_decoy_shape(self):
        """
        A decoy-BB84 shape reduces at x = 2340 and costs 94240.25 key bits over a 1e9 block,
        leaving most of a megabit.
        """
        x = _core.ps_optical(*DECOY)
        cost = _core.ps_cost(1e9, x)
        self.assertClose(cost, 47120.124, atol=1e-2, msg="log2 g at x = 2340, n = 1e9")
        self.assertClose(2.0 * cost, 94240.25, atol=2e-2, msg="key cost in bits")
        self.assertGreater(
            _core.ps_length(1e6, 1e9, x),
            0.0,
            msg="ps_length not positive",
        )
        self.assertLess(_core.ps_length(1e6, 1e9, x), 1e6, msg="ps_length not below input")

    def test_width_binds(self):
        """
        The acceptance-test half-width at x = 2340 is 2.5569e-2 at n = 1e9 and 3.2934e-4 at
        1e13, falling with the block.
        """
        x = _core.ps_optical(*DECOY)
        seen = []

        for n in (1e9, 1e12, 1e13):
            budget = _core.ps_budget(1e-10, n, x)
            at = _core.ps_thirds(budget)[2]
            seen.append(_core.ps_width(0.05 * n, 10.0, at))
        self.assertClose(seen[0], 2.5569e-2, atol=1e-5, msg="width at n = 1e9")
        self.assertClose(seen[2], 3.2934e-4, atol=1e-7, msg="width at n = 1e13")
        self.assertMonotone(seen, rising=False, msg="width not falling with n")

    def test_width_qubit(self):
        """
        At x = 16, n = 1e9 the acceptance-test half-width is 2.4884e-3.
        """
        budget = _core.ps_budget(1e-10, 1e9, 16)
        at = _core.ps_thirds(budget)[2]
        self.assertClose(
            _core.ps_width(5e7, 10.0, at),
            2.4884e-3,
            atol=1e-6,
            msg="width at x = 16, n = 1e9",
        )

    def test_thirds_compose(self):
        """
        ps_thirds recomposes through ps_secrecy to 2^budget, with eps_AT squared.
        """
        budget = -50.0
        pa, bar, at = _core.ps_thirds(budget)
        self.assertClose(
            _core.ps_secrecy(2.0**pa, 2.0**bar, 2.0**at),
            2.0**budget,
            atol=1e-24,
            msg="thirds do not recompose to 2^budget",
        )
        self.assertClose(at, 2.0 * budget - math.log2(72.0), atol=1e-12, msg="eps_AT != 2*budget - log2 72")

    def test_length_deficit(self):
        """
        A 1000-bit key against a 94240-bit lift returns the raw deficit, not a clamped zero.
        """
        short = _core.ps_length(1e3, 1e9, _core.ps_optical(*DECOY))
        self.assertLess(short, 0.0, msg="deficit not negative")
        self.assertClose(short, 1e3 - 94240.25, atol=2e-2, msg="deficit in bits")


class PostselectFamilies(Question):
    """
    Every protocol this tree ships, and the condition each one fails.
    """

    def test_families_refused(self):
        """
        ps_family refuses all fifteen qkd family names, each naming its own reason.
        """
        for name, needle in REFUSED:
            self.assertFails(
                NotImplementedError,
                needle,
                lambda n=name: _core.ps_family(n),
                msg=f"family {name}",
            )

    def test_general_attacks(self):
        """
        bb84, mdi and sixstate are refused naming Lim, Curty and Scarani, whose proofs are
        already against general attacks.
        """
        for name, paper in (("bb84", "Lim"), ("mdi", "Curty"), ("sixstate", "Scarani")):
            self.assertFails(
                NotImplementedError,
                paper,
                lambda n=name: _core.ps_family(n),
                msg=f"{name} does not name {paper}",
            )

    def test_pairing_invariance(self):
        """
        Mode pairing is the one family refused on permutation invariance rather than
        dimension.
        """
        self.assertFails(
            NotImplementedError,
            "only family here that fails on permutation invariance",
            lambda: _core.ps_family("pairing"),
            msg="pairing C1",
        )

    def test_cv_excluded(self):
        """
        The gaussian refusal names Leverrier: cv_finite is collective-only and
        infinite-dimensional.
        """
        self.assertFails(
            NotImplementedError,
            "Leverrier",
            lambda: _core.ps_family("gaussian"),
            msg="gaussian does not name Leverrier",
        )

    def test_allowed_names(self):
        """
        ps_family returns 16 for "qubit" and 52 for "three-state".
        """
        for name, want in ALLOWED.items():
            self.assertEqual(_core.ps_family(name), want, msg=f"reference dimension for {name}")

    def test_unknown_name(self):
        """
        An unrecognised family raises and lists both the accepted and the refused names.
        """
        self.assertFails(
            ValueError,
            "unknown family",
            lambda: _core.ps_family("bb84-wcp"),
            msg="bb84-wcp accepted",
        )

    def test_not_wired(self):
        """
        None of the 14 ps_* entry points appears in any module under qkd/.
        """
        names = [n for n in dir(_core) if n.startswith("ps_")]
        hits = []

        for leaf in sorted(os.listdir(os.path.join(ROOT, "qkd"))):
            if not leaf.endswith(".py"):
                continue

            with open(os.path.join(ROOT, "qkd", leaf)) as handle:
                text = handle.read()

            hits += [f"{leaf}:{n}" for n in names if re.search(rf"\b{n}\b", text)]
        self.assertEqual(len(names), 14, msg="ps_* entry point count")
        self.assertEqual(hits, [], msg="qkd/ reaches ps_*")


class PostselectGuards(Guarded):
    """
    The correction's direction, and every argument refused.
    """

    def test_epsilon_upward(self):
        """
        ps_epsilon raises the failure probability, and rises with n and with x.
        """
        base = _core.ps_epsilon(1e-200, 1e9, 16)
        self.assertGreater(base, 1e-200, msg="ps_epsilon not above input")
        self.assertGreater(
            _core.ps_epsilon(1e-200, 1e12, 16),
            base,
            msg="not rising with n",
        )
        self.assertGreater(
            _core.ps_epsilon(1e-250, 1e9, 20),
            _core.ps_epsilon(1e-250, 1e9, 16),
            msg="not rising with x",
        )

    def test_length_downward(self):
        """
        ps_length is below its input at x = 2, 16, 52 and 2340.
        """
        for x in (2, 16, 52, 2340):
            self.assertLess(_core.ps_length(1e7, 1e9, x), 1e7, msg=f"ps_length not below input at x={x}")

    def test_budget_stricter(self):
        """
        ps_budget is below log2(eps) at x = 2, 16 and 52.
        """
        for x in (2, 16, 52):
            self.assertLess(
                _core.ps_budget(1e-10, 1e9, x),
                math.log2(1e-10),
                msg=f"budget not tighter at x={x}",
            )

    def test_epsilon_vacuous(self):
        """
        ps_epsilon refuses a lift that does not land below 1.
        """
        self.assertFails(
            ValueError,
            "does not land below 1",
            lambda: _core.ps_epsilon(1e-10, 1e9, 16),
            msg="ps_epsilon(1e-10, 1e9, 16)",
        )

    def test_secrecy_vacuous(self):
        """
        ps_secrecy refuses a composition at or above 1.
        """
        self.assertFails(
            ValueError,
            "not below 1",
            lambda: _core.ps_secrecy(0.1, 0.1, 0.2),
            msg="ps_secrecy(0.1, 0.1, 0.2)",
        )

    def test_trivial_x(self):
        """
        ps_dim(1, 1) is refused as a no-op.
        """
        self.assertFails(
            ValueError,
            "no-op",
            lambda: _core.ps_dim(1, 1),
            msg="ps_dim(1, 1)",
        )

    def test_cost_domain(self):
        """
        ps_cost refuses a fractional, zero, oversized or non-finite block and an
        out-of-range reduction dimension.
        """
        self.assertSlots(
            _core.ps_cost,
            (1e9, 16),
            [
                (0, "whole number of signals", [0.0, -1.0, 1.5, 1e16, float("nan")]),
                (1, "reduction dimension", [0, 1, 1 << 21]),
            ],
            msg="ps_cost",
        )

    def test_dim_domain(self):
        """
        ps_dim, ps_tagged and ps_squash refuse a zero or oversized dimension in every slot.
        """
        self.assertSlots(
            _core.ps_dim,
            (2, 2),
            [
                (0, "d_a must be a dimension", [0, 2000]),
                (1, "d_b must be a dimension", [0, 2000]),
            ],
            msg="ps_dim",
        )
        self.assertSlots(
            _core.ps_tagged,
            (3, 3, 2, 2),
            [
                (0, "n_int must be a dimension", [0, 2000]),
                (1, "n_ph must be a count", [2000]),
                (2, "d_a must be a dimension", [0, 2000]),
            ],
            msg="ps_tagged",
        )
        self.assertSlots(
            _core.ps_squash,
            (2, 1, 8),
            [
                (1, "cutoff must be a count", [2000]),
                (2, "flags must be a count", [2000]),
            ],
            msg="ps_squash",
        )

    def test_log2_domain(self):
        """
        ps_thirds and ps_width refuse a positive or non-finite logarithm.
        """
        self.assertSlots(
            _core.ps_thirds,
            (-50.0,),
            [(0, "negative log2", [0.0, 1.0, float("inf"), float("nan")])],
            msg="ps_thirds",
        )
        self.assertSlots(
            _core.ps_width,
            (1e6, 10.0, -50.0),
            [
                (1, "at least one POVM element", [0.0, 0.5, float("nan")]),
                (2, "negative log2", [0.0, 1.0]),
            ],
            msg="ps_width",
        )

    def test_lift_gated(self):
        """
        ps_lift refuses all fifteen qkd family names, and on "qubit" spends both key length
        and security parameter.
        """
        args = (1e6, 1e-140, 1e-140, 1e-280, 1e9)

        for name, _ in REFUSED:
            self.assertFails(
                NotImplementedError,
                name,
                lambda n=name: _core.ps_lift(n, *args),
                msg=f"ps_lift refuses {name}",
            )

        short, lifted = _core.ps_lift("qubit", *args)
        self.assertLess(short, 1e6, msg="ps_lift length not spent")
        self.assertGreater(lifted, 1e-140, msg="ps_lift epsilon not raised")


if __name__ == "__main__":
    rc = Exam(
        "PostselectTheorem",
        "The postselection theorem as CKR09 and NTZLT print it",
        "postselect_theorem.md",
    ).run(load(PostselectTheorem))
    rc |= Exam(
        "PostselectCost",
        "What the lift costs in key bits, in epsilon and in acceptance-test width",
        "postselect_cost.md",
    ).run(load(PostselectCost))
    rc |= Exam(
        "PostselectFamilies",
        "Every family this tree ships, and the condition each one fails",
        "postselect_families.md",
    ).run(load(PostselectFamilies))
    rc |= Exam(
        "PostselectGuards",
        "The direction the correction must be applied in, and the arguments refused",
        "postselect_guards.md",
    ).run(load(PostselectGuards))
    sys.exit(rc)
