import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import F_REC
from kit.checks import Guarded
from qkd import _core

# SIX-STATE is implemented, against Scarani & Renner, Phys. Rev. Lett. 100,
# 200501 (2008), arXiv:0708.0709, Lemmas 1 to 3, for a SINGLE-QUBIT source, as a key
# LENGTH in bits per block. Decoy composition refused: see src/sixstate.rs. SARG04 and
# B92 refuse outright. The paper's "Six-states" paragraph departs from the Lemmas
# twice, both optimistic; test_edge_binds measures the gap.

# EPS: composed deviation of the final key; EPS_EC: the error-correction failure inside it.
EPS = 1e-10
EPS_EC = 1e-15

# EPS - EPS_EC in thirds: hash slack, smoothing, estimation; estimation halved over the two samples.
SLOT = (EPS - EPS_EC) / 3.0
EACH = 0.5 * SLOT

# BIAS 0.9 sits below the interior optimum at every block size tried.
QBER = 0.02
BIAS = 0.9

# The one-way collective-attack threshold of Rev. Mod. Phys. 81, 1301 (2009),
# Appendix A, computed to f64 by test/sixstate.py's SIX_ZERO.
SIX_ZERO = 0.12619308327682110

# The two refusal exams differ only in the SOURCE they append.
SIX_BLOCK = (1e12, BIAS, QBER, QBER, 1.0, EPS, EPS_EC)


def finite(total, bias=BIAS, qber=QBER, f_ec=1.0, eps=EPS):
    """
    The finite-key rate per EMITTED signal at a depolarising point.
    """
    out = _core.sixstate_finite(total, bias, qber, qber, f_ec, eps, EPS_EC, "single")

    return out[1]


def fraction(total, bias=BIAS, qber=QBER, f_ec=1.0):
    """
    The same rate per SIFTED signal, the unit sixstate_secret is written in.
    """

    return finite(total, bias, qber, f_ec) / _core.sixstate_sift(bias)[1]


def asymptote(qber=QBER, f_ec=1.0):
    """
    The asymptotic single-qubit secret fraction.
    """

    return _core.sixstate_secret(qber, qber, qber, f_ec)


def counts(total, bias=BIAS):
    """
    The paper's sifting split: the raw key length and one monitor sample.
    """
    off = 0.5 * (1.0 - bias)

    return total * _core.sixstate_sift(bias)[1], total * off * off


def printed(total, bias=BIAS, qber=QBER):
    """
    H(X|E) per the paper's six-states paragraph, BOTH rates raised by their own width.
    """
    n_key, m_test = counts(total, bias)
    wide = _core.sixstate_width(n_key, EACH)
    other = _core.sixstate_width(m_test, EACH)

    return _core.sixstate_bound(min(qber + wide, 0.5), min(qber + other, 0.5))


def derived(total, bias=BIAS, qber=QBER, f_ec=1.0):
    """
    H(X|E) as sixstate_length minimises it, with the rates it minimised at.
    """
    n_key, m_test = counts(total, bias)
    out = _core.sixstate_length(n_key, m_test, qber, qber, f_ec, EPS, EPS_EC)

    return out[1], out[2], out[3]


class QubitWidth(Question):
    """
    Scarani & Renner's Lemma 3, the statistical width of one finite sample.
    """

    def test_width_shrinks(self):
        """
        The width falls with the sample size and rises as the failure probability tightens.
        """
        seen = [_core.sixstate_width(m, EACH) for m in (1e4, 1e6, 1e8, 1e10)]

        self.assertMonotone(seen, rising=False, msg=f"width: {seen}")

        tight = [_core.sixstate_width(1e8, e) for e in (1e-3, 1e-6, 1e-11)]

        self.assertMonotone(tight, rising=True, msg=f"width over eps: {tight}")

    def test_width_types(self):
        """
        The width carries a method-of-types logarithm, not a Hoeffding deviation: $m w^2$ grows
        with $m$, and $w$ decays slower than $1/\\sqrt{m}$.
        """
        sizes = (1e4, 1e6, 1e8, 1e10)
        rad = [m * _core.sixstate_width(m, EACH) ** 2 for m in sizes]

        self.assertMonotone(rad, rising=True, msg=f"m w^2: {rad}")

        scaled = [_core.sixstate_width(m, EACH) * m**0.5 for m in sizes]

        self.assertMonotone(scaled, rising=True, msg=f"w sqrt(m): {scaled}")

    def test_width_domain(self):
        """
        An empty sample and a failure probability outside (0, 1) are refused.
        """
        for bad in (0.0, -1.0):
            self.assertFails(
                ValueError,
                "m must be",
                _core.sixstate_width,
                bad,
                EACH,
                msg=f"m={bad}",
            )

        for bad in (0.0, 1.0):
            self.assertFails(
                ValueError,
                "eps must be",
                _core.sixstate_width,
                1e6,
                bad,
                msg=f"eps={bad}",
            )


class QubitEntropy(Question):
    """
    Eve's residual uncertainty H(X|E), and which edge of a sample binds it.
    """

    def test_bound_matches(self):
        """
        CROSS-ENGINE. Scarani & Renner's closed form equals one minus the Rev. Mod. Phys.
        Appendix A Holevo bound at the same three error rates, a check across two papers.
        """
        for key in (0.0, 0.01, 0.05, 0.1, 0.2, 0.4):
            for test in (0.5 * key, key, 0.5 * (key + 0.5), 0.5):
                got = _core.sixstate_bound(key, test)
                want = 1.0 - _core.sixstate_eve(test, test, key)
                self.assertClose(got, want, atol=1e-14, msg=f"({key}, {test}): {got} != {want}")

    def test_bound_monotone(self):
        """
        The bound falls with the monitor-basis rate and RISES with the key-basis rate, putting
        the compatible set's minimum at the key rate's lower edge.
        """
        rising = [_core.sixstate_bound(k, 0.25) for k in (0.02, 0.05, 0.1, 0.2, 0.4)]

        self.assertMonotone(rising, rising=True, msg=f"rises with e_key: {rising}")

        falling = [_core.sixstate_bound(0.05, t) for t in (0.03, 0.05, 0.1, 0.3, 0.5)]

        self.assertMonotone(falling, rising=False, msg=f"falls with e_test: {falling}")

    def test_bound_region(self):
        """
        Over the WHOLE physical region the bound never falls with the key-basis rate and never
        rises with the monitor rate, licensing a corner read of the compatible set's minimum.
        """
        grid = [i * 0.0125 for i in range(41)]
        rise, fall = 0, 0
        for i, key in enumerate(grid):
            for j, test in enumerate(grid):
                if key > 2.0 * test:
                    continue

                here = _core.sixstate_bound(key, test)
                if i + 1 < len(grid) and grid[i + 1] <= 2.0 * test:
                    rise += _core.sixstate_bound(grid[i + 1], test) < here - 1e-12

                if j + 1 < len(grid):
                    fall += _core.sixstate_bound(key, grid[j + 1]) > here + 1e-12
        self.assertEqual(rise, 0, msg=f"key-rate violations {rise}")
        self.assertEqual(fall, 0, msg=f"monitor-rate violations {fall}")

    def test_bound_physical(self):
        """
        A key-basis rate above twice the monitor rate, marginals no Bell-diagonal state has, is
        refused, the expression still returning numbers there.
        """
        self.assertFails(
            ValueError,
            "Bell-diagonal",
            _core.sixstate_bound,
            0.2,
            0.05,
            msg="bound(0.2, 0.05)",
        )
        self.assertClose(
            _core.sixstate_bound(0.1, 0.05),
            0.9,
            msg="bound(0.1, 0.05)",
        )

    def test_bound_extremes(self):
        """
        A noiseless channel leaves Eve one bit of uncertainty and a random monitor basis leaves
        none.
        """
        self.assertClose(_core.sixstate_bound(0.0, 0.0), 1.0, msg="bound(0, 0)")
        self.assertClose(
            _core.sixstate_bound(0.0, 0.5),
            0.0,
            msg="bound(0, 0.5)",
        )


class QubitLength(Question):
    """
    The key length in bits, Scarani & Renner Lemmas 1 to 3 composed.
    """

    def test_length_units(self):
        """
        The length is floored to whole bits and never negative: positive at 1e12 signals, zero
        at 1e4.
        """
        n_key, m_test = counts(1e12)
        out = _core.sixstate_length(n_key, m_test, QBER, QBER, 1.0, EPS, EPS_EC)

        self.assertGreater(out[0], 0.0, msg=f"length {out[0]}")
        self.assertClose(out[0], math.floor(out[0]), msg=f"length {out[0]}")

        small = counts(1e4)
        none = _core.sixstate_length(small[0], small[1], QBER, QBER, 1.0, EPS, EPS_EC)

        self.assertClose(none[0], 0.0, msg=f"length {none[0]}")

    def test_edge_binds(self):
        """
        The compatible-set minimum puts the key-basis rate a full width LOWER and the monitor
        rate higher, so the paper's printed instantiation, raising both, over-reports H(X|E) by
        a gap that closes with the block.
        """
        gaps = []
        for total in (1e6, 1e8, 1e10, 1e12):
            hxe, key, test = derived(total)
            wide = _core.sixstate_width(counts(total)[0], EACH)
            self.assertLess(key, QBER, msg=f"e_key {key}")
            self.assertGreater(test, QBER, msg=f"e_test {test}")
            self.assertClose(key, QBER - wide, atol=1e-15, msg=f"e_key {key}, width {wide}")

            gaps.append(printed(total) - hxe)

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"gap {gap}")
        self.assertMonotone(gaps, rising=False, msg=f"gaps: {gaps}")

    def test_block_monotone(self):
        """
        The length and the length per signal both rise with the block.
        """
        totals = (1e8, 1e10, 1e12, 1e14, 1e16)
        lens = []
        for total in totals:
            n_key, m_test = counts(total)
            got = _core.sixstate_length(n_key, m_test, QBER, QBER, 1.0, EPS, EPS_EC)
            lens.append(got[0])
        self.assertMonotone(lens, rising=True, msg=f"length: {lens}")

        rates = [lens[i] / totals[i] for i in range(len(totals))]

        self.assertMonotone(rates, rising=True, msg=f"rate: {rates}")

    def test_epsilon_price(self):
        """
        A weaker secrecy parameter buys a longer key.
        """
        lens = [finite(1e10, eps=e) for e in (1e-12, 1e-10, 1e-5)]

        self.assertMonotone(lens, rising=True, msg=f"length over eps: {lens}")

    def test_leak_charged(self):
        """
        Error correction charged at the observed key-basis rate shortens the key as f_EC rises
        from 1.
        """
        lens = [finite(1e12, f_ec=f) for f in (1.0, 1.1, F_REC)]

        self.assertMonotone(lens, rising=False, msg=f"length over f_EC: {lens}")

    def test_length_budget(self):
        """
        An error-correction failure probability at or above the composed deviation is refused.
        """
        n_key, m_test = counts(1e12)

        self.assertFails(
            ValueError,
            "eps_ec",
            _core.sixstate_length,
            n_key,
            m_test,
            QBER,
            QBER,
            1.0,
            1e-10,
            1e-10,
            msg="eps_ec = eps = 1e-10",
        )


class QubitFinite(Question):
    """
    One six-state run end to end, and the limit it converges onto.
    """

    def test_below_asymptotic(self):
        """
        THE SANITY PROPERTY. The finite rate never exceeds the same-parameter asymptotic rate at
        any block size, basis bias or error rate.
        """
        for qber in (0.0, 0.01, 0.05, 0.1, 0.12):
            for bias in (0.4, 0.5, 0.7, 0.9, 0.99):
                share = _core.sixstate_sift(bias)[1]
                want = _core.sixstate_secret(qber, qber, qber, F_REC) * share
                for total in (1e6, 1e10, 1e14, 1e20):
                    got = finite(total, bias, qber, F_REC)
                    self.assertLessEqual(
                        got,
                        want,
                        msg=f"Q {qber}, bias {bias}, N {total}: {got} > {want}",
                    )

    def test_rate_converges(self):
        """
        MEASURED CONVERGENCE. The rate per sifted signal rises onto the asymptotic secret
        fraction, the gap falling over ninefold per two decades of block to 1.634e-8 at 1e22.
        """
        totals = (1e12, 1e14, 1e16, 1e18, 1e20, 1e22)
        want = asymptote()
        seen = [fraction(total) for total in totals]
        gaps = [want - got for got in seen]

        self.assertMonotone(seen, rising=True, msg=f"fraction: {seen}")

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"gap {gap}")
        self.assertMonotone(gaps, rising=False, msg=f"gaps: {gaps}")
        self.assertClose(gaps[-1], 1.634e-8, atol=1e-10, msg=f"gap at 1e22: {gaps[-1]}")

        ratio = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
        for step in ratio:
            self.assertGreater(step, 9.0, msg=f"ratios: {ratio}")

    def test_block_shrinks(self):
        """
        The rate falls as the block shrinks and is exactly zero at 1e5 signals, never negative.
        """
        seen = [finite(total) for total in (1e5, 1e6, 1e8, 1e10, 1e12)]

        self.assertMonotone(seen, rising=True, msg=f"rate: {seen}")
        self.assertClose(seen[0], 0.0, msg=f"rate at 1e5: {seen[0]}")
        self.assertGreater(seen[1], 0.0, msg=f"rate at 1e6: {seen[1]}")

    def test_bias_optimum(self):
        """
        The basis bias has an interior optimum: 0.95 beats 0.7 beats the uniform third, and
        0.999 starves the monitor bases to zero.
        """
        rates = {b: finite(1e8, bias=b) for b in (0.34, 0.7, 0.95, 0.999)}

        self.assertGreater(rates[0.95], rates[0.7], msg=f"rates: {rates}")
        self.assertGreater(rates[0.7], rates[0.34], msg=f"rates: {rates}")
        self.assertClose(rates[0.999], 0.0, msg=f"rate at 0.999: {rates[0.999]}")

    def test_sift_accounting(self):
        """
        The block splits by the paper's three-basis sifting, $b^2$ to the key basis and
        $((1-b)/2)^2$ to each monitor basis, mismatches discarded.
        """
        out = _core.sixstate_finite(1e12, 0.8, QBER, QBER, 1.0, EPS, EPS_EC, "single")
        off = 0.5 * (1.0 - 0.8)

        self.assertClose(out[2], 1e12 * 0.64, atol=1e-3, msg=f"key share {out[2]}")
        self.assertClose(out[3], 1e12 * off * off, atol=1e-3, msg=f"monitor share {out[3]}")
        self.assertClose(out[1], out[0] / 1e12, atol=1e-18, msg=f"rate {out[1]}")

    def test_threshold_holds(self):
        """
        At and above the 12.6% asymptotic crossing the finite rate is zero at every block.
        """
        for total in (1e10, 1e16, 1e22):
            for qber in (SIX_ZERO, 0.13, 0.2):
                got = finite(total, qber=qber)
                self.assertClose(got, 0.0, msg=f"N {total}, Q {qber} certified {got}")


class QubitRefusals(Guarded):
    """
    What the qubit families refuse, and the missing piece each names.
    """

    def test_decoy_refused(self):
        """
        The six-state decoy composition is refused: the photon-number inversion is basis-blind,
        but Scarani and Renner's Lemma 3 wants a per-round single-photon state decoy counts do
        not identify.
        """
        self.assertFails(
            NotImplementedError,
            "STATISTICAL TRANSFER",
            _core.sixstate_finite,
            *SIX_BLOCK,
            "decoy",
            msg="source=decoy",
        )

    def test_source_named(self):
        """
        An unrecognised source is refused by name.
        """
        self.assertFails(
            ValueError,
            "unknown source",
            _core.sixstate_finite,
            *SIX_BLOCK,
            "coherent",
            msg="source=coherent",
        )

    def test_sarg_refused(self):
        """
        A SARG04 block is refused, routing to sarg_length and naming Eq. (39)'s two-photon term,
        which that length drops rather than bounds, and the missing complementary-basis sample.
        """
        self.assertFails(
            NotImplementedError,
            "TWO-photon gain Q2",
            _core.sarg_finite,
            1e12,
            msg="sarg_finite(1e12)",
        )
        self.assertFails(
            NotImplementedError,
            "no complementary-basis sample",
            _core.sarg_finite,
            1e12,
            msg="sarg_finite(1e12)",
        )

    def test_b92_refused(self):
        """
        Plain B92 is refused for want of a joint confidence region, and strong-reference B92 for
        taking its phase error as an input.
        """
        self.assertFails(
            NotImplementedError,
            "joint confidence region",
            _core.b92_finite,
            1e12,
            False,
            msg="b92_finite plain",
        )
        self.assertFails(
            NotImplementedError,
            "malformed",
            _core.b92_finite,
            1e12,
            True,
            msg="b92_finite strong reference",
        )

    def test_route_named(self):
        """
        Each refusal cites the route in rather than stopping at unimplemented.
        """
        for args, needle in (
            ((1e12, False), "npj Quantum Information 6, 104 (2020)"),
            ((1e12, True), "Phys. Rev. Lett. 93, 120501 (2004)"),
        ):
            self.assertFails(
                NotImplementedError,
                needle,
                _core.b92_finite,
                *args,
                msg=f"B92 route {needle}",
            )
        self.assertFails(
            NotImplementedError,
            "Phys. Rev. A 73, 012337 (2006)",
            _core.sarg_finite,
            1e12,
            msg="SARG04 route",
        )

    def test_block_checked(self):
        """
        Every refusal checks its block size first, refusing an empty block as a bad argument,
        not an unimplemented protocol.
        """
        for fn, args in ((_core.sarg_finite, (0.0,)), (_core.b92_finite, (0.0, False))):
            self.assertBad("n_total", fn, args, msg="n_total=0")


if __name__ == "__main__":
    rc = Exam(
        "QubitWidth",
        "Scarani & Renner, Phys. Rev. Lett. 100, 200501 (2008), Lemma 3",
        "finitequbit_width.md",
    ).run(load(QubitWidth))
    rc |= Exam(
        "QubitEntropy",
        "Six-state H(X|E), against Rev. Mod. Phys. 81, 1301 (2009) Appendix A",
        "finitequbit_entropy.md",
    ).run(load(QubitEntropy))
    rc |= Exam(
        "QubitLength",
        "The finite key length in bits, Scarani & Renner Lemmas 1 to 3",
        "finitequbit_length.md",
    ).run(load(QubitLength))
    rc |= Exam(
        "QubitFinite",
        "One six-state run end to end, and its measured asymptotic limit",
        "finitequbit_finite.md",
    ).run(load(QubitFinite))
    rc |= Exam(
        "QubitRefusals",
        "SARG04, B92 and the six-state decoy composition, each refused by name",
        "finitequbit_refusals.md",
    ).run(load(QubitRefusals))
    sys.exit(rc)
