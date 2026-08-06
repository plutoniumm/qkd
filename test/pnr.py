import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

import qkd as q
from kit.checks import Guarded
from qkd import _core

# The occupancy derivation these check against is src/pnr.rs's module header.
# Nothing is pinned to this engine's own output: every reference is a closed form or another
# engine.

# Fock cutoffs. At mu <= 3 the Poisson mass past level 79 is below 1e-70.
CUT_BIG = 80

CUT_MID = 40

CUT_LOW = 12

# Doubling, so the 1/N rate reads off as a ratio of successive deficits.
LADDER = (64, 128, 256, 512, 1024, 2048, 4096)

# Mean photon numbers for the coherent-state folds.
FLUXES = (0.01, 0.1, 0.5, 1.0, 3.0)

# Thermal: geometric in n, the case decoy.rs's sender-side Poisson inversion cannot address.
NBAR = 0.8


def thermal(nbar, levels):
    """
    Thermal photon-number distribution on levels 0..levels-1; 400 levels leaves under 1e-40.
    """

    return [(nbar**n) / ((1.0 + nbar) ** (n + 1)) for n in range(levels)]


def poisson(mu, levels):
    """
    Photon-number distribution of a coherent state of mean photon number ``mu``.
    """

    return [math.exp(-mu) * mu**n / math.factorial(n) for n in range(levels)]


def exact(elems, n, k, eta, dark):
    """
    The array POVM's diagonal from the inclusion-exclusion form, in exact rationals: in f64
    it cancels catastrophically.
    """
    eta, dark = Fraction(eta), Fraction(dark)
    total = Fraction(0)

    for i in range(k + 1):
        silent = elems - k + i
        term = (1 - dark) ** silent * (1 - silent * eta / elems) ** n
        total += (-1) ** i * Fraction(math.comb(k, i)) * term

    return float(Fraction(math.comb(elems, k)) * total)


def deficit(elems, n):
    """
    How far a unit-efficiency array of ``elems`` elements falls short of reporting ``n``.
    """
    diag = _core.pnr_array(n + 1, elems, 1.0, 0.0).diagonal()

    return 1.0 - float(diag[n][n])


def settings(delta):
    """
    Moroder's attenuator triple c = 0, delta, sqrt(delta), as transmittances: c = 1 - atten.
    """

    return (1.0, 1.0 - delta, 1.0 - math.sqrt(delta))


def observe(attens, probs, eta=1.0, dark=0.0):
    """
    The three no-click probabilities a receiver would record at ``attens``.
    """

    return tuple(_core.pnr_noclick(eta, dark, a, probs) for a in attens)


class ArrayPovm(Question):
    def test_array_click(self):
        """
        At one element the array POVM reproduces `click_prob` across efficiency, dark count
        and coherent-state flux.
        """
        worst = 0.0

        for eta in (0.1, 0.4, 0.75, 1.0):
            for dark in (0.0, 1e-6, 1e-3, 0.05):
                povm = _core.pnr_array(CUT_BIG, 1, eta, dark)

                for mu in FLUXES:
                    got = povm.coherent(mu)
                    want = _core.click_prob(math.sqrt(mu), 0.0, eta, dark)
                    worst = max(worst, abs(float(got[1]) - want))
                    worst = max(worst, abs(float(got[0]) - (1.0 - want)))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"click_prob gap {worst:.3e}")

    def test_array_threshold(self):
        """
        At one element the no-click element is (1 - d)(1 - eta)^n on the Fock diagonal.
        """
        diag = _core.pnr_array(CUT_LOW, 1, 0.35, 0.004).diagonal()
        worst = 0.0

        for n in range(CUT_LOW):
            want = 0.996 * 0.65**n
            worst = max(worst, abs(float(diag[0][n]) - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"no-click diagonal gap {worst:.3e}")

    def test_array_inclusion(self):
        """
        The occupancy recursion agrees with the closed inclusion-exclusion form of the same
        POVM, evaluated in exact rationals.
        """
        worst = 0.0

        for elems in (1, 2, 3, 4, 5):
            for eta, dark in ((0.4, 0.0), (0.4, 0.05), (1.0, 0.01)):
                diag = _core.pnr_array(9, elems, eta, dark).diagonal()

                for n in range(9):
                    for k in range(elems + 1):
                        want = exact(elems, n, k, eta, dark)
                        worst = max(worst, abs(float(diag[k][n]) - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"recursion gap {worst:.3e}")

    def test_array_coherent(self):
        """
        Folding the POVM against a Poisson photon-number distribution agrees with the closed
        binomial count a balanced splitter gives directly.
        """
        worst = 0.0

        for elems in (1, 2, 4, 8, 16):
            for eta in (0.3, 0.9):
                for dark in (0.0, 1e-3):
                    povm = _core.pnr_array(90, elems, eta, dark)

                    for mu in (0.2, 1.0, 4.0):
                        got = np.asarray(povm.coherent(mu))
                        want = np.asarray(_core.pnr_coherent(elems, eta, dark, mu))
                        worst = max(worst, float(np.max(np.abs(got - want))))

        self.assertClose(worst, 0.0, atol=1e-14, msg=f"binomial gap {worst:.3e}")

    def test_array_complete(self):
        """
        The array's elements sum to the identity on the truncated space and each one is
        positive semi-definite.
        """
        povm = _core.pnr_array(CUT_LOW, 6, 0.7, 1e-3)
        total = sum(povm.element(k) for k in range(povm.outcomes))
        gap = float(np.max(np.abs(total - np.eye(CUT_LOW))))

        self.assertClose(gap, 0.0, atol=1e-14, msg=f"identity gap {gap:.3e}")
        self.assertClose(povm.defect(), 0.0, atol=1e-14, msg=f"diagonal defect {povm.defect():.3e}")

        floor = min(float(np.linalg.eigvalsh(povm.element(k)).min()) for k in range(povm.outcomes))

        self.assertGreaterEqual(floor, 0.0, msg=f"least eigenvalue {floor:.3e}")

    def test_array_vacuum(self):
        """
        The no-click element is (1 - d)^N (1 - eta)^n at every element count.
        """
        worst = 0.0

        for elems in (1, 2, 8, 64):
            for dark in (0.0, 1e-3, 0.02):
                diag = _core.pnr_array(8, elems, 0.55, dark).diagonal()

                for n in range(8):
                    want = (1.0 - dark) ** elems * 0.45**n
                    worst = max(worst, abs(float(diag[0][n]) - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"no-click gap {worst:.3e}")

    def test_array_dark(self):
        """
        On vacuum the array reports a binomial count over its elements.
        """
        worst = 0.0

        for elems in (1, 3, 12):
            for dark in (1e-3, 0.05):
                diag = _core.pnr_array(4, elems, 0.7, dark).diagonal()

                for k in range(elems + 1):
                    want = math.comb(elems, k) * dark**k * (1.0 - dark) ** (elems - k)
                    worst = max(worst, abs(float(diag[k][0]) - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"vacuum row gap {worst:.3e}")

    def test_array_distinct(self):
        """
        At unit efficiency the probability of reporting the true number is exactly
        `pnr_distinct`, the probability that no two photons share a bin.
        """
        diag = _core.pnr_array(9, 8, 1.0, 0.0).diagonal()
        worst = 0.0

        for n in range(9):
            worst = max(worst, abs(float(diag[n][n]) - _core.pnr_distinct(8, n)))

        self.assertClose(worst, 0.0, atol=0.0, msg=f"occupancy gap {worst:e}")

    def test_array_efficiency(self):
        """
        The no-click probability falls monotonically with element efficiency and is exactly
        zero at unit efficiency on three photons.
        """
        seq = [float(_core.pnr_array(CUT_LOW, 4, eta, 0.0).diagonal()[0][3]) for eta in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]

        self.assertMonotone(seq, rising=False, msg="no-click against efficiency")
        self.assertClose(seq[-1], 0.0, atol=0.0, msg=f"no-click at eta = 1 is {seq[-1]:e}")

    def test_array_elements(self):
        """
        The probability of reporting the true photon number rises monotonically with the
        element count, at fixed per-element efficiency.
        """
        seq = [float(_core.pnr_array(6, elems, 0.9, 0.0).diagonal()[5][5]) for elems in (8, 16, 32, 64, 128, 256)]

        self.assertMonotone(seq, rising=True, msg="five-photon resolution against N")

    def test_array_converge(self):
        """
        Doubling the element count halves the deficit from true photon-number resolution,
        which is first order in 1/N with coefficient $n(n-1)/2$.
        """
        for n in (2, 3, 5):
            gaps = [deficit(elems, n) for elems in LADDER]

            self.assertMonotone(gaps, rising=False, msg=f"n={n}: deficit against N")

            ratio = gaps[-2] / gaps[-1]

            self.assertClose(ratio, 2.0, atol=0.01, msg=f"n={n}: deficit ratio {ratio:.4f}")

            scaled = LADDER[-1] * gaps[-1]
            want = 0.5 * n * (n - 1)

            self.assertClose(scaled, want, atol=0.01 * want, msg=f"n={n}: N*deficit {scaled:.4f} != {want}")


class Resolution(Question):
    def test_tes_identity(self):
        """
        Unit efficiency, no background and a zero-width readout give Pi_k = |k><k| to the
        last bit.
        """
        povm = _core.pnr_tes(CUT_LOW, 1.0, 0.0, 0.0)
        gap = float(np.max(np.abs(povm.diagonal() - np.eye(CUT_LOW))))

        self.assertClose(gap, 0.0, atol=0.0, msg=f"identity gap {gap:e}")

    def test_tes_complete(self):
        """
        With efficiency, background and a finite readout width the elements still sum to the
        identity and stay positive semi-definite.
        """
        povm = _core.pnr_tes(CUT_LOW, 0.9, 0.02, 0.2)
        total = sum(povm.element(k) for k in range(povm.outcomes))
        gap = float(np.max(np.abs(total - np.eye(CUT_LOW))))

        self.assertClose(gap, 0.0, atol=1e-14, msg=f"identity gap {gap:.3e}")
        self.assertGreaterEqual(povm.floor(), 0.0, msg=f"least diagonal entry {povm.floor():.3e}")

    def test_tes_confuse(self):
        """
        The readout confusion matrix is stochastic on the reported number and its interior
        diagonal is flat under a linear response.
        """
        conf = _core.pnr_confuse(10, 0.3, 0.0)
        sums = np.asarray(conf).sum(axis=0)
        gap = float(np.max(np.abs(sums - 1.0)))

        self.assertClose(gap, 0.0, atol=1e-15, msg=f"column sum gap {gap:.3e}")

        core = np.diag(np.asarray(conf))[1:-1]

        self.assertClose(
            float(core.max() - core.min()),
            0.0,
            atol=1e-15,
            msg="interior diagonal spread",
        )

    def test_tes_width(self):
        """
        Resolution falls monotonically with the readout width and is exact at width zero.
        """
        seq = [float(_core.pnr_tes(CUT_LOW, 1.0, 0.0, sig).diagonal()[3][3]) for sig in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)]

        self.assertClose(seq[0], 1.0, atol=0.0, msg=f"zero width resolves {seq[0]!r}")
        self.assertMonotone(seq, rising=False, strict=False, msg="resolution against width")

    def test_tes_saturate(self):
        """
        A saturating response resolves high numbers worse than low ones, costing the top
        interior number more than half against a linear response.
        """
        lin = np.diag(np.asarray(_core.pnr_confuse(10, 0.3, 0.0)))[1:-1]
        sat = np.diag(np.asarray(_core.pnr_confuse(10, 0.3, 3.0)))[1:-1]

        self.assertMonotone(list(sat), rising=False, msg="saturated resolution against n")
        self.assertGreater(
            float(lin[-1] - sat[-1]),
            0.5,
            msg=f"top interior cost {float(lin[-1] - sat[-1]):.3f}",
        )

    def test_tes_mean(self):
        """
        The mean reported number is the mean detected number plus the readout's vacuum bias
        weighted by the chance the detected number was zero.
        """
        counts = np.arange(CUT_MID)

        for eta, bg, sig in ((0.8, 0.05, 0.25), (1.0, 0.5, 0.25), (0.9, 0.02, 0.2)):
            conf = np.asarray(_core.pnr_confuse(CUT_MID - 1, sig, 0.0))
            bias = float(counts @ conf[:, 0])
            diag = np.asarray(_core.pnr_tes(CUT_MID, eta, bg, sig).diagonal())

            for n in (0, 1, 3, 7, 12):
                got = float(counts @ diag[:, n])
                want = eta * n + bg + (1.0 - eta) ** n * math.exp(-bg) * bias

                self.assertClose(got, want, atol=2e-9, msg=f"n={n}: mean report {got:.10f} != {want:.10f}")

    def test_tes_floor(self):
        """
        The vacuum bias is upward and grows with the readout width, a count having nothing
        below zero to absorb into.
        """
        counts = np.arange(CUT_MID)
        seq = [
            float(counts @ np.asarray(_core.pnr_confuse(CUT_MID - 1, sig, 0.0))[:, 0]) for sig in (0.1, 0.25, 0.5, 0.8)
        ]

        self.assertMonotone(seq, rising=True, msg="vacuum bias against width")

        got = float(counts @ np.asarray(_core.pnr_tes(CUT_MID, 1.0, 0.0, 0.3).diagonal())[:, 0])

        self.assertGreater(got, 0.0, msg=f"vacuum report {got:.3e}")

    def test_tes_absorb(self):
        """
        The top outcome absorbs, so backgrounds of 0.1, 2 and 20 counts leave completeness
        intact and a background of 20 lands past 0.9 in that outcome.
        """
        for bg in (0.1, 2.0, 20.0):
            povm = _core.pnr_tes(CUT_LOW, 1.0, bg, 0.2)

            self.assertClose(povm.defect(), 0.0, atol=1e-14, msg=f"background {bg} defect {povm.defect():.3e}")

        heavy = np.asarray(_core.pnr_tes(CUT_LOW, 1.0, 20.0, 0.2).diagonal())

        self.assertGreater(
            float(heavy[CUT_LOW - 1].min()),
            0.9,
            msg=f"absorbing outcome holds {float(heavy[CUT_LOW - 1].min()):.3f}",
        )


class DetectorDecoy(Question):
    """
    Detector decoy: Moroder, Curty & Lutkenhaus, New J. Phys. 11, 045008 (2009),
    arXiv:0811.0027.
    """

    def test_decoy_noclick(self):
        """
        The forward model is the paper's Eq. (17), the photon-number distribution read
        against the kernel (1 - c*eta)^n behind a dark-count floor.
        """
        probs = thermal(NBAR, 400)
        worst = 0.0

        for eta, dark, atten in ((1.0, 0.0, 0.4), (0.6, 1e-3, 0.9), (0.25, 0.02, 1.0)):
            got = _core.pnr_noclick(eta, dark, atten, probs)
            kernel = 1.0 - atten * eta
            want = (1.0 - dark) * sum(p * kernel**n for n, p in enumerate(probs))
            worst = max(worst, abs(got - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"Eq. (17) gap {worst:.3e}")

    def test_decoy_poisson(self):
        """
        For a signal already Poissonian at the receiver, attenuating by t is the same as
        sending intensity t*mu, so `decoy_gain` reproduces the attenuator sweep.
        """
        probs = poisson(0.5, CUT_BIG)
        worst = 0.0

        for atten in (1.0, 0.3, 0.05):
            got = 1.0 - _core.pnr_noclick(1.0, 0.0, atten, probs)
            want = _core.decoy_gain(0.5, atten, 0.0, 0.0)[0]
            worst = max(worst, abs(got - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"decoy_gain gap {worst:.3e}")

    def test_decoy_reuse(self):
        """
        On that Poissonian signal `decoy_bounds` runs verbatim on receiver-side intensities,
        and its single-photon yield stays under `decoy_ideal`'s infinite-decoy ceiling.
        """
        probs = poisson(0.5, CUT_BIG)
        gains = [1.0 - _core.pnr_noclick(1.0, 0.0, t, probs) for t in (1.0, 0.2, 0.02)]
        got = _core.decoy_bounds(0.5, 0.1, 0.01, gains[0], 0.02, gains[1], 0.02, gains[2], 0.5)
        ceiling = _core.decoy_ideal(1.0, 0.0, 0.02)

        self.assertFinite(got, msg="decoy_bounds triple")
        self.assertGreater(got[0], 0.9, msg=f"y1 bound {got[0]:.6f}")
        self.assertLessEqual(
            got[0],
            ceiling[0] + 1e-12,
            msg=f"y1 bound {got[0]:.6f} over ceiling {ceiling[0]:.6f}",
        )

    def test_decoy_bracket(self):
        """
        Proposition 2.1's bounds bracket the true single- and two-photon weights of a thermal
        state, the vacuum weight read off f(0) exactly.
        """
        probs = thermal(NBAR, 400)
        attens = settings(0.01)
        got = _core.pnr_decoy(1.0, 0.0, attens, observe(attens, probs), 1.0)

        self.assertClose(got[0], probs[0], atol=1e-14, msg=f"p0 gap {abs(got[0] - probs[0]):.3e}")
        self.assertLessEqual(got[1], probs[1], msg=f"lower bound {got[1]:.6f} over p1 {probs[1]:.6f}")
        self.assertGreaterEqual(got[2], probs[1], msg=f"upper bound {got[2]:.6f} under p1 {probs[1]:.6f}")
        self.assertLessEqual(got[3], probs[2], msg=f"lower bound {got[3]:.6f} over p2 {probs[2]:.6f}")
        self.assertGreaterEqual(got[4], probs[2], msg=f"upper bound {got[4]:.6f} under p2 {probs[2]:.6f}")

    def test_decoy_converge(self):
        """
        Both brackets close as the probing settings approach the transparent one, the
        proposition's c1 = delta, c2 = sqrt(delta) limit.
        """
        probs = thermal(NBAR, 400)
        first = []
        second = []

        for delta in (0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4):
            attens = settings(delta)
            got = _core.pnr_decoy(1.0, 0.0, attens, observe(attens, probs), 1.0)
            first.append(got[2] - got[1])
            second.append(got[4] - got[3])

        self.assertMonotone(first, rising=False, msg="single-photon bracket against delta")
        self.assertMonotone(second, rising=False, msg="two-photon bracket against delta")
        self.assertLess(first[-1], 1e-4, msg=f"single-photon bracket {first[-1]:.3e}")

    def test_decoy_refuses(self):
        """
        A detector of less than unit efficiency is refused: no attenuator reaches the c = 0
        the vacuum weight is read off.
        """
        probs = thermal(NBAR, 400)
        attens = settings(0.01)
        vacs = observe(attens, probs, eta=0.8)

        self.assertFails(
            ValueError,
            "not 0",
            _core.pnr_decoy,
            0.8,
            0.0,
            attens,
            vacs,
            1.0,
            msg="eta = 0.8",
        )

    def test_decoy_counting(self):
        """
        A number-resolving detector reaches the inversion through its mean spurious count per
        gate, as the Poisson chance of at least one, and an equivalent threshold detector
        gives the same three no-click probabilities.
        """
        probs = [0.6, 0.3, 0.08, 0.02]
        att = q.DecoyAttenuator()
        det = q.PnrDetector(eta=1.0, background=0.02, sigma=0.0, saturation=3.0, cutoff=CUT_MID)
        got = att.noclick(det, probs)
        worst = 0.0

        for atten, vac in zip(att.settings, got):
            row = _core.pnr_tes(CUT_MID, atten * det.eta, det.background, 0.0, det.saturation).diagonal()[0]
            want = float(sum(r * p for r, p in zip(row, probs)))
            worst = max(worst, abs(vac - want))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"vacuum element gap {worst:.3e}")

        same = q.ClickDetector(eta=det.eta, dark=1.0 - math.exp(-det.background))

        self.assertEqual(att.noclick(same, probs), got, msg="gate against converted count")

        bound = att.bounds(det, got)

        self.assertClose(bound.p0, probs[0], atol=1e-14, msg=f"p0 gap {abs(bound.p0 - probs[0]):.3e}")
        self.assertLessEqual(bound.p1_lo, probs[1], msg=f"lower bound {bound.p1_lo:.6f} over p1")
        self.assertGreaterEqual(bound.p1_hi, probs[1], msg=f"upper bound {bound.p1_hi:.6f} under p1")

    def test_decoy_array(self):
        """
        A multiplexed array reaches the inversion through the whole receiver's per-gate silent
        probability and not through one element's.
        """
        probs = [0.6, 0.3, 0.08, 0.02]
        att = q.DecoyAttenuator()
        one = q.ThresholdArray(elements=1, eta=1.0, dark=1e-3, cutoff=CUT_MID)

        self.assertClose(one.gate(), one.dark, atol=0.0, msg=f"one element composed to {one.gate()!r}")
        self.assertEqual(
            att.noclick(one, probs),
            att.noclick(q.ClickDetector(eta=one.eta, dark=one.dark), probs),
            msg="one element against ClickDetector",
        )

        worst = 0.0
        naive = 0.0

        for elems in (2, 8, 64):
            arr = q.ThresholdArray(elements=elems, eta=1.0, dark=1e-3, cutoff=CUT_MID)
            want = 1.0 - (1.0 - arr.dark) ** elems

            self.assertClose(arr.gate(), want, atol=1e-15, msg=f"{elems} elements composed to {arr.gate()!r}")

            for atten, vac in zip(att.settings, att.noclick(arr, probs)):
                row = _core.pnr_array(CUT_MID, elems, atten * arr.eta, arr.dark).diagonal()[0]
                held = float(sum(r * p for r, p in zip(row, probs)))
                worst = max(worst, abs(vac - held))
                naive = max(naive, abs(_core.pnr_noclick(arr.eta, arr.dark, atten, probs) - held))

        self.assertClose(worst, 0.0, atol=1e-15, msg=f"array element gap {worst:.3e}")
        self.assertGreater(naive, 1e-3, msg=f"per-element gap {naive:.3e}")

    def test_decoy_readout(self):
        """
        A finite readout width, a single absorbing outcome and a component carrying neither
        field are each refused by the forward model and by the inversion.
        """
        probs = [0.6, 0.3, 0.08, 0.02]
        att = q.DecoyAttenuator()
        cases = (
            (NotImplementedError, "no per-gate dark", q.PnrDetector(eta=1.0, sigma=0.3)),
            (ValueError, "one absorbing outcome", q.PnrDetector(eta=1.0, cutoff=1)),
            (ValueError, "must carry a per-gate", q.Homodyne()),
        )

        for kind, text, det in cases:
            self.assertFails(
                kind,
                text,
                lambda det=det: att.noclick(det, probs),
                msg=f"{det.__class__.__name__} at the forward model",
            )

            self.assertFails(
                kind,
                text,
                lambda det=det: att.bounds(det, (0.9, 0.91, 0.95)),
                msg=f"{det.__class__.__name__} at the inversion",
            )


class PnrGuards(Guarded):
    def test_guard_array(self):
        """
        `pnr_array` refuses a cutoff, element count, efficiency or dark rate outside its
        domain, and a POVM too large to hold.
        """
        cases = (
            (0, "cutoff must be in", (0, 513, 1000)),
            (1, "elements must be in", (0, 4097)),
            (2, "eta must be in", (0.0, 1.5, float("nan"))),
            (3, "dark must be in", (1.0, -0.1, float("inf"))),
        )

        self.assertSlots(_core.pnr_array, (16, 4, 0.5, 0.01), cases, msg="pnr_array domain")
        self.assertBad(
            "diagonal entries",
            _core.pnr_array,
            (512, 4096, 0.5, 0.0),
            msg="512 x 4096 POVM",
        )

    def test_guard_tes(self):
        """
        `pnr_tes` refuses a cutoff, efficiency, background, readout width or saturation scale
        outside its domain.
        """
        cases = (
            (0, "cutoff must be in", (0, 513)),
            (1, "eta must be in", (0.0, 1.5)),
            (2, "background must be", (-1.0, float("nan"))),
            (3, "sigma must be", (-0.1, float("inf"))),
            (4, "sat must be", (-1.0,)),
        )

        self.assertSlots(_core.pnr_tes, (16, 0.9, 0.01, 0.2, 0.0), cases, msg="pnr_tes domain")

    def test_guard_decoy(self):
        """
        `pnr_decoy` refuses an unordered or out-of-range setting triple and a zero mass cap,
        and `pnr_noclick` refuses anything that is not a photon-number distribution.
        """
        good = (1.0, 0.0, (1.0, 0.99, 0.9), (0.5, 0.5, 0.6), 1.0)

        self.assertBad(
            "0 < c1 < c2 < 1",
            _core.pnr_decoy,
            (1.0, 0.0, (1.0, 0.9, 0.99), (0.5, 0.5, 0.6), 1.0),
            msg="unordered settings",
        )
        self.assertBad(
            "cap must be > 0",
            _core.pnr_decoy,
            good[:4] + (0.0,),
            msg="cap = 0",
        )
        self.assertBad(
            "atten1 must be in",
            _core.pnr_decoy,
            (1.0, 0.0, (1.0, 1.5, 0.9), (0.5, 0.5, 0.6), 1.0),
            msg="atten1 = 1.5",
        )
        self.assertBad(
            "probs is empty",
            _core.pnr_noclick,
            (0.5, 0.0, 0.5, []),
            msg="empty distribution",
        )
        self.assertBad(
            "which is not a probability",
            _core.pnr_noclick,
            (0.5, 0.0, 0.5, [0.5, -0.1]),
            msg="weight = -0.1",
        )
        self.assertBad(
            "more than one photon-number distribution",
            _core.pnr_noclick,
            (0.5, 0.0, 0.5, [0.9, 0.9]),
            msg="distribution sums to 1.8",
        )

    def test_guard_povm(self):
        """
        The returned POVM refuses an outcome it does not carry, a distribution on the wrong
        cutoff, and a coherent state whose tail the cutoff would silently drop.
        """
        povm = _core.pnr_array(8, 4, 0.5, 0.0)

        self.assertBad(
            "is past the",
            povm.element,
            (9,),
            msg="outcome 9 of 4",
        )
        self.assertBad(
            "must be given on the same",
            povm.fold,
            ([0.5, 0.5],),
            msg="2 levels against cutoff 8",
        )
        self.assertBad(
            "past this POVM's cutoff",
            povm.coherent,
            (40.0,),
            msg="mu = 40 against cutoff 8",
        )
        self.assertBad(
            "mu must be >= 0",
            povm.coherent,
            (-1.0,),
            msg="mu = -1",
        )


if __name__ == "__main__":
    rc = Exam(
        "PnrArray",
        "Multiplexed threshold array: POVM, threshold limit, PNR limit",
        "pnr_array.md",
    ).run(load(ArrayPovm))
    rc |= Exam(
        "PnrResolution",
        "Number-resolving readout with a parametrised confusion matrix",
        "pnr_resolution.md",
    ).run(load(Resolution))
    rc |= Exam(
        "PnrDecoy",
        "Detector decoy: PNR statistics from a threshold detector and an attenuator",
        "pnr_decoy.md",
    ).run(load(DetectorDecoy))
    rc |= Exam(
        "PnrGuards",
        "Argument domain of the photon-number-resolving layer",
        "pnr_guards.md",
    ).run(load(PnrGuards))
    sys.exit(rc)
