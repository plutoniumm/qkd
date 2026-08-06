import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.forms import background, bisect, h2
from kit.fuzzing import loguni, rng
import qkd as q
from qkd import _core, gaussian
from qkd import pairs as pr

# Ma, Fung & Lo, Phys. Rev. A 76, 012307 (2007), quant-ph/0703122, Table 1: the
# entanglement-PDC simulation set. 249 MHz at 710 nm reaches the per-second rate alone.
DET = 0.145
E_DET = 0.015
Y_DARK = 6.02e-6
F_REC = 1.22
# Per DETECTOR and per gate: 1 - (1 - DARK)^2 = Y_DARK, the receiver figure Ma-Fung-Lo state.
DARK = 3.0100045300684997e-06
PUMP = 249e6

# Fibre attenuation, dB/km. The position sweep's own figure, not an anchor.
ALPHA = 0.2

# Acin, Brunner, Gisin, Massar, Pironio & Scarani, Phys. Rev. Lett. 98, 230501
# (2007): the collective-attack device-independent QBER threshold, 7.1%.
DI_QBER = 0.071

TSIRELSON = 2.0 * math.sqrt(2.0)

# Tomamichel & Leverrier, "A largely self-contained and complete security proof
# for quantum key distribution", Quantum 1, 14 (2017), arXiv:1506.08458: the four points
# marked on their Figure 7, printed by the authors' own examples.py in the arXiv source
# package. (delta, m, k, ell, nu, the total epsilon printed beside it).
TL_POINTS = (
    (0.010, 36123, 5911, 15323, 0.06943318811415392, 9.778763272730244e-11),
    (0.025, 51114, 8484, 17009, 0.0580348064984291, 9.698272881235974e-11),
    (0.050, 128989, 20941, 27878, 0.03685663259331777, 9.872424026202253e-11),
    (0.075, 731680, 104258, 85453, 0.016324856748689963, 9.9261091464143e-11),
)

# Figure 7's own setting: mutually unbiased qubit measurements on Alice's side, a 1e-10
# target, and their leakage approximation r = 1.1*h(delta)*(m - k).
TL_CBAR = 0.5
TL_EPS = 1e-10
TL_FEC = 1.1


def leverrier(m, delta, k):
    """
    The (t, leak, eps_pa) their examples.py derives from a block length, on a HEURISTIC split of
    theirs -- alpha = m^-1/2, beta = 0.9*(1 - alpha) -- that is not a step of Theorem 3.
    """
    alpha = m**-0.5
    beta = (1.0 - alpha) * 0.9
    tag = float(math.ceil(-math.log2(alpha * TL_EPS)))
    leak = float(math.ceil(TL_FEC * h2(delta) * (m - k)))

    return (tag, leak, (1.0 - alpha - beta) * TL_EPS)


def refusal():
    """
    The text pair_finite raises for a coincidence-counted link.
    """
    try:
        _core.pair_finite(1e6, 0.01, TL_FEC, TL_CBAR, TL_EPS, "coincidence")
    except NotImplementedError as exc:
        return str(exc)

    return ""


def couple(lam, eta_a, eta_b):
    """
    One two-mode squeezer's four joint click outcomes, (no/no, yes/no, no/yes, yes/yes).
    """
    a = 1.0 + eta_a * lam
    b = 1.0 + eta_b * lam
    c = 1.0 + lam * (eta_a + eta_b - eta_a * eta_b)

    return (
        1.0 / c,
        1.0 / b - 1.0 / c,
        1.0 / a - 1.0 / c,
        1.0 - 1.0 / a - 1.0 / b + 1.0 / c,
    )


def exact(lam, eta_a, eta_b):
    """
    (gain, error) from an independent enumeration of the four detectors, owing nothing to the
    engine's algebra.
    """
    joint = couple(lam, eta_a, eta_b)
    keys = ((0, 0), (1, 0), (0, 1), (1, 1))
    gain = 0.0
    errs = 0.0
    for s1, w1 in zip(keys, joint):
        for s2, w2 in zip(keys, joint):
            if not (s1[0] or s2[0]) or not (s2[1] or s1[1]):
                continue

            pa = 0.5 if (s1[0] and s2[0]) else (0.0 if s1[0] else 1.0)
            pb = 0.5 if (s2[1] and s1[1]) else (0.0 if s1[1] else 1.0)
            gain += w1 * w2
            errs += w1 * w2 * (pa * (1.0 - pb) + (1.0 - pa) * pb)

    return gain, errs / gain


def sweep(eta_a, eta_b, e_d=E_DET, y0=Y_DARK):
    """
    The optimised rate at one pair of arm efficiencies, bits per pump pulse.
    """

    return pr.optimum(eta_a, eta_b, y0, y0, e_d, F_REC, 0.5)[3]


def reach(share):
    """
    Total fibre length at which the rate crosses zero, for a source `share` of the way from Alice to
    Bob.
    """

    def rate(km):
        eta_a = DET * 10.0 ** (-ALPHA * share * km / 10.0)
        eta_b = DET * 10.0 ** (-ALPHA * (1.0 - share) * km / 10.0)

        return sweep(eta_a, eta_b)

    return bisect(rate, 1.0, 900.0)


def arm(km):
    """
    One arm as a q.Fiber; a zero-length arm is stated as T = 1 rather than as a length of nothing.
    """
    if km <= 0.0:
        return q.Fiber(T=1.0)

    return q.Fiber(length=km, alpha=ALPHA)


def build(share, km, brightness=None):
    """
    A q.pairs link with the source `share` of the way along a `km` span.
    """

    return pr.PairLink(
        source=q.PairSource(brightness=brightness, rate=PUMP),
        detectors=(
            q.ClickDetector(eta=DET, dark=DARK),
            q.ClickDetector(eta=DET, dark=DARK),
        ),
        channels=(arm(share * km), arm((1.0 - share) * km)),
        analyser=q.BasisAnalyser(misalign=E_DET),
        security=q.SymmetryBound(f=F_REC),
    ).run()


class Model(Question):
    """
    The photon-pair forward model against an independent enumeration.
    """

    def test_exact(self):
        """
        Gain and error rate reproduce a four-detector enumeration at zero background, which fixes
        Ma-Fung-Lo Eq. (10)'s denominator exponent -- the squared reading misses by 7% at lam = 0.1.
        """
        for lam in (1e-4, 1e-2, 0.05, 0.3, 1.0, 3.0):
            for eta in (0.02, 0.145, 0.6, 1.0):
                want = exact(lam, eta, eta)
                gain = pr.gain(lam, eta, eta)
                err = pr.qber(lam, eta, eta)
                self.assertClose(gain, want[0], atol=1e-14, msg=f"gain at lam={lam}, eta={eta}")

                # E is E*Q over a gain as small as 1e-7: the division amplifies round-off from
                # 3.8e-16 to 8.2e-10, which is the ratio of the two tolerances.
                self.assertClose(
                    err * gain,
                    want[1] * want[0],
                    atol=1e-14,
                    msg=f"E*Q at lam={lam}, eta={eta}",
                )
                self.assertClose(err, want[1], atol=1e-8, msg=f"qber at lam={lam}, eta={eta}")

    def test_asymmetric(self):
        """
        Gain and error rate agree with the enumeration at three unequal arm pairs.
        """
        for eta_a, eta_b in ((0.9, 0.01), (0.145, 0.02), (0.3, 0.7)):
            want = exact(0.08, eta_a, eta_b)
            self.assertClose(
                pr.gain(0.08, eta_a, eta_b),
                want[0],
                atol=1e-12,
                msg=f"gain at {eta_a}, {eta_b}",
            )
            self.assertClose(
                pr.qber(0.08, eta_a, eta_b),
                want[1],
                atol=1e-12,
                msg=f"qber at {eta_a}, {eta_b}",
            )

    def test_limits(self):
        """
        A vanishing brightness leaves the misalignment as the error rate, and a link with no
        coincidences reads the chance rate 1/2.
        """

        self.assertClose(
            pr.qber(1e-9, DET, DET, 0.0, 0.0, E_DET),
            E_DET,
            atol=1e-6,
            msg="a single pair errs only by misalignment",
        )
        self.assertClose(
            pr.qber(0.0, DET, DET, 0.0, 0.0, E_DET),
            0.5,
            atol=0.0,
            msg="no coincidences: not the chance rate",
        )
        self.assertClose(pr.gain(0.0, DET, DET), 0.0, atol=0.0, msg="no pairs, no coincidences")

    def test_correlated(self):
        """
        The correlated fraction is the closed form 2*(1/c - 1/(a*b)) and never exceeds the gain.
        """
        for lam in (1e-3, 0.1, 1.0, 5.0):
            a = 1.0 + DET * lam
            b = 1.0 + DET * lam
            c = 1.0 + lam * (2.0 * DET - DET * DET)
            got = _core.pair_correlated(lam, DET, DET)
            self.assertClose(got, 2.0 * (1.0 / c - 1.0 / (a * b)), atol=1e-14, msg=f"lam={lam}")
            self.assertGreaterEqual(pr.gain(lam, DET, DET), got, msg=f"correlated exceeds gain at {lam}")

    def test_background(self):
        """
        Ma-Fung-Lo carry no background factor on the correlated term, so the QBER is exact at
        Y0 = 0 and rises with the background.
        """
        for y0, want in ((1e-6, 3.5e-7), (1e-3, 3.5e-4), (1e-2, 3.5e-3)):
            got = pr.qber(0.05, DET, DET, y0, y0, 0.0)
            base = exact(0.05, DET, DET)[1]
            self.assertLess(
                abs(got - base) / max(want, 1e-12),
                40.0,
                msg=f"background {y0} moved the QBER by {abs(got - base):g}",
            )
        self.assertGreater(
            pr.qber(0.05, DET, DET, 0.2, 0.2, 0.0),
            pr.qber(0.05, DET, DET, 0.0, 0.0, 0.0),
            msg="background lowered the QBER",
        )

    def test_scaling(self):
        """
        The optimised rate is linear in the end-to-end transmittance, log-log slope 1.0 within 0.05
        over 6 to 25 dB.
        """
        xs = []
        ys = []
        for db in range(6, 26):
            eta = DET * 10.0 ** (-db / 20.0)
            rate = sweep(eta, eta)
            self.assertGreater(rate, 0.0, msg=f"no key at {db} dB end to end")

            xs.append(-db / 10.0 * math.log(10.0))
            ys.append(math.log(rate))

        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = num / sum((x - mx) ** 2 for x in xs)
        self.assertClose(slope, 1.0, atol=0.05, msg=f"log-log slope {slope}")

    def test_brightness(self):
        """
        The optimal mean pair number falls over 25 dB but stays within half its back-to-back value,
        which is Ma-Fung-Lo's stated result.
        """
        mus = []
        for db in (0, 5, 10, 15, 20, 25):
            eta = DET * 10.0 ** (-db / 10.0)
            mus.append(2.0 * pr.optimum(eta, eta, Y_DARK, Y_DARK, E_DET, F_REC, 0.5)[0])
        self.assertMonotone(mus, rising=False, msg="optimal brightness must not rise")
        self.assertGreater(
            mus[-1] / mus[0],
            0.5,
            msg=f"optimal mu tracked the transmittance: {mus}",
        )

    def test_forecast(self):
        """
        The forward CHSH model reaches Ekert's Eq. (4) at his own settings, its correlations, S and
        visibility bridge being Eqs. (2), (3) and pr.chsh.
        """
        full = pr.predict(1.0)
        self.assertClose(full.s, -TSIRELSON, atol=1e-15, msg=f"Eq. (4) reads {full.s}")
        self.assertLess(full.s, 0.0, msg="Eq. (2) carries the singlet's minus sign")

        for v in (0.0, 0.3, 0.918, 1.0):
            got = pr.predict(v)
            a1, a3, b1, b3 = got.angles
            legs = ((a1, b1), (a1, b3), (a3, b1), (a3, b3))
            self.assertClose(abs(got.s), pr.chsh(v), atol=1e-15, msg=f"magnitude at v={v}")
            self.assertEqual(
                got.correlations,
                tuple(pr.correlate(a, b, v) for a, b in legs),
                msg=f"correlations not Eq. (2) at v={v}",
            )
            self.assertClose(
                got.s,
                sum(got.correlations) - 2.0 * got.correlations[1],
                atol=1e-15,
                msg=f"S not Eq. (3) at v={v}",
            )
            self.assertClose(
                1.0 - 2.0 * got.qber,
                v,
                atol=1e-15,
                msg=f"visibility bridge at v={v}",
            )


class Anchors(Question):
    """
    Published numbers: Ma, Fung & Lo 2007 and Acin et al. 2007.
    """

    def test_squeezing(self):
        """
        The source brightness is the two-mode squeezed vacuum's sinh^2(r).
        """
        for r in (0.2, 0.5, 1.0, 1.5):
            half = gaussian.Epr(r).keep([0])
            nbar = 0.5 * (half.cov[0, 0] + half.cov[1, 1]) - 0.5
            self.assertClose(
                nbar,
                math.sinh(r) ** 2,
                atol=1e-12,
                msg=f"Epr({r}) marginal nbar = {nbar}",
            )

    def test_shape(self):
        """
        The BBM92 rate is the Shor-Preskill shape cow_rate ships, kept a separate engine.
        """
        for gain, err in ((1e-3, 0.01), (0.02, 0.05), (0.5, 0.08)):
            got = _core.pair_rate(gain, err, err, F_REC, 1.0)
            self.assertClose(
                got,
                _core.cow_rate(gain, err, err, F_REC),
                atol=1e-15,
                msg=f"shape drift at gain={gain}, e={err}",
            )

    def test_table(self):
        """
        Ma-Fung-Lo's Table 1 hardware distils back to back at QBER 0.0574, optimal mu 0.1176 and
        gain 2.777e-3.
        """
        lam, gain, err, rate = pr.optimum(DET, DET, Y_DARK, Y_DARK, E_DET, F_REC, 0.5)
        self.assertGreater(rate, 0.0, msg="Table 1 hardware must distil")
        self.assertClose(err, 0.0574, atol=5e-4, msg=f"QBER at optimum {err}")
        self.assertClose(2.0 * lam, 0.1176, atol=5e-3, msg=f"optimal mu {2 * lam}")
        self.assertClose(gain, 2.777e-3, atol=1e-5, msg=f"gain at optimum {gain}")

    def test_holevo(self):
        """
        Acin et al.'s Holevo bound is exactly 1 at no violation, exactly 0 at Tsirelson's bound, and
        falls monotonically between.
        """

        self.assertClose(pr.holevo(2.0), 1.0, atol=0.0, msg="no violation, no secrecy")
        self.assertClose(pr.holevo(TSIRELSON), 0.0, atol=1e-15, msg="maximal violation excludes Eve")
        self.assertClose(pr.holevo(1.0), 1.0, atol=0.0, msg="a local correlation is worth nothing")

        chis = [pr.holevo(2.0 + 0.05 * i) for i in range(17)]
        self.assertMonotone(chis, rising=False, msg="Eve must lose as S rises")

    def test_threshold(self):
        """
        The device-independent QBER threshold under a depolarised singlet is 7.1%, reproducing Acin
        et al. 2007 to three figures.
        """

        def margin(qber):
            s = TSIRELSON * (1.0 - 2.0 * qber)

            return 1.0 - _core.pair_holevo(s) - h2(qber)

        got = bisect(margin, 1e-6, 0.15)
        self.assertClose(got, DI_QBER, atol=5e-4, msg=f"threshold {got}")


class Position(Question):
    """
    Where the source sits, at constant total fibre.
    """

    def test_position(self):
        """
        Reach is 308.1 km from the midpoint, 241.5 from the quarter point and 183.0 from a party's
        lab, at equal total fibre.
        """
        mid = reach(0.5)
        quarter = reach(0.25)
        end = reach(0.0)
        self.assertClose(mid, 308.1, atol=0.5, msg=f"midpoint reach {mid}")
        self.assertClose(quarter, 241.5, atol=0.5, msg=f"quarter reach {quarter}")
        self.assertClose(end, 183.0, atol=0.5, msg=f"endpoint reach {end}")
        self.assertClose(mid / end, 1.683, atol=5e-3, msg=f"midpoint gain in reach {mid / end}")
        self.assertMonotone([end, quarter, mid], msg="reach did not extend inward")

    def test_symmetric(self):
        """
        Swapping the two arms leaves the rate unchanged.
        """
        for km in (50.0, 150.0, 250.0):
            self.assertClose(
                build(0.25, km).key_rate,
                build(0.75, km).key_rate,
                atol=1e-15,
                msg=f"arm swap moved the rate at {km} km",
            )

    def test_product(self):
        """
        At fixed brightness and no background the gain is set by the product of the two arms alone,
        to 0.5%.
        """
        for km in (100.0, 200.0):
            mid = build(0.5, km, brightness=0.01)
            end = build(0.0, km, brightness=0.01)
            clean = (
                pr.gain(0.01, mid.eta_a, mid.eta_b),
                pr.gain(0.01, end.eta_a, end.eta_b),
            )
            self.assertClose(
                mid.eta_a * mid.eta_b,
                end.eta_a * end.eta_b,
                atol=1e-15,
                msg="arm products differ at equal fibre",
            )
            self.assertClose(
                clean[0],
                clean[1],
                atol=5e-3 * clean[0],
                msg=f"gain moved over 0.5% at {km} km",
            )

    def test_darkwall(self):
        """
        At 200 km the same total fibre gives 2.86% QBER from the middle and 16.10% from a party's
        lab, and only the first distils.
        """
        mid = build(0.5, 200.0, brightness=0.01)
        end = build(0.0, 200.0, brightness=0.01)
        self.assertClose(mid.qber, 0.0286, atol=5e-4, msg=f"midpoint QBER {mid.qber}")
        self.assertClose(end.qber, 0.1610, atol=5e-4, msg=f"endpoint QBER {end.qber}")
        self.assertGreater(mid.key_rate, 0.0, msg=f"midpoint rate {mid.key_rate} at 200 km")
        self.assertClose(end.key_rate, 0.0, atol=0.0, msg="a lab source must not, at the same fibre")

    def test_second(self):
        """
        per_second is the per-pulse rate times the pump repetition rate.
        """
        res = build(0.5, 50.0)
        self.assertClose(
            res.per_second,
            res.key_rate * PUMP,
            atol=1e-9,
            msg=f"per_second {res.per_second}",
        )


class Security(Guarded):
    """
    The device-independence position, in shipped output rather than in prose.
    """

    def test_diagnostic(self):
        """
        The CHSH value a run reports is labelled a diagnostic, and the plan's device_independent row
        is False.
        """
        info = build(0.5, 50.0).explain
        self.assertEqual(info["chsh"]["label"], "diagnostic", msg="CHSH must not read as derived")
        self.assertEqual(
            info["device_independent"]["value"],
            False,
            msg="the plan must deny device independence",
        )
        self.assertIn(
            "not device independent",
            info["trust"],
            msg=f"trust line: {info['trust']!r}",
        )

    def test_absent(self):
        """
        No device-independent key rate is exposed anywhere, a coincidence-counted click layer's
        post-selection on detection being the fair-sampling assumption.
        """
        for name in ("device", "di_rate", "ekert", "e91", "violation"):
            self.assertFalse(
                hasattr(pr, name),
                msg=f"qkd.pairs.{name} exists",
            )

        for name in ("DI_ETA_PARTIAL", "DI_ETA_SINGLET"):
            self.assertGreater(
                getattr(pr, name) / q.ClickDetector().eta,
                4.0,
                msg=f"{name} against the default eta",
            )
        self.assertGreater(
            pr.DI_ETA_SINGLET,
            pr.DI_ETA_PARTIAL,
            msg="singlet threshold below partial",
        )
        self.assertFalse(
            hasattr(pr, "DI_EFFICIENCY"),
            msg="DI_EFFICIENCY is back",
        )

    def test_network(self):
        """
        q.PairLink offers run() and explain(), and q.Network refuses it by name.
        """
        plan = q.PairLink(
            source=q.PairSource(brightness=0.01),
            detectors=(q.ClickDetector(), q.ClickDetector()),
            channels=(q.Fiber(length=10.0), q.Fiber(length=10.0)),
        )

        nodes = {
            "a": q.Node(),
            "b": q.Node(),
        }
        self.assertTrue(
            callable(plan.run) and callable(plan.explain),
            msg="no run()/explain() on PairLink",
        )
        self.assertFails(
            ValueError,
            "PairLink",
            lambda: q.Network(
                nodes=nodes,
                edges=[q.Hop(ends=("a", "b"), link=plan, clock=1e8)],
            ).run(),
            msg="refusal does not name PairLink",
        )

    def test_tsirelson(self):
        """
        A CHSH value past Tsirelson's bound is refused, the refusal naming the detection efficiency
        a real claim needs.
        """

        self.assertFails(
            ValueError,
            "Tsirelson",
            _core.pair_holevo,
            2.9,
            msg="past Tsirelson must raise",
        )
        self.assertFails(
            ValueError,
            "detection efficiency",
            _core.pair_holevo,
            3.0,
            msg="the refusal must name the loophole cost",
        )

    def test_pumping(self):
        """
        A continuous-wave pump is refused, the pulsed gain carrying no accidental-coincidence term.
        """

        self.assertFails(
            NotImplementedError,
            "accidental",
            lambda: q.PairSource(pumping="continuous"),
            msg="continuous pumping must raise",
        )

    def test_finite(self):
        """
        q.FiniteSize on a pair link is refused, the refusal naming coincidences.
        """

        self.assertFails(
            NotImplementedError,
            "COINCIDENCES",
            lambda: pr.PairLink(
                source=q.PairSource(brightness=0.01),
                detectors=(q.ClickDetector(), q.ClickDetector()),
                channels=(q.Fiber(length=25.0), q.Fiber(length=25.0)),
                security=q.FiniteSize(),
            ).run(),
            msg="FiniteSize must be refused",
        )

    def test_timing(self):
        """
        Detector dead time and jitter are refused rather than approximated.
        """

        def run(det):
            return pr.PairLink(
                source=q.PairSource(brightness=0.01),
                detectors=(det, q.ClickDetector()),
                channels=(q.Fiber(length=25.0), q.Fiber(length=25.0)),
            ).run()

        self.assertFails(
            ValueError,
            "COINCIDENCE",
            lambda: run(q.ClickDetector(dead_time=1e-6)),
            msg="dead time must be refused",
        )
        self.assertFails(
            ValueError,
            "coincidence window",
            lambda: run(q.ClickDetector(jitter=1e-10)),
            msg="detector jitter must be refused",
        )

    def test_slots(self):
        """
        Out-of-domain arguments raise rather than returning a wrong number.
        """

        self.assertSlots(
            _core.pair_gain,
            (0.1, 0.2, 0.3, 1e-6, 1e-6),
            (
                (0, "lam", (-1.0, float("nan"))),
                (1, "eta_a", (-0.1, 1.5)),
                (3, "y0_a", (1.0, -0.1)),
            ),
            msg="pair_gain",
        )
        self.assertSlots(
            _core.pair_rate,
            (0.1, 0.02, 0.02, 1.22, 0.5),
            (
                (0, "gain", (-0.1, 1.5)),
                (1, "e_bit", (0.6, -0.1)),
                (3, "f_ec", (0.5,)),
            ),
            msg="pair_rate",
        )

    def test_background(self):
        """
        The receiver background the pair layer derives is the two-gate composition q.Link
        performs.
        """
        for dark in (1e-6, 1e-4, 1e-2):
            self.assertClose(
                pr.background(q.ClickDetector(dark=dark)),
                background(dark),
                atol=0.0,
                msg=f"background drift at dark={dark}",
            )

    def test_forward(self):
        """
        predict() and run() report one modelled violation and one disturbance, and every rate name
        on the Forecast raises.
        """
        link = pr.PairLink(
            source=q.PairSource(brightness=0.01, rate=PUMP),
            detectors=(q.ClickDetector(eta=DET, dark=DARK),) * 2,
            channels=(arm(25.0), arm(25.0)),
            analyser=q.BasisAnalyser(misalign=E_DET),
        )
        seen = link.predict()
        res = link.run()
        self.assertClose(
            abs(seen.s),
            res.chsh,
            atol=1e-14,
            msg=f"predict {abs(seen.s)} against run {res.chsh}",
        )
        self.assertClose(seen.qber, res.qber, atol=1e-14, msg="and one modelled disturbance")

        for name in ("key_rate", "rate", "key", "bound", "secure", "safe", "margin"):
            self.assertFails(
                AttributeError,
                "bounds nothing",
                lambda n=name: getattr(seen, n),
                msg=f"Forecast.{name} answered",
            )
        self.assertFails(
            ValueError,
            "quadruple",
            lambda: pr.predict(1.0, angles=pr.settings()),
            msg="six settings accepted as four",
        )

    def test_launder(self):
        """
        A Forecast handed to q.ViolationBound(source="measured") is refused as circular, while a
        bare magnitude still prices.
        """
        seen = pr.predict(0.95)

        def run(s):
            return pr.PairLink(
                source=q.PairSource(brightness=0.01),
                detectors=(q.ClickDetector(),) * 2,
                channels=(q.Fiber(length=10.0),) * 2,
                security=q.ViolationBound(s=s, source="measured"),
            ).run()

        for needle in ("MEASURED", "circularity", "modelled", "q.SymmetryBound"):
            self.assertFails(
                ValueError,
                needle,
                lambda: run(seen),
                msg=f"the refusal must name {needle}",
            )
        self.assertGreaterEqual(
            run(abs(seen.s)).key_rate,
            0.0,
            msg="a bare magnitude was refused",
        )

    def test_attack(self):
        """
        Both plans carry an attack row after security: collective on the violation branch, unstated
        on the basis-symmetry one.
        """
        plain = build(0.5, 50.0).explain
        priced = pr.PairLink(
            source=q.PairSource(brightness=0.01, rate=PUMP),
            detectors=(q.ClickDetector(eta=DET, dark=DARK),) * 2,
            channels=(arm(25.0), arm(25.0)),
            security=q.ViolationBound(s=2.7, source="measured"),
        ).explain()
        for info in (plain, priced):
            keys = list(info)
            self.assertIsInstance(info["attack"], str, msg=f"attack row {info['attack']!r}")
            self.assertEqual(
                keys[keys.index("security") + 1],
                "attack",
                msg="attack row does not follow security",
            )
        self.assertIn("unstated", plain["attack"], msg="basis symmetry states no class")
        self.assertIn("pair_rate", plain["attack"], msg="and names the engine that states none")
        self.assertNotIn(
            "collective",
            plain["attack"],
            msg=f"basis-symmetry row: {plain['attack']!r}",
        )
        self.assertIn("collective", priced["attack"], msg="the CHSH price is a collective bound")
        self.assertIn(
            "entropy accumulation",
            priced["attack"],
            msg="entropy accumulation unnamed",
        )


class Finite(Guarded):
    """
    The entanglement-based finite-key length, and what a coincidence-counted link is refused for.
    """

    def test_published(self):
        """
        At the four block lengths Tomamichel and Leverrier's Figure 7 marks, pair_length and
        pair_secpar return the length and the epsilon their own code printed.
        """
        for delta, m, k, ell, nu, total in TL_POINTS:
            tag, leak, eps_pa = leverrier(m, delta, k)
            self.assertEqual(
                _core.pair_length(m, k, nu, delta, TL_CBAR, leak, tag, eps_pa),
                float(ell),
                msg=f"delta={delta} against published {ell}",
            )
            got = _core.pair_secpar(m, k, nu, delta, TL_CBAR, leak, tag, ell)
            self.assertClose(
                got[3] / total,
                1.0,
                atol=1e-12,
                msg=f"delta={delta} eps against {total}",
            )
            self.assertLessEqual(got[3], TL_EPS, msg=f"delta={delta} must meet its own target")

    def test_optimised(self):
        """
        pair_finite at those points is never shorter than the published length and at most a part in
        a thousand longer.
        """
        for delta, m, _, ell, _, _ in TL_POINTS:
            got = _core.pair_finite(m, delta, TL_FEC, TL_CBAR, TL_EPS, "deterministic")
            self.assertGreaterEqual(
                got[0],
                float(ell),
                msg=f"delta={delta}: {got[0]} below {ell}",
            )
            self.assertLess(
                got[0] / ell,
                1.001,
                msg=f"delta={delta}: {got[0]} over {ell}",
            )
            self.assertClose(
                got[1],
                got[0] / m,
                msg=f"delta={delta}: rate is not ell/m",
            )

    def test_budget(self):
        """
        Every length pair_finite returns costs no more than the epsilon it was asked for, read back
        through pair_secpar.
        """
        for delta in (0.01, 0.05):
            for m in (1e5, 1e7, 1e9):
                ell, _, k, nu, tag = _core.pair_finite(m, delta, TL_FEC, TL_CBAR, TL_EPS, "deterministic")
                leak = TL_FEC * (m - k) * h2(delta)
                total = _core.pair_secpar(m, k, nu, delta, TL_CBAR, leak, tag, ell)[3]
                self.assertLessEqual(
                    total,
                    TL_EPS,
                    msg=f"delta={delta} m={m:.0e} spends {total}",
                )

    def test_asymptote(self):
        """
        The rate per sifted round rises with the block towards log2(1/cbar) - 2h(delta), reaching
        98% of it at 1e12.
        """
        for delta in (0.01, 0.05):
            limit = 1.0 - 2.0 * h2(delta)
            rates = [
                _core.pair_finite(m, delta, 1.0, TL_CBAR, TL_EPS, "deterministic")[1]
                for m in (1e5, 1e6, 1e8, 1e10, 1e12)
            ]
            self.assertMonotone(rates, msg=f"delta={delta}: {rates}")
            self.assertLess(rates[-1], limit, msg=f"delta={delta}: {rates[-1]} at {limit}")
            self.assertGreater(
                rates[-1] / limit,
                0.98,
                msg=f"delta={delta}: {rates[-1]} of {limit}",
            )

    def test_marker(self):
        """
        At each block length Figure 7 marks, the length per sifted round has just reached half the
        asymptotic 1 - 2h(delta).
        """
        for delta, m, _, _, _, _ in TL_POINTS:
            rate = _core.pair_finite(m, delta, TL_FEC, TL_CBAR, TL_EPS, "deterministic")[1]
            half = 0.5 * (1.0 - 2.0 * h2(delta))
            self.assertGreaterEqual(rate, half, msg=f"delta={delta}: {rate} has not reached {half}")
            self.assertLess(rate, 1.02 * half, msg=f"delta={delta}: {rate} is well past {half}")

    def test_vanishes(self):
        """
        The length falls to nothing exactly where log2(1/cbar) meets 2h(delta).
        """
        dead = 2.0 ** (-2.0 * h2(0.02))
        self.assertGreater(
            _core.pair_finite(1e8, 0.02, 1.0, 0.98 * dead, TL_EPS, "deterministic")[0],
            0.0,
            msg="no key just inside the overlap",
        )
        self.assertClose(
            _core.pair_finite(1e8, 0.02, 1.0, 1.02 * dead, TL_EPS, "deterministic")[0],
            0.0,
            atol=0.0,
            msg="and just outside it there is none",
        )

    def test_inverse(self):
        """
        pair_length and pair_secpar invert each other to within the one bit the floor gives away.
        """
        m, k, nu, delta = 1e6, 6.0e4, 0.02, 0.01
        leak = TL_FEC * (m - k) * h2(delta)
        for eps_pa in (1e-13, 1e-11, 3.3e-11):
            ell = _core.pair_length(m, k, nu, delta, TL_CBAR, leak, 44.0, eps_pa)
            back = _core.pair_secpar(m, k, nu, delta, TL_CBAR, leak, 44.0, ell)[1]
            self.assertLessEqual(back, eps_pa, msg=f"flooring the length can only tighten {eps_pa}")
            self.assertGreater(
                back,
                eps_pa / math.sqrt(2.0),
                msg=f"{back} below {eps_pa}/sqrt(2)",
            )

    def test_terms(self):
        """
        The parameter-estimation and correctness terms are Eq. (58)'s exponential and Theorem 2's
        2^-t, and the total is their sum.
        """
        m, k, nu, delta = 5e5, 3.0e4, 0.03, 0.02
        leak = TL_FEC * (m - k) * h2(delta)
        pe, pa, ec, total = _core.pair_secpar(m, k, nu, delta, TL_CBAR, leak, 40.0, 1000.0)
        want = 2.0 * math.exp(-(m - k) * k * k * nu * nu / (m * (k + 1.0)))
        self.assertClose(pe / want, 1.0, msg="eps_pe is Eq. (58)'s exponential")
        self.assertClose(ec, 2.0**-40.0, msg="eps_ec is Theorem 2's 2^-t")
        self.assertClose(total, pe + pa + ec, msg=f"total {total} against {pe + pa + ec}")

    def test_cbar(self):
        """
        Alice's complementarity enters the length as log2(1/cbar) per key round, and cbar = 1 is
        refused.
        """
        m, k, nu, delta = 1e6, 5.0e4, 0.02, 0.01
        args = (m, k, nu, delta)
        pair = [_core.pair_length(*args, cbar, 0.0, 40.0, 1e-11) for cbar in (0.5, 0.25)]
        self.assertClose(
            pair[1] - pair[0],
            (m - k) * 1.0,
            atol=1.0,
            msg=f"halving cbar bought {pair[1] - pair[0]}",
        )
        self.assertFails(
            ValueError,
            "cbar must be in (0, 1)",
            _core.pair_length,
            m,
            k,
            nu,
            delta,
            1.0,
            0.0,
            40.0,
            1e-11,
            msg="cbar = 1 accepted",
        )

    def test_coincidence(self):
        """
        A coincidence-counted pair link is refused by name, the message naming three missing pieces
        and four papers.
        """

        self.assertFails(
            NotImplementedError,
            'detection="coincidence" is refused',
            _core.pair_finite,
            1e6,
            0.01,
            TL_FEC,
            TL_CBAR,
            TL_EPS,
            "coincidence",
            msg="a coincidence link returned a length",
        )
        text = refusal()
        for needle in ("DETERMINISTIC DETECTION", "Eq. (108)", "Lemma 16", "Corollary 15"):
            self.assertIn(needle, text, msg=f"the refusal must name {needle}")
        cites = (
            "arXiv:1506.08458",
            "Phys. Rev. A 89, 012325",
            "New J. Phys. 18, 053001",
            "arXiv:2607.10659",
        )
        for needle in cites:
            self.assertIn(needle, text, msg=f"the refusal must cite {needle}")
        self.assertIn("pair_rate", text, msg="pair_rate unnamed")

    def test_squasher(self):
        """
        The refusal does not blame the squashing model, naming the three Gittsovich theorems, the
        vacuum flag and Eq. (134) instead.
        """
        text = refusal()
        self.assertIn(
            "SQUASHING MODEL IS NOT THE BLOCKER",
            text,
            msg="the squasher is blamed",
        )
        for needle in ("Theorem 10", "Theorem 13", "Theorem 14"):
            self.assertIn(needle, text, msg=f"{needle} unnamed")
        self.assertIn(
            "vacuum flag",
            text,
            msg="vacuum flag unnamed",
        )
        self.assertIn(
            "Eq. (134)",
            text,
            msg="Eq. (134) unnamed",
        )
        self.assertIn(
            "ASYMPTOTIC",
            text,
            msg="ASYMPTOTIC unnamed",
        )

    def test_asymmetry(self):
        """
        The refusal states the two-sided gap as a requirement, naming Alice's side as the half that
        does not transfer.
        """
        text = refusal()
        self.assertIn("ONE-SIDED", text, msg="Part II's split covers one receiver")
        self.assertIn(
            "ALICE'S",
            text,
            msg="ALICE'S unnamed",
        )
        self.assertIn("Lemma 17", text, msg="Lemma 17 unnamed")
        self.assertIn(
            "Eq. (139)",
            text,
            msg="Eq. (139) unnamed",
        )
        self.assertIn("Eq. (118)", text, msg="Eq. (118) unnamed")
        for needle in ("Eqs. (121)", "(122)"):
            self.assertIn(needle, text, msg=f"{needle} unnamed")
        self.assertIn(
            "arXiv:0804.0891",
            text,
            msg="arXiv:0804.0891 unnamed",
        )
        self.assertIn(
            "ASYMPTOTICALLY",
            text,
            msg="ASYMPTOTICALLY unnamed",
        )

    def test_budget_domain(self):
        """
        A security target at or above one half is refused, at pair_finite and at pair_length's
        eps_pa.
        """

        self.assertFails(
            ValueError,
            "eps must be < 1/2",
            _core.pair_finite,
            1e6,
            0.01,
            TL_FEC,
            TL_CBAR,
            0.6,
            "deterministic",
            msg="eps = 0.6 accepted",
        )
        self.assertFails(
            ValueError,
            "eps_pa must be",
            _core.pair_length,
            1e6,
            5.0e4,
            0.02,
            0.01,
            TL_CBAR,
            0.0,
            40.0,
            0.5,
            msg="eps_pa = 0.5 accepted",
        )

    def test_short(self):
        """
        A block too short for its security parameter raises rather than returning zero, and an
        unknown detection model is refused.
        """

        self.assertFails(
            ValueError,
            "too short for the security parameter",
            _core.pair_finite,
            50.0,
            0.01,
            TL_FEC,
            TL_CBAR,
            TL_EPS,
            "deterministic",
            msg="fifty rounds returned a length",
        )
        self.assertFails(
            ValueError,
            "unknown detection",
            _core.pair_finite,
            1e6,
            0.01,
            TL_FEC,
            TL_CBAR,
            TL_EPS,
            "heralded",
            msg="heralded detection accepted",
        )

    def test_fuzz(self):
        """
        Across 150 random blocks, error rates, complementarities and targets, no length pair_finite
        returns overspends its epsilon.
        """
        r = rng(11)
        ran, zero = 0, 0
        for _ in range(150):
            m = loguni(r, 1e3, 1e12)
            delta = r.uniform(0.0, 0.11)
            f_ec = r.uniform(1.0, 1.3)
            cbar = r.uniform(0.2, 0.9)
            eps = loguni(r, 1e-15, 1e-3)
            try:
                ell, _, k, nu, tag = _core.pair_finite(m, delta, f_ec, cbar, eps, "deterministic")
            except ValueError:
                continue

            ran += 1
            if ell <= 0.0:
                zero += 1
                continue

            leak = f_ec * (m - k) * h2(delta)
            total = _core.pair_secpar(m, k, nu, delta, cbar, leak, tag, ell)[3]
            self.assertLessEqual(
                total,
                eps * (1.0 + 1e-12),
                msg=f"m={m:.3e} delta={delta:.4f} cbar={cbar:.3f} eps={eps:.3e} overspends",
            )
        self.assertGreater(ran, 110, msg=f"{ran} of 150 draws ran")
        self.assertGreater(ran - zero, 40, msg=f"{ran - zero} of {ran} keyed")

    def test_slots(self):
        """
        Out-of-domain arguments to the finite-key entry points raise rather than returning a
        wrong number.
        """

        self.assertSlots(
            _core.pair_length,
            (1e6, 5.0e4, 0.02, 0.01, TL_CBAR, 0.0, 40.0, 1e-11),
            (
                (1, "k must be < m", (1e6, 2e6)),
                (2, "nu must be in", (0.0, 0.6)),
                (3, "delta", (0.6, -0.1)),
                (4, "cbar must be in", (0.0, 1.5)),
                (5, "leak", (-1.0,)),
                (6, "t must be", (0.0, -1.0)),
                (7, "eps_pa must be", (0.6, 0.0)),
            ),
            msg="pair_length",
        )
        self.assertSlots(
            _core.pair_secpar,
            (1e6, 5.0e4, 0.02, 0.01, TL_CBAR, 0.0, 40.0, 1000.0),
            (
                (0, "m must be", (0.0, -1.0)),
                (7, "ell", (-1.0,)),
            ),
            msg="pair_secpar",
        )


if __name__ == "__main__":
    rc = Exam(
        "PairModel",
        "Photon-pair gain and error rate against an independent enumeration",
        "pairs_model.md",
    ).run(load(Model))
    rc |= Exam(
        "PairAnchors",
        "Published entanglement-based numbers: Ma-Fung-Lo 2007 and Acin 2007",
        "pairs_anchors.md",
    ).run(load(Anchors))
    rc |= Exam(
        "PairPosition",
        "Where the source sits, at constant total fibre",
        "pairs_position.md",
    ).run(load(Position))
    rc |= Exam(
        "PairFinite",
        "The entanglement-based finite-key length of Tomamichel & Leverrier 2017",
        "pairs_finite.md",
    ).run(load(Finite))
    rc |= Exam(
        "PairSecurity",
        "The device-independence position and what a pair link refuses",
        "pairs_security.md",
    ).run(load(Security))
    sys.exit(rc)
