import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.forms import bisect, h2
from qkd import _core

# Tsirelson's bound. test_shared pins the engine's spelling in pairs.rs as the
# same f64, not a second one that agrees today.
TSIRELSON = 2.0 * math.sqrt(2.0)

# Acin, Brunner, Gisin, Massar, Pironio & Scarani, Phys. Rev. Lett. 98, 230501
# (2007): 7.1% in the device-independent scenario, "to be compared to" the
# well-known 11% their Eq. (12) discussion reaches under the standard
# characterised-device assumption.
DI_QBER = 0.071
BBM92_QBER = 0.110

# (2 - sqrt(2))/4, the error rate at which a depolarised singlet stops
# violating the CHSH inequality.
LOST_QBER = 0.14644660940672627

# Lim, Portmann, Tomamichel, Renner & Gisin, Phys. Rev. X 3, 031006 (2013),
# arXiv:1208.0023, Fig. 2: the secret fraction of their Eq. (2) against the
# efficiency of Charlie's operation, at Q_tol = 1% and S_tol = 2 sqrt(2) V, for
# V = 0.999 and V = 0.99 "from left to right". These are the two zero crossings,
# computed from Eq. (2) here rather than read off the figure.
ETA_999 = 0.1133521877
ETA_99 = 0.3544212507

# Ekert, Phys. Rev. Lett. 67, 661 (1991), the paragraph before Eq. (1):
# Alice's three analyser angles and Bob's three, radians from the vertical
# x axis. Written out here so the engine's tuple is checked, not echoed.
ALICE = (0.0, 0.25 * math.pi, 0.5 * math.pi)
BOB = (0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi)


def holevo(s):
    """
    Eve's information against a CHSH value: Acin et al. 2007 Eq. (4) reads the violation as the
    Bell-diagonal state of weights (1 +/- C)/2 with C = sqrt((S/2)^2 - 1), and their Eq. (3) is
    that state's entropy. Transcribed from the paper, not the engine.
    """
    if s <= 2.0:
        return 1.0

    weight = math.sqrt((0.5 * s) ** 2 - 1.0)

    return h2(0.5 * (1.0 + weight))


def werner(qber):
    """
    The CHSH value a depolarised singlet of that bit error rate shows, S = 2 sqrt(2) (1 - 2Q),
    Acin et al. 2007.
    """

    return TSIRELSON * (1.0 - 2.0 * qber)


def pironio(qber, chsh):
    """
    Pironio et al., New J. Phys. 11, 045021 (2009), Eq. (12) literally: r >= 1 - h(Q) - h((1 +
    sqrt((S/2)^2 - 1))/2), bits per key round.
    """

    return 1.0 - h2(qber) - holevo(chsh)


def correlate(a, b, v):
    """
    Ekert Eq. (2) at visibility v, E(a, b) = -v cos(a - b).
    """

    return -v * math.cos(a - b)


def overlap(s):
    """
    Lim, Portmann, Tomamichel, Renner & Gisin, Phys. Rev. X 3, 031006 (2013), Lemma 6: the
    effective overlap of Alice's two measurements a CHSH value certifies, c* <= 1/2 + (S/8)
    sqrt(8 - S^2). Transcribed from the paper.
    """

    return 0.5 + s * math.sqrt(max(0.0, 8.0 - s * s)) / 8.0


def charged(s, eta):
    """
    Their Lemma 7, the post-selection charge on that overlap: c*_X <= 1/2 + (c*_Xtilde -
    1/2)/eta, with eta the surviving fraction.
    """

    return 0.5 + (overlap(s) - 0.5) / eta


def secret(s, eta, qber):
    """
    Their Eq. (2), the asymptotic secret fraction at the Shannon limit: 1 - log2(1 + (S/(4 eta))
    sqrt(8 - S^2)) - 2h(Q).
    """

    priced = math.log2(1.0 + s * math.sqrt(max(0.0, 8.0 - s * s)) / (4.0 * eta))

    return 1.0 - priced - 2.0 * h2(qber)


class Protocol(Question):
    """
    Ekert 1991's own settings, correlations and CHSH combination.
    """

    def test_settings(self):
        """
        The engine's analyser angles are Ekert's three per party, and the two shared
        orientations are the key-generating ones, at perfect anticorrelation.
        """
        angles = _core.ekert_settings()
        self.assertEqual(angles, ALICE + BOB, msg="the settings are Ekert's six, in his order")

        matched = []
        for i, a in enumerate(ALICE):
            for j, b in enumerate(BOB):
                if abs(a - b) < 1e-15:
                    matched.append((i, j))
        self.assertEqual(matched, [(1, 0), (2, 1)], msg="only (a2, b1) and (a3, b2) coincide")

        for i, j in matched:
            self.assertClose(
                _core.ekert_correlate(ALICE[i], BOB[j], 1.0),
                -1.0,
                atol=0.0,
                msg=f"a shared orientation must be totally anticorrelated, {i} {j}",
            )
        self.assertClose(
            len(matched) / 9.0,
            2.0 / 9.0,
            atol=0.0,
            msg="drawing all six uniformly sifts 2 of 9 setting pairs",
        )

    def test_violation(self):
        """
        Ekert Eq. (3) at those settings reaches his Eq. (4)'s -2 sqrt(2), in his sign convention
        rather than the modern one.
        """
        got = _core.ekert_chsh(ALICE[0], ALICE[2], BOB[0], BOB[2], 1.0)
        self.assertClose(got, -TSIRELSON, atol=1e-15, msg=f"Eq. (4) reads {got}")
        self.assertLess(got, 0.0, msg="Eq. (2) carries the singlet's minus sign")

        for v in (0.0, 0.3, 0.71, 1.0):
            self.assertClose(
                _core.ekert_chsh(ALICE[0], ALICE[2], BOB[0], BOB[2], v),
                -TSIRELSON * v,
                atol=1e-15,
                msg=f"the violation must scale with visibility, v={v}",
            )

    def test_correlate(self):
        """
        The correlation coefficient is Ekert Eq. (2) at every angle and visibility, not only at
        his own settings.
        """
        for a in (0.0, 0.4, 1.3, 2.9, -1.1):
            for b in (0.0, 0.8, 2.2, 5.0):
                for v in (0.0, 0.45, 1.0):
                    self.assertClose(
                        _core.ekert_correlate(a, b, v),
                        correlate(a, b, v),
                        atol=1e-15,
                        msg=f"E({a}, {b}) at v={v}",
                    )

    def test_ceiling(self):
        """
        No choice of the four CHSH angles pushes the modelled violation past Tsirelson's bound.
        """
        step = math.pi / 12.0
        for i in range(24):
            for j in range(24):
                for k in range(6):
                    got = _core.ekert_chsh(i * step, j * step, k * step, 0.0, 1.0)
                    self.assertLess(
                        abs(got),
                        TSIRELSON + 1e-12,
                        msg=f"angles {i} {j} {k} exceeded Tsirelson at {got}",
                    )


class Bound(Question):
    """
    The CHSH-priced rate: the local bound, Tsirelson, and the shape between.
    """

    def test_local(self):
        """
        A CHSH value at or below the local bound yields no key at any error rate, gain or
        reconciliation: Eve's price is the whole one-bit budget.
        """
        for s in (0.0, 1.0, 1.9, 2.0):
            for err in (0.0, 0.01, 0.2):
                self.assertClose(
                    _core.ekert_rate(1.0, err, s, 1.0, 1.0, "measured"),
                    0.0,
                    atol=0.0,
                    msg=f"no violation must mean no key, s={s}, e={err}",
                )
        self.assertClose(holevo(2.0), 1.0, atol=0.0, msg="the local bound costs a full bit")

    def test_maximal(self):
        """
        At Tsirelson's bound the rate is the error-correction leak alone, Eve being excluded
        outright, and a noiseless link keys every round.
        """

        self.assertClose(
            _core.ekert_rate(1.0, 0.0, TSIRELSON, 1.0, 1.0, "measured"),
            1.0,
            atol=0.0,
            msg="maximal violation and no error must key every round",
        )

        for err in (0.01, 0.05, 0.1):
            for f_ec in (1.0, 1.22):
                self.assertClose(
                    _core.ekert_rate(1.0, err, TSIRELSON, f_ec, 1.0, "measured"),
                    1.0 - f_ec * h2(err),
                    atol=1e-15,
                    msg=f"only the leak may remain at Tsirelson, e={err}",
                )

    def test_monotone(self):
        """
        The rate rises with the violation across the whole interval between the local bound and
        Tsirelson's.
        """
        rates = [_core.ekert_rate(1.0, 0.0, 2.0 + 0.05 * i, 1.0, 1.0, "measured") for i in range(17)]
        self.assertMonotone(rates[1:], msg="a larger violation must buy more key")
        self.assertClose(rates[0], 0.0, atol=0.0, msg="the local bound buys none")

    def test_sign(self):
        """
        Ekert's negative CHSH value and the modern positive one price identically: only the
        magnitude carries the violation.
        """
        for s in (2.1, 2.5, TSIRELSON):
            self.assertClose(
                _core.ekert_rate(1.0, 0.02, -s, 1.0, 1.0, "measured"),
                _core.ekert_rate(1.0, 0.02, s, 1.0, 1.0, "measured"),
                atol=0.0,
                msg=f"the two sign conventions disagreed at s={s}",
            )
            self.assertClose(
                _core.ekert_qber(-s),
                _core.ekert_qber(s),
                atol=0.0,
                msg=f"the error bridge disagreed at s={s}",
            )

    def test_units(self):
        """
        Gain and sifting scale the rate linearly, putting it in the photon-pair engine's bits
        per pump pulse.
        """
        base = _core.ekert_rate(1.0, 0.02, 2.7, 1.0, 1.0, "measured")
        for gain in (0.5, 0.01, 1e-4):
            for sift in (1.0, 2.0 / 9.0):
                self.assertClose(
                    _core.ekert_rate(gain, 0.02, 2.7, 1.0, sift, "measured"),
                    gain * sift * base,
                    atol=1e-15,
                    msg=f"units drifted at gain={gain}, sift={sift}",
                )


class Anchors(Question):
    """
    Published numbers: Ekert 1991, Acin et al. 2007 and Pironio et al. 2009.
    """

    def test_shared(self):
        """
        The engine's Tsirelson bound is bit-identical to the photon-pair layer's.
        """

        self.assertClose(_core.pair_chsh(1.0), TSIRELSON, atol=0.0, msg="one Tsirelson, not two")
        self.assertClose(
            _core.ekert_qber(TSIRELSON),
            0.0,
            atol=0.0,
            msg="maximal violation must read zero error, exactly",
        )

    def test_werner(self):
        """
        The engine's error bridge inverts Acin et al.'s S = 2 sqrt(2) (1 - 2Q) exactly, and the
        violation is lost at a bit error rate of 14.645%.
        """
        for qber in (0.0, 0.001, 0.02, 0.071, 0.1464, 0.5):
            self.assertClose(
                _core.ekert_qber(werner(qber)),
                qber,
                atol=1e-15,
                msg=f"the bridge is not its own inverse at Q={qber}",
            )
        self.assertClose(
            _core.ekert_qber(2.0),
            LOST_QBER,
            atol=1e-15,
            msg="the violation is lost at (2 - sqrt(2))/4",
        )
        self.assertClose(
            _core.ekert_qber(2.0),
            0.25 * (2.0 - math.sqrt(2.0)),
            atol=1e-15,
            msg="that number written out",
        )

    def test_pironio(self):
        """
        The rate is Pironio et al. Eq. (12) term for term at unit gain, unit sifting and
        Shannon-limit reconciliation.
        """
        for qber in (0.0, 0.01, 0.03, 0.06, 0.09):
            for s in (2.2, 2.5, 2.75, TSIRELSON):
                self.assertClose(
                    _core.ekert_rate(1.0, qber, s, 1.0, 1.0, "measured"),
                    max(0.0, pironio(qber, s)),
                    atol=1e-14,
                    msg=f"Eq. (12) drift at Q={qber}, S={s}",
                )

    def test_threshold(self):
        """
        The CHSH-priced rate vanishes at a 7.1% bit error rate against the photon-pair engine's
        11%, which is what the state assumption was worth.
        """
        got = _core.ekert_threshold(1.0)
        pair = bisect(lambda x: _core.pair_rate(1.0, x, x, 1.0, 1.0), 1e-9, 0.5)
        self.assertClose(got, DI_QBER, atol=5e-4, msg=f"device-independent {got}")
        self.assertClose(got, 0.0714918, atol=1e-6, msg=f"to seven figures, {got}")
        self.assertClose(pair, BBM92_QBER, atol=5e-4, msg=f"basis symmetry {pair}")
        self.assertLess(got, pair, msg="the CHSH price must be the worse deal")
        self.assertClose(
            _core.ekert_rate(1.0, got, werner(got), 1.0, 1.0, "measured"),
            0.0,
            atol=1e-12,
            msg="the threshold must be where the rate actually vanishes",
        )

    def test_reconcile(self):
        """
        A less efficient reconciliation lowers that threshold monotonically, reaching 6.394% at
        the photon-pair engine's own f = 1.22.
        """
        thresholds = [_core.ekert_threshold(f) for f in (1.0, 1.05, 1.1, 1.22, 1.5)]
        self.assertMonotone(thresholds, rising=False, msg="a worse code must not raise the threshold")
        self.assertClose(thresholds[3], 0.0639374, atol=1e-6, msg=f"f = 1.22 gives {thresholds[3]}")

    def test_price(self):
        """
        Along the depolarised-singlet line the CHSH term charges Eve as though the error were
        doubled, approaching the doubled binary entropy as the error vanishes and reaching 1.69
        times the plain one at the threshold.
        """
        for qber in (1e-6, 1e-5, 1e-4):
            excess = holevo(werner(qber)) / h2(2.0 * qber) - 1.0
            self.assertGreater(excess, 0.0, msg=f"the CHSH price must exceed it at Q={qber}")
            self.assertLess(excess, 2.0 * qber, msg=f"the two must converge as Q -> 0, at {qber}")

        thr = _core.ekert_threshold(1.0)
        ratio = holevo(werner(thr)) / h2(thr)
        self.assertClose(ratio, 1.692, atol=1e-3, msg=f"the ratio at threshold {ratio}")


class Reduction(Question):
    """
    Where E91 reduces to BBM92, and what the reduction costs.
    """

    def test_reduce(self):
        """
        The model evaluation's basis-symmetry leg is the photon-pair engine's rate bit for bit:
        the reduction is shared code, not a claim.
        """
        for gain in (1e-3, 0.02, 0.5):
            for err in (0.0, 0.01, 0.05, 0.08):
                point = _core.ekert_point(gain, err, 1.22, 0.5)
                self.assertClose(
                    point[2],
                    _core.pair_rate(gain, err, err, 1.22, 0.5),
                    atol=0.0,
                    msg=f"reduction drift at gain={gain}, e={err}",
                )

    def test_diagnostic(self):
        """
        The violation the model evaluation reports is the same number a photon-pair run already
        carries as its labelled diagnostic.
        """
        for err in (0.0, 0.01, 0.05, 0.1464, 0.5):
            self.assertClose(
                _core.ekert_point(1.0, err, 1.0, 1.0)[0],
                _core.pair_chsh(1.0 - 2.0 * err),
                atol=0.0,
                msg=f"the two layers must model one violation, e={err}",
            )

    def test_cost(self):
        """
        Pricing Eve by the violation is never cheaper than pricing her by the phase error, and
        the two agree only on a noiseless link.
        """
        for err in (0.0, 1e-4, 0.01, 0.03, 0.05, 0.07, 0.12):
            chsh, pair = _core.ekert_point(0.4, err, 1.0, 1.0)[1:]
            self.assertLessEqual(chsh, pair, msg=f"the CHSH price undercut the phase price at {err}")

            if err > 0.0 and pair > 0.0:
                self.assertLess(chsh, pair, msg=f"the two must separate once there is error, {err}")

        equal = _core.ekert_point(0.4, 0.0, 1.0, 1.0)
        self.assertClose(equal[1], equal[2], atol=0.0, msg="a noiseless link loses nothing")

    def test_accumulated(self):
        """
        Entropy accumulation's own E91 section gives this tree's phase-error rate, not its
        CHSH-priced one, so the length that argument would carry to a finite block is the
        entanglement-based one.
        """
        for err in (0.0, 0.005, 0.02, 0.05, 0.09, 0.11):
            for fec in (1.0, 1.1, 1.22):
                leak = fec * h2(err)
                self.assertClose(
                    max(0.0, 1.0 - h2(err) - leak),
                    _core.pair_rate(1.0, err, err, fec, 1.0),
                    atol=0.0,
                    msg=f"Theorem 5.1 must be pair_rate at e={err}, f_ec={fec}",
                )
                self.assertLessEqual(
                    _core.ekert_rate(1.0, err, werner(err), fec, 1.0, "measured"),
                    _core.pair_rate(1.0, err, err, fec, 1.0),
                    msg=f"and the CHSH price must not beat it at e={err}",
                )

    def test_agree(self):
        """
        Handing the measured entry point the violation the model predicts returns the model
        evaluation's own number.
        """
        for gain in (0.05, 1.0):
            for err in (0.0, 0.02, 0.06):
                point = _core.ekert_point(gain, err, 1.1, 0.25)
                self.assertClose(
                    _core.ekert_rate(gain, err, point[0], 1.1, 0.25, "measured"),
                    point[1],
                    atol=0.0,
                    msg=f"the two entry points drifted at gain={gain}, e={err}",
                )


class Overlap(Guarded):
    """
    What a violation is worth to a key length: Alice's measurement overlap, which is not a price
    on Eve and is never subtracted from one.
    """

    def test_endpoints(self):
        """
        The certified overlap is exactly 1 at the local bound and exactly 1/2 at Tsirelson's,
        the value of a mutually unbiased qubit pair.
        """
        self.assertClose(
            _core.ekert_overlap(2.0, 1.0)[0],
            1.0,
            atol=0.0,
            msg="the local bound must certify no complementarity at all",
        )
        self.assertClose(
            _core.ekert_overlap(TSIRELSON, 1.0)[0],
            0.5,
            atol=0.0,
            msg="Tsirelson's bound must certify a conjugate pair",
        )
        self.assertClose(
            _core.eur_quality(_core.ekert_overlap(TSIRELSON, 1.0)[0]),
            _core.eur_conjugate(2),
            atol=0.0,
            msg="and that is the same one bit eur.rs reads off two qubit MUBs",
        )

    def test_transcript(self):
        """
        Both halves are the paper's two lemmas at every violation and surviving fraction,
        transcribed from the paper rather than read off the engine.
        """
        for i in range(29):
            s = 2.0 + (TSIRELSON - 2.0) * i / 28.0
            for eta in (1.0, 0.8, 0.45, 0.2, 0.05):
                got = _core.ekert_overlap(s, eta)
                self.assertClose(got[0], overlap(s), atol=1e-15, msg=f"Lemma 6 at s={s}")
                self.assertClose(got[1], charged(s, eta), atol=1e-15, msg=f"Lemma 7 at s={s}, eta={eta}")

    def test_monotone(self):
        """
        More violation certifies a smaller overlap across the whole domain, and the overlap
        never falls below the conjugate-pair value of 1/2.
        """
        span = [_core.ekert_overlap(2.0 + (TSIRELSON - 2.0) * i / 120.0, 1.0)[0] for i in range(121)]
        self.assertMonotone(span, rising=False, msg="Lemma 6 must fall with the violation")
        for c in span:
            self.assertGreaterEqual(c, 0.5, msg=f"an overlap of {c} is below a conjugate pair's")

    def test_ceiling(self):
        """
        A certified complementarity is never worth more than an assumed one, so the quality a
        CHSH value buys is at most the one bit pair_length reads off cbar = 1/2, with equality
        only at a maximal violation.
        """
        for i in range(1, 40):
            s = 2.0 + (TSIRELSON - 2.0) * i / 40.0
            quality = _core.eur_quality(_core.ekert_overlap(s, 1.0)[0])
            self.assertLess(quality, 1.0, msg=f"a violation of {s} bought more than a qubit MUB pair")
        self.assertClose(
            _core.eur_quality(_core.ekert_overlap(TSIRELSON, 1.0)[0]),
            1.0,
            atol=0.0,
            msg="equality is reached at Tsirelson's bound and nowhere below it",
        )

    def test_charge(self):
        """
        The post-selection charge divides the whole excess over 1/2 by the surviving fraction,
        so it cannot help, and it cannot move a maximal violation.
        """
        for eta in (1.0, 0.6, 0.3, 0.1):
            got = _core.ekert_overlap(TSIRELSON, eta)
            self.assertClose(got[1], 0.5, atol=0.0, msg=f"eta={eta} moved a zero excess")
        base = _core.ekert_overlap(2.7, 1.0)
        self.assertClose(base[1], base[0], atol=0.0, msg="a lossless key set pays nothing")
        charges = [_core.ekert_overlap(2.7, eta)[1] for eta in (1.0, 0.8, 0.5, 0.25, 0.1)]
        self.assertMonotone(charges, rising=True, msg="a smaller surviving fraction must cost")
        self.assertGreater(charges[-1], 1.0, msg="and eventually eats the violation outright")

    def test_secret(self):
        """
        Feeding the charged overlap through the entropic uncertainty relation reproduces the
        paper's own Eq. (2) exactly, and at a maximal violation that fraction is BB84's 1 -
        2h(Q), which is what the authors claim.
        """
        for qber in (0.0, 0.01, 0.03, 0.07):
            for eta in (1.0, 0.7, 0.4):
                for v in (1.0, 0.999, 0.99):
                    s = TSIRELSON * v
                    quality = _core.eur_quality(_core.ekert_overlap(s, eta)[1])
                    self.assertClose(
                        _core.eur_asymptotic(quality, qber, qber),
                        secret(s, eta, qber),
                        atol=1e-14,
                        msg=f"Eq. (2) at v={v}, eta={eta}, Q={qber}",
                    )
            self.assertClose(
                secret(TSIRELSON, 1.0, qber),
                1.0 - 2.0 * h2(qber),
                atol=1e-15,
                msg=f"a maximal violation must reach BB84's secret fraction at Q={qber}",
            )

    def test_crossing(self):
        """
        The paper's Fig. 2 curves cross zero at the two surviving fractions its caption orders
        from left to right, so a coincidence-counted link, whose surviving fraction is its
        coincidence probability, is far below both.
        """
        crossings = []
        for v in (0.999, 0.99):
            crossings.append(bisect(lambda eta, s=TSIRELSON * v: secret(s, eta, 0.01), 1e-9, 1.0))
        self.assertClose(crossings[0], ETA_999, atol=1e-9, msg="V = 0.999 crosses here")
        self.assertClose(crossings[1], ETA_99, atol=1e-9, msg="V = 0.99 crosses here")
        self.assertLess(crossings[0], crossings[1], msg="the caption orders them left to right")
        for eta in (1e-2, 1e-3, 1e-4):
            self.assertLess(
                secret(TSIRELSON * 0.999, eta, 0.01),
                0.0,
                msg=f"a coincidence probability of {eta} must leave no key",
            )


class Security(Guarded):
    """
    What the engine refuses, in shipped messages rather than in prose.
    """

    def test_modelled(self):
        """
        A violation read off a state model is refused as an input to the bound, and the refusal
        names the circularity and both alternatives.
        """
        args = (1.0, 0.02, 2.7, 1.0, 1.0, "modelled")
        for needle in ("modelled", "circular", "ekert_point", "pair_rate"):
            self.assertBad(
                needle,
                _core.ekert_rate,
                args,
                msg=f"the refusal must name {needle}",
            )

    def test_device(self):
        """
        Device independence is refused outright, naming the semidefinite program, the entropy
        accumulation and the detection efficiency that would each have to arrive first.
        """
        args = (1.0, 0.02, 2.7, 1.0, 1.0, "device-independent")
        needles = (
            "semidefinite",
            "entropy accumulation",
            "fair sampling",
            "coherent",
            "0.924",
        )
        for needle in needles:
            self.assertBad(
                needle,
                _core.ekert_rate,
                args,
                exc=NotImplementedError,
                msg=f"the refusal must name {needle}",
            )

    def test_source(self):
        """
        An unrecognised origin for the violation names the three the engine accepts rather than
        defaulting to one.
        """
        args = (1.0, 0.02, 2.7, 1.0, 1.0, "")
        for needle in ("measured", "modelled", "device-independent"):
            self.assertBad(
                needle,
                _core.ekert_rate,
                args,
                msg=f"the refusal must list {needle}",
            )

    def test_tsirelson(self):
        """
        A violation past Tsirelson's bound is refused by every entry point that takes one,
        rather than priced.
        """
        for bad in (2.9, -2.9, 4.0):
            self.assertBad(
                "Tsirelson",
                _core.ekert_qber,
                (bad,),
                msg=f"ekert_qber priced {bad}",
            )
            self.assertBad(
                "Tsirelson",
                _core.ekert_rate,
                (1.0, 0.02, bad, 1.0, 1.0, "measured"),
                msg=f"ekert_rate priced {bad}",
            )
        self.assertBad(
            "s must be finite",
            _core.ekert_qber,
            (float("nan"),),
            msg="a lost violation must raise, not read as no violation",
        )

    def test_finite(self):
        """
        A finite-key E91 is refused outright, and the message names four missing pieces, both
        published routes from a violation to a length, and the entanglement-based length that
        does ship.
        """

        self.assertFails(
            NotImplementedError,
            "finite-key E91 is refused",
            _core.ekert_finite,
            1e6,
            msg="no published length reads this engine's s",
        )
        text = ""
        try:
            _core.ekert_finite(1e6)
        except NotImplementedError as exc:
            text = str(exc)
        for needle in ("(1)", "(2)", "(3)", "(4)"):
            self.assertIn(needle, text, msg=f"the refusal enumerates its pieces, missing {needle}")
        self.assertIn("ASYMPTOTIC", text, msg="the CHSH price is a per-round Holevo quantity")
        self.assertIn("entropy accumulation", text, msg="and names the route across that gap")
        self.assertIn("NEITHER READS A CHSH VALUE", text, msg="and what it prices E91 by instead")
        self.assertIn("Lim, Portmann, Tomamichel, Renner & Gisin", text, msg="and the other length")
        self.assertIn("ekert_overlap", text, msg="and the one piece of it this file computes")
        self.assertIn("ARRIVES AS AN ARGUMENT", text, msg="s carries no round count to bound")
        self.assertIn("b92_finite", text, msg="the same composition is refused one protocol over")
        self.assertIn("pair_finite", text, msg="and the entanglement-based length is named")
        self.assertIn("fair sampling", text, msg="and the post-selection s would be read through")
        self.assertIn("0.1134", text, msg="and the surviving fraction that charge demands")
        self.assertFails(
            ValueError,
            "n_total",
            _core.ekert_finite,
            0.0,
            msg="an empty block is refused before the family question is reached",
        )

    def test_domain(self):
        """
        The certified overlap is refused below the local bound, where the paper's expression
        turns back and would report a perfect complementarity from no violation.
        """
        for bad in (0.0, 1.0, 1.999, -1.5):
            self.assertBad(
                "monotonic decreasing",
                _core.ekert_overlap,
                (bad, 1.0),
                msg=f"ekert_overlap certified {bad}",
            )
        self.assertBad(
            "Tsirelson",
            _core.ekert_overlap,
            (2.9, 1.0),
            msg="ekert_overlap certified a value no state reaches",
        )
        self.assertBad(
            "eta must be > 0",
            _core.ekert_overlap,
            (2.7, 0.0),
            msg="a round set nothing survives certifies nothing",
        )
        self.assertClose(
            _core.ekert_overlap(2.0, 1.0)[0],
            overlap(2.0),
            atol=0.0,
            msg="the local bound itself is inside the domain the paper states",
        )

    def test_absent(self):
        """
        The engine exposes no entry point whose name would read as a device-independent bound.
        """
        for name in ("ekert_device", "ekert_di", "ekert_certify", "ekert_secure"):
            self.assertFalse(hasattr(_core, name), msg=f"_core.{name} would claim what is refused")
        self.assertTrue(hasattr(_core, "ekert_rate"), msg="the collective-attack rate does ship")

    def test_slots(self):
        """
        Out-of-domain arguments raise rather than returning a wrong number.
        """

        self.assertSlots(
            _core.ekert_rate,
            (1.0, 0.02, 2.7, 1.0, 1.0, "measured"),
            (
                (0, "gain", (-0.1, 1.5)),
                (1, "e_bit", (0.6, -0.1)),
                (3, "f_ec", (0.5, 0.0)),
                (4, "sift", (-0.1, 1.5)),
            ),
            msg="ekert_rate",
        )
        self.assertSlots(
            _core.ekert_point,
            (1.0, 0.02, 1.0, 1.0),
            (
                (0, "gain", (-0.1, 1.5)),
                (1, "e_bit", (0.6, -0.1)),
                (2, "f_ec", (0.5,)),
            ),
            msg="ekert_point",
        )
        self.assertSlots(
            _core.ekert_correlate,
            (0.3, 1.2, 0.9),
            (
                (0, "a", (float("nan"), float("inf"))),
                (2, "v", (-0.1, 1.5)),
            ),
            msg="ekert_correlate",
        )
        self.assertSlots(
            _core.ekert_threshold,
            (1.0,),
            ((0, "f_ec", (0.9, float("nan"))),),
            msg="ekert_threshold",
        )


if __name__ == "__main__":
    rc = Exam(
        "EkertProtocol",
        "Ekert 1991's analyser settings, correlations and CHSH combination",
        "ekert_protocol.md",
    ).run(load(Protocol))
    rc |= Exam(
        "EkertBound",
        "The CHSH-priced key rate between the local bound and Tsirelson's",
        "ekert_bound.md",
    ).run(load(Bound))
    rc |= Exam(
        "EkertAnchors",
        "Published numbers: Ekert 1991, Acin et al. 2007, Pironio et al. 2009",
        "ekert_anchors.md",
    ).run(load(Anchors))
    rc |= Exam(
        "EkertReduction",
        "Where E91 reduces to BBM92, and what the reduction costs",
        "ekert_reduction.md",
    ).run(load(Reduction))
    rc |= Exam(
        "EkertOverlap",
        "What a violation buys a key LENGTH: Lim et al.'s effective overlap",
        "ekert_overlap.md",
    ).run(load(Overlap))
    rc |= Exam(
        "EkertSecurity",
        "What the E91 engine refuses, and the device-independence position",
        "ekert_security.md",
    ).run(load(Security))
    sys.exit(rc)
