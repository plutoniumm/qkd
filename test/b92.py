import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.checks import Guarded
from kit.forms import bisect, h2, plob
from qkd import _core

# Three entry points on two receivers.
#
# Plain B92: Tamaki & Lutkenhaus, Phys. Rev. A 69, 032316 (2004), quant-ph/0308048,
# extending Tamaki, Koashi & Imoto, Phys. Rev. Lett. 90, 167904 (2003) to a lossy,
# noisy channel. Single-photon source, phase error derived from loss and bit error;
# unambiguous discrimination takes over at loss L = overlap, where the bound reaches
# 1/2 and the rate reaches zero unrefused.
#
# Strong reference: Koashi, Phys. Rev. Lett. 93, 120501 (2004), quant-ph/0403131, is
# b92_point(reference=True) and its e_ph stays supplied, as COW's is; Tamaki,
# Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302 (2009),
# quant-ph/0607082, sort the reference by photon number and b92_strong derives e_ph
# from Eq. (15).
#
# The operating point below is this exam's, no paper's: 10 dB span, default detector
# efficiency, 98% fringe, dark count four orders below a gate.
T_CHAN = 0.1
ETA_DET = 0.2
VIS = 0.98
DARK = 1e-6
E_PHASE = 0.05
F_REC = 1.16

# Shor & Preskill, Phys. Rev. Lett. 85, 441 (2000): the root of h2(e) = 1/2.
SHOR = 0.11002786443835955

# Tamaki & Lutkenhaus's Fig. 2 to the two digits they print, recomputed by b92_limit.
TAMAKI = {
    0.0: 0.034,
    0.2: 0.023,
    0.5: 0.012,
}

# b92_limit's second return is the amplitude overlap c; Fig. 2(b) plots c^2, which
# reads 0.46410, 0.56250, 0.73316.
OPTIMA = {
    0.0: 0.68125,
    0.2: 0.75,
    0.5: 0.85625,
}

# Koashi 2004's strong-reference threshold, recorded only: that branch's e_ph is an input.
KOASHI = 0.01

# Tamaki, Lutkenhaus, Koashi & Batuwantudawe, Phys. Rev. A 80, 032302 (2009), Sec. IV:
# their Fig. 2 hardware, with dark count, fibre and eta from Gobby, Yuan & Shields.
SRP_NOISE = 1.7e-6
SRP_DB = 0.21
SRP_ETA = 0.045
SRP_KAPPA = 10**-0.92
SRP_WIDTH = 3.2

# Their Fig. 2 curves (a), (b), (c): the km each reaches zero at, keyed by intensity.
SRP_REACH = {
    1e5: 55.0,
    10**6.59: 100.0,
    1e10: 122.0,
}

# "la = 124 ... the maximum distance among the combinations of the parameter set that
# we have tried" -- a maximum over kappa and the window as well as mu.
SRP_TOP = 124.0
SRP_BEST = (0.18, 5.0)

# p_fil comes off a threshold gate: linear in t only while 2*eta*t*mu*(1 + V) << 1.
SLOPE_LO = 1e-4
SLOPE_HI = 0.5
SLOPE_MU = 0.05

# Azuma at bounded difference 1, Eq. (A3) of Phys. Rev. A 80, 032302 (2009), rescaled
# by Eq. (A4) onto the code pairs and Eq. (A5) onto the test pairs; no key length.
AZUMA_N = 1e9
AZUMA_T = 0.05
AZUMA_EPS = 1e-10


def detect(mu, t=T_CHAN, eta=ETA_DET, vis=VIS, dark=DARK):
    return _core.b92_detect(mu, t, eta, vis, dark)


def ceiling(mu, t=T_CHAN, eta=ETA_DET, vis=VIS, dark=DARK):
    return _core.b92_ceiling(mu, t, eta, vis, dark)


def point(mu, t=T_CHAN, eph=E_PHASE):
    return _core.b92_point(mu, t, ETA_DET, VIS, DARK, eph, F_REC, True)


def rate(mu, t=T_CHAN, eph=E_PHASE, f_ec=F_REC, dark=DARK):
    """
    The rate alone at one signal strength, uncapped by the beam-splitting ceiling.
    """
    fil, err = detect(mu, t, ETA_DET, VIS, dark)

    return _core.b92_rate(fil, err, eph, f_ec)


def ideal(over, loss):
    """
    Tamaki and Lutkenhaus's n_fil/N: the clean single-photon conclusive gain at one overlap and loss.
    """
    return 0.5 * (1.0 - over * over) * (1.0 - loss)


def closed(over, loss):
    """
    Tamaki and Lutkenhaus's closed zero-bit-error phase bound, alpha^2/(beta^2 - alpha^2) times L/(1 - L), capped at 1/2.
    """
    return min(0.5, 0.5 * (1.0 - over) / over * loss / (1.0 - loss))


def log_slope(near, far, mu=SLOPE_MU):
    """
    Slope of log rate against log transmission, on a dark-free detector.
    """
    ratio = rate(mu, near, dark=0.0) / rate(mu, far, dark=0.0)

    return math.log(ratio) / math.log(near / far)


def two_state(mu, t, eph=None, ref=False, f=1.0):
    """
    The two-state link: one weak pulse per bit into a nulling receiver over a lossy channel, perfect fringe and no dark counts.
    """

    return q.Link(
        modulation=q.TwoStateKeying(mu=mu, reference=ref),
        channel=q.Channel(T=t),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=0.0),
            receiver=q.NullingReceiver(visibility=1.0),
        ),
        security=q.DiscriminationBound(e_phase=eph, f=f),
    )


def strong(dist, mu, kappa=SRP_KAPPA, width=SRP_WIDTH, f_ec=1.0):
    """
    The photon-number-resolving strong-reference variant at one distance in km, over the paper's 0.21 dB/km fibre.
    """

    return _core.b92_strong(kappa, mu, 10 ** (-SRP_DB * dist / 10.0), SRP_ETA, SRP_NOISE, width, f_ec)


# Hand-rolled: kit.forms.bisect's fixed 200 halvings is a measured 3.3x on b92_strong,
# and 60 resolves the crossing past the 0.1 km the rate's flatness there allows.
def crossing(mu, kappa=SRP_KAPPA, width=SRP_WIDTH):
    """
    The km at which the strong-reference rate reaches zero, by bisection on its positivity.
    """
    lo, hi = 0.0, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if strong(mid, mu, kappa, width)[3] > 0.0:
            lo = mid
        else:
            hi = mid

    return lo


def span_cap(dist):
    """
    The repeaterless capacity of the same span in bits per mode, Bob's efficiency inside it.
    """

    return plob(10 ** (-SRP_DB * dist / 10.0) * SRP_ETA)


def azuma_back(width, share, n=AZUMA_N):
    """
    Azuma's inequality as the paper scales it, 2*exp(-N*share^2*width^2/2): share is 1 - t on the code pairs and t on the test pairs.
    """

    return 2.0 * math.exp(-n * share * share * width * width / 2.0)


def refusal(reference):
    """
    The text b92_finite refuses one branch with, as a string.
    """
    try:
        _core.b92_finite(AZUMA_N, reference)
    except NotImplementedError as exc:
        return str(exc)

    return ""


def entropy_root(want):
    """
    The root of h2(e) = want below 1/2.
    """

    return bisect(h2, 0.0, 0.5, want)


class B92States(Question):
    """
    Two non-orthogonal states, Bob's receiver and Eve's discrimination.
    """

    def test_overlap(self):
        """
        The state overlap is exp(-2 mu) and unambiguous discrimination of two equiprobable pure states succeeds with probability 1 - c.
        """
        for mu in (0.01, 0.23, 0.5, 1.0, 4.0):
            self.assertClose(
                _core.b92_overlap(mu),
                math.exp(-2.0 * mu),
                msg=f"overlap at mu={mu}",
            )
        self.assertClose(_core.b92_overlap(0.23), 0.631283645506926, msg="Koashi's optimal amplitude")

        for over in (0.0, 0.25, 0.631283645506926, 1.0):
            self.assertClose(_core.b92_usd(over), 1.0 - over, msg=f"IDP at c={over}")
        self.assertClose(_core.b92_usd(1.0), 0.0, msg="identical states are never discriminated")
        self.assertClose(_core.b92_usd(0.0), 1.0, msg="orthogonal states always are")

    def test_receiver_idp(self):
        """
        The displacement receiver reaches the unambiguous-discrimination optimum as mu -> 0 and trails it by (1 + c)/2 as the signal brightens.
        """
        ratios = []
        for mu in (1e-3, 1e-2, 0.1, 1.0):
            fil = _core.b92_detect(mu, 1.0, 1.0, 1.0, 0.0)[0]
            over = _core.b92_overlap(mu)
            self.assertClose(
                fil / _core.b92_usd(over),
                0.5 * (1.0 + over),
                atol=1e-14,
                msg=f"receiver against IDP at mu={mu}",
            )

            ratios.append(fil / _core.b92_usd(over))

        weak = _core.b92_detect(1e-6, 1.0, 1.0, 1.0, 0.0)[0] / _core.b92_usd(_core.b92_overlap(1e-6))
        self.assertClose(
            weak,
            1.0,
            atol=1e-5,
            msg="IDP-optimal as mu -> 0, where 1 - exp(-4 mu) also loses six digits",
        )
        self.assertMonotone(ratios, rising=False, msg="the gap to IDP widens with the signal")

    def test_conclusive_form(self):
        """
        With no dark counts the conclusive rate is (1 - c^2)/2 over the states reaching the detector, and the bit error is (1 - visibility)/2 at weak flux and the dark count on the nulled gate at a perfect fringe.
        """
        for mu in (0.01, 0.1, 1.0):
            for t in (1.0, 0.3, 0.05):
                fil = _core.b92_detect(mu, t, ETA_DET, 1.0, 0.0)[0]
                over = math.exp(-2.0 * ETA_DET * t * mu)
                self.assertClose(
                    fil,
                    0.5 * (1.0 - over * over),
                    msg=f"conclusive rate at mu={mu}, t={t}",
                )

        for vis in (0.9, 0.98, 1.0):
            err = _core.b92_detect(1e-6, T_CHAN, ETA_DET, vis, 0.0)[1]
            self.assertClose(err, 0.5 * (1.0 - vis), atol=1e-7, msg=f"weak-flux error at V={vis}")

        fil, err = _core.b92_detect(0.1, T_CHAN, ETA_DET, 1.0, DARK)
        hit = 1.0 - (1.0 - DARK) * math.exp(-ETA_DET * 4.0 * T_CHAN * 0.1)
        self.assertClose(fil, 0.5 * (hit + DARK), msg="conclusive rate carries the dark gate")
        self.assertClose(err, DARK / (hit + DARK), msg="every error is a dark click")

    def test_usd_outruns(self):
        """
        Eve's unambiguous discrimination outruns Bob's conclusive rate at every signal strength, even lossless.
        """
        for mu in (1e-3, 1e-2, 0.1, 0.23, 1.0, 5.0):
            fil = _core.b92_detect(mu, 1.0, 1.0, 1.0, 0.0)[0]
            eve = _core.b92_usd(_core.b92_overlap(mu))
            self.assertLess(fil, eve, msg=f"Bob {fil:.6f} should trail Eve {eve:.6f} at mu={mu}")


class B92Tamaki(Question):
    """
    The Tamaki-Lutkenhaus bound: a derived phase error and the published depolarising thresholds.
    """

    def test_special_case(self):
        """
        The solved phase bound reproduces the paper's closed zero-bit-error special case over the stated overlap and loss ranges.
        """
        worst = 0.0
        for i in range(1, 100):
            over = i / 100.0
            for loss in (0.0, 0.01, 0.05, 0.2, 0.5, 0.8):
                if loss >= over:
                    continue

                got = _core.b92_phase(over, loss, ideal(over, loss), 0.0)
                worst = max(worst, abs(got - closed(over, loss)))
        self.assertLess(worst, 1e-6, msg=f"worst departure from the closed form was {worst:.3e}")
        self.assertClose(
            _core.b92_phase(0.5, 0.1, ideal(0.5, 0.1), 0.0),
            0.05555555628095259,
            atol=1e-9,
            msg="one point of it written out",
        )

    def test_usd_boundary(self):
        """
        The phase bound reaches exactly 1/2 at loss L = c, where unambiguous discrimination takes the protocol over.
        """
        for over in (0.818730753077982, 0.631283645506926, 0.367879441171442):
            below = _core.b92_phase(over, over * 0.99, ideal(over, over * 0.99), 0.0)
            at = _core.b92_phase(over, over, ideal(over, over), 0.0)
            above = _core.b92_phase(over, over * 1.01, ideal(over, over * 1.01), 0.0)
            self.assertLess(below, 0.5, msg=f"still finite below L = c at c={over}")
            self.assertClose(at, 0.5, atol=1e-9, msg=f"exactly one half at c={over}")
            self.assertClose(above, 0.5, msg=f"and saturated past it at c={over}")
            self.assertGreater(
                _core.b92_rate(ideal(over, over * 0.99), 0.0, below, 1.0),
                0.0,
                msg=f"positive key just inside the boundary at c={over}",
            )
            self.assertClose(
                _core.b92_rate(ideal(over, over), 0.0, at, 1.0),
                0.0,
                msg=f"and none at it at c={over}",
            )
        self.assertClose(
            _core.b92_usd(0.631283645506926),
            0.368716354493074,
            msg="the surviving fraction at the boundary is the IDP success rate",
        )

    def test_depol_limit(self):
        """
        The tolerable depolarising rate reproduces Tamaki and Lutkenhaus's three values, at an optimum overlap reported as the amplitude whose square their Fig. 2(b) plots.
        """
        got, best = [], []
        for loss, want in sorted(TAMAKI.items()):
            limit, over = _core.b92_limit(loss, 1.0)
            got.append(limit)
            best.append(over)
            self.assertClose(limit, want, atol=6e-4, msg=f"published {want} at loss {loss}")
        self.assertClose(got[0], 0.033786080453637, atol=1e-9, msg="L = 0 in full")
        self.assertClose(got[1], 0.023131775858018, atol=1e-9, msg="L = 0.2 in full")
        self.assertClose(got[2], 0.012147788916536, atol=1e-9, msg="L = 0.5 in full")
        self.assertMonotone(got, rising=False, msg="more loss, less noise tolerated")
        self.assertMonotone(best, msg="the optimum overlap widens with loss")

        plotted = [0.4641015625, 0.5625, 0.7331640625]
        for loss, over, want in zip(sorted(OPTIMA), best, plotted):
            self.assertClose(
                over,
                OPTIMA[loss],
                atol=1e-9,
                msg=f"the amplitude overlap c at loss {loss}",
            )
            self.assertClose(
                over * over,
                want,
                atol=1e-9,
                msg=f"and its square, which is what Fig. 2(b) plots, at loss {loss}",
            )

    def test_error_tolerance(self):
        """
        The bit error tolerated at those thresholds sits under the Shor-Preskill 11.00%.
        """
        errs = []
        for loss in (0.0, 0.2, 0.5):
            depol, over = _core.b92_limit(loss, 1.0)
            errs.append(_core.b92_channel(over, loss, depol * 0.999)[1])
        self.assertClose(errs[0], 0.04041336, atol=1e-7, msg="tolerated bit error, L=0")
        self.assertClose(errs[1], 0.03387136, atol=1e-7, msg="at L = 0.2")
        self.assertClose(errs[2], 0.02902925, atol=1e-7, msg="at L = 0.5")

        for loss, err in zip((0.0, 0.2, 0.5), errs):
            self.assertLess(err, SHOR, msg=f"under Shor-Preskill at loss {loss}")

    def test_channel_form(self):
        """
        The depolarising channel gives the conclusive rate and bit error the Bloch-shrinking factor predicts.
        """
        for over in (0.3, 0.681, 0.9):
            for loss in (0.0, 0.4):
                gain, err = _core.b92_channel(over, loss, 0.0)
                self.assertClose(gain, ideal(over, loss), msg=f"clean gain at c={over}, L={loss}")
                self.assertClose(err, 0.0, msg=f"and no errors at c={over}, L={loss}")

                for depol in (0.01, 0.05, 0.2):
                    lam = 1.0 - 4.0 * depol / 3.0
                    flip = 0.25 * (1.0 - lam)
                    total = 0.5 * (1.0 - over * over) * lam + 2.0 * flip
                    gain, err = _core.b92_channel(over, loss, depol)
                    self.assertClose(gain, (1.0 - loss) * total, msg=f"gain at p={depol}")
                    self.assertClose(err, flip / total, msg=f"bit error at p={depol}")

    def test_plain_derived(self):
        """
        The end-to-end plain rate agrees with the gain, bit error and phase error it derives, and reaches zero past the depolarising threshold.
        """
        over, loss = 0.68125, 0.2
        seq = []
        for depol in (0.0, 0.002, 0.005, 0.01, 0.02):
            gain, err, eph, key = _core.b92_plain(over, loss, depol, 1.0)

            back = _core.b92_channel(over, loss, depol)
            self.assertClose(gain, back[0], msg=f"the gain it used at p={depol}")
            self.assertClose(err, back[1], msg=f"the bit error it used at p={depol}")
            self.assertClose(
                eph,
                _core.b92_phase(over, loss, gain, err),
                msg=f"the phase bound it used at p={depol}",
            )
            self.assertClose(
                key,
                _core.b92_rate(gain, err, eph, 1.0),
                msg=f"the rate equation it used at p={depol}",
            )

            seq.append(key)
        self.assertMonotone(seq, rising=False, msg="noise only ever costs key")

        limit = _core.b92_limit(loss, 1.0)[0]
        self.assertGreater(
            _core.b92_plain(over, loss, limit * 0.9, 1.0)[3],
            0.0,
            msg="key below the threshold",
        )
        self.assertClose(
            _core.b92_plain(over, loss, limit * 1.1, 1.0)[3],
            0.0,
            msg="and none above it",
        )

    def test_phase_rises(self):
        """
        The derived phase error rises with channel loss and with the bit error rate, and saturates at 1/2.
        """
        # The clamp is deliberate: h2 falls again past 1/2, so an unclamped 0.53 reads as key.
        over = 0.6
        walk = [_core.b92_phase(over, loss, ideal(over, loss), 0.0) for loss in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
        self.assertMonotone(walk, msg="loss is what feeds the phase error")
        self.assertClose(walk[0], 0.0, msg="a lossless clean channel makes none")

        noisy = [
            _core.b92_phase(over, 0.3, _core.b92_channel(over, 0.3, p)[0], e)
            for p, e in [(0.0, 0.0), (0.01, 0.006), (0.02, 0.012), (0.03, 0.018)]
        ]
        self.assertMonotone(noisy, msg="and so is the bit error")

        for value in walk + noisy:
            self.assertLessEqual(value, 0.5, msg=f"never past one half, got {value}")


class B92Reach(Question):
    """
    Where plain B92 stops, through q.Link: transmittance, decibels, kilometres.
    """

    def test_point_plain(self):
        """
        A q.Link prices plain B92, its rate crossing zero at the unambiguous-discrimination success probability: 4.3331 dB, 21.665 km at 0.2 dB/km.
        """
        mu = 0.23
        over = q.TwoStateKeying(mu=mu).overlap
        edge = two_state(mu, 0.5).run().explain["discrimination"]["value"]
        self.assertClose(edge, 1.0 - over, msg="the crossing is one less the state overlap")
        self.assertClose(edge, 0.368716354493074, msg="the transmittance plain B92 dies at")

        for scale in (0.5, 0.9, 0.99, 1.0):
            key = two_state(mu, edge * scale).run().key_rate
            self.assertClose(key, 0.0, msg=f"no key at {scale:.2f} of the crossing")

        rising = []
        for scale in (1.01, 1.1, 1.5, 2.0):
            key = two_state(mu, min(1.0, edge * scale)).run().key_rate
            rising.append(key)
            self.assertGreater(key, 0.0, msg=f"key at {scale:.2f} of the crossing")
        self.assertMonotone(rising, msg="and more of it as the fibre improves")

        budget = -10.0 * math.log10(edge)
        self.assertClose(budget, 4.333075987416624, atol=1e-9, msg="a decibel budget")
        self.assertClose(budget / 0.2, 21.66537993708312, atol=1e-8, msg="and kilometres of fibre")

        weaker = two_state(0.1, 0.5).run().explain["discrimination"]["value"]
        self.assertGreater(
            -10.0 * math.log10(weaker),
            budget,
            msg="a weaker signal reaches further and carries less",
        )

    def test_plain_supplied(self):
        """
        A supplied phase error is honoured only as a floor, and the plain branch's derived error leaves less key than Koashi's supplied one on the same gain, bit error and ceiling.
        """
        free = two_state(0.23, 0.8).run().key_rate
        tight = two_state(0.23, 0.8, eph=1e-6).run().key_rate
        loose = two_state(0.23, 0.8, eph=0.4).run().key_rate
        self.assertClose(tight, free, msg="a tighter bound than the proof gives is not")
        self.assertLess(loose, free, msg="a looser one is honoured and costs key")
        self.assertGreater(free, 0.0, msg="and the branch returns a number at all")

        plain = two_state(0.23, 0.6, eph=E_PHASE, f=F_REC).run()
        koashi = two_state(0.23, 0.6, eph=E_PHASE, f=F_REC, ref=True).run()
        self.assertClose(plain.p_click, koashi.p_click, msg="same gain either way")
        self.assertClose(plain.qber, koashi.qber, msg="same bit error either way")
        self.assertClose(
            plain.explain["ceiling"]["value"],
            koashi.explain["ceiling"]["value"],
            msg="same ceiling either way",
        )
        self.assertLess(
            plain.key_rate,
            koashi.key_rate,
            msg="but the derived phase error costs more than the supplied one here",
        )


class B92Guards(Guarded):
    """
    Every argument of the B92 surface outside its domain.
    """

    def test_plain_priced(self):
        """
        Neither the rate equation nor the operating point refuses plain B92.
        """
        fil, err = detect(0.1)
        self.assertGreaterEqual(
            _core.b92_rate(fil, err, E_PHASE, F_REC),
            0.0,
            msg="the rate equation prices it",
        )

        priced = _core.b92_point(0.1, T_CHAN, ETA_DET, VIS, DARK, E_PHASE, F_REC, False)
        self.assertEqual(len(priced), 4, msg="the operating point prices it too")
        self.assertGreaterEqual(priced[3], 0.0, msg="and never returns a negative rate")

    def test_ceiling_caps(self):
        """
        A rate the beam-splitting attack has outrun is capped at the ceiling, not refused.
        """
        capped = []
        for mu in (1.0, 2.0, 4.0, 8.0):
            fil, err, best, key = point(mu)
            capped.append(key)
            self.assertLessEqual(key, best, msg=f"rate {key:.8f} over ceiling {best:.8f} at mu={mu}")
        self.assertClose(capped[2], ceiling(4.0), msg="a bright signal follows the ceiling down")
        self.assertGreater(rate(4.0), ceiling(4.0), msg="the equation alone would have returned more")
        self.assertMonotone(capped[1:], rising=False, msg="and the cap falls with the overlap")

    def test_lossless_optimum(self):
        """
        A lossless channel has no interior optimum overlap and is refused.
        """
        self.assertFails(
            ValueError,
            "no interior optimum",
            _core.b92_optimum,
            1.0,
            ETA_DET,
            VIS,
            DARK,
            msg="t = 1 leaves Eve's beam-splitting share empty",
        )

    def test_phase_domain(self):
        """
        A phase error above 1/2 is refused: both proofs require twice the phase errors not exceed the conclusive count.
        """
        self.assertBad(
            "e_ph must be in [0, 1/2]",
            _core.b92_rate,
            (0.01, 0.01, 0.6, F_REC),
            msg="past 1/2 the entropy term turns back upward",
        )
        self.assertClose(
            _core.b92_rate(0.01, 0.0, 0.5, 1.0),
            0.0,
            msg="a phase error of exactly 1/2 pays the whole bit",
        )

    def test_strong_slots(self):
        """
        Every argument of the strong-reference layer is refused outside its domain, the noise weight being a probability and not a per-gate dark count.
        """
        self.assertSlots(
            _core.b92_strong,
            (SRP_KAPPA, 1e5, 0.1, SRP_ETA, SRP_NOISE, SRP_WIDTH, 1.0),
            [
                (0, "kappa must be > 0", [0.0, -1.0]),
                (1, "mu must be > 0", [0.0, float("inf")]),
                (2, "t must be in (0, 1]", [0.0, 1.5]),
                (3, "eta must be in (0, 1]", [0.0, 1.5]),
                (4, "noise must be in [0, 1]", [-0.1, 1.5]),
                (5, "width must be > 0", [0.0, -1.0]),
                (6, "f_ec must be >= 1", [0.5, 0.0]),
            ],
            msg="b92_strong",
        )

    def test_azuma_slots(self):
        """
        Every argument of the deviation bound is refused outside its domain, the test fraction at both ends.
        """
        self.assertSlots(
            _core.b92_azuma,
            (AZUMA_N, AZUMA_T, AZUMA_EPS),
            [
                (0, "n_total must be > 0", [0.0, -1.0]),
                (1, "test must be in (0, 1)", [0.0, 1.0]),
                (2, "eps must be in (0, 1)", [0.0, 1.0]),
            ],
            msg="b92_azuma",
        )

    def test_strong_refused(self):
        """
        The three configurations the proof is not written for are refused by name: a reference no brighter than the signal, one too weak for D1 to sort, and a window past Appendix B's conditions.
        """
        self.assertBad(
            "not below 1",
            _core.b92_strong,
            (1.0, 0.5, 0.5, 0.5, 0.0, SRP_WIDTH, 1.0),
            msg="a reflectivity of one leaves Bob no local oscillator",
        )
        self.assertBad(
            "no photon-number window",
            _core.b92_window,
            (SRP_KAPPA, 1e5, 1e-9, SRP_ETA, SRP_WIDTH),
            msg="a reference that never arrives leaves the qubit space undefined",
        )
        self.assertBad(
            "past Appendix B's conditions",
            _core.b92_window,
            (100.0, 1000.0, 1.0, 1.0, SRP_WIDTH),
            msg="past them the entrywise maximum leaves the endpoints",
        )

    def test_slots(self):
        """
        Every argument of the receiver, the rate equation, the operating point and the Tamaki-Lutkenhaus layer is refused outside its domain.
        """
        self.assertSlots(
            _core.b92_detect,
            (0.1, T_CHAN, ETA_DET, VIS, DARK),
            [
                (0, "mu must be > 0", [0.0, -1.0, float("inf")]),
                (1, "t must be in (0, 1]", [0.0, 1.5]),
                (2, "eta must be in (0, 1]", [0.0, 1.5]),
                (3, "visibility must be in [0, 1]", [-0.1, 1.5]),
                (4, "dark must be in [0, 1)", [1.0, -0.1]),
            ],
            msg="b92_detect",
        )
        self.assertSlots(
            _core.b92_rate,
            (0.01, 0.01, E_PHASE, F_REC),
            [
                (0, "p_fil must be in [0, 1]", [-0.1, 1.5]),
                (1, "e_bit must be in [0, 1/2]", [0.6, -0.1]),
                (2, "e_ph must be in [0, 1/2]", [0.6, -0.1]),
                (3, "f_ec must be >= 1", [0.9, float("nan")]),
            ],
            msg="b92_rate",
        )
        self.assertSlots(
            _core.b92_point,
            (0.1, T_CHAN, ETA_DET, VIS, DARK, E_PHASE, F_REC, True),
            [
                (0, "mu must be > 0", [0.0, -1.0]),
                (1, "t must be in (0, 1]", [0.0, 1.5]),
                (5, "e_ph must be in [0, 1/2]", [0.6]),
                (6, "f_ec must be >= 1", [0.5]),
            ],
            msg="b92_point",
        )
        self.assertSlots(
            _core.b92_phase,
            (0.6, 0.2, ideal(0.6, 0.2), 0.01),
            [
                (0, "overlap must be in (0, 1)", [0.0, 1.0, -0.1, 1.5]),
                (1, "loss must be in [0, 1)", [1.0, -0.1]),
                (2, "gain must be in (0, 1]", [0.0, 1.5]),
                (3, "e_bit must be in [0, 1/2]", [0.6, -0.1]),
            ],
            msg="b92_phase",
        )
        self.assertSlots(
            _core.b92_channel,
            (0.6, 0.2, 0.01),
            [
                (0, "overlap must be in (0, 1)", [0.0, 1.0]),
                (1, "loss must be in [0, 1)", [1.0, -0.1]),
                (2, "depol must be in [0, 1]", [-0.1, 1.5]),
            ],
            msg="b92_channel",
        )
        self.assertSlots(
            _core.b92_plain,
            (0.6, 0.2, 0.01, 1.0),
            [
                (0, "overlap must be in (0, 1)", [0.0, 1.0]),
                (1, "loss must be in [0, 1)", [1.0]),
                (2, "depol must be in [0, 1]", [1.5]),
                (3, "f_ec must be >= 1", [0.9]),
            ],
            msg="b92_plain",
        )
        self.assertSlots(
            _core.b92_limit,
            (0.2, 1.0),
            [
                (0, "loss must be in [0, 1)", [1.0, -0.1]),
                (1, "f_ec must be >= 1", [0.9]),
            ],
            msg="b92_limit",
        )


class B92Overlap(Question):
    """
    The optimum overlap, where Bob's discrimination crosses Eve's.
    """

    def test_optimum(self):
        """
        The optimum overlap is interior and sits where Bob's conclusive rate crosses the share of rounds Eve's split light leaves opaque.
        """
        mu, over, best = _core.b92_optimum(T_CHAN, ETA_DET, VIS, DARK)
        self.assertClose(mu, 1.5711192813345294, atol=1e-9, msg="optimum signal strength")
        self.assertClose(over, 0.043186015014066305, atol=1e-12, msg="optimum overlap")
        self.assertClose(best, 0.05913012788338273, atol=1e-12, msg="the ceiling there")
        self.assertClose(over, math.exp(-2.0 * mu), msg="the overlap names the same point")
        self.assertClose(
            detect(mu)[0],
            math.exp(-2.0 * (1.0 - T_CHAN) * mu),
            atol=1e-9,
            msg="the two branches cross at the optimum",
        )

    def test_ceiling_shape(self):
        """
        The ceiling turns at the optimum overlap and, taken at its own optimum, rises with channel transmission.
        """
        mu = _core.b92_optimum(T_CHAN, ETA_DET, VIS, DARK)[0]
        below = [ceiling(mu * f) for f in (0.1, 0.25, 0.5, 0.8, 1.0)]
        above = [ceiling(mu * f) for f in (1.0, 1.25, 2.0, 4.0, 10.0)]
        self.assertMonotone(below, msg="too close together and Bob rarely discriminates")
        self.assertMonotone(above, rising=False, msg="too far apart and Eve discriminates instead")

        best = [_core.b92_optimum(t, ETA_DET, VIS, DARK)[2] for t in (0.01, 0.1, 0.5, 0.9)]
        self.assertMonotone(best, msg="less loss, more ceiling")

    def test_ceiling_binds(self):
        """
        No rate exceeds the beam-splitting ceiling across a sweep of signal strength, and the operating point reports the receiver's own gain and bit error.
        """
        # Below the b92_optimum crossing at 1.5711 the ceiling is p_fil itself; only
        # test_ceiling_caps runs past it, where the opaque arm binds.
        for i in range(1, 30):
            mu = 0.05 * i
            fil, err, best, key = point(mu)
            self.assertLessEqual(key, best, msg=f"rate {key:.8f} over ceiling {best:.8f} at mu={mu}")
            self.assertClose(best, ceiling(mu), msg="the point reports the same ceiling")
            self.assertClose(fil, detect(mu)[0], msg="the point reports the same gain")
            self.assertClose(err, detect(mu)[1], msg="and the same bit error rate")

    def test_linear_loss(self):
        """
        The key rate is linear in channel transmission across the window that scaling is asymptotic in, and departs from it once the gate saturates.
        """
        self.assertClose(log_slope(SLOPE_HI, 0.01), 1.0, atol=5e-3, msg="log-log slope")
        self.assertClose(log_slope(0.1, SLOPE_LO), 1.0, atol=5e-3, msg="over four decades")

        # Linearity is the small-eta*t*mu statement, so the window is asserted, not implied.
        self.assertLess(
            2.0 * ETA_DET * SLOPE_HI * SLOPE_MU * (1.0 + VIS),
            0.05,
            msg="the window keeps the gate far from saturation, at 0.0198 of a photon",
        )
        self.assertGreater(
            abs(log_slope(1.0, SLOPE_LO, mu=1.0) - 1.0),
            5e-3,
            msg="a bright signal over the same decades leaves the tolerance",
        )
        self.assertGreater(
            abs(log_slope(0.1, SLOPE_LO, mu=1.0) - 1.0),
            5e-3,
            msg="and so does one inside the window's transmittances",
        )


class B92Tolerance(Question):
    """
    The shared rate equation and the bit error rate it tolerates.
    """

    def test_rate_form(self):
        """
        The rate is p_fil*(1 - h2(e_ph) - f_ec*h2(e_bit)) clamped at zero, with f_ec = 1 the published form at the Shannon limit.
        """
        for mu in (0.05, 0.5, 1.5):
            fil, err = detect(mu)
            want = fil * (1.0 - h2(E_PHASE) - F_REC * h2(err))
            self.assertClose(
                _core.b92_rate(fil, err, E_PHASE, F_REC),
                max(0.0, want),
                msg=f"rate equation at mu={mu}",
            )
        self.assertClose(
            _core.b92_rate(0.1, 0.4, 0.4, F_REC),
            0.0,
            msg="a rate the entropies have eaten clamps rather than going negative",
        )

        fil, err = detect(0.2)
        self.assertClose(
            _core.b92_rate(fil, err, E_PHASE, 1.0),
            fil * (1.0 - h2(E_PHASE) - h2(err)),
            msg="f_ec = 1 is the published form",
        )

    def test_shor_preskill(self):
        """
        At equal bit and phase error rates and Shannon-limit reconciliation the tolerable error is the Shor-Preskill 11.00%.
        """
        root = entropy_root(0.5)
        self.assertClose(root, SHOR, atol=1e-12, msg="the symmetric threshold")
        self.assertClose(_core.b92_tolerance(root, 1.0), root, atol=1e-12, msg="a fixed point")
        self.assertLess(
            TAMAKI[0.0],
            SHOR,
            msg="B92's published threshold sits well under the symmetric one",
        )
        self.assertLess(KOASHI, TAMAKI[0.0], msg="and the coherent-state one under that")

    def test_tolerance_falls(self):
        """
        The tolerable bit error rate falls as the phase bound rises and is the root of 1 - h2(e_ph) - f_ec*h2(e) = 0.
        """
        seq = [_core.b92_tolerance(e, F_REC) for e in (0.0, 0.01, 0.05, 0.1, 0.2)]
        self.assertMonotone(seq, rising=False, msg="a worse phase bound tolerates less")
        self.assertClose(
            _core.b92_tolerance(0.05, F_REC),
            0.1521420942696841,
            atol=1e-12,
            msg="the exam's operating point",
        )
        self.assertClose(
            _core.b92_tolerance(0.5, F_REC),
            0.0,
            msg="a phase error of 1/2 leaves nothing to tolerate",
        )
        self.assertClose(
            _core.b92_tolerance(0.0, 1.0),
            0.5,
            msg="no phase error and no reconciliation penalty tolerates everything",
        )

        for eph in (0.01, 0.05, 0.1, 0.2):
            for f_ec in (1.0, 1.16, 1.5):
                err = _core.b92_tolerance(eph, f_ec)
                self.assertClose(
                    1.0 - h2(eph) - f_ec * h2(err),
                    0.0,
                    atol=1e-12,
                    msg=f"root at e_ph={eph}, f_ec={f_ec}",
                )

    def test_reconciliation(self):
        """
        Charging error correction above the Shannon limit lowers the tolerable depolarising rate, so the published thresholds are the f_ec = 1 column.
        """
        for loss in (0.0, 0.2, 0.5):
            shannon = _core.b92_limit(loss, 1.0)[0]
            real = _core.b92_limit(loss, F_REC)[0]
            self.assertLess(real, shannon, msg=f"f_ec = {F_REC} costs reach at {loss}")
        self.assertClose(
            _core.b92_limit(0.0, F_REC)[0],
            0.03145428262882888,
            atol=1e-9,
            msg="the zero-loss threshold at this exam's reconciliation",
        )


class B92Strong(Question):
    """
    Strong-reference B92 on the receiver that derives its phase error.
    """

    def test_window_form(self):
        """
        D1's window is centred on the mean photon number of the arriving reference, 3.2 standard deviations each side, holding that many sigma of Poisson weight.
        """
        span = 10 ** (-SRP_DB * 10.0 / 10.0)
        mean = span * SRP_ETA * 1e5
        lo, hi, share = _core.b92_window(SRP_KAPPA, 1e5, span, SRP_ETA, SRP_WIDTH)
        self.assertClose(0.5 * (lo + hi), mean, atol=1.0, msg="the window is not centred on the arriving reference")
        self.assertClose(
            0.5 * (hi - lo),
            SRP_WIDTH * math.sqrt(mean),
            atol=1.0,
            msg="the half-width is not 3.2 standard deviations",
        )
        self.assertClose(share, math.erf(SRP_WIDTH / math.sqrt(2.0)), atol=5e-5, msg="not the normal weight")
        wide = _core.b92_window(SRP_KAPPA, 1e5, span, SRP_ETA, 6.0)[2]
        self.assertGreater(wide, share, msg="a wider window kept no more of the reference")

    def test_monitor_form(self):
        """
        The five monitored rates are Eq. (17) term for term, the windowed three sifted by the Poisson share.
        """
        span = 10 ** (-SRP_DB * 10.0 / 10.0)
        seen = span * SRP_ETA * SRP_KAPPA
        share = _core.b92_window(SRP_KAPPA, 1e5, span, SRP_ETA, SRP_WIDTH)[2]
        rates = _core.b92_monitor(SRP_KAPPA, 1e5, span, SRP_ETA, SRP_NOISE, SRP_WIDTH)
        stray = math.exp(-seen) * SRP_NOISE
        self.assertClose(
            rates[0],
            math.exp(-2.0 * seen) * 2.0 * seen * (1.0 - SRP_NOISE) + stray,
            msg="the conclusive rate is not Eq. (17)",
        )
        self.assertClose(rates[4], 0.5 * stray, msg="the error rate is not half the single-photon branch")
        self.assertClose(rates[1], rates[0] * share, msg="the windowed conclusive rate is not sifted by the share")
        self.assertClose(rates[3], rates[4] * share, msg="the windowed error rate is not sifted by the share")
        self.assertClose(
            rates[2],
            math.exp(-2.0 * seen) * (1.0 - SRP_NOISE) * share,
            msg="the vacuum rate is not Eq. (17)",
        )

    def test_phase_derived(self):
        """
        The phase error is derived rather than supplied: the largest value the implicit Eq. (15) admits, rising with loss and with noise.
        """
        span, mu = 0.05, 1e10
        derived = _core.b92_derived(SRP_KAPPA, mu, span, SRP_ETA, SRP_NOISE, SRP_WIDTH)
        self.assertEqual(
            derived,
            _core.b92_strong(SRP_KAPPA, mu, span, SRP_ETA, SRP_NOISE, SRP_WIDTH, 1.0)[2],
            msg="the rate charged a different phase error than the bound returns",
        )
        self.assertMonotone(
            [strong(dist, mu)[2] for dist in (0.0, 40.0, 80.0, 100.0, 120.0)],
            msg="the derived phase error did not rise with the loss",
        )
        self.assertMonotone(
            [_core.b92_derived(SRP_KAPPA, mu, span, SRP_ETA, p, SRP_WIDTH) for p in (0.0, 1e-6, 1e-5, 1e-4)],
            msg="the derived phase error did not rise with the noise",
        )

    def test_published_reach(self):
        """
        The three curves of Fig. 2 reach zero at 55, 100 and 122 km, reproduced to better than a third of a kilometre with nothing fitted.
        """
        for mu, want in SRP_REACH.items():
            got = crossing(mu)
            self.assertClose(got, want, atol=0.35, msg=f"mu = {mu:.4e} reached {got:.4f} km, not {want}")
        self.assertMonotone(
            [crossing(mu) for mu in sorted(SRP_REACH)],
            msg="a brighter reference did not buy distance",
        )

    def test_reach_saturates(self):
        """
        Raising the reference intensity alone stops buying distance, the paper's own 124 km being a maximum over the signal strength and the window too.
        """
        fixed = crossing(1e14)
        self.assertClose(fixed, 122.28, atol=0.05, msg="the fixed-window saturation moved")
        self.assertLess(fixed - crossing(1e10), 1.0, msg="four more decades of reference bought a kilometre")
        best = crossing(1e14, SRP_BEST[0], SRP_BEST[1])
        self.assertClose(best, SRP_TOP, atol=0.35, msg=f"the parameter-set maximum reached {best:.4f} km")
        self.assertGreater(best, fixed, msg="the Fig. 2 window was already the best one")

    def test_linear_transmission(self):
        """
        The rate is proportional to channel transmission over the span the paper claims it for, steepening only near the crossing.
        """
        mu = 1e10
        near = _core.b92_strong(SRP_KAPPA, mu, 0.3, SRP_ETA, SRP_NOISE, SRP_WIDTH, 1.0)[3]
        far = _core.b92_strong(SRP_KAPPA, mu, 0.1, SRP_ETA, SRP_NOISE, SRP_WIDTH, 1.0)[3]
        self.assertClose(math.log(near / far) / math.log(3.0), 1.0, atol=0.05, msg="not linear in transmission")
        steep = []
        for lo, hi in ((0.1, 0.3), (0.01, 0.05), (0.005, 0.01)):
            top = _core.b92_strong(SRP_KAPPA, mu, hi, SRP_ETA, SRP_NOISE, SRP_WIDTH, 1.0)[3]
            bot = _core.b92_strong(SRP_KAPPA, mu, lo, SRP_ETA, SRP_NOISE, SRP_WIDTH, 1.0)[3]
            steep.append(math.log(top / bot) / math.log(hi / lo))
        self.assertMonotone(steep, msg="the slope did not steepen towards the crossing")

    def test_under_capacity(self):
        """
        This variant reaches at most seven per cent of the repeaterless capacity of the same span, which b92_ceiling beats.
        """
        worst = 0.0
        for dist in range(0, 126, 6):
            for mu in (1e5, 1e10, 1e12):
                worst = max(worst, strong(float(dist), mu)[3] / span_cap(float(dist)))
        self.assertLess(worst, 0.08, msg=f"the derived variant reached {worst:.4f} of capacity")
        self.assertGreater(worst, 0.06, msg="the comparison stopped measuring anything")


class B92Finite(Question):
    """
    Appendix A's deviation bound, and the two refusals standing where a key length would be.
    """

    def test_azuma_inverts(self):
        """
        Both widths put back into Azuma's inequality as the paper scales it return the failure probability asked for.
        """
        code, test = _core.b92_azuma(AZUMA_N, AZUMA_T, AZUMA_EPS)
        self.assertClose(
            azuma_back(code, 1.0 - AZUMA_T) / AZUMA_EPS,
            1.0,
            atol=1e-9,
            msg="the code-pair width did not invert Eq. (A4)",
        )
        self.assertClose(
            azuma_back(test, AZUMA_T) / AZUMA_EPS,
            1.0,
            atol=1e-9,
            msg="the test-pair width did not invert Eq. (A5)",
        )
        root = math.sqrt(2.0 * math.log(2.0 / AZUMA_EPS) / AZUMA_N)
        self.assertClose(code, root / (1.0 - AZUMA_T), msg="the code width is not the shared root over 1 - t")
        self.assertClose(test, root / AZUMA_T, msg="the test width is not the shared root over t")

    def test_azuma_scales(self):
        """
        The width falls as 1/sqrt(N) and rises as sqrt(log(1/eps)): martingale scaling, not Chernoff.
        """
        wide = [_core.b92_azuma(n, AZUMA_T, AZUMA_EPS)[0] for n in (1e6, 1e8, 1e10, 1e12)]
        self.assertMonotone(wide, rising=False, msg="a longer block did not narrow the width")
        for lo, hi in zip(wide, wide[1:]):
            self.assertClose(lo / hi, 10.0, atol=1e-9, msg="two decades of block did not buy one of width")
        tight = [_core.b92_azuma(AZUMA_N, AZUMA_T, e)[1] for e in (1e-4, 1e-8, 1e-12)]
        self.assertMonotone(tight, msg="a smaller failure probability did not widen the interval")
        self.assertClose(
            tight[2] / tight[0],
            math.sqrt(math.log(2e12) / math.log(2e4)),
            atol=1e-12,
            msg="the epsilon dependence is not a square root of a logarithm",
        )

    def test_azuma_splits(self):
        """
        The two widths cross at an even split, are smallest there, and each diverges as its own pool empties.
        """
        even = _core.b92_azuma(AZUMA_N, 0.5, AZUMA_EPS)
        self.assertClose(even[0], even[1], msg="an even split did not give two equal widths")
        worst = [max(_core.b92_azuma(AZUMA_N, t, AZUMA_EPS)) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(min(worst), worst[2], msg="the wider of the two is not smallest at t = 1/2")
        self.assertGreater(
            _core.b92_azuma(AZUMA_N, 1e-4, AZUMA_EPS)[1],
            1e3 * even[1],
            msg="an empty test pool did not blow the test width up",
        )
        self.assertGreater(
            _core.b92_azuma(AZUMA_N, 1.0 - 1e-4, AZUMA_EPS)[0],
            1e3 * even[0],
            msg="an empty code pool did not blow the code width up",
        )

    def test_length_refused(self):
        """
        Neither branch returns a key length, the block size is checked first, and the two refusals differ.
        """
        self.assertFails(
            NotImplementedError,
            "joint confidence region",
            _core.b92_finite,
            AZUMA_N,
            False,
            msg="plain B92 has no finite-key form",
        )
        self.assertFails(
            NotImplementedError,
            "malformed",
            _core.b92_finite,
            AZUMA_N,
            True,
            msg="a supplied phase error has nothing to bound",
        )
        self.assertNotEqual(refusal(True), refusal(False), msg="one message served both branches")
        self.assertFails(
            ValueError,
            "n_total",
            _core.b92_finite,
            0.0,
            True,
            msg="an empty block is a bad argument, not an unimplemented protocol",
        )

    def test_strong_gaps(self):
        """
        The strong-reference refusal enumerates four missing pieces and names b92_azuma, the paper it stops inside and the asymptotic argument its rate is read off.
        """
        text = refusal(True)
        for needle in ("(1)", "(2)", "(3)", "(4)"):
            self.assertIn(needle, text, msg=f"the refusal enumerates its pieces, missing {needle}")
        self.assertIn("b92_azuma", text, msg="the one finite-N width that ships is not named")
        self.assertIn("a slightly relaxed version", text, msg="the paper's own words for the gap are not quoted")
        self.assertIn("Phys. Rev. A 80, 032302 (2009)", text, msg="the derived branch's paper is not named")
        self.assertIn(
            "Phys. Rev. Lett. 85, 441 (2000)",
            text,
            msg="the asymptotic argument Eq. (16) is read off is not named",
        )
        self.assertIn("smooth min-entropy", text, msg="the missing step is not named as the entropy it needs")

    def test_plain_sources(self):
        """
        The plain refusal names three published finite-key analyses of a different B92, flags the journal-only one as unread, and separates them from the extended-B92 proofs.
        """
        text = refusal(False)
        for needle in (
            "arXiv:1504.05628",
            "Phys. Rev. A 88, 062306 (2013)",
            "npj Quantum Information 6, 104 (2020)",
        ):
            self.assertIn(needle, text, msg=f"a published analysis is unnamed: {needle}")
        self.assertIn("THREE", text, msg="the count of published analyses did not move with the sources")
        self.assertIn("NO arXiv version", text, msg="the journal-only source is not flagged as one")
        self.assertIn("SUM TO THE IDENTITY", text, msg="the lossless POVM is not named as the blocker it is")
        self.assertIn("COLLECTIVE", text, msg="the attack class of the ISIT analysis is not stated")
        for needle in ("arXiv:2001.05940", "arXiv:2510.11488", "Phys. Rev. A 80, 032327 (2009)"):
            self.assertIn(needle, text, msg=f"the extended-B92 route is unnamed: {needle}")


if __name__ == "__main__":
    rc = Exam(
        "B92States",
        "Two non-orthogonal states, Bob's displacement receiver, and Eve's discrimination",
        "b92_states.md",
    ).run(load(B92States))
    rc |= Exam(
        "B92Tamaki",
        "The Tamaki-Lutkenhaus bound: a derived phase error and the published thresholds",
        "b92_tamaki.md",
    ).run(load(B92Tamaki))
    rc |= Exam(
        "B92Reach",
        "Where plain B92 stops, in transmittance, decibels and kilometres",
        "b92_reach.md",
    ).run(load(B92Reach))
    rc |= Exam(
        "B92Guards",
        "Every argument domain, and a plain protocol that is priced rather than refused",
        "b92_guards.md",
    ).run(load(B92Guards))
    rc |= Exam(
        "B92Overlap",
        "The optimum overlap, where Bob's discrimination crosses Eve's",
        "b92_overlap.md",
    ).run(load(B92Overlap))
    rc |= Exam(
        "B92Strong",
        "Strong-reference B92 with the phase error derived, against Phys. Rev. A 80, 032302 (2009)",
        "b92_strong.md",
    ).run(load(B92Strong))
    rc |= Exam(
        "B92Tolerance",
        "The shared rate equation and the bit error rate it tolerates",
        "b92_tolerance.md",
    ).run(load(B92Tolerance))
    rc |= Exam(
        "B92Finite",
        "Appendix A's deviation bound, and the key length no B92 paper writes",
        "b92_finite.md",
    ).run(load(B92Finite))
    sys.exit(rc)
