import dataclasses
import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.forms import bisect
import qkd as q
from qkd import _core, attacks

# Loss-tolerant QKD with state-preparation flaws. Tamaki, Curty, Kato, Lo & Azuma,
# "Loss-tolerant quantum cryptography with imperfect sources", Phys. Rev. A 90, 052314
# (2014), arXiv:1312.3514 is the primary source; the device model and the four-state form
# are Pereira, Curty & Tamaki, "Quantum key distribution with flawed and leaky sources",
# npj Quantum Information 5, 62 (2019), arXiv:1902.02126, doi 10.1038/s41534-019-0180-9,
# whose Eq. (33) and Sec. IV A Y_Z these exams reproduce.
#
# arXiv:1902.02126 IS NOT "WANG, TAMAKI & CURTY" AND ITS ARTICLE NUMBER IS 62, NOT 64;
# Wang, Tamaki & Curty is Sci. Rep. 11, 1678 (2021), on MDI-QKD with leaky sources.
# `dev/novel.md` W10.
#
# Both flaw magnitudes are the papers': Tamaki's caption calls 0.063 rad "an experimentally
# available value" and cites Honjo, Inoue & Takahashi, Opt. Lett. 29, 23 (2004) for it.
DELTA_A = 0.063
DELTA_B = 0.126

# Pereira Sec. IV. Her Y_Z carries P_ZA P_ZB; the engine takes it separately as q_sift.
P_DARK = 1e-7
F_FLAW = 1.16
Q_SIFT = 0.25

# Both papers' channel models are written at tilt = -delta, Pereira Eq. (E2) fixing it;
# TILT_FLAT = 0 is in neither. tol_rate and std_rate default to -delta.
TILT_FLAT = 0.0

# Zero crossings, MEASURED by bisection over [0, 90] dB at DELTA_A, P_DARK, F_FLAW.
# THE TILT IS PART OF THE ANCHOR: at 28.766343 dB the standard rate is 2.4e-12 at
# tilt = -delta and 1.3e-5 at tilt = 0. Pinned at 1e-6, the rounding of a six-decimal
# constant, NOT a model tolerance.
ZERO_TOL = 57.775328
ZERO_STD = 28.766343
FLAT_TOL = 57.843443
FLAT_STD = 29.659930


def loss_db(db):
    """
    Overall transmittance at a system loss of ``db`` decibels, the axis both papers plot.
    """
    return 10.0 ** (-db / 10.0)


def pereira_bit(delta, eta, dark):
    """
    Pereira, Curty & Tamaki arXiv:1902.02126 Eq. (33) for the Z-basis bit error rate, typed from the paper.
    """
    top = (
        2.0 * (1.0 - eta / 2.0) * dark
        + eta / 2.0
        + (eta / 4.0) * (math.cos(2.0 * delta) + math.cos(delta)) * (dark - 1.0)
    )

    return top / (4.0 * (1.0 - eta / 2.0) * dark + eta)


def pereira_yield(eta, dark):
    """
    Pereira, Curty & Tamaki arXiv:1902.02126 Sec. IV A single-photon Z yield, less the P_ZA P_ZB the engine
    takes as q_sift.
    """
    return 4.0 * (1.0 - eta / 2.0) * dark + eta


def coin_trig(e_bit, coin):
    """
    Tamaki, Lo, Fung & Qi Eq. (6) as trigonometry: with e = sin^2(theta) the phase error is sin^2 of a sum of
    angles.
    """
    span = math.acos(max(-1.0, 1.0 - 2.0 * coin))

    return min(0.5, math.sin(math.asin(math.sqrt(e_bit)) + span) ** 2)


def tamaki_coef(g):
    """
    Tamaki arXiv:1312.3514 Appendix C's virtual-state coefficients C_00, C_10, C_01, C_11 at flaw ``g``, typed
    from the paper.
    """
    s, k = math.sin(g / 2.0), math.cos(g / 2.0)

    return (
        (1.0 + s + k) / (2.0 * math.sqrt(1.0 + s)),
        (1.0 + s - k) / (2.0 * math.sqrt(1.0 + s)),
        (1.0 - s - k) / (2.0 * math.sqrt(1.0 - s)),
        (1.0 - s + k) / (2.0 * math.sqrt(1.0 - s)),
    )


def tamaki_phase(delta):
    """
    Tamaki Appendix C's phase error rate at zero dark count, [C_10(3d/2)^2 + C_01(3d/2)^2]/2 on his 3d/2
    substitution for Bob's modulator.
    """
    _, c10, c01, _ = tamaki_coef(1.5 * delta)

    return (c10 * c10 + c01 * c01) / 2.0


def tamaki_excess(dark):
    """
    Tamaki's Appendix C channel at zero loss, minus one: 2.5*dark - dark^2 against Pereira's 2*dark.
    """
    return 2.5 * dark - dark * dark


# Mizutani, Curty, Lim, Imoto & Tamaki, "Finite-key security analysis of quantum key
# distribution with imperfect light sources", New J. Phys. 17, 093011 (2015),
# arXiv:1504.08151, is the finite-key analysis these exams run. Its Sec. V models the two
# modulators as dtheta_A = xi*theta_A/pi and dtheta_B = -dtheta_A -- this module's delta at
# tilt = -delta and no other -- and calls xi = 0.147 "a phase modulation error of 8.42
# degrees".
#
# 0.147 IS THE PREPRINT'S NUMBER AND arXiv v2's IS 0.145. Mizutani attributes it to
# "an updated version of a commercial plug&play system (ID Quantique Clavis2)" and cites
# the 2014 e-print of Xu, Wei, Sajeed, Kaiser, Sun, Tang, Qian, Makarov & Lo,
# "Experimental quantum key distribution with source flaws", Phys. Rev. A 92, 032305
# (2015), arXiv:1408.3667, whose v1 reads "delta <= 0.147" and whose v2 reads 0.145. The
# APS record itself was not read (link.aps.org returns 403).
#
# THE DECOY LAYER OF THEIR SEC. IV IS NOT IMPLEMENTED and no figure of theirs is reproduced.
# What is checked is their EQUATIONS: Eq. (37) against a numerical inverse, Eq. (36) against
# the asymptotic estimator it collapses onto, Lemma 4's g_A as typed, and Eq. (43) term by
# term.
MIZ_FLAW = 0.147
MIZ_DEG = 8.42

# pz*pz is Q_SIFT, so the finite length per emitted photon and the asymptotic rate are taken
# on one sifting share.
PZ_EVEN = 0.5

# eps_sec and eps_c are Mizutani Sec. V's; EPS_PH is this exam's and sits below eps_s^2.
EPS_S = 1e-8
EPS_C = 1e-15
EPS_PH = 1e-18

# Where the finite length crosses zero, MEASURED by bisection at DELTA_A, tilt = -delta,
# P_DARK, F_FLAW, PZ_EVEN and the three epsilons above. Mizutani plots decoy-state curves
# and this module has no decoy layer. Pinned at 1e-6, the rounding of a six-decimal
# constant, not a model tolerance.
BLOCK_TEN = 44.905985
BLOCK_TWELVE = 55.045740
BLOCK_FOURTEEN = 57.448692


def loss_eta(delta, tilt, eta, dark):
    """
    Bob's conditional X-basis outcome split for Alice's three sent states, the six counts Mizutani Eq. (36)
    reads, at unit block size.
    """
    turn = tilt + math.pi / 2.0 + delta / 2.0
    seen = []
    for sig in _core.flaw_angles(delta)[:3]:
        p0 = (1.0 + math.cos(sig - turn)) / 2.0
        floor = (1.0 - eta / 2.0) * dark
        gain = eta / 2.0 * (1.0 - dark / 2.0)
        both = eta / 4.0 * dark
        q0 = floor + gain * p0 + both * (1.0 - p0)
        q1 = floor + gain * (1.0 - p0) + both * p0
        seen.append((q0 / (q0 + q1), q1 / (q0 + q1)))

    return seen


def miz_counts(delta, eta, dark, tilt, pz, n1):
    """
    The six counts Lambda_(Omega,s) an honest channel gives, in Mizutani's order: Omega = 3, 4, 5 against
    Bob's two X outcomes, weighted by Q(Omega).
    """
    share = miz_share(pz)
    seen = loss_eta(delta, tilt, eta, dark)
    lower = [n1 * share[l] * seen[l][0] for l in range(3)]
    upper = [n1 * share[l] * seen[l][1] for l in range(3)]

    return lower + upper


def miz_share(pz):
    """
    Mizutani's Q(3) = Q(4) = pz(1-pz)/2 and Q(5) = (1-pz)^2, the setting probabilities his Eqs. (97)-(99)
    divide each count by.
    """
    px = 1.0 - pz

    return [pz * px / 2.0, pz * px / 2.0, px * px]


def miz_coefs(delta):
    """
    Mizutani's C_(1,l) for a pure qubit source, assembled from flaw_inverse and his Eq. (34).
    """
    ang = _core.flaw_angles(delta)
    inv = _core.flaw_inverse(delta)
    first = (math.cos(ang[0] / 2.0), math.sin(ang[0] / 2.0))
    second = (math.cos(ang[1] / 2.0), math.sin(ang[1] / 2.0))
    weights = (
        first[0] * second[0] + first[1] * second[1],
        first[0] * second[1] + first[1] * second[0],
        first[0] * second[0] - first[1] * second[1],
    )

    return [sum(weights[i] * inv[3 * i + l] for i in range(3)) for l in range(3)]


def miz_slack(delta, pz, n1, eps):
    """
    The width Mizutani Eq. (36) adds over its point estimate: each coefficient's magnitude against the Azuma
    half-width, plus one Delta per virtual state.
    """
    coefs = miz_coefs(delta)
    share = miz_share(pz)
    weight = pz * pz / 4.0
    wide = _core.flaw_azuma(n1, _core.flaw_eps(eps))
    total = 0.0
    for sign in (1.0, -1.0):
        rows = [weight * (1.0 + sign * coefs[0]), weight * (1.0 + sign * coefs[1]), weight * sign * coefs[2]]
        total += sum(abs(rows[l]) * wide / share[l] for l in range(3)) + wide

    return total


def finite_rate(delta, db, n_emit, tilt=None):
    """
    Mizutani's finite key length over ``n_emit`` emitted single photons at flaw ``delta`` and loss ``db``, at
    tilt = -delta.
    """
    turn = -delta if tilt is None else tilt

    return _core.flaw_finite(delta, loss_db(db), P_DARK, turn, PZ_EVEN, n_emit, F_FLAW, EPS_S, EPS_C, EPS_PH)


def tol_rate(delta, db, tilt=None):
    """
    Loss-tolerant key rate at flaw ``delta`` and system loss ``db``, on the shared setting.
    """
    turn = -delta if tilt is None else tilt

    return _core.flaw_tolerant(delta, loss_db(db), P_DARK, turn, F_FLAW, Q_SIFT)


def std_rate(delta, db, tilt=None):
    """
    Standard, non-loss-tolerant key rate at the same flaw, loss and setting.
    """
    turn = -delta if tilt is None else tilt

    return _core.flaw_standard(delta, loss_db(db), P_DARK, turn, F_FLAW, Q_SIFT)


# The overall transmittance the engine reads is the channel times the detector's own
# efficiency, so LINK_ETA is what the engine is called with when the two are compared.
LINK_T = 0.5
LINK_ETA = LINK_T * 0.9


def flaw_link(**kw):
    """
    The flawed-source link at the published configuration, tilt = -delta.
    """
    flaw = kw.pop("flaw", None) or q.SourceFlaw.opposed(DELTA_A)
    detector = kw.pop("detector", None) or q.ClickDetector(eta=0.9, dark=P_DARK)

    return q.Link(
        modulation=q.FlawedKeying(flaw=flaw, sift=Q_SIFT),
        channel=q.Channel(T=kw.pop("t", LINK_T)),
        bob=q.Bob(detector=detector, receiver=kw.pop("receiver", None)),
        security=kw.pop("security", None)
        or q.FlawBound(
            analysis=kw.pop("analysis", "tolerant"),
            f=F_FLAW,
            block=kw.pop("block", None),
        ),
        **kw,
    )


class FlawGeometry(Question):
    """
    Alice's flawed states on the Bloch sphere.
    """

    def test_ideal_rectangle(self):
        """
        `flaw_angles(0)` is the ideal BB84 set and at any flaw the four deviations over delta are 0, 1, 1/2
        and 3/2.
        """
        got = _core.flaw_angles(0.0)
        want = (0.0, math.pi, math.pi / 2, 3.0 * math.pi / 2)
        for a, b in zip(got, want):
            self.assertClose(a, b, atol=0.0, msg=f"ideal angle {a} != {b}")

        for delta in (DELTA_A, DELTA_B, 0.5):
            got = _core.flaw_angles(delta)
            ideal = _core.flaw_angles(0.0)
            for k, want in enumerate((0.0, 1.0, 0.5, 1.5)):
                share = (got[k] - ideal[k]) / delta
                self.assertClose(
                    share,
                    want,
                    atol=1e-14,
                    msg=f"deviation share {k} at delta={delta} is {share}",
                )

    def test_virtual_pair(self):
        """
        The two virtual X states stay antipodal, both rotate by delta/2, and their priors sum to 1 and split
        by sin(delta/2)/2.
        """
        for delta in (0.0, DELTA_A, 0.5, 2.0):
            p0, t0, p1, t1 = _core.flaw_virtual(delta)
            half = math.sin(delta / 2.0)
            self.assertClose(p0 + p1, 1.0, atol=1e-15, msg=f"priors at {delta}")
            self.assertClose(t1 - t0, math.pi, atol=1e-14, msg=f"not antipodal at {delta}")
            self.assertClose(
                t0 - math.pi / 2.0,
                delta / 2.0,
                atol=1e-14,
                msg=f"virtual rotation at {delta}",
            )
            self.assertClose(
                (p1 - p0) / 2.0,
                half / 2.0,
                atol=1e-15,
                msg=f"prior split at {delta}",
            )

    def test_triangle_open(self):
        """
        `flaw_triangle` is 2 at zero flaw, peaks at 3*sqrt(3)/2 at delta = pi/3 and stays positive out to
        3.14.
        """
        dets = [_core.flaw_triangle(k * math.pi / 60.0) for k in range(60)]
        self.assertClose(dets[0], 2.0, atol=1e-15, msg="ideal determinant")
        self.assertClose(
            _core.flaw_triangle(math.pi / 3.0),
            1.5 * math.sqrt(3.0),
            atol=1e-14,
            msg="peak determinant",
        )
        self.assertGreater(min(dets), 0.0, msg=f"min determinant {min(dets)}")
        self.assertLess(_core.flaw_triangle(3.14), 4e-3, msg="determinant near pi")

    def test_fidelity_closed(self):
        """
        `flaw_fidelity` matches sqrt(1 - [sin^2(delta/2) + sin^3(delta/2)]/2) to 1e-15 and is exactly 1 at
        zero flaw.
        """
        self.assertClose(_core.flaw_fidelity(0.0), 1.0, atol=0.0, msg="ideal fidelity")

        for k in range(40):
            delta = k * 0.07
            half = math.sin(delta / 2.0)
            want = math.sqrt(1.0 - (half * half + half**3) / 2.0)
            self.assertClose(
                _core.flaw_fidelity(delta),
                want,
                atol=1e-15,
                msg=f"fidelity closed form at delta={delta}",
            )


class LossTolerant(Question):
    """
    Tamaki's theorem and the loss independence it buys.
    """

    def test_theorem_exact(self):
        """
        Inverting the rejected data and measuring the virtual states give the same phase error rate to 1e-15
        over flaw, loss, dark count and tilt.
        """
        worst = 0.0
        for k in range(16):
            delta = k * 0.19
            for eta in (1.0, 0.15, 1e-3, 1e-6):
                for dark in (0.0, P_DARK, 1e-3):
                    for tilt in (0.0, -delta, 0.3, -1.0):
                        got = _core.flaw_phase(delta, eta, dark, tilt)
                        arb = _core.flaw_direct(delta, eta, dark, tilt)
                        worst = max(worst, abs(got - arb))
        self.assertLess(worst, 1e-15, msg=f"inversion drifted from the arbiter by {worst}")

    def test_phase_flat(self):
        """
        The phase error is 9.920716e-04 at 0 dB against 2.979953e-03 at 40 dB, and is loss-independent at zero
        dark count.
        """
        near = _core.flaw_phase(DELTA_A, loss_db(0.0), P_DARK, -DELTA_A)
        far = _core.flaw_phase(DELTA_A, loss_db(40.0), P_DARK, -DELTA_A)
        self.assertClose(near, 9.920716e-04, atol=1e-9, msg=f"phase at 0 dB {near}")
        self.assertClose(far, 2.979953e-03, atol=1e-8, msg=f"phase at 40 dB {far}")
        self.assertLess(far / near, 3.1, msg=f"phase grew {far / near}x over 40 dB")

        clean = [_core.flaw_phase(DELTA_A, loss_db(db), 0.0, -DELTA_A) for db in (0.0, 55.0)]
        self.assertClose(
            clean[1],
            clean[0],
            atol=1e-15,
            msg=f"dark-free phase moved with loss: {clean}",
        )

    def test_phase_cancels(self):
        """
        At tilt = 0 the phase error is independent of the flaw to 1e-15 across the domain.
        """
        base = _core.flaw_phase(0.0, 0.15, P_DARK, TILT_FLAT)
        for k in range(30):
            delta = k * 0.1
            got = _core.flaw_phase(delta, 0.15, P_DARK, TILT_FLAT)
            self.assertClose(got, base, atol=1e-15, msg=f"phase moved at delta={delta}")

    def test_phase_reads_tilt(self):
        """
        The dark-free phase error is sin^2(tilt/2) at every flaw and transmittance, reading (1 - cos delta)/2
        at the published tilt = -delta.
        """
        for delta in (DELTA_A, DELTA_B, 0.5, 1.0):
            got = _core.flaw_phase(delta, 0.15, 0.0, -delta)
            self.assertClose(
                got,
                (1.0 - math.cos(delta)) / 2.0,
                atol=1e-15,
                msg=f"added-flaw phase at delta={delta} is {got}",
            )

        for delta in (0.0, DELTA_A, DELTA_B, 0.5, 1.5):
            for turn in (0.0, -DELTA_A, 0.2, -0.7, 1.1):
                for eta in (1.0, 0.3, 1e-4):
                    got = _core.flaw_phase(delta, eta, 0.0, turn)
                    self.assertClose(
                        got,
                        math.sin(turn / 2.0) ** 2,
                        atol=1e-15,
                        msg=f"phase at delta={delta}, tilt={turn}, eta={eta} is {got}",
                    )

    def test_tamaki_ratio(self):
        """
        The engine's dark-free sin^2(delta/2) and Tamaki Appendix C's sin^2(3*delta/8) stand in a 4/3 ratio of
        error angles, 9.919218567e-04 against 5.580367924e-04 at DELTA_A.
        """
        for delta in (DELTA_A, DELTA_B, 0.25, 0.5, 1.0):
            got = _core.flaw_phase(delta, 0.15, 0.0, -delta)
            printed = tamaki_phase(delta)
            self.assertClose(
                printed,
                math.sin(3.0 * delta / 8.0) ** 2,
                atol=1e-15,
                msg=f"Tamaki's coefficients are not sin^2(3d/8) at {delta}",
            )

            ratio = math.asin(math.sqrt(got)) / math.asin(math.sqrt(printed))
            self.assertClose(
                ratio,
                4.0 / 3.0,
                atol=1e-13,
                msg=f"error angles differ by {ratio}, not 4/3, at delta={delta}",
            )
        self.assertClose(
            _core.flaw_phase(DELTA_A, 0.15, 0.0, -DELTA_A),
            9.919218567e-04,
            atol=1e-12,
            msg="the published-configuration phase error moved",
        )
        self.assertClose(
            tamaki_phase(DELTA_A),
            5.580367924e-04,
            atol=1e-12,
            msg="Tamaki's printed phase error moved",
        )

    def test_basis_flat(self):
        """
        The single-photon detection probability does not move with the flaw or the tilt to 1e-16.
        """
        base = _core.flaw_channel(0.0, 0.15, P_DARK, 0.0)[0]
        for delta in (0.0, 0.5, 2.0, 3.0):
            for tilt in (0.0, -delta, 1.1, -2.2):
                got = _core.flaw_channel(delta, 0.15, P_DARK, tilt)[0]
                self.assertClose(
                    got,
                    base,
                    atol=1e-16,
                    msg=f"yield moved at delta={delta}, tilt={tilt}",
                )

    def test_pereira_channel(self):
        """
        `flaw_channel` at tilt = -delta reproduces Pereira, Curty & Tamaki's Y_Z and Eq. (33) bit error rate
        to a part in 1e9 wherever their form stays below 1/2.
        """
        worst = 0.0
        for k in range(35):
            delta = k * 0.05
            for eta in (1.0, 0.5, 0.15, 1e-2, 1e-4):
                for dark in (0.0, P_DARK, 1e-5, 1e-3):
                    y1, e_bit = _core.flaw_channel(delta, eta, dark, -delta)
                    want = pereira_bit(delta, eta, dark)
                    self.assertClose(
                        y1,
                        pereira_yield(eta, dark),
                        atol=1e-15,
                        msg=f"yield at delta={delta}, eta={eta}, dark={dark}",
                    )

                    if want <= 0.5:
                        worst = max(worst, abs(e_bit - want) / max(1e-300, want))
        self.assertLess(worst, 1e-9, msg=f"bit error drifted from Eq. (33) by {worst}")

    def test_model_excess(self):
        """
        Pereira's yield overshoots a probability by 2*dark at unit efficiency against Tamaki Appendix C's
        2.5*dark - dark^2.
        """
        for dark in (1e-7, 1e-4, 1e-2):
            got = _core.flaw_channel(0.4, 1.0, dark, 0.3)[0]
            self.assertClose(
                got - 1.0,
                2.0 * dark,
                atol=1e-15,
                msg=f"excess at dark={dark} is {got - 1.0}",
            )
            self.assertGreater(
                tamaki_excess(dark),
                got - 1.0,
                msg=f"Tamaki's excess is not the larger at dark={dark}",
            )
        self.assertClose(
            tamaki_excess(1e-2) / 1e-2,
            2.49,
            atol=1e-15,
            msg="Tamaki's excess is not 2.5*dark - dark^2",
        )


class CoinAmplified(Question):
    """
    The standard analysis, and the division by the yield that collapses it with loss.
    """

    def test_coin_divides(self):
        """
        The coin imbalance is 5.269405e-04 at unit yield and 10 dB of loss multiplies it by exactly ten.
        """
        base = _core.flaw_coin(DELTA_B, 1.0)
        self.assertClose(base, 5.269405e-04, atol=1e-9, msg=f"source imbalance {base}")

        for db in (10.0, 20.0, 25.0):
            got = _core.flaw_coin(DELTA_B, loss_db(db))
            self.assertClose(
                got,
                base * 10.0 ** (db / 10.0),
                atol=1e-14,
                msg=f"coin at {db} dB is {got}",
            )

    def test_coin_source(self):
        """
        The imbalance is half the fidelity deficit of Alice's two bases at unit yield and scales as
        delta^2/32.
        """
        for delta in (0.01, DELTA_A, DELTA_B):
            got = _core.flaw_coin(delta, 1.0)
            self.assertClose(
                got,
                (1.0 - _core.flaw_fidelity(delta)) / 2.0,
                atol=1e-16,
                msg=f"imbalance at delta={delta}",
            )
            self.assertClose(
                got / (delta * delta),
                1.0 / 32.0,
                atol=2e-3,
                msg=f"quadratic scaling at delta={delta}",
            )

    def test_coin_trig(self):
        """
        `flaw_gllp` agrees with the trigonometric form of the Bloch-sphere bound to 1e-13 over the whole
        square.
        """
        worst = 0.0
        for e_bit in (0.0, 1e-6, 0.01, 0.05, 0.2, 0.5):
            for coin in (0.0, 1e-6, 1e-3, 0.05, 0.2, 0.45, 0.5):
                got = _core.flaw_gllp(e_bit, coin)
                worst = max(worst, abs(got - coin_trig(e_bit, coin)))
        self.assertLess(worst, 1e-13, msg=f"algebra and trigonometry differ by {worst}")

    def test_coin_ends(self):
        """
        `flaw_gllp` returns e_bit at a zero coin and 1/2 at a vacuous one, rising weakly between and never
        dipping below e_bit.
        """
        for e_bit in (0.0, 1e-6, 0.02, 0.3, 0.5):
            self.assertClose(
                _core.flaw_gllp(e_bit, 0.0),
                e_bit,
                atol=0.0,
                msg=f"balanced coin moved e_bit={e_bit}",
            )
            self.assertClose(
                _core.flaw_gllp(e_bit, 0.5),
                0.5,
                atol=0.0,
                msg=f"vacuous coin at e_bit={e_bit}",
            )

        seq = [_core.flaw_gllp(0.01, k * 0.01) for k in range(51)]
        self.assertMonotone(seq, rising=True, strict=False, msg="bound fell with the coin")
        self.assertGreaterEqual(min(seq), 0.01, msg="bound dipped below the bit error")


class TwoAnalyses(Question):
    """
    The two rates side by side.
    """

    def test_ideal_bb84(self):
        """
        At zero flaw both rates equal `bb84_rate` to 1e-15 and the phase error equals the bit error to 2e-16.
        """
        for db in (10.0, 20.0, 40.0):
            eta = loss_db(db)
            y1, e_bit = _core.flaw_channel(0.0, eta, P_DARK, 0.0)
            phase = _core.flaw_phase(0.0, eta, P_DARK, 0.0)
            want = _core.bb84_rate(Q_SIFT, y1, e_bit, y1, phase, F_FLAW)
            self.assertClose(phase, e_bit, atol=2e-16, msg=f"phase != bit at {db} dB")
            self.assertClose(
                tol_rate(0.0, db, 0.0)[0] / want,
                1.0,
                atol=1e-15,
                msg=f"tolerant rate left bb84_rate at {db} dB",
            )
            self.assertClose(
                std_rate(0.0, db, 0.0)[0] / want,
                1.0,
                atol=1e-15,
                msg=f"standard rate left bb84_rate at {db} dB",
            )

    def test_shape(self):
        """
        The loss-tolerant rate is `bb84_rate`'s Shor-Preskill shape on this engine's observables to 1e-15.
        """
        for delta in (0.0, DELTA_A, DELTA_B, 0.4):
            for db in (5.0, 20.0):
                rate, y1, e_bit, phase = tol_rate(delta, db)
                self.assertClose(
                    rate / _core.bb84_rate(Q_SIFT, y1, e_bit, y1, phase, F_FLAW),
                    1.0,
                    atol=1e-15,
                    msg=f"shape drift at delta={delta}, {db} dB",
                )

    def test_bound_order(self):
        """
        The loss-tolerant rate is never below the standard one across 30 flaws and 24 losses.
        """
        for k in range(30):
            delta = k * 0.02
            for j in range(24):
                db = j * 2.5

                was = std_rate(delta, db)[0]
                self.assertGreaterEqual(
                    tol_rate(delta, db)[0],
                    was - 1e-13 * was,
                    msg=f"order broke at delta={delta}, {db} dB",
                )

    def test_crossover(self):
        """
        The two analyses stop at 28.766343 and 57.775328 dB at tilt = -delta and at 29.659930 and 57.843443 at
        tilt = 0, gaps of 29.008984 and 28.183513, against a shared 57.855244 dB at zero flaw.
        """
        got = bisect(lambda db: tol_rate(DELTA_A, db)[0], 0.0, 90.0)
        was = bisect(lambda db: std_rate(DELTA_A, db)[0], 0.0, 90.0)
        self.assertClose(got, ZERO_TOL, atol=1e-6, msg=f"tolerant reach {got} dB")
        self.assertClose(was, ZERO_STD, atol=1e-6, msg=f"standard reach {was} dB")
        self.assertClose(got - was, 29.008984, atol=1e-6, msg=f"gap is {got - was} dB")

        flat = bisect(lambda db: tol_rate(DELTA_A, db, TILT_FLAT)[0], 0.0, 90.0)
        turn = bisect(lambda db: std_rate(DELTA_A, db, TILT_FLAT)[0], 0.0, 90.0)
        self.assertClose(flat, FLAT_TOL, atol=1e-6, msg=f"tolerant reach {flat} dB")
        self.assertClose(turn, FLAT_STD, atol=1e-6, msg=f"standard reach {turn} dB")
        self.assertClose(flat - turn, 28.183513, atol=1e-6, msg=f"gap is {flat - turn} dB")
        self.assertGreater(
            tol_rate(DELTA_A, ZERO_TOL, TILT_FLAT)[0],
            0.0,
            msg="tilt = 0 dead at ZERO_TOL",
        )
        self.assertGreater(
            std_rate(DELTA_A, ZERO_STD, TILT_FLAT)[0],
            1e-6,
            msg="tilt = 0 dead at ZERO_STD",
        )
        self.assertLess(
            std_rate(DELTA_A, ZERO_STD)[0],
            1e-11,
            msg="standard not near zero at ZERO_STD",
        )

        pure = bisect(lambda db: tol_rate(0.0, db, 0.0)[0], 0.0, 90.0)
        bare = bisect(lambda db: std_rate(0.0, db, 0.0)[0], 0.0, 90.0)
        self.assertClose(pure, bare, atol=1e-9, msg=f"reaches differ: {pure} vs {bare}")
        self.assertClose(pure, 57.855244, atol=1e-5, msg=f"ideal reach {pure} dB")

    def test_flaw_sweep(self):
        """
        At 20 dB the loss-tolerant rate falls by 1.1491x from zero flaw to 0.126 rad where the standard one
        falls by 22.096x.
        """
        rows = [(delta, tol_rate(delta, 20.0)[0], std_rate(delta, 20.0)[0]) for delta in (0.0, DELTA_A, DELTA_B)]
        self.assertClose(rows[0][1], 2.4982621e-03, atol=1e-9, msg="tolerant at zero flaw")
        self.assertClose(rows[1][1], 2.3982251e-03, atol=1e-9, msg="tolerant at 0.063")
        self.assertClose(rows[2][1], 2.1740307e-03, atol=1e-9, msg="tolerant at 0.126")
        self.assertClose(rows[1][2], 1.4697135e-03, atol=1e-9, msg="standard at 0.063")
        self.assertClose(rows[2][2], 1.1306666e-04, atol=1e-10, msg="standard at 0.126")
        self.assertClose(
            rows[0][1] / rows[2][1],
            1.1491,
            atol=1e-3,
            msg=f"tolerant ratio {rows[0][1] / rows[2][1]}",
        )
        self.assertClose(
            rows[0][2] / rows[2][2],
            22.096,
            atol=1e-2,
            msg=f"standard ratio {rows[0][2] / rows[2][2]}",
        )

    def test_loss_sweep(self):
        """
        The loss-tolerant rate falls a decade per 10 dB, 2.3991163e-01 at 0 dB against 2.3902135e-04 at 30,
        where the standard one is exactly zero by 30.
        """
        rows = [tol_rate(DELTA_A, db)[0] for db in (0.0, 10.0, 20.0, 30.0)]
        self.assertClose(rows[0], 2.3991163e-01, atol=1e-8, msg="tolerant at 0 dB")
        self.assertClose(rows[3], 2.3902135e-04, atol=1e-10, msg="tolerant at 30 dB")

        for step, want in enumerate((10.000338, 10.003378, 10.033518)):
            slope = rows[step] / rows[step + 1]
            self.assertClose(
                slope,
                want,
                atol=1e-5,
                msg=f"tolerant slope over step {step} is {slope}",
            )
        self.assertGreater(std_rate(DELTA_A, 20.0)[0], 0.0, msg="standard dead at 20 dB")
        self.assertClose(
            std_rate(DELTA_A, 30.0)[0],
            0.0,
            atol=0.0,
            msg="standard still distilling at 30 dB",
        )

    def test_not_conflatable(self):
        """
        The two fourth slots are different quantities, a coin imbalance against a phase error, and the bound
        built from the coin sits above the exact value.
        """
        for delta in (DELTA_A, DELTA_B):
            for db in (10.0, 20.0):
                phase = tol_rate(delta, db)[3]
                coin = std_rate(delta, db)[3]
                bound = _core.flaw_gllp(std_rate(delta, db)[2], coin)
                self.assertNotAlmostEqual(
                    phase,
                    coin,
                    places=4,
                    msg=f"coin read as a phase error at delta={delta}, {db} dB",
                )
                self.assertGreater(
                    bound,
                    phase,
                    msg=f"standard bound not above the exact value at {db} dB",
                )

    def test_clamped_floor(self):
        """
        Both rates clamp at zero with the fourth slot still reporting: a phase error above 0.48, a coin of
        exactly 0.5.
        """
        rate, _, _, phase = tol_rate(0.5, 80.0)
        self.assertClose(rate, 0.0, atol=0.0, msg="tolerant not clamped")
        self.assertGreater(phase, 0.48, msg=f"phase {phase}")

        rate, _, e_bit, coin = std_rate(0.5, 80.0)
        self.assertClose(rate, 0.0, atol=0.0, msg="standard not clamped")
        self.assertClose(coin, 0.5, atol=0.0, msg=f"coin {coin}")
        self.assertClose(
            _core.flaw_gllp(e_bit, coin),
            0.5,
            atol=0.0,
            msg="gllp below 0.5 at a vacuous coin",
        )


class FlawGuards(Guarded):
    """
    Argument domains of the loss-tolerant engine.
    """

    def test_flaw_domain(self):
        """
        `flaw_angles`, `flaw_virtual` and `flaw_triangle` refuse a delta outside [0, pi), pi included.
        """
        for fn in (_core.flaw_angles, _core.flaw_virtual, _core.flaw_triangle):
            for bad in (-1e-9, math.pi, 4.0, float("nan"), float("inf")):
                self.assertBad(
                    "delta must be in [0, pi)",
                    fn,
                    (bad,),
                    msg=f"{fn.__name__} took delta={bad}",
                )
        self.assertClose(
            _core.flaw_triangle(math.pi - 1e-9),
            2.0e-9,
            atol=1e-12,
            msg="triangle near pi",
        )

    def test_channel_slots(self):
        """
        The channel refuses a flaw outside its domain, an opaque or superunit transmittance, a dark count of
        one and a non-finite tilt.
        """
        cases = (
            (0, "delta must be in [0, pi)", (-0.1, math.pi)),
            (1, "eta must be in (0, 1]", (0.0, 1.5, float("nan"))),
            (2, "dark must be in [0, 1)", (1.0, -1e-9)),
            (3, "tilt must be finite", (float("inf"), float("nan"))),
        )
        self.assertSlots(
            _core.flaw_channel,
            (0.1, 0.5, 1e-7, 0.0),
            cases,
            msg="flaw_channel",
        )
        self.assertSlots(
            _core.flaw_phase,
            (0.1, 0.5, 1e-7, 0.0),
            cases,
            msg="flaw_phase",
        )

    def test_coin_slots(self):
        """
        `flaw_coin` refuses a yield at or below zero, `flaw_gllp` an error rate or imbalance above 1/2, and
        both rates an f_ec below 1 or a q_sift outside (0, 1].
        """
        self.assertBad(
            "y1 must be > 0",
            _core.flaw_coin,
            (0.1, 0.0),
            msg="flaw_coin took y1 = 0",
        )
        self.assertBad(
            "y1 must be > 0",
            _core.flaw_coin,
            (0.1, -0.5),
            msg="flaw_coin took a negative y1",
        )
        self.assertBad(
            "e_bit must be in [0, 1/2]",
            _core.flaw_gllp,
            (0.6, 0.1),
            msg="flaw_gllp took e_bit = 0.6",
        )
        self.assertBad(
            "coin must be in [0, 1/2]",
            _core.flaw_gllp,
            (0.1, 0.6),
            msg="flaw_gllp took coin = 0.6",
        )

        for fn in (_core.flaw_tolerant, _core.flaw_standard):
            self.assertSlots(
                fn,
                (0.1, 0.5, 1e-7, 0.0, 1.16, 0.25),
                (
                    (4, "f_ec must be >= 1", (0.9, 0.0)),
                    (5, "q_sift must be in (0, 1]", (0.0, 1.5)),
                ),
                msg="rate guards",
            )


class FlawLink(Question):
    """
    The q.Link route.
    """

    def test_link_exact(self):
        """
        Both analyses reach q.Link bit for bit against the engine, with eta_ab the channel times the detector
        efficiency.
        """
        for name in ("tolerant", "standard"):
            res = flaw_link(analysis=name).run()
            want = getattr(_core, f"flaw_{name}")(DELTA_A, LINK_ETA, P_DARK, -DELTA_A, F_FLAW, Q_SIFT)
            self.assertEqual(res.key_rate, want[0], msg=f"{name} left the engine")
            self.assertEqual(res.p_click, want[1], msg=f"{name} y1 left the engine")
            self.assertEqual(res.qber, want[2], msg=f"{name} e_bit left the engine")
            self.assertEqual(res.explain["eta_ab"]["value"], LINK_ETA, msg="eta_ab != T * eta_det")

    def test_analysis_named(self):
        """
        q.FlawBound has no default analysis and refuses a third, the two runs reporting e_phase and coin under
        names of their own.
        """
        with self.assertRaises(TypeError):
            q.FlawBound()

        self.assertFails(
            ValueError,
            "analysis must be 'tolerant'",
            lambda: q.FlawBound(analysis="exact"),
            msg="a third analysis was accepted",
        )

        tol = flaw_link(analysis="tolerant").run().explain
        std = flaw_link(analysis="standard").run().explain
        self.assertEqual(tol["analysis"]["value"], "tolerant", msg="analysis row")
        self.assertEqual(std["analysis"]["value"], "standard", msg="analysis row")
        self.assertIn("e_phase", tol, msg="the exact estimate is reported")
        self.assertNotIn("coin", tol, msg="and no coin beside it")
        self.assertIn("coin", std, msg="the imbalance is reported")
        self.assertNotIn("e_phase", std, msg="e_phase reported beside the coin")
        self.assertGreater(
            std["e_phase_bound"]["value"], tol["e_phase"]["value"], msg="standard bound below the exact estimate"
        )

    def test_signed_tilt(self):
        """
        The link reads the SIGNED tilt off q.SourceFlaw, flat in the flaw at 0 and costlier at -delta, and
        refuses a q.BasisAnalyser beside it.
        """
        signed = flaw_link().run()
        cancel = flaw_link(flaw=q.SourceFlaw(delta=DELTA_A, tilt=TILT_FLAT)).run()
        other = flaw_link(flaw=q.SourceFlaw(delta=DELTA_B, tilt=TILT_FLAT)).run()
        self.assertEqual(signed.explain["tilt"]["value"], -DELTA_A, msg="tilt != -delta")
        self.assertClose(
            cancel.explain["e_phase"]["value"],
            other.explain["e_phase"]["value"],
            msg="e_phase moved with delta at tilt = 0",
        )
        self.assertGreater(
            signed.explain["e_phase"]["value"],
            cancel.explain["e_phase"]["value"],
            msg="signed e_phase not above tilt = 0",
        )
        self.assertLess(signed.key_rate, cancel.key_rate, msg="signed rate not below tilt = 0")

        self.assertFails(
            ValueError,
            "q.BasisAnalyser.misalign is an UNSIGNED probability",
            flaw_link(receiver=q.BasisAnalyser()).run,
            msg="BasisAnalyser accepted",
        )

    def test_no_intensity(self):
        """
        q.FlawedKeying carries only flaw and sift, and q.SourceFlaw.coin is the one entry point taking a
        supplied y1.
        """
        named = {f.name for f in dataclasses.fields(q.FlawedKeying)}
        self.assertEqual(named, {"flaw", "sift"}, msg=f"FlawedKeying fields {named}")
        for gone in ("decoy", "mu", "intensities", "source"):
            self.assertNotIn(gone, named, msg=f"FlawedKeying carries {gone}")

        self.assertIn("y1", inspect.signature(q.SourceFlaw.coin).parameters, msg="SourceFlaw.coin takes no y1")

    def test_mismatch_barred(self):
        """
        A second detector efficiency is refused naming arXiv:2412.09684, both analyses needing Bob's
        inconclusive operator basis independent.
        """
        pair = q.ClickDetector(eta=0.9, dark=P_DARK, partner=q.ClickDetector(eta=0.4, dark=P_DARK))
        self.assertFails(
            NotImplementedError,
            "BASIS INDEPENDENT",
            flaw_link(detector=pair).run,
            msg="mismatched detector accepted",
        )
        self.assertFails(
            NotImplementedError,
            "arXiv:2412.09684",
            flaw_link(detector=pair).run,
            msg="refusal omits the composition",
        )

    def test_asymptotic_only(self):
        """
        A declared block is refused naming flaw_finite, Mizutani Eq. (43) on a SINGLE-PHOTON source, which
        ships and returns a positive length.
        """
        self.assertFails(
            NotImplementedError,
            "flaw_finite is the length that ships",
            flaw_link(block=q.KeyBlock(n=1e10)).run,
            msg="block accepted",
        )
        for name in ("flaw_azuma", "flaw_eps", "flaw_inverse", "flaw_errors", "flaw_length", "flaw_finite"):
            self.assertTrue(hasattr(_core, name), msg=f"_core.{name} left the engine")

        length = _core.flaw_finite(DELTA_A, LINK_ETA, P_DARK, -DELTA_A, PZ_EVEN, 1e12, F_FLAW, EPS_S, EPS_C, EPS_PH)
        self.assertGreater(length[0], 0.0, msg="flaw_finite length not positive")

    def test_closed_form(self):
        """
        A pulse train, a detector memory, an impairment and q.Asymptotic security are each refused by name,
        and explain() equals the run's dict at any seed.
        """
        rows = (
            (NotImplementedError, "emits no pulse train", dict(alice=q.Alice(symbol_rate=1e9))),
            (
                ValueError,
                "no slot train for a memory",
                dict(detector=q.ClickDetector(eta=0.9, dark=P_DARK, afterpulse=0.02)),
            ),
            (ValueError, "no term of this family reads an impairment", dict(impairments=(q.Backflash(prob=0.01),))),
            (NotImplementedError, "takes q.FlawBound security", dict(security=q.Asymptotic())),
        )
        for kind, needle, kw in rows:
            self.assertFails(kind, needle, flaw_link(**kw).run, msg=f"{needle!r} was accepted")

        link = flaw_link()
        self.assertEqual(link.explain(), link.run().explain, msg="explain() != run().explain")
        self.assertEqual(link.run(10, 0).key_rate, link.run(99, 7).key_rate, msg="seed moved the rate")

    def test_link_ordering(self):
        """
        On the link the tolerant rate beats the standard one at 3, 10 and 20 dB and the ratio grows with loss.
        """
        gaps = []
        for loss in (3.0, 10.0, 20.0):
            tol = flaw_link(t=10.0 ** (-loss / 10.0)).run().key_rate
            std = flaw_link(t=10.0 ** (-loss / 10.0), analysis="standard").run().key_rate
            self.assertGreater(tol, std, msg=f"standard >= tolerant at {loss} dB")
            gaps.append(tol / std)

        self.assertEqual(gaps, sorted(gaps), msg=f"gaps {gaps} not rising")


class FiniteRegion(Question):
    """
    The finite-size confidence region, and the matrix entry the paper repeats.
    """

    def test_inverse_solves(self):
        """
        `flaw_inverse` inverts the matrix Mizutani Eq. (101) builds from Alice's three Bloch vectors to 4e-15
        across the flaw domain.
        """
        for delta in (0.0, DELTA_A, DELTA_B, 0.5, 1.2, 2.9):
            inv = _core.flaw_inverse(delta)
            ang = _core.flaw_angles(delta)
            rows = [[0.5, 0.5 * math.sin(ang[i]), 0.5 * math.cos(ang[i])] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    cell = sum(inv[3 * i + k] * rows[k][j] for k in range(3))
                    want = 1.0 if i == j else 0.0
                    self.assertClose(cell, want, atol=4e-15, msg=f"A^-1 A[{i}][{j}] left the identity at {delta}")

            self.assertFinite(inv, msg=f"the inverse left f64 at delta={delta}")
            self.assertGreater(_core.flaw_triangle(delta), 0.0, msg=f"the triangle closed at {delta}")

    def test_printed_entry(self):
        """
        Mizutani Eq. (37)'s repeated third-row entry differs from the sigma_Z cofactor by more than 0.1 and
        misses the identity by more than 5%.
        """
        for delta in (0.0, DELTA_B, 0.8):
            ang = _core.flaw_angles(delta)
            inv = _core.flaw_inverse(delta)
            span = [math.sin(t) for t in ang[:3]]
            twice = 2.0 / _core.flaw_triangle(delta)
            self.assertClose(inv[6], twice * (span[2] - span[1]), atol=1e-14, msg="row 3 column 1 left the page")
            self.assertClose(inv[7], twice * (span[0] - span[2]), atol=1e-14, msg="row 3 column 2 left the page")
            self.assertClose(inv[8], twice * (span[1] - span[0]), atol=1e-14, msg="row 3 column 3 left the cofactor")

            printed = twice * (span[0] - span[2])
            self.assertGreater(abs(inv[8] - printed), 0.1, msg=f"the printed repeat agreed at delta={delta}")

            rows = [[0.5, 0.5 * math.sin(ang[i]), 0.5 * math.cos(ang[i])] for i in range(3)]
            wrong = [inv[6], inv[7], printed]
            drift = max(abs(sum(wrong[k] * rows[k][j] for k in range(3)) - (1.0 if j == 2 else 0.0)) for j in range(3))
            self.assertGreater(drift, 0.05, msg=f"the printed entry still inverted the matrix at delta={delta}")

    def test_azuma_width(self):
        """
        `flaw_azuma` is sqrt(2*n1*ln(1/eps)) to 1e-14 and `flaw_eps` gives each of the eight bounds an eighth.
        """
        for n1 in (1e4, 1e8, 1e12):
            for eps in (1e-3, 1e-10, 1e-20):
                self.assertClose(
                    _core.flaw_azuma(n1, eps) / math.sqrt(2.0 * n1 * math.log(1.0 / eps)),
                    1.0,
                    atol=1e-14,
                    msg=f"the Azuma width left its closed form at n={n1}",
                )

        self.assertClose(_core.flaw_eps(8e-10) / 1e-10, 1.0, atol=1e-14, msg="flaw_eps share")
        self.assertClose(
            _core.flaw_azuma(4e8, 1e-10) / _core.flaw_azuma(1e8, 1e-10),
            2.0,
            atol=1e-14,
            msg="the width must grow as the root of the block",
        )
        self.assertClose(_core.flaw_azuma(0.0, 1e-10), 0.0, atol=0.0, msg="width at n1 = 0")

    def test_region_collapse(self):
        """
        `flaw_errors` is the asymptotic estimate over the block plus each coefficient's magnitude against the
        Azuma width, to a part in 1e10.
        """
        for delta in (0.0, DELTA_A, DELTA_B, 0.9):
            for tilt in (-delta, 0.0, 0.4):
                for pz in (0.5, 0.8):
                    for n1 in (1e8, 1e14):
                        exact = _core.flaw_phase(delta, 0.3, P_DARK, tilt)
                        if exact >= 0.5:
                            continue

                        counts = miz_counts(delta, 0.3, P_DARK, tilt, pz, n1)
                        got = _core.flaw_errors(delta, pz, counts, n1, 1e-10)
                        want = n1 * pz * pz * exact + miz_slack(delta, pz, n1, 1e-10)
                        self.assertClose(
                            got / want, 1.0, atol=1e-10, msg=f"the region left its parts at {delta}/{tilt}"
                        )

    def test_region_pessimistic(self):
        """
        The finite phase-error count never falls below the asymptotic one and the gap falls by 10 per two
        decades of block.
        """
        gaps = []
        for n1 in (1e8, 1e10, 1e12, 1e14):
            counts = miz_counts(DELTA_B, 0.3, P_DARK, -DELTA_B, PZ_EVEN, n1)
            block = n1 * PZ_EVEN * PZ_EVEN
            got = _core.flaw_errors(DELTA_B, PZ_EVEN, counts, n1, 1e-10) / block
            exact = _core.flaw_phase(DELTA_B, 0.3, P_DARK, -DELTA_B)
            self.assertGreater(got, exact, msg=f"the finite bound undercut the exact estimate at n={n1}")
            gaps.append(got - exact)

        self.assertEqual(gaps, sorted(gaps, reverse=True), msg=f"gaps {gaps} not falling")
        for pair in zip(gaps, gaps[1:]):
            self.assertClose(pair[0] / pair[1], 10.0, atol=0.2, msg=f"gap ratio {pair[0] / pair[1]}")

    def test_region_slots(self):
        """
        `flaw_errors` refuses counts that are not six, a negative count, a degenerate pz, an empty block and
        an eps_ph outside (0, 1).
        """
        rows = (
            (ValueError, "must hold 6 single-photon detection counts", (0.1, 0.5, [1.0] * 5, 1e6, 1e-10)),
            (ValueError, "counts must be >= 0", (0.1, 0.5, [1.0, -1.0] + [1.0] * 4, 1e6, 1e-10)),
            (ValueError, "pz must be in (0, 1)", (0.1, 1.0, [1.0] * 6, 1e6, 1e-10)),
            (ValueError, "n1 must be > 0", (0.1, 0.5, [1.0] * 6, 0.0, 1e-10)),
            (ValueError, "eps_ph must be in (0, 1)", (0.1, 0.5, [1.0] * 6, 1e6, 1.0)),
        )
        for kind, needle, args in rows:
            self.assertFails(kind, needle, _core.flaw_errors, *args, msg=f"{needle!r} was accepted")


class FiniteLength(Question):
    """
    Mizutani's key length over a block.
    """

    def test_length_parts(self):
        """
        Eq. (43) is the vacuum count plus the single-photon count discounted by the phase entropy, less the
        leakage and log2(2/(eps_s^2 - eps_est)) + log2(2/eps_c).
        """
        for block in (1e6, 1e9):
            for e_phase in (0.0, 0.01, 0.2):
                leak = 1.16 * block * 0.05
                got = _core.flaw_length(0.0, block, e_phase, leak, EPS_S, EPS_C, EPS_PH)
                budget = math.log2(2.0 / (EPS_S * EPS_S - EPS_PH)) + math.log2(2.0 / EPS_C)
                entropy = (
                    0.0 if e_phase <= 0.0 else -e_phase * math.log2(e_phase) - (1 - e_phase) * math.log2(1 - e_phase)
                )
                want = math.floor(block * (1.0 - entropy) - leak - budget)
                self.assertClose(got, want, atol=0.0, msg=f"Eq. (43) left its terms at e_ph={e_phase}")

        credit = _core.flaw_length(1e5, 1e6, 0.1, 0.0, EPS_S, EPS_C, EPS_PH)
        bare = _core.flaw_length(0.0, 1e6, 0.1, 0.0, EPS_S, EPS_C, EPS_PH)
        self.assertClose(credit - bare, 1e5, atol=1.0, msg=f"vacuum credit {credit - bare}")

    def test_secrecy_squared(self):
        """
        An eps_est at or above eps_s^2 is refused by name and spending less of the square buys length.
        """
        self.assertFails(
            ValueError,
            "eps_est must be below eps_s^2",
            _core.flaw_length,
            0.0,
            1e6,
            0.01,
            0.0,
            1e-8,
            EPS_C,
            1e-10,
            msg="eps_est >= eps_s^2 accepted",
        )
        near = _core.flaw_length(0.0, 1e9, 0.01, 0.0, 1e-8, EPS_C, 0.9e-16)
        far = _core.flaw_length(0.0, 1e9, 0.01, 0.0, 1e-8, EPS_C, 1e-30)
        self.assertGreater(far, near, msg="far <= near")

    def test_block_approaches(self):
        """
        The length per emitted photon rises to `flaw_tolerant` at the same sifting share from below, agreeing
        to 1e-5 at 1e18 emissions.
        """
        asymptotic = _core.flaw_tolerant(DELTA_B, 0.3, P_DARK, -DELTA_B, F_FLAW, PZ_EVEN * PZ_EVEN)[0]
        rates = []
        for n_emit in (1e10, 1e12, 1e14, 1e16, 1e18):
            out = _core.flaw_finite(DELTA_B, 0.3, P_DARK, -DELTA_B, PZ_EVEN, n_emit, F_FLAW, EPS_S, EPS_C, EPS_PH)
            rates.append(out[0] / n_emit)
            self.assertLess(rates[-1], asymptotic, msg=f"finite rate above asymptotic at {n_emit}")

        self.assertEqual(rates, sorted(rates), msg=f"rates {rates} not rising")
        self.assertClose(rates[-1] / asymptotic, 1.0, atol=1e-5, msg=f"limit ratio {rates[-1] / asymptotic}")

    def test_finite_slots(self):
        """
        `flaw_finite` returns five slots against the asymptotic four, the first a whole number of bits, on the
        block's own yield and pz^2 share.
        """
        out = _core.flaw_finite(DELTA_B, 0.3, P_DARK, -DELTA_B, PZ_EVEN, 1e12, F_FLAW, EPS_S, EPS_C, EPS_PH)
        self.assertEqual(len(out), 5, msg=f"flaw_finite arity {len(out)}")
        self.assertEqual(len(tol_rate(DELTA_B, 3.0)), 4, msg="tol_rate arity")

        length, n1, m1, e_bit, e_phase = out
        self.assertClose(m1 / (n1 * PZ_EVEN * PZ_EVEN), 1.0, atol=1e-12, msg="m1 != n1 * pz^2")
        self.assertClose(
            n1 / (1e12 * _core.flaw_channel(DELTA_B, 0.3, P_DARK, -DELTA_B)[0]),
            1.0,
            atol=1e-12,
            msg="n1 != block * y1",
        )
        self.assertClose(e_bit, _core.flaw_channel(DELTA_B, 0.3, P_DARK, -DELTA_B)[1], atol=1e-15, msg="e_bit moved")
        self.assertGreater(
            e_phase, _core.flaw_phase(DELTA_B, 0.3, P_DARK, -DELTA_B), msg="e_phase not above the asymptotic"
        )
        self.assertEqual(length, math.floor(length), msg=f"length {length} not whole")

    def test_short_block(self):
        """
        A block too short to pay the epsilon budget returns zero and one that detects nothing is refused by
        name.
        """
        short = _core.flaw_finite(DELTA_B, 0.3, P_DARK, -DELTA_B, PZ_EVEN, 1e2, F_FLAW, EPS_S, EPS_C, EPS_PH)
        self.assertEqual(short[0], 0.0, msg="short block returned a length")
        self.assertFails(
            ValueError,
            "block detects nothing",
            _core.flaw_finite,
            DELTA_B,
            1e-300,
            0.0,
            -DELTA_B,
            PZ_EVEN,
            1e-300,
            F_FLAW,
            EPS_S,
            EPS_C,
            EPS_PH,
            msg="underflowed block returned a length",
        )

    def test_block_reach(self):
        """
        The finite length dies at 44.905985, 55.045740 and 57.448692 dB at 1e10, 1e12 and 1e14 emissions, all
        below the asymptotic ZERO_TOL.
        """
        asymptotic = bisect(lambda db: tol_rate(DELTA_A, db)[0], 0.0, 90.0)
        self.assertClose(asymptotic, ZERO_TOL, atol=1e-6, msg=f"asymptotic reach {asymptotic} dB")

        blocks = ((1e10, BLOCK_TEN), (1e12, BLOCK_TWELVE), (1e14, BLOCK_FOURTEEN))
        for n_emit, want in blocks:
            got = bisect(lambda db, n=n_emit: finite_rate(DELTA_A, db, n)[0], 0.0, 90.0)
            self.assertClose(got, want, atol=1e-6, msg=f"reach at n={n_emit} is {got} dB")
            self.assertLess(got, asymptotic, msg=f"finite reach {got} >= asymptotic at n={n_emit}")

    def test_paper_setting(self):
        """
        Mizutani Sec. V's 0.147 rad is 8.42 degrees to 5e-3 and is this module's delta, the deviation at
        Alice's pi setting.
        """
        self.assertClose(math.degrees(MIZ_FLAW), MIZ_DEG, atol=5e-3, msg="0.147 rad is not 8.42 degrees")
        ang = _core.flaw_angles(MIZ_FLAW)
        self.assertClose(ang[1] - math.pi, MIZ_FLAW, atol=1e-15, msg="the pi setting does not deviate by delta")

        out = _core.flaw_finite(MIZ_FLAW, 0.3, P_DARK, -MIZ_FLAW, PZ_EVEN, 1e12, F_FLAW, EPS_S, EPS_C, EPS_PH)
        clean = _core.flaw_finite(0.0, 0.3, P_DARK, 0.0, PZ_EVEN, 1e12, F_FLAW, EPS_S, EPS_C, EPS_PH)
        self.assertLess(out[0], clean[0], msg="flawed length above flaw-free")
        self.assertGreater(out[0] / clean[0], 0.5, msg=f"length ratio {out[0] / clean[0]}")


class NoComposition(Question):
    """
    Why no key rate here reads a mismatched receiver.
    """

    def test_one_efficiency(self):
        """
        Bob is ONE number: dark-free, y1 is the transmittance to 2e-16 at every flaw and tilt, and a second
        efficiency is refused.
        """
        for delta in (0.0, 0.2, 0.6):
            for tilt in (0.0, -delta, 0.3):
                for eta in (0.2, 0.5, 0.9):
                    y1, _e = _core.flaw_channel(delta, eta, 0.0, tilt)
                    self.assertClose(y1, eta, atol=2e-16, msg=f"y1 left eta at delta={delta}, tilt={tilt}")

        self.assertFails(
            TypeError,
            "positional",
            _core.flaw_phase,
            0.3,
            0.5,
            1e-6,
            -0.3,
            0.2,
            msg="flaw_phase took a second efficiency",
        )

    def test_mismatch_apart(self):
        """
        `attacks.mismatch_rate` prices a PAIR of efficiencies where all four entry points here read one and
        refuse a second; the composition is arXiv:2412.09684 (2024).
        """
        taken = inspect.signature(attacks.mismatch_rate).parameters
        self.assertIn("hi", taken, msg="mismatch_rate takes no pair")
        self.assertIn("lo", taken, msg="mismatch_rate takes no pair")
        self.assertIn("e_phase", taken, msg="mismatch_rate takes no e_phase")

        rows = (
            ("flaw_phase", (0.3, 0.5, 1e-6, -0.3, 0.2)),
            ("flaw_channel", (0.3, 0.5, 1e-6, -0.3, 0.2)),
            ("flaw_tolerant", (0.3, 0.5, 1e-6, -0.3, 1.1, 0.5, 0.2)),
            ("flaw_standard", (0.3, 0.5, 1e-6, -0.3, 1.1, 0.5, 0.2)),
        )
        for name, args in rows:
            self.assertFails(
                TypeError,
                "positional",
                getattr(_core, name),
                *args,
                msg=f"{name} took a second efficiency",
            )


if __name__ == "__main__":
    rc = Exam(
        "FlawGeometry",
        "Tier A: Alice's flawed states on the Bloch sphere and the triangle they must form",
        "flaws_geometry.md",
    ).run(load(FlawGeometry))
    rc |= Exam(
        "LossTolerant",
        "Tamaki's exact phase-error estimate, and what it stops depending on",
        "flaws_tolerant.md",
    ).run(load(LossTolerant))
    rc |= Exam(
        "CoinAmplified",
        "The standard analysis: a source flaw divided by the probability of detection",
        "flaws_coin.md",
    ).run(load(CoinAmplified))
    rc |= Exam(
        "TwoAnalyses",
        "The two rates side by side, the ideal limit they share and the loss where they part",
        "flaws_compare.md",
    ).run(load(TwoAnalyses))
    rc |= Exam(
        "FlawGuards",
        "Argument domains of the loss-tolerant engine",
        "flaws_guards.md",
    ).run(load(FlawGuards))
    rc |= Exam(
        "FlawLink",
        "The q.Link route: the analysis named, the tilt carried and what it refuses",
        "flaws_link.md",
    ).run(load(FlawLink))
    rc |= Exam(
        "FiniteRegion",
        "The confidence region the inversion runs over, and the matrix entry the paper repeats",
        "flaws_region.md",
    ).run(load(FiniteRegion))
    rc |= Exam(
        "FiniteLength",
        "Mizutani's key length over a block, and the asymptotic rate it walks up to",
        "flaws_finite.md",
    ).run(load(FiniteLength))
    rc |= Exam(
        "NoComposition",
        "The one efficiency this device model reads, and the pair the mismatch analysis wants",
        "flaws_nocompose.md",
    ).run(load(NoComposition))
    sys.exit(rc)
