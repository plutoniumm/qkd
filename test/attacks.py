import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import THA_CLOCK as LUC_CLOCK
from kit.anchors import THA_N as LUC_N
from kit.anchors import THA_TABLE
from kit.anchors import THA_TARGET as LUC_TARGET
from kit.forms import bisect
from qkd import _core, attacks, impairments

# Qin, Kumar and Alleaume, PRA 94, 012325 (2016), arXiv:1511.01007, Sec. VII.3.1. ALPHA is a half-range in
# sqrt(N_0), not an attenuation; ETA_BOB is Qin's receiver, not kit.anchors' GYS one.
ETA_BOB = 0.55
V_ELE = 0.015
XI_SYS = 0.1
ALPHA = 20.0
BETA = 0.95

# Assumed: fixed SNR, Qin's V_A being unpublished (optimised per Jouguet, PRA 84, 062317 (2011)).
SNR = 0.75

# Lydersen et al., Nat. Photonics 4, 686 (2010), arXiv:1008.4593, Figs. 3-4.
# W, measured on an ID Quantique id3110 Clavis2 blinded by 1.08 mW CW.
CLAVIS_ALWAYS = (808e-6, 932e-6)
CLAVIS_NEVER = (647e-6, 697e-6)
CLAVIS_BLIND = (397e-6, 765e-6)

# Zhao, Fung, Qi, Chen and Lo, PRA 78, 042333 (2008), Table I: per-detector counts at two time shifts, then
# same-basis totals and errors. Neither key length is anchored: K_L = 1297 (Eq. (3)) moves over 1230-1331 with the
# decoy convention, and K_U = 1131 (Eq. (4)) reproduces at 1229.6 under every reading tried.
SHIFT_A = (10992, 1541)
SHIFT_B = (1231, 4059)
BASIS_A = (6243, 383)
BASIS_B = (2684, 144)

# Weier et al., New J. Phys. 13, 073024 (2011), arXiv:1101.5289, Table I, measured: (mu_B_eff, Alice-Bob QBER %,
# Bob-Eve QBER %, I_EB); I_EB = 1 - h(QBER_BE), one measurement written twice.
WEIER = (
    (0.37, 1.10, 48.17, 0.001),
    (0.49, 1.12, 47.54, 0.002),
    (0.83, 1.09, 45.24, 0.007),
    (1.88, 1.01, 38.43, 0.039),
    (5.29, 1.11, 21.00, 0.259),
    (9.75, 1.12, 6.91, 0.638),
    (16.52, 1.25, 1.17, 0.908),
)

# Unverified: read off Weier Fig. 2b at mu_B = 16.52; literal Eqs. (13)-(14) give 0.9309, both diagonals 0.8816.
WEIER_CURVE = 0.882

# Lucamarini, Choi, Ward, Dynes, Yuan and Shields, Phys. Rev. X 5, 031030 (2015), arXiv:1506.01989, Table I row 1:
# 60 dB isolator + 2 x 35 dB attenuator (Eq. (15)) + 40 dB reflection = 170 dB; other rows in test/exptrojan.py.
LUC_ISO = THA_TABLE[0][1]
LUC_REFLECT = THA_TABLE[0][2]
LUC_ATTEN = THA_TABLE[0][3]
LUC_ISOLATOR = THA_TABLE[0][4]


def modulation(dist):
    """
    Alice's variance at ``dist`` km holding SNR fixed, this exam's V_A rule.
    """
    t = 10 ** (-0.02 * dist)

    return SNR * (1.0 + ETA_BOB * t * XI_SYS + V_ELE) / (ETA_BOB * t)


def printed(v_a, t, delta, weight=1.0):
    """
    Eq. (17) of arXiv:1511.01007 as printed, ``weight`` scaling its variance term.
    """
    var = attacks.sat_var(v_a, t, ETA_BOB, V_ELE, XI_SYS)
    gap = ALPHA - delta
    a = math.erf(gap / math.sqrt(2.0 * var))
    b = math.exp(-(gap**2) / (2.0 * var))
    bracket = (
        weight * var * (1.0 + a - b * b / math.pi)
        - 2.0 * math.sqrt(2.0 * var / math.pi) * gap * a * b
        + gap * gap * (1.0 - a * a)
        - 4.0
        - 4.0 * V_ELE
    )

    return bracket / (ETA_BOB * t * (1.0 + a) ** 2) - v_a


def doubled(blind, signal):
    """
    Eve's information under the reading that counts both diagonal detectors.
    """
    par, diag = attacks.blank_probs(blind)
    sig_p = 1.0 - math.exp(-signal / 2.0)
    sig_d = 1.0 - math.exp(-signal / 4.0)
    same = (1.0 - par) * sig_p + 2.0 * (1.0 - diag) * sig_d
    other = sig_p + 2.0 * (1.0 - diag) * sig_d

    return 1.0 - attacks._entropy(same / (same + other))


def hidden(gap):
    """
    assess with a Blanking at ``gap``.
    """

    return attacks.assess(
        blank=attacks.Blanking(16.52, 0.1, gap),
        window=5e-9,
        dead=2e-6,
        qber=0.0125,
    )


class Continuous(Question):
    """
    Homodyne saturation and local-oscillator calibration against CV-QKD.
    """

    def test_saturation_null(self):
        """
        Published anchor, Qin arXiv:1511.01007 Sec. VII.3.1: at zero displacement a full
        intercept-resend with 0.1 SNU technical noise estimates xi = 2.1 SNU and an unbiased T-hat
        at gain G = 2.
        """
        t_hat, xi_hat = attacks.sat_estimate(5.0, 0.5, ETA_BOB, V_ELE, ALPHA, 0.0, XI_SYS)

        self.assertClose(t_hat, 0.5, atol=1e-15, msg=f"T-hat {t_hat}")
        self.assertClose(xi_hat, 2.1, atol=1e-14, msg=f"xi-hat {xi_hat} vs 2.1 SNU")

        far = attacks.sat_estimate(5.0, 0.5, ETA_BOB, V_ELE, 1e6, 0.0, XI_SYS)

        self.assertClose(far[1], 2.1, atol=1e-9, msg=f"xi-hat at alpha 1e6: {far[1]}")

    def test_saturation_printed(self):
        """
        Eq. (17) of arXiv:1511.01007 as printed returns -3.2955 against the stated 2.1 SNU at zero
        displacement, and doubling its Var(X_B,lin) term, as Eq. (B14) and Qin's thesis Eq. (7.10)
        do, matches sat_estimate to 1e-12 at every displacement.
        """
        naive = printed(5.0, 0.5, 0.0)

        self.assertClose(
            naive,
            -3.2954545454545454,
            atol=1e-12,
            msg=f"printed Eq. (17) at Delta 0: {naive:.4f}",
        )

        for delta in (0.0, 10.0, 18.0, 19.0, 19.9):
            fixed = printed(5.0, 0.5, delta, 2.0)
            ours = attacks.sat_estimate(5.0, 0.5, ETA_BOB, V_ELE, ALPHA, delta, XI_SYS)
            self.assertClose(
                fixed,
                ours[1],
                atol=1e-12,
                msg=f"Delta {delta}: doubled {fixed} vs sat_estimate {ours[1]}",
            )

        wide = attacks.sat_clip(3.0, 40.0)

        self.assertClose(wide, 3.0, atol=1e-12, msg=f"sat_clip(3, 40): {wide}")

    def test_saturation_clip(self):
        """
        sat_clip matches the censored-Gaussian second moment by Simpson integration over ten
        standard deviations and never exceeds the unclipped variance.
        """
        var, gap = 4.0, 1.5
        sigma = math.sqrt(var)
        grid = 20001
        span = 10.0 * sigma
        step = 2.0 * span / (grid - 1)
        mean, second = 0.0, 0.0
        for i in range(grid):
            x = -span + i * step
            w = 1.0 if i in (0, grid - 1) else (4.0 if i % 2 else 2.0)
            dens = math.exp(-x * x / (2.0 * var)) / math.sqrt(2.0 * math.pi * var)
            y = min(x, gap)
            mean += w * dens * y
            second += w * dens * y * y
        mean *= step / 3.0
        second *= step / 3.0

        self.assertClose(
            attacks.sat_clip(var, gap),
            second - mean * mean,
            atol=1e-8,
            msg=f"Simpson moment {second - mean * mean}",
        )

        for delta in (0.0, 5.0, 15.0, 19.0, 20.0):
            clipped = attacks.sat_clip(3.0, ALPHA - delta)
            self.assertLessEqual(clipped, 3.0, msg=f"Delta {delta}: clipped {clipped}")

    def test_saturation_bias(self):
        """
        The estimated excess noise falls through zero as the displacement nears the detector limit,
        and every target below 2.1 SNU is reproduced by some displacement to 1e-9, Qin's
        Proposition.
        """
        args = (5.0, 0.5, ETA_BOB, V_ELE, ALPHA)
        far = [attacks.sat_estimate(*args, d, XI_SYS)[1] for d in (15.0, 17.0, 19.0)]

        self.assertMonotone(far, rising=False, msg=f"xi-hat over Delta: {far}")

        for want in (1.5, 0.5, 0.05, 0.0):
            delta = bisect(lambda d: attacks.sat_estimate(*args, d, XI_SYS)[1], 0.0, 21.0, want)
            got = attacks.sat_estimate(*args, delta, XI_SYS)[1]
            self.assertClose(got, want, atol=1e-9, msg=f"xi-hat {got} vs target {want}")

        deep = attacks.sat_estimate(*args, 20.0, XI_SYS)

        self.assertLess(deep[1], 0.0, msg=f"xi-hat at Delta 20: {deep[1]}")
        self.assertLess(deep[0], 0.5, msg=f"T-hat at Delta 20: {deep[0]}")

    def test_saturation_break(self):
        """
        qkd's asymptotic key rate is positive on the channel the saturated estimator reports and
        negative on the true one.
        """
        t_hat, xi_hat = attacks.sat_estimate(5.0, 0.5, ETA_BOB, V_ELE, ALPHA, 19.0, XI_SYS)
        seen = _core.cv_rate(5.0, t_hat, xi_hat, ETA_BOB, V_ELE, BETA, True, True)
        real = _core.cv_rate(5.0, 0.5, 2.1, ETA_BOB, V_ELE, BETA, True, True)

        self.assertGreater(seen[2], 0.0, msg=f"reported rate {seen[2]:.5f}")
        self.assertLess(real[2], 0.0, msg=f"true rate {real[2]:.5f}")

        read = attacks.sat_break(5.0, 0.5, ETA_BOB, V_ELE, ALPHA, 19.0, XI_SYS)

        self.assertClose(read.eve["xi"], 2.1, atol=1e-14, msg=f"eve xi {read.eve['xi']}")
        self.assertLess(
            read.observed["xi"],
            read.eve["xi"],
            msg=f"observed xi {read.observed['xi']} vs eve {read.eve['xi']}",
        )

    def test_saturation_levelii(self):
        """
        Published anchor, partly: holding T-hat by solving Qin Eq. (19) for the gain, the
        displacement minimising the reported excess noise at 31 km is 19.54 against their 19.5, with
        V_A from this exam's fixed-SNR rule.
        """
        dist = 31.0
        t = 10 ** (-0.02 * dist)
        v_a = modulation(dist)
        best = None
        for i in range(1, 4000):
            delta = 20.0 * i / 4000.0
            gain = attacks.sat_gain(v_a, t, ETA_BOB, V_ELE, ALPHA, delta, XI_SYS)
            pair = attacks.sat_estimate(v_a, t, ETA_BOB, V_ELE, ALPHA, delta, XI_SYS, gain)
            self.assertClose(pair[0], t, atol=1e-12, msg=f"Delta {delta}: T-hat {pair[0]} vs T {t}")

            if best is None or pair[1] < best[0]:
                best = (pair[1], delta)

        self.assertClose(best[1], 19.5, atol=0.1, msg=f"minimising Delta {best[1]:.2f} vs 19.5")
        self.assertLess(best[0], 0.0, msg=f"min xi-hat at 31 km: {best[0]}")

        near = modulation(10.0)
        shy = min(
            attacks.sat_estimate(
                near,
                10**-0.2,
                ETA_BOB,
                V_ELE,
                ALPHA,
                20.0 * i / 500.0,
                XI_SYS,
                attacks.sat_gain(near, 10**-0.2, ETA_BOB, V_ELE, ALPHA, 20.0 * i / 500.0, XI_SYS),
            )[1]
            for i in range(1, 500)
        )

        self.assertGreater(shy, 0.5, msg=f"min xi-hat at 10 km: {shy:.3f}")

    def test_saturation_domain(self):
        """
        sat_estimate and sat_gain refuse a negative displacement, a non-positive linear range, a
        displacement clipping T-hat to zero, and an unreachable level II gain.
        """
        args = (5.0, 0.5, ETA_BOB, V_ELE)

        self.assertFails(
            ValueError,
            "delta",
            attacks.sat_estimate,
            *args,
            ALPHA,
            -1.0,
            msg="Delta = -1",
        )
        self.assertFails(
            ValueError,
            "alpha",
            attacks.sat_estimate,
            *args,
            0.0,
            0.0,
            msg="alpha = 0",
        )
        self.assertFails(
            ValueError,
            "T-hat = 0",
            attacks.sat_estimate,
            *args,
            ALPHA,
            60.0,
            msg="Delta = 60",
        )
        self.assertFails(
            ValueError,
            "level II",
            attacks.sat_gain,
            *args,
            ALPHA,
            25.0,
            msg="level II at Delta = 25",
        )

    def test_calibration_anchor(self):
        """
        Published anchor, Jouguet, Kunz-Jacques and Diamanti, PRA 87, 062313 (2013), arXiv:1304.7024
        Eq. (9): 2.1 SNU at T = eta = 0.5 behind a shot-noise ratio of 1.5 reports exactly 1/15 SNU
        and 61/40 reports zero, both derived from Eqs. (7) and (8) and printed nowhere.
        """
        real = attacks.resend_xi(XI_SYS, 1.0)

        self.assertClose(real, 2.1, atol=1e-15, msg=f"xi_PIR {real} vs 2.1 SNU")

        seen = attacks.calib_xi(real, 1.5, 0.5, 0.5)

        self.assertClose(seen, 1.0 / 15.0, atol=1e-14, msg=f"reported xi {seen:.6f} vs 1/15")

        null = attacks.calib_ratio(real, 0.5, 0.5)

        self.assertClose(null, 61.0 / 40.0, atol=1e-14, msg=f"zero-noise ratio {null} vs 61/40")
        self.assertClose(
            attacks.calib_xi(real, null, 0.5, 0.5),
            0.0,
            atol=1e-14,
            msg=f"xi at ratio {null}",
        )

    def test_calibration_raw(self):
        """
        calib_xi is unclamped: the reported excess noise equals the true one at ratio 1, falls
        monotonically with the ratio, and goes negative past 1.525.
        """
        real = attacks.resend_xi(XI_SYS, 1.0)

        self.assertClose(
            attacks.calib_xi(real, 1.0, 0.5, 0.5),
            real,
            atol=1e-14,
            msg="ratio 1",
        )

        seen = [attacks.calib_xi(real, k, 0.5, 0.5) for k in (1.0, 1.2, 1.525, 1.8)]

        self.assertMonotone(seen, rising=False, msg=f"reported xi over ratio: {seen}")
        self.assertLess(seen[-1], 0.0, msg=f"reported xi at ratio 1.8: {seen[-1]}")

        share = [attacks.resend_xi(XI_SYS, m) for m in (0.0, 0.25, 0.5, 1.0)]

        self.assertMonotone(share, msg=f"xi over resent fraction: {share}")

    def test_calibration_break(self):
        """
        qkd's key rate is positive on the reported 0.0667 SNU and negative on the true 2.1 SNU, and
        the Reading reports the shot-noise ratio beside them.
        """
        read = attacks.calib_break(XI_SYS, 1.5, 0.5, 0.5)
        seen = _core.cv_rate(5.0, 0.5, read.observed["xi"], 0.5, 0.01, BETA, True, True)
        real = _core.cv_rate(5.0, 0.5, read.eve["xi"], 0.5, 0.01, BETA, True, True)

        self.assertGreater(seen[2], 0.0, msg=f"reported rate {seen[2]:.5f}")
        self.assertLess(real[2], 0.0, msg=f"true rate {real[2]:.5f}")
        self.assertClose(read.observed["shot"], 1.5, atol=1e-15, msg=f"observed shot {read.observed['shot']}")
        self.assertFails(
            ValueError,
            "ratio",
            attacks.calib_xi,
            2.1,
            0.0,
            0.5,
            0.5,
            msg="ratio = 0",
        )


class Threshold(Question):
    """
    Blinding, detection-efficiency mismatch and dead-time blanking against click receivers.
    """

    def test_blinding_anchor(self):
        """
        Published anchor, Lydersen et al., Nat. Photonics 4, 686 (2010), arXiv:1008.4593: the
        measured Clavis2 thresholds give P_always/P_never = 1.4405 < 2, their Eq. (1), and both
        detectors blind at 397 and 765 microwatt, below the 1.08 mW control power.
        """
        ratio = attacks.blind_ratio(CLAVIS_ALWAYS, CLAVIS_NEVER)

        self.assertClose(ratio, 932.0 / 647.0, atol=1e-12, msg=f"control ratio {ratio:.4f}")
        self.assertLess(ratio, 2.0, msg=f"Eq. (1) ratio {ratio:.4f}")
        self.assertLess(
            max(CLAVIS_BLIND),
            1.08e-3,
            msg=f"blinding power {max(CLAVIS_BLIND)} W",
        )
        self.assertGreater(
            min(CLAVIS_NEVER),
            max(CLAVIS_BLIND) * 0.8,
            msg=f"P_never {min(CLAVIS_NEVER)} vs P_blind {max(CLAVIS_BLIND)}",
        )

    def test_blinding_control(self):
        """
        blind_break refuses a detector set violating Eq. (1) and a gain Eve cannot cover, and the
        trigger share halves when Bob's basis choice is passive.
        """
        self.assertFails(
            ValueError,
            "Eq. (1)",
            attacks.blind_break,
            0.01,
            0.0,
            (2.0e-3,),
            (0.9e-3,),
            msg="ratio 2.2",
        )
        self.assertClose(
            attacks.blind_share(0.01),
            0.02,
            atol=1e-15,
            msg="active trigger share",
        )
        self.assertClose(
            attacks.blind_share(0.01, True),
            0.01,
            atol=1e-15,
            msg="passive trigger share",
        )
        self.assertFails(
            ValueError,
            "exceeds 1",
            attacks.blind_break,
            0.75,
            0.0,
            CLAVIS_ALWAYS,
            CLAVIS_NEVER,
            msg="gain 0.75",
        )

    def test_blinding_break(self):
        """
        Under detector control the observed QBER is Eve's choice while she holds one bit per sifted
        bit, and a 1% gain costs a 2% trigger share, 1% passive.
        """
        read = attacks.blind_break(0.01, 0.012, CLAVIS_ALWAYS, CLAVIS_NEVER)

        self.assertEqual(read.eve["info"], 1.0, msg=f"eve info {read.eve['info']}")
        self.assertEqual(read.observed["qber"], 0.012, msg=f"observed qber {read.observed['qber']}")
        self.assertClose(read.observed["trigger"], 0.02, atol=1e-15, msg="trigger share")

        wide = attacks.blind_break(0.01, 0.012, CLAVIS_ALWAYS, CLAVIS_NEVER, True)

        self.assertClose(
            wide.observed["trigger"],
            0.01,
            atol=1e-15,
            msg="passive trigger share",
        )

    def test_shift_qber(self):
        """
        Qi, Fung, Lo and Ma, QIC 7, 73 (2007), arXiv:quant-ph/0512080 Eq. (2): the faked-state QBER
        rises from 0 at r = 0 to 1/2 at r = 1 and crosses intercept-resend's 1/4 at their stated r =
        0.2.
        """
        self.assertEqual(attacks.shift_qber(0.0), 0.0, msg="qber at r = 0")
        self.assertClose(attacks.shift_qber(1.0), 0.5, atol=1e-15, msg="qber at r = 1")
        self.assertClose(attacks.shift_qber(0.2), 0.25, atol=1e-15, msg="qber at r = 0.2")

        vals = [attacks.shift_qber(r) for r in (0.0, 0.1, 0.5, 1.0)]

        self.assertMonotone(vals, msg=f"qber over r: {vals}")

    def test_shift_anchor(self):
        """
        Published anchor, Qi et al. Eq. (4): their measured mismatch of 2 bounds the rate at h(2/3)
        = 0.9183 against a naive 1, symmetric under r to 1/r, with no induced error.
        """
        r = attacks.shift_ratio(2.0, 1.0)

        self.assertClose(r, 0.5, atol=1e-15, msg=f"r {r}")

        bound = attacks.shift_bound(r)

        self.assertClose(bound, 0.9183, atol=5e-5, msg=f"bound {bound:.5f} vs 0.9183")
        self.assertEqual(attacks.shift_ratio(1.0, 2.0), r, msg="shift_ratio(1, 2)")
        self.assertClose(
            attacks.shift_leak(r),
            1.0 - bound,
            atol=1e-15,
            msg="shift_leak + shift_bound",
        )

        read = attacks.shift_break(2.0, 1.0)

        self.assertEqual(read.observed["qber"], 0.0, msg=f"observed qber {read.observed['qber']}")
        self.assertGreater(
            read.observed["naive"],
            read.eve["bound"],
            msg=f"naive {read.observed['naive']} vs bound {read.eve['bound']}",
        )

    def test_shift_balance(self):
        """
        Published anchor, Zhao, Fung, Qi, Chen and Lo, PRA 78, 042333 (2008) Table I: shift_balance
        derives the equalising mixture 23.031% and 3479.07 detections per detector against their
        23.0% and 3479, and that mixture gives the overall QBER 5.681% against 5.68%.
        """
        p, count = attacks.shift_balance(*SHIFT_A, *SHIFT_B)

        self.assertClose(p, 0.230, atol=5e-4, msg=f"mixture {p * 100:.3f}% vs 23.0%")
        self.assertClose(count, 3479.0, atol=0.5, msg=f"per-detector count {count:.2f}")
        self.assertClose(
            p * SHIFT_A[1] + (1.0 - p) * SHIFT_B[1],
            count,
            atol=1e-9,
            msg="mixed second-detector count",
        )

        sift = p * BASIS_A[0] + (1.0 - p) * BASIS_B[0]
        errs = p * BASIS_A[1] + (1.0 - p) * BASIS_B[1]

        self.assertClose(errs / sift, 0.0568, atol=5e-5, msg=f"overall QBER {errs / sift:.5f}")
        self.assertFails(
            ValueError,
            "same detector",
            attacks.shift_balance,
            100.0,
            10.0,
            90.0,
            20.0,
            msg="both shifts favour one detector",
        )

    def test_shift_measured(self):
        """
        Zhao's two measured shifts give mismatches 7.133 and 3.297, rate ceilings 0.5378 and 0.7827
        under Qi Eq. (4), and their per-shift QBERs 6.135% and 5.365%.
        """
        for counts, want in ((SHIFT_A, 7.133), (SHIFT_B, 3.297)):
            ratio = max(counts) / min(counts)
            self.assertClose(ratio, want, atol=5e-4, msg=f"measured mismatch {ratio:.4f}")

            bound = attacks.shift_bound(attacks.shift_ratio(*counts))
            self.assertLess(bound, 1.0, msg=f"bound {bound} at mismatch {ratio:.3f}")

        self.assertClose(
            attacks.shift_bound(attacks.shift_ratio(*SHIFT_A)),
            0.5378,
            atol=5e-4,
            msg="the -250 ps ceiling",
        )
        self.assertClose(
            attacks.shift_bound(attacks.shift_ratio(*SHIFT_B)),
            0.7827,
            atol=5e-4,
            msg="the +500 ps ceiling",
        )

        for basis, want in ((BASIS_A, 0.06135), (BASIS_B, 0.05365)):
            qber = basis[1] / basis[0]
            self.assertClose(qber, want, atol=5e-6, msg=f"per-shift QBER {qber:.5f} vs {want}")

    def test_mismatch_rate(self):
        """
        Fung, Tamaki, Qi, Lo and Ma, QIC 9, 131 (2009), arXiv:0802.3788 Eqs. (32) to (34): the two
        forms join at matched detectors and otherwise differ by (1 - share) h(e_bit), and at Zhao's
        7.13 mismatch and 5.68% QBER 1 - 2h(e) overstates the rate 4.07 times.
        """
        both = (
            attacks.mismatch_rate(0.2, 0.2, 0.05, 0.05),
            attacks.mismatch_rate(0.2, 0.2, 0.05, 0.05, False),
        )

        self.assertClose(both[0], both[1], atol=1e-15, msg=f"forms at matched detectors: {both}")
        self.assertClose(
            both[0],
            1.0 - 2.0 * attacks._entropy(0.05),
            atol=1e-15,
            msg="matched vs 1 - 2h(0.05)",
        )

        share = 2.0 * 1541.0 / (10992.0 + 1541.0)
        keep = attacks.mismatch_rate(10992.0, 1541.0, 0.0568, 0.0568)
        general = attacks.mismatch_rate(10992.0, 1541.0, 0.0568, 0.0568, False)

        self.assertClose(
            keep - general,
            (1.0 - share) * attacks._entropy(0.0568),
            atol=1e-14,
            msg=f"form gap {keep - general}",
        )

        naive = 1.0 - 2.0 * attacks._entropy(0.0568)

        self.assertClose(naive / keep, 4.067, atol=5e-3, msg=f"overstatement {naive / keep:.4f}")

        with self.assertRaises(TypeError):
            attacks.mismatch_rate(0.2, 0.1, 0.05)

    def test_blank_leak(self):
        """
        Published anchor, Weier et al., New J. Phys. 13, 073024 (2011) Table I: the model sits above
        all seven measured I_EB and reproduces the top point 0.908 to 0.023, I_EB = 1 - h(QBER_BE)
        holds to 6e-4, and the doubled diagonal reading lands on Fig. 2b.
        """
        got = [attacks.blank_leak(row[0], 0.1) for row in WEIER]

        self.assertMonotone(got, msg=f"leak over mu_B: {got}")

        for row, leak in zip(WEIER, got):
            self.assertClose(
                1.0 - attacks._entropy(row[2] / 100.0),
                row[3],
                atol=6e-4,
                msg=f"mu_B {row[0]}: 1 - h(QBER_BE) vs I_EB {row[3]}",
            )
            self.assertGreater(leak, row[3], msg=f"mu_B {row[0]}: model {leak:.4f} vs {row[3]}")
            self.assertLess(
                attacks.blank_error(row[0], 0.1),
                row[2] / 100.0,
                msg=f"mu_B {row[0]}: blank_error vs QBER_BE {row[2]}%",
            )

        self.assertClose(got[-1], 0.908, atol=0.03, msg=f"top point {got[-1]:.4f} vs 0.908")

        curve = doubled(16.52, 0.1)

        self.assertClose(
            curve,
            WEIER_CURVE,
            atol=5e-3,
            msg=f"doubled reading {curve:.4f}",
        )

        span = max(row[1] for row in WEIER) - min(row[1] for row in WEIER)

        self.assertLess(span, 0.25, msg=f"measured QBER span {span:.2f}")

    def test_blank_model(self):
        """
        The blanking model has Weier's structure: the parallel detector's blind probability composes
        from two diagonal halves, zero intensity blinds and leaks nothing, and the error rate is
        independent of signal intensity to 2e-3.
        """
        par, diag = attacks.blank_probs(4.0)

        self.assertGreater(par, diag, msg=f"par {par} vs diag {diag}")
        self.assertClose(1.0 - par, (1.0 - diag) ** 2, atol=1e-14, msg="1 - par vs (1 - diag)^2")
        self.assertEqual(attacks.blank_probs(0.0), (0.0, 0.0), msg="blank_probs(0)")
        self.assertClose(attacks.blank_leak(0.0, 0.1), 0.0, atol=1e-12, msg="blank_leak(0, 0.1)")

        thin = attacks.blank_error(16.52, 0.01)
        thick = attacks.blank_error(16.52, 0.2)

        self.assertClose(thin, thick, atol=2e-3, msg=f"signal dependence {abs(thin - thick):.2e}")

    def test_blank_hidden(self):
        """
        blank_hidden passes Weier's 200 ns lead against a 5 ns window and a 2 microsecond or 400 ns
        dead time, and refuses a pulse inside the window or older than the dead time.
        """
        self.assertTrue(attacks.blank_hidden(200e-9, 5e-9, 2e-6), msg="200 ns lead")
        self.assertFails(
            ValueError,
            "acceptance window",
            attacks.blank_hidden,
            2e-9,
            5e-9,
            2e-6,
            msg="2e-9 s lead",
        )
        self.assertFails(
            ValueError,
            "dead time",
            attacks.blank_hidden,
            3e-6,
            5e-9,
            2e-6,
            msg="3e-6 s lead",
        )
        self.assertTrue(
            attacks.blank_hidden(200e-9, 5e-9, 400e-9),
            msg="200 ns lead, 400 ns dead time",
        )

    def test_blank_deadtime(self):
        """
        The dead time impairments.saturate charges as lost clicks the blanking attack charges as
        Eve's information, leaving the QBER unmoved at 16.52 and 0.37 photons.
        """
        dead = 2e-6
        rate = 1e5
        honest = impairments.saturate(rate, dead)

        self.assertLess(honest, rate, msg=f"saturated rate {honest}")

        read = attacks.blank_break(16.52, 0.1, 0.0125)

        self.assertEqual(read.observed["qber"], 0.0125, msg=f"observed qber {read.observed['qber']}")
        self.assertGreater(read.eve["info"], 0.9, msg=f"eve info {read.eve['info']}")

        quiet = attacks.blank_break(0.37, 0.1, 0.0110)

        self.assertClose(
            quiet.observed["qber"] - 0.0110,
            0.0,
            atol=1e-15,
            msg=f"observed qber {quiet.observed['qber']}",
        )
        self.assertLess(
            quiet.eve["info"],
            0.01,
            msg=f"eve info {quiet.eve['info']}",
        )


class Steered(Question):
    """
    The transmitted-oscillator compositions: a shot-noise ratio derived from Bob's own
    calibration line, and a clipping point that moves with it.
    """

    def test_ratio_derived(self):
        """
        lo_ratio reads the shot-noise ratio off tlo_calib's calibration line and returns the two
        zero-signal variances, equal on an honest line and split by the misnormalisation otherwise.
        """
        ratio, zero, null = attacks.lo_ratio(1.05, 1.0, 2.0, 0.1)
        line = _core.tlo_calib(1.05, 1.0, 2.0, 0.1)

        self.assertClose(ratio, line[2], msg=f"ratio {ratio} vs tlo_calib {line[2]}")
        self.assertClose(ratio, 1.05, msg=f"ratio {ratio}")
        self.assertClose(zero, line[4], msg=f"zero {zero} vs tlo_calib {line[4]}")
        self.assertClose(null, 1.0 + line[3], msg=f"null {null}")
        self.assertGreater(null, zero, msg=f"null {null} vs zero {zero}")

        flat, seen, honest = attacks.lo_ratio(1.0, 1.0, 2.0, 0.1)

        self.assertClose(flat, 1.0, msg=f"honest ratio {flat}")
        self.assertClose(seen, honest, msg=f"zero {seen} vs null {honest}")

    def test_break_matches(self):
        """
        lo_break matches calib_break at equal ratios and additionally reports the zero and null
        variances.
        """
        derived = attacks.lo_break(1.63, 1.0, XI_SYS, 0.5, 0.6)
        dialled = attacks.calib_break(XI_SYS, 1.63, 0.5, 0.6)

        self.assertClose(
            derived.observed["xi"],
            dialled.observed["xi"],
            msg="observed xi, derived vs dialled",
        )
        self.assertClose(derived.eve["xi"], dialled.eve["xi"], msg="eve xi, derived vs dialled")
        self.assertEqual(derived.attack, "oscillator", msg=f"attack {derived.attack}")

        for name in ("zero", "null"):
            self.assertIn(name, derived.observed, msg=f"observed lacks {name}")

    def test_line_hides(self):
        """
        At the zeroing ratio 1.63 Bob reports zero excess noise over a 2.1 SNU intercept-resend
        while the two zero-signal variances differ by 1 - 1/1.63 = 39% of a shot-noise unit.
        """
        real = attacks.resend_xi(XI_SYS)
        ratio = attacks.calib_ratio(real, 0.5, 0.6)
        read = attacks.lo_break(ratio, 1.0, XI_SYS, 0.5, 0.6)

        self.assertClose(ratio, 1.63, msg=f"zeroing ratio {ratio}")
        self.assertClose(read.observed["xi"], 0.0, atol=1e-14, msg=f"observed xi {read.observed['xi']}")
        self.assertClose(read.eve["xi"], 2.1, msg=f"eve xi {read.eve['xi']}")
        self.assertClose(
            read.observed["null"] - read.observed["zero"],
            1.0 - 1.0 / 1.63,
            msg="null - zero",
        )

    def test_range_moves(self):
        """
        sat_range scales alpha by sqrt(ratio) as tlo_range does: identity at ratio 1, a factor 10 at
        20 dB.
        """

        self.assertClose(
            attacks.sat_range(ALPHA, 100.0),
            _core.tlo_range(ALPHA, 100.0),
            msg="sat_range left tlo_range",
        )
        self.assertClose(attacks.sat_range(ALPHA, 100.0), 10.0 * ALPHA, msg="sat_range at ratio 100")
        self.assertClose(attacks.sat_range(ALPHA, 1.0), ALPHA, msg="ratio 1")

    def test_range_reaches(self):
        """
        A displacement of 2 alpha that sat_estimate refuses as clipped at the quoted alpha estimates
        once alpha carries a 20 dB oscillator loss.
        """
        v_a = modulation(0.0)

        self.assertFails(
            ValueError,
            "clipped away entirely",
            attacks.sat_estimate,
            v_a,
            0.5,
            ETA_BOB,
            V_ELE,
            ALPHA,
            2.0 * ALPHA,
            msg="Delta = 2 alpha",
        )

        wide = attacks.sat_range(ALPHA, 100.0)
        t_hat, _xi = attacks.sat_estimate(v_a, 0.5, ETA_BOB, V_ELE, wide, 2.0 * ALPHA, XI_SYS)

        self.assertGreater(t_hat, 0.0, msg=f"T-hat {t_hat}")

    def test_steered_domains(self):
        """
        lo_ratio refuses a zero sampled power and sat_range a zero shot-noise ratio.
        """

        self.assertFails(
            ValueError,
            "p_eff",
            attacks.lo_ratio,
            1.0,
            0.0,
            msg="p_eff = 0",
        )
        self.assertFails(
            ValueError,
            "ratio",
            attacks.sat_range,
            ALPHA,
            0.0,
            msg="ratio = 0",
        )


class Source(Question):
    """
    Trojan-horse light injection: the isolation a transmitter needs, what leaks past it, and the
    key rate this module will not write.
    """

    def test_lucamarini_budget(self):
        """
        Lucamarini arXiv:1506.01989 Sec. IV's 170 dB three ways: summed over their split by Eq.
        (15), solved from their 1e-6 target by Eq. (16), and spent back to 1e-6 by Eq. (4).
        """

        self.assertClose(
            attacks.probe_isolation(LUC_REFLECT, LUC_ISOLATOR, 1, LUC_ATTEN),
            LUC_ISO,
            msg="Eq. (15) isolation",
        )
        self.assertClose(
            attacks.probe_budget(LUC_TARGET, LUC_N, LUC_CLOCK),
            LUC_ISO,
            atol=1e-12,
            msg="Eq. (16) budget",
        )
        self.assertClose(
            attacks.probe_photons(LUC_N, LUC_CLOCK, LUC_ISO),
            LUC_TARGET,
            atol=1e-20,
            msg="Eq. (4) photons",
        )

    def test_double_pass(self):
        """
        probe_isolation follows Eq. (15) term by term: reflection once, attenuator and filter twice,
        isolator once per stage.
        """
        base = attacks.probe_isolation(10.0)

        self.assertClose(base, 10.0, msg="reflection")
        self.assertClose(
            attacks.probe_isolation(10.0, atten=1.0) - base,
            2.0,
            msg="attenuator",
        )
        self.assertClose(
            attacks.probe_isolation(10.0, bandpass=1.0) - base,
            2.0,
            msg="bandpass",
        )
        self.assertClose(
            attacks.probe_isolation(10.0, isolator=1.0, stages=3) - base,
            3.0,
            msg="isolator, 3 stages",
        )

    def test_delta_first(self):
        """
        Eq. (7)'s coin imbalance is mu_out/2 to first order and not a probability, peaking at 0.5335
        near 3*pi/4.
        """

        self.assertClose(
            attacks.probe_delta(LUC_TARGET),
            0.5 * LUC_TARGET,
            atol=1e-15,
            msg="Delta at the target leak",
        )
        self.assertClose(attacks.probe_delta(0.0), 0.0, msg="Delta at mu 0")
        self.assertClose(
            attacks.probe_delta(0.75 * math.pi),
            0.5335098698541367,
            msg="Delta at 3 pi/4",
        )
        self.assertGreater(attacks.probe_delta(0.75 * math.pi), 0.5, msg="Delta at 3 pi/4")

    def test_phase_rises(self):
        """
        probe_phase equals the measured error rate at zero leak, rises with the leak and as the
        single-photon yield falls, and clamps at 1/2.
        """

        self.assertClose(attacks.probe_phase(0.01, 0.0, 1.0), 0.01, msg="phase at mu 0")
        self.assertMonotone(
            [attacks.probe_phase(0.01, mu, 1.0) for mu in (0.0, 1e-6, 1e-4, 1e-2)],
            msg="phase over mu",
        )
        self.assertMonotone(
            [attacks.probe_phase(0.01, 1e-4, y) for y in (1.0, 0.1, 0.01, 1e-3)],
            msg="phase over y1",
        )
        self.assertClose(attacks.probe_phase(0.01, 1.0, 0.01), 0.5, msg="phase at mu 1, y1 0.01")
        self.assertLessEqual(attacks.probe_phase(0.4, 0.5, 0.5), 0.5, msg="phase at qber 0.4")

    def test_probe_reading(self):
        """
        probe_break leaves the observed QBER unmoved while Eve's phase error rate rises above it,
        and the Reading refuses key_rate.
        """
        read = attacks.probe_break(1e-4, 0.0125, 0.5)

        self.assertEqual(read.attack, "injection", msg=f"attack {read.attack}")
        self.assertEqual(read.observed["qber"], 0.0125, msg=f"observed qber {read.observed['qber']}")
        self.assertGreater(
            read.eve["phase"],
            read.observed["qber"],
            msg=f"eve phase {read.eve['phase']}",
        )

        with self.assertRaises(AttributeError) as caught:
            read.key_rate
        self.assertIn(
            "does not bound the second",
            str(caught.exception),
            msg="key_rate refusal",
        )

    def test_rate_refused(self):
        """
        probe_rate raises NotImplementedError naming its missing pieces: THA-exposed decoys, a
        per-basis yield, a four-state BB84 map, and flaws.rs's no-side-channel assumption.
        """
        for needle in (
            "not touched by the THA",
            "min[Y_X, Y_Y]",
            "basis-blind y1",
            "four BB84 settings",
            "no side channel, no Trojan-horse leakage",
        ):
            self.assertFails(
                NotImplementedError,
                needle,
                attacks.probe_rate,
                1e-6,
                0.01,
                0.5,
                msg=f"refusal lacks {needle}",
            )

    def test_probe_domains(self):
        """
        The injection functions refuse a fractional isolator stage, a zero modulator clock, a QBER
        past 1/2 and a zero single-photon yield.
        """

        self.assertFails(
            ValueError,
            "whole number",
            attacks.probe_isolation,
            10.0,
            60.0,
            1.5,
            msg="stages = 1.5",
        )
        self.assertFails(
            ValueError,
            "clock",
            attacks.probe_photons,
            LUC_N,
            0.0,
            LUC_ISO,
            msg="clock = 0",
        )
        self.assertFails(
            ValueError,
            "qber",
            attacks.probe_phase,
            0.6,
            1e-6,
            1.0,
            msg="qber = 0.6",
        )
        self.assertFails(
            ValueError,
            "y1",
            attacks.probe_phase,
            0.01,
            1e-6,
            0.0,
            msg="y1 = 0",
        )


class Contract(Question):
    """
    What a Reading refuses, what assess emits, and what stays shared with impairments.
    """

    def test_reading_barred(self):
        """
        A Reading raises on key_rate, rate, key, secure, safe and margin with the reason, keeps
        observed and eve reachable, and raises a plain AttributeError on an unknown name.
        """
        read = attacks.shift_break(2.0, 1.0)

        for name in ("key_rate", "rate", "key", "secure", "safe", "margin"):
            with self.assertRaises(AttributeError) as caught:
                getattr(read, name)
            self.assertIn(
                "does not bound the second",
                str(caught.exception),
                msg=f"{name} refusal",
            )

        self.assertIn("info", read.eve, msg="eve lacks info")
        self.assertIn("qber", read.observed, msg="observed lacks qber")

        with self.assertRaises(AttributeError) as caught:
            read.nonsense
        self.assertEqual(str(caught.exception), "nonsense", msg="unknown attribute message")

    def test_reading_table(self):
        """
        Reading.table prints both sides, the mechanism note and Eve's two rows in one block.
        """
        text = attacks.blind_break(0.01, 0.01, CLAVIS_ALWAYS, CLAVIS_NEVER).table()

        for needle in ("observed", "eve", "Lydersen", "blinding"):
            self.assertIn(needle, text, msg=f"table lacks {needle}")

        rows = [line for line in text.splitlines() if line.startswith("eve")]

        self.assertEqual(len(rows), 2, msg=f"eve rows {len(rows)}")

    def test_assess_rows(self):
        """
        assess returns {} with no descriptors and otherwise one Reading per catalogue name, never a
        budget Entry.
        """
        self.assertEqual(attacks.assess(), {}, msg="assess()")

        out = attacks.assess(
            sat=attacks.Saturation(ALPHA, 19.0),
            calib=attacks.Calibration(1.5),
            blind=attacks.Blinding(CLAVIS_ALWAYS, CLAVIS_NEVER),
            mismatch=attacks.Mismatch(2.0, 1.0),
            blank=attacks.Blanking(16.52, 0.1, 200e-9),
            lo=attacks.Oscillator(1.63, 1.0),
            probe=attacks.Injection(LUC_REFLECT, LUC_ISOLATOR, 1, LUC_ATTEN),
            v_a=5.0,
            t=0.5,
            eta=ETA_BOB,
            v_el=V_ELE,
            xi=XI_SYS,
            gain=0.01,
            qber=0.0125,
            window=5e-9,
            dead=2e-6,
            y1=0.5,
        )

        self.assertEqual(
            sorted(out),
            [
                "blanking",
                "blinding",
                "calibration",
                "injection",
                "oscillator",
                "saturation",
                "timeshift",
            ],
            msg=f"keys {sorted(out)}",
        )

        for name, read in out.items():
            self.assertEqual(read.attack, name, msg=f"{name}: attack {read.attack}")
            self.assertIsInstance(read, attacks.Reading, msg=f"{name}: {type(read)}")
            self.assertNotIn("xi", read.observed.keys() - {"xi"}, msg="no stray xi")

        self.assertEqual(
            [row[0] for row in attacks.catalogue()],
            [
                "saturation",
                "calibration",
                "oscillator",
                "blinding",
                "timeshift",
                "blanking",
                "injection",
            ],
            msg="catalogue names",
        )

    def test_assess_yield(self):
        """
        assess refuses an Injection with no single-photon yield, naming the basis-blind y1, and with
        one reaches probe_photons through Eq. (15).
        """

        self.assertFails(
            ValueError,
            "basis-blind y1",
            attacks.assess,
            None,
            None,
            None,
            None,
            None,
            None,
            attacks.Injection(LUC_REFLECT, LUC_ISOLATOR, 1, LUC_ATTEN),
            msg="y1 = None",
        )

        out = attacks.assess(
            probe=attacks.Injection(LUC_REFLECT, LUC_ISOLATOR, 1, LUC_ATTEN),
            qber=0.0125,
            y1=0.5,
        )

        self.assertClose(
            out["injection"].eve["leaked"],
            LUC_TARGET,
            atol=1e-20,
            msg="eve leaked",
        )

    def test_assess_timing(self):
        """
        assess refuses a blanking pulse inside the window, older than the dead time, or declared
        with no dead time, and scores one with no timing.
        """
        self.assertFails(
            ValueError,
            "dead time",
            hidden,
            3e-6,
            msg="3e-6 s lead",
        )
        self.assertFails(
            ValueError,
            "acceptance window",
            hidden,
            2e-9,
            msg="2e-9 s lead",
        )
        self.assertIn("blanking", hidden(200e-9), msg="200 ns lead")
        self.assertFails(
            ValueError,
            "q.DeadTime",
            attacks.assess,
            None,
            None,
            None,
            None,
            attacks.Blanking(16.52, 0.1, 200e-9),
            msg="gap with dead = None",
        )

        out = attacks.assess(blank=attacks.Blanking(16.52, 0.1), qber=0.0125)

        self.assertIn("blanking", out, msg="gap = None")

    def test_entropy_shared(self):
        """
        attacks._entropy equals impairments._entropy over a grid and is zero out of range.
        """
        for e in (0.0, 1e-6, 0.01, 0.0568, 0.11, 0.25, 0.5, 0.9, 1.0):
            self.assertEqual(
                attacks._entropy(e),
                impairments._entropy(e),
                msg=f"entropy at {e}",
            )

        self.assertEqual(attacks._entropy(1.5), 0.0, msg="entropy at 1.5")


if __name__ == "__main__":
    rc = Exam(
        "AttacksContinuous",
        "Homodyne saturation and local-oscillator calibration against CV-QKD",
        "attacks_continuous.md",
    ).run(load(Continuous))
    rc |= Exam(
        "AttacksThreshold",
        "Blinding, efficiency mismatch and dead-time blanking against click receivers",
        "attacks_threshold.md",
    ).run(load(Threshold))
    rc |= Exam(
        "AttacksSteered",
        "A shot-noise unit and a clipping point set by a transmitted oscillator",
        "attacks_steered.md",
    ).run(load(Steered))
    rc |= Exam(
        "AttacksSource",
        "Trojan-horse light injection into Alice's transmitter",
        "attacks_source.md",
    ).run(load(Source))
    rc |= Exam(
        "AttacksContract",
        "What a Reading refuses and what assess emits",
        "attacks_contract.md",
    ).run(load(Contract))
    sys.exit(rc)
