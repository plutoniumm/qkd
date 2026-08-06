import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

from kit.anchors import ETA_COW, F_COW
from kit.cache import memo
from kit.checks import Guarded
from qkd import _core

# The COW' vacuum-decoy phase error, bounded two ways.
#
# GAO ET AL., Opt. Express 30, 23783 (2022), arXiv:2107.09329, bound the unmeasured component
# of |a> orthogonal to vacuum by 1: `_core.sdp_analytic`.
# SEKSARIA & PRABHAKAR, "Short reach by theorem: the certifiable key rate of COW QKD"
# (2026), Sec. VI D: the phase error is LINEAR in Bob's sub-POVM, so its worst case over the
# eight monitoring gains is an SDP, `_core.sdp_phase`, never looser (test_below_analytic).
#
# Fig. 1's operating point: 0.2 dB/km, eta_D = 0.8 (ETA_COW), p_d = 1e-8 (the review's, not
# Gao's), f = 1.1 (F_COW), rate optimised over (mu, t_B); pinned numbers are this exam's own.
# Reach runs 93 and 103 km against Fig. 1's 103 and 121 km: sdp_gains' Mach-Zehnder monitor sees
# half a pulse's flux. The whole pulse (3 dB) gives 102 km analytic and Fig. 1(b)'s 0.235 and
# 0.155 at zero distance.
ALPHA = 0.2
P_DARK = 1e-8

# EXAM-SIDE (mu, t_B) peaks from this exam's coarse search; Fig. 1 prints neither parameter.
PEAK_SDP = {
    0.0: (0.062503, 0.4389),
    20.0: (0.019662, 0.4750),
    40.0: (0.0072782, 0.4833),
}
PEAK_GAO = {
    0.0: (0.019101, 0.4861),
    20.0: (0.0073559, 0.4861),
    40.0: (0.0029277, 0.4750),
}

# EXAM-SIDE: the certified rate's operating point at 100 km, past the analytic reach.
FAR = (100.0, 0.00054757, 0.3472)

# Measured rate ratio at the three peaks, against the review's "roughly a factor of two out to 80 km".
DOUBLING = (2.087, 1.879, 1.876)

# Regression anchor at mu = 0.024, zero distance, symmetric splitter; this implementation, not a paper.
PINNED = 0.19597124

# LI, CAO, XIE, YIN & CHEN, Phys. Rev. Research 6, 013022 (2024), arXiv:2309.16136,
# composable finite key against COHERENT attacks. BLOCK in two-pulse ROUNDS; DECOY per decoy
# sequence, fixed here where Li optimises; EPS_SEC is TOTAL, EPS_COR additive to it.
BLOCK = 1e11
DECOY = 0.05
EPS_SEC = 1e-10
EPS_COR = 1e-15

# EXAM-SIDE: the finite key's share of the ceiling per PEAK_GAO point at BLOCK, and over BLOCKS at 20 km.
SHARES = {
    0.0: 0.985244,
    20.0: 0.962385,
    40.0: 0.905200,
}
GROWTH = (0.051668, 0.598822, 0.874026, 0.962385, 0.988404, 0.996362)
BLOCKS = (1e8, 1e9, 1e10, 1e11, 1e12, 1e13)

# Azuma's deviation over Kato's at k = 1e11, eps = 1e-11, at three counts; they coincide at k/2.
BIAS = ((2e3, 3352.9464), (1e6, 157.7400), (1e9, 5.0248))


def channel(dist):
    """
    Transmittance to the detector at ``dist`` km, without Bob's monitoring splitter, which
    sdp_gains applies.
    """
    return ETA_COW * 10 ** (-ALPHA * dist / 10.0)


def top_eig(a, b, d):
    """
    Largest eigenvalue of the symmetric 2x2 [[a, b], [b, d]], in closed form.
    """
    mid = 0.5 * (a + d)
    half = 0.5 * (a - d)

    return mid + math.sqrt(half * half + b * b)


def lmi_cap(mats, start):
    """
    ``min t`` s.t. ``t*I - A_k >> 0`` over the 2x2 blocks in ``mats``: the largest eigenvalue.
    """
    flat = []
    for one in mats:
        flat.extend(-x for x in one)
        flat.extend([1.0, 0.0, 0.0, 1.0])

    return _core.sdp_lmi([1.0], [2] * len(mats), flat, [start], 1e-12)


@memo
def phase_pair(dist, mu, t_b):
    """
    ``(sdp, analytic)`` phase-error bounds at one operating point, memoised.
    """
    eta = channel(dist)
    q0, q1 = _core.sdp_gains(mu, t_b, eta, P_DARK)
    bound = _core.sdp_phase(mu, q0, q1, 1e-9)[0]

    return bound, _core.sdp_analytic(mu, q0, q1)


def rate(dist, mu, t_b, kind):
    """
    Key rate per pulse, two pulses to the sequence, under bound ``kind``.
    """
    eta = channel(dist)
    q0, q1 = _core.sdp_gains(mu, t_b, eta, P_DARK)
    if kind == "sdp":
        try:
            e_ph = _core.sdp_phase(mu, q0, q1, 1e-8)[0]
        except ValueError:
            return 0.0
    else:
        e_ph = _core.sdp_analytic(mu, q0, q1)

    if e_ph >= 0.5:
        return 0.0

    gain, e_bit = _core.sdp_data(mu, t_b, eta, P_DARK)

    return 0.5 * _core.cow_rate(gain, e_bit, e_ph, F_COW)


def survey(dist, kind, nmu=24, nsplit=7):
    """
    Best rate over a coarse (mu, t_B) grid at ``dist`` km.
    """
    best = 0.0
    for i in range(nmu):
        mu = 10 ** (-6.0 + 5.4 * i / (nmu - 1))
        for j in range(nsplit):
            t_b = 0.05 + 0.9 * j / (nsplit - 1)
            best = max(best, rate(dist, mu, t_b, kind))

    return best


def floor_ep(dist, kind, nmu=14, nsplit=4):
    """
    Smallest phase-error bound any operating point reaches at ``dist`` km; key needs it below
    1/2.
    """
    low = 1.0
    eta = channel(dist)
    for i in range(nmu):
        mu = 10 ** (-5.5 + 4.6 * i / (nmu - 1))
        for j in range(nsplit):
            t_b = 0.05 + 0.9 * j / (nsplit - 1)
            q0, q1 = _core.sdp_gains(mu, t_b, eta, P_DARK)
            if kind == "gao":
                low = min(low, _core.sdp_analytic(mu, q0, q1))
                continue

            try:
                low = min(low, _core.sdp_phase(mu, q0, q1, 1e-7)[0])
            except ValueError:
                continue

    return low


def widen(dist, mu, t_b, rounds, decoy=DECOY):
    """
    The record ``(q0, q1, lo)`` with the two DECOY gains widened to Kato bounds over ``rounds``,
    the two BIT gains left observed.
    """
    eta = channel(dist)
    eps = _core.sdp_eps(EPS_SEC)
    q0, q1 = _core.sdp_gains(mu, t_b, eta, P_DARK)
    hi0, hi1, lo = list(q0), list(q1), [0.0, 0.0]
    sent = rounds * decoy
    for j in (0, 1):
        hi0[j] = (sent * q0[j] + _core.sdp_kato(sent, sent * q0[j], eps, True)) / sent
        hi1[j] = (sent * q1[j] + _core.sdp_kato(sent, sent * q1[j], eps, True)) / sent
        lo[j] = max(0.0, (sent * q0[j] - _core.sdp_kato(sent, sent * q0[j], eps, False)) / sent)

    return hi0, hi1, lo


def keylength(dist, mu, t_b, rounds, decoy=DECOY):
    """
    Li's composable finite-key length in bits over ``rounds`` two-pulse rounds, zero at the
    phase-error abort.
    """
    hi0, hi1, lo = widen(dist, mu, t_b, rounds, decoy)
    expect = _core.sdp_interval(mu, hi0, hi1, lo)
    gain, e_z = _core.sdp_data(mu, t_b, channel(dist), P_DARK)
    n_z = rounds * (1.0 - 2.0 * decoy) * gain
    phase = _core.sdp_sample(expect, n_z, _core.sdp_eps(EPS_SEC))
    if phase >= 0.5:
        return 0.0

    return _core.sdp_length(n_z, e_z, phase, F_COW, EPS_SEC, EPS_COR)


def ceiling(dist, mu, t_b, decoy=DECOY):
    """
    The asymptotic rate in bits per ROUND, decoy rounds removed: twice rate()'s per-pulse
    figure.
    """

    return 2.0 * rate(dist, mu, t_b, "gao") * (1.0 - 2.0 * decoy)


class SdpSolver(Question):
    """
    The bare interior-point solver on LMIs with closed-form answers.
    """

    def test_largest_eigenvalue(self):
        """
        Minimising t subject to $tI - A \\succeq 0$ returns the largest eigenvalue of a
        symmetric 2x2 within the reported gap, and over two blocks the larger of the two.
        """
        for a, b, d in ((2.0, 0.3, 1.0), (1.0, -0.9, 1.0), (5.0, 0.0, 0.25)):
            want = top_eig(a, b, d)
            got, gap, iters, v = lmi_cap([[a, b, b, d]], want + 5.0)
            self.assertClose(got, want, atol=gap + 1e-11, msg="LMI value != the top eigenvalue")
            self.assertClose(v[0], want, atol=gap + 1e-11, msg="LMI argument != the top eigenvalue")
            self.assertGreater(iters, 0, msg="iters = 0")

        one = (2.0, 0.3, 1.0)
        two = (1.4, 0.9, 1.4)
        want = max(top_eig(*one), top_eig(*two))
        got, gap, _, _ = lmi_cap(
            [
                [one[0], one[1], one[1], one[2]],
                [two[0], two[1], two[1], two[2]],
            ],
            10.0,
        )

        self.assertClose(got, want, atol=gap + 1e-11, msg="two-block cap != the larger eigenvalue")

    def test_gap_tightens(self):
        """
        A tighter tolerance returns a smaller gap and a value no further from the closed form,
        and at every tolerance the gap covers the real error with the point feasible.
        """
        want = top_eig(2.0, 0.3, 1.0)
        gaps = []
        errs = []
        for tol in (1e-3, 1e-6, 1e-9, 1e-12):
            flat = [-2.0, -0.3, -0.3, -1.0, 1.0, 0.0, 0.0, 1.0]
            got, gap, _, _ = _core.sdp_lmi([1.0], [2], flat, [8.0], tol)
            gaps.append(gap)
            errs.append(abs(got - want))
        self.assertMonotone(gaps, rising=False, msg="reported gap not falling")
        self.assertMonotone(errs, rising=False, msg="real error not falling")
        self.assertLess(errs[-1], 1e-10, msg="error above 1e-10 at tol 1e-12")

        want = top_eig(3.0, 1.1, 0.5)
        flat = [-3.0, -1.1, -1.1, -0.5, 1.0, 0.0, 0.0, 1.0]
        for tol in (1e-2, 1e-4, 1e-8):
            got, gap, _, _ = _core.sdp_lmi([1.0], [2], flat, [9.0], tol)
            self.assertLessEqual(abs(got - want), gap + 1e-12, msg="gap below the real error")
            self.assertGreaterEqual(got, want - 1e-12, msg="point below the optimum")


class SdpStates(Question):
    """
    The COW' assembly: four coherent sequences, their overlaps, the honest monitoring record.
    """

    def test_gram_overlaps(self):
        """
        The four sequences' Gram matrix is symmetric, with $e^{-\\mu}$ between sequences
        differing in both bins, $e^{-\\mu/2}$ in one, and least eigenvalue $(1 -
        e^{-\\mu/2})^2$.
        """
        for mu in (0.01, 0.2, 1.0):
            g = _core.sdp_gram(mu)
            one = math.exp(-0.5 * mu)
            two = math.exp(-mu)
            self.assertClose(g[0], 1.0, msg="Gram diagonal != 1")
            self.assertClose(g[1], two, msg="|0>|0> against |a>|a>")
            self.assertClose(g[2], one, msg="|0>|0> against |0>|a>")
            self.assertClose(g[11], two, msg="|0>|a> against |a>|0>")

            for j in range(4):
                for k in range(4):
                    self.assertClose(g[j * 4 + k], g[k * 4 + j], msg="Gram matrix not symmetric")

        for mu in (0.005, 0.05, 0.5):
            g = _core.sdp_gram(mu)
            one = math.exp(-0.5 * mu)
            want = (1.0 - one) ** 2
            flat = [g[i] for i in range(16)]
            flat.extend([-1.0 if i % 5 == 0 else 0.0 for i in range(16)])
            got, gap, _, _ = _core.sdp_lmi([-1.0], [4], flat, [0.5 * want], 1e-13)
            self.assertClose(-got, want, atol=gap + 1e-12, msg="least Gram eigenvalue != (1 - exp(-mu/2))^2")

    def test_gains_law(self):
        """
        The eight honest gains follow the threshold click law, dark floors on the vacuum
        sequence and the decoy's destructive port, while the data gain rises with intensity and
        its error rate is dark counts alone.
        """
        mu = 0.02
        t_b = 0.5
        eta = channel(20.0)
        q0, q1 = _core.sdp_gains(mu, t_b, eta, P_DARK)
        seen = eta * (1.0 - t_b)

        self.assertClose(q0[0], P_DARK, msg="q0[0] != the dark floor")
        self.assertClose(q1[0], P_DARK, msg="q1[0] != the dark floor")
        self.assertClose(q1[1], P_DARK, msg="q1[1] != the dark floor")

        want = 1.0 - (1.0 - P_DARK) * math.exp(-seen * mu)

        self.assertClose(q0[1], want, msg="q0[1] != the whole-flux gain")

        half = 1.0 - (1.0 - P_DARK) * math.exp(-0.25 * seen * mu)

        self.assertClose(q0[2], half, msg="q0[2] != the quarter-flux gain")
        self.assertClose(q1[3], half, msg="q1[3] != the quarter-flux gain")

        far = channel(40.0)
        gains = [_core.sdp_data(m, 0.5, far, P_DARK)[0] for m in (1e-4, 1e-3, 1e-2)]

        self.assertMonotone(gains, rising=True, msg="data gain not rising with mu")

        clean = _core.sdp_data(1e-2, 0.5, far, 1e-14)[1]

        self.assertLess(clean, 1e-10, msg="e_bit above 1e-10 at dark 1e-14")

        dirty = _core.sdp_data(1e-2, 0.5, far, 1e-5)[1]

        self.assertGreater(dirty, clean, msg="dark e_bit not above the clean one")

    def test_honest_floor(self):
        """
        The honest phase error is $1/2 - e^{-\\mu}(e^x + 1)/4$ at monitor mean photon number x,
        flooring just under $(1 - e^{-\\mu})/2$ with perfect detectors and rising with dark
        counts.
        """
        mu = 0.02
        t_b = 0.5
        eta = channel(20.0)
        x = 0.25 * eta * (1.0 - t_b) * mu
        want = 0.5 - math.exp(-mu) * (math.exp(x) + 1.0) / 4.0
        got = _core.sdp_honest(mu, t_b, eta, 0.0)
        edge = 0.5 * (1.0 - math.exp(-mu))

        self.assertClose(got, want, msg="the closed form of the honest phase error")
        self.assertLess(got, edge, msg="honest phase error at or above mu/2")
        self.assertGreater(got, 0.95 * edge, msg="honest phase error below 0.95 mu/2")

        rise = [_core.sdp_honest(mu, t_b, eta, d) for d in (0.0, 1e-8, 1e-6, 1e-4)]

        self.assertMonotone(rise, rising=True, msg="honest phase error not rising with dark")

    def test_split_product(self):
        """
        The monitoring record and honest phase error depend on transmittance and splitter only
        through $\\eta(1 - t_B)$, the data line through the other product.
        """
        mu = 0.03
        arm = ETA_COW * (1.0 - 0.5)
        want = _core.sdp_gains(mu, 0.5, ETA_COW, P_DARK)
        floor = _core.sdp_honest(mu, 0.5, ETA_COW, P_DARK)
        for eta in (0.5, 0.9, 1.0):
            t_b = 1.0 - arm / eta
            self.assertEqual(
                _core.sdp_gains(mu, t_b, eta, P_DARK),
                want,
                msg=f"sdp_gains at eta = {eta}",
            )
            self.assertClose(
                _core.sdp_honest(mu, t_b, eta, P_DARK),
                floor,
                atol=0.0,
                msg=f"sdp_honest at eta = {eta}",
            )

        wide = _core.sdp_data(mu, 1.0 - arm, 1.0, P_DARK)[0]
        held = _core.sdp_data(mu, 0.5, ETA_COW, P_DARK)[0]

        self.assertGreater(wide, held, msg="wide data gain not above the held one")


class SdpBound(Question):
    """
    The certified worst case against the analytic estimator it replaces and the honest channel
    it contains.
    """

    def test_below_analytic(self):
        """
        THE LOAD-BEARING ONE. The certified worst case sits at or below Gao's Cauchy-Schwarz
        bound at every operating point.
        """
        for dist in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0):
            for k in (0.3, 0.1, 0.03):
                mu = k * channel(dist)
                bound, plain = phase_pair(dist, mu, 0.5)
                self.assertLessEqual(
                    bound,
                    plain + 1e-9,
                    msg=f"{dist} km, mu = {mu:.4g}: {bound:g} > {plain:g}",
                )

    def test_above_honest(self):
        """
        The certified worst case sits at or above the honest channel's phase error, one point of
        the set it maximises over.
        """
        for dist in (0.0, 40.0, 80.0):
            for k in (0.3, 0.1, 0.03):
                mu = k * channel(dist)
                bound = phase_pair(dist, mu, 0.5)[0]
                seen = _core.sdp_honest(mu, 0.5, channel(dist), P_DARK)
                self.assertGreaterEqual(bound, seen - 1e-12, msg=f"{dist} km: {bound:g} < honest {seen:g}")

    def test_empty_lever(self):
        """
        With COW's own three sequences at honest gains, raising the vacuum-sequence row alone
        drives the certified phase error from 0.274 through the 1/2 abort: the empty sequence
        makes this COW' rather than COW.
        """
        mu, t_b = PEAK_SDP[0.0]
        q0, q1 = _core.sdp_gains(mu, t_b, channel(0.0), P_DARK)
        seen = []
        for lift in (P_DARK, 1e-6, 1e-4, 1e-3, 1e-2):
            rows = (list(q0), list(q1))
            rows[0][0] = lift
            rows[1][0] = lift
            seen.append(_core.sdp_phase(mu, rows[0], rows[1], 1e-9)[0])
        self.assertMonotone(seen, rising=True, msg="phase error not rising with the empty row")
        self.assertClose(seen[0], 0.2742, atol=5e-3, msg="honest bound != 0.2742")
        self.assertLess(seen[-2], 0.5, msg="1e-3 lift already aborts")
        self.assertGreater(seen[-1], 0.5, msg="1e-2 lift does not abort")

    def test_bound_pinned(self):
        """
        One certified value is pinned at 0.19597124, a tighter tolerance returns a smaller
        certified gap, and every bound stays within its own gap of the tightest.
        """
        mu = 0.024
        q0, q1 = _core.sdp_gains(mu, 0.5, ETA_COW, P_DARK)
        bound, gap, iters = _core.sdp_phase(mu, q0, q1, 1e-9)

        self.assertClose(bound, PINNED, atol=1e-7, msg="certified bound != PINNED")
        self.assertLess(gap, 1e-9, msg="gap above 1e-9")
        self.assertGreater(iters, 0, msg="iters = 0")

        mu = 0.01
        q0, q1 = _core.sdp_gains(mu, 0.5, channel(40.0), P_DARK)
        rows = [_core.sdp_phase(mu, q0, q1, t) for t in (1e-4, 1e-6, 1e-8, 1e-10)]
        gaps = [r[1] for r in rows]

        self.assertMonotone(gaps, rising=False, msg="certified gap not falling")

        best = rows[-1][0]
        for value, gap, _ in rows:
            self.assertGreaterEqual(value, best - 1e-12, msg="looser ask returned a tighter bound")
            self.assertLessEqual(value, best + gap + 1e-12, msg="bound outside its own gap")

    def test_abort_raw(self):
        """
        Past the monitoring line's reach the bound returns above 1/2 unclamped, so the depth of
        the failure is readable.
        """
        mu = 0.3 * channel(130.0)
        bound = phase_pair(130.0, mu, 0.5)[0]

        self.assertGreater(bound, 0.5, msg="bound clamped at or below 1/2")
        self.assertLess(bound, 1.5, msg="bound above 1.5")


class SdpReach(Question):
    """
    Figure 1 of the review: what the exact worst case buys in rate and distance.
    """

    def test_rate_doubles(self):
        """
        At each bound's rate optimum the certified rate is 2.087, 1.879 and 1.876 times the
        analytic one, Fig. 1(a)'s headline, falling as the square of the transmittance.
        """
        got = []
        fast = {}
        for dist in (0.0, 20.0, 40.0):
            fast[dist] = rate(dist, *PEAK_SDP[dist], "sdp")
            slow = rate(dist, *PEAK_GAO[dist], "gao")
            self.assertGreater(slow, 0.0, msg=f"{dist} km: analytic {slow}")

            got.append(fast[dist] / slow)

        for saw, want in zip(got, DOUBLING):
            self.assertClose(saw, want, atol=0.01, msg=f"ratio {saw:.3f}")
            self.assertGreater(saw, 1.8, msg="ratio below 1.8")
            self.assertLess(saw, 2.2, msg="ratio above 2.2")

        span = 10 ** (ALPHA * 20.0 / 10.0)
        slope = math.log(fast[20.0] / fast[40.0]) / math.log(span)

        self.assertGreater(slope, 1.8, msg=f"log slope {slope:.3f}")
        self.assertLess(slope, 2.2, msg="log slope above 2.2")

    def test_peaks_local(self):
        """
        The pinned operating points are local maxima in both the intensity and the splitter.
        """
        for dist, kind, peak in ((0.0, "sdp", PEAK_SDP), (0.0, "gao", PEAK_GAO)):
            mu, t_b = peak[dist]
            here = rate(dist, mu, t_b, kind)

            for scale in (0.7, 1.4):
                self.assertLess(
                    rate(dist, mu * scale, t_b, kind),
                    here,
                    msg=f"{kind}: intensity {mu * scale:.4g} is worse",
                )

            for shift in (-0.12, 0.12):
                self.assertLess(
                    rate(dist, mu, t_b + shift, kind),
                    here,
                    msg=f"{kind}: splitter {t_b + shift:.3f} is worse",
                )

    def test_reach_extends(self):
        """
        The certified rate survives 100 km on this exam's channel, where the analytic one is
        live at 90 km and gone by 95 km.
        """
        dist, mu, t_b = FAR

        self.assertGreater(survey(90.0, "gao", 60, 19), 0.0, msg="analytic rate zero at 90 km")
        self.assertEqual(survey(95.0, "gao", 60, 19), 0.0, msg="analytic rate live at 95 km")
        self.assertEqual(survey(dist, "gao", 60, 19), 0.0, msg="analytic rate live at 100 km")
        self.assertGreater(rate(dist, mu, t_b, "sdp"), 0.0, msg="certified rate zero at 100 km")

    def test_abort_later(self):
        """
        The analytic bound reaches the $E_p \\ge 1/2$ abort between 110 and 125 km, where the
        certified one is still below it.
        """
        rows = {}
        for dist in (110.0, 125.0):
            rows[dist] = (floor_ep(dist, "gao"), floor_ep(dist, "sdp"))
            slow, fast = rows[dist]
            self.assertLess(fast, slow, msg=f"{dist} km: {fast:g} vs {slow:g}")
        self.assertLess(rows[110.0][0], 0.5, msg="analytic floor above 1/2 at 110 km")
        self.assertGreater(rows[125.0][0], 0.5, msg="analytic floor below 1/2 at 125 km")
        self.assertLess(rows[125.0][1], 0.5, msg="certified floor above 1/2 at 125 km")


class SdpGuards(Guarded):
    """
    Argument domains, and the records the solver refuses rather than answers wrongly.
    """

    def test_gains_domain(self):
        """
        `sdp_gains` and `sdp_honest` refuse an unphysical intensity, splitter, transmittance
        or dark rate.
        """
        ok = (0.02, 0.5, 0.8, 1e-8)
        cases = (
            (0, "mu", (0.0, -1.0, float("nan"))),
            (1, "t_b", (-0.1, 1.5)),
            (2, "eta", (0.0, 1.5)),
            (3, "dark", (1.0, -1e-9)),
        )

        self.assertSlots(_core.sdp_gains, ok, cases, msg="sdp_gains")
        self.assertSlots(_core.sdp_honest, ok, cases, msg="sdp_honest")

    def test_record_shape(self):
        """
        Three monitoring records are refused: not four gains per detector, two detectors
        together clicking more than once, which no sub-POVM does, and a zero gain, which leaves
        the constraint set no interior.
        """
        good = [1e-8, 1e-4, 1e-5, 1e-5]
        over = [0.6, 1e-4, 1e-5, 1e-5]
        none = [0.0, 1e-4, 1e-5, 1e-5]

        self.assertBad(
            "four sequence gains",
            _core.sdp_phase,
            (0.02, good[:3], good),
            msg="a short record at the constructive port",
        )
        self.assertBad(
            "four sequence gains",
            _core.sdp_analytic,
            (0.02, good, good + [0.1]),
            msg="a long one at the destructive port",
        )
        self.assertBad(
            "sub-POVM",
            _core.sdp_phase,
            (0.02, over, [0.6] + good[1:]),
            msg="a vacuum sequence clicking 120% of the time",
        )
        self.assertBad("interior", _core.sdp_phase, (0.02, none, good), msg="zero at D_M0")
        self.assertBad("interior", _core.sdp_phase, (0.02, good, none), msg="zero at D_M1")

    def test_kato_domain(self):
        """
        `sdp_kato` refuses a block of no rounds, a failure probability outside (0, 1), and
        more clicks than there were rounds to click in.
        """
        self.assertSlots(
            _core.sdp_kato,
            (1e6, 1e3, 1e-10, True),
            (
                (0, "k", (0.0, -1.0, float("nan"))),
                (2, "eps", (0.0, 1.0, -1e-9)),
            ),
            msg="sdp_kato",
        )
        self.assertBad(
            "must not exceed k",
            _core.sdp_kato,
            (1e6, 1e6 + 1.0, 1e-10, True),
            msg="more clicks than rounds",
        )

    def test_interval_shape(self):
        """
        `sdp_interval` refuses a mis-shaped record, a decoy lower bound above its upper bound,
        as records from two blocks arrive, and a bit sequence clicking more than once.
        """
        good = [1e-8, 1e-4, 1e-5, 1e-5]

        self.assertBad(
            "two lower bounds",
            _core.sdp_interval,
            (0.02, good, good, [1e-8]),
            msg="one lower bound where two are needed",
        )
        self.assertBad(
            "exceeds its upper bound",
            _core.sdp_interval,
            (0.02, good, good, [1e-7, 1e-4]),
            msg="a lower bound above its upper one",
        )
        self.assertBad(
            "sub-POVM",
            _core.sdp_interval,
            (0.02, good[:2] + [0.6, 1e-5], good[:2] + [0.6, 1e-5], [1e-8, 1e-4]),
            msg="a bit sequence clicking 120% of the time",
        )

    def test_certified_finite(self):
        """
        `sdp_finite` refuses naming its three missing pieces and checks its rounds slot first,
        while the analytic chain beside it runs.
        """
        self.assertFails(
            NotImplementedError,
            "THE CONSTRAINT SET IS THE WRONG SHAPE",
            _core.sdp_finite,
            1e11,
            msg="the equality record against the Kato box",
        )
        self.assertFails(
            NotImplementedError,
            "THE EPSILONS DO NOT COMPOSE",
            _core.sdp_finite,
            1e11,
            msg="six bounded gains against ten",
        )
        self.assertFails(
            NotImplementedError,
            "THE CLAIM IS NOT IN PRINT",
            _core.sdp_finite,
            1e11,
            msg="the source states the composition as a conditional",
        )
        self.assertBad("rounds", _core.sdp_finite, (0.0,), msg="rounds slot unguarded")
        mu, t_b = PEAK_GAO[20.0]

        self.assertGreater(keylength(20.0, mu, t_b, BLOCK), 0.0, msg="analytic keylength = 0")

    def test_lmi_start(self):
        """
        `sdp_lmi` refuses a start outside the feasible region, where its barrier is undefined,
        and a block short of its matrices.
        """
        flat = [-2.0, -0.3, -0.3, -1.0, 1.0, 0.0, 0.0, 1.0]

        self.assertBad(
            "strictly feasible",
            _core.sdp_lmi,
            ([1.0], [2], flat, [0.0]),
            msg="a start below the cap",
        )
        self.assertBad(
            "mats must hold",
            _core.sdp_lmi,
            ([1.0], [2], flat[:6], [9.0]),
            msg="a block short of its matrices",
        )

    def test_solver_refuses(self):
        """
        A record the interior-point path cannot resolve raises rather than returning the loose
        bound it stopped at.
        """
        mu = 1e-5
        q0, q1 = _core.sdp_gains(mu, 0.5, channel(100.0), P_DARK)

        self.assertFails(
            ValueError,
            "did not resolve",
            _core.sdp_phase,
            mu,
            q0,
            q1,
            1e-9,
            msg="mu = 1e-5 at 100 km",
        )


# `herm_eig` returns eigenVECTORS; `herm_cone` is sdp_lmi's scheme on a complex Hermitian block.
# Exposed for numpy arbitration here, not as an API.
# THREE CONES, THREE EXAMS: SdpSolver (real symmetric), HermSolver (complex Hermitian) and
# LpEpigraph (diagonal) share no fixture, as each engine's header keeps them apart.


def hermit(seed, n):
    """
    A pseudorandom Hermitian matrix of side n, as flat (real, imaginary) planes.
    """
    rng = random.Random(seed)
    re = [[0.0] * n for _ in range(n)]
    im = [[0.0] * n for _ in range(n)]
    for i in range(n):
        re[i][i] = rng.uniform(-2.0, 2.0)
        for j in range(i + 1, n):
            a = rng.uniform(-1.0, 1.0)
            b = rng.uniform(-1.0, 1.0)
            re[i][j] = a
            re[j][i] = a
            im[i][j] = b
            im[j][i] = -b

    flat_re = [re[i][j] for i in range(n) for j in range(n)]
    flat_im = [im[i][j] for i in range(n) for j in range(n)]

    return flat_re, flat_im


def spectrum(re, im):
    """
    The eigenvalues numpy reads from the same matrix, ascending.
    """
    n = int(round(len(re) ** 0.5))
    a = np.asarray(re).reshape(n, n) + 1j * np.asarray(im).reshape(n, n)

    return np.linalg.eigvalsh(a)


class SdpFinite(Question):
    """
    Li, Cao, Xie, Yin & Chen's composable finite key over the four-sequence record: Kato on the
    decoy counts, Gao's estimator over the interval, an entropic-uncertainty length.
    """

    def test_kato_beats_azuma(self):
        """
        Kato's deviation never exceeds Azuma's, equals it at half the rounds, and beats it 5x,
        158x and 3353x as the count falls toward the dark-count floor.
        """
        rounds = 1e11
        eps = 1e-11
        azuma = math.sqrt(0.5 * rounds * math.log(1.0 / eps))
        for count, want in BIAS:
            got = _core.sdp_kato(rounds, count, eps, True)
            self.assertClose(azuma / got, want, atol=1e-4 * want, msg=f"count = {count:g}")

        half = _core.sdp_kato(rounds, 0.5 * rounds, eps, True)

        self.assertClose(half, azuma, atol=1e-9 * azuma, msg="kato at k/2 != azuma")
        for count in (1e2, 1e5, 1e8, 1e10, 5e10, 9e10):
            for way in (True, False):
                self.assertLessEqual(
                    _core.sdp_kato(rounds, count, eps, way),
                    azuma * (1.0 + 1e-12),
                    msg=f"count = {count:g}, upper = {way}",
                )

    def test_kato_directions(self):
        """
        The two Kato directions differ, the upper larger below half the rounds and smaller
        above, and both widen as the failure probability tightens.
        """
        rounds = 1e11
        for count in (1e2, 1e5, 1e9):
            up = _core.sdp_kato(rounds, count, 1e-11, True)
            down = _core.sdp_kato(rounds, count, 1e-11, False)
            self.assertGreater(up, down, msg=f"count = {count:g} sits below half")

        for count in (6e10, 9.9e10):
            up = _core.sdp_kato(rounds, count, 1e-11, True)
            down = _core.sdp_kato(rounds, count, 1e-11, False)
            self.assertLess(up, down, msg=f"count = {count:g} sits above half")

        for way in (True, False):
            widths = [_core.sdp_kato(rounds, 1e6, e, way) for e in (1e-4, 1e-7, 1e-11, 1e-15)]
            self.assertMonotone(widths, rising=True, msg=f"upper = {way}")

    def test_interval_reduces(self):
        """
        Li's expanded estimator with its interval collapsed onto the observed gains equals
        Gao's grouped one to 1e-14 at every pinned operating point.
        """
        for dist, (mu, t_b) in sorted(PEAK_GAO.items()):
            q0, q1 = _core.sdp_gains(mu, t_b, channel(dist), P_DARK)
            self.assertClose(
                _core.sdp_interval(mu, q0, q1, [q0[0], q0[1]]),
                _core.sdp_analytic(mu, q0, q1),
                atol=1e-14,
                msg=f"{dist} km",
            )

    def test_interval_widens(self):
        """
        The interval bound sits above the exact one at every block size and falls toward it,
        within 0.2% at 1e13 rounds.
        """
        mu, t_b = PEAK_GAO[20.0]
        q0, q1 = _core.sdp_gains(mu, t_b, channel(20.0), P_DARK)
        exact = _core.sdp_analytic(mu, q0, q1)
        seen = []
        for rounds in BLOCKS:
            hi0, hi1, lo = widen(20.0, mu, t_b, rounds)
            seen.append(_core.sdp_interval(mu, hi0, hi1, lo))
        self.assertMonotone(seen, rising=False, msg="interval bound not falling with the block")
        for value in seen:
            self.assertGreaterEqual(value, exact - 1e-15, msg="interval bound below the exact one")
        self.assertClose(seen[-1], exact, atol=2e-3 * exact, msg="1e13 interval bound off the exact one")

    def test_length_below_rate(self):
        """
        The finite key reaches 0.9852, 0.9624 and 0.9052 of the asymptotic rate at the three
        pinned operating points at a 1e11-round block, the shortfall growing with distance.
        """
        shares = []
        for dist, (mu, t_b) in sorted(PEAK_GAO.items()):
            share = keylength(dist, mu, t_b, BLOCK) / BLOCK / ceiling(dist, mu, t_b)
            self.assertLess(share, 1.0, msg=f"{dist} km reaches {share:.6f} of the ceiling")
            self.assertClose(share, SHARES[dist], atol=1e-4, msg=f"{dist} km")
            shares.append(share)
        self.assertMonotone(shares, rising=False, msg="share not falling with distance")

    def test_length_converges(self):
        """
        The length per round rises toward the asymptotic rate with the block, from 5.2% of it at
        1e8 rounds to 99.6% at 1e13, never reaching it.
        """
        mu, t_b = PEAK_GAO[20.0]
        cap = ceiling(20.0, mu, t_b)
        shares = [keylength(20.0, mu, t_b, n) / n / cap for n in BLOCKS]

        self.assertMonotone(shares, rising=True, msg="share not rising with the block")
        for got, want in zip(shares, GROWTH):
            self.assertClose(got, want, atol=1e-4, msg=f"share {want}")
        self.assertLess(shares[-1], 1.0, msg="1e13 share at or above 1")

    def test_length_budget(self):
        """
        Halving eps_sec costs exactly two bits and halving eps_cor exactly one, the length is a
        whole number of bits, and a phase error at the abort leaves nothing to hash.
        """
        base = _core.sdp_length(1e6, 0.01, 0.1, 1.1, EPS_SEC, EPS_COR)

        self.assertClose(
            base - _core.sdp_length(1e6, 0.01, 0.1, 1.1, 0.5 * EPS_SEC, EPS_COR),
            2.0,
            atol=1e-9,
            msg="2*log2(5/eps_sec)",
        )
        self.assertClose(
            base - _core.sdp_length(1e6, 0.01, 0.1, 1.1, EPS_SEC, 0.5 * EPS_COR),
            1.0,
            atol=1e-9,
            msg="log2(2/eps_cor)",
        )
        self.assertEqual(base, math.floor(base), msg="length is not an integer")
        self.assertEqual(
            _core.sdp_length(1e6, 0.01, 0.49, 1.1, EPS_SEC, EPS_COR),
            0.0,
            msg="length != 0 at e_ph = 0.49",
        )

    def test_length_reach(self):
        """
        The finite key distils at 40 km and is zero at 95 and 110 km, where the asymptotic bound
        has also stopped.
        """
        mu, t_b = PEAK_GAO[40.0]

        self.assertGreater(keylength(40.0, mu, t_b, BLOCK), 0.0, msg="keylength = 0 at 40 km")
        for dist in (95.0, 110.0):
            for k in (0.3, 0.1, 0.03):
                self.assertEqual(
                    keylength(dist, k * channel(dist), 0.5, BLOCK),
                    0.0,
                    msg=f"{dist} km, mu = {k} eta",
                )
                self.assertEqual(survey(dist, "gao"), 0.0, msg=f"{dist} km asymptotic")


class HermSolver(Question):
    """
    The complex Hermitian eigendecomposition and the cone solver that reads it.
    """

    def test_eig_spectrum(self):
        """
        `herm_eig` eigenvalues of pseudorandom Hermitian matrices match numpy's eigvalsh to
        1e-11 with reconstruction and unitarity residuals under 1e-13, and the identity returns
        exact values and orthonormal vectors.
        """
        for seed, n in ((1, 4), (2, 7), (3, 12), (4, 20)):
            re, im = hermit(seed, n)
            got = sorted(_core.herm_eig(re, im)[0])
            want = spectrum(re, im)
            self.assertClose(
                max(abs(a - b) for a, b in zip(got, want)),
                0.0,
                atol=1e-11,
                msg=f"seed {seed} side {n} spectrum",
            )

        worst = 0.0
        for seed, n in ((5, 3), (6, 9), (7, 16), (8, 24)):
            re, im = hermit(seed, n)
            back, unit = _core.herm_residual(re, im)
            worst = max(worst, back, unit)
        self.assertClose(worst, 0.0, atol=1e-13, msg="reconstruction and unitarity residual")

        n = 6
        re = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        im = [0.0] * (n * n)
        vals, _vr, _vi = _core.herm_eig(re, im)
        back, unit = _core.herm_residual(re, im)

        self.assertClose(max(abs(v - 1.0) for v in vals), 0.0, msg="identity eigenvalues")
        self.assertClose(max(back, unit), 0.0, atol=1e-15, msg="identity residual")

    def test_cone_scalar(self):
        """
        On a diagonal problem with a closed-form optimum the cone solver reaches it.
        """
        # min Tr(rho W), W = diag(3, 1), Tr(rho) = 1: optimum 1.
        w_re = [3.0, 0.0, 0.0, 1.0]
        w_im = [0.0] * 4
        terms = [0, 0, 1.0, 0.0, 1, 1, 1.0, 0.0]
        dual, gap, _iters, status, y = _core.herm_cone(w_re, w_im, terms, [2], [1.0], [-10.0], 1e-11)

        self.assertClose(dual, 1.0, atol=1e-8, msg="closed-form optimum")
        self.assertClose(y[0], 1.0, atol=1e-8, msg="y[0] != 1.0")
        self.assertEqual(status, "converged", msg="status != converged")
        self.assertLess(gap, 1e-10, msg="gap above 1e-10")

    def test_cone_duality(self):
        """
        The certified dual on the trace-one problem is the least eigenvalue, and a loose
        tolerance returns a smaller dual than a tight one, never larger.
        """
        n = 5
        re, im = hermit(11, n)
        terms = []
        for i in range(n):
            terms += [i, i, 1.0, 0.0]

        dual, _gap, _iters, _status, _y = _core.herm_cone(re, im, terms, [n], [1.0], [-40.0], 1e-10)
        vals = spectrum(re, im)

        # Tr(rho W) over the trace-one states is minimised by the smallest eigenvalue of W.
        self.assertClose(dual, float(vals[0]), atol=1e-8, msg="dual != lambda_min")

        n = 6
        re, im = hermit(12, n)
        terms = []
        for i in range(n):
            terms += [i, i, 1.0, 0.0]

        loose = _core.herm_cone(re, im, terms, [n], [1.0], [-40.0], 1e-1)[0]
        tight = _core.herm_cone(re, im, terms, [n], [1.0], [-40.0], 1e-11)[0]
        exact = float(spectrum(re, im)[0])

        self.assertLessEqual(loose, tight + 1e-12, msg="loose value above the tight one")
        self.assertLessEqual(loose, exact + 1e-12, msg="loose dual above the exact value")


class HermSplit(Question):
    """
    The same cone over several blocks, the shape of an INEQUALITY feasible set: a side-one block
    is a sign condition on one dual variable.
    """

    def test_one_block(self):
        """
        On the trace-one problem `herm_split` over one block returns the same dual, status and
        point as `herm_cone`, and it is the least eigenvalue.
        """
        n = 5
        re, im = hermit(11, n)
        flat = []
        wide = []
        for i in range(n):
            flat += [i, i, 1.0, 0.0]
            wide += [0, i, i, 1.0, 0.0]

        one = _core.herm_cone(re, im, flat, [n], [1.0], [-40.0], 1e-10)
        many = _core.herm_split([n], re, im, wide, [n], [1.0], [-40.0], 1e-10)

        self.assertClose(many[0], one[0], atol=1e-14, msg="herm_split dual != herm_cone's")
        self.assertEqual(many[3], one[3], msg="herm_split status != herm_cone's")
        self.assertClose(many[4][0], one[4][0], atol=1e-14, msg="herm_split point != herm_cone's")
        self.assertClose(one[0], float(spectrum(re, im)[0]), atol=1e-8, msg="dual != lambda_min")

    def test_sign_block(self):
        """
        A matrix block with a sign block minimises Tr(rho W) over SUBNORMALISED states at
        $\\min(0, \\lambda_{min}(W))$, the shape Kanitschar Eq. (21)'s trace window takes.
        """
        for seed, n in ((11, 4), (12, 6), (13, 8)):
            re, im = hermit(seed, n)
            terms = []
            for i in range(n):
                terms += [0, i, i, -1.0, 0.0]
            terms += [1, 0, 0, -1.0, 0.0]
            planes = re + [0.0]
            zeros = im + [0.0]
            dual, _gap, _iters, status, y = _core.herm_split(
                [n, 1], planes, zeros, terms, [n + 1], [-1.0], [40.0], 1e-11
            )
            want = min(0.0, float(spectrum(re, im)[0]))
            self.assertClose(dual, want, atol=1e-7, msg=f"seed {seed}: min(0, lambda_min) = {want}")
            self.assertGreater(y[0], -1e-12, msg="y[0] negative")
            self.assertEqual(status, "converged", msg=f"seed {seed} converged")

    def test_split_duality(self):
        """
        Three diagonal blocks make a linear programme whose certified value stays below the
        closed-form optimum 3, a loose tolerance returning less than a tight one.
        """
        # max y0 + y1 over 3 - y0 - y1 >= 0, 2 - y0 >= 0, y1 >= 0, whose optimum is 3.
        planes = [3.0, 2.0, 0.0]
        zeros = [0.0, 0.0, 0.0]
        terms = [0, 0, 0, 1.0, 0.0, 1, 0, 0, 1.0, 0.0, 0, 0, 0, 1.0, 0.0, 2, 0, 0, -1.0, 0.0]
        args = ([1, 1, 1], planes, zeros, terms, [2, 2], [1.0, 1.0], [0.5, 0.5])
        tight = _core.herm_split(*args, 1e-12)[0]
        loose = _core.herm_split(*args, 1e-1)[0]

        self.assertLess(tight, 3.0, msg="tight value at or above 3.0")
        self.assertClose(tight, 3.0, atol=1e-6, msg=f"tight {tight}")
        self.assertLessEqual(loose, tight + 1e-12, msg="loose value above the tight one")
        self.assertLessEqual(loose, 3.0, msg="loose value above 3.0")

    def test_split_refusals(self):
        """
        Malformed split problems are refused by name: a side-zero block, planes not matching the
        declared sides, terms not five per nonzero, an entry naming a missing block, and a start
        outside the cone.
        """
        good = ([1, 1], [1.0, 1.0], [0.0, 0.0], [0, 0, 0, 1.0, 0.0], [1], [1.0], [0.5])

        self.assertFails(
            ValueError,
            "no block of side zero",
            lambda: _core.herm_split([0], [], [], [], [], [], []),
            msg="a block of side zero",
        )
        self.assertFails(
            ValueError,
            "concatenated planes",
            lambda: _core.herm_split([1, 2], [1.0, 1.0], [0.0, 0.0], [0, 0, 0, 1.0, 0.0], [1], [1.0], [0.5]),
            msg="planes short of the declared sides",
        )
        self.assertFails(
            ValueError,
            "five per nonzero",
            lambda: _core.herm_split(good[0], good[1], good[2], [0, 0, 0, 1.0], good[4], good[5], good[6]),
            msg="four numbers per entry",
        )
        self.assertFails(
            ValueError,
            "names block",
            lambda: _core.herm_split(good[0], good[1], good[2], [7, 0, 0, 1.0, 0.0], good[4], good[5], good[6]),
            msg="a block index past the end",
        )
        self.assertFails(
            ValueError,
            "not strictly feasible",
            lambda: _core.herm_split(*good[:6], [2.0]),
            msg="a start outside the cone",
        )


class HermGuards(Guarded):
    """
    What the Hermitian substrate refuses.
    """

    def test_not_square(self):
        """
        `herm_eig` refuses a plane of non-square length, a matrix whose triangles disagree
        rather than symmetrising it, and real and imaginary planes of different lengths.
        """

        self.assertFails(
            ValueError,
            "not a square matrix",
            _core.herm_eig,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            msg="three entries name no matrix",
        )
        self.assertFails(
            ValueError,
            "not Hermitian",
            _core.herm_eig,
            [1.0, 2.0, 0.5, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            msg="an asymmetric real matrix",
        )
        self.assertFails(
            ValueError,
            "same length",
            _core.herm_eig,
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
            msg="mismatched planes",
        )

    def test_cone_start(self):
        """
        `herm_cone` refuses a start that is not strictly dual feasible, where its barrier is
        undefined, and terms disagreeing with the declared nonzero counts.
        """

        self.assertFails(
            ValueError,
            "not strictly dual feasible",
            _core.herm_cone,
            [1.0, 0.0, 0.0, 1.0],
            [0.0] * 4,
            [0, 0, 1.0, 0.0, 1, 1, 1.0, 0.0],
            [2],
            [1.0],
            [5.0],
            1e-9,
            msg="a shift past the smallest eigenvalue",
        )
        self.assertFails(
            ValueError,
            "four per nonzero",
            _core.herm_cone,
            [1.0, 0.0, 0.0, 1.0],
            [0.0] * 4,
            [0, 0, 1.0],
            [2],
            [1.0],
            [-5.0],
            1e-9,
            msg="a truncated coordinate list",
        )


# Cutting-plane models with NEGATIVE maxima, (dim, cuts, levels, peak); two symmetric planes peak
# at the simplex midpoint, levels[0] + (g_00 + g_01)/2.
NEGATIVE = (
    (2, [1.0, -1.0, -1.0, 1.0], [-1.0, -1.0], -1.0),
    (2, [1.0, -1.0, -1.0, 1.0], [-0.1, -0.1], -0.1),
    (2, [0.5, 0.25, 0.25, 0.5], [-1.0, -1.0], -0.625),
    (2, [1.0, 2.0, 2.0, 1.0], [-4.0, -4.0], -2.5),
)

# Regression pins for the two shipped `lp::polytope_cap` assemblies, where the shift must stay inert.
SHIPPED = (0.36412672764950227, 0.5357742769434696)


def epigraph(dim, cuts, levels, shift):
    """
    `lp::polytope_cap`'s assembly over the bare simplex as `(c, A, b, start)` for `lp_dual`,
    lifted by `shift` as the engine lifts it.
    """
    k = len(levels)
    n = dim + 1 + k
    m = k + 1
    c = [0.0] * n
    a = [0.0] * (m * n)
    b = [0.0] * m
    c[dim] = -1.0
    for row in range(k):
        at = row * n
        for j in range(dim):
            a[at + j] = cuts[row * dim + j]

        a[at + dim] = -1.0
        a[at + dim + 1 + row] = -1.0
        b[row] = -levels[row] - shift

    for j in range(dim):
        a[k * n + j] = 1.0

    b[k] = 1.0
    share = 2.0 / k
    start = [share] * k + [0.0]
    start[k] = -max(sum(share * cuts[row * dim + j] for row in range(k)) for j in range(dim)) - 1.0

    return c, a, b, start


def epifloor(dim, cuts, levels):
    """
    The shift's floor: each cut plane at its smallest coordinate.
    """

    return min(levels[r] + min(cuts[r * dim + j] for j in range(dim)) for r in range(len(levels)))


def epimodel(dim, cuts, levels, x):
    """
    `min_k (levels[k] + g_k . x)`, the concave model the cap is written over.
    """

    return min(levels[r] + sum(cuts[r * dim + j] * x[j] for j in range(dim)) for r in range(len(levels)))


class LpEpigraph(Question):
    """
    `lp::polytope_cap`'s epigraph variable: `tau >= 0` empties the primal wherever the model's
    maximum is negative, so the engine shifts the floor to zero and back; `epigraph` transcribes
    that assembly.
    """

    def test_clamp_empties_primal(self):
        """
        A model peaking at -1 leaves the clamped primal empty, so the unshifted dual runs along
        a ray above 1e6, its negative far below the peak.
        """
        dim, cuts, levels, peak = NEGATIVE[0]

        self.assertClose(epimodel(dim, cuts, levels, [0.5, 0.5]), peak, msg="model peak != -1 at the midpoint")

        value, _gap, iters, _y = _core.lp_dual(*epigraph(dim, cuts, levels, 0.0), 1e-9)

        self.assertGreater(value, 1e6, msg="dual at or below 1e6")
        self.assertGreater(iters, 0, msg="iters = 0")
        self.assertLess(-value, peak - 1e6, msg="-(b . y) not far below the peak")

    def test_shift_recovers_maximum(self):
        """
        Shifting by the model's floor and back bounds four negative-peak models from above
        within 1e-9 of the closed-form peak, where the unshifted assembly reads over 1e12 below.
        """
        for dim, cuts, levels, peak in NEGATIVE:
            lift = -epifloor(dim, cuts, levels)
            self.assertGreater(lift, 0.0, msg=f"peak {peak} has its floor below zero")

            plain = -_core.lp_dual(*epigraph(dim, cuts, levels, 0.0), 1e-9)[0]
            self.assertLess(plain, peak - 1e12, msg=f"peak {peak} unshifted reads far below it")

            bound = -_core.lp_dual(*epigraph(dim, cuts, levels, lift), 1e-9)[0] - lift
            self.assertGreaterEqual(bound, peak, msg=f"peak {peak} shifted bounds it from above")
            self.assertClose(bound, peak, atol=1e-9, msg=f"peak {peak} shifted is tight on it")

    def test_floor_bounds_model(self):
        """
        The floor sits at or below the model everywhere on the simplex, over a hundred
        pseudorandom sets of one to three cuts.
        """
        rng = random.Random(90412)
        for _ in range(100):
            dim = rng.randrange(2, 6)
            cuts = []
            levels = []
            for _ in range(rng.randrange(1, 4)):
                cuts += [rng.uniform(-3.0, 3.0) for _ in range(dim)]
                levels.append(rng.uniform(-3.0, 3.0))

            low = epifloor(dim, cuts, levels)
            for _ in range(8):
                draw = [rng.random() for _ in range(dim)]
                total = sum(draw)
                x = [d / total for d in draw]
                self.assertGreaterEqual(epimodel(dim, cuts, levels, x), low, msg="model below the floor")

    def test_shift_stays_inert(self):
        """
        A model with its floor at or above zero, as every shipped assembly has, is assembled
        unshifted: a dominating plane pair caps at its peak and both rrdps caps return their
        pins.
        """
        dim = 2
        cuts = [1.0, -1.0, -1.0, 1.0]
        levels = [2.0, 2.0]

        self.assertGreaterEqual(epifloor(dim, cuts, levels), 0.0, msg="floor below zero")

        flat = -_core.lp_dual(*epigraph(dim, cuts, levels, 0.0), 1e-9)[0]

        self.assertClose(flat, epimodel(dim, cuts, levels, [0.5, 0.5]), atol=1e-9, msg="cap != the model peak")

        self.assertClose(_core.rrdps_simplex(16, 2)[0], SHIPPED[0], atol=1e-12, msg="rrdps_simplex off its pin")
        self.assertClose(
            _core.rrdps_polytope(16, 0.1, 0.5)[0], SHIPPED[1], atol=1e-12, msg="rrdps_polytope off its pin"
        )


if __name__ == "__main__":
    rc = Exam(
        "SdpSolver",
        "The interior-point solver on problems whose optimum is known in closed form",
        "sdp_solver.md",
    ).run(load(SdpSolver))
    rc |= Exam(
        "SdpStates",
        "The four COW' sequences, their overlaps, and the honest monitoring record",
        "sdp_states.md",
    ).run(load(SdpStates))
    rc |= Exam(
        "SdpBound",
        "The certified worst case against the analytic estimator and the honest channel",
        "sdp_bound.md",
    ).run(load(SdpBound))
    rc |= Exam(
        "SdpReach",
        "Figure 1: what the exact worst case buys in rate and in distance",
        "sdp_reach.md",
    ).run(load(SdpReach))
    rc |= Exam(
        "SdpFinite",
        "Li et al.'s composable finite key over the four-sequence record",
        "sdp_finite.md",
    ).run(load(SdpFinite))
    rc |= Exam(
        "HermSolver",
        "The complex Hermitian eigendecomposition and the cone solver built on it",
        "herm_solver.md",
    ).run(load(HermSolver))
    rc |= Exam(
        "HermSplit",
        "The same cone over several blocks, which is what an inequality feasible set needs",
        "herm_split.md",
    ).run(load(HermSplit))
    rc |= Exam(
        "HermGuards",
        "Shapes and starting points the Hermitian substrate refuses",
        "herm_guards.md",
    ).run(load(HermGuards))
    rc |= Exam(
        "SdpGuards",
        "Every argument domain, and the records the solver refuses to answer",
        "sdp_guards.md",
    ).run(load(SdpGuards))
    rc |= Exam(
        "LpEpigraph",
        "The epigraph variable the cutting-plane cap maximises, and the clamp that empties its primal",
        "lp_epigraph.md",
    ).run(load(LpEpigraph))
    sys.exit(rc)
