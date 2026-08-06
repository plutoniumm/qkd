import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

from kit.cache import memo
from kit.checks import Guarded
from qkd import _core


def fiber(d):
    return 10 ** (-0.02 * d)


def rate(m=4, alpha=0.4, d=0.0, xi=0.01, eta=1.0, vel=0.0, beta=0.95):
    return _core.dm_rate(m, alpha, fiber(d), xi, eta, vel, beta)


def key(**kw):
    return rate(**kw)[2]


def paper(m=4, alpha=0.4, d=0.0, xi=0.01, beta=0.95):
    """
    Denys et al.'s own rate: Eq. (Z-PSK) into dm_holevo with an unbounded alphabet, i.e. the Gaussian I(X;Y).
    """
    va, zl, w, _zg = _core.dm_moments(m, alpha)
    t = fiber(d)
    z = max(0.0, math.sqrt(t) * (zl - math.sqrt(2 * xi * w)))

    return _core.dm_holevo(va, t, xi, z, float("inf"), beta)[2]


def gauss(va, d=0.0, xi=0.01, beta=0.95):
    return _core.cv_rate(va, fiber(d), xi, 1.0, 0.0, beta, False, True)[2]


# Hand-rolled, not kit.forms.bisect: ~22 halvings against its fixed 200, measured ~9x faster.
def reach(score, **kw):
    """
    Longest fibre, in km, at which ``score`` is still strictly positive.
    """
    lo, hi = 0.0, 400.0
    if score(d=lo, **kw) <= 0.0:
        return 0.0

    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        if score(d=mid, **kw) > 0.0:
            lo = mid
        else:
            hi = mid

    return lo


def best(score, **kw):
    """
    Ternary search for the amplitude maximising ``score``.
    """
    lo, hi = 1e-3, 3.0
    for _ in range(40):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if score(alpha=left, **kw) < score(alpha=right, **kw):
            lo = left
        else:
            hi = right

    mid = 0.5 * (lo + hi)

    return mid, score(alpha=mid, **kw)


def peak(d=0.0, xi=0.01, beta=0.95, hi=40.0):
    """
    The same search for a Gaussian modulation, over modulation variance.
    """
    lo = 1e-3
    for _ in range(60):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if gauss(left, d, xi, beta) < gauss(right, d, xi, beta):
            lo = left
        else:
            hi = right

    return 0.5 * (lo + hi)


class Constellation(Question):
    """
    The M-PSK ring of coherent states and its moments.
    """

    def test_states_ring(self):
        """
        `dm_states` lays m amplitudes of modulus $\\alpha$ at equal angles, the first on the
        positive real axis.
        """
        pts = _core.dm_states(8, 0.4)
        self.assertEqual(len(pts), 8, msg=f"len {len(pts)}")

        for re, im in pts:
            self.assertClose(math.hypot(re, im), 0.4, msg="state modulus")
        self.assertClose(pts[0][1], 0.0, msg="Im of first state")

        angles = [math.atan2(im, re) % (2 * math.pi) for re, im in pts]
        gaps = [angles[k + 1] - angles[k] for k in range(7)]

        for g in gaps:
            self.assertClose(g, 2 * math.pi / 8, msg=f"angle gap {g}")

    def test_penalty_positive(self):
        """
        $w$ is positive and finite for $m$ from 3 to 64 and $\\alpha$ from 0.1 to 3.
        """
        for m in (3, 4, 6, 16, 64):
            for alpha in (0.1, 0.4, 1.0, 3.0):
                w = _core.dm_moments(m, alpha)[2]
                self.assertGreater(w, 0.0, msg=f"w must be > 0 at m={m}, a={alpha}")
                self.assertFinite(w, msg=f"w must be finite at m={m}, a={alpha}")

    def test_moments_saturate(self):
        """
        At $\\alpha = 0.4$ the $m = 6$ correlation is within 1.8e-6 relative of the $m \\to \\infty$ value, both pinned
        to 1e-12.
        """
        z6 = _core.dm_moments(6, 0.4)[1]
        z32 = _core.dm_moments(32, 0.4)[1]
        self.assertClose(z6, 0.85205574403046, atol=1e-12, msg="m = 6 correlation")
        self.assertClose(z32, 0.85205722260945, atol=1e-12, msg="m -> inf correlation")


class Information(Question):
    """
    Exact constellation I(X;Y) against the Gaussian log2(1 + SNR) the literature substitutes.
    """

    def test_info_regression(self):
        """
        8-PSK at $\\alpha = 1$, $T = 1$, $\\xi = 0.01$, $\\beta = 0.95$ turns a 1.9% information gap (0.99640674
        Gaussian against 0.97753325 exact) into a 60% key-rate gap (0.047832 against 0.029903), 2.08% at the optimum
        $\\alpha = 0.6805$, while QPSK at $\\alpha = 0.4$, 0 km differs by 0.043% and at $\\alpha = 0.35$, 20 km by
        0.0016%.
        """
        exact, approx = _core.dm_info(8, 1.0, 1.0, 0.01)
        self.assertClose(exact, 0.977533252932, atol=1e-9, msg="exact I at alpha = 1")
        self.assertClose(approx, 0.996406735, atol=1e-8, msg="Gaussian I at alpha = 1")
        self.assertClose(key(m=8, alpha=1.0), 0.029903, atol=1e-6, msg="dm_rate key at alpha = 1")
        self.assertClose(paper(m=8, alpha=1.0), 0.047832, atol=1e-6, msg="paper() key at alpha = 1")
        self.assertClose(
            paper(m=8, alpha=0.6805) / key(m=8, alpha=0.6805) - 1.0,
            0.0208,
            atol=1e-3,
            msg="gap at alpha = 0.6805",
        )

        exact, approx = _core.dm_info(4, 0.4, 1.0, 0.01)
        self.assertClose(exact / approx, 0.999574, atol=1e-5, msg="QPSK at 0 km")

        exact, approx = _core.dm_info(4, 0.35, fiber(20.0), 0.005)
        self.assertClose(exact / approx, 0.999984, atol=1e-5, msg="QPSK at 20 km")

    def test_info_saturates(self):
        """
        The exact information rises monotonically to $\\log_2 m$ -- 2 bits for QPSK, 6 for 64-PSK at $\\alpha = 100$ --
        while the Gaussian form passes 13 bits.
        """
        infos = [_core.dm_info(8, a, 1.0, 0.01)[0] for a in (0.5, 1.0, 2.0, 4.0)]
        self.assertMonotone(infos, msg="I against alpha")
        self.assertClose(_core.dm_info(4, 100.0, 1.0, 0.01)[0], 2.0, msg="QPSK I at alpha = 100")

        # The shortfall from log2(64) is physical: 2.3e-3 at alpha = 50, 2.0e-11 at alpha = 100.
        self.assertClose(
            _core.dm_info(64, 100.0, 1.0, 0.01)[0],
            6.0,
            atol=1e-10,
            msg="64-PSK at 6 bits",
        )
        self.assertGreater(_core.dm_info(4, 100.0, 1.0, 0.01)[1], 13.0, msg="Gaussian I at alpha = 100")

    def test_info_gaussian(self):
        """
        The Gaussian branch returns $\\log_2(1 + \\mathrm{SNR})$ exactly with noise variance $2 + T\\xi$, one vacuum
        unit for the mode and one for Bob's balanced splitter.
        """
        for va in (0.2, 2.0, 20.0):
            for t in (1.0, 0.3):
                want = math.log2(1 + t * va / (2 + t * 0.01))
                got = _core.dm_info(4, math.sqrt(va / 2), t, 0.01)[1]
                self.assertClose(got, want, msg=f"Gaussian SNR form at va={va}, T={t}")


def integrate(m, alpha, t, xi, steps=80, span=9.0):
    """
    I(X;Y) by trapezoidal integration on a uniform grid, independent of _core's Gauss-Hermite rule.
    """
    s2 = 2.0 + t * xi
    reach = 2.0 * alpha * math.sqrt(t)
    off = []
    for j in range(m):
        ang = 2.0 * math.pi * j / m
        off.append((reach * (1.0 - math.cos(ang)), -reach * math.sin(ang)))

    edge = span * math.sqrt(s2)
    h = 2.0 * edge / steps
    acc = 0.0
    for ia in range(steps + 1):
        u = -edge + ia * h
        wa = 0.5 if ia in (0, steps) else 1.0
        for ib in range(steps + 1):
            v = -edge + ib * h
            wb = 0.5 if ib in (0, steps) else 1.0
            tot = 0.0
            for dx, dy in off:
                arg = (dx * dx + dy * dy) + 2.0 * (dx * u + dy * v)
                tot += math.exp(-arg / (2.0 * s2))

            gauss = math.exp(-(u * u + v * v) / (2.0 * s2))
            acc += wa * wb * gauss * math.log2(tot)

    return math.log2(m) - acc * h * h / (2.0 * math.pi * s2)


class Quadrature(Question):
    """
    The Gauss-Hermite rule underneath dm_info.
    """

    def test_info_positive(self):
        """
        Over 297 cases spanning every accepted m, both node branches and $T$ from 1 to 1e-6, the exact information stays
        positive and below both $\\log_2 m$ and the Gaussian form.
        """
        count = 0
        sizes = (3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64)
        for m in sizes:
            for alpha in (1e-6, 0.05, 0.2, 0.4, 0.4000001, 1.0, 3.0, 20.0, 100.0):
                for t in (1.0, 0.1, 1e-6):
                    count += 1
                    exact, approx = _core.dm_info(m, alpha, t, 0.01)
                    self.assertGreater(exact, 0.0, msg=f"I > 0 at m={m}, a={alpha}")
                    self.assertLessEqual(exact, math.log2(m) + 1e-9, msg=f"I <= log2(m) at m={m}")
                    self.assertLessEqual(exact, approx + 1e-9, msg=f"I <= Gaussian I at m={m}")
        self.assertEqual(count, 297, msg=f"cases swept {count}")

    def test_nodes_ordered(self):
        """
        The quadrature agrees with an independent trapezoidal integration to 5e-11 on both node branches.
        """
        cases = (
            (3, 0.4, 1.0, 0.01),
            (4, 0.4, 1.0, 0.01),
            (4, 0.35, fiber(20.0), 0.005),
            (8, 1.0, 1.0, 0.01),
            (6, 0.7, 0.1, 0.05),
        )
        for m, alpha, t, xi in cases:
            self.assertClose(
                _core.dm_info(m, alpha, t, xi)[0],
                integrate(m, alpha, t, xi),
                atol=5e-11,
                msg=f"quadrature against trapezoid at m={m}, alpha={alpha}",
            )

    def test_weights_normalise(self):
        """
        A dark channel returns $\\log_2 m (1 - S^2/\\pi)$ with $S$ the weight sum, so a zero to 1e-11 asserts $S =
        \\sqrt\\pi$; the 16- and 64-node rules one ulp either side of their switch agree to 1e-11, residual 4.2e-12 at
        $m = 3$.
        """
        for m in (3, 4, 8, 16, 64):
            for alpha in (0.3, 0.4, 0.9):
                dark = _core.dm_info(m, alpha, 1e-14, 0.0)[0]
                self.assertLess(abs(dark), 1e-11, msg=f"weights sum to sqrt(pi) at m={m}, a={alpha}")

        for m in (3, 4, 8, 64):
            few = _core.dm_info(m, 0.4, 1.0, 0.01)[0]
            full = _core.dm_info(m, math.nextafter(0.4, 1.0), 1.0, 0.01)[0]
            self.assertClose(few, full, atol=1e-11, msg=f"16- vs 64-node rule at m={m}")


class Anchors(Question):
    """
    Denys, Brown & Leverrier, Quantum 5, 540 (2021), arXiv:2103.13945, from its LaTeX source and ancillary
    implementation. Pinned against ``paper()``, not ``dm_rate``, which uses the exact I(X;Y).
    """

    def test_zstar_source(self):
        """
        Denys et al. Eq. (Z-PSK) and Fig. 2: 4-PSK at $\\alpha = 0.35$, $\\xi = 0.01$ certifies
        $Z^*(T) = \\sqrt T \\cdot 0.7146537$ from z_lin = 0.73463916 and w = 0.01997093,
        matching the SDP of Ghorai et al., PRX 9, 021059 (2019).
        """
        _va, zl, w, _zg = _core.dm_moments(4, 0.35)
        self.assertClose(zl, 0.7346391595118, atol=1e-12, msg="z_lin at alpha = 0.35")
        self.assertClose(w, 0.0199709339456, atol=1e-12, msg="w at alpha = 0.35")
        self.assertClose(
            rate(alpha=0.35, d=0.0)[3],
            0.7146536978231,
            atol=1e-12,
            msg="Z* at T = 1",
        )
        self.assertClose(
            rate(alpha=0.35, d=30.0)[3],
            math.sqrt(fiber(30.0)) * (zl - math.sqrt(2 * 0.01 * w)),
            atol=1e-14,
            msg="dm_rate's own Z* is Eq. (Z-PSK) rebuilt from the moments",
        )

    def test_psk_distance(self):
        """
        Denys et al. Fig. 3 left at $\\alpha = 0.4$, $\\xi = 0.01$, $\\beta = 0.95$: the M = 4, 5, 6 rates are 0.051785,
        0.054158, 0.054253 bit/symbol, reaching zero at 26.48, 28.40, 28.48 km, and the M = 6 and M = 32 reaches agree
        to 2.5 m, as Section 5 states.
        """
        for m, want in ((4, 0.051784903441), (5, 0.054158253938), (6, 0.054253414000)):
            self.assertClose(paper(m=m, alpha=0.4), want, atol=1e-11, msg=f"{m}-PSK at 0 km")

        for m, want in ((4, 26.4752), (5, 28.3990), (6, 28.4779)):
            self.assertClose(
                reach(paper, m=m, alpha=0.4),
                want,
                atol=1e-3,
                msg=f"{m}-PSK maximum distance",
            )

        # atol 3 m, observed 2.480e-3 km: the M = 6 deficit, not quadrature (paper() runs no Gauss-Hermite).
        self.assertClose(
            reach(paper, m=6, alpha=0.4),
            reach(paper, m=32, alpha=0.4),
            atol=3e-3,
            msg="M = 6 against M = 32 reach",
        )

    def test_ghorai_curve(self):
        """
        The Ghorai et al. PRX 9, 021059 (2019) QPSK curve as replotted in Fig. 9 of Lin,
        Upadhyaya & Lutkenhaus, PRX 9, 041064 (2019): qkd tracks the digitised curve to better
        than 0.5% and dies at 74.1 km against 73.6.
        """
        # Tolerances pin qkd's own values, not the digitisation; reaches to the printed 0.01 km.
        self.assertClose(
            paper(alpha=0.35, d=20.0, xi=0.005),
            0.015537004712,
            atol=1e-11,
            msg="20 km",
        )
        self.assertClose(
            paper(alpha=0.35, d=40.0, xi=0.005),
            0.003610620205,
            atol=1e-11,
            msg="40 km",
        )
        self.assertClose(
            paper(alpha=0.35, d=60.0, xi=0.005),
            0.000575552823,
            atol=1e-11,
            msg="60 km",
        )
        self.assertClose(
            reach(paper, alpha=0.35, xi=0.005),
            74.11,
            atol=0.01,
            msg="xi = 0.005 reach",
        )
        self.assertClose(reach(paper, alpha=0.35, xi=0.01), 23.64, atol=0.01, msg="xi = 0.01 reach")

    def test_qpsk_vanishes(self):
        """
        Denys et al. Fig. 5 caption at 50 km, $\\xi = 0.02$, $\\beta = 0.95$: the best QPSK amplitude, 0.2993, leaves
        -0.00942 bit/symbol while Gaussian modulation still keys.
        """
        alpha, out = best(paper, d=50.0, xi=0.02)
        self.assertLess(out, 0.0, msg=f"QPSK key at 50 km {out}")
        self.assertClose(out, -0.0094166, atol=1e-6, msg="the best QPSK rate at 50 km")
        self.assertClose(alpha, 0.2993, atol=1e-3, msg="best alpha")
        self.assertGreater(gauss(5.0, 50.0, 0.02), 0.03, msg="Gaussian rate at 50 km")

    def test_sdp_gap(self):
        """
        A declared miss against the SDP of Lin, Upadhyaya & Lutkenhaus, PRX 9, 041064 (2019) Fig. 9: at $\\alpha =
        0.35$, $\\xi = 0.005$ this bound reaches 0.80 of it at 0 km and 0.40 at 40 km, and nothing by 40 km at $\\xi =
        0.01$ (not like for like: LUL19 coarse-grains to quadrants, $\\beta = 0.95$ assumes soft information).
        """
        lul = {
            (0.005, 0.0): 0.085557727454,
            (0.005, 20.0): 0.025344818994,
            (0.005, 40.0): 0.009009988870,
            (0.005, 60.0): 0.003232015558,
            (0.005, 80.0): 0.001274013540,
            (0.01, 0.0): 0.071935555792,
            (0.01, 20.0): 0.019382350327,
            (0.01, 40.0): 0.006382179109,
        }
        for (xi, d), sdp in lul.items():
            ours = paper(alpha=0.35, d=d, xi=xi)
            self.assertLess(ours, sdp, msg=f"paper() {ours} against LUL19 {sdp} at {d} km")

        # atol 1e-3: both sides are exact; the slack is the 3-decimal anchors' 5e-4.
        self.assertClose(
            paper(alpha=0.35, d=0.0, xi=0.005) / lul[(0.005, 0.0)],
            0.799,
            atol=1e-3,
            msg="ratio to LUL19 at 0 km",
        )
        self.assertClose(
            paper(alpha=0.35, d=40.0, xi=0.005) / lul[(0.005, 40.0)],
            0.401,
            atol=1e-3,
            msg="ratio to LUL19 at 40 km",
        )
        self.assertLess(paper(alpha=0.35, d=40.0, xi=0.01), 0.0, msg="rate at 40 km, xi = 0.01")


class Limit(Question):
    """
    The tie to cv_rate: the Gaussian correlation returns it exactly and a growing constellation climbs to it.
    """

    def test_holevo_matches(self):
        """
        Fed $\\sqrt{T(V^2 - 1)}$ and an unbounded alphabet, `dm_holevo` reproduces `cv_rate` to 5e-12 over 80 grid
        points, both in SNU with $\\xi$ referred to the channel input.
        """
        # This grid agrees to 9.6e-14; a 2916-point superset to V_A = 100, xi = 1e-4 parts by 3.0e-11.
        count = 0
        for va in (0.02, 0.5, 2.0, 5.0, 20.0):
            for t in (1.0, 0.5, 0.1, 0.01):
                for xi in (0.0, 0.01, 0.05, 0.3):
                    count += 1
                    z = math.sqrt(t * ((va + 1) ** 2 - 1))
                    got = _core.dm_holevo(va, t, xi, z, float("inf"), 0.95)
                    want = _core.cv_rate(va, t, xi, 1.0, 0.0, 0.95, False, True)

                    for i in range(3):
                        self.assertClose(
                            got[i],
                            want[i],
                            atol=5e-12,
                            msg=f"component {i} at {va},{t},{xi}",
                        )
        self.assertEqual(count, 80, msg=f"cases swept {count}")

    def test_holevo_alphabet(self):
        """
        Four states cap $I(X;Y)$ at 2 bits where the unbounded form claims 3.459 at the same SNR, lowering the key
        without touching Eve's bound.
        """
        z = math.sqrt(440.0)
        wide = _core.dm_holevo(20.0, 1.0, 0.0, z, float("inf"), 0.95)
        four = _core.dm_holevo(20.0, 1.0, 0.0, z, 2.0, 0.95)
        self.assertClose(wide[0], 3.4594316186, atol=1e-9, msg="unbounded alphabet")
        self.assertEqual(four[0], 2.0, msg="I at bits = 2")
        self.assertLess(four[2], wide[2], msg="key at bits = 2")
        self.assertEqual(four[1], wide[1], msg="chi at bits = 2")

    def test_limit_closes(self):
        """
        Over pure loss the 8-PSK rate reaches 83.77%, 94.89%, 98.72% and 99.69% of the Gaussian rate at $V_A = 0.2$,
        0.05, 0.01 and 0.002.
        """
        want = {
            0.002: 0.996870,
            0.01: 0.987150,
            0.05: 0.948931,
            0.2: 0.837694,
        }
        for va in (0.2, 0.05, 0.01, 0.002):
            alpha = math.sqrt(va / 2)
            got = key(m=8, alpha=alpha, d=15.05, xi=0.0) / gauss(va, 15.05, 0.0)
            self.assertClose(got, want[va], atol=1e-5, msg=f"ratio to Gaussian at {va}")


class Shape(Question):
    """
    The penalty grows with distance and the optimal modulation shrinks, both opposite to Gaussian modulation.
    """

    def test_gap_widens(self):
        """
        At $\\alpha = 0.4$, $\\xi = 0.005$ the QPSK rate falls monotonically from 49.9% of the Gaussian rate at the
        transmitter to 13.9% at 50 km.
        """
        ratios = []
        for d in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            ratios.append(key(alpha=0.4, d=d, xi=0.005) / gauss(0.32, d, 0.005))
        self.assertMonotone(ratios, rising=False, msg=f"ratios {ratios}")
        self.assertClose(ratios[0], 0.498969, atol=1e-4, msg="ratio at the transmitter")
        self.assertClose(ratios[-1], 0.139466, atol=1e-4, msg="ratio at 50 km")

    def test_optimal_small(self):
        """
        The optimal QPSK $V_A$ falls from 0.538 to 0.166 SNU over 0 to 80 km, where Gaussian modulation wants $V_A$
        above 5 at 40 km.
        """
        vas = []
        for d in (0.0, 20.0, 40.0, 60.0, 80.0):
            vas.append(2 * best(key, d=d)[0] ** 2)
        self.assertMonotone(vas, rising=False, msg=f"optimal V_A {vas}")
        self.assertClose(vas[0], 0.5382, atol=1e-3, msg="optimal V_A at the transmitter")
        self.assertClose(vas[-1], 0.1657, atol=1e-3, msg="optimal V_A at 80 km")
        self.assertGreater(peak(40.0), 5.0, msg="Gaussian optimal V_A at 40 km")

    def test_alpha_penalty(self):
        """
        Past the optimum the QPSK key falls monotonically with $\\alpha$ and is negative by $\\alpha = 0.7$ at 20 km,
        $\\xi = 0.01$, where the Gaussian rate still climbs.
        """
        keys = [key(alpha=a, d=20.0) for a in (0.43, 0.5, 0.6, 0.7, 0.9)]
        self.assertMonotone(keys, rising=False, msg=f"keys {keys}")
        self.assertGreater(keys[0], 0.0, msg="key at alpha = 0.43")
        self.assertLess(keys[3], 0.0, msg="key at alpha = 0.7")

        rates = [gauss(2 * a**2, 20.0) for a in (0.43, 0.5, 0.6, 0.7, 0.9)]
        self.assertMonotone(rates, msg=f"Gaussian rates {rates}")


class Guards(Guarded):
    """
    Out-of-domain constellations, correlations and arguments raise rather than return a rate.
    """

    def test_size_domain(self):
        """
        m outside 3..64 raises `ValueError`, a two-state constellation not being isotropic, and 3 and 64 are accepted.
        """
        for bad in (0, 1, 2, 65, 1000):
            self.assertFails(ValueError, "m must be in", _core.dm_moments, bad, 0.4, msg=f"m = {bad}")
        self.assertFinite(_core.dm_moments(3, 0.4), msg="m = 3")
        self.assertFinite(_core.dm_moments(64, 0.4), msg="m = 64")

    def test_alpha_domain(self):
        """
        $\\alpha$ of 0, -0.4, 1e-9, 101, NaN or infinity raises `ValueError`.
        """
        for bad in (0.0, -0.4, 1e-9, 101.0, float("nan"), float("inf")):
            self.assertBad(
                "alpha must be in",
                _core.dm_states,
                (4, bad),
                msg=f"alpha = {bad}",
            )

    def test_rate_domain(self):
        """
        Transmittance outside (0, 1], negative excess noise, efficiency outside (0, 1], negative
        electronic noise and reconciliation efficiency outside [0, 1] each raise a `ValueError`
        naming the parameter.
        """

        self.assertSlots(
            _core.dm_rate,
            (4, 0.4, 0.5, 0.01, 1.0, 0.0, 0.95),
            [
                (2, "t must be", [0.0, -0.1, 1.5, float("nan")]),
                (3, "xi must be", [-1e-9]),
                (4, "eta must be", [0.0, 1.5, -0.2]),
                (5, "vel must be", [-1e-9]),
                (6, "beta must be", [1.5]),
            ],
            msg="dm_rate",
        )

    def test_holevo_domain(self):
        """
        A correlation stronger than the marginals carry raises rather than reach a sub-vacuum symplectic eigenvalue that
        understates Eve, and so does an alphabet entropy that is zero, negative or NaN.
        """

        self.assertSlots(
            _core.dm_holevo,
            (2.0, 1.0, 0.0, 1.0, 2.0, 0.95),
            [
                (3, "unphysical", [3.0, 10.0, 1e6]),
                (3, "z must be", [-0.1]),
                (4, "bits must be", [0.0, -1.0, float("nan")]),
            ],
            msg="dm_holevo",
        )
        self.assertFinite(
            _core.dm_holevo(2.0, 1.0, 0.0, math.sqrt(8.0), float("inf"), 0.95),
            msg="pure-state corner",
        )

    def test_zstar_clamped(self):
        """
        The correlation bound falls with excess noise and clamps at zero rather than going negative, and at $\\xi$ up to
        1e12 or $T$ down to 1e-12 every component stays finite with a negative rate.
        """
        zs = [rate(xi=x)[3] for x in (0.1, 1.0, 5.0, 10.0)]
        self.assertMonotone(zs, rising=False, msg=f"Z* {zs}")
        self.assertEqual(rate(xi=1e6)[3], 0.0, msg="Z* at xi = 1e6")

        for xi in (1.0, 10.0, 1e3, 1e6, 1e12):
            out = rate(xi=xi)
            self.assertFinite(out, msg=f"every component finite at xi = {xi}")
            self.assertLess(out[2], 0.0, msg=f"key at xi = {xi}")

        for t in (1e-12, 1e-9, 1e-6, 1e-3):
            out = _core.dm_rate(4, 0.4, t, 0.01, 1.0, 0.0, 0.95)
            self.assertFinite(out, msg=f"rate at T = {t}")
            self.assertLess(out[2], 0.0, msg=f"key at T = {t}")
        self.assertLess(
            abs(_core.dm_rate(4, 0.4, 1e-12, 0.01, 1.0, 0.0, 0.95)[3]),
            1e-6,
            msg="Z* at T = 1e-12",
        )


# dm_secure: Lin, Upadhyaya & Lutkenhaus, PRX 9, 041064 (2019); Winick, Lutkenhaus & Coles, Quantum 2, 77 (2018).
# `upper` is step 1 (Frank-Wolfe) and proves nothing; `bound` is step 2, the number the exams pin.
# Cutoff assumed, LUL19 Sec. III B; not built: Upadhyaya, van Himbeeck, Lin & Lutkenhaus, PRX Quantum 2, 020325 (2021).


def field(pair, n):
    """
    A flat (real, imaginary) pair from the core as one complex numpy matrix.
    """

    return np.asarray(pair[0]).reshape(n, n) + 1j * np.asarray(pair[1]).reshape(n, n)


def psd_root(a):
    """
    The positive square root of a Hermitian positive-semidefinite matrix.
    """
    w, v = np.linalg.eigh(a)

    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.conj().T


def kraus(m, nc, cut=0.0, phase=0.0):
    """
    K = sum_z |z> (x) I_A (x) sqrt(R_z), built in numpy from the core's region operators.
    """
    nb = nc + 1
    rows = []
    for z in range(m):
        rows.append(np.kron(np.eye(m), psd_root(field(_core.dm_region(m, nc, z, cut, phase), nb))))

    return np.vstack(rows)


def relent(m, nc, rho, eps, cut=0.0, phase=0.0):
    """
    D(G_eps(rho) || Z[G_eps(rho)]) in bits, formed explicitly on the m^2 (nc+1) dimensional Kraus image.
    """
    k = kraus(m, nc, cut, phase)
    dp = k.shape[0]
    d = rho.shape[0]
    img = k @ rho @ k.conj().T
    sig = (1 - eps) * img + eps * np.trace(img).real * np.eye(dp) / dp
    flat = np.zeros_like(sig)
    for j in range(m):
        flat[j * d : (j + 1) * d, j * d : (j + 1) * d] = sig[j * d : (j + 1) * d, j * d : (j + 1) * d]

    def nlogn(a):
        w = np.clip(np.linalg.eigvalsh(a), 1e-300, None)

        return float(np.sum(w * np.log2(w)))

    return nlogn(sig) - nlogn(flat)


@memo
def secure(km=20.0, xi=0.01, alpha=0.7, nc=6, steps=6, cut=0.0, eps=1e-10):
    """
    One memoised dm_secure run at a fibre distance in km.
    """

    return _core.dm_secure(4, alpha, fiber(km), xi, nc, cut, 0.0, 0.95, eps, steps, 1e-8)


@memo
def relaxed(km=20.0, xi=0.01, alpha=0.7, nc=6, w=1e-4, scale=1.0, tests=1e10, cut=0.0, norms=None, steps=10):
    """
    One memoised dm_relaxed run over Kanitschar Eq. (21)'s set; `scale` multiplies both acceptance half-widths, `norms`
    overrides the clipped pair.
    """
    eta = fiber(km)
    pair = norms if norms else _core.dm_clip(5.0, math.sqrt(eta) * alpha)
    wide = scale * _core.dm_accept(pair[0], tests, 0.7e-10, True)
    tight = scale * _core.dm_accept(pair[1], tests, 0.7e-10, True)

    return _core.dm_relaxed(4, alpha, eta, xi, nc, w, wide, tight, pair[0], pair[1], cut, 0.0, 0.95, 1e-10, steps, 1e-8)


@memo
def trusted(eta=0.5, xi=0.01, alpha=0.7, nc=6, eta_d=0.6, v_el=0.05, steps=6):
    """
    One memoised dm_trusted run; `eta` is the channel alone, and eta_d, v_el must not be folded into it.
    """

    return _core.dm_trusted(4, alpha, eta, xi, nc, eta_d, v_el, 0.0, 0.0, 0.95, 1e-10, steps, 1e-8)


class Numerics(Question):
    """
    The operators the numerical proof is assembled from.
    """

    def test_regions_complete(self):
        """
        Without postselection the four region operators sum to the identity to 1e-14, each positive semidefinite with
        1/4 on its diagonal, and a radial cut and then a phase guard each remove trace while staying positive.
        """
        nb = 9
        total = sum(field(_core.dm_region(4, 8, z), nb) for z in range(4))
        self.assertClose(
            float(np.abs(total - np.eye(nb)).max()),
            0.0,
            atol=1e-14,
            msg="sum of R_z against the identity",
        )

        worst = 0.0
        for z in range(4):
            r = field(_core.dm_region(4, 8, z), nb)
            worst = min(worst, float(np.linalg.eigvalsh(r).min()))
            self.assertClose(
                float(np.abs(np.diag(r).real - 0.25).max()),
                0.0,
                atol=1e-14,
                msg=f"diagonal of R_{z}",
            )
        self.assertGreaterEqual(worst, -1e-14, msg=f"min eigenvalue {worst}")

        plain = field(_core.dm_region(4, 8, 0), nb)
        cut = field(_core.dm_region(4, 8, 0, 0.6), nb)
        guard = field(_core.dm_region(4, 8, 0, 0.6, 0.2), nb)
        self.assertGreaterEqual(
            float(np.linalg.eigvalsh(cut).min()),
            -1e-14,
            msg="min eigenvalue with cut",
        )
        self.assertLess(
            np.trace(cut).real,
            np.trace(plain).real,
            msg="trace with cut",
        )
        self.assertLess(np.trace(guard).real, np.trace(cut).real, msg="trace with guard")

    def test_state_physical(self):
        """
        The honest bipartite state has unit trace, is positive semidefinite and has Alice's coherent-state Gram
        marginal, and Bob's conditional mean photon number is $\\eta\\alpha^2 + \\eta\\xi/2$, the displaced thermal
        state of LUL19 Sec. IV A.
        """
        nc = 8
        nb = nc + 1
        alpha = 0.7
        rho = field(_core.dm_honest(4, alpha, 0.5, 0.02, nc), 4 * nb)
        marg = np.zeros((4, 4), dtype=complex)
        want = np.zeros((4, 4), dtype=complex)
        for x in range(4):
            for y in range(4):
                marg[x, y] = np.trace(rho[x * nb : (x + 1) * nb, y * nb : (y + 1) * nb])
                ax = alpha * np.exp(2j * np.pi * x / 4)
                ay = alpha * np.exp(2j * np.pi * y / 4)
                want[x, y] = np.exp(-abs(ax) ** 2 / 2 - abs(ay) ** 2 / 2 + np.conj(ay) * ax) / 4
        self.assertClose(float(np.trace(rho).real), 1.0, atol=1e-9, msg="unit trace")
        self.assertGreaterEqual(float(np.linalg.eigvalsh(rho).min()), -1e-12, msg="positive semidefinite")
        self.assertClose(
            float(np.abs(marg - want).max()),
            0.0,
            atol=1e-9,
            msg="Alice's fixed marginal",
        )

        nc = 10
        nb = nc + 1
        num = np.diag(np.arange(nb).astype(float))
        for eta, xi in ((1.0, 0.0), (0.4, 0.01), (0.1, 0.05)):
            rho = field(_core.dm_honest(4, 0.7, eta, xi, nc), 4 * nb)
            got = 4 * np.trace(num @ rho[0:nb, 0:nb]).real
            self.assertClose(
                got,
                eta * 0.49 + eta * xi / 2,
                atol=1e-8,
                msg=f"<n> at eta={eta}, xi={xi}",
            )


class Objective(Question):
    """
    The reduced relative-entropy objective and its gradient against the explicit m^2 (nc+1) Kraus image.
    """

    def test_matches_numpy(self):
        """
        The reduced objective agrees with an explicit construction of the Kraus image to 1e-12.
        """
        nc = 6
        nb = nc + 1
        rho = field(_core.dm_honest(4, 0.7, 0.5, 0.02, nc), 4 * nb)
        rho = rho / np.trace(rho).real
        parts = (rho.real.ravel().tolist(), rho.imag.ravel().tolist())
        for eps in (1e-4, 1e-8, 1e-10):
            got = _core.dm_relent(4, nc, parts[0], parts[1], 0.0, 0.0, eps)[0]
            self.assertClose(got, relent(4, nc, rho, eps), atol=1e-12, msg=f"f_eps at eps={eps}")

    def test_gradient_matches(self):
        """
        The analytic gradient agrees with a central finite difference of the explicit objective to 1e-3 relative along
        four random Hermitian directions.
        """
        nc = 6
        nb = nc + 1
        d = 4 * nb
        base = field(_core.dm_honest(4, 0.7, 0.5, 0.02, nc), d)
        base = base / np.trace(base).real
        rho = 0.8 * base + 0.2 * np.eye(d) / d
        eps = 1e-8
        _f, _p, gr, gi = _core.dm_relent(4, nc, rho.real.ravel().tolist(), rho.imag.ravel().tolist(), 0.0, 0.0, eps)
        grad = np.asarray(gr).reshape(d, d) + 1j * np.asarray(gi).reshape(d, d)
        rng = np.random.default_rng(11)
        worst = 0.0
        for _ in range(4):
            step = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            step = (step + step.conj().T) / 2
            step *= 1e-6 / np.abs(step).max()
            diff = (relent(4, nc, rho + step, eps) - relent(4, nc, rho - step, eps)) / 2
            worst = max(worst, abs(diff - float(np.real(np.trace(step @ grad)))) / abs(diff))
        self.assertLess(worst, 1e-3, msg=f"relative gradient error {worst:.2e}")

    def test_objective_bracketed(self):
        """
        The perturbed relative entropy stays between zero and $\\log_2 4$ bits times the pass probability, at three
        excess noises with and without postselection.
        """
        nc = 6
        nb = nc + 1
        for xi in (0.002, 0.01, 0.05):
            for cut in (0.0, 0.5):
                rho = field(_core.dm_honest(4, 0.7, 0.4, xi, nc), 4 * nb)
                rho = rho / np.trace(rho).real
                val, pas, _gr, _gi = _core.dm_relent(
                    4,
                    nc,
                    rho.real.ravel().tolist(),
                    rho.imag.ravel().tolist(),
                    cut,
                    0.0,
                    1e-10,
                )
                self.assertGreater(val, 0.0, msg=f"f_eps > 0 at xi={xi}, cut={cut}")
                self.assertLess(
                    val,
                    2.0 * pas + 1e-9,
                    msg=f"f_eps {val} against 2 p_pass at xi={xi}",
                )


class Certified(Question):
    """
    The certified bound: its ceilings, early stopping, and reach past the analytic bound.
    """

    def test_below_step_one(self):
        """
        The certified bound stays under the Frank-Wolfe value it linearises about from 0 to 80 km and under the
        repeaterless capacity $-\\log_2(1 - \\eta)$ to 120 km, and one Frank-Wolfe step certifies less than fifteen with
        both still under step 1.
        """
        for km in (0.0, 30.0, 80.0):
            out = secure(km=km)
            self.assertLessEqual(
                out[0],
                out[2] + 1e-9,
                msg=f"bound against upper at {km} km",
            )

        # Not kit.forms.plob, here and in test_trusted_below_step_one: exact above eta = 0.02, and docstrings ship it.
        for km in (20.0, 60.0, 120.0):
            got = secure(km=km)[0]
            cap = -math.log2(1.0 - fiber(km))
            self.assertLess(got, cap, msg=f"PLOB capacity at {km} km")

        few = secure(steps=1)
        many = secure(steps=15)
        self.assertLess(few[1], many[1], msg="dual value, 1 vs 15 steps")
        self.assertLessEqual(few[0], few[2] + 1e-9, msg="bound against upper at 1 step")
        self.assertLessEqual(many[0], many[2] + 1e-9, msg="bound against upper at 15 steps")

    def test_beats_analytic(self):
        """
        At 60 km and $\\xi = 0.01$ `dm_rate` is negative and `dm_secure` certifies key.
        """
        loose = _core.dm_rate(4, 0.4, fiber(60.0), 0.01, 1.0, 0.0, 0.95)[2]
        tight = secure(km=60.0)[0]
        self.assertLess(loose, 0.0, msg=f"dm_rate key {loose} at 60 km")
        self.assertGreater(tight, 0.0, msg=f"dm_secure key {tight} at 60 km")

    def test_falls_with_loss(self):
        """
        The certified rate falls monotonically with fibre length and with excess noise.
        """
        span = [secure(km=km)[0] for km in (0.0, 20.0, 40.0, 60.0, 80.0)]
        noise = [secure(km=40.0, xi=xi)[0] for xi in (0.002, 0.01, 0.03)]
        self.assertMonotone(span, rising=False, msg="rate against distance")
        self.assertMonotone(noise, rising=False, msg="rate against excess noise")

    def test_cutoff_settles(self):
        """
        From nc = 4 to 8 the step-1 objective moves under 2%, the certificate stays within 20% of it and the constraint
        violation under 1e-4, which sizes the cutoff assumption rather than removing it.
        """
        upper = []
        lower = []
        for nc in (4, 6, 8):
            out = secure(nc=nc, steps=8)
            upper.append(out[2])
            lower.append(out[0])
            self.assertLess(out[5], 1e-4, msg=f"constraint violation at nc={nc}")

        # Only the step-1 value measures the cutoff; the bound adds solver looseness.
        self.assertLess(
            max(upper) - min(upper),
            0.02 * max(upper),
            msg=f"upper {upper}",
        )
        self.assertLess(
            max(upper) - min(lower),
            0.2 * max(upper),
            msg=f"upper {upper}, bound {lower}",
        )

    def test_postselection_pays(self):
        """
        At 80 km and $\\xi = 0.02$ a radial postselection cut lowers the pass probability and raises the certified rate,
        as LUL19 Sec. IV E reports.
        """
        plain = secure(km=80.0, xi=0.02, nc=8, steps=10)
        picked = secure(km=80.0, xi=0.02, nc=8, steps=10, cut=0.5)
        self.assertLess(picked[3], plain[3], msg="p_pass with cut")
        self.assertGreater(picked[0], plain[0], msg="bound with cut")

    def test_cost_is_charged(self):
        """
        `dm_cost`'s `p_pass` and `delta_ec` equal the certified run's bit-for-bit, `delta_ec` is LUL19 Sec. IV D's $(1 -
        \\beta)H(Z) + \\beta H(Z\\vert X)$ at every $\\beta$, and with no cut `p_pass` plus the dropped weight is one,
        an identity internal to the engine and not a published number.
        """
        run = secure()
        eta = fiber(20.0)
        p_pass, delta, h_z, h_zx = _core.dm_cost(4, 0.7, eta, 0.01, 6)
        self.assertEqual(p_pass, run[3], msg="p_pass")
        self.assertEqual(delta, run[4], msg="delta_ec")
        self.assertGreater(h_zx, 0.0, msg="H(Z|X)")
        self.assertLessEqual(h_zx, h_z, msg="H(Z|X) against H(Z)")
        self.assertLessEqual(h_z, 2.0 + 1e-15, msg="H(Z)")
        for beta in (0.0, 0.5, 0.95, 1.0):
            got = _core.dm_cost(4, 0.7, eta, 0.01, 6, 0.0, 0.0, beta)
            mixed = (1.0 - beta) * got[2] + beta * got[3]
            self.assertClose(got[1], mixed, atol=1e-15, msg=f"Sec. IV D at beta = {beta}")
            self.assertEqual(got[2:], (h_z, h_zx), msg=f"entropies at beta = {beta}")

        weight = _core.dm_weight(4, 0.7, eta, 0.01, 6)
        self.assertClose(p_pass + weight, 1.0, atol=1e-12, msg="p_pass + weight")
        picked = _core.dm_cost(4, 0.7, eta, 0.01, 6, 0.5)
        self.assertLess(picked[0], p_pass, msg="p_pass with cut")
        self.assertLess(picked[3], h_zx, msg="H(Z|X) with cut")


class Trusted(Question):
    """
    Lin & Lutkenhaus's trusted receiver: the POVM the noise moves into, the four observables it
    changes, and the distance keeping it out of the channel buys.
    """

    def test_noisy_complete(self):
        """
        At three receiver settings the trusted region operators sum to the identity to 1e-13, each positive semidefinite
        with 1/m on its diagonal.
        """
        nb = 9
        for eta_d, v_el in ((0.6, 0.05), (0.3, 0.2), (0.9, 0.001)):
            total = sum(field(_core.dm_noisy(4, 8, z, eta_d, v_el), nb) for z in range(4))
            self.assertClose(
                float(np.abs(total - np.eye(nb)).max()),
                0.0,
                atol=1e-13,
                msg=f"sum of R_z at eta_d={eta_d}, v_el={v_el}",
            )

            for z in range(4):
                r = field(_core.dm_noisy(4, 8, z, eta_d, v_el), nb)
                self.assertGreaterEqual(
                    float(np.linalg.eigvalsh(r).min()),
                    -1e-13,
                    msg=f"R_{z} positive at eta_d={eta_d}",
                )
                self.assertClose(
                    float(np.abs(np.diag(r).real - 0.25).max()),
                    0.0,
                    atol=1e-13,
                    msg=f"diagonal of R_{z} at eta_d={eta_d}",
                )

    def test_noisy_ideal_limit(self):
        """
        As $\\eta_d \\to 1$ with $v_{el} = 0$ the trusted region operator converges on `dm_region` linearly in $1 -
        \\eta_d$, a decade per decade.
        """
        nb = 7
        ideal = field(_core.dm_region(4, 6, 0), nb)
        gaps = []
        for eta_d in (0.99, 0.999, 0.9999):
            noisy = field(_core.dm_noisy(4, 6, 0, eta_d, 0.0), nb)
            gaps.append(float(np.abs(noisy - ideal).max()))

        self.assertMonotone(gaps, rising=False, msg=f"gaps {gaps}")

        for lo, hi in zip(gaps[1:], gaps[:-1]):
            self.assertClose(hi / lo, 10.0, atol=0.2, msg=f"linear convergence, ratio {hi / lo:.4f}")

    def test_moments_ideal_limit(self):
        """
        LL20 Sec. IV B's F_Q, F_P, S_Q and S_P are exactly Hermitian and, at an ideal receiver, within 1e-3 of $q$, $p$,
        $q^2 + 1/2$ and $p^2 + 1/2$ away from the cutoff edge.
        """
        nb = 7
        a = np.diag(np.sqrt(np.arange(1, nb)), 1)
        quad = (a + a.conj().T) / math.sqrt(2.0)
        mom = (a - a.conj().T) / (1j * math.sqrt(2.0))
        want = (quad, mom, quad @ quad + 0.5 * np.eye(nb), mom @ mom + 0.5 * np.eye(nb))
        names = ("F_Q", "F_P", "S_Q", "S_P")

        for which in range(4):
            for eta_d, v_el in ((0.6, 0.05), (0.9999, 0.0)):
                got = field(_core.dm_observe(6, which, eta_d, v_el), nb)
                self.assertClose(
                    float(np.abs(got - got.conj().T).max()),
                    0.0,
                    atol=0.0,
                    msg=f"{names[which]} is Hermitian at eta_d={eta_d}",
                )

            got = field(_core.dm_observe(6, which, 0.9999, 0.0), nb)
            edge = nb - 3
            gap = float(np.abs(got[:edge, :edge] - want[which][:edge, :edge]).max())
            self.assertLess(gap, 1e-3, msg=f"{names[which]} against its ideal form, gap {gap:.3e}")

    def test_trusted_buys_distance(self):
        """
        At $\\eta_d = 0.6$, $v_{el} = 0.05$ and $\\eta$ from 0.9 to 0.05 the trusted proof certifies key while `dm_rate`
        with $v_{el}$ referred to the channel input as $\\xi + 2v_{el}/(\\eta_d\\eta)$ certifies none.
        """
        for eta in (0.9, 0.5, 0.2, 0.05):
            got = trusted(eta=eta)[0]
            self.assertGreater(got, 0.0, msg=f"trusted key at eta={eta}")

            referred = 0.01 + 2.0 * 0.05 / (0.6 * eta)
            loose = _core.dm_rate(4, 0.7, 0.6 * eta, referred, 1.0, 0.0, 0.95)[2]
            self.assertLess(loose, 0.0, msg=f"referred dm_rate key at eta={eta}")

    def test_trusted_below_step_one(self):
        """
        The certified trusted key stays under its Frank-Wolfe value and under the repeaterless capacity $-\\log_2(1 -
        \\eta_d\\eta)$ at the transmittance Bob sees.
        """
        for eta in (0.9, 0.5, 0.2):
            out = trusted(eta=eta)
            self.assertLessEqual(out[0], out[2] + 1e-9, msg=f"bound against upper at eta={eta}")
            cap = -math.log2(1.0 - 0.6 * eta)
            self.assertLess(out[0], cap, msg=f"PLOB capacity at eta={eta}")

    def test_trusted_falls_with_loss(self):
        """
        The certified trusted key falls monotonically with $\\eta$ from 0.9 to 0.05, the dual point reported converged
        at each.
        """
        keys = []
        for eta in (0.9, 0.5, 0.2, 0.05):
            out = trusted(eta=eta)
            keys.append(out[0])
            self.assertEqual(out[8], "converged", msg=f"dual status at eta={eta}")

        self.assertMonotone(keys, rising=False, msg=f"keys {keys}")


class Numerical(Guarded):
    """
    What the numerical proof refuses.
    """

    def test_cutoff_domain(self):
        """
        `dm_secure` and `dm_cost` refuse a photon-number cutoff outside [2, 30] and a constellation outside [2, 8]
        states.
        """
        cases = [
            (4, "nc must be in", [0, 1, 31, 400]),
            (0, "accepts m in", [0, 1, 9, 16]),
        ]
        self.assertSlots(_core.dm_secure, (4, 0.7, 0.5, 0.01, 6), cases, msg="dm_secure sizes")
        self.assertSlots(_core.dm_cost, (4, 0.7, 0.5, 0.01, 6), cases, msg="dm_cost sizes")

    def test_perturbation_domain(self):
        """
        `dm_secure` refuses a perturbation outside Winick-Lutkenhaus-Coles Theorem 2's range and zero Frank-Wolfe steps.
        """

        self.assertSlots(
            _core.dm_secure,
            (4, 0.7, 0.5, 0.01, 6, 0.0, 0.0, 0.95, 1e-10, 6, 1e-8),
            [
                (8, "Theorem 2 does not apply", [0.5]),
                (9, "steps must be at least 1", [0]),
            ],
            msg="dm_secure controls",
        )

    def test_relaxed_weight(self):
        """
        `dm_relaxed` refuses a weight outside (0, 1); zero is `dm_secure`'s equality problem.
        """
        args = (4, 0.7, 0.5, 0.01, 6)
        for bad in (0.0, 1.0, 2.0, -1e-6):
            self.assertFails(
                ValueError,
                "w must be in (0, 1)",
                lambda v=bad: _core.dm_relaxed(*args, v, 1e-3, 1e-2, 49.0, 2351.0),
                msg=f"w = {bad}",
            )

    def test_relaxed_slots(self):
        """
        `dm_relaxed` refuses the sizes and controls `dm_secure` does, plus a non-positive clipped operator norm.
        """

        self.assertSlots(
            _core.dm_relaxed,
            (4, 0.7, 0.5, 0.01, 6, 1e-4, 1e-3, 1e-2, 49.0, 2351.0, 0.0, 0.0, 0.95, 1e-10, 6, 1e-8),
            [
                (4, "nc must be in", [0, 1, 31]),
                (0, "accepts m in", [0, 1, 9]),
                (8, "x_n must be > 0", [0.0, -1.0]),
                (9, "x_n2 must be > 0", [0.0, -1.0]),
                (13, "Theorem 2 does not apply", [0.5]),
                (14, "steps must be at least 1", [0]),
            ],
            msg="dm_relaxed sizes and controls",
        )

    def test_pure_loss(self):
        """
        `dm_secure` refuses pure loss ($\\xi = 0$), where the linearisation collapses, rather than return a valid
        worthless certificate.
        """

        self.assertFails(
            ValueError,
            "linearisation collapsed",
            _core.dm_secure,
            4,
            0.7,
            fiber(40.0),
            0.0,
            8,
            0.0,
            0.0,
            0.95,
            1e-10,
            8,
            1e-8,
            msg="xi = 0",
        )

    def test_detector_domain(self):
        """
        All three trusted entry points refuse $\\eta_d$ outside (0, 1) and negative $v_{el}$, an ideal receiver being
        `dm_secure` at $v_{el} = 0$.
        """

        for fn, ok, slot in (
            (_core.dm_noisy, (4, 6, 0, 0.6, 0.05), 3),
            (_core.dm_observe, (6, 0, 0.6, 0.05), 2),
            (_core.dm_trusted, (4, 0.7, 0.5, 0.01, 6, 0.6, 0.05), 5),
        ):
            self.assertSlots(
                fn,
                ok,
                [
                    (slot, "eta_d must be in (0, 1)", [0.0, 1.0, -0.5, 2.0]),
                    (slot + 1, "v_el must be >= 0", [-1e-9, -1.0]),
                ],
                msg=f"{fn.__name__} detector",
            )

    def test_trusted_selectors(self):
        """
        `dm_noisy` refuses a key symbol outside the constellation or a phase guard that closes the sector, and
        `dm_observe` an observable outside LL20's four.
        """

        self.assertSlots(
            _core.dm_noisy,
            (4, 6, 0, 0.6, 0.05),
            [(2, "z must name one of the 4 key symbols", [4, 9])],
            msg="dm_noisy key symbol",
        )
        self.assertSlots(
            _core.dm_observe,
            (6, 0, 0.6, 0.05),
            [(1, "which must name one of F_Q, F_P, S_Q, S_P", [4, 17])],
            msg="dm_observe selector",
        )
        self.assertFails(
            ValueError,
            "phase must be below",
            _core.dm_noisy,
            4,
            6,
            0,
            0.6,
            0.05,
            0.0,
            1.0,
            msg="dm_noisy phase = 1.0",
        )

    def test_phase_guard(self):
        """
        `dm_region` refuses a phase guard wide enough to close a sector.
        """

        self.assertFails(
            ValueError,
            "phase must be below",
            _core.dm_region,
            4,
            6,
            0,
            0.0,
            1.0,
            msg="dm_region phase = 1.0",
        )


class Relaxed(Question):
    """
    Kanitschar Eq. (21)'s relaxed feasible set against dm_secure's equality problem, and the key length it feeds.
    """

    def test_widening_costs(self):
        """
        Raising the weight outside the cutoff or widening the acceptance interval lowers both the step-1 objective and
        the certified entropy monotonically.
        """
        weights = [relaxed(w=w) for w in (1e-5, 1e-4, 1e-3, 1e-2)]
        widths = [relaxed(scale=s) for s in (0.3, 1.0, 3.0, 10.0)]
        self.assertMonotone([o[2] for o in weights], rising=False, msg="objective against the weight")
        self.assertMonotone([o[2] for o in widths], rising=False, msg="objective against the acceptance width")
        self.assertMonotone([o[0] for o in weights], rising=False, msg="entropy against the weight")
        self.assertMonotone([o[0] for o in widths], rising=False, msg="entropy against the acceptance width")

    def test_under_equality(self):
        """
        At the acceptance a ten-billion-round block buys, the relaxed minimum sits 0.01 to 0.5 bit below
        `dm_secure`'s equality minimum, the overclaim reading `dm_secure` into Theorem 6 Eq. (9) would make, measured
        rather than contained since the two sets' observables differ.
        """
        loose = relaxed()
        tight = secure()
        equal = tight[1] / tight[3]
        self.assertLess(loose[0], equal, msg=f"relaxed {loose[0]} against equality {equal}")
        self.assertGreater(equal - loose[0], 0.01, msg=f"gap {equal - loose[0]}")
        self.assertLess(equal - loose[0], 0.5, msg=f"gap {equal - loose[0]}")

    def test_certificate_holds(self):
        """
        Step 2 stays under step 1 and under $\\log_2\\lvert Z\\rvert = 2$ bits at one and ten steps, and one step
        certifies less than ten.
        """
        many = relaxed()
        few = relaxed(steps=1)
        for run, name in ((many, "ten steps"), (few, "one step")):
            self.assertLessEqual(run[1], run[2] + 1e-9, msg=f"step 2 against step 1 at {name}")
            self.assertLessEqual(run[0], 2.0 + 1e-9, msg=f"entropy {run[0]} at {name}")
            self.assertGreater(run[0], 0.0, msg=f"entropy {run[0]} at {name}")
        self.assertLess(few[1], many[1], msg="step 2, 1 vs 10 steps")

    def test_length_assembles(self):
        """
        Theorem 6 assembles from the shipped pieces -- weight, clipped norms, acceptance half-widths, relaxed minimum
        and composed $\\epsilon$ -- into a positive key length `dm_length` accepts as `source="relaxed"`.
        """
        total = _core.dm_secpar(0.2e-10, 0.2e-10, 0.7e-10, 0.1e-10, 0.7e-10)
        run = relaxed(alpha=0.55, cut=0.5, w=1e-5, tests=1e11)
        signals = 1e12
        keyed = 0.9 * signals * run[3]
        leak = keyed * run[4] + math.log2(2.0 / 0.2e-10)
        length, raw, aep, price = _core.dm_length(signals, keyed, run[0], 4, 1e-5, 0.7e-10, 0.2e-10, leak, "relaxed")
        self.assertClose(total / 1e-10, 1.0, atol=1e-14, msg=f"composed eps {total}")
        self.assertGreater(length, 0.0, msg=f"length {length}")
        self.assertClose(length / raw, 1.0, atol=1e-9, msg=f"length {length}, raw {raw}")
        self.assertGreater(price, aep, msg=f"price {price}, aep {aep}")

    def test_weight_is_deficit(self):
        """
        `dm_weight` equals the trace the truncation drops to 1e-15 and falls by over two orders per two photons of
        cutoff.
        """
        seen = []
        for nc in (6, 8, 10):
            rho = field(_core.dm_honest(4, 0.7, 0.5, 0.01, nc), 4 * (nc + 1))
            left = 1.0 - float(np.trace(rho).real)
            got = _core.dm_weight(4, 0.7, 0.5, 0.01, nc)
            self.assertClose(got, left, atol=1e-15, msg=f"weight at nc={nc}")
            seen.append(got)
        self.assertMonotone(seen, rising=False, msg=f"weights {seen}")
        self.assertLess(seen[2] * 100.0, seen[1], msg=f"weights {seen}")

    def test_clip_exceeds_printed(self):
        """
        The clipped operator norms are 2 and 3.84 times the constants Kanitschar Sec. V F prints, their own Appendix B
        stating the larger outright, and the printed pair certifies more than the engine's.
        """
        wide, square = _core.dm_clip(5.0)
        self.assertClose(wide, 49.0, atol=1e-12, msg="2 M^2 - 1 at their own M = 5")
        self.assertClose(square, 2351.0, atol=1e-12, msg="4 M^4 - 6 M^2 + 1 at M = 5")
        self.assertClose(wide / (25.0 - 0.5), 2.0, atol=1e-12, msg="x_n over printed")
        self.assertClose(square / (625.0 - 12.5), 3.8384, atol=1e-3, msg="x_n2 over printed")
        printed = relaxed(norms=(24.5, 612.5))
        honest = relaxed()
        self.assertGreater(printed[0], honest[0], msg="entropy, printed against clipped norms")

    def test_printed_rows_empty(self):
        """
        Kanitschar Eq. (21) as printed, at the paper's own block, testing share, soft limit, cutoff and weight, demands
        twice the acceptance half-width fit inside the dimension-reduction shift and misses by 319 on both observables,
        so the printed set is empty and Appendix E's slack form is what the engine assembles.
        """
        tests = 0.1 * 1e12
        weight, ratio = _core.dm_wchoice(20, 25.0, tests, 1e-8 * tests, math.log2(0.1e-10))
        self.assertClose(ratio, 5.391, atol=1e-3, msg="r at nc = 20")
        self.assertClose(weight / 6.878e-8, 1.0, atol=1e-3, msg=f"weight {weight}")
        for norm in _core.dm_clip(5.0):
            half = _core.dm_accept(norm, tests, 0.7e-10, True)
            self.assertClose(
                2.0 * half / (weight * norm),
                319.0,
                atol=1.0,
                msg=f"2 mu = {2 * half} against w ||Gamma|| = {weight * norm}",
            )

    def test_operators_arbitrate(self):
        """
        The displaced number operator and its square agree with numpy's $(a^\\dagger - \\beta^*)(a - \\beta)$, and the
        honest state reads them back as $\\eta\\xi/2$ and $2\\bar n^2 + \\bar n$, the displaced thermal moments.
        """
        nc = 10
        nb = nc + 1
        eta, xi, alpha = 0.5, 0.02, 0.7
        big = nb + 2
        ladder = np.diag(np.sqrt(np.arange(1, big)), 1)
        for pair in ((0.3, -0.2), (0.0, 0.0), (-0.9, 0.4)):
            one = field(_core.dm_number(nc, *pair)[:2], nb)
            two = field(_core.dm_number(nc, *pair)[2:], nb)
            beta = pair[0] + 1j * pair[1]
            want = (ladder.conj().T - np.conj(beta) * np.eye(big)) @ (ladder - beta * np.eye(big))
            self.assertClose(float(np.abs(one - want[:nb, :nb]).max()), 0.0, atol=1e-13, msg=f"n_beta at {pair}")
            self.assertClose(
                float(np.abs(two - (want @ want)[:nb, :nb]).max()),
                0.0,
                atol=1e-12,
                msg=f"its square at {pair}",
            )

        rho = field(_core.dm_honest(4, alpha, eta, xi, nc), 4 * nb)
        nbar = 0.5 * eta * xi
        for j in range(4):
            beta = math.sqrt(eta) * alpha * np.exp(2j * math.pi * j / 4)
            one = field(_core.dm_number(nc, beta.real, beta.imag)[:2], nb)
            two = field(_core.dm_number(nc, beta.real, beta.imag)[2:], nb)
            block = rho[j * nb : (j + 1) * nb, j * nb : (j + 1) * nb]
            self.assertClose(4 * np.trace(one @ block).real, nbar, atol=1e-9, msg=f"<n_beta> for symbol {j}")
            self.assertClose(
                4 * np.trace(two @ block).real,
                2 * nbar * nbar + nbar,
                atol=1e-9,
                msg=f"<n_beta^2> for symbol {j}",
            )

    def test_honest_is_feasible(self):
        """
        The honest truncated state lies inside the relaxed set: every moment within its interval, the trace within the
        subnormalisation window, and half the trace distance to Alice's marginal under $\\sqrt w$.
        """
        nc, alpha, eta, xi, weight = 10, 0.7, 0.5, 0.01, 1e-6
        nb = nc + 1
        rho = field(_core.dm_honest(4, alpha, eta, xi, nc), 4 * nb)
        wide, square = _core.dm_clip(5.0, math.sqrt(eta) * alpha)
        nbar = 0.5 * eta * xi
        for j in range(4):
            beta = math.sqrt(eta) * alpha * np.exp(2j * math.pi * j / 4)
            block = rho[j * nb : (j + 1) * nb, j * nb : (j + 1) * nb]
            for which, norm, seen in ((0, wide, nbar), (2, square, 2 * nbar * nbar + nbar)):
                op = field(_core.dm_number(nc, beta.real, beta.imag)[which : which + 2], nb)
                got = np.trace(op @ block).real
                self.assertLessEqual(got, seen / 4 + 1e-12, msg=f"symbol {j} under its upper row")
                self.assertGreaterEqual(
                    got,
                    seen / 4 - weight * norm,
                    msg=f"symbol {j} above the shifted lower row",
                )

        trace = float(np.trace(rho).real)
        self.assertLessEqual(trace, 1.0, msg=f"trace {trace}")
        self.assertGreaterEqual(trace, 1.0 - weight, msg=f"trace {trace}")
        marg = np.zeros((4, 4), dtype=complex)
        for x in range(4):
            for y in range(4):
                marg[x, y] = np.trace(rho[x * nb : (x + 1) * nb, y * nb : (y + 1) * nb])
                ax = alpha * np.exp(2j * np.pi * x / 4)
                ay = alpha * np.exp(2j * np.pi * y / 4)
                marg[x, y] -= np.exp(-abs(ax) ** 2 / 2 - abs(ay) ** 2 / 2 + np.conj(ay) * ax) / 4
        drift = 0.5 * float(np.abs(np.linalg.eigvalsh(marg)).sum())
        self.assertLessEqual(drift, math.sqrt(weight), msg=f"marginal drift {drift}")


class Finite(Question):
    """
    The finite-key refusal, one message per asymptotic bound.
    """

    def message(self, certified):
        """
        The refusal text of one branch.
        """
        try:
            _core.dm_finite(1e9, certified)
        except NotImplementedError as err:
            text = str(err)
        else:
            text = ""

        self.assertNotEqual(text, "", msg="dm_finite refuses")

        return text

    def test_analytic_refusal(self):
        """
        The analytic refusal names Lupo and Ouyang's heterodyne receiver with finite range and bin count, and where each
        sits here: `q.Heterodyne.clip` and a `q.ADC` that `q.Link` refuses by name.
        """
        text = self.message(False)
        for needle in (
            "ANALYTIC",
            "arXiv:2108.00428",
            "PRX Quantum 3, 010341 (2022)",
            "P_0(R)",
            "bins per quadrature",
            "q.Heterodyne.clip",
            "q.ADC is refused by name",
            "arXiv:2103.13945v3 Eq. (25)",
            "asymptotic",
        ):
            self.assertIn(needle, text, msg=f"analytic refusal names {needle}")

    def test_certified_refusal(self):
        """
        The certified refusal names all four shipped pieces and the protocol steps a `q.Link` cannot state: the energy
        test, the block split and the clipped detector.
        """
        text = self.message(True)
        for needle in (
            "CERTIFIED",
            "arXiv:2301.08686",
            "PRX Quantum 4, 040306 (2023)",
            "Delta(w)",
            "energy",
            "Hoeffding",
            "Tr P + Tr N <= 2 sqrt(w)",
            "arXiv:2006.04661",
            "asymptotic",
            "all four pieces",
            "dm_energy",
            "dm_accept",
            "dm_secpar",
            "dm_length",
            "dm_relaxed",
            "WHAT REFUSES IS EVERY REMAINING PIECE BEING A PROTOCOL STEP",
        ):
            self.assertIn(needle, text, msg=f"certified refusal names {needle}")

    def test_branches_differ(self):
        """
        The analytic and certified messages differ, each keeps its own citation, and the analytic one points at
        `certified = true`.
        """
        analytic = self.message(False)
        certified = self.message(True)
        self.assertNotEqual(analytic, certified, msg="analytic == certified")
        self.assertIn("certified = true", analytic, msg="certified = true in analytic")
        self.assertNotIn("2301.08686", analytic, msg="2301.08686 in analytic")
        self.assertNotIn("2108.00428", certified, msg="2108.00428 in certified")

    def test_block_first(self):
        """
        A bad `n_total` raises `ValueError` before either `NotImplementedError` refusal.
        """
        for bad in (0.0, -1.0, float("nan")):
            self.assertFails(
                ValueError,
                "n_total must be > 0",
                lambda v=bad: _core.dm_finite(v, False),
                msg=f"n_total = {bad}",
            )

    def test_dimension_price(self):
        """
        The charges the certified refusal quotes, 0.101 bit at $w = 10^{-4}$ and 1.34e-2 at $10^{-6}$, are read back off
        `dm_price`.
        """
        text = self.message(True)
        self.assertClose(_core.dm_price(1e-4, 4), 0.101, atol=1e-3, msg="Delta(1e-4) over four symbols")
        self.assertClose(_core.dm_price(1e-6, 4), 1.34e-2, atol=1e-4, msg="Delta(1e-6) over four symbols")
        self.assertIn("0.101 bit per pulse at w = 1e-4", text, msg="0.101 in message")
        self.assertIn("1.34e-2 bit at w = 1e-6", text, msg="1.34e-2 in message")
        self.assertIn("dm_price", text, msg="dm_price in message")


class Budget(Question):
    """
    The finite-size pieces that ship: Kanitschar Eq. (4)'s charge, Theorem 3's energy test,
    Theorem 4's acceptance width, Theorem 6's epsilon, and the Eq. (9) they assemble.
    """

    def charge(self, w, alphabet):
        """
        Kanitschar Eq. (4), recomputed independently of the engine.
        """
        root = math.sqrt(w)
        share = root / (1.0 + root)
        binary = 0.0 if share <= 0.0 else -share * math.log2(share) - (1 - share) * math.log2(1 - share)

        return root * math.log2(alphabet) + (1.0 + root) * binary

    def poisson(self, nc, beta):
        """
        The regularised upper incomplete gamma at integer order as a Poisson tail sum, not the engine's gamma ladder.
        """
        acc = 0.0
        term = math.exp(-beta)
        for k in range(nc + 1):
            acc += term
            term *= beta / (k + 1)

        return acc

    def test_price_form(self):
        """
        `dm_price` equals Kanitschar Eq. (4) evaluated here across five weights and four key-map sizes, and is exactly
        zero at zero weight.
        """
        for w in (1e-8, 1e-6, 1e-4, 1e-2, 0.5):
            for alphabet in (2, 3, 4, 8):
                got = _core.dm_price(w, alphabet)
                self.assertClose(got, self.charge(w, alphabet), msg=f"Delta({w}) over {alphabet}")
        self.assertEqual(_core.dm_price(0.0, 4), 0.0, msg="Delta(0)")

    def test_price_quoted(self):
        """
        `dm_price` gives the 0.101 and 1.34e-2 bit the certified refusal quotes and rises with both the weight and the
        key-map size.
        """
        self.assertClose(_core.dm_price(1e-4, 4), 0.101, atol=1e-3, msg="Delta(1e-4) over four symbols")
        self.assertClose(_core.dm_price(1e-6, 4), 1.34e-2, atol=1e-4, msg="Delta(1e-6) a decade down")
        rising = [_core.dm_price(w, 4) for w in (1e-8, 1e-6, 1e-4, 1e-2)]
        wider = [_core.dm_price(1e-4, z) for z in (2, 4, 8, 16)]
        self.assertMonotone(rising, rising=True, msg=f"Delta against w: {rising}")
        self.assertMonotone(wider, rising=True, msg=f"Delta against |Z|: {wider}")

    def test_price_dominates(self):
        """
        At a weight of 1e-4 and $10^{12}$ signals the dimension-reduction charge, 0.101 bit per pulse, exceeds the
        equipartition correction by over three orders.
        """
        _ln, _raw, aep, price = _core.dm_length(1e12, 1e12, 0.5, 4, 1e-4, 0.7e-10, 0.2e-10, 0.0, "relaxed")
        self.assertGreater(price / aep, 1e3, msg=f"price {price}, aep {aep}")
        self.assertEqual(price, _core.dm_price(1e-4, 4), msg=f"price {price}")

    def test_energy_ratio(self):
        """
        Theorem 3's $r$ is one over the regularised upper incomplete gamma to 1e-11 against the Poisson tail identity,
        and never below one.
        """
        for nc in (2, 5, 10, 20, 30):
            for beta in (0.5, 5.0, 12.0):
                r = _core.dm_energy(nc, beta, 1e6, 0.0, 1e-3)[1]
                self.assertClose(r * self.poisson(nc, beta), 1.0, atol=1e-11, msg=f"r at nc={nc}, beta={beta}")
                self.assertGreaterEqual(r, 1.0, msg=f"r = {r}")

    def test_energy_trade(self):
        """
        More test rounds and a larger claimed weight each lower the energy-test failure probability, while Eq. (4)
        charges the larger weight back (Kanitschar Sec. VI B).
        """
        longer = [_core.dm_energy(20, 5.0, k, 10.0, 1e-4)[0] for k in (1e6, 1e7, 1e8)]
        heavier = [_core.dm_energy(20, 5.0, 1e7, 10.0, w)[0] for w in (1e-5, 1e-4, 1e-3)]
        charged = [_core.dm_price(w, 4) for w in (1e-5, 1e-4, 1e-3)]
        self.assertMonotone(longer, rising=False, msg=f"eps_ET against k_T: {longer}")
        self.assertMonotone(heavier, rising=False, msg=f"eps_ET against w: {heavier}")
        self.assertMonotone(charged, rising=True, msg=f"Delta against w: {charged}")

    def test_energy_formula(self):
        """
        The returned logarithm is Eq. (5), $\\log_2(l_T + 1)$ minus $k_T$ times the Bernoulli Kullback-Leibler
        divergence in bits between the observed and claimed failure fractions.
        """
        log2, r, div = _core.dm_energy(20, 5.0, 1e9, 10.0, 1e-4)
        seen = 10.0 / 1e9
        claim = 1e-4 / r
        want = seen * math.log2(seen / claim) + (1 - seen) * math.log2((1 - seen) / (1 - claim))
        self.assertClose(div / want, 1.0, atol=1e-12, msg=f"divergence {div}")
        self.assertClose(log2 / (math.log2(11.0) - 1e9 * div), 1.0, atol=1e-13, msg=f"log2 eps_ET {log2}")
        self.assertLess(log2, -1000.0, msg=f"log2 eps_ET {log2}")

    def test_energy_domain(self):
        """
        `dm_energy` refuses $l_T/k_T \\ge w/r$, where Theorem 3 does not hold, $l_T \\ge k_T$, and a radius leaving the
        incomplete gamma outside f64.
        """
        self.assertFails(
            ValueError,
            "l_t / k_t < w / r",
            lambda: _core.dm_energy(20, 5.0, 1e6, 1e4, 1e-4),
            msg="l_t / k_t above w / r",
        )
        self.assertFails(
            ValueError,
            "l_t must be below k_t",
            lambda: _core.dm_energy(20, 5.0, 1e6, 1e6, 1e-2),
            msg="l_t = k_t",
        )
        self.assertFails(
            ValueError,
            "outside f64 range",
            lambda: _core.dm_energy(20, 900.0, 1e6, 0.0, 1e-3),
            msg="beta = 900",
        )

    def test_weight_inverts(self):
        """
        `dm_wchoice`'s $w_\\epsilon$ (Sec. VI B) inverts Eq. (5): fed back to the energy test it returns the target
        logarithm to 1e-6.
        """
        want = math.log2(1e-11)
        for k in (1e8, 1e9, 1e10):
            w, _r = _core.dm_wchoice(20, 5.0, k, k * 1e-8, want)
            back = _core.dm_energy(20, 5.0, k, k * 1e-8, w)[0]
            self.assertClose(back / want, 1.0, atol=1e-6, msg=f"round trip at k_T = {k}")
            self.assertLess(w, 1.0, msg=f"w = {w}")

    def test_weight_falls(self):
        """
        More test rounds buy a smaller weight and a smaller dimension-reduction charge at a fixed failure probability,
        and a positive target logarithm is refused.
        """
        want = math.log2(1e-11)
        weights = [_core.dm_wchoice(20, 5.0, k, k * 1e-8, want)[0] for k in (1e8, 1e9, 1e10, 1e11)]
        charges = [_core.dm_price(w, 4) for w in weights]
        self.assertMonotone(weights, rising=False, msg=f"w against k_T: {weights}")
        self.assertMonotone(charges, rising=False, msg=f"Delta against k_T: {charges}")
        self.assertFails(
            ValueError,
            "finite NEGATIVE log2 eps_ET",
            lambda: _core.dm_wchoice(20, 5.0, 1e9, 10.0, 0.5),
            msg="log2 eps_ET = 0.5",
        )

    def test_accept_halves(self):
        """
        Theorem 4's acceptance half-width is Hoeffding's, the semidefinite form exactly half the general one, and it
        narrows as one over the square root of the rounds.
        """
        span = math.log(2 / 7e-11)
        wide = _core.dm_accept(49.0, 1e9, 7e-11, False)
        half = _core.dm_accept(49.0, 1e9, 7e-11, True)
        self.assertClose(wide / math.sqrt(2 * 49.0**2 / 1e9 * span), 1.0, atol=1e-14, msg="the general half-width")
        self.assertClose(half / wide, 0.5, atol=1e-15, msg="semidefinite / general")
        pair = [_core.dm_accept(49.0, m, 7e-11, False) for m in (1e6, 1e8)]
        self.assertClose(pair[0] / pair[1], 10.0, atol=1e-12, msg="width ratio, 1e6 vs 1e8 rounds")

    def test_secpar_shape(self):
        """
        Kanitschar's five demonstration epsilons compose to their stated 1e-10 through a correctness term outside a
        maximum over two branches, which a plain sum overstates and a plain maximum understates.
        """
        parts = (0.2e-10, 0.2e-10, 0.7e-10, 0.1e-10, 0.7e-10)
        got = _core.dm_secpar(*parts)
        self.assertClose(got / 1e-10, 1.0, atol=1e-14, msg=f"composed eps {got}")
        self.assertGreater(sum(parts), got, msg="sum of parts")
        self.assertLess(max(parts), got, msg="max of parts")

    def test_secpar_branches(self):
        """
        Each of the two maximised branches, privacy amplification and testing, can bind the composed parameter.
        """
        pa = _core.dm_secpar(1e-10, 1e-9, 1e-10, 1e-12, 1e-12)
        test = _core.dm_secpar(1e-10, 1e-12, 1e-12, 1e-9, 1e-9)
        self.assertClose(pa / (1e-10 + 0.5e-9 + 1e-10), 1.0, atol=1e-14, msg=f"PA branch {pa}")
        self.assertClose(test / (1e-10 + 2e-9), 1.0, atol=1e-14, msg=f"testing branch {test}")

    def test_length_assembles(self):
        """
        `dm_length` returns Theorem 6 Eq. (9) in bits over the block, rebuilt here from the equipartition and dimension
        terms the same call returns, and floors only the reported length.
        """
        leak = 0.15 * 0.9e12
        length, raw, aep, price = _core.dm_length(1e12, 0.9e12, 0.35, 4, 1e-8, 0.7e-10, 0.2e-10, leak, "relaxed")
        want = 0.9e12 * (0.35 - aep - price) - leak - 2 * math.log2(1 / 0.2e-10)
        self.assertClose(raw / want, 1.0, atol=1e-13, msg="Eq. (9) times N")
        self.assertEqual(length, math.floor(raw), msg=f"length {length}, raw {raw}")
        form = 2 * math.log2(7.0) * math.sqrt(math.log2(2 / 0.7e-10) / 0.9e12)
        self.assertClose(aep / form, 1.0, atol=1e-13, msg=f"aep {aep}")

    def test_length_approaches(self):
        """
        At fixed keying fraction and leakage share the rate per signal rises with the block and converges from below on
        the entropy less leakage and dimension charge, to 1e-4 at $10^{14}$.
        """
        rates = []
        for n in (1e8, 1e10, 1e12, 1e14):
            leak = 0.15 * n
            length = _core.dm_length(n, n, 0.35, 4, 1e-8, 0.7e-10, 0.2e-10, leak, "relaxed")[1]
            rates.append(length / n)

        limit = 0.35 - 0.15 - _core.dm_price(1e-8, 4)
        self.assertMonotone(rates, rising=True, msg=f"rates {rates}")
        self.assertLess(rates[-1], limit, msg=f"rate {rates[-1]} vs {limit}")
        self.assertClose(rates[-1] / limit, 1.0, atol=1e-4, msg=f"rate {rates[-1]} vs {limit}")
        self.assertGreater(limit, 0.19, msg=f"limit {limit}")

    def test_length_source(self):
        """
        `dm_length` refuses an entropy from the asymptotic feasible set, whose minimum overstates the key since that set
        is contained in the relaxed one, and an unknown source.
        """
        args = (1e12, 0.9e12, 0.35, 4, 1e-8, 0.7e-10, 0.2e-10, 0.0)
        for bad in ("certified", "asymptotic", "dm_secure", "dm_trusted"):
            self.assertFails(
                NotImplementedError,
                "CONTAINED",
                lambda s=bad: _core.dm_length(*args, s),
                msg=f"source={bad}",
            )
        self.assertFails(
            ValueError,
            "unknown source",
            lambda: _core.dm_length(*args, "measured"),
            msg="source=measured",
        )

    def test_length_guards(self):
        """
        `dm_length` refuses an entropy above $\\log_2\\lvert Z\\rvert$ and keyed rounds above `n_total`, and `dm_price`
        a one-symbol key map.
        """
        self.assertFails(
            ValueError,
            "exceeds log2|Z|",
            lambda: _core.dm_length(1e12, 1e12, 2.5, 4, 1e-8, 0.7e-10, 0.2e-10, 0.0, "relaxed"),
            msg="entropy 2.5 at |Z| = 4",
        )
        self.assertFails(
            ValueError,
            "must not exceed n_total",
            lambda: _core.dm_length(1e10, 1e12, 0.3, 4, 1e-8, 0.7e-10, 0.2e-10, 0.0, "relaxed"),
            msg="keyed 1e12 over n_total 1e10",
        )
        self.assertFails(
            ValueError,
            "alphabet must be at least 2",
            lambda: _core.dm_price(1e-4, 1),
            msg="alphabet = 1",
        )


if __name__ == "__main__":
    rc = Exam(
        "DmcsConstellation",
        "The M-PSK ensemble, its effective state, and the gap to Gaussian modulation",
        "dmcs_constellation.md",
    ).run(load(Constellation))
    rc |= Exam(
        "DmcsInformation",
        "Exact constellation mutual information against the Gaussian substitution",
        "dmcs_information.md",
    ).run(load(Information))
    rc |= Exam(
        "DmcsQuadrature",
        "The Gauss-Hermite rule: ordered nodes, normalised weights, no negative information",
        "dmcs_quadrature.md",
    ).run(load(Quadrature))
    rc |= Exam(
        "DmcsAnchors",
        "Published QPSK bounds reproduced, and how far they sit below the SDP proofs",
        "dmcs_anchors.md",
    ).run(load(Anchors))
    rc |= Exam(
        "DmcsLimit",
        "The Gaussian limit: exact tie to cv_rate, and convergence from below",
        "dmcs_limit.md",
    ).run(load(Limit))
    rc |= Exam(
        "DmcsShape",
        "Penalty growing with distance and the small, shrinking optimal modulation",
        "dmcs_shape.md",
    ).run(load(Shape))
    rc |= Exam(
        "DmcsNumerics",
        "Region operators, the source-replacement state, and the moments they carry",
        "dmcs_numerics.md",
    ).run(load(Numerics))
    rc |= Exam(
        "DmcsObjective",
        "The relative entropy and its gradient against numpy and a finite difference",
        "dmcs_objective.md",
    ).run(load(Objective))
    rc |= Exam(
        "DmcsCertified",
        "The proved bound: what it may not exceed, and how much reach it buys",
        "dmcs_certified.md",
    ).run(load(Certified))
    rc |= Exam(
        "DmcsTrusted",
        "The trusted receiver: its POVM, its four observables, and the distance it buys",
        "dmcs_trusted.md",
    ).run(load(Trusted))
    rc |= Exam(
        "DmcsNumerical",
        "Cutoffs, perturbations and channels the numerical proof refuses",
        "dmcs_numerical.md",
    ).run(load(Numerical))
    rc |= Exam(
        "DmcsRelaxed",
        "Kanitschar's relaxed feasible set: what widening it costs and the length it feeds",
        "dmcs_relaxed.md",
    ).run(load(Relaxed))
    rc |= Exam(
        "DmcsFinite",
        "The finite-key refusal: a published analysis on each bound, and what is missing from it",
        "dmcs_finite.md",
    ).run(load(Finite))
    rc |= Exam(
        "DmcsBudget",
        "The finite-size pieces that ship: the dimension price, the energy test and the length",
        "dmcs_budget.md",
    ).run(load(Budget))
    rc |= Exam(
        "DmcsGuards",
        "Non-isotropic constellations, unphysical correlations and out-of-domain inputs",
        "dmcs_guards.md",
    ).run(load(Guards))
    sys.exit(rc)
