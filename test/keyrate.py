import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from qkd import _core


def rate(va=5.0, t=0.3, xi=0.02, eta=0.6, vel=0.1, beta=0.95, hom=True, trusted=True):
    return _core.cv_rate(va, t, xi, eta, vel, beta, hom, trusted)


def key(**kw):
    return rate(**kw)[2]


def entropy(x):
    """
    G(x) = (x+1)log2(x+1) - x log2 x, the thermal entropy (not binary), written literally.
    """
    if x <= 0.0:
        return 0.0

    return (x + 1.0) * math.log2(x + 1.0) - x * math.log2(x)


def transparent(va, xi, eta, vel):
    """
    Heterodyne chi_BE at T = 1 from Lodewyck Eq. (20) by hand: chi_line = xi, sqrt(B) = V*xi + 1,
    A - 2*sqrt(B) = xi^2, all non-negative sums.
    """
    v = va + 1.0
    det = (1.0 + (1.0 - eta) + 2.0 * vel) / eta
    root = v * xi + 1.0
    span = math.sqrt(xi * xi + 4.0 * v * xi + 4.0)
    q = v + xi + det
    delta = xi * (v - det)
    wide = math.sqrt(delta * delta + 4.0 * (v + root * det) * q)
    top = (wide + abs(delta)) / (2.0 * q)
    nus = ((span + xi) / 2.0, (span - xi) / 2.0, top, (v + root * det) / (q * top))

    return sum(w * entropy((n - 1.0) / 2.0) for w, n in zip((1, 1, -1, -1), nus))


def finite(n=1e9, pe=0.5, eps=1e-10, t=0.3, xi=0.02, hom=True, trusted=True):
    return _core.cv_finite(5.0, t, xi, 0.6, 0.1, 0.95, hom, trusted, n, pe, eps, eps, eps)


# Energy-test floors in mean photons: Alice V_A/2 = 2.5, Bob (T*(V_A + xi) + chi_det)/2 = 2.0864.
D_A, D_B = 2.5, 2.1


def general(n=1e9, pe=0.5, eps=1e-10, k=1e6, da=D_A, db=D_B, t=0.3, xi=0.02, hom=False, trusted=True):
    """
    cv_general on `finite`'s link, heterodyne. Returns (length, raw, cutoff, eps_coll, toll).
    """

    return _core.cv_general(5.0, t, xi, 0.6, 0.1, 0.95, hom, trusted, n, pe, eps, k, da, db)


class Quantile(Question):
    def test_z_textbook(self):
        """
        `z_pe` reproduces the textbook two-sided quantiles 1.959964 at 5%, 2.575829 at 1%
        and 3.290527 at 0.1%, to 1e-5.
        """
        self.assertClose(_core.z_pe(0.05), 1.959964, atol=1e-5, msg="5% two-sided z")
        self.assertClose(_core.z_pe(0.01), 2.575829, atol=1e-5, msg="1% two-sided z")
        self.assertClose(_core.z_pe(0.001), 3.290527, atol=1e-5, msg="0.1% two-sided z")

    def test_z_finite_size(self):
        """
        At the eps = 1e-10 the CV-QKD literature uses, `z_pe` gives 6.466951.
        """
        self.assertClose(_core.z_pe(1e-10), 6.466951, atol=1e-5, msg="6.5 sigma")

    def test_z_seam(self):
        """
        Acklam's approximation switches branch at eps = 0.0485; straddling it by 1e-10 does
        not step z or reverse the ordering.
        """
        lo = _core.z_pe(0.0484999999)
        hi = _core.z_pe(0.0485000001)
        self.assertClose(lo, hi, atol=1e-7, msg="gap across the seam")
        self.assertGreater(lo, hi, msg="ordering across the seam")

    def test_z_extremes(self):
        """
        The tail branch survives eps to 1e-300 without hitting the 1 - eps/2 == 1 rounding
        wall, and eps near 1 gives a small positive z.
        """
        deep = _core.z_pe(1e-300)
        shallow = _core.z_pe(0.999999999)
        self.assertFinite(deep, msg=f"z at 1e-300 is {deep!r}")
        self.assertGreater(deep, 30.0, msg=f"z at 1e-300 is {deep!r}")
        self.assertGreater(shallow, 0.0, msg=f"z at eps -> 1 is {shallow!r}")
        self.assertLess(shallow, 1e-8, msg=f"z at eps -> 1 is {shallow!r}")

    def test_z_guards(self):
        """
        eps outside (0, 1), NaN and infinity included, raises ValueError.
        """
        for bad in (0.0, 1.0, 1.5, -0.1, float("nan"), float("inf")):
            self.assertFails(ValueError, "eps must be", _core.z_pe, bad, msg=f"eps = {bad}")


class Monotone(Question):
    def test_rate_rises_transmittance(self):
        """
        The key rate rises with channel transmittance.
        """
        keys = [key(t=x) for x in (0.05, 0.1, 0.3, 0.6, 0.9)]
        self.assertMonotone(keys, msg="key against T")

    def test_rate_falls_electronic(self):
        """
        The rate falls as trusted electronic noise rises.
        """
        keys = [key(vel=x) for x in (0.0, 0.05, 0.1, 0.3, 1.0)]
        self.assertMonotone(keys, rising=False, msg="key against v_el")

    def test_heterodyne_information(self):
        """
        Heterodyne carries more mutual information than homodyne at the same hardware.
        """
        for t in (0.05, 0.2, 0.5, 0.95):
            for eta in (0.3, 0.7, 1.0):
                hom = rate(t=t, eta=eta, hom=True)[0]
                het = rate(t=t, eta=eta, hom=False)[0]
                self.assertGreater(het, hom, msg=f"I_AB at T={t}, eta={eta}")

    def test_heterodyne_wins_short(self):
        """
        With an ideal receiver, perfect reconciliation and T = 0.9, heterodyne beats
        homodyne.
        """
        hom = key(t=0.9, eta=1.0, vel=0.0, beta=1.0, hom=True)
        het = key(t=0.9, eta=1.0, vel=0.0, beta=1.0, hom=False)
        self.assertGreater(het, hom, msg=f"het {het!r} against hom {hom!r}")

    def test_homodyne_wins_long(self):
        """
        The ordering reverses at T = 0.1, beta = 0.85.
        """
        hom = key(t=0.1, eta=1.0, vel=0.0, beta=0.85, hom=True)
        het = key(t=0.1, eta=1.0, vel=0.0, beta=0.85, hom=False)
        self.assertGreater(hom, het, msg=f"hom {hom!r} against het {het!r}")


class Conditioning(Question):
    """
    Against 60-digit Lodewyck et al., PRA 76, 042305 (2007), Eqs. (19)-(24).
    """

    def test_small_eigenvalue(self):
        """
        The small symplectic eigenvalue is sqrt(B)/nu1, not (A - sqrt(A^2 - 4B))/2, whose lost
        digits past A^2 ~ 4.5e15 B understate Eve: keys +4.542207 and 0.205440 against 60-digit
        -2.385982 and 0.053971.
        """
        flip = rate(va=4.273e11, t=0.9388, xi=5.776, eta=0.573, vel=0.02546)
        small = rate(va=6.095e8, t=0.8018, xi=0.01072, eta=0.562, vel=0.00156)
        self.assertClose(flip[1], 19.341069172909820, atol=1e-12, msg="chi_BE against 60 digits")
        self.assertClose(flip[2], -2.3859820570583565, atol=1e-12, msg="key against 60 digits")
        self.assertClose(small[2], 0.05397080747188556, atol=1e-14, msg="key against 60 digits")

    def test_entropy_large(self):
        """
        With G(x) = log2(x) + (x+1)log2(1 + 1/x) above x = 1, V_A = 1e50 gives the 60-digit
        chi_BE 81.610030 and key -3.956908, where the literal form (zero past x ~ 9e15) gives 0
        and +77.65.
        """
        point = rate(va=1e50)
        self.assertClose(point[1], 81.6100295403018, atol=1e-9, msg="chi_BE against 60 digits")
        self.assertClose(point[2], -3.956907848041647, atol=1e-9, msg="key against 60 digits")

    def test_overflow_safe(self):
        """
        Over V_A = 1e3 to 1e300 the key is never positive, and at 1e200, past f64 range (~1e78
        homodyne, ~1e154 heterodyne), chi_BE is +inf and the key -inf.
        """
        for e in range(3, 301):
            for hom in (True, False):
                point = rate(va=10.0**e, hom=hom)
                self.assertLessEqual(point[2], 0.0, msg=f"V_A = 1e{e} hom={hom} key {point[2]!r}")

        broke = rate(va=1e200)
        self.assertEqual(broke[1], float("inf"), msg=f"chi_BE at 1e200 is {broke[1]!r}")
        self.assertEqual(broke[2], float("-inf"), msg=f"key at 1e200 is {broke[2]!r}")
        self.assertFinite(broke[0], msg=f"I_AB at 1e200 is {broke[0]!r}")

    def test_perfect_channel(self):
        """
        At T = 1, xi = 0 (A^2 = 4B, C^2 = 4D) chi_BE is exactly 0 and the key positive and
        continuous across the corner.
        """
        for hom in (True, False):
            point = rate(t=1.0, xi=0.0, hom=hom)
            self.assertEqual(point[1], 0.0, msg=f"chi_BE at hom={hom} is {point[1]!r}")
            self.assertGreater(point[2], 0.0, msg=f"key at hom={hom} is {point[2]!r}")

        near = [key(t=1.0, xi=x) for x in (0.0, 1e-12, 1e-9, 1e-6)]
        self.assertMonotone(near, rising=False, msg="key against xi at T = 1")
        self.assertClose(near[0], near[2], atol=1e-5, msg="step at the corner")

    def test_transparent_limit(self):
        """
        chi_BE matches `transparent` to 7.2e-15 over V_A 5 to 1e4, xi 1e-9 to 0.5 and three
        detectors, where the textbook A, B, C, D miss by 3.6e-4 and by -2.6e-7 (wrong sign).
        """
        for va in (5.0, 40.0, 1e3, 1e4):
            for eta, vel in ((1.0, 0.0), (0.6, 0.1), (0.85, 0.05)):
                for xi in (1e-9, 1e-6, 1e-4, 1e-2, 0.1, 0.5):
                    got = rate(va=va, t=1.0, xi=xi, eta=eta, vel=vel, hom=False)
                    self.assertClose(
                        got[1],
                        transparent(va, xi, eta, vel),
                        atol=1e-12,
                        msg=f"V_A={va:g} xi={xi:g} eta={eta}: chi_BE = {got[1]!r}",
                    )

    def test_transparent_joins(self):
        """
        Approaching T = 1 the Holevo bound falls monotonically to its T = 1 value, the last
        decade moving it under 1e-4, where the textbook form is non-monotone at V_A = 1e4.
        """
        drop = (1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 0.0)
        for hom in (True, False):
            for va, xi in ((5.0, 1e-3), (40.0, 1e-4), (1e4, 1e-9), (5.0, 0.05)):
                near = [rate(va=va, t=1.0 - d, xi=xi, hom=hom)[1] for d in drop]
                self.assertMonotone(
                    near,
                    rising=False,
                    strict=False,
                    msg=f"V_A={va:g} xi={xi:g} hom={hom}: {near}",
                )
                self.assertClose(
                    near[-1],
                    near[-3],
                    atol=1e-4,
                    msg=f"V_A={va:g} xi={xi:g} hom={hom} steps at the corner",
                )


class Guards(Guarded):
    def test_recon_floor(self):
        """
        f_ec below the Shannon limit 1 is refused (it multiplies h2(e), unlike beta in [0, 1]
        on I_AB), and dps_rate(0.5, 0.02, 0.2, 1.0) is 0.141516.
        """
        for bad in (0.0, 0.5, 0.95, 0.999999):
            self.assertBad(
                "f_ec must be >= 1",
                _core.dps_rate,
                (0.5, 0.02, 0.2, bad),
                msg=f"f_ec = {bad}",
            )
        self.assertClose(
            _core.dps_rate(0.5, 0.02, 0.2, 1.0),
            0.1415158723439881,
            msg="rate at f_ec = 1",
        )

    def test_bounds_vacuum(self):
        """
        cv_bounds refuses sigma2_hat below 1, Bob's residual variance in SNU with the vacuum
        (twice a vacuum-1/2 variance).
        """
        for bad in (0.5, 0.999999, 0.0, -1.0):
            self.assertBad(
                "sigma2_hat must be >= 1",
                _core.cv_bounds,
                (0.5, bad, 1e6, 5.0, 1e-10),
                msg=f"sigma2_hat = {bad}",
            )

        t_lo, xi_hi = _core.cv_bounds(0.5, 1.0, 1e6, 5.0, 1e-10)
        self.assertGreater(xi_hi, 0.0, msg=f"xi_max {xi_hi!r}")
        self.assertGreater(t_lo, 0.0, msg=f"t_min {t_lo!r}")

    def test_dead_channel(self):
        """
        As T goes to zero the rate falls monotonically with every component finite, chi_BE
        overtaking beta*I_AB at T = 2.16e-10 and cv_rate returning -4.0e-14 unclamped at 1e-12.
        """
        # 3e-15 cancellation floor in chi_BE: the sign holds, later digits do not, below 1e-13 the trend breaks.
        keys = [key(t=t) for t in (1e-12, 1e-9, 1e-6, 1e-3)]
        self.assertMonotone(keys, msg="key against T")
        self.assertFinite(keys, msg=f"keys {keys}")
        self.assertLess(abs(keys[2]), 1e-6, msg=f"key at T = 1e-6 is {keys[2]!r}")
        self.assertFinite(rate(t=1e-12), msg="components at T = 1e-12")

    def test_huge_noise(self):
        """
        Excess noise from 1 to 1e12 gives a finite negative rate, not NaN.
        """
        for xi in (1.0, 1e3, 1e6, 1e12):
            point = rate(xi=xi)
            self.assertFinite(point, msg=f"xi = {xi}")
            self.assertLess(point[2], 0.0, msg=f"xi = {xi} key {point[2]!r}")

    def test_tiny_modulation(self):
        """
        As V_A goes to zero the mutual information goes to zero, the rate to a finite
        negative floor, and nothing diverges.
        """
        points = [rate(va=v) for v in (1e-9, 1e-6, 1e-3, 1e-1)]
        for point in points:
            self.assertFinite(point, msg=f"point {point}")
        self.assertMonotone([p[0] for p in points], msg="I_AB against V_A")
        self.assertLess(points[0][0], 1e-9, msg=f"I_AB at V_A = 1e-9 is {points[0][0]!r}")
        self.assertLess(points[0][2], 0.0, msg=f"key at V_A = 1e-9 is {points[0][2]!r}")

    def test_rate_domain(self):
        """
        Out-of-domain arguments raise ValueError naming the parameter: T <= 0, eta outside
        (0, 1], non-positive V_A, negative xi, beta outside [0, 1].
        """
        for bad in (0.0, -0.1, 1.5, float("nan")):
            self.assertFails(ValueError, "t must be", lambda x=bad: rate(t=x), msg=f"T = {bad}")

        for bad in (0.0, 1.5, -0.2):
            self.assertFails(ValueError, "eta must be", lambda x=bad: rate(eta=x), msg=f"eta = {bad}")
        self.assertFails(ValueError, "va must be", lambda: rate(va=-1.0), msg="V_A = -1")
        self.assertFails(ValueError, "xi must be", lambda: rate(xi=-1e-9), msg="xi = -1e-9")
        self.assertFails(ValueError, "vel must be", lambda: rate(vel=-1e-9), msg="v_el = -1e-9")
        self.assertFails(ValueError, "beta must be", lambda: rate(beta=1.5), msg="beta = 1.5")

    def test_dps_domain(self):
        """
        dps_rate rejects a click probability outside [0, 1], a negative mean photon number
        and a negative error-correction factor.
        """
        for bad in (-0.1, 1.1, float("inf")):
            self.assertBad(
                "p_click must be",
                _core.dps_rate,
                (bad, 0.01, 0.2, 1.16),
                msg=f"p_click = {bad}",
            )
        self.assertBad(
            "mu must be",
            _core.dps_rate,
            (0.5, 0.01, -0.2, 1.16),
            msg="mu = -0.2",
        )
        self.assertBad(
            "f_ec must be",
            _core.dps_rate,
            (0.5, 0.01, 0.2, -1.0),
            msg="f_ec = -1",
        )
        self.assertBad(
            "qber must be",
            _core.dps_rate,
            (0.5, -0.01, 0.2, 1.16),
            msg="qber = -0.01",
        )


class Finite(Question):
    def test_bounds_tighten(self):
        """
        T_min climbs towards T and xi_max falls towards xi as the block grows, converging to
        0.3 and 0.02 at n = 1e14.
        """
        runs = [finite(n=n) for n in (1e6, 1e8, 1e10, 1e12, 1e14)]
        self.assertMonotone([r[3] for r in runs], msg="T_min against n")
        self.assertMonotone([r[4] for r in runs], rising=False, msg="xi_max against n")
        self.assertClose(runs[-1][3], 0.3, atol=1e-5, msg="T_min at n = 1e14")
        self.assertClose(runs[-1][4], 0.02, atol=1e-4, msg="xi_max at n = 1e14")

    def test_delta_shrinks(self):
        """
        The smooth min-entropy penalty is strictly positive and falls like 1/sqrt(n).
        """
        deltas = [finite(n=n)[5] for n in (1e6, 1e8, 1e10, 1e12, 1e14)]
        self.assertMonotone(deltas, rising=False, msg="Delta against n")
        self.assertFinite(deltas, msg=f"deltas {deltas}")
        self.assertGreater(deltas[-1], 0.0, msg=f"Delta at n = 1e14 is {deltas[-1]!r}")

        # Delta's O(1/n) term offsets this ratio by 0.021.
        self.assertClose(deltas[0] / deltas[1], 10.0, atol=0.05, msg="Delta ratio per two decades")

    def test_asymptotic_limit(self):
        """
        At n = 1e18 the finite-size rate is the asymptotic rate times 1 - pe_fraction.
        """
        want = key()
        for pe in (0.1, 0.5, 0.9):
            got = finite(n=1e18, pe=pe)[2]
            self.assertClose(got, (1.0 - pe) * want, atol=1e-6, msg=f"pe_fraction = {pe}")

    def test_dead_fiber(self):
        """
        At n = 50 the interval reaches T = 0: T_min exactly 0, xi_max and chi_BE infinite, rate 0.
        """
        _i, chi, rk, tmin, ximax, delta = finite(n=50.0)
        self.assertEqual(tmin, 0.0, msg=f"T_min {tmin!r}")
        self.assertEqual(rk, 0.0, msg=f"rate {rk!r}")
        self.assertTrue(math.isinf(ximax), msg=f"xi_max {ximax!r}")
        self.assertTrue(math.isinf(chi), msg=f"chi_BE {chi!r}")
        self.assertFinite(delta, msg=f"Delta {delta!r}")

    def test_finite_domain(self):
        """
        Block size, estimation fraction and the three epsilons are validated: n positive,
        the fractions strictly inside (0, 1).
        """
        self.assertFails(ValueError, "n_total must be", lambda: finite(n=-1.0), msg="n = -1")

        for bad in (0.0, 1.0, 1.5):
            self.assertFails(
                ValueError,
                "pe_fraction must be",
                lambda x=bad: finite(pe=x),
                msg=f"pe_fraction = {bad}",
            )

        for bad in (0.0, 1.0):
            self.assertFails(ValueError, "must be in (0, 1)", lambda x=bad: finite(eps=x), msg=f"eps = {bad}")

    def test_finite_detection(self):
        """
        Both trust models share t_min and xi_max, heterodyne's interval is wider than
        homodyne's, and untrusted detection costs finite key.
        """
        runs = {(h, tr): finite(n=1e10, hom=h, trusted=tr) for h in (True, False) for tr in (True, False)}
        for h in (True, False):
            self.assertClose(
                runs[(h, True)][3],
                runs[(h, False)][3],
                msg=f"t_min across trust at hom={h}",
            )
            self.assertClose(
                runs[(h, True)][4],
                runs[(h, False)][4],
                msg=f"xi_max across trust at hom={h}",
            )
        self.assertLess(
            runs[(False, True)][3],
            runs[(True, True)][3],
            msg="t_min heterodyne against homodyne",
        )
        self.assertGreater(
            runs[(False, True)][4],
            runs[(True, True)][4],
            msg="xi_max heterodyne against homodyne",
        )
        self.assertLess(
            runs[(True, False)][2],
            runs[(True, True)][2],
            msg="key untrusted against trusted",
        )


class General(Question):
    """
    cv_general: the arithmetic of the Gaussian de Finetti reduction, Leverrier, Phys. Rev.
    Lett. 118, 200501 (2017), Eq. (5). Returns a key length in bits, not a rate.
    """

    def test_certificate_recomputed(self):
        """
        K^4 * eps_coll/50 recovers the general-attack target (Eq. (5) inverted) at targets 1e-6
        to 1e-20.
        """
        for target in (1e-6, 1e-10, 1e-15, 1e-20):
            _ln, _raw, cut, eps, _toll = general(eps=target)

            self.assertClose(cut**4 / 50.0 * eps, target, atol=1e-6 * target, msg=f"target {target}")

    def test_lift_costs_key(self):
        """
        The general key length is 90.7% of the collective one at n = 1e9, 99.0% at 1e11 and
        99.7% at 1e12.
        """
        want = {
            1e9: 0.90735,
            1e11: 0.98957,
            1e12: 0.99650,
        }
        for n, ratio in want.items():
            gen = general(n=n)[0]
            coll = finite(n=n, hom=False)[2] * n

            self.assertLess(gen, coll, msg=f"n = {n}: {gen!r} against {coll!r}")

            self.assertClose(gen / coll, ratio, atol=1e-4, msg=f"n = {n} share of the collective length")

    def test_epsilon_pays_the_bill(self):
        """
        n * cv_finite at eps_coll/3, less the toll, rebuilds `raw` exactly.
        """
        for n in (1e9, 1e11, 1e12):
            _ln, raw, _cut, eps, toll = general(n=n)
            rebuilt = n * finite(n=n, eps=eps / 3.0, hom=False)[2] - toll

            self.assertEqual(raw, rebuilt, msg=f"n = {n}: {raw!r} against {rebuilt!r}")

    def test_cutoff_linear_in_block(self):
        """
        K/(n*(d_A + d_B)) stays within 2% of 1 from n = 1e7 to 1e15.
        """
        for n in (1e7, 1e9, 1e12, 1e15):
            cut = general(n=n)[2]

            self.assertClose(cut / (n * (D_A + D_B)), 1.0, atol=0.02, msg=f"K at n = {n}")

    def test_toll_is_not_the_cost(self):
        """
        The toll ceil(2 log2 C(K+4, 4)) is 248 bits at n = 1e9 and 328 at 1e12, within a bit of
        8 log2 K - 2 log2 24, and under 1e-5 of the key length.
        """
        for n, bits in ((1e9, 248.0), (1e12, 328.0)):
            length, _raw, cut, _eps, toll = general(n=n)
            grows = 8.0 * math.log2(cut) - 2.0 * math.log2(24.0)

            self.assertEqual(toll, bits, msg=f"toll at n = {n}")

            self.assertClose(toll, grows, atol=1.0, msg=f"toll against 8 log2 K at n = {n}")

            self.assertLess(toll / length, 1e-5, msg=f"toll share at n = {n}")

    def test_block_floor_rises(self):
        """
        This link distils under a collective attack from about 4e6 symbols and under a
        general one only between 1.5e7 and 2e7.
        """
        self.assertGreater(finite(n=5e6, hom=False)[2], 0.0, msg="collective rate at 5e6")
        self.assertEqual(general(n=5e6)[0], 0.0, msg="general length at 5e6")
        self.assertEqual(general(n=1.5e7)[0], 0.0, msg="general length at 1.5e7")
        self.assertGreater(general(n=2e7)[0], 0.0, msg="general length at 2e7")

    def test_more_test_modes_help(self):
        """
        As k_test grows K falls and the key length rises, saturating by 1e8.
        """
        runs = [general(k=k) for k in (1e3, 1e4, 1e6, 1e8)]

        self.assertMonotone([r[0] for r in runs], msg="length against k_test")

        self.assertMonotone([r[2] for r in runs], rising=False, msg="K against k_test")

        self.assertClose(runs[-1][0] / runs[-2][0], 1.0, atol=1e-4, msg="length ratio at saturation")

    def test_higher_threshold_costs(self):
        """
        Raising d_A raises K, shrinks eps_coll and shortens the key.
        """
        runs = [general(da=d) for d in (2.5, 5.0, 25.0)]

        self.assertMonotone([r[2] for r in runs], msg="K against d_a")

        self.assertMonotone([r[3] for r in runs], rising=False, msg="eps_coll against d_a")

        self.assertMonotone([r[0] for r in runs], rising=False, msg="length against d_a")

    def test_dead_fibre_clamped(self):
        """
        A block too short to bound the channel gives length exactly 0 while raw keeps the
        deficit: at n = 50 raw is minus the toll.
        """
        length, raw, cut, eps, toll = general(n=50.0)

        self.assertEqual(length, 0.0, msg=f"length {length!r}")

        self.assertEqual(raw, -toll, msg=f"raw {raw!r} against -toll {-toll!r}")

        self.assertFinite((cut, eps, toll), msg=f"K, eps_coll, toll = {(cut, eps, toll)}")

    def test_homodyne_refused(self):
        """
        hom = True raises NotImplementedError: Eq. (5) assumes U(n) covariance, the heterodyne
        no-switching protocol.
        """

        self.assertFails(
            NotImplementedError,
            "hom must be false",
            lambda: general(hom=True),
            msg="hom = True",
        )

    def test_thin_test_sample_refused(self):
        """
        k_test at or below 2 ln(8/eps_target) = 50.2, where no finite cutoff exists, is refused
        naming that bound.
        """
        for bad in (1.0, 40.0, 50.0):
            self.assertFails(
                ValueError,
                "k_test must exceed",
                lambda x=bad: general(k=x),
                msg=f"k_test = {bad}",
            )

    def test_thin_sample_bound_insufficient(self):
        """
        k_test = 51 clears 2 ln(8/eps_target) but not the bound near 220 at the settled eps
        ~1e-47 and is refused, while k_test = 1e3 runs.
        """

        self.assertFails(
            ValueError,
            "no collective budget certifies",
            lambda: general(k=51.0),
            msg="k_test = 51",
        )

        self.assertGreater(general(k=1e3)[0], 0.0, msg="length at k_test = 1e3")

    def test_threshold_below_state_refused(self):
        """
        d_a or d_b below the state's mean photon number, which understates K and lengthens the
        key, raises naming that value.
        """
        self.assertFails(
            ValueError,
            "d_a must be at least the honest 2.5",
            lambda: general(da=2.4999),
            msg="d_a = 2.4999",
        )

        self.assertFails(
            ValueError,
            "d_b must be at least the honest 2.086",
            lambda: general(db=2.0),
            msg="d_b = 2.0",
        )

    def test_general_domain(self):
        """
        cv_general refuses eps_target outside (0, 1), non-positive k_test, d_a and d_b, and a
        bad argument shared with cv_finite.
        """
        for bad in (0.0, 1.0, -1.0):
            self.assertFails(
                ValueError,
                "eps_target must be",
                lambda x=bad: general(eps=x),
                msg=f"eps_target = {bad}",
            )

        for name, kw in (("k_test", "k"), ("d_a", "da"), ("d_b", "db")):
            self.assertFails(
                ValueError,
                f"{name} must be",
                lambda a=kw: general(**{a: -1.0}),
                msg=f"{name} = -1",
            )
        self.assertFails(ValueError, "t must be", lambda: general(t=-1.0), msg="T = -1")


if __name__ == "__main__":
    rc = Exam(
        "KeyrateQuantile",
        "z_pe: two-sided normal quantiles, branch seam, tails and guards",
        "keyrate_quantile.md",
    ).run(load(Quantile))
    rc |= Exam(
        "KeyrateMonotone",
        "Which way each hardware parameter moves the asymptotic key rate",
        "keyrate_monotone.md",
    ).run(load(Monotone))
    rc |= Exam(
        "KeyrateConditioning",
        "Cancellation-sensitive steps against a 60-digit evaluation",
        "keyrate_conditioning.md",
    ).run(load(Conditioning))
    rc |= Exam(
        "KeyrateGuards",
        "Degenerate limits stay finite and out-of-domain inputs raise",
        "keyrate_guards.md",
    ).run(load(Guards))
    rc |= Exam(
        "KeyrateFinite",
        "Finite-size worst-case bounds, entropy penalty and clamping",
        "keyrate_finite.md",
    ).run(load(Finite))
    rc |= Exam(
        "KeyrateGeneral",
        "The Gaussian de Finetti lift to general attacks, and what it refuses",
        "keyrate_general.md",
    ).run(load(General))
    sys.exit(rc)
