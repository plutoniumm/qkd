import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np
from numpy.polynomial import laguerre

from kit.checks import Guarded
from qkd import gaussian as g
from qkd._core import FockState

GRID = np.linspace(-6.5, 6.5, 201)
STEP = float(GRID[1] - GRID[0])
MID = len(GRID) // 2
FINE = np.linspace(-6.0, 6.0, 301)
WIDE = np.linspace(-8.0, 8.0, 201)
WSTEP = float(WIDE[1] - WIDE[0])
ROOT = float(np.sqrt(np.pi))


def wig(st, axis=GRID):
    """
    Wigner grid as a (len(ps), len(xs)) array, matching the Gaussian layer.
    """
    flat = st.wigner(list(axis), list(axis))

    return np.array(flat, dtype=np.float64).reshape(len(axis), len(axis))


def hus(st, axis=GRID):
    flat = st.husimi(list(axis), list(axis))

    return np.array(flat, dtype=np.float64).reshape(len(axis), len(axis))


def neg(st, axis=GRID):
    return st.negativity(list(axis), list(axis))


def cat_form(x0, odd, axis=GRID):
    """
    Closed-form Wigner of the cat (|beta> +- |-beta>) with beta = x0/sqrt(2).
    """
    xs, ps = np.meshgrid(axis, axis)
    sign = -1.0 if odd else 1.0
    lobes = np.exp(-((xs - x0) ** 2) - ps**2) + np.exp(-((xs + x0) ** 2) - ps**2)
    fringe = 2.0 * np.exp(-(xs**2) - ps**2) * np.cos(2.0 * x0 * ps)

    return (lobes + sign * fringe) / np.pi / (2.0 * (1.0 + sign * np.exp(-(x0**2))))


def mixture(x0, cutoff=30):
    """
    Equal classical mixture of the two coherent lobes a cat superposes.
    """
    hot = FockState.coherent(x0, 0.0, cutoff)
    cold = FockState.coherent(-x0, 0.0, cutoff)
    (hr, hi), (cr, ci) = hot.matrix(), cold.matrix()
    re = [0.5 * (a + b) for a, b in zip(hr, cr)]
    im = [0.5 * (a + b) for a, b in zip(hi, ci)]

    return FockState.from_matrix(re, im)


class Basics(Guarded):
    """
    Construction in the truncated number basis.
    """

    def test_vacuum_peaks(self):
        """
        The vacuum Wigner peaks at 1/pi and its Husimi at 1/(2 pi) under dx dp, both
        integrating to 1.
        """
        vac = FockState.fock(0, 12)
        w = wig(vac)
        q = hus(vac)

        self.assertClose(w[MID, MID], 1.0 / np.pi, msg="W(0,0) != 1/pi")
        self.assertClose(q[MID, MID], 0.5 / np.pi, msg="Q(0,0) != 1/(2 pi)")
        self.assertClose(float(w.max()), 1.0 / np.pi, msg="the Wigner peak is off the origin")
        self.assertNormalized(w, STEP, STEP, msg="vacuum Wigner integral != 1")
        self.assertNormalized(q, STEP, STEP, msg="vacuum Husimi integral != 1")

    def test_fock_ladder(self):
        """
        |n> carries all its population on level n, with mean photon number n, purity 1,
        parity (-1)^n and nothing discarded.
        """
        for n in range(5):
            st = FockState.fock(n, 12)
            pops = np.array(st.populations())

            self.assertClose(pops[n], 1.0, msg=f"|{n}> population off level {n}")
            self.assertClose(st.photons(), float(n), msg=f"photons != {n}")
            self.assertClose(st.purity(), 1.0, msg="purity != 1")
            self.assertClose(st.parity(), (-1.0) ** n, msg=f"parity != {(-1) ** n}")
            self.assertClose(st.discarded(), 0.0, msg="discarded != 0")

    def test_truncation_fires(self):
        """
        An 18-photon coherent state reads truncated at cutoffs 8 and 20 and clean at 60, and
        tail() fires on a state displaced against its ceiling that discarded() calls clean.
        """
        tight = FockState.coherent(6.0, 0.0, 8)
        edge = FockState.coherent(6.0, 0.0, 20)
        roomy = FockState.coherent(6.0, 0.0, 60)

        self.assertGreater(tight.discarded(), 0.9, msg="discarded below 0.9 at 8 levels")
        self.assertGreater(edge.trunc_error(), 0.1, msg="trunc_error below 0.1 at 20 levels")

        # 1e-12 is the atol on photons() below at a measured 1.23x; observed 1.88e-13 at 60.
        self.assertLess(roomy.trunc_error(), 1e-12, msg="trunc_error above 1e-12 at 60 levels")

        # The Poisson tail past level 60 -- 1.2e-4 of it at cutoff 40, 2.3e-13 here.
        self.assertClose(roomy.photons(), 18.0, atol=1e-12, msg="photons != |beta|^2")

        # Renormalised: the trace says nothing about truncation.
        self.assertClose(float(np.array(tight.populations()).sum()), 1.0, msg="trace != 1")

        moved = FockState.fock(1, 30).displace(6.0, 0.0)
        still = FockState.fock(1, 30).displace(1.0, 0.0)

        self.assertGreater(moved.tail(3), 0.1, msg="tail(3) below 0.1")
        self.assertGreater(moved.trunc_error(), moved.discarded(), msg="tail did not dominate")
        self.assertLess(still.trunc_error(), 1e-12, msg="trunc_error above 1e-12 at x = 1")

    def test_matrix_validated(self):
        """
        from_matrix refuses non-Hermitian, non-unit-trace, negative-eigenvalue and non-square
        matrices, and the cutoff, level index, GKP squeezing and axis slots go the same way.
        """
        ok = FockState.from_matrix([0.6, 0.0, 0.0, 0.4], [0.0] * 4)
        self.assertEqual(ok.cutoff, 2, msg="a valid pair built no state")

        self.assertBad(
            "Hermitian",
            FockState.from_matrix,
            ([0.5, 0.2, -0.2, 0.5], [0.0] * 4),
            msg="non-Hermitian accepted",
        )
        self.assertBad(
            "unit trace",
            FockState.from_matrix,
            ([0.5, 0.0, 0.0, 0.9], [0.0] * 4),
            msg="trace 1.4 accepted",
        )

        # Nothing downstream catches this: a non-positive matrix's Wigner still integrates to 1.
        self.assertBad(
            "positive semidefinite",
            FockState.from_matrix,
            ([0.5, 0.9, 0.9, 0.5], [0.0] * 4),
            msg="negative eigenvalue accepted",
        )
        self.assertBad(
            "square",
            FockState.from_matrix,
            ([1.0, 0.0, 0.0], [0.0] * 3),
            msg="non-square accepted",
        )

        self.assertFails(ValueError, "cutoff", FockState.fock, 0, 0, msg="cutoff 0 accepted")
        self.assertBad("does not fit", FockState.fock, (9, 4), msg="|9> accepted under 4 levels")
        self.assertBad("delta", FockState.gkp, (0, 0.0, 40), msg="zero squeezing width accepted")
        self.assertBad("logical", FockState.gkp, (2, 0.35, 40), msg="GKP qutrit accepted")
        self.assertBad(
            "at least two points",
            FockState.fock(0, 4).wigner,
            ([0.0], [0.0, 1.0]),
            msg="one-point axis accepted",
        )
        self.assertBad(
            "must increase",
            FockState.fock(0, 4).wigner,
            ([1.0, 0.0], [0.0, 1.0]),
            msg="descending axis accepted",
        )

    def test_axis_uniform(self):
        """
        A non-uniform axis is refused on p as on x, and |1> scores 0.4262 on an even grid.
        """
        st = FockState.fock(1, 20)
        even = np.linspace(-6.0, 6.0, 241)
        ramp = np.linspace(-1.0, 1.0, 241)
        curved = np.sign(ramp) * ramp**2 * 6.0
        stalled = np.concatenate([np.full(120, -6.0), np.linspace(-6.0, 6.0, 121)])

        self.assertClose(neg(st, even), 0.4262, atol=1e-3, msg="uniform-axis volume != 0.4262")
        self.assertBad(
            "evenly spaced",
            st.negativity,
            (list(curved), list(curved)),
            msg="quadratic axis accepted",
        )
        self.assertBad(
            "evenly spaced",
            st.negativity,
            (list(stalled), list(stalled)),
            msg="stalled axis accepted",
        )
        self.assertBad(
            "evenly spaced",
            st.wigner,
            (list(even), list(curved)),
            msg="curved p axis accepted",
        )

    def test_spectrum_settles(self):
        """
        The 512-level thermal entropy is g(nbar) to 1e-6 in under 5 s, with the spectrum
        still non-negative.
        """
        big = FockState.thermal(3.0, 512)
        clock = time.perf_counter()
        got = big.entropy()
        spent = time.perf_counter() - clock

        self.assertClose(got, 4.0 * math.log2(4.0) - 3.0 * math.log2(3.0), atol=1e-6, msg="entropy != g(nbar)")
        self.assertLess(spent, 5.0, msg=f"{spent:.2f}s of sweeps")
        self.assertGreaterEqual(min(big.eigenvalues()), -1e-12, msg="negative eigenvalue")

    def test_kernel_range(self):
        """
        A coherent state at x = 28 in 512 levels peaks at 1/pi with no NaN, and the Wigner
        kernel stays finite out to radius 150.
        """
        far = FockState.coherent(28.0, 0.0, 512)
        axis = np.linspace(25.0, 31.0, 21)
        peak = np.array(far.wigner([27.9, 28.0, 28.1], [-0.1, 0.0, 0.1]))

        # 1e-8 is the atol on peak[4] below at a measured 0.325x; observed 3.95e-9 at 512.
        self.assertLess(far.discarded(), 1e-8, msg="discarded above 1e-8 at 512 levels")
        self.assertFinite(peak, msg=f"NaN in the kernel: {peak}")
        self.assertClose(peak[4], 1.0 / np.pi, atol=1e-8, msg="peak != 1/pi")
        self.assertFinite(far.negativity(list(axis), list(axis)), msg="volume is not a number")

        for r in (30.0, 60.0, 150.0):
            grid = [r, r + 0.05]
            self.assertFinite(FockState.fock(1, 400).wigner(grid, grid), msg=f"kernel blew up at radius {r}")

    def test_states_physical(self):
        """
        Every constructor returns a physical state, the pure ones at purity 1 and a lossy
        thermal state below it.
        """
        for st in (
            FockState.fock(3, 20),
            FockState.coherent(1.0, -0.5, 30),
            FockState.cat(2.5, 0.0, True, 30),
            FockState.squeezed(0.6, 40),
            FockState.gkp(1, 0.35, 60),
        ):
            self.assertTrue(st.physical(1e-9), msg="not a state")
            self.assertClose(st.purity(), 1.0, atol=1e-6, msg="purity != 1")

        mixed = FockState.thermal(0.5, 40).loss(0.7)

        self.assertTrue(mixed.physical(1e-9), msg="lossy thermal state is not a state")
        self.assertLess(mixed.purity(), 1.0, msg="thermal purity at or above 1")


class PhaseSpace(Question):
    """
    Wigner grids for non-Gaussian states against their analytic forms.
    """

    def test_fock_laguerre(self):
        """
        The Wigner function of |n> is (-1)^n L_n(2 r^2) e^{-r^2}/pi pointwise for n up to 4.
        """
        xs, ps = np.meshgrid(GRID, GRID)
        r2 = xs**2 + ps**2

        for n in range(5):
            got = wig(FockState.fock(n, 12))
            coef = np.zeros(n + 1)
            coef[n] = 1.0
            want = (-1.0) ** n * laguerre.lagval(2.0 * r2, coef) * np.exp(-r2) / np.pi

            self.gridClose(got, want, atol=1e-12, msg=f"|{n}> off the Laguerre form")

    def test_rotation_exact(self):
        """
        Rotating a cat in the number basis is a diagonal phase: it matches a directly built
        rotated cat to 1e-9 and discards nothing extra.
        """
        turned = FockState.cat(3.0, 0.0, False, 30).rotate(np.pi / 2.0)
        direct = FockState.cat(0.0, 3.0, False, 30)

        self.gridClose(wig(turned), wig(direct), atol=1e-9, msg="rotate != a built rotated cat")
        self.assertClose(
            turned.discarded(),
            FockState.cat(3.0, 0.0, False, 30).discarded(),
            msg="a diagonal phase truncated",
        )


class Shadow(Question):
    """
    Cross-layer agreement with the Gaussian engine, and Hudson's theorem as a regression.
    """

    def test_coherent_agrees(self):
        """
        Coherent, squeezed, thermal, displaced squeezed and Kraus-lossy squeezed states match the
        Gaussian layer's Wigner pointwise (loss against thermal-loss at xi = 0), squeezed to 1e-6
        and the rest to 1e-9.
        """
        cases = (
            ("coherent", FockState.coherent(1.0, 0.5, 30), g.Coherent(1.0, 0.5), 1e-9),
            ("squeezed vacuum", FockState.squeezed(0.5, 40), g.Squeezed(0.5), 1e-6),
            ("thermal", FockState.thermal(0.5, 40), g.Thermal(0.5), 1e-9),
            (
                "displaced squeezed",
                FockState.squeezed(0.5, 40).displace(1.2, -0.6),
                g.Squeezed(0.5).displace(0, x=1.2, p=-0.6),
                1e-6,
            ),
            (
                "pure loss",
                FockState.squeezed(0.7, 40).loss(0.6),
                g.Squeezed(0.7).thermal_loss(0, T=0.6, xi=0.0, ref="input"),
                1e-6,
            ),
        )
        for name, here, there, atol in cases:
            self.gridClose(wig(here), there.wigner(0, GRID, GRID), atol=atol, msg=f"layers differ on {name}")

    def test_squeezed_agrees(self):
        """
        Moments match the Gaussian layer: coherent mean (1.0, 0.5) on a bona fide I/2, squeezed
        variances e^-2r/2 (x) and e^2r/2 (p), thermal (nbar + 1/2) I with populations as
        eigenvalues and the same entropy, displacement through squeezing, unit trace after loss.
        """
        mean, cov = FockState.coherent(1.0, 0.5, 30).moments()

        self.assertClose(mean[0], 1.0, msg="<x> != 1.0")
        self.assertClose(mean[1], 0.5, msg="<p> != 0.5")
        self.assertPhysical(np.array(cov).reshape(2, 2), msg="the Gaussian shadow is not bona fide")
        self.assertClose(cov[0], 0.5, msg="var x != 1/2")

        _, cov = FockState.squeezed(0.5, 40).moments()

        # Squeezing spreads population up the ladder: 1.4e-9 of residual at cutoff 30, 7e-13 here.
        self.assertClose(cov[0], 0.5 * np.exp(-1.0), atol=1e-11, msg="var x != e^-2r/2")
        self.assertClose(cov[3], 0.5 * np.exp(1.0), atol=1e-11, msg="var p != e^2r/2")

        hot = FockState.thermal(0.5, 40)
        _, cov = hot.moments()

        self.gridClose(
            np.array(hot.eigenvalues()),
            np.sort(np.array(hot.populations())),
            atol=1e-12,
            msg="eigenvalues != populations",
        )
        self.assertClose(cov[0], 1.0, msg="var x != nbar + 1/2")
        self.assertClose(hot.entropy(), g.Thermal(0.5).entropy(), atol=1e-9, msg="entropies differ")

        mean, _ = FockState.squeezed(0.5, 40).displace(1.2, -0.6).moments()

        self.assertClose(mean[0], 1.2, atol=1e-6, msg="<x> != 1.2")
        self.assertClose(mean[1], -0.6, atol=1e-6, msg="<p> != -0.6")

        lost = FockState.squeezed(0.7, 40).loss(0.6)

        self.assertClose(float(np.array(lost.populations()).sum()), 1.0, msg="trace != 1 after loss")

    def test_hudson(self):
        """
        Seven Gaussian states inside their cutoff score zero negativity volume and zero
        non-Gaussianity to 1e-6, with no Wigner value below -1e-6.
        """
        for st in (
            FockState.fock(0, 12),
            FockState.coherent(1.0, 0.5, 30),
            FockState.squeezed(0.5, 40),
            FockState.squeezed(0.5, 40).rotate(0.7),
            FockState.squeezed(0.5, 40).displace(1.2, -0.6),
            FockState.thermal(0.5, 40),
            FockState.squeezed(0.7, 40).loss(0.6),
        ):
            self.assertLess(st.trunc_error(), 1e-7, msg=f"trunc_error = {st.trunc_error():.2e}")

            # Worst of seven: displaced squeezed at 6.82e-7, a truncated-kernel artefact.
            self.assertClose(neg(st), 0.0, atol=1e-6, msg="Gaussian negativity volume above 1e-6")
            self.assertGreaterEqual(float(wig(st).min()), -1e-6, msg="Gaussian Wigner below -1e-6")

            # Worst 8.4e-9: sees the state's truncation, not the grid's.
            self.assertClose(st.nongauss(), 0.0, atol=1e-6, msg="Gaussian non-Gaussianity above 1e-6")

    def test_hudson_truncated(self):
        """
        At cutoff 40 the r = 1.0 squeezed vacuum fails the Hudson assertions by two orders and
        r = 1.5 scores 0.7 bits of non-Gaussianity; 120 levels restore both.
        """
        tame = FockState.squeezed(0.5, 40)
        bites = FockState.squeezed(1.0, 40)
        harder = FockState.squeezed(1.5, 40)
        roomy = FockState.squeezed(1.0, 120)

        self.assertLess(abs(neg(tame)), 1e-6, msg="4.3 dB volume above 1e-6")
        self.assertGreater(neg(bites), 1e-3, msg=f"8.7 dB volume {neg(bites)}")
        self.assertLess(float(wig(bites).min()), -1e-4, msg="8.7 dB Wigner minimum above -1e-4")
        self.assertLess(bites.trunc_error(), 1e-5, msg="8.7 dB trunc_error above 1e-5")

        # trunc_error bounds the state, not the kernel.
        self.assertGreater(neg(bites) / bites.trunc_error(), 1e3, msg="volume within 1e3 of trunc_error")

        self.assertGreater(harder.nongauss(), 0.7, msg=f"13 dB non-Gaussianity {harder.nongauss()}")
        self.assertGreaterEqual(float(wig(roomy).min()), -1e-6, msg="120-level Wigner below -1e-6")
        self.assertClose(roomy.nongauss(), 0.0, atol=1e-6, msg="120-level non-Gaussianity above 1e-6")

    def test_gaussian_shadow(self):
        """
        The Gaussian shadow of |1> from moments() is Thermal(1), non-negative everywhere, while
        |1> is negative at the origin.
        """
        st = FockState.fock(1, 12)
        mean, cov = st.moments()
        shade = g.State(g.GaussianState.from_moments(mean, cov))

        self.assertPhysical(np.array(cov).reshape(2, 2), msg="the shadow is not bona fide")
        self.gridClose(
            shade.wigner(0, GRID, GRID),
            g.Thermal(1.0).wigner(0, GRID, GRID),
            atol=1e-12,
            msg="the shadow of |1> != Thermal(1)",
        )
        self.assertGreaterEqual(float(shade.wigner(0, GRID, GRID).min()), 0.0, msg="the shadow goes negative")
        self.assertLess(wig(st)[MID, MID], 0.0, msg="W(0,0) of |1> is not negative")


class Witness(Question):
    """
    Wigner negativity as the resource-theoretic witness, plus a second witness detecting
    something strictly weaker. Negativity is equivalent to generalised contextuality
    (Spekkens, PRL 101, 020401 (2008)), and Wigner-positive states under Gaussian dynamics
    with homodyne readout are efficiently simulable (Mari & Eisert, PRL 109, 230503 (2012);
    Veitch et al., NJP 15, 013037 (2013)). The |W| - 1 volume integral is Kenfack &
    Zyczkowski's indicator (J. Opt. B 6, 396 (2004)); its logarithm is the monotone of
    Albarelli et al. (PRA 98, 052350 (2018)).
    """

    def test_fock_anchor(self):
        """
        The negativity volume of |1> is 4/sqrt(e) - 2 = 0.4261226, the vacuum scores zero,
        and the volume grows over |0> to |4>.
        """
        vols = [neg(FockState.fock(n, 12), FINE) for n in range(5)]

        self.assertClose(vols[1], 4.0 * np.exp(-0.5) - 2.0, atol=1e-4, msg="|1> volume != 4/sqrt(e) - 2")
        self.assertClose(vols[0], 0.0, atol=1e-6, msg="vacuum volume != 0")
        self.assertMonotone(vols, msg="volume not rising with n")

    def test_cat_closed_form(self):
        """
        Even and odd cats match their closed-form Wigner functions pointwise, where a
        classical mixture of the same two lobes scores zero negativity but 1.55 bits of
        relative-entropy non-Gaussianity (Genoni & Paris, PRA 82, 052341 (2010)).
        """
        for odd in (False, True):
            self.gridClose(
                wig(FockState.cat(3.0, 0.0, odd, 30)),
                cat_form(3.0, odd),
                atol=1e-7,
                msg="the cat Wigner is off its closed form",
            )

        mixed = mixture(3.0)

        self.assertClose(neg(mixed), 0.0, atol=1e-6, msg="mixture volume above 1e-6")
        self.assertGreaterEqual(float(wig(mixed).min()), -1e-6, msg="mixture Wigner below -1e-6")
        self.assertGreater(mixed.nongauss(), 1.0, msg="mixture non-Gaussianity below 1 bit")
        self.assertGreater(neg(FockState.cat(3.0, 0.0, False, 30)), 0.5, msg="cat volume below 0.5")

        # atol 1e-6: exact 1/2 purity up to 7.6e-9 of overlap + cutoff-30 truncation.
        self.assertClose(mixed.purity(), 0.5, atol=1e-6, msg="mixture purity != 1/2")

    def test_gkp_lattice(self):
        """
        |0_L> peaks in x at even multiples of sqrt(pi) and |1_L> at the odd ones, both peak
        in p at every multiple, and a 9 dB GKP state is strongly Wigner negative.
        """
        zero = FockState.gkp(0, 0.35, 60)
        wzero = wig(zero, WIDE)
        wone = wig(FockState.gkp(1, 0.35, 60), WIDE)
        mx0 = wzero.sum(axis=0) * WSTEP
        mx1 = wone.sum(axis=0) * WSTEP
        mp0 = wzero.sum(axis=1) * WSTEP
        at0 = int(np.argmin(np.abs(WIDE)))
        at1 = int(np.argmin(np.abs(WIDE - ROOT)))
        at2 = int(np.argmin(np.abs(WIDE - 2.0 * ROOT)))

        self.assertGreater(mx0[at0], 0.5, msg="|0_L> x marginal at 0 below 0.5")
        self.assertGreater(mx0[at2], 0.1, msg="|0_L> x marginal at 2 sqrt(pi) below 0.1")

        # Node residual 8.8e-6 and 1.3e-5 here, 2.2e-5 worst over levels 50-100.
        self.assertLess(abs(mx0[at1]), 1e-4, msg="|0_L> node at sqrt(pi) above 1e-4")
        self.assertGreater(mx1[at1], 0.5, msg="|1_L> x marginal at sqrt(pi) below 0.5")
        self.assertLess(abs(mx1[at0]), 1e-4, msg="|1_L> node at 0 above 1e-4")
        self.assertGreater(mp0[at1], 0.1, msg="p marginal at sqrt(pi) below 0.1")
        self.assertGreater(neg(zero, WIDE), 0.5, msg="9 dB GKP volume below 0.5")

        # Bound = node tolerance; 5.97e-7 at 60 is a lattice-parity dip, 59 reads 2.45e-6.
        self.assertLess(zero.trunc_error(), 1e-4, msg="trunc_error above 1e-4 at 60 levels")

    def test_parity_witness(self):
        """
        The parity equals pi W(0,0) to 1e-9, negative for an odd cat and positive for a coherent
        state.
        """
        for st in (
            FockState.fock(1, 12),
            FockState.cat(3.0, 0.0, True, 30),
            FockState.coherent(1.0, 0.5, 30),
        ):
            self.assertClose(st.parity(), np.pi * float(wig(st)[MID, MID]), atol=1e-9, msg="parity != pi W(0,0)")

        self.assertLess(FockState.cat(3.0, 0.0, True, 30).parity(), 0.0, msg="odd cat parity not negative")
        self.assertGreater(FockState.coherent(1.0, 0.5, 30).parity(), 0.0, msg="coherent parity not positive")

    def test_loss_kills(self):
        """
        An odd cat's negativity volume falls monotonically under pure loss, from 0.5577
        intact to 0.0108 at eta = 0.6 and zero by eta = 0.4, with the state still physical.
        """
        base = FockState.cat(2.5, 0.0, True, 30)
        vols = [base.loss(eta).negativity(list(GRID), list(GRID)) for eta in (1.0, 0.8, 0.6)]

        self.assertMonotone(vols, rising=False, msg="volume not falling with loss")

        # alpha = 2.5 fits under 24 levels: GRID quadrature alone sets these.
        self.assertClose(vols[0], 0.5577, atol=1e-4, msg="intact volume != 0.5577")
        self.assertClose(vols[2], 0.0108, atol=1e-4, msg="eta = 0.6 volume != 0.0108")

        # atol 1e-9: sum of |W| - W, zero to rounding once W >= 0; 1.2e-12 here.
        self.assertClose(neg(base.loss(0.4)), 0.0, atol=1e-9, msg="eta = 0.4 volume != 0")
        self.assertTrue(base.loss(0.6).physical(1e-9), msg="the lossy cat is not a state")


if __name__ == "__main__":
    rc = Exam(
        "FockBasics",
        "Truncated number-basis construction, validation and truncation",
        "fock_basics.md",
    ).run(load(Basics))
    rc |= Exam(
        "FockPhaseSpace",
        "Wigner and Husimi grids for non-Gaussian states against analytic forms",
        "fock_phase_space.md",
    ).run(load(PhaseSpace))
    rc |= Exam(
        "FockShadow",
        "Cross-layer agreement with the Gaussian engine, and Hudson as a regression",
        "fock_shadow.md",
    ).run(load(Shadow))
    rc |= Exam(
        "FockWitness",
        "Wigner negativity and non-Gaussianity: what each witness detects",
        "fock_witness.md",
    ).run(load(Witness))
    sys.exit(rc)
