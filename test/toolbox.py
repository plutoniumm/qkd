import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load, omega

import numpy as np

from kit.checks import Guarded
from qkd import gaussian as g

GRID = np.linspace(-5.0, 5.0, 201)
STEP = float(GRID[1] - GRID[0])


class PhaseSpace(Question):
    def test_wigner_marginal(self):
        """
        The vacuum Wigner integrated over p has mean 0 and variance 1/2.
        """
        w = g.Vacuum(1).wigner(0, GRID, GRID)
        px = w.sum(axis=0) * STEP

        mean = float(np.sum(GRID * px) * STEP)
        var = float(np.sum((GRID - mean) ** 2 * px) * STEP)

        self.assertClose(mean, 0.0, atol=1e-6, msg="marginal mean")
        self.assertClose(var, 0.5, atol=1e-6, msg="marginal variance")

    def test_husimi_peak(self):
        """
        The vacuum Husimi peaks at 1/(2*pi), half the Wigner peak 1/pi: the d^2 alpha
        convention's 1/pi times the dx dp Jacobian 1/2.
        """
        q = g.Vacuum(1).husimi(0, GRID, GRID)
        w = g.Vacuum(1).wigner(0, GRID, GRID)
        peak = float(q[100, 100])

        self.assertClose(peak, 1.0 / (2.0 * np.pi), msg="Q(0,0)")
        self.assertClose(w[100, 100], 1.0 / np.pi, msg="W(0,0)")
        self.assertClose(peak, float(q.max()), msg=f"Q max {float(q.max())} vs origin {peak}")

    def test_negativity(self):
        """
        Hudson's theorem: a squeezed displaced state's Wigner is non-negative, volume 0.
        """
        st = g.Squeezed(0.7).displace(0, x=0.9, p=0.4)
        w = st.wigner(0, GRID, GRID)

        self.assertEqual(st.negativity(), 0.0, msg="Gaussian negativity")
        self.assertGreaterEqual(float(w.min()), 0.0, msg=f"Wigner minimum {float(w.min())}")


class Evolution(Question):
    def test_rotation(self):
        """
        evolve(I, t) matches rotate(0, -t): H = (x^2 + p^2)/2 turns clockwise where rotate
        turns counterclockwise.
        """
        st = g.Squeezed(0.8).displace(0, x=0.6, p=-0.2)

        out = st.evolve(np.eye(2), 0.7)
        want = st.rotate(0, -0.7)

        self.assertTrue(np.allclose(out.cov, want.cov), msg="cov vs rotate(0, -t)")
        self.assertTrue(np.allclose(out.mean, want.mean), msg="mean vs rotate(0, -t)")

    def test_squeezer(self):
        """
        G = [[0, 1], [1, 0]] evolves the vacuum symplectically to variances e^2t/2, e^-2t/2.
        """
        t = 0.3
        G = np.array([[0.0, 1.0], [1.0, 0.0]])

        out = g.Vacuum(1).evolve(G, t)
        s = g._expm(g._omega(1) @ G * t)

        self.assertSymplectic(s, msg="expm(Omega G t)")
        self.assertPhysical(out.cov, msg="evolved state not bona fide")
        self.assertClose(out.cov[0, 0], 0.5 * np.exp(2 * t), msg="var_x")
        self.assertClose(out.cov[1, 1], 0.5 * np.exp(-2 * t), msg="var_p")

    def test_purity_kept(self):
        """
        A thermal state keeps purity 0.5 under a generic quadratic Hamiltonian.
        """
        st = g.Thermal(nbar=0.5)
        G = np.array([[1.0, 0.3], [0.3, 0.7]])

        out = st.evolve(G, 0.9)

        self.assertClose(out.purity(), st.purity(), msg=f"purity {out.purity()} vs {st.purity()}")
        self.assertClose(out.purity(), 0.5, msg="thermal purity")

    def test_validation(self):
        """
        evolve rejects a non-symmetric G and a wrong-shaped G with ValueError.
        """
        st = g.Vacuum(1)
        bad = np.array([[0.0, 1.0], [0.0, 0.0]])

        with self.assertRaises(ValueError):
            st.evolve(bad, 0.5)

        with self.assertRaises(ValueError):
            st.evolve(np.eye(4), 0.5)

    def test_epr_coupled(self):
        """
        x1x2 + p1p2 on Epr(0.8) stays bona fide and pure under a symplectic map.
        """
        G = np.zeros((4, 4))
        G[0, 2] = G[2, 0] = 1.0
        G[1, 3] = G[3, 1] = 1.0

        out = g.Epr(0.8).evolve(G, 0.4)
        s = g._expm(g._omega(2) @ G * 0.4)

        self.assertSymplectic(s, msg="two-mode S")
        self.assertPhysical(out.cov, msg="evolved EPR not bona fide")
        self.assertClose(out.purity(), 1.0, msg=f"EPR purity {out.purity()}")


class Info(Question):
    def test_entropy_thermal(self):
        """
        Thermal(0.5) entropy is 1.5*log2(1.5) - 0.5*log2(0.5) = 1.37744 bits.
        """
        want = 1.5 * np.log2(1.5) - 0.5 * np.log2(0.5)

        self.assertClose(g.Thermal(nbar=0.5).entropy(), want, msg="thermal entropy")

    def test_epr_entropy(self):
        """
        Epr(0.8) has entropy 0 while its half carries Thermal(sinh^2 r)'s entropy.
        """
        st = g.Epr(0.8)
        half = st.keep([0]).entropy()
        want = g.Thermal(nbar=float(np.sinh(0.8) ** 2)).entropy()

        self.assertClose(st.entropy(), 0.0, msg="EPR global entropy")
        self.assertClose(half, want, msg="reduced entropy")
        self.assertGreater(half, 0.0, msg=f"entanglement entropy {half}")

    def test_spectrum_ceiling(self):
        """
        Epr(r).spectrum() gives nu = 0.5 twice through r = 8 and first refuses at r = 8.17,
        ahead of the constructor's 8.45, the guard scaling with max|V| not a fixed 1e-9.
        """
        for r in (4.44, 5.0, 6.0, 7.0, 8.0):
            nus = g.Epr(r).spectrum()
            self.assertTrue(np.allclose(nus, 0.5, atol=1e-2), msg=f"nu at r = {r}: {nus}")

        self.assertTrue(
            np.allclose(g.Epr(5.0).spectrum(), 0.5, atol=1e-8),
            msg="Epr(5) nu at 1e-8",
        )

        # Not vacuous: Epr(5) is past the old absolute 1e-9.
        eigs = np.linalg.eigvals(1j * omega(2) @ g.Epr(5.0).cov)
        self.assertGreater(float(np.max(np.abs(eigs.imag))), 1e-9, msg="Epr(5) residue")

        # 99.0 not None: a guard that never fires must read as a boundary far out.
        guard = 99.0
        built = 99.0

        for step in range(800, 930):
            try:
                st = g.Epr(step / 100.0)
            except ValueError:
                built = min(built, step / 100.0)
                continue
            try:
                st.spectrum()
            except ValueError:
                guard = min(guard, step / 100.0)

        self.assertClose(guard, 8.17, atol=0.5, msg=f"first refusal at r = {guard}")
        self.assertLessEqual(guard, built, msg=f"guard {guard} after the constructor {built}")

    def test_keep_drop(self):
        """
        keep([1]) of Epr(0.8) is the thermal state cosh(1.6)*I/2, and drop([0]) agrees.
        """
        st = g.Epr(0.8)
        kept = st.keep([1])
        want = 0.5 * np.cosh(1.6) * np.eye(2)

        self.assertTrue(np.allclose(kept.cov, want), msg=f"kept cov {kept.cov}")
        self.assertTrue(np.allclose(st.drop([0]).cov, kept.cov), msg="drop([0]) vs keep([1])")
        self.assertPhysical(kept.cov, msg="partial trace not bona fide")
        self.assertTrue(np.allclose(kept.mean, 0.0), msg=f"kept mean {kept.mean}")

    def test_keep_order(self):
        """
        keep([1, 0]) puts mode 1 first.
        """
        st = g.Vacuum(2).displace(1, x=0.3, p=-0.1)
        out = st.keep([1, 0])

        self.assertClose(out.mean[0], 0.3, msg="mode 1 x")
        self.assertClose(out.mean[1], -0.1, msg="mode 1 p")
        self.assertClose(out.mean[2], 0.0, msg="mode 0 x")

    def test_keep_checks(self):
        """
        keep rejects out-of-range, duplicate and empty mode lists with ValueError.
        """
        st = g.Vacuum(2)

        with self.assertRaises(ValueError):
            st.keep([2])

        with self.assertRaises(ValueError):
            st.keep([0, 0])

        with self.assertRaises(ValueError):
            st.keep([])


class Derived(Guarded):
    def test_methods_survive(self):
        """
        Displace, squeeze, rotate, beamsplitter, thermal loss, conditioning and both
        detectors all run on a state from keep, drop or evolve.
        """
        base = g.Epr(0.8).bs(0, 1, t=0.7)

        for st in (base.keep([1, 0]), base.drop([]), base.evolve(np.eye(4), 0.3)):
            out = (
                st.displace(0, x=0.2, p=-0.1)
                .squeeze(1, r=0.3)
                .rotate(0, theta=0.4)
                .bs(0, 1, t=0.6)
                .thermal_loss(0, T=0.5, xi=0.01, ref="input")
            )

            self.assertPhysical(out.cov, msg="chained cov")
            self.assertEqual(out.heterodyne(0, shots=4, seed=1).shape, (4, 2), msg="heterodyne shape")
            self.assertEqual(out.homodyne(0, angle=0.3, shots=6, seed=1).shape, (6,), msg="homodyne shape")
            self.assertEqual(out.condition(1, 0.0, 0.3).n_modes, 1, msg="conditioned mode count")

    def test_matches_direct(self):
        """
        Epr(0.8).keep([0]) and Thermal(sinh^2 0.8) give the same covariance, mean, purity
        and homodyne samples down one thermal-loss channel.
        """
        nbar = float(np.sinh(0.8) ** 2)

        got = g.Epr(0.8).keep([0]).thermal_loss(0, T=0.4, xi=0.03, ref="input")
        want = g.Thermal(nbar).thermal_loss(0, T=0.4, xi=0.03, ref="input")

        self.assertTrue(np.allclose(got.cov, want.cov), msg="covariance")
        self.assertTrue(np.allclose(got.mean, want.mean), msg="mean")
        self.assertClose(got.purity(), want.purity(), msg="purity")
        self.assertTrue(
            np.array_equal(
                got.homodyne(0, angle=0.0, shots=64, seed=7),
                want.homodyne(0, angle=0.0, shots=64, seed=7),
            ),
            msg="homodyne samples at seed 7",
        )

    def test_moments_validated(self):
        """
        from_moments refuses diag(0.1, 0.1) -- quadrature product 0.01 below 1/4, caught
        only by V + i*Omega/2 >= 0 -- plus a wrong shape and a non-symmetric covariance.
        """
        core = g.GaussianState
        ok = core.from_moments([0.0, 0.0], [0.5, 0.0, 0.0, 0.5])

        self.assertEqual(ok.n_modes, 1, msg=f"n_modes {ok.n_modes}")
        self.assertBad(
            "bona fide",
            core.from_moments,
            ([0.0, 0.0], [0.1, 0.0, 0.0, 0.1]),
            msg="sub-vacuum covariance accepted",
        )
        self.assertBad(
            "symmetric",
            core.from_moments,
            ([0.0, 0.0], [0.5, 0.2, -0.2, 0.5]),
            msg="non-symmetric covariance accepted",
        )
        self.assertBad(
            "cov must be",
            core.from_moments,
            ([0.0, 0.0], [0.5, 0.0, 0.0]),
            msg="mismatched cov shape accepted",
        )
        self.assertBad(
            "mean must have",
            core.from_moments,
            ([0.0], [0.5]),
            msg="odd mean length accepted",
        )


if __name__ == "__main__":
    rc = Exam(
        "ToolboxPhaseSpace",
        "Wigner/Husimi grids: normalization, marginals, peaks, Hudson's theorem",
        "toolbox_phase_space.md",
    ).run(load(PhaseSpace))
    rc |= Exam(
        "ToolboxEvolution",
        "Quadratic-Hamiltonian evolution: symplectic, purity-preserving, validated",
        "toolbox_evolution.md",
    ).run(load(Evolution))
    rc |= Exam(
        "ToolboxInfo",
        "Purity, entropy, symplectic spectrum and partial trace closed forms",
        "toolbox_info.md",
    ).run(load(Info))
    rc |= Exam(
        "ToolboxDerived",
        "keep/drop/evolve return full native states, validated as bona fide",
        "toolbox_derived.md",
    ).run(load(Derived))
    sys.exit(rc)
