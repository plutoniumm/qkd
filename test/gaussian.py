import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

from kit.checks import Guarded
from qkd import _core
from qkd import gaussian as g

SHOTS = 124_000


def moments(vx, vp):
    """
    A single-mode diagonal covariance handed straight to the native gate.
    """

    return _core.GaussianState.from_moments([0.0, 0.0], [vx, 0.0, 0.0, vp])


def admits(vx, vp):
    """
    Whether the core accepts diag(vx, vp) as a bona fide covariance matrix.
    """

    try:
        moments(vx, vp)
    except ValueError:
        return False

    return True


class BonaFide(Question):
    """
    V + i*Omega/2 >= 0, gated at the commutator scale and not at the state's energy.
    """

    def test_slack_is_absolute(self):
        """
        Four diagonals a tolerance relative to max|V_ii| admitted -- products 0.150 at
        V_pp = 1e4, 4.9e-97 and 1e-300 at 1e6 -- are refused by the absolute gate.
        """
        for vx, vp in ((1.5e-5, 1e4), (4.9e-97, 1e6), (1e-300, 1e6), (0.1, 1.0)):
            self.assertFalse(admits(vx, vp), msg=f"diag({vx:g}, {vp:g}) has product {vx * vp:g}")

    def test_gate_sits_at_heisenberg(self):
        """
        A diagonal single-mode covariance is admitted at V_xx*V_pp = 0.26 and refused at
        0.24, at V_pp from 1 to 1e6.
        """
        for vp in (1e0, 1e2, 1e4, 1e6):
            self.assertTrue(admits(1.04 * 0.25 / vp, vp), msg=f"product 0.26 at V_pp = {vp:g}")
            self.assertFalse(admits(0.96 * 0.25 / vp, vp), msg=f"product 0.24 at V_pp = {vp:g}")

    def test_pure_states_admitted(self):
        """
        A pure squeezed vacuum is bona fide and its exact boundary diagonal admitted, at r up
        to 16 (139 dB, against the ~15 dB made), where the smallest eigenvalue is an exact
        zero the Cholesky resolves only to about eps*tr(V).
        """
        for r in (0.0, 1.0, 4.0, 8.0, 16.0):
            made = g.Squeezed(r)
            self.assertPhysical(made.cov, msg=f"Squeezed({r}) not bona fide")
            self.assertTrue(
                admits(np.exp(-2.0 * r) / 2.0, np.exp(2.0 * r) / 2.0),
                msg=f"boundary at r = {r} refused",
            )

    def test_bright_thermal_admitted(self):
        """
        A thermal state of mean photon number up to 1e12 is admitted.
        """
        for nbar in (1e0, 1e4, 1e8, 1e12):
            self.assertTrue(admits(nbar + 0.5, nbar + 0.5), msg=f"thermal nbar = {nbar:g}")


class States(Guarded):
    """
    Constructors are bona fide: hbar = 1, xpxp ordering, vacuum variance 1/2.
    """

    def test_vacuum(self):
        """
        Vacuum(2) has covariance I/2, zero mean, and saturates dx^2 dp^2 = 1/4 on every mode.
        """
        st = g.Vacuum(2)
        self.assertTrue(np.allclose(st.cov, 0.5 * np.eye(4)), msg="vacuum covariance != I/2")
        self.assertTrue(np.allclose(st.mean, 0.0), msg="vacuum mean != 0")
        self.assertPhysical(st.cov, msg="vacuum not bona fide")

        for i in range(2):
            vx, vp = st.cov[2 * i, 2 * i], st.cov[2 * i + 1, 2 * i + 1]
            self.assertClose(vx * vp, 0.25, msg=f"mode {i} product != 1/4")

    def test_epr_ceiling(self):
        """
        epr(r) is bona fide and re-admitted by from_moments to r = 8, and raises at both
        signs from r = 12: the ceiling is representability near r = 9.2 (80 dB), not the
        r = 356 where the covariance overflows.
        """
        for r in (0.0, 1.0, 4.0, 8.0):
            st = _core.GaussianState.epr(r)
            cov = st.cov()
            self.assertFinite(cov, msg=f"epr({r}) covariance not finite")
            self.assertEqual(
                _core.GaussianState.from_moments(st.mean(), cov).n_modes,
                2,
                msg=f"epr({r}) not re-admitted",
            )

        for r in (0.0, 1.0, 4.0):
            self.assertPhysical(
                np.array(_core.GaussianState.epr(r).cov()).reshape(4, 4),
                msg=f"epr({r}) not bona fide",
            )

        for r in (12.0, 100.0, 356.0, 400.0, 1e300):
            self.assertFails(
                ValueError,
                "past the two-mode squeezing",
                _core.GaussianState.epr,
                r,
                msg=f"epr({r}) did not raise",
            )
            self.assertFails(
                ValueError,
                "past the two-mode squeezing",
                _core.GaussianState.epr,
                -r,
                msg=f"epr({-r}) did not raise",
            )

    def test_squeeze_ceiling(self):
        """
        squeeze(r) is finite to r = 355 and raises at 356, 400, 710 and 1e300; the guard is
        on the RESULT, not on r.
        """
        vac = _core.GaussianState.vacuum(1)
        for r in (0.0, 16.0, 100.0, 355.0):
            self.assertFinite(vac.squeeze(0, r).cov(), msg=f"squeeze({r}) covariance not finite")

        for r in (356.0, 400.0, 710.0, 1e300):
            self.assertBad(
                "non-finite state",
                vac.squeeze,
                (0, r),
                msg=f"squeeze({r}) did not raise",
            )

    def test_broken_state_samples(self):
        """
        epr(400).homodyne raises at the constructor, while epr(4) draws a marginal of
        non-zero width.
        """
        self.assertFails(
            ValueError,
            "past the two-mode squeezing",
            lambda: _core.GaussianState.epr(400.0).homodyne(0, 0.0, 4, 7),
            msg="epr(400).homodyne did not raise",
        )

        draws = _core.GaussianState.epr(4.0).homodyne(0, 0.0, 64, 7)
        self.assertFinite(draws[:8], msg="epr(4) draws not finite")
        self.assertGreater(len(set(draws)), 1, msg="marginal has no width")


class Operations(Question):
    """
    Symplectic operations preserve physicality and act as the algebra says.
    """

    def test_squeeze_inverse(self):
        """
        Squeezing by r then by -r returns the original vacuum covariance.
        """
        st = g.Vacuum(1).squeeze(0, r=0.8).squeeze(0, r=-0.8)
        self.assertTrue(
            np.allclose(st.cov, 0.5 * np.eye(2)),
            msg="squeeze round trip != identity",
        )
        self.assertPhysical(st.cov, msg="round trip not bona fide")

    def test_rotate_swaps(self):
        """
        Rotating a squeezed state by pi/2 exchanges var_x and var_p.
        """
        st = g.Squeezed(0.8).rotate(0, theta=np.pi / 2)
        self.assertClose(st.cov[0, 0], 0.5 * np.exp(1.6), msg="var_x != e^+2r/2")
        self.assertClose(st.cov[1, 1], 0.5 * np.exp(-1.6), msg="var_p != e^-2r/2")
        self.assertPhysical(st.cov, msg="rotated state not bona fide")

    def test_bs_identity(self):
        """
        A beamsplitter with t = 1.0 is the identity on covariance and mean.
        """
        st = g.Vacuum(2).squeeze(0, r=0.5).displace(0, x=0.3, p=-0.2)
        out = st.bs(0, 1, t=1.0)
        self.assertTrue(np.allclose(out.cov, st.cov), msg="t=1 moved the covariance")
        self.assertTrue(np.allclose(out.mean, st.mean), msg="t=1 moved the mean")

    def test_bs_splits(self):
        """
        A t = 0.5 beamsplitter on coherent + vacuum sends sqrt(t) each way and keeps vacuum
        covariance.
        """
        st = g.Vacuum(2).displace(0, x=1.0, p=0.0).bs(0, 1, t=0.5)
        amp0 = float(np.linalg.norm(st.mean[0:2]))
        amp1 = float(np.linalg.norm(st.mean[2:4]))
        self.assertClose(amp0, np.sqrt(0.5), msg="transmitted amplitude != sqrt(t)")
        self.assertClose(amp1, np.sqrt(0.5), msg="reflected amplitude != sqrt(1-t)")
        self.assertTrue(
            np.allclose(st.cov, 0.5 * np.eye(4)),
            msg="covariance != I/2",
        )

    def test_displace_mean_only(self):
        """
        Displacement moves the mean by (x, p) and never touches the covariance.
        """
        st = g.Squeezed(0.8)
        out = st.displace(0, x=1.0, p=0.5)
        self.assertTrue(np.allclose(out.cov, st.cov), msg="displace moved the covariance")
        self.assertClose(out.mean[0], 1.0, msg="mean x != 1.0")
        self.assertClose(out.mean[1], 0.5, msg="mean p != 0.5")
        self.assertPhysical(out.cov, msg="displaced state not bona fide")


class Loss(Question):
    """
    Thermal-loss channel: fixed points, sqrt(T) scaling, and the explicit xi plane.
    """

    def test_identity_channel(self):
        """
        thermal_loss with T = 1 and xi = 0 is the identity channel.
        """
        st = g.Squeezed(0.6).displace(0, x=0.7, p=-0.3)
        out = st.thermal_loss(0, T=1.0, xi=0.0, ref="input")
        self.assertTrue(np.allclose(out.cov, st.cov), msg="identity channel moved the covariance")
        self.assertTrue(np.allclose(out.mean, st.mean), msg="identity channel moved the mean")

    def test_vacuum_fixed_point(self):
        """
        The vacuum is a fixed point of pure loss at any transmittance.
        """
        for t in (0.9, 0.5, 0.1):
            out = g.Vacuum(1).thermal_loss(0, T=t, xi=0.0, ref="input")
            self.assertTrue(
                np.allclose(out.cov, 0.5 * np.eye(2)),
                msg=f"vacuum covariance moved at T={t}",
            )
            self.assertTrue(np.allclose(out.mean, 0.0), msg=f"vacuum mean moved at T={t}")

    def test_coherent_scaling(self):
        """
        Pure loss scales a coherent amplitude by sqrt(T): T = 0.36 sends (1.2, -0.4) to
        (0.72, -0.24).
        """
        out = g.Coherent(1.2, -0.4).thermal_loss(0, T=0.36, xi=0.0, ref="input")
        self.assertClose(out.mean[0], 0.72, msg="mean x != 0.72")
        self.assertClose(out.mean[1], -0.24, msg="mean p != -0.24")

    def test_loss_toward_vacuum(self):
        """
        Pure loss pulls both variances toward the vacuum 1/2, from either side.
        """
        st = g.Squeezed(0.8)
        out = st.thermal_loss(0, T=0.5, xi=0.0, ref="input")
        vx, vp = out.cov[0, 0], out.cov[1, 1]
        self.assertGreater(vx, st.cov[0, 0], msg="squeezed variance did not rise")
        self.assertLess(vx, 0.5, msg="squeezed variance above 1/2")
        self.assertLess(vp, st.cov[1, 1], msg="antisqueezed variance did not fall")
        self.assertGreater(vp, 0.5, msg="antisqueezed variance below 1/2")
        self.assertPhysical(out.cov, msg="lossy state not bona fide")

    def test_excess_noise_physical(self):
        """
        Thermal loss at positive xi yields a bona fide state and physical() agrees.
        """
        out = g.Squeezed(0.8).thermal_loss(0, T=0.3, xi=0.05, ref="input")
        self.assertPhysical(out.cov, msg="noisy output not bona fide")
        self.assertTrue(out.physical(), msg="physical() disagrees")

    def test_reference_plane(self):
        """
        xi = 0.02 at ref="input" gives the same covariance as xi = 0.008 at ref="output" for
        T = 0.4.
        """
        alice = g.Squeezed(0.8).thermal_loss(0, T=0.4, xi=0.02, ref="input")
        bob = g.Squeezed(0.8).thermal_loss(0, T=0.4, xi=0.4 * 0.02, ref="output")
        self.assertTrue(
            np.allclose(alice.cov, bob.cov),
            msg="xi_out = T*xi_in differs",
        )

    def test_ref_validation(self):
        """
        An unknown ref raises ValueError and an omitted one TypeError; the plane is never
        defaulted.
        """
        st = g.Vacuum(1)

        with self.assertRaises(ValueError):
            st.thermal_loss(0, T=0.5, xi=0.01, ref="sideways")

        with self.assertRaises(TypeError):
            st.thermal_loss(0, T=0.5, xi=0.01)


class Sampling(Question):
    """
    Homodyne and heterodyne sampling reproduce the Gaussian moments, seed-deterministic.
    """

    # atol 0.02 at SHOTS = 124000 is 5 sigma on the binding case, the heterodyne variance
    # (se 4.02e-3); the other moments run 7-27 sigma.

    def test_homodyne_coherent(self):
        """
        Homodyne x on Coherent(1.0, 0.0) has sample mean 1.0 and variance 1/2.
        """
        x = g.Coherent(1.0, 0.0).homodyne(mode=0, angle=0.0, shots=SHOTS, seed=3)

        self.momentsClose(x, mean=1.0, var=0.5, atol=0.02, msg="x moments off")

    def test_homodyne_angle(self):
        """
        Homodyne at angle pi/2 measures p: mean 0.0 for Coherent(1.0, 0.0).
        """
        st = g.Coherent(1.0, 0.0)
        p = st.homodyne(mode=0, angle=np.pi / 2, shots=SHOTS, seed=3)

        self.momentsClose(p, mean=0.0, var=0.5, atol=0.02, msg="p moments off")

    def test_homodyne_squeezed(self):
        """
        Homodyne x on Squeezed(0.5) has variance e^-1/2 = 0.184, below vacuum.
        """
        x = g.Squeezed(0.5).homodyne(mode=0, angle=0.0, shots=SHOTS, seed=3)

        self.momentsClose(x, mean=0.0, var=0.5 * np.exp(-1.0), atol=0.02, msg="squeezed moments")

    def test_heterodyne_vacuum(self):
        """
        Heterodyne on vacuum returns (shots, 2) with per-quadrature variance 1.0, the vacuum
        1/2 plus the 1/2 Husimi penalty.
        """
        xp = g.Vacuum(1).heterodyne(mode=0, shots=SHOTS, seed=3)
        self.assertEqual(xp.shape, (SHOTS, 2), msg="heterodyne shape")

        self.momentsClose(xp[:, 0], mean=0.0, var=1.0, atol=0.02, msg="x column")

        self.momentsClose(xp[:, 1], mean=0.0, var=1.0, atol=0.02, msg="p column")

    def test_seeding(self):
        """
        The same seed reproduces identical samples and a different seed does not.
        """
        st = g.Coherent(0.3, 0.1)
        a = st.homodyne(mode=0, angle=0.0, shots=1000, seed=42)
        b = st.homodyne(mode=0, angle=0.0, shots=1000, seed=42)
        c = st.homodyne(mode=0, angle=0.0, shots=1000, seed=43)
        self.assertTrue(np.array_equal(a, b), msg="same seed differed")
        self.assertFalse(np.array_equal(a, c), msg="different seeds matched")


class Conditioning(Guarded):
    """
    Homodyne conditioning by Schur complement.
    """

    def test_epr_steering(self):
        """
        Conditioning Epr(1.0) on x of mode 1 leaves mode 0 with x variance
        1/(2*cosh(2)) = 0.133, below the vacuum 1/2.
        """
        cond = g.Epr(1.0).condition(mode=1, angle=0.0, outcome=0.0)
        want = 1.0 / (2.0 * np.cosh(2.0))
        self.assertClose(cond.cov[0, 0], want, msg="conditional var_x != 1/(2 cosh 2)")

    def test_conditional_physical(self):
        """
        The conditional state is still a bona fide Gaussian state.
        """
        cond = g.Epr(1.0).condition(mode=1, angle=0.0, outcome=0.0)
        self.assertPhysical(cond.cov, msg="conditional state not bona fide")
        self.assertTrue(cond.physical(), msg="physical() disagrees")

    def test_mode_removed(self):
        """
        Conditioning removes the measured mode: two modes drop to one.
        """
        cond = g.Epr(1.0).condition(mode=1, angle=0.0, outcome=0.0)
        self.assertEqual(cond.cov.shape, (2, 2), msg="covariance not 2x2")
        self.assertEqual(cond.mean.shape, (2,), msg="mean not length 2")

    def test_mean_follows_outcome(self):
        """
        The conditional x mean is the outcome times the Schur gain tanh(2r).
        """
        cond = g.Epr(1.0).condition(mode=1, angle=0.0, outcome=0.7)
        mean_x = float(cond.mean[0])
        self.assertClose(mean_x, np.tanh(2.0) * 0.7, msg="mean gain != tanh(2r)")

    def test_last_mode_refused(self):
        """
        Conditioning the only mode raises, while homodyne on it still samples and two modes
        still drop to one.
        """
        lone = _core.GaussianState.vacuum(1)
        self.assertBad(
            "zero-mode object",
            lone.condition,
            (0, 0.0, 0.3),
            msg="lone mode conditioned",
        )
        self.assertEqual(
            len(lone.homodyne(0, 0.0, 4, 7)),
            4,
            msg="lone mode draw count",
        )
        self.assertEqual(
            _core.GaussianState.vacuum(2).condition(0, 0.0, 0.3).n_modes,
            1,
            msg="two modes did not drop to one",
        )

    def test_vacuum_untouched(self):
        """
        Conditioning on an uncorrelated vacuum mode leaves the survivor in the vacuum.
        """
        cond = g.Vacuum(2).condition(mode=1, angle=0.0, outcome=0.7)
        self.assertTrue(
            np.allclose(cond.cov, 0.5 * np.eye(2)),
            msg="covariance != I/2",
        )
        self.assertTrue(np.allclose(cond.mean, 0.0), msg="mean != 0")


if __name__ == "__main__":
    rc = Exam(
        "GaussianBonaFide",
        "V + i*Omega/2 >= 0 is gated at hbar, not at the state's energy",
        "gaussian_bona_fide.md",
    ).run(load(BonaFide))
    rc |= Exam(
        "GaussianStates",
        "Phase 1 Gaussian engine: constructors are bona fide with documented moments",
        "gaussian_states.md",
    ).run(load(States))
    rc |= Exam(
        "GaussianOps",
        "Symplectic operations preserve physicality and the covariance algebra",
        "gaussian_ops.md",
    ).run(load(Operations))
    rc |= Exam(
        "GaussianLoss",
        "Thermal-loss channel: fixed points, sqrt(T) scaling, explicit xi plane",
        "gaussian_loss.md",
    ).run(load(Loss))
    rc |= Exam(
        "GaussianSampling",
        "Homodyne/heterodyne sampling moments and seeded reproducibility",
        "gaussian_sampling.md",
    ).run(load(Sampling))
    rc |= Exam(
        "GaussianConditioning",
        "Homodyne conditioning: EPR steering, mode removal, correlated mean update",
        "gaussian_cond.md",
    ).run(load(Conditioning))
    sys.exit(rc)
