import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

from kit.checks import Guarded
from qkd import fock as f
from qkd import gaussian as g
from qkd._core import FockState, GaussianState

XS = np.linspace(-5.0, 5.0, 121)
PS = np.linspace(-4.0, 4.0, 97)
STEP = (float(XS[1] - XS[0]), float(PS[1] - PS[0]))


def raw(st, xs=XS, ps=PS):
    """
    A core state's Wigner grid, reshaped into (len(ps), len(xs)).
    """
    flat = st.wigner(list(xs), list(ps))

    return np.array(flat, dtype=np.float64).reshape(len(ps), len(xs))


def mixture(x0, cutoff=30):
    """
    Equal classical mixture of the two coherent lobes a cat superposes.
    """
    hot = f.Coherent(x0, 0.0, cutoff=cutoff).matrix()
    cold = f.Coherent(-x0, 0.0, cutoff=cutoff).matrix()

    return f.Density(0.5 * (hot + cold))


class Surface(Guarded):
    def test_engine_agreement(self):
        """
        Every number qkd.fock reports equals _core.FockState's bit for bit, on a rectangular
        window.
        """
        st = f.Cat(1.5, 0.4, odd=True, cutoff=30)
        core = FockState.cat(1.5, 0.4, True, 30)

        re, im = core.matrix()
        want = np.array(re).reshape(30, 30) + 1j * np.array(im).reshape(30, 30)
        mean, cov = core.moments()

        self.assertEqual(st.cutoff, 30, msg="cutoff")
        self.gridClose(st.matrix().real, want.real, atol=0.0, msg="matrix real part")
        self.gridClose(st.matrix().imag, want.imag, atol=0.0, msg="matrix imaginary part")
        self.gridClose(st.moments()[0], np.array(mean), atol=0.0, msg="mean")
        self.gridClose(st.moments()[1], np.array(cov).reshape(2, 2), atol=0.0, msg="covariance")
        self.assertEqual(st.wigner(XS, PS).shape, (97, 121), msg="Wigner shape")
        self.gridClose(st.wigner(XS, PS), raw(core), atol=0.0, msg="Wigner grid")
        self.assertEqual(
            st.negativity(XS, PS).value,
            core.negativity(list(XS), list(PS)),
            msg="negativity volume",
        )

    def test_grid_orientation(self):
        """
        W[i, j] is W(xs[j], ps[i]) on a rectangular grid, integrates to 1, and matches the
        Gaussian layer's Wigner and Husimi pointwise with Q inside its bounds.
        """
        w = f.Coherent(2.0, -1.0).wigner(XS, PS)
        i, j = np.unravel_index(int(np.argmax(w)), w.shape)

        self.assertClose(XS[j], 2.0, atol=STEP[0], msg="peak column x")
        self.assertClose(PS[i], -1.0, atol=STEP[1], msg="peak row p")
        self.assertClose(float(w.sum() * STEP[0] * STEP[1]), 1.0, atol=1e-4, msg="W integral")

        here = f.Coherent(1.2, -0.7)
        there = g.Coherent(1.2, -0.7)

        self.gridClose(here.wigner(XS, PS), there.wigner(0, XS, PS), atol=1e-9, msg="Wigner across layers")
        self.gridClose(here.husimi(XS, PS), there.husimi(0, XS, PS), atol=1e-9, msg="Husimi across layers")
        self.assertHusimi(here.husimi(XS, PS), msg="Q outside its bounds")

    def test_immutable_chain(self):
        """
        rotate, displace and loss return new states and leave the original alone.
        """
        st = f.Coherent(1.0, 0.0, cutoff=30)
        before = st.matrix()

        moved = st.displace(0.5, 0.5).rotate(0.3).loss(0.8)

        self.assertIsNot(moved, st, msg="mutated in place")
        self.gridClose(st.matrix().real, before.real, atol=0.0, msg="original matrix moved")
        self.assertClose(
            moved.photons(),
            0.8 * f.Coherent(1.0, 0.0, cutoff=30).displace(0.5, 0.5).photons(),
            atol=1e-9,
            msg="photons after loss",
        )
        self.gridClose(st.loss(1.0).matrix().real, before.real, atol=1e-12, msg="loss(1.0) vs identity")

    def test_shadow_bridge(self):
        """
        shadow() of Thermal(1.5) carries the (nbar + 1/2) covariance moments() hands
        GaussianState.from_moments, its entropy, and zero non-Gaussianity.
        """
        st = f.Thermal(1.5)
        mean, cov = st.moments()

        native = GaussianState.from_moments(mean.reshape(-1).tolist(), cov.reshape(-1).tolist())
        shade = st.shadow()

        self.gridClose(shade.cov, cov, atol=0.0, msg="shadow covariance")
        self.gridClose(np.array(native.cov()).reshape(2, 2), cov, atol=0.0, msg="from_moments covariance")
        self.assertClose(shade.entropy(), g.Thermal(1.5).entropy(), atol=1e-6, msg="shadow entropy")
        self.assertClose(st.non_gaussianity(), 0.0, atol=1e-6, msg="thermal non-Gaussianity")

    def test_gkp_scales(self):
        """
        The derived cutoff holds discarded under GKP_TAIL to delta = 0.15 and refuses
        delta = 0.08; the fixed 60 levels over-report the negativity by a third at 0.2.
        """
        grid = np.linspace(-6.0, 6.0, 121)
        loose = f.Gkp(0, 0.2, cutoff=60)
        tight = f.Gkp(0, 0.2)

        self.assertGreater(loose.discarded(), 1e-3, msg=f"60-level discard {loose.discarded()}")

        for delta in (0.35, 0.25, 0.2, 0.15):
            self.assertLess(f.Gkp(0, delta).discarded(), f.GKP_TAIL, msg=f"discarded at delta = {delta}")

        near = float(tight.negativity(grid, grid))
        far = float(tight.at_cutoff(tight.cutoff + 80).negativity(grid, grid))
        gap = abs(float(loose.negativity(grid, grid)) - far)

        self.assertClose(near, far, atol=5e-3, msg=f"negativity {near} vs {far}")
        self.assertGreater(gap, 0.1, msg=f"60-level gap {gap}")
        self.assertBad("512 levels", f.Gkp, (0, 0.08), msg="delta = 0.08 accepted")


class Truncation(Question):
    """
    A truncated density matrix is renormalised: physically valid, wrong numbers.
    """

    def test_error_moves(self):
        """
        A state too big for its cutoff shows in discarded(), one pushed against the ceiling
        in tail(), and loss moves both back.
        """
        tight = [f.Coherent(2.0, 0.0, cutoff=d).discarded() for d in (30, 20, 12, 8)]
        room = f.Coherent(1.0, 0.0, cutoff=20)
        pushed = room.displace(3.0, 0.0)

        self.assertMonotone(tight, rising=True, msg=f"discarded across cutoffs: {tight}")
        self.assertGreater(pushed.tail(), room.tail(), msg="tail after a displacement")
        self.assertLess(pushed.loss(0.3).tail(), pushed.tail(), msg="tail after loss")
        self.assertEqual(f.Number(3, cutoff=20).discarded(), 0.0, msg="Number(3) discarded")

    def test_volume_carries_grid(self):
        """
        A negativity comes back as a Volume carrying the window, step and truncation error,
        and reads as its number under float().
        """
        st = f.Number(1, cutoff=30)
        vol = st.negativity(XS, PS)

        self.assertEqual(vol.points, (121, 97), msg="point counts")
        self.assertEqual(vol.span, (-5.0, 5.0, -4.0, 4.0), msg="window")
        self.assertClose(vol.step[0], STEP[0], msg="step in x")
        self.assertEqual(vol.trunc_error, st.trunc_error(), msg="truncation error")
        self.assertEqual(float(vol), vol.value, msg="float(Volume)")

    def test_no_cheap_guard(self):
        """
        Exactly Gaussian states pass physical() and outscore Number(10) at a lower
        volume/trunc_error ratio (no Volume.ratio), rebuilt at cutoff 120 an artefact moves and
        Number(3) does not, and a bare density matrix has no recipe.
        """
        wrong = {
            "Squeezed(1.5)": f.Squeezed(1.5),
            "Squeezed(2.0)": f.Squeezed(2.0),
            "Coherent(9, 0)": f.Coherent(9.0, 0.0),
            "Coherent(11, 0)": f.Coherent(11.0, 0.0),
        }
        seen = {}

        for name, st in wrong.items():
            vol = st.negativity()
            seen[name] = (vol.value, vol.value / vol.trunc_error)
            self.assertTrue(st.physical(), msg=f"{name} physical()")
            self.assertGreater(vol.value, 0.1, msg=f"{name} volume: {vol}")

        real = f.Number(10).negativity().value
        worst = seen["Coherent(11, 0)"]

        self.assertGreater(worst[0], real, msg=f"volumes: {seen}")
        self.assertLess(worst[1], seen["Squeezed(1.5)"][1], msg=f"ratios: {seen}")
        self.assertFalse(hasattr(f.Volume, "ratio"), msg="Volume.ratio is back")

        axis = np.linspace(-8.0, 8.0, 161)
        fake = f.Squeezed(1.5)
        real = f.Number(3, cutoff=40)
        moved = abs(float(fake.negativity(axis, axis)) - float(fake.at_cutoff(120).negativity(axis, axis)))
        still = abs(float(real.negativity(axis, axis)) - float(real.at_cutoff(120).negativity(axis, axis)))

        self.assertGreater(moved, 0.1, msg=f"artefact moved {moved}")
        self.assertLess(still, 1e-6, msg=f"resource moved {still}")
        self.assertEqual(
            f.Number(1).rotate(0.3).loss(0.9).at_cutoff(60).cutoff,
            60,
            msg="cutoff after at_cutoff(60)",
        )
        self.assertFails(
            ValueError,
            "no recipe",
            mixture(2.0).at_cutoff,
            60,
            msg="an explicit density matrix rebuilt",
        )

    def test_purity_truncated(self):
        """
        Thermal(200) at cutoff 40 keeps 18% of its norm and reports ten times the true
        1/(1 + 2*nbar), with `least` = kept^2 * value below it.
        """
        hot = f.Thermal(200.0)
        cool = f.Thermal(0.5)
        true = 1.0 / (1.0 + 2.0 * 200.0)

        self.assertClose(hot.purity().value, 0.025083, atol=1e-6, msg="truncated purity")
        self.assertLess(hot.purity().least, true, msg=f"bound {hot.purity().least} vs true {true}")
        self.assertClose(cool.purity().value, 0.5, msg="Thermal(0.5) purity")
        self.assertClose(cool.purity().least, 0.5, msg="Thermal(0.5) bound")

    def test_window_explicit(self):
        """
        The default window is five sigma of the moments capped at sqrt(2*cutoff) + 5, a
        half-width of 2.0 reports a smaller volume, and Coherent(11, 0) at cutoff 40 asks for
        25.65 and is held to 13.94, outside which its Wigner is 5e-33.
        """
        st = f.Number(1, cutoff=30)
        wide = f.window(st)
        near = f.window(st, reach=2.0)
        full = st.negativity(wide, wide)
        cut = st.negativity(near, near)

        # atol 1e-12: window()'s edge is this exact product.
        self.assertClose(wide[-1], 5.0 * np.sqrt(st.moments()[1][0, 0]), atol=1e-12, msg="default window edge")
        self.assertLess(cut.value, full.value, msg=f"cut {cut.value} vs full {full.value}")
        self.assertEqual(cut.span[1], 2.0, msg="reported span")

        far = f.Coherent(11.0, 0.0)
        mean, cov = far.moments()
        spread = max(abs(mean[i]) + 5.0 * np.sqrt(cov[i, i]) for i in (0, 1))
        cap = math.sqrt(2.0 * far.cutoff) + 5.0
        band = np.linspace(cap, spread, 21)

        self.assertGreater(spread, 25.0, msg=f"moments ask for {spread}")
        self.assertClose(f.window(far)[-1], cap, atol=1e-12, msg=f"window edge vs cap {cap}")
        self.assertLess(
            float(np.abs(far.wigner(band, np.linspace(-spread, spread, 41))).max()),
            1e-12,
            msg="Wigner beyond the cap",
        )


class Witness(Question):
    def test_witness_truncation(self):
        """
        Exactly Gaussian states (true score zero) post 2.28 bits of S(shadow) - S(rho) at the
        default cutoff and under 1e-3 at cutoff 160, as a Divergence carrying the truncation error.
        """
        fake = f.Squeezed(2.0).non_gaussianity()
        real = f.Cat(2.0, 0.0, odd=True, cutoff=40).non_gaussianity()
        settled = f.Squeezed(1.5).at_cutoff(160).non_gaussianity()

        self.assertGreater(fake.value, 2.0, msg=f"Squeezed(2.0) scores {fake}")
        self.assertGreater(fake.value, real.value, msg=f"{fake} vs the odd cat's {real}")
        self.assertEqual(fake.trunc_error, f.Squeezed(2.0).trunc_error(), msg="reported truncation error")
        self.assertEqual(fake.cutoff, 40, msg="reported cutoff")
        self.assertGreater(float(f.Squeezed(1.5).non_gaussianity()), 0.5, msg="Squeezed(1.5) non-Gaussianity")
        self.assertLess(settled.value, 1e-3, msg=f"at cutoff 160: {settled}")
        self.assertEqual(float(fake), fake.value, msg="float(Divergence)")


if __name__ == "__main__":
    rc = Exam(
        "TruncatedSurface",
        "Phase 5c API: qkd.fock over the engine, and against its Gaussian sibling",
        "truncated_surface.md",
    ).run(load(Surface))
    rc |= Exam(
        "TruncatedCutoff",
        "Phase 5c API: the cutoff as part of the state, and the volume that carries it",
        "truncated_cutoff.md",
    ).run(load(Truncation))
    rc |= Exam(
        "TruncatedWitness",
        "Phase 5c API: Wigner negativity and non-Gaussianity, and what separates them",
        "truncated_witness.md",
    ).run(load(Witness))
    sys.exit(rc)
