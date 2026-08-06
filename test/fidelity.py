import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

from qkd import gaussian as g


def thermal_pair(n1, n2):
    """
    Fidelity of two thermal states summed in the Fock basis, no covariance matrix: 1/((n1+1)(n2+1)(1 -
    sqrt(x1 x2))^2).
    """
    x1 = n1 / (n1 + 1.0)
    x2 = n2 / (n2 + 1.0)
    root = np.sqrt(x1 * x2)

    return 1.0 / ((n1 + 1.0) * (n2 + 1.0) * (1.0 - root) ** 2)


def make(seed):
    """
    A pseudo-random one-mode Gaussian state, physical by construction and mixed.
    """
    rng = np.random.default_rng(seed)
    st = g.Thermal(float(rng.uniform(0.0, 2.0)))
    st = st.squeeze(0, float(rng.uniform(-1.0, 1.0)))
    st = st.rotate(0, float(rng.uniform(0.0, 2.0 * np.pi)))

    return st.displace(0, float(rng.normal()), float(rng.normal()))


class Anchors(Question):
    """
    Fock-basis sums and overlaps pinning hbar = 1, vacuum-variance-1/2 against the closed form -- Scutaru,
    J. Phys. A 31, 3659 (1998), restated in Weedbrook et al., Rev. Mod. Phys. 84, 621 (2012) -- which writes
    det V = 1 for a pure state where internally det V = 1/4, hence the factor of 4 on every determinant.
    """

    def test_self_identity(self):
        """
        $F(\\rho, \\rho) = 1$ for coherent, squeezed, thermal, displaced-squeezed and vacuum states.
        """
        made = (
            g.Coherent(1.3, -0.7),
            g.Squeezed(0.9),
            g.Thermal(1.7),
            g.Squeezed(0.6).displace(0, 1.1, 0.4),
            g.Vacuum(1),
        )
        for st in made:
            self.assertClose(
                st.fidelity(st),
                1.0,
                msg=f"F(rho, rho) must be 1, not {st.fidelity(st)}",
            )

    def test_coherent_pair(self):
        """
        Two coherent states overlap as $\\exp(-\\lvert d\\rvert^2/2)$ under $\\alpha = (x + ip)/\\sqrt2$
        -- the factor of 2 that the $\\sqrt2$ in the quadratures hides.
        """
        for x, p in ((0.5, 0.0), (1.0, 1.0), (-2.0, 0.7), (0.0, 3.0)):
            a = g.Coherent(0.0, 0.0)
            b = g.Coherent(x, p)
            got = a.fidelity(b)
            self.assertClose(got, np.exp(-0.5 * (x * x + p * p)), msg=f"coherent ({x}, {p})")

    def test_vacuum_thermal(self):
        """
        $F(\\lvert 0\\rangle, \\rho_{th}) = 1/(\\bar n + 1)$, the thermal ground-state weight.
        """
        for nbar in (0.0, 0.25, 1.0, 4.0, 40.0):
            got = g.Vacuum(1).fidelity(g.Thermal(nbar))
            self.assertClose(got, 1.0 / (nbar + 1.0), msg=f"nbar = {nbar}")

    def test_thermal_pair(self):
        """
        Two thermal states match the Fock-basis sum $1/((n_1+1)(n_2+1)(1 - \\sqrt{x_1 x_2})^2)$, exercising
        the $\\Lambda$ term that is zero for every pure pair.
        """
        for n1, n2 in ((0.3, 1.4), (1.0, 1.0), (5.0, 0.2), (12.0, 9.0)):
            got = g.Thermal(n1).fidelity(g.Thermal(n2))
            self.assertClose(got, thermal_pair(n1, n2), msg=f"nbar {n1} vs {n2}")

    def test_vacuum_squeezed(self):
        """
        $F(\\lvert 0\\rangle, S(r)\\lvert 0\\rangle) = 1/\\cosh r$.
        """
        for r in (0.0, 0.4, 1.0, 2.5):
            got = g.Vacuum(1).fidelity(g.Squeezed(r))
            self.assertClose(got, 1.0 / np.cosh(r), msg=f"r = {r}")

    def test_coherent_thermal(self):
        """
        $F(\\lvert\\alpha\\rangle, \\rho_{th}) = \\exp(-\\lvert\\alpha\\rvert^2/(\\bar n+1))/(\\bar n+1)$,
        $\\pi$ times the thermal Husimi $Q$: the one anchor pinning both prefactor and exponent.
        """
        for nbar in (0.5, 2.0, 7.0):
            for x, p in ((1.0, 0.0), (0.8, -1.4)):
                got = g.Coherent(x, p).fidelity(g.Thermal(nbar))
                sq = 0.5 * (x * x + p * p)
                want = np.exp(-sq / (nbar + 1.0)) / (nbar + 1.0)
                self.assertClose(got, want, msg=f"nbar={nbar}, mean=({x}, {p})")


class Algebra(Question):
    """
    Agreement with the overlap wherever a pure state makes the two coincide.
    """

    def test_symmetry(self):
        """
        $F(a, b) = F(b, a)$ over 40 random mixed single-mode pairs.
        """
        for seed in range(40):
            a = make(seed)
            b = make(seed + 1000)
            self.assertClose(a.fidelity(b), b.fidelity(a), msg=f"asymmetric at seed {seed}")

    def test_range(self):
        """
        $0 \\le F \\le 1$ over 60 random mixed pairs, with no clamping to make it true.
        """
        for seed in range(60):
            a = make(seed)
            b = make(seed + 7)
            got = a.fidelity(b)
            self.assertGreaterEqual(got, 0.0, msg=f"F = {got} < 0 at seed {seed}")
            self.assertLessEqual(got, 1.0 + 1e-12, msg=f"F = {got} > 1 at seed {seed}")

    def test_unitary_invariance(self):
        """
        A common symplectic unitary leaves $F$ unchanged: $F(UaU^\\dagger, UbU^\\dagger) = F(a, b)$.
        """
        for seed in range(12):
            a = make(seed)
            b = make(seed + 500)
            base = a.fidelity(b)
            ua = a.squeeze(0, 0.4).rotate(0, 0.9).displace(0, 0.6, -0.3)
            ub = b.squeeze(0, 0.4).rotate(0, 0.9).displace(0, 0.6, -0.3)
            self.assertClose(ua.fidelity(ub), base, msg=f"unitary moved F at seed {seed}")

    def test_pure_overlap(self):
        """
        $F = \\mathrm{Tr}(\\rho\\sigma)$ with a pure state on one side, where $\\Lambda$ vanishes.
        """
        for seed in range(20):
            mixed = make(seed)
            pure = g.Squeezed(0.3).displace(0, 0.4, -0.2)
            self.assertClose(pure.fidelity(mixed), pure.overlap(mixed), msg=f"seed {seed}")

    def test_two_mode_pure(self):
        """
        Two pure two-mode squeezed vacua take the n-mode pure branch and match the Fock-basis overlap
        $1/(\\cosh^2 r_1 \\cosh^2 r_2 (1 - t_1 t_2)^2)$ with $t = \\tanh r$.
        """
        for r1, r2 in ((0.5, 0.8), (0.2, 0.2), (1.1, 0.05)):
            got = g.Epr(r1).fidelity(g.Epr(r2))
            tan = np.tanh(r1) * np.tanh(r2)
            want = 1.0 / (np.cosh(r1) ** 2 * np.cosh(r2) ** 2 * (1.0 - tan) ** 2)
            self.assertClose(got, want, msg=f"EPR {r1} vs {r2}")


class Channels(Question):
    """
    Two states under a shared channel, not two ways of writing one: where a convention error that survives
    every algebraic check fails.
    """

    def test_noise_monotone(self):
        """
        A shared thermal-loss channel never lowers $F$, over a grid of $(T, \\xi)$ at both planes.
        """
        for seed in range(6):
            a = make(seed)
            b = make(seed + 900)
            base = a.fidelity(b)
            for t in (0.8, 0.5, 0.2):
                for xi in (0.01, 0.1, 0.5):
                    for ref in ("input", "output"):
                        out = a.thermal_loss(0, T=t, xi=xi, ref=ref)
                        got = out.fidelity(b.thermal_loss(0, T=t, xi=xi, ref=ref))
                        self.assertGreaterEqual(
                            got,
                            base - 1e-12,
                            msg=f"T={t}, xi={xi}, ref={ref}, seed={seed}: " f"{got:.9f} < {base:.9f}",
                        )

    def test_coherent_loss(self):
        """
        Pure loss scales a coherent pair's amplitude difference by $\\sqrt T$, so $F$ goes from
        $\\exp(-\\lvert d\\rvert^2/2)$ to $\\exp(-T\\lvert d\\rvert^2/2)$ -- an equality, not a bound.
        """
        for t in (1.0, 0.6, 0.25, 0.04):
            a = g.Coherent(1.4, -0.6).thermal_loss(0, T=t, xi=0.0, ref="input")
            b = g.Coherent(0.2, 0.9).thermal_loss(0, T=t, xi=0.0, ref="input")
            sq = (1.4 - 0.2) ** 2 + (-0.6 - 0.9) ** 2
            self.assertClose(a.fidelity(b), np.exp(-0.5 * t * sq), msg=f"coherent pair at T={t}")

    def test_thermal_channel(self):
        """
        Two thermal states converge as $T$ falls, both reaching the vacuum with $F \\to 1$.
        """
        a = g.Thermal(3.0)
        b = g.Thermal(0.1)
        seq = []
        for t in (1.0, 0.5, 0.2, 0.05, 1e-3, 1e-6):
            out = a.thermal_loss(0, T=t, xi=0.0, ref="input")
            seq.append(out.fidelity(b.thermal_loss(0, T=t, xi=0.0, ref="input")))
        self.assertMonotone(seq, rising=True, strict=True, msg=f"{seq}")
        self.assertClose(seq[-1], 1.0, atol=1e-5, msg="both must reach the vacuum")


class Distance(Question):
    """
    The bracket is Fuchs and van de Graaf, IEEE Trans. Inf. Theory 45, 1216 (1999): 1 - sqrt(F) <= D <=
    sqrt(1 - F) for the squared fidelity F.
    """

    def test_pure_distance(self):
        """
        A pure pair has $D = \\sqrt{1 - F}$ exactly: $\\sqrt{1 - \\exp(-\\lvert d\\rvert^2/2)}$ for coherent
        states, 0 for identical ones.
        """
        a = g.Coherent(0.0, 0.0)
        self.assertClose(a.trace_distance(a), 0.0, msg="a state is 0 from itself")

        for x, p in ((0.5, 0.0), (1.0, -1.0), (3.0, 2.0)):
            b = g.Coherent(x, p)
            want = np.sqrt(1.0 - np.exp(-0.5 * (x * x + p * p)))
            self.assertClose(a.trace_distance(b), want, msg=f"coherent ({x}, {p})")

    def test_bounds_bracket(self):
        """
        The Fuchs-van de Graaf bracket stays ordered in [0, 1], tight on a pure pair.
        """
        for seed in range(30):
            lo, hi = make(seed).trace_bounds(make(seed + 11))
            self.assertGreaterEqual(lo, 0.0, msg=f"lower {lo} < 0 at seed {seed}")
            self.assertLessEqual(hi, 1.0 + 1e-12, msg=f"upper {hi} > 1 at seed {seed}")
            self.assertLessEqual(lo, hi + 1e-12, msg=f"bracket inverted at seed {seed}")

        a = g.Squeezed(0.5)
        b = g.Squeezed(0.5).displace(0, 1.0, 0.3)
        lo, hi = a.trace_bounds(b)
        self.assertClose(hi, a.trace_distance(b), msg="upper end is tight when pure")
        self.assertLessEqual(lo, a.trace_distance(b) + 1e-12, msg="lower end must hold")

    def test_mixed_refused(self):
        """
        A mixed pair has no closed form: `trace_distance` raises and names `trace_bounds`.
        """
        a = g.Thermal(1.0)
        b = g.Thermal(2.0)
        self.assertFails(
            NotImplementedError,
            "trace_bounds",
            lambda: a.trace_distance(b),
            msg="a mixed pair must be refused by name",
        )


class Limits(Question):
    """
    The general n-mode mixed case raises rather than ship a formula these moments cannot prove: Banchi,
    Braunstein and Pirandola, Phys. Rev. Lett. 115, 260501 (2015).
    """

    def test_mixed_refused(self):
        """
        A mixed two-mode pair raises `NotImplementedError` naming `keep()`.
        """
        a = g.Epr(0.6).thermal_loss(0, T=0.7, xi=0.05, ref="input")
        b = g.Epr(0.4).thermal_loss(1, T=0.8, xi=0.02, ref="input")
        self.assertFails(
            NotImplementedError,
            "keep()",
            lambda: a.fidelity(b),
            msg="a mixed 2-mode pair must be refused with a route out",
        )

    def test_mode_mismatch(self):
        """
        A one-mode against a two-mode state is a `ValueError` naming both counts.
        """
        self.assertFails(
            ValueError,
            "equal mode counts",
            lambda: g.Vacuum(1).fidelity(g.Vacuum(2)),
            msg="a mode-count mismatch must be refused",
        )

    def test_not_a_state(self):
        """
        A non-`State` argument is a `TypeError`, not a deep attribute error.
        """
        self.assertFails(
            TypeError,
            "State",
            lambda: g.Vacuum(1).fidelity(0.5),
            msg="a non-State argument must be refused up front",
        )
        self.assertFails(
            TypeError,
            "State",
            lambda: g.Vacuum(1).trace_distance(0.5),
            msg="the distance must validate before it reaches the moments",
        )

    def test_results_finite(self):
        """
        Every quantity stays finite for a hot, strongly squeezed, far-displaced pair, where the rationalised
        prefactor is what prevents the cancellation.
        """
        a = g.Thermal(1e4).squeeze(0, 3.0).displace(0, 50.0, -30.0)
        b = g.Thermal(1e3).squeeze(0, -2.0).displace(0, -20.0, 40.0)
        got = a.fidelity(b)
        self.assertFinite([got, a.overlap(b)], msg="hot pair must stay finite")
        self.assertGreaterEqual(got, 0.0, msg=f"hot pair F = {got} < 0")
        self.assertLessEqual(got, 1.0 + 1e-12, msg=f"hot pair F = {got} > 1")


if __name__ == "__main__":
    rc = Exam(
        "FidelityAnchors",
        "Fidelity against Fock-basis values the covariance formula never sees",
        "fidelity_anchors.md",
    ).run(load(Anchors))
    rc |= Exam(
        "FidelityAlgebra",
        "Symmetry, range, unitary invariance and agreement with the overlap",
        "fidelity_algebra.md",
    ).run(load(Algebra))
    rc |= Exam(
        "FidelityChannels",
        "Data processing: a shared channel can only raise the fidelity",
        "fidelity_channels.md",
    ).run(load(Channels))
    rc |= Exam(
        "FidelityDistance",
        "Trace distance where it is exact, and the bracket where it is not",
        "fidelity_distance.md",
    ).run(load(Distance))
    rc |= Exam(
        "FidelityLimits",
        "What the layer refuses to compute, and how loudly it says so",
        "fidelity_limits.md",
    ).run(load(Limits))
    sys.exit(rc)
