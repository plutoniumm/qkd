import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import ALPHA, ETA_BOB, E_DET
from kit.checks import Guarded
from kit.anchors import E_VAC as E0
from kit.anchors import F_REC as F_EC
from kit.anchors import Y_DARK as Y0
from kit.forms import bisect, h2
from qkd import _core

# MQZL Sec. 3.1: their Eq. (12) optimum, and the weak decoy of their Figure 2.
MU_OPT = 0.48
NU_WEAK = 0.05

# Standard (non-efficient) BB84 sifting.
Q_SIFT = 0.5

# Any positive triple summing to 1 does; the source register does not read it.
PROBS = (0.7, 0.2, 0.1)


def eta(length):
    return _core.decoy_eta(ALPHA, length, ETA_BOB)


def gain(mu, length):
    return _core.decoy_gain(mu, eta(length), Y0, E_DET)


def ideal(length):
    return _core.decoy_ideal(eta(length), Y0, E_DET)


def bounds(length, mu=MU_OPT, nu=NU_WEAK, nu2=0.0):
    """
    Vacuum+weak (nu2 = 0) or general two-decoy bounds at a fibre length.
    """
    q_mu, e_mu = gain(mu, length)
    q_nu, e_nu = gain(nu, length)
    if nu2 <= 0.0:
        q_vac, e_vac = Y0, E0
    else:
        q_vac, e_vac = gain(nu2, length)

    return _core.decoy_bounds(mu, nu, nu2, q_mu, e_mu, q_nu, e_nu, q_vac, e_vac)


def rate(length, mu=MU_OPT, nu=NU_WEAK):
    """
    GLLP rate from the vacuum+weak bounds, their Eq. (42).
    """
    q_mu, e_mu = gain(mu, length)
    _y1, e1, q1 = bounds(length, mu=mu, nu=nu)

    return Q_SIFT * (-q_mu * F_EC * h2(e_mu) + q1 * (1.0 - h2(e1)))


def rate_inf(length, mu=MU_OPT):
    """
    The same rate fed by infinite-decoy values, their Eq. (1).
    """
    q_mu, e_mu = gain(mu, length)
    y1, e1 = ideal(length)
    q1 = y1 * mu * math.exp(-mu)

    return Q_SIFT * (-q_mu * F_EC * h2(e_mu) + q1 * (1.0 - h2(e1)))


def cross(fn, hi=400.0):
    """
    The distance at which a rate function changes sign.
    """

    return bisect(fn, 0.0, hi)


def mu_opt(f):
    """
    Solve their Eq. (12), (1-mu)exp(-mu) = f*H2(e_det)/(1 - H2(e_det)).
    """
    want = f * h2(E_DET) / (1.0 - h2(E_DET))

    return bisect(lambda mu: (1.0 - mu) * math.exp(-mu), 0.0, 1.0, want)


class ChannelModel(Question):
    """
    The forward model the decoy bounds invert: Ma-Qi-Zhao-Lo Eqs. (5)-(11).
    """

    def test_gain_poisson_sum(self):
        """
        The closed-form gain Y0 + (1 - Y0)(1 - exp(-eta*mu)) equals the Poisson sum over Y_i =
        Y0 + eta_i - Y0*eta_i, with eta_i = 1 - (1-eta)^i.
        """
        e = eta(25.0)
        got, _ = gain(MU_OPT, 25.0)
        total = 0.0
        for i in range(120):
            step = 1.0 - (1.0 - e) ** i
            y_i = Y0 + step - Y0 * step
            total += y_i * MU_OPT**i * math.exp(-MU_OPT) / math.factorial(i)
        self.assertClose(got, total, atol=1e-15, msg="gain closed form vs Poisson sum")

    def test_gain_double_count(self):
        """
        The gain sums Eq. (7) exactly, below Eq. (10) by its double-count Y0*(1 - exp(-eta*mu))
        that puts Y1 above its ceiling, and stays in [0, 1].
        """
        e = eta(25.0)
        got, _ = gain(MU_OPT, 25.0)
        loose = Y0 + (1.0 - math.exp(-e * MU_OPT))
        self.assertLess(got, loose, msg=f"gain {got} vs Eq. (10) {loose}")
        self.assertClose(
            loose - got,
            Y0 * (1.0 - math.exp(-e * MU_OPT)),
            atol=1e-18,
            msg="double-count gap",
        )
        self.assertLessEqual(
            _core.decoy_gain(50.0, 1.0, 1.0, E_DET)[0],
            1.0,
            msg="gain at mu = 50, eta = Y0 = 1",
        )

    def test_qber_poisson_sum(self):
        """
        The same for the QBER: E_mu*Q_mu = e0*Y0 + e_det*(1 - exp(-eta*mu)) equals the Poisson
        sum of e_i*Y_i with e_i = (e0*Y0 + e_det*eta_i)/Y_i.
        """
        e = eta(25.0)
        q_mu, e_mu = gain(MU_OPT, 25.0)
        total = 0.0
        for i in range(120):
            step = 1.0 - (1.0 - e) ** i
            y_i = Y0 + step - Y0 * step
            e_i = (E0 * Y0 + E_DET * step) / y_i
            total += e_i * y_i * MU_OPT**i * math.exp(-MU_OPT) / math.factorial(i)
        self.assertClose(q_mu * e_mu, total, atol=1e-15, msg="QBER vs Poisson sum")

    def test_dark_count_floor(self):
        """
        With the channel dead the gain is Y0 alone and the QBER saturates at 1/2.
        """
        q_mu, e_mu = _core.decoy_gain(MU_OPT, 0.0, Y0, E_DET)
        self.assertClose(q_mu, Y0, atol=1e-18, msg="dead channel gain is Y0")
        self.assertClose(e_mu, E0, atol=1e-15, msg="dark counts give QBER 1/2")

    def test_ideal_noiseless(self):
        """
        Without background the infinite-decoy Y1 is eta and e1 the misalignment.
        """
        y1, e1 = _core.decoy_ideal(0.3, 0.0, E_DET)
        self.assertClose(y1, 0.3, atol=1e-15, msg="Y1 = eta with no background")
        self.assertClose(e1, E_DET, atol=1e-15, msg="e1 = e_detector with no background")


class GysAnchors(Question):
    """
    Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005): their Eq. (12) optima, Figure 1 deviations and
    Figure 2 distances, on the never-fitted GYS Table 1 constants.
    """

    def test_optimal_intensity(self):
        """
        Their Eq. (12) on the GYS misalignment gives mu_optimal = 0.48 at f(e) = 1.22 and 0.54
        at f(e) = 1, the two values of their Sec. 3.1.
        """
        # atol 5e-3 = MQZL's two-decimal rounding, not a solver floor.
        self.assertClose(mu_opt(F_EC), 0.48, atol=5e-3, msg="mu_opt at f = 1.22")
        self.assertClose(mu_opt(1.0), 0.54, atol=5e-3, msg="mu_opt at f = 1")

    def test_deviation_at_40km(self):
        """
        Their Figure 1 at 40 km, nu/mu = 25%: the vacuum+weak Y1 bound is 3.5% below the
        infinite-decoy yield and e1 16.8% above its error rate.
        """
        y1, e1, _q1 = bounds(40.0, nu=0.25 * MU_OPT)
        y_inf, e_inf = ideal(40.0)

        # atol 0.05 = rounding half-width of MQZL's one-decimal deviations.
        self.assertClose(100.0 * (y_inf - y1) / y_inf, 3.5, atol=0.05, msg="Y1 deviation at 40 km")
        self.assertClose(100.0 * (e1 - e_inf) / e_inf, 16.8, atol=0.05, msg="e1 deviation at 40 km")

    def test_max_distance(self):
        """
        Their Figure 2: the vacuum+weak rate of Eq. (42) at mu = 0.48, nu = 0.05 stays positive
        to a maximal secure distance of 140.55 km.
        """
        got = cross(rate)

        # atol 0.2 stays under the 1.5 km infinite-decoy gap.
        self.assertClose(got, 140.55, atol=0.2, msg="vacuum+weak maximal distance")

    def test_ideal_max_distance(self):
        """
        Their Figure 2, dashed: the infinite-decoy rate of Eq. (1) reaches 142.05 km, so two
        decoy states cost ~1.5 km of reach.
        """
        got = cross(rate_inf)
        self.assertClose(got, 142.05, atol=0.2, msg="infinite-decoy maximal distance")
        self.assertGreater(got, cross(rate), msg="ideal must outreach vacuum+weak")

    def test_positive_past_100km(self):
        """
        The GYS rate is positive at 120 km and falls monotonically with distance.
        """
        rates = [rate(d) for d in (0.0, 25.0, 50.0, 75.0, 100.0, 120.0)]
        self.assertGreater(rates[-1], 0.0, msg="must still key at 120 km")
        self.assertMonotone(rates, rising=False, msg="rate must fall with distance")
        self.assertFinite(rates, msg="every rate must be finite")


class Sandwich(Question):
    """
    Eqs. (21)/(25) bracket from the safe side and converge as the decoys weaken.
    """

    def _sweep(self):
        return [float(d) for d in range(0, 205, 5)]

    def test_bracket_ideal(self):
        """
        Over 0-200 km the vacuum+weak Y1 lower bound never exceeds the ideal Y1 and the e1 upper
        bound never falls below the ideal e1.
        """
        for d in self._sweep():
            y1, e1, _q1 = bounds(d)
            y_inf, e_inf = ideal(d)
            self.assertLessEqual(y1, y_inf, msg=f"Y1 bound above truth at {d} km")
            self.assertGreaterEqual(e1, e_inf, msg=f"e1 bound below truth at {d} km")

    def test_gain_consistency(self):
        """
        The returned Q1 bound is Q1 = Y1*mu*exp(-mu), the Poisson weight of Eq. (8).
        """
        for d in (0.0, 50.0, 140.0):
            y1, _e1, q1 = bounds(d)
            self.assertClose(
                q1,
                y1 * MU_OPT * math.exp(-MU_OPT),
                atol=1e-18,
                msg=f"Q1 vs Y1 at {d} km",
            )

    def test_vacuum_decoy_optimal(self):
        """
        Their Sec. 3.3: at fixed nu1 a vacuum second decoy gives the tightest bounds.
        """
        for d in (0.0, 40.0, 100.0, 140.0):
            best = bounds(d, nu=0.1, nu2=0.0)
            for nu2 in (0.01, 0.03, 0.05):
                worse = bounds(d, nu=0.1, nu2=nu2)
                self.assertGreater(best[0], worse[0], msg=f"vacuum Y1 not tightest at {d} km")
                self.assertLess(best[1], worse[1], msg=f"vacuum e1 not tightest at {d} km")

    def test_tightness(self):
        """
        As the weak decoy goes to zero the bounds converge onto the infinite-decoy values, the
        deviation shrinking linearly in nu as their Sec. 3.4 predicts.
        """
        y_inf, e_inf = ideal(50.0)
        devs = []
        for nu in (0.1, 0.01, 1e-3, 1e-4):
            y1, e1, _q1 = bounds(50.0, nu=nu)
            devs.append(((y_inf - y1) / y_inf, e1 - e_inf))
        self.assertMonotone([d[0] for d in devs], rising=False, msg="Y1 deviation must shrink")
        self.assertMonotone([d[1] for d in devs], rising=False, msg="e1 deviation must shrink")
        self.assertClose(devs[-1][0], 0.0, atol=1e-4, msg="Y1 converges at nu = 1e-4")
        self.assertClose(devs[-1][1], 0.0, atol=1e-5, msg="e1 converges at nu = 1e-4")

        # atol 0.05 admits the second-order term in nu; a nu^2 law reads 100.
        self.assertClose(devs[1][0] / devs[2][0], 10.0, atol=0.05, msg="deviation is linear in nu")


class Infeasible(Question):
    """
    e1 is an upper bound: an unusable one is 1/2, where 1 - h2(e1) vanishes.
    """

    def test_e1_never_zeroed(self):
        """
        A negative Eq. (25) numerator returns e1 = 1/2, not the 0.0 a downward clamp gives, and
        e1 stays in [0, 1/2] for e_nu2 up to 1/2.
        """
        args = (0.5, 0.1, 0.05, 1e-3, 0.03, 2e-4, 0.02, 1e-4, 0.45)
        _y1, e1, _q1 = _core.decoy_bounds(*args)
        self.assertEqual(e1, E0, msg="infeasible data prove nothing about e1")

        for e_nu2 in (0.0, 0.2, 0.45, 0.5):
            got = _core.decoy_bounds(*args[:-1], e_nu2)[1]
            self.assertGreaterEqual(got, 0.0, msg=f"e1 stays a rate at {e_nu2}")
            self.assertLessEqual(got, E0, msg=f"e1 is capped at 1/2 at {e_nu2}")

    def test_e1_docstring_path(self):
        """
        The vacuum's e_nu2 = 0.5 at nu2 above 0 (src/decoy.rs prescribes it for nu2 = 0 alone)
        pushes e1 from 0.0466 to the 1/2 cap.
        """
        args = (0.5, 0.1, 0.05, 1e-3, 0.03, 2e-4, 0.02, 1e-4)
        honest = _core.decoy_bounds(*args, 0.0)[1]
        mislabelled = _core.decoy_bounds(*args, E0)[1]
        self.assertGreater(honest, 0.0, msg="the consistent labelling bounds e1")
        self.assertGreater(mislabelled, honest, msg="mislabelling must never sharpen the bound")


class Guards(Guarded):
    """
    Inputs violating Eq. (15) or the physical ranges raise: (nu1-nu2)(mu-nu1-nu2) changes sign
    there and the "lower" bound would come back above the truth.
    """

    def _args(self, mu=0.5, nu1=0.1, nu2=0.0):
        return (mu, nu1, nu2, 0.01, 0.03, 0.005, 0.03, Y0, E0)

    def test_reject_intensities(self):
        """
        The intensity triple is rejected wherever Eq. (15) fails or a value leaves its range and
        the message names the slot, as are gains and error rates outside [0, 1].
        """
        cases = (
            (dict(nu1=0.05, nu2=0.05), "nu2 < nu1", "equal intensities"),
            (dict(nu1=0.02, nu2=0.05), "nu2 < nu1", "swapped intensities"),
            (dict(nu1=0.5), "nu1 + nu2 < mu", "decoy at signal strength"),
            (dict(nu1=0.3, nu2=0.25), "nu1 + nu2 < mu", "decoy pair summing past mu"),
            (dict(mu=-0.5), "mu", "negative signal intensity"),
            (dict(nu1=0.0), "nu1", "zero weak decoy carries no information"),
            (dict(nu2=-0.01), "nu2", "negative decoy intensity"),
        )
        for kw, needle, why in cases:
            self.assertBad(
                needle,
                _core.decoy_bounds,
                self._args(**kw),
                msg=f"{why} must be rejected",
            )

        for slot, needle, bad in ((3, "q_mu", 1.5), (6, "e_nu1", -0.1)):
            args = list(self._args())
            args[slot] = bad
            self.assertBad(needle, _core.decoy_bounds, args, msg=f"{needle} outside [0, 1]")

    def test_vacuous_bound(self):
        """
        Statistics too poor to prove anything give Y1 = Q1 = 0 and e1 capped at 1/2, past which
        1 - H2(e1) rises again.
        """
        y1, e1, q1 = _core.decoy_bounds(0.5, 0.1, 0.0, 0.5, 0.03, 1e-9, 0.03, 0.0, E0)
        self.assertEqual(y1, 0.0, msg="vacuous Y1 bound must be exactly 0")
        self.assertEqual(q1, 0.0, msg="vacuous Q1 bound must be exactly 0")
        self.assertEqual(e1, 0.5, msg="vacuous e1 bound must cap at 1/2")

    def test_reject_bad_channel(self):
        """
        The channel helpers guard their own ranges: transmittance, background, length.
        """
        cases = (
            (_core.decoy_gain, (0.5, 1.5, Y0, E_DET), "eta", "transmittance above 1"),
            (_core.decoy_ideal, (0.3, -1e-6, E_DET), "y0", "negative background rate"),
            (
                _core.decoy_eta,
                (ALPHA, -10.0, ETA_BOB),
                "length",
                "negative fibre length",
            ),
            (
                _core.decoy_eta,
                (ALPHA, 25.0, 1.5),
                "eta_bob",
                "receiver efficiency above 1",
            ),
        )
        for fn, args, needle, why in cases:
            self.assertBad(needle, fn, args, msg=why)


class SourceModel(Guarded):
    """
    The source register on the three entry points that invert Poisson weights: the default is
    the idealisation, and each named imperfection refuses.
    """

    def _entries(self):
        """
        (function, good argument tuple) for each entry point carrying a source.
        """
        return (
            (_core.decoy_bounds, (0.5, 0.1, 0.0, 0.01, 0.03, 0.005, 0.03, Y0, E0)),
            (_core.decoy_counts, (0.5, 0.1, 0.0, PROBS, (1e5, 2e4, 1e3), 1e-10)),
            (_core.decoy_errors, (0.5, 0.1, 0.0, PROBS, (1e3, 2e2, 5e1), 1e-10)),
        )

    def test_default_ideal(self):
        """
        Omitting source, passing None and passing "ideal" give identical results.
        """
        for fn, args in self._entries():
            base = fn(*args)
            self.assertEqual(fn(*args, None), base, msg=f"{fn.__name__} shifted on an explicit None")
            self.assertEqual(fn(*args, "ideal"), base, msg=f"{fn.__name__} shifted on ideal")

    def test_reject_sources(self):
        """
        Every entry point refuses the discrete-phase, intensity-error and correlated sources, and
        an unknown name lists the register rather than falling back to the ideal source.
        """
        cases = (
            ("discrete-phase", NotImplementedError, "pseudo-Fock"),
            ("intensity-error", NotImplementedError, "not Poisson at any mean"),
            ("correlated", NotImplementedError, "correlation range"),
            ("poissonian", ValueError, "unknown source"),
        )
        for fn, args in self._entries():
            for name, exc, needle in cases:
                self.assertFails(
                    exc,
                    needle,
                    fn,
                    *args,
                    name,
                    msg=f"{fn.__name__} must refuse the {name} source",
                )

    def test_refusals_direction(self):
        """
        Every refusal states the direction of the error it prevents.
        """
        args = (0.5, 0.1, 0.0, 0.01, 0.03, 0.005, 0.03, Y0, E0)
        for name in ("discrete-phase", "intensity-error", "correlated"):
            self.assertFails(
                NotImplementedError,
                "Direction:",
                _core.decoy_bounds,
                *args,
                name,
                msg=f"{name} must state the direction of its error",
            )


if __name__ == "__main__":
    rc = Exam(
        "DecoyModel",
        "Fibre channel model feeding the decoy bounds (Ma-Qi-Zhao-Lo Eqs. 5-11)",
        "decoy_model.md",
    ).run(load(ChannelModel))
    rc |= Exam(
        "DecoyAnchors",
        "Tier A: GYS published numbers from Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005)",
        "decoy_anchors.md",
    ).run(load(GysAnchors))
    rc |= Exam(
        "DecoySandwich",
        "Finite-decoy bounds bracket the infinite-decoy limit and converge onto it",
        "decoy_sandwich.md",
    ).run(load(Sandwich))
    rc |= Exam(
        "DecoyInfeasible",
        "Infeasible measurements degrade the bounds toward proving nothing",
        "decoy_infeasible.md",
    ).run(load(Infeasible))
    rc |= Exam(
        "DecoyGuards",
        "Intensity-ordering and range guards on the decoy interface",
        "decoy_guards.md",
    ).run(load(Guards))
    rc |= Exam(
        "DecoySource",
        "The source-model register: the ideal source is the default, and the "
        "three source-side idealisations refuse by name",
        "decoy_source.md",
    ).run(load(SourceModel))
    sys.exit(rc)
