import decimal
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, load

from kit.checks import Guarded
from qkd import _core

ZETA = 58.0

ZETA_EC = 28.0

EPS = 2.0**-58 / 6

F_EC = 1.16

E_BIT = 0.01

N_EM = 1e13


def length(eta, mu, t, n_em=N_EM, kato=True, e_bit=E_BIT):
    """
    MTT23 Eq. (24): key length in bits for a coherent source of mean photon number `mu` behind
    transmission `eta`, at code-round probability `t`. The defaults are their Sec. V operating
    point.
    """
    tail = _core.dps_source(mu)
    counts = _core.dps_counts(n_em, eta, mu, t)

    return _core.dps_finite(
        n_em,
        counts[1],
        counts[2],
        e_bit,
        tail[0],
        tail[1],
        tail[2],
        t,
        EPS,
        EPS,
        ZETA,
        ZETA_EC,
        F_EC,
        kato,
        "pnr",
    )


def optimum(eta, n_em=N_EM, kato=True):
    """
    The length maximised over mu and t by four nested grid passes, returned as (length, mu, t).
    """
    lo, hi, low, high = 1e-6, 1e-1, 0.02, 0.999
    best = (-1.0, 0.0, 0.0)
    for _ in range(4):
        for i in range(81):
            mu = lo * (hi / lo) ** (i / 80)
            for j in range(81):
                t = low + (high - low) * j / 80
                got = length(eta, mu, t, n_em, kato)
                if got > best[0]:
                    best = (got, mu, t)
        lo, hi = best[1] / 1.6, best[1] * 1.6
        low, high = max(0.01, best[2] - 0.05), min(0.9995, best[2] + 0.05)

    return best


def deviate(a, n, m, eps):
    """
    Kato's deviation term [b + a(2m/n - 1)]sqrt(n) at the b his constraint Eq. (89) fixes, or
    None where that b does not exist.
    """
    le = math.log(eps)
    squared = a * a - le * (1.0 + 4.0 * a / (3.0 * math.sqrt(n))) ** 2 / 2.0
    if squared < 0.0:
        return None

    b = math.sqrt(squared)
    if b < abs(a):
        return None

    return (b + a * (2.0 * m / n - 1.0)) * math.sqrt(n)


def tail(a, lam, digits=60):
    """
    sum_{nu >= a} e^{-lam} lam^nu / nu! to `digits` decimal places, as the reference the f64
    engine is held against.
    """
    with decimal.localcontext() as ctx:
        ctx.prec = digits
        one = decimal.Decimal(1)
        head = decimal.Decimal(0)
        term = decimal.Decimal(-lam).exp()
        for nu in range(a):
            if nu > 0:
                term *= decimal.Decimal(lam) / nu
            head += term

        return float(one - head)


class Finite(Guarded):
    """
    MTT23: Mizutani, Takeuchi & Tamaki, "Finite-key security analysis of
    differential-phase-shift quantum key distribution", Phys. Rev. Research 5, 023132 (2023),
    arXiv:2301.09844. Its Kato machinery and refused receiver.
    """

    def test_epsilon_composition(self):
        """
        The paper's own $\\zeta' = 28$, $\\zeta = 58$ and $\\epsilon_1 = \\epsilon_2 =
        2^{-58}/6$ compose to exactly $2^{-27}$ under its Eq. (50).
        """
        eps_sec = 2.0**-ZETA_EC + math.sqrt(2.0) * math.sqrt(3 * EPS + 3 * EPS + 2.0**-ZETA)
        self.assertClose(eps_sec, 2.0**-27, atol=0.0, msg="eps_sec must be 2^-27 to the last bit")

    def test_kato_solves_its_problem(self):
        """
        `dps_kato` returns the pair minimising Kato's deviation term subject to his
        failure-probability constraint, against a direct scan.
        """
        for n, m in ((1.8e7, 98.66), (3.57e7, 160.0), (1e6, 50.0), (4096.0, 3.0)):
            a, b = _core.dps_kato(n, m, EPS)
            root = math.sqrt(n)
            scan = min(
                d
                for d in (deviate(-0.5 * root + 5.5 * root * i / 200000, n, m, EPS) for i in range(200001))
                if d is not None
            )
            self.assertClose(deviate(a, n, m, EPS), scan, atol=1e-9 * abs(scan), msg=f"a* not optimal at n={n}")
            self.assertClose(b, deviate(a, n, m, EPS) / root - a * (2 * m / n - 1), atol=1e-9, msg="b* off a*")

    def test_kato_constraint_is_tight(self):
        """
        The returned pair sits on Kato's constraint surface with equality, so the failure
        probability it is certified at is the one asked for.
        """
        for n, m in ((1.8e7, 98.66), (1e6, 50.0)):
            a, b = _core.dps_kato(n, m, EPS)
            got = math.exp(-(2 * b * b - 2 * a * a) / (1.0 + 4.0 * a / (3.0 * math.sqrt(n))) ** 2)
            self.assertClose(got / EPS, 1.0, atol=1e-9, msg=f"constraint slack at n={n}")

    def test_kato_beats_azuma(self):
        """
        The Kato branch of `dps_phase` returns a smaller phase-error bound than the Azuma branch
        at every point of an $\\eta$ ladder.
        """
        for eta in (1.0, 0.5, 0.2, 0.1, 0.05):
            mu, t = 9.3e-3 * eta, 0.9
            tail3 = _core.dps_source(mu)
            counts = _core.dps_counts(N_EM, eta, mu, t)
            args = (N_EM, counts[1], counts[2], E_BIT) + tail3 + (t, EPS, EPS)
            fast = _core.dps_phase(*args, True)
            slow = _core.dps_phase(*args, False)
            self.assertTrue(fast < slow, msg=f"Kato must be tighter than Azuma at eta={eta}")

    def test_azuma_loses_reach(self):
        """
        Replacing Kato by Azuma costs the whole key at $\\eta = 0.1$, the divergence the paper's
        Fig. 5 reports.
        """
        self.assertTrue(optimum(1.0, kato=False)[0] > 0.0, msg="Azuma must hold key at eta = 1")
        self.assertClose(optimum(0.1, kato=False)[0], 0.0, atol=0.0, msg="Azuma must lose key at eta = 0.1")
        self.assertTrue(optimum(0.1)[0] > 0.0, msg="Kato must hold key at eta = 0.1")

    def test_source_tail_is_exact(self):
        """
        `dps_source` matches a 60-digit evaluation of Eq. (51) to twelve digits at mean photon
        numbers spanning eight decades.
        """
        for mu in (1e-6, 9e-5, 9.3e-3, 0.1, 1.0, 3.0):
            got = _core.dps_source(mu)
            for a in (1, 2, 3):
                want = tail(a, 3.0 * mu)
                self.assertClose(got[a - 1] / want, 1.0, atol=1e-12, msg=f"q{a} at mu={mu}")

    def test_source_tail_is_ordered(self):
        """
        $q_1 \\ge q_2 \\ge q_3$: the bounds fall with the photon number they count from, because
        Eq. (3) reads each as a tail, not a weight.
        """
        for mu in (1e-6, 1e-3, 0.5, 2.0):
            got = _core.dps_source(mu)
            self.assertMonotone(list(got), rising=False, msg=f"q1 >= q2 >= q3 at mu={mu}")

    def test_optimal_intensity(self):
        """
        The length peaks at the mean photon numbers the paper's Sec. V prints, 9.3e-3 at $\\eta
        = 1$ and 9.4e-4 at $\\eta = 0.1$, to its two figures.
        """
        self.assertClose(optimum(1.0)[1], 9.3e-3, atol=5e-5, msg="mu_opt at eta = 1")
        self.assertClose(optimum(0.1)[1], 9.4e-4, atol=5e-6, msg="mu_opt at eta = 0.1")

    def test_reach_falls_short(self):
        """
        At the paper's 77 km the shipped length is 1.12 Mbit where its own text claims 3 Mbit,
        and 3 Mbit is reached at 73.4 km instead.
        """
        near = optimum(0.5 * 10.0**-1.54)[0]
        far = optimum(0.5 * 10.0**-1.46)[0]
        self.assertClose(near / 1e6, 1.116, atol=5e-3, msg="length at 77 km")
        self.assertTrue(far > 3e6 > near, msg="3 Mbit must land between 73 km and 77 km")

    def test_length_rises_with_block(self):
        """
        The length per emitted pulse rises with $N_{em}$ at a fixed operating point.
        """
        rates = [length(0.1, 9.4e-4, 0.9, n_em=n) / n for n in (1e11, 1e12, 1e13, 1e14)]
        self.assertMonotone(rates, rising=True, msg="rate must rise with N_em")

    def test_length_falls_with_error(self):
        """
        The length falls as the sample-round bit error rate rises, clamping at zero rather than
        going negative.
        """
        lengths = [length(0.5, 4.7e-3, 0.95, e_bit=e) for e in (0.0, 0.01, 0.02, 0.03, 0.04)]
        self.assertMonotone(lengths, rising=False, strict=False, msg="length must fall with e_bit")
        self.assertClose(lengths[-1], 0.0, atol=0.0, msg="a broken channel must clamp at zero")

    def test_threshold_receiver_refused(self):
        """
        A threshold receiver is refused by name, and the message enumerates what a click record
        cannot supply.
        """
        args = (N_EM, 1e6, 1e5, E_BIT, 3e-4, 4e-8, 4e-12, 0.9, EPS, EPS, ZETA, ZETA_EC, F_EC, True)
        for needle in (
            'detector="threshold" is refused',
            "PHOTON-NUMBER-RESOLVING",
            "EXACTLY ONE photon",
            "THE PROTOCOL IS BLOCK-WISE",
            "CODE/SAMPLE COIN",
            "SYMMETRIC PAIR",
            "src/pnr.rs",
            "dps_rate",
        ):
            self.assertFails(
                NotImplementedError,
                needle,
                _core.dps_finite,
                *(args + ("threshold",)),
                msg=f"the refusal must name {needle!r}",
            )

    def test_unknown_receiver_refused(self):
        """
        A receiver name other than the two the engine knows is refused rather than silently
        treated as resolving.
        """
        args = (N_EM, 1e6, 1e5, E_BIT, 3e-4, 4e-8, 4e-12, 0.9, EPS, EPS, ZETA, ZETA_EC, F_EC, True)
        for name in ("", "PNR", "pnr ", "snspd"):
            self.assertFails(
                ValueError,
                "detector must be",
                _core.dps_finite,
                *(args + (name,)),
                msg=f"detector={name!r} must be refused",
            )

    def test_tail_order_enforced(self):
        """
        Source bounds handed over out of order are refused, because reading them as three
        independent weights would understate the phase error.
        """
        self.assertBad(
            "q1 >= q2 >= q3",
            _core.dps_phase,
            (N_EM, 1e6, 1e5, E_BIT, 4e-8, 3e-4, 4e-12, 0.9, EPS, EPS, True),
            msg="q2 above q1 must be refused",
        )
        self.assertBad(
            "q1 >= q2 >= q3",
            _core.dps_phase,
            (N_EM, 1e6, 1e5, E_BIT, 3e-4, 4e-12, 4e-8, 0.9, EPS, EPS, True),
            msg="q3 above q2 must be refused",
        )

    def test_empty_run_refused(self):
        """
        A run that detected nothing is refused rather than reported as a zero phase error.
        """
        self.assertBad(
            "n_code + n_samp must be > 0",
            _core.dps_phase,
            (N_EM, 0.0, 0.0, E_BIT, 3e-4, 4e-8, 4e-12, 0.9, EPS, EPS, True),
            msg="no detected round must be refused",
        )
        self.assertBad(
            "n_code must be > 0",
            _core.dps_finite,
            (N_EM, 0.0, 1e5, E_BIT, 3e-4, 4e-8, 4e-12, 0.9, EPS, EPS, ZETA, ZETA_EC, F_EC, True, "pnr"),
            msg="no code round must be refused",
        )

    def test_prediction_below_half(self):
        """
        Kato's prediction must sit below half the trials, which is the condition his analytic
        optimum is solved under.
        """
        for bad in (5e5, 1e6, 2e6):
            self.assertBad(
                "m must be < n/2",
                _core.dps_kato,
                (1e6, bad, EPS),
                msg=f"m = {bad} at n = 1e6 must be refused",
            )

    def test_domain_guards(self):
        """
        Every argument of `dps_phase` is refused by name outside its own domain.
        """
        ok = (N_EM, 1e6, 1e5, E_BIT, 3e-4, 4e-8, 4e-12, 0.9, EPS, EPS, True)
        self.assertSlots(
            _core.dps_phase,
            ok,
            (
                (0, "n_em", (0.0, -1.0, float("nan"))),
                (1, "n_code", (-1.0, float("inf"))),
                (2, "n_samp", (-1.0, float("nan"))),
                (3, "e_bit", (-1e-9, 0.5000001, 1.0)),
                (7, "t", (0.0, 1.0, -0.5)),
                (8, "eps1", (0.0, 1.0, 2.0)),
                (9, "eps2", (0.0, 1.0, -1e-9)),
            ),
            msg="dps_phase",
        )


if __name__ == "__main__":
    sys.exit(
        Exam(
            "DpsFinite",
            "Phase 9: MTT23 finite-key DPS length, and the receiver it refuses",
            "dps_finite.md",
        ).run(load(Finite))
    )
