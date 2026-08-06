import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, load

from kit.fuzzing import (
    AXIS,
    DECADES,
    DOWN,
    NASTY,
    SEEDS,
    UP,
    Fuzzed,
    Sweep,
    grid,
    halve,
    loguni,
    rng,
    square,
    unit,
)
from qkd import Connector, Coupling, Splice, _core, budget
from qkd._core import FockState, GaussianState

# Exterior values: one ulp outside the domain, then the non-finite ones.
NONNEG = (-5e-324,) + NASTY

POSITIVE = (0.0, -1.0) + NASTY

CLOSED = (UP, -5e-324) + NASTY

OPEN = (0.0, UP) + NASTY

OPENNEG = (0.0, UP, -5e-324) + NASTY

SIGNED = (-5e-324, -1.0) + NASTY

HALF = (math.nextafter(0.5, 1.0), 1.0, -5e-324) + NASTY

# run_symbols past (n, seed): va t xi eta vel lw_alice lw_lo cfo symbol_rate pilot_db pilot_frac block frames.
PIPE = (5.0, 0.5, 0.01, 0.6, 0.1, 0.0, 0.0, 0.0, 1e9, 12.0, 0.25, 32, False)


def hardware(r):
    """
    One CV hardware point -- V_A, T, xi, eta, v_el -- drawn in that order.
    """

    return (
        loguni(r, 1e-3, 1e3),
        unit(r),
        loguni(r, 1e-9, 1.0),
        unit(r, 1e-3),
        r.uniform(0.0, 1.0),
    )


class FuzzCv(Fuzzed):
    """
    cv_rate, cv_bounds, cv_finite, and qkd.budget, which has no Rust guard behind it.
    """

    def test_cv_interior(self):
        """
        cv_rate over 600 interior hardware points is finite, with I_AB, chi_BE >= 0 and key = beta*I_AB - chi_BE.
        """
        s = Sweep(1, 600)
        for r in s:
            arg = hardware(r)
            beta = r.uniform(0.0, 1.0)
            hom = r.random() < 0.5
            got = _core.cv_rate(*arg, beta, hom, r.random() < 0.5)
            self.assertFinite(got, msg=f"cv_rate{arg + (beta, hom)} is not finite")
            self.assertGreaterEqual(got[0], 0.0, msg=f"I_AB < 0 at {arg}")
            self.assertGreaterEqual(got[1], -1e-12, msg=f"chi_BE < 0 at {arg}")
            self.assertClose(
                got[2],
                beta * got[0] - got[1],
                msg=f"key != beta*I_AB - chi_BE at {arg}",
            )
        self.assertSwept(s, 600, msg="interior sweep draw count")

    def test_cv_boundary(self):
        """
        cv_rate is finite on its closed edges and one ulp inside them, and at T = 1 its key neither
        rises nor steps as xi leaves 0.
        """
        edges = (
            (1.0, 0.0, 1.0, 0.0, 1.0),
            (DOWN, 0.0, 1.0, 0.0, 1.0),
            (1.0, 5e-324, 1.0, 0.0, 1.0),
            (1.0, 0.0, DOWN, 0.0, 1.0),
            (1.0, 0.0, 1.0, 5e-324, 1.0),
            (1.0, 0.0, 1.0, 0.0, DOWN),
            (1.0, 0.0, 1.0, 0.0, 0.0),
        )
        for hom in (True, False):
            for t, xi, eta, vel, beta in edges:
                got = _core.cv_rate(2.0, t, xi, eta, vel, beta, hom, True)
                self.assertFinite(got, msg=f"edge T={t} xi={xi} eta={eta} beta={beta}")

            near = [_core.cv_rate(2.0, 1.0, x, 1.0, 0.0, 1.0, hom, True)[2] for x in (0.0, 5e-324, 1e-15, 1e-12)]
            self.assertMonotone(near, rising=False, strict=False, msg="xi = 0 corner")
            self.assertClose(near[0], near[3], atol=1e-6, msg="no step at xi = 0")

    def test_cv_exterior(self):
        """
        cv_rate's 33 exterior points, and cv_bounds and cv_finite outside their domains, raise a
        ValueError naming the parameter.
        """
        ok = (2.0, 0.5, 0.01, 0.6, 0.1, 0.95, True, True)
        cases = (
            (0, "va must be", (0.0, -1e-300) + NASTY),
            (1, "t must be", OPENNEG),
            (2, "xi must be", SIGNED),
            (3, "eta must be", OPENNEG),
            (4, "vel must be", SIGNED),
            (5, "beta must be", (UP, -5e-324, 2.0) + NASTY),
        )
        self.assertSlots(_core.cv_rate, ok, cases, msg="cv_rate")
        self.assertSwept(
            sum(len(b) for _, _, b in cases),
            33,
            msg="exterior sweep case count",
        )

        est = (0.5, 1.0, 1e6, 5.0, 1e-10)
        bounds = (
            (0, "t_hat must be", NONNEG),
            (1, "sigma2_hat must be", (DOWN, 0.0, -1.0) + NASTY),
            (2, "m must be", POSITIVE),
            (3, "va must be", POSITIVE),
            (4, "eps must be", (0.0, 1.0, UP) + NASTY),
        )
        self.assertSlots(_core.cv_bounds, est, bounds, msg="cv_bounds")

        good = (5.0, 0.3, 0.02, 0.6, 0.1, 0.95, True, True)
        wild = (0.0, 1.0, UP, float("nan"))
        block = (
            (8, "n_total must be", (0.0,)),
            (9, "pe_fraction must be", wild),
            (10, "must be in (0, 1)", wild),
        )
        self.assertSlots(
            _core.cv_finite,
            good + (1e9, 0.5, 1e-10, 1e-10, 1e-10),
            block,
            msg="cv_finite",
        )

    def test_cv_extreme(self):
        """
        Over 80 points with V_A across ten decades I_AB is finite, chi_BE and key are not NaN, and
        past V_A = 1e3 or xi = 1 the key is not positive.
        """
        count = 0
        for va in DECADES:
            for xi in (0.0, 1e-12, 1.0, 1e12):
                for hom in (True, False):
                    i_ab, chi, key = _core.cv_rate(va, 0.5, xi, 0.6, 0.1, 0.95, hom, True)
                    count += 1
                    self.assertFinite(i_ab, msg=f"I_AB at V_A={va} xi={xi}")
                    self.assertFalse(math.isnan(chi), msg=f"chi_BE is NaN at V_A={va}")
                    self.assertFalse(math.isnan(key), msg=f"key is NaN at V_A={va}")

                    if va > 1e3 or xi > 1.0:
                        self.assertLessEqual(key, 0.0, msg=f"V_A={va} xi={xi} yielded key")
        self.assertSwept(count, 80, msg="extreme sweep count")

    def test_cv_estimate(self):
        """
        Over 400 blocks cv_bounds gives t_lo in [0, t_hat] and xi_hi >= 0, and cv_finite a key >= 0,
        t_min in [0, T] and Delta > 0.
        """
        r = rng(2)
        count = 0
        for _ in range(200):
            hat = r.uniform(0.0, 1.0)
            s2 = 1.0 + loguni(r, 1e-9, 1e3)
            m = loguni(r, 1.0, 1e14)
            arg = (hat, s2, m, loguni(r, 1e-3, 1e6), loguni(r, 1e-12, 0.5))
            t_lo, xi_hi = _core.cv_bounds(*arg)
            count += 1
            self.assertRange(t_lo, 0.0, hat, msg=f"t_lo at {arg}")
            self.assertGreaterEqual(xi_hi, 0.0, msg=f"xi_hi < 0 at {arg}")

        for _ in range(200):
            arg = hardware(r)
            eps = loguni(r, 1e-12, 0.4)
            out = _core.cv_finite(
                *arg,
                0.95,
                r.random() < 0.5,
                True,
                loguni(r, 1e2, 1e16),
                r.uniform(0.05, 0.95),
                eps,
                eps,
                eps,
            )
            count += 1
            self.assertFinite(out[0], msg=f"I_AB not finite at {arg}")
            self.assertGreaterEqual(out[2], 0.0, msg=f"finite key below 0 at {arg}")
            self.assertRange(out[3], 0.0, arg[1], msg=f"t_min at {arg}")
            self.assertFinite(out[5], msg=f"Delta not finite at {arg}")
            self.assertGreater(out[5], 0.0, msg=f"Delta <= 0 at {arg}")
        self.assertSwept(count, 400, msg="estimator sweep count")

    def test_cv_quantile(self):
        """
        z_pe over 400 failure probabilities is finite, falling and positive, above 30 at eps =
        1e-300 where 1 - eps/2 rounds to 1, 1.959964 at eps = 0.05, and refuses eps at 0 or 1.
        """
        r = rng(21)
        eps = sorted(loguni(r, 1e-300, 0.999) for _ in range(400))
        zs = [_core.z_pe(e) for e in eps]
        self.assertFinite(zs, msg="z_pe not finite")
        self.assertMonotone(zs, rising=False, strict=False, msg="z_pe not falling")
        self.assertGreater(min(zs), 0.0, msg="z_pe not positive")
        self.assertGreater(_core.z_pe(1e-300), 30.0, msg="z_pe(1e-300) below 30")
        self.assertClose(_core.z_pe(0.05), 1.959964, atol=1e-5, msg="z_pe(0.05) != 1.959964")
        self.assertSlots(
            _core.z_pe,
            (0.05,),
            ((0, "eps must be", (0.0, 1.0, UP, -5e-324) + NASTY),),
            msg="z_pe",
        )

    def test_budget_exterior(self):
        """
        budget.assemble raises on 35 non-finite loss or hardware parameters and on a loss chain
        whose decibels underflow T, and admits v_err = +inf with an infinite total.
        """

        def fit(**kw):
            kw.setdefault("v_a", 5.0)
            kw.setdefault("t", 0.5)

            return budget.assemble(**kw)

        count = 0
        for kind in (Connector, Coupling, Splice):
            for bad in (math.nan, math.inf):
                count += 1
                self.assertFails(
                    ValueError,
                    "loss must be finite",
                    lambda k=kind, x=bad: fit(v_err=2e-3, losses=(k(loss=x),)),
                    msg=f"{kind.__name__}(loss={bad}) assembled",
                )
            self.assertFails(
                ValueError,
                "nonnegative",
                lambda k=kind: k(loss=float("-inf")),
                msg=f"{kind.__name__}(loss=-inf) constructed",
            )

        # v_err admits +inf.
        cases = (
            ("v_a", NASTY, lambda x: fit(v_a=x, v_err=2e-3)),
            ("t must be in (0, 1]", NASTY, lambda x: fit(t=x)),
            ("v_err", (math.nan, -math.inf), lambda x: fit(v_err=x)),
            ("xi", NASTY, lambda x: fit(v_err=2e-3, xi=x)),
            ("dac_bits", NASTY, lambda x: fit(dac_bits=x)),
            ("adc_bits", NASTY, lambda x: fit(adc_bits=x)),
            ("rin", NASTY, lambda x: fit(rin=x, bandwidth=1e8)),
            ("bandwidth", NASTY, lambda x: fit(rin=-155.0, bandwidth=x)),
            ("mu", NASTY, lambda x: fit(adc_bits=12, mu=x)),
            ("ratio", NASTY, lambda x: fit(adc_bits=12, ratio=x)),
        )
        for needle, bads, build in cases:
            for bad in bads:
                count += 1
                self.assertFails(
                    ValueError,
                    needle,
                    build,
                    bad,
                    msg=f"assemble {needle} = {bad} accepted",
                )
        self.assertSwept(count, 35, msg="budget exterior case count")
        self.assertFails(
            ValueError,
            "underflows the transmittance",
            lambda: fit(v_err=2e-3, adc_bits=12, losses=(Coupling(loss=1e5),)),
            msg="1e5 dB chain accepted",
        )
        self.assertEqual(
            fit(v_err=math.inf).total,
            math.inf,
            msg="v_err = inf total",
        )


class FuzzRate(Fuzzed):
    """
    dps_rate, bb84_rate, cow_rate, cow_ceiling and cow_visibility.
    """

    def test_dps_contract(self):
        """
        dps_rate over 300 points is finite and inside [0, p_click]; an f_ec below 1, a
        probability outside [0, 1] and mu <= 0 all raise.
        """
        s = Sweep(3, 300)
        for r in s:
            p = r.uniform(0.0, 1.0)
            arg = (
                p,
                r.uniform(0.0, 1.0),
                loguni(r, 1e-6, 10.0),
                1.0 + loguni(r, 1e-9, 1.0),
            )
            got = _core.dps_rate(*arg)
            self.assertFinite(got, msg=f"dps_rate{arg} is not finite")
            self.assertRange(got, 0.0, p + 1e-12, msg=f"dps_rate{arg}")

        cases = (
            (0, "p_click must be", (-5e-324, UP) + NASTY),
            (1, "qber must be", (-5e-324, UP) + NASTY),
            (2, "mu must be", (0.0, -5e-324) + NASTY),
            (3, "f_ec must be >= 1", (0.0, 0.5, DOWN, -1.0, float("nan"))),
        )
        self.assertSlots(_core.dps_rate, (0.5, 0.02, 0.2, 1.16), cases, msg="dps_rate")
        self.assertSwept(s, 300, msg="dps sweep draw count")

    def test_bb84_contract(self):
        """
        bb84_rate over 300 points is finite and inside [0, q_sift*q1]; q1 above q_mu, an error
        rate past 1/2 and an f_ec below 1 raise.
        """
        s = Sweep(4, 300)
        for r in s:
            q_mu = r.uniform(0.0, 1.0)
            q1 = r.uniform(0.0, q_mu)
            sift = r.uniform(1e-6, 1.0)
            arg = (
                sift,
                q_mu,
                r.uniform(0.0, 0.5),
                q1,
                r.uniform(0.0, 0.5),
                1.0 + r.uniform(0.0, 1.0),
            )
            got = _core.bb84_rate(*arg)
            self.assertFinite(got, msg=f"bb84_rate{arg} is not finite")
            self.assertRange(got, 0.0, sift * q1 + 1e-12, msg=f"bb84_rate{arg}")
        self.assertBad(
            "q1 must not exceed q_mu",
            _core.bb84_rate,
            (0.5, 0.1, 0.02, math.nextafter(0.1, 1.0), 0.02, 1.16),
            msg="q1 above q_mu accepted",
        )

        cases = (
            (2, "e_mu must be in [0, 1/2]", HALF),
            (4, "e1 must be in [0, 1/2]", HALF),
            (5, "f_ec must be >= 1", (0.0, DOWN, -1.0) + NASTY),
        )
        self.assertSlots(
            _core.bb84_rate,
            (0.5, 0.1, 0.02, 0.05, 0.02, 1.16),
            cases,
            msg="bb84_rate",
        )
        self.assertSwept(s, 300, msg="bb84 sweep draw count")

    def test_cow_contract(self):
        """
        Over 300 points cow_rate stays in [0, q_z], cow_ceiling in [0, 1] and cow_visibility
        in [-1, 1]; a monitoring line that never clicked and an error rate past 1/2 raise.
        """
        s = Sweep(5, 300)
        for r in s:
            q_z = r.uniform(0.0, 1.0)
            arg = (
                q_z,
                r.uniform(0.0, 0.5),
                r.uniform(0.0, 0.5),
                1.0 + r.uniform(0.0, 1.0),
            )
            got = _core.cow_rate(*arg)
            cap = _core.cow_ceiling(r.uniform(0.0, 1.0), unit(r), loguni(r, 1e-3, 10.0))
            vis = _core.cow_visibility(r.uniform(0.0, 10.0), r.uniform(0.0, 10.0))
            self.assertFinite(got, msg=f"cow_rate{arg} is not finite")
            self.assertRange(got, 0.0, q_z + 1e-12, msg=f"cow_rate{arg}")
            self.assertRange(cap, 0.0, 1.0, msg="cow_ceiling outside [0, 1]")
            self.assertRange(vis, -1.0, 1.0, msg="cow_visibility outside [-1, 1]")

        cases = (
            (1, "e_bit must be in [0, 1/2]", HALF),
            (2, "e_phase must be in [0, 1/2]", HALF),
        )
        self.assertSlots(_core.cow_rate, (0.5, 0.02, 0.02, 1.16), cases, msg="cow_rate")
        self.assertBad(
            "monitoring line never clicked",
            _core.cow_visibility,
            (0.0, 0.0),
            msg="dark monitoring line accepted",
        )
        self.assertSlots(
            _core.cow_visibility,
            (0.5, 0.5),
            ((0, "p_m0 must be", NONNEG),),
            msg="cow_visibility",
        )
        self.assertSwept(s, 300, msg="cow sweep draw count")


class FuzzDecoy(Fuzzed):
    """
    The decoy forward model and the single-photon bounds that invert it.
    """

    def test_decoy_interior(self):
        """
        Over 400 self-consistent decoy experiments, eta and Y1 lie in [0, 1], the QBER and the
        e1 bound in [0, 1/2], and Q1 is no larger than Q_mu.
        """
        s = Sweep(6, 400)
        for r in s:
            eta = _core.decoy_eta(r.uniform(0.0, 0.5), r.uniform(0.0, 400.0), r.uniform(0.0, 1.0))
            y0 = loguni(r, 1e-9, 1e-2)
            det = r.uniform(0.0, 0.2)
            mu = r.uniform(0.05, 1.0)
            nu1 = r.uniform(0.01, 0.9) * mu
            nu2 = r.uniform(0.0, 0.9) * nu1
            if nu1 + nu2 >= mu or nu2 >= nu1:
                continue

            q_mu, e_mu = _core.decoy_gain(mu, eta, y0, det)
            q_n1, e_n1 = _core.decoy_gain(nu1, eta, y0, det)
            q_n2, e_n2 = _core.decoy_gain(nu2, eta, y0, det) if nu2 > 0.0 else (y0, 0.5)
            y1, e1, q1 = _core.decoy_bounds(mu, nu1, nu2, q_mu, e_mu, q_n1, e_n1, q_n2, e_n2)
            s.keep()
            self.assertRange(eta, 0.0, 1.0, msg="decoy_eta outside [0, 1]")
            self.assertLessEqual(q_mu, 1.0, msg="Q_mu above 1")
            self.assertLessEqual(e_mu, 0.5, msg="e_mu above 1/2")
            self.assertFinite((y1, e1, q1), msg=f"decoy_bounds at mu={mu} nu={nu1},{nu2}")
            self.assertRange(y1, 0.0, 1.0, msg="Y1 outside [0, 1]")
            self.assertRange(e1, 0.0, 0.5, msg="e1 outside [0, 1/2]")
            self.assertLessEqual(q1, q_mu + 1e-12, msg="Q1 above Q_mu")
        self.assertHoles(s, 200, msg="decoy sweep kept count")

    def test_decoy_ceiling(self):
        """
        Y1 from decoy_bounds on decoy_gain's own gains stays under decoy_ideal's infinite-decoy
        ceiling, and e1 above its floor, over 1000 channels plus one pinned reproducer.
        """
        s = Sweep(7, 1001)
        pinned = (0.1259083008573008, 0.0021856562547432584, 0.0010489833831033714)
        for k, r in enumerate(s):
            if k == 0:
                mu, nu1, nu2 = pinned
                eta, y0, det = 0.004879903732625898, 0.0023251437467200937, 0.03
            else:
                eta = loguni(r, 1e-4, 1e-1)
                y0 = loguni(r, 1e-4, 1e-2)
                det = r.uniform(0.0, 0.2)
                mu = r.uniform(0.05, 1.0)
                nu1 = r.uniform(0.01, 0.8) * mu
                nu2 = r.uniform(0.0, 0.8) * nu1
            if nu1 + nu2 >= mu or nu2 >= nu1:
                continue

            gains = []
            for x in (mu, nu1, nu2):
                gains.extend(_core.decoy_gain(x, eta, y0, det) if x > 0.0 else (y0, 0.5))
            y1, e1, _q1 = _core.decoy_bounds(mu, nu1, nu2, *gains)
            top, floor = _core.decoy_ideal(eta, y0, det)
            s.keep()
            self.assertLessEqual(
                y1,
                top * (1.0 + 1e-9),
                msg=f"Y1 bound {y1:.12g} above the infinite-decoy ceiling {top:.12g} at "
                f"mu={mu!r} nu1={nu1!r} nu2={nu2!r} eta={eta!r} y0={y0!r}",
            )

            if y1 > 0.0:
                self.assertGreaterEqual(
                    e1,
                    min(floor, 0.5) * (1.0 - 1e-9),
                    msg=f"e1 below the ideal floor at mu={mu}",
                )
        self.assertHoles(s, 500, msg="ceiling sweep kept count")

    def test_decoy_boundary(self):
        """
        decoy_bounds is finite with Y1 >= 0 at nu2 = 0, nu1 + nu2 just below mu and denormal
        intensities, and decoy_eta and decoy_ideal reach their closed-form edge values.
        """
        cases = (
            (0.5, 0.1, 0.0),
            (0.5, math.nextafter(0.5, 0.0), 0.0),
            (0.5, 0.3, 0.19999999999999),
            (5e-324 * 4, 5e-324 * 2, 5e-324),
        )
        for mu, nu1, nu2 in cases:
            q_mu, e_mu = _core.decoy_gain(mu, 0.1, 1e-6, 0.02)
            got = _core.decoy_bounds(mu, nu1, nu2, q_mu, e_mu, 0.01, 0.02, 1e-6, 0.5)
            self.assertFinite(got, msg=f"mu={mu} nu1={nu1} nu2={nu2}")
            self.assertGreaterEqual(got[0], 0.0, msg="Y1 negative")
        self.assertClose(_core.decoy_eta(0.2, 0.0, 1.0), 1.0, msg="decoy_eta(0 km) != 1")
        self.assertClose(_core.decoy_eta(0.2, 0.0, 0.0), 0.0, msg="decoy_eta(det=0) != 0")
        self.assertClose(
            _core.decoy_ideal(0.0, 0.0, 0.0)[1],
            0.5,
            msg="decoy_ideal floor != 0.5",
        )
        self.assertClose(_core.decoy_ideal(1.0, 1.0, 0.0)[0], 1.0, msg="decoy_ideal ceiling != 1")

    def test_decoy_exterior(self):
        """
        decoy_bounds refuses nu2 >= nu1, nu1 + nu2 >= mu and out-of-range intensities or gains,
        and decoy_eta, decoy_ideal and decoy_gain refuse their exterior.
        """
        ok = (0.5, 0.2, 0.1, 0.3, 0.02, 0.1, 0.02, 1e-6, 0.5)
        self.assertFails(
            ValueError,
            "nu2 < nu1",
            lambda: _core.decoy_bounds(0.5, 0.1, 0.1, *ok[3:]),
            msg="nu1 == nu2 accepted",
        )
        self.assertFails(
            ValueError,
            "nu2 < nu1",
            lambda: _core.decoy_bounds(0.5, 0.1, math.nextafter(0.1, 1.0), *ok[3:]),
            msg="nu2 above nu1 accepted",
        )
        self.assertFails(
            ValueError,
            "nu1 + nu2 < mu",
            lambda: _core.decoy_bounds(0.3, 0.2, 0.1, *ok[3:]),
            msg="nu1 + nu2 >= mu accepted",
        )

        cases = (
            (0, "mu must be", POSITIVE),
            (1, "nu1 must be", POSITIVE),
            (2, "nu2 must be", NONNEG),
            (3, "q_mu must be", CLOSED),
            (4, "e_mu must be", CLOSED),
        )
        self.assertSlots(_core.decoy_bounds, ok, cases, msg="decoy_bounds")

        guards = (
            (_core.decoy_eta, (0.2, 10.0, 0.5), 0, "alpha must be", NONNEG),
            (_core.decoy_eta, (0.2, 10.0, 0.5), 1, "length must be", NONNEG),
            (_core.decoy_ideal, (0.5, 1e-6, 0.02), 0, "eta must be", CLOSED),
            (_core.decoy_gain, (0.5, 0.1, 1e-6, 0.02), 2, "y0 must be", CLOSED),
        )
        self.assertGuards(guards, msg="decoy")


class FuzzRelay(Fuzzed):
    """
    The two-arm equivalent noise and its rate, the post-relay covariance, and the
    interferometric front end.
    """

    def test_relay_noise(self):
        """
        Over 300 two-arm attacks the equivalent noise is positive and cvmdi_least stays under
        cvmdi_floor; two pure-loss arms sit on the floor, and cvmdi_rate raises at the
        chi = 4 pole of the symmetric branch.
        """
        s = Sweep(8, 300)
        for r in s:
            ta, tb = unit(r, 1e-3), unit(r, 1e-3)
            wa = 1.0 + loguni(r, 1e-9, 10.0)
            wb = 1.0 + loguni(r, 1e-9, 10.0)
            # Entangling-cloner cap: sqrt((omega_a - 1)(omega_b - 1)).
            cap = math.sqrt((wa - 1.0) * (wb - 1.0))
            g = r.uniform(-1.0, 1.0) * cap
            gp = r.uniform(-1.0, 1.0) * cap
            try:
                chi = _core.cvmdi_noise(ta, tb, wa, wb, g, gp)
            except ValueError:
                continue

            least = _core.cvmdi_least(ta, tb)
            s.keep()
            self.assertFinite(chi, msg=f"chi at tau={ta},{tb} omega={wa},{wb}")
            self.assertGreater(chi, 0.0, msg="chi not positive")
            self.assertLessEqual(least, _core.cvmdi_floor(ta, tb), msg="cvmdi_least above cvmdi_floor")

            if chi > max(least, 4.0) * (1.0 + 1e-9):
                self.assertFinite(
                    _core.cvmdi_rate(ta, tb, chi),
                    msg=f"rate at tau={ta},{tb} chi={chi}",
                )
        self.assertHoles(s, 150, msg="relay noise kept count")
        self.assertClose(
            _core.cvmdi_noise(0.5, 0.5, 1.0, 1.0, 0.0, 0.0),
            _core.cvmdi_floor(0.5, 0.5),
            msg="pure-loss chi != cvmdi_floor",
        )
        self.assertBad(
            "is at or below",
            _core.cvmdi_rate,
            (0.5, 0.5, 4.0),
            msg="chi = 4 accepted",
        )

    def test_relay_cov(self):
        """
        Over 150 relay points the post-relay covariance satisfies V + i*Omega/2 >= 0 once
        halved into the internal convention, and cvmdi_point obeys key = beta*I_AB - chi.
        """
        s = Sweep(9, 150)
        for r in s:
            va = loguni(r, 1e-3, 1e5)
            ta, tb = unit(r, 1e-2), unit(r, 1e-2)
            wa = 1.0 + loguni(r, 1e-9, 5.0)
            wb = 1.0 + loguni(r, 1e-9, 5.0)
            beta = r.uniform(0.0, 1.0)
            try:
                v = _core.cvmdi_cov(va, ta, tb, wa, wb, 0.0, 0.0)
                i_ab, chi, key = _core.cvmdi_point(va, ta, tb, wa, wb, 0.0, 0.0, beta)
            except ValueError:
                continue

            s.keep()
            self.assertFinite(v, msg=f"covariance at va={va} tau={ta},{tb}")
            self.assertPhysical(square(halve(v)), msg=f"unphysical relay state at va={va}")
            self.assertFinite((i_ab, chi), msg=f"relay point at va={va}")
            self.assertGreaterEqual(i_ab, -1e-12, msg="I_AB below 0")
            self.assertClose(key, beta * i_ab - chi, atol=1e-9, msg="key != beta*I_AB - chi")
        self.assertHoles(s, 100, msg="relay covariance kept count")

    def test_relay_optics(self):
        """
        Over 400 front-end points mode_overlap lies in [0, 1], hom_visibility at most 1/2 rather
        than 1, relay_ports sum to the incoming flux, relay_counts lie in [0, 1] and relay_error in
        [0, 1/2].
        """
        s = Sweep(10, 400)
        for r in s:
            fw = loguni(r, 1e-3, 1e3)
            over = _core.mode_overlap(fw, loguni(r, 1e-3, 1e3), r.uniform(-100.0, 100.0))
            mu_a, mu_b = loguni(r, 1e-6, 10.0), loguni(r, 1e-6, 10.0)
            vis = _core.hom_visibility(over, mu_a, mu_b)
            ports = _core.relay_ports(mu_a, mu_b, r.uniform(-20.0, 20.0), r.uniform(0.0, 1.0))
            counts = _core.relay_counts(ports[0], ports[1], unit(r, 1e-3), r.uniform(0.0, 0.999))
            err = _core.relay_error(mu_a, mu_b, r.uniform(0.0, 1.0), unit(r, 1e-3), r.uniform(0.0, 0.999))
            self.assertRange(over, 0.0, 1.0, msg="mode_overlap outside [0, 1]")
            self.assertLessEqual(vis, 0.5 + 1e-12, msg=f"HOM visibility {vis} above 1/2")
            self.assertGreaterEqual(min(ports), -1e-12, msg="negative port intensity")
            self.assertClose(
                ports[0] + ports[1],
                mu_a + mu_b,
                atol=1e-9 * (mu_a + mu_b),
                msg="ports do not sum to the flux",
            )
            self.assertGreaterEqual(min(counts), 0.0, msg="negative relay count")
            self.assertLessEqual(max(counts), 1.0, msg="relay count above 1")
            self.assertRange(err, 0.0, 0.5 + 1e-12, msg="relay_error outside [0, 1/2]")
        self.assertClose(_core.mode_overlap(1.0, 1.0, 0.0), 1.0, msg="mode_overlap(1, 1, 0) != 1")
        self.assertClose(_core.hom_visibility(1.0, 0.2, 0.2), 0.5, msg="hom_visibility != 1/2")
        self.assertSwept(s, 400, msg="optics sweep draw count")

    def test_relay_exterior(self):
        """
        Relay arguments outside their domains raise: a transmissivity at 0 or above 1, a
        lossless arm asked to carry xi, a sub-vacuum environment variance, a correlation at
        the positivity cap, and correlations past the uncertainty principle.
        """
        sub = (DOWN, 0.0, -1.0) + NASTY
        arms = (0.5, 0.5, 1.0, 1.0, 0.0, 0.0)
        guards = (
            (_core.relay_omega, (0.5, 0.01), 0, "tau must be", OPENNEG),
            (_core.cvmdi_floor, (0.5, 0.5), 0, "tau_a must be", OPENNEG),
            (_core.cvmdi_noise, arms, 2, "environment variance", sub),
        )
        self.assertGuards(guards, msg="relay")
        self.assertBad(
            "lossless arm",
            _core.relay_omega,
            (1.0, 5e-324),
            msg="lossless arm with xi accepted",
        )
        self.assertClose(_core.relay_omega(1.0, 0.0), 1.0, msg="relay_omega(1, 0) != 1")
        self.assertBad(
            "must be < sqrt(omega_a*omega_b)",
            _core.cvmdi_noise,
            (0.5, 0.5, 2.0, 2.0, 2.0, 0.0),
            msg="correlation at the cap accepted",
        )
        self.assertBad(
            "uncertainty principle",
            _core.cvmdi_noise,
            (0.5, 0.5, 1.0, 1.0, 0.9, 0.9),
            msg="g = gp = 0.9 accepted",
        )


class FuzzDmcs(Fuzzed):
    """
    The M-PSK constellation, its moments, its mutual information, and the rate its certified
    correlation bound supports.
    """

    def test_dm_interior(self):
        """
        Over 120 constellations of 3 to 64 states every amplitude lies on the circle, V_A =
        2*alpha^2, the weight is >= 0, z_lin <= z_gauss and I <= min(I_gauss, log2 m).
        """
        s = Sweep(12, 120)
        for r in s:
            m = r.randint(3, 64)
            alpha = loguni(r, 1e-5, 5.0)
            pts = _core.dm_states(m, alpha)
            va, z_lin, w, z_g = _core.dm_moments(m, alpha)
            exact, gauss = _core.dm_info(m, alpha, unit(r), loguni(r, 1e-9, 2.0))
            self.assertEqual(len(pts), m, msg=f"m = {m} states")
            self.assertClose(
                pts[0][0],
                alpha,
                atol=1e-12 * max(1.0, alpha),
                msg="first state not real",
            )

            for re, im in pts:
                self.assertClose(
                    math.hypot(re, im),
                    alpha,
                    atol=1e-9 * max(1.0, alpha),
                    msg="state off the circle",
                )
            self.assertClose(
                va,
                2.0 * alpha * alpha,
                atol=1e-12 * max(1.0, va),
                msg="V_A != 2 alpha^2",
            )
            self.assertGreaterEqual(w, -1e-12, msg="weight below 0")
            self.assertLessEqual(
                z_lin,
                z_g * (1.0 + 1e-9),
                msg="z_lin above z_gauss",
            )
            self.assertRange(
                exact,
                -1e-12,
                min(gauss + 1e-9, math.log2(m) + 1e-9),
                msg="I outside [0, min(gauss, log2 m)]",
            )
        self.assertSwept(s, 120, msg="constellation sweep draw count")

    def test_dm_rate(self):
        """
        Over 180 M-PSK rates the certified correlation z is clamped at 0, I_AB <= log2(m), chi_BE
        is not NaN and key <= beta*I_AB.
        """
        s = Sweep(13, 180)
        for r in s:
            m = r.choice((3, 4, 5, 6, 8, 12, 16, 32, 64))
            arg = (
                m,
                loguni(r, 1e-4, 3.0),
                unit(r, 1e-3),
                loguni(r, 1e-9, 1.0),
                unit(r, 1e-2),
                r.uniform(0.0, 0.5),
            )
            beta = r.uniform(0.0, 1.0)
            i_ab, chi, key, z = _core.dm_rate(*arg, beta)
            self.assertFinite((i_ab, z), msg=f"dm_rate{arg} is not finite")
            self.assertGreaterEqual(z, 0.0, msg="z negative")
            self.assertRange(i_ab, -1e-12, math.log2(m) + 1e-9, msg="I_AB")
            self.assertFalse(math.isnan(chi), msg=f"chi_BE is NaN at {arg}")
            self.assertLessEqual(key, beta * i_ab + 1e-9, msg=f"key above beta*I_AB at {arg}")
        self.assertSwept(s, 180, msg="dm rate sweep draw count")

    def test_dm_boundary(self):
        """
        dm_moments is finite at m in {3, 64} and alpha in {1e-6, 100}, a perfect channel keys, and
        dm_holevo evaluates at bits = inf and at z = 0, where the discriminant vanishes.
        """
        for m in (3, 64):
            for alpha in (1e-6, 100.0):
                self.assertFinite(_core.dm_moments(m, alpha), msg=f"moments at m={m} alpha={alpha}")

        for m in (3, 4, 64):
            got = _core.dm_rate(m, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0)
            self.assertFinite((got[0], got[3]), msg=f"perfect channel at m={m}")
            self.assertGreater(got[2], 0.0, msg=f"no key on a perfect link at m={m}")

        wide = _core.dm_holevo(2.0, 1.0, 0.0, math.sqrt(2.0 * 2.0 + 2.0 * 2.0), float("inf"), 1.0)
        self.assertFinite(wide, msg="bits = inf not finite")
        self.assertGreaterEqual(wide[1], -1e-9, msg="chi_BE below 0")

        # Self-comparison on purpose: fails only on NaN, where +inf is a legal chi_BE.
        self.assertClose(
            _core.dm_holevo(2.0, 1.0, 0.0, 0.0, 4.0, 1.0)[1],
            _core.dm_holevo(2.0, 1.0, 0.0, 0.0, 4.0, 1.0)[1],
            msg="chi_BE is NaN at z = 0",
        )

    def test_dm_exterior(self):
        """
        Every M-PSK entry point raises on 2 or 65 states, alpha outside [1e-6, 100], a negative z,
        bits <= 0, and channel arguments outside their domains.
        """
        alphas = (
            0.0,
            math.nextafter(1e-6, 0.0),
            math.nextafter(100.0, 200.0),
            1e6,
        ) + NASTY

        holevo = (2.0, 0.5, 0.01, 1.0, 4.0, 0.95)
        rate = (4, 0.5, 0.5, 0.01, 0.6, 0.0, 0.95)
        guards = (
            (_core.dm_states, (4, 0.5), 0, "m must be in", (0, 1, 2, 65, 1000)),
            (_core.dm_moments, (4, 0.5), 1, "alpha must be in", alphas),
            (_core.dm_holevo, holevo, 3, "z must be", SIGNED),
            (_core.dm_holevo, holevo, 4, "bits must be", (0.0, -1.0, float("nan"))),
            (_core.dm_info, (4, 0.5, 0.5, 0.01), 2, "t must be", OPENNEG),
            (_core.dm_rate, rate, 4, "eta must be", OPENNEG),
        )
        self.assertGuards(guards, msg="dmcs")


class FuzzClick(Fuzzed):
    """
    Threshold detection: the per-pulse click probability, the delay interferometer's ports,
    and the full run whose QBER is an output.
    """

    def test_click_scalar(self):
        """
        Over 400 points click_prob returns a probability and interfere's ports are
        non-negative and sum to half the incoming flux whatever the visibility, nulling at
        perfect contrast with equal amplitudes.
        """
        s = Sweep(14, 400)
        for r in s:
            re, im = r.uniform(-30.0, 30.0), r.uniform(-30.0, 30.0)
            p = _core.click_prob(re, im, unit(r, 1e-9), r.uniform(0.0, 0.999))
            old = (r.uniform(-30.0, 30.0), r.uniform(-30.0, 30.0))
            vis = r.uniform(0.0, 1.0)
            hi, lo = _core.interfere(re, im, old[0], old[1], vis)
            flux = (re * re + im * im + old[0] * old[0] + old[1] * old[1]) / 2.0
            self.assertRange(p, 0.0, 1.0, msg="click_prob outside [0, 1]")
            self.assertGreaterEqual(hi, -1e-9, msg="constructive port negative")
            self.assertGreaterEqual(lo, -1e-9, msg=f"destructive port {lo}")
            self.assertClose(
                hi + lo,
                flux,
                atol=1e-9 * max(1.0, flux),
                msg="ports do not sum to the flux",
            )
        self.assertClose(
            _core.interfere(1.0, 0.0, 1.0, 0.0, 1.0)[1],
            0.0,
            msg="destructive port != 0",
        )
        self.assertClose(_core.click_prob(0.0, 0.0, 1.0, 0.0), 0.0, msg="click_prob(0) != 0")
        self.assertSwept(s, 400, msg="click scalar sweep draw count")

    def test_click_extreme(self):
        """
        Across ten decades of amplitude click_prob is finite and at most 1, and interfere returns
        finite ports or raises, as it does at 1e300.
        """
        for mag in DECADES:
            p = _core.click_prob(mag, 0.0, 1.0, 0.0)
            self.assertFinite(p, msg=f"click_prob at |alpha| = {mag}")
            self.assertLessEqual(p, 1.0, msg=f"click_prob {p} above 1 at {mag}")

            # Finite or raise: at 1e300 the intensity is 1e600.
            try:
                hi, lo = _core.interfere(mag, 0.0, mag, 0.0, 1.0)
            except ValueError:
                continue
            self.assertFinite(
                (hi, lo),
                msg=f"interfere at {mag} returned ({hi}, {lo})",
            )
        self.assertBad(
            "overflows f64",
            _core.interfere,
            (1e300, 0.0, 1e300, 0.0, 1.0),
            msg="interfere at 1e300 accepted",
        )

    def test_click_run(self):
        """
        Over 12 runs every rate and the QBER lie in [0, 1], visibility stays under the configured
        contrast and mu_bob = t*mu, an ideal receiver gives QBER 0, and the per-slot record sums
        to the counters.
        """
        s = Sweep(15, 12)
        for r in s:
            mu = loguni(r, 1e-3, 1.0)
            t = unit(r, 1e-3)
            vis = r.uniform(0.0, 1.0)
            out = _core.run_clicks(
                6000,
                r.getrandbits(63),
                mu,
                t,
                unit(r, 1e-2),
                r.uniform(0.0, 0.1),
                vis,
                r.randint(1, 3),
                loguni(r, 1e2, 1e7),
                1e9,
                0.0,
                r.uniform(0.0, 0.1),
                0,
                False,
            )
            self.assertEqual(out.sifted + 0, out.sifted, msg="sifted is not an integer")
            self.assertLessEqual(out.doubles, out.sifted, msg="doubles above sifted")
            self.assertLessEqual(out.errors, out.sifted, msg="errors above sifted")
            self.assertRange(out.qber, 0.0, 1.0, msg="qber outside [0, 1]")
            self.assertRange(out.sift_rate, 0.0, 1.0, msg="sift_rate outside [0, 1]")
            self.assertLessEqual(out.visibility, vis + 1e-9, msg="visibility above the configured contrast")
            self.assertClose(out.mu_bob, t * mu, atol=1e-12 * max(1.0, t * mu), msg="mu_bob != t*mu")
            self.assertGreaterEqual(out.v_phase, 0.0, msg="v_phase negative")

        clean = _core.run_clicks(4000, 7, 0.2, 1.0, 1.0, 0.0, 1.0, 1, 0.0, 1e9, 0.0, 0.0, 0, False)
        self.assertEqual(clean.qber, 0.0, msg="qber != 0 on a perfect receiver")
        self.assertEqual(clean.doubles, 0, msg="doubles != 0")
        self.assertClose(clean.visibility, 1.0, msg="visibility != 1")

        kept = _core.run_clicks(4000, 9, 0.2, 0.5, 0.9, 1e-4, 0.98, 2, 1e4, 1e9, 0.0, 0.01, 0, True)
        self.assertEqual(len(kept.record_d0), kept.n_slots, msg="record_d0 length != n_slots")
        self.assertEqual(len(kept.record_d1), kept.n_slots, msg="record_d1 length != n_slots")
        self.assertEqual(len(kept.record_bit), kept.n_slots, msg="record_bit length != n_slots")
        self.assertEqual(
            kept.clicks,
            kept.clicks_d0 + kept.clicks_d1,
            msg="clicks != d0 + d1",
        )
        self.assertEqual(
            sum(kept.record_d0) + sum(kept.record_d1),
            kept.clicks,
            msg="record sum != clicks",
        )
        self.assertClose(
            kept.click_rate,
            kept.clicks / kept.n_slots,
            msg="click_rate != clicks/n_slots",
        )
        self.assertSwept(s, 12, msg="click run sweep draw count")

    def test_click_exterior(self):
        """
        run_clicks refuses a dark or afterpulse probability of 1, a zero delay, a train too short
        for its delay, and a non-positive mu or symbol rate.
        """
        ok = (1000, 1, 0.2, 0.5, 0.9, 0.01, 0.98, 1, 0.0, 1e9, 0.0, 0.0, 0, False)
        cases = (
            (2, "mu must be", POSITIVE),
            (3, "t must be", OPEN),
            (4, "eta must be", OPEN),
            (5, "dark must be", (1.0, UP, -5e-324) + NASTY),
            (6, "visibility must be", CLOSED),
            (7, "delay must be >= 1", (0,)),
            (8, "linewidth must be", NONNEG),
            (9, "symbol_rate must be", POSITIVE),
            (10, "dead_time must be", NONNEG),
            (11, "afterpulse must be", (1.0, UP, -5e-324) + NASTY),
        )
        self.assertSlots(_core.run_clicks, ok, cases, msg="run_clicks")
        self.assertFails(
            ValueError,
            "n must exceed delay",
            lambda: _core.run_clicks(3, 1, 0.2, 0.5, 0.9, 0.01, 0.98, 2, 0.0, 1e9, 0.0, 0.0, 0, False),
            msg="n = 3 with delay 2 accepted",
        )

        port = (1.0, 0.0, 1.0, 0.0, 0.5)
        guards = (
            (_core.interfere, port, 0, "re_now must be", NASTY),
            (_core.interfere, port, 4, "vis must be", NASTY),
        )
        self.assertGuards(guards, msg="interfere")


class FuzzGauss(Fuzzed):
    """
    GaussianState: every state the layer hands back satisfies V + i*Omega/2 >= 0.
    """

    def test_state_fuzz(self):
        """
        Over 400 random chains up to five operations deep on one to three modes, every state
        is finite, satisfies V + i*Omega/2 >= 0, keeps its mode count, and obeys
        dx^2 dp^2 >= 1/4 on every mode.
        """
        s = Sweep(16, 400)
        for r in s:
            n = r.randint(1, 3)
            st = GaussianState.vacuum(n)
            if n == 2 and r.random() < 0.5:
                st = GaussianState.epr(r.uniform(-4.0, 4.0))
            if n == 1 and r.random() < 0.5:
                st = GaussianState.thermal(loguni(r, 1e-9, 1e6))

            for _ in range(r.randint(1, 5)):
                mode = r.randrange(n)
                pick = r.choice(("displace", "squeeze", "rotate", "bs", "loss"))
                if pick == "displace":
                    st = st.displace(mode, r.uniform(-50.0, 50.0), r.uniform(-50.0, 50.0))
                elif pick == "squeeze":
                    st = st.squeeze(mode, r.uniform(-4.0, 4.0))
                elif pick == "rotate":
                    st = st.rotate(mode, r.uniform(-20.0, 20.0))
                elif pick == "bs" and n > 1:
                    other = (mode + 1 + r.randrange(n - 1)) % n
                    st = st.bs(mode, other, r.uniform(0.0, 1.0))
                elif pick == "loss":
                    st = st.thermal_loss(
                        mode,
                        r.uniform(0.0, 1.0),
                        loguni(r, 1e-9, 1e3),
                        r.random() < 0.5,
                    )

            cov = square(st.cov())
            self.assertFinite(st.mean(), msg="mean not finite")
            self.assertFinite(st.cov(), msg="covariance not finite")

            # atol 1e-8: the residual tracks ||V||, which thermal-loss chains push to 9.3e6.
            self.assertPhysical(
                cov,
                atol=1e-8,
                msg="not bona fide",
            )
            self.assertUncertainty(cov, msg="uncertainty product below 1/4")
            self.assertEqual(st.n_modes, n, msg="mode count changed")
        self.assertSwept(s, 400, msg="Gaussian chain sweep draw count")

    def test_gauss_boundary(self):
        """
        Thermal loss and the beamsplitter stay physical at their closed t edges, thermal loss at
        t = 5e-324 and at t = 1, xi = 0 and thermal(0) give I/2, and epr(0) is bona fide.
        """
        vac = GaussianState.vacuum(1).cov()
        for t in (0.0, 5e-324, DOWN, 1.0):
            got = GaussianState.vacuum(1).thermal_loss(0, t, 0.0, True)
            self.assertPhysical(square(got.cov()), msg=f"thermal loss at t = {t}")

            pair = GaussianState.vacuum(2).bs(0, 1, t)
            self.assertPhysical(square(pair.cov()), msg=f"beamsplitter at t = {t}")

        dead = GaussianState.vacuum(1).thermal_loss(0, 5e-324, 1e3, True).cov()

        self.gridClose([dead], [vac], atol=1e-12, msg="t = 5e-324 covariance != I/2")

        keep = GaussianState.vacuum(1).thermal_loss(0, 1.0, 0.0, True).cov()

        self.gridClose(
            [keep],
            [vac],
            atol=1e-12,
            msg="t = 1, xi = 0 covariance != I/2",
        )

        self.gridClose(
            [GaussianState.thermal(0.0).cov()],
            [vac],
            atol=0.0,
            msg="thermal(0) != I/2",
        )
        self.assertPhysical(square(GaussianState.epr(0.0).cov()), msg="epr(0) not bona fide")

    def test_gauss_extreme(self):
        """
        squeeze and epr at r up to 1e300 give a finite covariance or raise, cosh(2r) overflowing
        at r = 356 and the squeezer's map at r = 710, and thermal states over ten decades of nbar
        stay finite and physical.
        """
        broken = []
        for r in (0.0, 1.0, 8.0, 100.0, 355.0, 356.0, 400.0, 710.0, 1e6, 1e300):
            for name, make in (
                ("squeeze", lambda x: GaussianState.vacuum(1).squeeze(0, x)),
                ("epr", GaussianState.epr),
            ):
                try:
                    cov = make(r).cov()
                except ValueError:
                    continue

                if not all(math.isfinite(x) for x in cov):
                    broken.append(f"{name}({r!r}) -> {cov[:4]}")
        self.assertEqual(
            len(broken),
            0,
            msg="neither finite nor raised: " + "; ".join(broken),
        )

        for nbar in DECADES:
            hot = GaussianState.thermal(nbar)
            self.assertFinite(hot.cov(), msg=f"thermal({nbar}) covariance")
            self.assertPhysical(square(hot.cov()), msg=f"thermal({nbar}) unphysical")

    def test_gauss_measure(self):
        """
        Homodyne and heterodyne draws are finite and counted, and condition removes the measured
        mode but refuses the only one, which vacuum(0) and from_moments would not re-admit.
        """
        for r in Sweep(17, 40):
            st = GaussianState.thermal(loguni(r, 1e-6, 10.0)).squeeze(0, r.uniform(-2.0, 2.0))
            draws = st.homodyne(0, r.uniform(-10.0, 10.0), 4000, r.getrandbits(63))
            both = st.heterodyne(0, 2000, r.getrandbits(63))
            self.assertFinite(draws[:8], msg="homodyne draws not finite")
            self.assertFinite(both[:8], msg="heterodyne draws not finite")
            self.assertEqual(len(draws), 4000, msg="homodyne length != 4000")
            self.assertEqual(len(both), 4000, msg="heterodyne length != 4000")

        pair = GaussianState.epr(1.0)
        left = pair.condition(1, 0.0, 0.7)
        self.assertEqual(left.n_modes, 1, msg="n_modes != 1 after conditioning")
        self.assertPhysical(square(left.cov()), msg="conditional state not bona fide")
        self.assertBad(
            "zero-mode object",
            GaussianState.vacuum(1).condition,
            (0, 0.0, 0.3),
            msg="condition on the only mode accepted",
        )

    def test_gauss_exterior(self):
        """
        Out-of-domain arguments raise: a zero-mode state, a mode index past the end, a
        beamsplitter on one mode twice, a transmittance one ulp outside [0, 1], a negative xi
        or nbar, non-finite displacements and angles, and a from_moments matrix that is
        odd-length, antisymmetric or not bona fide.
        """
        self.assertFails(
            ValueError,
            "at least one mode",
            GaussianState.vacuum,
            0,
            msg="vacuum(0) accepted",
        )

        one = GaussianState.vacuum(1)
        two = GaussianState.vacuum(2)
        self.assertBad(
            "out of range",
            one.displace,
            (1, 0.0, 0.0),
            msg="mode index past the end accepted",
        )
        self.assertBad(
            "two distinct modes",
            two.bs,
            (0, 0, 0.5),
            msg="bs(0, 0) accepted",
        )

        chan = (0, 0.5, 0.0, True)
        guards = (
            (two.bs, (0, 1, 0.5), 2, "t must be in [0, 1]", CLOSED),
            (one.thermal_loss, chan, 1, "t must be in [0, 1]", CLOSED),
            (one.thermal_loss, chan, 2, "xi must be", SIGNED),
            (GaussianState.thermal, (1.0,), 0, "nbar must be", SIGNED),
            (one.displace, (0, 0.0, 0.0), 1, "must be finite", NASTY),
            (one.squeeze, (0, 0.0), 1, "must be finite", NASTY),
            (GaussianState.epr, (0.0,), 0, "must be finite", NASTY),
        )
        self.assertGuards(guards, msg="gaussian")
        self.assertBad(
            "even, non-zero length",
            GaussianState.from_moments,
            ([0.0], [0.5]),
            msg="odd mean length accepted",
        )
        self.assertBad(
            "must be symmetric",
            GaussianState.from_moments,
            ([0.0, 0.0], [1.0, 0.3, -0.3, 1.0]),
            msg="antisymmetric covariance accepted",
        )
        self.assertBad(
            "bona fide",
            GaussianState.from_moments,
            ([0.0, 0.0], [4.9e-97, 0.0, 0.0, 1e6]),
            msg="unphysical bright diagonal accepted",
        )
        self.assertBad(
            "bona fide",
            GaussianState.from_moments,
            ([0.0, 0.0], [0.4, 0.0, 0.0, 0.4]),
            msg="sub-vacuum covariance accepted",
        )


class FuzzFock(Fuzzed):
    """
    FockState. A renormalised truncated state passes every invariant; trunc_error is swept beside them.
    """

    def test_fock_interior(self):
        """
        Over 120 states from every constructor rho is physical with eigenvalues >= 0, purity,
        parity, entropy, non-Gaussianity and truncation lie in range, and populations survive a
        from_matrix round trip and a rotation.
        """
        s = Sweep(18, 120)
        for r in s:
            cut = r.randint(4, 40)
            pick = r.choice(("fock", "coherent", "cat", "squeezed", "thermal"))
            if pick == "fock":
                st = FockState.fock(r.randrange(cut), cut)
            elif pick == "coherent":
                st = FockState.coherent(r.uniform(-2.0, 2.0), r.uniform(-2.0, 2.0), cut)
            elif pick == "cat":
                st = FockState.cat(r.uniform(0.2, 2.0), r.uniform(-2.0, 2.0), r.random() < 0.5, cut)
            elif pick == "squeezed":
                st = FockState.squeezed(r.uniform(-1.2, 1.2), cut)
            else:
                st = FockState.thermal(loguni(r, 1e-6, 3.0), cut)

            pops = st.populations()
            eigs = st.eigenvalues()
            mean, cov = st.moments()
            self.assertTrue(st.physical(1e-9), msg=f"{pick} at cutoff {cut} is not a state")
            self.assertFinite(pops, msg=f"{pick} populations")
            self.assertGreaterEqual(min(eigs), -1e-9, msg=f"{pick} has a negative eigenvalue")
            self.assertRange(st.purity(), 0.0, 1.0 + 1e-9, msg=f"{pick} purity")
            self.assertRange(st.parity(), -1.0 - 1e-9, 1.0 + 1e-9, msg=f"{pick} parity")
            self.assertGreaterEqual(st.entropy(), -1e-12, msg=f"{pick} entropy below 0")
            self.assertGreaterEqual(st.nongauss(), -1e-6, msg=f"{pick} non-Gaussianity below 0")
            self.assertGreaterEqual(st.photons(), -1e-12, msg=f"{pick} photon number below 0")
            self.assertFinite(mean, msg=f"{pick} first moments")
            self.assertUncertainty(square(cov), msg=f"{pick} uncertainty product")
            self.assertRange(st.trunc_error(), 0.0, 1.0, msg=f"{pick} truncation error")
            self.assertRange(st.discarded(), 0.0, 1.0, msg=f"{pick} discarded norm")

            re, im = st.matrix()
            again = FockState.from_matrix(re, im)
            spun = st.rotate(r.uniform(-10.0, 10.0))
            self.assertEqual(again.cutoff, cut, msg=f"{pick} round trip changed the cutoff")

            self.gridClose(
                [again.populations()],
                [pops],
                atol=1e-12,
                msg=f"{pick} from_matrix populations",
            )

            self.gridClose(
                [spun.populations()],
                [pops],
                atol=1e-9,
                msg=f"{pick} rotation moved a population",
            )
        self.assertSwept(s, 120, msg="Fock constructor sweep draw count")

    def test_fock_grids(self):
        """
        Over eight states W integrates to 1, Q lies in [0, 1/(2 pi)], the negativity volume is
        >= 0 and 0 for a thermal state, parity = pi*W(0, 0), and an odd cat has parity -1.
        """
        made = (
            ("vacuum", FockState.fock(0, 20)),
            ("|3>", FockState.fock(3, 24)),
            ("coherent", FockState.coherent(1.5, -1.0, 40)),
            ("even cat", FockState.cat(2.0, 0.0, False, 40)),
            ("odd cat", FockState.cat(2.0, 0.0, True, 40)),
            ("squeezed", FockState.squeezed(0.7, 40)),
            ("thermal", FockState.thermal(1.0, 50)),
            ("gkp", FockState.gkp(0, 0.4, 60)),
        )
        step = AXIS[1] - AXIS[0]
        for name, st in made:
            w = grid(st.wigner(AXIS, AXIS), len(AXIS))
            q = grid(st.husimi(AXIS, AXIS), len(AXIS))
            self.assertNormalized(w, step, step, atol=5e-3, msg=f"{name}: W norm")
            self.assertHusimi(q, msg=f"{name}: Q outside [0, 1/(2 pi)]")
            self.assertGreaterEqual(
                st.negativity(AXIS, AXIS),
                -5e-3,
                msg=f"{name}: negativity < 0",
            )

            mid = len(AXIS) // 2
            self.assertClose(
                st.parity(),
                math.pi * w[mid][mid],
                atol=5e-3,
                msg=f"{name}: parity != pi*W(0, 0)",
            )
        self.assertClose(made[4][1].parity(), -1.0, atol=1e-9, msg="odd cat parity != -1")
        self.assertClose(
            made[6][1].negativity(AXIS, AXIS),
            0.0,
            atol=5e-3,
            msg="thermal negativity != 0",
        )

    def test_fock_boundary(self):
        """
        FockState holds at cutoffs 1 and 512 with n = cutoff - 1, GKP delta 0.05 and 2, loss at
        eta = 0 and 1, thermal(0), and a tail past the ladder, which clamps to 1.
        """
        self.assertEqual(FockState.fock(0, 1).cutoff, 1, msg="cutoff != 1")
        self.assertClose(FockState.fock(511, 512).photons(), 511.0, msg="photons != 511")

        for delta in (0.05, 2.0):
            self.assertTrue(
                FockState.gkp(0, delta, 40).physical(1e-9),
                msg=f"gkp at delta = {delta}",
            )

        dark = FockState.fock(3, 12).loss(0.0)
        self.assertClose(dark.populations()[0], 1.0, atol=1e-12, msg="p0 != 1 after total loss")

        kept = FockState.fock(3, 12).loss(1.0)
        self.assertClose(kept.populations()[3], 1.0, atol=1e-12, msg="p3 != 1 at eta = 1")
        self.assertClose(FockState.fock(0, 8).tail(10**9), 1.0, msg="tail(1e9) != 1")
        self.assertClose(FockState.fock(0, 8).tail(0), 0.0, msg="tail(0) != 0")
        self.assertClose(
            FockState.thermal(0.0, 8).populations()[0],
            1.0,
            msg="thermal(0) != I/2",
        )

    def test_fock_exterior(self):
        """
        Outside the domain every entry point raises: a cutoff of 0 or 513, a state that does
        not fit, an odd cat at zero amplitude, a GKP delta outside [0.05, 2], a third logical
        state, and a matrix that is not square, Hermitian, unit-trace or PSD.
        """
        one = FockState.fock(0, 4)
        deltas = (math.nextafter(0.05, 0.0), math.nextafter(2.0, 3.0), 0.0) + NASTY
        axes = ([0.0], [0.0, 0.0], [1.0, 0.0], [0.0, float("nan")])
        guards = (
            (FockState.fock, (0, 20), 1, "cutoff must be in 1..=512", (0, 513, 10**6)),
            (FockState.thermal, (1.0, 20), 0, "nbar must be", SIGNED),
            (FockState.gkp, (0, 0.3, 20), 1, "delta must be in [0.05, 2]", deltas),
            (one.loss, (0.5,), 0, "eta must be in [0, 1]", CLOSED),
            (one.rotate, (0.0,), 0, "must be finite", NASTY),
            (one.displace, (0.0, 0.0), 0, "must be finite", NASTY),
            (one.physical, (1e-9,), 0, "tol must be finite", NASTY),
            (one.wigner, ([0.0, 1.0], [0.0, 1.0]), 0, "xs", axes),
        )
        self.assertGuards(guards, msg="fock")
        self.assertBad(
            "does not fit under a cutoff",
            FockState.fock,
            (5, 5),
            msg="fock(5, 5) accepted",
        )
        self.assertBad(
            "an odd cat needs a non-zero amplitude",
            FockState.cat,
            (0.0, 0.0, True, 20),
            msg="odd cat at beta = 0 accepted",
        )
        self.assertBad(
            "logical must be 0 or 1",
            FockState.gkp,
            (2, 0.3, 20),
            msg="gkp logical 2 accepted",
        )
        self.assertBad(
            "square and non-empty",
            FockState.from_matrix,
            ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            msg="non-square matrix accepted",
        )
        self.assertBad(
            "must be Hermitian",
            FockState.from_matrix,
            ([0.5, 0.3, -0.3, 0.5], [0.0, 0.0, 0.0, 0.0]),
            msg="antisymmetric matrix accepted",
        )
        self.assertBad(
            "unit trace",
            FockState.from_matrix,
            ([0.5, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0]),
            msg="sub-normalised matrix accepted",
        )
        self.assertBad(
            "positive semidefinite",
            FockState.from_matrix,
            ([0.5, 0.9, 0.9, 0.5], [0.0, 0.0, 0.0, 0.0]),
            msg="non-PSD matrix accepted",
        )

    def test_fock_extreme(self):
        """
        An amplitude or squeezing too large for its ladder raises, nbar up to 1e300 saturates to
        maximally mixed, and squeezing to r = 710.4 reports trunc_error above 1/2.
        """
        for mag in (1e2, 1e3, 1e150, 1e300):
            self.assertBad(
                "no weight inside the cutoff",
                FockState.coherent,
                (mag, 0.0, 60),
                msg=f"|alpha| = {mag}",
            )
            self.assertBad(
                "fell entirely outside the cutoff",
                FockState.fock(0, 20).displace,
                (mag, 0.0),
                msg=f"a displacement by {mag}",
            )

        for nbar in (1e6, 1e30, 1e300):
            hot = FockState.thermal(nbar, 16)
            self.assertTrue(hot.physical(1e-9), msg=f"thermal({nbar}) unphysical")

            # physical() passes a NaN trace (NaN > tol is false); only this norm check catches one.
            self.assertClose(
                sum(hot.populations()),
                1.0,
                atol=1e-9,
                msg=f"thermal({nbar}) normalisation",
            )
            self.assertClose(
                hot.populations()[0],
                1.0 / 16.0,
                atol=1e-6,
                msg="p0 != 1/16",
            )

        for r in (5.0, 20.0, 100.0, 710.4):
            tight = FockState.squeezed(r, 20)
            self.assertTrue(tight.physical(1e-9), msg=f"squeezed({r}) unphysical")
            self.assertGreater(
                tight.trunc_error(),
                0.5,
                msg=f"squeezed({r}) trunc_error <= 1/2",
            )

        for r in (710.5, 1e3, 1e300):
            self.assertBad(
                "no weight inside the cutoff",
                FockState.squeezed,
                (r, 20),
                msg=f"r = {r}",
            )


class FuzzPipe(Fuzzed):
    """
    run_symbols, its f64 CPU mirror, and the queries that decide which precision a caller
    gets.
    """

    def test_symbols_fuzz(self):
        """
        Over 24 runs every run_symbols statistic but v_err is finite, sigma2 >= 1 SNU, pilot_snr >=
        0, v_wrap <= pi^2 and saturating at pi^2/3 = 3.2899, v_err > 2*v_wrap past v_wrap = 3.2, and
        three pinned runs read back their configuration.
        """
        s = Sweep(19, 24)
        for r in s:
            arg = (
                8192,
                r.getrandbits(63),
                loguni(r, 1e-2, 1e2),
                unit(r, 1e-3),
                loguni(r, 1e-9, 0.5),
                unit(r, 1e-2),
                r.uniform(0.0, 1.0),
                loguni(r, 1.0, 1e6),
                loguni(r, 1.0, 1e6),
                r.uniform(-4.5e8, 4.5e8),
                1e9,
                r.uniform(-10.0, 30.0),
                r.uniform(0.05, 0.45),
                32,
                False,
            )
            out = _core.run_symbols(*arg)
            self.assertFinite(
                (
                    out.t_hat,
                    out.t_chan,
                    out.sigma2_hat,
                    out.xi_hat,
                    out.xi_ideal,
                    out.v_wrap,
                    out.pilot_snr,
                    out.cfo,
                ),
                msg=f"run_symbols va={arg[2]} t={arg[3]} xi={arg[4]}",
            )
            self.assertGreaterEqual(
                out.sigma2_hat,
                1.0 - 1e-2,
                msg=f"sigma2 = {out.sigma2_hat} below the vacuum",
            )
            self.assertGreaterEqual(out.t_chan, 0.0, msg="t_chan negative")
            self.assertGreaterEqual(out.pilot_snr, 0.0, msg="pilot_snr negative")

            # pi^2, not pi^2/3: the moment is about zero, not the mean.
            self.assertLessEqual(
                out.v_wrap,
                math.pi * math.pi,
                msg=f"v_wrap {out.v_wrap} above pi^2",
            )
            self.assertGreaterEqual(out.v_err, 0.0, msg="v_err negative")

            if out.v_wrap > 3.2:
                self.assertGreater(
                    out.v_err,
                    2.0 * out.v_wrap,
                    msg=f"v_err {out.v_err} at v_wrap {out.v_wrap:.5f}",
                )
            self.assertEqual(out.n_used, 8192, msg="n_used != 8192")

        loud = _core.run_symbols(65536, 5, *PIPE)
        self.assertClose(loud.t_chan, 0.5, atol=0.02, msg="t_chan != 0.5")

        frames = _core.run_symbols(2048, 11, 5.0, 0.5, 0.01, 0.6, 0.1, 1e4, 1e4, 3e7, 1e9, 12.0, 0.25, 32, True)
        self.assertEqual(len(frames.frames_x), frames.n_used, msg="frames_x length != n_used")
        self.assertEqual(len(frames.frames_p), frames.n_used, msg="frames_p length != n_used")

        # .tolist() first: `+` on numpy arrays is elementwise, not concatenation.
        self.assertFinite(
            frames.frames_x[:8].tolist() + frames.frames_p[:8].tolist(),
            msg="frames not finite",
        )
        self.assertEqual(len(loud.frames_x), 0, msg="frames_x not empty")
        self.assertClose(loud.xi_ideal, 0.01, atol=0.02, msg="xi_ideal != 0.01")
        self.assertClose(
            2.0 * loud.t_ideal * loud.t_ideal / 0.6,
            0.5,
            atol=0.02,
            msg="oracle slope != 0.5",
        )

        blocked = _core.run_symbols(4096, 3, 0.0, 1.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 1e9, 12.0, 0.25, 32, False)
        self.assertEqual(blocked.t_hat, 0.0, msg="t_hat != 0")
        self.assertEqual(blocked.xi_hat, 0.0, msg="xi_hat != 0")
        self.assertClose(blocked.sigma2_hat, 1.1, atol=0.05, msg="sigma2 != 1.1")
        self.assertSwept(s, 24, msg="pipeline sweep draw count")

    def test_symbols_exterior(self):
        """
        run_symbols refuses a T or eta outside (0, 1], a negative modulation, xi, v_el or
        linewidth, a CFO at or past Nyquist, a non-positive symbol rate, a non-finite pilot
        setting, and a run of fewer than two DSP blocks.
        """
        ok = (4096, 1) + PIPE
        cases = (
            (0, "at least two DSP blocks", (0, 32, 63)),
            (2, "va must be", NONNEG),
            (3, "t must be", OPEN),
            (4, "xi must be", NONNEG),
            (5, "eta must be", OPEN),
            (6, "vel must be", NONNEG),
            (7, "lw_alice must be", NONNEG),
            (8, "lw_lo must be", NONNEG),
            (9, "cfo must satisfy", (5e8, -5e8, 6e8) + NASTY),
            (10, "symbol_rate must be", POSITIVE),
            (11, "pilot_db and pilot_frac must be finite", NASTY),
            (12, "pilot_db and pilot_frac must be finite", NASTY),
            (13, "block must be >= 2", (0, 1)),
        )
        self.assertSlots(_core.run_symbols, ok, cases, msg="run_symbols")

    def test_backend_fuzz(self):
        """
        backend_info names a device, supports accepts only f32 and f64, cpu_words is pure and
        counter-indexed and matched bit for bit by gpu_words where an adapter answers, a
        zero-length dispatch is refused, and cpu_symbols stays finite over six runs.
        """
        name, prec, device = _core.backend_info()
        self.assertIn(name, ("cpu", "gpu"), msg=f"unknown backend {name!r}")
        self.assertIn(prec, ("f32", "f64"), msg=f"unknown precision {prec!r}")
        self.assertGreater(len(device), 0, msg="empty device string")
        self.assertTrue(_core.supports(prec), msg="resolved precision unsupported")
        self.assertSlots(
            _core.supports,
            ("f64",),
            ((0, "unknown precision", ("f16", "double", "", "F64")),),
            msg="supports",
        )
        self.assertIn(_core.gpu_ready(), (True, False), msg="gpu_ready is not a bool")
        self.assertGreater(len(_core.gpu_probe()), 0, msg="empty gpu_probe string")

        words = _core.cpu_words(7, 3, 0, 4, 0)
        self.assertEqual(len(words), 16, msg="word count != 16")
        self.assertEqual(
            words.tolist(),
            _core.cpu_words(7, 3, 0, 4, 0).tolist(),
            msg="cpu_words is not a pure function",
        )
        self.assertEqual(
            _core.cpu_words(7, 3, 2, 2, 0).tolist(),
            words[8:].tolist(),
            msg="offset base != slice",
        )

        if _core.gpu_ready():
            self.assertEqual(
                _core.gpu_words(7, 3, 0, 4, 0).tolist(),
                words.tolist(),
                msg="gpu_words != cpu_words",
            )
            self.assertFinite(_core.gpu_empty(), msg="gpu_empty not finite")
            self.assertGreaterEqual(_core.gpu_empty(), 0.0, msg="gpu_empty negative")
            self.assertBad(
                "no gpu compute path",
                _core.gpu_words,
                (7, 3, 0, 0, 0),
                exc=RuntimeError,
                msg="zero-length dispatch accepted",
            )

        # Repeats test_symbols_fuzz's leading draws; do not share them, the tails differ in length.
        s = Sweep(20, 6)
        for r in s:
            arg = (
                4096,
                r.getrandbits(63),
                loguni(r, 1e-2, 1e2),
                unit(r, 1e-3),
                loguni(r, 1e-9, 0.5),
                unit(r, 1e-2),
                r.uniform(0.0, 1.0),
                0.0,
                0.0,
                1e9,
                r.uniform(0.0, 20.0),
                r.uniform(0.05, 0.45),
            )
            got = _core.cpu_symbols(*arg)
            self.assertFinite(
                (got.t_hat, got.t_chan, got.sigma2, got.xi, got.pilot_snr, got.host_ms),
                msg=f"cpu_symbols va={arg[2]} t={arg[3]}",
            )
            self.assertGreaterEqual(got.sigma2, 1.0 - 1e-2, msg=f"sigma2 = {got.sigma2} below the vacuum")
            self.assertGreaterEqual(got.t_hat, 0.0, msg="t_hat negative")
            self.assertEqual(got.n_used, 4096, msg="n_used != 4096")
            self.assertGreater(got.groups, 0, msg="groups == 0")
            self.assertEqual(got.walk_ms, 0.0, msg="walk_ms != 0")
            self.assertEqual(got.fused_ms, 0.0, msg="fused_ms != 0")

            if _core.gpu_ready():
                fast = _core.gpu_symbols(*arg)
                self.assertFinite((fast.t_hat, fast.sigma2), msg="gpu_symbols not finite")
                self.assertClose(
                    fast.t_hat,
                    got.t_hat,
                    atol=1e-3 * max(1.0, got.t_hat),
                    msg="gpu t_hat != cpu t_hat",
                )
        self.assertSlots(
            _core.cpu_symbols,
            (4096, 1, 5.0, 0.5, 0.01, 0.6, 0.1, 0.0, 0.0, 1e9, 12.0, 0.25),
            ((0, "at least two blocks of 256 symbols", (0, 255, 511)),),
            msg="cpu_symbols",
        )
        self.assertSwept(s, 6, msg="backend sweep draw count")


if __name__ == "__main__":
    print(f"fuzz seeds: {SEEDS}")
    rc = Exam(
        "FuzzCv",
        "cv_rate, cv_bounds and cv_finite over the interior and the edges, "
        "and the pure-Python budget that feeds them",
        "fuzz_cv.md",
    ).run(load(FuzzCv))
    rc |= Exam(
        "FuzzRate",
        "dps, bb84 and cow rates: bounded above by their own prefactors, refusing f_ec < 1",
        "fuzz_rate.md",
    ).run(load(FuzzRate))
    rc |= Exam(
        "FuzzDecoy",
        "Decoy forward model and single-photon bounds, swept against the infinite-decoy ceiling",
        "fuzz_decoy.md",
    ).run(load(FuzzDecoy))
    rc |= Exam(
        "FuzzRelay",
        "Relay noise, covariance and optics",
        "fuzz_relay.md",
    ).run(load(FuzzRelay))
    rc |= Exam(
        "FuzzDmcs",
        "M-PSK constellations, their mutual information and the correlation bound",
        "fuzz_dmcs.md",
    ).run(load(FuzzDmcs))
    rc |= Exam(
        "FuzzClick",
        "Threshold detection: click probability, interferometer ports and full runs",
        "fuzz_click.md",
    ).run(load(FuzzClick))
    rc |= Exam(
        "FuzzGauss",
        "GaussianState: every returned state must satisfy V + i*Omega/2 >= 0",
        "fuzz_gauss.md",
    ).run(load(FuzzGauss))
    rc |= Exam(
        "FuzzFock",
        "FockState: physicality, quasi-probability invariants and truncation error",
        "fuzz_fock.md",
    ).run(load(FuzzFock))
    rc |= Exam(
        "FuzzPipe",
        "run_symbols, the f64 CPU mirror and the backend adapter's queries",
        "fuzz_pipe.md",
    ).run(load(FuzzPipe))
    sys.exit(rc)
