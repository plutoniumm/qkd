import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

import qkd as q
from kit.cache import memo
from kit.links import sim_link
from kit.noise import quiet
from kit.procs import threads
from qkd import _core, budget, impairments

RATE = 100e6
TONE = 180e6 / RATE

# 400k: at 50k the reduction tree has one shape under 1 and 8 threads.
CHILD = (
    "import array, hashlib\n"
    "import qkd._core as c\n"
    "o = c.run_symbols(400000, 4, 5.0, 0.5, 0.01, 0.6, 0.1, 1e4, 1e4, 4e6, "
    f"{RATE!r}, 12.0, {TONE!r}, 32, True)\n"
    "b = array.array('d', list(o.frames_x) + list(o.frames_p))\n"
    "print(hashlib.sha256(b).hexdigest(), repr(o.v_err), repr(o.t_hat), "
    "repr(o.sigma2_hat), repr(o.cfo))\n"
)


@memo
def run(
    n=200_000,
    seed=1,
    va=5.0,
    t=0.5,
    xi=0.01,
    eta=0.6,
    vel=0.1,
    lw=0.0,
    cfo=0.0,
    db=12.0,
    block=32,
    frames=False,
    jit=0.0,
):
    """
    One memoised `run_symbols` at the metro defaults.
    """

    return _core.run_symbols(n, seed, va, t, xi, eta, vel, lw, lw, cfo, RATE, db, TONE, block, frames, jit)


def slope(out, v_a=5.0):
    """
    Phase excess noise rebuilt from the run's own slope ratio, not exp(-v_err/2).
    """
    kappa = out.t_hat / out.t_ideal

    return (v_a + max(0.0, out.xi_ideal)) * (1.0 / (kappa * kappa) - 1.0)


def band(var, n, k=5.0):
    """
    k sampling deviations of -2*ln(E[cos phi]) as a fraction of var; carries exp(var/2).
    """
    kappa = math.exp(-var / 2.0)
    spread = (1.0 + math.exp(-2.0 * var)) / 2.0 - math.exp(-var)

    return k * 2.0 * math.sqrt(spread / n) / (kappa * var)


def fading(jit):
    """
    (<sqrt eta>^2, Var(sqrt eta), xi_timing) at pulse width 1, so jit is the ratio.
    """
    eff, var = impairments.jitter_fading(jit, 1.0)

    return eff, var, impairments.timing(5.0, jit, 1.0)


def wired(dsp, lw=1e4):
    """
    The same configuration through `q.Link`.
    """

    return sim_link(q.Asymptotic(beta=0.95), lw=lw, xi=0.01, dsp=dsp)


class Pipeline(Question):
    """
    Phase 2 pipeline: amplitude convention, reproducibility, estimators, pilot SNR.
    """

    def test_vacuum_normalization(self):
        """
        With no modulation, unit transmittance and no excess or phase noise, Bob's
        per-quadrature variance is $1 + v_{el}$ at every $\\eta$.
        """
        # atol 0.02 = 6.4 sigma of the 3.1e-3 seed spread (seeds 1..12).
        for eta in (0.4, 0.6, 1.0):
            out = run(n=100_000, va=0.0, t=1.0, xi=0.0, eta=eta, vel=0.1)
            self.assertClose(out.sigma2_hat, 1.1, atol=0.02, msg=f"eta={eta}: sigma2_hat {out.sigma2_hat}")

    def test_residual_conditioning(self):
        """
        Moments taken about the nominal slope, not the origin, keep `sigma2_hat` converged up to
        $V_A = 10^{12}$ while `v_err` falls as $1/V_A$.
        """
        # 200k: at 100k the xi step sits 2.9 sigma inside its 1e-5 band.
        runs = [run(n=200_000, va=va, lw=0.0) for va in (1e4, 1e6, 1e9, 1e12)]
        errs = [o.v_err for o in runs]

        self.assertMonotone(errs, rising=False, msg=f"v_err: {errs}")

        # atol 0.1 = 23 sigma; ratio 100.0016, sd 4.3e-3 over seeds 1..24.
        self.assertClose(errs[0] / errs[1], 100.0, atol=0.1, msg=f"v_err ratio {errs[0] / errs[1]}")

        # atols = drift + 6.1, 8.2 sigma: 1.6e-7 +- 1.4e-7, 1.3e-6 +- 1.1e-6 over seeds 1..24.
        for i, out in enumerate(runs[1:]):
            self.assertClose(
                out.sigma2_hat,
                runs[0].sigma2_hat,
                atol=1e-6,
                msg=f"sigma2_hat drifted at step {i}: {out.sigma2_hat!r}",
            )
            self.assertClose(
                out.xi_hat,
                runs[0].xi_hat,
                atol=1e-5,
                msg=f"xi_hat drifted at step {i}: {out.xi_hat!r}",
            )

    def test_chunk_invariance(self):
        """
        A 200k run reproduces a 100k run's frames bit for bit but for the last, neighbourless
        block, symbol $i$ being a pure function of (seed, stage, $i$).
        """
        short = run(n=100_000, seed=9, lw=1e4, frames=True)
        long = run(n=200_000, seed=9, lw=1e4, frames=True)
        keep = short.n_used - 32

        self.assertTrue(
            np.array_equal(short.frames_x[:keep], long.frames_x[:keep]),
            msg="frames_x, 100k vs 200k",
        )
        self.assertTrue(
            np.array_equal(short.frames_p[:keep], long.frames_p[:keep]),
            msg="frames_p, 100k vs 200k",
        )

    def test_thread_invariance(self):
        """
        Fixed index-order folding makes one and eight workers agree bit for bit, frames and
        scalars, at a 4 MHz offset.
        """
        one = threads(CHILD, 1)
        many = threads(CHILD, 8)

        self.assertEqual(one[0], many[0], msg="frames hash, 1 vs 8 threads")
        self.assertEqual(one[1:], many[1:], msg=f"scalars: {one[1:]} vs {many[1:]}")

    def test_frames_parity(self):
        """
        `v_err`, `t_hat` and `sigma2_hat` are bit-identical with the recovered frames kept or
        dropped.
        """
        off = run(n=200_000, seed=4, lw=1e4)
        on = run(n=200_000, seed=4, lw=1e4, frames=True)

        self.assertEqual(off.v_err, on.v_err, msg="v_err, frames on vs off")
        self.assertEqual(off.t_hat, on.t_hat, msg="t_hat, frames on vs off")
        self.assertEqual(off.sigma2_hat, on.sigma2_hat, msg="sigma2_hat, frames on vs off")

    def test_pilot_snr(self):
        """
        Pilot SNR against the adjacent empty DFT bin is positive and tracks pilot power one for
        one in dB.
        """
        snrs = [run(n=100_000, seed=3, db=db).pilot_snr for db in (0.0, 4.0, 8.0, 12.0)]

        self.assertGreater(snrs[0], 0.0, msg=f"pilot_snr {snrs[0]}")
        self.assertEqual(snrs, sorted(snrs), msg=f"pilot_snr: {snrs}")

        # atol 0.5 on 15.849 = 0.027 bias + 5.3 sigma; ratio 15.876, sd 0.089 over seeds 1..12.
        self.assertClose(snrs[3] / snrs[0], 10**1.2, atol=0.5, msg=f"12 dB ratio {snrs[3] / snrs[0]}")


class Recovery(Question):
    """
    Residual phase error and the xi it leaves.
    """

    def test_estimator_slope_ratio(self):
        """
        The excess over the oracle run matches $(V_A + \\xi)(1/\\kappa^2 - 1)$ at the run's own
        slope ratio $\\kappa$ to 0.2% over `v_err` = 0.0025 to 0.60 -- Kish, Quantum 8, 1382
        (2024), Eqs. (80)-(82).
        """
        outs = [run(n=1_000_000, seed=7, lw=lw) for lw in (1e3, 1e4, 1e5, 5e5, 1e6)]
        offs = []
        for out in outs:
            meas = out.xi_hat - out.xi_ideal
            offs.append(abs(slope(out) - meas) / meas)
            self.assertLess(
                offs[-1],
                2e-3,
                msg=f"v_err={out.v_err:.5f}: slope rebuild off by {offs[-1]:.5f}",
            )
        self.assertClose(max(offs), min(offs), atol=2e-4, msg=f"offs: {offs}")

    def test_estimator_form_gate(self):
        """
        The default form tracks `xi_hat - xi_ideal` to 2% at `v_err` = 0.054, 0.27, 0.60 and
        2.81, where the literature form sits 6%, 24%, 58% and 944% below.
        """
        outs = [run(n=400_000, seed=7, lw=lw) for lw in (1e5, 5e5, 1e6, 1e7)]
        for out in outs:
            meas = out.xi_hat - out.xi_ideal
            est = quiet(5.0, out.v_err, max(0.0, out.xi_ideal))
            lit = quiet(5.0, out.v_err, form="literature")
            self.assertLessEqual(
                abs(est - meas) / meas,
                0.02,
                msg=f"v_err={out.v_err:.4f}: default form {est:.4f} vs {meas:.4f}",
            )
            self.assertLess(
                abs(est - meas),
                abs(lit - meas),
                msg=f"v_err={out.v_err:.4f}: est {est:.4f}, lit {lit:.4f}, meas {meas:.4f}",
            )

        # 1.5 rad^2 = the retired WRAP_LIMIT; the wrapped moment saturated, not the form.
        last = outs[-1]

        self.assertGreater(last.v_err, 1.5, msg=f"v_err {last.v_err}")

    def test_xi_from_phase_noise(self):
        """
        Measured excess noise matches Marie-Alleaume's $2V_A(1 - e^{-v_{err}/2})$,
        `form="literature"`, to 25% at 1, 10 and 100 kHz, inside `PHASE_LIMIT`.
        """
        for lw in (1e3, 1e4, 1e5):
            out = run(n=1_000_000, seed=7, lw=lw)
            got = out.xi_hat - out.xi_ideal
            want = budget.phase(5.0, out.v_err, form="literature")
            self.assertLessEqual(
                out.v_err,
                budget.PHASE_LIMIT,
                msg=f"lw={lw:.0e}: v_err={out.v_err:.5f}",
            )
            self.assertLessEqual(
                abs(got - want) / want,
                0.25,
                msg=f"lw={lw:.0e}: measured {got:.5f}, model {want:.5f}, v_err={out.v_err:.5f}",
            )

    def test_phase_domain_edge(self):
        """
        The literature form, lacking Kish Eq. (82)'s renormalisation, misses the measured excess
        by the closed ratio $(e^v + e^{v/2})/2$ -- 5.5% at `v_err` = 0.054, 23% at 0.27, 58% at
        0.60, 940% at 2.81 -- and holds to 10% only inside `budget.PHASE_LIMIT`, a range
        Marie-Alleaume Eqs. (10)-(11) do not state.
        """
        outs = [run(n=400_000, seed=7, lw=lw) for lw in (1e5, 5e5, 1e6, 1e7)]
        errs = [o.v_err for o in outs]
        gap = [(o.xi_hat - o.xi_ideal) / quiet(5.0, o.v_err, form="literature") - 1.0 for o in outs]

        self.assertMonotone(gap, msg=f"gap: {gap}")
        self.assertLessEqual(
            errs[0],
            budget.PHASE_LIMIT,
            msg=f"v_err {errs[0]:.4f}",
        )
        self.assertLessEqual(gap[0], 0.10, msg=f"v_err={errs[0]:.4f}: gap {gap[0]:.3f}")

        # atol 0.03 = the sampling error in V_A + xi at 400k symbols; the ratio is not fitted.
        for err, got in zip(errs, gap):
            want = (math.exp(err) + math.exp(err / 2.0)) / 2.0 - 1.0
            self.assertClose(
                (1.0 + got) / (1.0 + want),
                1.0,
                atol=0.03,
                msg=f"v_err={err:.4f}: gap {got:.4f}, closed {want:.4f}",
            )
        self.assertGreater(gap[3], 5.0, msg=f"v_err={errs[3]:.4f}: gap {gap[3]:.3f}")

    def test_phase_domain_warns(self):
        """
        Past `PHASE_LIMIT` the literature form warns naming `v_err` and still returns the closed
        form, inside it is quiet, and `assemble()` inherits the warning.
        """
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            edge = budget.phase(5.0, budget.PHASE_LIMIT, form="literature")
        self.assertEqual(len(seen), 0, msg=f"warnings at PHASE_LIMIT: {seen}")
        self.assertGreater(edge, 0.0, msg=f"xi {edge}")

        with self.assertWarns(budget.PhaseDomainWarning) as caught:
            got = budget.phase(5.0, 2.55, form="literature")

        text = str(caught.warning)

        self.assertIn("2.55", text, msg=f"warning: {text}")
        self.assertClose(
            got,
            10.0 * (1.0 - math.exp(-1.275)),
            atol=1e-12,
            msg=f"xi {got}",
        )

        with self.assertWarns(budget.PhaseDomainWarning):
            bud = budget.assemble(v_a=5.0, t=0.5, v_err=2.55, phase_form="literature")
        self.assertClose(bud.total, got, atol=1e-12, msg=f"total {bud.total}")

    def test_estimator_domain_warns(self):
        """
        The default form is silent at `PHASE_LIMIT`, at `v_err` = 1.5 and at 12, and `budget`
        has no `WRAP_LIMIT`.
        """
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            budget.phase(5.0, budget.PHASE_LIMIT)
            budget.phase(5.0, 1.5)
            wide = budget.phase(5.0, 12.0, 0.01)
        self.assertEqual(len(seen), 0, msg=f"warnings: {seen}")
        self.assertClose(
            wide,
            5.01 * math.expm1(12.0),
            atol=1e-9,
            msg=f"xi {wide}",
        )
        self.assertFalse(
            hasattr(budget, "WRAP_LIMIT"),
            msg="budget.WRAP_LIMIT exists",
        )

    def test_estimator_recovers_variance(self):
        """
        $-2\\ln E[\\cos\\varphi]$ returns a wrapped normal's variance within five deviations of
        its sampling law at 4e6 draws from $\\sigma^2 = 0.05$ to 12 rad^2, while the wrapped
        second moment saturates at $\\pi^2/3 = 3.2899$: 3% low at 1.5, 22% at 3.13, 48% at 6,
        73% at 12.
        """
        rng = np.random.default_rng(7)
        draws = 4_000_000
        # Seeds 1..24 span 0.33-2.83 deviations; a flat 5e-3 band is 7.1 at 0.05 but 0.84 at 12.
        for want in (0.05, 0.28, 0.64, 1.5, 3.13, 6.0, 12.0):
            phi = rng.normal(0.0, math.sqrt(want), draws)
            wrapped = phi - 2.0 * math.pi * np.round(phi / (2.0 * math.pi))
            circ = -2.0 * math.log(float(np.mean(np.cos(wrapped))))
            moment = float(np.mean(wrapped * wrapped))
            self.assertClose(
                circ / want,
                1.0,
                atol=band(want, draws),
                msg=f"sigma^2 = {want}: circular estimator returned {circ:.5f}",
            )

            # 0.01 = 7 standard errors, 2*pi^2/sqrt(45*4e6); pi^2/3 is a mean, not a ceiling.
            self.assertLessEqual(
                moment,
                math.pi * math.pi / 3.0 + 0.01,
                msg=f"sigma^2 = {want}: wrapped moment {moment:.5f} passed pi^2/3",
            )

            if want >= 1.5:
                self.assertLessEqual(
                    moment / want,
                    0.98,
                    msg=f"sigma^2 = {want}: wrapped moment {moment:.5f}",
                )

        phi = rng.normal(0.0, math.sqrt(50.0), draws)
        wrapped = phi - 2.0 * math.pi * np.round(phi / (2.0 * math.pi))

        self.assertClose(
            float(np.mean(wrapped * wrapped)),
            math.pi * math.pi / 3.0,
            atol=1e-2,
            msg="wrapped moment at sigma^2 = 50",
        )

    def test_wrapped_is_not_the_budget_input(self):
        """
        `v_err` and `v_wrap` agree to 0.1% while narrow, and at 10 MHz, 2.81 against 2.34, only
        `v_err` reproduces the measured 78.76 SNU, to 1.0%, where `v_wrap` gives 46.77.
        """
        near = run(n=200_000, seed=7, lw=1e4)

        self.assertClose(
            near.v_err / near.v_wrap,
            1.0,
            atol=1e-3,
            msg=f"v_err {near.v_err}, v_wrap {near.v_wrap}",
        )

        wide = run(n=400_000, seed=7, lw=1e7)
        meas = wide.xi_hat - wide.xi_ideal
        xi = max(0.0, wide.xi_ideal)

        self.assertGreater(
            wide.v_err - wide.v_wrap,
            0.3,
            msg=f"v_err {wide.v_err}, v_wrap {wide.v_wrap}",
        )
        self.assertLessEqual(
            abs(quiet(5.0, wide.v_err, xi) - meas) / meas,
            0.02,
            msg=f"v_err={wide.v_err:.4f}: measured {meas:.4f}",
        )
        self.assertGreater(
            abs(quiet(5.0, wide.v_wrap, xi) - meas) / meas,
            0.20,
            msg=f"v_wrap={wide.v_wrap:.4f}: measured {meas:.4f}",
        )

    def test_v_err_grows_with_linewidth(self):
        """
        `v_err` rises with linewidth at fixed block size, 100 kHz reaching over 10x the
        noiseless run.
        """
        errs = [run(n=200_000, seed=7, lw=lw).v_err for lw in (0.0, 1e3, 1e4, 1e5)]

        self.assertEqual(errs, sorted(errs), msg=f"v_err not monotone: {errs}")
        self.assertGreater(errs[3], 10.0 * errs[0], msg=f"v_err: {errs}")

    def test_v_wrap_falls_with_pilot(self):
        """
        With the lasers noiseless, 12 dB buys $10^{1.2}$ in `v_wrap` once the block phase
        estimate is linear, 20 to 32 dB, and 5% more across 0 to 12 dB, where it is not.
        """
        # 16.646 = 10^1.2 * 1.0533/1.0029, the wrapped correction; atol 0.75 = 0.176 + 5*0.115, seeds 1..24.
        errs = [run(n=100_000, seed=3, db=db).v_wrap for db in (0.0, 4.0, 8.0, 12.0)]

        self.assertEqual(errs, sorted(errs, reverse=True), msg=f"v_wrap: {errs}")
        self.assertClose(
            errs[0] / errs[3],
            16.646,
            atol=0.75,
            msg=f"0-12 dB ratio {errs[0] / errs[3]:.4f}",
        )

        # atol 0.08 = 0.0155 + 5.5*0.0118 over seeds 1..24.
        loud = [run(n=100_000, seed=3, db=db).v_wrap for db in (20.0, 32.0)]

        self.assertClose(
            loud[0] / loud[1],
            10**1.2,
            atol=0.08,
            msg=f"20-32 dB ratio {loud[0] / loud[1]:.5f}",
        )

    def test_pinned_short_circuits(self):
        """
        A pinned `q.PilotPhase(v_err=...)` returns the closed-form $\\xi_{phase}$ for $10^8$
        symbols in under 0.1 s, simulating none.
        """
        t0 = time.perf_counter()
        res = wired(q.DSP(phase=q.PilotPhase(v_err=2e-3))).run(symbols=10**8)
        wall = time.perf_counter() - t0

        self.assertLess(wall, 0.1, msg=f"wall {wall:.3f}s")
        self.assertEqual(res.dsp.symbols, 0, msg=f"symbols {res.dsp.symbols}")
        self.assertClose(
            res.explain["xi_phase"]["value"],
            budget.phase(5.0, 2e-3, 0.01),
            atol=1e-15,
            msg="xi_phase",
        )
        self.assertEqual(res.explain["xi_phase"]["label"], "pinned", msg="xi_phase label")

    def test_link_wiring(self):
        """
        The derived path's `xi_input` is the pinned channel $\\xi$ plus `budget.phase(v_err)` on
        $(V_A + \\xi)$ and nothing else, a trusted $v_{el}$ staying in $\\chi_{det}$.
        """
        res = wired(q.DSP()).run(symbols=200_000, seed=3)
        out = run(n=200_000, seed=3, lw=1e4)

        self.assertClose(res.dsp.v_err, out.v_err, atol=1e-15, msg="v_err")
        self.assertClose(
            res.explain["xi_input"]["value"],
            0.01 + budget.phase(5.0, out.v_err, 0.01),
            atol=1e-15,
            msg="xi_input",
        )
        self.assertEqual(res.explain["v_err"]["label"], "derived", msg="v_err label")

    def test_link_names_the_form(self):
        """
        `explain()` names the $\\xi_{phase}$ form, `estimator` on every `q.Link` run, derived,
        pinned or dry.
        """
        derived = wired(q.DSP()).run(symbols=200_000, seed=3)
        pinned = wired(q.DSP(phase=q.PilotPhase(v_err=2e-3))).run()
        for res in (derived, pinned):
            row = res.explain["phase_form"]
            self.assertEqual(row["value"], "estimator", msg=f"form row: {row}")
            self.assertEqual(row["label"], "default", msg=f"form label: {row}")
        self.assertIn(
            "phase_form",
            wired(q.DSP()).explain(),
            msg="phase_form in dry-run explain()",
        )

    def test_phase_noise_costs_key(self):
        """
        The derived path's key rate lands strictly below the same link with `v_err` pinned at 0.
        """
        real = wired(q.DSP()).run(symbols=200_000, seed=3)
        ideal = wired(q.DSP(phase=q.PilotPhase(v_err=0.0))).run()

        self.assertGreater(real.key_rate, 0.0, msg=f"key_rate {real.key_rate}")
        self.assertLess(real.key_rate, ideal.key_rate, msg=f"key_rate {real.key_rate}, ideal {ideal.key_rate}")

    def test_pinned_transmittance(self):
        """
        A pinned `v_err` charges both halves of Kish App. E Eq. (78): `T_claimed` and
        `res.budget.inferred` equal `budget.infer(T, v_err)`, and $T_{claimed}(V_A +
        \\xi_{total}) = T(V_A + \\xi_{channel})$ at every variance.
        """
        for v_err in (0.0, 2e-3, 0.005, 0.01, 0.27):
            res = wired(q.DSP(phase=q.PilotPhase(v_err=v_err))).run()
            got = res.explain["T_claimed"]["value"]
            self.assertClose(
                got,
                budget.infer(0.5, v_err),
                atol=0.0,
                msg=f"v_err={v_err}: T_claimed {got}",
            )
            self.assertClose(
                res.budget.inferred,
                got,
                atol=0.0,
                msg=f"v_err={v_err}: inferred {res.budget.inferred}",
            )
            self.assertClose(
                got * (5.0 + res.explain["xi_total"]["value"]),
                0.5 * (5.0 + 0.01),
                atol=1e-12,
                msg=f"v_err={v_err}: signal",
            )

    def test_pinned_costs_rate(self):
        """
        Charging the noise half against an unattenuated $T$ over-claims the metro rate by 0.34%
        at `v_err` = 2e-3, 0.88% at 0.005, 1.89% at 0.01 and 4.98% at 0.02.
        """
        det = (0.6, 0.1, 0.95, False, True)
        pts = ((2e-3, 0.3427), (5e-3, 0.8828), (1e-2, 1.8930), (2e-2, 4.9763))
        for v_err, want in pts:
            res = wired(q.DSP(phase=q.PilotPhase(v_err=v_err))).run()
            loose = _core.cv_rate(5.0, 0.5, res.explain["xi_total"]["value"], *det)[2]
            self.assertGreater(
                loose,
                res.key_rate,
                msg=f"v_err={v_err}: loose {loose}, key_rate {res.key_rate}",
            )
            self.assertClose(
                100.0 * (loose / res.key_rate - 1.0),
                want,
                atol=1e-3,
                msg=f"v_err={v_err}: over-claim {100.0 * (loose / res.key_rate - 1.0):.4f}%",
            )

    def test_bracket_placement(self):
        """
        A described DAC raises $\\xi_{phase}$ over the channel-only term by $\\mathrm{row}(e^v -
        1)$, 3.2286e-06 SNU at 8 bits and `v_err` = 0.05, an Alice-plane row sitting inside
        `budget.phase`'s $(V_A + \\xi)$ bracket.
        """
        for bits in (8, 10, 12):
            for v_err in (2e-3, 0.05):
                res = sim_link(
                    q.Asymptotic(beta=0.95),
                    xi=0.01,
                    dsp=q.DSP(phase=q.PilotPhase(v_err=v_err)),
                    iq=q.IQModulator(bits=bits),
                ).run()
                row = res.explain["xi_dac"]["value"]
                step = res.explain["xi_phase"]["value"] - budget.phase(5.0, v_err, 0.01)

                # A ratio: step differences two 1e-2 numbers, round-off 4e-9 relative.
                self.assertClose(
                    step / (row * math.expm1(v_err)),
                    1.0,
                    atol=1e-6,
                    msg=f"{bits} bits at v_err={v_err}: bracket deficit {step:.6e}",
                )

    def test_sampled_not_twice(self):
        """
        The fitted slope already carries $\\kappa$, so the sampled path's `T_claimed` is the
        clamped `res.est.T` with no second `budget.infer`, within 1% of `budget.infer(T, v_err)`
        at 200k symbols.
        """
        for seed in (1, 3, 5, 7, 11):
            res = wired(q.DSP()).run(symbols=200_000, seed=seed)
            got = res.explain["T_claimed"]["value"]
            self.assertClose(
                got,
                min(1.0, max(1e-12, res.est.T)),
                atol=0.0,
                msg=f"seed {seed}: T_claimed {got}",
            )
            self.assertClose(
                got / budget.infer(0.5, res.dsp.v_err),
                1.0,
                atol=0.01,
                msg=f"seed {seed}: T_claimed {got}, v_err {res.dsp.v_err}",
            )


class Carrier(Question):
    """
    The carrier-frequency-offset stage: coarse acquisition, then the fine slope.
    """

    def test_offset_is_acquired(self):
        """
        Coarse acquisition keeps `v_err` within 5% of the no-offset run at 0.39, 1.17, 10 and 45
        MHz, past the unwrapper's $\\pm\\pi$ ambiguity at half a cycle per 32-symbol block.
        """
        step = RATE / (4 * 32)
        base = run(n=1_000_000, seed=7, lw=1e4)
        for f in (step / 2, 1.5 * step, 1e7, 4.5e7):
            out = run(n=1_000_000, seed=7, lw=1e4, cfo=f)
            self.assertClose(
                out.v_err / base.v_err,
                1.0,
                atol=0.05,
                msg=f"cfo={f:.4g}: v_err {out.v_err:.5e}, base {base.v_err:.5e}",
            )

    def test_offset_is_reported(self):
        """
        With the lasers noiseless `SimOut.cfo` returns the offset to 0.1 Hz from 100 kHz to 45
        MHz, past coarse acquisition's 781.25 kHz grid.
        """
        for f in (1e5, 390625.0, 1e7, 4.5e7):
            out = run(n=1_000_000, seed=7, lw=0.0, cfo=f)
            self.assertClose(out.cfo, f, atol=0.1, msg=f"cfo={f:.6g} reported as {out.cfo!r}")

    def test_offset_floor_is_the_walk(self):
        """
        A 10 kHz laser pair's Wiener walk, not the estimator, sets the offset residual: it
        matches to 0.1 Hz with 10 MHz present or absent.
        """
        for seed in (1, 5, 7):
            quiet_run = run(n=1_000_000, seed=seed, lw=1e4)
            loud = run(n=1_000_000, seed=seed, lw=1e4, cfo=1e7)
            self.assertClose(
                loud.cfo - 1e7,
                quiet_run.cfo,
                atol=0.1,
                msg=f"seed={seed}: residual {loud.cfo - 1e7:.3f} Hz, walk {quiet_run.cfo:.3f} Hz",
            )

    def test_offset_past_nyquist_refused(self):
        """
        `run_symbols` refuses an offset at or past $R/2$, where one sample per symbol aliases $f
        + R$ onto $f$.
        """
        for bad in (RATE / 2, -RATE / 2, RATE, -8e8):
            self.assertFails(
                ValueError,
                "cfo must satisfy",
                lambda f=bad: run(n=100_000, cfo=f),
                msg=f"cfo={bad:.4g}",
            )

        edge = run(n=100_000, cfo=RATE / 2 - 1.0)

        self.assertFinite(edge.cfo, msg="cfo at R/2 - 1 Hz")


class Fading(Question):
    """
    Sampling-instant jitter: the half of the fading identity the slope reads, and the half the
    additive row omits.
    """

    def test_jitter_is_off_by_default(self):
        """
        Jitter named at 0.0 and jitter not named return all six scalars bit for bit.
        """
        args = (200_000, 7, 5.0, 0.5, 0.01, 0.6, 0.1, 1e4, 1e4, 0.0)
        tail = (RATE, 12.0, TONE, 32, False)
        off = _core.run_symbols(*args, *tail)
        zero = _core.run_symbols(*args, *tail, 0.0)
        for name in ("v_err", "t_hat", "t_chan", "sigma2_hat", "xi_hat", "xi_ideal"):
            self.assertEqual(
                getattr(off, name),
                getattr(zero, name),
                msg=f"{name}",
            )

    def test_jitter_domain(self):
        """
        A negative, NaN or infinite jitter is refused, and so is a finite one whose square
        overflows, $\\exp(-\\infty g^2)$ being NaN at $g = 0$.
        """
        for bad in (-1e-3, float("nan"), float("inf")):
            self.assertFails(
                ValueError,
                "jitter must be >= 0",
                lambda j=bad: run(n=100_000, jit=j),
                msg=f"jitter={bad}",
            )
        self.assertFails(
            ValueError,
            "must square to a finite width ratio",
            lambda: run(n=100_000, jit=1e200),
            msg="jitter=1e200",
        )

    def test_pilot_arm_is_untouched(self):
        """
        `v_err`, `v_wrap`, `pilot_snr` and `cfo` are bit-identical at every jitter, a mistimed
        sample moving a CW pilot in phase (1.3e-6 rad^2 at 180 MHz and a tenth of a 10 ps
        width), not amplitude.
        """
        base = run(n=400_000, seed=7, lw=1e4)
        for jit in (0.1, 0.5, 1.0):
            out = run(n=400_000, seed=7, lw=1e4, jit=jit)
            self.assertEqual(
                (out.v_err, out.v_wrap, out.pilot_snr, out.cfo),
                (base.v_err, base.v_wrap, base.pilot_snr, base.cfo),
                msg=f"jitter={jit}: pilot-arm scalars",
            )

    def test_slope_reads_the_overlap(self):
        """
        The fitted transmittance tracks $T\\langle\\sqrt\\eta\\rangle^2$ from
        `impairments.jitter_fading` against the same-seed unfaded run at j/w = 0.1, 0.25 and
        0.5.
        """
        base = run(n=400_000, seed=7)
        # atol 2e-3 = -1.8e-4 mean + 6.3 sd of 2.9e-4 at j/w = 0.5, seeds 1..16.
        for jit in (0.1, 0.25, 0.5):
            eff = fading(jit)[0]
            out = run(n=400_000, seed=7, jit=jit)
            self.assertClose(
                out.t_chan / base.t_chan / eff,
                1.0,
                atol=2e-3,
                msg=f"j/w={jit}: t ratio {out.t_chan / base.t_chan:.6f}, <sqrt eta>^2 {eff:.6f}",
            )

    def test_residual_is_the_timing_row(self):
        """
        `xi_hat` rises by `impairments.timing` plus the channel $\\xi$ renormalised by
        $1/\\langle\\sqrt\\eta\\rangle^2$, at $V_A = 10^5$ to lift the step clear of shot noise.
        """
        base = run(n=1_000_000, seed=7, va=1e5)
        # atol 0.03 = +2.5e-3 mean + 5.6 sd of 4.9e-3, seeds 1..12; gain floor sqrt(14/n) = 3.7e-3.
        for jit in (0.1, 0.25, 0.5):
            eff, var = fading(jit)[:2]
            out = run(n=1_000_000, seed=7, va=1e5, jit=jit)
            step = 1e5 * var / eff + 0.01 * (1.0 / eff - 1.0)
            self.assertClose(
                (out.xi_hat - base.xi_hat) / step,
                1.0,
                atol=0.03,
                msg=f"j/w={jit}: xi_hat step {out.xi_hat - base.xi_hat:.6e}, identity {step:.6e}",
            )

    def test_row_leaves_transmittance_at_T(self):
        """
        The timing row against an unattenuated $T$ over-states the recovered signal by 0.50% at
        j/w = 0.1, 3.1% at 0.25 and 12.5% at 0.5.
        """
        base = run(n=1_000_000, seed=7)
        # atol 1e-2 = +2.6e-3 mean + 4.1 sd of 1.8e-3, seeds 1..12.
        for jit, want in ((0.1, 1.00499), (0.25, 1.03119), (0.5, 1.12472)):
            step = fading(jit)[2]
            out = run(n=1_000_000, seed=7, jit=jit)
            ratio = 0.5 * (5.0 + base.xi_hat + step) / (out.t_chan * (5.0 + out.xi_hat))
            self.assertClose(
                ratio / want,
                1.0,
                atol=1e-2,
                msg=f"j/w={jit}: over-statement {100.0 * (ratio - 1.0):.3f}%, want {100.0 * (want - 1.0):.3f}%",
            )
            self.assertGreater(
                ratio,
                1.0,
                msg=f"j/w={jit}: ratio {ratio}",
            )


if __name__ == "__main__":
    rc = Exam(
        "DspPipeline",
        "Phase 2: symbol pipeline conventions, reproducibility and estimators",
        "dsp_pipeline.md",
    ).run(load(Pipeline))
    rc |= Exam(
        "DspRecovery",
        "Phase 2 gate: pilot phase recovery and the xi it leaves behind",
        "dsp_recovery.md",
    ).run(load(Recovery))
    rc |= Exam(
        "DspCarrier",
        "Phase 2: carrier-frequency acquisition and what it leaves behind",
        "dsp_carrier.md",
    ).run(load(Carrier))
    rc |= Exam(
        "DspFading",
        "Sampling jitter in the loop: the two halves of the fading identity",
        "dsp_fading.md",
    ).run(load(Fading))
    sys.exit(rc)
