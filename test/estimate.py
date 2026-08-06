import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

import qkd as q
from kit.cache import memo
from kit.links import frames, sim_link as link
from qkd import _core

SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

# The link as cv_bounds sees it; CHI is chi_det at the channel-output plane.
VA = 5.0
ETA = 0.6
VEL = 0.1
BETA = 0.95
EPS = 1e-10
T_CH = 0.5
XI_CH = 0.02
CHI = (1.0 + (1.0 - ETA) + 2.0 * VEL) / ETA


@memo
def clean(symbols, seed):
    """
    Perfect-reference run: 0 linewidth, 30 dB pilot, v_err ~ 3e-5, sampling only.
    """

    return link(q.Asymptotic(beta=0.95), lw=0.0, db=30.0).run(symbols, seed)


@memo
def finite(n, symbols, seed=1):
    """
    One run of `symbols` symbols at 10 kHz linewidth, claimed at block size n.
    """
    security = q.FiniteSize(beta=0.95, eps=1e-10, n=n)

    return link(security).claim(frames(symbols, seed))


def rms(values, truth):
    return math.sqrt(sum((v - truth) ** 2 for v in values) / len(values))


def measured(t=T_CH, xi=XI_CH):
    """
    Bob's residual variance at the channel-output plane, SNU; the pipeline reports it times eta/2.
    """

    return 1.0 + CHI + t * xi


def sample(t, s2, m, seed):
    """
    One round: m pairs from y = t*x + N(0, s2), reduced to the ML estimates.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, math.sqrt(VA), m)
    y = t * x + rng.normal(0.0, math.sqrt(s2), m)
    slope = float(x @ y / (x @ x))
    resid = y - slope * x

    return slope, float(resid @ resid / m)


def covers(eps, trials, m=4000):
    """
    Fraction of `trials` rounds with t_lo <= sqrt(T) and xi_hi >= the input-referred excess.
    """
    root = math.sqrt(T_CH)
    s2 = measured()
    total = (s2 - 1.0) / T_CH
    hits = 0
    for seed in range(trials):
        t_hat, s2_hat = sample(root, s2, m, seed)
        t_lo, xi_hi = _core.cv_bounds(t_hat, s2_hat, m, VA, eps)
        hits += t_lo <= root and xi_hi >= total

    return hits / trials


def stale(m, eps=EPS):
    """
    The superseded claim: Leverrier's sigma^2 = 1 + T*xi, no detector noise.
    """
    t_lo, xi_hi = _core.cv_bounds(math.sqrt(T_CH), 1.0 + T_CH * XI_CH, m, VA, eps)

    return t_lo * t_lo, xi_hi


def honest(m, eps=EPS):
    """
    The same estimator on the variance Bob measures, chi_det referred back out.
    """
    t_lo, xi_hi = _core.cv_bounds(math.sqrt(T_CH), measured(), m, VA, eps)
    t_min = t_lo * t_lo

    return t_min, max(0.0, xi_hi - CHI / t_min)


def probe_at(bound, n, eps=EPS):
    """
    Probe, not a protocol rate: i_ab from the true (T_CH, XI_CH), chi from the bound.
    """
    n_key = 0.5 * n
    delta = 7.0 * math.sqrt(math.log2(2.0 / eps) / n_key) + (2.0 / n_key) * math.log2(1.0 / eps)
    i_ab = _core.cv_rate(VA, T_CH, XI_CH, ETA, VEL, BETA, False, True)[0]
    chi = _core.cv_rate(VA, bound[0], bound[1], ETA, VEL, BETA, False, True)[1]

    return max(0.0, 0.5 * (BETA * i_ab - chi - delta))


class Estimator(Question):
    """
    Phase 4a: the ML channel estimators through q.Link, their sampling law, and the interval the
    key rate is claimed at.
    """

    def test_estimator_consistency(self):
        """
        With a perfect phase reference the estimators recover the pinned $T = 0.5$ and
        input-referred $\\xi = 0.02$ from $10^6$ symbols, each inside five sigma.
        """
        res = clean(1_000_000, seed=1)

        # atol = 5 sd of the estimate over seeds 1..24: sd(T_hat) = 9.11e-4, sd(xi_hat) = 7.31e-3.
        self.assertClose(res.est.T, 0.5, atol=5e-3, msg=f"T_hat = {res.est.T:.6f}")
        self.assertClose(res.est.xi, 0.02, atol=3.7e-2, msg=f"xi_hat = {res.est.xi:.6f}")
        self.assertLess(res.dsp.v_err, 1e-4, msg=f"v_err = {res.dsp.v_err:.2e}")

    def test_error_falls_as_root_n(self):
        """
        Over 12 seeds at $N = 10^4$ to $10^6$ the RMS error of both estimators falls by a factor
        in [4, 25], which brackets $1/\\sqrt N = 10$ and excludes $N^{-1/4}$ and $1/N$.
        """
        errs = []
        xerr = []
        for n in (10_000, 100_000, 1_000_000):
            runs = [clean(n, seed) for seed in SEEDS]
            errs.append(rms([r.est.T for r in runs], 0.5))
            xerr.append(rms([r.est.xi for r in runs], 0.02))
        self.assertMonotone(errs, rising=False, msg=f"T error not falling: {errs}")
        self.assertMonotone(xerr, rising=False, msg=f"xi error not falling: {xerr}")
        self.assertTrue(
            4.0 <= errs[0] / errs[2] <= 25.0,
            msg=f"T error fell by {errs[0] / errs[2]:.2f} over two decades",
        )
        self.assertTrue(
            4.0 <= xerr[0] / xerr[2] <= 25.0,
            msg=f"xi error fell by {xerr[0] / xerr[2]:.2f} over two decades",
        )

    def test_bounds_bracket_truth(self):
        """
        At seeds 1 to 4, $t_{min} \\le T$ and $\\xi_{max} \\ge \\xi$ as applied.
        """
        for seed in (1, 2, 3, 4):
            res = finite(1e6, 1_000_000, seed=seed)
            self.assertLessEqual(
                res.est.t_min,
                res.oracle.T,
                msg=f"seed {seed}: t_min {res.est.t_min:.6f} above true T",
            )
            self.assertGreaterEqual(
                res.est.xi_max,
                res.oracle.xi,
                msg=f"seed {seed}: xi_max {res.est.xi_max:.6f} below true xi " f"{res.oracle.xi:.6f}",
            )

    def test_bounds_only_under_finite_size(self):
        """
        `q.Asymptotic` reports no interval; `q.FiniteSize` fills `t_min` and `xi_max`, and
        `LinkResult` carries the same pair.
        """
        loose = link(q.Asymptotic(beta=0.95)).claim(frames(200_000, 1))
        tight = finite(2e5, 200_000)

        self.assertIsNone(loose.est.t_min, msg="asymptotic t_min")
        self.assertIsNone(loose.est.xi_max, msg="asymptotic xi_max")
        self.assertClose(tight.est.t_min, tight.t_min, msg="t_min")
        self.assertClose(tight.est.xi_max, tight.xi_max, msg="xi_max")


class Honesty(Question):
    """
    Phase 4b: res.oracle is what the channel did, and the gap is finite-size cost.
    """

    def test_oracle_is_the_channel(self):
        """
        `res.oracle` reports the pinned $T$ and the input-referred $\\xi$ including $\\xi_{phase}$,
        and the estimates differ from it.
        """
        res = finite(1e8, 200_000)

        self.assertClose(res.oracle.T, 0.5, msg=f"oracle T = {res.oracle.T}")
        self.assertClose(res.oracle.xi, res.explain["xi_input"]["value"], msg=f"oracle xi = {res.oracle.xi}")
        self.assertGreater(abs(res.est.T - res.oracle.T), 0.0, msg="T_hat == oracle T")
        self.assertGreater(res.oracle.xi, 0.02, msg=f"oracle xi = {res.oracle.xi}")

    def test_est_never_beats_oracle(self):
        """
        Under `q.FiniteSize` the reported rate is positive and never exceeds the oracle rate at 12
        seeds.
        """
        for seed in SEEDS:
            res = finite(1e8, 200_000, seed=seed)
            self.assertLessEqual(
                res.key_rate,
                res.oracle.key_rate,
                msg=f"seed {seed}: reported {res.key_rate:.6f} beats oracle " f"{res.oracle.key_rate:.6f}",
            )
            self.assertGreater(res.key_rate, 0.0, msg=f"seed {seed}: no key at all")

    def test_explain_labels_both(self):
        """
        `explain()` labels `T_est`, `xi_est`, `t_min`, `xi_max`, `xi_oracle` and `key_oracle`
        "derived", and `stages` names the estimation step.
        """
        res = finite(1e8, 200_000)
        info = res.explain
        self.assertEqual(info["T_est"]["value"], res.est.T, msg="T_est must be the estimate")
        self.assertEqual(
            info["key_oracle"]["value"],
            res.oracle.key_rate,
            msg="key_oracle must be the oracle rate",
        )
        self.assertEqual(
            info["stages"],
            "symbol pipeline + pilot DSP + parameter estimation",
            msg="stages must name the estimation step",
        )

        for key in ("T_est", "xi_est", "t_min", "xi_max", "xi_oracle", "key_oracle"):
            self.assertEqual(info[key]["label"], "derived", msg=f"{key} must be labelled derived")


class Bounds(Question):
    """
    Phase 4c: cv_bounds fed the variance Bob measures, not the Leverrier one.
    """

    def test_plane_is_bobs(self):
        """
        `sigma2` at Bob's ADC times $2/\\eta$ is $1 + \\chi_{det} + \\hat T\\hat\\xi$, the
        channel-output plane `cv_bounds` takes.
        """
        res = finite(1e8, 200_000)
        want = 1.0 + CHI + res.est.T * res.est.xi
        got = res.est.sigma2 * 2.0 / ETA

        self.assertClose(got, want, msg=f"sigma2 * 2/eta = {got:.6f}, want {want:.6f}")
        self.assertGreater(got, 1.0 + CHI, msg=f"sigma2 * 2/eta = {got:.6f}")

    def test_finite_uses_bounds(self):
        """
        `cv_finite`'s (t_min, xi_max) is `cv_bounds` fed Bob's measured variance with
        $\\chi_{det}$ referred back out, rebuilt here to match.
        """
        out = _core.cv_finite(VA, T_CH, XI_CH, ETA, VEL, BETA, False, True, 1e8, 0.5, EPS, EPS, EPS)
        t_min, xi_max = honest(0.5e8)
        self.assertClose(out[3], t_min, msg=f"t_min {out[3]:.9f} != {t_min:.9f}")
        self.assertClose(out[4], xi_max, msg=f"xi_max {out[4]:.9f} != {xi_max:.9f}")

    def test_interval_covers(self):
        """
        Over 800 Monte Carlo rounds the worst-case pair covers the truth at 0.979 for
        $\\epsilon = 0.05$ and 0.883 for $\\epsilon = 0.2$, each above $1 - \\epsilon$.
        """
        loose = covers(0.2, 800)
        tight = covers(0.05, 800)

        self.assertGreaterEqual(tight, 0.95, msg=f"coverage at eps 0.05 = {tight:.4f}")
        self.assertGreaterEqual(loose, 0.80, msg=f"coverage at eps 0.2 = {loose:.4f}")
        self.assertGreater(tight, loose, msg=f"coverage {tight:.4f} <= {loose:.4f}")

    def test_detector_costs_key(self):
        """
        Against the detector-free variance, Bob's measured variance widens the interval and lowers
        the probe rate ($I_{AB}$ true, $\\chi$ bounded) from 0.0662 to 0.0536 bit/symbol at $n =
        10^7$ (24%), 6.3% at $10^8$, 6.9x at $10^6$.
        """
        for n in (1e6, 1e7, 1e8, 1e9):
            old = stale(0.5 * n)
            new = honest(0.5 * n)
            self.assertLess(new[0], old[0], msg=f"n={n:.0e}: t_min must fall, not rise")
            self.assertGreater(new[1], old[1], msg=f"n={n:.0e}: xi_max must rise, not fall")
            self.assertLess(
                probe_at(new, n),
                probe_at(old, n),
                msg=f"n={n:.0e}: the wider interval must give the lower probe",
            )

        cost = probe_at(stale(5e6), 1e7) / probe_at(honest(5e6), 1e7)
        self.assertTrue(
            1.15 <= cost <= 1.35,
            msg=f"the idealisation overstated n=1e7 by {100 * (cost - 1):.1f}%",
        )

    def test_no_key_at_small_block(self):
        """
        At $n = 10^4$ to $10^6$ the reported rate is zero and the oracle rate positive.
        """
        for n in (1e4, 1e5, 1e6):
            res = finite(n, 200_000)
            self.assertEqual(res.key_rate, 0.0, msg=f"n={n:.0e}: key_rate = {res.key_rate}")
            self.assertGreater(res.oracle.key_rate, 0.0, msg=f"n={n:.0e}: oracle key_rate = {res.oracle.key_rate}")


if __name__ == "__main__":
    rc = Exam(
        "EstimatorRecovery",
        "Phase 4: channel estimation from simulated frames and its 1/sqrt(N) law",
        "estimate_recovery.md",
    ).run(load(Estimator))
    rc |= Exam(
        "EstimateVsOracle",
        "Phase 4: the reported rate against the god's-eye rate, and the gap",
        "estimate_finite.md",
    ).run(load(Honesty))
    rc |= Exam(
        "EstimateBounds",
        "Phase 4: cv_bounds planes, Monte Carlo coverage, and what the old estimator overstated",
        "estimate_bounds.md",
    ).run(load(Bounds))
    sys.exit(rc)
