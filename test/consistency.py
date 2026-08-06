import inspect
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.anchors import BETA, HARDWARE as GRID
from kit.cache import memo
from kit.forms import untrusted
from kit.noise import assembled
from qkd import _core, budget

XI = 0.01

# (V_A, T, xi at the channel input, eta, v_el) for run_symbols; lasers noiseless, no carrier offset.
PLAN = (5.0, 0.5, 0.01, 0.6, 0.1)

RATE = 100e6

TONE = 1.8

# 24 seeds per point: a standard error is then 1/5 of a deviation.
SEEDS = 24

SIZES = (25600, 102400, 409600)

# Trusted: the untrusted rate at this receiver is negative and clamps.
TRUTH = _core.cv_rate(*PLAN, BETA, True, True)[2]


def label(t, eta, vel, hom):
    kind = "hom" if hom else "het"

    return f"T={t:.4f} eta={eta} v_el={vel} {kind}"


def paranoid(va, t, xi, eta, vel, hom):
    return _core.cv_rate(va, t, xi, eta, vel, BETA, hom, False)


def folded(va, t, xi, eta, vel, hom):
    # Untrusted by hand: the receiver merged into the channel.
    total, merged = untrusted(t, xi, eta, vel, 1.0 if hom else 2.0)

    return _core.cv_rate(va, total, merged, 1.0, 0.0, BETA, hom, True)


def hardware(t):
    return assembled(2.0, t)


def shown(info):
    """
    The value of every labelled entry of a report, keyed by name.
    """

    return {k: v["value"] for k, v in info.items() if isinstance(v, dict)}


def swap(la, lb, va=1e5, beta=0.98):
    """
    A CV relay: two senders, two fibre arms, one Bell measurement Eve owns.
    """

    return q.Swap(
        alice=q.Sender(modulation=q.GaussianModulation(v_a=va)),
        bob=q.Sender(modulation=q.GaussianModulation(v_a=va)),
        relay=q.Relay(bell=q.BellDetector(eta=1.0, v_el=0.0)),
        channels=(q.Fiber(length=la), q.Fiber(length=lb)),
        security=q.Asymptotic(beta=beta),
    )


def sample(n, seed, db):
    """
    One symbol-level run of the shared plan at the named pilot depth.
    """

    return _core.run_symbols(n, seed, *PLAN, 0.0, 0.0, 0.0, RATE, db, TONE, 32, False)


@memo
def spread(n, db=30.0):
    """
    (mean, sd) over SEEDS runs of n symbols of the errors in T, xi and the key rate; a 30 dB pilot
    leaks 1/60 of the default's.
    """
    rows = [[], [], []]
    for seed in range(1, SEEDS + 1):
        out = sample(n, seed, db)

        # An unbiased xi_hat can be negative, which cv_rate refuses; clamped here only.
        got = _core.cv_rate(
            PLAN[0],
            out.t_chan,
            max(out.xi_hat, 0.0),
            PLAN[3],
            PLAN[4],
            BETA,
            True,
            True,
        )
        rows[0].append(out.t_chan - PLAN[1])
        rows[1].append(out.xi_hat - PLAN[2])
        rows[2].append(got[2] - TRUTH)

    return tuple((statistics.fmean(r), statistics.pstdev(r)) for r in rows)


@memo
def leak(db, n=204800):
    """
    The pilot's excess-noise contribution divided by its power.
    """
    runs = [sample(n, seed, db) for seed in range(1, 17)]
    got = [out.xi_hat - out.xi_ideal for out in runs]

    return statistics.fmean(got) / 10.0 ** (-db / 10.0)


def rate(va, t, xi, eta, vel, hom):
    det = q.Homodyne if hom else q.Heterodyne
    link = q.Link(
        modulation=q.GaussianModulation(v_a=va),
        channel=q.Channel(T=t, xi=xi, ref="input"),
        bob=q.Bob(detector=det(eta=eta, v_el=vel, trusted=False)),
        security=q.Asymptotic(beta=BETA),
    )

    return link.run().key_rate


class Consistency(Question):
    """
    One layer models Bob's electronic noise.
    """

    def test_merged_channel(self):
        """
        Untrusted mode equals a trusted run on the merged channel T -> eta*T, xi -> xi +
        mu*v_el/(eta*T), mu = 1 homodyne and 2 heterodyne, over the (T, eta, v_el, V_A) grid:
        Laudenbach & Pacher arXiv:1904.01970 Sec. 2.
        """
        for t, eta, vel, va in GRID:
            for hom in (True, False):
                got = paranoid(va, t, XI, eta, vel, hom)[2]
                want = folded(va, t, XI, eta, vel, hom)[2]
                tag = label(t, eta, vel, hom)
                self.assertClose(
                    got,
                    want,
                    atol=1e-12,
                    msg=f"{tag}: {got:.6f} vs {want:.6f}",
                )

    def test_iab_invariant(self):
        """
        Trusted and untrusted I_AB agree to 1e-12 over the grid on both detectors.
        """
        for t, eta, vel, va in GRID:
            for hom in (True, False):
                unt = paranoid(va, t, XI, eta, vel, hom)[0]
                tru = _core.cv_rate(va, t, XI, eta, vel, BETA, hom, True)[0]
                tag = label(t, eta, vel, hom)
                self.assertClose(
                    unt,
                    tru,
                    atol=1e-12,
                    msg=f"{tag}: I_AB {unt:.6f} vs {tru:.6f}",
                )

    def test_untrusted_costs(self):
        """
        The untrusted rate never exceeds the trusted one, equal at eta = 1, v_el = 0.
        """
        for t, eta, vel, va in GRID:
            for hom in (True, False):
                unt = paranoid(va, t, XI, eta, vel, hom)[2]
                tru = _core.cv_rate(va, t, XI, eta, vel, BETA, hom, True)[2]
                tag = label(t, eta, vel, hom)
                self.assertLessEqual(unt, tru + 1e-12, msg=f"{tag}: {unt:.6f} > {tru:.6f}")

        ideal = paranoid(4.0, 0.4, XI, 1.0, 0.0, False)[2]
        same = _core.cv_rate(4.0, 0.4, XI, 1.0, 0.0, BETA, False, True)[2]

        self.assertClose(ideal, same, atol=1e-12, msg=f"ideal receiver: {ideal} vs {same}")

    def test_mu_doubling(self):
        """
        The untrusted heterodyne rate matches the mu = 2 fold to 1e-12 and not mu = 1 (Laudenbach
        arXiv:1703.09278 Eqs. (9.93), (9.94)).
        """
        t, eta, vel, va = 0.4, 0.6, 0.1, 4.0
        one = _core.cv_rate(va, eta * t, XI + vel / (eta * t), 1.0, 0.0, BETA, False, True)
        two = folded(va, t, XI, eta, vel, False)[2]
        got = paranoid(va, t, XI, eta, vel, False)[2]

        self.assertGreater(one[2] - two, 1e-3, msg=f"mu=1 {one[2]:.6f}, mu=2 {two:.6f}")
        self.assertClose(
            got,
            two,
            atol=1e-12,
            msg=f"untrusted {got:.6f} vs mu=2 {two:.6f}",
        )

    def test_eta_referral(self):
        """
        The untrusted homodyne rate matches xi + v_el/(eta*T) to 1e-12 and not xi + v_el/T.
        """
        t, eta, vel, va = 0.4, 0.6, 0.1, 4.0
        got = paranoid(va, t, XI, eta, vel, True)[2]
        with_eta = folded(va, t, XI, eta, vel, True)[2]
        no_eta = _core.cv_rate(va, eta * t, XI + vel / t, 1.0, 0.0, BETA, True, True)

        self.assertClose(got, with_eta, atol=1e-12, msg=f"untrusted {got:.6f} vs {with_eta:.6f}")
        self.assertGreater(
            no_eta[2] - with_eta,
            1e-3,
            msg=f"v_el/T {no_eta[2]:.6f} vs v_el/(eta*T) {with_eta:.6f}",
        )

    def test_budget_silent(self):
        """
        budget.assemble takes no vel or trusted argument, budget.detector does not exist, and an
        assembled budget has no detector row.
        """
        names = set(inspect.signature(budget.assemble).parameters)
        self.assertEqual(names & {"vel", "trusted"}, set(), msg=f"assemble params {names}")
        self.assertFalse(hasattr(budget, "detector"), msg="budget.detector")

        bud = hardware(0.8)
        self.assertNotIn("detector", bud.at("input"), msg="detector row")

    def test_single_count(self):
        """
        A hardware budget through an untrusted homodyne Link matches the merged channel to 1e-12.
        """
        t, eta, vel, va = 0.8, 0.98, 0.01, 2.0
        bud = hardware(t)
        got = rate(va, t, bud.total, eta, vel, True)
        want = folded(va, t, bud.total, eta, vel, True)[2]

        self.assertGreater(want, 0.0, msg=f"merged rate {want:.6f}")
        self.assertClose(got, want, atol=1e-12, msg=f"Link {got:.6f} vs merged {want:.6f}")

    def test_double_count(self):
        """
        Adding mu*v_el/T to a hardware budget by hand lowers the Link rate by over 1e-3 on both
        detectors.
        """
        t, eta, vel, va = 0.8, 0.98, 0.01, 2.0
        bud = hardware(t)
        for hom in (True, False):
            mu = 1.0 if hom else 2.0
            once = rate(va, t, bud.total, eta, vel, hom)
            twice = rate(va, t, bud.total + mu * vel / t, eta, vel, hom)
            tag = label(t, eta, vel, hom)

            self.assertGreater(
                once - twice,
                1e-3,
                msg=f"{tag}: single {once:.6f} vs doubled {twice:.6f}",
            )


class Trust(Question):
    """
    The untrusted rate never exceeds the trusted one anywhere in parameter space.
    """

    def test_trust_never_pays(self):
        """
        Over 20000 random configurations (V_A across six decades, T across four, xi and v_el to 1
        SNU, eta from 0.01 to 1, both detectors) the untrusted rate never exceeds the trusted one,
        slack 0.
        """
        rng = random.Random(20260903)
        count = 0
        worst = 0.0
        for _ in range(20000):
            va = 10.0 ** rng.uniform(-3.0, 3.0)
            t = 10.0 ** rng.uniform(-4.0, 0.0)
            xi = rng.choice((0.0, 10.0 ** rng.uniform(-4.0, 0.0)))
            eta = 10.0 ** rng.uniform(-2.0, 0.0)
            vel = rng.choice((0.0, 10.0 ** rng.uniform(-3.0, 0.0)))
            for hom in (True, False):
                try:
                    unt = _core.cv_rate(va, t, xi, eta, vel, BETA, hom, False)[2]
                    tru = _core.cv_rate(va, t, xi, eta, vel, BETA, hom, True)[2]
                except ValueError:
                    continue

                count += 1
                worst = max(worst, unt - tru)
                self.assertLessEqual(
                    unt,
                    tru,
                    msg=f"V_A={va:.4g} T={t:.4g} xi={xi:.4g} eta={eta:.4g} v_el={vel:.4g} "
                    f"hom={hom}: {unt:.9f} > {tru:.9f}",
                )

        self.assertGreater(count, 35000, msg=f"accepted {count}")
        self.assertLessEqual(worst, 0.0, msg=f"worst {worst:.3e}")

    def test_trust_is_free(self):
        """
        At eta = 1 and v_el = 0 the trusted and untrusted cv_rate outputs are bit-identical over
        2000 random configurations.
        """
        rng = random.Random(8675309)
        for _ in range(2000):
            va = 10.0 ** rng.uniform(-3.0, 3.0)
            t = 10.0 ** rng.uniform(-4.0, 0.0)
            xi = 10.0 ** rng.uniform(-4.0, 0.0)
            for hom in (True, False):
                try:
                    unt = _core.cv_rate(va, t, xi, 1.0, 0.0, BETA, hom, False)
                    tru = _core.cv_rate(va, t, xi, 1.0, 0.0, BETA, hom, True)
                except ValueError:
                    continue

                self.assertEqual(
                    unt,
                    tru,
                    msg=f"V_A={va:.4g} T={t:.4g} xi={xi:.4g} hom={hom}: {unt} vs {tru}",
                )


class Clamping(Question):
    """
    SwapResult carries two CV relay quantities and the Python layer clamps one.
    """

    def test_one_clamp(self):
        """
        At a relay position that cannot distil, SwapResult.key_rate is max(0, key_bound) = 0.0 while
        explain()'s key_bound = key_raw and attack.key_rate = key_attack stay negative.
        """
        res = swap(5.0, 5.0).run()
        info = shown(res.explain)

        self.assertEqual(res.key_rate, 0.0, msg=f"key_rate {res.key_rate}")
        self.assertLess(info["key_bound"], 0.0, msg=f"key_bound {info['key_bound']}")
        self.assertEqual(
            info["key_bound"],
            info["key_raw"],
            msg=f"key_bound {info['key_bound']} vs key_raw {info['key_raw']}",
        )
        self.assertEqual(
            res.key_rate,
            max(0.0, info["key_bound"]),
            msg=f"key_rate {res.key_rate} vs max(0, key_bound)",
        )
        self.assertLess(
            res.attack.key_rate,
            0.0,
            msg=f"attack.key_rate clamped: {res.attack.key_rate}",
        )
        self.assertEqual(
            res.attack.key_rate,
            info["key_attack"],
            msg=f"attack.key_rate {res.attack.key_rate} vs key_attack {info['key_attack']}",
        )

    def test_two_quantities(self):
        """
        At a live relay position key_bound and key_attack differ by over 0.1, and lowering beta to
        0.90 lowers key_attack and leaves key_bound bit-identical.
        """
        live = shown(swap(0.1, 5.0).run().explain)
        slow = shown(swap(0.1, 5.0, beta=0.90).run().explain)

        self.assertGreater(live["key_bound"], 0.0, msg=f"key_bound {live['key_bound']}")
        self.assertGreater(
            abs(live["key_bound"] - live["key_attack"]),
            0.1,
            msg=f"key_bound {live['key_bound']} vs key_attack {live['key_attack']}",
        )
        self.assertEqual(
            live["key_bound"],
            slow["key_bound"],
            msg=f"key_bound {live['key_bound']} vs {slow['key_bound']} at beta 0.90",
        )
        self.assertLess(
            slow["key_attack"],
            live["key_attack"],
            msg=f"key_attack {slow['key_attack']} at beta 0.90 vs {live['key_attack']}",
        )

    def test_bound_converges(self):
        """
        At beta = 1 the gap between key_bound and key_attack falls tenfold per decade of V_A from
        1e3 to 1e5.
        """
        gaps = []
        for va in (1e3, 1e4, 1e5):
            info = shown(swap(0.1, 5.0, va=va, beta=1.0).run().explain)
            gaps.append(abs(info["key_bound"] - info["key_attack"]))

        self.assertMonotone(gaps, rising=False, msg=f"gaps {gaps}")

        for i in range(len(gaps) - 1):
            step = gaps[i] / gaps[i + 1]

            # atol 0.5: deterministic second-order term in 1/V_A, worst miss 0.37.
            self.assertClose(step, 10.0, atol=0.5, msg=f"decade {i}: {step:.4f}")


class Estimated(Question):
    """
    The DSP estimates recover the (T, xi) and key rate of the channel the simulation was handed.
    """

    def test_estimator_unbiased(self):
        """
        With noiseless lasers and a 30 dB pilot the estimated T and xi sit within five standard
        errors of the simulated channel from 25600 to 409600 symbols; at the 12 dB default xi is
        biased by +1.2e-2 SNU.
        """
        for n in SIZES:
            errs = spread(n)
            self.assertLessEqual(
                abs(errs[0][0]),
                5.0 * errs[0][1] / math.sqrt(SEEDS),
                msg=f"n={n}: T error {errs[0][0]:+.3e}, se {errs[0][1] / math.sqrt(SEEDS):.3e}",
            )
            self.assertLessEqual(
                abs(errs[1][0]),
                5.0 * errs[1][1] / math.sqrt(SEEDS),
                msg=f"n={n}: xi error {errs[1][0]:+.3e}, se {errs[1][1] / math.sqrt(SEEDS):.3e}",
            )

    def test_estimator_narrows(self):
        """
        Over sixteen times the symbols the spread of the T estimate narrows between 2.5 and 8 times,
        measured 4.2 to 5.6 against the square-root law's 4.0.
        """
        sds = [spread(n)[0][1] for n in SIZES]
        self.assertMonotone(sds, rising=False, msg=f"spreads {sds}")

        step = sds[0] / sds[-1]
        self.assertGreater(step, 2.5, msg=f"narrowing {step:.3f}")
        self.assertLess(step, 8.0, msg=f"narrowing {step:.3f}")

    def test_rate_converges(self):
        """
        Trusted homodyne cv_rate on the estimates lands within three standard errors of cv_rate on
        the truth at 409600 symbols.
        """
        errs = spread(SIZES[-1])
        mean, sd = errs[2]

        self.assertGreater(TRUTH, 0.0, msg=f"TRUTH {TRUTH}")
        self.assertLessEqual(
            abs(mean),
            3.0 * sd / math.sqrt(SEEDS),
            msg=f"key error {mean:+.3e}, se {sd / math.sqrt(SEEDS):.3e}",
        )

    def test_pilot_leaks(self):
        """
        The pilot leak xi_hat - xi_ideal divided by 10^(-pilot_db/10) holds 0.155 SNU to 2.6% at
        pilot depths 3, 6 and 9 dB, asserted to 7%.
        """
        ks = [leak(db) for db in (3.0, 6.0, 9.0)]
        self.assertMonotone(ks, rising=False, msg=f"ks {ks}")

        # atol 0.07: second-order, not sampling; mean 0.025, sd 0.009, worst 0.036 over six 16-seed blocks.
        self.assertClose(
            max(ks) / min(ks),
            1.0,
            atol=0.07,
            msg=f"ks {ks}",
        )
        self.assertGreater(ks[0], 0.1, msg=f"ks[0] {ks[0]:.4f}")


if __name__ == "__main__":
    rc = Exam(
        "Consistency",
        "Cross-layer: budget and key rate must model the untrusted detector once",
        "consistency.md",
    ).run(load(Consistency))
    rc |= Exam(
        "TrustModel",
        "Trusting the receiver may only ever pay, and costs nothing when ideal",
        "consistency_trust.md",
    ).run(load(Trust))
    rc |= Exam(
        "RelayClamp",
        "One clamp on the relay's bound, none on anything else it reports",
        "consistency_clamp.md",
    ).run(load(Clamping))
    rc |= Exam(
        "Estimated",
        "The sampled pipeline recovers the channel the closed form was given",
        "consistency_estimated.md",
    ).run(load(Estimated))
    sys.exit(rc)
