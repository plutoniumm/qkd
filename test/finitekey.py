import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import ALPHA, ETA_BOB, E_DET, F_REC, P_DARK, Y_DARK
from kit.checks import Guarded
from kit.forms import h2
from kit.links import basis
import qkd as q
from qkd import _core

# Finite-key decoy-state BB84, Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A
# 89, 022307 (2014). A key length in bits per block, not a rate; q.Link divides by the block.
MU, NU1, NU2 = 0.5, 0.1, 0.0
PROBS = (0.7, 0.15, 0.15)

# bb84_length's eps_sec is the total, 21*eps in Lim; EPS_SEC and EPS_COR are q.KeyBlock's defaults.
EPS_SEC = 1e-10
EPS_COR = 1e-15
EPS = EPS_SEC / 21.0

# Zhang, Zhao, Razavi & Ma, Phys. Rev. A 95, 012333 (2017), Table 3, BB84 row:
# a Chernoff bound, which Lim's Hoeffding bound must not exceed.
ZR_RATE = 3.04e-6
ZR_INTS = (0.370, 0.126, 0.0)
ZR_PROBS = (0.650, 0.250, 0.100)


def gains(length):
    """
    Per-intensity (gain, error rate) of the GYS receiver at a fibre length.
    """
    eta = _core.decoy_eta(ALPHA, length, ETA_BOB)
    out = []
    for mu_k in (MU, NU1, NU2):
        if mu_k <= 0.0:
            out.append((Y_DARK, 0.5))
        else:
            out.append(_core.decoy_gain(mu_k, eta, Y_DARK, E_DET))

    return out


def counts(total, length, sift=0.5):
    """
    Expected (n_key, m_key, n_test, m_test) for `total` emitted pulses, q.Link's form without Alice.
    """
    key = _core.key_share(sift)
    rows = [[], [], [], []]
    for p_k, (gain, err) in zip(PROBS, gains(length)):
        sent = total * p_k * gain
        rows[0].append(sent * key)
        rows[1].append(sent * key * err)
        rows[2].append(sent * (sift - key))
        rows[3].append(sent * (sift - key) * err)

    return tuple(tuple(row) for row in rows)


def bounds(total, length, sift=0.5):
    """
    (s0, s1, phi, n_key, e_key) at one block size and fibre length.
    """
    n_key, m_key, n_test, m_test = counts(total, length, sift)
    args = (MU, NU1, NU2, PROBS)
    s0, s1 = _core.decoy_counts(*args, n_key, EPS)
    _, other = _core.decoy_counts(*args, n_test, EPS)
    v1 = _core.decoy_errors(*args, m_test, EPS)
    phi = _core.bb84_phase(v1, other, s1, EPS_SEC)
    kept = sum(n_key)

    return s0, s1, phi, kept, sum(m_key) / kept


def length(total, dist, sift=0.5, eps=EPS_SEC, fer=None):
    """
    Lim's key length in bits at one block size and fibre length.
    """
    s0, s1, phi, kept, err = bounds(total, dist, sift)
    out = _core.bb84_length(s0, s1, phi, kept, err, F_REC, eps, EPS_COR)
    if fer is not None:
        out *= 1.0 - fer

    return out


def link(dist, block=None, sift=0.5, ints=(MU, NU1, NU2), probs=PROBS, alice=None):
    """
    The GYS-receiver basis-keyed link, asymptotic unless given a q.KeyBlock.
    """

    return basis(dist, alice=alice, intens=ints, probs=probs, sift=sift, block=block)


def asymptote(dist):
    """
    (y1, e1) from the rate-based decoy bounds at the same gains, the infinite-block limit.
    """
    (q_mu, e_mu), (q_nu, e_nu), _ = gains(dist)

    return _core.decoy_bounds(MU, NU1, NU2, q_mu, e_mu, q_nu, e_nu, Y_DARK, 0.5)[:2]


def weight(n):
    """
    Lim's tau_n, the probability the source emitted exactly n photons.
    """

    return sum(p * math.exp(-k) * k**n / math.factorial(n) for p, k in zip(PROBS, (MU, NU1, NU2)))


class LimEstimator(Question):
    """
    The counted-event decoy bounds Lim's key length is built from.
    """

    def test_single_limit(self):
        """
        Cross-engine: s1 per pulse rises onto the rate-based Y1 times tau_1 and the key share
        from below as the block grows.
        """
        dist = 50.0
        want = weight(1) * asymptote(dist)[0] * _core.key_share(0.5)
        seen = [bounds(total, dist)[1] / total for total in (1e10, 1e12, 1e16)]
        self.assertMonotone(seen, rising=True, msg=f"s1 per pulse must rise: {seen}")

        for got in seen:
            self.assertLessEqual(got, want, msg=f"s1 per pulse {got} above {want}")
        self.assertClose(seen[-1], want, atol=1e-7, msg="s1 must reduce to tau_1 * y1 * share")

    def test_phase_limit(self):
        """
        Cross-engine: the phase-error bound falls onto the rate-based e_1 from above as the
        block grows.
        """
        dist = 50.0
        want = asymptote(dist)[1]
        seen = [bounds(total, dist)[2] for total in (1e10, 1e12, 1e16)]
        self.assertMonotone(seen, rising=False, msg=f"phi must fall with N: {seen}")

        for phi in seen:
            self.assertGreaterEqual(phi, want, msg=f"phi {phi} below e_1 {want}")
        self.assertClose(seen[-1], want, atol=5e-5, msg="phi must reduce to e_1")

    def test_vacuum_clamp(self):
        """
        s0 reads zero on a short block, never falls with N, and reaches tau_0 * Y_0 times the key
        share from below.
        """
        totals = (1e9, 1e10, 1e12, 1e14, 1e16)
        seen = [bounds(total, 50.0)[0] for total in totals]
        want = weight(0) * Y_DARK * _core.key_share(0.5)
        self.assertClose(seen[0], 0.0, msg="a short block certifies no vacuum")
        self.assertMonotone(seen, rising=True, strict=False, msg=f"s0 must not fall with N: {seen}")

        # Residual: Hoeffding width on the total detection count, closing as 1/sqrt(N), 1.1% at 1e16.
        rel = [want - seen[i] / totals[i] for i in (-2, -1)]

        for gap in rel:
            self.assertGreater(gap, 0.0, msg=f"s0 must stay below tau_0 * Y_0: {gap}")
        self.assertLess(rel[1], rel[0], msg=f"the gap must close with N: {rel}")
        self.assertClose(seen[-1] / totals[-1], want, atol=1e-8, msg="s0 must reduce to tau_0 * Y_0")

    def test_sampling_term(self):
        """
        The phase bound exceeds the raw test-basis error ratio by a positive sampling penalty
        that shrinks with N.
        """
        gaps = []
        for total in (1e11, 1e13):
            _, key, test, errs = counts(total, 50.0)
            args = (MU, NU1, NU2, PROBS)
            s1 = _core.decoy_counts(*args, key, EPS)[1]
            other = _core.decoy_counts(*args, test, EPS)[1]
            v1 = _core.decoy_errors(*args, errs, EPS)
            gaps.append(_core.bb84_phase(v1, other, s1, EPS_SEC) - v1 / other)

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"the sampling term must be positive: {gap}")
        self.assertLess(gaps[1], gaps[0], msg=f"it must shrink with N: {gaps}")

    def test_empty_test(self):
        """
        With no certified test sample the phase bound is 1/2, and with no certified error 0.
        """
        self.assertClose(
            _core.bb84_phase(0.0, 0.0, 1e6, EPS_SEC),
            0.5,
            msg="an empty test basis certifies nothing",
        )
        self.assertClose(
            _core.bb84_phase(0.0, 1e6, 1e6, EPS_SEC),
            0.0,
            msg="a test basis with no certified error takes no sampling width",
        )


class LimLength(Question):
    """
    The key length itself: Lim Eq. (1), in bits for one block.
    """

    def test_length_units(self):
        """
        The length is floored to whole bits, positive at 1e12 pulses and zero at 1e7.
        """
        out = length(1e12, 50.0)
        self.assertGreater(out, 0.0, msg="this block must certify a key")
        self.assertClose(out, math.floor(out), msg="a key length is whole bits")
        self.assertClose(length(1e7, 50.0), 0.0, msg="a short block certifies none")

    def test_block_monotone(self):
        """
        Length and length per pulse both rise with the block size.
        """
        totals = (1e10, 1e11, 1e12, 1e13, 1e14)
        lens = [length(total, 50.0) for total in totals]
        rates = [lens[i] / totals[i] for i in range(len(totals))]
        self.assertMonotone(lens, rising=True, msg=f"length must rise: {lens}")
        self.assertMonotone(rates, rising=True, msg=f"rate must rise: {rates}")

    def test_reach_monotone(self):
        """
        At a fixed block the length falls with fibre length to zero at 150 km.
        """
        lens = [length(1e12, dist) for dist in (25.0, 50.0, 75.0, 100.0, 150.0)]
        self.assertMonotone(lens, rising=False, msg=f"length must fall: {lens}")
        self.assertClose(lens[-1], 0.0, msg="150 km certifies nothing at this block")

    def test_epsilon_price(self):
        """
        A weaker secrecy parameter buys a longer key.
        """
        lens = [length(1e12, 50.0, eps=eps) for eps in (1e-15, 1e-10, 1e-5)]
        self.assertMonotone(lens, rising=True, msg=f"a weaker eps pays: {lens}")

    def test_frame_errors(self):
        """
        A frame error rate scales the length, a failed frame yielding and leaking nothing.
        """
        full = length(1e12, 50.0)
        half = length(1e12, 50.0, fer=0.5)
        self.assertClose(half, 0.5 * full, atol=1e-9, msg="fer scales the length")

    def test_epsilon_share(self):
        """
        bb84_eps is the engine's per-bound share, and 21 of them recompose eps_sec.
        """
        for eps in (1e-15, EPS_SEC, 1e-5):
            each = _core.bb84_eps(eps)
            self.assertClose(21.0 * each, eps, msg=f"21 shares of {eps} give {each}")
        self.assertClose(_core.bb84_eps(EPS_SEC), EPS, msg="the share this file already pins")

    def test_share_spends(self):
        """
        Six bounds at bb84_eps(eps_sec) plus the correctness term rebuild Eq. (1)'s length.
        """
        s0, s1, phi = 900.0, 4.0e4, 0.03
        n_key, e_key = 1.0e5, 0.02
        got = _core.bb84_length(s0, s1, phi, n_key, e_key, F_REC, EPS_SEC, EPS_COR)
        each = _core.bb84_eps(EPS_SEC)
        cost = 6.0 * math.log2(1.0 / each) + math.log2(2.0 / EPS_COR)
        want = math.floor(s0 + s1 * (1.0 - h2(phi)) - F_REC * n_key * h2(e_key) - cost)
        self.assertClose(got, want, msg=f"rebuilt {want} against the engine's {got}")

    def test_count_guard(self):
        """
        Certified vacuum and single-photon counts above the sifted count are refused.
        """
        self.assertFails(
            ValueError,
            "n_key",
            _core.bb84_length,
            600.0,
            600.0,
            0.02,
            1000.0,
            0.02,
            F_REC,
            EPS_SEC,
            EPS_COR,
            msg="certified counts above the sifted count are refused",
        )


class FiniteBb84(Question):
    """
    The q.Link surface: q.SplittingAttack(block=q.KeyBlock(...)).
    """

    def test_below_asymptotic(self):
        """
        The finite rate never exceeds the asymptotic rate of the same link, at any distance,
        block size or basis bias.
        """
        for dist in (25.0, 50.0, 100.0):
            for sift in (0.5, 0.7, 0.9):
                asym = link(dist, sift=sift).run().key_rate
                for total in (1e10, 1e12, 1e14):
                    got = link(dist, q.KeyBlock(n=total), sift=sift).run().key_rate
                    self.assertLessEqual(
                        got,
                        asym,
                        msg=f"{dist} km, sift {sift}, N {total}: {got} > {asym}",
                    )

    def test_rate_converges(self):
        """
        The reported rate rises with the block and settles by 1e14.
        """
        rates = [link(50.0, q.KeyBlock(n=n)).run().key_rate for n in (1e10, 1e12, 1e14)]
        self.assertMonotone(rates, rising=True, msg=f"rate must rise: {rates}")
        self.assertClose(rates[-1], rates[-2], atol=1e-5, msg="the rate must settle by 1e14")

    def test_bias_optimum(self):
        """
        Sifting at 0.8 beats 0.5, and 0.98 starves the test basis to zero key.
        """
        block = q.KeyBlock(n=1e12)
        rates = {s: link(50.0, block, sift=s).run().key_rate for s in (0.5, 0.8, 0.98)}
        self.assertGreater(rates[0.8], rates[0.5], msg=f"biasing must pay at 0.8: {rates}")
        self.assertClose(rates[0.98], 0.0, msg=f"rate at sift 0.98: {rates[0.98]}")

    def test_paths_agree(self):
        """
        The sampled pulse train and the closed-form expected counts reach the same length to 5%,
        labelled measured and expected.
        """
        total = int(2e7)
        near = dict(dist=1.0, ints=(0.5, 0.1, 0.0))
        shut = q.Bob(
            detector=q.ClickDetector(eta=0.5, dark=P_DARK),
            receiver=q.BasisAnalyser(misalign=0.01),
        )
        block = q.KeyBlock(n=total)
        closed = link(near["dist"], block)
        closed.bob = shut
        opened = link(near["dist"], block, alice=q.Alice())
        opened.bob = shut
        want = closed.run()
        got = opened.run(symbols=total, seed=7)
        self.assertGreater(want.key_length, 0.0, msg="the closed form must certify")
        self.assertClose(
            got.key_length / want.key_length,
            1.0,
            atol=0.05,
            msg=f"paths disagree: {got.key_length} vs {want.key_length}",
        )
        self.assertEqual(want.explain["counts"], "expected", msg="closed form label")
        self.assertEqual(got.explain["counts"], "measured", msg="sampled label")

    def test_named_carrier(self):
        """
        q.PolarisationKeying reaches the same finite-key path, frame drift widening the phase
        bound and costing key.
        """
        block = q.KeyBlock(n=1e12)
        rows = []
        for drift in (0.0, 0.05):
            res = q.Link(
                modulation=q.PolarisationKeying(
                    decoy=q.Decoy(intensities=(MU, NU1, NU2), probs=PROBS),
                    frame=q.ReferenceFrame(drift=drift),
                ),
                channel=q.Fiber(length=50.0, alpha=ALPHA),
                bob=q.Bob(
                    detector=q.ClickDetector(eta=ETA_BOB, dark=P_DARK),
                    receiver=q.BasisAnalyser(misalign=E_DET),
                ),
                security=q.SplittingAttack(f=F_REC, block=block),
            ).run()
            rows.append(res)
        self.assertGreater(rows[0].key_length, 0.0, msg="a held frame must certify")
        self.assertLess(
            rows[1].key_length,
            rows[0].key_length,
            msg=f"drift must cost key: {rows[1].key_length} vs {rows[0].key_length}",
        )
        self.assertGreater(rows[1].phi, rows[0].phi, msg="drift must widen the phase bound")

    def test_result_rows(self):
        """
        A finite run reports the rate as length over block, s0, s1, phi and n_key, and the
        asymptotic rate in explain.
        """
        res = link(50.0, q.KeyBlock(n=1e12)).run()
        self.assertClose(
            res.key_rate,
            res.key_length / 1e12,
            atol=1e-18,
            msg="the rate is the length over the block",
        )

        for name in ("s0", "s1", "phi", "n_key"):
            self.assertIsNotNone(getattr(res, name), msg=f"{name} must be reported")
        self.assertGreater(
            res.explain["key_asymptotic"]["value"],
            res.key_rate,
            msg="explain must carry the asymptotic rate beside the finite one",
        )

    def test_asymptotic_intact(self):
        """
        Leaving the block off returns the asymptotic GLLP rate byte for byte and no length.
        """
        res = link(50.0).run()
        self.assertClose(res.key_rate, 0.00019458124262760696, msg="the GLLP rate must not move")
        self.assertIsNone(res.key_length, msg="an asymptotic run reports no length")


class FiniteGuards(Guarded):
    """
    What the finite-key path refuses, and the family that never accepts a block.
    """

    def test_sifting_domain(self):
        """
        A sifting factor below 1/2 (no key/test split) or of 1 (no test basis) is refused.
        """
        self.assertFails(
            ValueError,
            "below 1/2",
            link(50.0, q.KeyBlock(n=1e12), sift=0.4).run,
            msg="sift below 1/2 is not invertible",
        )
        self.assertFails(
            ValueError,
            "no test basis",
            link(50.0, q.KeyBlock(n=1e12), sift=1.0).run,
            msg="sift = 1 samples no phase error",
        )

    def test_block_agrees(self):
        """
        A declared block differing from the sampled run size is refused.
        """
        sampled = link(50.0, q.KeyBlock(n=1e9), alice=q.Alice())
        self.assertFails(
            ValueError,
            "two block sizes",
            lambda: sampled.run(symbols=1000, seed=1),
            msg="the block must match the run",
        )

    def test_cow_refuses(self):
        """
        COW refuses a key block under q.SplittingAttack and q.PhaseBound, its phase error being
        an input, and q.PhaseBound's block defaults to None.
        """
        cow = q.Link(
            modulation=q.IntensityKeying(mu=0.5),
            channel=q.Fiber(length=50.0, alpha=ALPHA),
            bob=q.Bob(
                detector=q.ClickDetector(eta=ETA_BOB, dark=P_DARK),
                receiver=q.CoherenceMonitor(),
            ),
            security=q.SplittingAttack(f=1.22, block=q.KeyBlock()),
        )
        self.assertFails(
            NotImplementedError,
            "PhaseBound",
            cow.run,
            msg="COW never takes a key block",
        )
        self.assertIsNone(q.PhaseBound(e_phase=0.2).block, msg="no block unless one is declared")
        self.assertFails(
            NotImplementedError,
            "malformed rather than unimplemented",
            q.Link(
                modulation=q.IntensityKeying(mu=0.5),
                channel=q.Fiber(length=50.0, alpha=ALPHA),
                bob=q.Bob(
                    detector=q.ClickDetector(eta=ETA_BOB, dark=P_DARK),
                    receiver=q.CoherenceMonitor(),
                ),
                security=q.PhaseBound(e_phase=0.2, block=q.KeyBlock()),
            ).run,
            msg="a declared block reaches the engine's own refusal",
        )

    def test_share_domain(self):
        """
        bb84_eps refuses eps_sec outside (0, 1).
        """
        for bad in (0.0, 1.0, -1e-12, float("nan"), float("inf")):
            self.assertBad("eps_sec", _core.bb84_eps, (bad,), msg=f"eps_sec = {bad}")

    def test_block_domain(self):
        """
        q.KeyBlock refuses a non-positive size, an epsilon outside the unit interval and a frame
        error rate outside it.
        """
        cases = (
            ("n", dict(n=0.0)),
            ("eps_sec", dict(eps_sec=0.0)),
            ("eps_sec", dict(eps_sec=1.0)),
            ("eps_cor", dict(eps_cor=1.0)),
            ("fer", dict(fer=1.5)),
        )
        for needle, kwargs in cases:
            self.assertFails(
                ValueError,
                needle,
                lambda k=kwargs: q.KeyBlock(**k),
                msg=f"KeyBlock must refuse {kwargs}",
            )
        self.assertFails(
            ValueError,
            "KeyBlock",
            lambda: q.SplittingAttack(block=1e12),
            msg="block must be a component, not a number",
        )


class LimAnchor(Question):
    """
    Tier A, bound: against a published finite-key rate computed with a tighter concentration
    inequality on the same protocol.
    """

    def test_chernoff_bound(self):
        """
        At Zhang, Zhao, Razavi & Ma's Table 3 parameter set the Hoeffding-based Lim bound must
        not exceed their Chernoff-based rate, at any basis bias.
        """
        block = q.KeyBlock(n=1e10, eps_sec=EPS_SEC, eps_cor=EPS_COR)
        best = 0.0
        for step in range(0, 20):
            sift = 0.5 + 0.024 * step
            got = link(100.0, block, sift=sift, ints=ZR_INTS, probs=ZR_PROBS).run().key_rate
            best = max(best, got)
        self.assertLessEqual(best, ZR_RATE, msg=f"Lim {best} above the Chernoff rate {ZR_RATE}")

    def test_chernoff_gap(self):
        """
        Lim's deviation on the total detection count swamps the vacuum line at 1e10 (zero key),
        while 1e12 clears the published rate.
        """
        args = dict(sift=0.5, ints=ZR_INTS, probs=ZR_PROBS)
        short = link(100.0, q.KeyBlock(n=1e10), **args).run()
        long = link(100.0, q.KeyBlock(n=1e12), **args).run()
        self.assertClose(short.key_rate, 0.0, msg="1e10 certifies nothing here")
        self.assertGreater(long.key_rate, ZR_RATE, msg=f"1e12 must clear {ZR_RATE}: {long.key_rate}")
        self.assertClose(short.explain["s0"]["value"], 0.0, msg="the vacuum is swamped")


if __name__ == "__main__":
    rc = Exam(
        "FiniteEstimator",
        "Counted-event decoy bounds under Lim, Curty, Walenta, Xu & Zbinden (2014)",
        "finitekey_estimator.md",
    ).run(load(LimEstimator))
    rc |= Exam(
        "FiniteLength",
        "The finite key length in bits, Lim et al. Eq. (1)",
        "finitekey_length.md",
    ).run(load(LimLength))
    rc |= Exam(
        "FiniteBb84",
        "Finite-key BB84-WCP through q.Link with a q.KeyBlock",
        "finitekey_bb84.md",
    ).run(load(FiniteBb84))
    rc |= Exam(
        "FiniteGuards",
        "What the finite-key path refuses, and the families that never reach it",
        "finitekey_guards.md",
    ).run(load(FiniteGuards))
    rc |= Exam(
        "FiniteAnchor",
        "Tier A bound: below Zhang, Zhao, Razavi & Ma, PRA 95, 012333 (2017), Table 3",
        "finitekey_anchor.md",
    ).run(load(LimAnchor))
    sys.exit(rc)
