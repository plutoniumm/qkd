import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, load

from kit.checks import Guarded
from kit.forms import h2
from qkd import _core

# Round-robin differential phase shift, Sasaki, Yamamoto & Koashi, Nature 509, 475 (2014).
# Privacy amplification reads the packet length and the photon number, nothing an adversary
# can touch.
#
# The Nature 2014 paper was never posted to arXiv. Its bound is reproduced, with the
# equation numbers cited below, by Takesue, Sasaki, Tamaki & Koashi, Nature Photonics 9, 827
# (2015), arXiv:1505.07914, Eqs. (2)-(4), which is the source used here.

# Yin, Wang, Chen, Han, Wang, Guo & Han, Nature Communications 9, 457 (2018),
# arXiv:1702.01260, Table 1: the largest bit error rate at which each analysis still yields
# key, single photon per packet. "original RRDPS" is Sasaki, Yamamoto & Koashi's
# h2(1/(L-1)); "Eq. (1) without E" is Yin's collective bound with the error-rate constraint
# dropped. Their L = 3 entry in the first column is a dash, pinned below as exactly zero.
YIN_ORIGINAL = {
    5: 0.0289,
    16: 0.165,
    32: 0.24,
    64: 0.301,
}
YIN_COLLECTIVE = {
    3: 0.0546,
    5: 0.122,
    16: 0.244,
    32: 0.300,
    64: 0.346,
}

# Yin et al., Methods, "Calculations for an existing experiment": the L = 65 run of Wang,
# Yin, Chen, He, Song, Li, Zhang, Zhou, Guo & Han, Nature Photonics 9, 832 (2015), read back
# through both bounds. Every number on these three lines is theirs, F_RRDPS included; I_TEN
# is their ten-photon collective bound to the three figures they print.
L_WANG, MU_WANG, NU_WANG = 65, 0.037, 10
Q_WANG, E_WANG, I_TEN = 8.435e-4, 0.058, 0.513
R_ORIGINAL, R_COLLECTIVE, F_RRDPS = 5e-8, 1.44e-6, 1.1

# Yin et al., Results: "we assume that dark counting rate d = 10^-6 per pulse and the optical
# misalignment parameter e_mis = 0.015 or 0.15, which are typical and realistic".
D_YIN = 1e-6
E_CLEAN, E_NOISY = 0.015, 0.15

# ETA and MU are THIS EXAM'S choices, not Yin et al.'s.
ETA, MU = 0.1, 0.005

# Published one-way ceilings, none of which any block size moves: Shor & Preskill's BB84 root
# of 1 = 2 h2(e); the six-state bound quoted by Renner, Gisin & Kraus, Phys. Rev. A 72,
# 012332 (2005) Sec. V.1; and the domain of `dps_rate`, Waks, Takesue & Yamamoto,
# Phys. Rev. A 73, 012344 (2006).
E_BB84, E_SIX, E_DPS = 0.110, 0.126, 6.0 / 38.0


def tail(l, mu, nu):
    """
    Poisson probability of more than ``nu`` photons at mean ``l * mu``.
    """
    mean = l * mu
    kept = sum(math.exp(-mean) * mean**i / math.factorial(i) for i in range(nu + 1))

    return max(0.0, 1.0 - kept)


def charged(l, mu, nu, q):
    """
    Zhang, Yuan, Cao & Ma Eq. (13) summed over every photon number with a representable
    weight, without the engine's early stop at half the delays.
    """
    mean = l * mu
    weights = [math.exp(-mean)]
    while len(weights) < 2 * l or weights[-1] > 1e-300:
        weights.append(weights[-1] * mean / len(weights))

    src = max(0.0, 1.0 - sum(weights[: nu + 1]))
    if src >= q:
        return 1.0

    acc = sum(w * h2(min(n / (l - 1), 0.5)) for n, w in enumerate(weights) if n > nu)
    acc += max(0.0, 1.0 - sum(weights))

    return min(1.0, (acc + (q - src) * h2(min(nu / (l - 1), 0.5))) / q)


def point(l, nu, e_mis=E_CLEAN, mu=MU):
    """
    ``(q, e_bit, cost)`` at the exam's hardware, the cost being GLLP tagging per sifted bit.
    """
    q, e_bit = _core.rrdps_counts(l, mu, ETA, D_YIN, e_mis)
    cost = _core.rrdps_tag(q, _core.rrdps_src(l, mu, nu), _core.rrdps_leak(l, nu))

    return q, e_bit, cost


def train(l, nu, e_mis=E_CLEAN, mu=MU):
    """
    Key rate in bits per pulse for one ``(l, nu_th)`` pair at the exam's hardware.
    """
    q, e_bit, cost = point(l, nu, e_mis, mu)

    return _core.rrdps_rate(q, e_bit, cost, F_RRDPS, l)


def best(l, e_mis=E_CLEAN, mu=MU):
    """
    The best rate any threshold reaches at one train length.
    """

    return max(train(l, nu, e_mis, mu) for nu in range(1, max(2, (l - 1) // 2 + 1)))


class RrdpsLeakage(Guarded):
    """
    `rrdps_leak(l, nu_th)` returns bits: no error rate, no visibility, no test basis.
    """

    def test_leak_has_no_channel_argument(self):
        """
        Moving the bit error rate changes `rrdps_rate` only through q*f_ec*h2(e)/l, never
        through privacy amplification.
        """
        q, _, cost = point(128, 4)
        rates = [_core.rrdps_rate(q, e, cost, F_RRDPS, 128) for e in (0.0, 0.05, 0.1)]
        gaps = [
            q * F_RRDPS * (h2(0.05) - h2(0.0)) / 128.0,
            q * F_RRDPS * (h2(0.1) - h2(0.05)) / 128.0,
        ]

        self.assertClose(rates[0] - rates[1], gaps[0], atol=1e-18, msg=f"gap {rates[0] - rates[1]}")
        self.assertClose(rates[1] - rates[2], gaps[1], atol=1e-18, msg=f"gap {rates[1] - rates[2]}")

    def test_leak_falls_with_train_length(self):
        """
        `rrdps_leak(l, 1)` falls monotonically in l, is h2(1/1023) at l = 1024, and scales as
        log2(l)/l with no floor.
        """
        got = [_core.rrdps_leak(l, 1) for l in (4, 8, 16, 32, 64, 128, 256, 512, 1024)]

        self.assertMonotone(got, rising=False, msg="leakage not falling in l")
        self.assertClose(got[-1], h2(1.0 / 1023.0), atol=1e-15, msg="l = 1024 is h2(1/1023)")
        self.assertLess(got[-1] / (math.log2(1024) / 1024), 2.0, msg="log2(l)/l scaling")

    def test_leak_saturates_at_a_whole_bit(self):
        """
        `rrdps_leak` is exactly one bit once the threshold reaches half the delays, and stays
        there past it.
        """
        for l in (17, 33, 65):
            half = (l - 1) // 2
            self.assertClose(
                _core.rrdps_leak(l, half),
                1.0,
                atol=0.0,
                msg=f"l = {l} saturates at half",
            )
            self.assertClose(
                _core.rrdps_leak(l, l - 1),
                1.0,
                atol=0.0,
                msg=f"l = {l} must not fall back past half",
            )

    def test_source_tail_is_the_poisson_one(self):
        """
        `rrdps_src` is the Poisson tail of the packet's own mean l*mu, not of mu.
        """
        for l, nu in ((16, 2), (65, 10), (256, 5)):
            self.assertClose(
                _core.rrdps_src(l, MU, nu),
                tail(l, MU, nu),
                atol=1e-15,
                msg=f"e_src at l = {l}, nu_th = {nu}",
            )

    def test_phase_mixes_the_error_rates(self):
        """
        `rrdps_phase` is the tagged share plus nu/(l - 1) on the rest, capped at one half.
        """
        for l, nu in ((32, 3), (128, 6), (65, 10)):
            q, _ = _core.rrdps_counts(l, MU, ETA, D_YIN, E_CLEAN)
            share = min(1.0, _core.rrdps_src(l, MU, nu) / q)
            want = min(0.5, share + (1.0 - share) * nu / (l - 1))
            self.assertClose(
                _core.rrdps_phase(q, l, MU, nu),
                want,
                atol=1e-15,
                msg=f"e_ph at l = {l}, nu_th = {nu}",
            )

    def test_three_published_forms_are_ordered(self):
        """
        `rrdps_gllp` <= `rrdps_tag` <= h2(`rrdps_phase`) at every train length and threshold.
        """
        worst = 0.0
        for l in (16, 32, 64, 128, 256):
            for nu in range(1, l):
                q, _ = _core.rrdps_counts(l, MU, ETA, D_YIN, E_CLEAN)
                fine = _core.rrdps_gllp(q, l, MU, nu)
                mixed = _core.rrdps_tag(q, _core.rrdps_src(l, MU, nu), _core.rrdps_leak(l, nu))
                gross = h2(_core.rrdps_phase(q, l, MU, nu))
                worst = max(worst, fine - mixed, mixed - gross)

        self.assertLess(worst, 1e-12, msg=f"worst inversion {worst}")

    def test_gllp_charges_each_photon_number(self):
        """
        `rrdps_gllp` matches Eq. (13) summed out to four thousand photon numbers, including
        the thresholds past half the delays where it collapses onto the whole-key form.
        """
        worst = 0.0
        for l in (16, 64, 256):
            for nu in range(1, l):
                for mu in (0.001, 0.01, 0.05):
                    q, _ = _core.rrdps_counts(l, mu, ETA, D_YIN, E_CLEAN)
                    got = _core.rrdps_gllp(q, l, mu, nu)
                    worst = max(worst, abs(got - charged(l, mu, nu, q)))

        self.assertLess(worst, 1e-12, msg=f"worst deviation from Eq. (13) {worst}")

    def test_tagging_reduces_to_the_leak(self):
        """
        `rrdps_tag` passes the leakage bound through unchanged at a zero over-threshold
        probability, and charges a whole bit on a wholly tagged key.
        """
        for leak in (0.0, 0.25, 0.5, 1.0):
            self.assertClose(
                _core.rrdps_tag(0.01, 0.0, leak),
                leak,
                atol=0.0,
                msg=f"leak {leak} not passed through",
            )

        self.assertClose(_core.rrdps_tag(0.01, 0.01, 0.0), 1.0, atol=0.0, msg="tagged key leaks under a bit")

    def test_collective_bound_beats_the_original(self):
        """
        `rrdps_collective(l, 1)` is exactly 1 at l = 2 and below both `rrdps_leak(l, 1)` and
        a whole bit at every longer train, Yin et al.'s corollary.
        """
        self.assertClose(_core.rrdps_collective(2, 1), 1.0, atol=1e-12, msg="l = 2 under a whole bit")

        for l in (3, 5, 16, 32, 64, 128, 512):
            got = _core.rrdps_collective(l, 1)
            self.assertLess(got, _core.rrdps_leak(l, 1), msg=f"l = {l}: {got} not tighter")
            self.assertLess(got, 1.0, msg=f"l = {l}: {got} at or above a bit")


class RrdpsTolerance(Guarded):
    """
    The tolerable bit error rate, which climbs with the packet length rather than sitting at
    a ceiling.
    """

    def test_original_table_reproduces(self):
        """
        `rrdps_tolerance` off the original bound matches Yin et al. Table 1's first column,
        their dashed L = 3 entry reading exactly zero.
        """
        self.assertClose(
            _core.rrdps_tolerance(_core.rrdps_leak(3, 1), 1.0),
            0.0,
            atol=0.0,
            msg="l = 3 tolerance is not 0",
        )

        for l, want in YIN_ORIGINAL.items():
            self.assertClose(
                _core.rrdps_tolerance(_core.rrdps_leak(l, 1), 1.0),
                want,
                atol=5e-4,
                msg=f"Table 1 original column, l = {l}",
            )

    def test_collective_table_reproduces(self):
        """
        `rrdps_tolerance` off the collective bound matches Yin et al. Table 1's third column.
        """
        for l, want in YIN_COLLECTIVE.items():
            self.assertClose(
                _core.rrdps_tolerance(_core.rrdps_collective(l, 1), 1.0),
                want,
                atol=5e-4,
                msg=f"Table 1 collective column, l = {l}",
            )

    def test_tolerance_passes_every_qubit_ceiling(self):
        """
        The tolerance rises with the train length past BB84's 11.0%, six-state's 12.6% and
        differential phase shift's 6/38, and exceeds 0.45 at l = 4096.
        """
        got = [_core.rrdps_tolerance(_core.rrdps_leak(l, 1), 1.0) for l in (8, 16, 32, 64)]

        self.assertMonotone(got, rising=True, msg="tolerance not climbing with l")

        for l, bar in ((16, E_BB84), (16, E_SIX), (32, E_DPS)):
            self.assertGreater(
                _core.rrdps_tolerance(_core.rrdps_leak(l, 1), 1.0),
                bar,
                msg=f"l = {l} under {bar}",
            )

        self.assertGreater(
            _core.rrdps_tolerance(_core.rrdps_leak(4096, 1), 1.0),
            0.45,
            msg="l = 4096 tolerance under 0.45",
        )

    def test_tolerance_is_where_the_rate_closes(self):
        """
        `rrdps_rate` is zero at the rate `rrdps_tolerance` returns, positive one per cent below.
        """
        for l in (32, 128):
            cost = _core.rrdps_leak(l, 1)
            edge = _core.rrdps_tolerance(cost, F_RRDPS)
            self.assertClose(
                _core.rrdps_rate(0.5, edge, cost, F_RRDPS, l),
                0.0,
                atol=1e-15,
                msg=f"l = {l} rate non-zero at its tolerance",
            )
            self.assertGreater(
                _core.rrdps_rate(0.5, edge * 0.99, cost, F_RRDPS, l),
                0.0,
                msg=f"l = {l} rate zero below its tolerance",
            )

    def test_reconciliation_costs_tolerance(self):
        """
        f_ec above 1 lowers the tolerance; a whole-bit leakage bound leaves 0, a zero one 0.5.
        """
        cost = _core.rrdps_leak(64, 1)
        shannon = _core.rrdps_tolerance(cost, 1.0)

        self.assertLess(_core.rrdps_tolerance(cost, F_RRDPS), shannon, msg="f_ec > 1 costs no tolerance")
        self.assertClose(_core.rrdps_tolerance(1.0, 1.0), 0.0, atol=0.0, msg="a whole bit leaves tolerance")
        self.assertClose(_core.rrdps_tolerance(0.0, 1.0), 0.5, atol=0.0, msg="no leakage leaves under 0.5")


class RrdpsPublished(Guarded):
    """
    Yin et al.'s worked reading of the L = 65 experiment of Wang, Yin, Chen, He, Song, Li,
    Zhang, Zhou, Guo & Han, Nature Photonics 9, 832 (2015).
    """

    def test_wang_run_under_the_original_bound(self):
        """
        Their gain and error rate at a ten-photon threshold give the 5e-8 they report.
        """
        src = _core.rrdps_src(L_WANG, MU_WANG, NU_WANG)
        cost = _core.rrdps_tag(Q_WANG, src, _core.rrdps_leak(L_WANG, NU_WANG))
        got = _core.rrdps_rate(Q_WANG, E_WANG, cost, F_RRDPS, L_WANG)

        self.assertClose(got, R_ORIGINAL, atol=5e-10, msg=f"R_1 {got:.3g} against 5e-8")

    def test_wang_run_under_the_collective_bound(self):
        """
        The same run under their ten-photon collective bound gives the 1.44e-6 they report,
        over twenty-five times the original.
        """
        src = _core.rrdps_src(L_WANG, MU_WANG, NU_WANG)
        got = _core.rrdps_rate(Q_WANG, E_WANG, _core.rrdps_tag(Q_WANG, src, I_TEN), F_RRDPS, L_WANG)

        self.assertClose(got, R_COLLECTIVE, atol=1e-8, msg=f"R_2 {got:.3g} against 1.44e-6")
        self.assertGreater(got / R_ORIGINAL, 25.0, msg=f"ratio {got / R_ORIGINAL:.1f} under 25")

    def test_wang_run_with_the_bound_derived(self):
        """
        Computing the ten-photon collective bound rather than quoting it reaches the same
        1.44e-6 and their I_AE = 0.513.
        """
        src = _core.rrdps_src(L_WANG, MU_WANG, NU_WANG)
        iae = _core.rrdps_collective(L_WANG, NU_WANG)
        got = _core.rrdps_rate(Q_WANG, E_WANG, _core.rrdps_tag(Q_WANG, src, iae), F_RRDPS, L_WANG)

        self.assertClose(got, R_COLLECTIVE, atol=1e-8, msg=f"R_2 {got:.3g} against 1.44e-6")
        self.assertClose(iae, I_TEN, atol=5e-4, msg=f"I_AE {iae:.4f} against 0.513")

    def test_receiver_error_rate_is_derived(self):
        """
        `rrdps_counts`'s bit error rate is (eta*mu*e_mis + d)/(eta*mu + 2d) at every l, the
        delay weights cancelling exactly.
        """
        want = (ETA * MU * E_CLEAN + D_YIN) / (ETA * MU + 2.0 * D_YIN)

        for l in (5, 32, 257, 1024):
            _, e_bit = _core.rrdps_counts(l, MU, ETA, D_YIN, E_CLEAN)
            self.assertClose(e_bit, want, atol=1e-15, msg=f"e_bit {e_bit} at l = {l}")

    def test_receiver_gain_grows_then_saturates(self):
        """
        The sifted rate per packet rises with the train length while the packet is dim and
        falls by l = 1024, Bob keeping only a packet where exactly one window fired.
        """
        got = [_core.rrdps_counts(l, 0.05, ETA, D_YIN, E_CLEAN)[0] for l in (8, 64, 256, 1024)]

        self.assertMonotone(got[:3], rising=True, msg=f"gain not climbing: {got[:3]}")
        self.assertLess(got[3], got[2], msg=f"gain at l = 1024 is {got[3]}")


class RrdpsTrainLength(Guarded):
    """
    The trade-off in L, whose optimum is interior.
    """

    def test_optimum_is_interior(self):
        """
        `rrdps_optimum` returns a train length strictly inside the search, a positive rate and
        a non-zero threshold.
        """
        l, nu, rate = _core.rrdps_optimum(2048, MU, ETA, D_YIN, E_CLEAN, F_RRDPS)

        self.assertGreater(l, 2, msg=f"optimum l = {l}")
        self.assertLess(l, 2048, msg=f"optimum l = {l} at the search bound")
        self.assertGreater(rate, 0.0, msg=f"optimum rate {rate}")
        self.assertGreater(nu, 0, msg=f"optimum nu_th = {nu}")

    def test_optimum_matches_the_composed_rate(self):
        """
        The rate `rrdps_optimum` reports is bit-for-bit the one the exposed functions compose
        at the train length and threshold it names.
        """
        for mu, e_mis in ((MU, E_CLEAN), (0.0064, E_NOISY), (0.02, 0.05)):
            l, nu, rate = _core.rrdps_optimum(1024, mu, ETA, D_YIN, e_mis, F_RRDPS)
            self.assertClose(rate, train(l, nu, e_mis, mu), atol=0.0, msg=f"sweep at mu = {mu}")

    def test_optimum_does_not_move_with_the_bound(self):
        """
        Widening the search from 1024 to 4096 returns the same train length, threshold, rate.
        """
        near = _core.rrdps_optimum(1024, MU, ETA, D_YIN, E_CLEAN, F_RRDPS)
        far = _core.rrdps_optimum(4096, MU, ETA, D_YIN, E_CLEAN, F_RRDPS)

        self.assertEqual(near, far, msg=f"{near} against {far}")

    def test_rate_rises_then_falls_in_length(self):
        """
        The best rate at each train length rises up to the optimum and falls past it.
        """
        peak = _core.rrdps_optimum(2048, MU, ETA, D_YIN, E_CLEAN, F_RRDPS)[0]
        rising = [best(l) for l in (16, 64, peak // 2, peak)]
        falling = [best(l) for l in (peak, peak * 2, peak * 4)]

        self.assertMonotone(rising, rising=True, msg=f"below the peak: {rising}")
        self.assertMonotone(falling, rising=False, msg=f"above the peak: {falling}")

    def test_threshold_rises_with_length(self):
        """
        The photon-number threshold `rrdps_optimum` picks rises with the search bound.
        """
        got = [_core.rrdps_optimum(l, MU, ETA, D_YIN, E_CLEAN, F_RRDPS)[1] for l in (64, 256, 1024)]

        self.assertMonotone(got, rising=True, msg=f"thresholds {got}")

    def test_no_key_returns_a_zero_row(self):
        """
        Hardware no train length can distil from returns (0, 0, 0.0), not the shortest train.
        """
        got = _core.rrdps_optimum(64, MU, ETA, D_YIN, 0.45, F_RRDPS)
        self.assertEqual(got, (0, 0, 0.0), msg=f"empty search returned {got}")


class RrdpsAgainstMonitoring(Guarded):
    """
    RRDPS distils at a bit error rate where both monitoring-based click protocols have none.
    """

    def test_key_where_cow_and_dps_have_none(self):
        """
        At Yin et al.'s 0.15 misalignment `cow_rate`, `dps_rate` and `bb84_rate` are exactly
        zero where `rrdps_optimum` is positive, all four in bits per pulse at one click rate.
        """
        l, _, rate = _core.rrdps_optimum(2048, MU, ETA, D_YIN, E_NOISY, F_RRDPS)
        q, e_bit = _core.rrdps_counts(l, MU, ETA, D_YIN, E_NOISY)
        click = q / l

        self.assertGreater(rate, 0.0, msg=f"rrdps rate {rate}")
        self.assertClose(_core.cow_rate(click, e_bit, e_bit, F_RRDPS), 0.0, atol=0.0, msg="cow_rate non-zero")
        self.assertClose(_core.dps_rate(click, e_bit, MU, F_RRDPS), 0.0, atol=0.0, msg="dps_rate non-zero")
        self.assertClose(
            _core.bb84_rate(0.5, click, e_bit, click, e_bit, F_RRDPS),
            0.0,
            atol=0.0,
            msg="bb84_rate non-zero",
        )

    def test_monitoring_protocols_work_when_clean(self):
        """
        `cow_rate` and `dps_rate` are positive at the 0.015 misalignment, so the comparison
        above is about the error rate.
        """
        l = 128
        q, e_bit = _core.rrdps_counts(l, 0.05, ETA, D_YIN, E_CLEAN)
        click = q / l

        self.assertGreater(_core.cow_rate(click, e_bit, e_bit, F_RRDPS), 0.0, msg="cow_rate zero when clean")
        self.assertGreater(_core.dps_rate(click, e_bit, 0.05, F_RRDPS), 0.0, msg="dps_rate zero when clean")

    def test_cow_would_need_a_bound_it_cannot_get(self):
        """
        The phase-error bound COW would need to distil there is under half its own bit error
        rate, and handed it `cow_rate` does distil.
        """
        l, _, _ = _core.rrdps_optimum(2048, MU, ETA, D_YIN, E_NOISY, F_RRDPS)
        q, e_bit = _core.rrdps_counts(l, MU, ETA, D_YIN, E_NOISY)
        needed = _core.rrdps_tolerance(F_RRDPS * h2(e_bit), 1.0)

        self.assertLess(needed, e_bit / 2.0, msg=f"needed {needed} above e_bit/2")
        self.assertGreater(
            _core.cow_rate(q / l, e_bit, needed * 0.99, F_RRDPS),
            0.0,
            msg="cow_rate zero at the bound it needs",
        )

    def test_dps_ceiling_is_a_ceiling(self):
        """
        `dps_rate` is exactly zero at 6/38 at every intensity, while `rrdps_tolerance` passes
        6/38 by a train of thirty-two pulses.
        """
        for mu in (0.001, 0.01, 0.1):
            self.assertClose(
                _core.dps_rate(0.5, E_DPS, mu, 1.0),
                0.0,
                atol=0.0,
                msg=f"dps_rate non-zero at 6/38, mu = {mu}",
            )

        self.assertGreater(
            _core.rrdps_tolerance(_core.rrdps_leak(32, 1), 1.0),
            E_DPS,
            msg="l = 32 tolerance under 6/38",
        )


class RrdpsGuards(Guarded):
    """
    What the engine refuses rather than approximating.
    """

    def test_short_train_is_refused(self):
        """
        Every entry point refuses a packet of fewer than two pulses, naming the pair.
        """
        for fn, args in (
            (_core.rrdps_leak, (1, 1)),
            (_core.rrdps_src, (0, 0.01, 1)),
            (_core.rrdps_collective, (1, 1)),
            (_core.rrdps_counts, (1, 0.01, 0.1, 1e-6, 0.02)),
        ):
            self.assertBad("PAIR", fn, args, msg=f"{args} names no pair")

    def test_long_train_is_refused(self):
        """
        A packet past the interferometer's switching range is refused rather than swept.
        """
        self.assertBad("interferometer", _core.rrdps_leak, (100000, 1), msg="l past the cap")
        self.assertBad(
            "interferometer",
            _core.rrdps_optimum,
            (9000, 0.01, 0.1, 1e-6, 0.02, 1.1),
            msg="l_max past the cap",
        )

    def test_multiphoton_collective_is_refused(self):
        """
        `rrdps_collective` refuses a train of fewer than nu + 1 delays, which Eq. (1) needs,
        and `rrdps_simplex` refuses a vacuum packet.
        """
        for l, n in ((32, 32), (32, 100), (2, 2), (65, 65)):
            self.assertBad(
                "simplex",
                _core.rrdps_collective,
                (l, n),
                msg=f"l = {l} accepted at n = {n}",
            )

        self.assertBad(
            "empty sum",
            _core.rrdps_simplex,
            (32, 0),
            msg="a vacuum packet",
        )

    def test_arguments_are_guarded(self):
        """
        Each `rrdps_rate` and `rrdps_counts` argument is refused on its own slot, by name.
        """
        self.assertSlots(
            _core.rrdps_rate,
            (0.01, 0.02, 0.3, 1.1, 64),
            (
                (0, "q", (-0.1, 1.5, float("nan"))),
                (1, "e_bit", (0.6, -0.01)),
                (2, "cost", (-0.1, 1.5)),
                (3, "f_ec", (0.9, 0.0)),
            ),
            msg="rrdps_rate",
        )
        self.assertSlots(
            _core.rrdps_counts,
            (64, 0.01, 0.1, 1e-6, 0.02),
            (
                (1, "mu", (0.0, -1.0)),
                (2, "eta", (0.0, 1.5)),
                (3, "dark", (1.0, -0.1)),
                (4, "e_mis", (0.6, -0.01)),
            ),
            msg="rrdps_counts",
        )

    def test_sifted_rate_stays_a_probability(self):
        """
        The linearised dark-count term peaks the sifted rate at 0.48 across the whole allowed
        domain, so the guard against leaving [0, 1] never fires.
        """
        top = 0.0
        for l in (2, 8, 64, 512, 4096):
            for mu in (1e-6, 1e-3, 0.03, 1.0, 10.0):
                for eta in (1e-3, 0.1, 1.0):
                    for dark in (0.0, 1e-4, 0.01, 0.6, 0.99):
                        top = max(top, _core.rrdps_counts(l, mu, eta, dark, 0.02)[0])

        self.assertLess(top, 0.5, msg=f"sifted rate peaks at {top}")
        self.assertGreater(top, 0.4, msg=f"sweep only reached {top}")


class RrdpsMultiphoton(Guarded):
    """
    Yin et al. Eq. (1) past one photon: a concave maximisation over the (nu+1)-simplex
    certified by a tangent plane, not by the search, so a stalled ascent is looser not wrong.
    """

    def test_multiphoton_meets_the_closed_form(self):
        """
        At one photon `rrdps_simplex` agrees with `rrdps_collective`'s bisection to 1e-11 and
        never reads the smaller.
        """
        for l in (2, 3, 5, 16, 32, 64, 65, 92, 128, 512, 1024, 4096):
            bound, witness, tangent, _, _ = _core.rrdps_simplex(l, 1)
            closed = _core.rrdps_collective(l, 1)

            self.assertClose(bound, closed, atol=1e-11, msg=f"l = {l}: {bound} against {closed}")
            self.assertGreaterEqual(bound, closed - 1e-14, msg=f"l = {l}: {bound} below the bisection")
            self.assertLessEqual(witness, bound + 1e-12, msg=f"l = {l}: witness {witness} over bound")
            self.assertLessEqual(bound, tangent + 1e-12, msg=f"l = {l}: bound {bound} over tangent")

    def test_ten_photon_bound_reproduces(self):
        """
        `rrdps_collective(65, 10)` is the 0.513 Yin et al.'s Methods print, under `rrdps_leak`
        at the same threshold.
        """
        got = _core.rrdps_collective(L_WANG, NU_WANG)

        self.assertClose(got, I_TEN, atol=5e-4, msg=f"I_AE {got:.4f} against 0.513")
        self.assertLess(got, _core.rrdps_leak(L_WANG, NU_WANG), msg=f"I_AE {got:.4f} not tighter")

    def test_collective_rises_and_stays_tighter(self):
        """
        `rrdps_collective` rises strictly with the photon number, is exactly 0 on vacuum, and
        stays below `rrdps_leak` at every count the two share.
        """
        for l in (16, 32, 65, 128):
            got = [_core.rrdps_collective(l, n) for n in range(0, 9)]

            self.assertMonotone(got, rising=True, msg=f"l = {l}: {got}")
            self.assertClose(got[0], 0.0, atol=0.0, msg=f"l = {l} vacuum leaks {got[0]}")

            for n in range(1, 9):
                self.assertLess(got[n], _core.rrdps_leak(l, n), msg=f"l = {l}, n = {n} not tighter")

    def test_multiphoton_tolerance_rises(self):
        """
        The tolerance off Eq. (1) exceeds the one off the original bound at the same l and nu.
        """
        for l, nu in ((32, 2), (64, 5), (65, 10), (128, 20)):
            tight = _core.rrdps_tolerance(_core.rrdps_collective(l, nu), 1.0)
            plain = _core.rrdps_tolerance(_core.rrdps_leak(l, nu), 1.0)
            self.assertGreater(tight, plain, msg=f"l = {l}, nu = {nu} tolerates more")

    def test_two_certificates_agree(self):
        """
        `rrdps_simplex`'s programme value sits under the one-plane tangent cap and within
        1e-9 of the witness, with both solvers having run.
        """
        for l, nu in ((16, 3), (32, 2), (65, 10), (128, 20), (256, 5)):
            bound, witness, tangent, steps, iters = _core.rrdps_simplex(l, nu)
            self.assertLessEqual(bound, tangent + 1e-12, msg=f"l = {l}, nu = {nu} under the cap")
            self.assertLess(tangent - witness, 1e-9, msg=f"l = {l}, nu = {nu} brackets to 1e-9")
            self.assertGreater(steps, 0, msg=f"l = {l}, nu = {nu} took an ascent")
            self.assertGreater(iters, 0, msg=f"l = {l}, nu = {nu} ran the programme")

    def test_tight_ask_is_refused(self):
        """
        `rrdps_simplex` refuses a bracket narrower than the ascent closes, naming its width.
        """
        self.assertBad(
            "reltol",
            _core.rrdps_simplex,
            (65, 10, 1e-15),
            msg="a bracket below the ascent's own floor",
        )


class RrdpsRefined(Guarded):
    """
    Matsuura's Eq. (60): the leakage charged as a conditional entropy rather than as h2 of a
    mixed error rate.
    """

    def test_refined_tightens(self):
        """
        `rrdps_polytope` certifies less leakage than the GLLP tagging form and tolerates a
        higher bit error rate, distilling at l = 32, p_src = 0.2 where the closed form does not.
        """
        for l, p_src in ((8, 0.1), (16, 0.1), (32, 0.2), (64, 0.1)):
            refined, _w, _t, plain, _s, _i = _core.rrdps_polytope(l, p_src, 0.5)
            self.assertLess(refined, plain, msg=f"l = {l}, p_src = {p_src}: {refined} against {plain}")
            self.assertGreater(
                _core.rrdps_tolerance(refined, 1.1),
                _core.rrdps_tolerance(plain, 1.1),
                msg=f"tolerable error at l = {l}",
            )

        dead = _core.rrdps_polytope(32, 0.2, 0.5)
        self.assertEqual(_core.rrdps_rate(0.5, 0.02, dead[3], 1.1, 32), 0.0, msg="closed form has key at l = 32")
        self.assertGreater(
            _core.rrdps_rate(0.5, 0.02, dead[0], 1.1, 32),
            0.0,
            msg="refined bound has no key at l = 32",
        )

    def test_bracket_closes(self):
        """
        `bound`, the only certificate of the three, sits between `witness` -- H(X|Y) at the
        distribution the ascent stopped on -- and the tangent cap, closing to under 1e-11.
        """
        for l, p_src in ((4, 0.05), (8, 0.1), (16, 0.2), (64, 0.05), (250, 0.02)):
            bound, witness, tangent, _p, steps, iters = _core.rrdps_polytope(l, p_src, 0.5)
            self.assertLessEqual(witness, bound + 1e-12, msg=f"the witness is below the bound at l = {l}")
            self.assertLessEqual(bound, tangent + 1e-12, msg=f"the programme is under the tangent at l = {l}")
            self.assertLess(bound - witness, 1e-11, msg=f"bracket at l = {l} is {bound - witness:.3e}")
            self.assertGreater(steps, 0, msg=f"the ascent ran at l = {l}")
            self.assertGreater(iters, 0, msg=f"the programme ran at l = {l}")

    def test_entropy_agrees(self):
        """
        `rrdps_entropy` is `rrdps_polytope`'s certified value bit for bit.
        """
        for l, p_src, q in ((8, 0.1, 0.5), (16, 0.25, 0.8), (32, 0.05, 0.2)):
            self.assertEqual(
                _core.rrdps_entropy(l, p_src, q),
                _core.rrdps_polytope(l, p_src, q, 1e-9)[0],
                msg=f"l = {l}, p_src = {p_src}, q = {q}",
            )

    def test_silent_source(self):
        """
        At p_src = 0 every row of `rrdps_polytope` is exactly zero and neither solver ran.
        """
        bound, witness, tangent, plain, steps, iters = _core.rrdps_polytope(16, 0.0, 0.5)
        self.assertEqual(bound, 0.0, msg=f"bound {bound}")
        self.assertEqual(witness, 0.0, msg=f"witness {witness}")
        self.assertEqual(tangent, 0.0, msg=f"tangent {tangent}")
        self.assertEqual(plain, 0.0, msg=f"closed form {plain}")
        self.assertEqual(steps + iters, 0, msg=f"solvers ran {steps} and {iters}")

    def test_bracket_refused(self):
        """
        A tolerance tighter than the bracket the ascent closed is refused, naming the width
        reached.
        """
        loose = _core.rrdps_polytope(16, 0.1, 0.5, 1e-4)[0]
        tight = _core.rrdps_polytope(16, 0.1, 0.5, 1e-9)[0]
        self.assertEqual(loose, tight, msg=f"loose {loose} against tight {tight}")

        self.assertFails(
            ValueError,
            "closed only to",
            _core.rrdps_polytope,
            16,
            0.1,
            0.5,
            1e-14,
            msg="a tolerance the bracket did not reach",
        )

    def test_packet_domain(self):
        """
        `rrdps_polytope` refuses a train past 250 pulses by name, one under 2 pulses, and a
        p_src or q outside [0, 1].
        """

        self.assertSlots(
            _core.rrdps_polytope,
            (16, 0.1, 0.5),
            [
                (0, "must not exceed 250 pulses for the refined bound", [251, 4096]),
                (0, "l must be at least 2 pulses", [0, 1]),
                (1, "p_src must be", [-0.1, 1.5]),
                (2, "q must be", [-0.1, 1.5]),
            ],
            msg="the refined bound's domain",
        )


class RrdpsProgramme(Guarded):
    """
    The linear-programme solver behind the multiphoton bound, which runs on the DUAL, so an
    early stop is loose rather than wrong.
    """

    def test_dual_value_never_passes_the_minimum(self):
        """
        `lp_dual` reaches 1.0 and 2.5 on two programmes whose optima are known by hand, from
        below in both.
        """
        got, _, _, _ = _core.lp_dual([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0], [0.0], 1e-12)
        self.assertClose(got, 1.0, atol=1e-11, msg="the cheapest vertex costs one")
        self.assertLessEqual(got, 1.0, msg="and is reached from below")

        pair, _, _, _ = _core.lp_dual([2.0, 3.0], [1.0, 1.0, 1.0, -1.0], [1.0, 0.0], [0.0, 0.0], 1e-12)
        self.assertClose(pair, 2.5, atol=1e-11, msg="the balanced point costs 2.5")
        self.assertLessEqual(pair, 2.5, msg="and is reached from below")

    def test_early_stop_is_loose_not_wrong(self):
        """
        A wider central-path gap moves `lp_dual`'s value further below the minimum, never
        above, and the gap it reports covers the distance.
        """
        vals = []
        for tol in (1e-1, 1e-3, 1e-6, 1e-9):
            got, gap, _, _ = _core.lp_dual([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0], [0.0], tol)
            vals.append(got)
            self.assertLessEqual(got, 1.0, msg=f"tol = {tol} stays below the minimum")
            self.assertLessEqual(1.0 - got, gap + 1e-12, msg=f"tol = {tol} gap covers the error")
        self.assertMonotone(vals, rising=True, msg="a tighter ask is a tighter bound")

    def test_programme_shapes_are_guarded(self):
        """
        `lp_dual` refuses by name a shape mismatch between matrix, objective and start, a
        start outside the cone, and an empty programme.
        """
        self.assertBad(
            "flattened row-major",
            _core.lp_dual,
            ([1.0, 2.0], [1.0, 1.0, 1.0], [1.0], [0.0]),
            msg="a carries the wrong number of entries",
        )
        self.assertBad(
            "one value per equality row",
            _core.lp_dual,
            ([1.0, 2.0], [1.0, 1.0], [1.0], [0.0, 0.0]),
            msg="start carries the wrong number of entries",
        )
        self.assertBad(
            "strictly feasible",
            _core.lp_dual,
            ([1.0, 2.0], [1.0, 1.0], [1.0], [5.0]),
            msg="the start sits outside the cone",
        )
        self.assertBad(
            "at least one variable",
            _core.lp_dual,
            ([], [], [], []),
            msg="a programme with nothing in it",
        )


if __name__ == "__main__":
    rc = Exam(
        "RrdpsLeakage",
        "Privacy amplification bounded by the packet length alone, no observable read",
        "rrdps_leakage.md",
    ).run(load(RrdpsLeakage))
    rc |= Exam(
        "RrdpsTolerance",
        "The tolerable bit error rate against Yin et al.'s published table",
        "rrdps_tolerance.md",
    ).run(load(RrdpsTolerance))
    rc |= Exam(
        "RrdpsPublished",
        "The whole rate expression against the L = 65 experiment as Yin et al. read it",
        "rrdps_published.md",
    ).run(load(RrdpsPublished))
    rc |= Exam(
        "RrdpsTrainLength",
        "The trade-off in L: a tighter bound against sifting, resolved at an interior optimum",
        "rrdps_length.md",
    ).run(load(RrdpsTrainLength))
    rc |= Exam(
        "RrdpsAgainstMonitoring",
        "Key at an error rate where the monitoring-based click protocols have none",
        "rrdps_monitoring.md",
    ).run(load(RrdpsAgainstMonitoring))
    rc |= Exam(
        "RrdpsMultiphoton",
        "Eq. (1) past one photon: a concave maximisation certified by a tangent plane",
        "rrdps_multiphoton.md",
    ).run(load(RrdpsMultiphoton))
    rc |= Exam(
        "RrdpsProgramme",
        "The linear programme behind it, and which side of the minimum it stops on",
        "rrdps_programme.md",
    ).run(load(RrdpsProgramme))
    rc |= Exam(
        "RrdpsRefined",
        "Matsuura's conditional-entropy bound, its bracket, and what it buys over tagging",
        "rrdps_refined.md",
    ).run(load(RrdpsRefined))
    rc |= Exam(
        "RrdpsGuards",
        "What the engine refuses, and with what message",
        "rrdps_guards.md",
    ).run(load(RrdpsGuards))
    sys.exit(rc)
