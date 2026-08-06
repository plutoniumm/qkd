import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd
from kit.cache import memo
from kit.gpu import NeedsGpu, PROBE, READY
from qkd import _core

# RNG agreement is exact integer equality; f32 is checked on a tolerance.

# (va, T, xi, eta, v_el, lw_alice, lw_lo, symbol_rate, pilot_db, pilot_frac) in SNU,
# xi at the channel input. T = 0.5 is 15 km at alpha = 0.2 dB/km, 10 kHz lasers, 100 MBd.
LINK = (4.0, 0.5, 0.05, 0.6, 0.1, 1e4, 1e4, 1e8, 10.0, 0.25)

BLOCK = 256  # symbols per DSP block, one thread each

WIDTH = 256  # threads per workgroup in symbols.wgsl; one f64 partial each

# Measured on an M2 at n = 1e6-1e7: slope 1.2e-7, variance 1.7e-7, pilot 3.6e-7.
SLOPE_TOL = 1e-5

VAR_TOL = 1e-5

SNR_TOL = 1e-5

# xi amplifies a sigma2 error by ~1/(T*xi); see test_excess_noise_matches.
XI_TOL = 1e-3


def rel(got, want):
    """
    Relative difference, falling back to absolute when the reference is zero.
    """
    scale = max(abs(float(want)), 1e-30)

    return abs(float(got) - float(want)) / scale


@memo
def pair(n, seed=7, link=LINK):
    """
    The same plan on both backends, memoised.
    """
    args = (int(n), seed) + tuple(link)

    return (_core.cpu_symbols(*args), _core.gpu_symbols(*args))


@memo
def split(n, seed=7, link=LINK):
    """
    The same plan through the split kernel pair, memoised the same way.
    """

    return _core.split_symbols(int(n), seed, *link)


class Threefry(NeedsGpu):
    def test_words_match_reference(self):
        """
        The GPU's Threefry blocks equal the CPU's exactly over 4096 counters.
        """
        want = _core.cpu_words(0xC0FFEE, 1, 0, 4096, 0)
        got = _core.gpu_words(0xC0FFEE, 1, 0, 4096, 0)

        self.assertEqual(len(got), 4 * 4096, msg=f"words for 4096 counters: {len(got)}")
        self.assertEqual(got.tolist(), want.tolist(), msg="GPU words vs CPU words")

    def test_words_every_stage(self):
        """
        The GPU and CPU words agree for each of the five noise stages.
        """
        for stage in (1, 2, 3, 4, 5):
            want = _core.cpu_words(987654321, stage, 0, 512, 0)
            got = _core.gpu_words(987654321, stage, 0, 512, 0)
            self.assertEqual(got.tolist(), want.tolist(), msg=f"stage {stage}")

    def test_words_high_counter(self):
        """
        Counters past 2^32 match, exercising the counter's high word.
        """
        base = (1 << 32) - 3
        want = _core.cpu_words(42, 2, base, 64, 0)
        got = _core.gpu_words(42, 2, base, 64, 0)

        self.assertEqual(got.tolist(), want.tolist(), msg="64-bit counter carry")

    def test_words_draw_axis(self):
        """
        A non-zero draw index selects a different block, identically on both.
        """
        want = _core.cpu_words(42, 2, 0, 64, 3)
        got = _core.gpu_words(42, 2, 0, 64, 3)

        self.assertEqual(got.tolist(), want.tolist(), msg="draw axis")
        self.assertNotEqual(got.tolist(), _core.gpu_words(42, 2, 0, 64, 0).tolist(), msg="draw 3 equals draw 0")

    def test_words_index_pure(self):
        """
        A word depends only on its counter, not on how many were asked for.
        """
        few = _core.gpu_words(5, 1, 0, 64, 0)
        many = _core.gpu_words(5, 1, 0, 4096, 0)

        self.assertEqual(few.tolist(), many[: len(few)].tolist(), msg="index -> value map vs dispatch size")

    def test_words_seeds_differ(self):
        """
        Two seeds give different words.
        """
        one = _core.gpu_words(1, 1, 0, 64, 0)
        two = _core.gpu_words(2, 1, 0, 64, 0)

        self.assertNotEqual(one.tolist(), two.tolist(), msg="seeds 1 and 2 gave one stream")


class Fused(NeedsGpu):
    def test_slope_matches(self):
        """
        The GPU slope sum(a.y)/sum(a.a) matches the f64 arbiter to SLOPE_TOL = 1e-5.
        """
        cpu, gpu = pair(1e6)

        self.assertLessEqual(rel(gpu.t_hat, cpu.t_hat), SLOPE_TOL, msg=f"slope {gpu.t_hat!r} vs {cpu.t_hat!r}")

    def test_variance_matches(self):
        """
        The residual variance of y - t_hat*a agrees with the arbiter to VAR_TOL = 1e-5.
        """
        cpu, gpu = pair(1e6)

        self.assertLessEqual(rel(gpu.sigma2, cpu.sigma2), VAR_TOL, msg=f"sigma2 {gpu.sigma2!r} vs {cpu.sigma2!r}")

    def test_excess_noise_matches(self):
        """
        Input-referred xi agrees with the CPU arbiter within XI_TOL = 1e-3, looser because
        (sigma2 - 1 - v_el)/t_hat^2 amplifies a sigma2 error by ~1/(T*xi), forty here.
        """
        cpu, gpu = pair(1e6)

        self.assertLessEqual(rel(gpu.xi, cpu.xi), XI_TOL, msg=f"xi {gpu.xi!r} vs {cpu.xi!r}")

    def test_pilot_snr_matches(self):
        """
        The pilot bin's SNR over the adjacent empty bin agrees to SNR_TOL = 1e-5.
        """
        cpu, gpu = pair(1e6)

        self.assertGreater(cpu.pilot_snr, 1.0, msg=f"cpu pilot snr {cpu.pilot_snr!r}")
        self.assertLessEqual(
            rel(gpu.pilot_snr, cpu.pilot_snr),
            SNR_TOL,
            msg=f"pilot snr {gpu.pilot_snr!r} vs {cpu.pilot_snr!r}",
        )

    def test_static_phase_matches(self):
        """
        With both linewidths zero the walk vanishes and slope and variance still agree.
        """
        link = (4.0, 0.5, 0.05, 0.6, 0.1, 0.0, 0.0, 1e8, 10.0, 0.25)
        cpu, gpu = pair(5e5, seed=11, link=link)

        self.assertLessEqual(rel(gpu.t_hat, cpu.t_hat), SLOPE_TOL, msg=f"slope {gpu.t_hat!r} vs {cpu.t_hat!r}")
        self.assertLessEqual(rel(gpu.sigma2, cpu.sigma2), VAR_TOL, msg=f"sigma2 {gpu.sigma2!r} vs {cpu.sigma2!r}")

    def test_no_modulation_matches(self):
        """
        With V_A = 0 there is nothing to regress on and both report a bare slope.
        """
        link = (0.0, 0.5, 0.05, 0.6, 0.1, 1e4, 1e4, 1e8, 10.0, 0.25)
        cpu, gpu = pair(5e5, seed=13, link=link)

        self.assertEqual(gpu.t_hat, 0.0, msg=f"slope at V_A = 0: {gpu.t_hat!r}")
        self.assertLessEqual(rel(gpu.sigma2, cpu.sigma2), VAR_TOL, msg=f"sigma2 {gpu.sigma2!r} vs {cpu.sigma2!r}")

    def test_odd_pilot_frequency(self):
        """
        A non-dyadic pilot frequency tracks the CPU SNR to SNR_TOL: frac*k hits 2.5e6 at
        k = 1e7, where an f32 ulp is a quarter cycle, hence the 64-bit fixed point.
        """
        link = (4.0, 0.5, 0.05, 0.6, 0.1, 1e4, 1e4, 1e8, 10.0, 0.31337)
        cpu, gpu = pair(1e6, seed=17, link=link)

        self.assertLessEqual(
            rel(gpu.pilot_snr, cpu.pilot_snr),
            SNR_TOL,
            msg=f"pilot snr {gpu.pilot_snr!r} vs {cpu.pilot_snr!r}",
        )

    def test_error_flat_in_n(self):
        """
        The variance stays inside VAR_TOL across an eightfold n, where a sequential f32 sum
        is 4.5e-4 wrong at n = 1e6 and 2.7e-1 at 1e8, linear in n.
        """
        small = pair(2.5e5, seed=23)
        large = pair(2e6, seed=23)

        for cpu, gpu in (small, large):
            self.assertLessEqual(rel(gpu.sigma2, cpu.sigma2), VAR_TOL, msg=f"sigma2 at n = {gpu.n_used}")

    def test_partials_are_tree(self):
        """
        One f64 partial comes back per workgroup, not one per symbol or one total.
        """
        _cpu, gpu = pair(1e6)
        blocks = gpu.n_used // BLOCK

        self.assertEqual(gpu.groups, -(-blocks // WIDTH), msg=f"groups {gpu.groups} for {blocks} blocks")
        self.assertGreater(gpu.groups, 1, msg=f"groups {gpu.groups}")

    def test_kernel_time_reported(self):
        """
        Timestamp queries report positive kernel time inside the wall clock.
        """
        _cpu, gpu = pair(1e6)

        if gpu.fused_ms <= 0.0:
            self.skipTest("adapter has no TIMESTAMP_QUERY")

        self.assertGreater(gpu.walk_ms, 0.0, msg=f"walk_ms {gpu.walk_ms}")
        self.assertLessEqual(
            gpu.walk_ms + gpu.fused_ms,
            gpu.host_ms,
            msg=f"kernel {gpu.walk_ms + gpu.fused_ms} vs host {gpu.host_ms}",
        )


class Reuse(NeedsGpu):
    def test_small_after_large(self):
        """
        A small run is bit-identical before and after a larger one: a run reading past its
        block count would fold in stale partials.
        """
        args = (250_000, 31) + tuple(LINK)

        first = _core.gpu_symbols(*args)
        _core.gpu_symbols(4_000_000, 31, *LINK)
        again = _core.gpu_symbols(*args)

        self.assertEqual(
            (first.t_hat, first.sigma2, first.pilot_snr),
            (again.t_hat, again.sigma2, again.pilot_snr),
            msg="small run after a large one",
        )

    def test_plan_is_rewritten(self):
        """
        Changing the link between calls changes the answer, and changing back restores it.
        """
        loud = (4.0, 0.5, 0.2, 0.6, 0.1, 1e4, 1e4, 1e8, 10.0, 0.25)

        base = _core.gpu_symbols(300_000, 5, *LINK)
        other = _core.gpu_symbols(300_000, 5, *loud)
        back = _core.gpu_symbols(300_000, 5, *LINK)

        self.assertNotEqual(base.xi, other.xi, msg=f"xi {base.xi} vs loud {other.xi}")
        self.assertEqual(base.xi, back.xi, msg=f"xi {base.xi} vs {back.xi}")

    def test_words_after_words(self):
        """
        A short Threefry request after a long one still matches the reference.
        """
        _core.gpu_words(5, 1, 0, 4096, 0)
        got = _core.gpu_words(5, 1, 0, 64, 0)

        self.assertEqual(got.tolist(), _core.cpu_words(5, 1, 0, 64, 0).tolist(), msg="words after a longer request")

    def test_fused_after_split(self):
        """
        A split run in between leaves the fused answer bit-identical, the split arm owning
        an extra phase array.
        """
        args = (300_000, 41) + tuple(LINK)

        first = _core.gpu_symbols(*args)
        _core.split_symbols(*args)
        again = _core.gpu_symbols(*args)

        self.assertEqual(
            (first.t_hat, first.sigma2),
            (again.t_hat, again.sigma2),
            msg="fused after a split run",
        )

    def test_empty_stays_cheap(self):
        """
        Repeated empty dispatches keep reporting a finite, positive latency.
        """
        laps = [_core.gpu_empty() for _ in range(5)]

        self.assertTrue(all(math.isfinite(x) and x > 0.0 for x in laps), msg=f"empty dispatches {laps!r}")


class Split(NeedsGpu):
    def test_split_matches_fused(self):
        """
        Storing the phase rather than re-deriving it gives the fused kernel's slope,
        variance and pilot SNR within the f32 tolerances: the two wrap modulo 2*pi at
        different points, so a wider gap means the split arm times other physics.
        """
        _cpu, fused = pair(1e6)
        got = split(1e6)

        self.assertLessEqual(rel(got.t_hat, fused.t_hat), SLOPE_TOL, msg=f"slope {got.t_hat!r} vs {fused.t_hat!r}")
        self.assertLessEqual(rel(got.sigma2, fused.sigma2), VAR_TOL, msg=f"sigma2 {got.sigma2!r} vs {fused.sigma2!r}")
        self.assertLessEqual(
            rel(got.pilot_snr, fused.pilot_snr),
            SNR_TOL,
            msg=f"pilot snr {got.pilot_snr!r} vs {fused.pilot_snr!r}",
        )

    def test_split_matches_arbiter(self):
        """
        The split pair's variance and xi match the f64 arbiter at VAR_TOL and XI_TOL.
        """
        cpu, _fused = pair(1e6)
        got = split(1e6)

        self.assertLessEqual(rel(got.sigma2, cpu.sigma2), VAR_TOL, msg=f"sigma2 {got.sigma2!r} vs {cpu.sigma2!r}")
        self.assertLessEqual(rel(got.xi, cpu.xi), XI_TOL, msg=f"xi {got.xi!r} vs {cpu.xi!r}")

    def test_split_reduces_the_same(self):
        """
        The split pair uses the same block grid and workgroup tree as the fused one.
        """
        _cpu, fused = pair(1e6)
        got = split(1e6)

        self.assertEqual(got.n_used, fused.n_used, msg=f"n_used {got.n_used} vs {fused.n_used}")
        self.assertEqual(got.groups, fused.groups, msg=f"groups {got.groups} vs {fused.groups}")

    def test_split_times_two_passes(self):
        """
        Both split passes report their own kernel time, inside the wall clock.
        """
        got = split(1e6)

        if got.fused_ms <= 0.0:
            self.skipTest("adapter has no TIMESTAMP_QUERY")

        self.assertGreater(got.walk_ms, 0.0, msg=f"walk_ms {got.walk_ms}")
        self.assertLessEqual(
            got.walk_ms + got.fused_ms,
            got.host_ms,
            msg=f"kernel {got.walk_ms + got.fused_ms} vs host {got.host_ms}",
        )


class Mirror(Question):
    def test_mirror_matches_pipeline(self):
        """
        The f64 arbiter reproduces run_symbols' oracle branch -- slope 1e-12, xi 1e-10,
        pilot SNR 1e-8 -- and arbitrates only while cpu_symbols mirrors the WGSL.
        """
        n = 500_000
        args = (n, 7) + tuple(LINK)

        mine = _core.cpu_symbols(*args)
        # LINK carries no carrier-frequency offset; the oracle branch cancels it.
        full = _core.run_symbols(n, 7, *LINK[:7], 0.0, *LINK[7:], BLOCK, False)

        self.assertClose(mine.t_hat, full.t_ideal, atol=1e-12, msg=f"slope {mine.t_hat} vs {full.t_ideal}")
        self.assertClose(mine.xi, full.xi_ideal, atol=1e-10, msg=f"xi {mine.xi} vs {full.xi_ideal}")
        self.assertClose(
            mine.pilot_snr,
            full.pilot_snr,
            atol=1e-8,
            msg=f"pilot snr {mine.pilot_snr} vs {full.pilot_snr}",
        )

    def test_probe_always_answers(self):
        """
        gpu_probe() names the adapter or says why there is none, never nothing.
        """

        self.assertTrue(PROBE, msg=f"gpu_probe() {PROBE!r}")
        self.assertIsInstance(READY, bool, msg=f"gpu_ready() {READY!r}")

    def test_absent_gpu_is_not_an_error(self):
        """
        With no adapter the four GPU entry points raise naming the missing compute path.
        """
        if READY:
            self.skipTest("this machine has a gpu compute path")

        for call in (
            lambda: _core.gpu_words(1, 1, 0, 16, 0),
            lambda: _core.gpu_empty(),
            lambda: _core.gpu_symbols(100_000, 1, *LINK),
            lambda: _core.split_symbols(100_000, 1, *LINK),
        ):
            with self.assertRaises(RuntimeError) as caught:
                call()
            self.assertIn("no gpu compute path", str(caught.exception), msg=f"raised {caught.exception}")

    def test_cpu_path_needs_no_gpu(self):
        """
        The f64 arbiter runs whatever the adapter situation is.
        """
        args = (100_000, 3) + tuple(LINK)
        out = _core.cpu_symbols(*args)

        self.assertFinite((out.t_hat, out.sigma2, out.xi, out.pilot_snr), msg="cpu_symbols statistics")

    def test_precision_not_device(self):
        """
        Dispatch is by precision: a GPU reports f32 and refuses f64, WGSL having none,
        while the f64 key rate stays reachable.
        """
        name, precision, _device = qkd.backend_info()

        self.assertTrue(qkd.supports("f32"), msg="f32 unsupported")

        if name == "gpu":
            self.assertEqual(precision, "f32", msg=f"gpu precision {precision}")
            self.assertFalse(qkd.supports("f64"), msg="the gpu claims f64")

        rate = _core.cv_rate(4.0, 0.5, 0.05, 0.6, 0.1, 0.95, False, True)

        self.assertFinite(rate, msg=f"cv_rate {rate}")


if __name__ == "__main__":
    rc = Exam(
        "GPU Threefry",
        "Threefry-4x32-20 in WGSL, bit-exact against the Rust reference",
        "gpu_rng.md",
    ).run(load(Threefry))
    rc |= Exam(
        "GPU symbols",
        "The fused per-symbol kernel against the f64 CPU arbiter",
        "gpu_symbols.md",
    ).run(load(Fused))
    rc |= Exam(
        "GPU buffer pool",
        "Buffers reused across calls must not carry one run's data into the next",
        "gpu_pool.md",
    ).run(load(Reuse))
    rc |= Exam(
        "GPU split kernels",
        "The split kernel pair against the fused kernel it is measured against",
        "gpu_kernels.md",
    ).run(load(Split))
    rc |= Exam(
        "GPU fallback",
        "Guards that hold with or without an adapter, and precision dispatch",
        "gpu_fallback.md",
    ).run(load(Mirror))
    sys.exit(rc)
