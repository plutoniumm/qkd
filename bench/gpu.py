import functools
import sys
import time

from MDB import LINK, Suite, Trial, load, stamp, symbols

import qkd as q
from qkd import _core

# resolve() memoises into a OnceLock, so only this first call pays for a wgpu
# instance and an adapter request. Timed before anything else touches it.
MARK = time.perf_counter()

q.backend_info()

COLD = (time.perf_counter() - MARK) * 1e3

# Second one-off, memoised the same way: a logical device plus five WGSL
# pipelines (words, walk, symbols, store, stats) over the threefry.wgsl prelude.
MARK = time.perf_counter()

READY = _core.gpu_ready()

BUILD = (time.perf_counter() - MARK) * 1e3

# The GPU computes the oracle branch, not the DSP-derotated one, so run_symbols
# -- hence every q.Link run -- stays on CPU. Never quote a single-number speedup
# from this file: the CPU rows carry whatever load the host was under.
# dev/notes.md appendices 1-4.

NO_CLOCK = "no TIMESTAMP_QUERY on this adapter"


def cpu(n):
    return _core.cpu_symbols(int(n), 1, *LINK)


def gpu(n):
    return _core.gpu_symbols(int(n), 1, *LINK)


def split(n):
    return _core.split_symbols(int(n), 1, *LINK)


def skip():
    """
    The rate cell for a GPU benchmark on a machine that has no GPU.
    """
    return f"skipped: {_core.gpu_probe()}"


def gated(fn):
    """
    A case that needs an adapter: with none it reports the probe's reason in the
    Rate cell instead of running. An absent GPU is a fallback, not an error.
    """

    @functools.wraps(fn)
    def wrap(self):
        if not READY:
            return skip()

        return fn(self)

    return wrap


class Probe(Trial):
    warmup = 3
    runs = 9

    def bench_cold(self):
        """
        The one-off first resolve, timed at import: wgpu instance plus adapter.
        """
        return f"{COLD:.1f} ms once"

    def bench_build(self):
        """
        The one-off compute context: logical device plus five WGSL pipelines.
        """
        return f"{BUILD:.1f} ms once"

    def bench_resolve(self):
        """
        Every later backend_info(), which is a memoised OnceLock read.
        """
        q.backend_info()

        return (1, "probe")

    def bench_ready(self):
        """
        The gpu_ready() query a dispatch policy would make per plan stage.
        """
        _core.gpu_ready()

        return (1, "call")

    @gated
    def bench_empty(self):
        """
        One workgroup with nothing to do, submitted and read back: pure latency.
        """
        ms = _core.gpu_empty()

        return f"{ms * 1e3:.0f} us/dispatch"


class Threefry(Trial):
    warmup = 2
    runs = 5

    def bench_cpu_words(self):
        """
        Threefry-4x32-20 blocks from the Rust reference, one thread.
        """
        _core.cpu_words(1, 1, 0, 1 << 18, 0)

        return (1 << 18, "block")

    @gated
    def bench_gpu_words(self):
        """
        The same blocks from WGSL, including dispatch and a 4 MB readback.
        """
        _core.gpu_words(1, 1, 0, 1 << 18, 0)

        return (1 << 18, "block")


class Cross(Trial):
    """
    Read the BEST column, not the median: contention only ever makes a run
    slower, so the minimum is the number about this machine. bench_anchor says
    how loaded it was.
    """

    warmup = 1
    runs = 5

    def bench_anchor(self):
        """
        run_symbols at 1e6: not a competitor here (it also runs the pilot DSP)
        but the contention anchor, 10.3 M symbol/s on a quiet M2. Far below
        that means a debug build or a busy machine, and no other row counts.
        """
        symbols(1e6)

        return (1e6, "symbol")

    def bench_cpu_1e3(self):
        """
        A thousand symbols on the CPU: three DSP blocks, essentially serial.
        """
        cpu(1e3)

        return (1e3, "symbol")

    @gated
    def bench_gpu_1e3(self):
        """
        A thousand symbols on the GPU: three threads, pure fixed cost.
        """
        gpu(1e3)

        return (1e3, "symbol")

    def bench_cpu_3e3(self):
        """
        Three thousand symbols on the CPU, bracketing the crossover from below.
        """
        cpu(3e3)

        return (3e3, "symbol")

    @gated
    def bench_gpu_3e3(self):
        """
        Three thousand symbols on the GPU, bracketing the crossover from below.
        """
        gpu(3e3)

        return (3e3, "symbol")

    def bench_cpu_1e4(self):
        """
        Ten thousand symbols on the f64 CPU arbiter.
        """
        cpu(1e4)

        return (1e4, "symbol")

    @gated
    def bench_gpu_1e4(self):
        """
        Ten thousand symbols on the f32 kernel: only 39 threads, all latency.
        """
        gpu(1e4)

        return (1e4, "symbol")

    def bench_cpu_1e5(self):
        """
        A hundred thousand symbols on the CPU.
        """
        cpu(1e5)

        return (1e5, "symbol")

    @gated
    def bench_gpu_1e5(self):
        """
        A hundred thousand symbols on the GPU.
        """
        gpu(1e5)

        return (1e5, "symbol")

    def bench_cpu_1e6(self):
        """
        One million symbols on the CPU, the default Link.run() size.
        """
        cpu(1e6)

        return (1e6, "symbol")

    @gated
    def bench_gpu_1e6(self):
        """
        One million symbols on the GPU.
        """
        gpu(1e6)

        return (1e6, "symbol")

    def bench_cpu_1e7(self):
        """
        Ten million symbols on the CPU.
        """
        cpu(1e7)

        return (1e7, "symbol")

    @gated
    def bench_gpu_1e7(self):
        """
        Ten million symbols on the GPU.
        """
        gpu(1e7)

        return (1e7, "symbol")


class Large(Trial):
    warmup = 1
    runs = 2

    def bench_cpu_1e8(self):
        """
        A hundred million symbols on the f64 CPU arbiter.
        """
        cpu(1e8)

        return (1e8, "symbol")

    @gated
    def bench_gpu_1e8(self):
        """
        A hundred million symbols on the f32 kernel.
        """
        gpu(1e8)

        return (1e8, "symbol")


class Split(Trial):
    warmup = 1
    runs = 3

    def setup(self):
        self.out = gpu(1e7) if READY else None

    @gated
    def bench_kernel(self):
        """
        Both kernels' own time at 1e7 symbols, as the adapter clock saw it.
        """
        out = gpu(1e7)
        span = out.walk_ms + out.fused_ms
        if span <= 0.0:
            return NO_CLOCK

        return f"{1e4 / span:.0f} M symbol/s in-kernel"

    @gated
    def bench_overhead(self):
        """
        Wall clock minus kernel time: two submits, two waits, two readbacks.
        """
        out = gpu(1e7)
        span = out.walk_ms + out.fused_ms

        return f"{out.host_ms - span:.2f} ms of {out.host_ms:.2f} ms"

    @gated
    def bench_latency(self):
        """
        The same split at 1e5 symbols, where the fixed cost sets the crossover:
        two submits, two full waits and two readbacks, paid again per call.
        """
        out = gpu(1e5)
        span = out.walk_ms + out.fused_ms

        return f"{out.host_ms - span:.2f} ms fixed, {span:.2f} ms kernel"

    @gated
    def bench_rng_share(self):
        """
        The walk pass (2 of a symbol's 7 Threefry blocks) against the fused one
        (all 7, plus every transcendental), pricing the phase stages alone. A
        decomposition, not the fusion measurement -- that one runs both arms.
        """
        out = self.out
        if out is None or out.fused_ms <= 0.0:
            return NO_CLOCK

        return f"walk {out.walk_ms:.2f} ms vs fused {out.fused_ms:.2f} ms"


class Fusion(Trial):
    """
    The trade is 29% of the counter RNG against 8 B/symbol, not a bandwidth
    trade: the pipeline runs at 0.3% of the roof. Sizes cap at 1e7 because the
    phase array is 4n bytes on a machine with 8 GB shared with the CPU. Compare
    in-kernel rows -- the split arm pays a first-run allocation the fused does
    not.
    """

    warmup = 2
    runs = 5

    def setup(self):
        self.fused = gpu(1e7) if READY else None
        self.split = split(1e7) if READY else None

    @gated
    def bench_fused_1e7(self):
        """
        Ten million symbols through the fused kernel, wall clock.
        """
        gpu(1e7)

        return (1e7, "symbol")

    @gated
    def bench_split_1e7(self):
        """
        The same ten million through the split pair, wall clock.
        """
        split(1e7)

        return (1e7, "symbol")

    def passes(self, out):
        """
        One arm's two kernel times and their sum, as the adapter's clock saw it.
        """
        if out is None or out.fused_ms <= 0.0:
            return NO_CLOCK

        return f"{out.walk_ms:.2f} + {out.fused_ms:.2f} = {out.walk_ms + out.fused_ms:.2f} ms"

    @gated
    def bench_kernel_fused(self):
        """
        The fused arm's own two passes at 1e7, on the adapter's clock.
        """
        return self.passes(self.fused)

    @gated
    def bench_kernel_split(self):
        """
        The split arm's own two passes at 1e7, the same way.
        """
        return self.passes(self.split)

    @gated
    def bench_fusion_win(self):
        """
        Split in-kernel time over fused: above 1 fusion wins, below 1 it loses.
        """
        one = self.fused
        two = self.split
        if one is None or two is None:
            return NO_CLOCK

        span = one.walk_ms + one.fused_ms
        other = two.walk_ms + two.fused_ms
        if span <= 0.0 or other <= 0.0:
            return NO_CLOCK

        return f"{other / span:.3f}x split/fused"

    @gated
    def bench_fusion_error(self):
        """
        How far the two arms' answers differ, which bounds what the timing means.
        """
        one = self.fused
        two = self.split
        if one is None or two is None:
            return skip()

        err = abs(two.sigma2 - one.sigma2) / abs(one.sigma2)

        return f"rel {err:.1e} in sigma2"


class Accuracy(Trial):
    warmup = 0
    runs = 1

    def setup(self):
        self.ref = cpu(1e7)
        self.got = gpu(1e7) if READY else None

    def rel(self, field):
        """
        One scalar of the f32 kernel against the f64 arbiter, relative.
        """
        got = getattr(self.got, field)
        ref = getattr(self.ref, field)

        return f"rel {abs(got - ref) / abs(ref):.1e}"

    @gated
    def bench_slope(self):
        """
        Relative GPU-vs-CPU difference in the regression slope at 1e7 symbols.
        """
        return self.rel("t_hat")

    @gated
    def bench_variance(self):
        """
        Relative difference in the residual variance, the reduction's own error.
        """
        return self.rel("sigma2")

    @gated
    def bench_excess(self):
        """
        Relative difference in xi, where (sigma2 - 1 - v_el) amplifies by 1/(T*xi).
        """
        return self.rel("xi")

    @gated
    def bench_partials(self):
        """
        Workgroup partials the host finished in f64, one per 256 DSP blocks.
        """
        return f"{self.got.groups} partials"


if __name__ == "__main__":
    meta = stamp(gpu=_core.gpu_probe())
    rc = Suite(
        "GPU setup",
        "One-off adapter, device and pipeline costs, and the dispatch latency floor",
        "gpu.md",
        meta=meta,
    ).run(load(Probe))
    rc |= Suite(
        "GPU Threefry",
        "Counter-RNG block throughput on both backends (dev/notes.md benchmark 2, GPU arm)",
        "gpu_rng.md",
        meta=meta,
    ).run(load(Threefry))
    rc |= Suite(
        "GPU crossover",
        "cpu_symbols against gpu_symbols over four decades: the measured crossover of dev/notes.md benchmark 8",
        "gpu_cross.md",
        meta=meta,
    ).run(load(Cross))
    rc |= Suite(
        "GPU at 1e8",
        "The same comparison at a hundred million symbols",
        "gpu_large.md",
        meta=meta,
    ).run(load(Large))
    rc |= Suite(
        "GPU dispatch",
        "Kernel time against wall clock from timestamp queries (dev/notes.md benchmark 6)",
        "gpu_split.md",
        meta=meta,
    ).run(load(Split))
    rc |= Suite(
        "GPU fusion",
        "The fused per-symbol kernel against a split pair that stores the phase (dev/notes.md benchmark 3)",
        "gpu_fusion.md",
        meta=meta,
    ).run(load(Fusion))
    rc |= Suite(
        "GPU accuracy",
        "What f32 costs in agreement with the f64 arbiter at 1e7 symbols",
        "gpu_error.md",
        meta=meta,
    ).run(load(Accuracy))
    sys.exit(rc)
