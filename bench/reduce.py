import math
import sys

import numpy as np

from MDB import Suite, Trial, load

# The quantity is the sum of squares, where nothing cancels and the running sum
# grows to ~n. "Naive sequential f32" is np.cumsum with a float32 accumulator,
# whose last element is the left-to-right sum; np.sum is numpy's pairwise tree.

N = 100_000_000

# The largest exact split of 1e8 near the 16k-element per-workgroup
# accumulation of dev/notes.md section 2 (~6100 partials, ~150 KB).
CHUNK = 15_625

FLOOR = math.sqrt(2.0 / N)


def rel(got, ref):
    return abs(float(got) - ref) / abs(ref)


class Reduce(Trial):
    warmup = 1
    runs = 5

    def setup(self):
        rng = np.random.default_rng(7)
        self.x = rng.standard_normal(N, dtype=np.float32)
        self.sq = np.multiply(self.x, self.x)
        self.out = np.empty(N, dtype=np.float32)
        self.wide = np.empty(N, dtype=np.float64)
        # f64 pairwise over exact f32 inputs; own error ~1e-16, confirmed
        # against math.fsum by bench_exact.
        self.ref = float(np.sum(self.sq, dtype=np.float64))
        self.total = float(np.sum(self.x, dtype=np.float64))
        self.scale = math.sqrt(N)

    def bench_naive(self):
        """
        Naive sequential f32 sum of squares, the textbook one-accumulator loop.
        """
        np.cumsum(self.sq, dtype=np.float32, out=self.out)

        return f"rel {rel(self.out[-1], self.ref):.2e}"

    def bench_pairwise(self):
        """
        numpy's pairwise f32 tree reduction over the same sum of squares.
        """
        got = np.sum(self.sq, dtype=np.float32)

        return f"rel {rel(got, self.ref):.2e}"

    def bench_partial(self):
        """
        f32 tree over 15625-element chunks then an f64 finish, the GPU pattern.
        """
        parts = np.sum(self.sq.reshape(-1, CHUNK), axis=1, dtype=np.float32)
        got = np.sum(parts, dtype=np.float64)

        return f"rel {rel(got, self.ref):.2e}"

    def bench_double(self):
        """
        Naive sequential f64 accumulation, the same loop one width wider.
        """
        np.cumsum(self.sq, dtype=np.float64, out=self.wide)

        return f"rel {rel(self.wide[-1], self.ref):.2e}"

    def bench_mean(self):
        """
        Naive sequential f32 sum of the samples themselves, where signs cancel.
        """
        np.cumsum(self.x, dtype=np.float32, out=self.out)
        err = abs(float(self.out[-1]) - self.total) / self.scale

        return f"{err:.2e} SE"

    def bench_scale(self):
        """
        Naive f32 error at n = 1e6, 1e7 and 1e8, showing how it grows with n.
        """
        parts = []
        for size in (1_000_000, 10_000_000, 100_000_000):
            view = self.sq[:size]
            np.cumsum(view, dtype=np.float32, out=self.out[:size])
            ref = float(np.sum(view, dtype=np.float64))
            parts.append(f"{rel(self.out[size - 1], ref):.1e}")

        return " / ".join(parts)

    def bench_exact(self):
        """
        math.fsum against the f64 reference on 1e6 terms, validating that ref.
        """
        view = self.sq[:1_000_000]
        got = math.fsum(view.tolist())

        return f"rel {rel(np.sum(view, dtype=np.float64), got):.2e}"


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Reduction",
            f"Summation error vs the sqrt(2/n) = {FLOOR:.2e} statistical floor "
            f"at n = {N:.0e} (dev/notes.md benchmark 4). The Rate column carries "
            "the measured relative error, not a throughput",
            "reduce.md",
            meta={
                "n": N,
                "floor": f"{FLOOR:.2e}",
            },
        ).run(load(Reduce))
    )
