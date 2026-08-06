import math
import sys

import numpy as np

from MDB import Suite, Trial, load

# sigma^2 = 2*pi*(dnu_alice + dnu_lo)*T_s = 1.26e-3 rad^2/sample at the
# guide/link.md defaults (10 kHz + 10 kHz, 100 MBd). Error is the worst phase
# error over all samples, wrapped into [-pi, pi]; the f64 reference is itself
# good to ~1e-12 rad.

CHUNK = 4096  # matches src/pipeline.rs

# Half the note's 1e8, because the f64 reference scan alone is 400 MB on an
# 8 GB machine. The error grows with n.
N = 12_207 * CHUNK

SIGMA = math.sqrt(2.0 * math.pi * 2.0e4 / 1.0e8)

TAU = 2.0 * math.pi


def wrap(a):
    return a - TAU * np.round(a / TAU)


def worst(got, ref):
    return float(np.max(np.abs(wrap(got - ref))))


class Walk(Trial):
    warmup = 1
    runs = 5

    def setup(self):
        rng = np.random.default_rng(5)
        inc = (SIGMA * rng.standard_normal(N)).astype(np.float32)
        self.inc = inc
        self.grid = inc.reshape(-1, CHUNK)
        self.ref = np.cumsum(inc, dtype=np.float64)
        self.out = np.empty(N, dtype=np.float32)
        self.reach = float(np.abs(self.ref[-1]))

    def _carries(self, kind):
        # f64 exclusive scan of the chunk sums, wrapped at every partial.
        sums = np.sum(self.grid, axis=1, dtype=kind)
        carry = np.empty(len(sums))
        run = 0.0
        for i, s in enumerate(sums):
            carry[i] = run
            run = run + float(s)
            run = run - TAU * round(run / TAU)

        return carry

    def bench_naive(self):
        """
        One monolithic f32 prefix sum over every sample, no wrap, no carries.
        """
        np.cumsum(self.inc, dtype=np.float32, out=self.out)

        return f"{worst(self.out, self.ref):.2e} rad"

    def bench_chunked(self):
        """
        f32 scan inside 4096-sample chunks on top of wrapped f64 carries.
        """
        inner = np.cumsum(self.grid, axis=1, dtype=np.float32)
        got = inner + self._carries(np.float32)[:, None]

        return f"{worst(got.ravel(), self.ref):.2e} rad"

    def bench_double(self):
        """
        The same chunked scan entirely in f64, what src/pipeline.rs does today.
        """
        inner = np.cumsum(self.grid, axis=1, dtype=np.float64)
        got = inner + self._carries(np.float64)[:, None]

        return f"{worst(got.ravel(), self.ref):.2e} rad"

    def bench_reach(self):
        """
        Where this walk ended up, against the sqrt(N)*sigma scale it wanders on.
        """
        return f"{self.reach:.0f} rad, scale {SIGMA * math.sqrt(N):.0f}"


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Phase walk",
            "Prefix-sum accuracy for the laser phase random walk at "
            f"n = {N:.0e} (dev/notes.md benchmark 5, accuracy half). The Rate "
            "column carries the worst wrapped phase error, not a throughput",
            "walk.md",
            meta={
                "chunk": CHUNK,
                "sigma2": f"{SIGMA**2:.2e}",
            },
        ).run(load(Walk))
    )
