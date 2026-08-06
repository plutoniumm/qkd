import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from MDB import Suite, Trial, load

# The roof bench/gpu.py's numbers are judged against. numpy's elementwise ufuncs
# are single-threaded, so the threaded rows fan slices across a pool (np.add
# releases the GIL); 128 MB per array, eight times the P-cluster's 16 MB L2.

N = 16_000_000

THREADS = os.cpu_count() or 8


class Roofline(Trial):
    warmup = 3
    runs = 11

    def setup(self):
        rng = np.random.default_rng(0)
        self.a = np.zeros(N)
        self.b = rng.standard_normal(N)
        self.c = rng.standard_normal(N)
        self.a32 = np.zeros(N, dtype=np.float32)
        self.b32 = self.b.astype(np.float32)
        self.c32 = self.c.astype(np.float32)
        self.pool = ThreadPoolExecutor(THREADS)
        self.cuts = [round(i * N / THREADS) for i in range(THREADS + 1)]

    def _add(self, i):
        lo, hi = self.cuts[i], self.cuts[i + 1]
        np.add(self.b[lo:hi], self.c[lo:hi], out=self.a[lo:hi])

    def _add32(self, i):
        lo, hi = self.cuts[i], self.cuts[i + 1]
        np.add(self.b32[lo:hi], self.c32[lo:hi], out=self.a32[lo:hi])

    def _copy(self, i):
        lo, hi = self.cuts[i], self.cuts[i + 1]
        np.copyto(self.a[lo:hi], self.b[lo:hi])

    def _read(self, i):
        lo, hi = self.cuts[i], self.cuts[i + 1]

        return float(np.sum(self.b[lo:hi]))

    def bench_read(self):
        """
        Read-only f64 reduction over 128 MB on one core: the pure load path.
        """
        float(np.sum(self.b))

        return (N * 8, "B")

    def bench_reads(self):
        """
        The same read-only reduction fanned across every core.
        """
        list(self.pool.map(self._read, range(THREADS)))

        return (N * 8, "B")

    def bench_copy(self):
        """
        f64 array copy on one core, 128 MB in and 128 MB out.
        """
        np.copyto(self.a, self.b)

        return (2 * N * 8, "B")

    def bench_copies(self):
        """
        The same f64 copy fanned across every core: the streaming roof.
        """
        list(self.pool.map(self._copy, range(THREADS)))

        return (2 * N * 8, "B")

    def bench_add(self):
        """
        f64 vector add on one core: two streaming reads and one write.
        """
        np.add(self.b, self.c, out=self.a)

        return (3 * N * 8, "B")

    def bench_adds(self):
        """
        The same f64 vector add fanned across every core: the CPU-side roof.
        """
        list(self.pool.map(self._add, range(THREADS)))

        return (3 * N * 8, "B")

    def bench_add32(self):
        """
        Threaded f32 vector add, testing whether halving the bytes buys rate.
        """
        list(self.pool.map(self._add32, range(THREADS)))

        return (3 * N * 4, "B")


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Roofline",
            "Achievable host memory bandwidth on this machine "
            "(dev/notes.md benchmark 1, host half; the GPU is measured in gpu.py)",
            "roofline.md",
            meta={"threads": THREADS},
        ).run(load(Roofline))
    )
