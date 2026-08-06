import sys

import numpy as np

from MDB import Suite, Trial, load, stamp

import qkd as q

N = 4_000_000

TRAFFIC = 3 * N * 8  # f64 vector add: read b, read c, write a


class Baseline(Trial):
    warmup = 3
    runs = 9

    def setup(self):
        rng = np.random.default_rng(0)
        self.a = np.zeros(N, dtype=np.float64)
        self.b = rng.standard_normal(N)
        self.c = rng.standard_normal(N)
        self.rng = rng

    def bench_backend_info(self):
        """
        Resolve the compute backend and report which one answered.
        """
        name, precision, device = q.backend_info()

        return f"{name}/{precision} on {device}"

    def bench_supports(self):
        """
        Round trip through the PyO3 boundary, measuring bare FFI call cost.
        """
        q.supports("f64")

        return (1, "call")

    def bench_vector_add(self):
        """
        numpy f64 vector add over 32 MB arrays; the memory-bandwidth reference.
        """
        np.add(self.b, self.c, out=self.a)

        return (TRAFFIC, "B")

    def bench_normals(self):
        """
        numpy standard normals, the CPU reference for Gaussian sample throughput.
        """
        self.rng.standard_normal(N, out=self.a)

        return (N, "sample")


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Baseline",
            "PyO3 call cost and numpy CPU reference points",
            "backend.md",
            meta=stamp(),
        ).run(load(Baseline))
    )
