import sys

import numpy as np

from MDB import Suite, Trial, load, symbols

# Box-Muller is used because numpy's ziggurat consumes a variable number of
# uniforms per sample, which would break the symbol -> counter map.
# bench_pipeline divides the draw count out of a whole run_symbols call, so it
# is a lower bound on a standalone generator, not a measurement of one.

N = 20_000_000

# Threefry blocks per symbol at n = 1e6 in 1024-symbol blocks, one block being
# one normals() draw: WALK_TX + WALK_LO (walk stores its increments, so the
# second pass regenerates none), 1 DETECT shared by pilot_bins and block_stats,
# 1 ALICE, 1 CHANNEL, 0.26 for cfo_coarse. It is n-dependent at both ends:
# cfo_coarse's preamble is capped at ACQUIRE = 2**18 so its share falls as 1/n,
# and past 33.5M symbols the shared buffer exceeds CACHE_MAX, block_stats draws
# its own DETECT block and the total becomes 6.26.
BLOCKS = 5.26

DRAWS = 4 * BLOCKS  # 4 normals per Threefry block

TAU = 2.0 * np.pi


class Normals(Trial):
    warmup = 2
    runs = 7

    def setup(self):
        self.rng = np.random.default_rng(11)
        self.buf = np.zeros(N)
        self.b32 = np.zeros(N, dtype=np.float32)
        self.u1 = self.rng.random(N // 2)
        self.u2 = self.rng.random(N // 2)
        self.v1 = self.u1.astype(np.float32)
        self.v2 = self.u2.astype(np.float32)

    def bench_uniform(self):
        """
        PCG64 uniforms, the raw material both transforms consume.
        """
        self.rng.random(N, out=self.buf)

        return (N, "sample")

    def bench_ziggurat(self):
        """
        numpy f64 standard normals: ziggurat, variable uniform consumption.
        """
        self.rng.standard_normal(N, out=self.buf)

        return (N, "sample")

    def bench_zig32(self):
        """
        The same ziggurat at f32 width.
        """
        self.rng.standard_normal(N, dtype=np.float32, out=self.b32)

        return (N, "sample")

    def bench_boxmuller(self):
        """
        Vectorised f64 Box-Muller over stored uniforms: 2 in, 2 out, no branch.
        """
        r = np.sqrt(-2.0 * np.log(self.u1))
        ang = TAU * self.u2
        np.multiply(r, np.cos(ang), out=self.buf[0::2])
        np.multiply(r, np.sin(ang), out=self.buf[1::2])

        return (N, "sample")

    def bench_bm32(self):
        """
        The same Box-Muller at f32 width, the precision the GPU path would use.
        """
        r = np.sqrt(-2.0 * np.log(self.v1))
        ang = np.float32(TAU) * self.v2
        np.multiply(r, np.cos(ang), out=self.b32[0::2])
        np.multiply(r, np.sin(ang), out=self.b32[1::2])

        return (N, "sample")

    def bench_log(self):
        """
        f64 natural log alone, one of Box-Muller's two transcendentals.
        """
        np.log(self.u1)

        return (N // 2, "call")

    def bench_sincos(self):
        """
        f64 sine and cosine of the same array, Box-Muller's other pair.
        """
        np.sin(self.u1)
        np.cos(self.u1)

        return (N, "call")

    def bench_pipeline(self):
        """
        Threefry plus Box-Muller inside run_symbols, about 21 normals per symbol.
        """
        symbols(1e6)

        return (DRAWS * 1_000_000, "normal")


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Gaussians",
            "Gaussian generation throughput and the cost of Box-Muller's fixed "
            "two-uniform contract (dev/notes.md benchmark 2, CPU half)",
            "normals.md",
            meta={"n": N},
        ).run(load(Normals))
    )
