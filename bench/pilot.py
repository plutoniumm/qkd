import sys

import numpy as np

from MDB import Suite, Trial, load

# Work is fixed at TOTAL samples either way, so rates compare across block
# sizes; the FFT computes all B bins and the dot product one, so per useful bin
# the dot product leads by a further factor of B. numpy's matmul dispatches to a
# threaded BLAS gemv and pocketfft does not, so bench_plain* is the control.

TOTAL = 4_194_304

SIZES = (256, 1024, 4096, 16_384, 65_536)

FRAC = 0.25


def tone(size):
    """
    The conjugated pilot carrier for one block: exp(-2j*pi*FRAC*k).
    """
    k = np.arange(size)

    return np.exp(-2j * np.pi * FRAC * k)


class Pilot(Trial):
    warmup = 2
    runs = 7

    def setup(self):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(TOTAL) + 1j * rng.standard_normal(TOTAL)
        sig += np.exp(2j * np.pi * FRAC * np.arange(TOTAL))
        self.sig = sig
        self.grid = {}
        self.tones = {}
        for size in SIZES:
            self.grid[size] = sig.reshape(-1, size)
            self.tones[size] = tone(size)

    def _dot(self, size):
        self.grid[size] @ self.tones[size]

        return (TOTAL, "sample")

    def _plain(self, size):
        np.sum(self.grid[size] * self.tones[size], axis=1)

        return (TOTAL, "sample")

    def _fft(self, size):
        bin_id = round(FRAC * size)
        np.fft.fft(self.grid[size], axis=1)[:, bin_id]

        return (TOTAL, "sample")

    def bench_dot256(self):
        """
        Single-bin DFT as a dot product, 256-sample blocks.
        """
        return self._dot(256)

    def bench_fft256(self):
        """
        Full FFT of 256-sample blocks, keeping one bin.
        """
        return self._fft(256)

    def bench_dot1k(self):
        """
        Single-bin DFT as a dot product, 1024-sample blocks.
        """
        return self._dot(1024)

    def bench_fft1k(self):
        """
        Full FFT of 1024-sample blocks, keeping one bin.
        """
        return self._fft(1024)

    def bench_dot4k(self):
        """
        Single-bin DFT as a dot product, 4096-sample blocks.
        """
        return self._dot(4096)

    def bench_fft4k(self):
        """
        Full FFT of 4096-sample blocks, keeping one bin.
        """
        return self._fft(4096)

    def bench_dot16k(self):
        """
        Single-bin DFT as a dot product, 16384-sample blocks.
        """
        return self._dot(16_384)

    def bench_fft16k(self):
        """
        Full FFT of 16384-sample blocks, keeping one bin.
        """
        return self._fft(16_384)

    def bench_plain1k(self):
        """
        The 1024-block dot without BLAS: one multiply pass, one reduce pass.
        """
        return self._plain(1024)

    def bench_plain4k(self):
        """
        The 4096-block dot without BLAS: one multiply pass, one reduce pass.
        """
        return self._plain(4096)

    def bench_plain16k(self):
        """
        The 16384-block dot without BLAS: one multiply pass, one reduce pass.
        """
        return self._plain(16_384)

    def bench_dot64k(self):
        """
        Single-bin DFT as a dot product, 65536-sample blocks.
        """
        return self._dot(65_536)

    def bench_fft64k(self):
        """
        Full FFT of 65536-sample blocks, keeping one bin.
        """
        return self._fft(65_536)


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Pilot extraction",
            "Single-bin DFT dot product against a full FFT at equal total work (dev/notes.md benchmark 7)",
            "pilot.md",
            meta={"samples": TOTAL},
        ).run(load(Pilot))
    )
