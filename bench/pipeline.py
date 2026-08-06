import sys

from MDB import Suite, Trial, load, stamp, symbols

# The pilot DSP branch stays on the host, so these rates are not comparable to a
# kernel rate; the comparable pair is cpu_symbols/gpu_symbols in bench/gpu.py.
# MDB.symbols runs MDB.LINK, the guide/link.md defaults at 20 km of fibre.


class Symbols(Trial):
    warmup = 1
    runs = 3

    def bench_1e4(self):
        """
        Ten thousand symbols: small enough that fixed per-call cost still shows.
        """
        symbols(1e4)

        return (1e4, "symbol")

    def bench_1e5(self):
        """
        A hundred thousand symbols through the full pipeline.
        """
        symbols(1e5)

        return (1e5, "symbol")

    def bench_1e6(self):
        """
        One million symbols, the default Link.run() size.
        """
        symbols(1e6)

        return (1e6, "symbol")

    def bench_1e7(self):
        """
        Ten million symbols.
        """
        symbols(1e7)

        return (1e7, "symbol")

    def bench_1e8(self):
        """
        A hundred million symbols, the size dev/notes.md is written about.
        """
        symbols(1e8)

        return (1e8, "symbol")

    def bench_frames(self):
        """
        One million symbols keeping every recovered frame: 16 B/symbol stored.
        """
        symbols(1e6, frames=True)

        return (1e6, "symbol")

    def bench_short(self):
        """
        One million symbols in 128-symbol DSP blocks, eight times more blocks.
        """
        symbols(1e6, block=128)

        return (1e6, "symbol")

    def bench_long(self):
        """
        One million symbols in 16384-symbol DSP blocks, coarser phase tracking.
        """
        symbols(1e6, block=16384)

        return (1e6, "symbol")

    def bench_floor(self):
        """
        The smallest legal run, four symbols in two blocks: fixed call cost.
        """
        symbols(4, block=2)

        return (1, "call")


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Symbol pipeline",
            "End-to-end throughput of _core.run_symbols "
            "(dev/notes.md benchmark 8; run_symbols is CPU by construction, DSP included)",
            "pipeline.md",
            meta=stamp(),
        ).run(load(Symbols))
    )
