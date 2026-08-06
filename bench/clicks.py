import os
import sys

from MDB import Suite, Trial, load, spin, stamp, symbols

from qkd import _core

# bench_pipe re-measures the symbol pipeline in-process, so only the ratio to it
# is trustworthy. walk()/raw_slots() are parallel and the dead-time scan
# sequential: its Amdahl fraction needs RAYON_NUM_THREADS=1. `chunk` sets grain
# only, never a result.

# 20 km of fibre at 0.2 dB/km, a 1 GBd DPS transmitter, InGaAs-class detectors.
CLICK = dict(
    seed=1,
    mu=0.2,
    t=0.398,
    eta=0.2,
    dark=1e-6,
    visibility=0.98,
    delay=1,
    linewidth=1e4,
    symbol_rate=1e9,
    dead_time=50e-9,
    afterpulse=0.01,
    chunk=0,
    keep_record=False,
)

THREADS = os.environ.get("RAYON_NUM_THREADS", "default")


def clicks(n, **kw):
    # An unknown key would otherwise be added to p, read by nothing, and the
    # trial would silently re-measure CLICK while its row claimed a variant.
    unknown = sorted(set(kw) - set(CLICK))

    assert not unknown, f"clicks() got unknown override(s): {', '.join(unknown)}"

    p = dict(CLICK)
    p.update(kw)

    return _core.run_clicks(
        int(n),
        p["seed"],
        p["mu"],
        p["t"],
        p["eta"],
        p["dark"],
        p["visibility"],
        p["delay"],
        p["linewidth"],
        p["symbol_rate"],
        p["dead_time"],
        p["afterpulse"],
        p["chunk"],
        p["keep_record"],
    )


class Throughput(Trial):
    warmup = 1
    runs = 3

    def bench_1e5(self):
        """
        A hundred thousand gates: small enough that rayon ramp-up still shows.
        """
        clicks(1e5)

        return (1e5, "gate")

    def bench_1e6(self):
        """
        One million gates.
        """
        clicks(1e6)

        return (1e6, "gate")

    def bench_1e7(self):
        """
        Ten million gates, one hundredth of a second of a 1 GBd transmitter.
        """
        clicks(1e7)

        return (1e7, "gate")

    def bench_2e7(self):
        """
        Twenty million gates, where the f64 phase array reaches 160 MB.
        """
        clicks(2e7)

        return (2e7, "gate")

    def bench_pipe(self):
        """
        The CV symbol pipeline at the same size, re-measured here so both
        engines are seen in the same machine state rather than across runs.
        """
        symbols(1e7)

        return (1e7, "symbol")

    def bench_floor(self):
        """
        The smallest legal run, four gates: the fixed per-call cost.
        """

        return spin(lambda: clicks(4), 2000)


class Passes(Trial):
    warmup = 1
    runs = 3

    def bench_base(self):
        """
        The reference configuration, repeated here so the variants below differ
        against a number taken seconds away rather than minutes.
        """
        clicks(1e7)

        return (1e7, "gate")

    def bench_mono(self):
        """
        A monochromatic laser, linewidth = 0, where src/click.rs skips the
        Wiener walk bit-identically: this subtracts the walk pass exactly and
        leaves the gate pass and the sequential scan.
        """
        clicks(1e7, linewidth=0.0)

        return (1e7, "gate")

    def bench_record(self):
        """
        Keeping the per-gate record: three vectors pushed inside the sequential
        scan instead of counters, i.e. the scan's own store cost.
        """
        clicks(1e7, keep_record=True)

        return (1e7, "gate")

    def bench_live(self):
        """
        Dead time zero, so no gate is ever skipped: the scan takes its longest
        path and the afterpulse branch stays armed.
        """
        clicks(1e7, dead_time=0.0)

        return (1e7, "gate")

    def bench_blind(self):
        """
        A microsecond of dead time, a thousand gates at 1 GBd, so the scan skips
        most of its work after every click.
        """
        clicks(1e7, dead_time=1e-6)

        return (1e7, "gate")

    def bench_quiet(self):
        """
        No afterpulsing: one branch off the scan, and none of the two Threefry
        blocks the parallel gate pass draws for it.
        """
        clicks(1e7, afterpulse=0.0)

        return (1e7, "gate")

    def bench_long(self):
        """
        An eight-symbol delay line instead of one, which only lengthens the
        phase difference the fringe is computed over.
        """
        clicks(1e7, delay=8)

        return (1e7, "gate")


class Grain(Trial):
    warmup = 1
    runs = 3

    def bench_fine(self):
        """
        A 4096-gate grain, 2441 chunks over ten million gates.
        """
        clicks(1e7, chunk=4096)

        return (1e7, "gate")

    def bench_default(self):
        """
        The crate's own default grain.
        """
        clicks(1e7)

        return (1e7, "gate")

    def bench_coarse(self):
        """
        A one-million-gate grain, ten chunks over ten million gates: fewer than
        one per core pair, exposing the tail of the last chunk.
        """
        clicks(1e7, chunk=1_000_000)

        return (1e7, "gate")


class Optics(Trial):
    warmup = 2
    runs = 5

    def bench_prob(self):
        """
        click_prob: one threshold detector's click probability for a coherent
        pulse, i.e. an exponential and two multiplies.
        """

        return spin(lambda: _core.click_prob(0.4, 0.1, 0.2, 1e-6))

    def bench_fringe(self):
        """
        interfere: the delay-line interferometer's two output intensities.
        """

        return spin(lambda: _core.interfere(0.4, 0.1, 0.35, -0.05, 0.98))


if __name__ == "__main__":
    meta = stamp(rayon=THREADS)
    rc = 0
    rc |= Suite(
        "Click throughput",
        "run_clicks against gate count, with the CV symbol pipeline measured beside it as the reference rate",
        "clicks_rate.md",
        meta=meta,
    ).run(load(Throughput))
    rc |= Suite(
        "Click passes",
        "Which of run_clicks' three passes -- parallel walk, parallel gates, "
        "sequential dead-time scan -- the time is in",
        "clicks_passes.md",
        meta=meta,
    ).run(load(Passes))
    rc |= Suite(
        "Click grain",
        "The chunk argument, which moves the parallel grain and cannot move a bit of the result",
        "clicks_grain.md",
        meta=meta,
    ).run(load(Grain))
    rc |= Suite(
        "Click optics",
        "The per-pulse closed forms beside the simulator",
        "clicks_optics.md",
        meta=meta,
    ).run(load(Optics))
    sys.exit(rc)
