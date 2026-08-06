import os
import resource
import subprocess
import sys

import numpy as np

from MDB import Suite, Trial, load, si, stamp, symbols

from qkd import _core

# Peak RSS is monotonic within a process, so each row spawns a child running one
# workload and is quoted as a delta over bench_base, the empty child (~60 MB).

# 20 km of fibre, a 1 GBd DPS transmitter (see bench/clicks.py).
CLICK = (0.2, 0.398, 0.2, 1e-6, 0.98, 1, 1e4, 1e9, 50e-9, 0.01)


def peak():
    """
    Peak RSS in bytes. Darwin reports ru_maxrss in bytes, Linux in kibibytes,
    and the scale is fixed here rather than at every call site.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return raw if sys.platform == "darwin" else raw * 1024


def cat(d):
    return _core.FockState.cat(2.0, 0.0, True, d)


def grid(n):
    return np.linspace(-6.0, 6.0, n).tolist()


def gates(n, record=False):
    return _core.run_clicks(int(n), 1, *CLICK, 0, record)


WORK = {
    "base": lambda: None,
    "fock40": lambda: cat(40).eigenvalues(),
    "fock120": lambda: cat(120).eigenvalues(),
    "fock400": lambda: cat(400).eigenvalues(),
    "loss400": lambda: cat(400).loss(0.5),
    "wig101": lambda: cat(40).wigner(grid(101), grid(101)),
    "wig401": lambda: cat(40).wigner(grid(401), grid(401)),
    "wig1001": lambda: cat(40).wigner(grid(1001), grid(1001)),
    "sym1e6": lambda: symbols(1e6),
    "sym1e7": lambda: symbols(1e7),
    "sym5e7": lambda: symbols(5e7),
    "frames": lambda: symbols(1e7, frames=True),
    "read": lambda: symbols(1e7, frames=True).frames_x,
    "click1e7": lambda: gates(1e7),
    "record": lambda: gates(1e7, True),
    "rates": lambda: [_core.cv_rate(4.0, 0.5, 0.01, 0.6, 0.1, 0.95, False, False) for _ in range(100_000)],
    "dm": lambda: [_core.dm_rate(64, 0.5, 0.5, 0.01, 0.6, 0.1, 0.95) for _ in range(1000)],
}


def child(name):
    out = WORK[name]()
    print(f"PEAK {peak()} {out is None}")

    return 0


class Footprint(Trial):
    """
    Best/Median time the whole child, interpreter startup included, so they are
    not comparable with the other benchmark files; the Rate column is the
    measurement. `runs` is 1 because a peak is deterministic.
    """

    warmup = 0
    runs = 1

    def setup(self):
        self.base = 0
        self.base = self.spawn("base")

    def spawn(self, name):
        got = subprocess.run(
            [sys.executable, os.path.abspath(__file__), name],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in got.stdout.splitlines():
            if line.startswith("PEAK "):
                return int(line.split()[1])

        raise RuntimeError(f"child {name} reported no peak: {got.stderr[-400:]}")

    def report(self, name):
        got = self.spawn(name)

        return f"{si(got, 'B')} (+{si(got - self.base, 'B')})"

    def bench_base(self):
        """
        An empty child: interpreter, numpy and the loaded core, nothing run.
        """

        return f"{si(self.spawn('base'), 'B')} (baseline)"

    def bench_fock40(self):
        """
        A cat at the shipped cutoff of 40, eigendecomposed: 2*d^2 f64, which
        the Jacobi solver copies twice.
        """

        return self.report("fock40")

    def bench_fock120(self):
        """
        The same at cutoff 120, nine times the matrix entries.
        """

        return self.report("fock120")

    def bench_fock400(self):
        """
        The same at cutoff 400, a hundred times the matrix entries of 40.
        """

        return self.report("fock400")

    def bench_loss400(self):
        """
        A pure-loss channel at cutoff 400, holding input and output at once.
        """

        return self.report("loss400")

    def bench_wig101(self):
        """
        A 101x101 Wigner grid at cutoff 40: 10201 f64.
        """

        return self.report("wig101")

    def bench_wig401(self):
        """
        A 401x401 Wigner grid, sixteen times the points.
        """

        return self.report("wig401")

    def bench_wig1001(self):
        """
        A 1001x1001 Wigner grid: 8 MB of f64.
        """

        return self.report("wig1001")

    def bench_sym1e6(self):
        """
        One million symbols through the pipeline, reducing on the fly.
        """

        return self.report("sym1e6")

    def bench_sym1e7(self):
        """
        Ten million symbols, where the f64 phase array reaches 80 MB.
        """

        return self.report("sym1e7")

    def bench_sym5e7(self):
        """
        Fifty million symbols, where the phase array reaches 400 MB: half what
        dev/notes.md's 1e8 run needs, on a machine with 8 GB.
        """

        return self.report("sym5e7")

    def bench_frames(self):
        """
        Ten million symbols with keep_frames on: 16 B/symbol held in Rust.
        """

        return self.report("frames")

    def bench_read(self):
        """
        The same run with `.frames_x` read, which memcpys ten million f64 into
        one numpy buffer and allocates nothing else.
        """

        return self.report("read")

    def bench_click(self):
        """
        Ten million detector gates, counters only.
        """

        return self.report("click1e7")

    def bench_record(self):
        """
        The same with keep_record on: three u8 vectors of one byte per gate.
        """

        return self.report("record")

    def bench_rates(self):
        """
        A hundred thousand cv_rate calls, which should allocate nothing.
        """

        return self.report("rates")

    def bench_dm(self):
        """
        A thousand dm_rate calls at M = 64, whose quadrature tables are the
        only heap this family touches.
        """

        return self.report("dm")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(child(sys.argv[1]))

    sys.exit(
        Suite(
            "Memory",
            "Peak resident memory per workload, measured one child process at a "
            "time. The Rate column carries the peak, not a throughput",
            "memory.md",
            meta=stamp(ram=si(8 * 1024**3, "B")),
        ).run(load(Footprint))
    )
