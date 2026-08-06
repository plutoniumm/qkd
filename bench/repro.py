import os
import subprocess
import sys

import numpy as np

from MDB import LINK, ROOT, Suite, Trial, load, symbols

# A draw is a pure function of (seed, stage, symbol index), so the answer cannot
# depend on chunking or thread count. RAYON_NUM_THREADS is read once at pool
# construction, so bench_threads spawns a process per thread count and its
# Best/Median columns are launch cost, not pipeline cost.

# Short of the tail: the last DSP block interpolates its phase flat past the end
# of the data, so a shorter run legitimately differs there.
KEEP = 900_000

PROBE = (
    "import qkd._core as c\n"
    "o = c.run_symbols(2000000, 1, "
    f"{', '.join(repr(v) for v in LINK[:7])}, 0.0, "
    f"{', '.join(repr(v) for v in LINK[7:])}, 1024, False)\n"
    "print(repr(o.v_err), repr(o.xi_hat), repr(o.t_hat))\n"
)


def scalars(out):
    return (out.v_err, out.t_hat, out.t_chan, out.sigma2_hat, out.xi_hat, out.cfo)


def gap(a, b):
    if a == b:
        return 0.0

    return abs(a - b) / max(abs(a), abs(b))


# Mirrors test/kit/procs.py::threads (same subprocess-per-thread-count shape), but
# threads() returns strings where this returns floats -- gap() does arithmetic on
# the result immediately, so folding the two needs a return-type parameter neither
# call site asks for. Left as two functions on purpose.
def probe(threads):
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(threads)
    text = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=env,
    ).stdout

    return [float(v) for v in text.split()]


class Repro(Trial):
    warmup = 1
    runs = 3

    def bench_rerun(self):
        """
        The same seed run twice in one process, comparing every scalar output.
        """
        one = scalars(symbols(1e6))
        two = scalars(symbols(1e6))
        bad = [i for i, v in enumerate(one) if v != two[i]]

        return "bit-exact" if not bad else f"{len(bad)}/{len(one)} differ"

    def bench_prefix(self):
        """
        The first 900k recovered frames of a 1e6-symbol run against a 2e6 one.
        """
        one = symbols(1e6, frames=True)
        two = symbols(2e6, frames=True)
        fx = np.asarray(one.frames_x[:KEEP]) == np.asarray(two.frames_x[:KEEP])
        fp = np.asarray(one.frames_p[:KEEP]) == np.asarray(two.frames_p[:KEEP])
        bad = int(np.count_nonzero(~fx) + np.count_nonzero(~fp))

        return "bit-exact" if not bad else f"{bad}/{2 * KEEP} differ"

    def bench_threads(self):
        """
        One rayon thread against eight, out of process; startup dominates time.
        """
        one = probe(1)
        many = probe(8)
        worst = max(gap(a, many[i]) for i, a in enumerate(one))

        return "bit-exact" if worst == 0.0 else f"differs {worst:.1e}"


if __name__ == "__main__":
    sys.exit(
        Suite(
            "Reproducibility",
            "Whether the counter-based scheme actually delivers chunk- and "
            "thread-count invariance (dev/notes.md section 1)",
            "repro.md",
            meta={"symbols": 2_000_000},
        ).run(load(Repro))
    )
