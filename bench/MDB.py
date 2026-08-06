import json
import os
import platform
import statistics
import sys
import time
import traceback

# Benchmark harness mirroring test/MDR.py. Nothing here pins physics -- these
# are not anchors -- and the numbers mean nothing without ./do bench's release
# build. A bench_* docstring becomes its "What it does" cell; its return value
# becomes the Rate cell:
#  None -> no rate; "12 GB/s" -> verbatim; n -> n ops per call, as ops/s;
#  (n, "sample") -> n units per call, as sample/s
# Timing is wall clock around the method body only; setup() is untimed.
#
# Reading the numbers:
# - Quote in-process ratios, never absolutes. A fixed arm order is not enough:
#   round-robin the arms rep by rep inside one process, shuffling order per
#   rep, and take the lowest of >=15 medians.
# - CPU time is valid only at fixed parallel structure -- it rewards
#   serialising the parallel overhead away, so grain, thread-count and
#   scheduling questions want wall clock.
# - Conversely, wall clock misreads a change to how much work is done: the
#   Fock layer's zero-diagonal skip reads as a pessimisation. Use CPU time.
# - Keep the terminal window focused; macOS demotes an unfocused app onto the
#   E cores (openscad#4850).
# - A single test file's wall clock is not an instrument.
# - Cargo's release profile silently adds -Cstrip=debuginfo; attributing a call
#   site needs -Cstrip=none -Csplit-debuginfo=packed.

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.dirname(HERE)

# Every bench/*.py imports this module first (for Suite/Trial/load), so the
# repo root goes on sys.path here once rather than once per file. Needed only
# for direct `python bench/x.py` invocation -- ./do bench already exports it
# via PYTHONPATH.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.environ.get("MDB_OUT", os.path.join(HERE, "_results"))

PREFIX = ["", "k", "M", "G", "T", "P"]

# (va, T, xi, eta, v_el, lw_alice, lw_lo, symbol_rate, pilot_db, pilot_frac):
# the guide/link.md defaults at 20 km of fibre, 10 kHz lasers, 100 MBd.
LINK = (4.0, 0.5, 0.01, 0.6, 0.1, 1e4, 1e4, 1e8, 0.0, 0.25)


def symbols(n, block=1024, frames=False):
    """
    One run_symbols call at the LINK defaults. cfo is spliced in after lw_lo:
    run_symbols takes it, cpu/gpu_symbols do not, so LINK stays the 10-tuple
    they share. The import is local so numpy-only files can use this harness
    without a built core.
    """
    from qkd import _core

    return _core.run_symbols(int(n), 1, *LINK[:7], 0.0, *LINK[7:], int(block), frames)


def spin(fn, reps=20_000):
    """
    Call `fn` reps times, for a case too cheap to time once.
    """
    for _ in range(reps):
        fn()

    return (reps, "call")


def stamp(**extra):
    """
    The meta header a suite carries: which backend answered, on what device, and
    the load average, which absolute rates move with.
    """
    from qkd import backend_info

    name, precision, device = backend_info()
    got = {
        "backend": f"{name}/{precision}",
        "device": device,
        "load": f"{os.getloadavg()[0]:.1f}",
    }
    got.update(extra)

    return got


def load(cls):
    """
    Cases as (name, doc, bound method), in definition order, which is the order
    the report table reads in.
    """
    trial = cls()
    trial.setup()
    cases = []
    for name in vars(cls):
        if not name.startswith("bench_"):
            continue

        fn = getattr(trial, name)
        cases.append((name, (fn.__doc__ or "").strip(), fn))

    return cases


def si(value, unit):
    """
    Format with an SI prefix, so 2.4e10 B/s reads as 24 GB/s.
    """
    x = float(value)
    i = 0
    while abs(x) >= 1000.0 and i < len(PREFIX) - 1:
        x = x / 1000.0
        i = i + 1

    return f"{x:.3g} {PREFIX[i]}{unit}"


def _ms(value):
    if value < 1.0:
        return f"{value * 1e3:.3g} us"

    if value < 1000.0:
        return f"{value:.3g} ms"

    return f"{value / 1000.0:.3g} s"


def _rate(work, median):
    # `median` is in ms.
    if work is None:
        return "—"

    if isinstance(work, str):
        return work

    if isinstance(work, tuple) and len(work) == 2:
        count, unit = work
    else:
        count, unit = work, "op"

    try:
        count = float(count)
    except (TypeError, ValueError):
        return str(work)

    if median <= 0.0:
        return "—"

    return si(count / (median / 1e3), unit) + "/s"


def machine():
    """
    The host identification stamped into every report.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


class Trial:
    """
    Every bench_* method on a subclass is timed. `warmup` runs are discarded
    (page faults, BLAS warmups, GPU pipeline creation), then `runs` are kept; a
    repetition over `budget` ms stops the case rather than hanging the suite.
    """

    warmup = 2
    runs = 7
    budget = 30_000.0

    def setup(self):
        return None


class Suite:
    """
    A named group of trials, run to a markdown report and console summary.
    """

    def __init__(self, name, desc, file, meta=None):
        self.name = name
        self.desc = desc
        self.meta = meta or {}
        os.makedirs(OUT_DIR, exist_ok=True)
        self.file = os.path.join(OUT_DIR, file)

    def run(self, cases) -> int:
        trial = cases[0][2].__self__ if cases else Trial()
        records = []
        t0 = time.perf_counter()
        for name, doc, fn in cases:
            records.append(self._one(trial, name, doc, fn))
        dt = time.perf_counter() - t0

        self._write(records, dt)
        bad = [r for r in records if r["status"] != "ok"]
        print(
            f"{'FAIL' if bad else 'DONE'}  {self.name}: "
            f"{len(records) - len(bad)}/{len(records)} in {dt * 1000:.0f}ms"
            f"  -> {os.path.relpath(self.file)}"
        )
        for r in bad:
            print(f"      ERROR {r['name']}: {r['detail']}")

        return 1 if bad else 0

    def _one(self, trial, name, doc, fn):
        rec = {
            "name": name,
            "doc": doc,
            "status": "ok",
            "detail": "",
            "runs": 0,
            "best": 0.0,
            "median": 0.0,
            "spread": 0.0,
            "rate": "—",
        }
        try:
            for _ in range(trial.warmup):
                fn()

            times = []
            work = None
            for _ in range(trial.runs):
                start = time.perf_counter()
                work = fn()
                lap = (time.perf_counter() - start) * 1e3
                times.append(lap)
                if lap > trial.budget:
                    break
        except Exception as err:  # one broken bench must not stop the suite
            rec["status"] = "error"
            rec["detail"] = "".join(traceback.format_exception_only(type(err), err)).strip()

            return rec

        rec["runs"] = len(times)
        rec["best"] = min(times)
        rec["median"] = statistics.median(times)
        rec["spread"] = statistics.pstdev(times) if len(times) > 1 else 0.0
        rec["rate"] = _rate(work, rec["median"])

        return rec

    def _write(self, records, dt):
        info = machine()
        head = f"{info['platform']} - {info['cpus']} cpus - python {info['python']}"
        extra = "".join(f" - {k} {v}" for k, v in self.meta.items())
        lines = [
            f"# {self.name}",
            "",
            self.desc,
            "",
            f"**{len(records)} benchmarks** in {dt * 1000:.0f}ms",
            "",
            f"`{head}{extra}`",
            "",
            "| Benchmark | What it does | Runs | Best | Median | Spread | Rate |",
            "| --------- | ------------ | ---: | ---: | -----: | -----: | ---- |",
        ]
        for r in records:
            what = " ".join(r["doc"].split())
            what = (what.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")) or "—"
            if r["status"] != "ok":
                lines.append(f"| `{r['name']}` | {what} | — | — | — | — | ⚠️ error |")
                continue

            lines.append(
                f"| `{r['name']}` | {what} | {r['runs']} | {_ms(r['best'])} "
                f"| {_ms(r['median'])} | {_ms(r['spread'])} | {r['rate']} |"
            )

        broken = [r for r in records if r["status"] != "ok"]
        if broken:
            lines += ["", "## Errors", ""]
            for r in broken:
                lines += [f"### `{r['name']}`", "", "```", r["detail"], "```", ""]

        with open(self.file, "w") as f:
            f.write("\n".join(lines))

        payload = {
            "name": self.name,
            "desc": self.desc,
            "machine": info,
            "meta": self.meta,
            "seconds": dt,
            "records": records,
        }
        with open(os.path.splitext(self.file)[0] + ".json", "w") as f:
            json.dump(payload, f, indent=2)
