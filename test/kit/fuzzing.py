import math
import random

from kit.checks import Guarded

# Fixed seeds, mixed with a per-test tag, so any failure replays.
SEEDS = (20240517, 8675309, 1618033988, 2718281828, 4294967291)

# Magnitudes spanning f64 from denormal to overflow.
DECADES = (1e-300, 1e-30, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e30, 1e150, 1e300)

# One ulp either side of a domain edge.
UP = math.nextafter(1.0, 2.0)

DOWN = math.nextafter(1.0, 0.0)

NASTY = (float("nan"), float("inf"), float("-inf"))

AXIS = [(-6.0 + 12.0 * k / 40.0) for k in range(41)]


def rng(tag):
    return random.Random(SEEDS[tag % len(SEEDS)] * 1_000_003 + tag)


def loguni(r, lo, hi):
    return 10.0 ** r.uniform(math.log10(lo), math.log10(hi))


def unit(r, lo=1e-4):
    """
    A transmittance-like number in (lo, 1], uniform or log-uniform.
    """
    return r.choice((r.uniform(lo, 1.0), loguni(r, lo, 1.0)))


def grid(flat, nx):
    return [flat[i : i + nx] for i in range(0, len(flat), nx)]


def halve(v):
    """
    Shot-noise-unit covariance entries into the internal hbar = 1 convention.
    """
    return [x * 0.5 for x in v]


def square(v):
    """
    A flat row-major square matrix as a list of rows. Rows are slices, never list()-wrapped:
    a numpy input must yield numpy rows.
    """
    n = int(round(math.sqrt(len(v))))

    return [v[i * n : (i + 1) * n] for i in range(n)]


class Sweep:
    """
    A counted random sweep over one engine's domain: iterating tallies ``drawn``, ``keep()``
    tallies the points a guard did not reject.

    Two sweeps that open alike are not one sweep: pooling them, or reordering the draws
    inside either, moves every point tested.
    """

    def __init__(self, tag, draws):
        self.r = rng(tag)
        self.draws = draws
        self.drawn = 0
        self.kept = 0

    def __iter__(self):
        for _ in range(self.draws):
            self.drawn += 1
            yield self.r

    def keep(self):
        self.kept += 1


class Fuzzed(Guarded):
    """
    A question that sweeps a public entry point against its physical domain.
    """

    def assertSwept(self, sweep, want, msg=None):
        """
        A sweep visited exactly ``want`` points. Takes a Sweep or a plain tally.
        """
        got = sweep.drawn if isinstance(sweep, Sweep) else sweep
        self.assertEqual(got, want, msg=msg)

    def assertHoles(self, sweep, least, msg=None):
        """
        A sweep whose domain has holes kept more than ``least`` of its draws.
        """

        self.assertGreater(sweep.kept, least, msg=msg)

    def assertGuards(self, rows, msg=None):
        """
        A table of (fn, good args, slot, needle, bad values) rows, one slot out of domain.
        """
        for fn, ok, slot, needle, bads in rows:
            self.assertSlots(fn, ok, ((slot, needle, bads),), msg=f"{msg} {needle}")

    def assertRange(self, got, lo, hi, msg=None):
        """
        ``got`` lies in [lo, hi], slack included in the bounds as passed.
        """

        self.assertGreaterEqual(got, lo, msg=f"{msg}: {got} below {lo}")
        self.assertLessEqual(got, hi, msg=f"{msg}: {got} above {hi}")
