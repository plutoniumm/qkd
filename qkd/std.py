import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn

import numpy as np

# ANNOTATION ONLY: components imports gaussian imports std, so a runtime import
# would close that cycle. Never "fix" it.
if TYPE_CHECKING:
    from .components import Channel

ArrayLike = Sequence[float] | np.ndarray

# The SHAPE belongs to whatever takes one: a Gaussian covariance is 2n x 2n in
# xpxp ordering, a Fock density matrix cutoff x cutoff.
MatrixLike = Sequence[Sequence[float]] | ArrayLike

Cell = float | bool | str | None

# J s and m/s, exact since the 2019 SI redefinition.
PLANCK = 6.62607015e-34
LIGHT_SPEED = 299792458.0


def q(value: Cell, label: str) -> dict[str, Cell]:
    """
    One explain() row: a resolved value beside where it came from. Labels are
    "pinned", "derived", "default" and "derived (run to compute)". link.py and
    topology.py alias this as _q.
    """

    return {
        "value": value,
        "label": label,
    }


def input_xi(ch: "Channel") -> float:
    """
    A pinned q.Channel's excess noise at the CHANNEL INPUT plane, in SNU.
    ref='output' names Bob's side and divides by the channel's own T; ref=None
    and ref='input' are already there. Takes a q.Channel alone -- a q.Fiber has
    no xi and no ref to read.
    """

    if ch.ref == "output":
        return ch.xi / ch.T

    return ch.xi


def finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, not {value}")


def positive(name: str, value: float) -> None:
    """
    Refuse a value at or below zero. NaN PASSES: finiteness is finite()'s
    question, kept separate so a caller may admit +inf without admitting nan.
    """

    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def nonneg(name: str, value: float) -> None:
    """
    Refuse a negative value; zero is admitted. NaN passes, as in positive().
    """

    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def atleast(name: str, value: float, floor: float) -> None:
    """
    Refuse a value under `floor`. NaN and +inf PASS; nan is caught by the
    engine that consumes it.
    """

    if value < floor:
        raise ValueError(f"{name} must be at least {floor:g}")


def unit(name: str, value: float, hi: float = 1.0) -> None:
    """
    Refuse a value outside (0, hi]: a transmittance, a quantum efficiency, a
    reconciliation efficiency. NaN is refused here, unlike positive().
    """

    if not 0.0 < value <= hi:
        raise ValueError(f"{name} must be in (0, {hi:g}]")


def open_unit(name: str, value: float, hi: float = 1.0) -> None:
    """
    Refuse a value outside (0, hi), BOTH ends barred: a failure probability, a
    secrecy or correctness parameter, the share of a block disclosed for
    parameter estimation. An epsilon of 0 claims a bound that never fails, an
    epsilon of 1 claims nothing, and a pe_fraction of 1 leaves no key. NaN is
    refused, as in unit() and fraction() and unlike positive().
    """

    if not 0.0 < value < hi:
        raise ValueError(f"{name} must be in (0, {hi:g})")


def fraction(name: str, value: float, hi: float = 1.0) -> None:
    """
    Refuse a value outside [0, hi): a probability barred from its top end, a
    per-gate dark count, a monitoring fraction. NaN refused.
    """

    if not 0.0 <= value < hi:
        raise ValueError(f"{name} must be in [0, {hi:g})")


def closed(name: str, value: float, hi: float = 1.0) -> None:
    """
    Refuse a value outside [0, hi], both ends admitted: a fringe contrast, a
    misalignment against the 0.5 of chance, a frame error rate. NaN refused.
    """

    if not 0.0 <= value <= hi:
        raise ValueError(f"{name} must be in [0, {hi:g}]")


def entropy(e: float) -> float:
    """
    Binary Shannon entropy h(e) in BITS, log base 2. e <= 0 and e >= 1 return
    0.0 rather than raising -- callers reach this with an error rate an
    estimator or a bound has already saturated. NaN comes back as nan.
    """

    if e <= 0.0 or e >= 1.0:
        return 0.0

    return -e * math.log2(e) - (1.0 - e) * math.log2(1.0 - e)


def dark_prob(rate: float, gate: float) -> float:
    """
    A free-running dark COUNT RATE `rate` in Hz over a gate of width `gate` in
    seconds to the per-gate PROBABILITY every `dark` field here is written in:
    1 - exp(-rate*gate), the Poisson complement the threshold click law is
    written against.

    THE GATE WIDTH IS NOT THE SYMBOL PERIOD. A gated receiver opens for far
    less than a clock cycle -- 100 ps of a 1 ns period is a 10% duty cycle --
    so 1/symbol_rate overstates the floor by that ratio. A datasheet rate in Hz
    is measured FREE-RUNNING.
    """
    nonneg("rate", rate)
    positive("gate", gate)
    finite("rate", rate)
    finite("gate", gate)

    return -math.expm1(-rate * gate)


def from_db(db: float) -> float:
    """
    An INSERTION LOSS in decibels to the transmittance that survives it,
    10^(-db/10): 0.25 dB is T = 0.944, not 1.059.

    THIS IS NOT THE GENERIC dB-TO-LINEAR CONVERSION. A relative intensity noise
    in dBc/Hz, a launch power in dBm and a Raman density in dBm/nm are all
    10^(+db/10), and two of the three carry a unit conversion as well; those
    sites are spelled out where they stand.
    """

    return 10.0 ** (-db / 10.0)


def to_db(t: float) -> float:
    """
    The inverse of from_db: a transmittance to the INSERTION LOSS in decibels
    that produced it, -10 log10(t), so t in (0, 1] gives a nonnegative loss.
    The loss convention, not the generic linear-to-dB one; a caller wanting a
    gain in dB wants the negative of this. Zero raises ValueError from
    math.log10 rather than returning +inf.
    """

    return -10.0 * math.log10(t)


def grid(rows: Sequence[Sequence[str]]) -> list[str]:
    """
    The house aligned text table: a sequence of equal-length string tuples to a
    list of LINES, each column padded to its widest cell, two spaces between
    columns, trailing padding stripped. Reproduced verbatim in
    docs/guide/budget.md, so the two-space separator and the rstrip() are
    pinned rather than cosmetic. A short row raises IndexError from the width
    scan rather than being padded.
    """

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for r in rows:
        cells = [r[i].ljust(widths[i]) for i in range(len(r))]
        lines.append("  ".join(cells).rstrip())

    return lines


class Barred:
    """
    A result object that refuses a named set of attributes with one sentence
    saying why. Mixed into the frozen dataclasses that do NOT carry a key rate
    -- attacks.Reading, topology.Route, topology.NetworkResult.

    Each subclass supplies its own `_barred` and `_refusal`, deliberately NOT
    pooled: attacks._BARRED is enumerated by name in docs/architecture.md and
    docs/guide/security.md, and topology.py's two messages refuse different
    things. A name absent from `_barred` raises the plain AttributeError(name)
    hasattr() and the copy protocol expect.
    """

    _barred: tuple[str, ...] = ()
    _refusal: str = ""

    def __getattr__(self, name: str) -> NoReturn:
        if name in self._barred:
            raise AttributeError(self._refusal)

        raise AttributeError(name)
