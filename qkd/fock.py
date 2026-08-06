import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import gaussian
from ._core import FockState
from .std import ArrayLike

CUTOFF = 40

# The core's own level cap; a derived cutoff is clamped to it.
MAX_LEVELS = 512

# gkp_levels numerator, calibrated to discard under GKP_TAIL of the norm.
GKP_SCALE = 8.0

# Norm a derived-cutoff GKP state may discard before Gkp refuses. A pinned
# cutoff is never refused.
GKP_TAIL = 1e-6

# Default depth for tail() and trunc_error().
EDGE = 3

# Default phase-space sample spacing for window().
STEP = 0.06


def gkp_levels(delta: float) -> int:
    """
    Levels a grid state of peak width `delta` needs to keep truncation under
    GKP_TAIL, clamped to MAX_LEVELS. Grows as 1/delta^2.
    """

    return min(MAX_LEVELS, int(math.ceil(GKP_SCALE / (delta * delta))))


def _axes(xs: ArrayLike, ps: ArrayLike) -> tuple[list[float], list[float]]:
    # Plain lists: tolist() measured faster than reaching the core's Vec<f64>
    # through an ndarray's sequence protocol.
    return (
        np.asarray(xs, dtype=np.float64).tolist(),
        np.asarray(ps, dtype=np.float64).tolist(),
    )


def _step(axis: Sequence[float]) -> float:
    # Uniform axis spacing, the same way the core computes it.
    return float((axis[-1] - axis[0]) / (len(axis) - 1))


def window(state: "State", reach: float | None = None, step: float = STEP) -> np.ndarray:
    """
    The default phase-space axis for a quasi-probability grid over `state`.
    reach is the half-width in x and p, None taking five standard deviations of
    the state's own moments, held to the sqrt(2*cutoff) + 5 outside which a
    state on levels 0..cutoff-1 has no support. step = 0.06 is chosen, not
    converged.
    """
    # Built as a half and mirrored: the Wigner kernel shares its inner loop
    # across a radius orbit only where v[i] == -v[n-1-i] bit for bit, and
    # np.linspace(-a, a, n) is antisymmetric only on a dyadic step.
    if reach is None:
        # cov is row-major 2x2, so its diagonal is cov[0] and cov[3].
        mean, cov = state._raw.moments()
        spread = max(abs(mean[i]) + 5.0 * math.sqrt(cov[3 * i]) for i in (0, 1))

        # Five sigma fails where the moments do: a state too big for its cutoff
        # has the moments of the renormalised matrix.
        reach = min(spread, math.sqrt(2.0 * state.cutoff) + 5.0)

    if not reach > 0.0:
        raise ValueError("reach must be positive")

    if not step > 0.0:
        raise ValueError("step must be positive")

    n = 2 * int(math.ceil(reach / step)) + 1
    half = np.linspace(-reach, 0.0, n // 2 + 1)

    return np.concatenate((half, -half[-2::-1]))


@dataclass(frozen=True)
class Volume:
    """
    A Wigner-negativity volume with the grid it was integrated on, span
    (x_lo, x_hi, p_lo, p_hi) and step (dx, dp). trunc_error bounds the STATE,
    not the volume -- truncation moves a volume by about 2*sqrt(discarded).
    Check convergence with `at_cutoff`.
    """

    value: float
    trunc_error: float
    cutoff: int
    points: tuple[int, int]
    span: tuple[float, float, float, float]
    step: tuple[float, float]

    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True)
class Divergence:
    """
    A relative-entropy non-Gaussianity in bits, with the state's truncation
    error and cutoff. Truncation moves both halves of S(shadow) - S(rho), so a
    Gaussian state can score above a cat. Check with `at_cutoff`.
    """

    value: float
    trunc_error: float
    cutoff: int

    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True)
class Purity:
    """
    Tr(rho^2) of the TRUNCATED, renormalised state, the norm discarded, and
    least = kept^2 * value, a lower bound with nothing useful above it. value is
    off by up to 1/kept^2 either way.
    """

    value: float
    discarded: float
    least: float

    def __float__(self) -> float:
        return self.value


class State:
    """
    Immutable single-mode state in the truncated number basis; every operation
    returns a new State. hbar = 1, vacuum variance 1/2, quasi-probabilities
    normalised in dx dp: the vacuum peaks at W = 1/pi and Q = 1/(2 pi).
    """

    # A truncated matrix is renormalised, so it passes every physical check
    # while being quantitatively wrong; only `discarded`, `tail` and
    # `trunc_error` see it. Check convergence with at_cutoff.
    __slots__ = ("_raw", "_make")

    def __init__(self, raw: FockState, make: Callable[[int], FockState] | None = None) -> None:
        self._raw = raw
        self._make = make

    @property
    def cutoff(self) -> int:
        return self._raw.cutoff

    def at_cutoff(self, cutoff: int) -> "State":
        """
        The same state rebuilt in `cutoff` levels: a witness that moves with
        the cutoff was an artefact. A Density() state carries no recipe and
        raises.
        """
        if self._make is None:
            raise ValueError(
                "a state built from an explicit density matrix carries no recipe to rebuild at another cutoff"
            )

        return State(self._make(cutoff), self._make)

    def _chain(self, step: Callable[[FockState], FockState]) -> Callable[[int], FockState] | None:
        make = self._make
        if make is None:
            return None

        return lambda d: step(make(d))

    def rotate(self, theta: float) -> "State":
        """
        Phase-space rotation by theta, counterclockwise. Diagonal, so it adds
        no truncation error.
        """

        return State(self._raw.rotate(theta), self._chain(lambda raw: raw.rotate(theta)))

    def displace(self, x: float = 0.0, p: float = 0.0) -> "State":
        """
        Displacement by (x, p), beta = (x + i p)/sqrt(2). The truncated operator
        is not unitary: the lost norm reaches `discarded`, not a renormalisation.
        """

        return State(self._raw.displace(x, p), self._chain(lambda raw: raw.displace(x, p)))

    def loss(self, eta: float) -> "State":
        """
        Pure-loss channel of transmittance eta. Only moves population down the
        ladder, so it is trace-preserving inside the cutoff and adds no
        truncation error.
        """

        return State(self._raw.loss(eta), self._chain(lambda raw: raw.loss(eta)))

    def matrix(self) -> np.ndarray:
        """
        The density matrix as a complex (cutoff, cutoff) ndarray.
        """
        d = self.cutoff
        re, im = self._raw.matrix()

        return re.reshape(d, d) + 1j * im.reshape(d, d)

    def populations(self) -> np.ndarray:
        # P(n) = rho_nn.
        return self._raw.populations()

    def eigenvalues(self) -> np.ndarray:
        # Ascending.
        return self._raw.eigenvalues()

    def photons(self) -> float:
        return float(self._raw.photons())

    def parity(self) -> float:
        """
        Photon-number parity, exactly pi * W(0, 0). A negative value certifies
        Wigner negativity at the origin without a grid; sufficient, not
        necessary.
        """

        return float(self._raw.parity())

    def purity(self) -> Purity:
        """
        Purity Tr(rho^2): 1 pure, 1/cutoff maximally mixed, but of the
        TRUNCATED, renormalised matrix -- see Purity.
        """
        value = float(self._raw.purity())
        lost = self.discarded()
        kept = 1.0 - lost

        return Purity(value=value, discarded=lost, least=kept * kept * value)

    def entropy(self) -> float:
        # Von Neumann entropy in bits.
        return float(self._raw.entropy())

    def physical(self, tol: float = 1e-9) -> bool:
        """
        Hermitian, unit trace and positive semidefinite to within tol: the
        Fock-space counterpart of the Gaussian bona fide condition.
        """

        return bool(self._raw.physical(tol))

    def discarded(self) -> float:
        # Fraction of the norm this state's construction threw away.
        return float(self._raw.discarded())

    def tail(self, levels: int = EDGE) -> float:
        """
        Population in the top `levels` levels: `discarded` catches a state born
        too big, this one pushed against the ceiling afterwards. levels cannot
        be 1 -- a parity-restricted state reads exactly zero on half its levels.
        """

        return float(self._raw.tail(levels))

    def trunc_error(self) -> float:
        """
        The worse of `discarded` and `tail`. A bound on the STATE, not on any
        witness measured from it -- see Volume.
        """

        return float(self._raw.trunc_error())

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """
        (mean, cov): [<x>, <p>] and the 2x2 covariance, vacuum variance 1/2.
        """
        mean, cov = self._raw.moments()

        return (
            np.array(mean, dtype=np.float64),
            np.array(cov, dtype=np.float64).reshape(2, 2),
        )

    def shadow(self) -> gaussian.State:
        """
        The qkd.gaussian.State on the same moments, which non_gaussianity
        scores against. Its entropy is never below this state's own.
        """
        mean, cov = self.moments()

        return gaussian.Moments(mean, cov)

    def wigner(self, xs: ArrayLike, ps: ArrayLike) -> np.ndarray:
        """
        Wigner function shaped (len(ps), len(xs)), W[i, j] = W(xs[j], ps[i]),
        the Gaussian layer's layout.
        """
        xs, ps = _axes(xs, ps)
        flat = self._raw.wigner(xs, ps)

        return flat.reshape(len(ps), len(xs))

    def husimi(self, xs: ArrayLike, ps: ArrayLike) -> np.ndarray:
        """
        Husimi Q, same layout as `wigner`. Normalised in dx dp, so the vacuum
        peaks at 1/(2 pi), inside the 1/pi bound.
        """
        xs, ps = _axes(xs, ps)
        flat = self._raw.husimi(xs, ps)

        return flat.reshape(len(ps), len(xs))

    def negativity(self, xs: ArrayLike | None = None, ps: ArrayLike | None = None) -> Volume:
        """
        Wigner-negativity volume, the integral of |W| minus 1, as a Volume.
        Strictly positive for |n> with n >= 1, odd cats and GKP; zero for every
        Gaussian state. Both axes default to `window(self)`.
        """
        xs = window(self) if xs is None else xs
        ps = xs if ps is None else ps
        xs, ps = _axes(xs, ps)
        value = self._raw.negativity(xs, ps)

        return Volume(
            value=float(value),
            trunc_error=self.trunc_error(),
            cutoff=self.cutoff,
            points=(len(xs), len(ps)),
            span=(float(xs[0]), float(xs[-1]), float(ps[0]), float(ps[-1])),
            step=(_step(xs), _step(ps)),
        )

    def non_gaussianity(self) -> Divergence:
        """
        Relative-entropy non-Gaussianity in bits, S(shadow) - S(rho), as a
        Divergence. Weaker than `negativity`: a mixture of two coherent states
        scores here and not there. Non-negative, so a negative return is noise.
        """

        return Divergence(
            value=float(self._raw.nongauss()),
            trunc_error=self.trunc_error(),
            cutoff=self.cutoff,
        )


def Vacuum(*, cutoff: int = CUTOFF) -> State:
    return Number(0, cutoff=cutoff)


def Number(n: int, *, cutoff: int = CUTOFF) -> State:
    """
    The number state |n>.
    """

    return State(FockState.fock(n, cutoff), lambda d: FockState.fock(n, d))


def Coherent(x: float, p: float, *, cutoff: int = CUTOFF) -> State:
    """
    The coherent state at (x, p), beta = (x + i p)/sqrt(2). Its Poissonian tail
    reaches every level, so `discarded` is non-zero.
    """

    return State(FockState.coherent(x, p, cutoff), lambda d: FockState.coherent(x, p, d))


def Cat(x: float, p: float, odd: bool = False, *, cutoff: int = CUTOFF) -> State:
    """
    The even (odd = False) or odd superposition of the coherent states at
    +-(x, p). Each occupies one parity class, so the odd cat's parity is
    exactly -1 and its W(0, 0) exactly -1/pi at any lobe size.
    """

    return State(FockState.cat(x, p, odd, cutoff), lambda d: FockState.cat(x, p, odd, d))


def Squeezed(r: float, *, cutoff: int = CUTOFF) -> State:
    """
    The squeezed vacuum, x squeezed for r > 0, matching gaussian.Squeezed(r).
    Gaussian, so its true negativity is zero at every r.
    """

    return State(FockState.squeezed(r, cutoff), lambda d: FockState.squeezed(r, d))


def Thermal(nbar: float, *, cutoff: int = CUTOFF) -> State:
    """
    The thermal state of mean photon number nbar: mixed and Gaussian.
    """

    return State(FockState.thermal(nbar, cutoff), lambda d: FockState.thermal(nbar, d))


def Gkp(logical: int, delta: float, *, cutoff: int | None = None) -> State:
    """
    An approximate finite-energy grid state, logical = 0 or 1. delta is the
    peak width in vacuum units, so the squeezing is -20 log10(delta) dB.
    cutoff = None derives it from gkp_levels and raises below delta ~ 0.12; a
    pinned cutoff is honoured as given.
    """
    if cutoff is not None:
        return State(
            FockState.gkp(logical, delta, cutoff),
            lambda d: FockState.gkp(logical, delta, d),
        )

    levels = gkp_levels(delta)
    st = State(
        FockState.gkp(logical, delta, levels),
        lambda d: FockState.gkp(logical, delta, d),
    )
    if st.discarded() > GKP_TAIL:
        raise ValueError(
            f"a grid state of delta = {delta} needs more than the {MAX_LEVELS} levels "
            f"the core allows: at {levels} it discards {st.discarded():.2e} of its "
            f"norm, over GKP_TAIL = {GKP_TAIL:g}. Pin cutoff= to take it anyway"
        )

    return st


def Density(rho: Sequence[Sequence[complex]] | np.ndarray) -> State:
    """
    A state from an explicit square complex density matrix, validated for
    shape, finiteness, Hermiticity, unit trace and positive semidefiniteness.
    `discarded` is 0 and `at_cutoff` raises.
    """
    m = np.asarray(rho, dtype=np.complex128)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"rho must be a square matrix, got shape {m.shape}")

    re = m.real.reshape(-1).tolist()
    im = m.imag.reshape(-1).tolist()

    return State(FockState.from_matrix(re, im))
