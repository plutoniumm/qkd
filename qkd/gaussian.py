from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

from ._core import GaussianState
from .std import ArrayLike, MatrixLike

# Purity has no cancellation, so a pure state lands within a few ulp of 1.
_PURE_TOL = 1e-9


_OMEGA: dict[int, np.ndarray] = {}


def _omega(n: int) -> np.ndarray:
    # Symplectic form in xpxp ordering. The cached array is shared across all
    # callers and is read-only.
    cached = _OMEGA.get(n)
    if cached is None:
        block = np.array([[0.0, 1.0], [-1.0, 0.0]])
        cached = np.kron(np.eye(n), block)
        cached.flags.writeable = False
        _OMEGA[n] = cached

    return cached


def _expm(a: MatrixLike) -> np.ndarray:
    # Scaling and squaring with a Taylor series; scipy is not a dependency.
    a = np.asarray(a, dtype=np.float64)
    norm = float(np.linalg.norm(a, np.inf))
    k = max(0, int(np.ceil(np.log2(max(norm, 1e-16)))) + 4)
    m = a / 2.0**k
    out = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for i in range(1, 18):
        term = term @ m / i
        out = out + term
    for _ in range(k):
        out = out @ out

    return out


def _gauss(mean: np.ndarray, v: np.ndarray, xs: ArrayLike, ps: ArrayLike) -> np.ndarray:
    # Normalized Gaussian density, out[i, j] taken at (xs[j], ps[i]).
    inv = np.linalg.inv(v)
    dx = np.asarray(xs, dtype=np.float64)[None, :] - mean[0]
    dp = np.asarray(ps, dtype=np.float64)[:, None] - mean[1]
    quad = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dp + inv[1, 1] * dp * dp

    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(np.linalg.det(v)))


def _pair(a: "State", b: "State") -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(b, State):
        raise TypeError(f"expected a State, got {type(b).__name__}")
    if a.n_modes != b.n_modes:
        raise ValueError(f"comparing states needs equal mode counts, got {a.n_modes} and {b.n_modes}")

    return a.mean - b.mean, a.cov + b.cov


def _shift(delta: np.ndarray, total: np.ndarray) -> float:
    # The unit convention cancels here; it lives in the determinant prefactors.
    quad = float(delta @ np.linalg.solve(total, delta))

    return float(np.exp(-0.5 * quad))


def _native(mean: ArrayLike, cov: MatrixLike) -> "State":
    # Back through the core constructor, which checks the bona fide condition.
    flat = np.asarray(mean, dtype=np.float64).reshape(-1).tolist()
    block = np.asarray(cov, dtype=np.float64).reshape(-1).tolist()

    return State(GaussianState.from_moments(flat, block))


class State:
    """
    Immutable n-mode Gaussian state over the native core; every operation
    returns a new State (xpxp ordering, hbar = 1, vacuum variance 1/2).
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: GaussianState) -> None:
        self._raw = raw

    @property
    def n_modes(self) -> int:
        return self._raw.n_modes

    @property
    def mean(self) -> np.ndarray:
        # A fresh buffer per call, aliasing no state, so it is left writable.
        return np.asarray(self._raw.mean(), dtype=np.float64)

    @property
    def cov(self) -> np.ndarray:
        # The core hands the covariance back flat; the reshape is a view.
        n = 2 * self._raw.n_modes

        return np.asarray(self._raw.cov(), dtype=np.float64).reshape(n, n)

    def displace(self, mode: int, x: float = 0.0, p: float = 0.0) -> "State":
        return State(self._raw.displace(mode, x, p))

    def squeeze(self, mode: int, r: float) -> "State":
        return State(self._raw.squeeze(mode, r))

    def rotate(self, mode: int, theta: float) -> "State":
        return State(self._raw.rotate(mode, theta))

    def bs(self, m1: int, m2: int, t: float) -> "State":
        return State(self._raw.bs(m1, m2, t))

    def thermal_loss(self, mode: int, T: float, xi: float = 0.0, *, ref: Literal["input", "output"]) -> "State":
        """
        Thermal-loss channel on one mode (`bs` against vacuum is pure loss).
        xi is in SNU with the vacuum at 1, not this class's own 1/2, so the
        engine adds xi/2. ref names its plane and is required even at xi = 0:
        'input' adds T*xi/2, 'output' adds xi/2.
        """
        if ref not in ("input", "output"):
            raise ValueError("ref must be 'input' or 'output'")

        return State(self._raw.thermal_loss(mode, T, xi, ref == "input"))

    def condition(self, mode: int, angle: float, outcome: float) -> "State":
        return State(self._raw.condition(mode, angle, outcome))

    def homodyne(self, mode: int, angle: float = 0.0, shots: int = 1, seed: int = 0) -> np.ndarray:
        return self._raw.homodyne(mode, angle, shots, seed)

    def heterodyne(self, mode: int, shots: int = 1, seed: int = 0) -> np.ndarray:
        flat = self._raw.heterodyne(mode, shots, seed)

        return flat.reshape(shots, 2)

    def physical(self, atol: float = 1e-9) -> bool:
        v = self.cov
        if not np.allclose(v, v.T, atol=atol):
            return False

        # Bona fide condition: V + i*Omega/2 >= 0 (Hermitian, so eigvalsh).
        eigs = np.linalg.eigvalsh(v + 0.5j * _omega(self.n_modes))

        return bool(eigs.min() >= -atol)

    def wigner(self, mode: int, xs: ArrayLike, ps: ArrayLike) -> np.ndarray:
        """
        Wigner of one mode's reduced state, shaped (len(ps), len(xs)) so
        W[i, j] = W(xs[j], ps[i]).
        """
        sub = self.keep([mode])

        return _gauss(sub.mean, sub.cov, xs, ps)

    def husimi(self, mode: int, xs: ArrayLike, ps: ArrayLike) -> np.ndarray:
        """
        Husimi Q of one mode's reduced state, over V + I/2. Normalized in
        dx dp, so 0 <= Q <= 1/(2*pi); the looser 1/pi is the d^2 alpha form,
        the factor 2 being that measure's Jacobian.
        """
        sub = self.keep([mode])

        return _gauss(sub.mean, sub.cov + 0.5 * np.eye(2), xs, ps)

    def _reduce(self, modes: Sequence[int]) -> "State":
        # keep() with the checks already made: drop()'s complement is in range
        # and duplicate-free by construction.
        idx = [q for m in modes for q in (2 * m, 2 * m + 1)]

        return _native(self.mean[idx], self.cov[np.ix_(idx, idx)])

    def keep(self, modes: Iterable[int]) -> "State":
        """
        Partial trace down to the listed modes, in the order requested.
        """
        modes = [int(m) for m in modes]
        if not modes:
            raise ValueError("keep needs at least one mode")
        if len(set(modes)) != len(modes):
            raise ValueError(f"duplicate modes in {modes}")
        for m in modes:
            if not 0 <= m < self.n_modes:
                raise ValueError(f"mode {m} out of range for {self.n_modes} modes")

        return self._reduce(modes)

    def drop(self, modes: Iterable[int]) -> "State":
        """
        Partial trace over the listed modes: the complement, order preserved.
        """
        gone = {int(m) for m in modes}
        for m in gone:
            if not 0 <= m < self.n_modes:
                raise ValueError(f"mode {m} out of range for {self.n_modes} modes")
        left = [m for m in range(self.n_modes) if m not in gone]
        if not left:
            raise ValueError("keep needs at least one mode")

        return self._reduce(left)

    def evolve(self, G: MatrixLike, t: float) -> "State":
        """
        Unitary evolution for time t under H = (1/2) R^T G R, G symmetric in
        xpxp ordering.
        """
        g = np.asarray(G, dtype=np.float64)
        d = 2 * self.n_modes
        if g.shape != (d, d):
            raise ValueError(f"G must be {d}x{d}, got {g.shape}")
        if not np.allclose(g, g.T, atol=1e-12):
            raise ValueError("G must be symmetric")
        s = _expm(_omega(self.n_modes) @ g * t)

        return _native(s @ self.mean, s @ self.cov @ s.T)

    def purity(self) -> float:
        """
        Purity 1 / (2^n sqrt(det V)) in internal units: 1 for a pure state.
        """
        n = self.n_modes

        return float(1.0 / (2.0**n * np.sqrt(np.linalg.det(self.cov))))

    def entropy(self) -> float:
        """
        Von Neumann entropy in bits: each mode contributes
        G((nu_snu - 1) / 2), nu_snu = 2 nu the SNU symplectic eigenvalue and
        G(x) = (x + 1) log2(x + 1) - x log2(x).
        """
        total = 0.0
        for nu in self.spectrum():
            x = max(2.0 * nu - 1.0, 0.0) / 2.0
            if x > 0.0:
                total += (x + 1.0) * np.log2(x + 1.0) - x * np.log2(x)

        return float(total)

    def spectrum(self) -> np.ndarray:
        """
        Symplectic eigenvalues in internal units, vacuum 0.5: moduli of the
        eigenvalues of i Omega V, ascending. Raises where the stray imaginary
        part exceeds the eigensolve's backward error, 1e-9 * max(1, max|V|).
        Treat r > 8 on an Epr state as unusable whether or not it raises.
        """
        v = self.cov
        eigs = np.linalg.eigvals(1j * _omega(self.n_modes) @ v)
        stray = float(np.max(np.abs(eigs.imag)))
        tol = 1e-9 * max(1.0, float(np.max(np.abs(v))))
        if stray > tol:
            raise ValueError(f"covariance is not bona fide (imag part {stray:g} against {tol:g})")
        nus = np.sort(np.abs(eigs.real))

        return 0.5 * (nus[::2] + nus[1::2])

    def negativity(self) -> float:
        """
        Wigner-negativity volume, identically 0.0 for a Gaussian state by
        Hudson's theorem.
        """
        return 0.0

    def overlap(self, other: "State") -> float:
        """
        Hilbert-Schmidt overlap Tr(rho sigma), exact at every n.
        """
        delta, total = _pair(self, other)

        return float(_shift(delta, total) / np.sqrt(np.linalg.det(total)))

    def fidelity(self, other: "State") -> float:
        """
        Fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2, SQUARED, so
        F = |<psi|phi>|^2 on a pure pair. Closed form in the moments at one
        mode for any Gaussian state; at n > 1 exact only where one side is pure
        and F collapses to the overlap, a mixed pair raising.
        """
        delta, total = _pair(self, other)
        shift = _shift(delta, total)
        if self.n_modes == 1:
            # Written for vacuum variance 1, so both 2x2 determinants lift by
            # 4. Rationalised against cancellation.
            big = 4.0 * float(np.linalg.det(total))
            lam = (4.0 * float(np.linalg.det(self.cov)) - 1.0) * (4.0 * float(np.linalg.det(other.cov)) - 1.0)
            root = np.sqrt(big + max(lam, 0.0)) + np.sqrt(max(lam, 0.0))

            return float(2.0 * root / big * shift)

        gap = min(abs(self.purity() - 1.0), abs(other.purity() - 1.0))
        if gap <= _PURE_TOL:
            # A pure side makes the product rank one, so F = Tr(rho sigma).
            return float(shift / np.sqrt(np.linalg.det(total)))

        raise NotImplementedError(
            f"no closed-form fidelity for a mixed {self.n_modes}-mode pair. Compare one "
            "mode at a time with keep(), or put a pure state on one side, where "
            "F = Tr(rho sigma) is exact"
        )

    def trace_distance(self, other: "State") -> float:
        """
        Trace distance D = (1/2) ||rho - sigma||_1, returned only where it is
        exact: both states pure, where D = sqrt(1 - F). A mixed pair raises;
        trace_bounds() gives the two-sided bracket instead.
        """
        delta, total = _pair(self, other)
        gap = max(abs(self.purity() - 1.0), abs(other.purity() - 1.0))
        if gap > _PURE_TOL:
            raise NotImplementedError(
                "trace distance is exact here only for a pure pair (purities differ "
                f"from 1 by up to {gap:.3e}). Use trace_bounds() for the bracket"
            )

        # Two pure states put F = Tr(rho sigma) at any n. Spelt out rather than
        # calling overlap() so _pair runs FIRST, which is this method's
        # TypeError/ValueError contract.
        f = _shift(delta, total) / np.sqrt(np.linalg.det(total))

        return float(np.sqrt(max(1.0 - f, 0.0)))

    def trace_bounds(self, other: "State") -> tuple[float, float]:
        """
        1 - sqrt(F) <= D <= sqrt(1 - F) as (lower, upper). Holds for every
        pair; the upper end is an equality on a pure one.
        """
        f = self.fidelity(other)
        lower = max(1.0 - np.sqrt(max(f, 0.0)), 0.0)
        upper = np.sqrt(max(1.0 - f, 0.0))

        return (float(lower), float(upper))


def Vacuum(n: int) -> State:
    return State(GaussianState.vacuum(n))


def Coherent(x: float, p: float) -> State:
    return Vacuum(1).displace(0, x, p)


def Thermal(nbar: float) -> State:
    return State(GaussianState.thermal(nbar))


def Squeezed(r: float) -> State:
    return Vacuum(1).squeeze(0, r)


def Epr(r: float) -> State:
    return State(GaussianState.epr(r))


def Moments(mean: ArrayLike, cov: MatrixLike) -> State:
    """
    The Gaussian state on the given moments, xpxp ordering and vacuum variance
    1/2. The bona fide condition is checked on the way in.
    """

    return _native(mean, cov)
