import sys

import numpy as np

from MDB import Suite, Trial, load, spin, stamp, symbols

from qkd import _core, fock

# Grid size and cutoff are swept separately against T = a * points * d^2; a
# hidden O(d^3) shows as a rising ns/point. husimi is the control -- same shape
# and row split, no sqrt and no divide -- so the ratio prices the Laguerre
# recursion. Cases drive `_core.FockState` directly; only bench_window prices
# `qkd.fock`'s shipped default window policy.

CUT = 40

BEAM = 2.0  # odd cat: photon number ~2.07, Wigner-negative

GRID = 101


def axis(n, reach=6.0):
    return np.linspace(-reach, reach, n).tolist()


def cat(d):
    return _core.FockState.cat(BEAM, 0.0, True, d)


class Kernel(Trial):
    warmup = 1
    runs = 5

    def setup(self):
        self.st = {d: cat(d) for d in (10, 20, 40, 80, 120)}
        self.ax = {n: axis(n) for n in (51, 101, 201, 401)}
        self.mid = self.ax[GRID]
        self.flat = self.st[CUT].wigner(self.mid, self.mid)

    def bench_g51(self):
        """
        Wigner on a 51x51 grid at cutoff 40, the smallest grid worth timing.
        """
        self.st[CUT].wigner(self.ax[51], self.ax[51])

        return (51 * 51, "point")

    def bench_g101(self):
        """
        Wigner on a 101x101 grid at cutoff 40.
        """
        self.st[CUT].wigner(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_g201(self):
        """
        Wigner on a 201x201 grid at cutoff 40, four times the points of 101.
        """
        self.st[CUT].wigner(self.ax[201], self.ax[201])

        return (201 * 201, "point")

    def bench_g401(self):
        """
        Wigner on a 401x401 grid at cutoff 40, sixteen times the points of 101.
        """
        self.st[CUT].wigner(self.ax[401], self.ax[401])

        return (401 * 401, "point")

    def bench_split(self):
        """
        The same 160801 points as bench_g401 in sixteen 101x101 calls. The
        kernel is a pure per-point map, so both do identical arithmetic and any
        gap is buffer handback, grid residency or the scheduler.
        """
        for _ in range(16):
            self.st[CUT].wigner(self.mid, self.mid)

        return (16 * GRID * GRID, "point")

    def bench_d10(self):
        """
        Wigner at cutoff 10 on the fixed 101x101 grid.
        """
        self.st[10].wigner(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_d20(self):
        """
        Wigner at cutoff 20 on the fixed 101x101 grid.
        """
        self.st[20].wigner(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_d40(self):
        """
        Wigner at cutoff 40, the shipped default, on the fixed 101x101 grid.
        """
        self.st[40].wigner(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_d80(self):
        """
        Wigner at cutoff 80 on the fixed 101x101 grid: four times the work of 40
        if the kernel is O(d^2), eight times if it is secretly O(d^3).
        """
        self.st[80].wigner(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_d120(self):
        """
        Wigner at cutoff 120 on the fixed 101x101 grid.
        """
        self.st[120].wigner(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_hus20(self):
        """
        Husimi at cutoff 20 on the same grid: the no-sqrt O(d^2) control.
        """
        self.st[20].husimi(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_hus40(self):
        """
        Husimi at cutoff 40; over the Wigner cost at the same cutoff, the ratio
        prices the Laguerre recursion's sqrt/divide.
        """
        self.st[40].husimi(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_hus120(self):
        """
        Husimi at cutoff 120 on the same grid.
        """
        self.st[120].husimi(self.mid, self.mid)

        return (GRID * GRID, "point")

    def bench_neg(self):
        """
        Negativity on a 201x201 grid at cutoff 40: one Wigner grid plus an
        absolute-value sum, so anything above bench_g201 is the quadrature.
        """
        self.st[CUT].negativity(self.ax[201], self.ax[201])

        return (201 * 201, "point")

    def bench_anchor(self):
        """
        THE CONTENTION ANCHOR, not a Fock measurement: 1e6 CV-pipeline symbols,
        10.3 M symbol/s on a quiet machine. Read it first -- absolute rates in
        this table move by whatever factor it is below 10.3, ratios do not.
        """
        symbols(1e6)

        return (1e6, "symbol")

    def bench_marshal(self):
        """
        Copying one 101x101 Wigner result: the grid comes back as a numpy
        array, so the PyO3 boundary costs a memcpy and no unboxing.
        """
        self.flat.copy()

        return (GRID * GRID, "point")


class Spectrum(Trial):
    warmup = 2
    runs = 5

    def setup(self):
        self.st = {d: cat(d) for d in (10, 40, 120)}
        self.rho = self.st[CUT].matrix()

    def bench_eig10(self):
        """
        Jacobi eigenvalues of a 10x10 density matrix.
        """

        return spin(lambda: self.st[10].eigenvalues(), 2000)

    def bench_eig40(self):
        """
        Jacobi eigenvalues at the shipped cutoff of 40.
        """

        return spin(lambda: self.st[40].eigenvalues(), 200)

    def bench_eig120(self):
        """
        Jacobi eigenvalues at cutoff 120: 27x the matrix entries of 40, so a
        cyclic Jacobi sweep should cost 27x if it is O(d^3).
        """

        return spin(lambda: self.st[120].eigenvalues(), 20)

    def bench_ent40(self):
        """
        Von Neumann entropy at cutoff 40, which is the eigensolver plus O(d).
        """

        return spin(lambda: self.st[40].entropy(), 200)

    def bench_nongauss(self):
        """
        Relative-entropy non-Gaussianity at cutoff 40: moments plus one entropy.
        """

        return spin(lambda: self.st[40].nongauss(), 200)

    def bench_physical(self):
        """
        The positive-semidefinite check at cutoff 40: the same eigensolver
        behind a Hermiticity and trace scan.
        """

        return spin(lambda: self.st[40].physical(1e-9), 200)

    def bench_coherent(self):
        """
        Coherent-state constructor at cutoff 40: one O(d) amplitude ladder and
        one O(d^2) outer product.
        """

        return spin(lambda: _core.FockState.coherent(1.0, 0.5, CUT), 2000)

    def bench_cat(self):
        """
        Cat-state constructor at cutoff 40.
        """

        return spin(lambda: cat(CUT), 2000)

    def bench_squeezed(self):
        """
        Squeezed-vacuum constructor at cutoff 40.
        """

        return spin(lambda: _core.FockState.squeezed(0.5, CUT), 2000)

    def bench_thermal(self):
        """
        Thermal-state constructor at cutoff 40, a diagonal not an outer product.
        """

        return spin(lambda: _core.FockState.thermal(0.5, CUT), 2000)

    def bench_gkp(self):
        """
        GKP constructor at its own default of 60 levels: a real-space
        quadrature over a comb, the only constructor that integrates anything.
        """

        return spin(lambda: _core.FockState.gkp(0, 0.3, 60), 20)

    def bench_matrix(self):
        """
        Exporting the 40x40 density matrix to Python as two numpy arrays.
        """

        return spin(lambda: self.st[CUT].matrix(), 200)

    def bench_build(self):
        """
        from_matrix at cutoff 40, which validates positivity too, so it pays
        for one eigensolve on top of the copy.
        """

        return spin(lambda: _core.FockState.from_matrix(*self.rho), 200)

    def bench_rotate(self):
        """
        Phase-space rotation at cutoff 40: O(d^2), with a sin_cos per entry.
        """

        return spin(lambda: self.st[CUT].rotate(0.3), 200)

    def bench_displace(self):
        """
        Displacement at cutoff 40: the Laguerre recursion once, then two dense
        O(d^3) complex matrix products.
        """

        return spin(lambda: self.st[CUT].displace(1.0, 0.5), 100)

    def bench_loss(self):
        """
        Pure-loss channel at cutoff 40, the O(d^3) Kraus sum.
        """

        return spin(lambda: self.st[CUT].loss(0.5), 100)

    def bench_loss120(self):
        """
        The same channel at cutoff 120: 27x slower confirms O(d^3), and much
        worse is a cache wall rather than an exponent.
        """

        return spin(lambda: self.st[120].loss(0.5), 10)

    def bench_moments(self):
        """
        First and second moments at cutoff 40, an O(d) diagonal walk.
        """

        return spin(lambda: self.st[CUT].moments(), 2000)


class Default(Trial):
    warmup = 1
    runs = 3

    def setup(self):
        self.st = {d: fock.Cat(BEAM, 0.0, odd=True, cutoff=d) for d in (10, 20, 40)}
        self.wide = fock.window(self.st[40])
        self.half = fock.window(self.st[40], step=2.0 * fock.STEP)
        self.pts = len(self.wide) ** 2

    def bench_window(self):
        """
        negativity() of an odd cat on the shipped default window and cutoff:
        what state.negativity() with no arguments costs.
        """
        self.st[40].negativity()

        return (self.pts, "point")

    def bench_coarse(self):
        """
        The same negativity at twice the default step, a quarter of the points.
        """
        self.st[40].negativity(self.half, self.half)

        return (len(self.half) ** 2, "point")

    def bench_cut10(self):
        """
        The same default window at cutoff 10 instead of 40.
        """
        self.st[10].negativity(self.wide, self.wide)

        return (self.pts, "point")

    def bench_cut20(self):
        """
        The same default window at cutoff 20 instead of 40.
        """
        self.st[20].negativity(self.wide, self.wide)

        return (self.pts, "point")


class Truth(Trial):
    """
    The Rate column carries the measured value, not a throughput. A squeezed
    vacuum is Gaussian, so Hudson's theorem puts its true negativity volume at
    exactly zero: whatever the grid reports is the truncation artefact.
    """

    warmup = 0
    runs = 1

    def setup(self):
        self.ax = axis(201)
        self.cat = cat(CUT)

    def bench_pure10(self):
        """
        Spurious negativity volume of a squeezed vacuum at cutoff 10 (true 0).
        """
        st = _core.FockState.squeezed(0.5, 10)

        return f"{st.negativity(self.ax, self.ax):+.2e}"

    def bench_pure20(self):
        """
        Spurious negativity volume of a squeezed vacuum at cutoff 20 (true 0).
        """
        st = _core.FockState.squeezed(0.5, 20)

        return f"{st.negativity(self.ax, self.ax):+.2e}"

    def bench_pure40(self):
        """
        Spurious negativity volume of a squeezed vacuum at cutoff 40, the
        shipped default (true 0).
        """
        st = _core.FockState.squeezed(0.5, 40)

        return f"{st.negativity(self.ax, self.ax):+.2e}"

    def bench_pure80(self):
        """
        Spurious negativity volume of a squeezed vacuum at cutoff 80 (true 0).
        """
        st = _core.FockState.squeezed(0.5, 80)

        return f"{st.negativity(self.ax, self.ax):+.2e}"

    def bench_step24(self):
        """
        Odd-cat negativity volume at a step of 0.24, four times the default.
        """

        return f"{self.volume(0.24):.6f}"

    def bench_step12(self):
        """
        Odd-cat negativity volume at a step of 0.12, twice the default.
        """

        return f"{self.volume(0.12):.6f}"

    def bench_step06(self):
        """
        Odd-cat negativity volume at the shipped step of 0.06.
        """

        return f"{self.volume(0.06):.6f}"

    def bench_step03(self):
        """
        Odd-cat negativity volume at a step of 0.03, half the default: what the
        coarser steps are converging towards.
        """

        return f"{self.volume(0.03):.6f}"

    def volume(self, step):
        """
        Negativity of the odd cat on a fixed 10.7-wide window at this step.
        """
        n = 2 * int(10.7 / step) + 1
        ax = axis(n, 10.7)

        return self.cat.negativity(ax, ax)


if __name__ == "__main__":
    meta = stamp()
    rc = 0
    rc |= Suite(
        "Fock kernel",
        "Wigner and Husimi grid cost, swept in grid size and in cutoff "
        "separately, with Husimi as the no-sqrt control",
        "fock_kernel.md",
        meta=meta,
    ).run(load(Kernel))
    rc |= Suite(
        "Fock spectrum",
        "Eigensolver, constructors and channels: the whole gridless half of the Fock layer",
        "fock_spectrum.md",
        meta=meta,
    ).run(load(Spectrum))
    rc |= Suite(
        "Fock defaults",
        "What qkd.fock's shipped window and cutoff cost per call",
        "fock_defaults.md",
        meta=meta,
    ).run(load(Default))
    rc |= Suite(
        "Fock accuracy",
        "What those defaults buy. The Rate column carries the measured negativity volume, not a throughput",
        "fock_truth.md",
        meta=meta,
    ).run(load(Truth))
    sys.exit(rc)
