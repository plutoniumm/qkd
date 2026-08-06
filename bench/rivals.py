import os
import sys

sys.path.append(os.environ.get("QUTIP_PATH", ""))

import numpy as np

from MDB import Suite, Trial, load, stamp

from qkd import _core

# QUTIP_PATH is appended to sys.path, not prepended, so qkd keeps its numpy:
# pip install --target /tmp/rivals qutip; QUTIP_PATH=/tmp/rivals ./do bench rivals
# Only cost is comparable: qkd x = (a + a^dag)/sqrt(2), QuTiP alpha = (x+iy)/2.

CUT = 40

GRID = 101

# Kept as a bare ndarray on purpose: qutip.wigner/qfunc below take AXIS directly,
# where fock.py's axis() and memory.py's grid() return .tolist() because every
# caller there wants a list. A shared helper would need a shape parameter for the
# one site here that wants the raw array (self.ax = AXIS.tolist() covers the rest).
AXIS = np.linspace(-6.0, 6.0, GRID)

try:
    import qutip
except ImportError:
    qutip = None


class Fock(Trial):
    warmup = 2
    runs = 5

    def setup(self):
        self.ax = AXIS.tolist()
        self.st = _core.FockState.coherent(1.0, 0.5, CUT)
        self.big = _core.FockState.coherent(1.0, 0.5, 100)
        if qutip is None:
            return None

        self.rho = qutip.ket2dm(qutip.coherent(CUT, 0.5 + 0.25j))
        self.rho2 = qutip.ket2dm(qutip.coherent(100, 0.5 + 0.25j))

        return None

    def bench_wigner(self):
        """
        qkd: Wigner of a cutoff-40 state on 101x101, rayon over rows.
        """
        self.st.wigner(self.ax, self.ax)

        return (GRID * GRID, "point")

    def bench_qwigner(self):
        """
        QuTiP: the same grid and cutoff, Clenshaw recurrence over numpy arrays.
        """
        qutip.wigner(self.rho, AXIS, AXIS)

        return (GRID * GRID, "point")

    def bench_wig100(self):
        """
        qkd: the same grid at cutoff 100.
        """
        self.big.wigner(self.ax, self.ax)

        return (GRID * GRID, "point")

    def bench_qwig100(self):
        """
        QuTiP: the same grid at cutoff 100.
        """
        qutip.wigner(self.rho2, AXIS, AXIS)

        return (GRID * GRID, "point")

    def bench_husimi(self):
        """
        qkd: Husimi Q of a cutoff-40 state on 101x101.
        """
        self.st.husimi(self.ax, self.ax)

        return (GRID * GRID, "point")

    def bench_qfunc(self):
        """
        QuTiP: qfunc on the same grid and cutoff.
        """
        qutip.qfunc(self.rho, AXIS, AXIS)

        return (GRID * GRID, "point")

    def bench_entropy(self):
        """
        qkd: von Neumann entropy at cutoff 40, by hand-written cyclic Jacobi.
        """
        for _ in range(100):
            self.st.entropy()

        return (100, "call")

    def bench_qentropy(self):
        """
        QuTiP: the same entropy, by LAPACK eigh through scipy.
        """
        for _ in range(100):
            qutip.entropy_vn(self.rho, base=2)

        return (100, "call")

    def bench_coherent(self):
        """
        qkd: build a coherent state's density matrix at cutoff 40.
        """
        for _ in range(1000):
            _core.FockState.coherent(1.0, 0.5, CUT)

        return (1000, "call")

    def bench_qcoherent(self):
        """
        QuTiP: build the same density matrix at cutoff 40.
        """
        for _ in range(1000):
            qutip.ket2dm(qutip.coherent(CUT, 0.5 + 0.25j))

        return (1000, "call")


class Absent(Trial):
    warmup = 0
    runs = 1

    def bench_absent(self):
        """
        QuTiP was not importable, so no comparison was made: see the header of
        bench/rivals.py for the install recipe.
        """

        return "blocked: qutip not installed"


if __name__ == "__main__":
    meta = stamp(qutip=getattr(qutip, "__version__", "absent"), numpy=np.__version__)
    sys.exit(
        Suite(
            "Rivals",
            "qkd's Fock layer against QuTiP's on the three operations both "
            "libraries implement. Values are not compared, only cost",
            "rivals.md",
            meta=meta,
        ).run(load(Fock if qutip is not None else Absent))
    )
