"""
Markdown-reporting test harness, adapted from qliff (originally qudit). The
Exam/Question/_Reporter machinery is carried over unchanged; the assertion
helpers are continuous-variable ones.

NB: `./do lint` exempts this file, and `./do lint --fix` must never be run
against it -- reformatting here ships. Unrelated to the 2026-08-13 autofixer
bug, which `view/lint_test.py` pins.

Usage (bottom of a test file)::
    if __name__ == "__main__":
        import sys
        sys.exit(Exam("Gaussian", "Symplectic covariance engine", "gaussian.md")
                 .run(load(MyQuestion)))

Auditing the tolerances these helpers carry
-------------------------------------------
Walk `test/*.py` (excluding this file) for calls to the nine helpers below --
assertClose, assertPhysical, assertSymplectic, assertUncertainty,
assertNormalized, assertHusimi, gridClose, samplesClose, momentsClose --
recording the tolerance and whether it is explicit or inherited. Monkey-patch
them to record the error at every call, then:

  usage  = observed max error / tolerance
  margin = (atol - mean_err) / sd_err   over seeds 1..24, under 5 sigma is open

Three rules:

1. Key by CALL SITE (file:line), never by test; a site otherwise inherits the
   maximum of any looser site sharing its test.
2. Mutate BOTH ways. At k=2 the assertion must fail in both directions.
3. Sigma is the sd of the observed ERROR, not of the value; the two differ by
   ~1.4x. Where a test's own comment states the rule it was set by, honour that.

Land where margin >= 5 sigma and usage > 0.5. Plain assertGreater/assertLess
calls sit outside the walk; magnitude claims against a round literal want the
same treatment, ordering claims between two computed quantities do not.
"""

from __future__ import annotations

import os
import time
import traceback
import unittest

OUT_DIR = os.environ.get("MDR_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_reports"))

load = unittest.defaultTestLoader.loadTestsFromTestCase


def omega(n):
    """
    Symplectic form Omega for n modes, in (x1,p1,...,xn,pn) ordering.
    """
    import numpy as np

    w = np.zeros((2 * n, 2 * n))
    for i in range(n):
        w[2 * i, 2 * i + 1] = 1.0
        w[2 * i + 1, 2 * i] = -1.0

    return w


class Question(unittest.TestCase):
    """
    Base test case with continuous-variable assertion helpers.

    Conventions follow docs/guide/conventions.md: hbar = 1, x = (a + a-dagger)/sqrt(2), so
    the vacuum has Delta x^2 = Delta p^2 = 1/2, and Omega is the standard symplectic form in
    xpxp ordering.
    """

    def assertClose(self, got, want, atol=1e-13, msg=None):
        """
        Assert a scalar matches within an absolute tolerance.
        """
        self.assertLessEqual(
            abs(float(got) - float(want)),
            atol,
            msg=msg or f"{got} != {want} (atol={atol})",
        )

    def assertPhysical(self, V, atol=1e-12, msg=None):
        """
        Covariance matrix satisfies the bona fide condition V + i*Omega/2 >= 0.

        A covariance matrix can be symmetric, positive-definite and still unphysical.
        """
        import numpy as np

        V = np.asarray(V, dtype=float)
        n = V.shape[0] // 2
        self.assertEqual(V.shape, (2 * n, 2 * n), msg="covariance must be 2n x 2n")

        asym = np.max(np.abs(V - V.T))
        self.assertLessEqual(asym, atol, msg=msg or f"covariance not symmetric ({asym:g})")

        eigs = np.linalg.eigvalsh(V + 0.5j * omega(n))
        self.assertGreaterEqual(
            float(eigs.min()),
            -atol,
            msg=msg or f"V + i*Omega/2 has eigenvalue {eigs.min():.3e} < 0 (unphysical)",
        )

    def assertSymplectic(self, S, atol=1e-12, msg=None):
        """
        Assert S is symplectic: S Omega S^T = Omega.
        """
        import numpy as np

        S = np.asarray(S, dtype=float)
        n = S.shape[0] // 2
        w = omega(n)
        err = np.max(np.abs(S @ w @ S.T - w))
        self.assertLessEqual(err, atol, msg=msg or f"S Omega S^T != Omega (max err {err:.3e})")

    def assertUncertainty(self, V, atol=1e-12, msg=None):
        """
        Every mode's marginal obeys Delta x^2 * Delta p^2 >= 1/4.
        """
        import numpy as np

        V = np.asarray(V, dtype=float)
        for i in range(V.shape[0] // 2):
            vx, vp = V[2 * i, 2 * i], V[2 * i + 1, 2 * i + 1]
            self.assertGreaterEqual(
                vx * vp,
                0.25 - atol,
                msg=msg or f"mode {i}: dx^2 dp^2 = {vx * vp:.6f} < 1/4",
            )

    def assertNormalized(self, W, dx, dp, atol=1e-8, msg=None):
        """
        Quasi-probability grid integrates to 1 over phase space.
        """
        import numpy as np

        total = float(np.asarray(W, dtype=float).sum() * dx * dp)
        self.assertLessEqual(
            abs(total - 1.0),
            atol,
            msg=msg or f"integral = {total:.6f} != 1 (atol={atol})",
        )

    def assertHusimi(self, Q, atol=1e-12, msg=None):
        """
        Husimi Q is strictly non-negative and bounded above by 1/(2 pi).

        1/pi under d^2 alpha, 1/(2 pi) under the dx dp normalisation these grids carry, the
        factor 2 being the Jacobian. Attained exactly, so atol covers round-off alone.
        """
        import numpy as np

        Q = np.asarray(Q, dtype=float)
        self.assertGreaterEqual(float(Q.min()), -atol, msg=msg or f"Q has negative value {Q.min():.3e}")
        self.assertLessEqual(
            float(Q.max()),
            0.5 / np.pi + atol,
            msg=msg or f"Q exceeds 1/(2 pi) ({Q.max():.6f})",
        )

    def gridClose(self, got, want, atol=1e-9, msg=None):
        """
        Two phase-space grids agree pointwise.
        """
        import numpy as np

        got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
        self.assertEqual(got.shape, want.shape, msg="grid shapes differ")

        err = float(np.max(np.abs(got - want)))
        self.assertLessEqual(err, atol, msg=msg or f"max pointwise err {err:.3e}")

    def samplesClose(self, samples, expected, atol=0.03, msg=None):
        """
        Empirical frequencies of ``samples`` match ``expected`` within ``atol``.

        ``samples`` is an iterable of outcomes (tuples/ints); ``expected`` maps an outcome
        key to its probability.
        """
        from collections import Counter

        samples = [tuple(s) if hasattr(s, "__iter__") else (s,) for s in samples]
        n = len(samples)
        self.assertGreater(n, 0, msg="no samples")

        freq = Counter(samples)
        for key, p in expected.items():
            k = tuple(key) if hasattr(key, "__iter__") else (key,)
            emp = freq.get(k, 0) / n
            self.assertLessEqual(
                abs(emp - p),
                atol,
                msg=msg or f"P{k}={emp:.3f} != {p:.3f} (atol={atol}, n={n})",
            )

    def momentsClose(self, samples, mean, var, atol=0.05, msg=None):
        """
        Continuous samples (e.g. homodyne quadratures) match mean and variance.
        """
        import numpy as np

        s = np.asarray(samples, dtype=float)
        self.assertGreater(s.size, 0, msg="no samples")
        self.assertLessEqual(
            abs(float(s.mean()) - mean),
            atol,
            msg=msg or f"mean {s.mean():.4f} != {mean:.4f}",
        )
        self.assertLessEqual(
            abs(float(s.var()) - var),
            atol,
            msg=msg or f"var {s.var():.4f} != {var:.4f}",
        )

    def assertFinite(self, values, msg=None):
        """
        Every component of a scalar or tuple result is a finite float.

        A cancellation or overflow yielding NaN or +-inf still looks like a rate.
        """
        import math

        items = list(values) if hasattr(values, "__iter__") else [values]
        self.assertGreater(len(items), 0, msg="no values")

        for i, v in enumerate(items):
            self.assertTrue(
                math.isfinite(float(v)),
                msg=msg or f"component {i} is {v}, not finite",
            )

    def assertMonotone(self, seq, rising=True, strict=True, msg=None):
        """
        A numeric sequence moves monotonically in the stated direction.

        ``rising`` picks increasing over decreasing, ``strict`` picks > over >=.
        """
        vals = [float(v) for v in seq]
        self.assertGreater(len(vals), 1, msg="need at least two points")

        for i in range(len(vals) - 1):
            lo, hi = vals[i], vals[i + 1]
            step = hi - lo
            ok = step > 0.0 if strict else step >= 0.0
            if not rising:
                ok = step < 0.0 if strict else step <= 0.0
            self.assertTrue(ok, msg=msg or f"index {i}: {lo:g} -> {hi:g} is not monotone")

    def assertFails(self, exc, needle, fn, *args, msg=None):
        """
        Calling ``fn(*args)`` raises ``exc`` and its message mentions ``needle``.

        Checking the type alone passes on the wrong error.
        """
        with self.assertRaises(exc) as caught:
            fn(*args)

        text = str(caught.exception)
        self.assertIn(needle, text, msg=msg or f"{exc.__name__}({text!r}) lacks {needle!r}")


class _Reporter(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.records = []  # (method, doc, status, detail)

    @staticmethod
    def _doc(test):
        return (test._testMethodDoc or "").strip()

    @staticmethod
    def _exc(err):
        return "".join(traceback.format_exception_only(err[0], err[1])).strip()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append((test._testMethodName, self._doc(test), "pass", ""))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append((test._testMethodName, self._doc(test), "fail", self._exc(err)))

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append((test._testMethodName, self._doc(test), "error", self._exc(err)))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append((test._testMethodName, self._doc(test), "skip", reason))


class Exam:
    """
    A named group of questions that runs to a markdown report + console summary.
    """

    def __init__(self, name, desc, file):
        self.name = name
        self.desc = desc
        os.makedirs(OUT_DIR, exist_ok=True)
        self.file = os.path.join(OUT_DIR, file)

    def run(self, suite) -> int:
        result = _Reporter()
        t0 = time.perf_counter()
        suite(result)
        dt = time.perf_counter() - t0
        self._write(result, dt)

        npass = sum(1 for r in result.records if r[2] == "pass")
        ntot = len(result.records)
        ok = result.wasSuccessful()
        print(
            f"{'PASS' if ok else 'FAIL'}  {self.name}: {npass}/{ntot} in {dt * 1000:.0f}ms"
            f"  -> {os.path.relpath(self.file)}"
        )
        if not ok:
            for nm, _doc, st, detail in result.records:
                if st in ("fail", "error"):
                    print(f"      {st.upper()} {nm}: {detail}")

        return 0 if ok else 1

    def _write(self, result, dt):
        emoji = {
            "pass": "✅ pass",
            "fail": "❌ fail",
            "error": "⚠️ error",
            "skip": "⏭️ skip",
        }
        npass = sum(1 for r in result.records if r[2] == "pass")
        lines = [
            f"# {self.name}",
            "",
            self.desc,
            "",
            f"**{npass}/{len(result.records)} passed** in {dt * 1000:.0f}ms",
            "",
            "| Test | What it does | Result |",
            "| ---- | ------------ | ------ |",
        ]
        for nm, doc, st, _detail in result.records:
            what = " ".join(doc.split())
            what = (what.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")) or "—"
            lines.append(f"| `{nm}` | {what} | {emoji.get(st, '?')} |")

        failures = [(nm, detail) for nm, _doc, st, detail in result.records if st in ("fail", "error") and detail]
        if failures:
            lines += ["", "## Failures", ""]
            for nm, detail in failures:
                lines += [f"### `{nm}`", "", "```", detail, "```", ""]

        with open(self.file, "w") as f:
            f.write("\n".join(lines))
