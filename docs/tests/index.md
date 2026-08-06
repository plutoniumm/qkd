# Testing

Tests are standalone Python scripts, not a `pytest` tree. Each builds an `Exam`,
runs a `unittest` suite through it, and writes one markdown report page. The
harness is `test/MDR.py`; the continuous-variable assertion helpers are on
`Question`.

## Running

```sh
./do test
```

| | |
| --- | --- |
| `./do test` | compiles the Rust core in place, runs every `test/*.py` except `MDR.py`, and fails if any exam fails. One report page per `Exam` |
| The build it uses | `./do develop`, unoptimised and roughly an order of magnitude slower than the release build `./do bench` makes and the wheels carry. A report's duration is a debug-build figure |
| `MDR_OUT` | read once at import in `test/MDR.py`; default `test/_reports` (gitignored) |
| Reports under `docs/tests/` | gitignored; only this page is checked in. A fresh checkout has no report pages, the sidebar lists whichever are present, and `ignoreDeadLinks` covers `/tests/`, so a link to a missing report does not fail the build |
| `./do lint --fix` on `test/MDR.py` | **do not**: its docstrings are copied verbatim into the report tables, so reformatting ships. `./do lint` exempts the file |

Publish the reports into this site:

```sh
MDR_OUT="$(pwd)/docs/tests" ./do test
```

## Report format

A `# Name` heading, the exam description, a pass count and duration, then one row
per test:

| Test | What it does | Result |
| ---- | ------------ | ------ |
| `test_vacuum_is_bona_fide` | Vacuum covariance I/2 is physical and saturates the uncertainty bound. | ✅ pass |

The "What it does" column is the test method's docstring, whitespace-collapsed and
escaped so that a cell containing `|0>` does not read as a column break.

| Marker | Meaning |
| --- | --- |
| ✅ pass | assertion suite completed |
| ❌ fail | an assertion failed |
| ⚠️ error | an exception other than an assertion escaped |
| ⏭️ skip | `skipTest`, e.g. a GPU-only test on a machine with no adapter |

Failures and errors also get a `## Failures` section with the formatted exception
for each, so a published report is self-contained.

## Writing an exam

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MDR import Exam, Question, load

class Channels(Question):
    def test_thermal_loss_stays_physical(self):
        """
        A thermal-loss channel maps bona fide covariances to bona fide covariances.
        """
        ...

if __name__ == "__main__":
    sys.exit(Exam("Channels", "Gaussian channels", "channels.md").run(load(Channels)))
```

| Piece | Is |
| --- | --- |
| `load` | `unittest.defaultTestLoader.loadTestsFromTestCase` |
| `Exam.run` | a shell exit code; a file with several exams ORs them together |
| Each exam | one report file and one page under `/tests/`, listed in the sidebar by filename |

## CV assertion helpers

`Question` extends `unittest.TestCase` with assertions about phase space.

| Helper | Checks | Default tolerance |
| --- | --- | --- |
| `assertClose(got, want)` | scalar equality, $\lvert got - want \rvert \le \text{atol}$ | `1e-13` |
| `assertPhysical(V)` | $V$ symmetric **and** $V + \tfrac{i}{2}\Omega \succeq 0$ — the bona fide condition | `1e-12` |
| `assertSymplectic(S)` | $S\,\Omega\,S^{\!\top} = \Omega$ | `1e-12` |
| `assertUncertainty(V)` | per mode, $\Delta x^2 \Delta p^2 \ge \tfrac14$ | `1e-12` |
| `assertNormalized(W, dx, dp)` | $\iint W\,dx\,dp = 1$ on a discretised grid | `1e-8` |
| `assertHusimi(Q)` | $0 \le Q \le 1/(2\pi)$ pointwise, the $dx\,dp$ bound rather than the $d^2\alpha$ one | `1e-12` |
| `gridClose(got, want)` | two phase-space grids agree pointwise (max abs error) | `1e-9` |
| `samplesClose(samples, expected)` | empirical outcome frequencies match a probability map; for PNR and sign-binned homodyne | `0.03` |
| `momentsClose(samples, mean, var)` | continuous samples (e.g. homodyne quadratures) match first and second moments | `0.05` |

`omega(n)` in `MDR.py` builds the symplectic form for $n$ modes in $xpxp$ ordering;
`assertPhysical` and `assertSymplectic` are written against it.

| Helper | Exercised over |
| --- | --- |
| `assertPhysical` | every Gaussian constructor and channel image |
| `gridClose`, `assertNormalized` | the Wigner and Husimi grids of both state layers |
| `momentsClose`, `samplesClose` | the homodyne, heterodyne and click samplers |

## Why `assertPhysical` is the important one

A covariance matrix can be symmetric and positive-definite and describe no physical
state ([the bona fide condition](/guide/conventions#the-bona-fide-condition)). A
thermal-loss channel that mixes noise in at the
[wrong plane](/guide/conventions#excess-noise-carries-a-plane) still returns a
symmetric, positive-definite matrix. Every channel and symplectic map runs through
`assertPhysical`, and a negative control keeps the check from being vacuous:

```python
def test_unphysical_covariance_is_rejected(self):
    V = np.eye(2) * 0.1  # below vacuum in both quadratures
    with self.assertRaises(AssertionError):
        self.assertPhysical(V, msg="sub-vacuum covariance must be rejected")
```

## Conventions are tested, not documented

The phase-space convention is global, so no local test would catch a change to it.
The [`Conventions` exam](/tests/conventions) pins [the contract](/guide/conventions).

## Current exams

One source file per layer, one exam per question about it; each report opens with
its exam's description. `grep -n 'Exam(' test/*.py` maps reports back to source
files.

Three entries check the harness or the toolchain rather than the physics.

| Exam | Checks |
| --- | --- |
| `test_version` in `Smoke` | `Cargo.toml`'s `[package]` version against `qkd.__version__`, which comes through PyO3 from `CARGO_PKG_VERSION`; a stale compiled `_core` fails at once |
| `test_absent_gpu_is_not_an_error` in `GPU fallback` | skips rather than fails on a machine that has an adapter; the skip shows as ⏭️ in the report |
| `ApiScope` | the *refusals*: every configuration this tree cannot compute must raise, naming the restriction |
