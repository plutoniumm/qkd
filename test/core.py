import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.procs import ROOT

import numpy as np

import qkd

# Cargo.toml is the source of truth; nothing else reads the other two.
SITES = {
    "Cargo.toml": r'version\s*=\s*"([^"]+)"',
    "setup.py": r'version\s*=\s*"([^"]+)"',
    "docs/package.json": r'"version"\s*:\s*"([^"]+)"',
}


def declared(root, name):
    """
    The version one release file declares, or None. Cargo.toml is read from its [package] table
    alone: every dependency below it declares one too.
    """
    text = open(os.path.join(root, *name.split("/"))).read()
    if name == "Cargo.toml":
        text = text.split("[package]", 1)[1].split("\n[", 1)[0]

    found = re.search(SITES[name], text)

    return None if found is None else found.group(1)


class Smoke(Question):
    def test_version(self):
        """
        `qkd.__version__` comes from the Rust crate and agrees with all three declaration sites,
        so a two-of-three bump fails here rather than shipping.
        """
        expected = declared(ROOT, "Cargo.toml")
        self.assertIsNotNone(expected, msg="Cargo.toml [package] declares no version")
        self.assertEqual(
            qkd.__version__,
            expected,
            msg=f"__version__ must match Cargo.toml ({expected})",
        )

        for name in SITES:
            got = declared(ROOT, name)
            self.assertEqual(
                got,
                expected,
                msg=f"{name} declares {got!r}, Cargo.toml {expected!r}",
            )

    def test_backend_resolves(self):
        """
        A compute backend resolves without error and reports name/precision/device.
        """
        name, precision, device = qkd.backend_info()
        self.assertIn(name, ("cpu", "gpu"), msg=f"unexpected backend {name!r}")
        self.assertIn(precision, ("f32", "f64"), msg=f"bad precision {precision!r}")
        self.assertTrue(device, msg="backend must describe its device")

    def test_supports_tracks_precision(self):
        """
        `supports()` agrees with the reported precision, and f32 is always available.
        """
        _, precision, _ = qkd.backend_info()
        self.assertTrue(qkd.supports("f32"), msg="f32 must always be available")
        self.assertEqual(
            qkd.supports("f64"),
            precision == "f64",
            msg="supports('f64') must track the reported precision",
        )

    def test_supports_rejects_unknown_precision(self):
        """
        `supports()` raises on anything that is not f32/f64.
        """
        with self.assertRaises(ValueError):
            qkd.supports("f16")


class Conventions(Question):
    """
    Phase-space conventions: hbar = 1, vacuum variance 1/2, bona fide.
    """

    def test_vacuum_is_bona_fide(self):
        """
        Vacuum covariance $I/2$ is bona fide and saturates $\\Delta x^2 \\Delta p^2 = 1/4$.
        """
        V = np.eye(2) * 0.5
        self.assertPhysical(V, msg="vacuum must be a valid covariance matrix")
        self.assertUncertainty(V, msg="vacuum must respect the uncertainty bound")
        self.assertClose(V[0, 0] * V[1, 1], 0.25, msg="vacuum must saturate dx^2 dp^2 = 1/4")

    def test_unphysical_covariance_is_rejected(self):
        """
        A sub-vacuum covariance fails the bona fide check, so `assertPhysical` is not vacuously
        true.
        """
        V = np.eye(2) * 0.1

        with self.assertRaises(AssertionError):
            self.assertPhysical(V, msg="sub-vacuum covariance must be rejected")


if __name__ == "__main__":
    rc = Exam(
        "Smoke",
        "Build, import, and compute-backend resolution",
        "smoke.md",
    ).run(load(Smoke))
    rc |= Exam(
        "Conventions",
        "Phase-space conventions: hbar=1, vacuum variance 1/2, bona fide covariances",
        "conventions.md",
    ).run(load(Conventions))
    sys.exit(rc)
