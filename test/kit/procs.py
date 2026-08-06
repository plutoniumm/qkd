import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def threads(child, n):
    """
    Run ``child`` in a fresh interpreter with rayon pinned to n threads, returning its
    printed fields. rayon reads RAYON_NUM_THREADS once per process.
    """

    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(n)
    done = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=env,
    )

    return done.stdout.split()
