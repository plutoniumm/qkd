#!/usr/bin/env python3
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .lint_checks import check_py

ROOT = Path(__file__).parent.parent

# The same trees `do`'s PY_DIRS hands to black and compileall.
PY_FILES = (
    sorted(ROOT.glob("qkd/**/*.py"))
    + sorted(ROOT.glob("test/*.py"))
    + sorted(ROOT.glob("test/kit/*.py"))
    + sorted(ROOT.glob("view/*.py"))
    + sorted(ROOT.glob("ci/*.py"))
    + sorted(ROOT.glob("bench/*.py"))
)

if __name__ == "__main__":
    fixes = "--fix" in sys.argv
    args = [(p, fixes) for p in PY_FILES]

    fork = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=fork) as ex:
        results = list(ex.map(check_py, args))

    all_errors: list[str] = []
    for errs, msgs in results:
        for m in msgs:
            print(m)
        all_errors.extend(errs)

    if all_errors:
        for e in sorted(all_errors):
            print(e)
        sys.exit(1)

    print(f"ok  {len(PY_FILES)} py")
