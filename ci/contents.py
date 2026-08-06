import functools
import pathlib
import sys
import tarfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Directory names a recursive glob must not descend into: a stale .py left in
# __pycache__ would become a required member no artifact ships.
PRUNE = ("__pycache__",)


def relname(path):
    """
    The archive member name for a path inside the source tree.

    Relative to ROOT, not the bare filename, so a subpackage keeps its
    directory rather than flattening onto a same-named module. Archive members
    use forward slashes on every platform, so the separator is joined in rather
    than taken from os.sep.
    """

    return "/".join(path.relative_to(ROOT).parts)


def kept(path):
    """
    Whether a globbed path is a build input rather than generated litter.
    """

    return not any(part in PRUNE for part in path.relative_to(ROOT).parts)


def modules():
    """
    The qkd/ Python an artifact has to carry, subpackages included.

    Recursive because setup.py takes find_packages(include=["qkd",
    "qkd.*"]): a subpackage ships, and a flat glob would let it ship
    unchecked -- a gate that verifies less without ever turning red.
    """

    return sorted(relname(p) for p in (ROOT / "qkd").rglob("*.py") if kept(p))


@functools.cache
def sources():
    """
    The src/ build inputs an sdist has to carry, split by extension.

    An sdist compiles on the installing machine and the shaders arrive by
    include_str!, so a dropped .wgsl breaks that build exactly as a .rs does.
    Two lists rather than one because main() guards each separately: a tree
    holding no shaders at all would pass a combined count. Recursive because
    MANIFEST.in says recursive-include src. Cached: nothing writes to src/
    while this runs.
    """

    src = ROOT / "src"
    rust = tuple(sorted(relname(p) for p in src.rglob("*.rs") if kept(p)))
    wgsl = tuple(sorted(relname(p) for p in src.rglob("*.wgsl") if kept(p)))

    return rust, wgsl


def members(path):
    """
    Member names, with an sdist's leading qkd-<version>/ stripped so both
    archive kinds check against one list of paths.
    """

    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())

    with tarfile.open(path) as archive:
        names = archive.getnames()

    return {n.split("/", 1)[1] for n in names if "/" in n}


def native(path):
    """
    A wheel ships the compiled core under the one name its platform can
    import; an sdist ships the Rust to compile instead.
    """

    if not path.name.endswith(".whl"):
        rust, wgsl = sources()

        return ["Cargo.toml", "Cargo.lock", "ci/smoke.py"] + sorted(rust + wgsl)

    if "win_" in path.name:
        return ["qkd/_core.pyd"]

    return ["qkd/_core.abi3.so"]


def main(paths):
    """
    Fail unless every artifact carries the whole package.
    """

    want = modules()

    if not paths:
        print("!! contents: no artifacts given to check")

        return 1

    if not want:
        print("!! contents: no qkd/*.py in the source tree to check against")

        return 1

    rust, wgsl = sources()

    if not rust:
        print("!! contents: no src/*.rs in the source tree to check against")

        return 1

    if not wgsl:
        print("!! contents: no src/*.wgsl in the source tree to check against")

        return 1

    rc = 0
    for path in paths:
        need = want + native(path)
        have = members(path)
        gone = [n for n in need if n not in have]
        if gone:
            print(f"!! {path.name} is missing {len(gone)}: {' '.join(gone)}")
            rc = 1
        else:
            print(f">> {path.name}: all {len(need)} required members present")

    return rc


if __name__ == "__main__":
    sys.exit(main([pathlib.Path(arg) for arg in sys.argv[1:]]))
