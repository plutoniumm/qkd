#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# scratch copy of the project handed to the Linux build containers (see
# stage_project), dropped on every exit path including the failing ones.
STAGE=""
trap 'if [ -n "$STAGE" ]; then rm -rf "$STAGE"; fi' EXIT

# every python tree black/compileall touch. src/ is Rust, docs/ belongs to npm.
PY_DIRS=(qkd test view ci bench)

# black at 88 exploded 2,650 assertion calls across 3.1 lines each -- the msg=
# every camelCase assert must carry does not fit beside its arguments. At 120,
# 1,656 of them are one line again. No pyproject.toml: setup.py's setup_requires
# is the build path and a PEP 517 file would change it.
PY_COLS=120

has() {
  for c in "$@"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      echo "missing required tool: $c" >&2
      exit 1
    fi
  done
}

ensure_build_deps() {
  if ! python -c "import setuptools_rust" >/dev/null 2>&1; then
    pip install -q -U setuptools setuptools-rust wheel
  fi
}

tok() {
  local file="./.vscode/token.env"

  if [ ! -f "$file" ]; then
    echo "token file not found: $file" >&2
    exit 1
  fi

  cat "$file"
}

# Resolve script selectors to absolute paths in SELECTED (a global, because a
# `$(...)` subshell would swallow the hard error below).
#
#   select_files test MDR.py                -> every test/*.py but the harness
#   select_files test MDR.py gaussian       -> test/gaussian.py
#
# `gaussian`, `gaussian.py` and `test/gaussian.py` all name the same file. An
# unmatched selector is a HARD error listing what is available: silently running
# nothing is exactly how `./do bench` rotted into a no-op.
SELECTED=()

select_files() {
  local dir="$1" harness="$2"
  shift 2
  local f name

  SELECTED=()

  if [ "$#" -eq 0 ]; then
    for f in "$ROOT/$dir"/*.py; do
      if [ ! -e "$f" ]; then
        continue
      fi
      if [ -n "$harness" ] && [ "$(basename "$f")" = "$harness" ]; then
        continue
      fi
      SELECTED+=("$f")
    done

    return 0
  fi

  for name in "$@"; do
    name="$(basename "$name")"
    f="$ROOT/$dir/${name%.py}.py"
    if [ ! -e "$f" ]; then
      echo "no such file: $dir/${name%.py}.py" >&2
      echo "available in $dir/: $(ls "$ROOT/$dir"/*.py 2>/dev/null \
        | xargs -n1 basename | tr '\n' ' ')" >&2
      exit 1
    fi
    SELECTED+=("$f")
  done
}

# Run each script as its own process, then print a summary naming every file
# that failed. `soft` mode reports failures without failing the build; `hard`
# propagates. An empty file list is a failure in hard mode, never a silent pass.
run_files() {
  local label="$1" mode="$2"
  shift 2
  local total=$# rc=0 f base
  local failed=()

  if [ "$total" -eq 0 ]; then
    echo "!! nothing to run in $label/ -- no *.py files matched" >&2
    if [ "$mode" = soft ]; then
      return 0
    fi

    return 1
  fi

  for f in "$@"; do
    base="$(basename "$f")"
    echo "-- $base"
    if ! python "$f"; then
      failed+=("$base")
      if [ "$mode" != soft ]; then
        rc=1
      fi
    fi
  done

  echo "" >&2
  if [ "${#failed[@]}" -eq 0 ]; then
    echo ">> $label: $total/$total file(s) ok" >&2
  else
    echo "!! $label: ${#failed[@]}/$total file(s) FAILED -> ${failed[*]}" >&2
  fi

  return $rc
}

develop() {
  has python cargo rustc
  ensure_build_deps
  cd "$ROOT"
  python setup.py build_ext --inplace
}

# Fast feedback loop: type/borrow-check the Rust core and byte-compile every
# python tree, WITHOUT linking the extension module. `cargo check` skips
# codegen and linking entirely, so this answers "does it compile" in a fraction
# of `develop`'s time; it cannot produce an importable _core, so it is a
# complement to `develop`, not a replacement.
check() {
  has python cargo
  cd "$ROOT"

  echo ">> cargo check (no codegen, no link) ..." >&2
  cargo check --lib

  echo ">> python syntax ..." >&2
  python -m compileall -q "${PY_DIRS[@]}"
  echo ">> check ok" >&2
}

# compile the core in RELEASE (optimized), for benchmarking.
# develop/build_ext produce DEBUG builds (~10-90x slower) -- never bench those.
# -Ctarget-cpu=native tunes to THIS machine (SIMD, scheduling); kept out of
# Cargo.toml so the PyPI wheel stays portable. lto/codegen-units/opt-level=3
# come from [profile.release] in Cargo.toml.
build_release() {
  has python cargo rustc
  cd "$ROOT"

  cargo rustc --release --lib --manifest-path Cargo.toml \
    --features pyo3/extension-module --crate-type cdylib -- \
    -Ctarget-cpu=native \
    -Clink-arg=-undefined \
    -Clink-arg=dynamic_lookup \
    -Clink-arg=-Wl,-install_name,@rpath/_core.abi3.so

  cp target/release/lib_core.dylib qkd/_core.abi3.so
}

# build EVERYTHING: all cp311-abi3 wheels (macOS + Linux + Windows, x86_64 and
# arm in each) into wheelhouse/ plus the sdist into dist/, WITHOUT uploading --
# the same build as `deploy`, stopping before twine. (Linux wheels need a
# container engine; colima is started on demand.)
build() {
  build_local_wheels
  echo ">> wheels in $ROOT/wheelhouse (not uploaded). sdist in $ROOT/dist." >&2
}

# MDR report tables for the docs site land in docs/tests by running:
#   MDR_OUT=$PWD/docs/tests ./do test
test_() {
  # resolve selectors BEFORE compiling, so a typo fails in milliseconds.
  select_files test MDR.py "$@"
  local files=("${SELECTED[@]+"${SELECTED[@]}"}")
  develop
  run_files test hard "${files[@]+"${files[@]}"}"
}

# base/ is an optional gitignored scratch dir for local-only benchmarks, and
# never fails the build.
#
# NB: build_release overwrites qkd/_core.abi3.so with the optimized build,
# so run `./do develop` before going back to `./do test`.
bench() {
  local rc=0

  select_files bench MDB.py "$@"
  local files=("${SELECTED[@]+"${SELECTED[@]}"}")
  build_release
  run_files bench hard "${files[@]+"${files[@]}"}" || rc=1

  if [ "$#" -eq 0 ] && [ -d "$ROOT/base" ]; then
    select_files base ""
    if [ "${#SELECTED[@]}" -gt 0 ]; then
      run_files base soft "${SELECTED[@]}" || true
    fi
  fi

  return $rc
}

docs() {
  has npm
  cd "$ROOT/docs"
  [ -d node_modules ] || npm install
  case "${1:-dev}" in
    build)  npm run build ;;
    deploy) npm run build && npm run deploy ;;
    *)      npm run dev ;;
  esac
}

# Drop generated artifacts. The default is the cheap, safe set (python bytecode,
# wheels, sdists, generated reports). `--all` additionally removes the cargo
# target dir and the compiled extension, which costs a full rebuild of the wgpu
# dependency graph -- do not do that casually.
clean() {
  cd "$ROOT"

  rm -rf build dist wheelhouse ./*.egg-info
  rm -rf test/_reports bench/_results
  find "${PY_DIRS[@]}" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

  if [ "${1:-}" = "--all" ]; then
    rm -rf target docs/.vitepress/cache docs/.vitepress/dist
    rm -f qkd/_core*.so qkd/_core*.pyd qkd/_core*.dylib
    echo ">> cleaned (incl. target/ and the compiled core -- ./do develop to rebuild)" >&2

    return
  fi

  echo ">> cleaned build/dist/wheelhouse/reports/__pycache__ (use --all for target/)" >&2
}

# Formatting only, and python only. `lint --fix` additionally runs view.lint's
# AST autofixer, which is the part with teeth (and the part that must never see
# test/MDR.py); this verb just normalises whitespace and can be run at any time.
# Rust is deliberately NOT formatted here: src/ is not rustfmt-normalised, so a
# `cargo fmt` would rewrite the whole core in one commit. eastwood, not rustfmt,
# is the Rust gate -- run `cargo fmt` by hand if that ever changes.
fmt() {
  has python
  cd "$ROOT"
  python -m black -l "$PY_COLS" -q "${PY_DIRS[@]}"
  echo ">> formatted: ${PY_DIRS[*]}" >&2
}

# --- cross-platform wheels (built locally, shipped straight to PyPI) --------
# `build` makes cp311-abi3 wheels for macOS (arm64 + x86_64), Linux
# (x86_64 + aarch64) AND Windows (x86_64 + arm64) right here; `deploy`
# twine-uploads whatever `build` produced (it never rebuilds). No GitHub, no
# CI. macOS wheels build directly with this Python
# (cibuildwheel needs a python.org framework build that isn't installed); Linux
# wheels build in manylinux containers via cibuildwheel + colima; Windows wheels
# are cross-compiled with cargo-xwin (xwin fetches the MSVC CRT + Windows SDK,
# lld does the linking, and pyo3 >= 0.29 binds python3.dll via raw-dylib, so no
# import lib is generated). One abi3 wheel per platform covers every Python
# >= 3.11 (tag set by setup.py's bdist_wheel options). No pyproject.toml.
#
# NB: the wgpu GPU backend is compiled in by default. It needs no vendor SDK at
# build time -- that is exactly why it was chosen over CUDA -- so every target
# below still cross-compiles. It resolves an adapter at RUNTIME and falls back
# to CPU when none answers, which is the normal case inside the build containers.

ensure_cibw() {
  python -c "import cibuildwheel" >/dev/null 2>&1 || pip install -q -U cibuildwheel
}

# give rustup a default toolchain + the x86_64 macOS cross target (idempotent).
ensure_rust() {
  has rustup
  rustup show active-toolchain >/dev/null 2>&1 || rustup default stable
  rustup target list --installed 2>/dev/null | grep -q '^x86_64-apple-darwin$' \
    || rustup target add x86_64-apple-darwin
}

# cargo-xwin + the windows-msvc rust target for the Windows cross build
# (idempotent). xwin downloads the MSVC CRT + Windows SDK into its cache on
# first use; Microsoft's license applies to that download.
#
# x86_64 ONLY, deliberately. A win_arm64 wheel cannot be run, cross-compiled or
# otherwise, by anything in this build chain, and an untested wheel is worse than
# no wheel: pip prefers it over the sdist, so a broken one removes the fallback
# that would otherwise have worked. Windows-on-ARM installs from the sdist.
ensure_xwin() {
  rustup target list --installed 2>/dev/null | grep -q '^x86_64-pc-windows-msvc$' \
    || rustup target add x86_64-pc-windows-msvc
  command -v cargo-xwin >/dev/null 2>&1 || cargo install --locked cargo-xwin
}

# bring the local container engine (colima) up if it isn't; needed for the Linux
# wheels. Returns non-zero when no engine is available.
#
# --vz-rosetta runs the x86_64 container on Rosetta 2 instead of qemu, and that
# is a correctness setting here, not a speed one. Under qemu user-mode emulation
# this build failed three separate ways on 2026-08-10: rust-lld took SIGSEGV
# linking a build script (twice, different crates), and once the core compiled
# and `pip install` of the finished wheel segfaulted with code 139. None of it
# was reproducible in the same place twice. Rosetta runs the same containers
# without any of it, and takes the x86_64 wheel from ~22 min to a few.
#
# NB: this only applies to a colima that `do` starts. An already-running VM is
# used as it is -- `colima stop && colima start --vz-rosetta ...` by hand, or
# expect the qemu failure modes above.
ensure_container() {
  docker info >/dev/null 2>&1 && return 0
  command -v colima >/dev/null 2>&1 || return 1
  echo ">> starting colima (Linux container engine) ..." >&2
  colima start --cpu 4 --memory 8 --vm-type vz --vz-rosetta >&2 || return 1
  COLIMA_STARTED=1
  docker info >/dev/null 2>&1
}

# Stop the VM iff ensure_container started it this run. A colima that was
# already up belongs to whoever started it -- stopping that one would take a
# 4-CPU/8-GB VM out from under another build with no way to tell it happened.
release_container() {
  [ "${COLIMA_STARTED:-0}" -eq 1 ] || return 0
  echo ">> stopping colima (started by this build) ..." >&2
  colima stop >&2 || echo "!! colima stop failed -- VM left running" >&2
  COLIMA_STARTED=0
}

# Hand the Linux containers a QUIET copy of the project, and prove it landed.
#
# cibuildwheel ships the project in with
#     tar -cf - . | docker exec -i <container> tar -xf -
# and its check=True inspects the pipeline's status, which is the container
# side's -- so a failure of the host tar is invisible to it. The host tar fails
# whenever a directory changes underneath it, and build/ and target/ change
# constantly during a build because the macOS and Windows wheels are being
# produced in this same tree at the same time. libarchive's response is to stop
# descending into every directory it has not reached yet, so /project arrives
# holding setup.py, Cargo.toml and src/ but an EMPTY qkd/. find_packages()
# then returns nothing, build_py never runs, and the wheel ships as a bare
# _core.abi3.so -- which still installs, and still imports, because python
# makes the leftover directory an implicit namespace package. Every attribute
# lookup on it fails. That was the 2026-08-09 linux-aarch64 failure, and the
# only reason it was caught is that ci/smoke.py touches an attribute.
#
# Copying from a directory nothing else is writing removes the race outright.
# It is also ~500x faster than copying the live tree, because that is almost
# entirely target/, which the container never reads -- it compiles into its own.
stage_project() {
  local dst="$1" f

  rm -rf "$dst"
  mkdir -p "$dst"

  tar -cf - -C "$ROOT" \
    --exclude ./.git \
    --exclude ./build \
    --exclude ./dist \
    --exclude ./target \
    --exclude ./wheelhouse \
    --exclude ./docs/node_modules \
    --exclude ./docs/.vitepress/cache \
    --exclude ./docs/.vitepress/dist \
    --exclude './qkd/_core.*' \
    --exclude '*.egg-info' \
    --exclude __pycache__ \
    . | tar -xf - -C "$dst"

  # this copy is the exact step that failed silently, so check what arrived
  # instead of trusting an exit code that has already been shown to lie.
  for f in setup.py MANIFEST.in Cargo.toml Cargo.lock README.md LICENSE \
    qkd/__init__.py src/lib.rs ci/smoke.py; do
    if [ ! -f "$dst/$f" ]; then
      echo "!! staged project copy is missing $f -- refusing to build from it" >&2
      exit 1
    fi
  done
}

# cibuildwheel config for the Linux wheels (all via CIBW_*; no pyproject.toml).
# Local builds skip musllinux (slow under x86 emulation) by default.
cibw_env() {
  export CIBW_BUILD="cp311-*"
  export CIBW_SKIP="*-musllinux* *-win32 *_i686 pp*"
  export CIBW_TEST_REQUIRES="numpy"
  export CIBW_TEST_SOURCES="ci/smoke.py"
  export CIBW_TEST_COMMAND="python ci/smoke.py"
  export CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28"
  export CIBW_MANYLINUX_AARCH64_IMAGE="manylinux_2_28"
  # the manylinux containers ship no Rust; install it before each build.
  export CIBW_BEFORE_ALL_LINUX="curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable"
  export CIBW_ENVIRONMENT_LINUX='PATH="$HOME/.cargo/bin:$PATH" PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1'

  # x86_64 ONLY, and appended at the call site rather than exported here.
  # rust-lld is rustc's default linker on x86_64-unknown-linux-gnu, and under
  # qemu user-mode emulation it segfaults intermittently -- twice on 2026-08-10,
  # on two different crates' build scripts, each time as `linking with cc failed:
  # signal: 11 (SIGSEGV) (core dumped)`. -Clinker-features=-lld drops the
  # -fuse-ld=lld / gcc-ld shim so the link goes through binutils ld.bfd, which
  # emulates cleanly. The flag is stable only on the target that defaults to
  # rust-lld: aarch64 rustc rejects it outright without -Zunstable-options, so it
  # must not go in the shared environment above. Do not substitute
  # -Clink-self-contained=no -- that also drops the self-contained crt objects
  # and nothing compiles at all.
  NO_LLD='RUSTFLAGS="-C linker-features=-lld"'
}

# build the two macOS wheels directly with this Python. arm64 is native; x86_64
# is a cross-compile (CARGO_BUILD_TARGET + ARCHFLAGS + the host-platform tag
# override). Sequential -- both passes share the build/ tree.
build_macos_wheels() {
  export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
  export MACOSX_DEPLOYMENT_TARGET=11.0

  rm -rf build
  _PYTHON_HOST_PLATFORM=macosx-11.0-arm64 \
    python setup.py bdist_wheel -d wheelhouse

  rm -rf build
  ARCHFLAGS="-arch x86_64" CARGO_BUILD_TARGET=x86_64-apple-darwin \
    _PYTHON_HOST_PLATFORM=macosx-11.0-x86_64 \
    python setup.py bdist_wheel -d wheelhouse
}

# setuptools-rust names the cross-built extension with the HOST suffix
# (_core.abi3.so); Windows CPython only imports `.pyd`. Repack the wheel with
# the binary renamed -- `wheel pack` recomputes RECORD hashes.
rename_ext_in_wheel() {
  local whl="$1" tmp dir
  tmp="$(mktemp -d)"
  python -m wheel unpack "$whl" -d "$tmp"
  dir="$(echo "$tmp"/qkd-*)"
  mv "$dir/qkd/_core.abi3.so" "$dir/qkd/_core.pyd"
  rm "$whl"
  python -m wheel pack "$dir" -d "$ROOT/wheelhouse"
  rm -rf "$tmp"
}

# cross-compile the two Windows wheels on macOS. The CARGO shim (ci/cargo-xwin.sh)
# routes compile commands through `cargo xwin`; _PYTHON_HOST_PLATFORM forges the
# wheel tag the same way the macOS x86_64 cross build does. Nothing here can RUN
# the result -- smoke-test on a real Windows box if in doubt.
build_windows_wheels() {
  export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

  rm -rf build
  CARGO="$ROOT/ci/cargo-xwin.sh" CARGO_BUILD_TARGET=x86_64-pc-windows-msvc \
    _PYTHON_HOST_PLATFORM=win-amd64 \
    python setup.py bdist_wheel -d wheelhouse
  rename_ext_in_wheel wheelhouse/qkd-*-win_amd64.whl
}

# The release matrix, and the ONE place it is written down. `build` refuses to
# produce an sdist unless every one of these exists, and `deploy` refuses to
# upload unless every one of these exists -- because `deploy` never rebuilds, so
# a short wheelhouse would ship silently and every user on a missing platform
# would fall back to compiling from source with no error anywhere. That is
# exactly what happened on 2026-08-09 when colima was wedged: three of five
# wheels, exit 0, and nothing said so.
#
# win_arm64 is deliberately absent -- see RELEASING.md.
WHEEL_MATRIX=(
  macosx_11_0_arm64
  macosx_11_0_x86_64
  manylinux_2_28_aarch64
  manylinux_2_28_x86_64
  win_amd64
)

# Fail unless wheelhouse/ holds one wheel per WHEEL_MATRIX entry. $1 names the
# caller, so the message says which verb refused.
check_matrix() {
  local who="$1" tag missing=()

  for tag in "${WHEEL_MATRIX[@]}"; do
    ls wheelhouse/*"$tag"*.whl >/dev/null 2>&1 || missing+=("$tag")
  done

  if [ "${#missing[@]}" -ne 0 ]; then
    echo "!! $who: wheelhouse/ is short of the release matrix" >&2
    echo "!! missing: ${missing[*]}" >&2
    echo "!! present:" >&2
    ls -1 wheelhouse/*.whl 2>/dev/null | sed 's|^|!!   |' >&2
    echo "!! Linux wheels need a container engine: run 'colima start' and check" >&2
    echo "!! 'docker info' succeeds, then re-run ./do build." >&2
    exit 1
  fi
}

# check_matrix counts the artifacts; this one reads inside them. It is the only
# gate the macOS and Windows wheels have at all -- `build` never runs those, and
# cannot: nothing here can execute a win_amd64 or a macOS x86_64 binary. The
# failure it exists for is silent by construction, because a wheel that lost its
# Python layer still installs and still imports.
check_contents() {
  local who="$1"

  if ! python ci/contents.py wheelhouse/*.whl dist/*.tar.gz; then
    echo "!! $who: an artifact does not carry the package the source tree describes" >&2
    exit 1
  fi
}

# build every buildable backend IN PARALLEL into wheelhouse/, + an sdist into
# dist/. macOS always; Linux (x86_64, aarch64) when the container engine is up.
# Does NOT upload.
build_local_wheels() {
  has python cargo
  # the conda env ships its own rust toolchain WITHOUT the cross-target std
  # libs (x86_64-apple-darwin, *-pc-windows-msvc). ensure_rust/ensure_xwin
  # install targets into RUSTUP's toolchain, so rustup's cargo must win here.
  export PATH="$HOME/.cargo/bin:$PATH"
  ensure_rust
  ensure_xwin
  ensure_build_deps
  cd "$ROOT"

  rm -rf wheelhouse dist build
  mkdir -p wheelhouse/_logs

  local pids=() names=() rc=0 i linux=0

  # Stage BEFORE anything else starts, while the tree is still quiet: the whole
  # point of the copy is that no other build is writing to what it reads.
  # cibuildwheel copies Path.cwd() rather than the package dir it is handed, so
  # the staged tree has to be the working directory, not an argument.
  if ensure_container; then
    ensure_cibw
    cibw_env
    STAGE="$(mktemp -d)"
    stage_project "$STAGE"
    linux=1
  else
    echo "!! ====================================================" >&2
    echo "!! NO CONTAINER ENGINE -- the two Linux wheels will NOT" >&2
    echo "!! be built. This is an INCOMPLETE release matrix." >&2
    echo "!! Fix: colima start   (then check 'docker info' works)" >&2
    echo "!! ====================================================" >&2
  fi

  echo ">> building macOS + Windows wheels (sequential -- shared build/ tree) ..." >&2
  { build_macos_wheels && build_windows_wheels; } >wheelhouse/_logs/macos-windows.log 2>&1 &
  pids+=($!); names+=(macos-windows)

  if [ "$linux" -eq 1 ]; then
    # One job, both arches in sequence -- native aarch64 first, then the slow
    # emulated x86_64. They share a single colima VM on a fixed 4 CPUs and 8 GB,
    # so running them at once buys no throughput, it only halves what each gets;
    # and the 144s the container copy used to take was accidentally keeping them
    # apart anyway, so stage_project turned an accidental stagger into a race for
    # the same pool. (This is a resource argument, not the fix for the qemu
    # linker segfault -- that one is cibw_env's RUSTFLAGS, and it bit under both
    # orderings.) aarch64 costs ~3 min, so the sequencing is nearly free.
    echo ">> building linux-aarch64 then linux-x86_64 (one shared VM) ..." >&2
    (
      cd "$STAGE"
      cibuildwheel --platform linux --archs aarch64 --output-dir "$ROOT/wheelhouse" \
        >"$ROOT/wheelhouse/_logs/linux-aarch64.log" 2>&1
      CIBW_ENVIRONMENT_LINUX="$CIBW_ENVIRONMENT_LINUX $NO_LLD" \
        cibuildwheel --platform linux --archs x86_64 --output-dir "$ROOT/wheelhouse" \
          >"$ROOT/wheelhouse/_logs/linux-x86_64.log" 2>&1
    ) &
    pids+=($!); names+=(linux)
  fi

  i=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      # glob, not one name: the linux job writes one log per arch under a
      # single job name, and the tail has to reach whichever of them failed.
      echo "!! ${names[$i]} build FAILED -- tail of wheelhouse/_logs/${names[$i]}*.log:" >&2
      tail -25 wheelhouse/_logs/"${names[$i]}"*.log >&2
      rc=1
    fi
    i=$((i + 1))
  done

  # Before the exit, not after: a failed build must not leave the VM up either.
  # Nothing below here needs docker -- the sdist and both gates are host-side.
  release_container

  [ "$rc" -eq 0 ] || { echo "wheel build failed" >&2; exit 1; }

  # An sdist beside a short wheelhouse is what makes an incomplete build LOOK
  # finished, so it is gated on the matrix rather than on rc alone.
  check_matrix build

  python setup.py sdist        # rust source + Cargo.lock ride along via MANIFEST
  check_contents build
  echo ">> built wheels:" >&2
  ls -1 wheelhouse/*.whl >&2
}

# upload whatever `./do build` produced -- no rebuild here.
deploy() {
  has python
  cd "$ROOT"
  ls wheelhouse/*.whl >/dev/null 2>&1 || { echo "no wheels in wheelhouse/ -- run ./do build first" >&2; exit 1; }
  ls dist/*.tar.gz >/dev/null 2>&1 || { echo "no sdist in dist/ -- run ./do build first" >&2; exit 1; }

  # Checked again here, not just in build: deploy never rebuilds, so a stale or
  # short wheelhouse from an earlier run would upload without a word.
  check_matrix deploy
  check_contents deploy

  pip install -q twine
  twine check wheelhouse/*.whl dist/*.tar.gz

  local tokval
  tokval="$(tok)"
  twine upload --skip-existing wheelhouse/*.whl dist/*.tar.gz -u __token__ -p "$tokval"
}

lint() {
  has python
  cd "$ROOT"

  if [ "${1:-}" = "--fix" ]; then
    python -m view.lint --fix
    python -m black -l "$PY_COLS" -q "${PY_DIRS[@]}"
    return
  fi

  # view.lint always flags the protected test/MDR.py (its docstrings feed the MDR
  # report tables and must not be reformatted). Show its output but fail only on
  # findings elsewhere, so the remaining linters still run.
  local vl
  vl="$(python -m view.lint || true)"
  printf '%s\n' "$vl"
  if printf '%s\n' "$vl" | grep -vF 'test/MDR.py' | grep -qE '\.py:[0-9]+:'; then
    echo "view.lint: findings outside test/MDR.py" >&2
    exit 1
  fi

  # The autofixer's own regression suite. Each fixer addresses the file by line
  # number, so before the pipeline re-parsed between them a file carrying two
  # kinds of violation had the second one placed against stale offsets -- a
  # blank line landing inside an unrelated docstring. Findings are not enough
  # to catch that; this pins the bytes --fix produces.
  python -m view.lint_test

  # NB: no -q here. `black --check -q` suppresses its own "would reformat"
  # lines, so a failure shows up as a bare exit code with no clue which file.
  python -m black -l "$PY_COLS" --check "${PY_DIRS[@]}"

  if command -v eastwood >/dev/null 2>&1; then
    eastwood src
    eastwood -lang python $(find qkd ci bench -name '*.py')
    if [ -d docs/_tut ]; then
      eastwood docs/_tut                                        # docs explainers (ts + svelte)
    fi
  fi

  if [ -d docs/node_modules ]; then
    (cd docs && npx --no-install svelte-check --tsconfig ./tsconfig.json)
  fi
}

usage() {
  cat <<'EOF'
usage: ./do <verb> [args]

  develop            compile the Rust core in place (debug) -- the import path
  check              cargo check + python syntax, no link (fast iteration)
  test [name ...]    develop, then run test/*.py; names select files
                     (./do test gaussian budget). MDR_OUT overrides the report dir
  bench [name ...]   release build, then run bench/*.py through the MDB harness;
                     tables land in bench/_results (MDB_OUT overrides)
  lint [--fix]       view.lint + black + eastwood + svelte-check
  fmt                black over the python trees only (no AST autofix, no rustfmt)
  clean [--all]      drop build/dist/wheelhouse/reports; --all also drops target/
                     and the compiled core
  docs [build|deploy]  vitepress dev server, or build/deploy the static site
  build              every cp311-abi3 wheel -> wheelhouse/, sdist -> dist/
  deploy             twine check + upload what build produced (never rebuilds)
  help               this text

`./do` needs `python` on PATH, which on this machine means the conda env:
  conda activate aaronson
EOF
}

case "${1:-}" in
  develop) develop ;;
  check)   check ;;
  build)   build ;;
  test)    shift; test_ "$@" ;;
  bench)   shift; bench "$@" ;;
  docs)    shift; docs "$@" ;;
  clean)   shift; clean "$@" ;;
  fmt)     fmt ;;
  deploy)  deploy ;;
  lint)    shift; lint "$@" ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 1
    ;;
esac
