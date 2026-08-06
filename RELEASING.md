# Releasing qkd

qkd ships as **cp311-abi3** binary wheels (PyO3 Rust core, PEP 384 stable ABI)
plus an sdist. One wheel per OS×arch covers **every Python ≥ 3.11**. Everything is
built **locally** and pushed straight to PyPI — no CI, no GitHub Actions.
`./do deploy` is PyPI only; the docs site deploys separately (see the last section).

```sh
conda activate aaronson
./do build           # build every target in parallel (wheelhouse/ + dist/)
./do deploy          # twine check + upload what `build` produced (no rebuild)
```

Both verbs are gated: neither will emit an sdist or upload anything unless the
full five-wheel matrix is present and every artifact still carries the package
the source tree describes. See [The release gates](#the-release-gates) — that
section is the part of this file worth reading before a first publish.

`./do help` lists every verb. The ones that matter around a release are `./do
test` (which takes file names — `./do test anchors budget` — so a pinned-number
regression can be re-run alone), `./do check` for a fast compile-only pass, and
`./do clean` to drop `build/`, `dist/`, `wheelhouse/` and the generated reports
(`test/_reports`, `bench/_results`) before a fresh build. `./do clean --all`
additionally removes `target/`, the compiled core and the VitePress cache, which
costs a full rebuild of the wgpu dependency graph.

`./do` calls `python`, and on this machine plain `python` exists **only** inside
the `aaronson` conda env (`/usr/local/Caskroom/miniconda/base/envs/aaronson`) —
the system otherwise has `python3` only. Activate the env or nothing works.

## What gets built

| OS                     | x86_64                         | arm64 / aarch64                |
| ---------------------- | ------------------------------ | ------------------------------ |
| macOS                  | ✅ cross-compiled               | ✅ native                       |
| Linux (manylinux_2_28) | ✅ emulated (qemu)              | ✅ native (container)           |
| Windows                | ✅ cross-compiled (cargo-xwin)  | ❌ sdist only — see below       |

Five wheels, and they are the five entries of `WHEEL_MATRIX` in `do` — the one
place the release matrix is written down, and what `check_matrix` holds a build
to.

Plus the **sdist** (`setup.py sdist`), which bundles `README.md`, `LICENSE`,
`Cargo.toml`, `Cargo.lock`, `src/*.rs` + `src/*.wgsl` and `ci/smoke.py` (see
`MANIFEST.in`). Anyone without a matching wheel (musllinux, BSD, Windows-on-ARM,
...) installs from it — that needs a Rust toolchain on their machine, which `pip`
invokes through `setuptools-rust` — and `ci/smoke.py` rides along so they can
verify what their machine just compiled. The `.wgsl` shaders are in there because
they are `include_str!`-ed into the core: without them the sdist does not
compile. `test/` and `bench/` are **not** shipped in either artifact.

### Why there is no Windows-on-ARM wheel

Nothing in this build chain can **run** a `win_arm64` wheel — it is
cross-compiled from a Mac and there is no Windows-on-ARM machine to smoke-test it
on. An untested wheel is worse than no wheel: `pip` prefers a matching wheel over
the sdist, so shipping a broken one *removes* the fallback that would otherwise
have worked. Windows-on-ARM therefore installs from the sdist, which compiles the
core locally through `setuptools-rust` and rides along with `ci/smoke.py` so the
user can verify what their own machine produced.

Add the target back only when there is a real Windows-on-ARM box to smoke-test
on. Everything needed is one `rustup target add aarch64-pc-windows-msvc` and one
stanza in `build_windows_wheels`.

## The GPU backend is in the default wheel

The Rust core is built with the `gpu` cargo feature, which is **on by default**
(`wgpu` 30 + `pollster`). Every wheel above therefore contains the wgpu backend.
This is deliberate and load-bearing for the whole pipeline:

- **wgpu needs no vendor SDK at build time.** That is precisely why it was
  chosen over CUDA/cuQuantum. A CUDA dependency would break the cargo-xwin
  Windows cross-compile and the manylinux container builds outright, and could
  never be tested on this Apple Silicon dev machine at all. wgpu compiles to
  Metal / Vulkan / DX12 backends with nothing but the Rust toolchain, so all
  five targets in `WHEEL_MATRIX` still cross-compile from one Mac.
- **The adapter is resolved at RUNTIME**, not at build time: `request_adapter`
  is polled once and the core falls back to the rayon f64 CPU backend when no
  adapter answers. That never fails — an absent GPU is a fallback, not an error.
- **Which means the smoke test proves less than it looks like it does.** The CPU
  fallback is the *normal* case inside the manylinux build containers and on any
  headless machine, so `ci/smoke.py` passing in-container does **not** prove the
  GPU path works. It proves the wheel imports, the native core loads, the
  adapter probe is safe (does not hang, crash, or lie about precision), and the
  physics the wheel ships still reproduces its pinned numbers. Real GPU coverage
  only happens when someone runs `ci/smoke.py` on a box with a displayed adapter
  and sees `backend=gpu/f32`. On this dev Mac it does — `backend=gpu/f32 on
  Apple M2 (Metal)` — but that is a manual observation, not a build gate: treat
  the GPU path as untested-by-CI.
- **WGSL has no `f64`**, so the GPU backend is f32-only and `ci/smoke.py`
  asserts exactly that: a backend reporting `gpu` with `f64` means the adapter
  lied, and the smoke fails.
- **It costs build time.** wgpu is most of the dependency graph — the crate
  closure goes from ~25 to 73–85 depending on target (109 packages in
  `Cargo.lock`). That is a rounding error on the native builds and genuinely
  annoying on the qemu-emulated linux-x86_64 wheel, which is already the long
  pole of the parallel build.

A `cudarc`/cuFFT backend for NVIDIA f64 could be **added alongside** wgpu later
for people who have the hardware. It must never replace it and must never become
a default feature: the moment a vendor SDK is required at build time, this entire
local cross-compile pipeline stops working.

## How it works (`./do build`)

- **macOS** wheels build directly with the active Python via `setup.py
  bdist_wheel` (arm64 native; x86_64 cross-compiled with `CARGO_BUILD_TARGET=
  x86_64-apple-darwin` + `ARCHFLAGS`). cibuildwheel isn't used on macOS because
  it requires a python.org *framework* Python that isn't installed.
  `MACOSX_DEPLOYMENT_TARGET=11.0`.
- **Linux** wheels build inside `manylinux_2_28` containers via
  [cibuildwheel] + **colima**. Rust is installed inside the container
  (`CIBW_BEFORE_ALL_LINUX`); aarch64 is native to the arm Mac, x86_64 runs under
  qemu emulation (slower). musllinux is skipped locally by default (see
  `CIBW_SKIP` in `do`) — drop the skip to build it too.
- The **Windows** wheel — one, `win_amd64` — is cross-compiled with
  [cargo-xwin]: `xwin` downloads the MSVC CRT + Windows SDK into a local cache
  (Microsoft's license applies to that download — the "war crime"), lld links,
  and pyo3 ≥ 0.29 binds `python3.dll` via **raw-dylib**, so no import library is
  synthesized and no Windows Python is needed (`generate-import-lib` is now a
  no-op, kept as an escape hatch). setuptools-rust is pointed at the
  `ci/cargo-xwin.sh` CARGO shim, which routes `build`/`rustc`/`check` through
  `cargo xwin` and everything else (metadata, `--version`) to plain cargo.
  `_PYTHON_HOST_PLATFORM=win-amd64` forges the wheel tag the same way the macOS
  x86_64 cross build does. setuptools-rust names the cross-built extension with
  the *host* suffix, so the wheel is unpacked, `qkd/_core.abi3.so` renamed to
  `_core.pyd`, and repacked (`wheel pack` recomputes RECORD hashes).
- **PATH order matters.** The conda env ships its own rust toolchain *without*
  the cross-target std libs, so `build_local_wheels` puts `~/.cargo/bin` first —
  rustup's cargo (the one `ensure_rust`/`ensure_xwin` add targets to) must win.
  `ci/cargo-xwin.sh` does the same thing for the same reason.
- The **abi3 tag** comes from `setup.py`'s `options={"bdist_wheel":
  {"py_limited_api": "cp311"}}`, so a wheel built under any Python is still
  tagged `cp311-abi3`.
- Every Linux wheel is smoke-tested in-container (`ci/smoke.py`; numpy only,
  under a second). It imports qkd, resolves a backend via `backend_info()`
  and checks `supports()` agrees with it, asserts the Gaussian core's vacuum is
  1/2 per quadrature and that a thermal-loss channel leaves a bona fide
  covariance matrix, and reproduces the pinned Lodewyck 2007 asymptotic key rate
  (K = 0.0352 bit/symbol). It also asserts `qkd.__file__` is not `None` and
  that every name in `qkd.__all__` resolves — see the staged-copy note below
  for the failure those two lines exist for — and takes one array-returning call
  (`homodyne`), because the `numpy` Rust crate resolves NumPy's C-API through the
  `_ARRAY_API` capsule at **runtime**: a built wheel meeting a real NumPy is the
  only place that binding can break, so no source-tree test can reach it. macOS
  and Windows wheels are never
  *executed* by the build, and cannot be: nothing here can run a `win_amd64` or a
  macOS x86_64 binary. Smoke them by hand:

  ```sh
  python -m venv /tmp/qsmoke && /tmp/qsmoke/bin/pip install -q \
    wheelhouse/qkd-*-macosx_11_0_arm64.whl numpy
  /tmp/qsmoke/bin/python ci/smoke.py
  ```
- Every artifact's **contents** are gated, executed or not, by `ci/contents.py`
  — the only gate the macOS and Windows wheels get at all. Detail in
  [The release gates](#the-release-gates).
- The **linux-x86_64** build alone adds `RUSTFLAGS="-C linker-features=-lld"` to
  `CIBW_ENVIRONMENT_LINUX`. **rust-lld**, rustc's default linker on
  `x86_64-unknown-linux-gnu`, segfaults intermittently under qemu user-mode
  emulation — `linking with cc failed: signal: 11 (SIGSEGV)`, on a different
  crate's build script each time. The flag drops the `-fuse-ld=lld` / `gcc-ld`
  shim so the link goes through binutils `ld.bfd`. It is scoped to that one arch
  because it is stable **only** on the target that defaults to rust-lld —
  aarch64 rustc rejects it without `-Zunstable-options`, which is what a shared
  setting would break. Do **not** substitute `-C link-self-contained=no`: that
  also drops the self-contained crt objects and nothing compiles at all.
- Builds run in **two parallel jobs** into `wheelhouse/`: one does macOS then
  Windows (they share the `build/` tree), the other does linux-aarch64 then
  linux-x86_64 (they share one colima VM with a fixed 4 CPUs and 8 GB, and the
  x86_64 half is a qemu emulation that segfaults its linker when squeezed). The
  two jobs use disjoint resources — host CPU against the VM — so they overlap
  safely. Logs land in
  `wheelhouse/_logs/{macos-windows,linux-aarch64,linux-x86_64}.log`; on failure
  `do` tails the offending job's logs. The sdist is built **last** — after every
  wheel job succeeds *and* after `check_matrix` agrees the wheelhouse is
  complete, because an sdist beside a short wheelhouse is precisely what makes an
  incomplete build look finished.
- The Linux jobs build from a **staged copy** of the project (`stage_project`),
  made before anything else starts and thrown away afterwards, never from the
  live tree. cibuildwheel ships the project into the container with
  `tar -cf - . | docker exec -i … tar -xf -` and checks the *pipeline's* status,
  which is the container side's — so a host-side tar failure is invisible to it.
  The host tar fails whenever a directory changes under it, and `build/` and
  `target/` change constantly while the macOS and Windows wheels are being built
  in the same tree. libarchive's response is to stop descending into every
  directory it has not reached yet, so `/project` arrives with `setup.py`,
  `Cargo.toml` and `src/` but an **empty** `qkd/`: `find_packages()` returns
  nothing, `build_py` never runs, and the wheel ships as a bare `_core.abi3.so`
  that still installs and still imports — as an implicit namespace package whose
  every attribute lookup fails. That shipped on 2026-08-09. Staging also makes
  the copy ~500× faster, since the live tree is almost entirely `target/`, which
  the container never reads.
- `./do deploy` re-runs `check_matrix` and `ci/contents.py`, then `twine check` +
  `twine upload --skip-existing` over `wheelhouse/*.whl` and `dist/*.tar.gz`,
  using the token in `.vscode/token.env`. It **never rebuilds**, and refuses to
  run if either directory is empty. Re-running the two gates here is not
  belt-and-braces: because `deploy` never rebuilds, the wheelhouse it uploads may
  be from an entirely different, older, failed run.

## The release gates

Three gates stand between a build and PyPI. Each exists because of a specific
failure that produced **no error anywhere** — that is the point of the section,
so the failure is written down beside the gate.

`WHEEL_MATRIX` and `check_matrix` are in `do`; the third is `ci/contents.py`,
invoked through `do`'s `check_contents`. Both checks run in `./do build` *and*
again in `./do deploy`.

### `WHEEL_MATRIX` — the matrix, written down once

```sh
WHEEL_MATRIX=(
  macosx_11_0_arm64
  macosx_11_0_x86_64
  manylinux_2_28_aarch64
  manylinux_2_28_x86_64
  win_amd64
)
```

Five targets, declared in one array in `do` and read by everything downstream.
`win_arm64` is deliberately absent (see above). Adding or retiring a platform is
an edit to this array plus a build stanza, and nothing else — before it existed,
"what a complete release looks like" was implicit in the build code and therefore
unavailable to any check.

### `check_matrix` — refuses a short wheelhouse

Fails unless `wheelhouse/` holds one wheel matching every `WHEEL_MATRIX` entry,
printing which tags are missing, which are present, and the colima hint. It gates
two moments: in `build`, immediately before `setup.py sdist`; in `deploy`, before
twine is even installed. `$1` names the caller so the message says which verb
refused.

**The failure it prevents.** On 2026-08-09 the container engine was wedged.
`build_local_wheels` printed its no-container banner, built the three
host-buildable wheels, and **exited 0** — nothing had failed, so no exit code
could say otherwise. That left three wheels of five in `wheelhouse/`, and `./do
deploy` never rebuilds, so it would have uploaded exactly that. The damage is
silent on the far side too: `pip` does not error when no wheel matches, it falls
back to the sdist, so every Linux user would simply have started compiling Rust
from source with no message anywhere in the chain saying a wheel was meant to
exist. Counting artifacts against a declared matrix is the only thing that
catches a build whose every step succeeded.

### `ci/contents.py` — refuses an artifact that lost its contents

`check_matrix` counts artifacts; this one reads **inside** them. It opens all
five wheels and the sdist and requires each to carry:

- every `qkd/*.py` the source tree holds — globbed from `qkd/` at check
  time, so a newly added module is covered the moment it exists and there is no
  second list to keep in step;
- the compiled core under the one name its platform can import:
  `qkd/_core.pyd` on Windows, `qkd/_core.abi3.so` everywhere else;
- for the sdist instead of a compiled core: `Cargo.toml`, `Cargo.lock`,
  `ci/smoke.py`, and **every `src/*.rs` and every `src/*.wgsl`**.

It is the only gate the macOS and Windows wheels have at all — `build` never
executes those and cannot: nothing here can run a `win_amd64` or a macOS x86_64
binary.

**The two failures it prevents.**

1. **A wheel that lost its Python layer.** The staged-copy race described above
   shipped a wheel that was a bare `_core.abi3.so` with an empty `qkd/`. It
   installs. It imports — Python turns the leftover directory into an implicit
   namespace package. Only then does every attribute lookup fail, one symbol at a
   time, on the user's machine. `ci/smoke.py` gained its `__file__` and
   `__all__` assertions for the same failure, but the smoke only runs on Linux;
   `contents.py` is what extends the check to the four artifacts nothing can
   execute.
2. **An sdist that cannot compile.** The six `.wgsl` shaders reach the Rust core
   through `include_str!`, which makes them *build inputs*, not data files. A
   `MANIFEST.in` regression dropping `recursive-include src *.wgsl` would produce
   an sdist that looks entirely healthy — plausible size, `twine check` clean,
   imports irrelevant because nothing is compiled yet — and then fails at `cargo
   build` on the installing user's machine, with a missing-file error naming a
   shader they have never heard of. Enumerating `src/*.rs` **and** `src/*.wgsl`
   from the live source tree is what makes that unshippable; the shaders were
   added to this check on 2026-08-10, having been outside it before.

## One-time machine setup

`./do` bootstraps most of this automatically (`ensure_rust`, `ensure_xwin`,
`ensure_cibw`, `ensure_container`, `ensure_build_deps`), but for reference, the
host needs:

- A **Rust toolchain** (`rustup default stable`) plus **two** cross targets:
  `x86_64-apple-darwin` (added by `ensure_rust`) and `x86_64-pc-windows-msvc`
  (added by `ensure_xwin`). `aarch64-pc-windows-msvc` is **not** added and must
  not be — see [Why there is no Windows-on-ARM
  wheel](#why-there-is-no-windows-on-arm-wheel). Adding it is one of the two
  steps that would start producing an untestable wheel.
- **cargo-xwin** (`cargo install --locked cargo-xwin`, auto-installed).
- **cibuildwheel** and **setuptools-rust** in the conda env (auto-installed).
- A **native arm64 colima/lima** for the Linux containers. NB: Homebrew here is
  the Intel prefix (`/usr/local`), whose x86_64 colima/lima refuse to run under
  Rosetta — so native arm64 binaries live in `~/.local` and
  `/usr/local/bin/{colima,limactl}` are symlinks to them. `./do build` starts
  colima on demand (`--cpu 4 --memory 8 --vm-type vz`) and **stops it again when
  the Linux wheels are done** — but only a VM it started itself, so an engine
  that was already up is left alone. If no engine is
  available it prints a banner, builds the three host-buildable wheels, and then
  **fails** at `check_matrix` rather than emitting a short release — which is
  not what it used to do; see [The release gates](#the-release-gates).
- **node/npm** (for the docs site) and the PyPI token at `.vscode/token.env`.

## Caveats

- **x86_64 Linux is emulated** on Apple Silicon (qemu) — that wheel is the slow
  one in the parallel build, and wgpu's dependency graph makes it slower still.
  aarch64 Linux and both macOS wheels are fast.
- musllinux is **not** built locally by default (slow under emulation).
- **Windows wheels are never executed here** (macOS can't run PE binaries):
  linking succeeding + abi3 + the smoke passing on Linux is the whole guarantee.
  If in doubt, `pip install` the wheel on a real Windows box and run
  `ci/smoke.py`.
- **No wheel is ever GPU-tested by the build** — see the GPU section above. The
  containers fall back to CPU and the smoke happily passes.
- **A partial build is now a failed build.** Without a container engine there are
  no Linux wheels, and `check_matrix` fails rather than producing a three-wheel
  release. Nothing here silently ships less than the matrix any more; if that is
  genuinely wanted, the matrix is the thing to edit, not the check.
- The `.vscode/token.env` PyPI token is sensitive — keep it out of git.

## Docs → GitHub Pages

The docs site also deploys **locally**, straight from the built VitePress dist;
`./do deploy` has nothing to do with it. `./do docs` wraps the npm scripts and
runs `npm install` first if `docs/node_modules` is missing:

| command           | what it runs                      |
| ----------------- | --------------------------------- |
| `./do docs`       | `npm run dev` (vitepress dev server) |
| `./do docs build` | `npm run build`                   |
| `./do docs deploy`| `npm run build && npm run deploy` |

Or by hand:

```sh
cd docs
npm run build        # vitepress build (prod base /qkd/, writes .nojekyll)
npm run deploy       # pushes .vitepress/dist to the gh-pages branch
```

The base is `/qkd/` in production and `/` in dev (`docs/.vitepress/config.mts`),
and a `buildEnd` hook drops `.nojekyll` into the output. `npm run deploy` uses
`gh-pages -d .vitepress/dist -t` (`-t` publishes dotfiles, so `.nojekyll`
survives; without it GitHub's Jekyll pass eats underscore-prefixed asset paths).
One-time repo setting: **Settings → Pages → Deploy from a branch → `gh-pages`**.
The site lands at <https://plutoniumm.github.io/qkd/>.

What ships on it: the index and getting-started pages, `usage.md` (the code-first
cheat sheet), a ten-page **Guide** track under `docs/guide/` plus
`architecture.md` and `roadmap.md`, and `studio.md`.
`docs/.vitepress/config.mts` holds the sidebar, which is the authoritative list
and also drives the prev/next reading order.

The generated MDR tables under `docs/tests/` are gitignored, so they are absent
on a fresh checkout; `ignoreDeadLinks` carries a `/^\/tests\//`
rule so the build does not fail over them. Regenerate them with
`MDR_OUT="$(pwd)/docs/tests" ./do test` before `./do docs deploy` if the
published tables should be current.

[cargo-xwin]: https://github.com/rust-cross/cargo-xwin
[cibuildwheel]: https://cibuildwheel.pypa.io/
