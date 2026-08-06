# Getting Started

Install and a first run. Every family: [Usage by example](/usage). Layering:
[Architecture](/architecture#the-layering).

## Install

```sh
pip install qkd
```

Binary wheels are on PyPI. From source:

```sh
git clone https://github.com/plutoniumm/qkd
cd qkd
./do develop
python -c "import qkd; print(qkd.__version__, qkd.backend_info())"
# 0.2.0 ('gpu', 'f32', 'Apple M2 (Metal)')
```

`./do develop` builds `qkd/_core.abi3.so` in place; the working tree is the import
path. Without a GPU adapter the tuple reads `('cpu', 'f64', ...)`.

| Needs | Why |
| --- | --- |
| Python 3.11+ with `numpy` | bulk returns are `ndarray` ([which](/guide/conventions#arrays)) |
| A Rust toolchain (`cargo`, `rustc`) | the core is compiled |
| `python` on `PATH` | `./do` calls `python`, not `python3` |

::: warning A bare `cargo build` will fail to link
A PyO3 `extension-module` cdylib has no Python symbols; `setup.py build_ext`
supplies `-undefined dynamic_lookup`. Use `./do develop`; `./do check` checks
without linking.
:::

## First run: a key rate you did not set

Neither $T$ nor $\xi$ is stated. The pilot DSP runs over 200 000 simulated
symbols; its residual phase error is the only $\xi$ row.

```python
import qkd as q

link = q.Link(
    modulation=q.GaussianModulation(v_a=4.0),
    channel=q.Fiber(length=15.0, alpha=0.2),
    alice=q.Alice(
        laser=q.Laser(linewidth=10e3),
        pilots=q.Pilots(power_db=12.0, freq=180e6),
        symbol_rate=100e6,
    ),
    bob=q.Bob(
        detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True),
        lo=q.LocalLO(linewidth=10e3),
    ),
    dsp=q.DSP(phase=q.PilotPhase(), block=32),
    security=q.FiniteSize(beta=0.95, n=1e9),
)
res = link.run(symbols=200_000, seed=7)

res.dsp.v_err       # 0.0075742    rad^2, what the pilot estimator left
res.budget.total    # 0.0304118    SNU at the channel input, derived from it
res.est.xi          # 0.0142987    what Alice and Bob estimate from the samples
res.key_rate        # 0.0285553    bit/symbol
```

`print(res.budget.table())` ([rows and planes](/guide/budget#the-per-source-table)):

```
source   xi @ input  xi @ bob    plane
channel  0.0000e+00  0.0000e+00  no channel xi declared: a pure-loss span
phase    3.0412e-02  1.5242e-02  Kish (80)-(82) inferred channel: input-referred, no /T
total    3.0412e-02  1.5242e-02
```

## First run: pinned numbers

Lodewyck et al. 2007, fully pinned:
[Usage](/usage#gaussian-modulation-cv-qkd). `ref=` is
[required](/guide/conventions#excess-noise-carries-a-plane); `trusted=` is a
[security model](/guide/security#trusted-vs-untrusted-is-a-security-model).

## Checking the backend

```python
>>> import qkd
>>> qkd.backend_info()      # (name, precision, device) for whatever resolve() probed
('gpu', 'f32', 'Apple M2 (Metal)')
>>> qkd.supports('f64')     # whether that backend carries the precision natively
False
>>> qkd.__gpu__             # whether the gpu feature was COMPILED IN
True
```

Every `q.Link` run is CPU f64 regardless: [the GPU path](/roadmap#gpu).

## Where to go next

The sidebar is the index.

## Development

Everything goes through `./do`; there is no Makefile, `pyproject.toml` or CI.

```sh
./do develop           # compile the Rust core in place (debug) -- the import path
./do check             # cargo check + python compileall, fast, no link
./do test [name ...]   # develop + run every test/*.py, emit markdown reports
./do bench [name ...]  # release build + run bench/*.py through the MDB harness
./do lint [--fix]      # view.lint + black --check + eastwood + svelte-check
./do fmt               # black over the python trees
./do docs              # vitepress dev server on :3000
./do build             # every wheel -> wheelhouse/, sdist -> dist/
./do deploy            # twine check + upload what build produced
```

| Gotcha | Detail |
| --- | --- |
| Selectors | `./do test anchors budget` runs two files; an unmatched selector errors. `python test/core.py` runs one |
| Timings | release-build figures unless stated; `./do develop` is unoptimised |
| `./do bench` | builds release with `-Ctarget-cpu=native` and overwrites `qkd/_core.abi3.so`; run `./do develop` before the suite |
| `./do build`, `./do deploy` | gated on the full wheel matrix and on artifact contents; see `RELEASING.md` |

The suite and its reports: [Testing](/tests/#running).
