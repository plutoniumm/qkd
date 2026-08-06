<img src="https://raw.githubusercontent.com/plutoniumm/qkd/HEAD/icon.svg" align="right" width="96" alt="qkd" />

# qkd

A QKD protocol simulator with the hardware in the loop, over a native Rust core.

**Documentation: <https://plutoniumm.github.io/qkd/>**

Excess noise `ξ` is an output: `qkd.budget` assembles it from hardware parameters.
On a Gaussian-modulation `q.Link`, pilot-assisted DSP runs over simulated symbols,
and its residual phase error is a `ξ` term. Once frames are estimated, the key rate
comes from Alice and Bob's estimates, not from the simulated channel.

## Install

```sh
pip install qkd
```

- Python 3.11 or later. `numpy` is a runtime dependency.
- Wheels: macOS arm64 and x86_64, Linux `manylinux_2_28` aarch64 and x86_64, Windows x86_64. Other platforms install from the sdist, which needs a Rust toolchain.
- From source: [Getting Started](https://plutoniumm.github.io/qkd/getting-started#install).

## Example

Neither `T` nor `ξ` is stated. The pilot DSP runs over 200 000 simulated symbols;
its residual phase error is the only `ξ` row.

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

One block per protocol family and per layer: [Usage by example](https://plutoniumm.github.io/qkd/usage).

## Coverage

- Continuous variable: Gaussian modulation (homodyne, heterodyne), *M*-PSK discrete modulation, CV-MDI.
- Discrete variable: BB84 with weak coherent pulses and decoy states, six-state, SARG04, B92, DPS, RRDPS, COW, COW′, MDI-BB84, mode pairing, BBM92, E91, loss-tolerant source flaws.
- Beside the protocols: trusted-node networks (`q.Network`), Gaussian and truncated-Fock states (`qkd.gaussian`, `qkd.fock`), implementation attacks (`qkd.attacks`), reconciliation and privacy amplification (`qkd.reconcile`), composable ε accounting (`qkd.security`).
- Entry point and security analysis per family: [Protocol coverage](https://plutoniumm.github.io/qkd/guide/protocols#matrix).
- Excluded by decision: twin-field, phase-matching and sending-or-not-sending; satellite and free-space channels; qudit protocols.

## Before quoting a number

- `qkd.gaussian` and `qkd.fock` use ħ = 1, vacuum variance 1/2. `qkd.budget`, `q.Link` and `q.Swap` use shot-noise units, vacuum variance 1. [Conventions](https://plutoniumm.github.io/qkd/guide/conventions)
- `thermal_loss` and a nonzero `q.Channel(xi=…)` require `ref="input"` or `ref="output"`. `ξ` at Bob is `T` times `ξ` at the channel input.
- `q.Homodyne` and `q.Heterodyne` default to `trusted=True`: `eta` and `v_el` are Bob's, not Eve's. [Security](https://plutoniumm.github.io/qkd/guide/security#trusted-vs-untrusted-is-a-security-model)
- A mismatched `q.Link` tree constructs, then raises on the first `run()`, `measure()`, `claim()` or `explain()`, naming the restriction. [Refusals](https://plutoniumm.github.io/qkd/roadmap#refusals)
- Most `q.Link` families are asymptotic. Finite-key routes per family: [Security](https://plutoniumm.github.io/qkd/guide/security#asymptotic).
- Agreement with a paper is graded on two axes: reproducing what it computed, and reproducing what it measured at a stated number of fitted parameters. [Validation](https://plutoniumm.github.io/qkd/guide/validation#two-tiers)
- The pilot DSP chain simulates heterodyne detection only.
- Every `q.Link` run is CPU. `backend_info()` reports the backend that resolved, not the one a run used. [GPU kernels](https://plutoniumm.github.io/qkd/roadmap#gpu)

## License

MIT
