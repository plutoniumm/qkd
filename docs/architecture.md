# Architecture

Layers, engines, and the Python surface that reaches each engine. Refusals and the
four permanent exclusions: [Status and roadmap](/roadmap#refusals).

## The thesis

| | Usual toolkit | qkd |
| --- | --- | --- |
| $T$ | input parameter | derived from fibre length, attenuation, coupling |
| $\xi$ | input parameter | **output**, assembled from laser linewidth, $v_{el}$, $\eta$, DAC bit depth, pilot ratio, symbol rate, fibre length and the DSP residual |
| DSP | ideal or absent | in the loop; its residual error is a noise term |
| Parameter estimation | skipped or asymptotic | finite $n$, with confidence bounds |
| Detector noise | folded into $\xi$ | trusted/untrusted split, set by configuration |

## The layering

| Level | What it is | Rule at its boundary |
| --- | --- | --- |
| Rust core modules | one kind of mathematics each, no state between calls | imports no Python |
| `_core`, the PyO3 boundary | free functions and `#[pyclass]` objects | validates ranges, does arithmetic, knows no protocol. An engine can be anchored here and reached by no component |
| `qkd/` components | frozen dataclasses validated at construction (`components.py`), immutable `State` wrappers (`gaussian.py`, `fock.py`), the budget assembler (`budget.py`), the catalogues (`impairments.py`, `attacks.py`, `reconcile.py`, `security.py`, `rates.py`, `dps.py`) | describe hardware; compute no key rate. `budget.assemble()` takes no $\eta$ and no `trusted` ([why](/guide/budget#why-v-el-is-not-a-budget-line)) |
| `q.Link`, `q.Swap`, `q.PairLink` | compilers: component tree to plan; refuse what has no bound; label every quantity `pinned`, `derived` or `default` | the only level that knows a protocol. Three shapes: a point-to-point link, a **measurement** in the middle, a **source** in the middle |
| `q.Network` | key management over a graph of the above | not physics. Vertices hold key in the clear; a multi-hop number is a throughput over trusted nodes, never a bound. Carries `q.Link` or `q.Swap`; refuses `q.PairLink` |

## The core modules

| Module | Owns | Reached through |
| --- | --- | --- |
| `gaussian.rs` | the symplectic covariance engine: $n$-mode state, symplectic operations, thermal loss, sampling, conditioning | `qkd.gaussian`; every `q.Link` |
| `fock.rs` | the truncated number basis: Fock, cat, GKP, coherent, squeezed and thermal states; Wigner, Husimi, negativity, parity, non-Gaussianity, loss | [`qkd.fock`](/guide/fock) |
| `keyrate.rs` | Devetak–Winter, Holevo, Leverrier finite size, the WTY bound; `cv_general`, the Gaussian de Finetti arithmetic | `q.Link`. `cv_general` has no `q.` path ([the reduction](/roadmap#definetti)) |
| `pipeline.rs`, `threefry.rs` | the per-symbol train and its DSP; the counter-based RNG both trains draw from | `q.Link`, CPU only |
| `click.rs`, `decoy.rs`, `discrete.rs` | threshold detection on that train; Ma–Qi–Zhao–Lo photon-number bookkeeping; the BB84-WCP and COW rates | `q.Link` with `q.DifferentialPhase`, `q.BasisKeying`, `q.IntensityKeying`, over `q.Decoy`. The DPS finite-key layer: `qkd.dps` |
| `sixstate.rs`, `sarg.rs` | a third mutually unbiased basis; the four BB84 states sifted by a non-orthogonal pair announcement | `q.BasisKeying(bases=3)`, `q.BasisKeying(announce="pair")`; lengths from `q.security.keylength` |
| `b92.rs` | two non-orthogonal states, Koashi's receiver, two phase-error analyses | `q.TwoStateKeying` with `q.NullingReceiver`. The photon-number-resolving `b92_strong` has no `q.` path |
| `dmcs.rs` | the $M$-PSK constellation, its mutual information, the analytic `dm_rate`; the relative-entropy proof `dm_secure` with its trusted and relaxed forms | `q.PhaseShiftKeying` with `q.Asymptotic` (analytic) or `q.CertifiedBound` (certified; `q.Heterodyne(..., trusted=True)` for the trusted form, `.relaxed()` for the relaxed set) |
| `pairs.rs`, `ekert.rs` | the two-squeezer pair source and the BBM92 observables; the CHSH violation as the security parameter | `q.PairLink` with `q.SymmetryBound` or `q.ViolationBound`; `qkd.pairs.finite` |
| `relay.rs` | CV-MDI equivalent noise, the post-relay covariance, two-source interference at the coupler | [`q.Swap`](/guide/relay) with `q.BellDetector`. `q.TwoModeBound(block=q.GaussianBlock(…))` gives Papanastasiou, Ottaviani & Pirandola (2017)'s rate per use, not a length |
| `mdi.rs` | the qubit Bell measurement between two weak-coherent senders: forward model, two-sender decoy bound, Curty's finite-key length | `q.Swap` with `q.Relay(bell=q.BellAnalyser(…))` and `q.TestBasisBound`; `block=q.RelayBlock(…)` gives a key **length** in bits over the block. Composes inside `q.Network` |
| `pairing.rs` | pairing two clicks *after* the announcement, and the $O(\sqrt\eta)$ that buys | `q.rates("pairing")`, `q.security`; [no component tree](/roadmap#core-only) |
| `rrdps.rs` | privacy amplification that reads no observable | `q.rates("rrdps")` at a caller-stated `cost`, including Matsuura, Sasaki & Koashi (2019), strictly tighter than the GLLP tagging form at every point; `q.security.keylength`; `q.security.allocate()`; [no component tree](/roadmap#core-only) |
| `flaws.rs` | one modulator angle: the exact reading and the quantum-coin bound, side by side | `q.SourceFlaw`; `q.Link` with `q.FlawedKeying` and `q.FlawBound(analysis=…)` |
| `pnr.rs` | detectors that count rather than click; POVMs derived rather than copied | `q.ThresholdArray`, `q.PnrDetector`. `q.Link` refuses both ([refusals](/roadmap#refusals)) |
| `tlo.rs` | the transmitted oscillator: shot-noise unit, phase variance, leakage, a monitor for a steered unit | `q.TransmittedLO`, `qkd.attacks`. `q.Link` refuses it ([refusals](/roadmap#refusals)) |
| `cascade.rs` | counted Cascade: disclosed parities and a frame error rate, never an $\varepsilon_{\text{cor}}$ | `qkd.reconcile.run_cascade` |
| `sdp.rs` | a linear objective under dense linear matrix inequalities, on the dual; certifies the COW′ phase error | `q.rates("cow-vacuum")`, `q.security`; [no component tree](/roadmap#core-only) |
| `herm.rs` | cyclic Jacobi eigenvectors for complex Hermitian matrices; the cone in which `dm_secure` certifies its dual point | no Python surface; exposed for `test/sdp.py` to arbitrate against numpy |
| `lp.rs` | a linear programme over the diagonal cone, dualised | no Python surface; consumed by `mdi.rs` and `rrdps.rs` |
| `postselect.rs` | the lift from an IID-collective proof to a coherent-attack one | no Python surface; [reaches no family](/roadmap#postselect) |
| `eur.rs` | the entropic uncertainty relation: Maassen–Uffink's memoryless form, the Berta *et al.* memory form, and the composable key length that follows. A relation with its own family table, not a fourth solver | no Python surface; `eur_family` refuses every shipped family |
| `std.rs`, `numeric.rs` | one spelling each of the shared helpers; the fixed-count bracket searches and the shared divergence, the step count always the caller's | none, by design |
| `backend.rs` + `*.wgsl` | `Precision`, the `Backend` trait, `Cpu` (rayon, f64), `Gpu` (wgpu, f32), `resolve()`, the WGSL pipelines | `backend_info()`, `supports()`, `q.__gpu__`. No engine module imports it; its CPU and GPU entry points are unexported plumbing |
| `lib.rs` | the PyO3 module `_core` | — |

Reachability is per function: a family runs while some functions of its engine
stay on `_core`. Check with an AST walk of `qkd/*.py` for `_core.<attr>` and
`from ._core import`. A grep over-counts: refusal literals and comments name
`_core` functions too.

`std.rs` is `crate::std` and shadows the `std` crate for paths written in `lib.rs`
alone.

## Crossing the PyO3 boundary

The `#[pyclass]` objects: `GaussianState`, `FockState`, `SimOut`, `ClickOut`,
`BasisOut`, `SymStats`, `Povm`.

| Crossing | As |
| --- | --- |
| Scalars | Python floats and tuples |
| Bulk returns | numpy arrays through `rust-numpy`, the buffer moved out of Rust or copied from one the source still owns ([which](/guide/conventions#arrays)) |
| NumPy's C API | resolved through the `_ARRAY_API` capsule at run time, not through CPython's stable ABI; `ci/smoke.py` makes one array-returning call |

## Compute: dispatch on precision, not device

**WGSL has no `f64`.** The `wgpu` backend is single-precision permanently, so a
caller asks for a precision.

```python
>>> import qkd

>>> qkd.backend_info()
('gpu', 'f32', 'Apple M2 (Metal)')

>>> qkd.supports('f64')
False
```

| Workload | Size | Character | Precision | Device |
| --- | --- | --- | --- | --- |
| DSP over $\sim 10^8$ symbols | huge | massively parallel, over 12–16-bit ADC samples, within f32's 24-bit mantissa | f32 | GPU (wgpu) — *kernels written and measured, not wired* |
| Covariance matrices, symplectic maps, Holevo bound | $2\times2$, $4\times4$ | cancellation-sensitive, negligible cost: nearly equal terms subtracted in f32 manufacture key out of rounding | f64 | CPU (rayon) |
| The SDP interior-point path | $4\times4$ blocks | strict feasibility is *tested by* a Cholesky factorisation, so a rounding error is a wrong feasibility verdict | f64 | CPU, single-threaded |

`Backend::supports` is one-way: f64 serves an f32 request, never the reverse.
`resolve()` falls back to `Cpu` when no adapter answers. Measurements, and the
stages kept on the CPU: [GPU kernels](/roadmap#gpu).

### Why wgpu and not CUDA

| | wgpu | CUDA |
| --- | --- | --- |
| Vendor SDK at build time | none | required |
| `./do build` wheel matrix from one Apple Silicon machine, no CI | macOS arm64/x86_64 direct, Linux via manylinux containers, Windows via `cargo-xwin` | breaks |
| Executable on the development machine | yes | never |

A `cudarc` + cuFFT backend for f64 on NVIDIA hardware may be **added alongside**
wgpu through the same `Precision` interface, never as a replacement. Every GPU
kernel is checked against the f64 CPU.

## Engines: how the four-engine proposal is superseded {#engines-how-notes-03-md-is-superseded}

Gaussian-modulation CV-QKD (GG02, after Grosshans–Grangier 2002) is Gaussian end to
end. Discrete-modulation proofs are bounded by convex optimisation, not tensor
networks: [three solvers](/roadmap#solver), on three cones.

| Engine (as proposed) | Representation | Needed for CV-QKD | Disposition |
| --- | --- | --- | --- |
| Symplectic covariance | $\bar{\mathbf r} \in \mathbb R^{2N}$, $V \in \mathbb R^{2N\times2N}$ | yes — it is the protocol | built |
| Truncated Fock | $\rho$ on levels $0\dots d-1$, single mode | no | **built** — one mode suffices for cat, Fock and approximate GKP states, their Wigner and Husimi functions, and a negativity volume that is not identically zero |
| MPS / MPDO over many modes | bond dimension $D$ | no | deferred — open-system dynamics, arbitrary $H(\hat x, \hat p)$, many-mode non-Gaussian states |
| Phase-space stochastic | $M$ trajectories, SDE integration | no | deferred, same |
| Real-grid wavefunction | $\psi(x)$ mesh, split-operator FFT | no | deferred, same |

## The Gaussian layer

Covariance, ordering, vacuum and the bona fide condition:
[Conventions](/guide/conventions). Operations: [Gaussian layer](/guide/gaussian).

### Thermal loss, not pure loss

Pure loss carries no excess noise:
[thermal loss and the `ref=` plane](/guide/gaussian#thermal-loss-and-the-required-ref-plane).

### Trap 1: which plane the excess noise is referred to

$\xi_{\text{Bob}} = T\,\xi_{\text{Alice}}$. A Bob-plane figure used as an
input-referred one is short by $T$, about $10^{-2}$ at 100 km. `thermal_loss` and
`q.Channel(xi=…)` require `ref=`; `Budget.at()` is the only way to move a plane
([the plane](/guide/conventions#excess-noise-carries-a-plane)).

## DSP in the loop

CPU only, heterodyne only: [DSP in the loop](/guide/link#dsp-in-the-loop).

| Stage | Job | Residual that feeds the budget |
| --- | --- | --- |
| Carrier frequency offset estimation | remove the Alice–Bob laser beat frequency | uncorrected drift within a frame |
| Pilot-assisted phase recovery | track the relative optical phase from the pilot tone | phase-estimate variance $\mathrm{Var}(\hat\phi)$ |
| Frame rotation | apply the estimated rotation to the payload symbols | interpolation error between pilots |

$\xi_{\text{phase}}$ scales with $V_A$, so the rate-maximising $V_A$ is not the
largest. Its two forms: [Noise budget](/guide/budget#the-two-forms-of-the-phase-term).

## Noise budget: $\xi$ as an output

[`qkd.budget`](/guide/budget) and [`qkd.impairments`](/guide/impairments). $v_{el}$
and $\eta$ are not budget rows ([why](/guide/budget#why-v-el-is-not-a-budget-line)).

### A rotation moves variance, it does not make it

A residual phase adds $\xi_{\text{phase}}$ and attenuates $T$ by $e^{-v}$; the two
halves conserve the Bob-plane signal ([`infer()`](/guide/budget#infer),
[fading rows](/guide/impairments#fading)). Charging one half alone overstates the
rate. The transmittance half is absent in three places:

- sampling jitter, kept in the loop as a per-symbol random gain whose mean the
  fitted slope absorbs;
- the estimated path, whose fitted slope already carries $e^{-v/2}$;
- `form="literature"`, which holds the transmittance estimate fixed and refuses to
  return a transmittance half.

### Trap 2: trusted vs untrusted is a security model, not a physics model

Identical hardware gives different key rates and reach under the two labellings, so
the split is `trusted=` on the detector. The key-rate layer owns it; a budget has no
$\eta$ and cannot form $\mu v_{el}/T$. Substitution, $\mu$ and the $I_{AB}$
invariant: [Security](/guide/security#trusted-vs-untrusted-is-a-security-model).

## Parameter estimation and key rate

$T$ and $\xi$ come from $m$ disclosed symbols and enter the rate at their worst case
within a confidence interval ([finite size](/guide/security#finite-size)).
`res.key_rate` is the claim from the estimates; `res.oracle` is read off the $T$ and
$\xi$ that were set ([estimates versus the oracle](/guide/link#estimates-versus-the-oracle)).

## The click-detector family

[The click-detector family](/guide/link#the-click-detector-family); per protocol:
[Protocol coverage](/guide/protocols).

### One pulse train, two receivers

Only the detector is discrete. Both receivers share the pulse train, the channel and
the optical phase walk: both draw from `threefry.rs` and share `pipeline.rs`'s
Wiener walk, a chunked scan on a fixed grid whose carries combine in index order. The
result depends on the plan, not the thread split.

| | CV branch | Click branch |
| --- | --- | --- |
| Alice encodes | Gaussian-modulated amplitudes, $V_A$ in SNU | one uniformly random phase bit per pulse, fixed $\mu$ |
| Bob's optics | balanced heterodyne against a local oscillator | delay-line interferometer, one arm delayed by $d$ symbol periods |
| Bob's detector | ADC record, two quadratures | two threshold detectors, one bit per gate |
| Phase noise appears as | residual $v_{\text{err}}$ after pilot recovery, hence $\xi_{\text{phase}}$ | differential phase $\theta_k - \theta_{k-d}$, hence fringe error, hence QBER |
| Rate from | Devetak–Winter, Holevo, symplectic eigenvalues | collision probabilities / GLLP with decoy |

Row four: one anchored laser-linewidth walk yields both the CV excess noise and the
click-family QBER. Row five is not shared: a covariance matrix supports a Holevo
bound only for a Gaussian measurement, and `Link` refuses a mismatched
`modulation`/`security` pair.

### Threshold detection, and the photon-number convention

A threshold detector fires with probability $1 - (1-p_d)e^{-\eta|\alpha|^2}$; the
exponent is the mean photon number.

| Layer | Coherent state | $|\alpha|^2$ |
| --- | --- | --- |
| Gaussian, $\hbar = 1$, vacuum variance $\tfrac12$ | `vacuum().displace(dx, dp)` | $(dx^2 + dp^2)/2$ |
| Pipeline, shot-noise units, vacuum variance $1$ | — | $(x^2 + p^2)/4$ |

`click.rs` is written in photons and reads no quadrature.

### Detector memory is why this stage is sequential

| Pass | Character |
| --- | --- |
| Per-slot interference, the two exponentials and the RNG draws behind raw click decisions | index-pure and parallel; the expensive half of the run |
| Dead time and afterpulsing | a sequential scan: a click blinds its own detector for $\lceil t_d f_{\text{sym}} \rceil$ gates and arms a spurious click on the first live gate after, which can itself blind and re-arm |

### The gap this family existed to close {#derived-qber}

QBER is `derived` wherever `Bob` has a `DelayInterferometer` and `Alice` a laser and
a symbol rate: from fringe contrast, the Wiener phase walk over the delay, dark
counts and the squashing coin that resolves double clicks. It is `pinned`, with the
pipeline not running, under `IndividualAttack(qber=…)`. Finite-key lengths:
[Security](/guide/security#finitekey).

### Two variants that change one factor {#two-variants-that-change-one-factor}

Six-state and SARG04 send BB84's photons through BB84's detectors and differ only at
sifting and privacy amplification: two modules under `q.BasisKeying`, not two
protocol stacks ([six-state](/guide/protocols#sixstate),
[SARG04](/guide/protocols#sarg)).

Both modules carry domains their source equations do not state. Past the point
where they certify anything, the SARG04 privacy-amplification terms keep returning
numbers, and both turn back upward; `E1_DEAD` and `E2_DEAD` cut each term at the
value that credits zero. Neither refuses its argument, so a distance sweep runs
through both.

### B92: two states, and a receiver that may say nothing

Koashi's receiver — a displacement by a phase-locked local oscillator, then one
threshold gate — reuses `interfere` from `click.rs`: the ports carry
$2t\mu(1 \pm V)$ and the weak-flux bit error is $(1-V)/2$. One rate equation serves
both variants; they differ in where $e_{\text{ph}}$ comes from.

| Variant | Phase error |
| --- | --- |
| Plain B92 | **derived** from the loss and the observed bit error, by solving Tamaki & Lütkenhaus (2004)'s implicit constraint. The bound reaches $1/2$ at loss $=$ overlap, where unambiguous state discrimination takes the protocol over, so the rate reaches zero and nothing raises |
| Strong-reference B92 | **supplied** on Koashi (2004), whose bound is over quantities of a virtual entanglement picture and has no closed form in $(\mu, T, \eta, V, \text{dark})$. Tamaki, Lütkenhaus, Koashi & Batuwantudawe (2009) **derive** it on a photon-number-resolving receiver: `b92_strong` reproduces their Fig. 2 at 55.0 / 100.216 / 121.689 km against a published 55 / 100 / 122, zero fitted. The first crossing is quoted to 0.1 km: the rate is non-monotone across a 0.286 km band, first zero 54.832 and last positive 55.118, so bisection on positivity returns 54.974 or 55.046 depending on the bracket. No `q.Bob` builds that receiver |

## The entanglement-based path {#pairs}

`q.PairLink` puts a **source** in the middle where `q.Swap` puts a **measurement**.
Source model, position and security models:
[Protocol coverage](/guide/protocols#pairs).

### The two are not two security levels of one protocol {#not-di}

Bennett, Brassard & Mermin (1992): the Bell test is not what makes Ekert's protocol
secure. With trusted, characterised qubit analysers the violation reads the same
disturbance as the phase error, and `ekert_point` returns both numbers side by
side. The CHSH price is the worse one wherever a state assumption is available
([the gap](/guide/protocols#e91)).

Nothing here is device independent. `ekert_rate` computes the device-independent
formula, takes $S$ as an argument, and makes the caller state its source
([refusals](/roadmap#refusals)).

## Anchored in the core, not reachable from the API {#core-only}

Why each engine without a component tree has none: [Status](/roadmap#core-only).

**A forward model and a bound never share a name.** `mdi_yield` is the
single-photon-pair yield given the hardware; `mdi_y11` is the least it could be
given only the observed gains. Composed the wrong way round, they overstate the key
rate. `cvmdi_rate` and `cvmdi_point` keep the same separation.

**RRDPS is the structural opposite of COW.** Its privacy amplification reads packet
length and photon number and nothing an adversary can touch, so there is nothing
to estimate and nothing for an attack to fake. The price is sifting: a packet of
$L$ pulses yields at most one bit. COW's phase error is not estimable from the record — the
zero-error attack leaves the data line and the monitoring line undisturbed — so
`cow_rate` takes `e_phase` as an input. `dps_rate` sits between them, deriving
Eve's information from the measured QBER, and reaches zero at $6/38 = 15.8\%$.

### Three solvers, and what each is not {#the-solver-and-what-it-is-not}

Cones, method, refusals, the `lp.rs` assembly trap and `dm_secure`'s limits:
[Three solvers, three cones](/roadmap#solver).

### The postselection technique reaches no family here {#postselect}

`ps_family` refuses every shipped family: [per family](/roadmap#postselect).

## Three catalogues beside the key rate {#catalogues}

None of these reaches a `q.Link` rate.

| Module | Imports `_core` | Page |
| --- | --- | --- |
| `qkd.budget` | no | [Noise budget](/guide/budget) |
| `qkd.impairments` | no | [Impairments](/guide/impairments) |
| `qkd.attacks` | yes, `tlo.rs` | [Attacks](/guide/impairments#attacks) |
| `qkd.reconcile` | yes, `cascade.rs`, written for it | [below](#reconciliation-keeps-three-quantities-apart) |
| `qkd.security` | yes | [the epsilon register](/guide/security#register) |
| `qkd.rates` | yes | asymptotic rates of the families with no component tree; no epsilon |
| `qkd.dps` | yes, `click.rs` | finite-key DPS on photon-number-resolving detectors, a different protocol from `q.DifferentialPhase` ([refusals](/roadmap#refusals)) |

### Attacks report two sets of numbers, and neither bounds the other {#attacks-report-two-sets-of-numbers-and-neither-bounds-the-other}

A `Reading` carries `observed`, what Alice and Bob's monitoring reports while the
attack runs, and `eve`, what the eavesdropper holds. Under every modelled attack the
observables sit at values an unattacked link would produce, so `Reading` has **no
key rate**: `key_rate`, `rate`, `key`, `secure`, `safe` and `margin` raise. Never
subtract the two. `mismatch_rate` does not compose with `flaws.rs`
([why](/guide/security#no-composition)).

### Reconciliation keeps three quantities apart

| Quantity | Is | Used as |
| --- | --- | --- |
| Code rate $R$ | information bits per channel use, a property of the code | selects a code |
| $f_{\text{EC}} \ge 1$ | leakage over the Shannon limit | **subtracted** in a key rate |
| $\beta \le 1$ | share of the capacity reached | **multiplies** $I_{AB}$ |

Beside them, the frame error rate: the probability a reconciled frame still differs.
Every function states which it returns. `bridge()` is the only conversion between
the two efficiencies, an identity rather than an approximation.

| Surface | Behaviour |
| --- | --- |
| `run_cascade()` | counts Cascade's disclosed parities: the mean leak per frame and the frames that kept a residual error. A sample mean with sampling error, not a bound |
| `CODES` | the published LDPC catalogue as its authors measured it |
| `pick()` | refuses when the SNR is under every threshold on file, rather than returning the lowest-rate code |
| Cascade efficiency | tabulated, not fitted. $f_{\text{EC}}$ is not monotone across the published grid, so a value between grid points raises |

Nothing here feeds `q.Link`; `fer` is supplied ([refusals](/roadmap#refusals)).

## Two engines that reuse the layers rather than extending them {#reuse}

| Engine | Reuses |
| --- | --- |
| Discrete modulation | the key-rate covariance path. An $M$-PSK constellation's marginal covariance equals a Gaussian modulation's of the same variance; only the certified correlation $z$ changes, so `dm_holevo` takes $z$ as an argument ([discrete modulation](/guide/protocols#discrete)) |
| The continuous-variable relay | Gaussian conditioning: two thermal-loss channels, a beamsplitter and two homodynes, conditioned through the Schur complement `State.condition` uses ([the relay](/guide/relay#cvmdi)) |

## Open scope questions

Each is a **swappable component axis**: an answer changes a default, not the
architecture.

| Question | Options | What it changes |
| --- | --- | --- |
| Protocol | Gaussian vs discrete modulation | both built, both on `q.Link`. On a `q.PhaseShiftKeying` link `q.Asymptotic` reaches the analytic bound and `q.CertifiedBound` reaches `dm_secure` |
| Detection | homodyne vs heterodyne | $\chi_{\text{det}}$, mutual information, and one or two quadratures recorded per symbol. Also the count the finite-size interval is built on: two pairs per heterodyne symbol, one per homodyne symbol |
| Local oscillator | transmitted vs locally generated | both described; only the local one runs a link. `q.Link` refuses `q.TransmittedLO`: the rate normalises to a unit an oscillator that crossed Eve's channel does not fix. A local oscillator needs pilot-based phase recovery, the DSP layer's job |
| Regime | asymptotic vs finite-size | whether parameter estimation is a confidence-interval computation or a substitution. Per family: [Security](/guide/security#asymptotic); refused lengths: [refusals](/roadmap#refusals) |
| Detector noise | trusted vs untrusted | key rate and maximum distance; a switch on the Gaussian path. The analytic discrete-modulation bound has no trusted form and refuses the flag; a relay detector has no trust flag, the topology being the security model; a `q.PairSource` has none, BBM92 trusting the source in no configuration |

Gaussian modulation, heterodyne detection, a local oscillator, finite size and
configurable trust form the smallest scope that still derives $\xi$ rather than
dialling it.

## Prior art

| Project | What it is | Relation |
| --- | --- | --- |
| [QOSST](https://github.com/qosst) / `qosst-sim` | open-source CV-QKD platform: locally generated LO, frequency-multiplexed pilots, RF-heterodyne detection, and the DSP and key-rate stack that drives real hardware | the closest reference point. QOSST runs an experiment and measures the noise; qkd predicts it from hardware parameters, over the same DSP structure |
| Strawberry Fields / The Walrus | Gaussian and Fock photonic simulation | the Gaussian layer overlaps. SF's `ThermalLossChannel(T, nbar)` can carry excess noise, but SF has no notion of excess noise as a quantity, no key rate and no QKD (zero hits for `qkd`, `key rate`, `BB84` or `holevo` across all four Xanadu sdists). SF was archived 2026-01-16; The Walrus survives |
| QuTiP, QuantumOptics.jl | general open-system dynamics | the reference for the deferred stochastic and Fock engines, not for the QKD path |

## Conventions

[Conventions](/guide/conventions), enforced by the
[`Conventions` exam](/tests/conventions).

## References

- Bennett, Brassard & Mermin, Phys. Rev. Lett. **68**, 557 (1992)
- Berta, Christandl, Colbeck, Renes & Renner, *Nature Physics* **6**, 659 (2010), [arXiv:0909.0950](https://arxiv.org/abs/0909.0950)
- Koashi, Phys. Rev. Lett. **93**, 120501 (2004)
- Maassen & Uffink, Phys. Rev. Lett. **60**, 1103 (1988)
- Matsuura, Sasaki & Koashi, Phys. Rev. A **99**, 042303 (2019), [arXiv:1812.10916](https://arxiv.org/abs/1812.10916)
- Papanastasiou, Ottaviani & Pirandola, Phys. Rev. A **96**, 042332 (2017), [arXiv:1707.04599](https://arxiv.org/abs/1707.04599)
- Tamaki & Lutkenhaus, Phys. Rev. A **69**, 032316 (2004)
- Tamaki, Lütkenhaus, Koashi & Batuwantudawe, Phys. Rev. A **80**, 032302 (2009)
