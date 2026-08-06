# Status and roadmap

Version 0.2.0.

## What runs through `q.Link` {#link}

| Protocol | Component | Derived, not dialled |
| --- | --- | --- |
| Gaussian modulation, homodyne and heterodyne | `q.GaussianModulation` | $\xi$, from hardware and the pilot DSP residual |
| Discrete modulation, $M$-PSK | `q.PhaseShiftKeying` | the certified Alice–Bob correlation, and the constellation's mutual information |
| Differential phase shift | `q.DifferentialPhase` | QBER, from fringe contrast, phase walk and dark counts |
| BB84 with weak coherent pulses | `q.BasisKeying`, `q.PolarisationKeying` | $Y_1$ and $e_1$, from measured decoy gains |
| Six-state | `q.BasisKeying(bases=3)` | the same $Y_1$/$e_1$, and a **tomographically complete** Bell-diagonal estimate |
| SARG04 | `q.BasisKeying(announce="pair")` | the same $Y_1$/$e_1$, the conclusive sifting fraction, and two cutoffs neither published equation states |
| Two-state keying (B92) | `q.TwoStateKeying` | on the plain branch, the **phase error**, from the loss and the overlap |
| Coherent one way | `q.IntensityKeying` | the data-line gain and error rate; the phase-error bound is an input |
| Decoy layer under all of the above | `q.Decoy` | the Ma–Qi–Zhao–Lo $Y_1$/$e_1$ bounds, protocol-blind |

Physics per row: [Protocol coverage](/guide/protocols).

| Narrower than it sounds | Detail |
| --- | --- |
| The click family is asymptotic **except BB84-WCP** | `q.SplittingAttack(block=q.KeyBlock(...))` runs Lim et al.'s composable finite-key length. Six-state, SARG04 and B92 **accept** the block and the engine refuses, naming the blocker; DPS and COW security components have no block field ([Security](/guide/security#finitekey)) |
| The [sampled pulse train](/guide/protocols#sampled) is BB84's alone | `q.Alice()` beside `bases=3` or `announce="pair"` raises |
| The DSP chain is heterodyne only and runs on the CPU | [the GPU path](#gpu) |
| Four budget rows reach `Link` automatically | the DSP **phase** term and its transmittance half; the **DAC** term on a link carrying a `q.DSP`; the **two RIN rows**. `q.Laser(rin=…)` and the detector `bandwidth` are charged together or refused by name. `q.ADC` is refused: the pilot DSP chain runs in floating point and models no quantiser, so a bit depth would be dropped. A detector `clip` is refused, naming two different claims: a noise term via `budget.assemble(adc_bits=…, ratio=det.full_scale())`, or an attack via `q.attacks.Saturation(alpha=det.clip, …)`, which returns a `Reading` and no rate. Everything else is assembled by hand onto `q.Channel(xi=…, ref="input")` |

## Beyond `q.Link` {#in-the-core-not-yet-in-the-api}

`q.Swap`, `q.PairLink`, `q.Network`, `qkd.fock` and the catalogues, with the engine each
reaches: [Architecture](/architecture#the-core-modules).

### The epsilon layer is a registry, not a derivation {#epsilon}

No two shipped families mean the same thing by "eps":
[the epsilon register](/guide/security#register).

## What has no Python surface {#core-only}

`herm.rs` and `lp.rs` are numerics under other engines, not protocols. `eur.rs` has
no surface: `eur_family` refuses every shipped family, naming the condition each
fails. `postselect.rs` reaches no shipped family ([below](#postselect)).
`cv_general` has no `q.` path ([below](#definetti)); nor has `b92_strong`, whose
receiver no `q.Bob` builds.

| Engine | Has no component tree because |
| --- | --- |
| `pairing.rs`, mode pairing / asynchronous MDI | the rate is bought by the pairing **span**, a temporal degree of freedom `q.Swap` has no notion of. The engine assumes a symmetric station where `q.Swap` carries two independent arms, so a tree would offer an asymmetry the engine refuses |
| `rrdps.rs`, round-robin DPS | RRDPS is *defined* by Bob picking a delay uniformly per round; `q.DelayInterferometer` carries one fixed delay. Three published cost assemblies ship side by side and no component field selects one |
| `sdp.rs`, COW′ | `q.IntensityKeying` describes the three-sequence COW source; COW′ reads a four-sequence record and is a different protocol |

### The postselection technique reaches no shipped family {#postselect}

`postselect.rs` implements Christandl, König & Renner (2009) in the corrected form
of Nahar, Tupkary, Zhao, Lütkenhaus & Tan (2024); omitting the correction moves the
number in the *insecure* direction. `ps_family` refuses every shipped family, each
naming the condition it fails:

| Family | Fails |
| --- | --- |
| BB84-WCP, MDI-BB84 | **C4**, vacuously: their finite-key proofs already hold against *general* attacks, and their epsilons are not the $(\varepsilon_{PA}, \bar\varepsilon, \varepsilon_{AT})$ triple the corollary consumes. A lifted number bounds nothing while looking conservative. Both also fail **C2** |
| Six-state | **C4**. `source="single"` is a qubit protocol, so $x = 16$, but Scarani & Renner's bound carries no such decomposition, and the decoy branch already refuses |
| B92, three-sequence COW, E91 | **C4** vacuously: no finite-key length to shorten. `_core.pair_length` (BBM92) refuses every `q.PairLink`. `_core.sarg_length` and `_core.dps_finite` fail C4 **non**-vacuously: their analyses already hold against general attacks. Four-sequence COW′'s `_core.sdp_length` already holds against coherent attacks via an entropic uncertainty relation |
| Mode pairing | **C1**, the only family failing on **permutation invariance** rather than dimension. A key bit pairs two clicked rounds chosen *after* the announcement within a maximal interval, so pairing depends on time order; symmetrising changes the protocol |
| RRDPS | **C2**. The exchangeable unit is the *packet*, which C1 accommodates, but nothing bounds the packet's dimension |
| Gaussian / CV | **C2** outright: infinite-dimensional modes, no squashing map. `cv_finite` is the one collective-only finite-key path; the CV route is a Gaussian de Finetti reduction, a different theorem ([`keyrate::cv_general`](#definetti)) |

Every optical family fails C2 on **both** sides: a phase-randomised weak-coherent
source and a threshold-detector receiver are each infinite-dimensional. A repair
needs NTZLT's tagged-state source map and their weight-preserving flag-state
squasher; they state the existing flag-state squasher cannot be used. Neither is
implemented.

### The Gaussian de Finetti reduction: the arithmetic ships, the protocol does not {#definetti}

Postselection costs $(n+1)^{d^2-1}$, superpolynomial in a dimension an optical mode
lacks. Trading $S_n$ for $U(n)$ lets an energy test truncate the Fock space
**globally**, at $K^4/50$ with $K$ linear in $n$. `keyrate::cv_general` implements
Leverrier (2017), Eq. (5), with `mode_energy`, `energy_cutoff`, `hash_toll` and
`collective_eps`. `cv_finite`'s body is the private unclamped `finite_core`;
`cv_finite` only clamps.

Eq. (5) is an implication; this computes its arithmetic, and `cv_finite` supplies
only part of the antecedent.

| Hypothesis | State here |
| --- | --- |
| $U(n)$ covariance, **enforced** by random conjugate linear-optical networks | `pipeline.rs` does not. Leverrier calls the step "computationally costly" and believes, without proof, it can be bypassed |
| The energy test as a **protocol step with an abort**, on $k$ modes *on top of* the $n$ keyed from | nothing in `qkd/` runs one. `k_test` buys the analysis and no hardware |
| `cv_finite`'s $\varepsilon$ being the $\varepsilon$ Eq. (5) consumes | **not shown.** Leverrier's is the composable diamond-distance parameter of Leverrier (2015); `cv_finite` is Devetak–Winter with a smooth-min-entropy correction whose three terms carry no correctness parameter |

Write it as *the reduction's arithmetic ships; the symmetrisation, the energy test
as a protocol step, and a composable collective proof underneath it do not*. Never
"implemented", never "not implemented". `cv_general` is reached by **no `q.`
path**; `cv_finite` remains the one collective-only finite-key path a caller can run.

Measured at $\varepsilon' = 10^{-10}$: the general-attack key is **90.7%** of the
collective one at $n = 10^9$, 99.0% at $10^{11}$, 99.6% at $10^{12}$. The hashing
toll is **248 bits against $3.08\times10^7$**. The cost is the $\varepsilon$
tightening: the minimum block size rises about **4–5×**.

The budget side is in **logarithms**: at the dimensions that matter the required
IID secrecy parameter underflows f64, and `0.0` reads as unconditional security.
`dmcs.rs`'s photon-number cutoff is **not** a squashing licence: `nc + 1` is not a
$d_B$ that C2 certifies, and passing it to `ps_dim` is insecure rather than merely
wrong.

### Three solvers, three cones {#solver}

No solver adds a dependency. Each carries its own `chol`/`chol_solve`; a shared
copy would sit beside barrier constants that must not be shared.

| Solver | Cone | For |
| --- | --- | --- |
| `sdp.rs` | a **linear** functional under linear matrix inequalities, every block dense, symmetric, single-digit dimension | the COW′ phase-error problem |
| `dmcs.rs::dm_secure`, on `herm.rs` | quantum **relative entropy** minimised over density operators | the discrete-modulation proof stack: both steps of Winick, Lütkenhaus & Coles (2018) for the protocol of Lin, Upadhyaya & Lütkenhaus (2019) |
| `lp.rs` | a linear programme over the **diagonal** cone; the Cholesky of the slack disappears | Curty's decoy grid for MDI-BB84 (`mdi::deck`) and Yin's multiphoton RRDPS bound after one reduction (`rrdps::simplex_cap`). `pairing.rs` does not use it |

All three are log-barrier path-following on the **dual**: every iterate is feasible
and its objective is a rigorous bound by weak duality. A stalled or early-stopped
run returns a **looser** bound, never a wrong one; a primal method reporting its
stopping vertex fails in the **insecure** direction. `sdp_phase` raises when the
path breaks down before the requested gap, when a zero gain leaves no interior,
when the dual value falls below the Cauchy–Schwarz floor, and when the certified
value fails to beat the closed form it tightens. `lp.rs` errs at or *below* the
primal minimum: consumers state a yield as a lower bound, an error or an
information as `min` of its negative.

::: danger Assembly trap in `lp.rs`
Never state one pair of inequalities as two rows carrying identical coefficients.
That leaves the dual an exactly null direction, the path runs $|y|$ out to
$\sim10^{19}$, and round-off in $Av = b$ is amplified past the objective: measured
returning **8.38** for a yield whose true value is $1.05\times10^{-2}$, certified,
with every dual slack positive. Use one row with a bounded slack; `mdi::deck` is the
worked form.
:::

`dm_secure` returns its two steps as **separate fields**: `bound` is step 2, the
Theorem 3 linearisation dualised, a lower bound at *any* dual-feasible point;
`upper` is step 1's Frank–Wolfe value and certifies nothing. It is rigorous **given
the photon-number cutoff**, which LUL19 call a working assumption; the dimension
reduction that removes it, Upadhyaya, van Himbeeck, Lin & Lütkenhaus (2021), is not
implemented. It **refuses pure loss** ($\xi = 0$), where $\rho_{AB}$ goes
rank-deficient.

Reach ($\alpha$ optimised, $\beta = 0.95$): `dm_rate` crosses zero at **26.5 km**
at $\xi = 0.01$ and **74 km** at $\xi = 0.005$; `dm_secure` reaches **142 km** and
**149 km** at a $10^{-4}$ bit/pulse floor. Figures of 225 to 341 km sit **below a
numerical floor**: the continuity price $\zeta$ is of order $10^{-6}$ bit at
$\varepsilon = 10^{-10}$ and $10^{-9}$ at $10^{-13}$, so under about $10^{-6}$
bit/pulse the crossing is an artefact of $\zeta$.

`q.CertifiedBound` reaches `dm_secure` from a `q.PhaseShiftKeying` link and returns
a `q.Certificate`: `key`, `bound` and `upper` are three different quantities, and
`key_rate` raises. `cutoff` has **no default**; the bound is rigorous only given it.
`viol` is the largest constraint violation at the step-1 state. Raise the cutoff
until `key` stops moving, and do not call the result cutoff-free.
What `sdp.rs` buys COW′: [Protocol coverage](/guide/protocols#cow-prime).

## Four settled exclusions {#exclusions}

**Decisions, not planned work.** Nothing below in [What is next](#next) addresses
them.

| Excluded | Reason |
| --- | --- |
| MDI-DPS / DPS-MDI | patented; purged from the repo entirely rather than left unimplemented |
| Twin-field, phase-matching, sending-or-not-sending | the closed-form rate every such paper plots pins the interference error, and that pinned number *is* the protocol: the residual of a phase reference shared between two free-running lasers a hundred kilometres apart, not an optical alignment that sits still. Pinning it assumes the protocol already worked. [Mode pairing](/guide/protocols#pairing) is **not** covered by this row: it is implemented, and reaches $O(\sqrt\eta)$ by a different route |
| Satellite and free-space channels | every rate function takes a scalar $T$; fading needs a *distribution* over it. Laser and fibre only: no turbulence, no pointing error, no sky background |
| Qudits and high-dimensional encodings | the click layer is two-detector throughout |

## What qkd will not do today {#refusals}

Every row raises and names its restriction.

| Protocol or capability | State |
| --- | --- |
| Finite-size security for **six-state** from `q.Link` | refused on the **source**, not the protocol. `sixstate_finite` is Scarani & Renner (2008) for a **single-photon** source, the only case they analyse; `q.BasisKeying(bases=3)` is weak-coherent, so `source="decoy"` refuses. The photon-number inversion is basis-blind and carries over, as does the pooled monitor statistic. All three routes fail at one place: Scarani & Renner's Lemma 3 wants an observed relative frequency on an identified sample of *one state*, and decoy hands over bounds on counts for a single-photon population nobody can point at round by round |
| Finite-size security for DPS from `q.Link` | refused. `qkd.dps.finite(..., detector=…)` runs Mizutani et al. (2023), against **general** attacks on photon-number-resolving detectors, a different protocol from `q.DifferentialPhase`'s train; `detector="threshold"` is refused. The Waks–Takesue–Yamamoto bound shipped on `q.Link` bounds a collision probability, not a smooth min-entropy, and has no finite-key form |
| Finite-size security for COW | not meaningful for the protocol `q.Link` ships: `e_phase` is an input, so the quantity a finite-size analysis bounds is supplied. The route is the vacuum-decoy variant, whose phase error [`sdp.rs` certifies](#solver): a *different protocol* |
| Finite-size security for **SARG04** on `q.Link` | `sarg_finite` refuses a block size; the length ships through `q.security.keylength` (`sarg_length`), which drops the two-photon $Q_2$ term — no counted bound on $s_2$ exists. Not implemented: Wang, Corrigan & Lutkenhaus, arXiv:2603.22448, Eq. (3) |
| Finite-size security for two-state keying | `b92_finite` refuses each branch for its own reason. **Plain B92**: the phase-error bound is an implicit constraint on rates maximised over a nuisance parameter, and a finite-key form needs a joint confidence region propagated through that maximisation. Three published finite-key B92 analyses exist and none covers the shipped protocol: Sasaki, Matsumoto & Uyematsu (ISIT 2015) and Bunandar et al. are **lossless**; Mafu, Garapo & Petruccione (2013) runs through smooth Rényi entropies rather than a phase-error bracket, and has **no arXiv version and no repository copy**. **Strong-reference B92**: Koashi's receiver takes a **supplied** $e_{ph}$ and the refusal calls the question malformed; the photon-number-resolving `_core.b92_strong` derives it, and its refusal names four missing pieces, of which the Appendix A deviation width ships as `_core.b92_azuma` |
| Finite-size security for a photon-pair link | **three proof gaps**. The length ships as `qkd.pairs.finite` (Tomamichel & Leverrier (2017), Theorem 3) and returns key at $m = 10^5$, $\varepsilon = 10^{-10}$ **at deterministic detection**. It refuses `detection="coincidence"`, which is every `q.PairLink`, naming each gap: TL17 Part II discharges the detection assumption for **Bob only**, replacing Alice's *measurement* with a *preparation*; $\bar c$ then needs an analyser bound, and the squashing model that supplies one **exists** (Gittsovich et al. (2014), Theorems 10/13/14) but $\bar c$ for TL17's *filtered* operators is unwritten; the per-sifted-round to per-pulse conversion is stated only on expectation and asymptotically |
| Finite-size security for discrete modulation | **all four pieces of Kanitschar Theorem 6 ship**: the dimension-reduction charge, the energy test, the acceptance test, and Eq. (21)'s relaxed feasible set as `_core.dm_relaxed`, whose minimum sits *below* the equality one (1.834309 against 1.894405 bit/kept round at 20 km), so substituting `certify()` over-claims by 3.17%. The **protocol** refuses: no `q.Link` sacrifices $k_T$ rounds, splits a block, or clips a detector at a soft limit, so `q.FiniteSize` is refused on this path |
| Trusted-detector discrete modulation on the **analytic** bound | `dm_rate` is Denys, Brown & Leverrier's closed form and fixes the receiver it is written for, so `Link._check_dm` refuses `trusted=True` there. `q.CertifiedBound` with `q.Heterodyne(..., trusted=True)` runs Lin & Lütkenhaus (2020), which folds detector noise into Bob's POVM rather than the state; `ln_radial` in `src/dmcs.rs` is the associated Laguerre polynomial its region operators and observables need |
| Homodyne discrete modulation | the Denys–Brown–Leverrier bound is written for heterodyne detection and no homodyne form is published |
| Collective- or general-attack bounds for phase keying | the WTY bound is individual attacks |
| A device-independent key rate | **a position, not a gap.** `ekert_rate` computes the device-independent *formula* and refuses `source="device-independent"`: the tight bound beyond CHSH is the NPA hierarchy, a **fourth** cone no shipped solver touches and one ruled out by decision; coherent attacks need entropy accumulation; coincidence post-selection **is** fair sampling. A claim needs detection efficiency above 0.924 (Pironio, Acín, Brunner, Gisin, Massar & Scarani (2009), Fig. 3) against `q.ClickDetector`'s 0.20. See [Protocol coverage](/guide/protocols#not-di) |
| A modelled CHSH violation read as a bound | `source="modelled"` prices one assumed state and bounds no unknown one; `ekert_point` computes that number under its own label |
| Reference-frame-independent QKD | no component and no bound. `q.ReferenceFrame` charges a link for a *drifting* polarisation frame; it does not implement the protocol that tolerates one |
| Asymmetric modulation on the covariance relay | the closed form carries one $V_A$ for both senders; refused rather than symmetrised |
| An asymmetric mode-pairing station | `pairing.rs` takes one `eta_s`; its source paper's simulation assumes $\eta_a = \eta_b$. The asymmetric variant is Lu, Wang, Li & Cao, [arXiv:2401.01727](https://arxiv.org/abs/2401.01727) |
| A symbol-level relay, two senders phase-locking through the midpoint | not implemented; `Swap.run()` takes no `symbols` and no `seed` |
| A **transmitted**-local-oscillator link end to end | `q.TransmittedLO` ships and constructs (`src/tlo.rs`: `shot`, `phase`, `leak`, `estimate`, `covariance`, `monitor`); `q.Link` refuses it naming **one** missing thing. `pipeline::run_tlo` draws the common-mode residual and charges **both** halves, the conserved product holding to 5.5e-4 relative. Missing: a shot-noise unit a key rate may be written in. An oscillator that crossed Eve's channel does not fix one; `shot(T, v_el)` returns the calibrated and operating units separately and nothing in `q.Link` reads the gap. Route: assemble both by hand and carry the total as `q.Channel(T=…, xi=…, ref="input")` |
| A component tree for the engines without one | [above](#core-only); each row states the component-shaped reason |
| The postselection technique on any shipped family | [above](#postselect); `ps_family` refuses every one by name |
| Intensity optimisation, anywhere | no optimiser exists. MDI-BB84's four- and seven-intensity protocols are basis-asymmetric intensity choices; every family is computed at the caller's pinned intensities, and the exams sweep fixed grids |
| A `q.Link` **rate** that reads a photon-number-resolving outcome | the detectors ship: `q.ThresholdArray` (multiplexed; `elements=1` reduces to `q.ClickDetector`'s click law) and `q.PnrDetector` (finite-resolution readout), each building its POVM through `povm()` and `outcomes(mu)`. `q.Link` refuses both by name: every bound it ships over a counting receiver is written on a **binary click** (`decoy_bounds` inverts click gains; the phase-error bounds are stated in a click error rate). `_core.b92_strong` and `qkd.dps.finite` are the two rates that read one |
| A `q.Network` edge carrying a photon-pair source | `q.Hop` constructs around one; `Network.run()` refuses it: "a hop carries a q.Link or a q.Swap, not PairLink" |
| Coincidence-window timing, and a continuous-wave-pumped pair source | `q.ClickDetector`'s timing fields are refused on the pair path: a coincidence window is a width *between* two receivers, and the pulsed gain carries no accidental-coincidence term |
| `flaws.rs` composed with `q.attacks.mismatch_rate` | both loss-tolerant analyses require Bob's inconclusive operator to be basis-independent, and a detection-efficiency mismatch violates that condition. The composition exists in the literature ([arXiv:2412.09684](https://arxiv.org/abs/2412.09684)) and needs its own proof and virtual states |
| A source-flaw link with a defaulted analysis or angle | `q.FlawBound(analysis=…)` has no default: the two analyses are different quantities, with return tuples shaped not to be subtracted. Neither `q.SourceFlaw` angle has a default: at `tilt = 0` Bob carries Alice's flaw and the phase error stops depending on `delta`, so a default would report a flawed source as flaw-free |
| Finite key on a `q.FlawBound` link | a declared `block` is refused by name |
| An attack composed with a key rate | `q.attacks` returns a `Reading`; `key_rate`, `rate`, `key`, `secure`, `safe` and `margin` are barred attributes |
| GPU dispatch from any Python-level API | kernels exist and are measured; nothing routes to them ([the GPU path](#gpu)) |
| FER-aware reconciliation end to end | `q.reconcile.run_cascade()` counts Cascade's disclosed parities and the frames that kept a residual error; nothing feeds that back, so `FiniteSize(fer=…)` takes a supplied number |
| `qkd.ideal`, zero-argument `q.Link()`, `res.frames` | **formally dropped** ([the disposition](/guide/link#dropped)). Of seven unshipped design-note surfaces, only `res.budget.at(plane)` was built |

## GPU kernels: measured, not dispatched {#gpu}

`test/gpu.py` checks each pipeline against the f64 CPU implementation.

| Pipeline | What it is |
| --- | --- |
| `words` | raw Threefry blocks, for comparison only |
| `walk` | the phase-walk block sums |
| `symbols` | the fused per-symbol kernel, one thread per 256-symbol DSP block |
| `store` / `stats` | the split store-versus-recompute variant, a benchmark arm |
| `threefry.wgsl` | a prelude concatenated ahead of each, not a pipeline |

| Quantity | Result |
| --- | --- |
| Threefry words vs CPU | **bit-exact**, integer equality across 4096 counters, every stage, and counters past $2^{32}$ |
| $\hat t$, $\hat\sigma^2$, pilot SNR vs f64 CPU | $1.3\times10^{-7}$, $9.9\times10^{-8}$, $3.6\times10^{-7}$ relative |
| $\xi$ vs f64 CPU | $8.0\times10^{-5}$ relative: $(\hat\sigma^2 - 1 - v_{el})$ amplifies the input error by roughly $1/(T\xi)$. Hence DSP in f32, covariance and key rate in f64 on the CPU |
| In-kernel throughput | 313–335 M symbol/s, against 10.3 M symbol/s for the CPU pipeline |
| Per-call fixed cost | $\approx 2.1$ ms, loaded machine |
| Crossover | between $10^3$ and $3\times10^3$ symbols |

| Caveat on the throughput rows | Detail |
| --- | --- |
| **Build** | **release** build, as `./do bench` produces. `./do develop` and `./do test` builds are roughly an order of magnitude slower and not comparable |
| **Load** | load 20–100: a `run_symbols` anchor read 5.2 M symbol/s against 10.3 M quiet. **No single-number speedup is claimed.** The quiet-machine crossover sits somewhat above the bracket, at a few thousand symbols |
| **Staleness** | the CPU rows predate later work on `run_symbols` and have not been re-measured together |

The prediction that "below $10^6$–$10^7$ symbols the CPU wins outright" is refuted
by about three orders of magnitude.

`q.Link` calls `pipeline::run_symbols`, which is CPU and rayon only.

| Stage on the CPU by decision | Reason |
| --- | --- |
| The exclusive scan of phase-walk carries | a monolithic f32 scan errs $4\times10^{-2}$ rad worst case, the size of the residual phase noise being measured. Portable WGSL lacks the forward-progress guarantees decoupled lookback needs |
| The pilot DSP across blocks: unwrapping, the CFO fit, interpolation | a scan across blocks rather than within one |

**The GPU computes the oracle branch, not the DSP-derotated one.**
`backend_info()` reports what *resolved*, not what ran.

## Known gaps {#gaps}

| Gap | State |
| --- | --- |
| The symbol pipeline is **compute-bound, not bandwidth-bound**: about 0.3% of memory bandwidth at roughly 85 ops/byte | buffer pooling bought 62 µs; the residual per-call cost is **two compute-pass launches**, which sets the crossover. The f32 precision argument (12–16-bit ADC samples) is unaffected |
| Naive f32 accumulation | 27% relative error at $n = 10^8$, growing linearly in $n$, not as $\sqrt n$. A GPU kernel requires a tree reduction with an f64 finish |
| The Tier B curve-fitting machinery of the [validation contract](/guide/validation): joint fits with free parameters bounded and reported | partially built; the [scoreboard](/guide/validation) holds the state |
| COW's phase-error bound on `q.Link` | supplied by the caller rather than derived from monitoring-line gains; the published sequential-attack bound's numeric coefficient is not reproduced. COW's *shape* and model-independent ceiling hold, with one input read from the literature. [`sdp.rs`](#solver) derives the quantity for the vacuum-decoy variant, a different protocol with no component |
| SARG04's two-photon term is reported **absent** | the Ma–Qi–Zhao–Lo inversion bounds $Y_1$ and nothing about $Y_2$, so the shipped rate sits below the infinite-decoy one the literature publishes. No finite-decoy $Y_2$ bound is invented; `sarg_yield` computes the infinite-decoy limit its source paper takes explicitly |
| One line of Tamaki's Appendix C is **unreconciled** | his $\delta \to 3\delta/2$ substitution lands at $3\delta/4$ where composing the two rotations explicitly lands at $\delta$. The explicit composition runs, and reproduces Wang's printed device model; the factor was not retrievable from either paper. The difference is $4/3$ in an angle whose effect is second order |
| B92's plain branch has **one seam** | Tamaki & Lütkenhaus is a single-photon proof and `b92_detect` is a coherent-state receiver. The rate is a defensible number for weak coherent pulses, not a security proof for them; their conclusion names coherent-state B92 open. `b92_plain` has no seam |

## The order things land in, and why {#the-order-things-land-in-and-why}

| Order | Because |
| --- | --- |
| Decoy before the protocols that use it | the $Y_1$/$e_1$ inversion is **protocol-blind**; BB84-WCP, six-state and SARG04 consume it and differ at the privacy-amplification factor alone; MDI-BB84 needs the [two-dimensional](/guide/protocols#mdi-decoy) one |
| Click physics before COW, BB84-WCP and B92 | the same optical train through the same threshold detectors; only Alice's encoding and Bob's sifting differ. See [Architecture](/architecture#the-click-detector-family) |
| `mdi.rs` before `pairing.rs` | mode pairing's phase error **is** `mdi_yield`'s second return at a symmetric station, passed in rather than recomputed |
| `pairs.rs` before `ekert.rs` | E91 and BBM92 differ only in what prices Eve; the CHSH branch stands where the phase error stood, and `ekert_point` returns both numbers side by side |
| The covariance relay with the covariance engine | it needed only Phase 1's Gaussian conditioning |
| The Fock layer independently | nothing on the QKD path waits on it; it serves the general CV-simulator goal |
| The Python layer last, per engine | a formula is anchored against a paper in the core without constructing a link, then grows a component. Engines that have not: [above](#core-only) |

## What is next {#next}

| # | Item | What is missing |
| --- | --- | --- |
| 1 | **Finite-size security outside Gaussian modulation**, the largest gap | BB84-WCP and MDI-BB84 report it on a component tree; `q.security.keylength` reports a key **length** for families with none. `sixstate_finite` stays in `qkd._core` for its single-photon-source restriction. Analysis gaps [above](#refusals): a per-round single-photon state for six-state on a decoy source, three pieces for SARG04, a bound with no smooth-min-entropy form for DPS on `q.Link`, a supplied phase error for COW, coincidence statistics for the pair link, a nuisance-parameter maximisation for B92 |
| 2 | **The symbol-level relay** | two independent senders phase-locking through an untrusted midpoint; simulating the lock rather than assuming it |
| 3 | **A GPU dispatch that answers the right question** | the measured kernels compute the oracle branch; a routable path must carry the DSP-derotated branch. **Ceiling 51.6×, defensible bracket 20–50×** at $n \ge 10^6$, new WGSL plus a dispatch policy. **Accuracy is not the obstacle**: `xi_hat - xi_ideal` runs 3.0e-2 against an f32 floor of 8.0e-5, 4.6 decades clear, and the pilot chain differs by ≤1.96e-6 rad in f32; the 4.0e-2 rad hazard is specific to a *monolithic* prefix scan, the chunked f32 scan with f64 host carries reading 4.8e-5 rad. The obstacles are structural: a third compute pass (`psi` needs every block's phase before any symbol derotates), `unwrap_arg` as a second carry scan, 64-bit fixed-point pilot accumulation via emulated `mul_hi`, and exams pinning exact draws keeping the CPU path. It **moves published numbers** (~1e-7 on DSP statistics, 8.0e-5 on $\xi$), and retiring the accuracy objection is **contested** by `CLAUDE.md`'s f32/f64 line, not adopted policy. **Next step is a measurement**: extend `src/symbols.wgsl` with the derotation arithmetic alone, feed it a host-computed `psi`, and time it against today's kernel |
| 4 | **Closing the reconciliation loop** | `q.reconcile` counts Cascade's disclosed parities and carries the published LDPC catalogue; deriving `fer` from a code and feeding it into `q.FiniteSize` is missing |

## Open scope questions {#open}

Five axes stay swappable: protocol, detection, local oscillator, asymptotic versus
finite-size, trusted versus untrusted detector noise
([Architecture](/architecture#open-scope-questions)). Open: whether finite-size
becomes the default reporting mode across the matrix.

## References

- Christandl, König & Renner, Phys. Rev. Lett. **102**, 020504 (2009), [arXiv:0809.3019](https://arxiv.org/abs/0809.3019)
- Curty, Xu, Cui, Lim, Tamaki & Lo, Nat. Commun. **5**, 3732 (2014)
- Gittsovich et al., Phys. Rev. A **89**, 012325 (2014)
- Leverrier, Phys. Rev. Lett. **114**, 070501 (2015), [arXiv:1408.5689](https://arxiv.org/abs/1408.5689)
- Leverrier, "Security of continuous-variable quantum key distribution via a Gaussian de Finetti reduction", Phys. Rev. Lett. **118**, 200501 (2017), [arXiv:1701.03393](https://arxiv.org/abs/1701.03393)
- Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A **89**, 022307 (2014)
- Lin & Lütkenhaus, Phys. Rev. Applied **14**, 064030 (2020), [arXiv:2006.06166](https://arxiv.org/abs/2006.06166)
- Lin, Upadhyaya & Lütkenhaus, PRX **9**, 041064 (2019), [arXiv:1905.10896](https://arxiv.org/abs/1905.10896)
- Lu, Wang, Li & Cao, [arXiv:2401.01727](https://arxiv.org/abs/2401.01727)
- Mafu, Garapo & Petruccione, Phys. Rev. A **88**, 062306 (2013) — no arXiv version
- Mizutani et al., PRResearch **5**, 023132 (2023)
- Nahar, Tupkary, Zhao, Lütkenhaus & Tan, PRX Quantum **5**, 040315 (2024), [arXiv:2403.11851](https://arxiv.org/abs/2403.11851)
- Pironio, Acín, Brunner, Gisin, Massar & Scarani, New J. Phys. **11**, 045021 (2009)
- Sasaki, Matsumoto & Uyematsu, ISIT 2015
- Scarani & Renner, Phys. Rev. Lett. **100**, 200501 (2008), [arXiv:0708.0709](https://arxiv.org/abs/0708.0709)
- Tomamichel & Leverrier, Quantum **1**, 14 (2017), [arXiv:1506.08458](https://arxiv.org/abs/1506.08458)
- Upadhyaya, van Himbeeck, Lin & Lütkenhaus, PRX Quantum **2**, 020325 (2021), [arXiv:2101.05799](https://arxiv.org/abs/2101.05799)
- Winick, Lütkenhaus & Coles, Quantum **2**, 77 (2018), [arXiv:1710.05511](https://arxiv.org/abs/1710.05511)
