# Protocol layer

`q.Link` describes an Alice → channel → Bob path of frozen component dataclasses;
`run()` returns a key rate. The modulation fixes the protocol; everything else is a
component swap. Coverage: [Protocol coverage](/guide/protocols#matrix).

Quantities are in **shot-noise units** ([the SNU boundary](/guide/conventions#the-snu-boundary)).
Every `Link` run is **CPU and f64**, including `run(symbols=…)`
([Architecture](/architecture)). An untrusted relay is
[`q.Swap`](/guide/relay#swap); a photon-pair source is `q.PairLink`.

```python
import qkd as q
```

## The component tree

```python
link = q.Link(
    modulation=q.GaussianModulation(v_a=5.0),
    channel=q.Fiber(length=25.0, alpha=0.2),
    bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True)),
    security=q.Asymptotic(beta=0.95),
)

res = link.run()
res.key_rate     # 0.09947 bit/symbol
```

`modulation`, `channel` and `bob` are required; `alice`, `dsp`, `security`,
`impairments` and `losses` are optional, `security` defaulting to
`Asymptotic(beta=0.95)`. Components validate at construction:

```python
q.Fiber(length=25.0, T=0.3)
# ValueError: give hardware description or derived value, not both

q.Channel(T=0.5, xi=0.01)
# ValueError: nonzero xi requires ref='input' or ref='output'
```

**`Link.__init__` validates nothing across components**: a mismatched tree
constructs and raises on the first `run()`, `measure()`, `claim()` or `explain()`
([what runs today](#what-runs-today)).

## Component reference

Defaults describe a metro link; sources:
[Validation](/guide/validation#where-the-defaults-come-from). **required** means no
default; `None` means *derive*.

### The default link {#defaults}

| Component | Field | Default | Unit |
| --- | --- | --- | --- |
| `GaussianModulation` | `v_a` | $5.0$ | SNU, at the channel input |
| `Alice` | `symbol_rate` | $10^8$ ($100$ MBd) | Bd |
| `Laser` | `linewidth` | $10^4$ ($10$ kHz) | Hz |
| `Laser` | `rin` | `None` — no intensity noise | dBc/Hz |
| `LocalLO` | `linewidth` | $10^4$ ($10$ kHz) | Hz |
| `IQModulator` | `bits` | $16$ | bit (DAC) |
| `ADC` | `bits` | $12$ | bit |
| `Pilots` | `power_db` | $12.0$ | dB over the per-quadrature modulation variance |
| `Pilots` | `tones` | $1$ | — |
| `Pilots` | `freq` | $1.8 \times 10^8$ ($180$ MHz) | Hz |
| `Fiber` | `alpha` | $0.2$ | dB/km |
| `Fiber` | `length` | `None` — one of `length` or `T` is required; $25$ km in this page's examples | km |
| `Heterodyne` / `Homodyne` | `eta` | $0.6$ | — |
| `Heterodyne` / `Homodyne` | `v_el` | $0.1$ | SNU, at Bob's plane |
| `Heterodyne` / `Homodyne` | `trusted` | `True` | — |
| `DSP` | `block` | $32$ | symbols |
| `PilotPhase` | `v_err` | `None` — derived from the symbol pipeline | rad² |
| `Asymptotic` / `FiniteSize` | `beta` | $0.95$ | — |
| `FiniteSize` | `eps` | $10^{-10}$ | — |
| `FiniteSize` | `n` | $10^{9}$ | symbols |
| `FiniteSize` | `pe_fraction` | $0.5$ | — |
| `FiniteSize` | `fer` | `None` — not modelled | — |

$\xi$ is an output; only a fully pinned `q.Channel` accepts one. Measured envelope:
$\xi \approx 0.005$–$0.03$ SNU at the channel input over $25$ km
([Tier B](/guide/validation#tier-b)). Click and discrete-modulation defaults are
below.

### Modulation

`modulation` selects the protocol family, and with it the legal receiver, the
security model and the populated `LinkResult` fields.

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `GaussianModulation.v_a` | SNU | `5.0` | $V_A > 0$ per quadrature, at the **channel input**. |
| `PhaseShiftKeying.states` | — | `4` | Constellation size $m \ge 3$: a two-state alphabet sits on a line, so its $x$ and $p$ variances differ and the analysis rests on a covariance whose blocks are multiples of the identity. |
| `PhaseShiftKeying.alpha` | photon amplitude | `0.4` | Modulus of each coherent amplitude, **not** a quadrature. $\alpha > 0$; derived `v_a` is $2\alpha^2$ SNU and `bits` is $\log_2 m$. |
| `DifferentialPhase.mu` | photons/pulse | `0.2` | $\mu > 0$, with **no upper bound**: $\mu \ge 1/2$ stays constructible and the WTY $(1-2\mu)$ concession clamps the rate to zero rather than the component refusing. |
| `TwoStateKeying.mu` | photons/pulse | `0.23` | Signal mean photon number, $\mu > 0$, fixing the overlap $\langle\alpha\lvert-\alpha\rangle = e^{-2\mu}$, the only dial. Weak states coincide and Bob rarely discriminates; bright ones separate and Eve discriminates for him. |
| `TwoStateKeying.reference` | — | `False` | Whether Alice sends a bright phase reference. `False` **derives** the phase-error rate from loss and overlap; `True` has no closed form and takes it on `q.DiscriminationBound(e_phase=…)`. |
| `BasisKeying.decoy` | — | `Decoy()` | The intensity set. No decoy-free variant: without one the single-photon yield is unbounded. |
| `BasisKeying.sift` | — | `None` | Fraction of slots surviving basis reconciliation, $\in (0, 1]$, multiplying the rate. `None` takes the uniform value for `bases`; a value is a biased choice. `announce="pair"` refuses one, its conclusive fraction sitting inside the gains. |
| `BasisKeying.bases` | — | `2` | `2` is BB84; `3` the three mutually unbiased bases. Nothing else. |
| `BasisKeying.announce` | — | `"basis"` | `"basis"` is BB84 and six-state; `"pair"` announces a non-orthogonal **pair** containing Alice's state, and Bob keeps rounds whose outcome excludes a member. |
| `PolarisationKeying.decoy` | — | `Decoy()` | As `BasisKeying.decoy`. |
| `PolarisationKeying.sift` | — | `0.5` | As `BasisKeying.sift`. |
| `PolarisationKeying.frame` | — | `ReferenceFrame()` | How Bob's polarisation reference is held against the fibre. The default is a perfectly held frame and reproduces `BasisKeying` **bit for bit**. See [polarisation encoding](/guide/protocols#polarisation). |
| `IntensityKeying.mu` | photons/pulse | `0.5` | Mean photon number of a *filled* slot, $\mu > 0$. Fixed, the rate falls linearly in $T$; the sequential-attack analysis is expressed by sweeping it against distance. |
| `IntensityKeying.decoy_frac` | — | `0.1` | Fraction of sequences carrying the monitoring pattern and therefore no key, $\in [0, 1)$. |
| `IntensityKeying.extinction` | dB | `None` | Modulator on/off contrast. `None`: the empty slot is empty. A value must be $> 0$; at $0$ dB the bit is not encoded. [Finite extinction](/guide/protocols#extinction). |
| `Decoy.intensities` | photons/pulse | `(0.5, 0.1, 0.0)` | $(\mu, \nu_1, \nu_2)$ with $0 \le \nu_2 < \nu_1$ and $\nu_1 + \nu_2 < \mu$, keeping the single-photon bound's denominator $(\nu_1-\nu_2)(\mu-\nu_1-\nu_2)$ positive. A trailing $0.0$ is vacuum-plus-weak, the tightest two-decoy choice. |
| `Decoy.probs` | — | `None` | Read by the [sampled path](/guide/protocols#sampled) alone. `None` is an equal three-way split; a value is three probabilities summing to $1$. |

#### The polarisation reference frame

Read by `PolarisationKeying` alone. The two `tracking` models take **disjoint**
parameters; a field of the other model raises.

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `ReferenceFrame.tracking` | — | `"tracked"` | `"tracked"` is a compensated receiver described by its residual `drift`; `"free"` is uncompensated, described by `rate` and `interval`. Nothing else is accepted. |
| `ReferenceFrame.drift` | rad | `0.0` | Residual rms **Jones** angle $\sigma$ after compensation. `tracking="tracked"` only. Not a Poincaré angle — a Jones rotation by $\theta$ is $2\theta$ on the sphere. |
| `ReferenceFrame.rate` | rad/$\sqrt{\text{s}}$ | `0.0` | Diffusion coefficient of a free-running frame. `tracking="free"` only, and then required positive. |
| `ReferenceFrame.interval` | s | `0.0` | Time since the frame was last aligned. `tracking="free"` only, and then required positive. |
| `ReferenceFrame.dispersion` | s/$\sqrt{\text{km}}$ | `0.0` | Fibre PMD coefficient $D_{\text{PMD}}$, accumulated as $D_{\text{PMD}}\sqrt{L}$; needs a `q.Fiber(length=…)` and refuses a pinned `q.Channel`. |
| `ReferenceFrame.width` | s | `0.0` | Pulse intensity-envelope rms width. Read with `dispersion`; one without the other raises. |

### Channel

`Fiber` describes hardware and derives $T$; `Channel` pins the whole channel. $\xi$
at the channel input and at Bob's plane differ by a factor of $T$
([Trap 1](/guide/gaussian#thermal-loss-and-the-required-ref-plane)).

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `Fiber.length` | km | — | `None` | $L > 0$. Mutually exclusive with `T`; one of the two is required. |
| `Fiber.alpha` | dB/km | — | `0.2` | $\alpha \ge 0$, giving $T = 10^{-\alpha L/10}$. |
| `Fiber.T` | — | — | `None` | Transmittance pinned directly, $T \in (0, 1]$, with no fibre model. |
| `Channel.T` | — | — | **required** | $T \in (0, 1]$. |
| `Channel.xi` | SNU | **`ref`** | `0.0` | $\xi \ge 0$ at the plane `ref` names. Stored as written; converted at link compile and labelled `derived` in `explain()["xi_input"]`. |
| `Channel.ref` | — | — | `None` | `"input"` (Alice's side) or `"output"` (Bob's side). Required whenever $\xi \ne 0$. |

Everything downstream is input-referred: `ref="output"` converts once, as
$\xi_{\text{input}} = \xi_{\text{output}}/T$.

### Optical losses

`losses=` is an optional tuple of interfaces, each multiplying $T$ on its own named
line. `site` sets the plane: the same decibel at Alice's output and at Bob's input
gives the *same* $T$ and *different* input-referred excess noise
([the loss chain](/guide/budget#the-loss-chain)).

```python
link = q.Link(
    modulation=q.GaussianModulation(v_a=5.0),
    channel=q.Fiber(length=25.0),
    bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
    losses=(
        q.Connector(loss=0.25, count=2, site="launch"),
        q.Splice(loss=0.02, count=6, site="span"),
        q.Coupling(loss=1.5, site="receive"),
    ),
)
```

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `Connector.loss` | dB | `0.25` | Insertion loss of **one** mated pair. The IEC 61753-1 grade C mean, and Thorlabs' typical for FC/APC. |
| `Splice.loss` | dB | `0.02` | Insertion loss of **one** fusion splice. The measured SMF-28 Ultra mean at 1550 nm on a core-aligning splicer. |
| `Coupling.loss` | dB | **required** | Insertion loss of **one** mode-crossing interface. No default: published values run from 0.8 dB to 3.1 dB and up. |
| `*.count` | — | `1` | Identical interfaces at this site, $\ge 1$. The row's total is `loss * count` on **one** named line, never `count` anonymous ones. |
| `*.site` | — | `"launch"` (`"span"` for `Splice`) | `"launch"` is Alice's output, `"receive"` Bob's input, `"span"` the fibre run between. |

```python
q.Connector(loss=-0.1)                    # ValueError: loss is an insertion loss
                                          #   in dB and must be nonnegative
q.Coupling(loss=1.0, site="middle")       # ValueError: site must be 'launch',
                                          #   'span' or 'receive'
```

A second descriptor of the **same kind at the same site** raises; `count=` declares
multiplicity. `losses=` needs a `q.Fiber` span:

```python
q.Link(..., channel=q.Channel(T=0.5, xi=0.01, ref="output"),
       losses=(q.Connector(),)).run()
# ValueError: itemised losses need a q.Fiber span: a pinned q.Channel already
#             fixes the whole transmittance and the plane its ref= names
```

### Detectors

`trusted` is a claim about the adversary, not a hardware number:
[Trap 2](/guide/security#trusted-vs-untrusted-is-a-security-model).

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `Homodyne.eta` / `Heterodyne.eta` | — | Bob | `0.6` | $\eta \in (0, 1]$. |
| `Homodyne.v_el` / `Heterodyne.v_el` | SNU | **Bob** | `0.1` | $v_{el} \ge 0$ at Bob's plane, never converted. Not a budget row ([why](/guide/budget#why-v-el-is-not-a-budget-line)); reaches the rate through $\chi_{\text{det}}$. |
| `Homodyne.clip` / `Heterodyne.clip` | $\sigma_{\text{shot}}$ | Bob | `None` | Linear half-range. `None` is the unbounded idealisation every rate assumes; a value is [refused by name](#what-runs-today). `full_scale()` returns $2\,\text{clip}$, `budget.adc`'s `ratio`; the same number is `q.attacks.Saturation`'s `alpha`. |
| `Homodyne.bandwidth` / `Heterodyne.bandwidth` | Hz | Bob | `None` | Read by the two RIN rows and `budget.assemble`'s Raman row; on `q.Link` charged only as half of the RIN pair, so a bandwidth without `q.Laser(rin=…)` is refused. |
| `Homodyne.trusted` / `Heterodyne.trusted` | — | — | `True` | Whether Eve is denied $\eta$ and $v_{el}$. Unvalidated. `trusted=False` folds the pair into the channel as $T \to \eta T$, $\xi \to \xi + \mu v_{el}/(\eta T)$. |
| `ClickDetector.eta` | — | Bob | `0.2` | $\eta \in (0, 1]$. |
| `ClickDetector.dark` | per gate, **per detector** | Bob | `1e-6` | $p_d \in [0, 1)$ of **one** detector in **one** gate, **not** the background yield $Y_0 \approx 2p_d$ that `Link` derives ([the convention](/guide/protocols#sampled)). For the GYS hardware pass $8.5\times10^{-7}$, never $1.7\times10^{-6}$. Not an excess noise. A datasheet rate in Hz enters through `q.ClickDetector.gated(rate, gate)`, $1 - e^{-Rt_g}$ over the **gate width**, not the clock period or `window`. |
| `ClickDetector.dead_time` | s | — | `0.0` | Blind interval after a click, $\ge 0$. Simulated click train only. |
| `ClickDetector.afterpulse` | — | — | `0.0` | Spurious click on the first live gate after a dead interval, $\in [0, 1)$. Simulated click train only. |
| `ClickDetector.jitter` | s | Bob | `0.0` | FWHM of the Gaussian core of the arrival-time response, $\ge 0$. `0.0` reproduces a jitter-free run **bit for bit**. Simulated path only ([timing jitter](/guide/protocols#jitter)). |
| `ClickDetector.window` | s | Bob | `0.0` | Acceptance window inside one symbol period, $\ge 0$, at most the period. `0.0` is contiguous binning: nothing is gated away and only bin migration remains. |
| `ClickDetector.tail_frac` | — | Bob | `0.0` | Weight $f$ of the one-sided exponential diffusion tail, $\in [0, 1]$. `0.0` is purely Gaussian: window loss and **no** bin error at any realistic width. |
| `ClickDetector.tail_time` | s | Bob | `0.0` | Time constant $\tau$ of that tail, $\ge 0$. Required positive whenever `tail_frac` is. |

### Receivers

`Bob.receiver` is the passive optics before the detector; `None` for a quadrature
receiver.

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `NullingReceiver.visibility` | — | `1.0` | Fringe contrast of the displacement interference, $\in [0, 1]$; weak-flux bit error $(1-V)/2$. **One gate per pulse**: `ClickDetector.dark` is per pulse and no two-port $Y_0$ is derived, uniquely among click receivers. |
| `DelayInterferometer.delay` | symbol periods | `1` | Long-arm delay $d \ge 1$, so slot $k$ beats against slot $k-d$. |
| `DelayInterferometer.visibility` | — | `1.0` | The interferometer's **own** fringe contrast $V \in [0, 1]$, before the laser phase walk. |
| `BasisAnalyser.misalign` | — | `0.033` | Probability a photon in the matching basis hits the wrong detector, $\in [0, 0.5]$. A **receiver** property. |
| `CoherenceMonitor.split` | — | `0.1` | Fraction of light tapped to the monitoring interferometer, $\in [0, 1)$. |
| `CoherenceMonitor.misalign` | — | `0.02` | Monitoring-line misalignment, $\in [0, 0.5]$. A diagnostic, never a security parameter ([intensity keying](/guide/protocols#cow)). |

### The parties

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `Alice.laser` | — | `None` | A `Laser`. Required to derive `v_err`. |
| `Alice.iq` | — | `None` | An `IQModulator`. **Charged when a `q.DSP` is present**, as a `dac` budget row on the estimate, so `bits=` moves the key rate. Without `dsp=` it is refused by name. |
| `Alice.pilots` | — | `None` | A `Pilots` tone set. Required to derive `v_err`. |
| `Alice.symbol_rate` | Hz | `100e6` | $f_{\text{sym}} > 0$. Consumed only when a pipeline runs. |
| `Bob.detector` | — | **required** | `Homodyne`, `Heterodyne` or `ClickDetector`. |
| `Bob.receiver` | — | `None` | `DelayInterferometer`, `BasisAnalyser`, `CoherenceMonitor` or `NullingReceiver`. The modulation fixes which is required; its presence decides whether a click error rate is simulated or pinned. |
| `Bob.adc` | — | `None` | An `ADC`. **Refused by name** on every `q.Link` path: the pilot DSP chain models no quantiser ([roadmap](/roadmap#link)). |
| `Bob.lo` | — | `None` | A `LocalLO` or a `TransmittedLO`. The DSP chain requires `LocalLO` to derive $V_{\text{err}}$; under a pinned `PilotPhase(v_err=…)` it is accepted and not read. A `TransmittedLO` **constructs and is refused** on every path. |
| `TransmittedLO.photons` | — | `1e9` | Oscillator photons per pulse at Alice's output. |
| `TransmittedLO.mux` | — | `1.0` | Extra transmittance of the oscillator arm alone. |
| `TransmittedLO.calibrated` | — | `None` | The oscillator photon number the receiver's shot-noise unit was characterised at. `None` means the operating one, i.e. no calibration discrepancy. |
| `TransmittedLO.delay` | s | `0.0` | The multiplexing delay, the **only** surviving phase term: signal and oscillator share one laser and one fibre. The linewidth multiplying it is `q.Alice(laser=q.Laser(linewidth=…))`. |
| `TransmittedLO.extinction` | — | `None` | Finite isolation between the two multiplexed modes, driving `leak()`. |
| `Laser.linewidth` | Hz | `10e3` | Lorentzian $\Delta\nu \ge 0$, driving the Wiener phase walk. |
| `Laser.rin` | dBc/Hz | `None` | `None` **declares no intensity noise**; a value is **charged** as `budget.rin_sig` and `budget.rin_lo`, which need a detection bandwidth. Either half alone is [refused by name](#what-runs-today). |
| `Laser.carrier` | Hz | `None` | Centre optical frequency, about $193.4\times10^{12}$ at 1550 nm. Read only as a **difference** against `LocalLO.carrier`: both ends name one or neither. `None` at both is the frequency-degenerate limit, offset zero. |
| `LocalLO.linewidth` | Hz | `10e3` | $\ge 0$. Beats against Alice's laser. |
| `LocalLO.carrier` | Hz | `None` | `beat(laser)` returns `laser.carrier - self.carrier`, **signed**, unclamped; the pipeline refuses an offset at or past Nyquist. One end alone raises, naming the missing end. |
| `IQModulator.bits` | bit | `16` | DAC resolution $\ge 1$. |
| `ADC.bits` | bit | `12` | Digitiser resolution $\ge 1$. |
| `Pilots.power_db` | dB | `12.0` | Tone power over the per-quadrature modulation variance. Unvalidated: the pilot does not compete for DAC range in the simulation. |
| `Pilots.tones` | — | `1` | $\ge 1$; above `1` raises `NotImplementedError` (no second-tone clock recovery). |
| `Pilots.freq` | Hz | `180e6` | Optical-baseband frequency $> 0$, outside the quantum band. |

### DSP {#dsp}

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `DSP.phase` | — | `PilotPhase()` | Phase recovery. |
| `DSP.block` | symbols | `32` | Symbols per phase-estimation block, $\ge 2$. Small blocks track a broad laser but average less pilot noise: the knob trading the two error terms of $V_{\text{err}}$. |
| `PilotPhase.v_err` | rad² | `None` | `None` derives it from the symbol pipeline; a value $\ge 0$ pins it and generates no symbols. |

The carrier-frequency offset has **no `q.DSP` parameter**: it is
`LocalLO.beat(laser)`, zero when neither end names a carrier. The stage always runs:
coarse acquisition off the pilot's delay-one autocorrelation, then the least-squares
slope of the residual block phases, resolving offsets below the coarse grid. The
offset is **signed**, drives a phase ramp
$\theta \mathrel{+}= 2\pi\,\nu_{\text{cfo}}k/f_{\text{sym}}$, and must satisfy
$2\lvert\nu_{\text{cfo}}\rvert < f_{\text{sym}}$; at zero the run is bit for bit the
run without the stage. `res.dsp.cfo` reports the fit in Hz, signed.

### Security

The security component is **per protocol family**; `Link` refuses a mismatched pair
([Security](/guide/security#other-click)).

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `Asymptotic.beta` | — | `0.95` | $\beta \in (0, 1]$. |
| `FiniteSize.beta` | — | `0.95` | $\beta \in (0, 1]$. |
| `FiniteSize.eps` | — | `1e-10` | Failure probability of **each** step of the $\varepsilon$ budget, $\in (0, 1)$. Not split between $\varepsilon_{PE}$, $\bar\varepsilon$ and $\varepsilon_{PA}$. |
| `FiniteSize.n` | symbols | `1e9` | Declared block size $N > 0$. |
| `FiniteSize.pe_fraction` | — | `0.5` | Share of the block disclosed for parameter estimation, $\in (0, 1)$. Never recovered: $n/N$ multiplies the rate. |
| `FiniteSize.fer` | — | `None` | **No numeric default**: a FER belongs to one LDPC code at one SNR and one $\beta$ (Hajomer: FER $0.59$ at $\beta = 0.925$, rate-0.05 MET-LDPC). `None` is *not modelled*, labelled so in `explain()`; a value scales the rate by $(1-\text{FER})$. |
| `IndividualAttack.qber` | — | `None` | $\in [0, 0.5)$. `None` derives it from the simulated interferometer and detectors; a value pins it and runs the closed form. |
| `IndividualAttack.f` | — | `1.16` | Error-correction inefficiency over the Shannon limit, $\ge 1$. |
| `SplittingAttack.f` | — | `1.22` | Error-correction inefficiency, $\ge 1$. |
| `SplittingAttack.block` | — | `None` | `None` is the asymptotic GLLP rate (Gottesman, Lo, Lütkenhaus & Preskill, *QIC* **5**, 325 (2004)). A `q.KeyBlock` gives a finite key **length**, never above the asymptotic rate ([finite key for BB84-WCP](/guide/security#finitekey)). |
| `KeyBlock.n` | pulses | `1e10` | Pulses **emitted**, decoy and test-basis slots included. On a sampled run it must equal `run(symbols=…)`. |
| `KeyBlock.eps_sec` | — | `1e-10` | Secrecy parameter, $\varepsilon_{sec} = 21\varepsilon$: each of the twenty-one bounds is taken at $\varepsilon_{sec}/21$. |
| `KeyBlock.eps_cor` | — | `1e-15` | Correctness parameter, a declared input with no simulated counterpart, separate from `eps_sec`. |
| `KeyBlock.fer` | — | `None` | As `FiniteSize.fer`, scaling the reported **length**. |
| `DiscriminationBound.e_phase` | — | `None` | On `reference=False`, `None` **derives** the phase-error rate from overlap, loss and bit error; a value is a **floor** under it, a looser one honoured, a tighter one not. On `reference=True` it is the supplied bound, as `PhaseBound.e_phase` is, and omitting it raises. |
| `DiscriminationBound.f` | — | `1.16` | $\ge 1$. The published two-state thresholds are at $f = 1$. |
| `PhaseBound.e_phase` | — | **required** | Upper bound on the phase-error rate, $\in (0, 0.5]$. **No default**: the zero-error attack holds bit error rate and visibility at their unattacked values, so the monitoring line cannot derive it. Zero is refused: it claims Eve learns nothing, is worth about $1.5\times$ the rate at $0.05$, and is false for a multiphoton source. |
| `PhaseBound.f` | — | `1.1` | $\ge 1$. |

## Pinning {#pinning}

`None` means *derive*; a value means *pin*. A component takes the hardware
description or the derived value, never both.

```python
q.Fiber(length=25.0, alpha=0.2)          # T derived from attenuation
q.Fiber(T=0.316)                         # T pinned, no fibre model
q.Channel(T=0.5, xi=0.01, ref="input")   # whole channel pinned: textbook mode
```

Pinning is also **the speed dial**: a pinned quantity skips the stage that produces
it, so pinning $T$ and $\xi$ collapses `run()` to the closed form — microseconds, no
symbols. A mixture runs only the stages left to compute.

`explain()` labels every quantity `pinned`, `derived` or `default`:

```python
q.Link(
    modulation=q.GaussianModulation(v_a=18.5),
    channel=q.Channel(T=0.302, xi=0.005, ref="input"),
    bob=q.Bob(detector=q.Homodyne(eta=0.606, v_el=0.041)),
    security=q.Asymptotic(beta=0.898),
).explain()
# {'protocol': 'gaussian modulation',
#  'detection': 'homodyne',
#  'security': 'Asymptotic',
#  'attack': 'Gaussian collective',
#  'stages': 'closed-form only',
#  'v_a':     {'value': 18.5,  'label': 'pinned'},
#  'T':       {'value': 0.302, 'label': 'pinned'},
#  'xi_input':{'value': 0.005, 'label': 'pinned'},
#  'eta':     {'value': 0.606, 'label': 'pinned'},
#  'v_el':    {'value': 0.041, 'label': 'pinned'},
#  'trusted': {'value': True,  'label': 'pinned'},
#  'beta':    {'value': 0.898, 'label': 'pinned'},
#  'T_claimed': {'value': 0.302, 'label': 'derived'}}
```

Plane conversion is a derivation like any other:

```python
q.Channel(T=0.302, xi=0.0015, ref="output")   # ... explain()['xi_input'] is
# {'value': 0.004967, 'label': 'derived'}      # xi / T, labelled derived
```

## Reproducing Lodewyck 2007

Lodewyck et al., PRA **76**, 042305 (2007), every stated number pinned: runnable at
[Usage](/usage#gaussian-modulation-cv-qkd), graded at
[Validation](/guide/validation#the-continuous-variable-rows). Their headline 2 kb/s
is after reconciliation-throughput overhead; the anchor is 12.3 kb/s at a 350 kHz
effective rate, $0.0352$ bit/symbol.

## The result object

`run()` returns a frozen `LinkResult`. A field a family does not produce is `None`,
not `0`. `Link.explain()` is a method; `LinkResult.explain` is the computed dict.

| Field | Populated for | Meaning |
| --- | --- | --- |
| `key_rate` | all | bits/symbol (Gaussian and discrete modulation) or bits/pulse (click family), clamped at $\ge 0$. The unclamped value is always `explain["key_raw"]` |
| `i_ab` | Gaussian and discrete modulation | Alice–Bob mutual information. On the discrete path the constellation's own $I(X;Y)$, with the $\log_2(1+\mathrm{SNR})$ substitute kept as `explain["i_ab_gauss"]` |
| `chi_be` | Gaussian and discrete modulation | Holevo bound on Eve; under `FiniteSize` the worst-case value |
| `t_min`, `xi_max`, `delta` | `FiniteSize` | worst-case channel and the $\Delta(n)$ penalty; `None` under `Asymptotic` |
| `p_click` | click family | probability a slot registered a detection — the click probability for phase keying, the signal gain $Q_\mu$ for basis keying, the bit-carrying gain $q_z$ for intensity keying |
| `qber` | click family | error rate on the sifted bits |
| `dsp` | `dsp=` configured | `Dsp(v_err, pilot_snr, cfo, removed, symbols)`. On the pinned branch `pilot_snr` is `nan` and `symbols` is `0` by construction |
| `est` | `dsp=` with unpinned `v_err` | `Est(T, xi, sigma2, t_min, xi_max)` from the recovered frames; the interval ends are `None` under `Asymptotic` |
| `budget` | Gaussian and discrete modulation | the assembled [`Budget`](/guide/budget), one named row per $\xi$ source over the run's own loss chain, answering `.at(plane)` for the [four planes](/guide/budget#the-four-planes). `None` for the click family |
| `leak` | click family with a `Backflash` impairment | backflash leakage, reported beside the rate and never folded into it |
| `clicks`, `sifted`, `doubles`, `slots` | simulated click pipeline | exact counts, with `clicks = sifted + doubles` |
| `sift_rate` | simulated click pipeline | identical to `p_click` here: under squashing every clicking slot is a key bit |
| `visibility` | simulated click pipeline | the **measured** fringe contrast, not the `DelayInterferometer` input |
| `mu_bob` | simulated click pipeline | mean photon number at Bob's plane, $\approx T\mu$ |
| `key_length`, `s0`, `s1`, `phi`, `n_key` | `SplittingAttack(block=q.KeyBlock(...))` | the finite key **length in bits for the whole block**, and the four quantities it is built from — vacuum and single-photon counts, phase-error rate, sifted key count. `key_rate` there is `key_length / n` |
| `oracle` | all | `Oracle(key_rate, i_ab, chi_be, T, xi)` at the channel's true parameters |
| `explain` | all | the resolved plan dict, same shape as `Link.explain()` |

A slot where **both** detectors fired keeps a random bit, so `doubles` are also in
`sifted` and `clicks` counts them twice ([double clicks](/guide/protocols#doubles)).

`res.oracle` is never `None`. Where nothing was estimated — pinned closed-form paths,
including a pinned `v_err` — the rate already came from the configured channel, and
the property returns the result itself. `.key_rate`, `.i_ab` and `.chi_be` are always
readable; `.T` and `.xi` exist only on a real `Oracle`.

### Estimates versus the oracle

Once frames are estimated, `res.key_rate` comes **from the estimates** — `est.T`
and `est.xi`, clipped to a physical channel and worst-cased by the security model.
`res.oracle` is the rate at the channel's true parameters. Under `FiniteSize` the
claim never exceeds the oracle; under `Asymptotic` nothing is worst-cased, and an
optimistic estimate puts it *above*.

On `test/estimate.py`'s metro link ($T = 0.5$, $\xi = 0.02$ at the channel input,
10 kHz linewidth, $10^6$ symbols, heterodyne trusted), frames fixed and the declared
block growing — one [`measure()` and four `claim()`s](#measure-once-claim-many-times):

| $N$ | `key_rate` | `oracle.key_rate` | gap | `delta` | `t_min` | `xi_max` |
| --- | --- | --- | --- | --- | --- | --- |
| $10^{6}$ | 0.000000 | 0.114855 | 0.114855 | 0.058042 | 0.486058 | 0.154807 |
| $10^{7}$ | 0.026640 | 0.114855 | 0.088215 | 0.018326 | 0.491554 | 0.104475 |
| $10^{8}$ | 0.032907 | 0.114855 | 0.081948 | 0.005792 | 0.491554 | 0.104475 |
| $10^{9}$ | 0.034888 | 0.114855 | 0.079967 | 0.001831 | 0.491554 | 0.104475 |

The oracle is fixed; the claimed rate climbs and plateaus. $\Delta(n)$ keeps
falling, but `t_min` and `xi_max` stop improving after $N = 10^7$: the interval is
built from the pairs **estimated from**, two per heterodyne symbol ($2\times10^6$
here), capped at $p_{\text{PE}}N$. The $n/N$ prefactor keeps the
parameter-estimation half. `explain` keys: `T_est`/`xi_est`, `t_min`/`xi_max`,
`T_oracle`/`xi_oracle`/`key_oracle`.

## Finite-size runs

```python
res = q.Link(
    modulation=q.GaussianModulation(v_a=5.0),
    channel=q.Channel(T=0.316, xi=0.01, ref="input"),
    bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True)),
    security=q.FiniteSize(beta=0.95, eps=1e-10, n=1e9, pe_fraction=0.5),
).run()

res.key_rate   # 0.040998   against 0.087859 asymptotic
res.t_min      # 0.315721   worst-case transmittance
res.xi_max     # 0.014763   worst-case excess noise
res.delta      # 0.001831   smooth-min-entropy penalty
```

Formulas: [Security](/guide/security#finite-size).

## DSP in the loop {#dsp-in-the-loop}

With `dsp=` and the hardware it reads, `run()` generates symbols, sends them through
the channel and a simulated pilot-assisted receiver, and **derives** the residual
phase-error variance.

```python
link = q.Link(
    modulation=q.GaussianModulation(v_a=5.0),
    alice=q.Alice(
        laser=q.Laser(linewidth=10e3),
        pilots=q.Pilots(power_db=12.0, freq=180e6),
        symbol_rate=100e6,
    ),
    channel=q.Fiber(length=25.0, alpha=0.2),
    bob=q.Bob(
        detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True),
        lo=q.LocalLO(linewidth=10e3),
    ),
    dsp=q.DSP(phase=q.PilotPhase(), block=32),
    security=q.Asymptotic(beta=0.95),
)

res = link.run(symbols=1_000_000, seed=1)
res.dsp        # Dsp(v_err=0.008182, pilot_snr=89.29, cfo=815.6, removed=True,
               #     symbols=1000000)
res.est        # Est(T=0.31413, xi=0.040642, sigma2=1.10383,
               #     t_min=None, xi_max=None)
res.key_rate   # 0.062120   from the estimates
res.oracle.key_rate
               # 0.062436   from the true T = 0.316228, xi = 0.041077
```

$V_{\text{err}} = 0.0082$ rad² gives
$\xi_{\text{phase}} = (V_A + \xi)(e^{V_{\text{err}}} - 1) = 0.0411$ SNU, all the
input-referred $\xi$ here, the fibre being noiseless. Under `Asymptotic` the rate
lands below the oracle on this seed and above about as often over seeds.

```python
res.explain["stages"]     # 'symbol pipeline + pilot DSP + parameter estimation'
res.explain["v_err"]      # {'value': 0.008182, 'label': 'derived'}
res.explain["phase_form"] # {'value': 'estimator', 'label': 'default'}
res.explain["xi_phase"]   # {'value': 0.041077, 'label': 'derived'}
res.explain["xi_input"]   # {'value': 0.041077, 'label': 'derived'}
res.explain["xi_est"]     # {'value': 0.040642, 'label': 'derived'}
res.explain["key_oracle"]
                          # {'value': 0.062436, 'label': 'derived'}
```

`link.explain()` on the link reports a derived quantity as
`{'value': None, 'label': 'derived'}`: nothing has been simulated. Only
`res.explain` carries numbers.

`res.est` carries the Leverrier maximum-likelihood estimates of $T$, $\xi$ and
$\sigma^2$, plus `t_min`/`xi_max` when the security model declares a block. The same
seed and plan reproduce `v_err`.

Every `Link` run uses the estimator form of $\xi_{\text{phase}}$, with no `Link`-level
selector; it tracks the measured value to within 2% from $V_{\text{err}} = 0.05$ to
$2.8$ rad². The literature form is `budget.phase(..., form="literature")` alone
([the two forms](/guide/budget#the-two-forms-of-the-phase-term)) and always smaller:
at $\xi = 0$ the estimator-to-literature ratio is
$(e^{V_{\text{err}}} + e^{V_{\text{err}}/2})/2$, 1.23 at $V_{\text{err}} = 0.27$ and 19
at $3.48$.

`PilotPhase(v_err=2e-3)` skips symbol generation: no hardware needed, `res.est` is
`None`, `res.oracle is res`, and `explain()["stages"]` reads
`'closed form only (v_err pinned)'`. Described hardware is **accepted and not read**,
so one link runs with the pipeline on and off. `Alice.symbol_rate` still reaches the
[impairment](/guide/impairments) terms.

The chain is heterodyne-only and requires the hardware it consumes:

```python
q.Link(..., bob=q.Bob(detector=q.Homodyne()), dsp=q.DSP(...))
# NotImplementedError: the pilot DSP chain simulates heterodyne detection

q.Link(..., dsp=q.DSP())          # with no Alice laser/pilots
# ValueError: deriving v_err needs Alice with a laser and pilots
```

## Measure once, claim many times {#measure-once-claim-many-times}

A `run()` is one pipeline pass under one security model, so a sweep over block
size, $\varepsilon$ or the trusted flag re-simulates the symbols per point.
`measure()` stops after the symbols, the pilot DSP and the estimators; `claim()` does
the security accounting on the frames it is handed.

```python
def link(security, channel=q.Fiber(length=25.0, alpha=0.2), impairments=()):
    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        alice=q.Alice(
            laser=q.Laser(linewidth=10e3),
            pilots=q.Pilots(power_db=12.0, freq=180e6),
            symbol_rate=100e6,
        ),
        channel=channel,
        bob=q.Bob(
            detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True),
            lo=q.LocalLO(linewidth=10e3),
        ),
        dsp=q.DSP(phase=q.PilotPhase(), block=32),
        security=security,
        impairments=impairments,
    )

frames = link(q.Asymptotic()).measure(symbols=1_000_000, seed=1)
frames.dsp.v_err     # 0.008182   the pipeline ran here, once
frames.est.xi        # 0.040642

link(q.Asymptotic(beta=0.95)).claim(frames).key_rate
                     # 0.062120
[link(q.FiniteSize(beta=0.95, n=n)).claim(frames).key_rate
 for n in (1e7, 1e8, 1e9)]
                     # [0.001922, 0.008189, 0.010170]
```

`run(symbols, seed)` *is* `claim(measure(symbols, seed))` on this family, bit for
bit: the asymptotic $0.062120$ above is [`run()`](#dsp-in-the-loop)'s on the same
link and seed.

`measure()` returns a `Frames` — symbol count, seed, `dsp`, `est` and the resolved
pipeline arguments. It is not exported (`q.Frames` does not exist), as `Dsp`, `Est`
and `Oracle` are not; only `claim()` consumes one.

`claim()` rebuilds the pipeline key from the link and refuses a mismatch:

```python
link(q.Asymptotic(), channel=q.Fiber(length=30.0, alpha=0.2)).claim(frames)
# ValueError: these frames came out of a different pipeline: the measurement is
#             only valid for the modulation, channel, losses, Alice, Bob and DSP
#             chain that produced it
```

The key is the pipeline's argument list: modulation and $V_A$, channel, itemised
losses, Alice's laser, pilots and symbol rate, Bob's `eta`, `v_el` and LO linewidth,
DSP block size, symbol count and seed. Three things are free to differ:

| Free to differ | Why it is not in the key |
| --- | --- |
| the `security=` component: `Asymptotic` or `FiniteSize`, `beta`, `eps`, `n`, `pe_fraction`, `fer` | accounting over recovered frames |
| `trusted=` on the detector | a [security-model choice](/guide/security#trusted-vs-untrusted-is-a-security-model) reaching $\chi_{\text{det}}$ alone; `eta` and `v_el` are in the key |
| `impairments=` | a descriptor yields an excess-noise row, and the [budget](/guide/budget) is assembled in the claiming half |

A descriptor never reaches the engine and still moves the rate, here and on the
closed-form path:

```python
bare = link(q.Asymptotic()).claim(frames)
res = link(q.Asymptotic(),
           impairments=(q.Modulator(ratio=1.02, angle=0.01),)).claim(frames)

res.dsp == frames.dsp                     # True    the frames are untouched
res.explain["xi_imbalance"]["value"]      # 0.0012549978750070725, the added row
res.budget.total > bare.budget.total      # True    the budget rose
res.explain["xi_claimed"]["value"] > res.est.xi
                                          # True    so did the plane claimed at
res.key_rate < bare.key_rate              # True    and the rate fell for it
res.oracle.key_rate < bare.oracle.key_rate
                                          # True    the oracle too
```

`est.T` and `est.xi` are the channel the pipeline was handed: `run_symbols` takes no
impairment argument, its arguments being the key `claim()` compares. Descriptor rows
are **added to the estimate** where the rate is taken, and the security model is
evaluated at that sum.

| `explain` key | What it is |
| --- | --- |
| `xi_claimed` | that sum, emitted whenever an estimator ran |
| `xi_total` | the assembled budget, input-referred, equal to `res.budget.total`, always reported. It differs from `xi_claimed` by the estimator's sampling error |

On a closed-form link the rows reach the rate directly: the same channel and receiver
without `alice=` and `dsp=` go from $0.099471$ to $0.097559$ under that descriptor.
The comparisons above are **orderings** because a claimed rate here comes from
$\hat\xi$, which scatters over seeds; `xi_imbalance` is a closed form in $V_A$, $d$
and $\theta_m$.

### `T_claimed`, the transmittance half {#t-claimed}

`explain["T_claimed"]` is the transmittance the rate was taken at; `T` stays the
physical loss chain. Emitted on every Gaussian- and discrete-modulation path, it
equals `T` unless one of two things adds a transmittance half:

| What moves it | By |
| --- | --- |
| a fading impairment — the `pol` and `clock` descriptors | that row's own $\langle\sqrt\eta\rangle^2$, through [`impairments.fading_factor`](/guide/impairments#fading) |
| a charged residual-phase row | [`budget.infer`](/guide/budget#infer)'s $e^{-v_{\text{err}}}$ |

Charging one half alone overstates the rate ([the two halves](/guide/budget#infer)).
`v_err` is **carried, not charged**, on the `Budget`, so `res.budget.inferred`
returns this number and `res.budget.T` does not.

In the dry-run `Link.explain()`, `T_claimed` is `None` wherever `v_err` is derived:
nothing has run.

Where the pipeline computes nothing the key is nearly empty: a pinned `v_err` carries
`('pinned', symbols, seed, v_err)`, a link with no `dsp=` `('closed', symbols, seed)`.
Neither depends on the channel, so `claim()` accepts those frames after the channel
moves and returns `run()`'s answer on the new link, to the last bit.

**Only the Gaussian-modulation path has the seam.** The click pipelines stay inside
`run()`, and the M-PSK path emits no symbols. Both methods refuse there with the
message `run()` would raise:

```python
q.Link(
    modulation=q.DifferentialPhase(mu=0.2),
    channel=q.Fiber(length=50.0, alpha=0.2),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.2, dark=1e-6),
        receiver=q.DelayInterferometer(delay=1, visibility=0.98),
    ),
    alice=q.Alice(laser=q.Laser(linewidth=10e3), symbol_rate=1e9),
    security=q.IndividualAttack(f=1.16),
).measure()
# NotImplementedError: DifferentialPhase runs its pipeline inside run(): only
#                      the Gaussian-modulation path separates a measurement from
#                      the security model claimed on it
```

## The click-detector family

Same `Link` shape, different components and security literature. Bob gains a
`receiver`, whose presence decides whether the error rate is simulated or pinned.

```python
res = q.Link(
    modulation=q.DifferentialPhase(mu=0.2),
    channel=q.Fiber(length=50.0, alpha=0.2),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.2, dark=1e-6),
        receiver=q.DelayInterferometer(delay=1, visibility=0.98),
    ),
    alice=q.Alice(laser=q.Laser(linewidth=10e3), symbol_rate=1e9),
    security=q.IndividualAttack(f=1.16),
).run()

res.explain["stages"]      # 'click pipeline + threshold detection'
res.explain["qber"]        # {'value': ..., 'label': 'derived'}
res.explain["visibility"]  # {'value': ..., 'label': 'derived'}
```

$P(\text{click}) = 1 - (1 - p_d)^2 e^{-\mu T \eta}$: two threshold detectors at the
interferometer outputs, so a pulse yields nothing only when both stay dark. Drop the
`receiver=` and pin `IndividualAttack(qber=0.01)`: the closed form runs, `stages`
reads `'closed-form only'` and `qber` is `pinned`. The bound:
[Security](/guide/security#the-wty-individual-attack-bound).

`dead_time` and `afterpulse` reach the simulated click train alone:

```python
q.Link(..., bob=q.Bob(detector=q.ClickDetector(dead_time=1e-8)),
       security=q.IndividualAttack(qber=0.01))
# ValueError: dead time and afterpulsing only reach the simulated click path:
#             leave qber unpinned to use them
```

## The two-state family {#two-state}

`q.TwoStateKeying` keys the bit onto one of two non-orthogonal coherent states. Bob's
`q.NullingReceiver` displaces the signal by a phase-locked local oscillator so the
tested hypothesis nulls; a click names the *other* state. No basis announcement, and
a closed form with no pulse train.

```python
res = q.Link(
    modulation=q.TwoStateKeying(mu=0.23),
    channel=q.Channel(T=0.6),
    bob=q.Bob(
        detector=q.ClickDetector(eta=1.0, dark=0.0),
        receiver=q.NullingReceiver(visibility=1.0),
    ),
    security=q.DiscriminationBound(f=1.16),
).run()

res.key_rate                       # 0.061259 bit/pulse
res.p_click                        # 0.212101 the conclusive fraction
res.explain["overlap"]             # 0.631284 exp(-2*mu)
res.explain["discrimination"]      # 0.368716 what optimal unambiguous
                                   #          discrimination hands Eve
res.explain["e_phase"]             # 0.194691 derived, label 'derived'
```

`overlap` is $e^{-2\mu}$; `discrimination` is $1 - e^{-2\mu}$, the unambiguous
discrimination success probability; `ceiling` is the beam-splitting cap the rate is
held under ([Protocol coverage](/guide/protocols#b92)).

`reference=True` is the one branch where `e_phase` cannot be derived:

```python
q.Link(
    modulation=q.TwoStateKeying(reference=True),
    channel=q.Channel(T=0.5),
    bob=q.Bob(detector=q.ClickDetector(), receiver=q.NullingReceiver()),
    security=q.DiscriminationBound(),
).run()
# NotImplementedError: the strong-reference branch takes a SUPPLIED e_phase: ...
#   Give q.DiscriminationBound(e_phase=...), or drop reference= and let the plain
#   branch derive it from the loss and the overlap
```

Supplied there, `DiscriminationBound(e_phase=…)` is the bound, as `PhaseBound.e_phase`
is; on the plain branch it is a **floor**.

## What runs today {#what-runs-today}

What each modulation takes: [Protocol coverage](/guide/protocols#matrix) and
[Security](/guide/security#other-click). Protocol refusals, finite key among them:
[roadmap](/roadmap#refusals). `q.Link`'s own configuration checks:

| Configuration | Result |
| --- | --- |
| `alice=q.Alice(laser=…/iq=…/pilots=…)` on a basis-keyed link | `ValueError`: a phase-randomised weak-coherent source is described by its intensity set |
| `bases=3` or `announce="pair"` with `alice=` or a `DeadTime` descriptor | `NotImplementedError`/`ValueError` naming the variant: the sampled train sifts two bases and announces one, and the afterpulsing gain form is BB84's yield law |
| `BasisKeying(sift=…)` outside $[1/2, 1)$ under a `q.KeyBlock` | `ValueError` ([the basis split](/guide/security#basis)) |
| `TwoStateKeying` with `alice=`, a `DeadTime`, or detector memory on `q.ClickDetector` | `NotImplementedError`/`ValueError`: the family is a closed form over Bob's receiver and emits no pulse train |
| `alice=` on an `IntensityKeying` link | `NotImplementedError`: only the basis-keyed family emits pulses |
| `Alice(laser=…/iq=…/pilots=…)`, `Bob(adc=…/lo=…)` **without** `dsp=` | `NotImplementedError: nothing on this path reads that hardware`. Add a `q.DSP`, or price the hardware with `qkd.budget.assemble(...)` and pass its total as `q.Channel(T=…, xi=…, ref="input")` |
| `dsp=` with any click modulation | `NotImplementedError: the DSP chain is Gaussian-modulation only` |
| `measure()` or `claim()` on any click or `PhaseShiftKeying` link | `NotImplementedError` ([the seam](#measure-once-claim-many-times)) |
| `claim()` given frames from a different pipeline | `ValueError` naming the mismatch |
| `Bob(detector=q.ThresholdArray(…))` or `q.PnrDetector(…)` | `NotImplementedError`, raised **before** the modulation is dispatched on ([roadmap](/roadmap#refusals)) |
| a click modulation with `Asymptotic`/`FiniteSize`, the wrong attack for the encoding, or any cross-family mix | `NotImplementedError` naming the mismatch |
| a quadrature detector with a `receiver=` | `ValueError: a quadrature detector describes its own front end` |
| a click detector with `Bob(adc=…)` or `Bob(lo=…)` | `ValueError: a click receiver has no ADC and no local oscillator` |

`IQModulator(bits=…)` is charged on a link carrying a `q.DSP`, as
`explain["xi_dac"]`. `Laser.rin` with a detector `bandwidth` is charged as
`explain["xi_rin_sig"]` and `explain["xi_rin_lo"]`: on a 100 MHz-bandwidth
heterodyne link at $T = 0.5$, $\xi = 0.01$, $V_A = 4$ the rate reads $0.1111$,
$0.1074$ and $0$ bit/symbol at $-155$, $-140$ and $-100$ dBc/Hz. `ADC(bits=…)`, a
`clip`, and half a RIN pair are refused by name ([roadmap](/roadmap#link)).

`run(symbols=1_000_000, seed=0)` is read by three paths — the CV symbol pipeline, the
simulated click train and the sampled basis-keyed pulse train — and ignored by the
closed forms. `seed=0` is a real seed: an unchanged plan reproduces bit for bit.
`measure(symbols, seed)` means the same.

`explain()["stages"]` takes one of six values:

| String | When |
| --- | --- |
| `'closed-form only'` | the default; nothing is simulated |
| `'closed form only (v_err pinned)'` | Gaussian modulation with `dsp=` and a pinned `v_err` |
| `'symbol pipeline + pilot DSP + parameter estimation'` | Gaussian modulation with `dsp=` and `v_err` derived |
| `'click pipeline + threshold detection'` | phase keying with the QBER derived |
| `'pulse train + sifting'` | basis keying with an `Alice`, after `run()` |
| `'pulse train (unrun)'` | the same plan through `explain()`, a dry run |

### Surfaces that were designed and dropped {#dropped}

Seven surfaces were specified in the design notes: **one built, six dropped**, not
deferred.

| Surface as designed | Disposition |
| --- | --- |
| `res.budget`, `res.budget.at(plane)` | **Built.** `LinkResult.budget` is a [`Budget`](/guide/budget) on the Gaussian- and discrete-modulation paths; `.at(plane)` takes [four hardware-named planes](/guide/budget#the-four-planes) and refers as the group action $\xi\,f(\text{end})/f(\text{start})$, `post_detection` taking $\eta$ as an argument |
| `q.Link()` with all defaults | **Dropped.** It would default a modulation, a channel and a receiver, which are what a run means |
| `res.frames`, "recovered quadratures, zero-copy numpy" | **Dropped.** Bulk returns are [numpy arrays](/guide/conventions#arrays): at $10^8$ symbols one array is $0.8$ GB, and a `keep_frames` run peaks near $2.4$ GB reading one quadrature and $3.2$ GB reading both. Frames stay opt-in on `_core.run_symbols(..., keep_frames=True)`. Distinct from [`measure()`](#measure-once-claim-many-times)'s `Frames`, which never carries quadratures |
| `qkd.ideal` — `q.ideal.DSP()`, `q.ideal.Detector()`, `q.ideal.PhaseRef()` | **Dropped.** [Pinning](#pinning) reaches every idealisation — `q.PilotPhase(v_err=0.0)`, `q.Heterodyne(eta=1.0, v_el=0.0)` |
| `res.explain()` as a method on the result | **Dropped.** The result's plan is resolved; `Link.explain()` is the dry-run method, `LinkResult.explain` the dict |
| A DSP stage as a plain Python callable — the "escape hatch" | **Dropped.** It forces a Rust→Python→Rust round trip per block, and reproducibility is stated for one binary under any thread count and chunking |
| `MQAM`, and free-form `Discrete(constellation)` | **Dropped.** A modulation component carries a security proof; an arbitrary constellation has none. `PhaseShiftKeying.holevo(..., z=…)` supplies a correlation bound of your own |

Three design-note names were renamed under the [naming rule](/guide/protocols#matrix):
`QPSK` is `q.PhaseShiftKeying`, `Cow` is `q.IntensityKeying`, `Bb84` is
`q.BasisKeying`.
