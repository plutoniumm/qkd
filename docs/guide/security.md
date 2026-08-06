# Security layer

Selected by the `security=` component, per protocol family. Shot-noise units, $\xi$
referred to the [channel input](/guide/conventions#excess-noise-carries-a-plane).

## Devetak–Winter

| | |
| --- | --- |
| Rate | $K_{\text{asym}} = \beta\,I_{AB} - \chi_{BE}$: Gaussian modulation, reverse reconciliation, Gaussian collective attacks |
| $\beta$ | reconciliation efficiency |
| $\chi_{BE}$ | Holevo bound on Eve's information about Bob's key, Eve purifying the channel |
| Arithmetic | f64 on the CPU; the subtraction is cancellation-sensitive ([precision, not device](/architecture#compute-dispatch-on-precision-not-device)) |

### Added-noise bookkeeping

| Term | Value, referred to the channel input |
| --- | --- |
| $\chi_{\text{line}}$ | $\dfrac1T - 1 + \xi$ |
| $\chi_{\text{hom}}$ | $\dfrac{(1-\eta) + v_{el}}{\eta}$ |
| $\chi_{\text{het}}$ | $\dfrac{1 + (1-\eta) + 2v_{el}}{\eta}$. $v_{el}$ enters **twice**, one per receiver; the extra $1$ is the vacuum at Bob's balanced splitter, the 3 dB heterodyne penalty |
| $\chi_{\text{tot}}$ | $\chi_{\text{line}} + \dfrac{\chi_{\text{det}}}{T}$ |

### Mutual information

| Detection | $I_{AB}$, with $V = V_A + 1$ |
| --- | --- |
| Homodyne | $\tfrac12\log_2\dfrac{V + \chi_{\text{tot}}}{1 + \chi_{\text{tot}}}$ |
| Heterodyne | $\log_2\dfrac{V + \chi_{\text{tot}}}{1 + \chi_{\text{tot}}}$ |

## The Holevo bound

Gaussian modulation of coherent states is equivalent to Alice holding one mode of an
EPR state of variance $V = V_A + 1$.

| | |
| --- | --- |
| $\gamma_{AB}$ after the channel (Lodewyck Eq. (19)) | $\begin{pmatrix}V\,\mathbb{1}_2 & \sqrt{T(V^2-1)}\,\sigma_z \\ \sqrt{T(V^2-1)}\,\sigma_z & T(V + \chi_{\text{line}})\,\mathbb{1}_2\end{pmatrix}$, $\sigma_z = \mathrm{diag}(1, -1)$ |
| $\nu_{1,2}^2$ | $\tfrac12\left[A \pm \sqrt{A^2 - 4B}\right]$ |
| $A$ | $V^2(1 - 2T) + 2T + T^2(V + \chi_{\text{line}})^2$ |
| $B$ | $T^2(V\chi_{\text{line}} + 1)^2$ |
| $\nu_{3,4}$ | from an analogous pair $(C, D)$: Lodewyck Eq. (24), heterodyne variant from Fossier |
| $\nu_5$ | $1$ exactly; $G(0) = 0$, so its term vanishes |
| $\chi_{BE}$ | $\sum_{i=1}^{2} G\!\left(\frac{\nu_i - 1}{2}\right) - \sum_{i=3}^{5} G\!\left(\frac{\nu_i - 1}{2}\right)$, $G(x) = (x{+}1)\log_2(x{+}1) - x\log_2 x$ |
| Where the detector enters | trusted: $\eta$ and $v_{el}$ appear **only** in $C$, $D$ and $\chi_{\text{tot}}$, never in $A$ or $B$ — Eve purifies the channel, not Bob's electronics |

Departures from the literal closed form, checked against a 60-digit evaluation
([conditioning exam](/tests/keyrate_conditioning)):

| Departure | Detail |
| --- | --- |
| $C$ and $D$ reparametrised | near $T = 1$ the conditional blocks are differences of nearly equal numbers. Literally, $\nu_2 - 1$ at $V_A = 10^4$, $\xi = 10^{-9}$, $\eta = 1$, $T = 1-3\times10^{-7}$ is $3.103\times10^{-6}$ against an exact $4.993\times10^{-6}$: 38 % on the small eigenvalue, which $G'(0) = \infty$ turns into $1.5\times10^{-5}$ bit/symbol of **understated** $\chi_{BE}$. Reparametrised: $10^{-11}$, all four $\nu - 1$ exactly $0$ at the identity channel, monotone in $V_A$, published anchors moved in the last bit only ($+2.2\times10^{-15}$ on Lodewyck's $\chi_{BE}$) |
| $G(x) \to \log_2 x + (x+1)\ln(1 + 1/x)/\ln 2$ above $x = 1$ | the literal form returns exactly zero once $x > 2^{53}$, where `x + 1 == x` |
| The guard | on the eigenvalues, $\nu \ge 1 - 10^{-6}$, not on the sign of $\chi_{BE}$ |

## Trusted vs untrusted is a security model

Whether Eve controls Bob's $\eta$ and $v_{el}$ is a claim about the **adversary**,
not the hardware; no experiment settles it. It is the `trusted=` flag on the
detector.

| | Trusted | Untrusted |
| --- | --- | --- |
| Assumption | detector inside Bob's secure lab | Eve controls detector imperfections |
| Detector noise | held out of Eve's information | attributed to Eve, inflates $\chi_{BE}$ |
| Key rate | higher, longer reach | conservative |
| Requires | a calibration argument that $\eta$, $v_{el}$ are stable and unmanipulable | nothing |

- The flag belongs to the rate function, not to the receiver component carrying it.
- Discrete modulation: the analytic bound refuses `trusted=True` ([why](/roadmap#refusals)); the [certified bound](/guide/protocols#relent) carries it.
- No click family carries the split: a threshold detector's $\eta$ and dark rate enter the gain and error rate, with no second labelling.

Untrusted is a substitution: fold the detector into the channel, give Bob a perfect
one.

| | |
| --- | --- |
| Substitution | $T \to \eta T$, $\xi \to \xi + \dfrac{\mu\,v_{el}}{\eta T}$, $\eta \to 1$, $v_{el} \to 0$ |
| $\mu$ | $1$ homodyne, $2$ heterodyne: two receivers' electronic noise does not halve at Bob's splitter (Laudenbach Eq. (9.93)) — the factor that puts $2v_{el}$ in $\chi_{\text{het}}$ |
| Source | Laudenbach & Pacher, Sec. 2, Eqs. (13)/(14), in input-referred form; Laudenbach 2018, Eqs. (7.38)–(7.41) and (9.92)–(9.97) |
| No $\xi$ term for $(1-\eta)$ | vacuum loss, carried by the smaller $T$ |
| Invariant | $I_{AB}$ is **exactly** equal under both labellings: $0.451838$ homodyne, $0.521943$ heterodyne at $V_A = 4$, $T = 0.4$, $\xi = 0.01$, $\eta = 0.6$, $v_{el} = 0.1$. Only $\chi_{BE}$ moves; `test/consistency.py` asserts it |

```python
import qkd as q

def rate(trusted):
    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.316, xi=0.01, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=trusted)),
        security=q.Asymptotic(beta=0.95),
    ).run()

rate(True).key_rate    # 0.087859   i_ab = 0.516558, chi_BE = 0.402871
rate(False).key_rate   # 0.0        i_ab = 0.516558, chi_BE = 0.929099
                       #            (raw value -0.438369)
```

`key_rate` clamps at zero; [the suite](/tests/anchors_cv) asserts untrusted never
exceeds trusted. The substitution is the **only** place modelling the split:
`budget.assemble()` has no `vel` and no `trusted`
([why](/guide/budget#why-v-el-is-not-a-budget-line)).

## Finite size

Leverrier, Grosshans & Grangier (2010), collective attacks. $T$ and $\xi$ come from
$m$ disclosed symbols of $N$ and are used at their worst case within a confidence
interval; disclosed pairs obey $y = t x + z$, $t = \sqrt T$,
$z \sim \mathcal N(0, \sigma^2)$, $\sigma^2 = 1 + T\xi$.

| Quantity | At failure probability $\epsilon_{PE}$ |
| --- | --- |
| $t_{\min}$ | $\hat t - z_{\epsilon_{PE}/2}\sqrt{\dfrac{\hat\sigma^2}{mV_A}}$ |
| $\sigma^2_{\max}$ | $\hat\sigma^2 + z_{\epsilon_{PE}/2}\dfrac{\hat\sigma^2\sqrt2}{\sqrt m}$ |
| $T_{\min}$, $\xi_{\max}$ | $t_{\min}^2$, $\dfrac{\sigma^2_{\max} - 1}{T_{\min}}$ |
| $z_{\epsilon/2}$ | two-sided Gaussian tail; at $\epsilon = 10^{-10}$ the literature's "6.5 sigma", computed as $6.467$ by Acklam's rational approximation |
| $K_{\text{finite}}$ | $\dfrac{n}{N}\Big(\beta I_{AB} - \chi_{BE}(T_{\min}, \xi_{\max}) - \Delta(n)\Big)$ |
| $\Delta(n)$ | $(2\dim\mathcal H_X + 3)\sqrt{\dfrac{\log_2(2/\bar\epsilon)}{n}} + \dfrac2n\log_2\dfrac{1}{\epsilon_{PA}}$; $\dim\mathcal H_X = 2$ for the binary reconciliation alphabet, prefactor 7 |

```python
for n in (1e5, 1e7, 1e9, 1e12):
    res = q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.316, xi=0.01, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True)),
        security=q.FiniteSize(beta=0.95, eps=1e-10, n=n, pe_fraction=0.5),
    ).run()
    print(n, res.key_rate, res.t_min, res.xi_max, res.delta)
```

| $N$ | `key_rate` | `t_min` | `xi_max` | `delta` |
| --- | --- | --- | --- | --- |
| $10^{5}$ | 0.000000 | 0.288757 | 0.530750 | 0.184454 |
| $10^{7}$ | 0.018441 | 0.313220 | 0.058010 | 0.018326 |
| $10^{9}$ | 0.040998 | 0.315721 | 0.014763 | 0.001831 |
| $10^{12}$ | 0.043834 | 0.315991 | 0.010150 | 0.000058 |
| asymptotic | 0.087859 | — | — | — |

At $N = 10^{12}$ the rate converges to half the asymptotic one: $n/N$ never returns
the parameter-estimation half. A lower `pe_fraction` recovers it at wider intervals.

| `cv_bounds(t_hat, sigma2_hat, m, va, eps)` | |
| --- | --- |
| Takes | the measured $\hat\sigma^2$, which carries the detection term, not the idealisation $\sigma^2 = 1 + T\xi$ |
| Cost of the idealisation | overstates the key $6.9\times$ at $n = 10^6$, 24 % at $10^7$, 6.3 % at $10^8$, falling as $1/\sqrt n$ |
| Plane | the pipeline reports $\hat\sigma^2$ at Bob's ADC; dividing out a heterodyne receiver's amplitude gain $\sqrt{\eta/2}$ lands it on the **channel output**, where `cv_bounds` needs it and where it is exactly $1 + \chi_{\text{det}} + \hat T\hat\xi$ |
| Same hardware, two planes | de-embedded, $\eta = 0.6$, $v_{el} = 0.1$ heterodyne give $\chi_{\text{het}} = 2.667$ SNU and a variance $\approx 3.67\times$ the idealisation's; at Bob's ADC, $\sigma^2 \approx 1.1$ |

| Epsilon convention | |
| --- | --- |
| Decomposition (Leverrier 2010, Eq. (5)) | $\epsilon = \epsilon_{PE} + \epsilon_{EC} + \bar\epsilon + \epsilon_{PA}$: parameter estimation, error correction ($\epsilon_{\text{cor}}$), smoothing, privacy amplification |
| `FiniteSize(eps=…)` | sets $\epsilon_{PE}$, $\bar\epsilon$ and $\epsilon_{PA}$ each to the value, as that paper's numerics do; not split ([register](#register)) |
| $\epsilon_{EC}$ | not modelled: qkd runs no reconciliation. A published four-term $\epsilon$ does not compare with qkd's three-term one |

### The count the interval is built on

$m$ counts scalar symbol/sample pairs — **two per heterodyne symbol, one per
homodyne symbol** — capped at the declared $p_{\text{PE}}N$. Bounding from the
nominal $p_{\text{PE}}N$ overstated the rate $3.2$–$3.6\times$ on a run whose
$2\times10^5$ symbols supplied $4\times10^5$ pairs against a declared $10^7$.

### Frame errors

| | |
| --- | --- |
| `FiniteSize(fer=…)` | scales the **whole** rate: $K = (1 - \text{FER})\frac{n}{N}(\beta I_{AB} - \chi_{BE} - \Delta(n))$. A failed frame is discarded before privacy amplification, yielding and leaking nothing |
| `fer=None` | *not modelled*, never a measured zero |
| Pinned versus simulated | the table above is a pinned channel; on the simulated path see [estimates versus the oracle](/guide/link#estimates-versus-the-oracle) |
| Attack class | **collective**; general attacks: [the Gaussian de Finetti reduction](/roadmap#definetti) |

## Click protocols use different mathematics

A covariance matrix yields a Holevo bound only under a Gaussian measurement; a
threshold click is a non-Gaussian POVM. The optics simulation is shared,
`security=` is per family, and `Link` refuses a mismatched pair by name.

### The WTY individual-attack bound

Waks, Takesue & Yamamoto (2006).

| | |
| --- | --- |
| Collision probability per bit | $P_{c0} = 1 - e^2 - \dfrac{(1-6e)^2}{2}$ |
| Rate | $R = p_{\text{click}}\left[-(1 - 2\mu)\log_2 P_{c0}(e) - f\,h_2(e)\right]$ |
| $(1 - 2\mu)$ | the photon-splitting concession for a weak-coherent source of mean $\mu$ photons per pulse |
| $f$ | $1.16$, the DPS literature's error-correction inefficiency |

| Domain | Behaviour |
| --- | --- |
| $e \ge 6/38 \approx 0.158$ | Eq. (34) saturates and proves nothing; the rate is $0$ |
| $\mu \ge 1/2$ | the concession consumes the key; the rate is $0$ |
| $e = 0$ | $P_{c0} = 1/2$, so $R = p_{\text{click}}(1 - 2\mu)$ |

Anchors: [DPS exam](/tests/anchors_dps). The QBER is derived from the click train or
pinned on `IndividualAttack(qber=…)`; `explain()` labels which ran.

### Every security component, and the family it belongs to {#other-click}

| Modulation | Security component | Eve priced by |
| --- | --- | --- |
| `GaussianModulation` | `Asymptotic`, `FiniteSize` | the Holevo bound on a Gaussian state |
| `PhaseShiftKeying` | `Asymptotic` (analytic bound, untrusted only), `CertifiedBound` | the Holevo bound on a finite constellation; the relative-entropy proof |
| `DifferentialPhase` | `IndividualAttack` | the WTY collision probability |
| `BasisKeying` | `SplittingAttack`: GLLP ([GLLP04](#references)) over the decoy bounds | the tagged multiphoton fraction plus $h_2$ of the phase error |
| `BasisKeying(bases=3)` | the same | the same, on a **tomographically complete** estimate |
| `BasisKeying(announce="pair")` | the same | the same, plus a **two-photon** term BB84 concedes |
| `IntensityKeying` | `PhaseBound`, `e_phase` required | a phase error nothing in the record estimates |
| `TwoStateKeying` | `DiscriminationBound` | a phase error derived from the loss, or supplied |
| `PairSource` (`q.PairLink`) | `SymmetryBound` | the phase error, from the conjugate basis |
| the same | `ViolationBound` | the **observed CHSH violation**, no source-state assumption |
| `BellAnalyser` (MDI-BB84, `q.Swap`) | `TestBasisBound` | the $Y_{11}^{Z}$ / $e_{11}^{X}$ decoy analysis |

`PhaseBound` refuses an `e_phase` below the multiphoton fraction
$1 - e^{-\mu}(1 + \mu)$ of a mean-$\mu$ source: a floor, not a proof.

## What the qubit-family bounds assume {#qubit}

Most share the Shor–Preskill shape $1 - h_2(e_{\text{ph}}) - f h_2(e_{\text{bit}})$
and differ in where $e_{\text{ph}}$ comes from.

| Family | Phase error from | Tolerance and cost |
| --- | --- | --- |
| **BB84** (`BasisKeying`) | the conjugate basis; two error rates leave Eve one free Bell-diagonal parameter | tolerable QBER **11.003%** |
| **Six-state** (`bases=3`) | three error rates, fixing all four Bell-diagonal weights | **12.6193%**, paid in sifting: $1/3$ against BB84's $1/2$ |
| **SARG04** (`announce="pair"`) | the conjugate basis, on conclusive exclusions of an announced pair | **9.689%** at one photon, **2.710%** at two; sifting $1/4$. The optimised splitting ceiling scales as $t^{3/2}$ against BB84's $t^{2}$ |
| **B92** (`TwoStateKeying`) | **derived from the loss** on the plain branch | the rate reaches zero at transmittance equal to the state overlap, the **noiseless** boundary. At overlap 0.68125 the crossing falls to 0.61884, 0.52015 and 0.42221 at depolarising rates 0.001, 0.005 and 0.01. `reference=True` has no closed form and prices a supplied bound |
| **BBM92** (`SymmetryBound`) | the conjugate basis of an entangled pair | **11.003%** |
| **E91** (`ViolationBound`) | none: $\chi(S)$ from the observed violation, no source-state assumption | **7.149%**. The gap from 11.003% is what the state assumption is worth |
| **MDI-BB84** (`TestBasisBound`) | the test basis, through the joint decoy inversion | the detector is Eve's by construction |
| **RRDPS** (`q.rates`) | nothing measured: packet length $L$ and photon number | rises without limit in $L$, past BB84's 11.0%, six-state's 12.6% and DPS's $6/38$; one packet of $L$ pulses yields at most one bit |
| **Mode pairing** (`q.rates`) | Ma & Razavi's $e^{X}_{11}$, from `mdi.rs` | pairs arrive at $O(\sqrt\eta)$, passing the repeaterless capacity |

Nothing here is device independent ([why](/guide/protocols#not-di)).

## Finite key for BB84-WCP {#finitekey}

Lim, Curty, Walenta, Xu & Zbinden (2014), through
`q.SplittingAttack(block=q.KeyBlock(…))`; without a block the asymptotic GLLP rate
returns byte for byte. `res.key_length` is bits for the block, `res.key_rate` is
`key_length / n`; `res.s0`, `res.s1`, `res.phi`, `res.n_key` carry the pieces.
[Example](/usage#finite-key-length).

**Quote this caveat with every number from this path.** Tupkary et al.: Lim's
phase-error estimate "goes through a Taylor approximation step. Thus the estimate is
**not a true bound**", while "the main essence of the result remains true after
suitable fixes". qkd evaluates the bound as published, with
[E6](/guide/validation#register)'s load-bearing printing error corrected; no fix for
the Taylor step is implemented.

### What is different from the continuous-variable path

| | Gaussian modulation | BB84-WCP |
| --- | --- | --- |
| Estimated | $\hat t$, $\hat\sigma^2$ from $m$ disclosed scalar pairs | integer detection and error **counts**, per intensity and basis |
| Interval | two-sided Gaussian tail, $z_{\epsilon/2}$ | Hoeffding, $\delta = \sqrt{(n/2)\ln(1/\epsilon)}$, on per-intensity counts |
| Unit | bits per symbol | **bits for the block**, reported beside the per-pulse rate |
| Smoothing | $\Delta(n)$, prefactor 7 | six explicit $\log_2$ terms in the length |
| Component | `q.FiniteSize` | `q.KeyBlock` on the family's security component |

The components are not shared: `FiniteSize.n` counts symbols and `KeyBlock.n`
emitted pulses; `beta` $\in [0,1]$ is an efficiency where the click families take
$f \ge 1$; `pe_fraction` has no meaning where the split is over intensities and
bases.

### The pieces

| Quantity | |
| --- | --- |
| $\tau_n$ | $\sum_k p_k e^{-k}k^n/n!$, the probability the source emitted exactly $n$ photons |
| $n_k^{\pm}$ | $(e^k/p_k)\left[n_k \pm \sqrt{(n_{\text{tot}}/2)\ln(1/\epsilon)}\right]$, the Hoeffding-corrected per-intensity count |
| $s_0$, $s_1$ | `decoy_counts`: lower bounds on vacuum and single-photon detections, clamped at zero |
| $v_1$ | `decoy_errors`: upper bound on single-photon bit errors |
| $\phi$ | `bb84_phase`: $v_1/s_1^{\text{test}}$ plus a sampling-**without**-replacement penalty, capped at $1/2$ |
| $\ell$ | `bb84_length`: $\lfloor s_0 + s_1[1 - h_2(\phi)] - \lambda_{EC} - 6\log_2(21/\epsilon_{\text{sec}}) - \log_2(2/\epsilon_{\text{cor}})\rfloor$ |

$s_0$ enters at full weight with no entropy factor: a vacuum emission tells Eve
nothing.

### The basis split is derived, not dialled {#basis}

`BasisKeying(sift=…)` is $q^2 + (1-q)^2$ at bias $q$, so
$q = [1 + \sqrt{2\,\text{sift} - 1}]/2$ and the key share is $q^2$. `sift` below
$1/2$ is refused (unreachable) and `sift = 1` too (no test basis). `run_basis`
names the basis with the draw that decided sifting, so
`key_sifted + test_sifted == sifted` exactly.

### What it costs, and why biasing is worth doing

GYS receiver, 50 km, $\mu = 0.5$, $\nu_1 = 0.1$, unbiased bases:

| $N$ | `key_rate` | ratio to asymptotic |
| --- | --- | --- |
| $10^{8}$ | 0.000000 | 0.000 |
| $10^{10}$ | $2.97\times10^{-5}$ | 0.153 |
| $10^{12}$ | $6.94\times10^{-5}$ | 0.356 |
| $10^{14}$ | $7.38\times10^{-5}$ | 0.379 |
| asymptotic | $1.95\times10^{-4}$ | — |

The rate settles near 0.38 of asymptotic. The residual is the test basis and the
decoy lines, not a finite-size penalty. Biasing recovers most of it at an interior
optimum:

| `sift` | asymptotic | finite, $N = 10^{12}$ | ratio |
| --- | --- | --- | --- |
| 0.5 | $1.95\times10^{-4}$ | $6.94\times10^{-5}$ | 0.356 |
| 0.7 | $2.72\times10^{-4}$ | $1.76\times10^{-4}$ | 0.646 |
| 0.8 | $3.11\times10^{-4}$ | $1.95\times10^{-4}$ | 0.625 |
| 0.9 | $3.50\times10^{-4}$ | $1.75\times10^{-4}$ | 0.499 |
| 0.98 | $3.81\times10^{-4}$ | 0 | 0 |

Past the optimum the test basis starves and $\phi$ saturates at $1/2$.

### Epsilon convention

| | |
| --- | --- |
| Composition | $\epsilon_{\text{sec}} = 21\epsilon$; each of the twenty-one bounds is taken at $\epsilon_{\text{sec}}/21$ |
| `KeyBlock(eps_sec=…)` | $10^{-10}$, matching `FiniteSize(eps=…)` |
| `KeyBlock(eps_cor=…)` | $10^{-15}$, separate from `eps_sec`: no simulated counterpart, qkd running no reconciliation |
| `KeyBlock(fer=…)` | scales the whole length; `None` is *not modelled* |

### Closed form or counted {#counted}

`explain()["counts"]` says which ran.

| Path | Selected by | Counts |
| --- | --- | --- |
| Closed form | `alice=None` | **expected**, from the analytic gains at `KeyBlock(n=…)` |
| Sampled | `alice=q.Alice(…)` | **measured**, integers off `run_basis`; `run(symbols=…)` must equal `KeyBlock(n=…)` |

Expected counts are a **typical case**, not a bound: a block whose signal line falls
low returns a shorter key. The two paths agree to about 1% at $N = 2\times10^7$ on a
short high-efficiency link.

## Source-preparation flaws {#flaws}

A qubit source whose phase modulator applies $\phi + \delta\phi/\pi$ for an intended
$\phi$ puts Alice's four single-photon components at Bloch polar angles $0$,
$\pi/2 + \delta/2$, $\pi + \delta$ and $3\pi/2 + 3\delta/2$. Tamaki, Curty, Kato, Lo &
Azuma (2014); device model Pereira, Curty & Tamaki (2019).

| | `flaw_tolerant` | `flaw_standard` |
| --- | --- | --- |
| Phase error | read **exactly**, by inverting observed yields against Alice's known Bloch vectors | **bounded** through a quantum coin whose imbalance Eve amplifies by the whole channel loss |
| Fourth return | the phase error rate | the coin imbalance |
| At zero flaw | agree to round-off; both reduce to the single-photon BB84 rate | |
| Smaller published flaw, tilt $-\delta$ | distils to **57.775328 dB** | stops at **28.766343 dB** |

The **29.008984 dB** gap is the price of the worst-case assumption, not a security
margin. The two tuples do not line up slot for slot; never subtract them.

| Loss tolerance | |
| --- | --- |
| Mechanism | the single-photon components stay in a two-dimensional Hilbert space, so loss opens no side channel. Three states whose Bloch vectors form a triangle span $\{\mathbb 1, \sigma_x, \sigma_z\}$; the transmission of any fourth, including the virtual state the phase error is defined on, follows by linear algebra |
| Numerically | the phase error barely moves across forty decibels of loss; the dark count moves it |

| Scope | |
| --- | --- |
| Source | a qubit only: no side channel, no Trojan-horse leakage, no mode dependency. Wang's $\theta$ and $\mu$ are absent, so their inequalities collapse to equalities |
| Signals | asymptotic, single photons; a weak-coherent source reaches it through the decoy layer, which supplies $y_1$ |
| Not composed | with `attacks.mismatch_rate` ([why](#no-composition)); the unreconciled line of Tamaki's Appendix C: [gaps](/roadmap#gaps) |

## Certifying COW′'s phase error {#sdp}

Plain COW's `e_phase` is supplied. COW′ adds the two-pulse vacuum decoy sequence, and
`src/sdp.rs` certifies its phase error as the value of a semidefinite programme
([problem](/guide/protocols#cow-prime)). Method, refusals and the other two solvers:
[three cones](/roadmap#solver).

The certified value is returned **raw**: at or above $1/2$ is the abort region,
reported unclamped so the depth of the failure stays readable. No `q.Link` reaches
it: `q.IntensityKeying` describes a three-sequence source and `sdp.rs` reads four.
The routes are `q.rates("cow-vacuum")` and `q.security.keylength`.

## Everything else is asymptotic, and says so {#asymptotic}

| Finite-key route | Families |
| --- | --- |
| A component | Gaussian modulation, `q.FiniteSize`; BB84-WCP, `q.KeyBlock`; MDI-BB84, `q.TestBasisBound(block=q.RelayBlock(…))`; CV-MDI, `q.TwoModeBound(block=q.GaussianBlock(…))` |
| `q.security.keylength`, by name | six-state, SARG04, RRDPS, mode pairing, COW′, discrete modulation, the pair link |
| Its own module | DPS, `qkd.dps.finite(..., detector=…)` |

Every other request refuses by name; the reasons are
[What qkd will not do today](/roadmap#refusals). What that table does not carry:

| Family | |
| --- | --- |
| **Discrete modulation** | analytic bound: Lupo & Ouyang (2022), a heterodyne confined to $q,p \in [-R, R]$ in $d$ bins, both charged as *noise rows*. Certified bound: Kanitschar, George, Lin, Upadhyaya & Lütkenhaus (2023), Theorem 6; its dimension-reduction charge $\Delta(w)$ is 0.101 bit/pulse at $w = 10^{-4}$, so dropping it is wrong, not loose. The 1.834309 / 1.894405 bit/kept round comparison is at $\alpha = 0.7$, $n_c = 6$, and the 3.17% over-claim is substitution into Eq. (9). `_core.dm_length` refuses `source="certified"` twice: the equality set sits inside the relaxed one, and the two are written over different observables, $\{q, p, n, d\}$ against Eq. (21)'s displaced $\{\hat n_\beta, \hat n_\beta^2\}$. The protocol steps stay caller obligations, of the standing the photon-number cutoff has |
| **DPS** | `qkd.dps.finite`, Mizutani, Takeuchi & Tamaki (2023), reproduces their $\varepsilon_{\rm sec}$ composition to the last bit and their $\mu_{\rm opt}$ to two figures, zero fitted. `detector="threshold"` names four gaps: the detected event is a photon number, not a click; the protocol is block-wise where `run_clicks` is a continuous train; `ClickOut` has no code/sample coin; the analysis wants one symmetric detector pair |
| **COW** | Korzh et al. 2015 published $\epsilon_{qkd} = 4\times10^{-9}$ at 307 km; González-Payo et al. 2020 declared that implementation insecure without touching the statistics. After the vacuum-decoy change the composable length is Li, Cao, Xie, Yin & Chen (2024) |
| **Six-state** | the pooled monitor statistic is the worst case, not an approximation: $h_2$ peaks at $1/2$, so the equal split minimises $H(X|E)$ at fixed sum, and Lim's Eq. (5) takes the one sample and population size a pooled sample has. The sampled-train refusal is `q.Link`'s, not the engine's: `_core.run_basis` takes `bases=3` and a per-basis misalignment triple, and `q.Link` passes neither |
| **SARG04** | `sarg_length` is Nian, Nie, Zhang & Lu, Commun. Theor. Phys. 76, 065101 (2024), Eq. (5), on Rusca et al., Appl. Phys. Lett. 112, 171104 (2018). No complementary-basis sample: $e_p \le \tfrac32 e_b$ |
| **E91 and BBM92** | `qkd.pairs.finite` (Tomamichel & Leverrier (2017), Theorem 3) reproduces the four points their own `examples.py` prints exactly. `_core.ekert_finite` refuses and names four missing pieces. `q.PairLink` carries no block-size component |
| **RRDPS, mode pairing** | no component tree; epsilons: [register](#register) |
| **The relay** | `q.TwoModeBound(block=q.GaussianBlock(…))` (Papanastasiou, Ottaviani & Pirandola (2017)): `sigma` is a coefficient in standard deviations, not a probability, with no default; its epsilons do not compose; `attack=` refuses `"collective"` and `"coherent"` by name. `q.RelayBlock` is not a `q.KeyBlock`: `n` counts pulse **pairs**, the length is per announced Bell state summed at a summed budget, and the secrecy budget splits into 266 shares against Lim's 21 |

## `qkd.security`, the epsilon register {#register}

**No two shipped families mean the same thing by "eps".** A registry, not a
derivation: every row is read off the engine it names.

| Family | What the declared parameter is | Rule | `form` | Equal shares |
| --- | --- | --- | --- | --- |
| `cv` | **per term.** `q.FiniteSize.eps` is handed to `eps_pe`, `eps_smooth` and `eps_pa` alike, so the composed secrecy parameter is $3\varepsilon$ and the declared number is **not** the total | sum, a union bound over three independent failure events | `sum` | none |
| `bb84` | the **total** | sum: secrecy plus correctness | `sum` | 21 |
| `sarg` | the **total** | sum: secrecy plus correctness | `sum` | 18 |
| `pairing` | the **total** | sum: secrecy plus correctness | `sum` | 24 |
| `pair` | the **total**, over one correctness term and two secrecy | sum, plus a hash length in bits | `sum` | none |
| `cvmdi` | **uncomposed**: POP17 add none of these together, and state parameter estimation as a coefficient in standard deviations | the secrecy half is a sum of the two modelled terms; no total composes | `sum` | none |
| `mdi` | the total **per announced Bell state** | per-state sum | `per-state` | 266 |
| `sixstate` | `eps_ec` sits **inside** `eps` | nested | `nested` | 3 |
| `cow-vacuum` | four terms on one common share | sum **at Li's weights 2/1/6/1**, each weight counting the bounds that term is spent on | `weighted` | 10 |
| `rrdps` | Takesue's $d$, over five terms **two of which are hash lengths in bits** | **max**, with a square root inside one branch | `max` | none |
| `dmcs` | five terms | **outer sum over a maximum**: $\varepsilon_{ec} + \max(\varepsilon_{pa}/2 + \bar\varepsilon,\ \varepsilon_{et} + \varepsilon_{at})$ | `outer-max` | none |
| `dps` | four terms, two of them hash lengths | **root** over a weighted sum, MTT23 Eq. (50), two terms charged three times each | `root` | none |

`families()` lists the registered families; `describe(family)` returns one record.
Both refuse an unregistered name.

`rule` is a sentence; `form` is the same composition as a token from
`security.FORMS`, and `Ledger.secrecy` dispatches on it. A row whose `form` names an
arm it does not reach refuses rather than falling through: the plain sum reads
**zero** for six-state, **half** the parameter for MDI and **above one** for RRDPS and
DPS. The token names the arm of the *secrecy half*, not the total — hence `cvmdi`
carries `sum` while its `rule` composes nothing.

### Three kinds, and a hash length is not a probability {#register-kinds}

| Kind | Is |
| --- | --- |
| `eps` | a failure probability in $(0, 1)$, as `src/std.rs::check_eps` |
| `bits` | a hash length $s$ standing for $2^{-s}$ — the only form Takesue's $\eta_x$, $\eta_z$ take |
| `count` | whole repetitions of a whole analysis, as Curty's sum over announced Bell states |

A term also carries a **role**: `secrecy`, `correctness`, `authentication`, or
`structure`, which is not charged but multiplies.

### Verify or compute, and only one of them is available per family {#register-two}

| | |
| --- | --- |
| `compose(family, **terms)` | **verifies** a stated budget. Every term is required; the refusal lists what the family accepts |
| `allocate(family, target)` | **computes** terms by the family's own published assignment. `rrdps` only; every other family refuses by name, having an equal-share divisor or none, and dividing a target misreports a budget that composes as a maximum, nests, or counts repetitions |

Both return a `Ledger`. `.secrecy` and `.correctness` are the two halves;
`.total` is their sum plus any authentication term. Each **refuses** a term the
family does not model, naming the file that would produce it. `cv` models no
correctness parameter — a frame error rate scales the rate and is not a probability
that two keys differ undetected — and `rrdps` charges error-correction leakage as a
count of bits sent, with no confidence interval.

`authenticate(ledger, messages, eps)` adds the forgery probability
$\text{messages}\times\varepsilon$ — the **epsilon** side. `qkd.reconcile`'s
`net_length` subtracts authentication in **bits** and never touches the parameter.
Never charge both.

A privacy-amplification term — `eps_pa`, `eps_amp`, `eps_hash` — is accepted only by
a family whose engine takes one as a separate argument; elsewhere it is refused by
name, the family's secrecy parameter already charging the hashing.

## Attacks outside every bound on this page {#attacks}

Every rate above bounds Eve given the observables. Under every `qkd.attacks` attack
the observables stay at values an unattacked link would give, so no bound here sees
them.

| Attack | Family | Mechanism |
| --- | --- | --- |
| `Saturation` | homodyne CV | the covariance matrix is invariant under a shift of the quadrature **mean**, which no CV-QKD estimator monitors: Eve intercept-resends every pulse and displaces Bob into his clipping region. Qin, Kumar & Alléaume (2016) |
| `Calibration` | homodyne CV | the shot-noise unit is over-estimated; the reported excess noise is the true one divided by that ratio, and past the zero-noise ratio it is **negative**, returned raw. Jouguet, Kunz-Jacques & Diamanti (2013) |
| `Blinding` | threshold detectors | under detector control the gain and QBER are whatever Eve reproduces, while she holds one bit per sifted bit. Lydersen et al. (2010); a detector set violating their Eq. (1) is refused, not modelled as a weaker attack |
| `Mismatch` | threshold detectors | a time shift routes the pulse through a long or short path unmeasured, so the QBER carries no trace at any mismatch. Qi, Fung, Lo & Ma (2007); surviving rate Fung, Tamaki, Qi, Lo & Ma (2009), Eqs. (32)–(34) |
| `Blanking` | threshold detectors | a bright pulse in the dead interval blinds the next gate and leaves gain and QBER unchanged. Weier et al. (2011). The $\tau_D$ `q.DeadTime` spends on rate, this attack spends on security |

A `Reading` carries `observed` and `eve` and no key rate: `key_rate`, `rate`, `key`,
`secure`, `safe` and `margin` raise. Never subtract the two. qkd's asymptotic rate is
positive on the channel the attacked estimator reports and negative on the channel
that is there. Module reference: [Impairments](/guide/impairments#attacks).

### The two engines that do not compose {#no-composition}

`src/flaws.rs` and `attacks.mismatch_rate` may not be chained. Both loss-tolerant
analyses require Bob's inconclusive operator $M_f$ to be the same in both bases; a
detection-efficiency mismatch violates exactly that. The composition exists
(arXiv:2412.09684, 2024) and needs its own proof and virtual states; neither is
implemented.

## References

| Key | Citation |
| --- | --- |
| Fung, Tamaki, Qi, Lo & Ma (2009) | QIC **9**, 131 (2009), [arXiv:0802.3788](https://arxiv.org/abs/0802.3788) |
| Gao et al. (2022) | Opt. Express **30**, 23783 (2022), [arXiv:2107.09329](https://arxiv.org/abs/2107.09329) |
| GLLP04 | Gottesman, Lo, Lütkenhaus & Preskill, *QIC* **5**, 325 (2004) |
| Jouguet, Kunz-Jacques & Diamanti (2013) | PRA **87**, 062313 (2013), [arXiv:1304.7024](https://arxiv.org/abs/1304.7024) |
| Kanitschar, George, Lin, Upadhyaya & Lütkenhaus (2023) | PRX Quantum **4**, 040306 (2023), [arXiv:2301.08686](https://arxiv.org/abs/2301.08686) |
| Laudenbach & Pacher | [arXiv:1904.01970](https://arxiv.org/abs/1904.01970) |
| Laudenbach 2018 | [arXiv:1703.09278](https://arxiv.org/abs/1703.09278) |
| Leverrier, Grosshans & Grangier (2010) | PRA **81**, 062343 (2010) |
| Li, Cao, Xie, Yin & Chen (2024) | Phys. Rev. Research **6**, 013022 (2024) |
| Lim, Curty, Walenta, Xu & Zbinden (2014) | Phys. Rev. A **89**, 022307 (2014), [arXiv:1311.7129](https://arxiv.org/abs/1311.7129) |
| Lupo & Ouyang (2022) | PRX Quantum **3**, 010341 (2022), [arXiv:2108.00428](https://arxiv.org/abs/2108.00428) |
| Lydersen et al. (2010) | Nat. Photonics **4**, 686 (2010), [arXiv:1008.4593](https://arxiv.org/abs/1008.4593) |
| Mizutani, Takeuchi & Tamaki (2023), MTT23 | Phys. Rev. Research **5**, 023132 (2023), [arXiv:2301.09844](https://arxiv.org/abs/2301.09844) |
| Papanastasiou, Ottaviani & Pirandola (2017), POP17 | Phys. Rev. A **96**, 042332 (2017), [arXiv:1707.04599](https://arxiv.org/abs/1707.04599) |
| Pereira, Curty & Tamaki (2019) | npj Quantum Information **5**, 62 (2019), [arXiv:1902.02126](https://arxiv.org/abs/1902.02126) |
| Qi, Fung, Lo & Ma (2007) | QIC **7**, 73 (2007), [quant-ph/0512080](https://arxiv.org/abs/quant-ph/0512080) |
| Qin, Kumar & Alléaume (2016) | PRA **94**, 012325 (2016), [arXiv:1511.01007](https://arxiv.org/abs/1511.01007) |
| Scarani & Renner (2008) | Phys. Rev. Lett. **100**, 200501 (2008), [arXiv:0708.0709](https://arxiv.org/abs/0708.0709) |
| Tamaki, Curty, Kato, Lo & Azuma (2014) | PRA **90**, 052314 (2014), [arXiv:1312.3514](https://arxiv.org/abs/1312.3514) |
| Tomamichel & Leverrier (2017) | Quantum **1**, 14 (2017), [arXiv:1506.08458](https://arxiv.org/abs/1506.08458) |
| Tupkary et al. | [arXiv:2502.10340v3](https://arxiv.org/abs/2502.10340) |
| Waks, Takesue & Yamamoto (2006) | PRA 73, 012344 (2006) |
| Weier et al. (2011) | New J. Phys. **13**, 073024 (2011), [arXiv:1101.5289](https://arxiv.org/abs/1101.5289) |
