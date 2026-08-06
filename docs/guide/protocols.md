# Protocol coverage

What runs, through which component, on which analysis. Grades:
[Validation](/guide/validation#board). Refusals:
[Status and roadmap](/roadmap#refusals). Finite key per family:
[Security](/guide/security#asymptotic).

## Three levels of reach {#reach}

An engine is reached through a component tree (`q.Link`, `q.Swap`, `q.PairLink`),
through a module taking its arguments by name (`q.rates`, `q.security`, `q.dps`,
`qkd.fock`), or from `qkd._core` alone. Per engine:
[Architecture](/architecture#the-core-modules). `q.ThresholdArray`,
`q.PnrDetector` and `q.TransmittedLO` construct and are refused by `q.Link`
([refusals](/roadmap#refusals)).

## The two families {#families}

Balanced receivers (quadratures, covariance matrices, Devetak–Winter) and threshold
detectors (clicks, collision probabilities, GLLP [GLLP04], phase-error bounds) share
the RNG, the Wiener phase walk and the thermal-loss channel, never the accounting.
`Link` refuses a mismatched modulation/security pair
([Architecture](/architecture#one-pulse-train-two-receivers),
[Security](/guide/security#click-protocols-use-different-mathematics)).

## The matrix {#matrix}

| Protocol | Encoding | Detection | Security implemented | Reachable from |
| --- | --- | --- | --- | --- |
| Gaussian modulation of coherent states | Gaussian in $x$ and $p$ | homodyne or heterodyne | Devetak–Winter, asymptotic and Leverrier finite-size, trusted or untrusted | `q.Link` |
| Discrete modulation, $M$-PSK | $m$ coherent amplitudes on a circle | heterodyne | [DBL21] analytic bound, untrusted only; the [relative-entropy proof](#relent), trusted or untrusted. Asymptotic, collective attacks | `q.Link` via `q.PhaseShiftKeying`, with `q.Asymptotic` or `q.CertifiedBound` |
| Differential phase shift | phase between adjacent pulses | delay interferometer + two threshold detectors | Waks–Takesue–Yamamoto individual attacks | `q.Link`. Three-pulse blocks on photon-number-resolving detectors are a different protocol: `q.dps` |
| Round-robin differential phase shift | phase between a *randomly chosen* pair in a packet of $L$ | variable-delay interferometer + two threshold detectors | [SYK14]: privacy amplification from $(L, \nu_{\text{th}})$ and **no observable at all** | `q.rates.keyrate("rrdps", …)` — [no component tree](/roadmap#core-only) |
| BB84 with weak coherent pulses | basis and bit, two intensities of decoy; carrier unnamed or polarisation | basis analyser + threshold detectors | GLLP with decoy-state $Y_1$/$e_1$ bounds | `q.Link`, closed form or [sampled](#sampled) |
| Six-state | the same bits keyed into **three** mutually unbiased bases | the same | [S09] App. A: a tomographically complete Bell-diagonal estimate, asymptotic | `q.Link` via `q.BasisKeying(bases=3)` |
| SARG04 | the same four BB84 states, a non-orthogonal **pair** announced instead of a basis | the same | [FTL06] unconditional rate, paying privacy amplification on one- **and** two-photon signals — though the decoy layer bounds only $Y_1$, so [$Q_2$ is reported absent](#sarg) | `q.Link` via `q.BasisKeying(announce="pair")` |
| Two-state keying (B92) | two non-orthogonal coherent states, one per bit | nulling receiver + one threshold gate | [TL04] on the plain branch, phase error **derived**; [K04] on the strong-reference branch, **supplied**. [TLKB09] derive it on a photon-number-resolving receiver: `_core.b92_strong`, no `q.` path | `q.Link` via `q.TwoStateKeying` |
| Coherent one way | intensity keying, data line plus monitoring tap, optional finite [extinction](#extinction) | threshold detectors on both lines | phase-error form, with the bound supplied rather than derived | `q.Link` |
| COW′, the vacuum-decoy variant | the same, with the four-sequence monitoring record read exactly | the same | the exact worst case over every measurement consistent with the record, by [semidefinite programme](#sdp) | `q.rates.keyrate("cow-vacuum", …)` — [no component tree](/roadmap#core-only) |
| Continuous-variable relay | two Gaussian senders into an untrusted relay | relay Bell measurement | Holevo bound on the conditioned two-sender state | `q.Swap` |
| Basis-keyed relay (MDI-BB84) | two decoy weak-coherent senders into an untrusted relay, key basis and test basis | relay Bell-state measurement, four threshold detectors | joint two-sender decoy $Y_{11}^{Z}$ / $e_{11}^{X}$ bounds; [C14] finite-key length | `q.Swap` with `q.Relay(bell=q.BellAnalyser(…))` — [below](#mdi-bb84) |
| Mode-pairing / asynchronous MDI | two independent weak-coherent senders, the two clicks that make a bit chosen **after** the announcement | single-photon interference at an untrusted station | [ZZWM22] asymptotic rate on a pairing rate and a sifted share | `q.rates.keyrate("pairing", …)` — [no component tree](/roadmap#core-only) |
| Entanglement-based basis keying | a photon-pair source, both parties measuring in two conjugate bases | threshold detectors at both ends, coincidence-counted | Shor–Preskill through basis symmetry, on the [MFL07] pair model | `q.PairLink` with `q.SymmetryBound` |
| CHSH-priced entanglement (E91) | the same pairs, a third analyser setting spent on a Bell test | the same | [A07]'s $\chi(S)$, which needs **no assumption about the source state** — and is never the cheaper price | `q.PairLink` with `q.ViolationBound` |
| Loss-tolerant source flaws | not a protocol — a *reading* of BB84's own data with a known modulator flaw | the same | [TCKLA14] exact phase-error inversion, beside the quantum-coin worst case it replaces | `q.Link` via `q.FlawedKeying` with `q.FlawBound` — [below](#flaws) |
| Truncated-Fock states | not a protocol — the non-Gaussian state layer | — | not a security layer | `qkd.fock` |

Relay rows: [Relay topologies](/guide/relay). Fock row:
[Non-Gaussian states](/guide/fock). Components are named for what the hardware does
to the light; the acronyms live here.

| Component | Known in the literature as |
| --- | --- |
| `GaussianModulation` | GG02, after Grosshans–Grangier 2002 |
| `PhaseShiftKeying` | $M$-PSK discrete modulation |
| `DifferentialPhase` | DPS |
| `BasisKeying` with a `Decoy` set | BB84-WCP |
| `BasisKeying(bases=3)` | the six-state protocol |
| `BasisKeying(announce="pair")` | SARG04 |
| `PolarisationKeying` | BB84-WCP with its carrier named |
| `IntensityKeying` | COW |
| `TwoStateKeying` | B92; with `reference=True`, Koashi's strong-reference variant |
| `q.Swap` | CV-MDI |
| `q.BellAnalyser` with basis- or polarisation-keyed senders | MDI-QKD, MDI-BB84 |
| `q.TestBasisBound` | the $Y_{11}^{Z}$ / $e_{11}^{X}$ decoy analysis that protocol is secured by |
| `q.PairSource` with `q.SymmetryBound` | BBM92, the entanglement-based BB84 |
| `q.PairSource` with `q.ViolationBound` | E91, priced by the CHSH violation |
| `q.rates.keyrate("rrdps", …)` | RRDPS |
| `q.rates.keyrate("pairing", …)` | mode-pairing QKD, also published as asynchronous MDI-QKD |
| `q.SourceFlaw`, `q.FlawedKeying` | loss-tolerant QKD with state-preparation flaws |
| `q.rates.keyrate("cow-vacuum", …)` | COW′, the two-pulse vacuum-decoy variant |

## Gaussian modulation {#gaussian}

Devetak–Winter with reverse reconciliation, $K = \beta I_{AB} - \chi_{BE}$
([Security](/guide/security)). $\xi$ is an output of the pilot-assisted receiver
([DSP in the loop](/guide/link#dsp-in-the-loop),
[the two forms of the phase term](/guide/budget#the-two-forms-of-the-phase-term)).
`q.FiniteSize` is accepted on this family alone.

## Discrete modulation {#discrete}

$m$ coherent states of equal amplitude on a circle, each with probability $1/m$.
Alice's marginal covariance is $(1 + V_A)\mathbb{1}_2$, identical to a Gaussian
modulation of the same variance; only the Alice–Bob correlation $z$ differs, which a
Gaussian modulation certifies as $\sqrt{V_A^2 + 2V_A}$ and a finite constellation
strictly less. The [DBL21] lower bound:

$$Z^\star = \sqrt{T}\left(z_{\text{lin}} - \sqrt{2\,\xi\,w}\right),$$

$z_{\text{lin}}$ and $w$ being the constellation's moments. `dm_holevo` takes $z$ as
an argument, so QAM, probabilistic shaping or a measured correlation plug in at the
same seam.

| Argument | Plane / units | Default | Description |
| --- | --- | --- | --- |
| `z` | **channel output**, SNU | **required** | Off-diagonal block of the Alice–Bob matrix; carries its factor of $\sqrt T$. $z = \sqrt{T(V_A+1)^2 - T}$ reproduces `cv_rate` at $\eta = 1$, $v_{el} = 0$, heterodyne. |
| `bits` | bits | **required** | Entropy of Alice's alphabet. No default: a covariance matrix carries no alphabet. $m$ equiprobable states give $\log_2 m$ at any brightness; a continuous Gaussian modulation passes infinity. |

| Quantity | Clamped? | Because |
| --- | --- | --- |
| $z_\star$ | at zero, on output | $\chi_{BE}$ depends on $z^2$, so $-0.5$ would read as $+0.5$ |
| `dm_holevo`'s `key` | no, as `cv_rate` | a negative value means no key |

| Property | What holds |
| --- | --- |
| Ordering in $m$ | the rate is $\le$ the Gaussian one at every $m$, approaching it from below |
| Mutual information | Gauss–Hermite quadrature over the real constellation, not $\log_2(1 + \mathrm{SNR})$, which is exact only for a continuous modulation and unbounded where a finite alphabet saturates at $\log_2 m$. `dm_rate` therefore returns slightly *less* than the papers it is checked against, whose own combination is the checked quantity |

### The quadrature, and the order it is verified to {#quadrature}

The node count tracks the modulation amplitude $\alpha$ rather than $m$.

| $\alpha$ | Nodes | Verified |
| --- | --- | --- |
| $\le 0.4$ | 16 | every constellation to $5\times10^{-12}$ or better at $T = 1$, $\xi = 0$, its hardest channel |
| $> 0.4$ | 64 | at $\alpha = 1$; every tight anchor runs on this branch, bit-identical |

Operating points sit at $\alpha = 0.35$–$0.4$. The crossover was set against the
exams; anchors off the 64-node branch go through `dm_holevo`, which never calls this
quadrature.

Newton starts from *Numerical Recipes*' `gauher` guesses, which fail at high order:

| Order | Second root, against an independent bisection finder |
| --- | --- |
| 192 | wrong by $1.2\times10^{-13}$ |
| 194 | wrong by $9.7\times10^{-10}$ |
| 196 | wrong by $3.6\times10^{-7}$ |
| 200 | lands in the *first* root's basin; the positive half of the rule collapses onto a single point |
| 384 | first visible symptom — `mutual()` returns $-0.093$ bits |

| Guard | |
| --- | --- |
| a sum-of-weights check | **blind to this**: the zeroth moment matched $\sqrt\pi$ to $10^{-15}$ at order 198 with a node wrong in the seventh digit, the outer weights being of order $10^{-155}$ |
| root acceptance | Newton must converge *and* land in a bracket proven to hold one root; otherwise bisection. Bit-identical below order 188 |
| order cap | **370**, checked at compile time against both node counts. Orders 2 to 370 agree with the independent finder to $3.6\times10^{-15}$; 371 overflows $w = 2/p'^2$ |

```python
import qkd as q

res = q.Link(
    modulation=q.PhaseShiftKeying(states=4, alpha=0.4),
    channel=q.Channel(T=0.5, xi=0.01, ref="input"),
    bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=False)),
    security=q.Asymptotic(beta=0.95),
).run()

res.explain["v_a"]          # {'value': 0.32, 'label': 'derived'}   2 alpha^2
res.explain["bits"]         # {'value': 2.0,  'label': 'derived'}   log2 m
res.explain["z_star"]       # the certified correlation
res.explain["z_gauss"]      # what a Gaussian modulation would have certified
res.explain["i_ab_gauss"]   # the log2(1 + SNR) substitute, kept beside i_ab
```

`dm_rate` coincides with the **linear-objective** SDP, falls short of the best known
QPSK result where a nonlinear objective buys more, and tightens with $m$ to sit close
to the Gaussian value by $m = 6$. The certified bound is
[`q.CertifiedBound`](#relent). Refused on this family — a trusted receiver on the
analytic bound, homodyne, `FiniteSize`: [roadmap](/roadmap#refusals).

## Phase keying {#dps}

A train of coherent pulses with random $0$ or $\pi$ phase; Bob's delay-line
interferometer compares each pulse with the one $d$ slots earlier. The bit is which
of two threshold detectors fires; a slot with no click is discarded.

A Mach–Zehnder has **two** 50:50 couplers, so the ports of slot $k$ carry
$(\alpha_k \pm \alpha_{k-d})/2$ — not $/\sqrt2$, the isolated two-mode
normalisation, which double-counts every pulse when applied at every slot. Fringe
contrast $V$ scales the interference term alone,

$$I_\pm = \frac{|\alpha_k|^2 + |\alpha_{k-d}|^2 \pm 2V\,\mathrm{Re}\!\left(\alpha_k \alpha_{k-d}^{*}\right)}{4},$$

so the interferometric error is $(1-V)/2$ in the weak-flux limit and the count rate
does not move: visibility and click rate are separate observables.

The QBER is derived from `DelayInterferometer.visibility`, the Wiener walk over $d$
symbol periods and `ClickDetector.dark`
([Architecture](/architecture#derived-qber)); worked link:
[Protocol layer](/guide/link#the-click-detector-family).

### Timing jitter is two things, and only one of them is a loss {#jitter}

| Symbol | Meaning |
| --- | --- |
| $J(u)$ | the arrival-time law |
| $T_s$, $T_w \le T_s$ | symbol period, and the centred acceptance window |
| $w_m = \int_{mT_s - T_w/2}^{\,mT_s + T_w/2} J(u)\,\mathrm{d}u$ | the share of one slot's light collected by the window $m$ slots away; $W = \sum_m w_m$ |
| **Window loss** $= 1 - W$ | detections in **no** window. A pure loss, indistinguishable from a smaller $\eta$, carrying no error |
| **Bin leak** $= \sum_{m\neq0} w_m / W$ | detections in the **wrong** slot's window. The differential bit there is independent of this slot's, so half are errors |

Jitter convolves the port intensities along the slot train,
$\tilde I_\pm(k) = \sum_m w_m I_\pm(k-m)$, exact for a coherent source since
photodetection is an inhomogeneous Poisson process. Leaked light is flat across the
two ports, so the measured contrast falls to

$$V_{\text{meas}} = \frac{w_0}{W}\,V\,e^{-\sigma_\phi^2/2}
= (1 - \text{leak})\,V\,e^{-\sigma_\phi^2/2},$$

and $e = (1 - V_{\text{meas}})/2$ holds unchanged, carrying a
$V\cdot\text{leak}/2$ jitter term beside the $(1-V)/2$ optical and $\sigma_\phi^2$
phase-walk terms.

#### Gaussian, or tailed? {#jitter-shape}

The **prompt** response — avalanche build-up, discriminator walk, clock
distribution, optical pulse width — is Gaussian. A thick silicon SPAD also detects
carriers photogenerated *outside* the high-field region, which diffuse in and arrive
late by an exponential time, never early. The mixture is

$$J(u) = (1 - f)\,\mathcal N(0, \sigma^2)
       + f\,\frac{1}{\tau}e^{-u/\tau}\,\Theta(u),
\qquad \sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}},$$

with $f = 0$ by default. Composing the two by convolution rather than by mixture
changes $w_m$ by $O(\sigma^2/\tau^2)$.

**A purely Gaussian jitter model produces window loss and no bin error**: a core
narrow enough for a realistic window loss puts the neighbouring slot tens of
standard deviations away. Diamanti *et al.*'s two published window statements
exclude it from the data side ([Validation](/guide/validation)).

| Core FWHM, 1 GHz train | Neighbour offset | Bin leak |
| --- | --- | --- |
| $79.335$ ps (fitted to Diamanti *et al.*) | $29.7\sigma$ | $7.2\times10^{-175}$ recomputed; `0.0` as shipped |
| below $\sim50$ ps | — | `0.0` by arithmetic, not a modelling cut |
| $300$ ps | — | $2.9\times10^{-13}$ |

| `jitter_split` choice | Reason |
| --- | --- |
| the Gaussian core is integrated to $9\sigma$ | its density there is below $10^{-18}$ of the peak, so the shipped `0.0` is the model, not an underflow (f64's smallest subnormal is $4.9\times10^{-324}$) |
| eight-point Gauss–Legendre panels, not a rational `erf` | that `erf`'s $\sim10^{-7}$ **absolute** floor would return signed noise where the true mass is $e^{-\text{hundreds}}$ |

```python
link = q.Link(
    modulation=q.DifferentialPhase(mu=0.2),
    channel=q.Fiber(length=100.0, alpha=0.2074),
    bob=q.Bob(
        detector=q.ClickDetector(
            eta=0.004,          # PRE-window: the gate's cost is derived
            dark=3.5e-8,
            jitter=79.3e-12,    # Gaussian core, FWHM
            window=100e-12,     # acceptance window inside the 1 ns period
            tail_frac=0.579,    # weight of the diffusion tail
            tail_time=295e-12,  # its time constant
        ),
        receiver=q.DelayInterferometer(delay=1, visibility=0.98),
    ),
    alice=q.Alice(laser=q.Laser(linewidth=0.0), symbol_rate=1e9),
    security=q.IndividualAttack(f=1.16),
)

res = link.run(symbols=1_000_000, seed=1)
res.explain["window_loss"]  # {'value': 0.539783, 'label': 'derived'}
res.explain["bin_leak"]     # {'value': 0.014951, 'label': 'derived'}
```

`eta` is the efficiency **before** the gate; the window's cost is derived from the
response. The two `explain` rows appear only when a response is described.

Jitter reaches the convolved slot train alone, as `dead_time` and `afterpulse` do,
so it is refused beside a pinned `IndividualAttack(qber=…)` and on the decoy
families. A window wider than the symbol period is refused.

### Double clicks are a bit, not a discard {#doubles}

A slot where both detectors fire is kept and given a **uniformly random bit**
(squashing), the coin drawn from the run's own Threefry stream so the result stays
reproducible. Discarding it would let an adversary who provokes double clicks shape
the sifted set. The coin is wrong half the time, worth roughly a factor of two on the
DPS rate here.

The rate is the Waks–Takesue–Yamamoto individual-attack bound
([Security](/guide/security#the-wty-individual-attack-bound)).

## Round-robin differential phase shift {#rrdps}

Alice sends $L$ pulses with random $0/\pi$ phases; Bob picks a delay $r$ uniformly
from $1..L-1$ and reads the phase difference of **one** random pair [SYK14].

**The privacy amplification reads no observable.** `rrdps_leak(l, nu_th)` is a
function of the packet length and a photon-number threshold alone
([against DPS and COW](/architecture#core-only)).

The price is sifting: a packet yields at most one bit, so every rate is divided by
$L$, and a longer train buys a tighter leakage bound at a throughput cost.
`rrdps_optimum` sweeps $l$ to `l_max` and $\nu_{\text{th}}$ to $(l-1)/2$; `L_MAX` is
4096, a train being $L$ pulses of a real modulator and the sweep quadratic in it.

**Three published assemblies of the same cost ship; they differ, and the literature
quotes all three.**

| Function | Assembly | Source |
| --- | --- | --- |
| `rrdps_phase` fed to $h_2$ | entropy form, the one the [SYK14] proof is printed in | [T15] Eqs. (3) and (4) |
| `rrdps_tag` | GLLP tagging | [Z17] Eq. (12); the same expression is [Y18] Methods |
| `rrdps_gllp` | per-photon-number tagging | [Z17] Eq. (13) |

`rrdps_gllp` $\le$ `rrdps_tag` $\le$ $h_2($`rrdps_phase`$)$ at every argument, the
second by concavity of $h_2$: the entropy form takes $h_2$ of a *mixed* error rate,
the tagging forms mix the entropies. All three are closed forms. At $\mu = 0.1$, $\eta = 0.1$,
$p_d = 10^{-6}$, $e_{\text{mis}} = 2\%$, $f = 1.16$ the optimum is $L = 92$,
$\nu_{\text{th}} = 15$ at $4.27\times10^{-4}$ bit/pulse, where the three costs read
$0.6506$, $0.6820$ and $0.8117$ bits per sifted bit.

`rrdps_collective` is [Y18] Eq. (1): $0.0527$ against `rrdps_leak`'s $0.0873$ at
$L = 92$, one photon. Past one photon: at $\nu = 1$ the simplex is an interval and a
bisection on a strictly decreasing derivative finds the maximum; at $\nu \ge 2$ it is
a concave maximisation over the $(\nu+1)$-simplex.

**Cut validity is the theorem; cut placement is a heuristic.** Eq. (1) maximises a
concave, positively homogeneous objective, so for *any* interior $y$

$$\max_x f(x) \;\le\; \max_x\left[f(y) + \nabla f(y)\cdot(x - y)\right] = f(y) - \nabla f(y)\cdot y + \max_i \nabla_i f(y).$$

The ascent picking $y$ may stall, cycle or stop early; the result is then looser by
the `bound - witness` gap, never wrong. [`src/lp.rs`](#lp) certifies the same
quantity from the same planes; a programme above the one-plane cap is refused as
unresolved.

| Field of `rrdps_simplex` | What it is |
| --- | --- |
| `bound` | the certified upper bound on Eve's information — the number `rrdps_collective` returns |
| `witness` | Eq. (1)'s objective where the ascent stopped. The maximum is *at least* this; it **proves nothing** and brackets from below |
| `tangent` | the same bound from one plane, no solver |
| `steps`, `iters` | ascent iterations and barrier Newton steps |

It refuses when the ascent does not close its bracket, when the programme fails to
match the tangent cap, when the certified value fails to beat `rrdps_leak`, and when
the bracket is wider than `reltol`. At $\nu = 1$ `rrdps_simplex(l, 1)` reproduces the
bisection to $9\times10^{-12}$ from above at every train length.

`q.rates.keyrate("rrdps", …)` selects the assembly by `cost`: `"entropy"`,
`"tagged"` and `"resolved"` are the three above, `"collective"` is
`rrdps_collective`. [MSK19]'s refined bound is the shape taking `p_src`, `q` and
`e_bit` in place of the receiver model. `rrdps_counts` gives $(q, e_{\text{bit}})$ from the
receiver model of [Y18] Methods — Bob's delay uniform on $1..l-1$, the packet kept
only when exactly one window clicks — and `rrdps_rate` assembles the rate in bits
per pulse.

## Basis keying with weak coherent pulses {#bb84}

BB84 on an attenuated laser. Poissonian pulses carry two or more photons at some
rate, which a photon-number-splitting adversary takes for free. The GLLP rate
concedes them, pays privacy amplification on single-photon detections only, and
charges error correction against the *whole* sifted gain:

$$R = q_{\text{sift}}\Big(q_1\left[1 - h_2(e_1)\right] - q_\mu\,f\,h_2(e_\mu)\Big).$$

$q_1$ and $e_1$ are not directly measurable; the decoy layer bounds them.

### The decoy layer {#decoy}

Alice varies the intensity between a signal $\mu$ and two decoys
$\nu_1 > \nu_2 \ge 0$. Eve cannot tell the intensity of a pulse, so the yields $Y_n$
are common to all three, and the measured gains and error rates invert into a lower
bound on $Y_1$ and an upper bound on $e_1$ [MQZL05]. $\nu_2 = 0$ is the optimal
vacuum+weak case.

The inversion is protocol-blind: BB84-WCP, [six-state](#sixstate) and
[SARG04](#sarg) consume the same $Y_1$ and $e_1$ and part at the
privacy-amplification factor ([the ordering](/roadmap#the-order-things-land-in-and-why)).

### The sampled pulse train {#sampled}

The bound consumes gains and error rates, produced two ways. `decoy_gain`, the
default, integrates the Poisson photon-number distribution against the detector
yields in closed form. `alice=q.Alice()` on a basis-keyed link samples instead:
pulses at one of three intensities reach two threshold detectors behind the basis
analyser, are sifted, and the gains and error rates are **divided out of integer
counts**. Both feed `decoy_bounds` and `bb84_rate`.

```python
res = q.Link(
    modulation=q.BasisKeying(
        decoy=q.Decoy(intensities=(0.48, 0.1, 0.0), probs=(0.8, 0.15, 0.05)),
    ),
    channel=q.Fiber(length=25.0, alpha=0.21),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
        receiver=q.BasisAnalyser(misalign=0.033),
    ),
    security=q.SplittingAttack(f=1.22),
    alice=q.Alice(),
).run(symbols=2_000_000, seed=5)

res.explain["stages"]     # 'pulse train + sifting'
res.qber                  # counted, not integrated
res.sifted, res.doubles   # exact integers
```

| Quantity | Closed form | Sampled |
| --- | --- | --- |
| Background $Y_0$ | $1 - (1 - p_d)^2$, derived from `ClickDetector.dark` | one dark gate per detector, $p_d$ each |
| Gain $Q_\mu$ | $Y_0 + h - Y_0 h$, $h = 1 - e^{-T\eta\mu}$ | clicks / pulses sent, per intensity |
| Error rate $E_\mu$ | $(e_0 Y_0 + e_{\text{det}} h) / Q_\mu$ | errors / sifted, per intensity |
| Double clicks | not represented | counted, kept, and given a uniformly random bit |
| Sifting | a factor $q_{\text{sift}}$ on the rate | a Bernoulli thinning of the clicks, checked by `sift_rate` |

The sampled path is BB84's alone ([Protocol layer](/guide/link#what-runs-today)).

`ClickDetector.dark` is $p_d$, **one detector in one gate**. `Link` derives
$Y_0 = 1 - (1 - p_d)^2$ over the two gates a bit reads — two detectors behind the
basis analyser, or one detector across both slots of an [intensity-keyed](#cow)
pair — and reports it beside `dark`. [MQZL05]'s $Y_0 = 1.7\times10^{-6}$ is twice
Gobby, Yuan and Shields' per-cycle $P_e = 8.5\times10^{-7}$: pass
$8.5\times10^{-7}$ and `q.Link` reproduces their $Y_0$ to one part in $10^{10}$.

The two paths agree at $p_d = 0$ and part at $O(p_d)$: the closed form charges
**every** background click at $e_0 = 1/2$; the simulation resolves the coincidence,
so a background click on the detector the signal already lit is not an error. The
simulated error is lower by

$$\Delta(E_\mu Q_\mu) = h\,e_{\text{det}}\,p_d + \tfrac{1}{2}h\,p_d(1 - p_d),$$

the double-count correction `decoy_gain` keeps and [MQZL05] drop.
`test_dark_convention` in `test/discrete.py` asserts both directions.

| Case, at $T\eta = 1/2$ | QBER |
| --- | --- |
| closed form, $p_d = 10^{-2}$ | $0.07288$ |
| sampled, $p_d = 10^{-2}$ | $0.06778$ — $9.5$ binomial sigmas below; no sample size closes it |
| closed form handed $p_d$ where it wants $Y_0$ | $0.05371$ — the background undercharged by half; the only way a sampled value lands *above* its twin |
| GYS hardware, $p_d \sim 10^{-6}$ | the gap is eleven orders below the gain |

A finite sample's numbers are kept as measured: a vacuum arm with no click reports a
gain of zero, which `decoy_bounds` reads as a *smaller* single-photon yield.
`Decoy.probs` exists because the weakest arm's count sets the bound's precision.

## Six-state keying {#sixstate}

`q.BasisKeying(bases=3)` keys the bits into **three** mutually unbiased qubit bases
[B98]. Bob flips his bit when he measured $\sigma_y$, so all three bases read as
correlated.

**Three bases make the channel estimate tomographically complete; that is the
advantage.** A Bell-diagonal state has four weights. BB84 measures two error rates
and leaves Eve one free parameter; six-state measures three and fixes all four.
`sixstate_bell` is that inversion — $\lambda_1 = 1 - (e_x + e_y + e_z)/2$ and its
cyclic partners — and refuses, rather than clamps, rates violating a triangle
inequality: a negative weight means the bases disagree on the shared state.

Eve's Holevo information is $h_2(e_x)$ for BB84 and $H(\lambda) - h_2(e_z)$, the
Shannon entropy of the four weights, here; thresholds and sifting:
[Security](/guide/security#qubit). The formulas are [S09] Appendix A,
Eqs. (A1)–(A6), whose Eq. (A9) is BB84's $I_E = h_2(e_x)$, so `bb84_rate` and
`sixstate_rate` differ in one factor. The Bell-diagonal parametrisation is [RGK05]
Eq. (10); unconditional security is [L01].

**Eq. (A4) runs, not Eq. (A5).** Same quantity, but (A5) divides by $e_z$, which
vanishes on a noiseless key basis; they agree to $5\times10^{-15}$ wherever (A5) is
defined.

`sixstate_eve(e_x, e_y, e_z)` is **not symmetric**: the key basis is Z, the third
slot, and only $e_z$ enters the subtracted term. `sixstate_holevo(Q)` is the
depolarising case $Q + (1-Q)h_2\!\left(\frac{1 - 3Q/2}{1 - Q}\right)$, the one a
decoy layer reaches, `decoy_bounds` returning one $e_1$.

```python
res = q.Link(
    modulation=q.BasisKeying(
        decoy=q.Decoy(intensities=(0.48, 0.1, 0.0)), bases=3
    ),
    channel=q.Fiber(length=25.0, alpha=0.21),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
        receiver=q.BasisAnalyser(misalign=0.033),
    ),
    security=q.SplittingAttack(f=1.22),
).run()

res.key_rate                # 5.221e-04, against BB84's 6.670e-04 on the same span
res.explain["sift"]         # {'value': 0.3333..., 'label': 'pinned'}
res.explain["chi_e1"]       # {'value': 0.1712, 'label': 'derived'}  Eve, at e1
```

`test_third_basis_pays` in `test/protocols.py` asserts both directions: at an
*equal* sifting factor six-state gives strictly more key on the same decoy bounds; at
the uniform factors, $1/3$ against $1/2$, it gives less. Biasing the basis choice
sends `sift` towards 1; `q.BasisKeying(bases=3, sift=0.81)` pins one. Refused on this
branch: [Protocol layer](/guide/link#what-runs-today),
[roadmap](/roadmap#refusals).

## SARG04 {#sarg}

`q.BasisKeying(announce="pair")` sends the four BB84 states and changes only the
classical post-processing [SARG04]. Alice announces one of four non-orthogonal
**pairs** containing the state she sent; Bob keeps the round only when his outcome is
orthogonal to one member, excluding it and naming the other. A BB84 rig runs SARG04
unchanged.

Sifting is $e_{\text{det}}/2 + 1/4$: misalignment **raises** it, a rotated outcome
being able to be orthogonal to a member the aligned one was not. Key survives on one-
**and** two-photon signals, a splitting adversary holding one photon of a pair still
facing two non-orthogonal states. Tolerances: [Security](/guide/security#qubit).

The unconditional rate is [FTL06] Eq. (39), on the Bell-diagonal relations of their
Theorems 1 and 2 (Eqs. (9) and (10)), after [TL06]. The detector model, yields and
gains are their Eqs. (40)–(43).

**Both privacy-amplification terms have a domain neither equation states.** Past
it, $1 - H(Z_1|X_1)$ and $1 - h_2(e_{p2})$ turn back *upward*, so a literal reading
credits key from a broken channel.

| Cutoff | Why there |
| --- | --- |
| two-photon, $e_2 = 1/6$ | the stationarity condition of Eq. (10)'s minimisation puts the root of $e_{p2}(e) = 1/2$ at $w = 1/6$ with $w = 1/3 - e$ |
| single-photon, $e_1 = 1/3$ | three independent statements: $H(Z_1|X_1)$ reaches one bit; the worst-case phase error $(3/2)e_1$ reaches $1/2$; [FTL06]'s general-POVM analysis shows Eve can always force $e_b \ge 1/3$ |

Neither cutoff refuses its argument, so a distance sweep runs through both
([Validation](/guide/validation#the-threshold-detector-and-qubit-rows)); each returns
the value that credits zero. Uncut, the single-photon term returns $+0.090$ at
$e_1 = 0.45$.

```python
res = q.Link(
    modulation=q.BasisKeying(
        decoy=q.Decoy(intensities=(0.16, 0.1, 0.0)), announce="pair"
    ),
    channel=q.Fiber(length=25.0, alpha=0.21),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
        receiver=q.BasisAnalyser(misalign=0.033),
    ),
    security=q.SplittingAttack(f=1.22),
).run()

res.key_rate            # 4.275e-05
res.qber                # 0.06256, the error rate on CONCLUSIVE rounds
res.explain["sift"]     # {'value': 0.2665, 'label': 'derived'}  -- 0.25 + e_det/2
res.explain["q2"]       # {'value': 0.0, 'label': 'absent'}
```

**`q2` is reported absent: nothing in the decoy layer bounds $Y_2$.** The shipped
rate sits *below* the infinite-decoy one the literature publishes.
`test_two_photon_absent` in `test/protocols.py` asserts that the infinite-decoy $Q_2$
is positive and that adding it would have paid. `sarg_yield` computes that limit,
which [FTL06] take explicitly; no finite-decoy $Y_2$ bound is invented.

`sarg_ceiling` is not a security bound: [BGKS05] Eq. (106), an *upper* bound over a
restricted class of incoherent attacks, with no decoy states, no dark counts, no
limiting distance, and three corrections its authors name dropped. It scales as
$t^{3/2}$ at $\mu_{\text{opt}} = 2\sqrt t$ against BB84's $t^2$ at
$\mu_{\text{opt}} = t$. [FTL06] reach the **opposite ordering** under general attacks
with decoy states — BB84 ahead at every distance — against that paper's Fig. 4. Both
readings are in `test/sarg.py`.

## Two-state keying {#b92}

`q.TwoStateKeying` sends $|\alpha\rangle$ or $|-\alpha\rangle$ [B92] into a
`q.NullingReceiver` ([Protocol layer](/guide/link#two-state)).

| Branch | Phase error | Source |
| --- | --- | --- |
| `reference=False`, the default | **derived** from the loss, the overlap and the observed bit error | [TL04], extending the loss-free proof of [TKI03] |
| `reference=True` | **supplied**, on `q.DiscriminationBound(e_phase=…)`, and labelled *pinned* as [`cow_rate`'s](#cow) is | [K04], implicit over a virtual entanglement picture, as is [TLKB09] Eq. (15) over experimental bounds; neither collapses to a closed form in $(\mu, T, \eta, V, p_d)$ |

### The unambiguous-discrimination boundary is where the rate dies {#usd}

USD breaks the protocol only past the loss at which Eve can run it and still meet
Bob's expected count, $L = |\langle\phi_0|\phi_1\rangle|$; [TL04] **price** that
boundary. Below it the derived phase-error bound is finite and the rate positive; at
it the bound reaches $1/2$ and the rate zero. `test_usd_boundary` in `test/b92.py`
and `test/protocols.py` pins that nothing raises.

**$L = c$ is the noiseless boundary.** Depolarising noise moves the crossing *in*
([Security](/guide/security#qubit)); quote it with the noise it was computed at.

`b92_limit(L, f_ec)` returns the largest depolarising rate at which plain B92
distils key, and the overlap attaining it.

| $L$ | Depolarising limit at $f = 1$ | Overlap $c$ attaining it | [TL04] Fig. 2 |
| --- | --- | --- | --- |
| $0$ | $0.03379$ | $0.68125$ | "$p \approx 0.034$" |
| $0.2$ | $0.02313$ | $0.75$ | "$0.023$" |
| $0.5$ | $0.01215$ | $0.85625$ | "$0.012$" |

Their Fig. 2(b) plots the **square** $|\langle\phi_0|\phi_1\rangle|^2$: $0.46410$,
$0.56250$ and $0.73316$ at those losses.

### The one seam, stated plainly {#b92-seam}

[TL04] is a **single-photon** proof; `b92_detect` is a coherent-state receiver.
`b92_point`'s plain branch takes the overlap from $e^{-2\mu}$, the loss from
$\eta T$ and the bit error from the fringe and dark floor, and prices the phase error
with the single-photon gain their model attaches — **not** the coherent conclusive
rate, which is larger and which their estimator answers non-monotonically. The rate
is Bob's conclusive rate times the CSS bracket, capped by the beam-splitting ceiling:
**not a security proof** for weak coherent pulses, which [TL04]'s conclusion names
open. `b92_plain` has no seam: single-photon source, their channel, bound and rate.

`b92_ceiling` charges the multiphoton pulses the proof does not model: Eve takes the
$(1-t)$ share, her share carries the two states at overlap $e^{-2(1-t)\mu}$, and
unambiguous discrimination hands her the bit with no disturbance. `b92_point` returns
$\min(\text{rate}, \text{ceiling})$, so a signal brightened past the `b92_optimum`
crossing follows the ceiling down to zero.

**The reference branch's $O(t)$ scaling holds only far from gate saturation**:
`p_fil` is $1 - e^{-2\eta t\mu(1 \pm V)}$ per port. On `test/b92.py`'s receiver
($\eta = 0.2$, $V = 0.98$, no dark counts) the chord slope over $t \in [10^{-4}, 1]$
is $0.99762$ at $\mu = 0.05$ and $0.95454$ at $\mu = 1$; over $[10^{-4}, 0.1]$:
[Validation](/guide/validation#the-threshold-detector-and-qubit-rows).

```python
res = q.Link(
    modulation=q.TwoStateKeying(mu=0.23),      # Koashi's optimum, overlap 0.6313
    channel=q.Channel(T=0.8),
    bob=q.Bob(
        detector=q.ClickDetector(eta=1.0, dark=1e-8),
        receiver=q.NullingReceiver(visibility=0.995),
    ),
    security=q.DiscriminationBound(f=1.16),
).run()

res.key_rate                        # 0.1016
res.qber                            # 0.003522
res.explain["overlap"]              # 0.6313, derived
res.explain["e_phase"]              # 0.1352, DERIVED on the plain branch
res.explain["discrimination"]       # 0.3687 -- the transmittance USD takes over at
res.explain["ceiling"]              # the beam-splitting cap
```

A supplied `e_phase` on the plain branch is a **floor** under the derived bound
(`test_supplied_floor`); `explain()["e_phase_floor"]` reads *absent* or *pinned*.
Refused: [Protocol layer](/guide/link#what-runs-today).

## Polarisation encoding {#polarisation}

`BasisKeying` leaves the carrier open and takes the analyser's misalignment as
given. `PolarisationKeying` names the carrier: fibre birefringence rotates the
polarisation, which Bob reads as basis misalignment.

Both bases are **linear** and turn by the same angle under a rotation of the linear
axes, so one angle serves the protocol. A Jones rotation by $\theta$ carries
$\ket H$ to $\cos\theta\ket H + \sin\theta\ket V$, so the wrong-detector
probability is $\sin^2\theta$, and for $\theta \sim \mathcal N(0, \sigma^2)$,

$$e_{\text{pol}} = \tfrac{1}{2}\left(1 - e^{-2\sigma^2}\right),
\qquad V_{\text{pol}} = 1 - 2e_{\text{pol}} = e^{-2\sigma^2}.$$

Three contrasts multiply. Adding error rates would double-count events where both
flip and push the total past $1/2$:

$$e = \tfrac{1}{2}\Big(1 - (1 - 2e_{\text{optics}})\,V_{\text{drift}}\,V_{\text{PMD}}\Big).$$

| Contrast | Form | Notes |
| --- | --- | --- |
| analyser optics | $1 - 2e_{\text{optics}}$ | `BasisAnalyser.misalign`, a property of the receiver |
| frame drift | $V_{\text{drift}}$, see [tracking](#tracking) | $\sigma$ is a **Jones** angle, not a Poincaré angle: the same rotation is $2\theta$ on the sphere |
| PMD | $V_{\text{PMD}} = \exp(-\Delta\tau^2/8w^2)$ at $\Delta\tau = D_{\text{PMD}}\sqrt L$ | normalised *field* overlap of the two principal states for a Gaussian envelope of intensity rms width $w$: the square root of `mode_overlap` at equal widths, as a first-order fringe contrast must be. Evaluated at the **worst** input polarisation |

### The reference-frame tracking model {#tracking}

The two models take disjoint parameters.

| `tracking` | Reads | Contrast | $e_{\text{pol}}$ at the stated setting |
| --- | --- | --- | --- |
| `"tracked"` | `drift` $= \sigma$, the residual rms angle the compensator leaves | $e^{-2\sigma^2}$ | $2.50\times10^{-5}$ at $\sigma = 5$ mrad |
| `"free"` | `rate` $= r$ in rad/$\sqrt{\text{s}}$ and `interval` $= T$ since the last alignment | $\dfrac{1 - e^{-2r^2T}}{2r^2T}$ | $4.97\times10^{-3}$ at $r = 0.1$, $T = 1$ s |

The free-running form is the *time average* of $e^{-2r^2 t}$ over $t \in [0, T]$,
a frame diffusing at $r$ having angle $\mathcal N(0, r^2 t)$ after $t$. The two rows
are the **same fibre**, a factor of **199** apart; `tracking` has no permissive
default.

```python
q.Link(
    modulation=q.PolarisationKeying(
        decoy=q.Decoy(intensities=(0.48, 0.1, 0.0)),
        frame=q.ReferenceFrame(drift=0.05),
    ),
    channel=q.Fiber(length=25.0, alpha=0.21),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
        receiver=q.BasisAnalyser(misalign=0.033),
    ),
    security=q.SplittingAttack(f=1.22),
).run()
# qber 0.0354 against 0.0330, and 89.8% of the key rate
```

| Reported | Label |
| --- | --- |
| `misalign_optics` | *pinned*, at the analyser's value |
| `pol_contrast`, `misalign` | *derived* |
| `tracking`, `drift` (or `drift_rate` and `drift_interval`), `dispersion`, `pulse_width`, `dgd` | the inputs that produced them |

The rotation is of the *linear* axes. A fibre's birefringence is an arbitrary $SU(2)$
element taking linear polarisation to elliptical, strictly worse for at least one
basis, so the misalignment here is a floor on a span's damage. `bases` and `announce`
are not fields here: six-state and SARG04 run through `q.BasisKeying`.

## Intensity keying {#cow}

Coherent one way encodes the bit in *which slot of a pair is empty*.

| Slot pair | Meaning |
| --- | --- |
| (empty, pulse) | logical 0 |
| (pulse, empty) | logical 1 |
| (pulse, pulse), a small fraction | decoy sequence, sent to a monitoring interferometer measuring coherence between adjacent non-empty pulses |

No active basis choice and no phase modulator on the data line. Its security
analysis was revised:

| | Original (Stucki et al.; Branciard, Gisin and Scarani) | After the sequential zero-error attack |
| --- | --- | --- |
| Eve bounded through | monitoring-line coherence | the phase error, supplied rather than derived |
| Rate scaling | linear in $\eta$ | $O(\eta^2)$ at most |
| Reach | hundreds of kilometres | below decoy BB84 at range |
| Due to | — | [GTWC20]; [TC21] |

COW's signal states are linearly independent and its vacuum slots break coherence
across the sequence, so Eve discriminates blocks unambiguously and resends them
separated by vacuum, holding *both* the data-line error rate and the monitoring
visibility at their unattacked values while knowing the key. qkd implements the
[G22] phase-error form,

$$R = q_z\left[1 - h_2(e_{\text{phase}}) - f\,h_2(e_{\text{bit}})\right],$$

so **`e_phase` is an input on `q.Link`.**

| Function | What it is |
| --- | --- |
| `cow_rate` | the phase-error form above; reads `e_phase` and never a visibility |
| `cow_visibility` | from the two monitoring detectors; *not* wired into `cow_rate`, a visibility of 1 being consistent with a full zero-error attack. A hardware diagnostic |
| `cow_ceiling` | the model-independent limit $(1 - f_{\text{dec}})(1 - e^{-t\mu})$ — Alice sends a bit signal and Bob sees a click — no security analysis exceeds |

The sequential-attack bound's cap on $\mu$, falling linearly with $t$ and forcing the
quadratic scaling, is not reproduced ([roadmap](/roadmap#gaps)). [COW′](#sdp) derives
`e_phase`.

### Finite modulator extinction {#extinction}

Intensity modulators reach $20$ to $30$ dB on/off contrast. With
$r = 10^{\mathrm{ER}/10}$ the empty slot carries $\mu_{\text{res}} = \mu / r$,
coherent with the signal. `IntensityKeying.extinction` is the dB figure, `None` by
default. The **on** slot keeps $\mu$ and the pair's flux grows as extinction worsens;
fixing the flux instead would let a rising residual pay for itself with a falling
gain.

#### The data line

The empty slot clicks with $p_{\text{res}} = 1 - e^{-t\mu_{\text{res}}}$, with no
dark term, $Q_0$ having charged it. Three outcomes, composed, not added:

| Outcome | Probability | Error |
| --- | --- | --- |
| signal slot alone clicks | $Q_0(1 - p_{\text{res}})$ | $e_0$ |
| empty slot alone clicks | $(1 - Q_0)p_{\text{res}}$ | always |
| both slots click | $Q_0 p_{\text{res}}$ | $1/2$, by the squashing coin |

giving $Q = Q_0 + p_{\text{res}} - Q_0 p_{\text{res}}$ and

$$e_{\text{bit}} = \frac{e_0 Q_0(1 - p_{\text{res}}) + (1 - Q_0)p_{\text{res}}
+ \tfrac{1}{2}Q_0 p_{\text{res}}}{Q}.$$

Ambiguous slots get a random bit, as on the phase-keyed path. $e_{\text{bit}}$
saturates at $1/2$; beyond it $1 - h_2(e)$ turns back upward.

#### The coherence check

**The fringe does not move.** A coherence check sits where both sides carry signal
pulses, so the measured visibility, `cow_visibility`, is what a perfect modulator
gives. What changes is the light it covers: a monitoring sequence emits $2\mu$, all
covered; a bit sequence emits $\mu(1 + 1/r)$, of which $\mu$ is covered.

$$m = \frac{1 + f_{\text{dec}}}{1 + f_{\text{dec}} + (1 - f_{\text{dec}})/r},
\qquad V_{\text{cert}} = m\,V_{\text{opt}}.$$

`cow_monitor` returns the pair. $V_{\text{cert}}$ is a **diagnostic and reaches no
rate**; `e_phase` stays supplied.

The residual raises the **gain**, to which a key rate is proportional, so counting
only the extra clicks would report *more* key for a *worse* modulator.

| `extinction` | $\mu_{\text{res}}$ | gain | $e_{\text{bit}}$ | monitored | rate vs ideal |
| --- | --- | --- | --- | --- | --- |
| `None` | $0$ | $0.30233$ | $2.000\%$ | $1$ | $1.00$ |
| $30$ dB | $5\times10^{-4}$ | $0.30258$ | $2.099\%$ | $0.9992$ | $0.95$ |
| $25$ dB | $1.58\times10^{-3}$ | $0.30312$ | $2.311\%$ | $0.9974$ | $0.85$ |
| $20$ dB | $5\times10^{-3}$ | $0.30483$ | $2.977\%$ | $0.9919$ | $0.54$ |
| $15$ dB | $1.58\times10^{-2}$ | $0.31022$ | $5.025\%$ | $0.9748$ | $0$ |

Parameterisation, from `test/discrete.py`: data line at $T = 1$,
$t_{\text{data}} = 0.72$, $\mu = 0.5$, $\eta_d = 0.8$, a $10\%$ monitoring tap,
$2\%$ misalignment, $p_d = 10^{-6}$ per detector per gate — so
$Y_0 = 2\times10^{-6}$, the pair's two slots — $e_{\text{phase}} = 0.20$,
$f = 1.1$. The ceiling rides on $\mu + \mu_{\text{res}}$, the residual being light
Alice sent.

## COW′ and the semidefinite programme {#sdp}

`src/sdp.rs` solves

$$\min\; c\cdot v \quad\text{subject to}\quad F_k(v) = F_k^0 + \sum_i v_i F_k^i \succeq 0,$$

every block dense, symmetric and single-digit. Method, refusals and the other two
solvers: [roadmap](/roadmap#solver), [Security](/guide/security#sdp).

### What it buys COW′ {#cow-prime}

COW′ is the two-pulse vacuum-decoy variant. Its four sequences have overlaps fixed
by $\mu$; Eve's channel and Bob's fixed detectors compose into one sub-POVM, and the
phase error is **linear** in it. The largest phase error consistent with the eight
monitoring gains is therefore an SDP value, solved by `sdp_phase`. The problem is
[SP26] Sec. VI D, replacing the analytic estimator of [G22] Eqs. (7) and (12),
shipped as `sdp_analytic`.

The SDP cannot be looser than Cauchy–Schwarz, which maximises over a superset of the
measurements the record allows; `test_below_analytic` in `test/sdp.py` would catch a
sign or index slip in the assembly.

Reach against the source is a [declared miss](/guide/validation#misses).

## The relative-entropy proof for discrete modulation {#relent}

`dm_secure` in `src/dmcs.rs`, on the complex-Hermitian numerics of `src/herm.rs`,
implements **both** steps of [WLC18] for protocol 2 — heterodyne, reverse
reconciliation — of [LUL19].

| Field | What it is |
| --- | --- |
| `upper` | step 1, Frank–Wolfe over $f(\rho) = D(\mathcal{G}(\rho)\,\|\,\mathcal{Z}[\mathcal{G}(\rho)])$; certifies nothing |
| `bound` | step 2, Theorem 3 linearised at the step-1 state and dualised; the proof |
| `key` | `bound - p_pass * delta_ec`, **unclamped** — a negative value is how far past the working distance the point sits |
| `viol`, `zeta` | Theorem 3's $\varepsilon'$, the largest constraint violation a truncated state cannot avoid; and the continuity price of the perturbation |

The dual point is certified by its **spectrum**, not by the Cholesky that accepted
it: the least eigenvalue is measured and the point shifted down until positive.
Limits and reach: [roadmap](/roadmap#solver).

`q.CertifiedBound(cutoff=…)` stands for `q.Asymptotic` on a `q.PhaseShiftKeying`
link, or is called directly. `certify(modulation, eta, xi)` is untrusted: `eta` is
the **total** transmittance from Alice's output to Bob's measurement, `xi` the
input-plane excess noise in the closed form's folding, $\xi + 2v_{el}/(\eta T)$.
`certify_trusted(modulation, eta, xi, eta_d, v_el)` is [LL20]: `eta` the channel
alone, the receiver's `eta_d` and `v_el` apart. Both misuses are the referring-plane
error: a bare span transmittance in `certify()` overstates the rate; a folded one in
`certify_trusted()` counts the loss twice.

## The linear programme {#lp}

`src/lp.rs` solves $\min c\cdot v$ s.t. $Av = b$, $v \ge 0$ over the diagonal cone,
on the dual, exposed as `_core.lp_dual` and reached by nothing in `qkd/`. Error side
and assembly trap: [roadmap](/roadmap#solver).

| Consumer | What it bounds |
| --- | --- |
| [`mdi_program`, `mdi_disturb`](#mdi-decoy) | MDI-BB84's $Y_{11}$ and $e_{11}$ from **all nine** cells of the decoy grid, against the closed form's seven |
| [`rrdps_simplex`, `rrdps_collective`](#rrdps) | [Y18] Eq. (1) past one photon, certifying the tangent planes an ascent placed |
| [mode pairing](#pairing) | **nothing.** `src/pairing.rs` does not import it |

## The postselection technique {#postselect}

The lift from an IID-collective proof to a **coherent**-attack one, for
permutation-invariant protocols on finite-dimensional systems, wired to nothing
([roadmap](/roadmap#postselect)). [CKR09] Theorem 1 is the original; what runs is
[NTZLT24] Corollary 3.1.

**`ps_lift` returns two costs as separate fields.**

| Cost | What it does |
| --- | --- |
| the factor $g_{n,x}$ | multiplies a **failure probability**, upward. It does **not** scale a key rate |
| $2\log_2 g_{n,x}$ | subtracts from the **key length** in bits |

It errs toward the larger epsilon, the larger $g$, the shorter key. `ps_secrecy`
implements [NTZLT24]'s **sum with the square root**, never [CKR09]'s max, which is
strictly smaller at every admissible allocation and so under-reports the failure
probability; no argument produces it.

**The epsilon the lift demands is usually not representable in f64**, so the budget
side is in logarithms. At [NTZLT24]'s worked point — $x = 52$,
$n = 1.08\times10^{13}$ — $\log_2 g$ is $1988.2$, and a coherent target of
$10^{-10}$ needs an IID secrecy parameter near $2^{-2021.4}$; f64 stops at
$2^{-1074}$, and `0.0` reads as unconditional security. `ps_budget` returns $\log_2$
of it, `ps_thirds` allocates in $\log_2$, and `ps_epsilon`, the one function forming
a probability, **refuses** a lifted epsilon at or above 1.

| Condition | |
| --- | --- |
| **C1** permutation invariance | enforceable by prepending a public random permutation of the rounds, at about $n\log n$ public random bits and no pre-shared key |
| **C2** finite dimension | failed by every optical protocol, on both sides ([roadmap](/roadmap#postselect)) |
| **C3** a fixed marginal | built from the source description, extending the technique to prepare-and-measure protocols; [CKR09]'s results are entanglement-based only |
| **C4** an IID proof of the right shape | a proof already stated against general attacks has no such decomposition; its epsilon fed here bounds nothing while looking conservative |

`ps_family` refuses every shipped family, naming the condition each fails
([per family](/roadmap#postselect)). The lift removes one assumption and adds device
assumptions on the way in, so it is no step toward [device independence](#not-di).

## The entropic uncertainty relation {#eur}

Nothing in `qkd/` reaches it: `eur_family` refuses every shipped protocol, naming
the condition each fails.

**Three statements, one chain, three different quantities.**

| Statement | Scope | Here |
| --- | --- | --- |
| [MU88] — $H(X) + H(Z) \ge q$, $q = \log_2(1/c)$, $c = \max_{x,z}\lvert\langle x\vert z\rangle\rvert^2$ | a **memoryless** adversary | `eur_quality`, `eur_conjugate`, `eur_misalign` compute $q$. **No rate comes from this form**: a QKD adversary has memory |
| [B10] — $H(X\vert B) + H(Z\vert B) \ge q + H(A\vert B)$ | **collective** attacks, in their words | `eur_memory`, and `eur_asymptotic` for the two-adversary form through Devetak–Winter. Not a coherent-attack number |
| [TR11] Theorem 1 — $H_{\min}^\varepsilon(X\vert B) + H_{\max}^\varepsilon(Z\vert C) \ge q$ | **coherent** attacks | `eur_smooth`. Reaches them with **no** de Finetti reduction, no [postselection technique](#postselect) and no asymptotic equipartition step |

The key length is [TLGR12] Eq. (2) — the smooth relation for $H_{\min}$, Serfling on
a random subset for $H_{\max}$, and the quantum leftover hash lemma to turn
$H_{\min}$ into hashed bits. `eur_length` is that formula and `eur_secret` is it
behind the family gate.

**Which side it errs on.** $q$ enters with a **plus**, so it must be a **lower**
bound on $-\log_2 c$: an overlap read too small *lengthens* the key, so
`eur_quality` takes an **upper** bound on $c$ and refuses $c > 1$. $H_{\max}$ enters
with a **minus**, so it must be an **upper** bound; `eur_smooth` and `eur_length`
refuse a result not below $nq$. $h_2$ turns downward past $1/2$, so a *worse*
tolerated error would report a *longer* key; `eur_length` saturates at $h_2 = 1$,
the correct bound, the Hamming ball of radius $n/2$ being the whole space.

**The continuous-variable branch.** [F14] give the overlap of position and momentum
bins of widths $dq$ and $dp$ through the 0th radial prolate spheroidal wave
function, a function of the **product** $dq\,dp$ alone since dilations are unitary.
`eur_binned` ships

$$q \ge \log_2\frac{2\pi}{dq\,dp},$$

that at $S = 1$. It is a **theorem**, not the small-bin approximation the papers call
it: $c$ is the largest eigenvalue of the positive trace-class operator
$Q[dq]\,P[dp]\,Q[dq]$, whose trace is $dq\,dp/2\pi$, so $c \le dq\,dp/2\pi$.
Against the Slepian eigenvalue it overstates the overlap by $6.9\times10^{-11}$
relative at $dq = dp = 0.01$, $6.9\times10^{-7}$ at $0.1$ and $0.69\%$ at $1.0$,
costing $q$ $1.0\times10^{-10}$, $1.0\times10^{-6}$ and $9.99\times10^{-3}$ bits.
**The special function is not evaluated**: a quadrature of the sinc kernel certifies
neither side of the eigenvalue, and a $q$ too large is insecure.

Two conventions. [F14] prints the constant **twice, inconsistently** — Results as
$(dq\,dp)/2\cdot S^2$, Appendix Eq. (A2) as $(1/2\pi)\,dq\,dp\cdot S^2$, a factor
of $\pi$ apart, the Results printing also dropping the square inside the norm.
**Take the appendix**, consistent with the paper's $dq\,dp \to 0$ limit. It sets
$[Q,P] = i$, $\hbar = 1$, vacuum variance $1/2$ — [this tree's
convention](/guide/conventions), not $\hbar = 2$. `eur_binned(1, 1)` is
$\log_2 2\pi = 2.6515$ bits, the differential relation.

## Entanglement-based basis keying {#pairs}

A photon-pair source sends one photon of each pair to each party; both measure in
two conjugate bases, and the key rides on the **coincidences**. Nothing is modulated
or prepared. The source may sit anywhere between the labs and is never trusted:
Alice and Bob certify the state by measuring in both bases.

| | |
| --- | --- |
| Component | `q.PairSource(brightness=…, rate=…, pumping="pulsed")`, run through `q.PairLink` |
| Source model | two independent two-mode squeezers, one per encoding mode pair: negative-binomial pair number, mean $\mu = 2\lambda$. $\lambda$ is `q.gaussian.Epr(r)`'s $\sinh^2 r$ |
| Rate | $R \ge q\,Q_\lambda\!\left[1 - h(\delta_p) - f\,h(\delta_b)\right]$, [MFL07] Eq. (5), with $\delta_b = \delta_p = E_\lambda$ by basis symmetry |
| $Q_\lambda$, $E_\lambda$ | derived from hardware, their Eqs. (9) and (10), not dialled |
| Security | `q.SymmetryBound`, or [`q.ViolationBound`](#e91). Asymptotic only |

### Where the source lives is the result {#position}

`channels=(alice's arm, bob's arm)` states the position. There is no `site=` field,
which could disagree with it.

At **constant total fibre** on [MFL07] Table 1 hardware ($\eta = 14.5\%$,
$e_d = 1.5\%$, $Y_0 = 6.02\times10^{-6}$, $f = 1.22$, $0.2$ dB/km):

| Source at | Zero-key crossing |
| --- | --- |
| the midpoint | **308.1 km** |
| the quarter point | **241.5 km** |
| a party's lab | **183.0 km** |

The gain depends on the **product** of the two arms, so this is no $\sqrt\eta$
effect and beats no repeaterless bound: the rate is linear in end-to-end
transmittance, log–log slope $1.02$. The reach is bought from **dark counts**: at
200 km the error is $2.86\%$ from the middle and $16.10\%$ from a lab, the second
past every threshold qkd ships.

A midpoint **source** is not the [relay](/guide/relay). A relay moves the *detectors*
to the middle and buys detector side-channel immunity; a pair source moves the
*source* and buys source-trust immunity, its detectors staying in the threat model.

## E91: pricing Eve by the violation {#e91}

E91 [E91] is the protocol above with a third setting spent on a CHSH test.
`q.ViolationBound` puts the observed violation where `q.SymmetryBound` puts a phase
error; that substitution is the difference.

| | `q.SymmetryBound` (BBM92) | `q.ViolationBound` (E91) |
| --- | --- | --- |
| Eve priced by | the phase error, under a **characterised-qubit** assumption | the observed CHSH magnitude, with **no assumption about the source state** |
| Bound | [MFL07], basis symmetry | [A07]: $\chi(S) = h\!\left(\tfrac{1 + \sqrt{S^2/4 - 1}}{2}\right)$, exactly $1$ at $S = 2$ and exactly $0$ at $S = 2\sqrt2$ |
| Zero-key error rate, $f = 1$ | $11.003\%$ | $7.149\%$ |
| Defaults | `e_phase=None`, taking the bit error; `f=1.22`; `sift=0.5` | `s` and `source` required; `f=1.22`; `sift=2/9` — Ekert's six settings drawn uniformly, two of the nine pairs coinciding and carrying the key |

**The violation is never the cheaper price**, and the $11.003\% \to 7.149\%$ gap is
what the state assumption was worth. `test_price_is_worse` in `test/protocols.py`
asserts the ordering on the same coincidences.

With trusted, characterised analysers E91's key rate **is** `pair_rate` [BBM92], and
`ekert_point` returns both numbers ([Architecture](/architecture#not-di)).

[E91]'s Eq. (3) sign convention makes $S$ negative at his settings, the modern one
positive; `ekert_rate` takes the magnitude. His Eq. (7), $-\sqrt2 \le S \le \sqrt2$,
is **not** the local bound of 2: it is the range over the intercept-and-resend family
his measure describes, a strict subset of local models. The rate is written against
the local bound 2, the weaker and safe one.

### `source=` is how the engine says this is not device independence {#not-di}

`q.ViolationBound.source` has **no default**: it decides whether the number is a
bound.

| `source` | What happens |
| --- | --- |
| `"measured"` | the rate is returned, and it is a collective-attack bound *given that estimate* |
| `"modelled"` | **refused.** An $S$ from a state model — `ekert_chsh`, `pair_chsh`, a visibility — prices one assumed state and bounds no unknown one. That model is the device-*dependent* assumption the CHSH bound avoids, so reading it as security is circular. `ekert_point` computes that number under its own label |
| `"device-independent"` | **refused**, naming three missing pieces — the tight bound beyond CHSH, entropy accumulation for coherent attacks, and an $S$ free of the detection loophole ([roadmap](/roadmap#refusals)). A claim needs detection efficiency above $0.924$ ([P09] Fig. 3) against `q.ClickDetector`'s default $0.20$ |
| anything else | refused, listing the three read |

```python
res = q.PairLink(
    source=q.PairSource(brightness=0.01),
    detectors=(q.ClickDetector(eta=0.2, dark=1e-6),
               q.ClickDetector(eta=0.2, dark=1e-6)),
    channels=(q.Fiber(length=10.0), q.Fiber(length=10.0)),
    security=q.ViolationBound(s=2.7, source="measured"),
).run()

res.key_rate                              # 3.093e-05
res.e_phase                               # None -- the violation stands where one would
res.explain["e_phase"]                    # {'value': None, 'label': 'absent'}
res.explain["chsh"]                       # 2.5965, labelled 'diagnostic'
res.explain["key_symmetry"]               # what basis symmetry would have paid
res.explain["device_independent"]         # False, labelled 'structural'
```

`qkd.pairs.chsh(v)` models $S$ from a visibility under a singlet-plus-white-noise
assumption; it is labelled *diagnostic* and is what `source="modelled"` refuses. A
violation past Tsirelson's bound is refused; one at or below 2 buys no key,
$\chi(2) = 1$.

Published device-independent demonstrations: [N22] over 2 m of trapped-ion
separation; [L26], 624 hours for 1.2 million heralded pairs at 11 km.

## Basis keying into an untrusted midpoint {#mdi-bb84}

MDI-BB84. Alice and Bob send decoy weak coherent pulses, a bit in one of two
conjugate bases. An untrusted station interferes the arrivals on a balanced coupler,
splits each output on a polarising beamsplitter, reads four threshold detectors and
announces **which pair fired together**: a Bell-state projection, revealing neither
bit. Detector side channels leave the threat model; the proof never assumes the
station behaved.

| | |
| --- | --- |
| State | asymptotic, or a finite-key **length** under `q.TestBasisBound(block=q.RelayBlock(…))`. `q.BellDetector` on the relay gives the [continuous-variable midpoint](/guide/relay), `q.BellAnalyser` this one. Composes inside `q.Network` |
| Reached from | `q.Swap(alice=…, bob=…, relay=q.Relay(bell=q.BellAnalyser(…)), channels=…, security=q.TestBasisBound(f=…))`, with `q.BasisKeying(decoy=…, sift=1.0)` on **both** senders ([why](/usage#mdi-bb84)) |
| Defaults | `q.TestBasisBound(f=1.16, block=None)`; `block=None` is the asymptotic rate |
| Exams | [MDI · Forward](/tests/mdi_forward), [MDI · Decoy](/tests/mdi_decoy), [MDI · Anchors](/tests/mdi_anchors), [MDI · Parts](/tests/mdi_parts), [MDI · Guards](/tests/mdi_guards) |

### The rate {#mdi-rate}

[XCQL13] Eq. (1):

$$R \;\ge\; Q_{11}^{Z}\left[1 - H\!\left(e_{11}^{X}\right)\right] - Q_{\mu\nu}^{Z}\,f\,H\!\left(E_{\mu\nu}^{Z}\right)$$

| Symbol | What it is |
| --- | --- |
| $Q_{11}^{Z}$ | **gain** of rounds where each sender emitted one photon, both in the key basis, and the station announced: $\mu_a\mu_b e^{-\mu_a-\mu_b}Y_{11}$, applied by `mdi_gain` |
| $e_{11}^{X}$ | that pair's error rate in the **test** basis, standing in for the key basis's phase error |
| $Q_{\mu\nu}^{Z}$, $E_{\mu\nu}^{Z}$ | key-basis gain and QBER at the signal intensities, observed |
| $f$ | error-correction inefficiency, $1.16$ throughout this literature |

| Three ways to get this wrong | |
| --- | --- |
| **Per pulse *pair*, not per pulse** | $\mu_a\mu_b e^{-\mu_a-\mu_b}$ is applied **once** |
| **No sifting prefactor** | $Q^{Z}$ already counts key-basis rounds only. [`bb84_rate`](#bb84)'s explicit $q_{\text{sift}}$ would charge sifting twice |
| **The two error rates are in different bases** | privacy amplification pays on $e_{11}^{X}$, error correction on $E^{Z}_{\mu\nu}$. `bb84_rate` is not reused: its signature cannot say which basis each argument came from |

[MR12]'s single-photon-source rate, $Y_{11}[1 - H(e_{11}) - fH(e_{11})]$ (their
Eq. (7)), is this expression at $Q_{11} = Q^Z = Y_{11}$; `test_single_photon`
asserts the reduction.

### The two bases are different physics {#mdi-bases}

Only the test basis needs indistinguishable arrivals. The polarising beamsplitters
separate rectilinear signals, which never interfere, so interference enters the key
basis only through dark counts.

| | Key basis (rectilinear) | Test basis (diagonal) |
| --- | --- | --- |
| Forward model | `mdi_rect`, [MR12] Eqs. (51)–(54) | `mdi_diag`, their Eqs. (35) and (39) |
| Hong–Ou–Mandel indistinguishability | not needed | **required** |
| QBER with no dark counts | exactly $e_d$, the misalignment | $\to \tfrac14 + e_d/2$ as the signal weakens |
| Announcement rate | — | **twice** the key basis's in the weak-signal limit: it accepts the multiphoton coincidences the polarising beamsplitters reject |

**A test-basis QBER near 25% is the protocol working**: two independent coherent
sources have uncorrelated multiphoton content, and [T14] measure about 26% beside a
key-basis QBER under 0.5%. The decoy layer recovers $e_{11}^{X}$ at the
misalignment. `q.BellAnalyser` carries **two** misalignments, `misalign` and
`misalign_test`; one number for both overstates a midpoint rate. The four- and
seven-intensity protocols give the two bases *different* intensities for this reason.

### The decoy layer is two-dimensional {#mdi-decoy}

Both senders vary intensity independently, so the unknown is a **matrix** $Y_{nm}$
and the observations a **grid** $Q^{\mu_i\nu_j}$. The one-dimensional
[`decoy_bounds`](#decoy) cannot reach it.

`mdi_y11` implements [XCQL13]'s three-intensity analytic bound over a $3\times3$
gain grid. Its four-cell double differences kill the $n = 0$ and $m = 0$ rows
outright, the outer weighting kills $n = 2$, the $(1,1)$ coefficient is $1$, and
every surviving $n \ge 3$ coefficient is non-positive, so dropping them only lowers
the result. An ordering condition fixes that sign; otherwise the parties are
exchanged, the same theorem relabelled.

| | |
| --- | --- |
| Two of the nine cells are unread **by `mdi_y11`** | the closed form uses the vacuum row and column, the decoy block and the signal-signal cell, not `(signal_a, decoy_b)` or `(decoy_a, signal_b)`. `mdi_program` reads all nine as a linear programme over the truncated $Y_{nm}$ on [`src/lp.rs`](#lp) and bounds tighter; `mdi_disturb` is its twin over $e_{nm}Y_{nm}$. Both refuse a certified value outside $[0,1]$ or one failing to beat the closed form ("The value is a valid bound and a useless one"). Neither has a `q.` path |
| Infeasible data | $e_{11}$ returns $1/2$, never $0$, which would claim the single photons were error-free |

### A numerical trap, and where it bites {#mdi-numerics}

[MR12]'s Eq. (35) bracket, $1 + 2y^2 - 4yI_0(x) + I_0(2x)$, **vanishes** through
$O(x^2)$ and $O(1-y)$: every piece is within $O(\mu\eta)$ of $1$ and the answer is
$O((\mu\eta)^2)$. Literal f64 returns rounding noise below $\mu\eta \approx 10^{-5}$,
breaking the $\tfrac14$ limit. The engine regroups it as

$$2(1-y)^2 + 4(1-y)\left[I_0(x) - 1\right] + \left[I_0(2x) - 4I_0(x) + 3\right],$$

each bracket summed from its series with the leading terms removed **before** they
form, as `relay.rs`'s `h_ent` and `std.rs`'s `g_ent` are. The limit then holds to
$10^{-9}$ at $\mu = 10^{-8}$.

The repeaterless bound has the same trap: $-\log_2(1-\eta)$ written literally is 11%
wrong by 160 dB and `0.0` by 176 dB. Use `-log1p(-eta)/ln 2`. [Mode
pairing](#pairing)'s source paper prints the literal form.

### What it refuses {#mdi-refusals}

| Refused | Why |
| --- | --- |
| An intensity ladder that does not decrease strictly | equal settings put a zero in the bound's denominator |
| A gain grid that is not nine cells | it is row-major with Alice's intensity as the row, and is never reshaped |
| $Q_{11} > Q^{Z}$ | an inverted decoy bound, not rescaled |
| A decoy setting at or below its own vacuum, in `mdi_e11` | the same denominator |
| A `trusted=` flag anywhere on `q.BellAnalyser` | structural: an untrusted midpoint has no trusted variant |
| An `e_phase` on `q.TestBasisBound` | nothing to supply: unlike [`q.PhaseBound`](#cow), the test basis measures it |
| Finite-size security **from `q.Swap`** | **runs.** `q.TestBasisBound(block=q.RelayBlock(…))` reaches `_core.mdi_length` — [C14], with the analytical two-decoy estimation, Serfling's sampling transfer and Claim 3 across all six branches. It reads the $3\times3$ joint intensity probabilities as the outer product of the senders' `q.Decoy` weights, the basis bias from `q.BasisKeying(bias=…)`, and the code-string share from `q.RelayBlock(code=…)`. Refused instead: a block with no `bias=`, and a `bias=` with no block |

## Mode pairing {#pairing}

Two weak-coherent senders into one untrusted **single-photon**-interference
station; the two clicks forming a key bit are chosen **after** the announcement.
[ZZWM22]: Eq. (4) the pairing rate, Eq. (7) the key rate, Eqs. (90)–(104) the
forward model. Also published as *asynchronous MDI-QKD*; the phase-drift model behind
the pairing window is [X21] Sec. III.

**Not twin-field QKD**, and not covered by that [exclusion](/roadmap#exclusions): it
shares the $O(\sqrt\eta)$ scaling and reaches it differently.

**Why the scaling changes.** Alice and Bob emit independently; the station announces
which rounds clicked; only then are two clicked rounds paired into a bit. A round
succeeds on its **own**, not jointly with a partner fixed in advance, so a pair
arrives at $O(\eta_s) = O(\sqrt\eta)$ instead of $O(\eta_s^2) = O(\eta)$. `span`,
the maximal pairing interval, buys this: at `span = 1` the scheme **is** time-bin
MDI-QKD and the exponent returns to 1. On [X21] Table 1 hardware ($\alpha = 0.165$ dB/km,
$\eta_D = 0.70$, $p_d = 10^{-8}$, $f = 1.1$, X-basis misalignment angle
$\sigma = \pi/10$), the log–log slope in end-to-end transmittance between 300 and
400 km is **$0.525$** at `span` $= 10^4$ and **$1.003$** at `span` $= 1$.

| Quantity | What it is | Trap |
| --- | --- | --- |
| `pairing_pairs`, $r_p$ | pairs per **round** | carries the whole transmittance scaling |
| `pairing_sift`, $r_s$ | fraction of those pairs that are usable Z-pairs | $\approx 1/8$ at every distance ($0.1250$ at 300 km) |
| `pairing_single`, $q_{11}$ | **fraction** of sifted signal pairs whose two non-empty slots each held one photon | a fraction, not a gain: `mdi_gain`'s Poisson prefactor is already taken, and applying it again charges the source twice |

$r_p$ and $r_s$ are both dimensionless and both multiply in `pairing_rate`; swapped,
the result is wrong by the channel loss and still looks like a key rate.

**The phase error is not computed here**: [ZZWM22] Eq. (104) takes $e^X_{(1,1)}$ from
[MR12], which is [`mdi_yield`](#mdi-bb84)'s second return at
$\eta_a = \eta_b = \eta_s$. [ZZWM22] print the dark-count factor as $1 - p_d^2$ where
[MR12] have $(1 - p_d)^2$, differing at $O(p_d)$, $10^{-8}$ here; `mdi_yield`
follows [MR12].

`pairing_length` implements [X21]'s key-length equation and fluctuation appendix,
and `pairing_yield` the one decoy step that transfers. Two links of the chain are
absent and **no linear programme retires either**: $s_{11}^x$ lacks the phase slice
$m$, which no optimiser can manufacture, and $s_{11}^z$ lacks [X21]'s declared
vacuum, which `pairing_bases` has no setting for. The programme's own constraint row
is upstream too: its equalities are pairing-resolved gains $Q_{ab}$ at named pair
intensities, and `pairing_click` takes one intensity with the four patterns at a
quarter apiece. On `pairing_yield` the closed form is the vertex: [`lp_dual`](#lp)
climbs to it from below, relative $10^{-14}$ at $(\nu, \mu) = (0.05, 0.4)$. Station
symmetry, component tree and optimiser: [roadmap](/roadmap#refusals).

## Loss-tolerant source flaws {#flaws}

A *reading* of BB84's data when Alice's phase modulator applies
$\phi + \delta\phi/\pi$ for an intended $\phi$. [TCKLA14], with the device model
and four-state form of [PCT19].

`flaw_tolerant` reads the phase error **exactly**; `flaw_standard` **bounds** it
through a quantum coin and returns the coin imbalance in that slot
([Security](/guide/security#flaws)). At $\delta = 0.1$ rad, $p_d = 10^{-6}$,
$f = 1.16$, over $0.2$ dB/km fibre, the two are $0.4919$ and $0.4688$ bit at 0 km,
$4.90\times10^{-3}$ and $1.82\times10^{-3}$ at 100 km, and $4.73\times10^{-4}$
against exactly zero at 150 km.

`flaw_triangle` is the spanning condition as a number: $2.0$ for the ideal states,
rising near $\delta = \pi/3$ and falling to zero at $\delta = \pi$, where the
inversion has no solution and the guard has already refused.

`flaw_direct` measures the two virtual states, which Alice never sends: not a
protocol step but the **arbiter** `flaw_phase` is checked against. [TCKLA14]'s
theorem is that the two agree; `test_theorem_exact` in `test/flaws.py` pins them to
$10^{-15}$ across flaw, loss, dark count and misalignment.

Scope, the refusal to compose with `qkd.attacks` and the unreconciled line of
[TCKLA14] Appendix C: [Security](/guide/security#flaws),
[roadmap](/roadmap#gaps).

[TCKLA14] Appendix C prints $+\sin(\delta/2)$ where Wang prints $-\sin(\delta/2)$
for the same state: the direction of the Bloch $x$ axis, moving no number, every
quantity being symmetric under relabelling both bits.

## Detector attacks {#attacks}

Mechanisms and sources: [Security](/guide/security#attacks); module:
[Impairments](/guide/impairments#attacks). `Mismatch` is also Makarov, Anisimov &
Skaar, Phys. Rev. A **74**, 022313 (2006). `mismatch_rate` is Fung, Tamaki, Qi, Lo &
Ma, QIC **9**, 131 (2009), [arXiv:0802.3788](https://arxiv.org/abs/0802.3788),
Eq. (33) with the data-discarding argument and Eq. (34) as their general estimate.

## Post-processing {#reconcile}

Three quantities kept apart, `bridge()`, `pick()`'s refusal and tabulated Cascade
efficiency: [Architecture](/architecture#reconciliation-keeps-three-quantities-apart),
[Usage](/usage#qkd-reconcile). Not modelled: slice reconciliation, BICONF, polar and
rateless codes, and ETSI key-delivery accounting.

## Settled exclusions {#exclusions}

Four settled exclusions: [Status and roadmap](/roadmap#exclusions).

## What is not implemented {#gaps}

Refused capabilities: [roadmap](/roadmap#refusals). Not implemented and stated
nowhere else:

| Not implemented | |
| --- | --- |
| A composable CV proof against **general** attacks | the Gaussian de Finetti arithmetic ships ([roadmap](/roadmap#definetti)). On `test/keyrate.py`'s link it moves the minimum viable block from about $4\times10^6$ symbols under a collective attack to between $1.5\times10^7$ and $2\times10^7$; Leverrier quotes no worked length, so this is [⚙ structural](/guide/validation#levels). The [postselection technique](#postselect) and the [entropic uncertainty relation](#eur) ship as engines and reach nothing |
| Four- and seven-intensity MDI protocols | Zhou, Yu & Wang, PRA **93**, 042324 (2016); Wang, Xu & Lo, PRX **9**, 041012 (2019). The three-intensity bound ships |
| Entanglement-based **CV**-QKD as a protocol | the EPR picture only *derives* the prepare-and-measure bound; [`q.PairLink`](#pairs) shares none of that machinery |
| Source-side flaws beyond a modulator angle | correlated pulses and intensity-setting errors. `probe_rate` refuses a Trojan-horse key rate naming four missing pieces, the first being that `q.Decoy` sets its intensities with a modulator behind the very isolators the bound prices, spending one isolation budget twice |
| Field-level simulation of the basis-keyed train | the [sampled path](#sampled) reduces each pulse to a detection probability: no interferometer, no phase walk, no detector memory |
| Squashing / basis-independent detector models | the click family works from click probabilities directly |

## Where these live {#code}

`Link` refuses a security model that does not belong to the modulation:

```python
q.Link(
    modulation=q.BasisKeying(decoy=q.Decoy(intensities=(0.5, 0.1, 0.0))),
    channel=q.Fiber(length=50.0),
    bob=q.Bob(detector=q.ClickDetector(), receiver=q.BasisAnalyser()),
    security=q.Asymptotic(),
).run()
# NotImplementedError: BasisKeying takes SplittingAttack security
```

Entry point per engine: [Architecture](/architecture#the-core-modules).

See the [component reference](/guide/link#component-reference) for what each
component takes.

## References

| | |
| --- | --- |
| [A07] | Acín, Brunner, Gisin, Massar, Pironio & Scarani, Phys. Rev. Lett. **98**, 230501 (2007) |
| [B92] | Bennett, Phys. Rev. Lett. **68**, 3121 (1992) |
| [B98] | Bruß, Phys. Rev. Lett. **81**, 3018 (1998) |
| [B10] | Berta, Christandl, Colbeck, Renes & Renner, *Nature Physics* **6**, 659 (2010), [arXiv:0909.0950](https://arxiv.org/abs/0909.0950) |
| [BBM92] | Bennett, Brassard & Mermin, Phys. Rev. Lett. **68**, 557 (1992) |
| [BGKS05] | Branciard, Gisin, Kraus & Scarani, Phys. Rev. A **72**, 032301 (2005) |
| [C14] | Curty, Xu, Cui, Lim, Tamaki & Lo, Nat. Commun. **5**, 3732 (2014) |
| [CKR09] | Christandl, König & Renner, *Post-selection technique for quantum channels with applications to quantum cryptography*, Phys. Rev. Lett. **102**, 020504 (2009), [arXiv:0809.3019](https://arxiv.org/abs/0809.3019) |
| [DBL21] | Denys, Brown & Leverrier, Quantum **5**, 540 (2021) |
| [E91] | Ekert, Phys. Rev. Lett. **67**, 661 (1991) |
| [F14] | Furrer, Berta, Tomamichel, Scholz & Christandl, *J. Math. Phys.* **55**, 122205 (2014), [arXiv:1308.4527](https://arxiv.org/abs/1308.4527) |
| [FTL06] | Fung, Tamaki & Lo, Phys. Rev. A **73**, 012337 (2006), [arXiv:quant-ph/0510025](https://arxiv.org/abs/quant-ph/0510025) |
| [G22] | Gao et al., Opt. Express **30**, 23783 (2022), [arXiv:2107.09329](https://arxiv.org/abs/2107.09329) |
| [GLLP04] | Gottesman, Lo, Lütkenhaus & Preskill, *QIC* **5**, 325 (2004) |
| [GTWC20] | González-Payo, Trényi, Wang and Curty, PRL **125**, 260510 (2020) |
| [K04] | Koashi, Phys. Rev. Lett. **93**, 120501 (2004), [arXiv:quant-ph/0403131](https://arxiv.org/abs/quant-ph/0403131) |
| [L01] | Lo, Quant. Inf. Comput. **1**, 81 (2001) |
| [L15] | Leverrier, Phys. Rev. Lett. **114**, 070501 (2015), [arXiv:1408.5689](https://arxiv.org/abs/1408.5689) |
| [L17] | Leverrier, Phys. Rev. Lett. **118**, 200501 (2017), [arXiv:1701.03393](https://arxiv.org/abs/1701.03393) |
| [L26] | Lu et al., Science (2026), [10.1126/science.aec6243](https://doi.org/10.1126/science.aec6243) |
| [LL20] | Lin & Lütkenhaus, Phys. Rev. Applied **14**, 064030 (2020), [arXiv:2006.06166](https://arxiv.org/abs/2006.06166) |
| [LUL19] | Lin, Upadhyaya & Lütkenhaus, PRX **9**, 041064 (2019), [arXiv:1905.10896](https://arxiv.org/abs/1905.10896) |
| [LWLC24] | Lu, Wang, Li & Cao, [arXiv:2401.01727](https://arxiv.org/abs/2401.01727) |
| [MFL07] | Ma, Fung & Lo, Phys. Rev. A **76**, 012307 (2007) |
| [MQZL05] | Ma, Qi, Zhao and Lo |
| [MR12] | Ma & Razavi |
| [MSK19] | Matsuura, Sasaki & Koashi, Phys. Rev. A **99**, 042303 (2019), [arXiv:1812.10916](https://arxiv.org/abs/1812.10916) |
| [MU88] | Maassen & Uffink, Phys. Rev. Lett. **60**, 1103 (1988) |
| [N22] | Nadlinger et al., Nature **607**, 682 (2022) |
| [NTZLT24] | Nahar, Tupkary, Zhao, Lütkenhaus & Tan, *Postselection technique for optical Quantum Key Distribution with improved de Finetti reductions*, PRX Quantum **5**, 040315 (2024), [arXiv:2403.11851](https://arxiv.org/abs/2403.11851) |
| [P09] | Pironio, Acín, Brunner, Gisin, Massar & Scarani, New J. Phys. **11**, 045021 (2009) |
| [PCT19] | Pereira, Curty & Tamaki, npj Quantum Information **5**, 62 (2019), [arXiv:1902.02126](https://arxiv.org/abs/1902.02126) |
| [RGK05] | Renner, Gisin & Kraus, Phys. Rev. A **72**, 012332 (2005) |
| [S09] | Scarani, Bechmann-Pasquinucci, Cerf, Dušek, Lütkenhaus & Peev, Rev. Mod. Phys. **81**, 1301 (2009) |
| [SARG04] | Scarani, Acín, Ribordy & Gisin, Phys. Rev. Lett. **92**, 057901 (2004) |
| [SP26] | Seksaria & Prabhakar, *Short reach by theorem: the certifiable key rate of COW QKD* (2026) |
| [SYK14] | Sasaki, Yamamoto & Koashi, Nature **509**, 475 (2014) |
| [T14] | Tang *et al.*, Phys. Rev. Lett. **113**, 190501 (2014) |
| [T15] | Takesue, Sasaki, Tamaki & Koashi, Nature Photonics **9**, 827 (2015), [arXiv:1505.07914](https://arxiv.org/abs/1505.07914) |
| [TC21] | Trényi and Curty, NJP **23**, 093005 (2021) |
| [TCKLA14] | Tamaki, Curty, Kato, Lo & Azuma, Phys. Rev. A **90**, 052314 (2014), [arXiv:1312.3514](https://arxiv.org/abs/1312.3514) |
| [TKI03] | Tamaki, Koashi & Imoto, Phys. Rev. Lett. **90**, 167904 (2003) |
| [TL04] | Tamaki & Lütkenhaus, Phys. Rev. A **69**, 032316 (2004), [arXiv:quant-ph/0308048](https://arxiv.org/abs/quant-ph/0308048) |
| [TL06] | Tamaki & Lo, Phys. Rev. A **73**, 010302(R) (2006) |
| [TLGR12] | Tomamichel, Lim, Gisin & Renner, *Nature Communications* **3**, 634 (2012), [arXiv:1103.4130](https://arxiv.org/abs/1103.4130) |
| [TLKB09] | Tamaki, Lütkenhaus, Koashi & Batuwantudawe, Phys. Rev. A **80**, 032302 (2009) |
| [TR11] | Tomamichel & Renner, Phys. Rev. Lett. **106**, 110506 (2011), [arXiv:1009.2015](https://arxiv.org/abs/1009.2015) |
| [UvHLL21] | Upadhyaya, van Himbeeck, Lin & Lütkenhaus, PRX Quantum **2**, 020325 (2021), [arXiv:2101.05799](https://arxiv.org/abs/2101.05799) |
| [WLC18] | Winick, Lütkenhaus & Coles, Quantum **2**, 77 (2018), [arXiv:1710.05511](https://arxiv.org/abs/1710.05511) |
| [X21] | Xie, Lu, Weng, Zhang et al., *Breaking the rate-loss bound of quantum key distribution with asynchronous two-photon interference*, [arXiv:2112.11635](https://arxiv.org/abs/2112.11635) |
| [XCQL13] | Xu, Curty, Qi and Lo, New J. Phys. **15**, 113007 (2013) |
| [Y18] | Yin, Wang, Chen, Han, Wang, Guo & Han, Nature Communications **9**, 457 (2018) |
| [Z17] | Zhang, Yuan, Cao & Ma, New J. Phys. **19**, 033013 (2017), [arXiv:1505.02481](https://arxiv.org/abs/1505.02481) |
| [ZZWM22] | Zeng, Zhou, Wu & Ma, *Mode-pairing quantum key distribution*, Nat. Commun. **13**, 3903 (2022), [arXiv:2201.04300](https://arxiv.org/abs/2201.04300) |
