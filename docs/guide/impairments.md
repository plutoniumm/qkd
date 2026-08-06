# Impairments

Noise sources outside Laudenbach's Section 9, and click-protocol observables that
are **not** excess noise. Companion to [`qkd.budget`](/guide/budget).

```python
from qkd import impairments as im
```

An impairment is imperfect hardware; an attack is Eve exploiting it. Each model
here returns a `budget.Entry` a rate is computed *from*, or an observable reported
beside one; [`qkd.attacks`](#attacks) returns a `Reading`, which is neither. The
same $\tau_d$ is a rate cost as `q.DeadTime` and a blanking attack's hiding place.

Descriptors are frozen dataclasses. $\xi$ contributors go through
[`assemble_extra()`](#assemble-extra), observables through [`security()`](#security),
and [`catalogue()`](#catalogue) lists both.

## Every model names a plane {#planes}

Rows are SNU and stored at `channel_input`, as `budget` stores its own, so
`res.budget.at(plane)` moves both by one factor ([four
planes](/guide/budget#the-four-planes)). `Stated at` is the plane the source wrote
its formula in; `Referred by` is the operation that moved it, recorded in
`Entry.plane_note`.

| Model | Formula | Stated at | Referred by | Source |
| --- | --- | --- | --- | --- |
| `raman` | $\xi = 2\langle N\rangle/(\eta_D T)$ | Bob | $/T$ | Kumar 2015, Eqs. (6)–(7) |
| `rayleigh` | $\xi = 2\langle n\rangle/T$, $\langle n\rangle = \tfrac12 f_{\text{bs}}P\tau/(hf)$ | Bob | $/T$ | Mandil 2024, Eq. (A1), into Laudenbach Eqs. (9.59)–(9.60) |
| `dephasing` | [`budget.phase`](/guide/budget#the-two-forms-of-the-phase-term) at $V_\phi = 2\pi\Delta\nu\,t$ | input | not divided | Qi 2015, Eqs. (9)–(10) |
| `polarisation` | $\xi = \operatorname{Var}(\sqrt\eta)\,V_A/\langle\sqrt\eta\rangle^2$, $\eta = \cos^2\Phi$ | Bob | $/T_{\text{eff}}$ | Usenko 2012, unnumbered; overlap from Sharma 2024 |
| `imbalance` | $\xi = \tfrac12 V_A\lvert d\,e^{i\theta_m} - 1\rvert^2$ | Bob | $T\eta_e$ stripped | Wang 2025, Eq. (E1) |
| `timing` | $\xi = \operatorname{Var}(\sqrt\eta)\,V_A/\langle\sqrt\eta\rangle^2$, $\eta = e^{-\delta^2/(2w^2)}$ | Bob | $/T_{\text{eff}}$ | Usenko 2012, unnumbered, on a Gaussian matched-filter overlap |

$T_{\text{eff}} = \langle\sqrt\eta\rangle^2$ is not $T$. `polarisation` and `timing`
are fading channels: Usenko splits each into a fixed channel of transmittance
$\langle\sqrt\eta\rangle^2$ plus a Bob-plane excess noise. That noise is referred to
the input by $T_{\text{eff}}$; the fibre's $T$ cancels. Both scale as the *fourth*
power of their small parameter, since $\langle\eta\rangle$ and
$\langle\sqrt\eta\rangle^2$ agree to second order.

### A fading row is two halves {#fading}

| Half | Where it is | Reaches |
| --- | --- | --- |
| the **noise** half, $\operatorname{Var}(\sqrt\eta)\,V_A/\langle\sqrt\eta\rangle^2$ | `polarisation()`, `timing()`, and the `polarisation`/`timing` rows of `assemble_extra()` | the budget, as a `budget.Entry` |
| the **transmittance** half, $\langle\sqrt\eta\rangle^2$ | `pol_fading()[0]`, `jitter_fading()[0]`, composed by `fading_factor()` | the plane the rate is claimed at — `explain["T_claimed"]`, never `explain["T"]` |

`fading_factor(pol=…, clock=…)` multiplies the halves present and returns `1.0`
when neither is given. No other descriptor has a transmittance half, and it takes
no other keyword.

The two halves are one channel:

$$T\,f\,(V_A + \xi_{\text{fade}}) = T\,V_A\,\langle\eta\rangle,$$

so charging the noise half against an unattenuated $T$ overstates the rate. `q.Link`
composes both into [`explain["T_claimed"]`](/guide/link#t-claimed).

`dephasing` is input-referred because its source states it there.
`dephasing(v_a, linewidth, delay, xi=0.0, form=…)` and
`Dephasing(linewidth, delay, form=…)` take [`budget.phase`](/guide/budget#the-two-forms-of-the-phase-term)'s
selector and its `"estimator"` default.

Observables carry no plane:

| Observable | Returns | Observed at | Source |
| --- | --- | --- | --- |
| `backflash_leak` | $P_{\text{learn}} = P_b P_{\text{sift}}$, the fraction of the sifted key Eve reads | Eve's tap on the fibre outside Bob | Singh 2025 |
| `backflash_rate` | $P_{\text{sec}} = P_{\text{sift}}(1 - P_b - f h(e))$, clamped at zero | the sifted key | Singh 2025 |
| `extinction` | $e' = (1 - \tfrac{4}{r+3})e + \tfrac{2}{r+3}$, a QBER | Bob's sifted bits | Huang 2012, Eqs. (1) and (5) |
| `visibility` | $V = \cos\Phi$, a fringe contrast | the receiver's interference | Sharma 2024 |
| `dgd` | $\langle\Delta\tau\rangle = D_{\text{PMD}}\sqrt L$, in seconds | the fibre span | Antonelli 2024 |
| `saturate` | detected click rate after dead-time losses | Bob's detector | Krause 2025, Eq. (1); Rogers 2007 |
| `afterpulse` | $(Q_\mu, E_\mu)$, a gain and a QBER | Bob's detector | Papapanos 2020, Eqs. (4), (7) and (8) |

### The default $f$ is not the source's {#backflash-inefficiency}

`backflash_rate` charges $f = 1.16$: Lütkenhaus 2000, Table I, Brassard–Salvail
bidirectional reconciliation at $e = 0.01$ and $0.05$, and `q.IndividualAttack`'s
default. Singh 2025 write "the usual value for $f$ is 1.15" and evaluate there. $h$
cancels, so the gap is a constant **0.862 % of the error-correction term at every
QBER**; on the anchor it moves the published "approximately 10 %" reduction to
10.024 % from 10.036 %. `f=1.15`, directly or through `security(..., f=1.15)`,
reproduces the paper.

Backflash is **zero-disturbance**: no excess noise, no observable moved. It is
reported as `LinkResult.leak` and never folded into a rate.

## Sourced, derived and measured {#provenance}

| Status | Means |
| --- | --- |
| **published** | the closed form and its constants come from the cited source |
| **qkd connects** | the source states an ingredient; an unpublished elementary step joins it to the result. Flagged in the docstring of the function that performs it, in the wording `budget.dac` uses for its own substitution |
| **measured input** | the parameter is a measurement of one device, with no model predicting it from anything more primitive |
| **diagnostic** | computed and reported, never converted into a $\xi$, because no verified conversion exists |

### What each model owes its source {#prior-art}

*Published* is what the primary source states, to the equation. *qkd's own step* is
everything joining it to the returned number, including constants fitted to close an
anchor.

| Mechanism | Status | Published | qkd's own step |
| --- | --- | --- | --- |
| **Raman** (`raman`) | published, on a measured input | Kumar Eqs. (6) and (7): the SASRS photon number for both launch geometries, and its conversion to a channel-input $\xi$. Behind it, Laudenbach Eq. (9.59) $\xi = 2\langle n\rangle$ with $\langle\hat n\rangle = \tfrac12 V$, and Eq. (9.60) converting optical power to photon number. | The parameterisation: Kumar Eq. (6) takes launch power, fibre and geometry. Laudenbach Eq. (9.63) takes a measured density and ships as [`budget.raman`](/guide/budget#the-raman-row), a factor of two apart and consumed by no `q.Link`. Anchored on the paper's figures, $1.274\times10^{-3}$ forward and $1.574\times10^{-3}$ backward against their $\sim1.3$ and $\sim1.6\times10^{-3}N_0$. Those quoted numbers are $2\langle N\rangle$ at Bob, not the $\xi_{\text{in}}$ of their Eq. (7), which pins the convention. $\beta$ is a spectrum, so `Coexistence.beta` takes the measurement (gap 1). |
| **Rayleigh** (`rayleigh`) | qkd connects | Mandil Eq. (A1), the fibre's impulse response $P_s(t) = P_0\tau\eta e^{-\alpha v_g t}$; the OTDR-measured $\eta = 8.0 \pm 0.1\ \mathrm{s^{-1}}$ for SMF-28 and $6.54 \pm 0.08$ for SMF-28 ULL; and the random scattered polarisation, which supplies the factor $\tfrac12$. | The round-trip integral of that response for a continuous-wave launch, $f_{\text{bs}}(L) = \eta\left(1 - e^{-2\alpha L}\right)/(\alpha v_g)$, saturating at $\eta/(\alpha v_g)$. Flagged in `rayleigh_fraction`'s docstring. Single backscatter only (gap 7). Subacius 2005, the canonical QKD statement, is paywalled and was not mined; nothing here rests on it. |
| **Dephasing** (`dephasing`) | published | Qi Eqs. (9)–(10): $\langle(\Delta\theta)^2\rangle = 2t/\tau_c$ with $\tau_c \simeq 1/(\pi\Delta\nu)$, hence $V_\phi = 2\pi\Delta\nu\,t$, and $\Delta\nu = \Delta\nu_A + \Delta\nu_B$ for two independent lasers. The map to $\xi$ is [either published phase form](/guide/budget#the-per-source-table). | Composition of two published results. The default estimator form is prior art, not a house derivation. |
| **Polarisation** (`polarisation`, `visibility`, `dgd`) | published; `dgd` diagnostic | Usenko's fading identity: a channel of fluctuating transmittance *is* a fixed channel of transmittance $\langle\sqrt\eta\rangle^2$ plus a Bob-plane $\operatorname{Var}(\sqrt\eta)(V - 1)$. The mode overlap is Sharma's, the mismatch entering the relay output mean photon number through $\cos\delta\cos\Phi$. $\langle\Delta\tau\rangle = D_{\text{PMD}}\sqrt L$ is standard; Antonelli applies it to a quantum channel. | The Gaussian average of the overlap over a drifting angle: $\langle\sqrt\eta\rangle = \langle\cos\Phi\rangle = e^{-\sigma^2/2}$ and $\langle\eta\rangle = \tfrac12(1 + e^{-2\sigma^2})$, so $\operatorname{Var}(\sqrt\eta) = \tfrac12\left(1 - e^{-\sigma^2}\right)^2$ and $\xi_{\text{pol}} \to V_A\sigma^4/2$ as $\sigma \to 0$. Sharma's $\Phi \le 11^\circ$ tolerance stands. **Sharma's $V < 0.37$ is not qkd's to use**: it is a *polarisation*-MDI threshold imported from Yuan 2014 and Xu 2013 and drawn as a dashed reference line; nothing in qkd rests on it. The PMD branch stops at the DGD (gap 2). |
| **Modulator** (`imbalance`, `extinction`) | published | Wang Eq. (E1), $\operatorname{Var}[\varepsilon_{q1}]_B = T\eta_e\frac{V_A}{2}\left[d^2\sin^2\theta_m + (d\cos\theta_m - 1)^2\right]$, with their measured floor $d \ge 0.9937$ and noise under 0.02 SNU for $\lvert\theta_m\rvert \le 5^\circ$. Huang Eqs. (1) and (5) for the extinction QBER. The $4/(r+3)$ admixture is *identity*, invariant under every unitary, so Eve learns nothing from it and privacy amplification shrinks only the remaining $1 - 4/(r+3)$. | Stripping $T\eta_e$ to reach the input plane, and reading the bracket as $\lvert d\,e^{i\theta_m} - 1\rvert^2$, the squared miss of the complex modulation gain. Amplitude imbalance and quadrature bias drift are therefore one term. The anchor fits $V_A = 5.25$ SNU, which the source does not tabulate. `extinction` holds for a four-modulator click transmitter and returns a QBER; it reaches [`security()`](#security), never `assemble_extra()` (gaps 3 and 4). |
| **Timing** (`timing`) | qkd connects | Usenko's fading identity, unchanged. | The matched-filter overlap $\rho(\delta) = e^{-\delta^2/(4w^2)}$ of a Gaussian envelope sampled at offset $\delta$, averaged over a jittering instant with $\langle e^{-a\delta^2}\rangle = (1 + 2a\sigma_j^2)^{-1/2}$. Both flagged in `jitter_fading`'s docstring. Only $u = (\sigma_j/w)^2$ survives, so tolerable absolute jitter scales with the symbol period. |
| **Backflash** (`backflash_leak`, `backflash_rate`) | published, on a measured input | Singh's $P_b = n_{\text{Eve}}/n_{\text{sift}}$, $P_{\text{learn}} = P_b P_{\text{sift}}$ and $P_{\text{sec}} = P_{\text{sift}}(1 - P_b - f\,h(e))$, with 1598 backflash events against 18000 sifted counts on a 74 % SNSPD, arriving within $< 5$ ns of the click. | $P_b$ belongs to one detector at one excess bias and gate width (gap 5), so `Backflash(prob)` takes the measurement. The anchor QBER is **fitted** at $e = 1.29\%$, a COW operating point the source does not state; with it and $f = 1.16$ the module reproduces their 10.0 % key-rate reduction, and the test message names the fit. Meda's $P_L = 9.8\%$ and $6\%$ are a plausibility range, never a default (gap 6). Li's ceiling — Eve extracts usable information from at most 95.7 % of backflash photons, the broadband spectrum degrading the extinction she can measure — is recorded, not implemented. |
| **Dead time** (`saturate`, `afterpulse`) | published | Non-paralysable $R = R^*/(1 + R^*\tau_d)$, Krause Eq. (1), and paralysable $R = R^*e^{-R^*\tau_d}$, both standard. Rogers settle which applies to QKD: one SPAD is non-paralysable, a *pair* serving one basis is paralysable. Afterpulsing is Papapanos Eqs. (4), (7) and (8) over the Ma–Lo decoy model. | At $p_{AP} = 0$ the expressions collapse onto the plain Ma–Lo gain and error rate, which the tests check. `q.ClickDetector(dead_time=…, afterpulse=…)` reaches the simulated click train, but no exam reproduces Rogers' conclusion, the transmission rate above which the *sifted* rate falls. Built, not anchored. |

`timing` is the CV **sampling** branch. Click-path detector jitter is a different
model, split into window loss and bin leak — [Protocols](/guide/protocols#jitter),
fitted in [Validation](/guide/validation#tier-b-qubit). A pure Gaussian fitted to
Diamanti et al.'s 100 ps window statement (192.13 ps FWHM) puts the neighbouring
slot 12.3 standard deviations away, leak $5.4\times10^{-31}$.

### The gaps {#gaps}

Each open gap is open because the published constant or closed form does not exist.

| # | Gap | Status, and why |
| --- | --- | --- |
| 1 | Spacing dependence of the Raman coefficient $\beta$ | **Open.** $\beta$ *is* the spacing dependence, a measured spectrum: Kumar report $1.5$ to $3.1\times10^{-9}\ \mathrm{km^{-1}nm^{-1}}$ across the C band with no fitted form, and silica's Raman gain has a broad 13 THz peak plus structure no two-parameter model captures. No published $\beta(\Delta\lambda)$ accurate enough to ship was found; **unverified**. |
| 2 | PMD to excess noise | **Open.** Converting a DGD into $\xi$ needs a depolarisation model for a pulse spanning both principal states; every treatment found is numerical or specific to entangled-pair spectra. To first order PMD on a narrowband CV signal is a *unitary* rotation, not depolarisation, so the drift model carries the load — here and in the [reference-frame tracking model](/guide/protocols#tracking). |
| 3 | Extinction ratio to a CV excess noise | **Open, narrowed.** A residual carrier of known amplitude is a *deterministic displacement*, not noise unless it drifts, so the CV parameter is a drifting displacement variance, a different measurement from an extinction ratio. On the click side COW carries [finite modulator extinction](/guide/protocols#extinction) as `q.IntensityKeying(extinction=…)`, and symbol-level BB84 derives its QBER from a [simulated pulse train](/guide/protocols#sampled). |
| 4 | Bias drift beyond the quadrature bias point | **Open; reduces to gap 3.** Wang's $\theta_m$ covers the orthogonality bias point. A DC null-bias drift of the child modulators appears as carrier leakage, gap 3's quantity. |
| 5 | $P_b$ from hardware | **Open.** Every source treats $P_b$ as measured for one detector at one excess bias and gate width. Meda give the spectral and temporal shape — broadband across 1530–1600 nm, with a temporal signature that fingerprints the detector model — but no model from bias voltage or avalanche charge. Whether the avalanche-charge scaling implied by SPAD physics is published as a usable formula is **unverified**. |
| 6 | The normalisation of Meda's $P_L$ | **Open.** Their 9.8 % and 6 % assume $\eta_{ch}\eta_{det} = 0.05$ on Eve's side. Which corrections sit in $P_L$ rather than the raw emission probability could not be reconstructed from the post-print. |
| 7 | Double backscattering on a one-way link | **Open.** On a one-way link single backscatter travels *away* from Bob; only the second-order forward term reaches him. It scales as $f_{\text{bs}}^2 \approx 10^{-6}$. No closed form was found; the caveat is in the module docstring. |
| 8 | The two connecting integrals | **Not a gap: a disclosure.** The continuous-wave integration of Eq. (A1) and the Gaussian matched-filter overlap are qkd's, flagged in code with the wording `budget.dac` uses. The overlap covers the CV sampling branch only; the click path's tailed arrival-time law and its window-loss/bin-leak split are also qkd's and also flagged. |
| 9 | Composition of `budget.phase` and `impairments.dephasing` | **Closed.** One mechanism at two idealisations: a run with both a DSP chain and a `Dephasing` charged the laser twice. See [the double-count guard](#in-a-link). |

| House rule | |
| --- | --- |
| No verified conversion | report the ingredient and stop — the **diagnostic** status: `dgd` returns a differential group delay in seconds and no $\xi$ |
| Constant exists only as a measurement | the descriptor takes the measurement ($\beta$, $P_b$) |
| A number fitted to close an anchor | the test message that uses it names the fit |

## Descriptors {#descriptors}

Frozen dataclasses of hardware numbers, **not validated at construction**, unlike
the [`q.` components](/guide/link#component-reference). `im.Backflash(prob=2.0)`
constructs, and raises only when a leak is computed from it. `Plane` is the plane
the source formula was stated in ([planes](#planes)).

### Fibre

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `Coexistence.channels` | — | Bob | **required** | Co-propagating classical channels. Launch powers add linearly in the Raman source term, so $\xi$ is linear in the count. |
| `Coexistence.launch` | dBm | Bob | **required** | Per-channel launch power. Kumar found a 25 km CV-QKD link tolerates up to $11.5$ dBm forward and $9.7$ dBm backward. |
| `Coexistence.beta` | 1/(km nm) | Bob | `3.0e-9` | Raman coefficient, measured: Kumar report $1.5$ to $3.1 \times 10^{-9}$ across the C band for a 1531.12 nm quantum channel. The default is their worst case. |
| `Coexistence.wavelength` | m | Bob | `1531.12e-9` | Quantum-channel wavelength $\lambda$, entering the source term as $\lambda^3$. The default is Kumar's measurement channel and pairs with the default $\beta$. |
| `Coexistence.demux` | — | Bob | `1.0` | Kumar Eq. (6)'s $\eta_D$, "transmittance of DEMUX (Add-Drop Module) place at Bob side" — **not** a detector efficiency, which is that paper's $\eta_B$ at Eq. (5). It **cancels** through Eq. (7) and moves no input-referred $\xi$. A filter *narrower than the quantum channel* would not cancel; no rate models one, so `q.Link` **refuses a declared value by name**. Carried so `raman_photons()` reproduces the source's Bob-plane figures. |
| `Coexistence.backward` | — | — | `False` | Counter-propagating classical channels: backward geometry $\left(1 - e^{-2\alpha L}\right)/(2\alpha)$ in place of forward $L e^{-\alpha L}$. |
| `Backscatter.power` | W | Bob | **required** | Power of the counter-propagating tone scattering back into Bob: a two-way architecture's outbound pulse train or a classical channel. |
| `Backscatter.coeff` | 1/s | Bob | `8.0` | Returned power at $t = 0$ over forward pulse energy, measured by OTDR: $8.0 \pm 0.1$ for SMF-28, $6.54 \pm 0.08$ for SMF-28 ULL. |
| `Backscatter.index` | — | Bob | `1.468` | Fibre group index $n$; $v_g = c/n$. |
| `Backscatter.wavelength` | m | Bob | `1550.12e-9` | The launch wavelength. Rayleigh scattering is elastic, so the return is in band and no spectral filter applies. |
| `Polarisation.drift` | rad | Bob | **required** | RMS mismatch angle $\sigma$ between signal and receiver reference. A static mismatch is pure loss; the drift produces excess noise. |
| `Polarisation.dispersion` | s/$\sqrt{\text{km}}$ | — | `0.0` | PMD coefficient $D_{\text{PMD}}$, reaching `dgd()` alone. `q.Link` **refuses a declared value by name**: no verified closed form converts a DGD into an excess noise. |

### Transmitter

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `Dephasing.linewidth` | Hz | input | **required** | The **sum** of both lasers' Lorentzian full widths; the beat phase diffuses at the sum rate. |
| `Dephasing.delay` | s | input | **required** | Time between the phase reference and the symbol it corrects. What diffuses over it is uncorrectable. |
| `Dephasing.form` | — | input | `"estimator"` | `budget.phase`'s [`form=`](/guide/budget#the-two-forms-of-the-phase-term). Also selects the row's `plane_note`, naming Kish or Marie & Alléaume. |
| `Modulator.ratio` | — | Bob | `1.0` | I/Q amplitude imbalance $d = b/a$; $1.0$ balanced. |
| `Modulator.angle` | rad | Bob | `0.0` | Quadrature bias-point offset $\theta_m$, $0.0$ orthogonal, and the axis that bias point drifts along. In $\lvert d e^{i\theta_m} - 1\rvert^2$ amplitude imbalance and *quadrature* bias drift are one term. A DC null-bias drift of the child modulators is not covered. |
| `Modulator.extinction` | — | — | `None` | Linear extinction ratio $r = I_{\text{on}}/I_{\text{off}}$, a QBER. A click `q.Link` charges it and [`security()`](#security) reports it; a quadrature link **refuses a declared value by name**, and `assemble_extra()` never reads it (gap 3). `None` omits the observable. Practical intensity modulators reach 20 to 40 dB; 20 dB alone costs 1.9 % QBER. |
| `Timing.jitter` | s | Bob | **required** | RMS sampling-instant error: clock jitter plus residual drift. |
| `Timing.width` | s | Bob | **required** | Field envelope parameter $w$ of $e^{-t^2/(2w^2)}$. Only $u = (\text{jitter}/w)^2$ reaches the answer. |

### Detector

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `Backflash.prob` | — | — | **required** | $P_b = n_{\text{Eve}}/n_{\text{sift}}$, the chance a sifted detection at Bob comes with a backflash photon reaching Eve. A leakage probability with no plane, checked against $[0, 1]$ by its consumers. |
| `DeadTime.dead` | s | — | **required** | Interval $\tau_d$ after an avalanche during which the detector is held below breakdown. Tens of nanoseconds for a gated InGaAs device, of order 100 ns for a free-running SPAD. |
| `DeadTime.afterpulse` | — | — | `0.0` | Probability $p_{AP}$ of a spurious count given a previous detection. Uncorrelated with its trigger, so it errs half the time, like a dark count. Of order 1 to 5 % for InGaAs/InP SPADs, essentially zero for superconducting nanowires. |
| `DeadTime.paralysable` | — | — | `False` | Whether an arrival inside the dead time extends it. `q.Link`'s decoy path **refuses a declared value by name**, taking a `q.DeadTime` only at `dead = 0`. One SPAD is non-paralysable, the default; a **pair** serving one basis is paralysable, since a click on either disables that basis's sifting. The non-paralysable rate saturates at $1/\tau_d$; the paralysable one peaks at $R^* = 1/\tau_d$ and then falls. |

```python
im.saturate(1e6, 1e-7)                    # 909090.9  non-paralysable
im.saturate(1e6, 1e-7, paralysable=True)  # 904837.4  same arrivals, fewer counts
im.afterpulse(0.5, 0.2, 1e-6, 0.02)       # (0.0970669, 0.0098091)  gain, QBER
```

At $R^*\tau_d = 0.1$ the branches agree to half a percent; above it they diverge
without limit.

## `assemble_extra()` {#assemble-extra}

Keyword-only. As in [`budget.assemble`](/guide/budget#assemble), a descriptor left
`None` omits its row; one present but perfect, such as `Modulator()` at ratio 1 and
angle 0, contributes `0.0`. Keywords name the descriptor's **role**, not its class.
Returns a tuple of `budget.Entry`, which concatenates onto a `Budget`'s entries.

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `v_a` | SNU | input | **required** | $V_A > 0$, else `ValueError`. Read by every model except `raman` and `rayleigh`, which are absolute photon fluxes. |
| `t` | — | — | **required** | $T \in (0, 1]$, else `ValueError`. Divides the two Bob-plane fluxes and nothing else. On a link with a [loss chain](/guide/budget#loss-sites) pass $T_{\text{launch}}T_{\text{span}}$, as `q.Link` does: `raman` and `rayleigh` are born inside the span. |
| `length` | km | — | `None` | Fibre length. Required by `coexist` and `probe`, else `ValueError`. |
| `alpha` | dB/km | — | `0.2` | Fibre attenuation, converted to nepers inside both scattering integrals. |
| `symbol` | s | — | `None` | Symbol period $\tau$, converting returned power into photons per detection mode. Required by `probe`. |
| `coexist` | — | Bob | `None` | A `Coexistence`. Emits `raman`, divided by $T$. |
| `probe` | — | Bob | `None` | A `Backscatter`. Emits `rayleigh`, divided by $T$. |
| `dephase` | — | input | `None` | A `Dephasing`. Emits `dephasing`, not divided. Its `form` picks the expression and the `plane_note`. |
| `pol` | — | Bob | `None` | A `Polarisation`. Emits `polarisation`, the **noise half alone**, referred by $T_{\text{eff}}$ ([the other half](#fading)). Reads `drift`; `dispersion` reaches `security()`. |
| `modulator` | — | Bob | `None` | A `Modulator`. Emits `imbalance`, input-referred once $T\eta_e$ is stripped. Reads `ratio` and `angle`; `extinction` reaches `security()`. |
| `clock` | — | Bob | `None` | A `Timing`. Emits `timing`, the **noise half alone**, referred by $T_{\text{eff}}$. |

```python
from qkd import budget, impairments as im

base = budget.assemble(v_a=5.0, t=10**-0.5, v_err=2e-3)
extra = im.assemble_extra(
    v_a=5.0, t=10**-0.5, length=25.0,
    dephase=im.Dephasing(linewidth=10e3, delay=1e-6),
    modulator=im.Modulator(ratio=0.98, angle=0.01),
    pol=im.Polarisation(drift=0.02),
)

full = budget.Budget(entries=base.entries + extra, T=10**-0.5)
full.total    # 0.335494
```

## `security()` {#security}

Everything that is not an excess noise, as a plain `{name: value}` dict on a
separate return path. A `None` descriptor is omitted. Operating points arrive
through `**kw`; a model missing its operating point is skipped, not defaulted.

| Parameter | Kind | Default | Description |
| --- | --- | --- | --- |
| `backflash` | descriptor | `None` | A `Backflash`. Emits `backflash_leak` and `backflash_rate`. |
| `dead` | descriptor | `None` | A `DeadTime`. Emits `click_rate` if `rate` is given, and `gain` with `qber` if `mu` is given. |
| `modulator` | descriptor | `None` | A `Modulator`. Emits `extinction` when its `extinction` field is not `None`. |
| `pol` | descriptor | `None` | A `Polarisation`. Emits `visibility`, and `dgd` when `dispersion` is positive and `length` is given. `visibility` is $\cos\sigma$ at the **rms** drift angle, not averaged over the drift distribution: the contrast at a typical mismatch, not the mean contrast. |
| `sift` | operating point | `1.0` | Sifted fraction $P_{\text{sift}}$, scaling both backflash outputs. |
| `qber` | operating point | `0.0` | Observed error rate, entering `backflash_rate` through $h(e)$ and `extinction` as the error the leak adds to. |
| `f` | operating point | `1.16` | Error-correction inefficiency charged by `backflash_rate`. **Not** the source's 1.15 — see [above](#backflash-inefficiency). |
| `rate` | operating point | omitted | Incident click rate $R^*$ in Hz, before dead-time losses. |
| `mu` | operating point | omitted | Mean photon number per pulse at the detector. |
| `eta` | operating point | `1.0` | Detection efficiency, used only by the afterpulse model. |
| `dark` | operating point | `0.0` | Dark-count probability per gate $p_{DC}$. |
| `edet` | operating point | `0.0` | Static detector misalignment error $e_{\text{det}}$, distinct from the afterpulse-induced error the model adds. |
| `length` | operating point | omitted | Fibre length in km, needed by `dgd` alone. |

Possible keys: `backflash_leak`, `backflash_rate`, `click_rate`, `gain`, `qber`,
`extinction`, `visibility`, `dgd`. An absent key means its descriptor or operating
point was not supplied, never that the quantity is zero.

## `catalogue()` {#catalogue}

`catalogue()` lists every model as `(name, callable, parameters)`: the $\xi$
contributors, then the observables, in the tables' order. A model not listed is
unreachable. Each $\xi$ function returns an **input-referred** value in SNU.

| Function | Arguments | Returns |
| --- | --- | --- |
| `raman_photons(power, length, ...)` | $P$ in W, $L$ in km | $\langle N\rangle$ per detection mode, at Bob |
| `raman(power, length, t, ...)` | as above plus $T$ | $\xi$ |
| `rayleigh_fraction(length, ...)` | $L$ in km | returned power fraction $f_{\text{bs}}$, dimensionless |
| `rayleigh_photons(power, length, symbol, ...)` | $P$ in W, $L$ in km, $\tau$ in s | $\langle n\rangle$ per detection mode, at Bob |
| `rayleigh(power, length, symbol, t, ...)` | as above plus $T$ | $\xi$ |
| `phase_variance(linewidth, delay)` | $\Delta\nu$ in Hz, $t$ in s | $V_\phi = 2\pi\Delta\nu t$ in rad² |
| `coherence(linewidth, delay)` | as above | $e^{-V_\phi/2}$, the beat note's coherence factor |
| `dephasing(v_a, linewidth, delay)` | $V_A$ in SNU, $\Delta\nu$ in Hz, $t$ in s | $\xi$ |
| `pol_fading(drift)` | $\sigma$ in rad | $(\langle\sqrt\eta\rangle^2, \operatorname{Var}\sqrt\eta)$ |
| `polarisation(v_a, drift)` | $V_A$ in SNU, $\sigma$ in rad | $\xi$ |
| `imbalance(v_a, ratio, angle)` | $V_A$ in SNU, $d$, $\theta_m$ in rad | $\xi$ |
| `jitter_fading(jitter, width)` | both in s | $(\langle\sqrt\eta\rangle^2, \operatorname{Var}\sqrt\eta)$ |
| `fading_factor(pol=…, clock=…)` | the two fading descriptors | the product of their $\langle\sqrt\eta\rangle^2$, `1.0` where neither is given |
| `timing(v_a, jitter, width)` | $V_A$ in SNU, both in s | $\xi$ |
| `backflash_leak(prob, sift)` | $P_b$, $P_{\text{sift}}$ | leaked fraction of the sifted key |
| `backflash_rate(sift, prob, qber, f)` | $P_{\text{sift}}$, $P_b$, $e$, $f$ | secure fraction, clamped at zero |
| `visibility(angle)` | $\Phi$ in rad | fringe contrast $\cos\Phi$ |
| `dgd(dispersion, length)` | $D_{\text{PMD}}$ in s/$\sqrt{\text{km}}$, $L$ in km | mean differential group delay in s |
| `extinction(ratio, qber)` | $r$, $e$ | QBER including the leaked admixture |
| `saturate(rate, dead, paralysable)` | $R^*$ in Hz, $\tau_d$ in s | detected rate in Hz |
| `afterpulse(mu, eta, dark, prob, edet)` | $\mu$, $\eta$, $p_{DC}$, $p_{AP}$, $e_{\text{det}}$ | $(Q_\mu, E_\mu)$ |

```python
im.phase_variance(20e3, 10e-9)   # 0.00125664  rad^2 over one 100 MBaud symbol
im.dephasing(5.0, 20e3, 10e-9)   # 0.00628121  SNU, input-referred
im.polarisation(5.0, 0.02)       # 4.0e-07     20 mrad rms drift
im.rayleigh_fraction(25.0)       # 0.00076558  == -31.2 dB over 25 km
im.dgd(0.1e-12, 100.0)           # 1e-12       1 ps of DGD over 100 km
```

$6.3\times10^{-3}$ SNU from two 10 kHz lasers over *one* 100 MBaud symbol period is
the order of the [measured envelope](/guide/budget#the-measured-envelope).
Linewidth and delay enter $V_\phi$ symmetrically.

`fading_factor` returns a transmittance, not a $\xi$, and is **not** in
`catalogue()`.

`backflash_leak`, `backflash_rate` and `afterpulse` require a probability in
$[0, 1]$, `extinction` a positive ratio, `saturate` a non-negative dead time. The
rest accept any input: an unphysical drift angle or a negative jitter returns a
number, not an exception.

## Reaching a `Link` {#in-a-link}

`q.Link(..., impairments=[...])` matches descriptors onto `assemble_extra`'s
keywords **by type**. An unrecognised object raises, as does a repeated type, e.g.
two `Dephasing` entries describing one laser twice.

```python
import qkd as q

res = q.Link(
    modulation=q.GaussianModulation(v_a=5.0),
    channel=q.Fiber(length=25.0, alpha=0.2),
    bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True)),
    impairments=[q.Polarisation(drift=0.02), q.Modulator(ratio=0.98)],
).run()

res.explain["xi_polarisation"]   # {'value': 4.0e-07,   'label': 'derived'}
res.explain["xi_imbalance"]      # {'value': 0.001,     'label': 'derived'}
res.explain["xi_total"]          # {'value': 0.0010004, 'label': 'derived'}
```

`xi_total` is the sum that reached the covariance. The [pinning
labels](/guide/link#pinning) apply unchanged; an impairment row is never a default.

### A descriptor moves the rate on both paths {#both-paths}

The example above is the closed form. On the [sampled
path](/guide/link#dsp-in-the-loop) no descriptor is simulated into the symbols; its
rows are **added to the estimate** ([Measure once, claim many
times](/guide/link#measure-once-claim-many-times)).

```python
def link(impairments=()):
    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        alice=q.Alice(laser=q.Laser(linewidth=10e3),
                      pilots=q.Pilots(power_db=12.0, freq=180e6),
                      symbol_rate=100e6),
        channel=q.Fiber(length=25.0, alpha=0.2),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True),
                  lo=q.LocalLO(linewidth=10e3)),
        dsp=q.DSP(phase=q.PilotPhase(), block=32),
        impairments=impairments,
    )

bare = link().run(symbols=1_000_000, seed=1)
res = link([q.Modulator(ratio=1.02, angle=0.01)]).run(symbols=1_000_000, seed=1)

res.explain["xi_imbalance"]["value"]   # 0.0012549978750070725, the descriptor's own
                                       #   row — a closed form in (V_A, d, theta),
                                       #   with no seed and no estimate in it
res.explain["xi_claimed"]              # the clamped estimate PLUS that row
res.explain["xi_total"]                # the assembled budget, beside it

res.explain["xi_claimed"]["value"] > res.est.xi     # True
res.key_rate < bare.key_rate                        # True
```

**Read the ordering, not the magnitude.** $\hat\xi$ scatters over seeds by more
than a small descriptor contributes: at $10^5$ symbols and 10 kHz linewidths, by
several times the charged row. The seed-independent claim is `xi_imbalance`.

The split holds in both directions, since threshold detection has no quadrature
variance for an excess noise to inflate:

```python
q.Link(..., modulation=q.GaussianModulation(), impairments=[q.Backflash(prob=0.09)])
# ValueError: no Gaussian-modulation term consumes Backflash

q.Link(..., modulation=q.DifferentialPhase(), impairments=[q.Dephasing(...)])
# ValueError: no click bound consumes Dephasing: it yields a Gaussian excess
#             noise, which threshold detection never sees
```

`budget.phase(v_err)` and `impairments.dephasing(linewidth, delay)` are **one
mechanism at two idealisations**: the residual an estimator leaves at finite pilot
power, and the diffusion no estimator removes. Where a [DSP
chain](/guide/link#dsp-in-the-loop) produced a residual, `Link` adds **only that
residual** and checks it against the diffusion floor $V_\phi = 2\pi\Delta\nu\,t$:

```python
q.Link(..., dsp=q.DSP(phase=q.PilotPhase(v_err=4e-4)),
       impairments=[q.Dephasing(linewidth=20e3, delay=1e-8)]).run()
# ValueError: v_err 0.0004 rad^2 sits below the 0.001257 rad^2 the declared
#             linewidth diffuses over that delay: no estimator removes a phase
#             that has already diffused
```

The floor is returned as `explain["v_floor"]`, not summed. No `xi_dephasing` row is
emitted on that path, and the check asserts only that the two are ordered.

Detector memory is described **either** on `q.ClickDetector(dead_time=…,
afterpulse=…)`, consumed slot by slot by the simulated click train, **or** through a
`DeadTime` descriptor on a closed-form path. Never both; `Link` names the conflict.

Reports: [Impairments · Optical](/tests/impairments_optical) ·
[Impairments · Detector](/tests/impairments_detector).

## `qkd.attacks` {#attacks}

Published attacks on the same hardware, in a separate module on a separate return
path.

```python
from qkd import attacks as at
```

| Attack | Descriptor | What Eve exploits | Family |
| --- | --- | --- | --- |
| `saturation` | `at.Saturation(alpha, delta, gain)` | a homodyne's finite linear range, and that no CV-QKD estimator monitors the quadrature **mean** | quadrature |
| `calibration` | `at.Calibration(ratio, resend)` | that every CV-QKD number is quoted in shot-noise units Bob calibrated himself | quadrature |
| `blinding` | `at.Blinding(always, never, passive)` | that a bright-light-held APD answers classical power and nothing else | threshold |
| `timeshift` | `at.Mismatch(hi, lo)` | that Bob's two detectors are not equally efficient at the same instant | threshold |
| `blanking` | `at.Blanking(blind, signal, gap)` | the dead time `q.DeadTime` describes — spent on security here, at no cost in rate | threshold |
| `oscillator` | `at.Oscillator(monitored, sampled, slope, floor, resend)` | that a **transmitted** oscillator sets the shot-noise unit Bob divides by, and that his monitor reads the pulse's peak rather than his sampling instant | quadrature |
| `injection` | `at.Injection(reflect, isolator, stages, atten, bandpass, injected, clock)` | that Alice's phase modulator back-reflects the setting she just applied — a Trojan-horse probe crosses it twice and disturbs no statistic Alice and Bob can form | source |

Descriptors are frozen dataclasses validated by their consumers. `at.catalogue()`
lists the attacks as `(name, callable, parameters)`. `at.assess(sat=…, calib=…,
blind=…, mismatch=…, blank=…, lo=…, probe=…, **operating_point)` returns
`{name: Reading}`, `None` omitting a row. `probe=` **requires** `y1` in the operating
point and raises without it: the phase-error price is written on the single-photon
yield.

`oscillator` is `calibration`'s arithmetic with the ratio **derived, not dialled**.
[`q.TransmittedLO`](/guide/link)'s `tlo_calib` turns the monitored power, the
sampled power and Bob's fitted line $Z(P) = \text{slope}\,P + \text{floor}$ into the
assumed-over-actual ratio; the two attacks agree wherever the ratios do. It is a
separate descriptor because a transmitted oscillator is a security model, not a
hardware description.

`injection` is the **source-side** attack, the only one not touching Bob.
`probe_isolation` reads a chain of positive dB suppressions — the insertion losses
`q.Connector`, `q.Splice` and `q.Coupling` carry — counting the filter and the
attenuator **twice**, since the probe crosses both ways, and the reflection once.
`probe_photons` converts that to photons per modulator setting, `probe_delta` to a
coin imbalance, `probe_phase` to a phase error rate. The dB hold at **one
wavelength**; qkd's component losses carry none, and Eve chooses the wavelength.
`probe_rate` raises rather than returning a number.

### A `Reading` is two books, and neither is a key rate {#reading}

```python
r = at.sat_break(v_a=18.5, t=0.302, eta=0.606, v_el=0.041,
                 alpha=20.0, delta=18.867, xi=0.005)

r.observed     # {'t': 0.146963, 'xi': 0.0049667}
r.eve          # {'xi': 2.005, 'resend': 1.0}
```

`observed` is what Alice and Bob's monitoring reports during the attack; `eve` is
what the eavesdropper does. Here the reported $\xi$ is the link's own $0.005$ while
Eve's full intercept-resend adds $2 + \xi_{\text{sys}} = 2.005$ SNU: entanglement
breaking at every distance, and the noise both source papers quote as undistillable
by any CV-QKD link.

These names raise instead of returning a rate:

```python
r.key_rate
# AttributeError: a Reading has no key rate: observed and eve are not the same
#   quantity, and the first does not bound the second -- under every attack
#   modelled here the observables stay at values an unattacked link would
#   produce. Read Reading.observed and Reading.eve separately, and never
#   subtract them
```

`key_rate`, `rate`, `key`, `secure`, `safe` and `margin` raise that message; any
other missing attribute raises the ordinary `AttributeError`. The gap between the
books is not a security margin ([Security](/guide/security#attacks)). `r.table()`
prints both books side by side with the sentence naming why neither bounds the
other.

### None of these $\xi$ values may enter a budget {#not-a-budget}

`at.sat_estimate` and `at.calib_xi` return SNU numbers spelled $\xi$ that must not
reach `budget.Entry`: they are estimator artefacts, not light in a fibre, and belong
to no plane.

```python
at.sat_estimate(18.5, 0.302, 0.606, 0.041, alpha=20.0, delta=0.0,  xi=0.005)
# (0.302000, 2.005000)     no displacement: the intercept-resend is in plain sight

at.sat_estimate(18.5, 0.302, 0.606, 0.041, alpha=20.0, delta=18.867, xi=0.005)
# (0.146963, 0.004967)     displaced into the clip: xi reads honest, T-hat does not
```

The calibration attack acts through the unit instead of the estimator. The
overestimate erasing a given excess noise has a closed form:

```python
at.calib_ratio(0.01, 0.5, 0.6)          # 1.003    the N'_0/N_0 that reports xi = 0
at.calib_xi(0.01, 1.003, 0.5, 0.6)      # 3.6e-16  and it does
```

`calib_xi` returns negatives raw. A negative apparent excess noise is the attack's
signature; clamping would erase it.

### The threshold-detector three {#attacks-threshold}

```python
at.shift_ratio(hi=0.2, lo=0.02)   # 0.100000  the efficiency mismatch r = min/max
at.shift_qber(0.1)                # 0.153846  what a faked-state attack induces
at.shift_leak(0.1)                # 0.560503  Eve's information, per sifted bit
at.shift_bound(0.1)               # 0.439497  the ceiling any rate must sit under
```

The time-shift attack induces **no** error at any mismatch: Eve reroutes the pulse
and never measures it. `shift_break` therefore reports the QBER unmoved beside the
naive rate a mismatch-blind proof would claim. `at.mismatch_rate(hi, lo, e_bit,
e_phase)` is what Alice and Bob are entitled to once the mismatch is accounted for.
Its `e_phase` has **no default**: equating it to `e_bit` assumes a channel symmetric
between the bases, which an efficiency mismatch is not.

Two of the detector attacks refuse rather than approximate. Blinding refuses when
the trigger powers do not separate the detectors:

```python
at.blind_break(gain=0.01, qber=0.01, always=(2.5e-3,), never=(1.0e-3,))
# ValueError: max(P_always)/min(P_never) = 2.5 is not below 2, so Lydersen
#   arXiv:1008.4593 Eq. (1) fails: a pulse able to fire the intended detector also
#   fires at least one other, and the attack leaves errors. This module has no model
#   for that partial control and will not approximate one
```

Blanking refuses a pulse timed outside the window it must hide in: inside Bob's
acceptance window it is counted, older than the dead time it is harmless.

```python
at.blank_hidden(gap=5e-9,   window=1e-9, dead=50e-9)   # True
at.blank_hidden(gap=100e-9, window=1e-9, dead=50e-9)
# ValueError: a blinding pulse 1e-07 s ahead of the slot is older than the 5e-08 s
#   dead time it has to survive: the detectors have recovered by the time Alice's
#   photon arrives and nothing is blinded
```

`blank_leak(16.52, 0.1)` returns $0.930949$ bit per sifted bit at the source's
blinding intensity. Two readings of the source's equations are defensible. The
module takes the literal one, which sits above the source's measured figure;
`blank_leak`'s docstring states both readings and their numbers. Verification here
*reproduces the published algebra*, not a measured constant — see
[Validation](/guide/validation#two-tiers).

Reports: [Attacks · Continuous](/tests/attacks_continuous) ·
[Attacks · Threshold](/tests/attacks_threshold) ·
[Attacks · Contract](/tests/attacks_contract).

## References

| Key | Source |
| --- | --- |
| Antonelli 2024 | Antonelli et al., [arXiv:2408.01754](https://arxiv.org/abs/2408.01754) |
| Huang 2012 | Huang, Yin, Wang, Li, Chen & Han, [arXiv:1206.6591](https://arxiv.org/abs/1206.6591) |
| Kish 2024 | Kish et al., *Quantum* **8**, 1382 (2024) |
| Krause 2025 | Krause et al., [arXiv:2507.10361](https://arxiv.org/abs/2507.10361) |
| Kumar 2015 | Kumar, Qin & Alléaume, New J. Phys. 17, 043027 (2015), [arXiv:1412.1403](https://arxiv.org/abs/1412.1403) |
| Laudenbach 2018 | Laudenbach et al. 2018, [arXiv:1703.09278](https://arxiv.org/abs/1703.09278) |
| Lütkenhaus 2000 | Lütkenhaus, Phys. Rev. A **61**, 052304 (2000) |
| Mandil 2024 | Mandil, Qian & Lo, [arXiv:2407.08009](https://arxiv.org/abs/2407.08009) |
| Marie & Alléaume 2017 | Marie & Alléaume, PRA **95**, 012316 (2017) |
| Papapanos 2020 | Papapanos et al., [arXiv:2010.03358](https://arxiv.org/abs/2010.03358) |
| Qi 2015 | Qi et al., PRX 5, 041009 (2015), [arXiv:1503.00662](https://arxiv.org/abs/1503.00662) |
| Rogers 2007 | Rogers et al., New J. Phys. 9, 319 (2007), [arXiv:0706.1449](https://arxiv.org/abs/0706.1449) |
| Sharma 2024 | Sharma et al., [arXiv:2409.05802](https://arxiv.org/abs/2409.05802) |
| Singh 2025 | Singh et al., IEEE Photonics J. 17, 7600206 (2025), [arXiv:2502.04081](https://arxiv.org/abs/2502.04081) |
| Subacius 2005 | Subacius, Zavriyev & Trifonov, Appl. Phys. Lett. **86**, 011103 (2005) |
| Usenko 2012 | Usenko et al., New J. Phys. 14, 093048 (2012), [arXiv:1208.4307](https://arxiv.org/abs/1208.4307) |
| Wang 2025 | Wang et al., [arXiv:2503.10168](https://arxiv.org/abs/2503.10168) |
| Xu 2013 | Xu et al., New J. Phys. **15**, 113007 (2013) |
| Yuan 2014 | Yuan et al., PRApplied **2**, 064006 (2014) |

Cited above without an identifier here: Meda, Li, Diamanti, Ma & Lo.
