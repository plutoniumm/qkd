# Noise budget

`qkd.budget` assembles $\xi$ as an **output** of hardware parameters, each row
tagged with the plane its source formula was stated in. SNU throughout ([the SNU
boundary](/guide/conventions#the-snu-boundary)); entries stored **input-referred**.
Also reachable as `q.budget`.

```python
from qkd import budget
```

## Assembling a budget

```python
bud = budget.assemble(
    v_a=5.0,          # SNU
    t=10**-0.5,       # 25 km at 0.2 dB/km
    v_err=2e-3,       # residual phase-error variance, rad^2
    rin=-155.0,       # dBc/Hz
    bandwidth=100e6,  # Hz
    dac_bits=16,
    adc_bits=12,
)

print(bud.table())
```

```
source   xi @ input  xi @ bob    plane
phase    1.0010e-02  3.1654e-03  Kish (80)-(82) inferred channel: input-referred, no /T
dac      9.5749e-10  3.0279e-10  Laudenbach (9.37): stated Bob-plane, /T applied
adc      3.1414e-06  9.9341e-07  Laudenbach (9.108): stated Bob-plane, /T applied
rin_sig  8.8914e-04  2.8117e-04  Laudenbach (9.9): stated Bob-plane, /T applied
rin_lo   6.4528e-08  2.0406e-08  Laudenbach (9.21): stated Bob-plane, /T applied
total    1.0902e-02  3.4476e-03
```

$1.090\times10^{-2}$ SNU at the channel input, inside the [measured
envelope](#the-measured-envelope) of 25 km systems; residual phase noise
dominates by an order of magnitude. The `phase` row has [two published
forms](#the-two-forms-of-the-phase-term).

A parameter left out omits its row rather than reporting a zero:

```python
budget.assemble(v_a=5.0, t=0.3, v_err=2e-3).table()
# source  xi @ input  xi @ bob    plane
# phase   1.0010e-02  3.0030e-03  Kish (80)-(82) inferred channel: input-referred, no /T
# total   1.0010e-02  3.0030e-03
```

### `assemble(...)`

Keyword-only; `v_a` and `t` required. `Plane` is where the **source formula was
stated**, not where the entry is stored: a Bob-plane row is already divided by $T$,
and `Entry.plane_note` records it.

| Parameter | Unit | Plane | Default | Description |
| --- | --- | --- | --- | --- |
| `v_a` | SNU | input | **required** | $V_A > 0$ at the channel input. |
| `t` | — | — | **required** | Transmittance of the **fibre span alone**, $T \in (0, 1]$. Itemised optics multiply in: `Budget.T` is `t` times the chain, and that product divides every Bob-plane formula. |
| `v_err` | rad² | input | `None` | Residual phase-error variance after pilot-assisted recovery ([which one](#which-v-err)). Emits `phase`, a property of the recovered phase and so not divided by $T$. |
| `xi` | SNU | input | `0.0` | Excess noise the channel already carried, renormalised alongside $V_A$ by `phase` and by nothing else. Not a row. |
| `phase_form` | — | — | `"estimator"` | `"estimator"` or `"literature"`; anything else raises. |
| `rin` | dBc/Hz | Bob | `None` | RIN of signal laser and LO. Emits `rin_sig` and `rin_lo`. |
| `bandwidth` | Hz | — | `None` | Detection bandwidth $B$, for `rin` and `raman_db`. |
| `raman_db` | dBm/nm | Bob | `None` | **Measured** spontaneous-Raman spectral density $N_{\text{Ram}}$ at the quantum channel. Emits `raman`. Referred by $T_{\text{launch}}T_{\text{span}}$, not the total — [loss sites](#loss-sites). |
| `wavelength` | m | — | `1550.12e-9` | $\lambda$, for `raman_db` alone: $hf = hc/\lambda$ and the filter width $\lambda^2B/c$. Default ITU channel 34, Laudenbach's $193.4$ THz. |
| `dac_bits` | bit | Bob | `None` | Alice's DAC resolution $b$. Emits `dac`. |
| `adc_bits` | bit | Bob | `None` | Bob's ADC resolution $n$. Emits `adc`. |
| `losses` | — | — | `()` | `q.Connector`, `q.Splice`, `q.Coupling` — see [the loss chain](#the-loss-chain). |
| `mu` | — | — | `2.0` | Detector multiplicity inside `adc`: $1$ homodyne, $2$ heterodyne, whose two ADCs' quantisation noise does not halve at the beamsplitter (Laudenbach Eq. (9.79)). |
| `ratio` | — | Bob | `10.0` | ADC full scale in shot-noise standard deviations, $r = R_U/\sigma_{\text{shot}}$; default $\pm5\sigma$. |

| Input | Result |
| --- | --- |
| negative `v_err` or `xi` | raises, not clamped. Both forms of $\xi_{\text{phase}}$ are monotone through zero, so the literature form returns $-6.49$ SNU at $V_{\text{err}} = -1$, $V_A = 5$ |
| `xi` with no `v_err` | raises: it reaches the `phase` row and nothing else |
| `xi` with `phase_form="literature"` | raises: that form renormalises nothing, so the channel noise would be dropped |
| `rin` or `raman_db` without `bandwidth`, or `bandwidth` with neither | `ValueError` |
| `vel=` or `trusted=` | `TypeError` — see [why $v_{el}$ is not a budget line](#why-v-el-is-not-a-budget-line) |

### `Budget` and `Entry`

`Budget` is a frozen dataclass, so its entries concatenate with
[`impairments.assemble_extra()`](/guide/impairments#assemble-extra)'s.

| Member | Type | Description |
| --- | --- | --- |
| `Budget.entries` | `tuple[Entry, ...]` | One per described source, in emission order. |
| `Budget.T` | `float` | Channel input to detector input — the span *and* every itemised optic. |
| `Budget.launch` | `float` | Transmittance of the optics at Alice's output; `1.0` when none declared. |
| `Budget.receive` | `float` | Transmittance of the optics at Bob's input; `1.0` when none declared. |
| `Budget.span` | `float` | $T/(\text{launch}\cdot\text{receive})$. |
| `Budget.losses` | `tuple[Loss, ...]` | The chain in optical order, the fibre's own attenuation among it as `fiber_span`. |
| `Budget.total` | `float` | Sum of every entry's $\xi$, input-referred, SNU. |
| `Budget.factor(plane, eta=None)` | `float` | Factor referring an input-referred noise to that plane. |
| `Budget.at(plane, eta=None)` | `dict[str, float]` | `{source: xi}` at one of the [four planes](#the-four-planes). |
| `Budget.refer(xi, start, end, eta=None)` | `float` | One number moved between two planes — for a $\xi$ this budget did not produce. |
| `Budget.table()` | `str` | The aligned text table above. |
| `Entry.source` | `str` | A row name from [the per-source table](#the-per-source-table). |
| `Entry.xi` | `float` | Input-referred, SNU. |
| `Entry.plane_note` | `str` | The citation and the plane its source paper stated it in. |
| `Loss.source` | `str` | `fiber_span`, `connector_launch`, `splice_span`, `coupling_receive`. |
| `Loss.db` | `float` | The row's insertion loss in dB, `count` included. |
| `Loss.site` | `str` | `"launch"`, `"span"` or `"receive"`. |
| `Loss.transmittance` | `float` | $10^{-\text{db}/10}$. |

## The per-source table

Bob-plane rows are divided by $T$ before storing.

| Source | Formula | Stated at | Reference |
| --- | --- | --- | --- |
| `phase` | $\xi = (V_A + \xi_{\text{ch}})(e^{V_{\text{err}}} - 1)$, or $2V_A(1 - e^{-V_{\text{err}}/2})$ under `phase_form="literature"` | input | [K24] App. E, Eqs. (76)–(82), by rearranging (80)–(82); [MA17] Eqs. (10)–(11) for the second, reproduced as [K24] App. F, Eq. (85) |
| `dac` | $\xi = T V_{\text{mod}}(d + d^2/2)^2$, $d = \pi/(2^b\sqrt{12})$ | Bob | [L18] Eq. (9.37) |
| `adc` | $\xi = \mu\,r^2/(12\cdot 2^{2n})$ | Bob | [L18] Eq. (9.108), quantisation term |
| `rin_sig` | $\xi = T V_{\text{mod}}\sqrt{\mathrm{RIN}\,B}$ | Bob | [L18] Eq. (9.9) |
| `rin_lo` | $\xi = \tfrac14\mathrm{RIN}\,B\,V(\hat q)$ | Bob | [L18] Eq. (9.21) |
| `raman` | $\xi = 2\,\Delta\lambda\,10^{N_{\text{Ram}}/10}\,\tau/(hf)\times10^{6}$ | Bob | [L18] Eq. (9.63), over Eq. (9.59) $\xi = 2\langle n\rangle$ and Eq. (9.60) $\langle n\rangle = P\tau/(hf)$ |

No `detector` row and no `trusted` flag — [why](#why-v-el-is-not-a-budget-line).

Two connecting substitutions are qkd's, not verbatim in the sources:

| Row | Substitution |
| --- | --- |
| `dac` | $\delta U = \mathrm{LSB}/\sqrt{12}$ for pure quantisation of an ideal $b$-bit DAC, the full DAC range driving one half-wave voltage ($gR_U = U_\pi$) |
| `adc` | Eq. (9.108)'s opto-electronic prefactor reduced to one `ratio` $r = R_U/\sigma_{\text{shot}}$ (default $10$, i.e. $\pm 5\sigma$); the components carry no $g$, $\rho$, $\tau$ or $P_{LO}$ |

### The `raman` row

$N_{\text{Ram}}$ is **measured**: what an OSA reads at the quantum wavelength with
the classical traffic running. [L18]'s $\Delta\lambda$ is the filter the light is
read through; `raman_width()` supplies $\lambda^2B/c$, since in a coherent receiver
only modes within $B$ of the LO beat into the quadrature. `raman_width(1e9)`
$= 8.015$ pm reproduces his $\Delta\lambda \approx 8$ pm at $B = 1$ GHz on the
$1550$ nm / $193.4$ THz channel.

A matched receiver, $\tau = 1/B$ through width $\lambda^2B/c$, **cancels $B$**:

$$\xi_{\text{Ram}} = \frac{2\lambda^{3}\,p_{\text{Ram}}}{hc^{2}},
\qquad p_{\text{Ram}} = 10^{N_{\text{Ram}}/10}\times10^{6}\ \mathrm{W/m}.$$

`assemble` still takes `bandwidth`: $B$ enters `raman()` as two separate
parameters, and either alone moves the answer.

**Two conventions, a factor of two apart.** [L18] carries no $\tfrac12$ for
polarisation: his footnote to Eq. (9.63) cancels the polarising beamsplitter
against the LO's doubled mixing bandwidth. [K21], behind
[`impairments.raman`](/guide/impairments#prior-art), keeps the half as the LO's
polarisation-mode selectivity. The same density reads twice as high here, the
conservative side; [`test_raman_convention`](/tests/budget_engine) pins the ratio at
$2$.

### The formulas individually

Each returns an **input-referred** $\xi$ in SNU.

| Function | Arguments | Returns |
| --- | --- | --- |
| `phase(v_a, v_err, xi=0.0, *, form="estimator")` | $V_A$ and $\xi_{\text{ch}}$ in SNU, $V_{\text{err}}$ in rad² | [either form](#the-two-forms-of-the-phase-term) |
| `dac(v_a, bits, t)` | $V_A$ in SNU, $b$ in bits, $T$ | $V_A(d + d^2/2)^2$ with $d = \pi/(2^b\sqrt{12})$ |
| `adc(bits, t, mu=2.0, ratio=10.0)` | $n$ in bits, $T$, multiplicity, $r$ | $\mu r^2/(12\cdot 2^{2n}\,T)$ |
| `rin_sig(v_a, rin_db, bandwidth, t)` | $V_A$ in SNU, RIN in dBc/Hz, $B$ in Hz, $T$ | $V_A\sqrt{\mathrm{RIN}\,B}$ |
| `rin_lo(rin_db, bandwidth, quad_var, t)` | RIN in dBc/Hz, $B$ in Hz, $V(\hat q)$ at Bob in SNU, $T$ | $\mathrm{RIN}\,B\,V(\hat q)/(4T)$ |
| `raman_width(bandwidth, wavelength=1550.12e-9)` | $B$ in Hz, $\lambda$ in m | $\Delta\lambda = \lambda^2B/c$ in m |
| `raman(n_ram, width, symbol, t, *, wavelength=1550.12e-9)` | $N_{\text{Ram}}$ in dBm/nm, $\Delta\lambda$ in m, $\tau$ in s, $T_{\text{launch}}T_{\text{span}}$ | $2\Delta\lambda\,10^{N/10}\tau\lambda/(hcT)\times10^{6}$ |

```python
budget.phase(5.0, 2e-3)                     # 0.010010
budget.phase(5.0, 2e-3, 0.01)               # 0.010030
budget.phase(5.0, 2e-3, form="literature")  # 0.009995
budget.dac(5.0, 16, 10**-0.5)               # 9.575e-10
budget.adc(12, 10**-0.5)                    # 3.141e-06
budget.raman_width(1e9)                     # 8.015e-12  (Laudenbach's 8 pm)
budget.raman(-80.0, 8.015e-12, 1e-9, 1.0)   # 1.251e-03  at Bob, T = 1
```

`assemble` supplies `rin_lo`'s $V(\hat q)$ as $TV_A + 1$, dropping $\xi_{\text{rest}}$
and the electronic term as second order.

## The two forms of the phase term {#the-two-forms-of-the-phase-term}

$$\xi_{\text{phase}}^{\text{estimator}} = (V_A + \xi)\left(e^{V_{\text{err}}} - 1\right),
\qquad
\xi_{\text{phase}}^{\text{literature}} = 2V_A\left(1 - e^{-V_{\text{err}}/2}\right)$$

$\xi$ is the channel's input-referred excess noise. Both forms are stated at the
channel input; neither is divided by $T$.

### Which $V_{\text{err}}$ {#which-v-err}

Both read $V_{\text{err}}$ through one channel, the surviving correlation
$\kappa = e^{-V_{\text{err}}/2}$:

$$V_{\text{err}} = -2\ln\mathbb{E}\!\left[\cos(\theta - \hat\theta)\right],$$

[K24] Eq. (25)'s $\bar r = \langle e^{i\theta}\rangle = e^{-\sigma^2/2}$. It is
**not** the second moment of the residual. `src/pipeline.rs` reports both:

| Field | Quantity | Ceiling | Feeds |
| --- | --- | --- | --- |
| `SimOut.v_err` | $-2\ln\mathbb{E}[\cos\varphi]$ | none | the budget |
| `SimOut.v_wrap` | the wrapped second moment about zero | $\pi^2/3 = 3.29$, the wrap interval's own variance, while the correlation keeps falling to zero | a phase tracker's own error budget |

Only $\theta \bmod 2\pi$ reaches the frames, so any moment of the residual is of a
*wrapped* angle ([K24], before Eq. (24)).

### Where the estimator form comes from

[K24] is the citation of record: it concerns a phase *variance* and names the model
it corrects.

| Source | Route |
| --- | --- |
| [K24] App. E, Eqs. (76)–(82) | the channel Alice and Bob **infer** through a phase-noise channel. The correlation carries $\bar r = e^{-\sigma^2/2}$, so the inferred transmittance is $\eta_I = \eta\,\bar r^{\,2} = \eta\,e^{-\sigma_\theta^2}$, and the inferred noise $\chi_I$ is renormalised by that factor, its signal part $\mu(e^{\sigma_\theta^2} - 1)$ with $\mu$ the variance at the referred plane |
| [S21] Eqs. (18)–(19) | a **deterministic** receiver angle: $\varepsilon' = \varepsilon/\cos^2\theta + \tan^2\theta\,\langle x_A^2\rangle = \xi + (V_A + \xi)(1/\cos^2\theta - 1)$, the same once $\cos^2\theta$ is identified with $\kappa^2$ |
| [U12] | the fading-channel form, independently; the passage is unnumbered |

$(V_A + \xi)(e^{V_{\text{err}}} - 1)$ never appears literally in [K24]: it follows
from Eqs. (80)–(82) by rearrangement, and is cited that way. `budget.phase` returns
the *increment* over $\xi$.

### `infer()`, the transmittance half {#infer}

[K24] Eq. (78) carries the **other** half of the rotation,
$\eta_I = \eta\,\bar r^{\,2}$:

```python
budget.infer(t, v_err, form="estimator")   # -> t * exp(-v_err)
```

in whatever plane `t` is stated. The halves pin:

$$\texttt{infer}(t, v)\,\bigl(V_A + \xi + \texttt{phase}(V_A, v, \xi)\bigr) \;\equiv\; t\,(V_A + \xi),$$

so charging the noise half against an unattenuated $t$ overstates the rate. **No
domain**; `v_err = inf` returns `0.0`. `form="literature"` is **refused**:

```python
budget.infer(0.5, 0.3, form="literature")
# ValueError: form='literature' has no transmittance half to return:
#   2*V_A*(1 - exp(-v_err/2)) is derived holding the transmittance estimate
#   fixed, the assumption Kish App. E names as its error. Use the default
#   form='estimator', or carry t as it stands
```

| | |
| --- | --- |
| `Budget.T` | the **physical** loss chain, unmoved |
| `Budget.inferred` | the same `T` as parameter estimation infers it, $T e^{-v_{\text{err}}}$. Equal to `T` where no phase row was formed, so it may be read unconditionally; **raises** where that row is the literature form |

`v_err` and `phase_form` are set and cleared together. A `q.Link` run fills them
with the estimator form and carries `v_err` into the `Budget` **carried rather than
charged**: `T` stays the physical chain, no $\xi$ moves for it, and the product
goes on [`explain["T_claimed"]`](/guide/link#t-claimed). A hand-built budget that
pins its own `T` and never reads `inferred` is the case the identity warns about.

### Where the difference comes from

The literature form is $2V_A(1-\kappa)$: the noise an observer who already knows
$T$ and $\xi$ attributes to the lost correlation. Alice and Bob **fit** the
channel; the random phase shrinks the fitted slope to $\hat t = t_0\,\kappa$, and
parameter estimation divides the residual variance by $\hat t^2$ ([L10] Eq. (20),
`src/pipeline.rs`). Through the regression,

$$\hat\sigma^2 = t_0^2 V_A\left(1 - \kappa^2\right) + t_0^2\,\xi + 1 + v_{el},
\qquad
\hat\xi - \xi_{\text{ideal}}
= \frac{\hat\sigma^2 - 1 - v_{el}}{\hat t^{\,2}} - \xi
= (V_A + \xi)\left(\frac{1}{\kappa^2} - 1\right).$$

The renormalisation is the difference, and why $\xi$ appears in one form and
not the other. [K24] make this criticism of [MA17] by name at the end of their
Appendix E.

### Which plane, and which denominator {#phase-plane-and-denominator}

The second-order answer depends on plane and denominator — [the referral
trap](/guide/conventions#excess-noise-carries-a-plane) at second order.

| Choice | What is shipped |
| --- | --- |
| Plane | The **channel input** (Alice), for both $V_A$ and $\xi$, in SNU with the vacuum at 1 per quadrature. Every budget entry lives there. |
| Slope | The least-squares $\hat t = \sum a y / \sum a^2$ of Bob's quadrature on Alice's symbol, and the residual variance about **that** fitted line. |
| Denominator | $\hat t^{\,2}$, the *fitted* slope, not the design slope $t_0^2$. |
| Receiver | Calibrated inside the estimator: $v_{el}$ subtracted before the division and $\eta$ carried inside the slope, i.e. $\hat\xi = (\hat\sigma^2 - 1 - v_{el})/\hat t^{\,2}$ and $\hat T = 2\hat t^{\,2}/\eta$, exactly as `src/pipeline.rs` computes them. |

The last row belongs to the **estimator**, not the security model: a rate from
these estimates may still hand $\eta$ and $v_{el}$ to Eve through `trusted=False`.
Referring to Bob's plane, or dividing by $t_0^2$, changes the second-order term.

### They agree to first order, then separate

With $\xi = 0$ both reduce to $V_A V_{\text{err}}$ as $V_{\text{err}} \to 0$, and
the relative gap opens **linearly**:

$$\frac{\xi^{\text{estimator}} - \xi^{\text{literature}}}{\xi^{\text{estimator}}}
= \tfrac34 V_{\text{err}} + O(V_{\text{err}}^2),
\qquad
\frac{\xi^{\text{estimator}}}{\xi^{\text{literature}}}
= \frac{e^{V_{\text{err}}} + e^{V_{\text{err}}/2}}{2}.$$

The ratio is **analytic**: with [this $V_{\text{err}}$](#which-v-err) both forms are
identities, not approximations. With $u = e^{-V_{\text{err}}/2}$ it is $(1+u)/(2u^2) > 1$ for every $u < 1$,
so the literature form is strictly the smaller at every $V_{\text{err}} > 0$.

| $V_{\text{err}}$ (rad²) | estimator | literature | literature is low by |
| --- | --- | --- | --- |
| $10^{-3}$ | 0.005003 | 0.004999 | 0.07% |
| $2\times10^{-3}$ | 0.010010 | 0.009995 | 0.15% |
| $10^{-2}$ | 0.050251 | 0.049875 | 0.75% |
| $0.05$ | 0.256355 | 0.246901 | 3.7% |
| $0.1$ | 0.525855 | 0.487706 | 7.3% |
| $0.28$ | 1.615649 | 1.306418 | 19.1% |
| $1.0$ | 8.591409 | 3.934693 | 54.2% |

$V_A = 5$ SNU, $\xi = 0$, input-referred.

### Measured against the simulator's own estimator

$\hat\xi - \xi_{\text{ideal}}$ — recovered frames against the same frames derotated
by the oracle phase — is the excess noise phase recovery cost. Linewidth sweep at
$V_A = 5$, $T = 0.5$, $\xi = 0.01$, $\eta = 0.6$, $v_{el} = 0.1$, $10^6$ symbols,
seed 7; positive means the expression sits *below* the measurement.

| $V_{\text{err}}$ | measured | literature is low by | estimator is off by |
| --- | --- | --- | --- |
| 0.0024 | 0.0126 | +3.4% | +3.1% |
| 0.0535 | 0.2771 | +5.0% | +0.8% |
| 0.1575 | 0.8557 | +13.0% | +0.3% |
| 0.2715 | 1.5745 | +24.0% | +0.9% |
| 0.3990 | 2.4676 | +36.5% | +0.6% |
| 0.5957 | 4.0817 | +58.5% | +0.2% |
| 1.1513 | 10.9099 | +149.3% | +0.8% |
| 2.0683 | 35.0527 | +443.9% | +1.4% |

| Column | Behaviour |
| --- | --- |
| literature | monotone, one-signed, following the analytic ratio above |
| estimator | **flat**: an identity in $V_{\text{err}}$, the residual being the sampling error of $V_A + \xi$ at $10^6$ symbols. Five seeds scatter each entry by about 2 points; the first row, at $0.0126$ SNU, is noisiest. At $V_{\text{err}} = 4.15$, excess noise 306 SNU, it lands within 2.3% |
| the run's **own** slope ratio $\hat t/t_{\text{ideal}}$ substituted for $\kappa$ | reproduces the measurement to better than $0.06\%$ at every point, the residual constant across linewidths for a given seed — [`test_estimator_slope_ratio`](/tests/dsp_recovery) |

A **modelling error, not a security hole**: a deployed system measures $\xi$, where
the estimator bias makes the measured value the *larger* and the claimed rate
*lower*. The literature form biases the **predicted** number, $\xi$ under and rate
over.

### The literature form's domain {#no-upper-limit}

| Form | Domain | Past it |
| --- | --- | --- |
| `"literature"` | `PHASE_LIMIT = 0.1` rad² | *saturates* at $2V_A$ — 10 SNU at $V_A = 5$ — while measured excess noise has no ceiling: 6.44 SNU at $V_{\text{err}} = 2.07$ against a measured 35.05, the gap past a factor of 19 by $3.48$. Emits a `PhaseDomainWarning` naming $V_{\text{err}}$ and the limit, and **still returns the number** |
| `"estimator"` | **none** | $e^{V_{\text{err}}} - 1$ *is* $1/\kappa^2 - 1$ for a phase distribution of any width and shape. Never warns |

The $0.1$ is [K24]'s figure but **not a domain they put on Eq. (85)**: their
sentence bounds the *further* linearisation $\xi_x = \xi_p \approx
\tilde\sigma_\Theta^2\,2V_A$ for $\tilde\sigma^2 < 0.1$, and Eq. (85) is printed
with no stated range. qkd uses it as the range over which the literature form
tracks the estimator one — a house reading of that sentence, not a quotation.

`warnings.simplefilter("error", PhaseDomainWarning)` escalates the warning;
`"ignore"` silences it. There is no `WRAP_LIMIT`: its 1.5 rad² was the **wrapped**
second moment saturating.

### Which one to select

```python
budget.assemble(v_a=5.0, t=0.5, v_err=2e-3, xi=0.01)              # estimator row
budget.assemble(v_a=5.0, t=0.5, v_err=2e-3, phase_form="literature")
```

| Task | `form=` | Why |
| --- | --- | --- |
| Predicting a rate, or comparing against anything this simulator measures | `"estimator"`, the default | it is what parameter estimation reports |
| Reproducing a published number computed with the other form | `"literature"` | [MA17] Eqs. (10)–(11) state $2V_A(1 - e^{-V_{\text{err}}/2})$, reproduced as [K24] App. F, Eq. (85). Every anchor fitted against that form selects it explicitly and says so in its report row |

`Entry.plane_note` names the expression behind each `phase` row.

[`Link` with a DSP chain](/guide/link#dsp-in-the-loop) derives $V_{\text{err}}$ and
applies the estimator form, the channel's $\xi$ as second argument. There is no
`Link`-level selector (`res.explain["phase_form"]` names the form): to reproduce a
literature-form number on a `Link`, assemble here and pin the total onto
`q.Channel(xi=…, ref="input")`. `impairments.dephasing` carries the same selector
and default.

## The loss chain {#the-loss-chain}

$T = \prod_i 10^{-\ell_i/10}$.

```python
import qkd as q

bud = budget.assemble(
    v_a=5.0,
    t=10**-0.5,           # the SPAN alone, 25 km at 0.2 dB/km
    v_err=2e-3,
    rin=-155.0,
    bandwidth=100e6,
    dac_bits=16,
    adc_bits=12,
    losses=(
        q.Connector(loss=0.25, count=2, site="launch"),
        q.Splice(loss=0.02, count=6, site="span"),
        q.Coupling(loss=1.5, site="receive"),
    ),
)

print(bud.table())
```

```
loss              dB      T         site
connector_launch  0.5000  0.891251  launch
fiber_span        5.0000  0.316228  span
splice_span       0.1200  0.972747  span
coupling_receive  1.5000  0.707946  receive
link              7.1200  0.194089

source   xi @ input  xi @ bob    plane
phase    1.0010e-02  1.9428e-03  Kish (80)-(82) inferred channel: input-referred, no /T
dac      9.5749e-10  1.8584e-10  Laudenbach (9.37): stated Bob-plane, /T applied
adc      5.1183e-06  9.9341e-07  Laudenbach (9.108): stated Bob-plane, /T applied
rin_sig  8.8914e-04  1.7257e-04  Laudenbach (9.9): stated Bob-plane, /T applied
rin_lo   8.0261e-08  1.5578e-08  Laudenbach (9.21): stated Bob-plane, /T applied
total    1.0904e-02  2.1164e-03
```

| On the run above | |
| --- | --- |
| 5.0 dB of fibre becomes 7.12 dB of link | each decibel has a named row |
| `adc` moves to $5.118\times10^{-6}$ from $3.141\times10^{-6}$ | stated at Bob's plane and referred to the input by the total $T$: 2.12 dB more link is 2.12 dB more input-referred quantisation noise |
| `phase`, `dac`, `rin_sig` do not move | Alice-plane quantities |
| a second descriptor of the same kind at the same site | raises rather than merging. `count=` multiplies the decibels on **one** line |

### What each one is, and what it costs

| Component | What it is physically | Typical | Source |
| --- | --- | --- | --- |
| `q.Connector` | A **mated pair** of ferrules: mode mismatch plus the lateral offset and air gap the sleeve leaves. | **0.25 dB** (FC/APC), 0.3 dB (FC/PC) | IEC 61753-1 grade C: mean $\le 0.25$ dB, max $\le 0.50$ dB, measured per IEC 61300-3-34. Thorlabs quotes 0.25 dB typical FC/APC, 0.3 dB FC/PC |
| | grades B / C / D, mean (max) | 0.12 (0.25) / 0.25 (0.50) / 0.50 (1.00) dB | IEC 61753-1 attenuation grades, the max holding for $>$97% of samples |
| `q.Splice` | A **fusion splice**: core eccentricity and mode-field mismatch. | **0.02 dB** core-aligning, 0.03–0.04 dB v-groove | Corning/AFL AN0041, SMF-28 Ultra at 1550 nm, measured per ANSI/TIA/EIA-455-8. Telcordia GR-20-CORE asks for a group mean $\le 0.10$ dB |
| `q.Coupling` | Light crossing between **two different modes** — fibre to chip, free space to fibre, fibre to a bulk bench: the overlap integral of two mode profiles. | **no default** | the five rows below |
| | silicon grating coupler | 3.1 dB, or 1.0–1.6 dB with a bottom reflector | Mu *et al.*, Appl. Sci. **10**, 1538 (2020), Table 2; Cheng *et al.*, Micromachines **11**, 666 (2020) |
| | silicon inverse-taper edge coupler | 1.3–1.5 dB | Mu *et al.* 2020, Tables 2–3 |
| | free-form coupler to standard SMF-28 | 0.8 dB | Ranno *et al.*, Photonics Res. **12**, 1055 (2024) |
| | thin-film lithium niobate edge coupler | 0.24–0.29 dB, but **to a 4.8 µm-MFD high-NA fibre**, not SMF-28 | Chen *et al.*, APL Photonics **9**, 116111 (2024) |
| | free space $\to$ single-mode fibre | $\sim$0.97 dB at the theoretical best (80% Airy–Gaussian overlap); $\sim$1.6 dB measured on a diffraction-limited beam | Jovanovic *et al.*, A&A **604**, A122 (2017), after Shaklan & Roddier, Appl. Opt. **27**, 2334 (1988) |

What CV-QKD labs charge themselves:

| System | Charged | Source |
| --- | --- | --- |
| QOSST, 25.2 km spool | **0.47 dB** to the connectors, and **0.23 dB per polarisation-maintaining mating sleeve** | [Quantum **8**, 1575 (2024)](https://doi.org/10.22331/q-2024-12-23-1575) |
| Hajomer *et al.*, 100 km | 15.4 dB over 0.146 dB/km fibre, leaving **0.8 dB** across two mode-field-diameter mismatch junctions | [Sci. Adv. **10**, eadi9474 (2024)](https://doi.org/10.1126/sciadv.adi9474) |

`q.Coupling` has no default: published values span an order of magnitude and depend
on what is coupled to what. Return loss is not modelled and there is no `polish=`:
FC/PC and FC/APC cost about the same *forward* loss and differ in **return** loss,
roughly 50 dB flat against 60 dB angled.

### Loss sites

Every loss carries a `site` — `"launch"` at Alice's output, `"receive"` at Bob's
input, `"span"` between them:

$$T \;=\; T_{\text{launch}}\, T_{\text{span}}\, T_{\text{receive}},$$

the same number however the decibels are distributed. A noise **born inside the
span** — Raman, Rayleigh backscatter — crosses Bob's receive optics *with the
signal*, so that loss cancels out of its shot-noise ratio:

$$\xi_{\text{input}} \;=\; \frac{\xi_{\text{span}}}{T_{\text{launch}}T_{\text{span}}}
\qquad\text{and not}\qquad \frac{\xi_{\text{span}}}{T}.$$

| Move 1.5 dB of coupling | $T$ | detector-plane rows | input-referred Raman |
| --- | --- | --- | --- |
| at Bob's input | unchanged | unchanged | unchanged |
| at Alice's output | unchanged | unchanged | **+1.5 dB** — Alice's coupling attenuates the signal *before* the scattering, Bob's attenuates signal and scattered light together *after* it |

`q.Link` hands fibre-born impairments $T_{\text{launch}}T_{\text{span}}$ and
everything else the total, so launch-side placement costs key at identical $T$
([`test_born_in_span`](/tests/budget_optics)). `assemble(raman_db=…)` is the only
row of its own split the same way ([`test_raman_span`](/tests/budget_optics) puts
`raman` and `adc` on opposite sides of the same $1.5$ dB of receive coupling).

## The four planes {#the-four-planes}

A **plane** is a place on the optical path a noise is quoted at, named after
hardware, never after a party. `.at(plane)` is the only way to move an entry.

| Plane | Where it is | Factor from the channel input |
| --- | --- | --- |
| `"channel_input"` | Alice's output, where $V_A$ is defined and measured | $1$ |
| `"channel_output"` | the far end of the span, before any receive-side optics | $T_{\text{launch}}T_{\text{span}}$ |
| `"detector_input"` | the detector's front face, after those optics, before $\eta$ | $T$ |
| `"post_detection"` | after the detector's quantum efficiency | $T\eta$ |

`"input"` and `"bob"` alias `channel_input` and `detector_input`; anything else
raises.

```python
bud.at("channel_input")     # {'phase': 0.010010, 'adc': 5.118e-06, ...}
bud.at("channel_output")    # {'phase': 0.002744, 'adc': 1.403e-06, ...}
bud.at("detector_input")    # {'phase': 0.001943, 'adc': 9.934e-07, ...}
bud.at("alice")             # ValueError: plane must be one of ('channel_input', ...)
```

### The same noise at two planes

`.refer(xi, start, end)` moves one number — a $\xi$ measured at the fibre end, or
quoted at Bob:

```python
bud.refer(1e-3, "detector_input", "channel_input")   # 0.005152
bud.refer(1e-3, "channel_output", "channel_input")   # 0.003648
```

The factor of 1.41 between the answers is the 1.5 dB of coupling between the two
planes:

$$\xi_{\text{end}} \;=\; \xi_{\text{start}}\,\frac{f(\text{end})}{f(\text{start})},$$

$f$ the factor column. Composition and $A \to B \to A$ as the identity hold over
every ordered pair ([`test_plane_compose`](/tests/budget_planes)). `.at(plane)` is
`refer()` row by row from the channel input.

### `post_detection` takes an $\eta$, as an argument

```python
bud.at("post_detection")
# ValueError: post_detection needs eta=: a budget carries no quantum efficiency
#             and will not assume a perfect detector

bud.at("post_detection", eta=0.6)   # every value multiplied by T * 0.6
```

$\eta$ belongs to the receiver: a query parameter, never a `Budget` field — no
`eta`, no `v_el` ([`test_plane_detector`](/tests/budget_planes)). Past the detector
is one factor of $\eta$; electronic noise and the heterodyne vacuum unit
are the key-rate layer's arithmetic, unreachable from `at()`.

### On a run

`res.budget` is the same object, assembled from the run: what the channel *did*,
beside `res.est` — what Alice and Bob *measured* from a finite sample — and
`res.oracle`. `None` for the click families.

```python
res = q.Link(
    modulation=q.GaussianModulation(v_a=5.0),
    channel=q.Fiber(length=25.0),
    bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
    impairments=(q.Coexistence(channels=4, launch=0.0),),
    losses=(q.Coupling(loss=1.5, site="receive"),),
).run()

res.budget.total                          # the xi the rate was computed at
res.budget.at("channel_input")            # {'channel': 0.0, 'raman': 0.018082}
res.budget.at("detector_input")["raman"]  # the same noise, one factor of T later
res.budget.losses                         # the itemised chain, fibre included
```

## Why $v_{el}$ is not a budget line {#why-v-el-is-not-a-budget-line}

```python
budget.assemble(v_a=5.0, t=10**-0.5, v_err=2e-3, vel=0.1)
# TypeError: assemble() got an unexpected keyword argument 'vel'
```

$v_{el}$ reaches the key-rate layer as `q.Homodyne(v_el=…)` or
`q.Heterodyne(v_el=…)`, never folded into $\xi$. The untrusted case is [the
substitution](/guide/security#trusted-vs-untrusted-is-a-security-model)
$\xi \to \xi + \mu v_{el}/(\eta T)$ ([LP19] Sec. 2, Eqs. (13)/(14)), applied by
`cv_rate(..., trusted=False)` and nowhere else.

| Layer | Owns | Pinned by |
| --- | --- | --- |
| budget | $\xi$ — what the channel and the transmitter did to the light. No $\eta$, so it could not form the term above | `test/consistency.py`: no untrusted budget can be built to feed $v_{el}$ twice |
| key rate | $(\eta, v_{el}, \text{trusted})$ — what the receiver and the threat model do to it | `test/consistency.py`: the untrusted rate equals a trusted rate on the hand-merged channel |

At $T = 10^{-0.5}$, $\eta = 0.6$, $v_{el} = 0.1$ heterodyne, that term is
$\approx 1.05$ SNU against a $10^{-2}$ SNU channel budget.

## The measured envelope

What real systems report ([defaults survey](/guide/validation#where-the-defaults-come-from)):

| System | $\xi$ | Plane | Source |
| --- | --- | --- | --- |
| Jouguet 2013, 80 km, TLO | 0.001–0.002 (0.007–0.008 worst case) | input | [arXiv:1210.6216](https://arxiv.org/abs/1210.6216) |
| Zhang 2020, 27–203 km, TLO | 0.0015–0.0086 (0.0383 worst case) | input | [arXiv:2001.02555](https://arxiv.org/abs/2001.02555) |
| QOSST benchmarks, 0–25 km, LLO | 0.0062–0.0095 | Bob | [arXiv:2404.18637](https://arxiv.org/abs/2404.18637) |
| Hajomer 2024, 100 km LLO | 2.12e-4 | output | [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| Hajomer 2024, 10 GBaud, DM | 0.0159–0.0718 | input | [arXiv:2305.19642](https://arxiv.org/abs/2305.19642) |
| Laudenbach 2018, worked example | 0.0653 | output | [arXiv:1703.09278](https://arxiv.org/abs/1703.09278) |

Zhang's values are transmitted-LO, low partly for want of two-laser phase noise;
QOSST and Hajomer are the LLO comparators. **Validation target:** a default 25 km
run lands at $\xi \approx 0.005$–$0.03$ SNU at the channel input
([`test_metro_envelope`](/tests/budget_tierb)).

## Reproducing Hajomer 2024

The closest published system to qkd's default configuration: CW light, locally
generated LO, heterodyne, frequency-multiplexed pilot, 100 km of ultra-low-loss
fibre.

```python
h = budget.assemble(
    v_a=8.41,          # published V_mod
    t=10**-1.54,       # published 15.4 dB
    v_err=7e-4,        # FITTED, rad^2
    rin=-155.0,        # FITTED, dBc/Hz
    bandwidth=100e6,   # FITTED, Hz
    dac_bits=16,       # published
    adc_bits=16,       # published
)

h.total            # 0.007385   input-referred
h.total * h.T      # 2.130e-04  Bob plane
```

Published channel-output $\xi = 2.12\times10^{-4}$ SNU; the budget lands 0.47%
high. Their $v_{el} = 0.06272$ SNU reaches a rate as `q.Heterodyne(v_el=0.06272)`,
not the budget. Three parameters are fitted — Tier B, with their allowed ranges and
plausibility in [Validation](/guide/validation#tier-b-cv).

The $v_{\text{err}}$ fit was made against the literature form and stands under the
estimator default: at $7\times10^{-4}$ rad² the forms differ by 0.05%, moving the
Bob-plane total from $2.1290\times10^{-4}$ to $2.1299\times10^{-4}$.

## Further impairments

[`qkd.impairments`](/guide/impairments) holds the sources beyond [L18] Section 9 and
the click-protocol observables that are **not** excess noise. Its
[`assemble_extra()`](/guide/impairments#assemble-extra) rows concatenate onto a
`Budget`.

## What the budget does not do yet

| Gap | State |
| --- | --- |
| CMRR ([L18] Eq. (9.80)) | not implemented |
| A measured Raman density on a run | `assemble(raman_db=…)` is **budget-only**: no `q.Link` consumes an $N_{\text{Ram}}$. A run reaches Raman through `q.Coexistence`, [K21]'s launch-power parameterisation. Declaring both would count one mechanism twice |
| Hardware rows on a `q.Link` | which rows a run charges and which it refuses: [Status](/roadmap#link). The rest are assembled here and pinned onto `q.Channel(xi=…, ref="input")`; a pinned `q.Channel` refuses `losses=`, so fold the chain into `T` |

## References

| | |
| --- | --- |
| [K24] | Kish et al., *Quantum* **8**, 1382 (2024), [arXiv:2206.13724](https://arxiv.org/abs/2206.13724) |
| [MA17] | Marie & Alléaume, *Phys. Rev. A* **95**, 012316 (2017) |
| [S21] | Shen et al., *Opt. Express* **29**, 30978 (2021), [arXiv:2107.01798](https://arxiv.org/abs/2107.01798) |
| [U12] | Usenko et al., *New J. Phys.* **14**, 093048 (2012) |
| [L18] | Laudenbach et al. 2018, [arXiv:1703.09278](https://arxiv.org/abs/1703.09278) |
| [L10] | Leverrier et al. 2010 |
| [LP19] | Laudenbach & Pacher, [arXiv:1904.01970](https://arxiv.org/abs/1904.01970) |
| [K21] | Kumar, Qin & Alléaume — see [Impairments](/guide/impairments) |
