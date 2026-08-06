# Non-Gaussian states

`qkd.fock` holds one mode as a density matrix on levels $0 \dots d-1$, with the
[conventions](/guide/conventions#the-snu-boundary) and `(len(ps), len(xs))` grid
layout of [`qkd.gaussian`](/guide/gaussian): the layers compare with no conversion.
[Scope](#scope).

```python
from qkd import fock
```

## Constructors {#constructors}

`cutoff` is keyword-only, $1 \le d \le 512$. **The cutoff is part of the state**:
the only check on a derived figure is convergence under `at_cutoff(d)`.

| `Coherent(11.0, 0.0)` at the default cutoff 40 | |
| --- | --- |
| Norm discarded | **99.79 %** |
| Reported Wigner negativity | **2.5072**, above every resource state here, on a state whose true negativity is zero |
| `physical()`, `purity()`, [Gaussian shadow](#shadow) | `True`, $1.0$, bona fide |
| Artefact versus discarded population | 1.4 to 6.7 orders of magnitude above it, so `trunc_error()` does not bound it |

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `Vacuum(*, cutoff)` | levels | `40` | `Number(0)`. |
| `Number(n, *, cutoff)` | — | **required** | $\lvert n\rangle$, $0 \le n < d$. Zero truncation error. |
| `Coherent(x, p, *, cutoff)` | internal quadratures | **required** | $\beta = (x + ip)/\sqrt2$, as `gaussian.Coherent(x, p)`. The Poissonian tail reaches every level, so `discarded()` is non-zero. |
| `Squeezed(r, *, cutoff)` | nepers | **required** | Squeezed vacuum, $x$ squeezed for $r > 0$, as `gaussian.Squeezed(r)`. True negativity zero at every $r$: the grid's calibration state. |
| `Thermal(nbar, *, cutoff)` | photons | **required** | $\bar n \ge 0$. Mixed *and* Gaussian, exercising both halves of the non-Gaussianity measure. |
| `Cat(x, p, odd=False, *, cutoff)` | internal quadratures | **required** | Even or odd superposition of the coherent states at $\pm(x, p)$. An odd cat at the origin raises: $\lvert\beta\rangle - \lvert-\beta\rangle$ vanishes. |
| `Gkp(logical, delta, *, cutoff)` | — / vacuum units | **required**, `cutoff=None` | Finite-energy grid state, `logical` $\in \{0, 1\}$, $\delta \in [0.05, 2]$; squeezing $-20\log_{10}\delta$ dB, so $\delta = 0.3$ is 10.5 dB. `None` derives the cutoff as `gkp_levels(delta)`; a pinned value is honoured; a *derived* cutoff that cannot hold the state raises. |
| `Density(rho)` | — | **required** | Square complex density matrix; its dimension is the cutoff and `discarded()` is 0. Validated for shape, finiteness, Hermiticity to $10^{-9}$, unit trace and positive semidefiniteness — the number-basis [bona fide condition](/guide/conventions#the-bona-fide-condition). |

### Module constants

| Name | Value | Bounds |
| --- | --- | --- |
| `CUTOFF` | `40` | Default cutoff for every constructor except `Gkp`. **Not** safe for a large displacement. |
| `MAX_LEVELS` | `512` | The native core's cap, so a derived cutoff clamps before the core refuses. |
| `GKP_SCALE`, `GKP_TAIL` | `8.0`, `1e-6` | The numerator of `gkp_levels`, and the discarded norm a *derived*-cutoff grid state may reach before its constructor raises. A pinned cutoff is never refused. |
| `gkp_levels(delta)` | `ceil(GKP_SCALE / delta**2)`, capped at `MAX_LEVELS` | Levels needed for a discarded population below `GKP_TAIL` track $7.1/\delta^2$ over $\delta \in [0.15, 0.6]$: 66, 89, 128 and 200 at $\delta$ = 0.35, 0.3, 0.25, 0.2. $\delta \lesssim 0.12$ is refused — no cutoff within the cap converges. |
| `EDGE` | `3` | Default `levels` for `tail()`, and the depth `trunc_error()` sums over. |
| `STEP` | `0.06` | Default phase-space sample spacing for `window()`. |

## Truncation {#truncation}

| Method | Catches |
| --- | --- |
| `discarded()` | norm thrown away **at construction** — a state born too big for its cutoff |
| `tail(levels=3)` | population in the **top few levels** — a state born inside its cutoff, then pushed against the ceiling by a displacement or a channel |
| `trunc_error()` | the worse of the two, as one number to threshold on |

A truncated matrix is renormalised, so it passes every physicality check. Neither
instrument bounds the kernel: both measure *population*, and the displaced-parity
kernel oscillates, so the truncated tail contributes to $\iint\lvert W\rvert$ with
alternating sign.

| Operation | Truncation cost |
| --- | --- |
| `rotate(theta)` | none — diagonal in the number basis |
| `loss(eta)` | none — the Kraus operators only move population down, so it is trace-preserving inside the cutoff and *shrinks* the tail |
| `displace(x, p)` | yes — the truncated displacement is not unitary: it loses norm and pushes population up. The loss is folded into `discarded()`, not renormalised away |

## Phase space {#phasespace}

$W$ and $Q$ as in the [Gaussian layer](/guide/gaussian#wigner-and-husimi), in
$dx\,dp$. A general density matrix has no closed form, so $W$ uses the
Cahill–Glauber displaced-parity kernel on a rescaled three-term Laguerre recursion.

```python
from qkd import fock

grid = [-1e-9, 0.0, 1e-9]
vac = fock.Vacuum(cutoff=24)

vac.wigner(grid, grid)[1, 1]   # 0.3183098861837907  == 1/pi
vac.husimi(grid, grid)[1, 1]   # 0.15915494309189535 == 1/(2 pi)
```

$W_{|1\rangle}(0,0) = -1/\pi$.

## Wigner negativity as the witness {#witness}

| Theorem | Statement |
| --- | --- |
| Spekkens 2008 | negativity of a quasi-probability representation and the failure of generalised non-contextuality are the same statement |
| Mari & Eisert 2012; Veitch et al. 2013 | Wigner-positive states, under Gaussian operations and homodyne readout, are efficiently classically simulable |

The computed quantity is the Kenfack–Życzkowski indicator,

$$\delta(\rho) = \iint \left|W(x,p)\right|\,dx\,dp - 1,$$

zero iff $W \ge 0$ everywhere. Its logarithm is the resource monotone of Albarelli
et al. 2018; qkd reports the volume.

### The anchor {#anchor}

$\delta(|1\rangle) = 4/\sqrt e - 2 = 0.4261226\ldots$ in closed form.

```python
from qkd import fock

fock.Number(1, cutoff=12).negativity().value   # 0.4262150
fock.Vacuum(cutoff=12).negativity().value      # -9.1e-07, i.e. zero
```

Four significant figures; the residual is the default grid's quadrature error —
[validation](/guide/validation) quotes the same state on a stated grid. A number
state discards nothing at any $d > n$. The volume grows monotonically from
$|0\rangle$ to $|4\rangle$.

### The grid is part of the answer {#grid}

`negativity()` returns a `Volume`, not a bare float: the integral runs over a
window nobody chose explicitly.

| Member | Description |
| --- | --- |
| `value` | $\iint\lvert W\rvert - 1$. `float(vol)` returns it. |
| `trunc_error` | The state's truncation error. |
| `cutoff` | The cutoff the state was held at. |
| `points` | `(len(xs), len(ps))`. |
| `span` | `(x_lo, x_hi, p_lo, p_hi)`. |
| `step` | `(dx, dp)`. |

| `window(state, reach=None, step=0.06)` | |
| --- | --- |
| `reach=None` | five standard deviations of the state's second moments plus its displacement, on the wider quadrature, **held to $\sqrt{2\,\text{cutoff}} + 5$** |
| Five | a Gaussian leaves $1.1\times10^{-6}$ of its mass outside a $5\sigma$ square and $1.4\times10^{-5}$ outside $4.5\sigma$, the latter already the size of a truncated state's spurious volume |
| The cap | a state too big for its cutoff has the *renormalised* matrix's moments: `Coherent(11, 0)` at cutoff 40 asks for reach 25.65, an $857^2$ grid. Levels $0..d-1$ cannot reach past the top Hermite function's turning point $\sqrt{2d-1}$ plus a Gaussian tail — the bound `gkp()` sizes its quadrature with. Capping moves volumes by at most $1.2\times10^{-4}$ |
| `step` | resolves the kink in $\lvert W\rvert$ at the zero contour; too coarse a step reports a smaller volume |

| State | `trunc_error` | Volume, true value $0$ |
| --- | --- | --- |
| `Squeezed(0.8)`, cutoff 25 | $2.2\times10^{-5}$ | $4.5\times10^{-3}$ |
| the same at cutoff 40 | — | $2.7\times10^{-4}$ |
| the same at cutoff 60 | — | $4.1\times10^{-6}$ |

Refining the grid does not move those; raising the cutoff does, monotonically. A
volume is evidence of negativity when it **stops changing** as the cutoff grows.
Volume over `trunc_error` is not a screen: a wholly spurious $2.505$ scores $2.5$,
a wholly spurious $4.9\times10^{-3}$ scores $1400$, a mostly physical GKP volume
$27$.

## Two witnesses {#pair}

`non_gaussianity()` is the relative-entropy non-Gaussianity $S(\rho_G) - S(\rho)$,
$\rho_G$ the Gaussian state with $\rho$'s first and second moments (Genoni & Paris
2010) — [`shadow()`](#shadow). It returns a `Divergence`: `value`, `trunc_error`,
`cutoff`, and `float()`.

Truncation moves *both* halves, so a Gaussian state can score large and
no cheap bound separates the columns:

| At cutoff 40 | Artefact | Resource |
| --- | --- | --- |
| `non_gaussianity()`, bits | `Squeezed(2.0)` $2.284$, `Coherent(11.0, 0.0)` $5.324$ | odd cat $2.16$, `Number(10)` $4.83$ |

Negativity implies non-Gaussianity; the converse fails. An equal mixture of two
coherent states is non-Gaussian with a strictly positive Wigner function:
`non_gaussianity()` sees it, `negativity()` does not. Loss separates them at a
calculable point:

```python
from qkd import fock

cat = fock.Cat(3.0, 0.0, True, cutoff=40)

for eta in (1.0, 0.7, 0.51, 0.5, 0.49):
    st = cat.loss(eta)
    print(eta, st.negativity().value, float(st.non_gaussianity()))
# 1.00   0.600926   2.5555
# 0.70   0.024912   1.3117
# 0.51   0.000048   1.0931
# 0.50  -0.000000   1.0798
# 0.49  -0.000000   1.0662
```

Negativity reaches zero at $\eta = \tfrac12$ and stays there;
non-Gaussianity falls smoothly through it. The same $\eta = \tfrac12$ kills
$|1\rangle$'s negativity, where the parity $\langle(-1)^n\rangle = 1 - 2\eta$
crosses zero. **Negativity is the resource, non-Gaussianity a symptom.**

### Parity {#parity}

`parity()` returns $\langle(-1)^{\hat n}\rangle = \pi\,W(0,0)$ — no grid, and the
quantity an experiment measures. A negative value certifies negativity at the
origin: sufficient, **not** necessary. The odd cat sits at $-1$, the even cat
at $+1$.

## The Gaussian shadow {#shadow}

`moments()` returns $(\langle x\rangle, \langle p\rangle)$ and the $2\times2$
covariance; `shadow()` is `gaussian.Moments(mean, cov)` over it.

```python
from qkd import fock

fock.Coherent(1.0, 0.0).shadow().cov
# array([[0.5, 0. ],
#        [0. , 0.5]])
```

| What the shadow is used for | |
| --- | --- |
| Hudson as a live test | a coherent, squeezed or thermal state built in both layers must give pointwise-equal Wigner grids |
| Non-Gaussian states | the shadow stays non-negative where the state it shadows does not |
| $S(\rho_G) \ge S(\rho)$ | a Gaussian maximises entropy at fixed second moments, so a negative `non_gaussianity()` is numerical noise |

## Scope {#scope}

No protocol consumes this layer: Gaussian-modulation CV-QKD is Gaussian end to end,
the discrete-modulation analysis works from a covariance matrix, and the click
family is written in photons.

| Limit | |
| --- | --- |
| Single mode | no two-mode gates, no tensor network, no MPS — [Architecture](/architecture#engines-how-notes-03-md-is-superseded) |
| GKP is the finite-energy approximation | a Gaussian-enveloped comb of Gaussian peaks in the position wavefunction, at $x_s = 2s\sqrt\pi$ for $\lvert 0_L\rangle$ and $(2s+1)\sqrt\pi$ for $\lvert 1_L\rangle$; not the ideal grid state |
| Operations | pure loss, rotation and displacement only. No thermal-loss channel, no cubic-phase gate, no Kerr evolution, no photon-number-resolving measurement |

## Read surface {#code}

```python
from qkd import fock

odd = fock.Cat(3.0, 0.0, True, cutoff=30)
odd.parity()                 # -1.0 exactly
odd.negativity().value       # 0.600926  at cutoff 30
odd.trunc_error()            # 9.1e-13 -- how much the cutoff cost

gkp = fock.Gkp(0, 0.3)       # cutoff derived: gkp_levels(0.3) == 89
gkp.cutoff                   # 89
gkp.negativity().value       # 0.926206  at cutoff 89
gkp.trunc_error()            # 9.4e-08

thermal = fock.Thermal(0.5, cutoff=30)
thermal.negativity().value   # -9.8e-07 -- Gaussian, so Hudson applies
float(thermal.non_gaussianity())   # 4.6e-13  -- and it is Gaussian too
```

The cat reads $0.600926$ at cutoff 30 and at `at_cutoff(45)`. The GKP state:

| Cutoff | `negativity().value` | `trunc_error()` |
| --- | --- | --- |
| 60 (the former fixed default) | 0.933283 | $2.6\times10^{-5}$ |
| 80 | 0.926781 | $4.2\times10^{-7}$ |
| **89** (`gkp_levels(0.3)`, the derived default) | **0.926206** | $9.4\times10^{-8}$ |
| 110 | 0.925878 | $2.2\times10^{-9}$ |

`trunc_error` improves four orders of magnitude while the volume moves in the third
decimal: at 60 the volume's error was $\approx 7\times10^{-3}$, about 280× the
`trunc_error`. The derived 89 cuts the overstatement from 0.8 % to 0.04 %.

| Method | Returns |
| --- | --- |
| `cutoff` | Levels this state is held on. A property |
| `at_cutoff(d)` | The same state on $d$ levels — the convergence check, and the only guard on a `Volume` |
| `matrix()` | The density matrix |
| `populations()` | $P(n) = \rho_{nn}$ |
| `eigenvalues()` | Spectrum, ascending |
| `photons()` | Mean photon number |
| `purity()` | A `Purity`: `value` is $\mathrm{Tr}(\rho^2)$ **of the truncated state**, with `discarded` and the certified lower bound `least` |
| `entropy()` | Von Neumann entropy in bits |
| `parity()` | $\langle(-1)^{\hat n}\rangle = \pi W(0,0)$ |
| `physical(tol=1e-9)` | Hermitian, unit trace and positive semidefinite |
| `moments()`, `shadow()` | The Gaussian shadow, as a `(mean, cov)` pair of arrays or as a `gaussian.State` |
| `wigner(xs, ps)`, `husimi(xs, ps)` | Phase-space grids |
| `negativity(xs=None, ps=None)` | A `Volume`; both axes default to `window(self)` |
| `non_gaussianity()` | A `Divergence`: $S(\rho_G) - S(\rho)$ in bits, with its truncation error and cutoff |
| `discarded()`, `tail(levels=3)`, `trunc_error()` | The truncation instrumentation |

Array dtypes, shapes and aliasing: [Bulk returns are numpy
arrays](/guide/conventions#arrays). `Volume`, `Divergence` and `Purity` are frozen
records carrying `__float__`.

| `purity()` is the truncated state's purity | |
| --- | --- |
| Mechanism | the constructors renormalise whatever fits under the cutoff, rescaling every retained element by $1/\text{kept}$ and $\mathrm{Tr}(\rho^2)$ by $1/\text{kept}^2$ |
| `Thermal(200)` at cutoff 40 | keeps 18 % of its norm and reports $0.025083$ against the true $1/(1 + 2\times200) = 0.002494$ — ten times high, on a state that passes every physical check |
| Sign | not one-signed: a state whose discarded tail is *purer* than its retained head reads too low |
| `least` | the one certified statement, $\mathrm{Tr}(P\rho P)^2 \le \mathrm{Tr}(\rho^2)$ for the truncating projector $P$, and $\mathrm{Tr}(P\rho P)^2 = \text{kept}^2\times$`value` |
| Above | nothing bounds it usefully — the discarded tail could be pure. `entropy()` and `eigenvalues()` carry the same distortion with no bound at all |

## References

- Spekkens 2008 — PRL **101**, 020401 (2008)
- Genoni and Paris 2010 — PRA **82**, 052341 (2010)
- Mari & Eisert 2012 — PRL **109**, 230503 (2012)
- Veitch et al. 2013 — NJP **15**, 013037 (2013)
- Albarelli et al. 2018 — PRA **98**, 052350 (2018)
