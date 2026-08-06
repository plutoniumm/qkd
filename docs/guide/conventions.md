# Conventions

Enforced by the [`Conventions`](/tests/conventions) exam.

| Convention | qkd |
| --- | --- |
| Trust | an explicit `trusted=` flag, never inferred — [Security](/guide/security#trusted-vs-untrusted-is-a-security-model) |
| Unsupported combination | raises, naming the restriction; never approximated — [refusals](/roadmap#refusals) |
| $\hbar$ | $1$ |
| Quadratures | $\hat x = (\hat a + \hat a^\dagger)/\sqrt2$, $\hat p = (\hat a - \hat a^\dagger)/(i\sqrt2)$, so $[\hat x, \hat p] = i$ |
| Vacuum | $\Delta x^2_{\text{vac}} = \Delta p^2_{\text{vac}} = \tfrac12$, covariance $\tfrac12\mathbb{1}$ |
| Coherent state $\alpha$ | mean $(\sqrt2\,\mathrm{Re}\,\alpha,\ \sqrt2\,\mathrm{Im}\,\alpha)$ |
| Ordering | **xpxp**: $\hat{\mathbf r} = (\hat x_1, \hat p_1, \dots, \hat x_N, \hat p_N)^\top$, mode $m$ at indices $2m$, $2m+1$ |
| Covariance | $V_{ij} = \tfrac12\langle\{\Delta\hat r_i, \Delta\hat r_j\}\rangle$ |

```python
from qkd import gaussian as g

g.Vacuum(1).cov
# array([[0.5, 0. ],
#        [0. , 0.5]])
```

## Ordering and the symplectic form

| | |
| --- | --- |
| Symplectic form | $\Omega = \mathbb{1}_N \otimes \begin{pmatrix}0 & 1\\ -1 & 0\end{pmatrix}$, block-diagonal in xpxp |
| Same $\Omega$ in xxpp | the single global block $\left(\begin{smallmatrix}0 & \mathbb{1}_N\\ -\mathbb{1}_N & 0\end{smallmatrix}\right)$ |
| Gaussian unitary | $V \mapsto S V S^\top$, $\bar{\mathbf r} \mapsto S\bar{\mathbf r}$, with $S\Omega S^\top = \Omega$ |

## The bona fide condition

$V + \tfrac{i}{2}\Omega \succeq 0$, strictly stronger than symmetry plus
positive-definiteness:

```python
import numpy as np

V = np.diag([0.4, 0.4])
omega = np.array([[0.0, 1.0], [-1.0, 0.0]])

np.linalg.eigvalsh(V)                    # array([0.4, 0.4])  -- positive definite
np.linalg.eigvalsh(V + 0.5j * omega)     # array([-0.1,  0.9]) -- not a state
```

| | |
| --- | --- |
| The negative eigenvalue | $\Delta x^2\Delta p^2 \ge 1/4$ violated, $0.4 \times 0.4 = 0.16$ |
| Applied by | `State.physical()`, and the [test harness](/tests/) after every channel and every symplectic map |
| Tolerance | **absolute**, not scaled by the state's own energy |
| Boundary | $V_{xx}V_{pp} = 1/4$ accepted, out to $V_{pp} = 10^6$ |
| Symplectic eigenvalues | $\nu_k \ge 1/2$ (internal units), $= 1/2$ for every mode iff the state is pure |

```python
from qkd import gaussian as g

g.Moments([0.0, 0.0], [[0.25e-6, 0.0], [0.0, 1e6]])   # accepted
g.Moments([0.0, 0.0], [[4.9e-97, 0.0], [0.0, 1e6]])
# ValueError: cov violates the bona fide condition V + i*Omega/2 >= 0
```

## The SNU boundary

$V_{\text{SNU}} = 2\,V_{\text{internal}}$, applied once, between the Gaussian layer
and the key-rate layer.

| Layer | Module | Units | Vacuum |
| --- | --- | --- | --- |
| Gaussian states, channels, sampling | `qkd.gaussian` | internal, $\hbar=1$ | $1/2$ |
| Number-basis states, Wigner, Husimi, negativity | `qkd.fock` | internal, $\hbar=1$ | $1/2$ |
| Excess-noise budget, key rate, security | `qkd.budget`, `qkd.Link`, `qkd.Swap` | SNU | $1$ |

$V_A$, $\xi$, $v_{el}$, $\chi_{\text{line}}$ and every published anchor are SNU;
`State.cov`, `State.spectrum()` and `State.purity()` are internal.
`fock.State.shadow()` needs no conversion.

```python
from qkd import gaussian as g

st = g.Coherent(1.0, 0.0).thermal_loss(0, T=0.5, xi=0.01, ref="input")

st.cov[0, 0]        # 0.5025      internal
2.0 * st.cov[0, 0]  # 1.005       SNU: vacuum 1 plus T*xi = 0.005
```

| Where the factor surfaces | Form |
| --- | --- |
| `thermal_loss` | multiplies the SNU `xi` argument by $0.5$ on the way in |
| `purity()` | $\mu = 1/(2^n\sqrt{\det V})$ internally versus $1/\sqrt{\det V_{\text{SNU}}}$ |
| `entropy()` | converts $\nu_{\text{SNU}} = 2\nu$ before applying $G$ |
| Bona fide, in SNU | $V + i\Omega \succeq 0$, and $\Delta x^2\Delta p^2 \ge 1$ |

## Reference states

| State | Internal ($\hbar=1$, vacuum $\tfrac12$) | SNU (vacuum $1$) |
| --- | --- | --- |
| `Vacuum(n)` | $\tfrac12\mathbb{1}_{2n}$ | $\mathbb{1}_{2n}$ |
| `Coherent(x, p)` | $\tfrac12\mathbb{1}_2$, mean $(x, p)$ | $\mathbb{1}_2$ |
| `Thermal(nbar)` | $(\bar n + \tfrac12)\mathbb{1}_2$ | $(2\bar n + 1)\mathbb{1}_2$ |
| `Squeezed(r)` | $\tfrac12\,\mathrm{diag}(e^{-2r}, e^{+2r})$ | $\mathrm{diag}(e^{-2r}, e^{+2r})$ |
| `Epr(r)` | $\tfrac12\begin{pmatrix}\cosh 2r\,\mathbb{1}_2 & \sinh 2r\,\sigma_z\\ \sinh 2r\,\sigma_z & \cosh 2r\,\mathbb{1}_2\end{pmatrix}$ | $\begin{pmatrix}V\mathbb{1}_2 & \sqrt{V^2-1}\,\sigma_z\\ \sqrt{V^2-1}\,\sigma_z & V\mathbb{1}_2\end{pmatrix}$, $V = \cosh 2r$ |

$\sigma_z = \mathrm{diag}(1, -1)$. The key-rate layer builds the SNU `Epr` at
$V = V_A + 1$: Gaussian-modulated coherent states are equivalent to Alice holding
half of a two-mode squeezed vacuum.

## Excess noise carries a plane

$\xi_{\text{Bob}} = T\,\xi_{\text{Alice}}$.

| | |
| --- | --- |
| qkd's plane | the **channel input** (Alice's side) everywhere, as Lodewyck, Fossier and Leverrier |
| Never defaulted | `thermal_loss` and `q.Channel(xi=...)` require `ref=` — see [the `ref=` plane](/guide/gaussian#thermal-loss-and-the-required-ref-plane) |

## Bulk returns are numpy arrays {#arrays}

Bulk returns cross the PyO3 boundary as `numpy.ndarray`, so `numpy` is a hard
runtime dependency. None aliases its source.

| Call | dtype | Shape | Buffer |
| --- | --- | --- | --- |
| `gaussian.State.mean` | `float64` | `(2n,)` | copied from the state |
| `gaussian.State.cov` | `float64` | `(2n, 2n)` | copied from the state, reshaped from flat |
| `gaussian.State.homodyne(...)` | `float64` | `(shots,)` | built for the call |
| `gaussian.State.heterodyne(...)` | `float64` | `(shots, 2)` | built for the call |
| `fock.State.populations()` | `float64` | `(cutoff,)` | built for the call |
| `fock.State.eigenvalues()` | `float64` | `(cutoff,)` | built for the call |
| `fock.State.wigner(xs, ps)`, `.husimi(xs, ps)` | `float64` | `(len(ps), len(xs))` | built for the call |
| `fock.State.matrix()` | `complex128` | `(cutoff, cutoff)` | copied from the state |
| `_core.SimOut.frames_x`, `.frames_p` | `float64` | `(n_used,)` | copied from the run |
| `_core.cpu_words(...)`, `_core.gpu_words(...)` | `uint32` | `(4n,)` | built for the call |

`cpu_words` and `gpu_words` are Threefry-4×32-20 words; `uint32` arithmetic wraps
at $2^{32}$. A dtype is the representation, not the precision computed at — see
[dispatch on precision, not
device](/architecture#compute-dispatch-on-precision-not-device).

## Where the literature disagrees

| Point | qkd | Alternative in the literature |
| --- | --- | --- |
| Vacuum variance | $1/2$ internally, $1$ in the QKD layer | $\hbar = 2$ conventions put it at $1$ throughout |
| Quadrature ordering | xpxp | xxpp |
| Plane of $\xi$ | channel input | channel output (Laudenbach 2018, Eq. (4.8)); divide by $T$ on import |
| Heterodyne 3 dB penalty | inside $\chi_{\text{het}}$ (Lodewyck/Fossier) | $T \to T/2$, $\xi \to \xi/2$ per quadrature (Laudenbach Eq. (5.10)) |
| Entropy function | $G(x) = (x{+}1)\log_2(x{+}1) - x\log_2 x$ at $(\nu{-}1)/2$ | $g(\nu)$ applied to $\nu$ directly; identical, $g(\nu) = G((\nu{-}1)/2)$ |
