# Swap and network topologies

Alice and Bob both *send*; an untrusted station between them measures and
announces, and the announcement correlates the senders. Neither detects anything.

| | |
| --- | --- |
| Bought | detector side channels leave the threat model: the proof never trusts the relay, so a relay detector has no `trusted=` flag ([the split](/guide/security#trusted-vs-untrusted-is-a-security-model)) |
| Paid | a two-arm channel and a harsher loss budget: **relay position matters, and the symmetric position is not the optimum** |

## Why it is called a swap {#why-swap}

Preparing and sending a state is equivalent to measuring half an entangled pair, so
this topology is an EPR source per end with a Bell measurement between them:
entanglement swapping, the time reversal of BBM92. Lo, Curty and Qi introduced it
so, on the time-reversed EPR idea of Biham, Huttner and Mor and of Inamori.

| | |
| --- | --- |
| Why the middle may be untrusted | a Bell measurement projects onto a **joint** property and reveals nothing about either input |
| Why not "relay" | **no photon travels from Alice to Bob.** Both terminate at the station, the joint measurement consumes them, and only a classical announcement leaves |
| **Caveat** | **no entanglement exists anywhere**: Alice and Bob prepare Gaussian-modulated coherent states. Swapping is the **virtual protocol of the security proof**; the hardware mixes the arrivals on a balanced beamsplitter and homodynes conjugate quadratures |

## `q.Swap` {#swap}

`Swap` resolves components into a plan, refuses what it cannot compute, and returns
a `SwapResult`. The qubit midpoint, `q.BellAnalyser` with `q.TestBasisBound`:
[MDI-BB84](/guide/protocols#mdi-bb84).

```python
import qkd as q

res = q.Swap(
    alice=q.Sender(modulation=q.GaussianModulation(v_a=1e5)),
    bob=q.Sender(modulation=q.GaussianModulation(v_a=1e5)),
    relay=q.Relay(bell=q.BellDetector(eta=1.0, v_el=0.0)),
    channels=(q.Channel(T=1.0), q.Channel(T=0.1)),
    security=q.Asymptotic(beta=1.0),
).run()

res.key_rate     # 0.077336  bits per relay use
res.chi          # 22.0      equivalent noise
```

| | |
| --- | --- |
| `run()` signature | **no `symbols`, no `seed`**: a closed form over the configured hardware |
| $V_A$ cap | $10^5$. `res.key_rate` takes no $V_A$, but `res.attack` evaluates the post-relay covariance at the configured one, and past $10^5$ its conditional blocks are differences of numbers that have spent their mantissa |
| Where the cap raises | at `run()`: `GaussianModulation` is shared with the point-to-point path, which has no cap |

```python
q.Sender(modulation=q.GaussianModulation(v_a=1e6))   # constructs fine
# ... Swap(...).run()
# ValueError: va must be <= 1e5, got 1000000: the conditional blocks cancel
#             catastrophically and the rate exceeds its asymptotic bound. Use
#             cvmdi_rate for the large-modulation limit
```

### The component tree

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `Swap.alice`, `.bob` | — | **required** | Two `Sender`s. Both send; neither measures. The rate is not symmetric under exchanging them. |
| `Swap.relay` | — | **required** | The untrusted `Relay`. |
| `Swap.channels` | — | **required** | `(alice's arm, bob's arm)`, each a `Channel` or `Fiber`. Exactly two. |
| `Swap.security` | — | `None` | On a `q.BellDetector` midpoint, `Asymptotic` or `q.TwoModeBound(block=q.GaussianBlock(…))`; on a `q.BellAnalyser` one, `q.TestBasisBound`. `None` always fails at `run()`: no security model is safe to fall back to. |
| `Swap.environment` | — | `None` | A `CorrelatedEnvironment`. `None` is the independent pair. |
| `Sender.modulation` | — | **required** | `GaussianModulation` beside a `BellDetector`; a basis- or polarisation-keyed modulation beside a `BellAnalyser`. |
| `Sender.pulse` | s | `None` | Intensity FWHM of this sender's temporal mode. **Refused on the covariance path**, which has no two-source mode model. Beside a `BellAnalyser`, widths on both senders derive `misalign_test`; one width alone, or widths beside a pinned `misalign_test`, is refused. |
| `Relay.bell` | — | **required** | A `BellDetector` or a `BellAnalyser`. |
| `BellDetector.eta` | — | **required** | Relay detection efficiency $\eta \in (0, 1]$. |
| `BellDetector.v_el` | SNU | **required** | Relay electronic noise $v_{el} \ge 0$. |
| `CorrelatedEnvironment.x`, `.p` | SNU | **required** | Eve's cross-arm environment correlations $g$, $g'$. No neutral default exists; the uncertainty principle caps the pair jointly, and a combination describing no state is refused. |

### What `run()` returns

| Field | Description |
| --- | --- |
| `key_rate` | **The bound**, bits per relay use: the worst attack compatible with the observed $\chi$, in the large-modulation limit at the **Shannon** reconciliation limit, so the configured $\beta$ does not reach it. Clamped at $\ge 0$. |
| `chi` | The equivalent noise the two arms carry. |
| `floor` | What two **pure-loss** arms would carry; see [below](#chi). |
| `least` | The domain bound, where the noise term vanishes. $\chi \le \text{least}$ describes no state and raises. |
| `attack` | `Attack(i_ab, chi_e, key_rate)` at the configured finite $V_A$ and $\beta$, **unclamped**. |
| `explain` | The labelled plan, with no `stages` key: a relay run has one stage. |

**Only `SwapResult.key_rate` is clamped.** `explain()["key_bound"]` and
`["key_raw"]` carry the unclamped bound; `Attack.key_rate`
(`explain()["key_attack"]`) is never clamped.

## The continuous-variable relay {#cvmdi}

Each sender sends a Gaussian-modulated coherent state down its own thermal-loss arm,
transmissivity $\tau_a$ or $\tau_b$. The relay mixes the arrivals on a balanced
beamsplitter, homodynes $x$ on one output and $p$ on the other, and broadcasts both
outcomes, which *condition* the joint state of the senders' kept modes.

| | |
| --- | --- |
| Implementation | two thermal-loss channels, a beamsplitter and two homodynes, conditioned through the Schur complement of [`State.condition`](/guide/gaussian#conditioning); no bespoke relay formula |
| Roles | the **encoder**'s variable becomes the key; the **decoder** combines his variable with the announcement to reconstruct it. Reconciliation direction is not symmetric |

### Everything depends on one number {#chi}

In the large-modulation limit the rate is a closed form in the two transmissivities
and one **equivalent noise** $\chi$.

| | |
| --- | --- |
| $\chi_{\text{loss}}$, two pure-loss arms | $\dfrac{2(\tau_a + \tau_b)}{\tau_a \tau_b}$ |
| Excess noise enters as | each arm's environment variance $\omega = 1 + \tau\xi/(1-\tau)$, $\xi$ at the **channel input**. A lossless arm has no environment mode and no excess noise |
| `res.floor` | $\chi_{\text{loss}}$ — a **reference point, not a lower bound**. Eve may correlate the arms' environments to *help* the Bell measurement, driving $\chi$ below its pure-loss value within the uncertainty principle |
| `res.least` | $\dfrac{(\tau_a+\tau_b)^2}{\tau_a\tau_b}$, where the noise term reaches zero; below it no covariance matrix exists and inputs raise. $4$ for equal arms at any loss, the pole of the symmetric branch's logarithm |
| Relay imperfections | **folded into the two arms**: $\tau \to \eta\tau$, $\omega \to \dfrac{\eta(1-\tau)\omega + (1-\eta) + v_{el}}{1 - \eta\tau}$ |
| Why the fold is exact | equal loss on every mode commutes with any passive linear network, so detector efficiency after the balanced coupler is a loss on both arms before it |

### Relay position is the result {#position}

Total transmittance fixed at $\tau_a\tau_b = 0.1$, relay sliding along it:
pure-loss arms, perfect relay detector, $V_A = 10^5$, **unclamped**
`res.attack.key_rate` (`res.key_rate` is $0$ on every row but the last):

| Relay position | $\tau_a$ (encoder arm) | Rate, bits per relay use |
| --- | --- | --- |
| at the decoder | $0.1$ | $-4.6123$ |
| | $0.2$ | $-3.5727$ |
| **symmetric** | $\sqrt{0.1} = 0.3162$ | $-2.8120$ |
| | $0.5$ | $-1.9631$ |
| | $0.9$ | $-0.4649$ |
| **at the encoder** | $1.0$ | $+0.0773$ |

The rate climbs monotonically and crosses zero only at the last step: the symmetric
relay is $2.9$ bits *below* the encoder-side one. Exchanging the arms at fixed total
loss costs key when the longer arm moves to the encoder's side.

Zero crossings against Pirandola et al., Nat. Photon. **9**, 397 (2015),
arXiv:1312.4104, are graded at
[Validation](/guide/validation#the-continuous-variable-rows). With the decoder's arm
lossless the rate is $\log_2\!\big(\tau/((1-\tau)e)\big)$, zero at
$\tau = e/(1+e)$, asserted to $10^{-6}$; the symmetric row, read from prose, to
$5\times10^{-3}$. With the relay adjacent to the encoder there is no crossing: the
rate stays positive past 300 km.

### What `res.attack` is, and is not {#point}

| | `res.key_rate` — `cvmdi_rate` | `res.attack` — `cvmdi_point` |
| --- | --- | --- |
| Is | a **security bound**: the worst attack compatible with the observed $\chi$ | **one specified attack**: the two environment variances and the correlations $g$, $g'$ the link was configured with |
| Modulation | the large-modulation limit; takes no $V_A$ | the finite $V_A$ configured |
| Reconciliation | the **Shannon** limit, so the configured $\beta$ does not reach it | the configured $\beta$, as $\text{key} = \beta I_{AB} - \chi_E$ |
| Returns | a rate | $(I_{AB}, \chi_E, \text{key})$ |

| | |
| --- | --- |
| Why unclamped | most of the $(\tau_a, \tau_b)$ plane cannot distil, and the depth of the deficit measures how far a position is from viable |
| Join | raising the modulation drives the point onto the asymptote, and is tested |
| `attack.key_rate` above `key_rate` | possible: `cvmdi_rate` reads the observed noise as the worst attack producing it, `cvmdi_point` prices one named Eve at finite $V_A$. A correlated attack of the helping sign hands the relay an entanglement resource. **That it also exceeds the repeaterless capacity is UNREPRODUCED**: `test_relay_bounded` finds neither rate above capacity in any of the 4,518 configurations the engine accepts out of 40,000 drawn, closest approach 0.41, and a 200,000-draw search maximising $R - \mathrm{PLOB}$ peaks at $-0.33$. `b92_ceiling` and `cow_rate` escape PLOB, pinned in `test/capacity.py`'s `Escapes`; neither is CV-MDI |
| Wrong sign | fights the correlations the Bell measurement establishes and costs more key than two independent entangling cloners, so the equivalent noise takes $g$ and $g'$, not only $\omega_a$, $\omega_b$ |
| PLOB | every pure-loss configuration sits strictly under $-\log_2(1-T)$, Pirandola et al., Nat. Commun. **8**, 15043 (2017) |

## Where it is not {#gaps}

Asymmetric modulation and a symbol-level relay: [roadmap](/roadmap#refusals).

Twin-field, phase-matching and sending-or-not-sending: [permanent exclusion](/roadmap#exclusions).

Graded numbers: [Validation](/guide/validation#the-scoreboard).

## `q.Network` {#network}

A graph of key-holding nodes. Every vertex is a `q.Node` holding key **in the
clear**; every edge is a `q.Hop` carrying one `q.Link` or `q.Swap`.

```python
import qkd as q

net = q.Network(
    nodes={"IITM": q.Node(), "ERNET": q.Node(), "NIC": q.Node()},
    edges=[
        q.Hop(ends=("IITM", "ERNET"), link=cow, clock=1e9),
        q.Hop(ends=("ERNET", "NIC"),  link=dps, clock=1e9),
    ],
)

res = net.run()
res.rates[("IITM", "ERNET")]      # bits per second
res.rates[("ERNET", "IITM")]      # the same; pairs read either way round

route = res.route("IITM", "NIC")
route.hops        # (("IITM", "ERNET"), ("ERNET", "NIC"))
route.trusts      # ("ERNET",)
route.bottleneck  # bits per second, the minimum over the hops
route.security    # "trusted-node key relay through ERNET; a key-management
                  #  throughput, not a quantum bound"
```

### An untrusted station is not a vertex {#not-a-vertex}

**A `q.Swap` is one edge.** Its untrusted midpoint is *interior* to that edge and
in no vertex set, segment key or route: the data structure cannot express it.

| | |
| --- | --- |
| Why | untrusted relays **do not compose**: chaining two swaps needs quantum memory or a repeater |
| What it buys | detector-side-channel immunity for **one** segment, at a worse loss budget. No added reach |
| What it costs in the model | two swap edges through one building each carry their own `q.Relay`; one detector pair serving two segments is not expressible |
| Consequence for routing | a swap segment is one edge between its senders; the router sees neither arm and no name to route *through* |

### There is no end-to-end key rate {#no-key-rate}

Neither `NetworkResult` nor `Route` has one; asking raises with the reason:

```python
res.key_rate
# AttributeError: a network has no key rate. Per-segment rates are in
#   res.segments and res.rates; a path number is res.route(a, b).bottleneck,
#   which is a KEY-MANAGEMENT THROUGHPUT and not a security bound, because no
#   quantum bound spans a trusted node
```

`route.bottleneck` is the minimum hop rate in bits per second: hop-by-hop key relay
spends one local key bit per relayed bit. **`route.trusts` names every intermediate
node, each of which held the end-to-end key in the clear.** It is empty only on a
single-hop route, the one case where the number is a quantum claim.

### The component tree {#network-tree}

| Parameter | Unit | Default | Description |
| --- | --- | --- | --- |
| `Network.nodes` | — | **required** | `{name: q.Node}`; the key is the name. A `q.Relay` is refused: an untrusted station holds no key |
| `Network.edges` | — | **required** | A list of `q.Hop`, not a dict keyed by pair, so no parallel `clocks=` mapping can disagree |
| `Hop.ends` | — | **required** | Two **distinct** node names. `(transmitter, receiver)` for a `q.Link`; `(alice, bob)` for a `q.Swap`, following its encoder convention |
| `Hop.link` | — | **required** | A `q.Link` or a `q.Swap`, verbatim, run and multiplied by the clock: a network reports what the standalone payload reports |
| `Hop.clock` | Hz | **required** | Symbols per second on the quadrature families, emitted pulses per second on the click ones |
| `Node` | — | — | Carries nothing; physical descriptions live on the edge |

**The clock is mandatory.** `LinkResult.key_rate` is bits per *symbol* on Gaussian
modulation and $M$-PSK, per emitted *pulse* on the click families; `SwapResult.key_rate`
is per *swap use*. A minimum over a mixed path would mix three units. It cannot come
from `q.Alice.symbol_rate`: `q.Link` refuses an `Alice` on the intensity-keyed path.

### What `Network.run()` returns {#network-returns}

| Field | Description |
| --- | --- |
| `segments` | `{(a, b): LinkResult \| SwapResult}`, per-shot rates in each payload's own unit. Reads either way round |
| `rates` | `{(a, b): bits per second}`, the only unit that compares across families |
| `nodes` | `{name: NodeReport}` with `degree`, `segments` and `total`, the sum of the incident segments in bits per second |
| `explain` | the labelled plan, with `trust`, `units`, `end_to_end` and `contention` rows always present |

`run(symbols=…, seed=…)` reaches the `q.Link` payloads alone; a `q.Swap` is a
closed form and takes neither.

### Routing {#routing}

`res.route(a, b)` returns the **widest** path — largest bottleneck, ties broken by
fewest hops and so fewest trusted nodes — key relay being limited by its narrowest
hop.

| Refused | |
| --- | --- |
| An unknown endpoint | names the nodes the network does have |
| A route from a node to itself | a route joins two distinct nodes |
| A disconnected pair | says so, noting that an untrusted swap station is never a vertex |

**Contention is not modelled.** A node relaying two concurrent routes splits its key
pool, so two `route()` calls on one result cannot both be achieved.

### What the graph layer does not model {#network-gaps}

| Absent | |
| --- | --- |
| Authentication key consumption | hop-by-hop relay spends key authenticating every round, so delivered key is strictly below `bottleneck` |
| Key pool dynamics | no buffering, scheduling, refill or contention |
| The relay operation itself | the XOR of local keys, its failure modes, its latency |
| Time | no availability, no outage, no drift |
| Parallel segments between one pair | refused: their key pools would add and nothing here tracks that |
| Point to multipoint | a receiver shared by several transmitters; hardware lives on the edge |
| Wavelength | no component carries one, so per-edge Raman crosstalk is not derived from a channel plan |
| Multi-path key splitting | `route()` returns one path |
| The control plane | key management APIs, SDN control, and everything above the physical layer |
