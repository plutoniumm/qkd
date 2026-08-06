# Usage

One runnable block per protocol family and per layer. Rules every block obeys:
[Conventions](/guide/conventions). API index: `qkd.__all__`.

## Choosing an entry point

Entry classes: [Protocol coverage](/guide/protocols#reach). Valid and refused
combinations: [What runs today](/guide/link#what-runs-today).

## Gaussian-modulation CV-QKD

Lodewyck et al. 2007, 25 km all-fibre homodyne, every number pinned:

```python
import qkd as q

def rate(security=q.Asymptotic(beta=0.898), trusted=True):
    return q.Link(
        modulation=q.GaussianModulation(v_a=18.5),
        channel=q.Channel(T=0.302, xi=0.005, ref="input"),
        bob=q.Bob(detector=q.Homodyne(eta=0.606, v_el=0.041, trusted=trusted)),
        security=security,
    ).run()

res = rate()
res.key_rate    # 0.035154090
res.i_ab        # 1.043649649
res.chi_be      # 0.902043295
```

`q.FiniteSize` and `trusted=False`:

```python
rate(q.FiniteSize(beta=0.898, n=1e9)).key_rate    # 0.015508724
rate(q.FiniteSize(beta=0.898, n=1e6)).key_rate    # 0.0
rate(trusted=False).key_rate                      # 0.0
```

## Gaussian modulation with the hardware in the loop

Hardware example: [First run](/getting-started#first-run-a-key-rate-you-did-not-set).
Reference: [DSP in the loop](/guide/link#dsp-in-the-loop). The blocks below share
one link over a pinned channel.

```python
def dsp(security):
    return q.Link(
        modulation=q.GaussianModulation(v_a=4.0),
        channel=q.Channel(T=0.5, xi=0.01, ref="input"),
        alice=q.Alice(laser=q.Laser(), pilots=q.Pilots(), symbol_rate=100e6),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1), lo=q.LocalLO()),
        dsp=q.DSP(block=32),
        security=security,
    )

res = dsp(q.FiniteSize(beta=0.95, n=1e9)).run(symbols=200_000, seed=7)

res.dsp.v_err          # 0.0075832    rad^2, feeds the phase term
res.dsp.pilot_snr      # 105.0021
res.dsp.cfo            # 1959.745     Hz, fitted
res.dsp.removed        # True
res.key_rate           # 0.0240635
```

### Estimates versus the oracle

```python
res.est.T             # 0.4989213     estimated
res.est.xi            # 0.0242803
res.est.t_min         # 0.4851647     worst channel at eps
res.est.xi_max        # 0.1346164
res.oracle.T          # 0.5           what was actually set
res.oracle.xi         # 0.0405243
res.oracle.key_rate   # 0.1254689     the circular number
res.key_rate          # 0.0240635     what they can claim
```

[More](/guide/link#estimates-versus-the-oracle)

### Measure once, claim many times

Gaussian modulation only;
[what may differ between the two calls](/guide/link#measure-once-claim-many-times).

```python
frames = dsp(q.Asymptotic()).measure(symbols=200_000, seed=7)
frames.dsp.v_err                                             # 0.0075832

dsp(q.Asymptotic(beta=0.95)).claim(frames).key_rate          # 0.1460320
dsp(q.FiniteSize(beta=0.95, n=1e9)).claim(frames).key_rate   # 0.0240635
dsp(q.FiniteSize(beta=0.95, n=1e7)).claim(frames).key_rate   # 0.0158162
```

## *M*-PSK discrete modulation

`q.Asymptotic` reaches the analytic bound; `q.CertifiedBound(cutoff=…).certify(…)`
and `certify_trusted()` the certified one:
[the relative-entropy proof](/guide/protocols#relent).

```python
res = q.Link(
    modulation=q.PhaseShiftKeying(states=4, alpha=0.4),
    channel=q.Channel(T=0.5, xi=0.01, ref="input"),
    bob=q.Bob(detector=q.Heterodyne(eta=1.0, v_el=0.0, trusted=False)),
    security=q.Asymptotic(beta=0.95),
).run()

res.key_rate    # 0.0084865
res.i_ab        # 0.1107575    the constellation's own I(X;Y)
```

```python
psk = q.PhaseShiftKeying(states=4, alpha=0.4)

round(psk.v_a, 6)                # 0.32     2*alpha^2, SNU
psk.bits                         # 2.0      log2(states)
len(psk.constellation())         # 4        complex amplitudes
psk.information(T=0.5, xi=0.01)  # (0.1107575, 0.1107648)  exact, gaussian-substitute
```

## BB84 with weak coherent pulses

Six-state and SARG04 reuse this `bb84` helper.

```python
def bb84(modulation, **security):
    return q.Link(
        modulation=modulation,
        channel=q.Fiber(length=25.0, alpha=0.21),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
            receiver=q.BasisAnalyser(misalign=0.033),
        ),
        security=q.SplittingAttack(f=1.22, **security),
    )

decoy = q.Decoy(intensities=(0.48, 0.1, 0.0))     # probs=None: equal three-way split
res = bb84(q.BasisKeying(decoy=decoy)).run()

res.key_rate    # 0.00066698    bit per emitted pulse
res.qber        # 0.03312354
res.p_click     # 0.00642937
```

### Finite key length

Quote these numbers with the caveat in
[Finite key for BB84-WCP](/guide/security#finitekey).

```python
def fin(n):
    return bb84(q.BasisKeying(decoy=decoy), block=q.KeyBlock(n=n)).run()

fin(1e12).key_rate     # 0.000155715    fin(1e12).key_length -> 155714902.0
fin(1e10).key_rate     # 0.000139456    fin(1e10).key_length ->   1394562.0
fin(1e7).key_rate      # 0.0            fin(1e7).key_length  ->         0.0
fin(1e10).s1           # 4115374.14     single-photon detections, lower bound
fin(1e10).phi          # 0.04309986     phase error rate, upper bound
```

## Six-state

[Protocol coverage](/guide/protocols#sixstate)

```python
res = bb84(q.BasisKeying(decoy=decoy, bases=3)).run()

res.key_rate    # 0.00052213    against BB84's 0.00066698 on the same hardware
res.qber        # 0.03312354    unchanged: the hardware did not move
```

## SARG04

[Protocol coverage](/guide/protocols#sarg)

```python
link = bb84(q.BasisKeying(
    decoy=q.Decoy(intensities=(0.2, 0.05, 0.0)),
    announce="pair",
))
res = link.run()

res.key_rate                          # 5.0980e-05
res.qber                              # 0.06243243   nearly twice BB84's
link.explain()["sift"]                # {'value': 0.2665, 'label': 'derived'}
```

## Polarisation-keyed BB84

[The reference frame](/guide/link#the-polarisation-reference-frame)

```python
res = q.Link(
    modulation=q.PolarisationKeying(
        decoy=decoy,
        frame=q.ReferenceFrame(tracking="free", rate=0.02, interval=60.0),
    ),
    channel=q.Fiber(length=50.0),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.1, dark=1e-6),
        receiver=q.BasisAnalyser(misalign=0.02),
    ),
    security=q.SplittingAttack(f=1.22),
).run()

res.key_rate    # 0.00053115
res.qber        # 0.03153360    the analyser's 0.02, plus the frame
```

## B92

[Protocol coverage](/guide/protocols#b92)

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

res.key_rate                          # 0.06125854
res.qber                              # 0.0
res.explain["e_phase"]["value"]       # 0.19469133   label 'derived'
res.explain["gain"]["value"]          # 0.21210147   label 'derived'

q.TwoStateKeying(mu=0.23).overlap    # 0.63128365   <alpha|-alpha>
```

## Differential phase shift

[Protocol coverage](/guide/protocols#dps)

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
).run(symbols=200_000, seed=1)

res.key_rate      # 0.00198769
res.qber          # 0.00500626    an output, not an input
res.visibility    # 0.97996924
res.clicks        # 800
res.sifted        # 799
res.doubles       # 1             kept and coin-flipped, not discarded
res.slots         # 199999
```

## Coherent one way

[Protocol coverage](/guide/protocols#cow)

```python
res = q.Link(
    modulation=q.IntensityKeying(mu=0.5, decoy_frac=0.1),
    channel=q.Fiber(length=100.0, alpha=0.2),
    bob=q.Bob(
        detector=q.ClickDetector(eta=0.1, dark=1e-6),
        receiver=q.CoherenceMonitor(split=0.1, misalign=0.02),
    ),
    security=q.PhaseBound(e_phase=0.15, f=1.1),
).run()

res.key_rate    # 9.0139e-05
res.qber        # 0.02212441
res.p_click     # 0.00040671
```

## BBM92 over a photon-pair source

`channels` states the source position. [Protocol coverage](/guide/protocols#pairs)

```python
def pair(source, security):
    return q.PairLink(
        source=source,
        detectors=(q.ClickDetector(eta=0.2, dark=1e-6),) * 2,
        channels=(q.Fiber(length=10.0), q.Fiber(length=10.0)),
        security=security,
    ).run()

res = pair(q.PairSource(brightness=0.01, rate=100e6), q.SymmetryBound(f=1.22))

res.key_rate     # 7.3611e-05    bit per pump pulse
res.per_second   # 7361.14       key_rate * source.rate
res.gain         # 0.00032568    coincidence gain
res.qber         # 0.04099233
res.e_phase      # 0.04099233    = qber: basis symmetry, asymptotically
res.chsh         # 2.59653952    the violation the same state would show
res.eta_a        # 0.12619147
```

`q.PairSource(brightness=None)` optimises the brightness:

```python
res = pair(q.PairSource(rate=100e6), q.SymmetryBound(f=1.22))

res.brightness   # 0.04430096    pairs per pump pulse per mode pair
res.key_rate     # 0.00017531    2.4x the pinned 0.01 above
```

## E91, priced by a CHSH violation

`source=` has no default. [Protocol coverage](/guide/protocols#e91)

```python
res = pair(q.PairSource(brightness=0.01, rate=100e6),
           q.ViolationBound(s=2.7, source="measured"))

res.key_rate    # 3.0929e-05    against BBM92's 7.3611e-05 on the same hardware
res.s           # 2.7
res.chsh        # 2.59653952
res.e_phase     # None          nothing on this branch is a phase error
```

## CV-MDI

[What `res.attack` is](/guide/relay#point);
[`chi`, `floor` and `least`](/guide/relay#chi).

```python
def swap(channels, environment=None):       # exactly two channels
    return q.Swap(
        alice=q.Sender(modulation=q.GaussianModulation(v_a=1e5)),
        bob=q.Sender(modulation=q.GaussianModulation(v_a=1e5)),
        relay=q.Relay(bell=q.BellDetector(eta=1.0, v_el=0.0)),
        channels=channels,
        security=q.Asymptotic(beta=1.0),
        environment=environment,            # q.CorrelatedEnvironment
    ).run()

res = swap((q.Channel(T=1.0), q.Channel(T=0.1)))

res.key_rate            # 0.07733589    bits per relay use, the bound
res.chi                 # 22.0          what the arms carry
res.floor               # 22.0          what two pure-loss arms would carry
res.least               # 12.1          below this, no state exists
res.attack.key_rate     # 0.07733319    one named Eve, unclamped, NOT the bound
```

Correlated environments, capped by the uncertainty principle:

```python
noisy = (q.Channel(T=0.9, xi=0.05, ref="input"),
         q.Channel(T=0.5, xi=0.05, ref="input"))
helped = swap(noisy, q.CorrelatedEnvironment(x=0.3, p=-0.3))

swap(noisy).chi     # 6.44         above the floor: independent cloners cost noise
helped.chi          # 6.02260064   below it: Eve donating entanglement
helped.key_rate     # 0.00954387   and the observable rate rises with it

swap(noisy, q.CorrelatedEnvironment(x=0.9, p=-0.9))
# ValueError: g = 0.9, gp = -0.9 violate the uncertainty principle for Eve's
#             environment pair: least symplectic eigenvalue 0.6674675786448736 < 1
```

## MDI-BB84 {#mdi-bb84}

Both senders need `sift=1.0`. `misalign` and `misalign_test` are
[different physics](/guide/protocols#mdi-bases).

```python
sender = q.Sender(
    modulation=q.BasisKeying(decoy=q.Decoy((0.5, 0.1, 0.0)), sift=1.0)
)

res = q.Swap(
    alice=sender,
    bob=sender,
    relay=q.Relay(
        bell=q.BellAnalyser(
            eta=0.145, dark=6.02e-6, misalign=0.015, misalign_test=0.015
        )
    ),
    channels=(q.Fiber(length=25.0, alpha=0.2),) * 2,
    security=q.TestBasisBound(f=1.16),
).run()

res.key_rate    # 9.14295489524867e-06   bit per pulse pair
res.p_click     # 0.0002544613731596217  key-basis gain at the signal setting
res.qber        # 0.01602283621399531    key-basis QBER
res.y1          # 0.0009047183333000796  the single-pair yield, bounded
res.e1          # 0.10025112854983996    its TEST-basis error rate
```

```python
q.Swap(..., security=q.Asymptotic()).run()
# NotImplementedError: a Bell-analysed midpoint takes q.TestBasisBound security:
#     the phase error rides on the single-photon-pair error rate of the TEST
#     basis, measured through the decoy grid. The finite-key form is a slot
#     on that same class: q.TestBasisBound(block=q.RelayBlock(...))

q.BasisKeying(decoy=q.Decoy((0.5, 0.1, 0.0)))     # no sift=
# ValueError: the midpoint rate is asymptotic in the biased-basis limit, where
#     the key-basis gain already counts key-basis rounds only, so a sifting
#     factor here would charge the same sifting twice. Pass
#     q.BasisKeying(..., sift=1.0)
```

Finite key length, `q.TestBasisBound(block=q.RelayBlock(…))`:
[what it refuses](/guide/protocols#mdi-refusals);
[`q.RelayBlock` is not a `q.KeyBlock`](/guide/security#asymptotic).

## Networks of trusted nodes

A route number is a throughput over trusted nodes, never a bound:
[`q.Network`](/guide/relay#network).

```python
def cow(db):
    return q.Link(
        modulation=q.IntensityKeying(mu=0.5, decoy_frac=0.1),
        channel=q.Fiber(T=10.0 ** (-db / 10.0)),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.1, dark=1e-6),
            receiver=q.CoherenceMonitor(split=0.1, misalign=0.02),
        ),
        security=q.PhaseBound(e_phase=0.15, f=1.1),
    )

net = q.Network(
    nodes={"alice": q.Node(), "middle": q.Node(), "bob": q.Node()},
    edges=[
        q.Hop(ends=("alice", "middle"), link=cow(6.0), clock=1e9),
        q.Hop(ends=("middle", "bob"), link=cow(10.0), clock=1e9),
    ],
)
res = net.run()
route = res.route("alice", "bob")

route.hops         # (('alice', 'middle'), ('middle', 'bob'))
route.trusts       # ('middle',)
route.bottleneck   # 942985.36    bit/s, the minimum over the hops
route.security     # 'trusted-node key relay through middle; a key-management
                   #  throughput, not a quantum bound'
```

## Components

Every parameter with its unit and default: [Protocol layer](/guide/link#component-reference).
`alpha=0.2` dB/km is a bare-fibre datasheet maximum; cabled ITU-T G.652 runs 0.275
to 0.35.

The same `xi`, two planes:

```python
def xi_seen(ref):
    return q.Link(
        modulation=q.GaussianModulation(v_a=18.5),
        channel=q.Channel(T=0.302, xi=0.005, ref=ref),
        bob=q.Bob(detector=q.Homodyne(eta=0.606, v_el=0.041)),
    ).explain()["xi_input"]["value"]

xi_seen("input")     # 0.005
xi_seen("output")    # 0.01655629
```

Security component defaults:

```python
q.Asymptotic(beta=0.95)
q.FiniteSize(beta=0.95, n=1e9, eps=1e-10, pe_fraction=0.5, fer=None)
q.IndividualAttack(qber=None, f=1.16)          # qber=x pins it, None simulates
q.SplittingAttack(f=1.22, block=None)          # block=q.KeyBlock(...) -> finite
q.KeyBlock(n=1e10, eps_sec=1e-10, eps_cor=1e-15, fer=None)
q.PhaseBound(e_phase=0.15, f=1.1)              # required, no safe default
q.DiscriminationBound(e_phase=None, f=1.16)    # e_phase=None derives it
q.SymmetryBound(e_phase=None, f=1.22, sift=0.5)
q.ViolationBound(s=2.7, source="measured", f=1.22, sift=2/9)
q.TestBasisBound(f=1.16)
```

Impairment descriptors on a link: [Reaching a `Link`](/guide/impairments#in-a-link).

## `LinkResult` and `explain()`

[Every field](/guide/link#the-result-object);
[`explain()` labels](/guide/link#pinning);
[`res.budget.at(plane)`](/guide/budget#the-four-planes).

## `qkd.gaussian` {#qkd-gaussian}

[The full layer](/guide/gaussian)

```python
import numpy as np
from qkd import gaussian as g

st = (g.Vacuum(2)                # also Coherent, Thermal, Squeezed, Epr, Moments
      .squeeze(0, r=0.8)
      .bs(0, 1, t=0.5)
      .rotate(1, theta=0.3)
      .displace(0, x=0.5, p=0.0)
      .thermal_loss(0, T=0.5, xi=0.01, ref="input"))

st.cov.shape    # (4, 4)
st.physical()   # True     bona fide: V + i*Omega/2 >= 0

st.homodyne(0, angle=0.0, shots=100_000, seed=3).shape   # (100000,)   float64
st.heterodyne(1, shots=1000, seed=1).shape               # (1000, 2)   (x, p)
st.condition(1, angle=0.0, outcome=0.3).n_modes          # 1
g.Epr(1.0).keep([0]).n_modes                             # 1   in the order asked
g.Epr(1.0).drop([0]).n_modes                             # 1   the complement

e = g.Epr(1.0)
round(e.entropy(), 9)     # 0.0        bits, global: pure two-mode squeezed vacuum
e.keep([0]).entropy()     # 2.3369093  entanglement entropy
e.spectrum()              # [0.5 0.5]  symplectic eigenvalues, vacuum = 0.5
e.negativity()            # 0.0        Hudson: zero for every Gaussian state

xs = np.linspace(-3.0, 3.0, 121)                # W[i, j] = W(xs[j], ps[i])
g.Coherent(1.0, 0.0).wigner(0, xs, xs).max()    # 0.31830989   1/pi
g.Coherent(1.0, 0.0).husimi(0, xs, xs).max()    # 0.15915494   1/(2 pi)

a, b = g.Coherent(1.0, 0.0), g.Coherent(0.0, 1.0)
a.fidelity(b)          # 0.36787944   SQUARED
a.overlap(b)           # 0.36787944   Tr(rho sigma)
a.trace_distance(b)    # 0.79506010   exact: both pure; a mixed pair raises
a.trace_bounds(b)[0]   # 0.39346934   1 - sqrt(F), the exact lower bracket
```

## `qkd.fock` {#qkd-fock}

[The full layer](/guide/fock). The last line is truncation and grid error
([why](/guide/fock#witness)).

```python
from qkd import fock as f

f.Vacuum(), f.Number(1), f.Coherent(x=1.0, p=0.0), f.Squeezed(r=0.5)
f.Thermal(nbar=1.0)
f.Gkp(logical=0, delta=0.3)      # cutoff derived from delta, not fixed
f.Density(np.eye(4) / 4)         # explicit matrix, validated not trusted
f.CUTOFF, f.MAX_LEVELS           # 40, 512

st = f.Cat(2.0, 0.0, odd=True)   # odd cat: parity exactly -1
st.rotate(0.5)                   # exact, diagonal in n
st.displace(x=1.0, p=0.0)        # truncated D is not unitary -- costs norm
st.loss(eta=0.5)                 # Kraus pure-loss channel

st.photons()                           # 2.07462944
st.parity()                            # -1.0        the cheap negativity witness
st.matrix().shape                      # (40, 40)    complex128
st.negativity().value                  # 0.49290448  Wigner volume
float(st.non_gaussianity())            # 2.11632312
float(st.at_cutoff(80).negativity())   # 0.49290448  unmoved: it is physics
float(f.Squeezed(0.5).negativity())    # -3.6686e-07
```

## `qkd.budget` {#qkd-budget}

[The full layer](/guide/budget);
[the two forms of `phase`](/guide/budget#the-two-forms-of-the-phase-term);
[why not $v_{el}$](/guide/budget#why-v-el-is-not-a-budget-line).

```python
import qkd as q
from qkd import budget as b

bud = b.assemble(
    v_a=4.0,
    t=0.1,                # THE SPAN ALONE -- losses multiply into it
    v_err=0.01,           # rad^2; omit and no phase row is emitted
    xi=0.005,             # what the channel already carried
    rin=-155.0, bandwidth=100e6,
    dac_bits=16, adc_bits=12,
    losses=(q.Connector(loss=0.25, count=2, site="launch"),),
)

bud.total                          # 0.04097350   SNU at the channel input
bud.T                              # 0.08912509
bud.at("detector_input")["phase"]  # 0.00358737
bud.refer(0.01, "detector_input", "channel_input")   # 0.11220185

b.phase(v_a=4.0, v_err=0.01, xi=0.005)                    # 0.04025092
b.phase(v_a=4.0, v_err=0.01, form="literature")           # 0.03990017
b.dac(v_a=4.0, bits=16, t=0.1)                            # 7.6599e-10
b.adc(bits=12, t=0.1)                                     # 9.9341e-06
b.rin_sig(v_a=4.0, rin_db=-155.0, bandwidth=100e6, t=0.1) # 0.00071131
b.chain(0.1, (q.Connector(loss=0.25, count=2),))          # (launch, span, receive, rows)
```

## `qkd.impairments` {#qkd-impairments}

[The full layer](/guide/impairments). `security()` rows are not excess noise;
never add them to `assemble_extra()` rows ([why](/guide/impairments#security)).

```python
import qkd as q
from qkd import impairments as imp

q.Coexistence(channels=4, launch=0.0)          # Raman from classical neighbours
q.Backscatter(power=1e-6)                      # Rayleigh from a probe
q.Dephasing(linewidth=10e3, delay=1e-6)        # laser phase diffusion
q.Polarisation(drift=0.02, dispersion=0.0)     # fading + PMD
q.Modulator(ratio=1.02, angle=0.01)            # IQ imbalance
q.Timing(jitter=5e-12, width=100e-12)          # clock
q.Backflash(prob=1e-3)                         # detector -> fibre leakage
q.DeadTime(dead=50e-9, afterpulse=0.02)        # detector memory

imp.phase_variance(10e3, 1e-6)      # 0.06283185   rad^2, Wiener
imp.coherence(10e3, 1e-6)           # 0.96907243   E[cos phi]
imp.dephasing(4.0, 10e3, 1e-6)      # 0.25939109   SNU, estimator form
imp.imbalance(4.0, 1.02, 0.01)      # 0.00100400
imp.saturate(1e6, 50e-9)            # 952380.95    observed click rate
imp.dgd(0.1e-12, 50.0)              # 7.0711e-13   s

[name for name, fn, params in imp.catalogue()]
# ['raman', 'rayleigh', 'dephasing', 'polarisation', 'imbalance', 'timing',
#  'backflash_leak', 'backflash_rate', 'extinction', 'visibility', 'dgd',
#  'saturate', 'afterpulse']

imp.security(backflash=q.Backflash(prob=1e-3), sift=0.5, qber=0.03)
# {'backflash_leak': 0.0005, 'backflash_rate': 0.3867527224576858}
```

## `qkd.attacks` {#qkd-attacks}

A `Reading` has no key rate, and no attacked $\xi$ enters a budget:
[`qkd.attacks`](/guide/impairments#attacks).

```python
from qkd import attacks

[name for name, fn, params in attacks.catalogue()]
# ['saturation', 'calibration', 'oscillator', 'blinding', 'timeshift',
#  'blanking', 'injection']

out = attacks.assess(
    sat=attacks.Saturation(alpha=20.0, delta=19.0),
    calib=attacks.Calibration(ratio=1.5),
    mismatch=attacks.Mismatch(hi=2.0, lo=1.0),
    v_a=5.0, t=0.5, eta=0.55, v_el=0.015, xi=0.1, qber=0.0125,
)
sorted(out)            # ['calibration', 'saturation', 'timeshift']
print(out["saturation"].table())
```

```
saturation
Qin arXiv:1511.01007: the covariance matrix is invariant under a shift of the quadrature mean, and no CV-QKD estimator monitors that mean, so Eve displaces Bob into his clipping region for free

side      quantity  value
observed  t         0.258633
observed  xi        0.127999
eve       xi        2.1
eve       resend    1
```

## `qkd.reconcile` {#qkd-reconcile}

$f_{EC}$, $\beta$ and the frame error rate are never interchanged:
[Post-processing](/guide/protocols#reconcile).

```python
from qkd import reconcile

reconcile.bridge(0.03, f_ec=1.16)     # 0.96139228   f_ec -> beta
reconcile.bridge(0.03, beta=0.95)     # 1.20721242   beta -> f_ec
reconcile.inefficiency(0.5, 0.03)     # 2.57212419

reconcile.codes()[0]
# Ldpc(name='met-0.1', rate=0.1, snr=0.156, dim=None, beta=0.959, block=None, fer=None)

code = reconcile.pick(0.5, dim=8)
code.name                             # 'met-0.1-d8'
code.beta                             # 0.931   at its own threshold
reconcile.code_beta(code, 0.161)      # 0.92905122   run at snr 0.161
```

Cascade, from a measured table and from a simulation:

```python
reconcile.cascade_point(0.03)
# (32, 512, 4096, 1.03945, 1.03902, 0.00011, 0.9906, 496.9)
#  k1  k2   k3    eta_EC   f_EC     fer      beta    rounds

reconcile.run_cascade(0.03, frames=8, seed=1)    # (3342.0, 0)  leaked bits, failures
# The seed indexes a Threefry stream in src/cascade.rs, not Python's MT19937:
# only sample statistics carry across, never a per-seed number.
```

Privacy amplification and authentication:

```python
reconcile.hash_length(1e5, 1e-10)       # 99935   bits out of a 1e5-bit min-entropy
reconcile.amplify_cost(1e-10)           # 64.43856190   the leftover-hash toll
reconcile.tag_length(1e6, 1e-12)        # 61
reconcile.round_cost(1e7, 1e6, 1e6)
# {'amplify': 64.43856189774725, 'authenticate': 171.0}
reconcile.net_rate(1e6, 1e7, 1.0, 1e6)  # 999764.56   bit/s once the layer is paid
reconcile.block_floor(0.01, 1e6)        # 21743.86    smallest block that pays its bill
```

## The native core

Signatures: `help(core.<name>)`. Names no `q.` path reaches:
[the roadmap](/roadmap#core-only). Unclamped returns:
[discrete modulation](/guide/protocols#discrete), [CV-MDI](/guide/relay#point).

<!-- skip -->
```python
from qkd import _core as core

core.cv_rate(v_a, T, xi, eta, v_el, beta, hom, trusted)   # (i_ab, chi_be, key)
core.cv_finite(...)          # (i_ab, chi_worst, key, t_min, xi_max, delta)
core.cv_bounds(...)          # (t_lo, xi_hi) worst-case channel at 1 - eps
core.dm_holevo(v_a, T, xi, z, bits, beta)   # any modulation, via its z
core.decoy_bounds(...)       # (y1_lo, e1_hi, q1_lo)
core.dps_rate(...)           # also bb84_rate, cow_rate, sarg_rate,
                             # sixstate_rate, b92_rate, ekert_rate, pair_rate
core.run_symbols(...)        # also run_clicks, run_basis: the samplers

core.z_pe(1e-10)             # 6.46695108    the literature's "6.5 sigma"
core.gpu_ready()             # whether the wgpu path is usable in this process
core.gpu_probe()             # the adapter, or the reason there is none
```

## Where to go next

The sidebar is the index.
