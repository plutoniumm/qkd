import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import _core, gaussian, std


@dataclass(frozen=True)
class GaussianModulation:
    v_a: float = 5.0

    def __post_init__(self) -> None:
        std.positive("v_a", self.v_a)


@dataclass(frozen=True)
class PhaseShiftKeying:
    """
    Bits keyed onto the phase of a coherent state from a ring of `states`
    amplitudes of modulus alpha (Denys, Brown & Leverrier, Quantum 5, 540).
    """

    # No trusted form: eta and v_el fold in untrusted only, as T -> eta*T,
    # xi -> xi + 2*v_el/(eta*T), so q.Link refuses q.Heterodyne(trusted=True).

    states: int = 4
    alpha: float = 0.4

    def __post_init__(self) -> None:
        if self.states < 3:
            raise ValueError("states must be at least 3")

        std.positive("alpha", self.alpha)

    @property
    def v_a(self) -> float:
        # Per-quadrature modulation variance in SNU.

        return 2.0 * self.alpha * self.alpha

    @property
    def bits(self) -> float:
        return math.log2(self.states)

    def constellation(self) -> tuple[complex, ...]:
        """
        The coherent amplitudes, first on the positive real axis. These are
        PHOTON amplitudes: alpha sits at (2 Re alpha, 2 Im alpha) in SNU.
        """

        return tuple(complex(re, im) for re, im in _core.dm_states(self.states, self.alpha))

    def information(self, T: float, xi: float) -> tuple[float, float]:
        """
        (exact, gauss) bits per symbol: I(X;Y), and the log2(1 + SNR) upper
        bound. The bound's excess grows with alpha, not with the constellation:
        at 8 states, 1.9% at alpha = 1 and 60% at alpha = 5.1. T and xi carry
        the receiver.
        """

        return _core.dm_info(self.states, self.alpha, T, xi)

    def holevo(self, T: float, xi: float, z: float, beta: float) -> tuple[float, float, float]:
        """
        (i_ab, chi_be, key) bits per symbol at a supplied correlation z, which
        is at the CHANNEL OUTPUT plane and already carries its sqrt(T).
        """

        return _core.dm_holevo(self.v_a, T, xi, z, self.bits, beta)


@dataclass(frozen=True)
class DifferentialPhase:
    mu: float = 0.2

    def __post_init__(self) -> None:
        # mu >= 1/2 stays constructible: WTY's (1 - 2*mu) clamps the rate to 0.
        std.positive("mu", self.mu)


@dataclass(frozen=True)
class Decoy:
    """
    The intensity set of a phase-randomised weak-coherent source.
    """

    # (signal, decoy, weak decoy), mean photons per pulse. mu = 0.5 and
    # nu2 = 0.0 are Ma, Qi, Zhao and Lo, Phys. Rev. A 72, 012326
    # (2005), Eq. (12) and Sec. 3.3; nu1 = 0.1 is qkd's, twice their 0.05.
    intensities: tuple[float, float, float] = (0.5, 0.1, 0.0)
    # None is an equal three-way split. A Bell-analysed midpoint's finite chain
    # reads the OUTER PRODUCT of the two senders' ladders as its 3x3 grid, so
    # one sender's marginal is not that grid.
    probs: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.intensities) != 3:
            raise ValueError("intensities must be (signal, decoy, weak decoy)")

        mu, nu1, nu2 = self.intensities
        if mu <= 0.0 or nu1 <= 0.0:
            raise ValueError("intensities must be positive (the weakest may be 0)")

        std.nonneg("intensities", nu2)

        if nu2 >= nu1:
            raise ValueError("decoy intensities must decrease: nu2 < nu1")

        if nu1 + nu2 >= mu:
            raise ValueError("decoys must be weaker than the signal: nu1 + nu2 < mu")

        if self.probs is None:
            return

        if len(self.probs) != 3:
            raise ValueError("probs must be one probability per intensity")

        if any(not 0.0 <= p <= 1.0 for p in self.probs):
            raise ValueError("probs must each be in [0, 1]")

        if abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError("probs must sum to 1")

    @property
    def signal(self) -> float:
        # Mean photons per pulse.

        return self.intensities[0]

    @property
    def weights(self) -> tuple[float, float, float]:
        if self.probs is None:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        return tuple(self.probs)


_ANNOUNCE = ("basis", "pair")

_ANALYSES = ("tolerant", "standard")


@dataclass(frozen=True)
class BasisKeying:
    """
    Bits keyed inside two or three conjugate bases on phase-randomised weak
    coherent pulses. Two bases with the basis announced is BB84-WCP; three is
    the six-state protocol; announcing a non-orthogonal PAIR of states instead
    of a basis is SARG04, which sends the same four states and differs in the
    classical post-processing alone.
    """

    decoy: Decoy = field(default_factory=Decoy)
    # Fraction of slots surviving basis reconciliation, multiplying the whole
    # rate. None takes the uniform value for `bases`. announce='pair' refuses
    # one: SARG04's conclusive fraction is already inside its gains.
    sift: float | None = None
    # 2 is BB84's conjugate pair, 3 the six-state protocol. The third basis
    # lifts the tolerable QBER 11.0% -> 12.6% and costs sifting, 1/3 not 1/2.
    bases: int = 2
    # 'basis' is BB84 and six-state; 'pair' is SARG04, whose non-orthogonal
    # pair announcement keeps a quarter of the rounds where a basis keeps half.
    announce: Literal["basis", "pair"] = "basis"
    # THIS SENDER'S PROBABILITY OF PICKING THE KEY BASIS, and NOT `sift`:
    # `sift` is the surviving share and multiplies a rate, `bias` is one party's
    # choice probability and is what a finite-key layer splits a block into code
    # string and test sample with. Only q.Swap reads one; q.Link refuses a
    # declared bias by name, so None is every q.Link path.
    bias: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decoy, Decoy):
            raise ValueError("decoy must be a q.Decoy intensity set")

        if self.bases not in (2, 3):
            raise ValueError("bases must be 2 (one conjugate pair) or 3 (three mutually unbiased bases)")

        if self.announce not in _ANNOUNCE:
            raise ValueError("announce must be 'basis' (BB84, six-state) or 'pair' (SARG04)")

        if self.announce == "pair" and self.bases != 2:
            raise ValueError(
                "announce='pair' is written for the four states of two "
                "conjugate bases: no three-basis form of it is implemented. "
                "Pass bases=2"
            )

        if self.bias is not None:
            std.open_unit("bias", self.bias)

        if self.announce == "pair":
            if self.sift is not None:
                raise ValueError(
                    "announce='pair' takes no sift: the conclusive fraction "
                    "e_det/2 + 1/4 already sits inside the gains. It is DERIVED "
                    "from Bob's misalignment and reported as sift"
                )

            return

        if self.sift is None:
            share = 0.5 if self.bases == 2 else _core.sixstate_sift(1.0 / 3.0)[0]
            object.__setattr__(self, "sift", share)

        std.unit("sift", self.sift)

    @property
    def mu(self) -> float:
        return self.decoy.signal


@dataclass(frozen=True)
class ReferenceFrame:
    """
    How Bob's polarisation reference is held against the fibre's
    birefringence. The defaults are a perfectly held frame.
    """

    # 'tracked' reads drift alone, 'free' reads rate and interval alone.
    tracking: Literal["tracked", "free"] = "tracked"
    # rms JONES angle in rad, so a matched-basis photon errs at sin^2(theta);
    # a Jones rotation by theta is 2 theta on the Poincare sphere.
    drift: float = 0.0
    # rad/sqrt(s).
    rate: float = 0.0
    # s since the frame was last aligned.
    interval: float = 0.0
    # PMD coefficient in s/sqrt(km), giving first-order depolarisation at the
    # WORST input polarisation. Read with width: both or neither.
    dispersion: float = 0.0
    # Pulse intensity-envelope rms width, s.
    width: float = 0.0

    def __post_init__(self) -> None:
        if self.tracking not in ("tracked", "free"):
            raise ValueError(
                "tracking must be 'tracked' (a compensated receiver, given its "
                "residual drift) or 'free' (an uncompensated one, given its "
                "diffusion rate and calibration interval)"
            )

        for name in ("drift", "rate", "interval", "dispersion", "width"):
            std.nonneg(name, getattr(self, name))

        if self.tracking == "tracked" and (self.rate != 0.0 or self.interval != 0.0):
            raise ValueError(
                "a tracked frame is described by its residual drift; rate and " "interval belong to tracking='free'"
            )

        if self.tracking == "free":
            if self.drift != 0.0:
                raise ValueError(
                    "a free-running frame accumulates its own angle; drift " "belongs to tracking='tracked'"
                )

            if self.rate <= 0.0 or self.interval <= 0.0:
                raise ValueError(
                    "tracking='free' needs a positive rate and interval; for a "
                    "frame that never drifts pass tracking='tracked' with "
                    "drift = 0"
                )

        if (self.dispersion > 0.0) != (self.width > 0.0):
            raise ValueError("dispersion and width are read together: give both or neither")


@dataclass(frozen=True)
class PolarisationKeying:
    """
    BasisKeying with the carrier named, in two conjugate LINEAR bases that turn
    by one angle together.
    """

    decoy: Decoy = field(default_factory=Decoy)
    sift: float = 0.5
    # The misalignment is derived from this, not taken from Bob; real SU(2)
    # birefringence makes it a floor on a span's damage, not a full account.
    frame: ReferenceFrame = field(default_factory=ReferenceFrame)

    def __post_init__(self) -> None:
        if not isinstance(self.decoy, Decoy):
            raise ValueError("decoy must be a q.Decoy intensity set")

        if not isinstance(self.frame, ReferenceFrame):
            raise ValueError("frame must be a q.ReferenceFrame")

        std.unit("sift", self.sift)

    @property
    def mu(self) -> float:
        return self.decoy.signal

    def contrast(self, dgd: float = 0.0) -> float:
        """
        Polarisation contrast 1 - 2e surviving this frame, given a mean
        differential group delay `dgd` in s already accumulated by q.Link.
        """
        frame = self.frame
        if frame.tracking == "free":
            walk = _core.pol_walk(frame.rate, frame.interval)

            return walk * _core.pol_contrast(0.0, dgd, frame.width)

        return _core.pol_contrast(frame.drift, dgd, frame.width)


@dataclass(frozen=True)
class IntensityKeying:
    """
    Bits keyed onto which of two adjacent slots carries light, some sequences
    filled in both: the coherent-one-way protocol, COW.
    """

    # Mean photons in a filled slot.
    mu: float = 0.5
    # Fraction of sequences carrying the monitoring pattern, which carry no key.
    decoy_frac: float = 0.1
    # Modulator on/off contrast in dB, None being the empty-slot idealisation.
    # Real modulators reach 20 to 30 dB.
    extinction: float | None = None

    def __post_init__(self) -> None:
        std.positive("mu", self.mu)

        std.fraction("decoy_frac", self.decoy_frac)

        if self.extinction is not None and self.extinction <= 0.0:
            raise ValueError("extinction must be a positive dB contrast; None is the empty-slot idealisation")

    @property
    def residual(self) -> float:
        # Mean photons in the nominally-empty slot.
        if self.extinction is None:
            return 0.0

        return _core.cow_residual(self.mu, self.extinction)


_SOURCES = ("coherent", "single")


@dataclass(frozen=True)
class TwoStateKeying:
    """
    Bits keyed onto one of TWO non-orthogonal states. No basis and no basis
    announcement: Bob either names which state was sent or says nothing, and
    the overlap exp(-2*mu) is the protocol's whole dial.

    `source` names what the two states are. The default pair is coherent,
    |alpha> and |-alpha>, and `mu` is then a mean photon number; source='single'
    is the single-photon pair the phase bound is proved for, where `mu` names
    the overlap and nothing else.
    """

    # Mean photons in the signal pulse, |alpha|^2. 0.23 is Koashi's optimal
    # amplitude, an overlap of 0.6313. On source='single' there is no photon
    # number to be mean of and this names the OVERLAP instead, through the same
    # exp(-2*mu).
    mu: float = 0.23
    # Whether Alice sends a BRIGHT PHASE REFERENCE beside every signal. False
    # DERIVES the phase error rate from the loss and the overlap; True is the
    # strong-reference variant, whose bound is supplied on
    # q.DiscriminationBound instead.
    reference: bool = False
    # 'coherent' is the pair |alpha>, |-alpha> read by a coherent-state
    # receiver whose gain and bit error carry a dark count and a fringe
    # contrast, and stands a beam-splitting ceiling under Tamaki &
    # Lutkenhaus's bound in place of the missing multiphoton term; 'single' is
    # the single-photon source that proof is written for and needs neither.
    source: Literal["coherent", "single"] = "coherent"
    # Tamaki & Lutkenhaus's depolarising rate p: rho -> L|V><V| + (1 - L)
    # [(1 - p) rho + (p/3) sum_i sigma_i rho sigma_i]. NOT a bit error rate and
    # not a QBER -- a BB84 error rate would be 2p/3. source='single' only.
    depol: float | None = None

    def __post_init__(self) -> None:
        std.positive("mu", self.mu)

        if self.source not in _SOURCES:
            raise ValueError(
                "source must be 'coherent' (two weak coherent states) or "
                "'single' (the single-photon source the phase bound is proved "
                "for)"
            )

        if self.source == "coherent":
            if self.depol is not None:
                raise ValueError(
                    "depol reaches no term of the coherent branch, which reads "
                    "its bit error off the receiver's own fringe contrast and "
                    "dark count. Give q.NullingReceiver(visibility=...) and "
                    "q.ClickDetector(dark=...), or pass source='single'"
                )

            return

        if self.reference:
            raise NotImplementedError(
                "reference=True is the strong-reference variant, whose phase "
                "error rate is supplied on q.DiscriminationBound; the "
                "single-photon analysis derives its own, and the two are "
                "different proofs, not two sources of one. Drop reference= or "
                "pass source='coherent'"
            )

        if self.depol is None:
            raise ValueError(
                "source='single' needs depol: the single-photon branch takes "
                "its gain and its bit error from a depolarising channel rather "
                "than from a receiver. Pass depol=0.0 for the noiseless span"
            )

        std.fraction("depol", self.depol)

    @property
    def overlap(self) -> float:
        # <alpha|-alpha> = exp(-2*mu), the overlap of Alice's two states. NOT
        # the overlap the gate sees, which carries eta and T as well.

        return _core.b92_overlap(self.mu)


@dataclass(frozen=True)
class PairSource:
    """
    A photon-pair source: two independent two-mode squeezers, one per encoding
    mode pair, feeding one photon of each pair to each party. A SOURCE, not a
    party, and it carries no trusted= flag: BBM92 trusts it in no
    configuration, Alice and Bob certifying the state in both bases.
    """

    # Pairs per pump pulse PER MODE PAIR, lambda = sinh^2(chi); the mean pair
    # number is 2*brightness. None asks qkd.pairs to optimise it.
    brightness: float | None = None
    # Pump repetition rate in Hz. Reaches bits per second and nothing else; the
    # rate functions are all per pump pulse.
    rate: float = 100e6
    pumping: str = "pulsed"

    def __post_init__(self) -> None:
        if self.brightness is not None:
            std.positive("brightness", self.brightness)

        std.positive("rate", self.rate)

        if self.pumping != "pulsed":
            raise NotImplementedError(
                "only pumping='pulsed' is implemented: the Ma-Fung-Lo gain and "
                "error rate are per pump pulse and carry no accidental-"
                "coincidence term, so a continuous-wave source would drop its "
                "dominant high-brightness error and inflate the key rate"
            )


@dataclass(frozen=True)
class SymmetryBound:
    """
    Asymptotic security for a photon-pair link. Alice and Bob measure in two
    conjugate bases on the same pairs, so basis symmetry makes the phase error
    the bit error; the name is the argument, not the hardware.
    """

    # None takes e_phase = e_bit, the asymptotic statement of basis symmetry
    # for BBM92 and what Ma, Fung & Lo Eq. (5) evaluates; a value pins a
    # different one. Unlike q.PhaseBound this relation exists, holding by
    # symmetry rather than by a proven bound.
    e_phase: float | None = None
    # Error-correction inefficiency over the Shannon limit. 1.22 is Ma, Fung &
    # Lo's own Table 1 value, itself Lutkenhaus, Phys. Rev. A 61, 052304 (2000),
    # Table I at e = 0.10.
    f: float = 1.22
    # Fraction of coincidences surviving basis reconciliation; 1/2 uniform.
    sift: float = 0.5

    def __post_init__(self) -> None:
        if self.e_phase is not None:
            std.closed("e_phase", self.e_phase, 0.5)

        std.atleast("f", self.f, 1.0)

        std.unit("sift", self.sift)


@dataclass(frozen=True)
class ViolationBound:
    """
    Asymptotic security for a photon-pair link priced by the observed CHSH
    VIOLATION rather than by the error rate. It stands where q.SymmetryBound
    stands: basis symmetry reads Eve off a phase error under a
    characterised-qubit assumption, a violation prices her with no assumption
    about the state at all. The violation is never the cheaper price -- the
    zero-key error rate falls from 11.003% to 7.149%.

    THIS IS NOT DEVICE INDEPENDENCE, and `source` is how it says so.
    """

    # The CHSH magnitude. Either sign is read: Ekert's own Eq. (3) convention
    # makes it negative at his settings and the modern one positive.
    s: float
    # WHERE `s` CAME FROM, and it has NO DEFAULT because the answer decides
    # whether this number is a bound at all. 'measured' is an estimate from
    # this run's CHSH rounds; 'modelled' and 'device-independent' are REFUSED
    # by the engine when the link runs.
    source: Literal["measured", "modelled", "device-independent"]
    # Error-correction inefficiency over the Shannon limit, as on
    # q.SymmetryBound; 1.22 keeps the two comparable.
    f: float = 1.22
    # Fraction of pairs surviving reconciliation. 2/9 is Ekert's own six
    # settings drawn uniformly -- two of the nine pairs coincide and carry the
    # key -- which is what makes the third setting a cost rather than free.
    sift: float = 2.0 / 9.0

    def __post_init__(self) -> None:
        std.atleast("f", self.f, 1.0)

        std.unit("sift", self.sift)

        # s and source are the engine's to check: Tsirelson's bound and the two
        # refusals come through q.PairLink verbatim.


@dataclass(frozen=True)
class Channel:
    """
    A thermal-loss span stated by numbers rather than hardware, against
    q.Fiber's length and attenuation. Itemised optics attach to q.Fiber only.
    """

    # A bare transmittance in (0, 1], not a decibel figure and not a loss.
    T: float
    # SHOT-NOISE UNITS. 0.0 is a pure-loss span and takes no ref.
    xi: float = 0.0
    # 'input' is the CHANNEL INPUT plane, Alice's side, which the library
    # carries xi on; 'output' is Bob's side, divided by T on the way in.
    ref: Literal["input", "output"] | None = None

    def __post_init__(self) -> None:
        std.unit("T", self.T)

        std.nonneg("xi", self.xi)

        if self.ref is not None and self.ref not in ("input", "output"):
            raise ValueError("ref must be 'input' or 'output'")

        # xi is stored at the plane the caller named; q.Link converts it.
        if self.xi != 0.0 and self.ref is None:
            raise ValueError("nonzero xi requires ref='input' or ref='output'")


@dataclass(frozen=True)
class Fiber:
    """
    The span itself: length in km with an attenuation, or a transmittance T
    outright, never both.
    """

    length: float | None = None
    # Attenuation in dB/km. 0.2 is a DATASHEET MAXIMUM FOR BARE FIBRE, not a
    # typical value: Corning SMF-28e+ 0.20 at 1550 nm, Ultra 0.18, ULL 0.16 to
    # 0.17. Installed ITU-T G.652 cable is 0.30 (cat D) to 0.35 (cat B), with
    # Appendix I offering 0.275 dB/km for a concatenated link.
    alpha: float = 0.2
    T: float | None = None

    def __post_init__(self) -> None:
        if self.length is not None and self.T is not None:
            raise ValueError("give hardware description or derived value, not both")

        if self.length is None and self.T is None:
            raise ValueError("Fiber needs either length or T")

        if self.length is not None:
            std.positive("length", self.length)

        std.nonneg("alpha", self.alpha)

        if self.T is not None:
            std.unit("T", self.T)

    @property
    def transmittance(self) -> float:
        if self.T is not None:
            return self.T

        return std.from_db(self.alpha * self.length)


_SITES = ("launch", "span", "receive")


def _optic(loss: float, count: int, site: str) -> None:
    if loss < 0.0:
        raise ValueError("loss is an insertion loss in dB and must be nonnegative")

    std.atleast("count", count, 1)

    if site not in _SITES:
        raise ValueError("site must be 'launch', 'span' or 'receive'")


@dataclass(frozen=True)
class Connector:
    """
    A mated connector pair. No polish= field: angled and flat ferrules cost
    about the same forward loss (0.25 vs 0.3 dB) and differ in RETURN loss.
    """

    # dB for ONE mated pair. 0.25 is the IEC 61753-1 grade C mean (B 0.12,
    # D 0.50) and Thorlabs' typical FC/APC; QOSST (Quantum 8, 1575 (2024))
    # charges 0.23 dB per PM mating sleeve and 0.47 dB per spool connector.
    loss: float = 0.25
    # A patch cord between two bulkheads is count=2.
    count: int = 1
    # 'launch', 'receive' or 'span': same total T, different input-referred
    # excess noise for anything born in the fibre.
    site: Literal["launch", "span", "receive"] = "launch"

    def __post_init__(self) -> None:
        _optic(self.loss, self.count, self.site)


@dataclass(frozen=True)
class Splice:
    # dB for ONE splice. 0.02 is the measured mean for SMF-28 Ultra to itself
    # at 1550 nm on a core-aligning splicer (Corning/AFL AN0041); v-groove
    # gives 0.03-0.04, Telcordia GR-20-CORE asks a group mean under 0.10, and a
    # mechanical splice is a few tenths, declared by passing that number.
    loss: float = 0.02
    count: int = 1
    # A transmitter pigtail sits at 'launch'.
    site: Literal["launch", "span", "receive"] = "span"

    def __post_init__(self) -> None:
        _optic(self.loss, self.count, self.site)


@dataclass(frozen=True)
class Coupling:
    """
    Light crossing between two guided or free modes: fibre to chip, free space
    to fibre, or fibre to bulk optics.
    """

    # No default: published values span an order of magnitude -- 3.1 dB
    # shallow-etched silicon grating, 1.3 dB silicon inverse-taper edge, 0.8 dB
    # best free-form to SMF-28, ~1 dB free-space injection at its best.
    loss: float
    # Identical interfaces at this site; a chip in line has two facets.
    count: int = 1
    site: Literal["launch", "span", "receive"] = "launch"

    def __post_init__(self) -> None:
        _optic(self.loss, self.count, self.site)


def _quad(eta: float, v_el: float, clip: float | None, bandwidth: float | None) -> None:
    # Called rather than inherited: a frozen dataclass does not inherit cleanly
    # across defaulted and non-defaulted fields.
    std.unit("eta", eta)
    std.nonneg("v_el", v_el)

    if clip is not None:
        std.finite("clip", clip)

        std.positive("clip", clip)

    if bandwidth is not None:
        std.finite("bandwidth", bandwidth)

        std.positive("bandwidth", bandwidth)


def _scale(clip: float | None) -> float | None:
    # The full range is 2*clip: the receiver is linear over +/-clip.
    if clip is None:
        return None

    return 2.0 * clip


@dataclass(frozen=True)
class Homodyne:
    """
    A balanced homodyne receiver, one quadrature per symbol. The default
    (eta, v_el) is a composite pair no single receiver has.
    """

    # Bob-side APPARATUS efficiency, not a bare photodiode QE. Fossier et al.,
    # New J. Phys. 11, 045023 (2009); published values run 0.44 to 0.84.
    eta: float = 0.6
    # Electronic-noise variance in SNU at Bob's input. Clearance
    # (shot + electronic)/electronic = (1 + v_el)/v_el = 11, i.e. 10.4 dB, by
    # Laudenbach et al., arXiv:1703.09278v3 Eqs. (9.96)-(9.97) at mu = 1; their
    # own worked example is 0.0396 SNU (Fig. 10.2). High end of Zhang et al.,
    # Phys. Rev. Lett. 125, 010502 (2020); 0.01-0.06 detectors are slower.
    v_el: float = 0.1
    # Security-model flag: whether eta and v_el are Bob's or Eve's.
    trusted: bool = True
    # LINEAR HALF-RANGE in shot-noise amplitude units: linear over
    # +/- clip*sigma_shot, None the unbounded idealisation every rate here is
    # written in. q.attacks.Saturation's `alpha`; full_scale() is
    # qkd.budget.adc's `ratio`. NO KEY RATE HERE MODELS CLIPPING, so q.Link
    # refuses a declared clip by name, as it refuses a q.ADC.
    clip: float | None = None
    # DETECTION BANDWIDTH in Hz, read by qkd.budget's rin_sig, rin_lo and
    # assemble's Raman row; assemble(rin=..., bandwidth=...) refuses one
    # without the other. q.Link refuses a declared bandwidth by name.
    bandwidth: float | None = None

    def __post_init__(self) -> None:
        _quad(self.eta, self.v_el, self.clip, self.bandwidth)

    def full_scale(self) -> float | None:
        """
        Full-scale range over the shot-noise standard deviation, 2*clip: the
        `ratio` of qkd.budget.adc and budget.assemble. None where no linear
        range was declared; budget's own default 10.0 is clip = 5.0.
        """

        return _scale(self.clip)


@dataclass(frozen=True)
class Heterodyne:
    """
    A phase-diverse receiver: both quadratures every symbol, at the cost of one
    vacuum unit entering the splitter. eta and v_el are as on q.Homodyne.
    """

    eta: float = 0.6
    # Does not halve at the splitter: two detectors each contribute their own,
    # so the untrusted substitution carries mu = 2 against homodyne's 1
    # (Laudenbach et al., arXiv:1703.09278v3, Eqs. 9.93-9.94). Dropping that
    # inflates the key by 0.2165 bit/symbol at the default operating point.
    v_el: float = 0.1
    trusted: bool = True
    # LINEAR HALF-RANGE in shot-noise amplitude units: linear over
    # +/- clip*sigma_shot, None the unbounded idealisation every rate here is
    # written in. q.attacks.Saturation's `alpha`; full_scale() is
    # qkd.budget.adc's `ratio`. NO KEY RATE HERE MODELS CLIPPING, so q.Link
    # refuses a declared clip by name, as it refuses a q.ADC.
    clip: float | None = None
    # DETECTION BANDWIDTH in Hz, read by qkd.budget's rin_sig, rin_lo and
    # assemble's Raman row; assemble(rin=..., bandwidth=...) refuses one
    # without the other. q.Link refuses a declared bandwidth by name.
    bandwidth: float | None = None

    def __post_init__(self) -> None:
        _quad(self.eta, self.v_el, self.clip, self.bandwidth)

    def full_scale(self) -> float | None:
        """
        Full-scale range over the shot-noise standard deviation, 2*clip: the
        `ratio` of qkd.budget.adc and budget.assemble. None where no linear
        range was declared; budget's own default 10.0 is clip = 5.0.
        """

        return _scale(self.clip)


@dataclass(frozen=True)
class ClickDetector:
    """
    A threshold detector and its gate. eta = 0.2 and dark = 1e-6 are round
    figures for the gated InGaAs/InP class, neither of them a measurement.
    """

    # Mid-band of a published 10-25%. Gobby, Yuan and Shields, Appl. Phys.
    # Lett. 84, 3762 (2004) measure ~12% at 1.55 um, entering a rate as 0.045.
    eta: float = 0.2
    # PER DETECTOR AND PER GATE, never a receiver total, and a probability
    # rather than a rate in Hz. Bob reads two gates per bit here, so q.Link
    # derives Y0 = 1 - (1 - dark)^2; Ma, Qi, Zhao and Lo's GYS Y0 = 1.7e-6 is
    # twice GYS's 8.5e-7, so pass 8.5e-7 rather than 1.7e-6. A datasheet rate
    # in Hz enters through gated(rate, gate), which needs the GATE WIDTH.
    dark: float = 1e-6
    dead_time: float = 0.0
    afterpulse: float = 0.0
    # FWHM of the Gaussian arrival-time core, s. Light missing every window is
    # an efficiency loss, light in a NEIGHBOUR's window an error.
    jitter: float = 0.0
    # Acceptance width in s inside one symbol period; 0 is contiguous binning.
    # It GATES a response and is nothing on its own: with jitter and tail_frac
    # both 0 no kernel is built, so q.Link refuses a window with neither beside
    # it, and refuses one outright off the sampled phase-keyed path, its only
    # reader.
    window: float = 0.0
    # Weight and time constant of the one-sided exponential diffusion tail a
    # thick silicon SPAD adds to the Gaussian core.
    tail_frac: float = 0.0
    tail_time: float = 0.0
    # OPTICAL filter FWHM in Hz in front of this detector; 0.0 is UNSTATED
    # rather than unfiltered, so q.Link refuses a q.Coexistence without it. NOT
    # qkd.budget.raman_width's number -- that is a DETECTION bandwidth, and a
    # threshold detector has no local oscillator to be the filter.
    # dnu = c dlambda/lambda^2 converts a datasheet width: 45 pm at 1550 nm is
    # 5.62 GHz.
    passband: float = 0.0
    # BOB'S OTHER THRESHOLD DETECTOR; None is the symmetric receiver every rate
    # here is written for. A partner makes DETECTION-EFFICIENCY MISMATCH
    # expressible as hardware, priced by q.attacks.Mismatch(hi, lo) and read by
    # pair() and background(). NO q.Link RATE READS IT -- charging a mismatch
    # means Fung, Tamaki, Qi, Lo & Ma's 2*min/(hi + lo) factor and a phase error
    # rate estimated apart from the bit error rate -- so q.Link refuses a
    # partner by name.
    partner: "ClickDetector | None" = None

    def __post_init__(self) -> None:
        std.unit("eta", self.eta)
        std.fraction("dark", self.dark)
        std.nonneg("dead_time", self.dead_time)
        std.fraction("afterpulse", self.afterpulse)
        if self.jitter < 0.0 or self.window < 0.0 or self.tail_time < 0.0:
            raise ValueError("jitter, window and tail_time must be nonnegative")

        std.nonneg("passband", self.passband)

        std.closed("tail_frac", self.tail_frac)

        if self.tail_frac > 0.0 and self.tail_time <= 0.0:
            raise ValueError("a tail_frac above 0 needs a positive tail_time")

        if self.partner is None:
            return

        if not isinstance(self.partner, ClickDetector):
            raise ValueError(
                "partner is Bob's other threshold detector and must be a "
                "q.ClickDetector; None is the symmetric receiver, the same "
                "detector twice"
            )

        if self.partner.partner is not None:
            raise ValueError("a partner names the SECOND of two detectors and carries none: there is no third")

    @classmethod
    def gated(cls, rate: float, gate: float, **kw: object) -> "ClickDetector":
        """
        A detector whose dark floor is given as a datasheet gives it: a
        FREE-RUNNING count rate `rate` in Hz over a gate `gate` seconds wide,
        rather than as the per-gate probability `dark` holds. Every other field
        passes through; `dark` may not.

        `gate` is the width the detector is BIASED ABOVE BREAKDOWN for, not the
        clock period and not q.ClickDetector.window: 1 kHz over a 100 ps gate
        is dark = 1e-10, and over a 1 ns clock period it is 1e-6.
        """
        if "dark" in kw:
            raise ValueError(
                "gated() derives dark from a count rate and a gate width; pass "
                "q.ClickDetector(dark=...) to state the per-gate probability "
                "directly, not both"
            )

        return cls(dark=std.dark_prob(rate, gate), **kw)

    def timed(self) -> bool:
        return self.jitter > 0.0 or self.tail_frac > 0.0

    def both(self) -> tuple["ClickDetector", "ClickDetector"]:
        """
        The receiver's two detectors, this one first; itself twice where no
        partner was named.
        """

        return (self, self if self.partner is None else self.partner)

    def pair(self) -> tuple[float, float]:
        """
        The two quantum efficiencies, LARGER FIRST, which is the (hi, lo) order
        q.attacks.Mismatch and attacks.mismatch_rate take. Equal on a symmetric
        receiver, where mismatch_rate's 2*min/(hi + lo) prefactor is 1.
        """
        one, two = self.both()

        return (max(one.eta, two.eta), min(one.eta, two.eta))

    def background(self) -> float:
        """
        Per-pulse background yield Y0 = 1 - (1 - dark_0)(1 - dark_1) out of the
        TWO gates a two-detector receiver reads per bit, collapsing to
        1 - (1 - dark)^2 with no partner.

        NOT the figure for q.NullingReceiver, which reads ONE gate per pulse:
        there `dark` is already the per-pulse number.
        """
        one, two = self.both()

        return 1.0 - (1.0 - one.dark) * (1.0 - two.dark)


@dataclass(frozen=True)
class ThresholdArray:
    """
    A multiplexed threshold array: `elements` identical threshold detectors
    behind a balanced splitter, spatial or temporal. What is read is HOW MANY
    elements fired, so two photons in one bin are counted once and the
    resolution is bounded by the element count. elements = 1 is
    q.ClickDetector's own click law exactly.

    UNIFORM ELEMENTS ONLY: the occupancy recursion behind the POVM is bin
    exchangeability, which a non-uniform array does not have. Temporal
    multiplexing is covered only where the loop's per-bin loss folds into one
    common eta.

    No q.Link family reads this: every key rate qkd ships over a counting
    receiver is written on a binary click, so q.Link refuses it by name.
    """

    elements: int = 8
    eta: float = 0.2
    # PER ELEMENT AND PER GATE, as on q.ClickDetector, and a probability rather
    # than a rate in Hz. That is the plane povm() reads it on; the WHOLE
    # RECEIVER's per-gate probability -- detector decoy's Moroder Eq. (16)
    # prefactor is the one consumer -- is gate(), and the two differ from two
    # elements up.
    dark: float = 1e-6
    # Fock levels the POVM elements are defined on, 0..cutoff-1. A TRUNCATION
    # rather than hardware: raise it until the state's photon-number tail sits
    # inside it, which povm().coherent(mu) refuses to ignore.
    cutoff: int = 40

    def __post_init__(self) -> None:
        std.atleast("elements", self.elements, 1)

        std.unit("eta", self.eta)

        std.fraction("dark", self.dark)

        std.atleast("cutoff", self.cutoff, 1)

    def povm(self) -> _core.Povm:
        """
        The measurement itself: elements + 1 outcomes on levels 0..cutoff-1,
        every element diagonal in the Fock basis. Carries diagonal(), element(k),
        defect(), floor(), fold(probs) and coherent(mu).
        """

        return _core.pnr_array(self.cutoff, self.elements, self.eta, self.dark)

    def outcomes(self, mu: float) -> np.ndarray:
        """
        Outcome distribution for a coherent state of mean photon number `mu`, in
        closed form: a balanced splitter leaves each bin Poisson and
        independent, so the count is binomial. NO Fock truncation, unlike
        povm().coherent(mu).
        """

        return _core.pnr_coherent(self.elements, self.eta, self.dark, mu)

    def distinct(self, n: int) -> float:
        """
        The probability that `n` photons land in distinct bins, which is this
        array's fidelity to true number resolution at unit efficiency. Zero once
        n exceeds the element count, by the pigeonhole.
        """

        return _core.pnr_distinct(self.elements, n)

    def gate(self) -> float:
        """
        The probability the WHOLE RECEIVER reports a spurious click in a gate
        holding no signal photon, 1 - (1 - dark)**elements.

        `dark` is the PER-ELEMENT number and is not this one above one element.
        Moroder Eq. (16)'s (1 - dark) prefactor is written on the receiver, so
        q.DecoyAttenuator reads gate() and never the field.
        """
        # 1 - (1 - dark)**elements cancels: at dark = 1e-6 it errs 2.9e-11
        # relative and does not return `dark` itself at one element.

        return -math.expm1(self.elements * math.log1p(-self.dark))


@dataclass(frozen=True)
class PnrDetector:
    """
    A number-resolving detector whose READOUT sets the resolution: efficiency
    thins the incident number, a background adds to it, and a pulse height of
    Gaussian width `sigma` is binned to the nearest photon peak. sigma = 0
    collapses the confusion to the identity. The top outcome ABSORBS -- it
    reads "cutoff - 1 or more" -- which keeps the elements complete once a
    background puts mass above the cutoff.

    THE READOUT MODEL IS PARAMETRISED, NOT FITTED TO A DEVICE. confusion()
    returns it so a calibrated sensor's measured matrix can be held against it.

    No q.Link family reads this, for the reason q.ThresholdArray gives.
    """

    eta: float = 0.9
    # Mean spurious COUNT per gate, not the per-gate probability q.ClickDetector
    # and q.ThresholdArray take.
    background: float = 0.0
    # Readout width in units of one photon's step.
    sigma: float = 0.0
    # Saturation scale: peak spacing falls as exp(-m/saturation), so one fixed
    # sigma merges the high-number peaks. 0 is a linear response.
    saturation: float = 0.0
    cutoff: int = 40

    def __post_init__(self) -> None:
        std.unit("eta", self.eta)

        std.nonneg("background", self.background)

        std.nonneg("sigma", self.sigma)

        std.nonneg("saturation", self.saturation)

        std.atleast("cutoff", self.cutoff, 1)

    def povm(self) -> _core.Povm:
        """
        The measurement itself: cutoff outcomes on levels 0..cutoff-1, the top
        one absorbing. Same interface as q.ThresholdArray.povm().
        """

        return _core.pnr_tes(self.cutoff, self.eta, self.background, self.sigma, self.saturation)

    def outcomes(self, mu: float) -> np.ndarray:
        """
        Outcome distribution for a coherent state of mean photon number `mu`,
        folded through the POVM. No closed form as q.ThresholdArray has, so
        this carries the Fock truncation and refuses a `mu` whose photon-number
        tail falls outside the cutoff.
        """

        return self.povm().coherent(mu)

    def confusion(self, top: int | None = None) -> np.ndarray:
        """
        The readout confusion alone, C[k, m] = P(report k | m detected), as a
        square array over 0..top. Columns sum to 1 and the top row absorbs;
        None takes the whole cutoff.
        """
        edge = self.cutoff - 1 if top is None else top

        return _core.pnr_confuse(edge, self.sigma, self.saturation)


def _gate(detector: object) -> float:
    # The engine's `dark` slot is the WHOLE RECEIVER's PER-GATE PROBABILITY;
    # q.DecoyAttenuator.bounds() sets out the three planes it is stated on.
    # Handing q.ThresholdArray's per-element field over instead of its gate()
    # understates the spurious floor, so the forward no-click comes out high and
    # the inverted photon-number weights low. `saturation` is inert here rather
    # than dropped: it acts only through the readout bin edges, which pnr.rs
    # collapses to the identity at sigma = 0, the one regime this converts in.
    if isinstance(detector, ThresholdArray):
        return detector.gate()

    dark = getattr(detector, "dark", None)
    if dark is not None:
        return dark

    if not isinstance(detector, PnrDetector):
        raise ValueError(
            "detector must carry a per-gate spurious-click probability: "
            "q.ClickDetector carries one as `dark`, q.ThresholdArray composes "
            "one from its per-element `dark`, and q.PnrDetector's mean spurious "
            "count per gate converts to one here"
        )

    if detector.sigma > 0.0:
        raise NotImplementedError(
            "a q.PnrDetector with sigma > 0 reports vacuum for a photon it did "
            "detect, so no per-gate dark probability stands for Moroder "
            "Eq. (16)'s (1 - dark) sum_n (1 - atten*eta)^n. Pass sigma=0.0, or "
            "a q.ClickDetector"
        )

    if detector.cutoff < 2:
        raise ValueError(
            "a q.PnrDetector at cutoff=1 carries one absorbing outcome and "
            "reads vacuum whatever arrives: there is no no-click statistic to "
            "invert. Raise cutoff"
        )

    return 1.0 - math.exp(-detector.background)


@dataclass(frozen=True)
class PhotonBounds:
    """
    What detector decoy certifies about the photon-number distribution arriving
    at Bob: the vacuum weight EXACTLY, and two-sided bounds on the one- and
    two-photon weights. Moroder, Curty & Lutkenhaus, New J. Phys. 11, 045008
    (2009), Proposition 2.1.

    p0 IS NOT A BOUND and the other four are: p0 is read straight off the
    transparent setting, and each interval is what two probing settings leave
    once the remaining mass is conceded to the worst photon number. Never
    average an interval to a point.
    """

    p0: float
    p1_lo: float
    p1_hi: float
    p2_lo: float
    p2_hi: float


@dataclass(frozen=True)
class DecoyAttenuator:
    """
    Bob's variable attenuator at THREE settings, the detector-decoy
    measurement: he re-runs the same no-click statistic behind three known
    attenuations and inverts the three numbers for the photon-number weights
    arriving at him. A measurement on Bob's own receiver, not a source-side
    intensity ladder -- q.Decoy is that.

    THE FIRST SETTING MUST BE TRANSPARENT AND THE DETECTOR PERFECT. Proposition
    2.1 anchors every later bound on x_0 = f(0), and c_0 = 1 - atten_0*eta is
    zero only at unit attenuation on a unit-efficiency detector; `bounds()`
    lets the engine's refusal through in its own words.

    No q.Link family reads this, and it has no rate: what it returns is an input
    to one, and no bound qkd ships takes a photon-number interval.
    """

    # The three transmittances (a0, a1, a2), the first transparent and the
    # other two strictly decreasing so that 0 < c1 < c2 < 1 in the
    # proposition's own variable c = 1 - atten*eta. (1.0, 0.99, 0.9) is its
    # stated convergence path c1 = delta, c2 = sqrt(delta) at delta = 0.01.
    settings: tuple[float, float, float] = (1.0, 0.99, 0.9)
    # The C of Proposition 2.1, the bound on sum_n x_n, which is 1 for a
    # photon-number distribution and less for a sub-normalised one.
    cap: float = 1.0

    def __post_init__(self) -> None:
        if len(self.settings) != 3:
            raise ValueError("settings must be three transmittances (transparent, first probe, second probe)")

        first, one, two = self.settings
        if first != 1.0:
            raise ValueError(
                "the first setting must be exactly 1.0, a transparent "
                "attenuator: Proposition 2.1 reads the vacuum weight off f(0)"
            )

        for name, value in (("settings[1]", one), ("settings[2]", two)):
            std.unit(name, value)

        if not two < one < 1.0:
            raise ValueError(
                "the two probing settings must decrease strictly below the "
                "transparent one, settings[2] < settings[1] < 1: they are the "
                "proposition's c1 < c2 written as transmittances"
            )

        std.positive("cap", self.cap)

    def bounds(self, detector: ClickDetector | ThresholdArray | PnrDetector, vacs: std.ArrayLike) -> PhotonBounds:
        """
        The photon-number weights `vacs` implies, as a q.PhotonBounds. `vacs`
        are the three observed NO-CLICK probabilities, one per setting and in
        the same order; `detector` supplies `eta` and the per-gate
        spurious-click PROBABILITY of Moroder Eq. (16).

        q.ClickDetector carries that probability as `dark`. q.ThresholdArray's
        `dark` is PER ELEMENT, so what enters is its gate(),
        1 - (1 - dark)**elements. q.PnrDetector carries a mean spurious COUNT
        per gate, entering as 1 - exp(-background), and its readout must be
        exact: a sigma above 0 is refused rather than converted.

        The engine's own refusals reach the caller unchanged: a first setting
        that does not reach c = 0, and probing settings that do not bracket.
        """

        return PhotonBounds(
            *_core.pnr_decoy(
                detector.eta,
                _gate(detector),
                tuple(self.settings),
                tuple(vacs),
                self.cap,
            )
        )

    def noclick(
        self, detector: ClickDetector | ThresholdArray | PnrDetector, probs: std.ArrayLike
    ) -> tuple[float, float, float]:
        """
        The forward half: the three no-click probabilities a photon-number
        distribution `probs` produces at these settings, (1 - dark) sum_n
        (1 - atten*eta)^n probs[n]. `detector` reaches `dark` exactly as
        bounds() does.
        """
        dark = _gate(detector)

        return tuple(_core.pnr_noclick(detector.eta, dark, atten, list(probs)) for atten in self.settings)


@dataclass(frozen=True)
class DelayInterferometer:
    """
    Bob's delay-line receiver: a Mach-Zehnder whose long arm holds `delay`
    symbol periods, so slot k beats against slot k - delay.
    """

    delay: int = 1
    # The interferometer's own fringe contrast, before the laser's phase walk.
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError("delay must be at least 1 symbol period")

        std.closed("visibility", self.visibility)


@dataclass(frozen=True)
class BasisAnalyser:
    """
    The basis-selecting optics in front of Bob's threshold detectors.
    """

    # The probability a matching-basis photon lands on the wrong detector: a
    # property of the receiver, not the channel. 0.033 is the QBER plateau of
    # Gobby, Yuan and Shields, Appl. Phys. Lett. 84, 3762 (2004), which they
    # attribute to modulator bias; the relabelling is Ma et al. Table 1's.
    misalign: float = 0.033

    def __post_init__(self) -> None:
        std.closed("misalign", self.misalign, 0.5)


@dataclass(frozen=True)
class CoherenceMonitor:
    """
    A coupler tapping `split` into a coherence-monitoring interferometer, the
    rest to the data-line detector.
    """

    # A diagnostic, never a security parameter: the zero-error attack holds its
    # visibility at the honest value. Enters as a flat (1 - split) on the data
    # line and nowhere else.
    split: float = 0.1
    # The 2% e_align of Gao et al., Opt. Express 30, 23783 (2022), Sec. IV.
    misalign: float = 0.02

    def __post_init__(self) -> None:
        std.fraction("split", self.split)

        std.closed("misalign", self.misalign, 0.5)


@dataclass(frozen=True)
class NullingReceiver:
    """
    Bob's two-state front end: he displaces the arriving signal by a
    phase-locked local oscillator so the tested hypothesis nulls, and a click on
    the single threshold gate names the OTHER state. ONE GATE PER PULSE, so
    q.ClickDetector's per-gate dark probability is the per-pulse one here and no
    two-port Y0 is derived, unlike every other click receiver.
    """

    # Fringe contrast of that interference, the same quantity
    # q.DelayInterferometer carries. The weak-flux bit error rate is
    # (1 - visibility)/2, the interferometric law both receivers obey.
    visibility: float = 1.0

    def __post_init__(self) -> None:
        std.closed("visibility", self.visibility)


@dataclass(frozen=True)
class SourceFlaw:
    """
    Alice's phase modulator applies `phi + delta*phi/pi` where she intended
    `phi`, and Bob's analyser sits `tilt` radians off her frame. Those two
    angles are the whole of the loss-tolerant analysis of a flawed qubit
    source: Tamaki, Curty, Kato, Lo & Azuma, Phys. Rev. A 90, 052314 (2014),
    with the device model of Pereira, Curty & Tamaki, npj Quantum Information
    5, 62 (2019).

    NEITHER ANGLE HAS A DEFAULT, AND `tilt` IS THE REASON: at `tilt = 0` Bob's
    modulator carries Alice's own flaw and phase() stops depending on `delta`,
    so a field defaulting to zero would report a flawed source as flaw-free.
    Both papers are written at `tilt = -delta`, where the two modulators'
    errors ADD, and opposed() is that configuration by name.

    THE TILT IS SIGNED AND q.BasisAnalyser.misalign IS NOT, that one being the
    probability a matching-basis photon lands on the wrong detector, in
    [0, 1/2] with no direction in it. A q.Link reads the tilt from HERE and
    refuses an analyser beside it, so one frame is never stated twice.

    A q.Link reaches both analyses: q.FlawedKeying(flaw=...) with
    q.FlawBound(analysis=...), which picks one and has no default.
    """

    # Alice's modulation deviation in radians, domain [0, pi): at pi her two
    # Z-basis states coincide. The engine states that range and its refusal
    # comes through unparaphrased.
    delta: float
    # Bob's analyser frame against Alice's, in radians, SIGNED and wrapped by
    # the engine. -delta is both papers' configuration; 0.0 is a Bob carrying
    # Alice's flaw, which cancels it.
    tilt: float

    @classmethod
    def opposed(cls, delta: float) -> "SourceFlaw":
        """
        The configuration both papers are written at, `tilt = -delta`: Alice's
        and Bob's modulation errors add instead of cancelling, and the
        dark-free phase error is exactly sin^2(delta/2).
        """

        return cls(delta=delta, tilt=-delta)

    def angles(self) -> tuple[float, float, float, float]:
        """
        Alice's four Bloch polar angles in radians, from +z in the X-Z plane,
        her two Z-basis states first: 0 and pi + delta, then pi/2 + delta/2
        and 3*pi/2 + 3*delta/2 for the X-basis pair. Reads `delta` alone --
        `tilt` is Bob's frame and moves none of the four.
        """

        return _core.flaw_angles(self.delta)

    def triangle(self) -> float:
        """
        How well conditioned the inversion phase() runs is: the determinant of
        the three Bloch vectors Alice sends, `2*cos(delta/2) + sin(delta)`.
        2.0 at zero flaw, falling to zero as delta approaches pi, where the
        three states go collinear. A diagnostic; nothing consumes it.
        """

        return _core.flaw_triangle(self.delta)

    def fidelity(self) -> float:
        """
        F(rho_Z, rho_X) between the single-photon states Alice's two bases
        emit. The ONLY thing the standard analysis knows about the flaw; the
        loss-tolerant one never asks for it.
        """

        return _core.flaw_fidelity(self.delta)

    def channel(self, eta: float, dark: float) -> tuple[float, float]:
        """
        `(y1, e_bit)` at these two angles: the single-photon detection
        probability and the Z-basis bit error rate. `eta` is the OVERALL
        transmittance from Alice's output to a detection, the receiver's own
        efficiency included, and `dark` is the per-gate dark probability.

        `y1` depends on neither angle: the basis-independent detection
        efficiency assumption holding inside the model. It is also why
        qkd.attacks's mismatch model does not compose with any of this -- a
        detection-efficiency mismatch is that assumption's violation.
        """

        return _core.flaw_channel(self.delta, eta, dark, self.tilt)

    def phase(self, eta: float, dark: float) -> float:
        """
        The phase error rate under the loss-tolerant analysis, EXACT rather
        than a bound: Bob's rejected-basis rates inverted against Alice's known
        Bloch vectors, an argument that never mentions the loss.

        Flat in `delta` at `tilt = 0` -- the cancellation the class docstring
        names, not a flaw-free source.
        """

        return _core.flaw_phase(self.delta, eta, dark, self.tilt)

    def coin(self, y1: float) -> float:
        """
        The quantum coin imbalance `(1 - F)/(2*y1)` of the standard analysis,
        capped at 1/2, for a single-photon detection probability `y1` supplied
        from outside -- a decoy layer's y1_lo, say.

        THIS IS THE ONLY ENTRY POINT HERE THAT TAKES A y1. tolerant() and
        standard() build their own through channel(), from a qubit model of the
        whole receiver, so the loss-tolerant branch has no decoy seam.
        """

        return _core.flaw_coin(self.delta, y1)

    def tolerant(self, eta: float, dark: float, f_ec: float, q_sift: float) -> tuple[float, float, float, float]:
        """
        `(rate, y1, e_bit, e_phase)` under the LOSS-TOLERANT analysis, bits
        per emitted single photon, at reconciliation inefficiency `f_ec` and
        sifting probability `q_sift`. Asymptotic, single-photon signals.

        The fourth slot is a phase error rate and standard()'s is a coin
        imbalance, so the two four-tuples cannot be lined up slot for slot.
        Their rates can be, and this one is never the smaller; the gap is the
        price of the worst-case assumption, not a margin over an attack.
        """

        return _core.flaw_tolerant(self.delta, eta, dark, self.tilt, f_ec, q_sift)

    def standard(self, eta: float, dark: float, f_ec: float, q_sift: float) -> tuple[float, float, float, float]:
        """
        `(rate, y1, e_bit, coin)` under the STANDARD, non-loss-tolerant
        analysis of the same channel at the same two angles.

        Eve may attribute every non-detection to the favourable coin outcome,
        so the source's imbalance is divided by y1 and collapses with
        distance while tolerant() does not. The fourth slot is that coin, NOT
        a phase error rate; _core.flaw_gllp(e_bit, coin) is the phase error
        bound it implies.
        """

        return _core.flaw_standard(self.delta, eta, dark, self.tilt, f_ec, q_sift)


@dataclass(frozen=True)
class FlawedKeying:
    """
    BB84's four states from a modulator that applies `phi + delta*phi/pi` where
    Alice intended `phi`, on a SINGLE-PHOTON source: Tamaki, Curty, Kato, Lo &
    Azuma, Phys. Rev. A 90, 052314 (2014), with the device model of Pereira,
    Curty & Tamaki, npj Quantum Information 5, 62 (2019).

    THERE IS NO INTENSITY FIELD AND THERE CANNOT BE ONE. Both analyses build
    their own single-photon detection probability out of a qubit model of the
    whole receiver, so a decoy bound has nowhere to enter and none is dropped:
    q.SourceFlaw.coin(y1) is the one entry point taking a supplied y1, and it
    serves the standard analysis alone.

    WHICH ANALYSIS A RUN IS PRICED UNDER IS q.FlawBound's, not this component's.
    """

    # Alice's modulation deviation and Bob's SIGNED frame, carried here rather
    # than borrowed from q.BasisAnalyser, whose `misalign` is an unsigned
    # probability with no direction in it.
    flaw: SourceFlaw
    # P_ZA P_ZB, the share Pereira's Y_Z carries and flaw_tolerant takes
    # separately as q_sift. 1/4 is her Sec. IV, where both parties pick Z half
    # the time; a Z-biased protocol raises it towards 1.
    sift: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.flaw, SourceFlaw):
            raise ValueError(
                "flaw must be a q.SourceFlaw, carrying Alice's modulation "
                "deviation and Bob's SIGNED frame tilt; both papers are written "
                "at q.SourceFlaw.opposed(delta), where tilt = -delta"
            )

        std.unit("sift", self.sift)


@dataclass(frozen=True)
class Laser:
    """
    Alice's source: its phase noise and its intensity noise.
    """

    # LORENTZIAN FWHM in Hz: V_phi = 2 pi dnu t holds for that component alone,
    # so a drift-broadened datasheet figure is far larger. 10 kHz is a
    # commodity ECL/micro-ITLA, viable at 2e-3 excess noise and ~7 dB pilot SNR
    # per Chin et al., npj Quantum Inf. 7, 20 (2021).
    linewidth: float = 10e3
    # Relative intensity noise in dBc/Hz. NONE DECLARES NO INTENSITY NOISE; a
    # value is charged, and budget.rin_sig and budget.rin_lo need a DETECTION
    # BANDWIDTH beside it, so q.Link refuses a declared rin with no
    # q.Homodyne(bandwidth=...) opposite it, and that bandwidth with no rin.
    # -155.0 is the reference figure and is FITTED and unpublished, 10-15 dB
    # better than any datasheet maximum and only a plateau at 10 MHz+; not
    # shot-noise-limited (2 h nu / P is -155.9 dB/Hz at 0.81 mW, 1550 nm).
    rin: float | None = None
    # CENTRE OPTICAL FREQUENCY in Hz, about 193.4e12 at 1550 nm. None is the
    # frequency-degenerate limit, where the carrier-frequency offset is
    # structurally zero. Read only as a DIFFERENCE against Bob's oscillator --
    # q.LocalLO.beat(laser) is the offset _core.run_symbols takes -- so both
    # ends must declare one.
    carrier: float | None = None

    def __post_init__(self) -> None:
        std.nonneg("linewidth", self.linewidth)

        if self.rin is not None:
            std.finite("rin", self.rin)

        if self.carrier is not None:
            std.finite("carrier", self.carrier)

            std.positive("carrier", self.carrier)


@dataclass(frozen=True)
class IQModulator:
    bits: int = 16

    def __post_init__(self) -> None:
        std.atleast("bits", self.bits, 1)


@dataclass(frozen=True)
class Pilots:
    """
    Frequency-multiplexed continuous-wave pilot tone(s). Defaults follow QOSST
    (arXiv:2404.18637) and Hajomer 2024 (arXiv:2305.08156).
    """

    # Tone power relative to the per-quadrature modulation variance.
    power_db: float = 12.0
    tones: int = 1
    # Optical-baseband frequency, outside the quantum band.
    freq: float = 180e6

    def __post_init__(self) -> None:
        std.atleast("tones", self.tones, 1)

        if self.tones > 1:
            raise NotImplementedError("clock recovery from a second tone lands later")

        std.positive("freq", self.freq)


@dataclass(frozen=True)
class PilotPhase:
    """
    Pilot-argument phase recovery.
    """

    # None derives the residual phase-error variance from the symbol pipeline;
    # a value pins it and generates no symbols. Read as -2*ln(E[cos phi]), what
    # budget.phase consumes: NOT the wrapped second moment above ~0.6 rad^2.
    v_err: float | None = None

    def __post_init__(self) -> None:
        if self.v_err is not None:
            std.nonneg("v_err", self.v_err)


@dataclass(frozen=True)
class DSP:
    """
    The receiver's digital chain: carrier-frequency-offset recovery, then
    pilot-argument phase recovery over blocks of `block` symbols.

    THE OFFSET IS NOT CONFIGURED HERE. It is the difference of the two declared
    carriers, q.Laser(carrier=...) minus q.LocalLO(carrier=...), which
    q.LocalLO.beat() takes and the chain then fits by least squares; the fitted
    value comes back in Hz and signed as res.dsp.cfo.
    """

    phase: PilotPhase = field(default_factory=PilotPhase)
    # Symbols per phase-estimation block, trading v_err's two error terms:
    # small blocks track a broad laser but average less pilot noise.
    block: int = 32

    def __post_init__(self) -> None:
        std.atleast("block", self.block, 2)


@dataclass(frozen=True)
class ADC:
    bits: int = 12

    def __post_init__(self) -> None:
        std.atleast("bits", self.bits, 1)


@dataclass(frozen=True)
class LocalLO:
    """
    An oscillator generated in Bob's own lab. The shot-noise unit is fixed by a
    vacuum he measures there and is trustworthy; q.TransmittedLO is the case
    where it is not.
    """

    linewidth: float = 10e3
    # Centre optical frequency in Hz, as on q.Laser and read only against it:
    # the beat IS the carrier-frequency offset the pilot chain estimates, and
    # beat() is the one place the subtraction is written.
    carrier: float | None = None

    def __post_init__(self) -> None:
        std.nonneg("linewidth", self.linewidth)

        if self.carrier is not None:
            std.finite("carrier", self.carrier)

            std.positive("carrier", self.carrier)

    def beat(self, laser: Laser) -> float:
        """
        The carrier-frequency offset in Hz between Alice's laser and this
        oscillator, laser.carrier - self.carrier. SIGNED: the recovery stage
        fits the unwrapped per-block pilot phase against block index by least
        squares, so it recovers a sign as well as a magnitude.

        NOT CLAMPED. The pipeline refuses an offset at or past the Nyquist
        frequency -- 2*|cfo| < symbol_rate, one complex sample per symbol --
        rather than aliasing it.
        """
        if self.carrier is None or laser.carrier is None:
            missing = "q.LocalLO" if self.carrier is None else "q.Laser"
            raise ValueError(
                f"{missing} carries no carrier frequency: an offset is the "
                "DIFFERENCE of two optical frequencies. Declare "
                "q.Laser(carrier=...) and q.LocalLO(carrier=...) together, or "
                "leave both off for the frequency-degenerate limit"
            )

        return laser.carrier - self.carrier


@dataclass(frozen=True)
class ShotNoise:
    """
    The shot-noise unit of a transmitted-oscillator receiver, calibrated against
    operating. `unit` is the whole content: at 1 the calibration still holds,
    and away from 1 every quadrature was divided by the wrong number.
    """

    # LO photon number per pulse at DETECTOR INPUT, before eta.
    arriving: float
    # LO photon number per pulse the shot-noise unit was characterised at.
    calibrated: float
    # arriving / calibrated: the operating unit as a multiple of the
    # characterised one. The detector efficiency cancels out of it.
    unit: float
    # calibrated / arriving, i.e. ASSUMED over ACTUAL, which is the orientation
    # q.attacks.Calibration already carries. 1.0 is no discrepancy.
    ratio: float
    # The oscillator arm's whole transmittance, span and multiplexer together.
    transmittance: float
    # Electronic noise in SNU referred to the OPERATING unit: it grows as the
    # oscillator arm loses light.
    v_el: float


@dataclass(frozen=True)
class TransmittedLO:
    """
    Alice's oscillator, sent down the same fibre as the signal, time- or
    polarisation-multiplexed against it.

    THIS IS A SECURITY-MODEL CHOICE, NOT A CHOICE OF HARDWARE DESCRIPTION.
    With q.LocalLO the shot-noise unit is fixed by a vacuum Bob measures inside
    his own lab; here it is inferred from light that crossed Eve's channel, so
    it moves with the channel and can be steered. shot() returns the calibrated
    unit and the operating one separately, and estimate() is what Alice and Bob
    conclude when the two differ.

    THERE IS NO linewidth FIELD. Signal and oscillator leave one laser and
    cross one fibre, so the laser phase is common mode and cancels; what
    survives is the multiplexing `delay`, multiplied by
    q.Alice(laser=q.Laser(linewidth=...)). phase(linewidth) is that variance;
    feeding it a q.LocalLO linewidth swaps two different quantities.
    """

    # Oscillator photon number per pulse at ALICE's output. 1e9 is the order
    # Lodewyck, Bloch, Garcia-Patron, Fossier, Karpov, Diamanti, Debuisschert,
    # Cerf, Tualle-Brouri, McLaughlin & Grangier, Phys. Rev. A 76, 042305
    # (2007), arXiv:0706.4255, Sec. III.1 run at.
    photons: float = 1e9
    # The oscillator arm's OWN transmittance beyond the span both modes cross.
    # 1.0 is a multiplexer that costs the oscillator nothing.
    mux: float = 1.0
    # Oscillator photon number per pulse the receiver's shot-noise unit was
    # characterised at. None is a back-to-back characterisation, at `photons`.
    calibrated: float | None = None
    # Seconds between the signal pulse and the oscillator pulse. 0.0 is
    # polarisation multiplexing with no time offset, carrying no laser phase
    # noise up to a mode dispersion this component does not model.
    delay: float = 0.0
    # Oscillator suppression in the signal mode, dB: the multiplexer's
    # polarisation extinction, or the switch's on/off contrast. None declares
    # no leak rather than a perfect one.
    extinction: float | None = None

    def __post_init__(self) -> None:
        std.positive("photons", self.photons)

        std.unit("mux", self.mux)

        if self.calibrated is not None:
            std.positive("calibrated", self.calibrated)

        std.nonneg("delay", self.delay)

        if self.extinction is not None and self.extinction <= 0.0:
            raise ValueError("extinction must be a positive dB suppression; None declares no leak")

    @property
    def reference(self) -> float:
        # The characterisation oscillator level, back-to-back where none is
        # named. One spelling, so shot() and leak() cannot disagree.

        return self.photons if self.calibrated is None else self.calibrated

    def shot(self, T: float, v_el: float) -> ShotNoise:
        """
        The shot-noise unit after the oscillator has crossed `T`, as a
        q.ShotNoise. `v_el` is the receiver's electronic noise in SNU AT
        CHARACTERISATION, which is what a datasheet quotes.

        Shot noise is proportional to the oscillator reaching the detector, so
        the electronic noise referred to the operating unit grows as the arm
        loses light: a receiver quoted at v_el = 0.01 back to back carries
        v_el = 1.0 once its oscillator has crossed 20 dB, and the
        untrusted-detector penalty 2*v_el/(eta*T) then grows with the square of
        the span. T = mux = 1 with no separate characterisation level is the
        locally-generated limit: unit = ratio = 1 and v_el unchanged.
        """
        n_bob, t_lo, unit, vel = _core.tlo_shot(self.photons, T, self.mux, self.reference, v_el)

        return ShotNoise(
            arriving=n_bob,
            calibrated=self.reference,
            unit=unit,
            ratio=1.0 / unit,
            transmittance=t_lo,
            v_el=vel,
        )

    def phase(self, linewidth: float) -> float:
        """
        Residual phase-error variance in rad^2 of a co-propagating reference,
        2*pi*linewidth*delay: the Wiener variance a Lorentzian line of that FWHM
        accumulates over the multiplexing delay. Feed it to qkd.budget.phase as
        v_err; it is dimensionless, so no plane attaches to it, and the excess
        noise it produces there is channel-input referred.

        A different quantity from the locally-generated phase budget, which two
        independent lasers set and the pilot chain estimates across a whole DSP
        block. A zero delay returns exactly 0.
        """

        return _core.tlo_phase(linewidth, self.delay)

    def leak(self, T: float) -> float | None:
        """
        The quadrature displacement in SNU amplitude a leaking oscillator puts
        on the signal mode after crossing `T`, or None where no extinction is
        declared. At 1e9 oscillator photons a 60 dB extinction still leaks 1e3
        and displaces the measured quadrature by 63 SNU.

        NOT a variance. What a leak costs is the residual of subtracting that
        displacement, needing the leak's own intensity noise and the estimator's
        tracking, neither of which qkd models -- so the excess noise a leak
        contributes is not returned. The displacement is, because it decides
        whether the receiver is inside its linear range at all.
        """
        if self.extinction is None:
            return None

        share = std.from_db(self.extinction)

        return _core.tlo_leak(self.photons * T * self.mux, share)

    def estimate(self, T: float, xi: float, eta: float, ratio: float) -> tuple[float, float, float, float, float]:
        """
        What Alice and Bob conclude when the shot-noise unit is off by `ratio`,
        assumed over actual, as (T, xi, xi_flat, gap, loss_db). `T`, `xi` and
        `eta` are the TRUE channel and detector, `xi` channel-input referred in
        the ACTUAL unit.

        The first pair is self-consistent: Bob divides his samples by the square
        root of the assumed unit, so the regression slope carries the same
        misnormalisation as the residual variance. `xi_flat` is the same
        quantity written with an unbiased slope, the convention
        qkd.attacks.calib_xi ships, and the two differ by exactly `ratio`.
        `gap` is the input-referred excess noise the misnormalisation hides,
        `loss_db` the apparent extra channel loss beside it.

        Negatives come back as they come: a negative estimated excess noise IS
        the signature.
        """

        return _core.tlo_estimate(T, xi, eta, ratio)

    def covariance(self, v_a: float, T: float, xi: float, eta: float, ratio: float) -> gaussian.State:
        """
        The two-mode state Bob reconstructs from that biased estimate, as a
        qkd.gaussian.State in the internal units -- vacuum 1/2, xpxp -- so the
        shot-noise unit is halved exactly once, at that boundary.

        Refuses a negative estimated excess noise by name, and the bona fide
        condition refuses the rest: a shot-noise unit pushed far enough makes
        the reconstruction not a quantum state at all.
        """

        return gaussian.State(_core.tlo_cov(v_a, T, xi, eta, ratio))

    def monitor(
        self, samples: float, T: float, eta: float, v_el: float, ratio: float = 1.0, sigmas: float = 5.0
    ) -> tuple[float, float, bool, float]:
        """
        The published countermeasure: Bob measures his unit from the oscillator
        he is using, by blocking the signal path on randomly chosen pulses.
        Returns (sigma, gap, detected, residual) -- the resolution at the null,
        the shortfall the monitor has to see, whether it sees it two-sided at
        `sigmas`, and the channel-input referred excess noise still hideable
        underneath the threshold.

        `residual` does not depend on `ratio`: it is a property of the monitor,
        not of the attack. Monitoring closes the loophole only as one over the
        square root of the sample count, so a link sensitive to the unit at the
        part-per-million level needs of order 1e11 vacuum samples.
        """

        return _core.tlo_monitor(int(samples), sigmas, ratio, v_el, T, eta)


@dataclass(frozen=True)
class Bob:
    """
    The receiving party. The refusals below are raised by q.Link, not here:
    what Bob may carry depends on the modulation he sits beside.
    """

    detector: Homodyne | Heterodyne | ClickDetector | PnrDetector | ThresholdArray
    # Every family refuses one; qkd.budget.assemble(adc_bits=...) prices the
    # bit depth instead.
    adc: ADC | None = None
    # q.TransmittedLO constructs here and q.Link refuses it by name: the choice
    # between the two is a security model, not a description of one oscillator.
    lo: LocalLO | TransmittedLO | None = None
    # Between fibre and detector; None for a quadrature receiver, whose
    # detector already describes its own front end.
    receiver: DelayInterferometer | BasisAnalyser | CoherenceMonitor | None = None

    def __post_init__(self) -> None:
        if self.detector is None:
            raise ValueError("Bob needs a detector")


@dataclass(frozen=True)
class Alice:
    laser: Laser | None = None
    iq: IQModulator | None = None
    pilots: Pilots | None = None
    symbol_rate: float = 100e6

    def __post_init__(self) -> None:
        std.positive("symbol_rate", self.symbol_rate)


@dataclass(frozen=True)
class Sender:
    """
    One of the two parties feeding an untrusted relay. Both SEND and neither
    measures; the rate is not symmetric under swapping them.
    """

    # q.GaussianModulation for the covariance midpoint, q.BasisKeying or
    # q.PolarisationKeying for the coincidence analyser; the topology reads
    # which one and picks the family.
    modulation: GaussianModulation | DifferentialPhase | BasisKeying | PolarisationKeying
    # Intensity FWHM of this sender's temporal mode, s, for the
    # two-independent-laser mode matching.
    pulse: float | None = None

    def __post_init__(self) -> None:
        if self.modulation is None:
            raise ValueError("a Sender needs a modulation")

        if self.pulse is not None and self.pulse <= 0.0:
            raise ValueError("pulse must be a positive width")


@dataclass(frozen=True)
class BellDetector:
    """
    The relay's CV Bell measurement: a balanced beamsplitter across the two
    arriving modes, homodyne on both ports in conjugate quadratures.
    """

    # No trusted= flag: the proof never assumes the relay behaved, so eta and
    # v_el fold into the arms as loss and noise Eve holds. Neither defaults.
    eta: float
    v_el: float

    def __post_init__(self) -> None:
        std.unit("eta", self.eta)
        std.nonneg("v_el", self.v_el)


@dataclass(frozen=True)
class BellAnalyser:
    """
    The untrusted midpoint's QUBIT Bell-state measurement: a balanced coupler, a
    polarising beamsplitter on each output and four threshold detectors. What is
    announced is which PAIR of them fired together, and nothing else. Linear
    optics separates two of the four Bell states, so a perfect one announces at
    most half the time. q.BellDetector is the continuous-variable sibling, two
    homodynes rather than four counters, and shares no formula with this.
    """

    eta: float
    # PER DETECTOR AND PER GATE, as on ClickDetector. The literature usually
    # quotes the RECEIVER total as Y0 = 6.02e-6; Ma & Razavi quote
    # p_d = 3.0e-6 per detector and say it is about half of it. This field is
    # the per-detector figure, so passing a receiver total doubles the floor.
    dark: float
    # Interferometric error of a matched announcement. TWO NUMBERS, not one:
    # the key basis needs no Hong-Ou-Mandel indistinguishability and the test
    # basis does, so one value for both makes a midpoint rate look good.
    # misalign_test=None derives it from the two senders' mode overlap;
    # misalign=None derives nothing, the key basis having no interference
    # visibility to read one off.
    misalign: float | None = None
    misalign_test: float | None = None
    # HOW MANY OF THE FOUR BELL STATES THIS ANALYSER ANNOUNCES; linear optics
    # separates two. NO DEFAULT: a finite key length is stated PER ANNOUNCED
    # STATE with eps_sec = sum_k eps_k, so the layer runs it `states` times at
    # eps_sec/`states` and adds. The asymptotic rate does not read it.
    states: int | None = None

    def __post_init__(self) -> None:
        std.unit("eta", self.eta)

        std.fraction("dark", self.dark)

        for name in ("misalign", "misalign_test"):
            got = getattr(self, name)
            if got is not None:
                std.closed(name, got, 0.5)

        if self.states is not None and self.states not in (1, 2):
            raise ValueError(
                "states must be 1 or 2: linear optics separates two of the four "
                "Bell states, so there is no third announcement for a key "
                "length to be summed over"
            )


@dataclass(frozen=True)
class TestBasisBound:
    """
    Asymptotic decoy-state security for a Bell-analysed midpoint: the key rides
    on announcements where BOTH senders emitted exactly one photon in the KEY
    basis, and its phase error is that same pair's error rate measured in a
    separate TEST basis. Xu, Curty, Qi & Lo, New J. Phys. 15, 113007 (2013),
    Eq. (1); the two bases are why it carries no e_phase field.
    """

    # Error-correction inefficiency over the Shannon limit, multiplying an
    # entropy that is SUBTRACTED, so below 1 inflates the rate. 1.16 is the
    # value this whole literature runs at.
    f: float = 1.16
    # None is the asymptotic midpoint rate. A q.RelayBlock asks for a key
    # LENGTH instead. It is not a q.KeyBlock: a midpoint's block counts PAIRS
    # rather than pulses, spends a differently split secrecy budget, and states
    # its length per announced Bell state.
    block: "RelayBlock | None" = None

    def __post_init__(self) -> None:
        std.atleast("f", self.f, 1.0)

        if self.block is not None and not isinstance(self.block, RelayBlock):
            raise ValueError(
                "block must be a q.RelayBlock, not a q.KeyBlock: this counts "
                "pulse PAIRS from two senders where a q.KeyBlock counts pulses "
                "from one, and splits a differently sized secrecy budget"
            )

    # NO e_phase FIELD, unlike q.PhaseBound (COW), which takes one because
    # nothing derives it: here the test basis measures it through the decoy
    # layer.


@dataclass(frozen=True)
class Relay:
    """
    The untrusted middle node, carrying the Bell measurement the two senders'
    modes are swapped through. WHICH measurement it carries selects the family:
    a q.BellDetector conditions a Gaussian state and takes q.GaussianModulation
    senders, a q.BellAnalyser counts polarisation coincidences and takes
    q.BasisKeying ones. They share no formula.
    """

    bell: BellDetector | BellAnalyser

    def __post_init__(self) -> None:
        if self.bell is None:
            raise ValueError(
                "a Relay takes a measurement: bell=q.BellDetector(...) for the "
                "continuous-variable relay, or bell=q.BellAnalyser(...) for "
                "the qubit one"
            )


@dataclass(frozen=True)
class CorrelatedEnvironment:
    """
    ONE NAMED EVE: the cross-covariance between the two relay arms' environment
    modes. Omitting it from the link is the independent pair.
    """

    # An assertion about the attack faced; SwapResult.attack is per-attack,
    # unclamped, never a bound on key_rate. x and p are capped jointly and both
    # need an environment above the vacuum: from channel xi or v_el, not eta.
    x: float
    p: float


@dataclass(frozen=True)
class Asymptotic:
    beta: float = 0.95

    def __post_init__(self) -> None:
        std.unit("beta", self.beta)


@dataclass(frozen=True)
class FiniteSize:
    """
    Composable finite-size security against collective attacks.
    """

    # Reconciliation efficiency.
    beta: float = 0.95
    # Failure probability of each step of the epsilon budget.
    eps: float = 1e-10
    # Block size in symbols.
    n: float = 1e9
    # Share of the block disclosed for parameter estimation.
    pe_fraction: float = 0.5
    # Frame error rate. No default: it belongs to one LDPC code at one SNR
    # targeting one beta. None means failures are NOT MODELLED; a value scales
    # the whole reported rate by (1 - fer).
    fer: float | None = None

    def __post_init__(self) -> None:
        std.unit("beta", self.beta)

        std.open_unit("eps", self.eps)

        std.positive("n", self.n)

        std.open_unit("pe_fraction", self.pe_fraction)

        if self.fer is not None:
            std.closed("fer", self.fer)


@dataclass(frozen=True)
class EntropyBound(std.Barred):
    """
    One run of the same two-step proof over Kanitschar Eq. (21)'s RELAXED
    feasible set, which is the one input of their Theorem 6 Eq. (9) that a
    q.Certificate may not stand in for.

    `entropy` is min H(X|E') in bits per KEPT round, the plane
    q.security.keylength takes it on. `bound` is the same minimum in bits per
    PULSE and is the proof; `upper` is step 1's Frank-Wolfe value and PROVES
    NOTHING. `viol` is reported and not charged, an inequality set needing no
    perturbation where an equality set does.

    IT IS NOT A KEY RATE AND CARRIES NO ERROR-CORRECTION COST. The equality set
    a q.Certificate minimises over is CONTAINED in this one, so its minimum is
    the LARGER and substituting one for the other overstates a key length
    rather than loosening it. `key`, `key_rate` and `rate` raise.
    """

    _barred = ("key", "key_rate", "rate")
    _refusal = (
        "an EntropyBound has no key: it is min H(X|E') over Kanitschar "
        "Eq. (21)'s relaxed set, in bits per kept round. "
        "q.security.keylength(ledger, entropy=..., ...) on family 'dmcs' makes "
        "a length of it. Read `entropy` by name, and check `zeta` before "
        "quoting one near the numerical floor"
    )

    entropy: float
    bound: float
    upper: float
    p_pass: float
    delta_ec: float
    viol: float
    zeta: float
    steps: int
    status: str


@dataclass(frozen=True)
class Certificate(std.Barred):
    """
    One run of the two-step relative-entropy proof, every rate in bits per
    pulse. THE TWO STEPS ARE SEPARATE FIELDS AND ONE IS NOT THE OTHER.

    `key` is the number a caller may quote: step 2's certified entropy less the
    error-correction cost. `bound` is that entropy on its own, the linearised
    dual value weak duality makes a lower bound whatever the interior-point path
    did, so a run that stalls returns a LOOSER key and never a wrong one.

    `upper` is step 1, the Frank-Wolfe value less the same cost. It is an UPPER
    bound on the key rate and PROVES NOTHING; `key_rate` and `rate` raise
    rather than resolving to either.

    TWO LIMITS TRAVEL WITH EVERY NUMBER TAKEN FROM THIS. It is rigorous GIVEN
    THE PHOTON-NUMBER CUTOFF, which the protocol's source paper calls a working
    assumption in as many words: `viol` is the largest constraint violation at
    the step-1 state, so raise q.CertifiedBound(cutoff=...) until `key` stops
    moving and do not call the result cutoff-free. And `zeta` is a NUMERICAL
    FLOOR, the continuity price of the eps perturbation -- of order 1e-6 bit at
    eps = 1e-10 and 1e-9 at 1e-13 -- so a `key` below about that size is that
    price rather than physics.
    """

    _barred = ("key_rate", "rate")
    _refusal = (
        "a Certificate has no key_rate: `key` is step 2 less error correction "
        "and is the certified rate; `upper` is step 1's Frank-Wolfe value and "
        "is an UPPER bound that proves nothing. Read the one you mean by name, "
        "and check `zeta` before quoting a key near the numerical floor"
    )

    key: float
    bound: float
    upper: float
    p_pass: float
    delta_ec: float
    viol: float
    zeta: float
    steps: int
    status: str


@dataclass(frozen=True)
class CertifiedBound:
    """
    Asymptotic security for a phase-keyed link by NUMERICAL PROOF rather than a
    closed form: the relative entropy minimised over the states compatible with
    the observations, Frank-Wolfe first and then the linearisation whose dual
    value is the rigorous bound. It stands where q.Asymptotic stands on a
    q.PhaseShiftKeying link and is not comparable to it term by term -- the
    closed form prices the alphabet against a Gaussian optimum and this one
    does not.

    At beta = 0.95 with the amplitude optimised, the closed form crosses zero
    at 26.5 km at xi = 0.01; this reaches 142 km at a 1e-4 bit/pulse floor.
    Larger figures sit below a 1e-6 bit/pulse floor and are the continuity
    price q.Certificate reports as `zeta`, not physics.

    TWO RECEIVER MODELS, one method each, and they are not the same bound.
    certify() is untrusted, as the closed form is: the receiver folds in
    through the total transmittance and the excess noise and nowhere else, so
    it takes the folded pair rather than a detector. certify_trusted() is Lin
    & Lutkenhaus, Phys. Rev. Applied 14, 064030 (2020), which puts Bob's
    efficiency and electronic noise in his POVM instead, so it takes the span
    and the receiver apart and must not be handed a pre-multiplied pair.
    Trusting the detector buys distance, not rigour -- the photon-number
    cutoff below is the same working assumption either way.

    PURE LOSS IS REFUSED, by the engine and in its own words: at xi = 0 the
    two-mode state goes rank deficient and the certificate comes back valid and
    worthless.
    """

    # Fock levels Bob's space is truncated at. NO DEFAULT: the bound is
    # rigorous given this number and the answer moves with it. The engine's own
    # range is [2, 30] and the cost is cubic in m^2*(cutoff + 1).
    cutoff: int
    # The postselection pair, both zero for no postselection at all, where the
    # region operators sum to the identity. `cut` is the radius below which Bob
    # discards and `phase` the ANGULAR guard band on each side of a sector
    # boundary, in radians -- a guard angle, not a phase noise.
    cut: float = 0.0
    phase: float = 0.0
    # Reconciliation efficiency, the share of the mutual information reached.
    beta: float = 0.95
    # The perturbation the continuity bound is taken at, paid for in `zeta`:
    # lowering it tightens the floor and loosens nothing.
    eps: float = 1e-10
    # Frank-Wolfe iterations. Step 2 linearises about wherever step 1 stopped,
    # so this trades run time against how loose the certificate is, never
    # against whether there is one.
    steps: int = 15
    gaptol: float = 1e-8

    def __post_init__(self) -> None:
        std.atleast("cutoff", self.cutoff, 2)

        std.nonneg("cut", self.cut)

        std.nonneg("phase", self.phase)

        std.unit("beta", self.beta)

        std.open_unit("eps", self.eps)

        std.atleast("steps", self.steps, 1)

        std.positive("gaptol", self.gaptol)

    @property
    def _knobs(self) -> tuple:
        """
        The tail dm_secure, dm_trusted and dm_relaxed each end with, in order:
        (cut, phase, beta, eps, steps, gaptol). ONE SPELLING, so the
        postselection pair and where the interior-point path stops cannot
        differ between the three bounds this component reaches.
        """

        return (self.cut, self.phase, self.beta, self.eps, self.steps, self.gaptol)

    def certify(self, modulation: PhaseShiftKeying, eta: float, xi: float) -> Certificate:
        """
        The certified rate for a q.PhaseShiftKeying constellation, as a
        q.Certificate.

        `eta` is the TOTAL transmittance from Alice's output to Bob's
        measurement -- span and receiver together, the receiver being untrusted
        here -- and `xi` the excess noise at the channel input in the folding
        the closed form uses, xi + 2*v_el/(eta*T). A bare span transmittance
        with a separate detector efficiency is the referring-plane error, and it
        makes the rate look good.

        The engine's refusals reach the caller unchanged: a cutoff outside its
        range, a constellation outside [2, 8] states, an eps outside the
        continuity theorem's domain, a dual point it could not certify, and pure
        loss.
        """

        return Certificate(
            *_core.dm_secure(
                modulation.states,
                modulation.alpha,
                eta,
                xi,
                self.cutoff,
                *self._knobs,
            )
        )

    def certify_trusted(
        self,
        modulation: PhaseShiftKeying,
        eta: float,
        xi: float,
        eta_d: float,
        v_el: float,
    ) -> Certificate:
        """
        The certified rate with Bob's receiver TRUSTED, as a q.Certificate.

        `eta` is the CHANNEL transmittance alone and `xi` the excess noise at
        its input; `eta_d` and `v_el` are Bob's calibrated efficiency and
        electronic noise, which stay his. Pre-multiplying the receiver into
        `eta` is the referring-plane error and counts the loss twice --
        certify() is the method that wants the folded pair.

        The saving is entirely in where `v_el` sits: untrusted it is referred
        back to the channel input as xi + 2*v_el/(eta_d*eta) and so grows as
        1/eta, and trusted it stays an additive term in Bob's own moments.
        """

        return Certificate(
            *_core.dm_trusted(
                modulation.states,
                modulation.alpha,
                eta,
                xi,
                self.cutoff,
                eta_d,
                v_el,
                *self._knobs,
            )
        )

    # THE FIVE BELOW PRICE A BLOCK AND NONE RETURNS A KEY LENGTH; q.Link reaches
    # no finite-size discrete-modulation rate at all. Kanitschar, George, Lin,
    # Upadhyaya & Lutkenhaus, PRX Quantum 4, 040306 (2023), arXiv:2301.08686,
    # Theorem 6 Eq. (9) needs four pieces -- the dimension-reduction charge, the
    # energy test, the acceptance test, and a minimum of H(X|E') over their
    # RELAXED feasible set of Eq. (21) -- and all four ship, relaxed() being the
    # fourth. Substituting certify()'s equality-set minimum into Eq. (9)
    # over-claims by 3.17% (1.894405 against 1.834309 bit/kept round at 20 km,
    # alpha 0.7, nc 6) rather than being merely loose.
    #
    # WHAT REFUSES IS THE PROTOCOL, NOT THE ARITHMETIC. dm_relaxed wants w, two
    # acceptance half-widths and two clipped operator norms, and no q.Link
    # sacrifices k_T rounds to an energy test, carries that split, or clips a
    # detector at a soft limit -- caller obligations of exactly the standing
    # certify()'s photon-number cutoff has. _core.dm_length refuses
    # source="certified" for TWO reasons: certify()'s set is CONTAINED in the
    # relaxed one, and the two are written over different observables,
    # {q, p, n, d} against Eq. (21)'s displaced {n_beta, n_beta^2}. The epsilon
    # shape is qkd.security family 'dmcs'.

    def price(self, modulation: PhaseShiftKeying, w: float) -> float:
        """
        Kanitschar Eq. (4)'s dimension-reduction charge Delta(w) in BITS PER
        PULSE, for a weight `w` claimed outside the photon-number cutoff.

        NOT a small correction and never droppable: over a four-symbol key map
        it is 0.101 bit at w = 1e-4 and still 1.34e-2 bit at 1e-6. The engine's
        `alphabet` is |Z|, the size of the KEY MAP, read off the modulation
        because q.PhaseShiftKeying describes the standard M-PSK key map. The two
        need not coincide, and a link whose key map is coarser than its
        constellation does not reach this method.
        """

        return _core.dm_price(w, modulation.states)

    def energy(self, beta_test: float, k_t: float, l_t: float, w: float) -> tuple:
        """
        Kanitschar Theorem 3, the noise-robust energy test, as
        `(log2 eps_ET, r, D)` for `k_t` rounds sacrificed to a heterodyne test
        at most `l_t` of which may land OUTSIDE the disc of squared radius
        `beta_test`, claiming the weight `w` outside the cutoff.

        THE FIRST RETURN IS A BASE-TWO LOGARITHM, not a probability: at the
        paper's own l_T/k_T and a block in the tens of billions the epsilon is
        not a representable f64, and returning 0.0 as a security parameter
        reads as unconditional security. The cutoff is this component's own, so
        test and certificate cannot be sized at two different ones.

        IT DOES NOT RUN THE TEST. Sacrificing `k_t` rounds is a protocol step
        and no q.Link has one, so the number is rigorous GIVEN that the test was
        run and passed on `k_t` rounds drawn at random from the block -- the
        same standing as `cutoff` itself.
        """

        return _core.dm_energy(self.cutoff, beta_test, k_t, l_t, w)

    def weight(self, beta_test: float, k_t: float, l_t: float, target: float) -> tuple:
        """
        `(w, r)`: the smallest weight compatible with a target energy-test
        failure probability, by inverting Kanitschar Eq. (5). `target` is
        `log2 eps_ET` and is NEGATIVE, matching energy()'s first return.

        Their Sec. VI B's w_eps, the middle term of their shipped rule
        `w = max(w_exp, w_eps, w_min)`.
        """

        return _core.dm_wchoice(self.cutoff, beta_test, k_t, l_t, target)

    def expected(self, modulation: PhaseShiftKeying, eta: float, xi: float) -> float:
        """
        The weight `w_exp` the HONEST channel leaves outside the photon-number
        cutoff, `1 - Tr rhobar`, the first term of Kanitschar Sec. VI B's rule
        `w = max(w_exp, w_eps, w_min)`.

        Exact rather than a bound: the truncation drops exactly that weight, so
        a declared `w` below this one is a claim the honest implementation
        itself fails.
        """

        return _core.dm_weight(modulation.states, modulation.alpha, eta, xi, self.cutoff)

    def clip(self, m_soft: float, beta: float) -> tuple:
        """
        The clipped infinity norms `(x_n, x_n2)` of the displaced number
        operator and its square over a soft detection limit `m_soft`, at a
        displacement of modulus `beta`. These are the two norms relaxed() takes
        and the `x` accept() will not derive.

        Kanitschar Eq. (25) clips a heterodyne outcome componentwise into
        [-M, M]^2, and these are the norms that clipping leaves. At beta = 0
        they are 2M^2 - 1 and 4M^4 - 6M^2 + 1, which are NOT the M^2 - 1/2 and
        M^4 - M^2/2 their Sec. V F prints -- 49 and 2351 against 24.5 and 612.5
        at their own M = 5. A narrower norm narrows the acceptance test and
        RAISES the key rate, so the printed constants are not the conservative
        reading.
        """

        return _core.dm_clip(m_soft, beta)

    def relaxed(
        self,
        modulation: PhaseShiftKeying,
        eta: float,
        xi: float,
        w: float,
        widths: tuple,
        norms: tuple,
    ) -> EntropyBound:
        """
        The certified minimum of H(X|E') over Kanitschar Eq. (21)'s RELAXED
        feasible set, as a q.EntropyBound. What q.security.keylength takes on
        family 'dmcs'; certify()'s minimum is over the CONTAINED equality set
        and overstates a length rather than loosening it.

        `widths` is Theorem 4's two acceptance half-widths `(mu_n, mu_n2)`,
        which accept() computes; `norms` is the pair of clipped infinity norms
        `(x_n, x_n2)`, which clip() computes. `w` is DECLARED, not derived --
        the same weight the energy test claims, at Sec. VI B's
        `w = max(w_exp, w_eps, w_min)`, whose first two terms are expected()
        and weight().

        UNTRUSTED, IDEAL RECEIVER ONLY, so `eta` is the channel and the
        receiver together and `xi` the excess noise at the channel input, the
        same folding certify() takes. The trusted receiver's relaxed set is not
        this one.
        """
        mu_n, mu_n2 = widths
        x_n, x_n2 = norms

        return EntropyBound(
            *_core.dm_relaxed(
                modulation.states,
                modulation.alpha,
                eta,
                xi,
                self.cutoff,
                w,
                mu_n,
                mu_n2,
                x_n,
                x_n2,
                *self._knobs,
            )
        )

    def accept(self, x: float, tests: float, eps_at: float, psd: bool = False) -> float:
        """
        Kanitschar Theorem 4's acceptance half-width mu_X for ONE observable,
        over `tests` rounds spent on that observable alone. `psd` says the
        operator is positive semidefinite with range [0, x] rather than
        [-x, x], which quarters the variance.

        `x` IS ||X||_inf AND IS THE CALLER'S. Hoeffding needs a bounded
        observable and none of the operators this bound is written over is
        bounded; Kanitschar buy boundedness by coarse-graining the detector at a
        soft limit and read `x` off the CLIPPED symbol. Nothing in qkd clips --
        q.Link refuses q.Heterodyne.clip by name -- so an unclipped operator
        norm is unavailable rather than conservative.
        """

        return _core.dm_accept(x, tests, eps_at, psd)


@dataclass(frozen=True)
class IndividualAttack:
    """
    Asymptotic security against individual attacks on a phase-keyed pulse
    train (Waks, Takesue & Yamamoto, PRA 73, 012344).
    """

    # None derives the error rate from the simulated interferometer; a value
    # pins it and short-circuits the click simulation into the closed form.
    qber: float | None = None
    # Error-correction inefficiency over the Shannon limit. 1.16 is qkd's,
    # from Lutkenhaus, Phys. Rev. A 61, 052304 (2000), Table I; WTY and
    # Diamanti et al. both evaluate at f = 1, which recovers their rates. The
    # single-photon zero-crossing is 6.09% at f = 1 and 5.63% here.
    f: float = 1.16

    def __post_init__(self) -> None:
        if self.qber is not None:
            std.fraction("qber", self.qber, 0.5)

        std.atleast("f", self.f, 1.0)


# The four security dataclasses carrying a block= field re-spell the
# isinstance guard rather than share a helper the way Connector/Splice/Coupling
# share _optic: one two-line check over four sites is +1 line behind an
# indirection, and assertFails pins the refusal literal against rewording.
@dataclass(frozen=True)
class KeyBlock:
    """
    A finite block of emitted pulses and its composable epsilon budget, turning
    a decoy-state rate into a key LENGTH (Lim et al., PRA 89, 022307 (2014)).
    """

    # Pulses EMITTED, decoy and test-basis slots included -- not sifted bits
    # and not key bits. The reported length carries the block size rather than
    # dividing it away.
    n: float = 1e10
    # Secrecy parameter. Lim's numerics use 1e-10; their budget collapses to
    # eps_sec = 21*eps, so each of the twenty-one bounds fails at eps_sec/21.
    eps_sec: float = 1e-10
    # Correctness parameter, the residual chance a reconciled key differs. A
    # declared input kept separate from eps_sec: qkd runs no reconciliation.
    eps_cor: float = 1e-15
    # Frame error rate, same meaning as on q.FiniteSize: None is NOT MODELLED,
    # a value scales the reported length by (1 - fer).
    fer: float | None = None

    def __post_init__(self) -> None:
        std.positive("n", self.n)

        for name, eps in (("eps_sec", self.eps_sec), ("eps_cor", self.eps_cor)):
            std.open_unit(name, eps)

        if self.fer is not None:
            std.closed("fer", self.fer)


@dataclass(frozen=True)
class GaussianBlock:
    """
    A finite block for a Gaussian relay midpoint, turning the midpoint's
    asymptotic rate into a finite-size RATE. Papanastasiou, Ottaviani &
    Pirandola, Phys. Rev. A 96, 042332 (2017).

    IT IS NOT A q.RelayBlock. That one counts pulse PAIRS and returns a key
    LENGTH per announced Bell state; this one counts uses and returns a rate
    per use, so nothing here is ever divided by `n`.

    ITS EPSILONS DO NOT COMPOSE. eps_smooth and eps_pa enter the smooth
    min-entropy penalty alone, there is no correctness term, and `sigma` is a
    confidence COEFFICIENT in standard deviations rather than a probability, as
    POP17 state it.
    """

    # Total channel uses, key and parameter-estimation rounds together.
    n: float = 1e9
    # Share of them spent on parameter estimation rather than key.
    pe_fraction: float = 0.5
    # Standard deviations the four confidence intervals are taken at. NO
    # DEFAULT: it is not a probability, the bound moves with it, and POP17 run
    # 6.5 while quoting 1e-10 beside it -- the same reason
    # q.CertifiedBound.cutoff has none.
    sigma: float = None
    eps_smooth: float = 1e-10
    eps_pa: float = 1e-10
    # The attack class the interval statistics are claimed against. Left
    # unvalidated so the engine's refusal reaches the caller verbatim: POP17
    # cover "gaussian" alone, and name what "collective" and "coherent" need.
    attack: str = "gaussian"

    def __post_init__(self) -> None:
        std.positive("n", self.n)

        std.open_unit("pe_fraction", self.pe_fraction)

        if self.sigma is None:
            raise ValueError(
                "sigma has no default: it is a confidence coefficient in "
                "standard deviations, not a probability. POP17 run 6.5"
            )

        std.positive("sigma", self.sigma)

        for name, eps in (("eps_smooth", self.eps_smooth), ("eps_pa", self.eps_pa)):
            std.open_unit(name, eps)


@dataclass(frozen=True)
class TwoModeBound:
    """
    Finite-size security for a Gaussian relay midpoint, where q.Asymptotic takes
    the asymptotic bound. The block is REQUIRED, leaving q.Asymptotic the only
    route to the asymptotic number.
    """

    block: "GaussianBlock"
    # Reconciliation efficiency, the share of the mutual information reached.
    beta: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.block, GaussianBlock):
            raise ValueError("block must be a q.GaussianBlock")

        std.unit("beta", self.beta)


@dataclass(frozen=True)
class RelayBlock:
    """
    A finite block for a Bell-analysed midpoint, turning the midpoint's
    asymptotic rate into a key LENGTH in bits.

    IT IS NOT A q.KeyBlock AND THE TWO ARE NOT INTERCHANGEABLE. Its `n` counts
    emitted pulse PAIRS where a single-sender block counts pulses, and its
    secrecy budget splits into 266 equal shares inside the engine where Lim's
    splits into 21: swapping them reports a length at the wrong budget and the
    wrong block size at once.

    THE LENGTH IS PER ANNOUNCED BELL STATE. `eps_sec` is the budget for the
    WHOLE announcement set and per_state() divides it by the count
    q.BellAnalyser(states=...) declares, so a relay separating two Bell states
    runs the length twice at half the budget and adds.
    """

    # Pulse PAIRS emitted, decoy and test-basis rounds included -- not
    # announcements, not sifted pairs and not key bits. The reported length
    # carries the block size rather than dividing it away.
    n: float = 1e12
    # Secrecy parameter for the whole announcement set. per_state() is the one
    # place it is divided; the engine's own split into its term count is the
    # paper's.
    eps_sec: float = 1e-10
    # Correctness parameter, the residual chance a reconciled key differs. A
    # declared input as on q.KeyBlock: qkd runs no reconciliation.
    eps_cor: float = 1e-15
    # Share of the signal-setting key-basis announcements forming the CODE
    # STRING, the rest disclosed to measure its error rate. Barred at zero,
    # which leaves no key, and admitted at 1.
    code: float = 0.9
    # Frame error rate, same meaning as on q.KeyBlock: None is NOT MODELLED, a
    # value scales the reported length by (1 - fer).
    fer: float | None = None

    def __post_init__(self) -> None:
        std.positive("n", self.n)

        for name, eps in (("eps_sec", self.eps_sec), ("eps_cor", self.eps_cor)):
            std.open_unit(name, eps)

        std.unit("code", self.code)

        if self.fer is not None:
            std.closed("fer", self.fer)

    def per_state(self, states: int | None) -> float:
        """
        The secrecy parameter one announcement is run at, eps_sec/states, where
        `states` is what q.BellAnalyser declares. ONE SPELLING OF THE DIVISION,
        so a sum over two announcements cannot be assembled at the full budget
        twice -- which would report a length secure at 2*eps_sec while claiming
        eps_sec.
        """
        if states is None:
            raise ValueError(
                "q.BellAnalyser carries no states, so there is nothing to "
                "divide the secrecy parameter by: a length is stated per "
                "announced Bell state, so how many the analyser announces is "
                "part of the claim. Declare q.BellAnalyser(states=1), or "
                "states=2 for the pair linear optics separates"
            )

        if states not in (1, 2):
            raise ValueError(
                "states must be 1 or 2: linear optics separates two of the four "
                "Bell states, so there is no third announcement to sum over"
            )

        return self.eps_sec / states


@dataclass(frozen=True)
class SplittingAttack:
    """
    Security against photon-number splitting: every multiphoton pulse conceded,
    so amplification pays only on the decoy-bounded singles.
    """

    # Error-correction inefficiency over the Shannon limit. 1.22 is
    # Lutkenhaus, Phys. Rev. A 61, 052304 (2000), Table I at e = 0.10, adopted
    # by Ma, Qi, Zhao and Lo; it is used at every error rate, so a link at the
    # default 3.3% misalignment is charged ~5% more leakage than its band.
    f: float = 1.22
    # None is the asymptotic GLLP rate. A q.KeyBlock switches the path to Lim's
    # finite key length, which can only ever be smaller.
    block: "KeyBlock | None" = None

    def __post_init__(self) -> None:
        std.atleast("f", self.f, 1.0)

        if self.block is not None and not isinstance(self.block, KeyBlock):
            raise ValueError("block must be a q.KeyBlock")


@dataclass(frozen=True)
class PhaseBound:
    """
    Security from an externally supplied upper bound on the phase-error rate.
    """

    # Not derivable from the monitoring visibility: the zero-error attack holds
    # visibility at 1 through a total break. Strictly positive -- zero claims
    # Eve learns nothing, worth ~1.5x the rate over a realistic 0.05.
    e_phase: float
    # Error-correction inefficiency over the Shannon limit. 1.1 is Gao et al.,
    # Opt. Express 30, 23783 (2022), Sec. IV, assumed rather than measured;
    # Korzh et al., Nat. Photon. 9, 163 (2015) measure 1.217 to 1.287 over
    # 104-307 km on this protocol, so the default is 10-17% optimistic.
    f: float = 1.1
    # None is the asymptotic rate. A q.KeyBlock asks for a key LENGTH and q.Link
    # REFUSES it, the engine calling the question malformed rather than
    # unimplemented: the supplied e_phase is the dominant term and sits outside
    # any budget a finite-size correction could attach to. The field exists so
    # that refusal is reachable, as q.DiscriminationBound.block is. The
    # four-sequence variant that DERIVES the phase error is
    # q.security.keylength on family 'cow-vacuum', a different protocol rather
    # than a setting on this one.
    block: "KeyBlock | None" = None

    def __post_init__(self) -> None:
        std.unit("e_phase", self.e_phase, 0.5)

        std.atleast("f", self.f, 1.0)

        if self.block is not None and not isinstance(self.block, KeyBlock):
            raise ValueError("block must be a q.KeyBlock")


@dataclass(frozen=True)
class DiscriminationBound:
    """
    Asymptotic security for a two-state link. Eve's best strategy is unambiguous
    state discrimination, so the phase error rate is priced by what that
    measurement could have done at this loss and this overlap: the bound reaches
    1/2 exactly at loss = overlap, where the attack takes the protocol over, and
    the rate arrives at zero on its own with nothing refused.
    """

    # None DERIVES the phase error rate from the overlap, the loss and the
    # observed bit error; a value is a FLOOR under that derived bound. On
    # q.TwoStateKeying(reference=True) there is nothing to derive and this is
    # the supplied bound outright, as q.PhaseBound's is.
    e_phase: float | None = None
    # Error-correction inefficiency over the Shannon limit. The published
    # thresholds are all the f = 1 column; 1.16 is qkd's, from Lutkenhaus,
    # Phys. Rev. A 61, 052304 (2000), Table I.
    f: float = 1.16
    # None is the asymptotic rate. A q.KeyBlock asks for a key LENGTH and is
    # REFUSED on both branches, for two different reasons the engine states
    # when q.Link runs; the field exists so that refusal is reachable.
    block: "KeyBlock | None" = None

    def __post_init__(self) -> None:
        if self.e_phase is not None:
            std.closed("e_phase", self.e_phase, 0.5)

        std.atleast("f", self.f, 1.0)

        if self.block is not None and not isinstance(self.block, KeyBlock):
            raise ValueError("block must be a q.KeyBlock")


@dataclass(frozen=True)
class FlawBound:
    """
    Which of the two analyses of a flawed source a q.Link is priced under.

    `analysis` HAS NO DEFAULT AND THAT IS THE WHOLE POINT. The two are analyses
    of one dataset and their four-tuples are shaped so they cannot be lined up
    -- flaw_tolerant's fourth slot is a phase error rate, flaw_standard's is a
    quantum coin imbalance -- so a link that chose for you would report one
    number and hide that the other exists. explain() carries the name beside it.

    'tolerant' reads the phase error EXACTLY, by inverting Bob's rejected-basis
    rates against Alice's known Bloch vectors; nothing in that argument
    mentions the loss. 'standard' bounds the same quantity through a quantum
    coin whose imbalance Eve may attribute to every non-detection, so it is
    divided by y1 and collapses with distance. They agree to round-off at zero
    flaw, and the gap between their rates is the price of the worst-case
    assumption, never a margin over an attack.
    """

    analysis: Literal["tolerant", "standard"]
    # Error-correction inefficiency over the Shannon limit. 1.16 is Pereira
    # Sec. IV's own f and qkd's house value, from Lutkenhaus, Phys. Rev. A 61,
    # 052304 (2000), Table I.
    f: float = 1.16
    # None is the asymptotic rate, which is the only one src/flaws.rs ships. A
    # q.KeyBlock asks for a key LENGTH and q.Link REFUSES it by name; the field
    # exists so that refusal is reachable rather than absent.
    block: "KeyBlock | None" = None

    def __post_init__(self) -> None:
        if self.analysis not in _ANALYSES:
            raise ValueError(
                "analysis must be 'tolerant' (Tamaki's exact inversion of the "
                "rejected data) or 'standard' (the quantum-coin bound Eve may "
                "amplify by the whole channel loss): there is no default and no "
                "link picks for you"
            )

        std.atleast("f", self.f, 1.0)

        if self.block is not None and not isinstance(self.block, KeyBlock):
            raise ValueError("block must be a q.KeyBlock")
