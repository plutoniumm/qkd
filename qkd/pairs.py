from dataclasses import dataclass

from . import _core, std
from .components import (
    BasisAnalyser,
    ClickDetector,
    Fiber,
    PairSource,
    SymmetryBound,
    ViolationBound,
)

_q = std.q

# Two detection-efficiency thresholds for a device-independent CHSH claim: ONE
# requirement under two source models, so neither may be quoted without naming
# its model. DI_ETA_PARTIAL, 86.5479% at plain CHSH with no noise
# preprocessing, is for a partially entangled two-qubit state optimised over
# its Schmidt angle: Woodhead, Acin & Pironio, "Device-independent quantum key
# distribution with asymmetric CHSH inequalities", Quantum 5, 443 (2021), table
# tab:eta-psitheta. DI_ETA_SINGLET, 92.4%, is for the maximally entangled state
# at Q = 0 and S = 2 sqrt(2) with a missing click read as the outcome -1:
# Pironio, Acin, Brunner, Gisin, Massar & Scarani, New J. Phys. 11, 045021
# (2009), Fig. 3 caption.
#
# Quoted in refusal messages only; nothing gates on either, qkd shipping no
# device-independent rate to gate.
DI_ETA_PARTIAL = 0.865
DI_ETA_SINGLET = 0.924

# Attribute names a caller wanting one number out of a forecast reaches for.
_BARRED = ("key_rate", "rate", "key", "bound", "secure", "safe", "margin")

_NO_BOUND = (
    "a Forecast carries no key rate and bounds nothing: predict() is a FORWARD "
    "model, so pricing Eve by its s prices ONE assumed state and bounds no "
    "unknown one -- the circularity ekert_rate refuses as source='modelled'. "
    "Take q.SymmetryBound, or q.ViolationBound on a MEASURED violation"
)

# What a q.ViolationBound run is told when its violation came out of predict()
# rather than off an instrument. Constructing is allowed; running is not.
_LAUNDERED = (
    "q.ViolationBound(s=...) reads a MEASURED CHSH magnitude and a pairs.Forecast "
    "is a modelled one, which is the circularity ekert_rate refuses as "
    "source='modelled'. Read Forecast.s beside the run's own diagnostic chsh row, "
    "or take q.SymmetryBound"
)

# The attack class each security model's engine is proved against, quoted from
# that engine's own header rather than classified from outside it. Neither
# pair_rate nor src/pairs.rs's header names one, so unstated is reported.
_ATTACK_PAIR = (
    "unstated: neither pair_rate nor the module header names one; the rate is "
    "Ma, Fung & Lo Eq. (5), Shor-Preskill through Koashi-Preskill"
)

# src/ekert.rs:155 is what source="measured" buys, on the chi of
# src/pairs.rs:232; :29 and :203 name what a coherent-attack claim would need.
_ATTACK_CHSH = (
    "collective, given the s supplied; coherent attacks would need entropy "
    "accumulation, which qkd does not implement"
)


@dataclass(frozen=True)
class Coincidences:
    """
    What the two receivers see per pump pulse: the coincidence gain, the error
    rate on it, and the arms and brightness that produced them. A FORWARD
    MODEL, never a bound inferred from observed counts.
    """

    gain: float
    qber: float
    eta_a: float
    eta_b: float
    brightness: float


@dataclass(frozen=True)
class Forecast(std.Barred):
    """
    What CHSH value an assumed state shows at stated analyser settings: Ekert
    Eq. (3) over Eq. (2)'s correlations. A FORWARD MODEL, and AN INPUT TO NO
    BOUND -- the state and the settings are ASSUMED, so nothing device
    independent is implied and a q.PairLink handed one refuses it by name.

    `s` is SIGNED and negative at Ekert's own settings, his Eq. (4)'s
    convention against the modern one `chsh` returns; the two magnitudes
    agree. `qber` is the bit error rate that magnitude reads back as under the
    Bell-diagonal model, `angles` the CHSH quadruple (a1, a3, b1, b3) it was
    evaluated at, and `correlations` the four E(a, b) that summed to it.
    """

    # std.Barred's __getattr__ reads these two. It refuses a RATE off a forward
    # model, a different refusal from attacks.Reading's.
    _barred = _BARRED
    _refusal = _NO_BOUND

    s: float
    visibility: float
    qber: float
    angles: tuple
    correlations: tuple


@dataclass(frozen=True)
class PairResult:
    """
    A photon-pair link's rate. key_rate is bits per PUMP PULSE, clamped at 0 in
    the engine; per_second multiplies by the source repetition rate. chsh is a
    DIAGNOSTIC and never a security claim -- see `holevo`.
    """

    key_rate: float
    per_second: float
    gain: float
    qber: float
    # None where the security model prices Eve by something else: a
    # q.ViolationBound run reads the violation and no phase error at all.
    e_phase: float | None
    brightness: float
    eta_a: float
    eta_b: float
    chsh: float
    # The violation the rate was CLAIMED at, None under q.SymmetryBound. NOT
    # `chsh`, which is modelled from the error rate either way.
    s: float | None = None
    explain: dict | None = None


def gain(brightness, eta_a, eta_b, y0_a=0.0, y0_b=0.0):
    """
    Coincidence probability per pump pulse, Ma, Fung & Lo Eq. (9).
    """

    return _core.pair_gain(brightness, eta_a, eta_b, y0_a, y0_b)


def qber(brightness, eta_a, eta_b, y0_a=0.0, y0_b=0.0, e_d=0.0):
    """
    Overall error rate on those coincidences, Ma, Fung & Lo Eq. (10).
    """

    return _core.pair_error(brightness, eta_a, eta_b, y0_a, y0_b, e_d)


def rate(q_lam, e_bit, e_phase, f_ec=1.22, sift=0.5):
    """
    Asymptotic BBM92 rate in bits per pump pulse, Ma, Fung & Lo Eq. (5).
    """

    return _core.pair_rate(q_lam, e_bit, e_phase, f_ec, sift)


def optimum(eta_a, eta_b, y0_a=0.0, y0_b=0.0, e_d=0.0, f_ec=1.22, sift=0.5):
    """
    The brightness maximising that rate: (brightness, gain, qber, key_rate).
    """

    return _core.pair_optimum(eta_a, eta_b, y0_a, y0_b, e_d, f_ec, sift)


def holevo(s):
    """
    Eve's Holevo information in bits against a CHSH value s, the tight
    collective-attack bound of Acin et al., Phys. Rev. Lett. 98, 230501 (2007).
    Exactly 1 at s = 2 and exactly 0 at s = 2 sqrt(2).

    NOT A KEY RATE, and qkd ships no device-independent one. The rate this
    belongs to is 1 - h(Q) - holevo(S); a coincidence-counted click layer
    post-selects on detection, and that post-selection IS the fair-sampling
    assumption a device-independent claim must do without. The efficiency such
    a claim needs moves with the source model -- DI_ETA_PARTIAL 0.865,
    DI_ETA_SINGLET 0.924 -- and q.ClickDetector's default 0.2 is below both.
    """

    return _core.pair_holevo(s)


def finite(m, delta, f_ec, cbar, eps, detection):
    """
    Finite-key BBM92 length in bits over a block of m SIFTED rounds, and that
    length per sifted round, as (length, rate, k, nu, t): Tomamichel &
    Leverrier, Quantum 1, 14 (2017), Theorem 3. NOT per pump pulse, which is
    what rate() returns.

    `cbar` has no default: 1/2 is the IDEAL pair of measurements, so a default
    would report a real analyser as perfectly complementary. detection is
    "deterministic" or "coincidence"; the second -- what every q.PairLink runs
    -- is refused by the engine, which names the three missing pieces.
    """

    return _core.pair_finite(m, delta, f_ec, cbar, eps, detection)


def chsh(v):
    """
    The CHSH value a singlet of visibility v reaches at optimal settings,
    2 sqrt(2) v. A diagnostic: it assumes the state and the settings, which is
    the device-dependent assumption device independence forbids.
    """

    return _core.pair_chsh(v)


def settings():
    """
    Ekert's own six analyser orientations in radians, (a1, a2, a3, b1, b2, b3),
    from the paragraph before his Eq. (1). Two of Alice's coincide with two of
    Bob's and those two pairs are the KEY-generating ones; (a1, a3) against
    (b1, b3) is the CHSH quadruple of his Eq. (3), which is the four predict()
    reads. Drawing all six uniformly leaves sift = 2/9.
    """

    return _core.ekert_settings()


def correlate(a, b, v):
    """
    Ekert Eq. (2) softened to a singlet of visibility v, E = -v*cos(a - b),
    with the analyser angles a and b in radians; his Eq. (1) defines it as
    P++ + P-- - P+- - P-+. A forward model, never a measurement.
    """

    return _core.ekert_correlate(a, b, v)


def disturbance(s):
    """
    The bit error rate a Bell-diagonal state showing CHSH magnitude |s|
    carries, (1 - |s|/(2 sqrt 2))/2. MODEL DEPENDENT: it is what lets a
    violation be read as a disturbance at all. s = 2 reads 14.645% and
    s = 2 sqrt 2 reads exactly 0.
    """

    return _core.ekert_qber(s)


def predict(v, angles=None):
    """
    The CHSH value an assumed singlet of visibility v would show at four
    stated analyser angles, as a Forecast. `angles` is the CHSH quadruple
    (a1, a3, b1, b3) in radians and defaults to the four of Ekert's six that
    enter his Eq. (3).

    A FORWARD MODEL. Its s is an input to no bound: q.ViolationBound reads a
    MEASURED violation, and handing it a Forecast is refused by name at run().
    """
    six = settings()
    quad = (six[0], six[2], six[3], six[5]) if angles is None else tuple(angles)
    if len(quad) != 4:
        raise ValueError(
            "angles is the CHSH quadruple (a1, a3, b1, b3) in radians, four numbers "
            f"and not {len(quad)}: two of Ekert's six settings generate the KEY and "
            "enter no CHSH combination"
        )

    a1, a3, b1, b3 = quad
    s = _core.ekert_chsh(a1, a3, b1, b3, v)
    legs = ((a1, b1), (a1, b3), (a3, b1), (a3, b3))

    return Forecast(
        s=s,
        visibility=v,
        qber=disturbance(s),
        angles=quad,
        correlations=tuple(correlate(a, b, v) for a, b in legs),
    )


def background(det):
    """
    One receiver's background probability per pump pulse. A conjugate-basis
    receiver reads TWO gates, so Y0 = 1 - (1 - dark)^2.
    """

    return 1.0 - (1.0 - det.dark) ** 2


def arms(channels, detectors):
    """
    Per-arm end-to-end efficiency (eta_a, eta_b): each fibre's transmittance
    times that receiver's own detection efficiency.
    """
    if len(channels) != 2 or len(detectors) != 2:
        raise ValueError("channels and detectors are each (alice's arm, bob's arm)")

    out = []
    for fib, det in zip(channels, detectors):
        if not isinstance(fib, Fiber):
            raise ValueError("each arm must be a q.Fiber")

        if not isinstance(det, ClickDetector):
            raise ValueError("each receiver must be a q.ClickDetector")

        out.append(fib.transmittance * det.eta)

    return tuple(out)


def _check(source, detectors, analyser, security):
    """
    What this closed form may honestly be handed.
    """
    if not isinstance(source, PairSource):
        raise ValueError("source must be a q.PairSource")

    if not isinstance(analyser, BasisAnalyser):
        raise ValueError("analyser must be a q.BasisAnalyser")

    if not isinstance(security, (SymmetryBound, ViolationBound)):
        raise NotImplementedError(
            "a photon-pair link takes q.SymmetryBound security, which prices Eve by "
            "the phase error, or q.ViolationBound, which prices her by the CHSH "
            "violation. NEITHER RUNS FINITE-SIZE over COINCIDENCES: qkd.pairs.finite "
            "implements Tomamichel & Leverrier's entanglement-based length and "
            "REFUSES a coincidence-counted link by name, naming three missing pieces, "
            "while _core.ekert_finite refuses a CHSH-priced length outright and "
            "names four"
        )

    if isinstance(security, ViolationBound) and isinstance(security.s, Forecast):
        raise ValueError(_LAUNDERED)

    for det in detectors:
        if det.dead_time != 0.0 or det.afterpulse != 0.0:
            raise ValueError(
                "dead time and afterpulsing reach no closed form here: a dead "
                "detector kills a COINCIDENCE, so the single-detector correction is "
                "not the pair correction and applying it understates the loss"
            )

        if det.timed() or det.window != 0.0:
            raise ValueError(
                "detector timing reaches no closed form here: a coincidence window is "
                "a width BETWEEN two receivers and q.ClickDetector's window is a width "
                "inside one symbol period at one of them. Fold the window loss into eta"
            )


def observe(source, channels, detectors, analyser=None):
    """
    The coincidence gain and error rate this hardware produces, at the source's
    own brightness or at the one maximising the rate when it declares none.
    """
    analyser = BasisAnalyser() if analyser is None else analyser
    eta_a, eta_b = arms(channels, detectors)
    y0_a = background(detectors[0])
    y0_b = background(detectors[1])
    lam = source.brightness
    if lam is None:
        lam = optimum(eta_a, eta_b, y0_a, y0_b, analyser.misalign)[0]

    return Coincidences(
        gain=gain(lam, eta_a, eta_b, y0_a, y0_b),
        qber=qber(lam, eta_a, eta_b, y0_a, y0_b, analyser.misalign),
        eta_a=eta_a,
        eta_b=eta_b,
        brightness=lam,
    )


class PairLink:
    """
    One photon-pair source and two receiving stations: the dual of q.Swap,
    which puts a MEASUREMENT in the middle where this puts a SOURCE. run()
    returns a result, explain() the labelled plan, and refusals land at run().

    `channels` is (alice's arm, bob's arm), and THE SOURCE POSITION IS STATED
    THERE AND NOWHERE ELSE: q.Fiber(T=1.0) on one side puts the source in that
    party's lab, two half-length arms put it at the midpoint. There is no
    site= field, because a second way to say it could disagree with the first.
    """

    def __init__(self, source, detectors, channels, analyser=None, security=None):
        self.source = source
        self.detectors = tuple(detectors)
        self.channels = tuple(channels)
        self.analyser = BasisAnalyser() if analyser is None else analyser
        self.security = SymmetryBound() if security is None else security

    def run(self):
        """
        The rate this topology supports. A closed form over the configured
        hardware, so there is nothing to seed and run() takes no symbols.
        """
        sec = self.security
        _check(self.source, self.detectors, self.analyser, sec)
        counts = observe(self.source, self.channels, self.detectors, self.analyser)
        seen = chsh(max(0.0, 1.0 - 2.0 * counts.qber))
        if isinstance(sec, ViolationBound):
            return self._violated(counts, seen)

        e_phase = counts.qber if sec.e_phase is None else sec.e_phase
        key = rate(counts.gain, counts.qber, e_phase, sec.f, sec.sift)
        info = _explain(self, counts, e_phase, key, seen)

        return PairResult(
            key_rate=key,
            per_second=key * self.source.rate,
            gain=counts.gain,
            qber=counts.qber,
            e_phase=e_phase,
            brightness=counts.brightness,
            eta_a=counts.eta_a,
            eta_b=counts.eta_b,
            chsh=seen,
            explain=info,
        )

    def predict(self):
        """
        What CHSH value this hardware would show, as a Forecast: the module's
        predict() at the visibility 1 - 2*qber the coincidence model gives,
        and at Ekert's own analyser settings. A FORWARD MODEL and no part of
        any rate.
        """
        _check(self.source, self.detectors, self.analyser, self.security)
        counts = observe(self.source, self.channels, self.detectors, self.analyser)

        # The module-level predict(), not this method: the hardware fixes the
        # visibility, so the method takes none.
        return predict(max(0.0, 1.0 - 2.0 * counts.qber))

    def _violated(self, counts, seen):
        """
        The CHSH-priced branch. `source` reaches the engine untouched, which
        is where a modelled violation and a device-independent claim are
        refused.
        """
        sec = self.security
        key = _core.ekert_rate(counts.gain, counts.qber, sec.s, sec.f, sec.sift, sec.source)
        info = _explain_chsh(self, counts, key, seen)

        return PairResult(
            key_rate=key,
            per_second=key * self.source.rate,
            gain=counts.gain,
            qber=counts.qber,
            # No phase error on this branch: the violation stands where one
            # would, and the two must not read as one row.
            e_phase=None,
            brightness=counts.brightness,
            eta_a=counts.eta_a,
            eta_b=counts.eta_b,
            chsh=seen,
            s=sec.s,
            explain=info,
        )

    def explain(self):
        """
        The resolved plan, every quantity labelled pinned, derived, structural
        or diagnostic.
        """

        return self.run().explain


def _shared(plan, counts, info):
    """
    The hardware rows both security models report, in one order so the two
    plans read against each other line by line.
    """
    pinned = plan.source.brightness is not None
    for i, tag in enumerate(("a", "b")):
        info[f"T_{tag}"] = _q(plan.channels[i].transmittance, "derived")

    info["eta_a"] = _q(counts.eta_a, "derived")
    info["eta_b"] = _q(counts.eta_b, "derived")
    info["brightness"] = _q(counts.brightness, "pinned" if pinned else "derived")
    info["pairs"] = _q(2.0 * counts.brightness, "derived")
    info["misalign"] = _q(plan.analyser.misalign, "pinned")
    info["f"] = _q(plan.security.f, "pinned")
    info["sift"] = _q(plan.security.sift, "pinned")
    info["gain"] = _q(counts.gain, "derived")
    info["qber"] = _q(counts.qber, "derived")

    return info


def _explain(plan, counts, e_phase, key, seen):
    """
    The resolved plan, every quantity labelled pinned or derived.
    """
    source = plan.source
    security = plan.security
    info = {
        "topology": "photon-pair source",
        "protocol": "entanglement-based basis keying",
        "measurement": "conjugate bases (threshold detectors)",
        "trust": (
            "structural: the source is never trusted, because both parties "
            "measure in both bases. The RECEIVERS are trusted and "
            "characterised, so this is not device independent"
        ),
        "security": type(security).__name__,
        "attack": _ATTACK_PAIR,
    }
    _shared(plan, counts, info)
    info["e_phase"] = _q(e_phase, "pinned" if security.e_phase is not None else "derived")
    # Modelled from the error rate under a singlet-plus-white-noise assumption,
    # so it certifies nothing and reaches no rate.
    info["chsh"] = _q(seen, "diagnostic")
    info["device_independent"] = _q(False, "structural")
    info["key_rate"] = _q(key, "derived")
    info["key_second"] = _q(key * source.rate, "derived")

    return info


def _explain_chsh(plan, counts, key, seen):
    """
    The same plan priced by the violation instead, with the two model
    evaluations that measure what dropping the state assumption costs.
    """
    source = plan.source
    security = plan.security
    info = {
        "topology": "photon-pair source",
        "protocol": "entanglement-based keying, priced by the violation",
        "measurement": "three analyser settings per party (threshold detectors)",
        "trust": (
            "structural: the violation prices Eve with no assumption about the "
            "SOURCE state, which is what it buys over basis symmetry. The "
            "RECEIVERS are still trusted and characterised and the coincidence "
            "count still post-selects on detection, so this is not device "
            "independent"
        ),
        "security": type(security).__name__,
        "attack": _ATTACK_CHSH,
    }
    _shared(plan, counts, info)
    # The engine refuses every origin but a measured one.
    info["s"] = _q(security.s, "pinned")
    info["source"] = _q(security.source, "pinned")
    # NO PHASE ERROR IS READ on this branch: s stands exactly where one would.
    info["e_phase"] = _q(None, "absent")
    # Modelled from the error rate, as on the basis-symmetry plan.
    info["chsh"] = _q(seen, "diagnostic")
    model = _core.ekert_point(counts.gain, counts.qber, security.f, security.sift)
    # Neither is a security claim: the first prices Eve by the MODELLED violation,
    # the second is basis symmetry on the same numbers, and their gap is what the
    # state assumption was worth.
    info["key_model"] = _q(model[1], "diagnostic")
    info["key_symmetry"] = _q(model[2], "diagnostic")
    info["e_threshold"] = _q(_core.ekert_threshold(security.f), "derived")
    info["device_independent"] = _q(False, "structural")
    info["key_rate"] = _q(key, "derived")
    info["key_second"] = _q(key * source.rate, "derived")

    return info
