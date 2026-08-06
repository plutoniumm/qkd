import math
from dataclasses import dataclass

from . import _core, std

_ROOT2 = math.sqrt(2.0)
_TWOPI = 2.0 * math.pi

# Eq. (19) of arXiv:1511.01007 caps Eve's level-II gain: (1 + A) = 2*sqrt2/g
# with A = erf(...) in (-1, 1), so g must sit in (0, 2*sqrt 2).
_GAIN_CAP = 2.0 * _ROOT2

# Enumerated by name in docs/architecture.md and docs/guide/security.md.
_BARRED = ("key_rate", "rate", "key", "secure", "safe", "margin")

_NO_RATE = (
    "a Reading has no key rate: observed and eve are not the same quantity, and "
    "the first does not bound the second -- under every attack modelled here the "
    "observables stay at values an unattacked link would produce. Read "
    "Reading.observed and Reading.eve separately, and never subtract them"
)


# std.entropy is the one implementation; a private copy would drift silently.
_entropy = std.entropy


@dataclass(frozen=True)
class Reading(std.Barred):
    """
    One attack's two sets of numbers, which are different quantities. observed
    is what Alice and Bob measure while the attack runs; eve is what the
    eavesdropper holds. note names the reason the first does not bound the
    second. There is deliberately no key rate here, and asking raises.
    """

    # std.Barred's __getattr__ reads these two; the tuple is this module's own,
    # not topology's.
    _barred = _BARRED
    _refusal = _NO_RATE

    attack: str
    observed: dict
    eve: dict
    note: str

    def table(self):
        """
        Aligned text table: one row per number, each side labelled.
        """
        rows = [("side", "quantity", "value")]
        for side, book in (("observed", self.observed), ("eve", self.eve)):
            for name, value in book.items():
                rows.append((side, name, f"{value:.6g}"))

        head = [self.attack, self.note, ""]

        return "\n".join(head + std.grid(rows))


@dataclass(frozen=True)
class Saturation:
    """
    Eve intercept-resends every pulse and displaces the coherent state she
    sends so that Bob's homodyne clips. alpha is the detector's linear
    half-range in shot-noise amplitude units, delta the displacement AT BOB
    (t*Delta_X, after the channel), gain the rescaling g she applies to her
    heterodyne result, sqrt 2 compensating its 3 dB. Qin, Kumar and Alleaume,
    PRA 94, 012325 (2016), arXiv:1511.01007.
    """

    alpha: float
    delta: float
    gain: float = _ROOT2


@dataclass(frozen=True)
class Calibration:
    """
    Eve reshapes the local-oscillator pulse so Bob's clock fires late, which
    lowers his detection slope and leaves the calibrated shot noise he applies
    above the one his detector has. ratio is N'_0/N_0 and the orientation is
    ASSUMED OVER ACTUAL: N'_0 is the shot noise Bob's calibration hands the
    estimator, N_0 the one his detector really has, so 1.0 is no attack and an
    attacked ratio is above 1.0. resend is the fraction mu of signal pulses she
    intercept-resends. Jouguet, Kunz-Jacques and Diamanti, PRA 87, 062313
    (2013), arXiv:1304.7024.
    """

    ratio: float
    resend: float = 1.0


@dataclass(frozen=True)
class Blinding:
    """
    Bright light holds Bob's APDs below breakdown, where they answer classical
    power and nothing else. always and never are the per-detector P_always and
    P_never in W; passive says Bob splits his basis on a coupler rather than a
    modulator, which doubles the trigger power Eve needs and removes the
    sifting loss. Lydersen et al., Nat. Photonics 4, 686 (2010),
    arXiv:1008.4593.
    """

    always: tuple
    never: tuple
    passive: bool = False


@dataclass(frozen=True)
class Mismatch:
    """
    Bob's two detectors do not answer with the same efficiency at the same
    instant, so which detector fires carries information about when the photon
    arrived. hi and lo are the two efficiencies at the instant Eve shifts to.
    Makarov, Anisimov and Skaar, PRA 74, 022313 (2006); Qi, Fung, Lo and Ma,
    QIC 7, 73 (2007), arXiv:quant-ph/0512080.

    THIS IS NOT q.ClickDetector(partner=...) AND THE TWO MUST NOT BE COMPOSED.
    That component DESCRIBES a receiver whose two detectors differ at their own
    sampling point; this descriptor PRICES AN ATTACK that moves the sampling
    point away from it, so hi and lo are the efficiencies at the instant Eve
    shifts to rather than the two a datasheet quotes.
    """

    hi: float
    lo: float


@dataclass(frozen=True)
class Blanking:
    """
    Eve fires a dim polarised pulse into the channel less than one dead time
    ahead of each slot, so the detectors it lands in are still recovering when
    Alice's photon arrives. blind is that pulse's mean photon number at an
    ideal receiver, signal is Alice's, gap is how far ahead of the slot it goes
    in s. Weier et al., New J. Phys. 13, 073024 (2011), arXiv:1101.5289. The
    hardware parameter it exploits is the tau_D of q.DeadTime; see
    blank_hidden.
    """

    blind: float
    signal: float
    gap: float = 0.0


@dataclass(frozen=True)
class Oscillator:
    """
    Eve reshapes the transmitted local oscillator so Bob's clock fires away
    from the peak of the pulse the clock is derived from. monitored is the
    oscillator power his monitor reports and sampled the power actually mixing
    at the sampling instant; slope and floor are the calibration line
    Z(P) = slope*P + floor he fitted offline, slope in shot noise per unit
    power and floor the zero-oscillator variance, which is electronic and does
    not move with the oscillator.

    This is q.Calibration's attack with its ratio DERIVED rather than dialled:
    src/tlo.rs's tlo_calib turns those four numbers into the assumed-over-actual
    ratio, lo_ratio reads it and lo_break lands it. Separate from q.Calibration
    because a transmitted oscillator is a security model, not a hardware
    description. resend is the fraction of signal pulses she intercept-resends
    behind it, as on q.Calibration.
    """

    monitored: float
    sampled: float
    slope: float = 1.0
    floor: float = 0.0
    resend: float = 1.0


@dataclass(frozen=True)
class Injection:
    """
    Eve sends bright light down the quantum channel into Alice's transmitter
    and reads what comes back out. The probe crosses her phase modulator twice,
    so the back-reflection carries the setting Alice just applied, and no error
    rate Alice and Bob can form is disturbed by it. This is the Trojan-horse
    attack of Gisin, Fasel, Kraus, Zbinden and Ribordy, PRA 73, 022320 (2006),
    arXiv:quant-ph/0507063, priced by the bound of Lucamarini, Choi, Ward,
    Dynes, Yuan and Shields, Phys. Rev. X 5, 031030 (2015), arXiv:1506.01989.

    Every field but the last two is a POSITIVE dB SUPPRESSION on one pass, and
    they are the insertion losses q.Connector, q.Splice and q.Coupling already
    carry read as suppressions: reflect is the back-reflection R of the deepest
    component Eve reaches, isolator the isolation I of one stage and stages how
    many stand in line, atten the attenuator A, bandpass the spectral filter F.
    Eq. (15) doubles the filter and the attenuator because the probe crosses
    them going in and coming back out, and counts the reflection once. injected
    is N, the photons per second Eve may send before the fibre takes laser
    damage -- 1e20 is that paper's figure -- and clock is Alice's modulator
    rate f_A, 1 GHz being theirs.

    THE dB ARE AT ONE WAVELENGTH AND qkd's COMPONENT LOSSES CARRY NONE.
    Jain, Stiller, Khan, Makarov, Marquardt and Leuchs, IEEE J. Sel. Topics
    Quantum Elect. 21, 3 (2015), arXiv:1408.0492 report wavelength regimes
    where the attacker gains considerable advantage over 1550 nm, and the
    wavelength is Eve's to choose.
    """

    reflect: float
    isolator: float = 0.0
    stages: int = 0
    atten: float = 0.0
    bandpass: float = 0.0
    injected: float = 1e20
    clock: float = 1e9


def sat_var(v_a, t, eta, v_el, xi=0.0, gain=_ROOT2):
    """
    Var(X_B,lin) in shot-noise units: Bob's quadrature variance under the
    intercept-resend with an unbounded detector, Qin arXiv:1511.01007 Eq. (15),
    eta*T*(G/2)*(V_A + 2 + xi_sys) + 1 + v_ele with G = g^2. The 2 is the
    entanglement-breaking cost of any intercept-resend, and at G = 2 it is
    exactly what a linear detector reports back as excess noise.
    """
    std.positive("v_a", v_a)

    std.unit("t", t)

    std.unit("eta", eta)

    std.nonneg("v_el", v_el)

    std.nonneg("xi", xi)

    std.positive("gain", gain)

    half = eta * t * gain * gain / 2.0

    return half * (v_a + 2.0 + xi) + 1.0 + v_el


def sat_clip(var, gap):
    """
    Var(X_B,sat) in shot-noise units, the variance a homodyne clipping at
    +/- alpha reports: Qin arXiv:1511.01007 Appendix B, Eq. (B14), for
    gap = alpha - Delta. NOT "Eq. (40)": the body ends at (20) and the appendices
    run A1-A6 then B1-B13, so B14 is the 40th on a flat count and the string
    "(40)" appears nowhere in the paper. The exact second central moment of a
    Gaussian of variance var censored gap above its mean. ONE-SIDED, as
    Appendix B is: it keeps the upper clip and drops the lower one at -alpha,
    which sits alpha + Delta below the mean. That tail is below double
    precision at Qin's alpha = 20 SNU, but this is NOT exact for an alpha of
    order sigma -- at alpha = 2 SNU and a Var(X_B,lin) of 2.97 the one-sided
    form overstates the clipped variance by 31%.
    """
    std.positive("var", var)

    std.finite("gap", gap)

    a = math.erf(gap / math.sqrt(2.0 * var))
    b = math.exp(-gap * gap / (2.0 * var))
    edge = gap * math.sqrt(var / _TWOPI) * a * b

    return var * ((1.0 + a) / 2.0 - b * b / _TWOPI) - edge + gap * gap * (1.0 - a * a) / 4.0


def sat_trans(t, var, gap, gain=_ROOT2):
    """
    T-hat, the channel transmittance Alice and Bob estimate under the
    saturation attack: Qin arXiv:1511.01007 Eq. (16), T*(G/8)*(1 + A)^2 with
    A = erf(gap/sqrt(2*Var(X_B,lin))). Clipping only ever costs correlation, so
    with the natural G = 2 this returns T when gap is wide and falls to zero as
    the displacement passes alpha.
    """
    std.unit("t", t)

    std.positive("var", var)

    std.positive("gain", gain)

    a = math.erf(gap / math.sqrt(2.0 * var))

    return t * gain * gain * (1.0 + a) ** 2 / 8.0


def sat_estimate(v_a, t, eta, v_el, alpha, delta, xi=0.0, gain=_ROOT2):
    """
    (T-hat, xi-hat), the channel Alice and Bob estimate while Eve runs a full
    intercept-resend behind a displaced, clipping homodyne. Qin
    arXiv:1511.01007 Eq. (16) for the transmittance and Eq. (7) evaluated at
    that transmittance and at the censored variance of Eq. (B14) for the excess
    noise. That composition is used rather than Eq. (17), whose Var(X_B,lin)
    term is printed at half the value Appendix B derives; Qin's PhD thesis
    Eq. (7.10) prints the factor 2 this form carries, and at Delta = 0 it is
    the form that returns the paper's stated xi_lin.
    """
    std.positive("alpha", alpha)

    std.nonneg("delta", delta)

    var = sat_var(v_a, t, eta, v_el, xi, gain)
    gap = alpha - delta
    t_hat = sat_trans(t, var, gap, gain)
    if t_hat <= 0.0:
        raise ValueError(
            f"the displacement Delta = {delta:g} leaves T-hat = 0: the correlation "
            "has been clipped away entirely, so Eq. (7) divides by zero. Qin's "
            "Eq. (16) reaches this as Delta grows past alpha"
        )

    clipped = sat_clip(var, gap)

    return t_hat, (clipped - 1.0 - v_el) / (eta * t_hat) - v_a


def sat_gain(v_a, t, eta, v_el, alpha, delta, xi=0.0):
    """
    The gain g that leaves the transmittance estimate unbiased, T-hat = T:
    Qin arXiv:1511.01007 Eq. (19), 2*sqrt2/g = 1 + erf(gap/sqrt(2*Var)), solved
    for g with Var itself a function of g. Eq. (19) omits the 2 under the root
    that Eq. (16) carries and the consistent form is used here. This is the
    level II criterion: an attack that also leaves the loss budget looking
    right.
    """
    std.positive("alpha", alpha)

    std.nonneg("delta", delta)

    gap = alpha - delta

    def miss(g):
        var = sat_var(v_a, t, eta, v_el, xi, g)

        return 1.0 + math.erf(gap / math.sqrt(2.0 * var)) - _GAIN_CAP / g

    lo, hi = 1e-9, _GAIN_CAP
    if miss(hi) <= 0.0:
        raise ValueError(
            f"no gain in (0, {_GAIN_CAP:.6g}] leaves T-hat = T at Delta = {delta:g}: "
            "Eq. (19) needs 1 + erf(gap/sqrt(2 Var)) = 2 sqrt2/g and the left side is "
            "at its floor, so the level II criterion cannot be met. Level I, which "
            "lets the transmittance estimate drop, still can"
        )

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if miss(mid) <= 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def sat_range(alpha, ratio):
    """
    q.Saturation.alpha carried from the shot-noise unit it is QUOTED against to
    the one the detector actually has: src/tlo.rs's tlo_range, alpha*sqrt(ratio)
    with ratio assumed over actual. Clipping happens at a fixed photocurrent, so
    in a transmitted-oscillator receiver alpha is a function of the oscillator
    reaching the detector rather than a constant of the hardware.

    Two different quantities reach this: an honest span's q.ShotNoise.ratio,
    calibrated over arriving, and the ratio lo_ratio derives under attack. Feed
    the result to sat_var, sat_estimate and sat_break as their alpha.
    """

    return _core.tlo_range(alpha, ratio)


def sat_break(v_a, t, eta, v_el, alpha, delta, xi=0.0, gain=_ROOT2):
    """
    The saturation attack as a Reading. observed is the (T, xi) pair parameter
    estimation returns; eve is the intercept-resend actually running, whose
    2 + xi_sys shot-noise units of excess noise are entanglement breaking at
    every distance. The gap between the two is not a correction: the observed
    xi is an artefact of a clipped estimator and refers to no plane.
    """
    t_hat, xi_hat = sat_estimate(v_a, t, eta, v_el, alpha, delta, xi, gain)

    return Reading(
        attack="saturation",
        observed={
            "t": t_hat,
            "xi": xi_hat,
        },
        eve={
            "xi": 2.0 + xi,
            "resend": 1.0,
        },
        note=(
            "Qin arXiv:1511.01007: the covariance matrix is invariant under a "
            "shift of the quadrature mean, and no CV-QKD estimator monitors "
            "that mean, so Eve displaces Bob into his clipping region for free"
        ),
    )


def resend_xi(xi, share=1.0):
    """
    Excess noise a partial intercept-resend actually adds, in shot-noise units:
    Jouguet arXiv:1304.7024 Eq. (7), xi + 2*mu, mu the intercepted fraction.
    At mu = 1 and a system xi of 0.1 this is 2.1, the number both that paper
    and Qin's quote as undistillable.
    """
    std.nonneg("xi", xi)

    std.closed("share", share)

    return xi + 2.0 * share


def calib_xi(xi, ratio, t, eta):
    """
    Excess noise Alice and Bob report, in units of the shot noise they believe
    they calibrated: Jouguet arXiv:1304.7024 Eqs. (4) and (8),
    (N_0/N'_0)[xi/N_0 + (1 - N'_0/N_0)/t-hat^2] with t-hat^2 = eta*T. ratio is
    N'_0/N_0, assumed over actual, and N_0 = 1 here because xi arrives already
    in units of the shot noise the detector really has. RETURNED RAW,
    including negative values: a negative apparent excess noise is the
    signature and clamping it would erase the evidence. It is not a physical
    noise and no budget may carry it.
    """
    std.finite("xi", xi)

    std.positive("ratio", ratio)

    std.unit("t", t)

    std.unit("eta", eta)

    return (xi + (1.0 - ratio) / (eta * t)) / ratio


def calib_ratio(xi, t, eta):
    """
    The shot-noise overestimate N'_0/N_0, assumed over actual, that drives the
    reported excess noise exactly to zero: Jouguet arXiv:1304.7024 Eq. (8)
    solved for the ratio, 1 + eta*T*xi. Anything above it reports a negative
    excess noise, which a receiver checking only that xi is small will accept.
    """
    std.nonneg("xi", xi)

    std.unit("t", t)

    std.unit("eta", eta)

    return 1.0 + eta * t * xi


def calib_break(xi, ratio, t, eta, share=1.0):
    """
    The local-oscillator calibration attack as a Reading. observed is the
    excess noise in the attacked shot-noise unit; eve is the real one, xi + 2mu
    at the true unit. The two differ by an estimator bias and a change of unit at
    once, so neither may be quoted without saying which N_0 it is divided by.
    """
    real = resend_xi(xi, share)

    return Reading(
        attack="calibration",
        observed={
            "xi": calib_xi(real, ratio, t, eta),
            "shot": ratio,
        },
        eve={
            "xi": real,
            "resend": share,
        },
        note=(
            "Jouguet arXiv:1304.7024: every CV-QKD quantity is quoted in shot "
            "noise units, so an eavesdropper who inflates the calibrated N_0 "
            "deflates the excess noise without touching the quantum channel"
        ),
    )


def lo_ratio(monitored, sampled, slope=1.0, floor=0.0):
    """
    (ratio, zero, null) read off Bob's own shot-noise calibration line:
    src/tlo.rs's tlo_calib. ratio is N'_0/N_0, assumed over actual, which is
    exactly the number q.Calibration dials and this one derives instead. zero is
    the normalised zero-signal variance a real-time monitor reads, 1/ratio +
    v_el, and null is the 1 + v_el it reads while the line still holds; both are
    in the ASSUMED unit.

    zero and null are the attack surface: equal in an honest run, pulled apart
    by anything that moves the sampling instant relative to the pulse, and
    compared by no q.Link path. Both sit on the OBSERVED side of lo_break, and
    unlike the two sides of a Reading these two MAY be compared -- comparing
    them is the countermeasure.
    """
    line = _core.tlo_calib(monitored, sampled, slope, floor)

    return line[2], line[4], 1.0 + line[3]


def lo_break(monitored, sampled, xi, t, eta, share=1.0, slope=1.0, floor=0.0):
    """
    The transmitted-oscillator calibration attack as a Reading, with the
    shot-noise ratio read off Bob's calibration line rather than declared.
    observed carries the excess noise in the attacked unit, that unit, and the
    two zero-signal variances a monitor would compare; eve carries the real
    excess noise xi + 2mu at the real unit. It is calib_break's arithmetic on a
    ratio that came from hardware, and the two agree wherever the ratios do.

    The countermeasure is Kunz-Jacques & Jouguet, PRA 91, 022307 (2015),
    arXiv:1406.7554: Bob measures the unit from the oscillator he is using, by
    blocking the signal path on randomly chosen pulses. It ships as tlo_monitor
    and q.TransmittedLO.monitor. It narrows a UNIFORM rescale as one over the
    square root of the sample count and closes no loophole outright: Huang,
    Kunz-Jacques, Jouguet, Weedbrook, Yin, Wang, Chen, Guo & Han, PRA 89,
    032304 (2014), arXiv:1402.6921, bias the shot-noise estimate even if it is
    done in real time. No q.Link path runs it.
    """
    ratio, zero, null = lo_ratio(monitored, sampled, slope, floor)
    real = resend_xi(xi, share)

    return Reading(
        attack="oscillator",
        observed={
            "xi": calib_xi(real, ratio, t, eta),
            "shot": ratio,
            "zero": zero,
            "null": null,
        },
        eve={
            "xi": real,
            "resend": share,
        },
        note=(
            "Jouguet arXiv:1304.7024 through src/tlo.rs: Bob's clock is derived "
            "from the oscillator, so reshaping the oscillator moves the sampling "
            "instant, and every quadrature is then divided by a shot-noise unit "
            "his detector does not have"
        ),
    )


def blind_ratio(always, never):
    """
    max_i P_always,i / min_i P_never,i for a set of blinded detectors. Lydersen
    arXiv:1008.4593 Eq. (1) requires it below 2: a trigger pulse of that power
    fires the one detector Eve aims at, and half of it, reaching each of the
    others, fires none of them. Powers in W.
    """
    if not always or not never:
        raise ValueError("blinding needs at least one P_always and one P_never")

    for name, book in (("P_always", always), ("P_never", never)):
        for power in book:
            std.positive(name, power)

    return max(always) / min(never)


def blind_share(gain, passive=False):
    """
    The fraction of slots Eve must fire a trigger pulse into to reproduce an
    observed gain. A faked state reaches an actively switched Bob's detector
    only when his basis matches hers, so it clicks with probability 1/2; a
    passively split Bob sees half the power in each basis and, at Lydersen's
    doubled trigger power, always clicks. Lydersen arXiv:1008.4593.
    """
    std.closed("gain", gain)

    return gain if passive else 2.0 * gain


def blind_break(gain, qber, always, never, passive=False):
    """
    Detector blinding as a Reading. observed is the gain and QBER Eve chooses
    to reproduce; eve is one bit per sifted bit, because a blinded APD reports
    the classical power it is handed and Bob's raw key becomes a copy of hers.
    """
    ratio = blind_ratio(always, never)
    if ratio >= 2.0:
        raise ValueError(
            f"max(P_always)/min(P_never) = {ratio:.6g} is not below 2, so "
            "Lydersen arXiv:1008.4593 Eq. (1) fails: a pulse able to fire the "
            "intended detector also fires at least one other, and the attack "
            "leaves errors. This module has no model for that partial control "
            "and will not approximate one"
        )

    share = blind_share(gain, passive)
    if share > 1.0:
        raise ValueError(
            f"reproducing a gain of {gain:.6g} needs a trigger on {share:.6g} of the "
            "slots, which exceeds 1: an actively switched Bob discards half of Eve's "
            "faked states, so a gain above 1/2 cannot be covered"
        )

    return Reading(
        attack="blinding",
        observed={
            "gain": gain,
            "qber": qber,
            "trigger": share,
        },
        eve={
            "info": 1.0,
            "control": ratio,
        },
        note=(
            "Lydersen arXiv:1008.4593: a blinded APD has no dark counts and no "
            "single-photon sensitivity, so the observed QBER is whatever Eve "
            "decides to reproduce and a dark-count floor must be simulated by "
            "her rather than measured by Bob"
        ),
    )


def shift_ratio(hi, lo):
    """
    The efficiency mismatch r = min/max in [0, 1] of two detectors read at one
    instant: Qi arXiv:quant-ph/0512080 Eq. (1) under the symmetric assumption
    of Makarov, PRA 74, 022313 (2006). r = 1 is matched detectors and r = 0 is
    a detector that is off, which is the case Eve wants.
    """
    std.positive("hi", hi)

    std.positive("lo", lo)

    return min(hi, lo) / max(hi, lo)


def shift_qber(r):
    """
    QBER a faked-state attack on mismatched detectors induces: Qi
    arXiv:quant-ph/0512080 Eq. (2) under the symmetric assumption, which
    reduces to 2r/(1 + 3r). It is 0 at r = 0 and 1/2 at r = 1, and crosses the
    1/4 of a plain intercept-resend at r = 1/5, which is Qi's stated threshold
    for the faked-state attack being the better one.
    """
    std.closed("r", r)

    return 2.0 * r / (1.0 + 3.0 * r)


def shift_leak(r):
    """
    Eve's information per sifted bit under the time-shift attack proper: Qi
    arXiv:quant-ph/0512080 Eq. (3), 1 - h(r/(1 + r)). Eve never measures, so she
    induces NO error at any r and the attack is invisible in the QBER.
    """
    std.closed("r", r)

    return 1.0 - _entropy(r / (1.0 + r))


def shift_bound(r):
    """
    Upper bound on any key rate under the time-shift attack, per sifted bit:
    Qi arXiv:quant-ph/0512080 Eq. (4), the conditional mutual information
    h(r/(1 + r)). A CEILING and not a rate. Alice and Bob see no errors and
    would claim 1, so any r below 1 already puts their claim above this bound.
    r/(1 + r) and 1/(1 + r) are complements and h is symmetric about 1/2, so
    the bound is the same whichever detector shift_ratio calls the fast one.
    """
    std.closed("r", r)

    return _entropy(r / (1.0 + r))


def shift_balance(hi_a, lo_a, hi_b, lo_b):
    """
    (p, count): the probability p with which Eve applies the first of two time
    shifts so that the two detectors' totals come out equal, and the resulting
    count per detector. This is the concealment, not the attack -- Zhao et al.,
    PRA 78, 042333 (2008) balance their two measured shifts this way, and it is
    what stops Alice and Bob noticing a lopsided detector.
    """
    for name, count in (("hi_a", hi_a), ("lo_a", lo_a), ("hi_b", hi_b), ("lo_b", lo_b)):
        std.nonneg(name, count)

    lead = (hi_a - lo_a) - (hi_b - lo_b)
    if lead == 0.0:
        raise ValueError(
            "the two shifts favour the detectors by the same margin, so no mixture of them balances the counts"
        )

    p = (lo_b - hi_b) / lead
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"balancing needs p = {p:.6g}, outside [0, 1]: both shifts favour the "
            "same detector, so mixing them cannot equalise the counts"
        )

    return p, p * hi_a + (1.0 - p) * hi_b


def mismatch_rate(hi, lo, e_bit, e_phase, discard=True):
    """
    The key rate Alice and Bob are entitled to once the mismatch is accounted
    for, per detected signal: Fung, Tamaki, Qi, Lo and Ma, QIC 9, 131 (2009),
    arXiv:0802.3788 Eq. (33) with discard=True, the data-discarding argument
    that equalises the two efficiencies, and Eq. (34) with discard=False, their
    general estimate. Both scale by 2*min/(hi + lo), Eq. (32). e_phase has NO
    default: taking it equal to e_bit assumes a channel symmetric between the
    bases, which is exactly what an efficiency mismatch is not.
    """
    std.positive("hi", hi)

    std.positive("lo", lo)

    std.closed("e_bit", e_bit)

    std.closed("e_phase", e_phase)

    share = 2.0 * min(hi, lo) / (hi + lo)
    if discard:
        return share * (1.0 - _entropy(e_phase) - _entropy(e_bit))

    return share * (1.0 - _entropy(e_phase)) - _entropy(e_bit)


def shift_break(hi, lo, qber=0.0):
    """
    The time-shift attack as a Reading. observed is the QBER, which the attack
    does not move at all, beside the naive rate a proof blind to the mismatch
    would claim from it; eve is her information and the ceiling that actually
    applies. Zhao et al., PRA 78, 042333 (2008) state the same comparison as a
    key length: K_L = 1297 bits is their Eq. (3), the GLLP length Alice and Bob
    believe they can extract while ignoring the attack, and K_U = 1131 bits
    their Eq. (4), the Renner maximum given Eve's side information. The two are
    not bounds on one quantity and must not be read as a bracket.
    """
    r = shift_ratio(hi, lo)

    return Reading(
        attack="timeshift",
        observed={
            "qber": qber,
            "naive": 1.0 - _entropy(qber),
        },
        eve={
            "info": shift_leak(r),
            "bound": shift_bound(r),
            "ratio": r,
        },
        note=(
            "Qi arXiv:quant-ph/0512080: Eve reroutes the pulse through a long "
            "or a short path and never measures it, so she introduces no error "
            "and the QBER carries no trace of her at any mismatch"
        ),
    )


def blank_probs(blind):
    """
    (P_parallel, P_diagonal), the probability that Eve's blinding pulse fires
    the detector aligned with its polarisation and each of the two diagonal
    ones, in the passive four-detector receiver of Weier arXiv:1101.5289:
    1 - exp(-mu/2) and 1 - exp(-mu/4). The orthogonal detector never fires,
    which is the whole mechanism -- it is the one left able to answer Alice.
    """
    std.nonneg("blind", blind)

    return 1.0 - math.exp(-blind / 2.0), 1.0 - math.exp(-blind / 4.0)


def blank_error(blind, signal):
    """
    Eve's error probability when she guesses that Bob's click came from the
    detector orthogonal to her blinding pulse: Weier arXiv:1101.5289 Eqs. (10),
    (13) and (14), p_par/(p_par + p_perp) with p = (1 - P_phi(mu_B))*P_theta(mu_S).
    It is independent of the signal intensity to first order, since both terms
    carry one factor of mu_S.
    """
    std.positive("signal", signal)

    par, diag = blank_probs(blind)
    sig_p, sig_d = blank_probs(signal)
    same = (1.0 - par) * sig_p + (1.0 - diag) * sig_d
    other = sig_p + (1.0 - diag) * sig_d

    return same / (same + other)


def blank_leak(blind, signal):
    """
    Eve's information per sifted bit under the dead-time blanking attack:
    Weier arXiv:1101.5289 Eq. (15), 1 - h(p_par/(p_par + p_perp)). Eqs. (13)
    and (14) admit two readings and this is the LITERAL one, the diagonal term
    entering once: it gives 0.9309 bit at their mu_B = 16.52, above the 0.908
    they measure. Carrying a factor 2 on that term, one per diagonal detector,
    gives 0.8816 instead and tracks the theory curve plotted in their Fig. 2b,
    which digitises to about 0.882 at that point. Both readings are defensible
    from the text; the literal one is the default and every number this module
    reports is on it.
    """

    return 1.0 - _entropy(blank_error(blind, signal))


def blank_hidden(gap, window, dead):
    """
    True when Eve's blinding pulse is invisible to Bob: Weier
    arXiv:1101.5289 requires Delta_tw/2 < t_i - t_B < tau_D, so the pulse lands
    outside the acceptance window that would have counted it and inside the
    dead time that carries its effect into the next slot. tau_D is the same
    hardware number q.DeadTime carries. Refuses rather than returning False,
    because a pulse outside that interval is a different experiment and not a
    weaker attack.
    """
    std.positive("gap", gap)

    std.nonneg("window", window)

    std.positive("dead", dead)

    if gap <= window / 2.0:
        raise ValueError(
            f"a blinding pulse {gap:g} s ahead of the slot falls inside Bob's "
            f"{window:g} s acceptance window and is counted. Weier arXiv:1101.5289 "
            "needs Delta_tw/2 < t_i - t_B"
        )

    if gap >= dead:
        raise ValueError(
            f"a blinding pulse {gap:g} s ahead of the slot is older than the "
            f"{dead:g} s dead time it has to survive: the detectors have "
            "recovered by the time Alice's photon arrives and nothing is "
            "blinded. Weier arXiv:1101.5289 needs t_i - t_B < tau_D"
        )

    return True


def blank_break(blind, signal, qber=0.0):
    """
    The dead-time blanking attack as a Reading. observed is the QBER and it
    does not move, Eve never touching Alice's photons. eve is her information
    and her own error rate against Bob.
    """

    return Reading(
        attack="blanking",
        observed={
            "qber": qber,
            "naive": 1.0 - _entropy(qber),
        },
        eve={
            "info": blank_leak(blind, signal),
            "error": blank_error(blind, signal),
        },
        note=(
            "Weier arXiv:1101.5289: the pulse is timed outside Bob's "
            "acceptance window, so the clicks it causes are discarded before "
            "any statistic is formed and only the dead time it leaves behind "
            "reaches the key"
        ),
    )


def probe_isolation(reflect, isolator=0.0, stages=0, atten=0.0, bandpass=0.0):
    """
    The round-trip suppression |gamma| in dB a probe pulse meets going in and
    coming back out: Lucamarini arXiv:1506.01989 Eq. (15), 2F + nI + 2A + R with
    every term a POSITIVE dB. The filter and the attenuator are doubled because
    the probe crosses them twice, and the reflection is counted once.

    These are the numbers q.Connector, q.Splice and q.Coupling already carry,
    read as suppressions rather than as forward losses. The paper's own
    combination -- one isolator at I = 60 dB, an attenuator at A = 35 dB and a
    back-reflection at R = 40 dB -- adds to 170 dB.
    """
    std.nonneg("reflect", reflect)

    std.nonneg("isolator", isolator)

    std.nonneg("atten", atten)

    std.nonneg("bandpass", bandpass)

    if stages < 0 or int(stages) != stages:
        raise ValueError("stages must be a whole number of isolators, and not negative")

    return 2.0 * bandpass + stages * isolator + 2.0 * atten + reflect


def probe_photons(injected, clock, isolation):
    """
    mu_out, the mean photon number of the Trojan state leaving Alice per
    modulator setting: Lucamarini arXiv:1506.01989 Eqs. (4) and (16), (N/f_A)
    times the transmittance behind |gamma| dB of suppression. N is capped by
    laser damage rather than by Eve's budget; probe_budget is the inverse. At
    that paper's N = 1e20 photons per second, f_A = 1 GHz and 170 dB this
    returns 1e-6, their target leak.
    """
    std.positive("injected", injected)

    std.positive("clock", clock)

    std.finite("isolation", isolation)

    return injected / clock * std.from_db(isolation)


def probe_budget(target, injected, clock):
    """
    The suppression in dB a transmitter needs to hold its leak at `target`:
    Lucamarini arXiv:1506.01989 Eq. (16) solved for gamma, 10*log10((N/f_A)
    /mu_out). At their N = 1e20, f_A = 1 GHz and mu_out = 1e-6 it is 170 dB,
    which is the number probe_isolation has to add up to.

    Returns a NEGATIVE dB where the target is already met with no isolation at
    all; that is not an error.
    """
    std.positive("target", target)

    std.positive("injected", injected)

    std.positive("clock", clock)

    return std.to_db(target * clock / injected)


def probe_delta(leaked):
    """
    Delta, the imbalance of the quantum coin a leak of mu_out photons leaves:
    Lucamarini arXiv:1506.01989 Eq. (7), (1 - exp(-mu_out)*cos(mu_out))/2. It
    reads the overlap of the coherent states Eve gets back from the four BB84
    settings, Eq. (5), and it is mu_out/2 to first order.

    It is NOT confined to [0, 1/2]: Delta peaks at 0.5335 near mu_out = 3*pi/4,
    so a caller must not read it as a probability. Past 1/2 the coin says
    nothing and probe_phase reports a phase error rate of 1/2.
    """
    std.nonneg("leaked", leaked)

    return 0.5 * (1.0 - math.exp(-leaked) * math.cos(leaked))


def probe_phase(qber, leaked, y1):
    """
    The phase error rate the virtual protocol pays once mu_out photons have left
    Alice's lab carrying her modulator setting: Lucamarini arXiv:1506.01989
    Eq. (7), e' = e + 4d(1 - d)(1 - 2e) + 4(1 - 2d)*sqrt(d(1 - d)e(1 - e)) with
    d = Delta/Y.

    y1 is Y, the SINGLE-PHOTON YIELD, and dividing by it is why a leak harmless
    at short range is not at long. Eq. (7) takes min[Y_X, Y_Y] because Eve
    chooses which basis to starve; qkd's decoy_bounds returns one
    basis-blind y1 with no min to take, so the caller states which basis theirs
    came from. See probe_rate.

    CLAMPED AT 1/2, the only clamp in this module: h is symmetric about 1/2, so
    an unclamped estimate past chance would hand a caller a SMALLER entropy for
    a WORSE leak.

    THIS DOES NOT COMPOSE WITH flaw_phase AND THE TWO ARE NOT ESTIMATES OF ONE
    NUMBER. src/flaws.rs models, in its own words, a qubit source with "no side
    channel, no Trojan-horse leakage, no mode dependency"; the leak priced here
    is exactly the side channel that assumption removes.
    """
    std.closed("qber", qber, 0.5)

    std.unit("y1", y1)

    part = probe_delta(leaked) / y1
    if part >= 0.5:
        return 0.5

    root = math.sqrt(part * (1.0 - part) * qber * (1.0 - qber))
    seen = qber + 4.0 * part * (1.0 - part) * (1.0 - 2.0 * qber) + 4.0 * (1.0 - 2.0 * part) * root

    return min(0.5, seen)


def probe_break(leaked, qber, y1):
    """
    The Trojan-horse attack as a Reading. observed is the QBER and it does not
    move at all: the probe never enters the forward signal, so Alice and Bob
    measure the link they always had while the setting of Alice's modulator
    leaves her lab on the back-reflection. eve is the leak, the coin imbalance
    it buys and the phase error rate that then applies.

    That phase error rate is not a key rate and must not be read as one; see
    probe_rate.
    """

    return Reading(
        attack="injection",
        observed={
            "qber": qber,
            "naive": 1.0 - _entropy(qber),
        },
        eve={
            "leaked": leaked,
            "coin": probe_delta(leaked),
            "phase": probe_phase(qber, leaked, y1),
        },
        note=(
            "Lucamarini arXiv:1506.01989: the probe crosses Alice's modulator "
            "twice and never touches the signal, so no statistic Alice and Bob "
            "can form carries a trace of it and the only defence is the "
            "insertion loss of the components between the fibre and the "
            "modulator"
        ),
    )


def probe_rate(leaked, qber, y1):
    """
    Refused; the message names the four missing pieces.
    """
    std.nonneg("leaked", leaked)

    std.closed("qber", qber, 0.5)

    std.unit("y1", y1)

    raise NotImplementedError(
        "a key rate under the Trojan-horse attack is refused, and four pieces are "
        "missing rather than one. (1) Lucamarini arXiv:1506.01989 Eq. (8) is the only "
        "decoy-state form and it assumes the decoy layer is out of reach -- \"Eve's "
        "only target in the THA considered here is Alice's PM and the devices used by "
        'Alice to implement the decoy-state technique are not touched by the THA" -- '
        "while q.Decoy sets its three intensities with a modulator behind the very "
        "isolators probe_isolation prices, and nothing here bounds an intensity side "
        "channel. (2) Eq. (7) divides Delta by min[Y_X, Y_Y], the single-photon yield "
        "per basis, because Eve chooses which basis to starve; decoy_bounds carries no "
        "basis argument and returns one basis-blind y1 with no min to take. (3) "
        "Eq. (5) pairs one coherent state with each of the four BB84 settings and "
        "Eq. (7)'s coin is the BB84 basis coin: q.TwoStateKeying sends two states, "
        'q.BasisKeying(announce="pair") announces non-orthogonal pairs rather than '
        "bases, and the click families key on a relation between consecutive slots, so "
        'none of them has that map. (4) src/flaws.rs models a qubit source with "no '
        'side channel, no Trojan-horse leakage, no mode dependency", so flaw_phase and '
        "probe_phase are estimates under contradictory source models. Use probe_budget "
        "for the suppression a transmitter needs, and probe_phase for Eq. (7) itself, "
        "naming the basis its y1 came from"
    )


def catalogue():
    """
    Every attack here, as (name, callable, parameters); each callable returns a
    Reading, never a key rate.
    """

    return (
        ("saturation", sat_break, ("v_a", "t", "eta", "v_el", "alpha", "delta")),
        ("calibration", calib_break, ("xi", "ratio", "t", "eta")),
        ("oscillator", lo_break, ("monitored", "sampled", "xi", "t", "eta")),
        ("blinding", blind_break, ("gain", "qber", "always", "never")),
        ("timeshift", shift_break, ("hi", "lo", "qber")),
        ("blanking", blank_break, ("blind", "signal", "qber")),
        ("injection", probe_break, ("leaked", "qber", "y1")),
    )


def assess(
    sat=None,
    calib=None,
    blind=None,
    mismatch=None,
    blank=None,
    lo=None,
    probe=None,
    **kw,
):
    """
    Every attack the given descriptors describe, as {name: Reading}. None omits
    a row. Nothing here is a budget.Entry and nothing here may become one: the
    xi a saturation or calibration attack shows Bob is an artefact of his
    estimator, not light in his fibre, and it belongs to no plane. kw is the
    operating point: v_a, t, eta, v_el, xi, qber, gain, signal, window, dead,
    y1.
    """
    out = {}
    if sat is not None:
        out["saturation"] = sat_break(
            kw["v_a"],
            kw["t"],
            kw["eta"],
            kw.get("v_el", 0.0),
            sat.alpha,
            sat.delta,
            kw.get("xi", 0.0),
            sat.gain,
        )
    if calib is not None:
        out["calibration"] = calib_break(
            kw.get("xi", 0.0),
            calib.ratio,
            kw["t"],
            kw["eta"],
            calib.resend,
        )
    if lo is not None:
        out["oscillator"] = lo_break(
            lo.monitored,
            lo.sampled,
            kw.get("xi", 0.0),
            kw["t"],
            kw["eta"],
            lo.resend,
            lo.slope,
            lo.floor,
        )
    if blind is not None:
        out["blinding"] = blind_break(
            kw.get("gain", 0.0),
            kw.get("qber", 0.0),
            blind.always,
            blind.never,
            blind.passive,
        )
    if mismatch is not None:
        out["timeshift"] = shift_break(mismatch.hi, mismatch.lo, kw.get("qber", 0.0))
    if blank is not None:
        if blank.gap > 0.0:
            if "dead" not in kw:
                raise ValueError(
                    "a Blanking with a declared gap needs dead=, the tau_D of "
                    "q.DeadTime: there is no default recovery time to assume"
                )

            blank_hidden(blank.gap, kw.get("window", 0.0), kw["dead"])

        out["blanking"] = blank_break(blank.blind, blank.signal, kw.get("qber", 0.0))

    if probe is not None:
        if "y1" not in kw:
            raise ValueError(
                "an Injection needs y1=, the single-photon yield its phase "
                "error rate is divided by: Lucamarini Eq. (7) takes "
                "min[Y_X, Y_Y] and there is no default yield to assume. Say "
                "which basis it came from -- decoy_bounds returns one "
                "basis-blind y1"
            )

        out["injection"] = probe_break(
            probe_photons(
                probe.injected,
                probe.clock,
                probe_isolation(
                    probe.reflect,
                    probe.isolator,
                    probe.stages,
                    probe.atten,
                    probe.bandpass,
                ),
            ),
            kw.get("qber", 0.0),
            kw["y1"],
        )

    return out
