import math
from dataclasses import dataclass

from . import std
from .budget import Entry, phase

_H = std.PLANCK
_C = std.LIGHT_SPEED
_NEPER = 10.0 / math.log(10.0)

# Per-form plane_note for the dephasing row, keyed like budget.PHASE_NOTES:
# the label names the form budget.phase() actually evaluated.
DEPHASE_NOTES = {
    "estimator": "Qi (9)-(10) into Kish (80)-(82): input-referred, no /T",
    "literature": "Qi (9)-(10) into Marie-Alleaume (10): input-referred, no /T",
}


@dataclass(frozen=True)
class Coexistence:
    """
    Classical DWDM traffic sharing the fibre. launch is per-channel power in
    dBm, adding linearly; backward is counter-propagating. beta is the Raman
    coefficient in 1/(km nm), default Kumar, Qin & Alleaume arXiv:1412.1403's
    worst C-band value.
    """

    channels: int
    launch: float
    beta: float = 3.0e-9
    wavelength: float = 1531.12e-9
    # Kumar Eq. (6)'s eta_D, "transmittance of DEMUX (Add-Drop Module) place at
    # Bob side" -- NOT a detector efficiency, which is that paper's separate
    # eta_B at Eq. (5). It cancels: raman_photons() multiplies it in and raman()
    # divides it back out through Eq. (7)'s /(eta_D T), so no excess noise
    # depends on it and raman_photons(demux=...) is the one caller it moves.
    # q.Link refuses anything but 1.0 rather than accept an isolation it does
    # not model.
    demux: float = 1.0
    backward: bool = False


@dataclass(frozen=True)
class Backscatter:
    """
    A counter-propagating tone that Rayleigh-scatters into Bob. power in W,
    coeff in 1/s, index the GROUP index of Mandil, Qian and Lo, arXiv:2407.08009
    Eq. (A1). The defaults are the SMF-28 pair and move together: ULL is 6.54
    with 1.4620.
    """

    power: float
    coeff: float = 8.0
    index: float = 1.468
    wavelength: float = 1550.12e-9


@dataclass(frozen=True)
class Backflash:
    """
    Detector breakdown-flash emission, a security side channel. prob is
    P_b = n_Eve / n_sift (Singh arXiv:2502.04081).
    """

    prob: float


@dataclass(frozen=True)
class Dephasing:
    """
    Free-running laser phase diffusion, the floor under budget.phase's v_err.
    linewidth is the SUM of the two lasers' Lorentzian FWHM in Hz; delay is the
    reference-to-symbol time in s. form picks budget.phase's form, and with it
    the plane_note that names the source assemble_extra reports.
    """

    linewidth: float
    delay: float
    form: str = "estimator"


@dataclass(frozen=True)
class Polarisation:
    """
    Polarisation drift and PMD. drift is the rms angle in rad against the
    receiver reference. dispersion, the PMD coefficient in s/sqrt(km), feeds
    dgd() and security() alone: no verified closed form maps a DGD onto an
    excess noise, so q.Link refuses a non-zero one by name.
    """

    drift: float
    dispersion: float = 0.0


@dataclass(frozen=True)
class Modulator:
    """
    IQ-modulator imperfections. ratio is d = b/a, the I-to-Q amplitude
    imbalance (1.0 balanced); angle is the quadrature bias-point offset in rad
    (0.0 orthogonal); extinction is the LINEAR r = I_on / I_off.

    THE THREE FIELDS REACH DIFFERENT PATHS AND NEITHER PATH READS BOTH. ratio
    and angle are the quadrature term imbalance() charges as an excess noise;
    extinction is a click-protocol QBER, charged by a q.BasisKeying or
    q.PolarisationKeying q.Link and reported by security(). A quadrature link
    refuses a declared r by name and a click link refuses a declared ratio or
    angle by name.
    """

    ratio: float = 1.0
    angle: float = 0.0
    extinction: float | None = None


@dataclass(frozen=True)
class Timing:
    """
    Sampling-instant error against a Gaussian pulse. jitter is the rms error in
    s; width is the field envelope parameter w in s of exp(-t^2/(2 w^2)).
    """

    jitter: float
    width: float


@dataclass(frozen=True)
class DeadTime:
    """
    Click-detector recovery. dead is in s; afterpulse is p_AP given a previous
    detection; paralysable means an event inside the dead time extends it --
    an SPAD is not, a pair sharing a basis is (Rogers arXiv:0706.1449).
    """

    dead: float
    afterpulse: float = 0.0
    # Read by saturate() alone, and says nothing at dead = 0, where both branches
    # are the same rate: q.Link refuses that pair by name rather than drop a flag.
    paralysable: bool = False


# std.entropy is the one implementation; test_entropy_shared pins this alias.
_entropy = std.entropy


def raman_photons(
    power,
    length,
    *,
    alpha=0.2,
    beta=3.0e-9,
    wavelength=1531.12e-9,
    demux=1.0,
    backward=False,
):
    """
    Mean spontaneous-Raman photons per detection mode at Bob's input: Kumar,
    Qin & Alleaume, New J. Phys. 17, 043027 (2015), Eq. (6); the 1/2 is the LO's
    polarisation-mode selectivity. KUMAR is first on this 2015 paper and QIN on
    the 2016 one (novel.md W26). beta per (km nm), length in km, alpha in dB/km.
    """
    lin = alpha / _NEPER
    if backward:
        span = (1.0 - math.exp(-2.0 * lin * length)) / (2.0 * lin)
    else:
        span = length * math.exp(-lin * length)

    flux = wavelength**3 / (_H * _C * _C) * beta * 1e9

    return 0.5 * flux * demux * power * span


# The five keyword-only parameters are re-declared rather than forwarded as **kw
# (-7 lines): this is the outer call site, and for hardware inputs the signature
# with its defaults IS the documentation.
def raman(
    power,
    length,
    t,
    *,
    alpha=0.2,
    beta=3.0e-9,
    wavelength=1531.12e-9,
    demux=1.0,
    backward=False,
):
    """
    Raman excess noise from co-propagating classical channels, input-referred:
    Kumar, Qin & Alleaume arXiv:1412.1403 Eq. (7), 2<N>/(eta_D T). The 2 is
    Laudenbach arXiv:1703.09278 Eq. (9.59), Bob-plane, and /T refers it to the
    input; eta_D cancels but is kept, so raman_photons() matches the paper.
    """
    photons = raman_photons(
        power,
        length,
        alpha=alpha,
        beta=beta,
        wavelength=wavelength,
        demux=demux,
        backward=backward,
    )

    return 2.0 * photons / (demux * t)


def rayleigh_fraction(length, *, alpha=0.2, coeff=8.0, index=1.468):
    """
    Fraction of a CW launch that returns by Rayleigh backscatter: OUR integration
    of Mandil, Qian and Lo, arXiv:2407.08009 Eq. (A1) over a round trip. alpha in
    dB/km, its default 0.2 a bare-fibre datasheet maximum.
    """
    lin = alpha / _NEPER
    speed = _C / index / 1e3

    return coeff * (1.0 - math.exp(-2.0 * lin * length)) / (lin * speed)


def rayleigh_photons(power, length, symbol, *, alpha=0.2, coeff=8.0, index=1.468, wavelength=1550.12e-9):
    """
    Mean Rayleigh-backscattered photons per detection mode: returned power f*P
    into photons per symbol by Laudenbach arXiv:1703.09278 Eq. (9.60), the 1/2
    polarisation-mode selectivity (arXiv:2407.08009 finds the scattered
    polarisation random). Elastic, so no filter applies.
    """
    frac = rayleigh_fraction(length, alpha=alpha, coeff=coeff, index=index)
    energy = _H * _C / wavelength

    return 0.5 * frac * power * symbol / energy


def rayleigh(
    power,
    length,
    symbol,
    t,
    *,
    alpha=0.2,
    coeff=8.0,
    index=1.468,
    wavelength=1550.12e-9,
):
    """
    Rayleigh-backscattering excess noise, input-referred. Same conversion as
    raman(): Laudenbach Eq. (9.59) xi = 2<n> at the Bob plane, then /T. For
    two-way architectures and counter-propagating classical channels; the
    one-way double-scattering term is NOT modelled.
    """
    photons = rayleigh_photons(
        power,
        length,
        symbol,
        alpha=alpha,
        coeff=coeff,
        index=index,
        wavelength=wavelength,
    )

    return 2.0 * photons / t


def backflash_leak(prob, sift=1.0):
    """
    Fraction of the sifted key Eve learns from detector backflash: P_b P_sift,
    Singh, IEEE Photonics J. 17, 7600206 (2025), arXiv:2502.04081. A security
    observable, not an excess noise.
    """
    std.closed("backflash prob", prob)

    return prob * sift


def backflash_rate(sift, prob, qber, f=1.16):
    """
    Sifted-key rate discounted for backflash leakage, clamped at zero. Singh
    2025 (arXiv:2502.04081): P_sift (1 - P_b - f h(e)). f defaults to qkd's
    house 1.16 (Lutkenhaus, PRA 61, 052304 (2000), Table I), which
    q.IndividualAttack also takes; pass f=1.15 for Singh's.
    """
    std.closed("backflash prob", prob)

    return max(0.0, sift * (1.0 - prob - f * _entropy(qber)))


def phase_variance(linewidth, delay):
    """
    Wiener phase-diffusion variance accumulated over a delay, in rad^2: Qi,
    PRX 5, 041009 (2015), Eqs. (9)-(10). Pass the SUM of the two linewidths;
    the beat phase diffuses at the sum rate. Zero is admitted on both: a line
    of no width and a delay of no length each accumulate no phase.
    """
    std.nonneg("linewidth", linewidth)
    std.nonneg("delay", delay)

    return 2.0 * math.pi * linewidth * delay


def coherence(linewidth, delay):
    """
    Lorentzian coherence decay of the beat note over a delay, exp(-V/2) over
    phase_variance(): the ceiling on any phase estimator. It is |E[exp(i phi)]|
    and so lies in (0, 1]; phase_variance's guards are what hold it there.
    """

    return math.exp(-0.5 * phase_variance(linewidth, delay))


def dephasing(v_a, linewidth, delay, xi=0.0, *, form="estimator"):
    """
    Excess noise from the linewidth-limited coherence floor, input-referred:
    budget.phase() over phase_variance() rather than a DSP residual, so it
    lower-bounds the same v_err.
    """

    return phase(v_a, phase_variance(linewidth, delay), xi, form=form)


def pol_fading(drift):
    """
    (<sqrt(eta)>^2, Var(sqrt(eta))) for a drifting polarisation overlap:
    eta = cos^2(theta) against the receiver reference, the visibility factor of
    Sharma arXiv:2409.05802, averaged over theta ~ N(0, s^2).
    """
    root = math.exp(-0.5 * drift * drift)
    mean = 0.5 * (1.0 + math.exp(-2.0 * drift * drift))

    return root * root, mean - root * root


def polarisation(v_a, drift):
    """
    Excess noise from polarisation drift, input-referred. Usenko, New J. Phys.
    14, 093048 (2012), arXiv:1208.4307: a fading channel is a fixed one of
    transmittance <sqrt(eta)>^2 plus a Bob-plane Var(sqrt(eta))*V_A, referred
    by that T_eff, NOT by T, which cancels. This is the NOISE half alone;
    pol_fading()[0] is the transmittance half and fading_factor() composes it.
    """
    eff, var = pol_fading(drift)

    return var * v_a / eff


def visibility(angle):
    """
    Interference visibility under a polarisation mode mismatch. Sharma,
    arXiv:2409.05802: V = cos(Phi) on balanced arms, e = (1 - V)/2.
    """

    return math.cos(angle)


def dgd(dispersion, length):
    """
    Mean differential group delay of a fibre span, in s: D_PMD sqrt(L) for
    randomly mode-coupled fibre (Antonelli arXiv:2408.01754).
    """

    return dispersion * math.sqrt(length)


def imbalance(v_a, ratio=1.0, angle=0.0):
    """
    Modulation-imbalance excess noise of an IQ modulator, input-referred: Wang,
    arXiv:2503.10168, Eq. (E1) with the leading T*eta_e stripped. The bracket
    |d exp(i theta_m) - 1|^2 carries bias drift and amplitude imbalance in one
    term.
    """
    miss = ratio * ratio * math.sin(angle) ** 2 + (ratio * math.cos(angle) - 1.0) ** 2

    return 0.5 * v_a * miss


def extinction(ratio, qber=0.0):
    """
    Click-protocol QBER of a transmitter with a finite extinction ratio.
    Huang, Yin, Wang, Li, Chen & Han, "Effect of Intensity Modulator Extinction
    on Practical Quantum Key Distribution System", arXiv:1206.6591, Eqs. (1)
    and (5): a four-path BB84 transmitter, one intensity modulator per state,
    leaks a maximally mixed 4/(r+3) fraction that errs half the time. A QBER,
    NOT a CV excess noise -- no verified form maps r onto xi.

    The 4 counts the four paths, so no six-state form of it is published;
    q.Link refuses one at bases = 3 rather than substituting this constant.
    """
    std.positive("extinction ratio", ratio)

    leak = 4.0 / (ratio + 3.0)

    return (1.0 - leak) * qber + leak / 2.0


def jitter_fading(jitter, width):
    """
    (<sqrt(eta)>^2, Var(sqrt(eta))) for a mistimed Gaussian pulse. OUR overlap,
    not verbatim in any source: a matched filter at offset delta ~ N(0, j^2)
    attenuates the amplitude by exp(-delta^2/(4 w^2)), so with u = (j/w)^2,
    <sqrt(eta)> = (1 + u/2)^(-1/2) and <eta> = (1 + u)^(-1/2).
    """
    u = (jitter / width) ** 2
    root = 1.0 / math.sqrt(1.0 + 0.5 * u)
    mean = 1.0 / math.sqrt(1.0 + u)

    return root * root, mean - root * root


def timing(v_a, jitter, width):
    """
    Excess noise from sampling-instant jitter, input-referred: Usenko
    arXiv:1208.4307 again, referred by T_eff, not by T. The NOISE half alone,
    as polarisation() is; jitter_fading()[0] is the transmittance half and
    fading_factor() composes it.
    """
    eff, var = jitter_fading(jitter, width)

    return var * v_a / eff


def fading_factor(*, pol=None, clock=None):
    """
    The transmittance halves of the fading rows assemble_extra() returns the
    noise halves of, multiplied together; 1.0 where neither is described. Each
    is <sqrt(eta)>^2, the fixed transmittance Usenko, New J. Phys. 14, 093048
    (2012), arXiv:1208.4307 splits a fading channel into alongside a Bob-plane
    Var(sqrt(eta))*V_A. The two halves are ONE channel and pin exactly:
    T*f*(V_A + xi_fade) == T*V_A*<eta>, so charging the noise half against an
    unattenuated T overstates the rate. polarisation() and timing() return the
    noise half alone.
    """
    out = 1.0
    if pol is not None:
        out = out * pol_fading(pol.drift)[0]

    if clock is not None:
        out = out * jitter_fading(clock.jitter, clock.width)[0]

    return out


def saturate(rate, dead, paralysable=False):
    """
    Detected click rate after dead-time losses. Krause arXiv:2507.10361 Eq. (1)
    non-paralysable, saturating at 1/tau_d; paralysable peaks there and falls.
    """
    std.nonneg("dead time", dead)

    if paralysable:
        return rate * math.exp(-rate * dead)

    return rate / (1.0 + rate * dead)


def afterpulse(mu, eta, dark, prob, edet=0.0):
    """
    (gain, QBER) of a click receiver whose detectors afterpulse: Papapanos,
    arXiv:2010.03358, Eqs. (4), (7) and (8), an afterpulse weighted e_0 = 1/2.
    The click sources are COMPOSED, not added as Q_mu is written, so p_AP = 0
    collapses onto decoy_gain. dark is p_DC, the RECEIVER's per-pulse floor.
    """
    std.closed("afterpulse prob", prob)

    floor = (1.0 + prob) * dark
    click = 1.0 - math.exp(-eta * mu)

    # (1 + p_AP) can carry a probability past 1; decoy_gain caps its own too.
    lit = min(click * (1.0 + prob), 1.0)
    gain = floor + lit - floor * lit
    err = (0.5 * floor + (edet + 0.5 * prob) * click) / gain

    return gain, err


def catalogue():
    """
    Every model here, as (name, callable, parameters). The first six return an
    input-referred excess noise in SNU; the rest have no xi form.
    """

    return (
        ("raman", raman, ("power", "length", "t")),
        ("rayleigh", rayleigh, ("power", "length", "symbol", "t")),
        ("dephasing", dephasing, ("v_a", "linewidth", "delay")),
        ("polarisation", polarisation, ("v_a", "drift")),
        ("imbalance", imbalance, ("v_a", "ratio", "angle")),
        ("timing", timing, ("v_a", "jitter", "width")),
        ("backflash_leak", backflash_leak, ("prob", "sift")),
        ("backflash_rate", backflash_rate, ("sift", "prob", "qber")),
        ("extinction", extinction, ("ratio", "qber")),
        ("visibility", visibility, ("angle",)),
        ("dgd", dgd, ("dispersion", "length")),
        ("saturate", saturate, ("rate", "dead")),
        ("afterpulse", afterpulse, ("mu", "eta", "dark", "prob")),
    )


def assemble_extra(
    *,
    v_a,
    t,
    length=None,
    alpha=0.2,
    symbol=None,
    coexist=None,
    probe=None,
    dephase=None,
    pol=None,
    modulator=None,
    clock=None,
):
    """
    The impairment entries this module can justify, input-referred, as
    budget.Entry values composing with budget.assemble(...).entries. None omits
    a row; a present descriptor contributes its value, including 0.0.
    Click-protocol observables are excluded; see security().
    """
    std.positive("v_a", v_a)
    std.unit("t", t)

    if coexist is not None and length is None:
        raise ValueError("coexist needs length")

    if probe is not None and (length is None or symbol is None):
        raise ValueError("probe needs length and symbol")

    entries = []
    if coexist is not None:
        watts = coexist.channels * 10.0 ** (coexist.launch / 10.0) * 1e-3
        entries.append(
            Entry(
                "raman",
                raman(
                    watts,
                    length,
                    t,
                    alpha=alpha,
                    beta=coexist.beta,
                    wavelength=coexist.wavelength,
                    demux=coexist.demux,
                    backward=coexist.backward,
                ),
                "Kumar (6)-(7): stated Bob-plane, /T applied",
            )
        )
    if probe is not None:
        entries.append(
            Entry(
                "rayleigh",
                rayleigh(
                    probe.power,
                    length,
                    symbol,
                    t,
                    alpha=alpha,
                    coeff=probe.coeff,
                    index=probe.index,
                    wavelength=probe.wavelength,
                ),
                "Sagnac (A1) + Laudenbach (9.59): Bob-plane, /T applied",
            )
        )
    if dephase is not None:
        entries.append(
            Entry(
                "dephasing",
                dephasing(v_a, dephase.linewidth, dephase.delay, form=dephase.form),
                DEPHASE_NOTES[dephase.form],
            )
        )
    if pol is not None:
        entries.append(
            Entry(
                "polarisation",
                polarisation(v_a, pol.drift),
                "Usenko: fading, referred by T_eff not by T",
            )
        )
    if modulator is not None:
        entries.append(
            Entry(
                "imbalance",
                imbalance(v_a, modulator.ratio, modulator.angle),
                "arXiv:2503.10168 (E1): T*eta_e stripped, input-referred",
            )
        )
    if clock is not None:
        entries.append(
            Entry(
                "timing",
                timing(v_a, clock.jitter, clock.width),
                "Gaussian overlap into Usenko: referred by T_eff",
            )
        )

    return tuple(entries)


def security(backflash=None, dead=None, modulator=None, pol=None, **kw):
    """
    Click-protocol observables and security flags as {name: value}, the
    counterpart of assemble_extra for what is not an excess noise. None omits a
    row. kw is the operating point: sift, qber, rate, mu, eta, dark, edet,
    length, f.
    """
    out = {}
    if backflash is not None:
        sift = kw.get("sift", 1.0)
        out["backflash_leak"] = backflash_leak(backflash.prob, sift)
        out["backflash_rate"] = backflash_rate(sift, backflash.prob, kw.get("qber", 0.0), kw.get("f", 1.16))
    if dead is not None:
        if "rate" in kw:
            out["click_rate"] = saturate(kw["rate"], dead.dead, dead.paralysable)
        if "mu" in kw:
            out["gain"], out["qber"] = afterpulse(
                kw["mu"],
                kw.get("eta", 1.0),
                kw.get("dark", 0.0),
                dead.afterpulse,
                kw.get("edet", 0.0),
            )
    if modulator is not None and modulator.extinction is not None:
        out["extinction"] = extinction(modulator.extinction, kw.get("qber", 0.0))
    if pol is not None:
        out["visibility"] = visibility(pol.drift)
        if pol.dispersion > 0.0 and "length" in kw:
            out["dgd"] = dgd(pol.dispersion, kw["length"])

    return out
