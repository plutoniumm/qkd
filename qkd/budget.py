import math
import warnings
from dataclasses import dataclass

from . import std
from .components import Connector, Coupling, Splice

_SQRT12 = math.sqrt(12.0)

_H = std.PLANCK
_C = std.LIGHT_SPEED

# A dBm/nm density enters Laudenbach Eq. (9.63) as 10^(N/10) * 1e6 W/m.
_MW_NM = 1.0e6

# Optical order: Alice's output (where V_A is defined), the far end of the
# span, the detector face before eta, after eta.
PLANES = (
    "channel_input",
    "channel_output",
    "detector_input",
    "post_detection",
)

# Accepted spellings of two PLANES names.
_ALIAS = {
    "input": "channel_input",
    "bob": "detector_input",
}

# Loss sites in optical order: Alice's output, the fibre run, Bob's input.
_ORDER = ("launch", "span", "receive")

# rad^2. Domain of the LITERATURE form of phase() only; past it, warn and
# return. The estimator form has no domain. The 0.1 is Kish App. F's own figure
# but NOT a domain the paper puts on Eq. (85): the sentence after it, "for small
# phase noise sigma^2 < 0.1, xi_x = xi_p ~ sigma^2 * 2*V_A", bounds the FURTHER
# linearisation, and Eq. (85) is printed with no stated range. Read here as the
# range over which the literature form tracks the estimator one, a house reading.
PHASE_LIMIT = 0.1

PHASE_FORMS = ("estimator", "literature")

# Per-form plane_note; q.Link emits the same row from its own DSP v_err.
PHASE_NOTES = {
    "estimator": "Kish (80)-(82) inferred channel: input-referred, no /T",
    "literature": "Marie-Alleaume (10): stated input-referred, no /T",
}


class PhaseDomainWarning(UserWarning):
    """
    phase() was called past the domain of the form it was asked for.
    """


@dataclass(frozen=True)
class Entry:
    source: str
    xi: float
    plane_note: str


@dataclass(frozen=True)
class Loss:
    """
    One line of the loss budget; db is the whole insertion loss, count folded
    in, and nonnegative.
    """

    source: str
    db: float
    site: str

    @property
    def transmittance(self):
        return std.from_db(self.db)


# Itemised loss components to the row name each emits; chain() refuses others.
_KINDS = {
    Connector: "connector",
    Coupling: "coupling",
    Splice: "splice",
}


def chain(t, items):
    """
    (launch, span, receive, rows) for a link whose fibre span alone transmits
    t, rows in optical order. Span-born noise is referred by launch*span, not
    by the total. Raises on an unknown, duplicate or non-finite loss, or a T
    that underflows.
    """
    std.unit("t", t)

    seen = {}
    for item in items:
        kind = _KINDS.get(type(item))
        if kind is None:
            raise ValueError(
                f"{type(item).__name__} is not an itemised loss: losses takes q.Connector, q.Splice or q.Coupling"
            )

        name = f"{kind}_{item.site}"
        if name in seen:
            raise ValueError(f"duplicate {name}: one descriptor per kind per site, with count= for how many")

        db = item.loss * item.count
        if not math.isfinite(db):
            raise ValueError(
                f"loss must be finite: {type(item).__name__}(loss={item.loss}, "
                f"count={item.count}) at {item.site} is {db} dB"
            )

        seen[name] = Loss(name, db, item.site)

    fiber = Loss("fiber_span", std.to_db(t), "span")
    rows = []
    parts = dict.fromkeys(_ORDER, 1.0)
    for site in _ORDER:
        if site == "span":
            rows.append(fiber)
            parts[site] = t

        for row in seen.values():
            if row.site == site:
                rows.append(row)
                parts[site] = parts[site] * row.transmittance

    total = parts["launch"] * parts["span"] * parts["receive"]
    if total <= 0.0:
        raise ValueError(
            "the loss chain underflows the transmittance to zero: "
            f"{sum(row.db for row in rows):.6g} dB leaves nothing to refer a budget through"
        )

    return parts["launch"], parts["span"], parts["receive"], tuple(rows)


@dataclass(frozen=True)
class Budget:
    """
    An excess-noise budget and its loss chain. entries are always
    input-referred, in SNU; T is the whole channel-input-to-detector
    transmittance.
    """

    entries: tuple[Entry, ...]
    T: float
    launch: float = 1.0
    receive: float = 1.0
    losses: tuple[Loss, ...] = ()
    # Set and cleared together, so `v_err is None` and `phase_form is None`
    # always agree.
    v_err: float | None = None
    phase_form: str | None = None

    @property
    def inferred(self):
        """
        This budget's T as parameter estimation infers it through the phase
        row, T*exp(-v_err); see infer(). Equal to T where no phase row was
        formed, so it may be read unconditionally. Raises on the literature
        form, which has no transmittance half.
        """
        if self.v_err is None:
            return self.T

        return infer(self.T, self.v_err, form=self.phase_form)

    @property
    def total(self):
        """
        Sum of all contributions, referred to the channel input, in SNU.
        """

        return sum(e.xi for e in self.entries)

    @property
    def span(self):
        return self.T / (self.launch * self.receive)

    def factor(self, plane, eta=None):
        """
        Factor referring an input-referred noise to `plane`: 1, T/receive, T,
        T*eta. post_detection without eta= raises.
        """
        name = _ALIAS.get(plane, plane)
        if name == "channel_input":
            return 1.0

        if name == "channel_output":
            return self.T / self.receive

        if name == "detector_input":
            return self.T

        if name != "post_detection":
            raise ValueError(f"plane must be one of {PLANES}, not {plane!r}")

        if eta is None:
            raise ValueError(
                "post_detection needs eta=: a budget carries no quantum "
                "efficiency and will not assume a perfect detector"
            )

        std.unit("eta", eta)

        return self.T * eta

    def at(self, plane, eta=None):
        """
        The budget as {source: xi} at `plane`, one of PLANES.
        """
        scale = self.factor(plane, eta)

        return {e.source: scale * e.xi for e in self.entries}

    def refer(self, xi, start, end, eta=None):
        """
        Refer an xi this budget did not produce from `start` to `end`.
        Detector-plane to input-referred divides by T.
        """

        return xi * self.factor(end, eta) / self.factor(start, eta)

    def table(self):
        """
        Aligned text table: the loss chain, then one xi row per source.
        """
        lines = []
        if len(self.losses) > 1:
            optics = [("loss", "dB", "T", "site")]
            for x in self.losses:
                optics.append((x.source, f"{x.db:.4f}", f"{x.transmittance:.6f}", x.site))
            optics.append(("link", f"{std.to_db(self.T):.4f}", f"{self.T:.6f}", ""))
            lines = std.grid(optics) + [""]

        rows = [("source", "xi @ input", "xi @ bob", "plane")]
        for e in self.entries:
            rows.append((e.source, f"{e.xi:.4e}", f"{self.T * e.xi:.4e}", e.plane_note))
        rows.append(("total", f"{self.total:.4e}", f"{self.T * self.total:.4e}", ""))

        return "\n".join(lines + std.grid(rows))


def phase(v_a, v_err, xi=0.0, *, form="estimator"):
    """
    Residual-phase noise, input-referred (channel input, SNU). Default is
    (V_A + xi)*expm1(v_err), which never appears literally in the source: it
    follows from the inferred channel of Kish, Quantum 8, 1382 (2024), App. E
    (76)-(82), by rearranging Eqs. (80)-(82). form="literature" is
    2*V_A*(1 - exp(-v_err/2)): Marie & Alleaume, PRA 95, 012316 (2017),
    (10)-(11), reproduced as Kish App. F (85), and under-predicts by
    (e^v + e^(v/2))/2. infer() returns the transmittance half of the same
    effect; charging this row against an unattenuated T is half the derivation.
    """
    if form not in PHASE_FORMS:
        raise ValueError(f"form must be one of {PHASE_FORMS}, not {form!r}")

    std.finite("v_a", v_a)

    std.finite("xi", xi)

    # v_err = -2*ln<cos phi> in rad^2, unbounded and +inf at zero mean cosine.
    # SimOut.v_wrap, the wrapped second moment, is a different quantity that
    # silently caps every xi. Never swap them.
    if math.isnan(v_err):
        raise ValueError("v_err must be a number: nan is not a phase variance")

    if v_err < 0.0:
        raise ValueError("v_err is a variance and must be nonnegative")

    if xi < 0.0:
        raise ValueError("xi is an excess noise and must be nonnegative")

    if form == "literature":
        if xi != 0.0:
            raise ValueError(
                f"form='literature' cannot consume xi = {xi:.4g} SNU: "
                "2*V_A*(1 - exp(-v_err/2)) renormalises nothing, so the channel "
                "noise would be dropped. Use the default form='estimator'"
            )

        if v_err > PHASE_LIMIT:
            warnings.warn(
                f"v_err = {v_err:.4g} rad^2 is past the {PHASE_LIMIT} rad^2 limit of "
                "form='literature': xi_phase = 2*V_A*(1 - exp(-v_err/2)) saturates at "
                f"2*V_A = {2.0 * v_a:.4g} SNU while the estimator it models diverges, so "
                "it under-predicts the excess noise (23% at v_err = 0.27, 19x at 3.48, "
                "from the closed ratio) and a budget built on it over-predicts the rate",
                PhaseDomainWarning,
                stacklevel=2,
            )

        return 2.0 * v_a * (1.0 - math.exp(-v_err / 2.0))

    return (v_a + xi) * math.expm1(v_err)


def infer(t, v_err, *, form="estimator"):
    """
    The transmittance parameter estimation infers through a residual phase,
    t*exp(-v_err), stated in whatever plane t was: the TRANSMITTANCE half of
    the effect phase() returns the noise half of. Kish, Quantum 8, 1382
    (2024), App. E (78) derives eta_I = eta*r_bar^2 from the surviving
    correlation r_bar = exp(-v_err/2). The two halves are one rotation, so
    infer(t, v)*(v_a + xi + phase(v_a, v, xi)) == t*(v_a + xi) identically:
    charging the noise half against an unattenuated t overstates the rate.
    v_err = +inf returns 0.0. There is no domain here; form="literature" is
    refused rather than approximated.
    """
    if form not in PHASE_FORMS:
        raise ValueError(f"form must be one of {PHASE_FORMS}, not {form!r}")

    std.unit("t", t)

    # Deliberately not pooled with phase()'s identical prologue; each carries its
    # own v_err gotcha. v_err, not v_wrap: r_bar = exp(-v_err/2) IS the mean
    # cosine, by definition.
    if math.isnan(v_err):
        raise ValueError("v_err must be a number: nan is not a phase variance")

    if v_err < 0.0:
        raise ValueError("v_err is a variance and must be nonnegative")

    if form == "literature":
        raise ValueError(
            "form='literature' has no transmittance half to return: "
            "2*V_A*(1 - exp(-v_err/2)) is derived holding the transmittance "
            "estimate fixed, the assumption Kish App. E names as its error. "
            "Use the default form='estimator', or carry t as it stands"
        )

    return t * math.exp(-v_err)


# Every row below forms its noise at BOB's plane and returns it /T, referred to
# the channel input. In dac and rin_sig the t cancels algebraically and stays:
# the pair is the referring discipline made visible, and cancelling it erases
# the plane the number is stated in.
def dac(v_a, bits, t):
    """
    DAC quantisation noise, Bob-plane, /T applied. Laudenbach arXiv:1703.09278
    Eq. (9.37) with delta_U = R_U/(2^bits*sqrt 12) (their Eq. (9.105)) and
    g*R_U = U_pi.
    """
    std.finite("v_a", v_a)

    std.finite("bits", bits)

    std.unit("t", t)

    d = math.pi / (2.0**bits * _SQRT12)
    bob = t * v_a * (d + d * d / 2.0) ** 2

    return bob / t


def adc(bits, t, mu=2.0, ratio=10.0):
    """
    ADC quantisation noise at Bob, Bob-plane, /T applied. Laudenbach
    arXiv:1703.09278 Eq. (9.108), quantisation term only: ratio is
    R_U/sigma_shot (default 10, +/-5 sigma of full range) and mu = 2 under
    heterodyne, their Eq. (9.79).
    """
    std.finite("bits", bits)

    std.finite("mu", mu)

    std.finite("ratio", ratio)

    std.unit("t", t)

    bob = mu * ratio**2 / (12.0 * 4.0**bits)

    return bob / t


def rin_sig(v_a, rin_db, bandwidth, t):
    """
    Signal-laser RIN excess noise, Bob-plane, /T applied. Laudenbach
    arXiv:1703.09278 Eq. (9.9), RIN linear from rin_db in dBc/Hz and bandwidth
    the detection bandwidth in Hz.
    """
    std.finite("v_a", v_a)

    std.finite("rin_db", rin_db)

    std.finite("bandwidth", bandwidth)

    std.unit("t", t)

    lin = 10.0 ** (rin_db / 10.0)
    bob = t * v_a * math.sqrt(lin * bandwidth)

    return bob / t


def rin_lo(rin_db, bandwidth, quad_var, t):
    """
    Local-oscillator RIN excess noise, Bob-plane, /T applied. Laudenbach
    arXiv:1703.09278 Eq. (9.21), quad_var the Bob-plane quadrature variance
    excluding the LO-RIN term itself.
    """
    std.finite("rin_db", rin_db)

    std.finite("bandwidth", bandwidth)

    std.finite("quad_var", quad_var)

    std.unit("t", t)

    lin = 10.0 ** (rin_db / 10.0)
    bob = lin * bandwidth * quad_var / 4.0

    return bob / t


def rin_rows(v_a, rin_db, bandwidth, t):
    """
    The two rows a described laser RIN and detection bandwidth emit, as
    (rin_sig, rin_lo) Entries, input-referred. rin_lo's V(q) is Bob's
    quadrature variance excluding the LO-RIN term itself, t*V_A + 1, which is
    a choice rather than a reading and is made here alone. bandwidth is in Hz
    and rin_db in dBc/Hz, so neither goes through std.from_db.
    """

    return (
        Entry(
            "rin_sig",
            rin_sig(v_a, rin_db, bandwidth, t),
            "Laudenbach (9.9): stated Bob-plane, /T applied",
        ),
        Entry(
            "rin_lo",
            rin_lo(rin_db, bandwidth, t * v_a + 1.0, t),
            "Laudenbach (9.21): stated Bob-plane, /T applied",
        ),
    )


def raman_width(bandwidth, wavelength=1550.12e-9):
    """
    Filter bandwidth in m matched to a detection bandwidth in Hz,
    dlambda = lambda^2 B/c: the conversion behind Laudenbach
    arXiv:1703.09278's "dlambda ~ 8 pm at B = 1 GHz, f = 193.4 THz,
    lambda = 1550 nm" under Eq. (9.62). In a coherent receiver the LO is the
    filter that counts, so this is the width that reaches xi however wide the
    optical passband in front of it happens to be.
    """
    std.finite("bandwidth", bandwidth)

    std.nonneg("bandwidth", bandwidth)

    std.finite("wavelength", wavelength)

    std.positive("wavelength", wavelength)

    return wavelength * wavelength * bandwidth / _C


def raman(n_ram, width, symbol, t, *, wavelength=1550.12e-9):
    """
    Spontaneous-Raman noise from classical channels sharing the fibre,
    Bob-plane, /T applied. Laudenbach arXiv:1703.09278 Eq. (9.63),
    xi = 2 dlambda 10^(N/10) tau/(hf) * 1e6: n_ram is the MEASURED spectral
    noise density N_Ram in dBm/nm, width the filter bandwidth dlambda in m
    (raman_width() derives it), symbol the integration time tau in s. Pass
    t = T_launch*T_span rather than the whole T -- the photons are born inside
    the span, so Bob's receive optics attenuate them along with the signal and
    cancel out of the ratio. Laudenbach carries NO 1/2 for polarisation, his
    footnote to (9.63) cancelling the PBS against the LO's doubled mixing
    bandwidth, where impairments.raman, on Kumar's 2015 paper, keeps it: one
    density reads a factor 2 higher here. Budget-only: no q.Link run consumes a
    measured density, q.Coexistence carrying Kumar's launch-power form instead,
    and declaring both would count one mechanism twice.
    """
    std.finite("n_ram", n_ram)

    std.finite("width", width)

    std.nonneg("width", width)

    std.finite("symbol", symbol)

    std.nonneg("symbol", symbol)

    std.finite("wavelength", wavelength)

    std.positive("wavelength", wavelength)

    std.unit("t", t)

    # hf = hc/lambda is the photon energy at the quantum channel, so <n> = P
    # tau/(hf) is Eq. (9.60) and the leading 2 is Eq. (9.59)'s xi = 2<n>.
    power = width * 10.0 ** (n_ram / 10.0) * _MW_NM
    bob = 2.0 * power * symbol * wavelength / (_H * _C)

    return bob / t


def assemble(
    *,
    v_a,
    t,
    v_err=None,
    xi=0.0,
    phase_form="estimator",
    rin=None,
    dac_bits=None,
    adc_bits=None,
    bandwidth=None,
    raman_db=None,
    wavelength=1550.12e-9,
    losses=(),
    mu=2.0,
    ratio=10.0,
):
    """
    The excess-noise budget of the named hardware, input-referred. None omits a
    row; a present parameter contributes its value, including 0.0. t is the
    SPAN ALONE, losses multiplying into it. xi and phase_form reach the phase
    row only. raman_db is N_Ram in dBm/nm and needs bandwidth, which reaches it
    as both the matched filter width and the integration time and therefore
    cancels; wavelength reaches raman_db alone. All checked finite, v_err
    apart, which admits +inf. v_err and phase_form are also carried onto the
    Budget so .inferred can return the transmittance half of the phase row; T
    stays the physical chain and no returned xi moves because of it.
    """
    std.finite("v_a", v_a)

    std.positive("v_a", v_a)

    std.unit("t", t)

    for name, value in (
        ("xi", xi),
        ("mu", mu),
        ("ratio", ratio),
        ("dac_bits", dac_bits),
        ("adc_bits", adc_bits),
        ("rin", rin),
        ("bandwidth", bandwidth),
        ("raman_db", raman_db),
        ("wavelength", wavelength),
    ):
        if value is not None:
            std.finite(name, value)

    if (rin is None) != (bandwidth is None):
        # bandwidth serves the raman row too, so it may stand without rin; rin
        # without it still raises.
        if raman_db is None or rin is not None:
            raise ValueError("rin and bandwidth must be given together")

    if raman_db is not None:
        if bandwidth is None:
            raise ValueError(
                "raman_db and bandwidth must be given together: Laudenbach (9.63) "
                "reads a spectral density through a filter, and the width that "
                "reaches xi is the lambda^2 B/c the LO mixes"
            )

        std.positive("bandwidth", bandwidth)

    if v_err is None and xi != 0.0:
        raise ValueError("xi only reaches the phase row, which needs v_err")

    launch, span, receive, rows = chain(t, losses)
    total = launch * span * receive
    entries = []
    if v_err is not None:
        note = PHASE_NOTES
        if phase_form not in note:
            raise ValueError(f"phase_form must be one of {PHASE_FORMS}")

        entries.append(
            Entry(
                "phase",
                phase(v_a, v_err, xi, form=phase_form),
                note[phase_form],
            )
        )
    if dac_bits is not None:
        entries.append(
            Entry(
                "dac",
                dac(v_a, dac_bits, total),
                "Laudenbach (9.37): stated Bob-plane, /T applied",
            )
        )
    if adc_bits is not None:
        entries.append(
            Entry(
                "adc",
                adc(adc_bits, total, mu=mu, ratio=ratio),
                "Laudenbach (9.108): stated Bob-plane, /T applied",
            )
        )
    if rin is not None:
        entries.extend(rin_rows(v_a, rin, bandwidth, total))
    if raman_db is not None:
        # Referred by launch*span, NOT by total: Raman light is born inside the
        # fibre. q.Link splits q.Coexistence the same way.
        entries.append(
            Entry(
                "raman",
                raman(
                    raman_db,
                    raman_width(bandwidth, wavelength),
                    1.0 / bandwidth,
                    launch * span,
                    wavelength=wavelength,
                ),
                "Laudenbach (9.63): Bob-plane, /(T_launch T_span), born in span",
            )
        )

    return Budget(
        entries=tuple(entries),
        T=total,
        launch=launch,
        receive=receive,
        losses=rows,
        # Carried, not charged: T stays the physical chain.
        v_err=v_err,
        phase_form=None if v_err is None else phase_form,
    )
