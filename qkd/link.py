import math
from dataclasses import dataclass, replace

from . import _core, budget, impairments, std
from .budget import Budget
from .impairments import (
    Backflash,
    Backscatter,
    Coexistence,
    DeadTime,
    Dephasing,
    Modulator,
    Polarisation,
    Timing,
)
from .components import (
    Asymptotic,
    BasisAnalyser,
    BasisKeying,
    Certificate,
    CertifiedBound,
    Channel,
    ClickDetector,
    CoherenceMonitor,
    DelayInterferometer,
    DifferentialPhase,
    DiscriminationBound,
    DSP,
    Fiber,
    FiniteSize,
    FlawBound,
    FlawedKeying,
    GaussianModulation,
    Heterodyne,
    Homodyne,
    IndividualAttack,
    IntensityKeying,
    NullingReceiver,
    PhaseBound,
    PhaseShiftKeying,
    PnrDetector,
    PolarisationKeying,
    SplittingAttack,
    ThresholdArray,
    TransmittedLO,
    TwoStateKeying,
)

_BASIS = (BasisKeying, PolarisationKeying)

_COUNTING = (PnrDetector, ThresholdArray)

# The uniform three-basis bias sixstate_finite is stated in.
_THIRD = 1.0 / 3.0

_EXTRA_KEY = {
    Coexistence: "coexist",
    Backscatter: "probe",
    Dephasing: "dephase",
    Polarisation: "pol",
    Modulator: "modulator",
    Timing: "clock",
}

# Impairments a threshold detector feels. q.Modulator enters for `extinction`
# alone -- a QBER, never a xi.
_FLAG_KEY = {
    Backflash: "backflash",
    Backscatter: "probe",
    Coexistence: "coexist",
    DeadTime: "dead",
    Modulator: "modulator",
}

# Dephasing's sentence is reproduced verbatim in docs/guide/impairments.md.
_CLICK_NO = {
    Dephasing: ("it yields a Gaussian excess noise, which threshold detection never sees"),
    Polarisation: (
        "q.Polarisation is the CV fading model and threshold detection has no "
        "quadrature variance to inflate. The click path's polarisation model "
        "is a QBER: q.PolarisationKeying(frame=q.ReferenceFrame(drift=...)), "
        "which errs a matching-basis photon at sin^2(theta). Declaring both "
        "charges one drift twice"
    ),
    Timing: (
        "q.Timing is the CV sampling branch, where a mistimed sample is a "
        "fading efficiency. A threshold detector's arrival-time error is a "
        "first-order QBER term and q.ClickDetector carries it already: jitter "
        "as the Gaussian core FWHM, tail_frac and tail_time as the diffusion "
        "tail, window as the acceptance width. Declaring both models one "
        "clock twice"
    ),
}

# Detector figures _run_clicks and _run_pulses pass ONCE for the whole receiver.
# The calls fix this list: a name comes off here in the same change that makes
# the call per-detector.
_SPLIT_DROPS = (
    "dead_time",
    "afterpulse",
    "jitter",
    "window",
    "tail_frac",
    "tail_time",
)

# Rows inside budget.phase's (V_A + xi): noise already on the light the residual
# rotation turns. Bob-plane rows (adc, rin_lo) stay outside. A name here is still
# summed into the total once.
_BRACKET = ("dac", "imbalance", "raman", "rayleigh")

_PLANES = {
    "pinned": "q.Channel(ref='input'): stated input-referred, no /T",
    "derived": "q.Channel(ref='output'): stated at the far end, /T applied",
    "default": "no channel xi declared: a pure-loss span",
}


_q = std.q

# q.Asymptotic selects the analytic Denys-Brown-Leverrier bound, q.CertifiedBound
# the two-step relative-entropy proof.
_DM_SECURITY = (Asymptotic, CertifiedBound)


def _clamp(est):
    """
    (T, xi) clipped for the rate formulas only; res.est keeps the raw ones,
    which may carry xi_hat < 0 or T_hat > 1.
    """

    # 1e-12 guards a division by an estimated T of zero; not a physical bound.
    return min(1.0, max(1e-12, est.T)), max(0.0, est.xi)


@dataclass(frozen=True)
class Dsp:
    # rad^2, as -2*ln(E[cos(theta - theta_hat)]), not the second moment.
    v_err: float
    # Pilot bin against the adjacent empty bin.
    pilot_snr: float
    # Fitted carrier-frequency offset, Hz.
    cfo: float
    removed: bool
    symbols: int


@dataclass(frozen=True)
class Est:
    """
    Channel parameters from the recovered frames, by the Leverrier 2010
    maximum-likelihood estimators (Eq. (20)).
    """

    T: float
    xi: float
    sigma2: float
    # Confidence-interval ends the rate is claimed at; None under Asymptotic.
    t_min: float | None = None
    xi_max: float | None = None


@dataclass(frozen=True)
class Frames:
    """
    One pass of the symbol pipeline, before any security model is applied.
    """

    # What measure() was asked for; the count used is dsp.symbols.
    symbols: int
    seed: int
    dsp: Dsp | None
    # None where nothing was estimated: pinned v_err, or no DSP chain.
    est: Est | None
    # The pipeline key Link.claim() rebuilds and compares.
    args: tuple


@dataclass(frozen=True)
class Oracle:
    """
    The rate at the T and xi that were set, nothing charged for learning them.
    Under FiniteSize key_rate <= this; under Asymptotic it can land above.
    """

    key_rate: float
    i_ab: float
    chi_be: float
    T: float
    xi: float


@dataclass(frozen=True)
class LinkResult:
    """
    What one run reports; a field the running family does not produce is None.
    """

    # BITS PER SYMBOL on Gaussian modulation and M-PSK, BITS PER EMITTED PULSE
    # on the click families. Clamped at zero; explain()["key_raw"] is unclamped.
    key_rate: float
    # In key_rate's own unit.
    i_ab: float | None = None
    chi_be: float | None = None
    p_click: float | None = None
    qber: float | None = None
    t_min: float | None = None
    xi_max: float | None = None
    delta: float | None = None
    dsp: Dsp | None = None
    est: Est | None = None
    # `Budget`, not `budget.Budget`: the field shadows the module and the
    # annotation evaluates eagerly before 3.14. None on the click families.
    budget: Budget | None = None
    explain: dict | None = None
    leak: float | None = None
    clicks: int | None = None
    sifted: int | None = None
    # A subset of sifted, so clicks = sifted + doubles (the squashing model).
    doubles: int | None = None
    slots: int | None = None
    sift_rate: float | None = None
    visibility: float | None = None
    mu_bob: float | None = None
    # `key_length` is BITS for the whole block, which is what Lim's analysis
    # bounds; `key_rate` carries length / block.
    key_length: float | None = None
    s0: float | None = None
    s1: float | None = None
    phi: float | None = None
    n_key: float | None = None
    # The ONLY place the certified branch's two steps are reachable apart:
    # `bound` is the proof and `upper` proves nothing. key_rate above is
    # `certificate.key` clamped.
    certificate: Certificate | None = None
    _oracle: Oracle | None = None

    @property
    def oracle(self):
        # Where nothing was estimated the reported rate already is the oracle.
        if self._oracle is None:
            return self

        return self._oracle


class Link:
    """
    A point-to-point key distribution link. Runs are CPU and f64 throughout.
    v_a and v_el are in SNU, v_err in rad^2, fibre length in km.
    """

    def __init__(
        self,
        modulation,
        channel,
        bob,
        alice=None,
        dsp=None,
        security=None,
        impairments=(),
        losses=(),
    ):
        # Selects the family; an unconsumed component is refused, not dropped.
        self.modulation = modulation
        # q.Channel (T pinned) or q.Fiber (length, attenuation). xi is carried
        # at the CHANNEL INPUT plane; ref='output' is divided by T on the way.
        self.channel = channel
        self.bob = bob
        self.alice = alice
        # Gaussian modulation only.
        self.dsp = dsp
        self.security = security if security is not None else Asymptotic()
        self.impairments = tuple(impairments)
        # Itemised optics along a q.Fiber span; a pinned q.Channel refuses them.
        self.losses = tuple(losses)
        # Single-slot caches of (key, value), compared by VALUE: every attribute
        # above is reassignable, so a set-once cache would answer for a link
        # that no longer exists.
        self._chain_at = None
        self._flags_at = None
        self._channel_at = None
        self._misalign_at = None

    def _flags(self):
        held = self._flags_at
        if held is not None and held[0] == self.impairments:
            return held[1]

        out = {}
        for item in self.impairments:
            key = _FLAG_KEY.get(type(item))
            if key is None:
                why = _CLICK_NO.get(type(item))
                raise ValueError(
                    f"no click bound consumes {type(item).__name__}: "
                    + (why if why is not None else "it reaches no term here")
                )

            if key in out:
                raise ValueError(f"duplicate {type(item).__name__} impairment")

            out[key] = item

        self._flags_at = (self.impairments, out)

        return out

    def _extra(self, t, v_err=None):
        """
        Excess noise from the impairment descriptors, input-referred, as
        (total, entries, floor, fade). fade is the TRANSMITTANCE half of the
        rows whose noise half is in entries, 1.0 where none was described.
        """
        if not self.impairments:
            return 0.0, (), None, 1.0

        kw = {}
        for item in self.impairments:
            key = _EXTRA_KEY.get(type(item))
            if key is None:
                raise ValueError(f"no Gaussian-modulation term consumes {type(item).__name__}")

            if key in kw:
                raise ValueError(f"duplicate {type(item).__name__} impairment")

            kw[key] = item

        # budget.phase(v_err) and dephasing are one mechanism at two
        # idealisations; the floor returns as a diagnostic. Unwrapped.
        floor = None
        walk = kw.get("dephase")
        if walk is not None and v_err is not None:
            floor = impairments.phase_variance(walk.linewidth, walk.delay)
            if v_err < floor:
                raise ValueError(
                    f"v_err {v_err:.4g} rad^2 sits below the {floor:.4g} rad^2 "
                    "the declared linewidth diffuses over that delay: no "
                    "estimator removes a phase that has already diffused"
                )

            del kw["dephase"]

        ch = self.channel
        length = getattr(ch, "length", None)
        alpha = getattr(ch, "alpha", 0.2)
        rate = None if self.alice is None else self.alice.symbol_rate
        shared = {
            "v_a": self.modulation.v_a,
            "length": length,
            "alpha": alpha,
            "symbol": None if rate is None else 1.0 / rate,
        }
        # Raman and Rayleigh photons are born inside the span, so the receive
        # loss cancels: they refer in by launch*span alone, not the total T.
        born = {}
        for key in ("coexist", "probe"):
            if key in kw:
                born[key] = kw.pop(key)

        launch, span, _, _ = self._chain()
        entries = ()
        if born:
            entries = impairments.assemble_extra(t=launch * span, **shared, **born)

        entries = entries + impairments.assemble_extra(t=t, **shared, **kw)

        return sum(e.xi for e in entries), entries, floor, self._fade()

    def _fade(self):
        """
        The transmittance half of the declared fading rows, as one factor. 1.0
        where neither descriptor is present.
        """
        kw = {}
        for item in self.impairments:
            key = _EXTRA_KEY.get(type(item))
            if key in ("pol", "clock"):
                kw[key] = item

        return impairments.fading_factor(**kw)

    def _dac(self, t):
        """
        The DAC quantisation row q.Alice(iq=...) describes, input-referred, or
        None where no IQ modulator was. Reads the bit depth and V_A; budget.adc
        needs a full-scale ratio no component carries.
        """
        alice = self.alice
        if alice is None or alice.iq is None:
            return None

        return budget.Entry(
            "dac",
            budget.dac(self.modulation.v_a, alice.iq.bits, t),
            "Laudenbach (9.37): stated Bob-plane, /T applied",
        )

    def _rin(self, t):
        """
        The two relative-intensity-noise rows q.Laser(rin=...) and the
        detector's bandwidth describe, input-referred, or () where either is
        absent.
        """
        alice = self.alice
        laser = None if alice is None else alice.laser
        band = self.bob.detector.bandwidth
        if laser is None or laser.rin is None or band is None:
            return ()

        return budget.rin_rows(self.modulation.v_a, laser.rin, band, t)

    def _charged(self, t, v_err=None):
        """
        Every input-referred row a Gaussian run charges beside the channel xi,
        as (total, entries, floor, fade): impairment rows with the DAC row and
        then the two RIN rows prepended. _claim_cv and explain() both read this
        one assembly, so the plan's phase term is charged over the rows the
        run's is.
        """
        extra, entries, floor, fade = self._extra(t, v_err)
        quant = self._dac(t)
        if quant is not None:
            # run_symbols simulates no DAC quantisation, so this lands on the
            # budget and the estimate rather than in _args.
            entries = (quant,) + tuple(entries)
            extra = extra + quant.xi

        noise = self._rin(t)
        if noise:
            # run_symbols simulates no intensity noise either.
            entries = tuple(noise) + tuple(entries)
            extra = extra + sum(e.xi for e in noise)

        return extra, entries, floor, fade

    def _optics(self, info):
        if not self.losses:
            return

        launch, span, receive, rows = self._chain()
        info["T_launch"] = _q(launch, "derived")
        info["T_span"] = _q(span, "derived")
        info["T_receive"] = _q(receive, "derived")
        for row in rows:
            label = "derived" if row.source == "fiber_span" else "pinned"
            info[f"loss_{row.source}"] = _q(row.db, label)

    def _bud(self, entries, v_err=None):
        # entries are already input-referred. v_err is CARRIED, NOT CHARGED -- T
        # stays the physical chain and no xi moves for it -- so that
        # res.budget.inferred can return the phase row's transmittance half.
        launch, span, receive, rows = self._chain()

        return budget.Budget(
            entries=tuple(entries),
            T=launch * span * receive,
            launch=launch,
            receive=receive,
            losses=rows,
            v_err=v_err,
            phase_form=None if v_err is None else "estimator",
        )

    def _bracket(self, entries):
        """
        The rows charged INSIDE budget.phase's (V_A + xi), summed. Alice- and
        channel-plane noise renormalises with V_A; a Bob-plane row (adc,
        rin_lo) is downstream of the rotation and stays outside.
        """

        return sum(e.xi for e in entries if e.source in _BRACKET)

    def _plane(self, t, v_err, fade):
        """
        The transmittance the key rate is claimed at: the physical chain faded
        by each fading row's own <sqrt(eta)>^2 and, where a residual phase was
        charged, attenuated by budget.infer's exp(-v_err). Charging one half of
        either identity alone overstates the rate.
        """
        out = max(1e-12, t * fade)
        if v_err is not None:
            out = budget.infer(out, v_err)

        return max(1e-12, out)

    def _check(self):
        # First of FOUR isinstance ladders over the same seven modulations --
        # _check, run, explain, _attack. A (check, run, explain) table measured
        # -34 lines of 6,108 and was DECLINED: tabling adds six methods. This
        # ladder alone raises, and its sixth arm is (*_BASIS, IntensityKeying)
        # where run and explain split them.
        mod = self.modulation
        if self.losses and isinstance(self.channel, Channel):
            raise ValueError(
                "itemised losses need a q.Fiber span: a pinned q.Channel "
                "already fixes the whole transmittance and the plane its ref= "
                "names"
            )

        self._receiver()
        self._chain()
        if isinstance(mod, GaussianModulation):
            return self._check_cv()

        if isinstance(mod, PhaseShiftKeying):
            return self._check_dm()

        if isinstance(mod, DifferentialPhase):
            return self._check_clicks()

        if isinstance(mod, TwoStateKeying):
            return self._check_two()

        if isinstance(mod, FlawedKeying):
            return self._check_flaw()

        if isinstance(mod, (*_BASIS, IntensityKeying)):
            return self._check_decoy()

        raise NotImplementedError("unsupported modulation component")

    def _receiver(self):
        """
        The two receivers no family reaches, refused before the modulation is
        dispatched on so the reason names the receiver, not the family.
        """
        if isinstance(self.bob.detector, _COUNTING):
            raise NotImplementedError(
                "no rate here reads a photon-number outcome, so a "
                f"{type(self.bob.detector).__name__} reaches none of them: "
                "every bound qkd ships over a counting receiver is written on "
                "a BINARY click, and a number-resolving POVM has nowhere to "
                "enter one. The component builds the measurement itself, "
                "povm() and outcomes(mu); a click link takes q.ClickDetector, "
                "which is q.ThresholdArray at one element"
            )

        if isinstance(self.bob.lo, TransmittedLO):
            raise NotImplementedError(
                "q.Link runs a locally generated oscillator alone. Missing for "
                "a transmitted one is a shot-noise unit a KEY RATE may be "
                "written in: q.TransmittedLO.shot(T, v_el) returns the "
                "calibrated unit and the operating one separately, and nothing "
                "here reads the gap. Real-time measurement is the published "
                "countermeasure and is not a bound -- Kunz-Jacques & Jouguet, "
                "Phys. Rev. A 91, 022307 (2015), arXiv:1406.7554; Huang, "
                "Kunz-Jacques, Jouguet, Weedbrook, Yin, Wang, Chen, Guo & Han, "
                "Phys. Rev. A 89, 032304 (2014), arXiv:1402.6921. Assemble the "
                "budget by hand -- q.TransmittedLO.phase(linewidth) through "
                "qkd.budget.phase, shot(T, v_el).v_el on the detector -- and "
                "carry the total as q.Channel(T=..., xi=..., ref='input')"
            )

    def _check_quad(self):
        # Returns whether any front-end hardware was described.
        det = self.bob.detector
        if self.bob.receiver is not None:
            raise ValueError("a quadrature detector describes its own front end")

        if det.clip is not None:
            raise NotImplementedError(
                "nothing on the quadrature path models clipping: every "
                "quadrature here is UNBOUNDED, so a declared linear half-range "
                f"of {det.clip:.4g} sigma would be dropped. Two routes carry it "
                "and they are different claims. As a NOISE TERM, "
                "qkd.budget.assemble(adc_bits=..., ratio=det.full_scale()) "
                "charges the quantiser over that full-scale range and its total "
                "goes on q.Channel(T=..., xi=..., ref='input'). As an ATTACK, "
                "q.attacks.Saturation(alpha=det.clip, delta=..., gain=...) is "
                "stated in these units and returns a Reading, which carries no "
                "key rate"
            )

        self._check_rin()
        self._check_rows()

        alice = self.alice
        hw = alice is not None and (alice.laser is not None or alice.iq is not None or alice.pilots is not None)

        return hw or self.bob.adc is not None or self.bob.lo is not None

    def _check_rows(self):
        """
        The impairment descriptors, and the three fields inside them that no
        assemble_extra row reads. Each has one reader in qkd.impairments and it
        is hand-called, so a declared value is refused by name rather than
        accepted and dropped.
        """
        for item in self.impairments:
            if type(item) not in _EXTRA_KEY:
                raise ValueError(f"no Gaussian-modulation term consumes {type(item).__name__}")

            if isinstance(item, Coexistence) and item.demux != 1.0:
                raise ValueError(
                    f"a demux of {item.demux:.4g} moves no excess noise: it is "
                    "Kumar Eq. (6)'s eta_D, and a DWDM demultiplexer passes the "
                    "quantum channel and the in-band Raman light at ONE "
                    "transmittance, so it divides back out of Eq. (7)'s "
                    "/(eta_D T). A filter NARROWER than the quantum channel "
                    "would not cancel, and no quadrature rate here models one. "
                    "impairments.raman_photons(demux=...) is the one caller it "
                    "changes. Leave demux at 1.0"
                )

            if isinstance(item, Polarisation) and item.dispersion != 0.0:
                raise ValueError(
                    f"a polarisation dispersion of {item.dispersion:.4g} "
                    "s/sqrt(km) reaches no term of a quadrature rate: "
                    "assemble_extra reads q.Polarisation(drift=...) alone, and "
                    "no verified closed form maps a differential group delay "
                    "onto an excess noise. impairments.dgd(dispersion, length) "
                    "and impairments.security(pol=..., length=...) report the "
                    "mean DGD, neither a key rate. q.ReferenceFrame("
                    "dispersion=..., width=...) is a DIFFERENT field of the "
                    "same name, read by the polarisation-keyed families. Drop "
                    "the dispersion"
                )

            if isinstance(item, Modulator) and item.extinction is not None:
                raise ValueError(
                    f"a modulator extinction ratio of {item.extinction:.4g} "
                    "reaches no term of a quadrature rate: assemble_extra reads "
                    "q.Modulator(ratio=..., angle=...) alone, and no verified "
                    "form maps a linear r onto xi. It is the LINEAR I_on/I_off, "
                    "and q.IntensityKeying(extinction=...) is a different field "
                    "of the same name, stated in dB. Take the descriptor to a "
                    "q.BasisKeying or q.PolarisationKeying link, report it with "
                    "impairments.security(modulator=..., qber=...), or drop the "
                    "extinction"
                )

    def _check_rin(self):
        """
        The two relative-intensity-noise rows read a laser RIN and a detection
        bandwidth TOGETHER, so half a declared pair is refused by name.
        budget.assemble draws the same line.
        """
        alice = self.alice
        laser = None if alice is None else alice.laser
        rin = None if laser is None else laser.rin
        band = self.bob.detector.bandwidth
        if (rin is None) == (band is None):
            return

        if rin is None:
            raise ValueError(
                f"a detection bandwidth of {band:.4g} Hz reaches nothing on "
                "its own: the only terms here that read one are the two "
                "relative-intensity-noise rows budget.rin_sig and "
                "budget.rin_lo, and each needs a laser RIN beside it. Declare "
                "q.Laser(rin=...) in dBc/Hz, or drop the bandwidth"
            )

        raise ValueError(
            f"a laser RIN of {rin:.4g} dBc/Hz reaches nothing on its own: "
            "budget.rin_sig integrates it over a DETECTION BANDWIDTH and "
            "budget.rin_lo reads that bandwidth again, so neither row forms "
            "without one. Declare "
            f"q.{type(self.bob.detector).__name__}(bandwidth=...) in Hz, or "
            "drop the rin"
        )

    def _check_cv(self):
        det = self.bob.detector
        if not isinstance(det, (Homodyne, Heterodyne)):
            raise NotImplementedError("GaussianModulation needs a Homodyne or Heterodyne detector")

        if not isinstance(self.security, (Asymptotic, FiniteSize)):
            raise NotImplementedError("Gaussian modulation takes Asymptotic or FiniteSize security")

        hw = self._check_quad()
        if self.dsp is None:
            if hw:
                raise NotImplementedError(
                    "nothing on this path reads that hardware: the pilot DSP "
                    "chain is the only consumer of a described front end and "
                    "this link has none. Add a q.DSP, which derives xi from the "
                    "laser, the pilots and Bob's local oscillator, or price the "
                    "hardware with qkd.budget.assemble(v_err=..., dac_bits=..., "
                    "adc_bits=...) and pass its total as q.Channel(T=..., "
                    "xi=..., ref='input')"
                )

            return

        if self.bob.adc is not None:
            raise NotImplementedError(
                "nothing on this path reads a q.ADC: the pilot DSP chain runs "
                "in floating point end to end and models no quantiser. Price "
                "it as a noise term with qkd.budget.assemble(adc_bits=...) and "
                "carry that on q.Channel instead"
            )

        self._check_dsp()

    def _check_dm(self):
        """
        The trusted refusal below fires on the ANALYTIC branch alone;
        q.CertifiedBound reaches _core.dm_trusted with the same detector.
        """
        det = self.bob.detector
        if not isinstance(det, Heterodyne):
            raise NotImplementedError(
                "the discrete-modulation bound is written for heterodyne "
                "detection; no homodyne form of it is published"
            )

        if det.trusted and self._certified() is None:
            raise NotImplementedError(
                "the analytic discrete-modulation bound has no trusted-detector "
                "form: Denys, Brown & Leverrier fix the receiver theirs is "
                "written for, so eta and v_el fold in untrusted only, as "
                "T -> eta*T and xi -> xi + 2*v_el/(eta*T). The trusted "
                "derivation is Lin & Lutkenhaus, PR Applied 14, 064030 (2020), "
                "which folds the noise into Bob's POVM rather than into the "
                "state, and it ships: pass q.CertifiedBound in place of "
                "q.Asymptotic, or q.Heterodyne(..., trusted=False) to stay on "
                "this one"
            )

        if not isinstance(self.security, _DM_SECURITY):
            # dm_finite always raises, one message per asymptotic bound, so
            # nothing falls through. q.Asymptotic's is the analytic one.
            _core.dm_finite(getattr(self.security, "n", 1.0), False)

        if self.dsp is not None:
            raise NotImplementedError("the pilot DSP chain runs on the Gaussian-modulation path")

        hw = self._check_quad()
        if hw:
            raise NotImplementedError(
                "nothing on the discrete-modulation path reads that hardware: "
                "the pilot DSP chain is the only consumer of a described front "
                "end and it is Gaussian-modulation only. Price them with "
                "qkd.budget.assemble(v_err=..., dac_bits=..., adc_bits=...) and "
                "pass its total as q.Channel(T=..., xi=..., ref='input')"
            )

    def _certified(self):
        """
        The security component selecting the certified bound, or None where the
        analytic Denys-Brown-Leverrier one runs.
        """
        sec = self.security

        return sec if isinstance(sec, CertifiedBound) else None

    def _check_front(self):
        if self.dsp is not None:
            raise NotImplementedError("the DSP chain is Gaussian-modulation only")

        if not isinstance(self.bob.detector, ClickDetector):
            raise NotImplementedError(f"{type(self.modulation).__name__} needs a ClickDetector")

        if self.bob.adc is not None or self.bob.lo is not None:
            raise ValueError("a click receiver has no ADC and no local oscillator")

        self._check_flags()

    def _check_flags(self):
        """
        The fields inside a q.Coexistence and a q.Modulator that no click bound
        reads. _flags() has already refused the descriptors that reach nothing.
        """
        coexist = self._flags().get("coexist")
        if coexist is not None and coexist.demux != 1.0:
            raise ValueError(
                f"a demux of {coexist.demux:.4g} reaches no term of a click "
                "rate, and for a different reason than the quadrature route "
                "gives: it is a loss on Bob's receive chain rather than a share "
                "of the background alone. Declare it as q.Loss on the receive "
                "side, where Link._chain() attenuates both, and leave demux at "
                "1.0"
            )

        item = self._flags().get("modulator")
        if item is None:
            return

        if item.ratio != 1.0 or item.angle != 0.0:
            raise ValueError(
                f"a modulator ratio of {item.ratio:.4g} and angle of "
                f"{item.angle:.4g} reach no term of a click rate: they are an "
                "IQ modulator's quadrature imbalance, a Gaussian excess noise "
                "that threshold detection never sees. A click family reads "
                "`extinction` alone, the LINEAR I_on/I_off. Give "
                "q.Modulator(extinction=...) with ratio at 1.0 and angle at "
                "0.0, or take the imbalance to a quadrature link"
            )

        if item.extinction is None:
            raise ValueError(
                "a q.Modulator on a click family is read for its `extinction` "
                "alone and none was declared: ratio and angle are quadrature "
                "terms a threshold detector cannot read. Give "
                "q.Modulator(extinction=r) as the LINEAR I_on/I_off, or drop "
                "the descriptor"
            )

        mod = self.modulation
        if isinstance(mod, IntensityKeying):
            raise ValueError(
                f"an extinction ratio of {item.extinction:.4g} is declared "
                "twice on this link: q.IntensityKeying(extinction=...) is a "
                "different field of the same name, stated in dB, and the COW "
                "rate already charges it. Drop the descriptor and state the "
                "ratio in dB on the modulation"
            )

        if not isinstance(mod, _BASIS):
            raise ValueError(
                f"an extinction ratio of {item.extinction:.4g} reaches no term "
                f"of the {type(mod).__name__} family: the 4/(r + 3) admixture "
                "is emitted by a FOUR-path transmitter, one intensity "
                "modulator per BB84 state, and dilutes a four-state "
                "misalignment. q.BasisKeying and q.PolarisationKeying are the "
                "two families that send such a set"
            )

        if self._bases() != 2:
            raise ValueError(
                f"an extinction ratio of {item.extinction:.4g} is refused at "
                "bases = 3: the 4 of 4/(r + 3) COUNTS the transmitter's four "
                "paths and six-state sends six. No six-path form is published, "
                "and the four-state one would understate the leak -- the "
                "insecure direction. Drop the extinction, or run bases = 2"
            )

    def _extinct(self):
        """
        The linear transmitter extinction ratio a click family charges as a
        QBER, or None where no q.Modulator was declared. _check_flags has
        already refused every descriptor this cannot answer for.
        """
        item = self._flags().get("modulator")

        return None if item is None else item.extinction

    def _split_fields(self):
        """
        What Bob's second threshold detector may not differ in on the two
        SAMPLED paths. _check_split covers the closed forms, where the whole
        partner is refused.
        """
        one, two = self.bob.detector.both()
        if two is one:
            return

        named = [n for n in _SPLIT_DROPS if getattr(one, n) != getattr(two, n)]
        if not named:
            return

        raise NotImplementedError(
            "a partner splits eta and dark and nothing else: _core.run_clicks "
            "and _core.run_basis read dead_time, afterpulse, jitter, window, "
            "tail_frac and tail_time ONCE for the receiver as a whole, so a "
            f"partner differing in {', '.join(named)} would be described and "
            "then dropped. Give both detectors the same value there"
        )

    def _check_split(self):
        """
        A partner reaching a closed form, which is every branch that does not
        sample the pulse train. Refused rather than silently halved.
        """
        if self.bob.detector.partner is None:
            return

        raise NotImplementedError(
            "no closed form here reads a second detector: decoy_gain, "
            "sarg_gain, sarg_yield, dps_rate and the afterpulsing gain are "
            "each written on ONE eta and ONE dark. The split is hardware in the "
            "two SAMPLED engines alone: _core.run_clicks, reached by "
            "q.DifferentialPhase with q.IndividualAttack(qber=None) and a "
            "q.DelayInterferometer, and _core.run_basis, reached by "
            "q.BasisKeying once a q.Alice is described. A mismatch in a closed "
            "form is a DIFFERENT BOUND, scaling by 2*min/(hi + lo): "
            "qkd.attacks.mismatch_rate(*det.pair(), e_bit, e_phase), which is "
            "not a q.Link key rate"
        )

    def _check_flux(self):
        """
        A background flux reaching a closed form that refuses a q.Alice by
        name. Left to run time, _flux would ask for the clock the family has
        already refused.
        """
        flags = self._flags()
        named = [type(flags[key]).__name__ for key in ("coexist", "probe") if key in flags]
        if not named:
            return

        raise ValueError(
            f"a q.{' and a q.'.join(named)} reaches no term of the "
            f"{type(self.modulation).__name__} closed form: a background flux "
            "is counted in a GATE, and this form refuses by name the "
            "q.Alice(symbol_rate=...) that carries the clock. Fold the "
            "background click probability into q.ClickDetector(dark=...) "
            "yourself, at the gate you have in mind, or take the flux to an "
            "engine that samples the pulse train -- _core.run_clicks, reached "
            "by q.DifferentialPhase with q.IndividualAttack(qber=None) and a "
            "q.DelayInterferometer, or _core.run_basis, reached by "
            "q.BasisKeying once a q.Alice is described"
        )

    def _check_gate(self, sampled):
        """
        The acceptance window, which gates an arrival-time response and is
        nothing on its own. ``sampled`` selects the one branch that reads a
        window; q.PairLink refuses both.
        """
        det = self.bob.detector
        if det.window == 0.0:
            return

        if not sampled:
            raise ValueError(
                f"an acceptance window of {det.window:.4g} s reaches no term of "
                "this family: the one engine that convolves an arrival-time "
                "response along the slot train is _core.run_clicks, reached by "
                "q.DifferentialPhase with q.IndividualAttack(qber=None) and a "
                "q.DelayInterferometer. Fold the window loss into eta yourself "
                "if you want it here, or drop the window"
            )

        if det.timed():
            return

        raise ValueError(
            f"an acceptance window of {det.window:.4g} s gates nothing on this "
            "detector: jitter and tail_frac are both 0, so _core.run_clicks "
            "builds no timing kernel. Declare the response beside it -- "
            "q.ClickDetector(jitter=...) as a Gaussian FWHM in s, or "
            "(tail_frac=..., tail_time=...) for the diffusion tail -- or drop "
            "the window"
        )

    def _check_clicks(self):
        # A pinned qber is a closed form over mu, T, eta and dark; unpinned
        # simulates the pulse train and consumes the rest.
        self._check_front()
        det = self.bob.detector
        sec = self.security
        alice = self.alice
        if self._dead() is not None:
            raise ValueError(
                "the phase-keyed path takes dead time and afterpulsing on "
                "q.ClickDetector, where the simulation consumes them slot by "
                "slot; a DeadTime descriptor would model the same memory twice"
            )

        if not isinstance(sec, IndividualAttack):
            raise NotImplementedError(
                "the phase-keyed family takes IndividualAttack security. The "
                "shipped Waks-Takesue-Yamamoto bound has no finite-key form: "
                "it bounds a collision probability, not a smooth min-entropy, "
                "so there is no correction term to attach to it. The "
                "finite-key DPS analysis that does exist -- Mizutani, Takeuchi "
                "& Tamaki, Phys. Rev. Research 5, 023132 (2023), THREE authors "
                "-- is against general attacks by a complementarity argument "
                "and assumes photon-number-resolving detectors, which this "
                "click layer does not model"
            )

        if sec.qber is not None:
            if self.bob.receiver is not None or alice is not None:
                raise ValueError(
                    "a pinned qber is a closed form: it cannot consume an " "interferometer or a source description"
                )

            if det.dead_time != 0.0 or det.afterpulse != 0.0:
                raise ValueError(
                    "dead time and afterpulsing only reach the simulated click " "path: leave qber unpinned to use them"
                )

            if det.timed():
                raise ValueError(
                    "detector timing jitter only reaches the simulated click "
                    "path, where the arrival-time response is convolved along "
                    "the slot train: leave qber unpinned to use it"
                )

            self._check_gate(False)
            self._check_split()
            self._check_flux()

            return

        if not isinstance(self.bob.receiver, DelayInterferometer):
            raise ValueError("deriving the qber needs Bob with a DelayInterferometer")

        if alice is None or alice.laser is None:
            raise ValueError("deriving the qber needs Alice with a laser")

        if alice.iq is not None or alice.pilots is not None:
            raise ValueError("a phase-keyed source has no IQ modulator and no pilots")

        self._check_gate(True)
        self._split_fields()

    def _check_two(self):
        self._check_front()
        det = self.bob.detector
        mod = self.modulation
        sec = self.security
        if not isinstance(self.bob.receiver, NullingReceiver):
            raise ValueError(
                "TwoStateKeying needs Bob with a NullingReceiver: the two "
                "states are told apart by displacing one of them to the vacuum, "
                "not by a basis analyser or a delay line"
            )

        if not isinstance(sec, DiscriminationBound):
            raise NotImplementedError("TwoStateKeying takes DiscriminationBound security")

        if det.dead_time != 0.0 or det.afterpulse != 0.0 or det.timed():
            raise ValueError(
                "dead time, afterpulsing and detector timing are memory the "
                "simulated click path carries, and this family is a closed form"
            )

        self._check_gate(False)
        if self._dead() is not None:
            raise ValueError(
                "a DeadTime descriptor reaches no term of this closed form; "
                "the two-state gains carry no detector memory"
            )

        self._check_split()
        self._check_flux()

        if self.alice is not None:
            raise NotImplementedError(
                "the two-state family is a closed form over Bob's receiver: it "
                "emits no pulse train, so a q.Alice reaches nothing in it"
            )

        if getattr(mod, "source", "coherent") == "single":
            self._check_plain()

        if mod.reference and sec.e_phase is None:
            raise NotImplementedError(
                "the strong-reference branch takes a SUPPLIED e_phase: Koashi "
                "bounds the phase error over a virtual entanglement picture "
                "rather than the hardware, and it collapses to no closed form "
                "in (mu, T, eta, visibility, dark). Give "
                "q.DiscriminationBound(e_phase=...), or drop reference= and let "
                "the plain branch derive it from the loss and the overlap"
            )

        if sec.block is not None:
            # b92_finite reads the block size and refuses in its own words,
            # with a message that depends on the branch.
            _core.b92_finite(sec.block.n, mod.reference)

    def _check_plain(self):
        """
        What the single-photon branch cannot read. It takes Tamaki and
        Lutkenhaus's channel whole, so every receiver number the coherent branch
        consumes reaches nothing here.
        """
        det = self.bob.detector
        if self.security.e_phase is not None:
            raise ValueError(
                "the single-photon branch DERIVES its phase error from the "
                "overlap, the loss and the depolarising rate, and applies no "
                "floor under it: q.DiscriminationBound(e_phase=...) is read on "
                "the coherent branch alone. Leave e_phase unset"
            )

        if det.dark != 0.0 or self.bob.receiver.visibility != 1.0:
            raise ValueError(
                "the single-photon branch reads its observables off the channel "
                "rather than off Bob's optics, so a dark count rate and a "
                "receiver fringe visibility reach no term of it. Give "
                "q.ClickDetector(eta=..., dark=0.0) and "
                "q.NullingReceiver(visibility=1.0), or drop source='single' for "
                "the coherent branch that consumes both"
            )

    def _check_flaw(self):
        """
        The flawed qubit source. Every observable is a closed form over
        Pereira's channel model: the two angles and the two detector numbers
        are the whole of the input.
        """
        self._check_front()
        det = self.bob.detector
        sec = self.security
        if not isinstance(sec, FlawBound):
            raise NotImplementedError(
                "the flawed-source family takes q.FlawBound security, which "
                "carries WHICH of the two analyses the run is priced under: "
                "src/flaws.rs ships flaw_tolerant, reading the phase error "
                "EXACTLY by inverting Bob's rejected-basis rates against "
                "Alice's known Bloch vectors, and flaw_standard, bounding the "
                "same quantity through a quantum coin Eve may amplify by the "
                "whole channel loss. No other security component here can name "
                "which number came back"
            )

        if self.bob.receiver is not None:
            raise ValueError(
                "Bob's analyser angle IS the flaw's tilt on this family and "
                "q.SourceFlaw(tilt=...) carries it, so a "
                f"{type(self.bob.receiver).__name__} would describe one frame "
                "twice. They are not the same quantity: q.BasisAnalyser."
                "misalign is an UNSIGNED probability in [0, 1/2], and both "
                "papers are written at the SIGNED tilt = -delta -- "
                "q.SourceFlaw.opposed(delta) is that configuration by name, and "
                "at tilt = 0 the phase error stops depending on delta. Give "
                "q.Bob(detector=q.ClickDetector(...)) with no receiver"
            )

        if det.partner is not None:
            raise NotImplementedError(
                "both analyses here need Bob's inconclusive operator to be "
                "BASIS INDEPENDENT, and a second detector with its own "
                "efficiency violates that condition, making both inapplicable "
                "rather than looser. The composition exists -- loss-tolerant "
                "QKD with detection-efficiency mismatch, arXiv:2412.09684 "
                "(2024) -- and needs its own proof and its own virtual states, "
                "neither implemented, so a phase error from here must not be "
                "handed to qkd.attacks.mismatch_rate as its e_phase"
            )

        if det.dead_time != 0.0 or det.afterpulse != 0.0 or det.timed():
            raise ValueError(
                "dead time, afterpulsing and detector timing are memory a "
                "simulated click train carries, and this family is a closed "
                "form over Pereira arXiv:1902.02126 Appendix E: its readout is "
                "one gate's click law in the two angles and a per-gate dark "
                "probability, with no slot train for a memory"
            )

        self._check_gate(False)

        if self.alice is not None:
            raise NotImplementedError(
                "the flawed-source family is a closed form over a SINGLE "
                "PHOTON: it emits no pulse train, so a q.Alice reaches nothing "
                "in it. The source is the flaw itself, q.SourceFlaw(delta=...)"
            )

        if self.impairments:
            raise ValueError(
                "no term of this family reads an impairment descriptor: every "
                "observable is a closed form over the two angles, the overall "
                "transmittance and the per-gate dark probability. A q.Modulator "
                "extinction is a FOUR-path BB84 transmitter's identity fraction "
                "and this source is a qubit"
            )

        if sec.block is not None:
            raise NotImplementedError(
                "flaw_finite is the length that ships, and it is NOT the "
                "analysis this bound selects: src/flaws.rs::flaw_finite is "
                "Mizutani arXiv:1504.08151 Eq. (43) on a SINGLE-PHOTON source "
                "with m0 = 0, reading the FORWARD channel model rather than "
                "either loss-tolerant analysis. The two analyses q.FlawBound "
                "selects are asymptotic, and neither becomes a length by "
                "attaching a correction term: flaw_phase is an EXACT inversion "
                "of three observed rates, so its finite-size form is that same "
                "inversion over a confidence REGION, plus the random-sampling "
                "correction between the virtual X-basis error rate and the "
                "phase error rate. Neither is implemented. Leave block unset, "
                "or take a finite-key length from q.SplittingAttack("
                "block=q.KeyBlock(...)) on BB84-WCP or q.TestBasisBound("
                "block=q.RelayBlock(...)) on the MDI-BB84 q.Swap"
            )

    def _run_flaw(self):
        # bits per EMITTED SINGLE PHOTON, neither a symbol nor a pulse: Pereira
        # Eq. (4) is written per single-photon emission, q_sift outside it.
        mod = self.modulation
        sec = self.security
        det = self.bob.detector
        # Alice's output to a detection, receiver efficiency included: the one
        # transmittance flaw_channel reads.
        eta = self._channel()[0] * det.eta
        read = mod.flaw.tolerant if sec.analysis == "tolerant" else mod.flaw.standard
        out = read(eta, det.dark, sec.f, mod.sift)
        info = self._explain_flaw(eta, out)

        return LinkResult(
            key_rate=max(0.0, out[0]),
            p_click=out[1],
            qber=out[2],
            explain=info,
        )

    def _explain_flaw(self, eta, out):
        # The fourth slot of `out` is a phase error rate under one analysis and
        # a quantum coin imbalance under the other, so each branch below emits
        # its own row rather than sharing a name.
        mod = self.modulation
        sec = self.security
        det = self.bob.detector
        flaw = mod.flaw
        t, _, t_label, _ = self._channel()
        rate, y1, e_bit, last = out
        info = {
            **self._head("flawed-source keying"),
            "analysis": _q(sec.analysis, "pinned"),
            "source": _q("single photon", "structural"),
            "delta": _q(flaw.delta, "pinned"),
            "tilt": _q(flaw.tilt, "pinned"),
            "T": _q(t, t_label),
            "eta": _q(det.eta, "pinned"),
            "dark": _q(det.dark, "pinned"),
            "eta_ab": _q(eta, "derived"),
            "sift": _q(mod.sift, "pinned"),
            "f": _q(sec.f, "pinned"),
            "y1": _q(y1, "derived"),
            "qber": _q(e_bit, "derived"),
        }
        if sec.analysis == "tolerant":
            # How well conditioned this branch's inversion is; nothing reads it.
            info["triangle"] = _q(flaw.triangle(), "derived")
            info["e_phase"] = _q(last, "derived")
        else:
            info["fidelity"] = _q(flaw.fidelity(), "derived")
            info["coin"] = _q(last, "derived")
            # Named apart from the other branch's e_phase: that one is the
            # phase error rate, this one an upper bound on it.
            info["e_phase_bound"] = _q(_core.flaw_gllp(e_bit, last), "derived")

        # flaw_tolerant and flaw_standard clamp in Rust, so this row and
        # key_rate agree rather than bracketing anything.
        info["key_raw"] = _q(rate, "derived")

        return info

    def _check_decoy(self):
        # Closed forms over the Poisson photon-number sum; an Alice on a
        # basis-keyed link selects the sampled path instead.
        self._check_front()
        det = self.bob.detector
        mod = self.modulation
        keyed = isinstance(mod, _BASIS)
        dead = self._dead()
        if dead is not None and dead.dead != 0.0:
            raise ValueError(
                "dead-time saturation needs one click stream at one rate, and "
                "these families interleave intensities through the same "
                "detector; give DeadTime(dead=0.0, afterpulse=...) here and "
                "take the saturation to the simulated path"
            )

        if dead is not None and dead.paralysable:
            raise ValueError(
                "paralysable=True qualifies what happens to an arrival inside "
                "the dead time, and this family takes a q.DeadTime only at "
                "dead = 0, where both branches return the same rate. They part "
                "company in impairments.saturate(rate, dead, paralysable) and "
                "impairments.security(dead=..., rate=...), neither of them a "
                "key rate. Leave paralysable False"
            )

        if det.dead_time != 0.0 or det.afterpulse != 0.0:
            raise ValueError(
                "dead time and afterpulsing are detector memory, which only the "
                "simulated click path carries; this family is a closed form"
            )

        if det.timed():
            raise ValueError(
                "detector timing jitter reaches the slot train the simulated "
                "click path convolves, and this family is a closed form; fold "
                "the window loss into eta yourself if you want it here"
            )

        self._check_gate(False)
        want = BasisAnalyser if keyed else CoherenceMonitor
        if not isinstance(self.bob.receiver, want):
            raise ValueError(f"{type(mod).__name__} needs Bob with a {want.__name__}")

        need = SplittingAttack if keyed else PhaseBound
        if not isinstance(self.security, need):
            raise NotImplementedError(f"{type(mod).__name__} takes {need.__name__} security")

        if keyed:
            self._check_bias()

            self._check_variant(dead)
        else:
            self._check_phase()
            self._check_flux()

        if keyed and self.security.block is not None:
            self._check_block()

        if self.alice is None:
            self._check_split()

            return

        if not keyed:
            raise NotImplementedError(
                "the intensity-keyed family is a closed form over its two "
                "lines' gains; only the basis-keyed one emits pulses"
            )

        alice = self.alice
        if alice.laser is not None or alice.iq is not None or alice.pilots is not None:
            raise ValueError(
                "a phase-randomised weak-coherent source is described by its "
                "intensity set: the laser's linewidth, an IQ modulator and "
                "pilot tones all reach zero terms of this bound"
            )

        if dead is not None:
            raise ValueError(
                "the sampled path interleaves three intensities through one "
                "detector, so it carries no detector memory: drop the DeadTime "
                "descriptor or leave Alice off and take the closed form"
            )

        self._split_fields()

    def _bases(self):
        # 2 everywhere but q.BasisKeying(bases=3).

        return getattr(self.modulation, "bases", 2)

    def _announced(self):
        # What Alice announces after the quantum phase, 'basis' or 'pair'.

        return getattr(self.modulation, "announce", "basis")

    def _attack(self):
        """
        The class of attack the engine behind this link's number is proved
        against, quoted from that engine's module header. A family whose header
        states no class says so here; the row is never guessed.
        """
        mod = self.modulation
        if isinstance(mod, GaussianModulation):
            # src/keyrate.rs:46 and :203. cv_finite takes the same worst-case
            # point through cv_core, so the finite-size branch is the same class.
            return "Gaussian collective"

        if isinstance(mod, PhaseShiftKeying):
            if self._certified() is not None:
                # src/dmcs.rs:550, which names the cutoff as part of the claim.
                return "collective, asymptotic, under a photon-number cutoff"

            # src/dmcs.rs:9, which names the receiver as part of the class.
            return "collective, asymptotic, ideal receiver"

        if isinstance(mod, DifferentialPhase):
            # src/keyrate.rs:317, the Waks-Takesue-Yamamoto bound.
            return "general individual"

        if isinstance(mod, TwoStateKeying):
            if mod.reference:
                # src/b92.rs: Koashi bounds the phase error over quantities of
                # a virtual picture, so here it is supplied, as COW's is.
                return "undefined: e_phase is supplied, not derived"

            if getattr(mod, "source", "coherent") == "single":
                # src/b92.rs:15, on the branch its own header calls seam-free.
                return "unconditional (Tamaki & Lutkenhaus 2004)"

            # src/b92.rs:15 for the proof, and its module header's last
            # paragraph for the seam.
            return (
                "unconditional for a single-photon source (Tamaki & "
                "Lutkenhaus 2004), read here off coherent hardware"
            )

        if isinstance(mod, FlawedKeying):
            # src/flaws.rs names a source, a regime and a tilt and no attack
            # class, and neither rate function names one either.
            return "unstated: neither flaw_tolerant nor flaw_standard names one"

        if isinstance(mod, _BASIS):
            if self._announced() == "pair":
                # src/sarg.rs:24, in its source's word.
                return "unconditional, per Fung, Tamaki & Lo (2006) Eq. (39)"

            if self._bases() == 3:
                # src/sixstate.rs:157.
                return "collective, one-way post-processing"

            # src/discrete.rs:17 and :90 state a source and a unit and no
            # attack class, and nothing downstream of them states one either.
            return "unstated: neither bb84_rate nor bb84_length names one"

        # COW (q.IntensityKeying) is the only modulation reaching here, _check
        # having raised on any other. A family added to the other three ladders
        # and forgotten here reports COW's string rather than raising, so add it
        # to test/protocols.py's reports() for AttackClass to walk.
        # src/discrete.rs:155: e_phase is an input upper bound, so no class.
        return "undefined: e_phase is supplied, not derived"

    def _check_bias(self):
        """
        The key-basis probability, a q.Swap field carried on a q.Link
        component. Nothing here reads one.
        """
        bias = getattr(self.modulation, "bias", None)
        if bias is None:
            return

        raise ValueError(
            f"a key-basis probability of {bias:.4g} reaches no branch of "
            "q.Link: both the asymptotic rates and the finite-key length read "
            "the sifting factor alone, and inverting sift = q^2 + (1 - q)^2 "
            "recovers |q - 1/2| and not its sign. A bias is read on the OTHER "
            "topology: q.Swap with q.TestBasisBound(block=q.RelayBlock(...)) "
            "takes one per sender. State the split here as "
            "q.BasisKeying(sift=...), or drop the bias"
        )

    def _check_variant(self, dead):
        """
        What the six-state and pair-announced branches cannot honour. Both run
        BB84's decoy layer unchanged and part company at the
        privacy-amplification factor, so what is refused here reads the basis
        structure rather than the yields.
        """
        if self._bases() == 2 and self._announced() == "basis":
            return

        name = "six-state" if self._bases() == 3 else "the pair announcement"
        if self.alice is not None:
            raise NotImplementedError(
                "the sampled pulse train sifts two bases and announces one, so "
                f"it cannot emit {name}: leave Alice off and take the closed form"
            )

        if self.security.block is not None:
            self._refuse_block()

        if dead is not None:
            raise ValueError(
                "the afterpulsing gain form is BB84's yield law, with the "
                f"sifting outside the yield, and {name} does not share it"
            )

    def _refuse_block(self):
        """
        The finite-key call each variant's engine refuses, made here so the
        refusal reaches the caller in the engine's words. Neither branch
        returns.
        """
        block = self.security.block
        if self._announced() == "pair":
            _core.sarg_finite(block.n)

        # sixstate_finite refuses source="decoy", the only source this family
        # has, so nothing below the bias is reached and the misalignment stands
        # in for the two error rates rather than a number being invented.
        miss = self._misalign()

        _core.sixstate_finite(
            block.n,
            _THIRD,
            miss,
            miss,
            self.security.f,
            block.eps_sec,
            block.eps_cor,
            "decoy",
        )

    def _check_block(self):
        """
        The finite-key path needs a sifting factor a symmetric pair of
        independent basis choices can produce. Only BB84 reaches here;
        _check_variant refuses the other two first.
        """
        mod = self.modulation
        if mod.sift < 0.5:
            raise ValueError(
                f"sift {mod.sift:.4g} is below 1/2, which no pair of "
                "independent basis choices reaches: sift = q^2 + (1 - q)^2 has "
                "its minimum 1/2 at q = 1/2. The finite-key length reads the "
                "key and test bases separately and cannot invert this one"
            )

        if mod.sift >= 1.0:
            raise ValueError(
                "sift = 1 leaves no test basis, so the phase error is never "
                "sampled and no length can be certified; bias the basis choice "
                "short of 1"
            )

    def _check_phase(self):
        # A plausibility floor, not a proof: the multiphoton fraction
        # 1 - exp(-mu)(1 + mu) is conceded, so a bound below it is refused.
        mu = self.modulation.mu
        floor = 1.0 - math.exp(-mu) * (1.0 + mu)
        if self.security.e_phase < floor:
            raise ValueError(
                f"e_phase {self.security.e_phase:.4g} is below the {floor:.4g} "
                f"multiphoton fraction of a mu = {mu:.4g} source, which is "
                "conceded to Eve before the channel is considered"
            )

        if self.security.block is not None:
            # cow_finite refuses in its own words, naming the four-sequence
            # variant that q.security reaches.
            _core.cow_finite(self.security.block.n)

    def _check_dsp(self):
        if not isinstance(self.dsp, DSP):
            raise NotImplementedError("dsp must be a q.DSP component")

        if not isinstance(self.bob.detector, Heterodyne):
            raise NotImplementedError("the pilot DSP chain simulates heterodyne detection")

        if self.dsp.phase.v_err is not None:
            # Pinned: the hardware the pipeline would have read is accepted and
            # ignored rather than refused, so the same link runs with the
            # pipeline on and off. q.Alice's symbol_rate and iq still reach
            # _extra and _dac.
            return

        alice = self.alice
        if alice is None or alice.laser is None or alice.pilots is None:
            raise ValueError("deriving v_err needs Alice with a laser and pilots")

        if self.bob.lo is None:
            raise ValueError("deriving v_err needs Bob with a local oscillator")

        if alice.laser.carrier is not None or self.bob.lo.carrier is not None:
            # beat()'s own words reach the caller: it names which end lacks one.
            self.bob.lo.beat(alice.laser)

    def _base(self):
        # The span's transmittance alone, before any itemised optics.
        ch = self.channel
        if isinstance(ch, Channel):
            return ch.T

        if isinstance(ch, Fiber):
            return ch.transmittance

        raise ValueError("channel must be a Channel or Fiber")

    def _chain(self):
        # (launch, span, receive, rows).
        key = (self.channel, self.losses)
        held = self._chain_at
        if held is not None and held[0] == key:
            return held[1]

        out = budget.chain(self._base(), self.losses)
        self._chain_at = (key, out)

        return out

    def _channel(self):
        """
        (T, xi, T label, xi label). T is the WHOLE PATH, Alice's output to
        Bob's detector, itemised optics included; xi is at the input plane.
        """
        key = (self.channel, self.losses)
        held = self._channel_at
        if held is not None and held[0] == key:
            return held[1]

        ch = self.channel
        launch, span, receive, _ = self._chain()
        t = launch * span * receive
        if not isinstance(ch, Channel):
            t_label = "pinned" if ch.T is not None else "derived"
            out = (t, 0.0, "derived" if self.losses else t_label, "default")
        elif ch.ref == "output":
            # xi_in = xi_out / T, and naming the far plane is itself a
            # derivation even where the noise there is zero.
            out = (t, std.input_xi(ch), "pinned", "derived")
        else:
            out = (t, ch.xi, "pinned", "pinned" if ch.xi != 0.0 else "default")

        self._channel_at = (key, out)

        return out

    def _pclick(self, t):
        """
        The probability a phase-keyed detection slot registered a click.
        Memory-free, and independent of visibility: the two ports sum to mu*T.
        """
        det = self.bob.detector
        mu = self.modulation.mu
        one, two = self._floors()

        # Two detectors at the delay-line ports: no click is both dark.
        return 1.0 - (1.0 - one) * (1.0 - two) * math.exp(-mu * t * det.eta)

    def _args(self, t, xi, symbols, seed):
        # BOTH the run_symbols argument tuple and the memo key claim() compares:
        # the "sampled" tail IS that arg list in order, so anything reaching the
        # engine must be added here. Impairments are excluded on purpose, so one
        # measurement serves any set of them.
        if self.dsp is None:
            return ("closed", int(symbols), int(seed))

        pin = self.dsp.phase.v_err
        if pin is not None:
            return ("pinned", int(symbols), int(seed), pin)

        det = self.bob.detector
        alice = self.alice
        lo = self.bob.lo
        tone = alice.pilots

        return (
            "sampled",
            int(symbols),
            int(seed),
            self.modulation.v_a,
            t,
            xi,
            det.eta,
            det.v_el,
            alice.laser.linewidth,
            lo.linewidth,
            # Carrier-frequency offset, Hz: SIGNED and unclamped, so run_symbols
            # still refuses 2*|cfo| >= symbol_rate rather than aliasing it. Zero
            # where neither end named a carrier.
            0.0 if lo.carrier is None else lo.beat(alice.laser),
            alice.symbol_rate,
            tone.power_db,
            tone.freq / alice.symbol_rate,
            int(self.dsp.block),
            False,
        )

    def _sim(self, args):
        # (dsp, est) for one resolved argument tuple. Both None without a DSP
        # chain; a pinned v_err short-circuits the pipeline, leaving no est.
        kind = args[0]
        if kind == "closed":
            return None, None

        if kind == "pinned":
            return Dsp(args[3], float("nan"), 0.0, False, 0), None

        out = _core.run_symbols(*args[1:])
        chain = Dsp(out.v_err, out.pilot_snr, out.cfo, True, int(out.n_used))

        return chain, Est(out.t_chan, out.xi_hat, out.sigma2_hat)

    def _seam(self):
        self._check()
        if not isinstance(self.modulation, GaussianModulation):
            raise NotImplementedError(
                f"{type(self.modulation).__name__} runs its pipeline inside "
                "run(): only the Gaussian-modulation path separates a "
                "measurement from the security model claimed on it"
            )

    def measure(self, symbols=1_000_000, seed=0):
        # The pipeline half alone, so one measurement serves any security model.
        # Gaussian modulation only; run() is claim(measure()).
        self._seam()

        return self._measure_cv(symbols, seed)

    def claim(self, frames):
        """
        Rebuilds the pipeline key and raises ValueError if it does not match
        the frames; only the security model and the impairments may differ.
        """
        self._seam()
        t, xi, _, _ = self._channel()
        want = self._args(t, xi, frames.symbols, frames.seed)
        if want != frames.args:
            raise ValueError(
                "these frames came out of a different pipeline: the measurement "
                "is only valid for the modulation, channel, losses, Alice, Bob "
                "and DSP chain that produced it"
            )

        return self._claim_cv(frames)

    def run(self, symbols=1_000_000, seed=0):
        """
        Simulate this link. Validated here, not at __init__. symbols and seed
        reach only the sampling paths: underived v_err, Alice, underived qber.
        """
        mod = self.modulation
        self._check()
        if isinstance(mod, GaussianModulation):
            return self._run_cv(symbols, seed)

        if isinstance(mod, PhaseShiftKeying):
            return self._run_dm()

        if isinstance(mod, DifferentialPhase):
            return self._run_dps(symbols, seed)

        if isinstance(mod, TwoStateKeying):
            return self._run_two()

        if isinstance(mod, FlawedKeying):
            return self._run_flaw()

        if isinstance(mod, _BASIS):
            return self._run_basis(symbols, seed)

        return self._run_cow()

    def _cv_args(self, t, xi):
        """
        The head cv_rate and cv_finite share: (V_A, T, xi, eta, v_el, beta,
        homodyne, trusted). ONE SPELLING, so the detection choice and the
        trusted flag cannot differ between the rate and the oracle beside it.
        """
        det = self.bob.detector

        return (
            self.modulation.v_a,
            t,
            xi,
            det.eta,
            det.v_el,
            self.security.beta,
            isinstance(det, Homodyne),
            det.trusted,
        )

    def _rate(self, t, xi, pairs=None):
        # (i_ab, chi_be, key, t_min, xi_max, delta), the last three None under
        # Asymptotic. `pairs`: scalars estimated from, 2 per heterodyne symbol.
        sec = self.security
        args = self._cv_args(t, xi)
        if not isinstance(sec, FiniteSize):
            i_ab, chi_be, key = _core.cv_rate(*args)

            return i_ab, chi_be, key, None, None, None

        # eps_pe, eps_smooth and eps_pa each take the given eps, unsplit
        # (Leverrier 2010's "all 1e-10").
        eps = (sec.eps, sec.eps, sec.eps)
        out = _core.cv_finite(*args, sec.n, sec.pe_fraction, *eps)
        if pairs is None or pairs >= sec.pe_fraction * sec.n:
            return out[:2] + (self._fer(out[2]),) + out[3:]

        # Re-entering at pe_fraction = pairs/n moves the estimation count
        # alone; Delta and the (1 - pe_fraction) throughput stay declared.
        wide = _core.cv_finite(*args, sec.n, pairs / sec.n, *eps)
        key = (1.0 - sec.pe_fraction) * (sec.beta * out[0] - wide[1] - out[5])

        return out[0], wide[1], self._fer(max(0.0, key)), wide[3], wide[4], out[5]

    def _fer(self, key):
        # The WHOLE rate scales by (1 - fer), not beta alone: a failed frame
        # yields no key and leaks none.
        fer = self.security.fer
        if fer is None:
            return key

        return (1.0 - fer) * key

    def _oracle(self, t, xi):
        i_ab, chi_be, key = _core.cv_rate(*self._cv_args(t, xi))

        return Oracle(max(0.0, key), i_ab, chi_be, t, xi)

    def _run_cv(self, symbols, seed):
        return self._claim_cv(self._measure_cv(symbols, seed))

    def _measure_cv(self, symbols, seed):
        t, xi, _, _ = self._channel()
        args = self._args(t, xi, symbols, seed)
        chain, est = self._sim(args)

        return Frames(
            symbols=int(symbols),
            seed=int(seed),
            dsp=chain,
            est=est,
            args=args,
        )

    def _claim_cv(self, frames):
        t, xi, _, xi_label = self._channel()
        chain, est = frames.dsp, frames.est
        v_err = None if chain is None else chain.v_err
        # Assembled BEFORE the phase row, which reads _bracket over them: a
        # phase term written on the channel xi alone leaves them outside its
        # (V_A + xi) and undercharges by row*(e^v - 1).
        extra, entries, floor, fade = self._charged(t, v_err)
        rows = [budget.Entry("channel", xi, _PLANES[xi_label])]
        step = None
        if chain is not None:
            # The phase term inflates (V_A + xi), not V_A alone (Kish, Quantum
            # 8, 1382 (2024), Eqs. (80)-(82)), and takes the CHANNEL xi together
            # with every row already on the light it rotates -- see _bracket.
            step = budget.phase(
                self.modulation.v_a,
                v_err,
                xi + self._bracket(entries),
            )
            xi = xi + step
            rows.append(budget.Entry("phase", step, budget.PHASE_NOTES["estimator"]))

        xi = xi + extra
        bud = self._bud(rows + list(entries), v_err)

        # A heterodyne symbol contributed two scalar pairs, a homodyne one a
        # single quadrature.
        pairs = None
        if est is not None:
            per = 1.0 if isinstance(self.bob.detector, Homodyne) else 2.0
            pairs = per * chain.symbols

        seen = (self._plane(t, v_err, fade), xi)
        if est is not None:
            # The estimator recovers the channel xi alone: run_symbols has no
            # impairment argument, so the rows add on here.
            #
            # NO infer() here: the fitted slope already carries exp(-v_err/2),
            # so t_hat IS the inferred channel and applying it again would
            # charge the phase row's transmittance half twice.
            t_hat, xi_hat = _clamp(est)
            seen = (max(1e-12, t_hat * fade), xi_hat + extra)

        i_ab, chi_be, key, t_min, xi_max, delta = self._rate(*seen, pairs)
        oracle = None if est is None else self._oracle(t, xi)
        if est is not None:
            est = replace(est, t_min=t_min, xi_max=xi_max)

        info = self._explain_cv(chain, est, oracle, pairs, step)
        for entry in entries:
            info[f"xi_{entry.source}"] = _q(entry.xi, "derived")

        if floor is not None:
            # Reported, never summed: v_err already contains it.
            info["v_floor"] = _q(floor, "derived")

        # Every assembled row, channel and phase included; equal to
        # res.budget.total.
        info["xi_total"] = _q(xi, "derived")
        # The transmittance half of the same plane. Equal to T where no fading
        # and no phase row was charged.
        info["T_claimed"] = _q(seen[0], "derived")
        if est is not None:
            # xi_est clamped plus every charged row. It differs from xi_total
            # by the estimator's own sampling error.
            info["xi_claimed"] = _q(seen[1], "derived")

        info["key_raw"] = _q(key, "derived")

        return LinkResult(
            key_rate=max(0.0, key),
            i_ab=i_ab,
            chi_be=chi_be,
            t_min=t_min,
            xi_max=xi_max,
            delta=delta,
            dsp=chain,
            est=est,
            budget=bud,
            explain=info,
            _oracle=oracle,
        )

    def _operating_dm(self):
        # The operating point both receiver models share, as (T, T_claimed, xi,
        # entries, budget). NO RATE: dm_rate folds eta and v_el in as untrusted
        # and nowhere else, so the trusted branch must not reach it.
        t, xi, _, xi_label = self._channel()
        extra, entries, _, fade = self._extra(t, None)
        bud = self._bud([budget.Entry("channel", xi, _PLANES[xi_label])] + list(entries))
        xi = xi + extra
        # The fading rows' transmittance half: a row charged as noise alone
        # against an unattenuated T is half the identity. No phase row here --
        # _check_dm refuses a DSP chain.
        plane = self._plane(t, None, fade)

        return t, plane, xi, entries, bud

    def _rows_dm(self, info, entries, xi):
        for entry in entries:
            info[f"xi_{entry.source}"] = _q(entry.xi, "derived")

        # No dac row -- _check_dm refuses a described front end.
        info["xi_total"] = _q(xi, "derived")

    def _point_dm(self):
        # (T_claimed, xi, info, dm_rate's tuple, budget) for both UNTRUSTED
        # branches. The certified branch computes it too rather than instead:
        # its rows are what that bound is read against.
        mod = self.modulation
        det = self.bob.detector
        t, plane, xi, entries, bud = self._operating_dm()
        out = _core.dm_rate(
            mod.states,
            mod.alpha,
            plane,
            xi,
            det.eta,
            det.v_el,
            self.security.beta,
        )
        info = self._explain_dm(t, xi, (out[0], out[1], out[3]), plane)
        self._rows_dm(info, entries, xi)

        return plane, xi, info, out, bud

    def _trusted_dm(self):
        # (T_claimed, xi, info, budget). Deliberately no dm_rate number beside
        # the certificate: that closed form fixes an UNTRUSTED receiver, so its
        # key, its I(X;Y) and its Holevo bound are read at a plane this run is
        # not taken on.
        t, plane, xi, entries, bud = self._operating_dm()
        info = self._explain_trust(t, plane)
        self._rows_dm(info, entries, xi)

        return plane, xi, info, bud

    def _run_dm(self):
        if self.bob.detector.trusted:
            return self._run_trust()

        # z_star lower-bounds the Alice-Bob correlation; against the Gaussian
        # one it is the alphabet's cost.
        t, xi, info, out, bud = self._point_dm()
        key = out[2]
        cert = None
        if self._certified() is not None:
            # Both numbers stay in the report: same protocol, same operating
            # point, differing by the proof alone.
            info["key_analytic"] = _q(key, "derived")
            cert = self._certify(t, xi, info)
            key = cert.key

        info["key_raw"] = _q(key, "derived")

        return LinkResult(
            key_rate=max(0.0, key),
            i_ab=out[0],
            chi_be=out[1],
            budget=bud,
            explain=info,
            certificate=cert,
        )

    def _run_trust(self):
        # i_ab and chi_be stay None rather than carry an untrusted number under
        # a trusted label; no key_analytic row, for the same reason.
        plane, xi, info, bud = self._trusted_dm()
        det = self.bob.detector
        cert = self.security.certify_trusted(self.modulation, plane, xi, det.eta, det.v_el)
        self._cert_rows(info, cert)
        info["key_raw"] = _q(cert.key, "derived")

        return LinkResult(
            key_rate=max(0.0, cert.key),
            budget=bud,
            explain=info,
            certificate=cert,
        )

    def _certify(self, t, xi, info):
        """
        The certificate at this operating point, as a q.Certificate, by the
        two-step method of Winick, Lutkenhaus & Coles, Quantum 2, 77 (2018).
        Step 2's `bound` is the proof and step 1's `upper` is not.
        """
        det = self.bob.detector
        # dm_rate's own untrusted substitution: eta and v_el fold into the
        # channel and nowhere else, the 2 being the heterodyne's two physical
        # receivers.
        eff = det.eta * t
        cert = self.security.certify(self.modulation, eff, xi + 2.0 * det.v_el / eff)
        self._cert_rows(info, cert)

        return cert

    def _cert_rows(self, info, cert):
        """
        The certified branch's rows. `bound` is step 2 and IS the proof;
        `upper` is step 1's Frank-Wolfe value and proves nothing, so it is
        labelled a diagnostic. `cert` is None on the plan.
        """
        sec = self.security
        shown = "derived" if cert is not None else "derived (run to compute)"
        info["cutoff"] = _q(sec.cutoff, "pinned")
        # The postselection pair: the radius below which Bob discards, and the
        # ANGULAR guard band each side of a sector boundary. The second is a
        # guard angle, not a phase noise, hence phase_guard.
        info["cut"] = _q(sec.cut, "pinned")
        info["phase_guard"] = _q(sec.phase, "pinned")
        info["eps"] = _q(sec.eps, "pinned")
        info["steps"] = _q(sec.steps, "pinned")
        info["bound"] = _q(None if cert is None else cert.bound, shown)
        info["upper"] = _q(None if cert is None else cert.upper, "diagnostic")
        info["p_pass"] = _q(None if cert is None else cert.p_pass, shown)
        info["delta_ec"] = _q(None if cert is None else cert.delta_ec, shown)
        # What the photon-number cutoff costs, and the continuity price of eps.
        # A key below the size of the second is that price and not physics.
        info["viol"] = _q(None if cert is None else cert.viol, shown)
        info["zeta"] = _q(None if cert is None else cert.zeta, shown)
        info["steps_taken"] = _q(None if cert is None else cert.steps, shown)
        info["solver"] = "unrun" if cert is None else cert.status

    def _run_dps(self, symbols, seed):
        sec = self.security
        t = self._channel()[0]
        if sec.qber is None:
            return self._run_clicks(t, symbols, seed)

        p_click = self._pclick(t)
        key = _core.dps_rate(p_click, sec.qber, self.modulation.mu, sec.f)
        info = self._explain_dps(click=p_click)
        info["key_raw"] = _q(key, "derived")

        return LinkResult(
            key_rate=max(0.0, key),
            p_click=p_click,
            qber=sec.qber,
            leak=self._leak(p_click),
            explain=info,
        )

    def _run_clicks(self, t, symbols, seed):
        # Error rate, sift rate and visibility are outputs. Doubles keep a
        # random bit, so p_click = sift_rate; mu stays ALICE's photon number.
        det = self.bob.detector
        optics = self.bob.receiver
        alice = self.alice
        # PORT ORDER, not pair()'s sorted (hi, lo): eta_d1 is the pi-phase
        # detector's, and sorting here would swap which physical port the
        # engine answers at.
        one, two = det.both()
        # Each port's floor carries every declared background flux, in PORT
        # ORDER again.
        dark_one, dark_two = self._floors()
        out = _core.run_clicks(
            int(symbols),
            int(seed),
            self.modulation.mu,
            t,
            one.eta,
            dark_one,
            optics.visibility,
            int(optics.delay),
            alice.laser.linewidth,
            alice.symbol_rate,
            det.dead_time,
            det.afterpulse,
            0,
            False,
            det.jitter,
            det.window,
            det.tail_frac,
            det.tail_time,
            two.eta,
            dark_two,
        )
        key = _core.dps_rate(out.sift_rate, out.qber, self.modulation.mu, self.security.f)
        info = self._explain_dps(out)
        info["key_raw"] = _q(key, "derived")

        return LinkResult(
            key_rate=max(0.0, key),
            p_click=out.sift_rate,
            qber=out.qber,
            leak=self._leak(out.sift_rate),
            clicks=int(out.clicks),
            sifted=int(out.sifted),
            doubles=int(out.doubles),
            slots=int(out.n_slots),
            sift_rate=out.sift_rate,
            visibility=out.visibility,
            mu_bob=out.mu_bob,
            explain=info,
        )

    def _sampled(self, out):
        # (gain, qber, y1, e1, q1): the signal pair and the three decoy bounds.
        # Both BasisOut getters build a fresh list per read, so the two vectors
        # are hoisted into locals.
        mu, nu1, nu2 = self.modulation.decoy.intensities
        gain = out.gain
        qber = out.qber
        y1, e1, q1 = _core.decoy_bounds(
            mu,
            nu1,
            nu2,
            gain[0],
            qber[0],
            gain[1],
            qber[1],
            gain[2],
            qber[2],
        )

        return gain[0], qber[0], y1, e1, q1

    def _dead(self):
        return self._flags().get("dead")

    def _leak(self, sift):
        # The share of the sifted key backflash hands Eve: a security flag
        # beside the rate, never in it, and no parameter estimate sees it.
        flag = self._flags().get("backflash")
        if flag is None:
            return None

        return impairments.backflash_leak(flag.prob, sift)

    def _gate(self):
        """
        The acceptance width one click gate integrates, in s: q.ClickDetector's
        window where one was declared and the whole symbol period otherwise,
        which is what run_clicks bins when it builds no arrival-time kernel.
        """
        det = self.bob.detector
        period = 1.0 / self.alice.symbol_rate
        if det.window > 0.0:
            return min(det.window, period)

        return period

    def _flux(self, split):
        """
        Every declared background as mean photons standing in ONE gate at a
        receiver taking `split` of the flux: 0.5 for the two-gate click
        receivers and 1.0 for q.NullingReceiver, which reads one gate per pulse.
        """
        flags = self._flags()
        coexist = flags.get("coexist")
        probe = flags.get("probe")
        if coexist is None and probe is None:
            return ()

        ch = self.channel
        length = getattr(ch, "length", None)
        if length is None or self.alice is None:
            raise ValueError(
                "a background flux needs q.Channel(length=...) and Alice's "
                "symbol_rate: Raman and Rayleigh photons are born along the "
                "fibre and counted in a gate, so neither a bare transmittance "
                "nor a closed-form run without a clock can carry one"
            )

        alpha = getattr(ch, "alpha", 0.2)
        period = 1.0 / self.alice.symbol_rate
        gate = self._gate()
        # The receive optics multiply because the quadrature route refers these
        # rows in by launch*span alone: same statement, one plane along.
        receive = self._chain()[2]
        out = []
        if coexist is not None:
            if self.bob.detector.passband <= 0.0:
                raise ValueError(
                    "a q.Coexistence on a click link needs "
                    "q.ClickDetector(passband=...): impairments.raman_photons "
                    "returns an OCCUPANCY per detection mode, and a background "
                    "click probability is that occupancy times an optical "
                    "passband times the gate. qkd.budget.raman_width's number "
                    "is a DETECTION bandwidth and is not it -- a threshold "
                    "detector has no local oscillator to be the filter. "
                    "dnu = c dlambda/lambda^2"
                )

            watts = coexist.channels * 10.0 ** (coexist.launch / 10.0) * 1e-3
            occ = impairments.raman_photons(
                watts,
                length,
                alpha=alpha,
                beta=coexist.beta,
                wavelength=coexist.wavelength,
                backward=coexist.backward,
            )
            # pol=2 undoes Kumar, Qin and Alleaume Eq. (6)'s 1/2, which is a
            # local oscillator's polarisation selectivity, not a detector's.
            out.append(_core.bg_raman(occ * receive, self.bob.detector.passband, gate, 2, split)[0])

        if probe is not None:
            photons = impairments.rayleigh_photons(
                probe.power,
                length,
                period,
                alpha=alpha,
                coeff=probe.coeff,
                index=probe.index,
                wavelength=probe.wavelength,
            )
            # The 2 undoes rayleigh_photons' own oscillator 1/2; `split` is
            # Bob's optics dividing an unpolarised return and is a DIFFERENT
            # half that happens to be the same number.
            out.append(_core.bg_rayleigh(2.0 * photons * receive, period, gate, split))

        return tuple(out)

    def _floors(self):
        """
        The per-gate floor each of Bob's two detectors reads: its dark
        probability with every declared background flux composed in.
        """
        one, two = self.bob.detector.both()
        flux = self._flux(0.5)
        if not flux:
            return one.dark, two.dark

        return (
            _core.bg_floor(one.dark, list(flux), one.eta),
            _core.bg_floor(two.dark, list(flux), two.eta),
        )

    def _onegate(self, det):
        """
        The same floor for a receiver reading ONE gate per pulse, where the
        whole flux reaches the one detector rather than half of it. Returned as
        declared where no flux was: composing an empty one moves every number
        this reader produces by a few ulp.
        """
        flux = self._flux(1.0)
        if not flux:
            return det.dark

        return _core.bg_floor(det.dark, list(flux), det.eta)

    def _background(self):
        """
        Per-pulse yield Y0 = 1 - (1 - dark_0)(1 - dark_1) out of the TWO gates
        a click receiver reads per bit, collapsing to 1 - (1 - dark)^2 where no
        partner was named. Ma, Qi, Zhao & Lo's GYS Y0 = 1.7e-6 is twice GYS's
        8.5e-7. Each gate's floor carries every declared background flux, so a
        coexisting channel or a Rayleigh return raises Y0 rather than forming a
        row of its own.
        """
        one, two = self._floors()

        return 1.0 - (1.0 - one) * (1.0 - two)

    def _gain(self, mu, eta, y0, miss, dead):
        # One intensity's gain and error rate; a DeadTime descriptor swaps in
        # the afterpulsing form. Every branch takes Y0, not `dark`.
        if self._announced() == "pair":
            # A "gain" here is the probability of a CONCLUSIVE round, so the 1/4
            # sifting is already inside it and no q_sift multiplies it.
            if mu <= 0.0:
                return _core.sarg_yield(eta, y0, miss, 0)

            return _core.sarg_gain(mu, eta, y0, miss)

        if dead is not None:
            return impairments.afterpulse(mu, eta, y0, dead.afterpulse, miss)

        if mu <= 0.0:
            return y0, 0.5

        return _core.decoy_gain(mu, eta, y0, miss)

    def _keyrate(self, gain, qber, q1, e1):
        """
        The one line the three basis-keyed analyses differ in: which privacy
        amplification the decoy-bounded single photons are charged.
        """
        f = self.security.f
        if self._announced() == "pair":
            # Fung, Tamaki & Lo Eq. (39) with its two-photon term at ZERO: the
            # Ma-Qi-Zhao-Lo inversion bounds Y1 from a three-intensity set and
            # nothing bounds Y2. A zero term costs rate, the safe direction.
            return _core.sarg_rate(gain, qber, q1, e1, 0.0, 0.0, f)

        sift = self.modulation.sift
        if self._bases() == 3:
            return _core.sixstate_rate(sift, gain, qber, q1, e1, f)

        return _core.bb84_rate(sift, gain, qber, q1, e1, f)

    def _sifted(self, gain):
        # The sifted rate a backflash leaks a share of. The pair announcement
        # carries its sifting inside the gain, so there the two are one number.
        sift = self.modulation.sift
        if sift is None:
            return gain

        return sift * gain

    def _contrast(self):
        # The carrier's polarisation contrast, None where none is named.
        mod = self.modulation
        if not isinstance(mod, PolarisationKeying):
            return None

        frame = mod.frame
        if frame.dispersion <= 0.0:
            return mod.contrast()

        length = getattr(self.channel, "length", None)
        if length is None:
            raise ValueError(
                "a PMD coefficient accumulates over a fibre span: give the "
                "channel as q.Fiber(length=...) or drop dispersion"
            )

        return mod.contrast(impairments.dgd(frame.dispersion, length))

    def _misalign(self):
        # The probability a matching-basis photon lands on the wrong detector.
        # Analyser misalignment and a named carrier's contribution compose as
        # CONTRASTS, not as error rates; a transmitter extinction ratio then
        # DILUTES that error rate, a third composition again.
        key = (self.bob, self.modulation, self.channel, self.impairments)
        held = self._misalign_at
        if held is not None and held[0] == key:
            return held[1]

        miss = self.bob.receiver.misalign
        contrast = self._contrast()
        if contrast is None:
            out = miss
        else:
            out = _core.pol_error(miss, contrast)

        ratio = self._extinct()
        if ratio is not None:
            # Huang et al. Eq. (5). Their next equation credits the identity
            # fraction back out of privacy amplification; charging it here as a
            # QBER alone declines that credit, the pessimistic direction.
            out = impairments.extinction(ratio, out)

        self._misalign_at = (key, out)

        return out

    def _decoy(self):
        # (eta, gain, qber, y1, e1, q1): Alice-to-Bob transmittance including
        # detector efficiency, then the signal gain and the decoy bounds.
        mod = self.modulation
        eta = self._channel()[0] * self.bob.detector.eta
        mu, nu1, nu2 = mod.decoy.intensities
        # One receiver, three intensities: the yield floor, the misalignment and
        # the detector memory do not vary with the intensity.
        y0 = self._background()
        miss = self._misalign()
        dead = self._dead()
        q_mu, e_mu = self._gain(mu, eta, y0, miss, dead)
        q_nu1, e_nu1 = self._gain(nu1, eta, y0, miss, dead)
        q_nu2, e_nu2 = self._gain(nu2, eta, y0, miss, dead)
        y1, e1, q1 = _core.decoy_bounds(mu, nu1, nu2, q_mu, e_mu, q_nu1, e_nu1, q_nu2, e_nu2)

        return eta, q_mu, e_mu, y1, e1, q1

    def _cow(self):
        # (t_data, gain, e_bit, q_z, ceiling). The tap precedes the detector,
        # so the data line sees T*(1 - split)*eta; ceiling rides on mu + mu/r.
        mod = self.modulation
        det = self.bob.detector
        data = self._channel()[0] * (1.0 - self.bob.receiver.split) * det.eta
        gain, e_bit = self._gain(mod.mu, data, self._background(), self._misalign(), self._dead())
        res = mod.residual
        # Applied after the honest gain, so both gain models reach it.
        if res > 0.0:
            gain, e_bit = _core.cow_data(gain, e_bit, res, data)

        q_z = (1.0 - mod.decoy_frac) * gain
        ceiling = _core.cow_ceiling(mod.decoy_frac, data, mod.mu + res)

        return data, gain, e_bit, q_z, ceiling

    def _run_basis(self, symbols, seed):
        if self.alice is not None:
            return self._run_pulses(symbols, seed)

        decoy = self._decoy()
        _, gain, qber, _, e1, q1 = decoy
        key = self._keyrate(gain, qber, q1, e1)
        # Threaded, not recomputed: the bounds cost three _gain calls each.
        info = self._explain_basis(decoy=decoy)
        block = self.security.block
        if block is not None:
            info["key_asymptotic"] = _q(key, "derived")
            length, s0, s1, phi, kept = self._run_finite(None, block.n, info)

            return LinkResult(
                key_rate=length / block.n,
                p_click=gain,
                qber=qber,
                leak=self._leak(self._sifted(gain)),
                slots=int(block.n),
                key_length=length,
                s0=s0,
                s1=s1,
                phi=phi,
                n_key=kept,
                explain=info,
            )

        info["key_raw"] = _q(key, "derived")

        return LinkResult(
            key_rate=max(0.0, key),
            p_click=gain,
            qber=qber,
            leak=self._leak(self._sifted(gain)),
            explain=info,
        )

    def _run_pulses(self, symbols, seed):
        # Gains and error rates divided out of integer counts. This path and the
        # closed form converge as dark -> 0; the closed form charges background
        # at 1/2.
        mod = self.modulation
        det = self.bob.detector
        t = self._channel()[0]
        block = self.security.block
        if block is not None and int(block.n) != int(symbols):
            raise ValueError(
                f"a declared block of {int(block.n)} pulses and a sampled run "
                f"of {int(symbols)} are two block sizes: the finite-key length "
                "is a statement about ONE block, so run(symbols=...) must match "
                "q.KeyBlock(n=...), or leave Alice off for the closed form"
            )

        # PORT ORDER, as in _run_clicks.
        one, two = det.both()
        dark_one, dark_two = self._floors()
        out = _core.run_basis(
            int(symbols),
            int(seed),
            tuple(mod.decoy.intensities),
            mod.decoy.weights,
            t,
            one.eta,
            dark_one,
            self._misalign(),
            mod.sift,
            0,
            two.eta,
            dark_two,
        )
        rates = self._sampled(out)
        gain, qber, _, e1, q1 = rates
        key = self._keyrate(gain, qber, q1, e1)
        info = self._explain_basis(out, rates=rates)
        shared = dict(
            p_click=gain,
            qber=qber,
            leak=self._leak(self._sifted(gain)),
            clicks=sum(out.clicks),
            sifted=sum(out.sifted),
            doubles=int(out.doubles),
            slots=int(out.n_pulses),
            sift_rate=out.sift_rate,
            explain=info,
        )
        if block is not None:
            info["key_asymptotic"] = _q(key, "derived")
            emitted = float(out.n_pulses)
            length, s0, s1, phi, kept = self._run_finite(out, emitted, info)

            return LinkResult(
                key_rate=length / emitted,
                key_length=length,
                s0=s0,
                s1=s1,
                phi=phi,
                n_key=kept,
                **shared,
            )

        info["key_raw"] = _q(key, "derived")

        return LinkResult(key_rate=max(0.0, key), **shared)

    def _shares(self):
        # (key share, test share) of the emitted pulses, both derived from the
        # one sifting factor; _core owns the inversion.
        sift = self.modulation.sift
        share = _core.key_share(sift)

        return share, sift - share

    def _expected(self, n_total):
        """
        (n_key, m_key, n_test, m_test) as EXPECTED counts, per intensity, from
        the closed-form gains: a typical block, not a bound on one.
        """
        mod = self.modulation
        eta = self._channel()[0] * self.bob.detector.eta
        y0 = self._background()
        miss = self._misalign()
        dead = self._dead()
        key, test = self._shares()
        rows = [[], [], [], []]
        for mu_k, p_k in zip(mod.decoy.intensities, mod.decoy.weights):
            gain, err = self._gain(mu_k, eta, y0, miss, dead)
            sent = n_total * p_k * gain
            rows[0].append(sent * key)
            rows[1].append(sent * key * err)
            rows[2].append(sent * test)
            rows[3].append(sent * test * err)

        return tuple(tuple(row) for row in rows)

    def _measured(self, out):
        """
        The same four triples as integer counts off a sampled run.
        """
        return (
            tuple(float(x) for x in out.key_sifted),
            tuple(float(x) for x in out.key_errors),
            tuple(float(x) for x in out.test_sifted),
            tuple(float(x) for x in out.test_errors),
        )

    def _length(self, counts, n_total):
        """
        Lim's key length in bits and the four quantities it is built from, as
        (length, s0, s1, phi, n_key, e_key).
        """
        mod = self.modulation
        block = self.security.block
        mu, nu1, nu2 = mod.decoy.intensities
        probs = mod.decoy.weights
        # Lim's budget collapses to eps_sec = 21*eps, so each of the twenty-one
        # bounds inside it is taken at eps_sec/21.
        eps = block.eps_sec / 21.0
        n_key, m_key, n_test, m_test = counts
        args = (mu, nu1, nu2, probs)
        s0, s1 = _core.decoy_counts(*args, n_key, eps)
        _, s1_test = _core.decoy_counts(*args, n_test, eps)
        v1 = _core.decoy_errors(*args, m_test, eps)
        phi = _core.bb84_phase(v1, s1_test, s1, block.eps_sec)
        kept = sum(n_key)
        if kept <= 0.0:
            return 0.0, s0, s1, phi, 0.0, 0.0

        err = min(0.5, sum(m_key) / kept)
        length = _core.bb84_length(
            s0,
            s1,
            phi,
            kept,
            err,
            self.security.f,
            block.eps_sec,
            block.eps_cor,
        )
        if block.fer is not None:
            # A failed frame yields no key and leaks none, so the WHOLE length
            # scales, as on q.FiniteSize.
            length *= 1.0 - block.fer

        return length, s0, s1, phi, kept, err

    def _run_finite(self, out, n_total, info):
        counts = self._expected(n_total) if out is None else self._measured(out)
        length, s0, s1, phi, kept, err = self._length(counts, n_total)
        block = self.security.block
        info["n_block"] = _q(n_total, "pinned" if out is None else "derived")
        info["eps_sec"] = _q(block.eps_sec, "pinned")
        info["eps_cor"] = _q(block.eps_cor, "pinned")
        info["key_share"] = _q(self._shares()[0], "derived")
        info["s0"] = _q(s0, "derived")
        info["s1"] = _q(s1, "derived")
        info["phi"] = _q(phi, "derived")
        info["n_key"] = _q(kept, "derived")
        info["e_key"] = _q(err, "derived")
        info["fer"] = _q(block.fer, "pinned" if block.fer is not None else "absent")
        info["key_length"] = _q(length, "derived")
        info["counts"] = "measured" if out is not None else "expected"

        return length, s0, s1, phi, kept

    def _e_phase(self, t, e_bit):
        """
        The phase-error bound b92_point priced this operating point at, rebuilt
        for the report because the engine returns the rate and not the bound.
        """
        mod = self.modulation
        floor = self.security.e_phase
        floor = 0.0 if floor is None else floor
        if mod.reference:
            return floor

        # A SECOND COPY of b92.rs::regain, evaluating their SINGLE-PHOTON gain
        # back rather than the coherent conclusive rate. Transcribed, so it can
        # drift; test/protocols.py's test_derived_phase pins the two together.
        over = mod.overlap
        loss = 1.0 - self.bob.detector.eta * t
        den = 1.0 - 2.0 * e_bit * over * over
        gain = 0.0
        if den > 0.0:
            share = 2.0 * e_bit * (1.0 - over * over) / den
            gain = 0.5 * (1.0 - loss) * ((1.0 - over * over) + share * over * over)

        if not 0.0 < over < 1.0 or loss >= 1.0 or gain <= 0.0:
            return 0.5

        return max(floor, _core.b92_phase(over, loss, gain, e_bit))

    def _run_two(self):
        # The conclusive rate, the bit error and (on the plain branch) the phase
        # error are all outputs of the receiver and the span.
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        t = self._channel()[0]
        if getattr(mod, "source", "coherent") == "single":
            return self._run_plain(t)

        floor = 0.0 if sec.e_phase is None else sec.e_phase
        out = _core.b92_point(
            mod.mu,
            t,
            det.eta,
            self.bob.receiver.visibility,
            self._onegate(det),
            floor,
            sec.f,
            mod.reference,
        )
        info = self._explain_two(out)
        # b92_point clamps in Rust, as cow_rate does, so the two agree here.
        info["key_raw"] = _q(out[3], "derived")

        return LinkResult(
            key_rate=max(0.0, out[3]),
            p_click=out[0],
            qber=out[1],
            leak=self._leak(out[0]),
            explain=info,
        )

    def _run_plain(self, t):
        """
        Plain B92 over Tamaki and Lutkenhaus's own channel: a single-photon
        source, their depolarising-with-loss map, their phase bound and their
        rate equation, with none of the coherent-state seam b92_point carries.
        """
        mod = self.modulation
        out = _core.b92_plain(
            mod.overlap,
            1.0 - self.bob.detector.eta * t,
            mod.depol,
            self.security.f,
        )
        info = self._explain_plain(t, out)
        # b92_plain clamps in Rust, as b92_point does, so this row agrees.
        info["key_raw"] = _q(out[3], "derived")
        # Not pooled with _run_two's tail: a shared _b92_result measured -6
        # lines over TWO call sites, and a shared _b92_loss for (1 - eta*t) +1.

        return LinkResult(
            key_rate=max(0.0, out[3]),
            p_click=out[0],
            qber=out[1],
            leak=self._leak(out[0]),
            explain=info,
        )

    def _run_cow(self):
        sec = self.security
        cow = self._cow()
        _, _, e_bit, q_z, _ = cow
        key = _core.cow_rate(q_z, e_bit, sec.e_phase, sec.f)
        info = self._explain_cow(cow)
        info["key_raw"] = _q(key, "derived")

        return LinkResult(
            key_rate=max(0.0, key),
            p_click=q_z,
            qber=e_bit,
            leak=self._leak(q_z),
            explain=info,
        )

    def explain(self):
        """
        The resolved plan as {"value", "label"} rows, emitting no symbols; a
        value only a run could fill reads "derived (run to compute)".
        """
        mod = self.modulation
        self._check()
        if isinstance(mod, GaussianModulation):
            return self._explain_cv()

        if isinstance(mod, PhaseShiftKeying):
            if self._certified() is None:
                # Every stage is a closed form: the plan is the run.
                return self._run_dm().explain

            if self.bob.detector.trusted:
                # No key_analytic row: dm_rate's receiver is not this one.
                _, _, info, _ = self._trusted_dm()
                self._cert_rows(info, None)

                return info

            # The certified branch is a Frank-Wolfe minimisation then an
            # interior-point solve, so the plan names its rows without running.
            _, _, info, out, _ = self._point_dm()
            info["key_analytic"] = _q(out[2], "derived")
            self._cert_rows(info, None)

            return info

        if isinstance(mod, DifferentialPhase):
            return self._explain_dps()

        if isinstance(mod, TwoStateKeying):
            return self._run_two().explain

        if isinstance(mod, FlawedKeying):
            return self._run_flaw().explain

        if isinstance(mod, _BASIS):
            return self._explain_basis()

        return self._explain_cow()

    def _explain_two(self, out):
        # `out` is b92_point's (p_fil, e_bit, ceiling, rate); explain() reaches
        # this through _run_two.
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        t, _, t_label, _ = self._channel()
        over = mod.overlap
        info = {
            **self._head("two-state keying"),
            "mu": _q(mod.mu, "pinned"),
            "reference": _q(mod.reference, "pinned"),
            "T": _q(t, t_label),
            "eta": _q(det.eta, "pinned"),
            # ONE gate per pulse, so this is the per-pulse background and no
            # two-port Y0 is derived.
            "dark": _q(det.dark, "pinned"),
            "visibility": _q(self.bob.receiver.visibility, "pinned"),
            "f": _q(sec.f, "pinned"),
            "overlap": _q(over, "derived"),
            "loss": _q(1.0 - det.eta * t, "derived"),
            # What optimal unambiguous discrimination hands Eve. The plain
            # protocol dies where the surviving fraction falls to it.
            "discrimination": _q(_core.b92_usd(over), "derived"),
            "gain": _q(out[0], "derived"),
            "qber": _q(out[1], "derived"),
            "e_phase_floor": _q(sec.e_phase, "pinned" if sec.e_phase is not None else "absent"),
            "e_phase": _q(self._e_phase(t, out[1]), "pinned" if mod.reference else "derived"),
            # The beam-splitting attack's cap, which the rate is held under.
            "ceiling": _q(out[2], "derived"),
        }
        self._optics(info)
        self._flag_row(info, self._leak(out[0]), "derived")

        return info

    def _explain_plain(self, t, out):
        """
        The single-photon branch's report. No fringe row and no ceiling: the
        beam-splitting cap prices multiphoton pulses this source does not emit.
        """
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        t_label = self._channel()[2]
        over = mod.overlap
        loss = 1.0 - det.eta * t
        limit, best = _core.b92_limit(loss, sec.f)
        info = {
            **self._head("two-state keying"),
            "source": "single photon",
            "T": _q(t, t_label),
            "eta": _q(det.eta, "pinned"),
            "f": _q(sec.f, "pinned"),
            # mu names the overlap on this branch, through exp(-2*mu).
            "mu": _q(mod.mu, "pinned"),
            "overlap": _q(over, "derived"),
            "depol": _q(mod.depol, "pinned"),
            "loss": _q(loss, "derived"),
            # What optimal unambiguous discrimination hands Eve, and the loss at
            # which a NOISELESS channel dies: the two coincide at
            # loss = overlap, and depolarising noise moves the crossing in.
            "discrimination": _q(_core.b92_usd(over), "derived"),
            "usd_boundary": _q(over, "structural"),
            "gain": _q(out[0], "derived"),
            "qber": _q(out[1], "derived"),
            "e_phase": _q(out[2], "derived"),
            # The largest depolarising rate this loss still distils key at, and
            # the AMPLITUDE overlap attaining it. Tamaki and Lutkenhaus's
            # Fig. 2(b) plots the SQUARE of that second number.
            "depol_limit": _q(limit, "derived"),
            "overlap_best": _q(best, "derived"),
        }
        self._optics(info)
        self._flag_row(info, self._leak(out[0]), "derived")

        return info

    def _explain_cv(self, chain=None, est=None, oracle=None, pairs=None, step=None):
        # `step` is _claim_cv's own budget.phase term, over the channel xi AND
        # the bracket rows; explain() passes none and rebuilds the same term.
        t, xi, t_label, xi_label = self._channel()
        det = self.bob.detector
        sec = self.security
        hom = isinstance(det, Homodyne)
        info = {
            **self._head("gaussian modulation", "homodyne" if hom else "heterodyne"),
            "v_a": _q(self.modulation.v_a, "pinned"),
            "T": _q(t, t_label),
            "xi_input": _q(xi, xi_label),
            "eta": _q(det.eta, "pinned"),
            "v_el": _q(det.v_el, "pinned"),
            "trusted": _q(det.trusted, "pinned"),
            "beta": _q(sec.beta, "pinned"),
        }
        self._optics(info)
        # The transmittance the rate will be claimed at, beside T, which stays
        # the physical chain. The DSP block below overwrites it.
        fade = self._fade()
        info["T_claimed"] = _q(self._plane(t, None, fade), "derived")
        if isinstance(sec, FiniteSize):
            info["n"] = _q(sec.n, "pinned")
            info["eps"] = _q(sec.eps, "pinned")
            info["pe_fraction"] = _q(sec.pe_fraction, "pinned")
            info["fer"] = _q(sec.fer, "default" if sec.fer is None else "pinned")

        if self.dsp is not None:
            pin = self.dsp.phase.v_err
            label = "pinned" if pin is not None else "derived"
            v_err = pin if chain is None else chain.v_err
            info["stages"] = (
                "closed form only (v_err pinned)"
                if pin is not None
                else "symbol pipeline + pilot DSP + parameter estimation"
            )
            unrun = label == "derived" and v_err is None
            shown = "derived (run to compute)" if unrun else label
            info["v_err"] = _q(v_err, shown)
            # A Link run is always budget.phase's estimator form.
            info["phase_form"] = _q("estimator", "default")
            va = self.modulation.v_a
            extra = step
            if extra is None and v_err is not None:
                # The run's bracket, rebuilt from the same assembly: written on
                # the channel xi alone this row reads LOW by row*(e^v - 1) per
                # bracket row, which overstates the rate.
                entries = self._charged(t, v_err)[1]
                extra = budget.phase(va, v_err, xi + self._bracket(entries))

            info["xi_phase"] = _q(extra, shown)
            if extra is not None:
                info["xi_input"] = _q(xi + extra, "derived")

            # Kish App. E (78)'s OTHER half: the variance a residual rotation
            # moves out of the fitted slope is the variance xi_phase reports.
            info["T_claimed"] = _q(
                None if v_err is None else self._plane(t, v_err, fade),
                shown,
            )

        if est is not None:
            info["T_est"] = _q(est.T, "derived")
            info["xi_est"] = _q(est.xi, "derived")
            if pairs is not None:
                # The smaller of what the run measured and the block discloses.
                nominal = pairs
                if isinstance(sec, FiniteSize):
                    nominal = min(pairs, sec.pe_fraction * sec.n)

                info["pe_pairs"] = _q(nominal, "derived")

            if est.t_min is not None:
                info["t_min"] = _q(est.t_min, "derived")
                info["xi_max"] = _q(est.xi_max, "derived")

        if oracle is not None:
            info["T_oracle"] = _q(oracle.T, "pinned")
            info["xi_oracle"] = _q(oracle.xi, "derived")
            info["key_oracle"] = _q(oracle.key_rate, "derived")

        return info

    def _explain_dm(self, t, xi, out, plane=None):
        # i_ab is this constellation's own I(X;Y); i_ab_gauss is the log2(1 +
        # SNR) upper bound the literature substitutes. `t` is the physical chain
        # and `plane` the transmittance the rate was taken at; the untrusted
        # substitution below reads the PLANE, not the chain.
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        i_ab, chi_be, z_star = out
        plane = t if plane is None else plane
        chan, t_label, xi_label = self._channel()[1:]
        # A SECOND COPY of the untrusted substitution dm_rate applies
        # internally, transcribed, so these rows can drift from the rate beside
        # them; test/protocols.py's test_effective_plane pins them together.
        t_eff = det.eta * plane
        xi_eff = xi + 2.0 * det.v_el / t_eff
        gauss = _core.dm_info(mod.states, mod.alpha, t_eff, xi_eff)[1]
        _, z_lin, w, z_gauss = _core.dm_moments(mod.states, mod.alpha)
        info = {
            **self._head("phase-shift keying", "heterodyne"),
            "states": _q(mod.states, "pinned"),
            "alpha": _q(mod.alpha, "pinned"),
            "v_a": _q(mod.v_a, "derived"),
            "bits": _q(mod.bits, "derived"),
            "T": _q(t, t_label),
            # T faded by every fading row's own <sqrt(eta)>^2: the plane T_eff
            # is built on.
            "T_claimed": _q(plane, "derived"),
            # The channel's own noise; impairment rows and xi_total come after.
            "xi_input": _q(chan, xi_label),
            "eta": _q(det.eta, "pinned"),
            "v_el": _q(det.v_el, "pinned"),
            "trusted": _q(det.trusted, "pinned"),
            "beta": _q(sec.beta, "pinned"),
            # The untrusted substitution T -> eta*T, xi -> xi + 2*v_el/(eta*T),
            # the 2 being the heterodyne's two physical receivers.
            "T_eff": _q(t_eff, "derived"),
            "xi_eff": _q(xi_eff, "derived"),
            # z_lin and w are at the CHANNEL INPUT plane; z_star and z_gauss at
            # the output, already carrying their sqrt(T).
            "z_lin": _q(z_lin, "derived"),
            "w": _q(w, "derived"),
            "z_star": _q(z_star, "derived"),
            "z_gauss": _q(math.sqrt(t_eff) * z_gauss, "derived"),
            "i_ab": _q(i_ab, "derived"),
            "i_ab_gauss": _q(gauss, "derived"),
            "chi_be": _q(chi_be, "derived"),
            "key_gauss": _q(sec.beta * gauss - chi_be, "derived"),
        }
        self._optics(info)

        return info

    def _explain_trust(self, t, plane):
        # The trusted certified branch's rows. eta_d and v_el enter the POVM,
        # never the channel, so no T_eff or xi_eff row forms and the untrusted
        # branch's i_ab, chi_be, i_ab_gauss, key_gauss, z_star and z_gauss are
        # absent rather than reported at a plane this run is not taken on. Never
        # a shared prefix with _explain_dm: a row added there would surface here
        # at that plane.
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        chan, t_label, xi_label = self._channel()[1:]
        _, z_lin, w, _ = _core.dm_moments(mod.states, mod.alpha)
        info = {
            **self._head("phase-shift keying", "heterodyne"),
            "states": _q(mod.states, "pinned"),
            "alpha": _q(mod.alpha, "pinned"),
            "v_a": _q(mod.v_a, "derived"),
            "bits": _q(mod.bits, "derived"),
            "T": _q(t, t_label),
            "T_claimed": _q(plane, "derived"),
            "xi_input": _q(chan, xi_label),
            "eta": _q(det.eta, "pinned"),
            "v_el": _q(det.v_el, "pinned"),
            "trusted": _q(det.trusted, "pinned"),
            "beta": _q(sec.beta, "pinned"),
            # z_lin and w are at the CHANNEL INPUT and carry no transmittance,
            # so they read the same under either receiver model.
            "z_lin": _q(z_lin, "derived"),
            "w": _q(w, "derived"),
        }
        self._optics(info)

        return info

    def _explain_dps(self, out=None, click=None):
        # `click` is _run_dps' own p_click on the pinned-qber branch. NOT
        # p_ideal below, which the simulated branch reports beside a sift rate.
        t, _, t_label, _ = self._channel()
        det = self.bob.detector
        sec = self.security
        info = {
            **self._head("differential phase"),
            "mu": _q(self.modulation.mu, "pinned"),
            "T": _q(t, t_label),
            "eta": _q(det.eta, "pinned"),
            "dark": _q(det.dark, "pinned"),
            # dark is ONE port's gate as declared; y0 is the per-slot floor out
            # of the two ports, every background flux composed in, and is what
            # _run_clicks and _pclick read.
            "y0": _q(self._background(), "derived"),
            "f": _q(sec.f, "pinned"),
        }
        self._optics(info)
        if sec.qber is not None:
            if click is None:
                click = self._pclick(t)

            info["p_click"] = _q(click, "derived")
            info["qber"] = _q(sec.qber, "pinned")
            self._flag_row(info, self._leak(click), "derived")

            return info

        optics = self.bob.receiver
        shown = "derived" if out is not None else "derived (run to compute)"
        info["stages"] = "click pipeline + threshold detection"
        info["delay"] = _q(optics.delay, "pinned")
        info["contrast"] = _q(optics.visibility, "pinned")
        info["linewidth"] = _q(self.alice.laser.linewidth, "pinned")
        info["symbol_rate"] = _q(self.alice.symbol_rate, "pinned")
        info["dead_time"] = _q(det.dead_time, "pinned")
        info["afterpulse"] = _q(det.afterpulse, "pinned")
        # Separate rows: window_loss is light that reached no window and reads
        # as a smaller eta; bin_leak reached the WRONG window and is half error.
        if det.timed():
            info["jitter"] = _q(det.jitter, "pinned")
            info["window"] = _q(det.window, "pinned")
            info["window_loss"] = _q(None if out is None else out.window_loss, shown)
            info["bin_leak"] = _q(None if out is None else out.bin_leak, shown)

        info["mu_bob"] = _q(None if out is None else out.mu_bob, shown)
        info["p_click"] = _q(None if out is None else out.sift_rate, shown)
        # Beside the measurement: the gap is the detector memory's cost.
        info["p_ideal"] = _q(self._pclick(t), "derived")
        info["sift_rate"] = _q(None if out is None else out.sift_rate, shown)
        info["double_rate"] = _q(None if out is None else out.doubles / out.n_slots, shown)
        info["visibility"] = _q(None if out is None else out.visibility, shown)
        info["v_phase"] = _q(None if out is None else out.v_phase, shown)
        info["qber"] = _q(None if out is None else out.qber, shown)
        self._flag_row(info, None if out is None else self._leak(out.sift_rate), shown)

        return info

    def _head(self, protocol, detection="click", stages="closed-form only"):
        """
        The rows an explain() opens with. `attack` stands immediately after
        `security` on every family. _explain_basis spells these out instead,
        its carrier and basis rows standing between protocol and detection.
        """

        return {
            "protocol": protocol,
            "detection": detection,
            "security": type(self.security).__name__,
            "attack": self._attack(),
            "stages": stages,
        }

    def _flag_row(self, info, leak, label):
        if not self._flags().get("backflash"):
            return

        info["backflash_leak"] = _q(leak, label)

    def _explain_basis(self, out=None, decoy=None, rates=None):
        # The closed-form plan, or a completed run's numbers in `out`. `decoy`
        # and `rates` are threaded through so a run does not price twice.
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        t, _, t_label, _ = self._channel()
        mu, nu1, nu2 = mod.decoy.intensities
        dead = self._dead()
        sampled = self.alice is not None
        if out is None:
            eta, gain, qber, y1, e1, q1 = decoy if decoy else self._decoy()
        else:
            eta = t * det.eta
            gain, qber, y1, e1, q1 = rates if rates else self._sampled(out)

        stage = "closed-form only"
        if sampled:
            stage = "pulse train + sifting" if out is not None else "pulse train (unrun)"

        pair = self._announced() == "pair"
        info = {
            "protocol": "basis keying",
            "carrier": _q(
                "polarisation" if isinstance(mod, PolarisationKeying) else "unnamed",
                "pinned",
            ),
            "bases": _q(self._bases(), "pinned"),
            "announce": _q(self._announced(), "pinned"),
            "detection": "click",
            "security": type(sec).__name__,
            "attack": self._attack(),
            "stages": stage,
            "mu": _q(mu, "pinned"),
            "nu1": _q(nu1, "pinned"),
            "nu2": _q(nu2, "pinned"),
            "T": _q(t, t_label),
            "eta": _q(det.eta, "pinned"),
            "dark": _q(det.dark, "pinned"),
            # dark is ONE detector per gate; y0 is the whole receiver per pulse,
            # the plane the decoy forward model works in.
            "y0": _q(self._background(), "derived"),
            # A pair announcement carries its sifting inside the gains, so there
            # this row is Bob's conclusive fraction and multiplies nothing.
            "sift": (_q(_core.sarg_sift(self._misalign()), "derived") if pair else _q(mod.sift, "pinned")),
            "f": _q(sec.f, "pinned"),
            "eta_ab": _q(eta, "derived"),
        }
        self._frame_rows(info)
        info["gain"] = _q(gain, "derived")
        info["qber"] = _q(qber, "derived")
        info["y1"] = _q(y1, "derived")
        info["e1"] = _q(e1, "derived")
        info["q1"] = _q(q1, "derived")
        if pair:
            # The two-photon term SARG04 exists for. Nothing bounds Y2 from a
            # three-intensity set, so none of it is certified here.
            info["q2"] = _q(0.0, "absent")
            # H(Z1|X1), the per-detection amplification cost, and beside it the
            # phase error rate the literature quotes but the rate never reads.
            cost, phase1 = _core.sarg_cost(e1)
            info["cost1"] = _q(cost, "derived")
            info["e_phase1"] = _q(phase1, "derived")

        if self._bases() == 3:
            # decoy_bounds returns ONE e1, not three, so the single-photon term
            # reads the depolarising closed form rather than a complete triple.
            info["depolarising"] = _q(True, "structural")
            info["chi_e1"] = _q(_core.sixstate_holevo(e1), "derived")

        if sampled:
            info["pulses"] = _q(None if out is None else int(out.n_pulses), "derived")
            info["sifted"] = _q(None if out is None else sum(out.sifted), "derived")
            info["sift_rate"] = _q(None if out is None else out.sift_rate, "derived")
            info["doubles"] = _q(None if out is None else int(out.doubles), "derived")

        self._optics(info)
        if dead is not None:
            info["afterpulse"] = _q(dead.afterpulse, "pinned")

        self._flag_row(info, self._leak(self._sifted(gain)), "derived")

        return info

    def _frame_rows(self, info):
        # The polarisation carrier's rows; misalign follows because here it is
        # derived, not pinned. A declared extinction ratio makes it derived on
        # the unnamed carrier too, so the row the rate reads and the row the
        # receiver was given are never one cell.
        mod = self.modulation
        ratio = self._extinct()
        if not isinstance(mod, PolarisationKeying):
            if ratio is None:
                info["misalign"] = _q(self.bob.receiver.misalign, "pinned")

                return

            info["misalign_optics"] = _q(self.bob.receiver.misalign, "pinned")
            info["extinction"] = _q(ratio, "pinned")
            info["misalign"] = _q(self._misalign(), "derived")

            return

        frame = mod.frame
        info["misalign_optics"] = _q(self.bob.receiver.misalign, "pinned")
        if ratio is not None:
            info["extinction"] = _q(ratio, "pinned")
        info["tracking"] = _q(frame.tracking, "pinned")
        if frame.tracking == "free":
            info["drift_rate"] = _q(frame.rate, "pinned")
            info["drift_interval"] = _q(frame.interval, "pinned")
        else:
            info["drift"] = _q(frame.drift, "pinned")

        if frame.dispersion > 0.0:
            length = getattr(self.channel, "length", None)
            info["dispersion"] = _q(frame.dispersion, "pinned")
            info["pulse_width"] = _q(frame.width, "pinned")
            info["dgd"] = _q(impairments.dgd(frame.dispersion, length), "derived")

        info["pol_contrast"] = _q(self._contrast(), "derived")
        info["misalign"] = _q(self._misalign(), "derived")

    def _explain_cow(self, cow=None):
        # `cow` is _run_cow's own tuple; explain() passes none and rebuilds it.
        mod = self.modulation
        det = self.bob.detector
        sec = self.security
        optics = self.bob.receiver
        t, _, t_label, _ = self._channel()
        data, gain, e_bit, q_z, ceiling = cow if cow else self._cow()
        dead = self._dead()
        info = {
            **self._head("intensity keying"),
            "mu": _q(mod.mu, "pinned"),
            "decoy_frac": _q(mod.decoy_frac, "pinned"),
            "T": _q(t, t_label),
            "eta": _q(det.eta, "pinned"),
            "dark": _q(det.dark, "pinned"),
            # Two gates per bit here too: the pair's slots, not two detectors.
            "y0": _q(self._background(), "derived"),
            "split": _q(optics.split, "pinned"),
            "misalign": _q(optics.misalign, "pinned"),
            "e_phase": _q(sec.e_phase, "pinned"),
            "f": _q(sec.f, "pinned"),
            "t_data": _q(data, "derived"),
            "gain": _q(gain, "derived"),
            "q_z": _q(q_z, "derived"),
            "qber": _q(e_bit, "derived"),
            "ceiling": _q(ceiling, "derived"),
        }
        if mod.extinction is not None:
            # A diagnostic reaching no rate: the share of the emission the check
            # speaks for. e_phase stays a supplied bound.
            covered, certified = _core.cow_monitor(1.0 - 2.0 * optics.misalign, mod.extinction, mod.decoy_frac)
            info["extinction"] = _q(mod.extinction, "pinned")
            info["residual"] = _q(mod.residual, "derived")
            info["monitored"] = _q(covered, "derived")
            info["certified"] = _q(certified, "derived")

        self._optics(info)
        if dead is not None:
            info["afterpulse"] = _q(dead.afterpulse, "pinned")

        self._flag_row(info, self._leak(q_z), "derived")

        return info
