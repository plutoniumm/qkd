from dataclasses import dataclass

from . import _core, std

# THE ASYMPTOTIC KEY RATES OF THE FAMILIES WITH NO COMPONENT TREE. Every other
# shipped family reaches its rate through q.Link, q.Swap or q.PairLink. These
# three do not, each engine's own header stating the component-shaped reason:
# src/pairing.rs wants a SYMMETRIC station and a pairing span q.Swap has no
# field for; src/rrdps.rs has Bob drawing a delay uniformly per packet where
# q.DelayInterferometer carries one fixed delay; src/sdp.rs reads a FOUR-sequence
# source where q.IntensityKeying describes three. Each is an objection to a
# component FIELD, not to a route taking the engine's argument names whole.
#
# NOTHING HERE COMPOSES AN EPSILON. Every number is asymptotic; no block size
# and no failure probability reaches any of it. The finite-key lengths are
# q.security's, and a rate from here is not one of them divided by a block.

# Which published assembly of the privacy amplification per sifted bit an RRDPS
# run charges. src/rrdps.rs orders "resolved" <= "tagged" <= "entropy" at every
# argument, and "collective" tightens the per-packet leakage "tagged" reads. All
# four are exact closed forms over the same (l, mu, nu_th); none is fitted.
COSTS = ("tagged", "resolved", "entropy", "collective")

# Which bound on the COW' phase error a run is priced at. "analytic" is Gao's
# Cauchy-Schwarz closed form and "certified" the semidefinite programme that
# tightens it; the second is at or below the first at every operating point.
BOUNDS = ("certified", "analytic")

# The default central-path tolerance the two certificate branches are asked for,
# matching the engines' own.
RELTOL = 1e-9


@dataclass(frozen=True)
class Family:
    """
    One asymptotic rate this layer exports: the argument shapes its engine
    reads, the unit the rate comes back in, and the engine and paper every row
    was taken from. A shape is `(names, tag)` and is taken whole, no count here
    having a safe default.
    """

    name: str
    units: str
    shapes: tuple
    source: str
    paper: str


# The register. Every row was read off the engine named in `source`.
REGISTER = {
    "pairing": Family(
        name="pairing",
        units="bits per ROUND; pairs already charges for the two rounds a key bit consumes",
        shapes=(
            (("eta", "mu", "dark", "span", "misalign", "f_ec"), "yield"),
            (("eta", "mu", "dark", "span", "misalign", "drift", "clock", "f_ec"), "drift"),
        ),
        source="src/pairing.rs::pairing_rate, its phase error from src/mdi.rs::mdi_yield",
        paper=(
            "Zeng, Zhou, Wu & Ma, Nature Communications 13, 3903 (2022), "
            "arXiv:2201.04300; the drift shape from Xie, Lu, Weng, Cao, Jia, "
            "Bao, Wang, Fu, Yin & Chen, PRX Quantum 3, 020315 (2022), "
            "arXiv:2112.11635, Sec. III"
        ),
    ),
    "rrdps": Family(
        name="rrdps",
        units="bits per PULSE; multiply by l for Takesue's per-packet G/N_em",
        shapes=(
            (("l", "mu", "eta", "dark", "e_mis", "nu_th", "f_ec", "cost"), "hardware"),
            (("l", "p_src", "q", "e_bit", "f_ec"), "refined"),
        ),
        source="src/rrdps.rs::rrdps_rate, its cost from the assembly named in COSTS",
        paper=(
            "Takesue, Sasaki, Tamaki & Koashi, Nature Photonics 9, 827 (2015), "
            "arXiv:1505.07914, Eqs. (2)-(4); the receiver model from Yin, Wang, "
            "Chen, Han, Wang, Guo & Han, Nature Communications 9, 457 (2018), "
            "arXiv:1702.01260, Methods"
        ),
    ),
    "cow-vacuum": Family(
        name="cow-vacuum",
        units="bits per PULSE, two pulses to the sequence",
        shapes=((("mu", "t_b", "eta", "dark", "f_ec", "bound"), "vacuum"),),
        source="src/sdp.rs::sdp_data and sdp_phase, into src/discrete.rs::cow_rate",
        paper=(
            'Seksaria & Prabhakar, "Short reach by theorem: the certifiable key '
            'rate of COW QKD" (2026), Sec. VI D, tightening Gao et al., Opt. '
            "Express 30, 23783 (2022), arXiv:2107.09329"
        ),
    ),
}

# Names belonging to a DIFFERENT protocol this layer holds something adjacent
# to; each says what the two are and where the other one runs.
NEARBY = {
    "cow": (
        "'cow' is the THREE-sequence coherent-one-way protocol and runs on q.Link "
        "with q.IntensityKeying and q.PhaseBound, its e_phase supplied rather than "
        "derived. This layer holds 'cow-vacuum', the four-sequence variant COW', "
        "whose empty |0>|0> sequence is what makes the phase error derivable. "
        "Different protocols; the numbers are not comparable"
    ),
    "cow-prime": (
        "the vacuum-decoy variant is registered as 'cow-vacuum', named for the "
        "empty sequence rather than the prime the papers mark it with. 'cow' is "
        "the three-sequence protocol and runs on q.Link with q.IntensityKeying"
    ),
}


@dataclass(frozen=True)
class KeyRate:
    """
    An asymptotic key rate in the unit its family states -- per ROUND for mode
    pairing, per PULSE for the two click families -- beside the engine's own
    working. NOT a finite-key length: q.security.keylength is where a block
    size and a failure probability reach these families.
    """

    family: str
    rate: float
    rows: tuple
    observed: tuple
    units: str

    @property
    def values(self):
        """
        The engine's own working beside the rate, by its own names.
        """

        return dict(self.rows)

    def explain(self):
        """
        Where every number came from: the arguments as pinned, the engine's
        working as derived, and the engine and paper the assembly was read off.
        """
        spec = describe(self.family)
        out = {
            "family": std.q(self.family, "pinned"),
            "source": std.q(spec.source, "default"),
            "paper": std.q(spec.paper, "default"),
            "units": std.q(self.units, "default"),
        }

        for name, value in self.observed:
            out[name] = std.q(value, "pinned")

        for name, value in self.rows:
            out[name] = std.q(value, "derived")

        out["rate"] = std.q(self.rate, "derived")

        return out


# Deliberately not shared with qkd.security.families, nor KeyRate.values with
# KeyLength.values: the two REGISTERs are different layers, asymptotic rates
# here and composed epsilons there.
def families():
    """
    Every family this module holds an asymptotic rate for, in name order.
    """

    return tuple(sorted(REGISTER))


def describe(family):
    """
    The Family record for `family`, or a refusal listing what is registered.
    """
    if family in NEARBY:
        raise ValueError(NEARBY[family])

    if family not in REGISTER:
        known = ", ".join(families())
        raise ValueError(
            f"no asymptotic rate is registered for '{family}'. Registered: {known}. "
            "These are the families with no component tree; every other shipped "
            "rate is reached from q.Link, q.Swap or q.PairLink"
        )

    return REGISTER[family]


def keyrate(family, reltol=None, **observed):
    """
    The asymptotic key rate of one family with no component tree, as a KeyRate.
    `observed` are the engine's own argument names, and which complete set is
    given selects the shape where a family has two. `reltol` is the tolerance
    the two certificate branches are asked for and is refused everywhere else.

    Clamped at zero by the engines, so a family that cannot distil there reads
    0.0 while explain() still carries the phase error that closed it.
    """
    spec = describe(family)
    seen = set(observed)

    for names, tag in spec.shapes:
        if seen == set(names):
            rate, rows = _measure(family, tag, observed, reltol)

            return KeyRate(
                family,
                rate,
                rows,
                tuple((n, observed[n]) for n in names),
                spec.units,
            )

    shapes = " or ".join("(" + ", ".join(n for n in names) + ")" for names, _t in spec.shapes)
    raise ValueError(
        f"family '{family}' reads {shapes}, and was given ({', '.join(sorted(seen))}). "
        f"The names are the engine's own, in {spec.source}, and a shape is taken "
        "whole because no argument here has a safe default"
    )


def window(family, clock, coherence):
    """
    The maximal pairing interval in ROUNDS that a laser coherence time affords
    at a system repetition rate, `floor(clock * coherence)` with `clock` in Hz
    and `coherence` in seconds. This is the span keyrate() takes.

    Runs for 'pairing' alone.
    """
    _only(family, "pairing", "a pairing interval")

    return _core.pairing_span(clock, coherence)


def optimum(family, l_max, mu, eta, dark, e_mis, f_ec):
    """
    The packet length and photon-number threshold that maximise the rate, as
    `(l, nu_th, rate)` in bits per pulse, searching `l` in 2..=l_max at fixed
    hardware. `(0, 0, 0.0)` where no packet length yields key.

    Runs for 'rrdps' alone, and charges the "tagged" cost. `mu` is an argument
    rather than a swept axis: a caller wanting the joint optimum sweeps it
    outside.
    """
    _only(family, "rrdps", "a free packet length")

    return _core.rrdps_optimum(_whole("l_max", l_max), mu, eta, dark, e_mis, f_ec)


def ladder(family, nu, mu, probs):
    """
    One party's PAIR intensity ladder: the six values a two-round pair intensity
    can take when each round draws from `{0, nu, mu}` with the probabilities
    `probs = (s_0, s_nu, s_mu)`, each beside the probability of drawing it,
    ascending.

    Runs for 'pairing' alone, and it is what does NOT transfer from the decoy
    layer: a decoy bound is written over the three intensities a ROUND carries,
    where the photon number here is Poisson in the SUM of the two slots and
    cannot see four of these six rungs.
    """
    _only(family, "pairing", "a pair intensity over two rounds")

    return _core.pairing_ladder(nu, mu, probs)


def bases(family, probs):
    """
    `(z, x, zero, dropped)`: how one party's pair of rounds is labelled, given
    the same per-round intensity probabilities `(s_0, s_nu, s_mu)`. A pair is Z
    when exactly one slot is empty, X when both are lit at the SAME intensity,
    '0' when both are empty, and otherwise dropped. The four sum to one.

    Runs for 'pairing' alone.
    """
    _only(family, "pairing", "a basis chosen after the announcement")

    return _core.pairing_bases(probs)


def overlaps(family, mu):
    """
    The Gram matrix of the four COW' sequences as a flat row-major 4x4 in the
    order `|0>|0>, |a>|a>, |0>|a>, |a>|0>`, every entry fixed by `mu = |a|^2`
    alone.

    Runs for 'cow-vacuum' alone, and it is the fourth row and column that make
    it that protocol rather than the three-sequence one q.Link runs.
    """
    _only(family, "cow-vacuum", "a fourth source sequence")

    return _core.sdp_gram(mu)


def catalogue():
    """
    Every family, as (name, units, shapes, source); the shape
    reconcile.catalogue() uses.
    """

    return tuple(
        (
            spec.name,
            spec.units,
            tuple(names for names, _t in spec.shapes),
            spec.source,
        )
        for _, spec in sorted(REGISTER.items())
    )


def _only(family, want, what):
    """
    Refuse a family whose engine states nothing of this shape, by name.
    """
    spec = describe(family)

    if family != want:
        raise NotImplementedError(
            f"family '{family}' does not reach one here: {what} is what '{want}' has "
            f"and this family does not, and nothing in {spec.source} states a "
            "quantity of that shape"
        )


def _whole(name, value):
    """
    A count the engine takes as an integer, refused rather than truncated.
    """
    if value != int(value) or value < 1:
        raise ValueError(
            f"{name} must be a whole number of at least 1, got {value}: it counts "
            "pulses or rounds, and truncating a fractional one changes the protocol"
        )

    return int(value)


def _named(name, value, allowed):
    """
    A selector the caller states because no engine field carries it.
    """
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(allowed)}, got {value!r}: the assemblies "
            "ship side by side because the literature quotes all of them, they are "
            "different numbers, and nothing here picks for you"
        )

    return value


def _tolerance(reads, reltol):
    """
    The central-path tolerance a certificate branch runs at, or a refusal where
    the branch reaches no solver.
    """
    if reltol is None:
        return RELTOL

    if not reads:
        raise ValueError(
            "reltol is the tolerance a CERTIFICATE is asked for, and this shape "
            "reaches no solver: it is read by the refined keyrate('rrdps', q=..., p_src=..., ...) "
            "and keyrate('cow-vacuum', bound='certified', ...) alone. Every other "
            "assembly here is a closed form"
        )

    std.positive("reltol", reltol)

    return reltol


def _measure(family, tag, observed, reltol):
    """
    One family's assembly, returning `(rate, rows)` with the engine's working
    under the engine's own names.
    """
    if family == "pairing":
        _tolerance(False, reltol)

        return _pairing(tag, observed)

    if family == "rrdps":
        return _rrdps(tag, observed, reltol)

    return _vacuum(observed, reltol)


def _pairing(tag, vals):
    """
    Zeng's Eq. (7) chain: the announcement probability, the pairing rate, the
    Z-pair share and its bit error rate, the single-photon-pair fraction, and
    the phase error from whichever shape the caller took.
    """
    eta = vals["eta"]
    mu = vals["mu"]
    dark = vals["dark"]
    span = _whole("span", vals["span"])
    click = _core.pairing_click(eta, mu, dark)
    pairs = _core.pairing_pairs(click, span)
    sift, e_z = _core.pairing_sift(eta, mu, dark)
    q11 = _core.pairing_single(eta, mu, dark)
    rows = [("click", click), ("pairs", pairs), ("sift", sift), ("q11", q11)]

    if tag == "drift":
        excursion, e_x = _core.pairing_drift(vals["clock"], span, vals["drift"], vals["misalign"])
        rows.append(("excursion", excursion))
    else:
        # Zeng Eq. (104) takes the phase error from Ma & Razavi at a symmetric
        # station, which is mdi_yield's second return at eta_a = eta_b.
        e_x = _core.mdi_yield(eta, eta, dark, vals["misalign"])[1]

    rows.extend([("e_x", e_x), ("e_z", e_z)])

    return _core.pairing_rate(pairs, sift, q11, e_x, e_z, vals["f_ec"]), tuple(rows)


def _rrdps(tag, vals, reltol):
    """
    Takesue's Eq. (4) divided by the packet length, its privacy-amplification
    cost from the named assembly. The refined shape reads a source's
    odd-photon-number probability and an observed sifted rate instead of the
    receiver model, Matsuura's Eq. (60) being written over those.
    """
    l = _whole("l", vals["l"])
    f_ec = vals["f_ec"]

    if tag == "refined":
        q = vals["q"]
        e_bit = vals["e_bit"]
        cost, witness, tangent, plain, steps, iters = _core.rrdps_polytope(
            l,
            vals["p_src"],
            q,
            _tolerance(True, reltol),
        )
        rows = (
            ("q", q),
            ("cost", cost),
            ("witness", witness),
            ("tangent", tangent),
            ("plain", plain),
            ("steps", steps),
            ("iterations", iters),
            ("tolerance", _core.rrdps_tolerance(cost, f_ec)),
        )

        return _core.rrdps_rate(q, e_bit, cost, f_ec, l), rows

    _tolerance(False, reltol)
    form = _named("cost", vals["cost"], COSTS)
    mu = vals["mu"]
    nu_th = _whole("nu_th", vals["nu_th"])
    q, e_bit = _core.rrdps_counts(l, mu, vals["eta"], vals["dark"], vals["e_mis"])
    src = _core.rrdps_src(l, mu, nu_th)

    if form == "resolved":
        cost = _core.rrdps_gllp(q, l, mu, nu_th)
    elif form == "entropy":
        cost = std.entropy(_core.rrdps_phase(q, l, mu, nu_th))
    elif form == "collective":
        cost = _core.rrdps_tag(q, src, _core.rrdps_collective(l, nu_th))
    else:
        cost = _core.rrdps_tag(q, src, _core.rrdps_leak(l, nu_th))

    rows = (
        ("q", q),
        ("e_bit", e_bit),
        ("e_src", src),
        ("cost", cost),
        ("tolerance", _core.rrdps_tolerance(cost, f_ec)),
    )

    return _core.rrdps_rate(q, e_bit, cost, f_ec, l), rows


def _vacuum(vals, reltol):
    """
    The COW' data line priced at one of the two phase-error bounds. The halving
    is the unit: sdp_data's gain is per two-pulse SEQUENCE and cow_rate returns
    bits per pulse.
    """
    mu = vals["mu"]
    t_b = vals["t_b"]
    eta = vals["eta"]
    dark = vals["dark"]
    kind = _named("bound", vals["bound"], BOUNDS)
    q0, q1 = _core.sdp_gains(mu, t_b, eta, dark)
    rows = [("honest", _core.sdp_honest(mu, t_b, eta, dark))]

    if kind == "certified":
        e_phase, gap, iters = _core.sdp_phase(mu, q0, q1, _tolerance(True, reltol))
        rows.extend([("gap", gap), ("iterations", iters)])
    else:
        _tolerance(False, reltol)
        e_phase = _core.sdp_analytic(mu, q0, q1)

    gain, e_bit = _core.sdp_data(mu, t_b, eta, dark)
    rows.extend([("e_phase", e_phase), ("gain", gain), ("e_bit", e_bit)])

    # Both bounds return e_phase raw and uncapped; at or above 1/2 the protocol
    # aborts, and cow_rate's own domain would refuse rather than report it.
    if e_phase >= 0.5:
        return 0.0, tuple(rows)

    return 0.5 * _core.cow_rate(gain, e_bit, e_phase, vals["f_ec"]), tuple(rows)
