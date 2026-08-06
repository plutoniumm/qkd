import math
from dataclasses import dataclass

from . import _core, std

# A REGISTRY, not a derivation. No two shipped families mean the same thing by
# "eps": q.FiniteSize.eps is PER TERM (qkd/link.py hands it to eps_pe,
# eps_smooth and eps_pa, so the composed secrecy parameter is 3*eps),
# bb84_length's eps_sec IS the total, mdi_length's is the total PER ANNOUNCED
# BELL STATE, sixstate_length nests eps_ec inside eps, and rrdps_finite's d is a
# MAXIMUM over five terms, two of them hash lengths in bits.
#
# Never converts an epsilon to bits: the leftover-hash charge is qkd.reconcile's,
# and a hashing term charged again here double-counts privacy amplification.

# "eps" is a failure probability in (0, 1), matching src/std.rs::check_eps.
# "bits" is a hash length s standing for 2**-s, the only form Takesue's eta_x and
# eta_z take. "count" is a whole number of repetitions of a whole analysis, as
# Curty's sum over announced Bell states is.
KINDS = ("eps", "bits", "count")

# Which half of the composable parameter a term is charged to. "structure" is
# not charged at all -- it multiplies.
ROLES = ("secrecy", "correctness", "authentication", "structure")

# Privacy-amplification term names. Accepted only by the family whose engine
# takes one as a separate argument.
HASHING = ("eps_pa", "eps_amp", "eps_hash")


@dataclass(frozen=True)
class Term:
    """
    One entry in a family's budget. The kind is load-bearing: a hash length and
    a failure probability are both floats and must never be added.
    """

    name: str
    kind: str
    role: str
    note: str = ""

    def check(self, value):
        """
        The value as a float, checked against this term's kind.
        """
        if self.kind == "eps":
            std.open_unit(self.name, value)
        elif self.kind == "bits":
            std.finite(self.name, value)

            std.positive(self.name, value)
        elif value != int(value) or value < 1:
            raise ValueError(
                f"{self.name} must be a whole number of at least 1, got "
                f"{value}: it counts repetitions of the whole analysis, not a "
                "probability"
            )

        return float(value)

    def eps(self, value):
        """
        The failure probability this term stands for: itself for kind "eps",
        2**-value for a hash length in bits, and the value unchanged for a
        count, which is never charged.
        """
        if self.kind == "bits":
            return 2.0**-value

        return value


# The token Ledger.secrecy dispatches on; Family.rule is a sentence and checks
# nothing. A form reaching no arm there refuses rather than falling through.
FORMS = ("sum", "per-state", "nested", "weighted", "max", "outer-max", "root")


@dataclass(frozen=True)
class Family:
    """
    One shipped finite-size analysis and the budget it declares. `rule` is the
    composition as a sentence, reported on every number this module returns;
    `form` is the same composition as a FORMS token. `shares` is the number of
    equal parts the engine divides its secrecy parameter into, or None.
    `unmodelled` names the terms the family does not produce, which is what
    makes a total refuse.
    """

    name: str
    rule: str
    form: str
    terms: tuple
    shares: int | None
    unmodelled: tuple
    source: str
    paper: str

    def spec(self, name):
        """
        The Term called `name`, or a refusal listing what this family accepts.
        """
        for term in self.terms:
            if term.name == name:
                return term

        taken = ", ".join(t.name for t in self.terms)
        if name in HASHING:
            raise ValueError(
                f"'{name}' is not a term of family '{self.name}': {self.source} "
                "already charges privacy amplification inside its secrecy "
                "parameter, so adding it again double-counts the hashing. "
                f"Accepted terms: {taken}"
            )

        raise ValueError(
            f"'{name}' is not a term of family '{self.name}'. Accepted terms: "
            f"{taken}. Term names follow the engine's own argument names, in "
            f"{self.source}"
        )


# The one term no engine produces, shared by every family.
_AUTH = Term(
    "eps_auth",
    "eps",
    "authentication",
    "declared by the caller; no engine produces one",
)

# No matching _COR: only three of the eight correctness rows below are one Term
# -- mdi's reads PER announced Bell state, cow-vacuum's states the additivity,
# and four declare an eps_ec or a hash length instead -- so hoisting costs +1
# line and reads as though it covered all eight.

# Every row read off the engine named in `source`; nothing inferred from shape.
REGISTER = {
    "cv": Family(
        name="cv",
        rule="sum (union bound over three independent failure events)",
        form="sum",
        terms=(
            Term(
                "eps_pe",
                "eps",
                "secrecy",
                "the channel estimate leaves its confidence interval",
            ),
            Term("eps_smooth", "eps", "secrecy", "the smoothing of the min-entropy"),
            Term("eps_pa", "eps", "secrecy", "the leftover hash lemma"),
            _AUTH,
        ),
        shares=None,
        unmodelled=("eps_cor",),
        source="src/keyrate.rs::cv_finite, via qkd/link.py",
        paper="Leverrier, Grosshans & Grangier, PRA 81, 062343 (2010)",
    ),
    "bb84": Family(
        name="bb84",
        rule="sum (secrecy plus correctness)",
        form="sum",
        terms=(
            Term("eps_sec", "eps", "secrecy", "split 21 ways inside the engine"),
            Term("eps_cor", "eps", "correctness", "the keys differ undetected"),
            _AUTH,
        ),
        shares=21,
        unmodelled=(),
        source="src/discrete.rs::bb84_length",
        paper="Lim, Curty, Walenta, Xu & Zbinden, Phys. Rev. A 89, 022307 (2014)",
    ),
    "sixstate": Family(
        name="sixstate",
        rule="nested (eps_ec sits inside eps; the rest splits three ways)",
        form="nested",
        terms=(
            Term("eps", "eps", "structure", "the composed deviation, declared whole"),
            Term(
                "eps_ec",
                "eps",
                "correctness",
                "error correction fails, strictly inside eps",
            ),
            _AUTH,
        ),
        shares=3,
        unmodelled=(),
        source="src/sixstate.rs::sixstate_length",
        paper="Scarani & Renner, Phys. Rev. Lett. 100, 200501 (2008), arXiv:0708.0709",
    ),
    "sarg": Family(
        name="sarg",
        rule="sum (secrecy plus correctness)",
        form="sum",
        terms=(
            Term("eps_sec", "eps", "secrecy", "split 18 ways inside the engine"),
            Term("eps_cor", "eps", "correctness", "the keys differ undetected"),
            _AUTH,
        ),
        shares=18,
        unmodelled=(),
        source="src/sarg.rs::sarg_length, sarg_counts, sarg_errors and sarg_eps",
        paper=(
            "Nian, Nie, Zhang & Lu, Commun. Theor. Phys. 76, 065101 (2024), "
            "Eq. (5), which has NO arXiv version and was read as the publisher "
            "PDF, over the one-decoy estimators of Rusca, Boaron, Grunenfelder, "
            "Martin & Zbinden, Appl. Phys. Lett. 112, 171104 (2018), "
            "arXiv:1801.03443"
        ),
    ),
    "mdi": Family(
        name="mdi",
        rule="per-state sum (each announced Bell state carries its own budget)",
        form="per-state",
        terms=(
            Term("eps_sec", "eps", "secrecy", "PER announced Bell state, split 266 ways"),
            Term("eps_cor", "eps", "correctness", "PER announced Bell state"),
            Term(
                "states",
                "count",
                "structure",
                "how many Bell states the relay identifies",
            ),
            _AUTH,
        ),
        shares=266,
        unmodelled=(),
        source="src/mdi.rs::mdi_length and mdi_eps",
        paper=("Curty, Xu, Cui, Lim, Tamaki & Lo, Nature Communications 5, 3732 " "(2014), arXiv:1307.1081"),
    ),
    "pairing": Family(
        name="pairing",
        rule="sum (secrecy plus correctness)",
        form="sum",
        terms=(
            Term("eps_sec", "eps", "secrecy", "split 24 ways inside the engine"),
            Term("eps_cor", "eps", "correctness", "the keys differ undetected"),
            _AUTH,
        ),
        shares=24,
        unmodelled=(),
        source="src/pairing.rs::pairing_length and pairing_eps",
        paper=("Xie, Lu, Weng, Cao, Jia, Bao, Wang, Fu, Yin & Chen, PRX Quantum " "3, 020315 (2022), arXiv:2112.11635"),
    ),
    "rrdps": Family(
        name="rrdps",
        rule="max (Takesue's two branches, with a square root inside one)",
        form="max",
        terms=(
            Term(
                "eps_1",
                "eps",
                "secrecy",
                "the double-click branch, Bob's photon number",
            ),
            Term("eps_2", "eps", "secrecy", "Alice's source, a binomial Eve is not in"),
            Term("eps_3", "eps", "secrecy", "the round-robin bound, a protocol constant"),
            Term("s_x", "bits", "secrecy", "hash length; eta_x = 2**-s_x"),
            Term("s_z", "bits", "secrecy", "hash length; eta_z = 2**-s_z"),
            _AUTH,
        ),
        shares=None,
        unmodelled=("eps_cor",),
        source="src/rrdps.rs::rrdps_secpar and rrdps_split",
        paper=("Takesue, Sasaki, Tamaki & Koashi, Nature Photonics 9, 827 (2015), " "arXiv:1505.07914"),
    ),
    "pair": Family(
        name="pair",
        rule="sum (one correctness term and two secrecy)",
        form="sum",
        terms=(
            Term("eps_pe", "eps", "secrecy", "parameter estimation, their Eq. (57)"),
            Term("eps_pa", "eps", "secrecy", "privacy amplification, their Eq. (58)"),
            Term("eps_ec", "eps", "correctness", "error verification, their Eq. (56)"),
            Term("t", "bits", "correctness", "hash length; eps_ec = 2**-t"),
            _AUTH,
        ),
        shares=None,
        unmodelled=(),
        source="src/pairs.rs::pair_secpar and pair_length",
        paper="Tomamichel & Leverrier, Quantum 1, 14 (2017), arXiv:1506.08458",
    ),
    "dmcs": Family(
        name="dmcs",
        rule="outer sum over a maximum: eps_ec + max(eps_pa/2 + eps_bar, eps_et + eps_at)",
        form="outer-max",
        terms=(
            Term("eps_ec", "eps", "correctness", "error verification"),
            Term("eps_pa", "eps", "secrecy", "privacy amplification, halved inside the max"),
            Term("eps_bar", "eps", "secrecy", "the smoothing parameter"),
            Term("eps_et", "eps", "secrecy", "the energy test, Theorem 3"),
            Term("eps_at", "eps", "secrecy", "the acceptance test, Theorem 4"),
            _AUTH,
        ),
        shares=None,
        unmodelled=(),
        source="src/dmcs.rs::dm_secpar",
        paper=(
            "Kanitschar, George, Lin, Upadhyaya & Lutkenhaus, PRX Quantum 4, "
            "040306 (2023), arXiv:2301.08686, Theorem 6"
        ),
    ),
    "cow-vacuum": Family(
        name="cow-vacuum",
        rule="sum, at Li's weights 2/1/6/1 on one common share",
        form="weighted",
        terms=(
            Term("eps", "eps", "secrecy", "smoothing; enters twice"),
            Term("eps_0", "eps", "secrecy", "the leftover-hash deviation"),
            Term("eps_1", "eps", "secrecy", "one per Kato estimation, and there are six"),
            Term("eps_2", "eps", "secrecy", "the expected-to-observed conversion"),
            Term(
                "eps_cor",
                "eps",
                "correctness",
                "separate and additive: eps_s = eps_cor + eps_sec",
            ),
            _AUTH,
        ),
        shares=10,
        unmodelled=(),
        source="src/sdp.rs::sdp_eps, sdp_kato, sdp_interval, sdp_sample and sdp_length",
        paper=("Li, Cao, Xie, Yin & Chen, Phys. Rev. Research 6, 013022 (2024), " "arXiv:2309.16136, Eq. (23)"),
    ),
    "dps": Family(
        name="dps",
        rule="root over a weighted sum: 2**-zeta_ec + sqrt(2)*sqrt(3*eps_1 + 3*eps_2 + 2**-zeta)",
        form="root",
        terms=(
            Term("eps_1", "eps", "secrecy", "Kato on the three-photon sum; charged THREE times"),
            Term("eps_2", "eps", "secrecy", "the source bounds; charged THREE times"),
            Term("zeta", "bits", "secrecy", "hash length; 2**-zeta sits under the root"),
            Term(
                "zeta_ec",
                "bits",
                "secrecy",
                "error-verification hash length; Eq. (50) adds 2**-zeta_ec OUTSIDE the root",
            ),
            _AUTH,
        ),
        shares=None,
        unmodelled=("eps_cor",),
        source="src/click.rs::dps_finite",
        paper=(
            "Mizutani, Takeuchi & Tamaki, Phys. Rev. Research 5, 023132 (2023), "
            "arXiv:2301.09844, Eq. (50); the length is Corollary 1's Eq. (24) on "
            "Theorem 3's Eq. (42)"
        ),
    ),
    "cvmdi": Family(
        name="cvmdi",
        rule="uncomposed (POP17 add none of these together)",
        form="sum",
        terms=(
            Term("eps_smooth", "eps", "secrecy", "the smoothing parameter in Delta(n)"),
            Term("eps_pa", "eps", "secrecy", "privacy amplification, inside Delta(n)"),
            _AUTH,
        ),
        shares=None,
        # eps_pe is unmodelled because POP17 state parameter estimation as a
        # CONFIDENCE COEFFICIENT in standard deviations rather than as a
        # probability, and a sigma is not a Term of any kind here.
        unmodelled=("eps_cor", "eps_pe"),
        source="src/relay.rs::cvmdi_finite",
        paper=("Papanastasiou, Ottaviani & Pirandola, Phys. Rev. A 96, 042332 " "(2017), arXiv:1707.04599"),
    ),
}

ABSENT = {
    "cv": (
        "src/keyrate.rs::cv_finite takes eps_pe, eps_smooth and eps_pa and no "
        "correctness parameter, and the only reconciliation failure quantity in "
        "the tree is the FRAME ERROR RATE (reconcile.Code.fer, "
        "q.FiniteSize.fer), which Link._fer scales the rate by rather than a "
        "probability that the two keys differ undetected. Read .secrecy, and "
        "declare eps_cor yourself if your reconciliation layer certifies one"
    ),
    "rrdps": (
        "src/rrdps.rs models no correctness parameter: e_bit reaches the key "
        "length only through the error-correction leakage, a count of bits "
        "Alice actually sent that carries no confidence interval. Takesue's d "
        "is the secrecy branch alone. Read .secrecy"
    ),
    "cvmdi": (
        "src/relay.rs::cvmdi_finite models neither a correctness parameter nor "
        "a parameter-estimation probability: eps_smooth and eps_pa enter the "
        "smooth-min-entropy penalty Delta(n) and nothing else, and POP17 state "
        "the four confidence intervals as a coefficient in STANDARD DEVIATIONS, "
        "which q.GaussianBlock carries as sigma and which is not a Term of any "
        "kind here. Read .secrecy"
    ),
    "dps": (
        "src/click.rs::dps_finite states ONE composed parameter, Eq. (50), and "
        "it charges the error-verification hash INSIDE it: 2**-zeta_ec is added "
        "outside the square root, not declared beside eps_sec the way "
        "bb84_length's eps_cor is. Read .secrecy, which is Eq. (50) whole"
    ),
}


@dataclass(frozen=True)
class Ledger:
    """
    A stated budget for one family, with the composition rule attached. Terms
    are (name, value) pairs in declaration order.

    .secrecy and .correctness are the two halves of the composable parameter and
    .total their sum plus any declared authentication term. Each refuses rather
    than defaulting a term the family does not model, naming the file that would
    have to produce it.
    """

    family: str
    terms: tuple

    @property
    def spec(self):
        """
        The Family record this ledger was composed against.
        """

        return REGISTER[self.family]

    @property
    def rule(self):
        """
        The composition rule, as a sentence.
        """

        return self.spec.rule

    @property
    def values(self):
        """
        The declared terms as a name-to-value mapping.
        """

        return dict(self.terms)

    @property
    def secrecy(self):
        """
        The composed secrecy parameter, by this family's own rule.
        """
        # The composition arms across this ladder, .correctness, .shares() and
        # keylength stay hand-written. Hanging them off Family as callables
        # measured at ~+60 lines and is DECLINED, {family: fn} dispatch at ~+50
        # (which reintroduces the .get(family, default) these ladders exist to
        # refuse): the registry already IS data, only the arithmetic is code.
        vals = self.values
        form = self.spec.form
        if form == "max":
            return _core.rrdps_secpar(
                vals["eps_1"],
                vals["eps_2"],
                vals["eps_3"],
                vals["s_x"],
                vals["s_z"],
            )

        if form == "nested":
            return vals["eps"] - vals["eps_ec"]

        if form == "weighted":
            # Li Eq. (23): eps_sec = 2*eps + eps_0 + 6*eps_1 + eps_2. Each weight
            # counts the BOUNDS that term is spent on -- six Kato estimations,
            # one per bounded gain -- so a plain sum of the four understates it.
            return 2.0 * vals["eps"] + vals["eps_0"] + 6.0 * vals["eps_1"] + vals["eps_2"]

        if form == "outer-max":
            # Kanitschar Theorem 6 adds eps_ec OUTSIDE a maximum over the two
            # branches, so the secrecy half is that maximum. Taken off the engine
            # and stripped of eps_ec, keeping one spelling of the composition.
            whole = _core.dm_secpar(
                vals["eps_ec"],
                vals["eps_pa"],
                vals["eps_bar"],
                vals["eps_et"],
                vals["eps_at"],
            )

            return whole - vals["eps_ec"]

        if form == "root":
            # MTT23 Eq. (50). zeta and zeta_ec are hash lengths in BITS, and
            # eps_1 and eps_2 are each charged THREE times under the root, so
            # neither a sum of the five nor a maximum over them is this number.
            inner = 3.0 * vals["eps_1"] + 3.0 * vals["eps_2"] + 2.0 ** -vals["zeta"]

            return 2.0 ** -vals["zeta_ec"] + math.sqrt(2.0) * math.sqrt(inner)

        odd = [t.name for t in self.spec.terms if t.role == "structure" or (t.role == "secrecy" and t.kind != "eps")]
        if form not in ("sum", "per-state") or (form == "sum" and odd):
            raise NotImplementedError(
                f"family '{self.family}' is registered as composing '{form}' "
                f"and declares {', '.join(t.name for t in self.spec.terms)}. A "
                "plain sum adds probabilities and multiplies by nothing, so a "
                "hash length in bits, a structure term, or a form no arm above "
                "states reaches no composition here"
            )

        total = sum(v for k, v in self.terms if self.spec.spec(k).role == "secrecy")
        if form == "per-state":
            return vals["states"] * total

        return total

    @property
    def correctness(self):
        """
        The composed correctness parameter, or a refusal naming why the family
        produces none.
        """
        if "eps_cor" in self.spec.unmodelled:
            raise ValueError(f"family '{self.family}' models no correctness parameter: " f"{ABSENT[self.family]}")

        vals = self.values
        if self.family == "mdi":
            return vals["states"] * vals["eps_cor"]

        return vals.get("eps_cor", vals.get("eps_ec"))

    @property
    def authentication(self):
        """
        The declared authentication term, or None. Never defaulted to zero: an
        undeclared classical channel is out of scope rather than free.
        """

        return self.values.get("eps_auth")

    @property
    def total(self):
        """
        The protocol-level parameter: secrecy plus correctness plus any declared
        authentication term. Refuses where the family models no correctness
        parameter.
        """
        out = self.secrecy + self.correctness
        auth = self.authentication

        return out + auth if auth is not None else out

    def shares(self):
        """
        (each, count): the failure probability every internal bound is run at
        and how many of them there are, by the engine's own divisor.
        """
        count = self.spec.shares
        if count is None:
            raise ValueError(_NOSHARE[self.family])

        vals = self.values
        if self.family == "mdi":
            return _core.mdi_eps(vals["eps_sec"]), count

        if self.family == "pairing":
            return _core.pairing_eps(vals["eps_sec"]), count

        if self.family == "sixstate":
            return (vals["eps"] - vals["eps_ec"]) / count, count

        if self.family == "cow-vacuum":
            # Off the TOTAL .secrecy recomposes: this family declares no eps_sec.
            return _core.sdp_eps(self.secrecy), count

        if self.family == "sarg":
            return _core.sarg_eps(vals["eps_sec"]), count

        if self.family == "bb84":
            return vals["eps_sec"] / count, count

        # Keyed on the family, not spec.form: bb84, sarg and pairing all compose
        # as "sum" and read three different engine divisors.
        raise NotImplementedError(
            f"family '{self.family}' declares {count} equal shares and no arm "
            f"here reads that divisor off {self.spec.source}. A share count "
            "reaches one named engine; falling through would report one "
            "engine's split under another's name"
        )

    def meets(self, target):
        """
        Whether the composed protocol parameter is at or under `target`.
        Propagates the refusal where no total composes.
        """
        std.open_unit("target", target)

        return self.total <= target

    def explain(self):
        """
        Every declared term beside its kind, its role and the probability it
        stands for, plus the composed halves and the rule that made them.
        Unmodelled halves come back as the refusal text, never as a number.
        """
        rows = {
            "family": std.q(self.family, "pinned"),
            "rule": std.q(self.spec.rule, "default"),
            "source": std.q(self.spec.source, "default"),
            "paper": std.q(self.spec.paper, "default"),
        }
        for name, value in self.terms:
            term = self.spec.spec(name)
            rows[name] = std.q(value, f"pinned ({term.kind}, {term.role})")
            if term.kind == "bits":
                rows[f"{name}_eps"] = std.q(term.eps(value), "derived")

        for half in ("secrecy", "correctness", "total"):
            try:
                rows[half] = std.q(getattr(self, half), "derived")
            except ValueError as exc:
                rows[half] = std.q(str(exc), "refused")

        try:
            each, count = self.shares()
            rows["per_bound"] = std.q(each, f"derived ({count} equal shares)")
        except ValueError as exc:
            rows["per_bound"] = std.q(str(exc), "refused")

        if self.authentication is None:
            rows["eps_auth"] = std.q(None, "not declared")

        return rows


_NOSHARE = {
    "cv": (
        "family 'cv' has no share count: eps_pe, eps_smooth and eps_pa are "
        "three independent arguments of src/keyrate.rs::cv_finite, not one "
        "number divided three ways. qkd/link.py passes q.FiniteSize.eps for all "
        "three, making the composed secrecy parameter 3*eps -- .secrecy is that "
        "number"
    ),
    "rrdps": (
        "family 'rrdps' divides its budget into no equal shares: src/rrdps.rs "
        "composes d <= max(eps_1, eta_z + sqrt2*sqrt(eps_2 + eps_3 + eta_x)), a "
        "maximum over two branches with a square root inside one, and two of "
        "the five terms are hash lengths in bits. Use allocate('rrdps', d), "
        "which calls Takesue's own assignment through _core.rrdps_split"
    ),
    "pair": (
        "family 'pair' has no share count: src/pairs.rs::pair_secpar takes "
        "eps_pe, eps_pa and eps_ec as three separate arguments, and t is a hash "
        "length in BITS standing for eps_ec = 2**-t. Read .secrecy and "
        ".correctness, the two halves Tomamichel & Leverrier state"
    ),
    "dmcs": (
        "family 'dmcs' divides its budget into no equal shares: Kanitschar "
        "Theorem 6 is eps_ec + max(eps_pa/2 + eps_bar, eps_et + eps_at), so "
        "eps_pa is HALVED, the two branches compose as a maximum, and eps_ec "
        "sits outside it -- five equal shares of a target reach three tenths of "
        "it. Read .secrecy for that maximum, or _core.dm_secpar for the whole "
        "composition"
    ),
    "cvmdi": (
        "family 'cvmdi' has no share count and composes no total: "
        "src/relay.rs::cvmdi_finite reads eps_smooth and eps_pa inside one "
        "smooth-min-entropy penalty Delta(n) and divides neither, and POP17 "
        "state parameter estimation as a confidence COEFFICIENT in standard "
        "deviations, which q.GaussianBlock carries as sigma and which is not a "
        "probability. Read .secrecy"
    ),
    "dps": (
        "family 'dps' divides its budget into no equal shares: MTT23 Eq. (50) "
        "is 2**-zeta_ec + sqrt(2)*sqrt(3*eps_1 + 3*eps_2 + 2**-zeta), where two "
        "of the four terms are hash lengths in BITS, two are charged three "
        "times each, and one whole branch sits under a square root. Declare the "
        "four with compose('dps', ...) and check them with Ledger.meets(target)"
    ),
}


def families():
    """
    Every family this module holds a budget for, in name order.
    """

    return tuple(sorted(REGISTER))


def describe(family):
    """
    The Family record for `family`, or a refusal listing what is registered.
    """
    if family not in REGISTER:
        known = ", ".join(families())
        raise ValueError(
            f"no epsilon budget is registered for '{family}'. Registered: "
            f"{known}. A family is registered when an engine in src/ declares "
            "security parameters and states how they compose"
        )

    return REGISTER[family]


def compose(family, **terms):
    """
    VERIFY a declared budget: check every term against its kind and return a
    Ledger that composes them by the family's own rule.
    """
    spec = describe(family)
    need = [t.name for t in spec.terms if t.role != "authentication"]
    if family == "cv":
        need = ["eps_pe", "eps_smooth", "eps_pa"]

    absent = [n for n in need if n not in terms]
    if absent:
        raise ValueError(
            f"family '{family}' needs {', '.join(need)} and is missing "
            f"{', '.join(absent)}. No term has a safe default: see "
            f"{spec.source} for what each one is"
        )

    rows = []
    for name in list(need) + [n for n in terms if n not in need]:
        rows.append((name, spec.spec(name).check(terms[name])))

    return Ledger(family, tuple(rows))


def allocate(family, target):
    """
    COMPUTE a budget from a target: return the terms reaching `target`, by the
    family's own published assignment. Runs only where an engine ships an
    assignment rather than a divisor, and refuses by name everywhere else.
    """
    spec = describe(family)
    std.open_unit("target", target)

    if family != "rrdps":
        has = (
            f"the division of its SECRECY parameter alone into {spec.shares} "
            "equal shares, which Ledger.shares() reports"
            if spec.shares is not None
            else f"no divisor at all: {_NOSHARE[family]}"
        )
        raise NotImplementedError(
            "qkd ships no published assignment of a protocol target across "
            f"family '{family}''s terms. What {spec.source} publishes is {has}; "
            "how much of a target goes to secrecy and how much to correctness "
            f"is the caller's declaration. Use compose('{family}', ...) and "
            "check it with Ledger.meets(target). allocate() runs for 'rrdps' "
            "alone, where Takesue fixes the assignment himself and "
            "_core.rrdps_split implements it"
        )

    eps1, eps2, eps3, s_x, s_z = _core.rrdps_split(target)

    return compose(
        "rrdps",
        eps_1=eps1,
        eps_2=eps2,
        eps_3=eps3,
        s_x=s_x,
        s_z=s_z,
    )


def authenticate(ledger, messages, eps):
    """
    Add the classical channel's forgery probability to a budget: the union
    bound `messages * eps` over a round's authenticated messages, `eps` being
    q.Authentication.eps, the PER-MESSAGE forgery probability. Returns a new
    Ledger. reconcile.net_length subtracts the authentication bill in BITS
    instead and never touches the security parameter.
    """
    std.positive("messages", messages)

    std.open_unit("eps", eps)

    if messages != int(messages):
        raise ValueError(f"messages must be a whole number, got {messages}: it counts one round's authenticated ones")

    total = messages * eps
    if not 0.0 < total < 1.0:
        raise ValueError(
            f"{int(messages)} messages at eps = {eps:g} compose to {total:g}, "
            "which is not a probability: the union bound has saturated. Use a "
            "smaller per-message eps, or fewer messages"
        )

    rows = [(n, v) for n, v in ledger.terms if n != "eps_auth"]

    return Ledger(ledger.family, tuple(rows) + (("eps_auth", total),))


# The finite-key lengths this layer exports, and the observations each reads.
# RRDPS carries TWO shapes and they are different claims: from a block the engine
# makes its own split of the target and reports the thresholds it derived; from
# thresholds already certified it is handed the ledger's own s_x and s_z, which
# are HASH LENGTHS IN BITS rather than probabilities.
LENGTHS = {
    "pair": (
        (
            ("m", "k", "nu", "delta", "cbar", "leak"),
            ("length",),
            "bits over a block of m SIFTED rounds, at deterministic detection",
            "pair",
        ),
    ),
    "sixstate": (
        (
            ("n", "m_test", "e_key", "e_test", "f_ec"),
            ("length", "hxe", "e_key_used", "e_test_used", "delta"),
            "bits over the raw key of n sifted single-photon signals",
            "sixstate",
        ),
    ),
    "pairing": (
        (
            ("s0", "s11", "phi", "n_z", "e_z", "f_ec"),
            ("length",),
            "bits over the block, per sifted Z-PAIR rather than per round",
            "pairing",
        ),
    ),
    "sarg": (
        (
            ("mu", "nu", "probs", "counts", "errors", "n_key", "e_key", "f_ec"),
            ("length", "s0", "s1", "s0_hi", "v1", "eps_each"),
            "bits over the block the ONE-DECOY record was counted from",
            "record",
        ),
        (
            ("s0", "s1", "v1", "n_key", "e_key", "f_ec"),
            ("length",),
            "bits over a block whose three photon-number counts are already certified",
            "certified",
        ),
    ),
    "cow-vacuum": (
        (
            ("n_z", "e_z", "phase", "f_ec"),
            ("length",),
            "bits over the block, per sifted Z round",
            "vacuum",
        ),
    ),
    "dmcs": (
        (
            ("n_total", "n_key", "entropy", "alphabet", "w", "leak", "source"),
            ("length", "raw", "aep", "price"),
            "bits over the whole block of n_total signals, key and test rounds together",
            "relaxed",
        ),
    ),
    "rrdps": (
        (
            ("n_em", "n_sift", "n_double", "l", "mu", "nu_th", "e_bit", "f_ec", "p_two"),
            ("length", "secpar", "tagged", "faults", "phase"),
            "bits over the block of n_em emitted packets",
            "block",
        ),
        (
            ("n_sift", "tagged", "faults", "phase", "e_bit", "f_ec"),
            ("length",),
            "bits over a block whose three thresholds are already certified",
            "thresholds",
        ),
    ),
}

# Families whose finite-key length already has a component tree. keylength()
# refuses them rather than becoming a second route to a number q.Link and q.Swap
# already report, which would let the two disagree.
COMPONENT = {
    "bb84": "q.Link with q.SplittingAttack(block=q.KeyBlock(...))",
    "mdi": "q.Swap with q.TestBasisBound(block=q.RelayBlock(...))",
    "cv": "q.Link with q.FiniteSize(...)",
}

# The one family whose registered budget reaches no length at all. Not a
# duplication refusal: nothing in the tree computes this one.
NOLENGTH = {
    "cvmdi": (
        "src/relay.rs::cvmdi_finite returns a finite-size RATE per channel use, "
        "not a length in bits over a block, and it already runs through q.Swap "
        "with q.TwoModeBound(block=q.GaussianBlock(...)). Dividing that rate by "
        "n would report a per-use number as a block total, and multiplying it "
        "by n would report a rate POP17 never integrated as one"
    ),
}

# The one length exported from a MODULE rather than from a component tree.
ELSEWHERE = {
    "dps": (
        "qkd.dps.finite(..., detector=...), which takes the receiver as a "
        "keyword argument with NO DEFAULT. A ledger declares failure "
        "probabilities and describes no receiver, so reaching the length from "
        "one would supply that argument here -- and src/click.rs::dps_finite "
        "refuses a threshold click record because it moves the bound in the "
        "INSECURE direction. compose('dps', ...) declares the epsilon shape; "
        "the length is stated where the receiver is"
    ),
}

# The families reaching the two Scarani & Renner estimators below.
SINGLE = ("sixstate",)


@dataclass(frozen=True)
class KeyLength:
    """
    A finite-key length in BITS over a whole block, beside the budget it was
    taken at. NOT a rate: the per-shot unit differs across these families -- a
    packet for rrdps, a sifted Z-pair for pairing, a sifted signal for sixstate
    -- so the caller divides by the count its own hardware emitted.
    """

    family: str
    bits: float
    ledger: "Ledger"
    rows: tuple
    observed: tuple
    units: str

    @property
    def values(self):
        """
        The engine's own working beside the length, by its own argument names.
        """

        return dict(self.rows)

    def explain(self):
        """
        The observations as pinned, the engine's working as derived, and the
        budget rows the ledger already labels.
        """
        out = {"family": std.q(self.family, "pinned")}
        out["source"] = std.q(describe(self.family).source, "default")
        out["units"] = std.q(self.units, "default")

        for name, value in self.observed:
            out[name] = std.q(value, "pinned")

        for name, value in self.rows:
            out[name] = std.q(value, "derived")

        for name, row in self.ledger.explain().items():
            out[f"budget_{name}"] = row

        return out


def keylength(ledger, **observed):
    """
    The finite-key length in BITS for a family whose budget is already declared.
    `ledger` fixes the security parameters and selects the engine; `observed`
    are that engine's own argument names, and which complete set is given
    selects the shape where a family has two.

    Runs for the LENGTHS families -- the lengths with no component tree. Every
    other family is refused by name.
    """
    spec = describe(ledger.family)
    family = ledger.family

    if family in ELSEWHERE:
        raise NotImplementedError(f"family '{family}' states its finite-key length at " f"{ELSEWHERE[family]}")

    if family in NOLENGTH:
        raise NotImplementedError(f"family '{family}' reaches no finite-key length here: " f"{NOLENGTH[family]}")

    if family in COMPONENT:
        raise NotImplementedError(
            f"family '{family}' already reports a finite-key length through "
            f"{COMPONENT[family]}, which reads the hardware rather than taking "
            f"counts. {spec.source} needs a described link, not a declared "
            "budget alone"
        )

    seen = set(observed)
    for names, back, units, tag in LENGTHS[family]:
        if seen == set(names):
            out = _measure(tag, ledger, [observed[n] for n in names])

            return KeyLength(
                family,
                out[0],
                ledger,
                tuple(zip(back, out)),
                tuple((n, observed[n]) for n in names),
                units,
            )

    shapes = " or ".join("(" + ", ".join(n for n in names) + ")" for names, _b, _u, _t in LENGTHS[family])
    raise ValueError(
        f"family '{family}' reads {shapes}, and was given "
        f"({', '.join(sorted(seen))}). The names are the engine's own, which "
        f"are in {spec.source}, and a shape is taken whole because no count "
        "here has a safe default"
    )


def _measure(tag, ledger, args):
    """
    The engine's tuple, scalars boxed, with the ledger read the way that
    family's own engine states its parameters.
    """
    vals = ledger.values

    if tag == "pair":
        # eps_pa is the PRIVACY-AMPLIFICATION half alone, not ledger.secrecy:
        # Eq. (58) enters as 2*log2(2*eps_pa), eps_pe beside it. t is in BITS.
        return (_core.pair_length(*args, vals["t"], vals["eps_pa"]),)

    if tag == "sixstate":
        # eps_ec sits INSIDE eps, so the engine takes the OUTER number and
        # ledger.secrecy -- which is eps - eps_ec -- is not it.
        return _core.sixstate_length(*args, vals["eps"], vals["eps_ec"])

    if tag == "pairing":
        # Secrecy and correctness side by side; the engine divides the first by 24.
        return (_core.pairing_length(*args, ledger.secrecy, ledger.correctness),)

    if tag == "record":
        mu, nu, probs, counts, errors, n_key, e_key, f_ec = args
        # sarg_length takes the COMPOSED eps_sec and divides by 18 internally,
        # so the share the estimators run at is visible only through sarg_eps.
        each = _core.sarg_eps(vals["eps_sec"])
        s0, s1, s0_hi = _core.sarg_counts(mu, nu, probs, counts, errors, each)
        v1 = _core.sarg_errors(mu, nu, probs, errors, each)
        length = _core.sarg_length(
            s0,
            s1,
            v1,
            n_key,
            e_key,
            f_ec,
            vals["eps_sec"],
            vals["eps_cor"],
        )

        return (length, s0, s1, s0_hi, v1, each)

    if tag == "certified":
        return (_core.sarg_length(*args, vals["eps_sec"], vals["eps_cor"]),)

    if tag == "relaxed":
        # `source` is an OBSERVATION rather than a pinned string, so the engine's
        # refusal of every set but Eq. (21)'s reaches the caller: a minimum over
        # the contained equality set overstates the length.
        return _core.dm_length(*args[:5], vals["eps_bar"], vals["eps_pa"], *args[5:])

    if tag == "vacuum":
        # Li's eps_sec is the TOTAL 2*eps + eps_0 + 6*eps_1 + eps_2, which
        # Ledger.secrecy recomposes; eps_cor is separate and additive.
        return (_core.sdp_length(*args, ledger.secrecy, ledger.correctness),)

    if tag == "thresholds":
        # s_x and s_z are hash lengths in BITS, handed over as declared:
        # converting them to probabilities is what Ledger.secrecy refuses to do.
        return (_core.rrdps_length(*args, vals["s_x"], vals["s_z"]),)

    if tag == "block":
        # The engine takes the target the split was made from, not any one term;
        # Ledger.secrecy recomposes it through _core.rrdps_secpar.
        return _core.rrdps_finite(*args[:8], ledger.secrecy, args[8])

    raise NotImplementedError(
        f"LENGTHS declares the shape tagged '{tag}' and nothing here calls an "
        "engine for it. A tag reaches one named engine; falling through would "
        "read one family's counts under another family's bound"
    )


def sample_width(family, m, eps):
    """
    The statistical width of one finite parameter-estimation sample: the state
    that produced the data lies within it of the observed statistics except with
    probability `eps`.

    NOT A HOEFFDING WIDTH. This bounds a relative FREQUENCY and pays a
    method-of-types logarithm, so it falls as sqrt(ln m / m); decoy.rs bounds a
    COUNT by an additive deviation. Runs for 'sixstate' alone.
    """
    _single(family, "statistical width")

    return _core.sixstate_width(m, eps)


def residual_entropy(family, e_key, e_test):
    """
    Eve's residual uncertainty about one sifted key bit, H(X|E) in bits, from
    the key-basis error rate and the common monitor-basis error rate. The BARE
    quantity, with no statistical widening -- keylength() minimises it over the
    set the finite samples leave compatible.

    Runs for 'sixstate' alone: one e_test for both monitor bases is Scarani &
    Renner's own restriction.
    """
    _single(family, "residual entropy")

    return _core.sixstate_bound(e_key, e_test)


def _single(family, what):
    spec = describe(family)

    if family not in SINGLE:
        raise NotImplementedError(
            f"{spec.source} states no {what} of this shape, so family "
            f"'{family}' does not reach one here. It is Scarani & Renner's, "
            "written for a single-photon six-state source"
        )


# The mode-pairing estimators below are Xie's, and stay off sample_width() and
# residual_entropy(): the Chernoff pair is a range about a COUNT where the
# six-state width bounds a relative frequency, and the sampling penalty reads two
# string lengths and the sampled rate rather than one sample size.
PAIRED = ("pairing",)

# "expected" is the range the expected value may take given an observed count --
# what a real run needs -- and "observed" the range an observed count may take
# given a known mean, what a simulation runs. NOT INVERSES: their widths differ
# at order ln(1/eps).
CHERNOFF = ("expected", "observed")


def confidence(family, x, eps, kind="expected"):
    """
    `(lo, hi)`: one Chernoff interval about a count, at failure probability
    `eps`. `kind` names which direction it runs and the two are not inverses --
    see CHERNOFF.

    Runs for 'pairing' alone.
    """
    _paired(family, "Chernoff interval about a count")

    if kind not in CHERNOFF:
        raise ValueError(
            f"kind must be one of {', '.join(CHERNOFF)}, got {kind!r}: the two "
            "directions differ at order ln(1/eps) and are not inverses, so "
            "which one is wanted is stated rather than guessed"
        )

    if kind == "observed":
        return _core.pairing_observe(x, eps)

    return _core.pairing_expect(x, eps)


def single_yield(family, nu, mu, q_nu, q_mu, q_vac):
    """
    Lower bound on one arm's single-photon yield from a three-setting intensity
    ladder held against the other arm's vacuum. `q_nu` is a LOWER bound on the
    expected per-round gain at (vacuum, decoy), `q_mu` an UPPER bound at
    (vacuum, signal) and `q_vac` an UPPER bound at (vacuum, vacuum), each with
    the block size and the joint setting probabilities already divided out.

    Runs for 'pairing' alone: a Z-pair has one lit slot per party, so its pair
    intensity collapses to one dimension.
    """
    _paired(family, "one-dimensional decoy ladder against a vacuum arm")

    return _core.pairing_yield(nu, mu, q_nu, q_mu, q_vac)


def sampling(family, n, k, lam, eps):
    """
    The random-sampling-without-replacement penalty: how far the error rate of
    the unsampled string of length `n` may exceed the rate `lam` observed in a
    sample of length `k`, except with probability `eps`.

    Runs for 'pairing' alone. A third published form of the transfer
    bb84_length and mdi_length make, each belonging to its own key-length
    formula; unlike the other two it does NOT vanish as `lam` goes to zero, so a
    test sample showing no errors certifies nothing here.
    """
    _paired(family, "sampling penalty between two strings")

    return _core.pairing_gamma(n, k, lam, eps)


def phase_bound(family, s_key, s_test, e_test, eps):
    """
    Upper bound on the single-photon-pair phase error rate of the key basis,
    the X-basis bit error rate widened by sampling(). `s_key` and `s_test` are
    the certified single-photon-pair counts in the two bases. This is the `phi`
    keylength() takes.

    Runs for 'pairing' alone.
    """
    _paired(family, "phase error read off the other basis's single-photon pairs")

    return _core.pairing_phase(s_key, s_test, e_test, eps)


def _paired(family, what):
    spec = describe(family)

    if family not in PAIRED:
        raise NotImplementedError(
            f"{spec.source} states no {what}, so family '{family}' does not "
            "reach one here. It is Xie's, written for the pairing analysis"
        )


# The three COW' estimators below are all Li's and run in one order: deviation()
# six times on the decoy click counts, interval() on the box they leave,
# conversion() to turn that EXPECTED rate into the observed bound keylength()
# takes.
VACUUM = ("cow-vacuum",)


def deviation(family, k, count, eps, upper):
    """
    Kato's deviation in COUNTS at its optimised parameters, over `k` rounds at
    failure probability `eps`. `upper` true bounds the EXPECTED value above the
    observed count, so the bound is `count + this`; false bounds it below,
    `count - this`. The two directions are not one function of a sign.

    Runs for 'cow-vacuum' alone, six times per record -- one per bounded decoy
    gain, which is Li's weight of 6 on eps_1.
    """
    _vacuum(family, "Kato deviation about a decoy click count")

    return _core.sdp_kato(k, count, eps, upper)


def interval(family, mu, q0, q1, lo):
    """
    The EXPECTED phase-error rate over the box the Kato bounds leave, Li's
    Eqs. (18)-(20). `q0` and `q1` are the gains at the two monitoring ports in
    the order |0>|0>, |a>|a>, |0>|a>, |a>|0>, with the two DECOY entries
    upper-bounded and the two BIT entries observed; `lo` is the two LOWER
    bounds at D_M0 alone, ordered |0>|0>, |a>|a>.

    Returned raw and uncapped: at or above 1/2 the protocol aborts, and this is
    the expected rate rather than the observed bound -- conversion() is the step
    between them and keylength() takes the second.
    """
    _vacuum(family, "phase error over a box of Kato-bounded gains")

    return _core.sdp_interval(mu, q0, q1, lo)


def conversion(family, expect, n_z, eps):
    """
    The expected-to-observed step, Li Eq. (21): an expected phase-error rate
    becomes the OBSERVED upper bound over `n_z` sifted rounds. This is the
    `phase` keylength() takes.

    Runs for 'cow-vacuum' alone. Kato at a = 0, which is Azuma's width: the
    optimised parameters need the observed count and here the observed count is
    the thing being bounded.
    """
    _vacuum(family, "expected-to-observed conversion")

    return _core.sdp_sample(expect, n_z, eps)


def _vacuum(family, what):
    spec = describe(family)

    if family not in VACUUM:
        raise NotImplementedError(
            f"{spec.source} states no {what}, so family '{family}' does not "
            "reach one here. It is Li's, written over the FOUR-sequence COW' "
            "alphabet whose empty |0>|0> sequence is what makes the phase error "
            "derivable"
        )


def catalogue():
    """
    Every family, as (name, rule, terms, shares); the shape
    reconcile.catalogue() uses.
    """

    return tuple(
        (
            spec.name,
            spec.rule,
            tuple(t.name for t in spec.terms),
            spec.shares,
        )
        for _, spec in sorted(REGISTER.items())
    )
