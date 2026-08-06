import math
from dataclasses import dataclass

from . import _core, std

# The classical post-processing layer: reconciliation, privacy amplification
# and authentication, in BITS. Closed forms in Python; the one simulation,
# run_cascade, is src/cascade.rs.
#
# THREE QUANTITIES live here and are never interchanged, and every function
# says which it returns:
#   * a CODE RATE R, information bits per channel use, a property of the code;
#   * a RECONCILIATION EFFICIENCY, either f_EC >= 1 (Martinez-Mateo Eq. (2),
#     leakage over the Shannon limit, SUBTRACTED in a key rate) or beta <= 1
#     (Jouguet Sec. II.B, share of the capacity reached, MULTIPLYING I_AB);
#   * a FRAME ERROR RATE, the probability a reconciled frame still differs.
# bridge() is the only conversion, Martinez-Mateo Eq. (4), an identity.
#
# Not modelled: slice reconciliation, BICONF, polar and rateless codes, the
# ETSI key-delivery accounting.
#
# qkd.security composes failure probabilities and never converts one to bits;
# this module counts bits and owns the leftover hash lemma, the one conversion
# between the two currencies. No engine in the tree returns a min-entropy, so
# hash_length takes one as an input and cannot be handed a key length.

_LN2 = math.log(2.0)

# Simpson nodes for biawgn_capacity: 401 over +/- 9 sigma reproduce the same
# 12 decimals as 3201, so this is convergence, not a tolerance.
_NODES = 401
_SPAN = 9.0

# Frame error rate the multi-edge codes of Jouguet, Kunz-Jacques & Leverrier,
# PRA 84, 062317 (2011), arXiv:1110.0100, Sec. II.C were measured at: "a quite
# high Frame Error Rate (FER) (about 1/3) but a null Bit Error Rate (BER) on
# the blocks where the decoding succeeded". Stated ONCE for the whole set, so
# every rotated row below carries the same number.
_MET_FER = 1.0 / 3.0

_NO_INTERP = (
    "Cascade efficiency is tabulated, not fitted: Martinez-Mateo et al., QIC "
    "15, 453 (2015), arXiv:1407.3257, Table 3 measures {grid} and nothing "
    "between them, and f_EC is not monotone across that grid (1.05182 at 0.5%, "
    "1.03945 at 3%, rising again after). Ask for a tabulated point, or run "
    "run_cascade() at the error rate you want"
)


@dataclass(frozen=True)
class Ldpc:
    """
    One published low-density parity-check code, as its authors measured it.
    rate is R in information bits per channel use; snr is the threshold below
    which the frame does not decode; dim is the multidimensional reconciliation
    dimension the threshold was measured with (0 for none, None for a bare
    binary-input channel with no CV rotation); beta is the efficiency the
    source prints; block is the codeword length; fer is the frame error rate
    where the source gives one.
    """

    name: str
    rate: float
    snr: float
    dim: int | None
    beta: float
    block: int | None = None
    fer: float | None = None


@dataclass(frozen=True)
class Cascade:
    """
    A Cascade configuration: the interactive parity-exchange reconciliation of
    Brassard & Salvail, EUROCRYPT '93. frame is the frame length n in bits and
    variant names a published parameter set in CASCADE_RULES; passes and reuse
    override that variant's pass count and its use of the subblocks the
    dichotomic search produces.
    """

    variant: str = "nearoptimal"
    frame: int = 1 << 14
    passes: int | None = None
    reuse: bool | None = None


@dataclass(frozen=True)
class Amplification:
    """
    Privacy amplification by two-universal hashing. eps is the distance from
    uniform the extracted key is allowed to have; modified picks the
    (T(r) | I_m) family of Hayashi & Tsurumaru over the plain Toeplitz matrix
    of Fung, Ma & Chau, which changes only the seed length.
    """

    eps: float = 1e-10
    modified: bool = False

    def __post_init__(self):
        std.fraction("eps", self.eps)


@dataclass(frozen=True)
class Authentication:
    """
    Authentication of the classical channel by the Wegman-Carter scheme, in
    the LFSR-based Toeplitz construction of Krawczyk that Fung, Ma & Chau,
    PRA 81, 012318 (2010) cost out. eps is the per-message forgery
    probability; recycle says the 2k-bit hash seed survives because the tag is
    one-time padded, so a message costs k bits rather than 3k.
    """

    eps: float = 1e-10
    recycle: bool = True

    def __post_init__(self):
        std.fraction("eps", self.eps)


# Jouguet, Kunz-Jacques & Leverrier, PRA 84, 062317 (2011), arXiv:1110.0100.
#
# Table I: asymptotic density-evolution thresholds on the binary-input AWGN
# channel. dim=None: the codes' own efficiencies on the BIAWGNC, before any CV
# rotation. Tables II and III: the same codes at block length 2^20 through the
# multidimensional reconciliation of Leverrier et al., PRA 77, 042325 (2008),
# arXiv:0712.3823; dim=0 is the column headed s and beta, no rotation at all.
#
# The rate-1/2 row is Sec. III of the same paper, quoted from Table VI of
# Richardson & Urbanke: "The SNR threshold given by Discretized Density
# Evolution is s* = 1.074. The corresponding efficiency on the BIAWGNC is
# 98.2%." It settles which capacity beta is measured against:
# 0.5/C_BIAWGN(1.074) = 0.9817 and 0.5/C_AWGN(1.074) = 0.9502.
CODES = (
    Ldpc("met-0.1", 0.1, 0.156, None, 0.959),
    Ldpc("met-0.05", 0.05, 0.074, None, 0.972),
    Ldpc("met-0.02", 0.02, 0.029, None, 0.981),
    Ldpc("met-0.5", 0.5, 1.074, None, 0.982),
    Ldpc("met-0.1-d0", 0.1, 0.271, 0, 0.579, 1 << 20, _MET_FER),
    Ldpc("met-0.1-d1", 0.1, 0.187, 1, 0.808, 1 << 20, _MET_FER),
    Ldpc("met-0.1-d2", 0.1, 0.169, 2, 0.887, 1 << 20, _MET_FER),
    Ldpc("met-0.1-d4", 0.1, 0.163, 4, 0.921, 1 << 20, _MET_FER),
    Ldpc("met-0.1-d8", 0.1, 0.161, 8, 0.931, 1 << 20, _MET_FER),
    Ldpc("met-0.05-d0", 0.05, 0.123, 0, 0.597, 1 << 20, _MET_FER),
    Ldpc("met-0.05-d1", 0.05, 0.082, 1, 0.883, 1 << 20, _MET_FER),
    Ldpc("met-0.05-d2", 0.05, 0.077, 2, 0.935, 1 << 20, _MET_FER),
    Ldpc("met-0.05-d4", 0.05, 0.076, 4, 0.948, 1 << 20, _MET_FER),
    Ldpc("met-0.05-d8", 0.05, 0.075, 8, 0.958, 1 << 20, _MET_FER),
    Ldpc("met-0.02-d0", 0.02, 0.047, 0, 0.600, 1 << 20, _MET_FER),
    Ldpc("met-0.02-d1", 0.02, 0.030, 1, 0.931, 1 << 20, _MET_FER),
    Ldpc("met-0.02-d2", 0.02, 0.029, 2, 0.963, 1 << 20, _MET_FER),
    Ldpc("met-0.02-d4", 0.02, 0.029, 4, 0.966, 1 << 20, _MET_FER),
    Ldpc("met-0.02-d8", 0.02, 0.029, 8, 0.969, 1 << 20, _MET_FER),
)

# Martinez-Mateo, Pacher, Peev, Ciurana & Martin, "Demystifying the information
# reconciliation protocol Cascade", QIC 15, 453 (2015), arXiv:1407.3257,
# Table 3: the near-optimal parameter set, frame n = 2^14, 14 passes, subblock
# reuse. Columns are (Q, k1, k2, k3, eta_EC, f_EC, frame error rate, beta,
# channel uses). eta_EC is Eq. (6), f_EC Eq. (2) and beta Eq. (3) -- three
# numbers for one run, and they are not the same quantity.
CASCADE_TABLE = (
    (0.005, 256, 1024, 4096, 1.05182, 1.04989, 9.2e-5, 0.9976, 168.6),
    (0.010, 128, 512, 4096, 1.04310, 1.04219, 8.0e-5, 0.9963, 208.8),
    (0.020, 64, 512, 4096, 1.04062, 1.04006, 9.3e-5, 0.9934, 407.6),
    (0.030, 32, 512, 4096, 1.03945, 1.03902, 1.1e-4, 0.9906, 496.9),
    (0.040, 32, 256, 4096, 1.04342, 1.04313, 9.4e-5, 0.9862, 500.2),
    (0.050, 16, 256, 4096, 1.04335, 1.04313, 8.9e-5, 0.9827, 432.6),
    (0.060, 16, 256, 4096, 1.04601, 1.04580, 1.1e-4, 0.9777, 606.6),
    (0.070, 16, 256, 4096, 1.05065, 1.05050, 8.7e-5, 0.9709, 796.9),
    (0.080, 8, 256, 4096, 1.05479, 1.05465, 9.7e-5, 0.9632, 550.3),
    (0.090, 8, 256, 4096, 1.05499, 1.05486, 1.0e-4, 0.9575, 690.4),
    (0.100, 8, 256, 4096, 1.05747, 1.05736, 1.0e-4, 0.9493, 840.3),
    (0.110, 8, 256, 4096, 1.06139, 1.06130, 1.0e-4, 0.9387, 998.4),
)

# The three error rates where Martinez-Mateo's own closed form for the
# near-optimal block sizes disagrees with its Table 3, as
# {Q: (k1, k2) the table prints}. The rule the text proposes is
# k1 = 2^ceil(a), k2 = 2^ceil((a + 12)/2) with a = log2(1/Q) - 1/2, and it
# reproduces nine of the twelve rows. Recorded rather than reconciled: the
# table is the measurement, the formula its summary.
CASCADE_MISFITS = {
    0.010: (128, 512),
    0.040: (32, 256),
    0.080: (8, 256),
}

# Martinez-Mateo Table 1, the block-size rules of every Cascade variant it
# simulates, as (passes, reuse); the sizes themselves are in blocks().
# "sugimoto" is listed for completeness and run_cascade REFUSES it: after two
# Cascade passes that version switches to BICONF, a different algorithm.
CASCADE_RULES = {
    "original": (4, False),
    "sugimoto": (2, False),
    "yan": (10, False),
    "optimised": (16, False),
    "poweroftwo": (14, True),
    "nearoptimal": (14, True),
}

# What each shipped finite-size engine charges for privacy amplification, as
# {family: (bits above amplify_cost, where the line is)}. Read off the engine
# named in each row, not inferred: three print 2*log2(1/(2 eps)), which is
# amplify_cost to the last bit, and the Gaussian path prints 2*log2(1/eps),
# two bits more. The family names are qkd.security's, so a budget composed
# there and a bill counted here address the same row. bb84 and rrdps are
# absent on purpose; see HASH_ABSENT.
PA_FORMS = {
    "cv": (2.0, "src/keyrate.rs::cv_finite, inside its delta"),
    "mdi": (0.0, "src/mdi.rs::mdi_length, inside its budget"),
    "pairing": (0.0, "src/pairing.rs::pairing_length, inside its budget"),
    "sixstate": (0.0, "src/sixstate.rs::sixstate_length, inside its delta"),
}

# The two families whose hashing charge is not a separable bit count.
HASH_ABSENT = {
    "bb84": (
        "src/discrete.rs::bb84_length charges 6*log2(21/eps_sec) + "
        "log2(2/eps_cor), and no part of that is 2*log2(1/(2 eps)): the 6 is "
        "not six independent epsilons but Lim's supplementary Eq. (B3) "
        "grouping, so the leftover-hash share is not separable from the "
        "smoothing and estimation terms"
    ),
    "rrdps": (
        "src/rrdps.rs::rrdps_length subtracts the two hash lengths s_x and "
        "s_z in BITS directly, eta = 2**-s being the probability each stands "
        "for, so its hashing charge is a declared bit count rather than a "
        "function of an eps. Read the two lengths off _core.rrdps_split(d), "
        "Takesue's own assignment"
    ),
}


# Binary Shannon entropy in bits, h(e) of Martinez-Mateo Eq. (2). std.entropy
# is the one implementation, bound to this module's own name.
_entropy = std.entropy


def awgn_capacity(snr):
    """
    Capacity of the additive white Gaussian noise channel with a Gaussian
    input, in bits per channel use: (1/2) log2(1 + SNR). I(A:B) for a Gaussian
    modulation over a Gaussian channel, Leverrier et al., PRA 77, 042325
    (2008), arXiv:0712.3823, Sec. VI, and the denominator the key-rate beta of
    q.Asymptotic multiplies.
    """
    std.positive("snr", snr)

    return 0.5 * math.log2(1.0 + snr)


def biawgn_capacity(snr):
    """
    Capacity of the BINARY-input AWGN channel at the same SNR = 1/sigma^2, in
    bits per channel use, by Simpson quadrature of
    1 - E_n[log2(1 + exp(-2(1 + sigma n)/sigma^2))]. A different channel from
    awgn_capacity: the codes of CODES are binary and their published beta is
    measured against THIS capacity, which the rate-1/2 row settles.
    """
    std.positive("snr", snr)

    sigma = 1.0 / math.sqrt(snr)
    step = 2.0 * _SPAN / (_NODES - 1)
    total = 0.0
    for i in range(_NODES):
        x = -_SPAN + i * step
        weight = 1.0 if i in (0, _NODES - 1) else (4.0 if i % 2 else 2.0)
        z = -2.0 * (1.0 + sigma * x) / (sigma * sigma)
        soft = (max(z, 0.0) + math.log1p(math.exp(-abs(z)))) / _LN2
        total += weight * math.exp(-0.5 * x * x) * soft

    total *= step / (3.0 * math.sqrt(2.0 * math.pi))

    return 1.0 - total


def efficiency(rate, snr, binary=True):
    """
    Reconciliation efficiency beta = R / C(snr), the share of the channel
    capacity a code of rate R reaches: Jouguet, Kunz-Jacques & Leverrier, PRA
    84, 062317 (2011), Sec. II.B. MULTIPLIES I_AB in a key rate. binary picks
    the binary-input capacity, which is the one the published beta of every
    code in CODES is measured against.
    """
    std.positive("rate", rate)

    cap = biawgn_capacity(snr) if binary else awgn_capacity(snr)
    if rate > cap:
        raise ValueError(
            f"no code of rate {rate:g} decodes at SNR {snr:g}: the capacity "
            f"there is {cap:.6g} bit per channel use, so the rate is above the "
            "Shannon limit. The channel coding theorem refusing, not a tolerance"
        )

    return rate / cap


def inefficiency(rate, qber):
    """
    Reconciliation inefficiency f_EC = (1 - R)/h(Q) over a binary symmetric
    channel: Martinez-Mateo et al., QIC 15, 453 (2015), arXiv:1407.3257,
    Eq. (2), with R = 1 - m/n the share of the frame not disclosed. It
    multiplies h(QBER) where a key rate SUBTRACTS it; f_EC below 1 is below the
    Slepian-Wolf bound and is refused.
    """
    std.closed("rate", rate)

    std.fraction("qber", qber, 0.5)

    if qber <= 0.0:
        raise ValueError(
            "f_EC is undefined at zero error rate: h(0) = 0, so Eq. (2) "
            "divides by zero. A frame with no errors still costs the parities "
            "that established it, a leakage in bits and not a ratio"
        )

    f = (1.0 - rate) / _entropy(qber)
    if f < 1.0:
        raise ValueError(
            f"a rate-{rate:g} code at QBER {qber:g} would leak "
            f"{1.0 - rate:.6g} bit per frame bit against a Slepian-Wolf "
            f"minimum of {_entropy(qber):.6g}, i.e. f_EC = {f:.6g} < 1. No "
            "protocol reconciles below the conditional entropy"
        )

    return f


def capacity_ratio(rate, qber):
    """
    The other reconciliation efficiency, beta = R/(1 - h(Q)): Martinez-Mateo
    Eq. (3), the share of the BSC capacity reached. Its denominator is the BSC
    capacity, not the AWGN capacity efficiency() uses, so the two betas are the
    same definition over different channels and must not be swapped.
    """
    std.closed("rate", rate)

    std.fraction("qber", qber, 0.5)

    cap = 1.0 - _entropy(qber)
    if rate > cap:
        raise ValueError(
            f"a rate-{rate:g} code is above the BSC({qber:g}) capacity of {cap:.6g}, so beta would exceed 1"
        )

    return rate / cap


def bridge(qber, f_ec=None, beta=None):
    """
    Convert between the two efficiencies at one error rate: Martinez-Mateo
    Eq. (4), 1 - f_EC h(Q) = beta (1 - h(Q)). An identity between the two
    definitions of Eqs. (2) and (3), not a fit. Give exactly one of f_ec or
    beta and the other comes back.
    """
    std.fraction("qber", qber, 0.5)

    if (f_ec is None) == (beta is None):
        raise ValueError("bridge takes exactly one of f_ec= or beta=")

    h = _entropy(qber)
    if h >= 1.0:
        raise ValueError(
            f"at QBER {qber:g} the BSC capacity 1 - h(Q) is zero, so Eq. (4) "
            "relates nothing: no key survives that error rate under either "
            "definition"
        )

    if beta is None:
        std.atleast("f_ec", f_ec, 1.0)

        return (1.0 - f_ec * h) / (1.0 - h)

    std.unit("beta", beta)

    return (1.0 - beta * (1.0 - h)) / h


def leak_ratio(rate, fer):
    """
    Information leaked per frame bit once failed frames are counted:
    Martinez-Mateo Eq. (5), leak_EC = (1 - eps)(1 - R) + eps. The whole frame
    is charged when reconciliation fails, which keeps the frame error rate out
    of the key rate as a separate factor.
    """
    std.closed("rate", rate)

    std.closed("fer", fer)

    return (1.0 - fer) * (1.0 - rate) + fer


def charged(f_ec, qber, fer):
    """
    Reconciliation efficiency with the frame error rate folded in:
    Martinez-Mateo Eq. (6), eta_EC = leak_EC/h(Q), written here as
    (1 - eps) f_EC + eps/h(Q) so it takes an f_EC rather than a code rate.
    Always at or above f_EC, and it diverges as the error rate goes to zero
    because a failed frame costs its whole length however few errors it had.
    """
    std.atleast("f_ec", f_ec, 1.0)

    std.fraction("qber", qber, 0.5)

    std.closed("fer", fer)

    h = _entropy(qber)
    if h <= 0.0:
        raise ValueError("eta_EC is undefined at zero error rate: Eq. (6) divides by h(Q)")

    return (1.0 - fer) * f_ec + fer / h


def codes(dim=None, rate=None):
    """
    The published code catalogue, filtered. dim selects a multidimensional
    reconciliation dimension (0 for no rotation, None for every row); rate
    selects one code rate. Returns the rows verbatim, in table order.
    """

    return tuple(c for c in CODES if (dim is None or c.dim == dim) and (rate is None or c.rate == rate))


def pick(snr, dim=8):
    """
    The highest-rate published code in CODES whose measured threshold sits at
    or below snr, for a given reconciliation dimension. Refuses an SNR under
    every threshold on file.
    """
    std.positive("snr", snr)

    fits = [c for c in codes(dim) if c.snr <= snr]
    if not fits:
        floors = sorted(c.snr for c in codes(dim))
        raise ValueError(
            f"SNR {snr:g} is below every d={dim} threshold on file "
            f"({', '.join(f'{s:g}' for s in floors)}): no code here decodes there, "
            "and the lowest-rate one would fail rather than run slowly"
        )

    return max(fits, key=lambda c: c.rate)


def code_beta(code, snr, binary=True):
    """
    The efficiency a published code actually reaches when run at snr rather
    than at its threshold: beta = R/C(snr). Refuses below the code's measured
    threshold, where the frame does not decode. Above it beta FALLS, a
    fixed-rate code leaving the surplus capacity on the table, so code.beta --
    the value at threshold -- is the largest this can return.
    """
    if snr < code.snr:
        raise ValueError(
            f"{code.name} was measured to decode down to SNR {code.snr:g} and "
            f"{snr:g} is below it: the frame does not decode, so the "
            "efficiency is undefined rather than small. pick() chooses a code "
            "that does decode"
        )

    return efficiency(code.rate, snr, binary)


def cascade_point(qber, atol=1e-9):
    """
    One measured row of Martinez-Mateo Table 3 as
    (k1, k2, k3, eta_EC, f_EC, fer, beta, rounds), for the near-optimal
    Cascade at frame 2^14 with 14 passes and subblock reuse. Refuses an error
    rate the table does not carry.
    """
    std.fraction("qber", qber, 0.5)

    for row in CASCADE_TABLE:
        if abs(row[0] - qber) <= atol:
            return row[1:]

    grid = ", ".join(f"{row[0]:g}" for row in CASCADE_TABLE)

    raise ValueError(_NO_INTERP.format(grid=grid))


def blocks(qber, frame, variant="nearoptimal", passes=None):
    """
    The block-size schedule of one published Cascade parameter set, one entry
    per pass, from Table 1 of Martinez-Mateo (which collects the rules of
    Brassard & Salvail, Sugimoto & Yamazaki and Yan et al. beside its own).
    The near-optimal rule is k1 = 2^ceil(a), k2 = 2^ceil((a + 12)/2),
    k3 = 4096, k_i = ceil(n/2), with a = log2(1/Q) - 1/2, and it returns nine
    of Table 3's twelve rows; see CASCADE_MISFITS.
    """
    std.fraction("qber", qber, 0.5)

    if qber <= 0.0:
        raise ValueError(
            "every Cascade block-size rule is a function of 1/Q and diverges "
            "at zero: a frame with no errors needs no passes"
        )

    if variant not in CASCADE_RULES:
        raise ValueError(f"unknown Cascade variant {variant!r}; known: {', '.join(sorted(CASCADE_RULES))}")

    count, _ = CASCADE_RULES[variant]
    if passes is not None:
        count = passes

    half = -(-frame // 2)
    a = math.log2(1.0 / qber) - 0.5
    if variant == "original":
        first = [math.ceil(0.73 / qber)]
        while len(first) < count:
            first.append(2 * first[-1])

        return tuple(first[:count])

    if variant == "sugimoto":
        k1 = math.floor(4.0 * _LN2 / (3.0 * qber))
        sizes = [k1, math.floor(4.0 * _LN2 / qber)]
    elif variant == "yan":
        k1 = math.ceil(0.8 / qber)
        sizes = [k1, 5 * k1]
    elif variant == "optimised":
        k1 = math.ceil(1.0 / qber)
        sizes = [k1, 2 * k1]
    elif variant == "poweroftwo":
        k1 = 2 ** math.ceil(math.log2(1.0 / qber))
        sizes = [k1, 4 * k1]
    else:
        sizes = [2 ** math.ceil(a), 2 ** math.ceil((a + 12.0) / 2.0), 4096]

    while len(sizes) < count:
        sizes.append(half)

    return tuple(sizes[:count])


def run_cascade(qber, frames=8, seed=1, cfg=None):
    """
    Simulate Cascade and COUNT the parities it discloses, returning
    (mean leak in bits per frame, frames left with a residual error). Divide by
    frame*h(Q) for the f_EC of Eq. (2). A SAMPLE MEAN over `frames` frames, so
    it carries sampling error and is not a bound.

    The residual count is a FRAME ERROR RATE and is NOT eps_cor: it counts
    frames whose keys differ, where eps_cor is the probability they differ and
    the error-verification hash does not notice. frame_bound() turns the count
    into a confidence limit and verify_failure() supplies the second factor;
    neither makes it a security parameter, the errors here being drawn i.i.d.
    from an honest binary symmetric channel rather than arranged by an
    adversary.

    seed keys the Threefry stream of src/cascade.rs and is not a random.Random
    seed: the same integer draws a different frame than Python's Mersenne
    Twister did, so only the sample statistics carry across.
    """
    cfg = cfg or Cascade()
    if cfg.variant == "sugimoto":
        raise ValueError(
            "the Sugimoto-Yamazaki version runs two Cascade passes and then "
            "BICONF, which selects random subsets of the whole frame rather "
            "than dividing it into blocks. That is a different algorithm and "
            "is not implemented here"
        )

    sizes = blocks(qber, cfg.frame, cfg.variant, cfg.passes)
    reuse = CASCADE_RULES[cfg.variant][1] if cfg.reuse is None else cfg.reuse

    return _core.cascade_run(qber, cfg.frame, sizes, reuse, int(seed), int(frames))


def _tail(frames, failed, prob):
    # P(X <= failed) for X binomial in `frames` trials at `prob`, through
    # log-gamma so a large frame count does not overflow the coefficient.
    total = 0.0
    for i in range(int(failed) + 1):
        term = (
            math.lgamma(frames + 1.0)
            - math.lgamma(i + 1.0)
            - math.lgamma(frames - i + 1.0)
            + i * math.log(prob)
            + (frames - i) * math.log1p(-prob)
        )
        total += math.exp(term)

    return total


def frame_bound(frames, failed, conf=0.95):
    """
    One-sided upper confidence limit on the frame error rate behind a
    counted run, by inverting the binomial tail: the largest rate under
    which `failed` or fewer failures in `frames` frames has probability
    1 - conf. At zero failures it is the closed form 1 - (1 - conf)**(1/n).
    A BOUND ON THE SIMULATED CHANNEL and on nothing else: run_cascade draws
    its errors i.i.d. at the given rate, so this prices the sampling and not
    an adversary, and it is a frame error rate rather than an eps_cor.
    """
    std.positive("frames", frames)

    std.nonneg("failed", failed)

    std.open_unit("conf", conf)

    if failed > frames:
        raise ValueError(f"{failed:g} frames failed out of {frames:g}: a run cannot lose more frames than it ran")

    if failed >= frames:
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _tail(frames, failed, mid) > 1.0 - conf:
            lo = mid
        else:
            hi = mid

    return hi


def verify_failure(fer, eps_hash):
    """
    Probability that two reconciled strings differ AND error verification
    does not notice: fer times eps_hash. This is eps_cor, the correctness
    parameter src/discrete.rs::bb84_length charges log2(2/eps_cor) for, and
    its two factors come from different places -- fer from the code, eps_hash
    from the two-universal family the strings are compared through, which
    poly_collision and tag_failure size. A BOUND only when `fer` is itself a
    bound, which run_cascade's count is not.
    """
    std.closed("fer", fer)

    std.open_unit("eps_hash", eps_hash)

    if fer <= 0.0:
        raise ValueError(
            "a frame error rate of zero makes eps_cor zero, certifying that the "
            "two keys never differ. No counted run says that: zero failures in n "
            "frames bounds the rate at 1 - (1 - conf)**(1/n), which is 0.0894 at "
            "32 frames and 95% confidence -- 961 times the 9.3e-5 Martinez-Mateo "
            "measure at a 2% error rate. Pass a bound, from frame_bound() or from "
            "a declared frame error rate, and not a count of zero"
        )

    return fer * eps_hash


def hash_length(h_min, eps):
    """
    Bits a two-universal hash may extract, the leftover hash lemma:
    Tomamichel, Schaffner, Smith & Renner, IEEE Trans. Inf. Theory 57, 5524
    (2011), arXiv:1002.2436, Eq. (1), floor(H_min(X|E) - 2 log2(1/(2 eps))).
    h_min is in bits and is an INPUT because NOTHING IN THE TREE RETURNS ONE:
    the finite-size engines return key LENGTHS with the hashing already charged
    inside them, so passing one as h_min charges the hashing twice.
    hash_charge() reports what each engine charged, which is the comparison
    that can be made instead.
    """
    std.nonneg("h_min", h_min)

    cost = amplify_cost(eps)
    ell = math.floor(h_min - cost)
    if ell <= 0:
        raise ValueError(
            f"the block carries {h_min:g} bit of smooth min-entropy and the "
            f"leftover hash lemma charges {cost:.6g} for an eps of {eps:g}, so no "
            "key comes out. A shorter hash does not help: the charge does not "
            "shrink with the output length"
        )

    return ell


def amplify_cost(eps):
    """
    The 2 log2(1/(2 eps)) bits privacy amplification costs, Tomamichel Eq. (1).
    Independent of the block length, so it is a fixed toll per distilled block.
    Papers that print 2 log2(1/eps) charge 2 bits more, the conservative
    reading of the same lemma.
    """
    std.fraction("eps", eps)

    if eps <= 0.0:
        raise ValueError("eps must be positive: 2 log2(1/(2 eps)) diverges")

    return 2.0 * math.log2(1.0 / (2.0 * eps))


def hash_charge(family, eps):
    """
    Bits a named finite-size engine charges for privacy amplification, so the
    two accountings of one step can be COMPARED rather than added; PA_FORMS
    records which engine and where each line was read from. eps is the failure
    probability that engine hands the lemma, not a composed secrecy parameter:
    _core.mdi_eps and _core.pairing_eps return it, _core.sixstate_eps returns
    it as the first of its pair, and the Gaussian path takes q.FiniteSize.eps
    unchanged. Refuses bb84 and rrdps.
    """
    if family in HASH_ABSENT:
        raise ValueError(HASH_ABSENT[family])

    if family not in PA_FORMS:
        known = ", ".join(sorted(PA_FORMS))
        named = ", ".join(sorted(HASH_ABSENT))
        raise ValueError(
            f"no privacy-amplification charge is recorded for '{family}'. "
            f"Recorded: {known}; refused by name: {named}. A family is recorded "
            "when an engine in src/ prints a leftover-hash line readable out of "
            "its budget on its own"
        )

    extra, _ = PA_FORMS[family]

    return amplify_cost(eps) + extra


def hash_distance(ell, h_min, smooth=0.0):
    """
    How far from uniform an ell-bit two-universal hash output sits, given the
    conditioning system: Tomamichel Lemma 1, (1/2) sqrt(2^(ell - H_min)), and
    Theorem 8 for the smoothed entropy, where the smoothing parameter adds on
    top. The inverse of hash_length.
    """
    std.nonneg("h_min", h_min)

    std.nonneg("smooth", smooth)

    return smooth + 0.5 * math.sqrt(2.0 ** (ell - h_min))


def toeplitz_seed(n, ell, modified=False):
    """
    Random bits needed to pick the hash function: n + ell - 1 for the plain
    Toeplitz matrix of Fung, Ma & Chau, PRA 81, 012318 (2010), Sec. III.A, and
    n - 1 for the modified family (T(r) | I_m) of Hayashi & Tsurumaru, IEEE
    Trans. Inf. Theory 62, 2213 (2016), arXiv:1311.5322, App. D, which is the
    one stated there to be universal_2. Seed, not key: it is public, and
    sending it costs an authenticated message rather than a secret.
    """
    std.positive("n", n)

    std.positive("ell", ell)

    if ell > n:
        raise ValueError(f"a hash to {ell:g} bits cannot take more out of a {n:g}-bit block than it was given")

    if modified:
        return n - 1.0

    return n + ell - 1.0


def poly_collision(chunks, bits):
    """
    Collision probability delta of the polynomial hash family
    (x_1..x_r) -> sum x_i alpha^(r-i) over a field of 2^bits elements:
    Tomamichel Sec. I.C, delta = (r - 1)/|F|. delta-almost two-universal rather
    than two-universal, so this is for authentication sizing rather than for
    privacy amplification.
    """
    std.positive("chunks", chunks)

    std.positive("bits", bits)

    return (chunks - 1.0) / 2.0**bits


def tag_failure(message, tag):
    """
    Forgery probability of one authenticated message under the LFSR-based
    Toeplitz construction: n 2^(-k+1), Krawczyk's Theorem 9 as quoted by Fung,
    Ma & Chau, PRA 81, 012318 (2010), Eq. (11), for a message of n bits and a
    tag of k bits.
    """
    std.positive("message", message)

    std.positive("tag", tag)

    return message * 2.0 ** (1.0 - tag)


def lfsr_failure(message, tag, seed):
    """
    The same forgery probability under Krawczyk's later construction, which
    lets the hash seed shrink to any length: 2^-k + (k + n - 1)/2^(seed/2),
    Fung Eq. (12). At seed = 2k it collapses to (n + k)/2^k against
    tag_failure's 2n/2^k, so it is a factor 2n/(n + k) SMALLER. tag_length is
    built on the larger of the two, so the tag it returns is the conservative
    one.
    """
    std.positive("message", message)

    std.positive("tag", tag)

    std.positive("seed", seed)

    return 2.0**-tag + (tag + message - 1.0) * 2.0 ** (-seed / 2.0)


def tag_length(message, eps):
    """
    Shortest tag meeting a forgery probability, from tag_failure inverted:
    k = ceil(1 + log2(n/eps)). Logarithmic in the message, so the
    authentication toll is set by how OFTEN a round authenticates.
    """
    std.positive("message", message)

    std.fraction("eps", eps)

    if eps <= 0.0:
        raise ValueError("eps must be positive: the tag length diverges")

    return math.ceil(1.0 + math.log2(message / eps))


def auth_cost(lengths, eps, recycle=True):
    """
    Secret bits an authenticated round consumes, summed over its messages.
    Fung Sec. III.B: a message costs a 2k-bit hash seed and a k-bit one-time
    pad for the tag, and because the tag is one-time padded the seed survives,
    so the NET cost is k. recycle=False charges the full 3k, for a round that
    draws a fresh seed each time.

    BITS ONLY. The same round's forgery probability is auth_failure, a separate
    return because the two compose by different rules: bits add into
    net_length, probabilities union-bound into a security parameter.
    """
    std.fraction("eps", eps)

    if not lengths:
        raise ValueError("an authenticated round has at least one message")

    tags = [tag_length(n, eps) for n in lengths]
    per = 1 if recycle else 3

    return float(per * sum(tags))


def auth_failure(lengths, eps):
    """
    Forgery probability of the same authenticated round auth_cost prices: the
    union bound over its messages of the probability each tag is forged, at
    the tag length tag_length picks for eps. STRICTLY BELOW len(lengths)*eps,
    because tag_length rounds up to a whole bit. A union bound needs no
    independence, so it holds over the shared seed auth_cost's recycle=True
    reuses: recycling is a cost switch, not an epsilon one.
    """
    std.fraction("eps", eps)

    if not lengths:
        raise ValueError("an authenticated round has at least one message")

    total = sum(tag_failure(n, tag_length(n, eps)) for n in lengths)
    if total >= 1.0:
        raise ValueError(
            f"{len(lengths)} messages at eps = {eps:g} compose to {total:g}, which "
            "is not a probability: the union bound has saturated and the round's "
            "authentication certifies nothing"
        )

    return total


def net_length(gross, costs, charged_ec=True):
    """
    Net key a block delivers: Fung Eq. (24), NR >= l - k_bs - k_ec - k_ev -
    k_pa, l being the length privacy amplification produced and the k's the
    secret bits the round spent. BITS ALONE: the security parameter those bits
    were certified at is untouched here, and the authentication term's share of
    it is auth_failure. Clamped at zero. charged_ec says the key length already
    had the reconciliation leakage subtracted, the usual Devetak-Winter
    accounting; Fung instead one-time-pads the parities and pays k_ec here, and
    both together are REFUSED.
    """
    std.nonneg("gross", gross)

    total = 0.0
    for name, bits in sorted(costs.items()):
        std.nonneg(name, bits)

        if name == "reconcile" and charged_ec:
            raise ValueError(
                "the reconciliation leakage is being charged twice: the key "
                "length already has f_EC h(Q) subtracted, and costs carries a "
                "'reconcile' entry as well. Fung, Ma & Chau one-time-pad the "
                "parities and charge them here INSTEAD of subtracting them, "
                "so pick one accounting: drop the entry, or pass "
                "charged_ec=False with a gross length that has not paid for "
                "error correction"
            )

        total += bits

    return max(0.0, gross - total)


def round_cost(sift, ell, announce, amplify=None, auth=None):
    """
    The classical bill for one distillation block, in bits, as {name: bits}:
    the leftover hash lemma's fixed toll, plus authentication tags for the
    three messages Fung, Ma & Chau's Table I charges for -- the announcement,
    the error-verification tag and the privacy-amplification seed. Feeds
    net_length as its costs. sift is the block's sifted length, ell the key
    length coming out of it and announce the announcement message length.
    """
    std.positive("sift", sift)

    std.positive("ell", ell)

    std.positive("announce", announce)

    amplify = amplify or Amplification()
    auth = auth or Authentication()
    seed = toeplitz_seed(sift, ell, amplify.modified)
    tags = auth_cost((announce, sift, seed), auth.eps, auth.recycle)

    return {
        "amplify": amplify_cost(amplify.eps),
        "authenticate": tags,
    }


def net_rate(gross, sift, cadence, announce, amplify=None, auth=None):
    """
    Delivered key rate in bit/s once the classical layer is paid, from a gross
    rate in bit/s and a block cadence in blocks per second. gross must already
    carry the reconciliation leakage, as every qkd key rate does, so only
    hashing and authentication are charged here.
    """
    std.positive("cadence", cadence)

    std.nonneg("gross", gross)

    costs = round_cost(sift, gross / cadence, announce, amplify, auth)

    return max(0.0, gross - cadence * sum(costs.values()))


def block_floor(per_bit, announce, amplify=None, auth=None, cap=1e12):
    """
    The smallest sifted block that can still pay its own classical bill, by
    bisection. per_bit is the block's key bits per sifted bit. Below this
    length a block delivers nothing however good the quantum layer is.
    """
    std.unit("per_bit", per_bit)

    def net(n):
        costs = round_cost(n, per_bit * n, announce, amplify, auth)

        return per_bit * n - sum(costs.values())

    if net(cap) <= 0.0:
        raise ValueError(
            f"a block of {cap:g} sifted bits at {per_bit:g} key bit per "
            "sifted bit still cannot pay for its own hashing and "
            "authentication: there is no block length that works at this rate"
        )

    lo, hi = 1.0, cap
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if net(mid) > 0.0:
            hi = mid
        else:
            lo = mid

    return hi


def catalogue():
    """
    Every model here, as (name, callable, parameters), in three groups:
    efficiencies and capacities, then bit counts, then failure probabilities.
    """

    return (
        ("awgn_capacity", awgn_capacity, ("snr",)),
        ("biawgn_capacity", biawgn_capacity, ("snr",)),
        ("efficiency", efficiency, ("rate", "snr")),
        ("inefficiency", inefficiency, ("rate", "qber")),
        ("capacity_ratio", capacity_ratio, ("rate", "qber")),
        ("bridge", bridge, ("qber", "f_ec", "beta")),
        ("leak_ratio", leak_ratio, ("rate", "fer")),
        ("charged", charged, ("f_ec", "qber", "fer")),
        ("code_beta", code_beta, ("code", "snr")),
        ("cascade_point", cascade_point, ("qber",)),
        ("blocks", blocks, ("qber", "frame", "variant")),
        ("run_cascade", run_cascade, ("qber", "frames", "seed")),
        ("frame_bound", frame_bound, ("frames", "failed", "conf")),
        ("verify_failure", verify_failure, ("fer", "eps_hash")),
        ("hash_length", hash_length, ("h_min", "eps")),
        ("amplify_cost", amplify_cost, ("eps",)),
        ("hash_charge", hash_charge, ("family", "eps")),
        ("hash_distance", hash_distance, ("ell", "h_min")),
        ("toeplitz_seed", toeplitz_seed, ("n", "ell")),
        ("poly_collision", poly_collision, ("chunks", "bits")),
        ("tag_failure", tag_failure, ("message", "tag")),
        ("lfsr_failure", lfsr_failure, ("message", "tag", "seed")),
        ("tag_length", tag_length, ("message", "eps")),
        ("auth_cost", auth_cost, ("lengths", "eps")),
        ("auth_failure", auth_failure, ("lengths", "eps")),
        ("round_cost", round_cost, ("sift", "ell", "announce")),
        ("net_length", net_length, ("gross", "costs")),
        ("net_rate", net_rate, ("gross", "sift", "cadence", "announce")),
        ("block_floor", block_floor, ("per_bit", "announce")),
    )
