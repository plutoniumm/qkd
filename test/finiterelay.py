import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.cache import memo
from kit.checks import Guarded
from kit.forms import fitslope, h2, poisson
from qkd import _core, topology

# Three published finite-key analyses, one per family, each a different SHAPE of finite-size
# accounting rather than the same machinery re-parameterised:
#
#   MDI-BB84       Curty, Xu, Cui, Lim, Tamaki & Lo, Nature Communications 5, 3732 (2014),
#                  arXiv:1307.1081. A TWO-DIMENSIONAL decoy table -- a 3x3 grid over joint
#                  intensity settings -- inverted by Gaussian elimination, with the
#                  multiplicative Chernoff bound stated in the OBSERVED value and Serfling's
#                  inequality carrying the test basis to the code string.
#   RRDPS          Takesue, Sasaki, Tamaki & Koashi, Nature Photonics 9, 827 (2015),
#                  arXiv:1505.07914, Methods. NO channel parameter is estimated at all: the
#                  privacy amplification reads the packet length and the source's photon
#                  statistics alone, leaving three binomial tails at exactly known rates, and
#                  the bit error rate reaches the length only as leakage.
#   Mode pairing   Xie, Lu, Weng, Cao, Jia, Bao, Wang, Fu, Yin & Chen, PRX Quantum 3, 020315
#                  (2022), arXiv:2112.11635. Chernoff in both directions -- expected to
#                  observed and back -- with a third published random-sampling transfer.
#
# The three sampling transfers are Lim's gamma, Curty's Upsilon and Xie's gamma^U, and they
# are not interchangeable: each belongs to its own key-length formula.

# Curty's simulation setting, from his Discussion: relay detection efficiency 14.5%,
# background count rate 6.02e-6, misalignment 1.5%, 0.2 dB/km, zeta = 1.16, weakest decoy
# 5e-4 ("difficult to generate a vacuum state due to imperfect extinction"), eps = 1e-10 and
# eps_cor = 1e-15. DARK is per detector, half the quoted background, as `test/mdi.py` records.
ALPHA = 0.2
DET = 0.145
MISALIGN = 0.015
FEC = 1.16
DARK = 6.02e-6 / 2.0
EPS_MDI = 1e-10
EPS_COR = 1e-15

# Intensities and their emission probabilities, per party. The signal and first decoy are
# this exam's choice; the 5e-4 floor is Curty's.
INTS = [0.3, 0.1, 5e-4]
SETS = (0.7, 0.2, 0.1)

# Each party picks the key basis half the time, so one basis carries a quarter of the pulse
# pairs; nine tenths of the signal-setting announcements form the code string.
BASIS = 0.5
CODE = 0.9

# Photon-number truncation of the forward model: the Poisson tail past twelve photons at
# mean 0.3 is below 1e-19.
CUT = 12

# RRDPS setting, from `test/rrdps.py`: Yin et al.'s dark rate and clean misalignment, with a
# short metro transmittance and an intensity that puts the packet's Poisson mean near one.
L_TRAIN, NU_TH = 32, 2
MU_TRAIN, ETA_TRAIN = 0.005, 0.1
D_TRAIN, E_TRAIN = 1e-6, 0.015
F_TRAIN = 1.1
D_SEC = 2.0**-50

# Takesue's own five values, fixed by hand in his Methods to reach d <= 2^-50: eps_1 = 2^-50,
# eps_2 = eps_3 = eta_x = 2^-103/3 and eta_z = 2^-51.
TAKESUE = (2.0**-50, 2.0**-103 / 3.0, 2.0**-103 / 3.0, 2.0**-103 / 3.0, 2.0**-51)

# Mode-pairing setting, from `test/pairing.py`: Xie's Table 1 hardware, his pi/10 X-basis
# misalignment angle, and the pairing interval his 625 MHz interference measurement supports.
A_PAIR, DET_PAIR = 0.165, 0.70
DARK_PAIR, F_PAIR = 1e-8, 1.1
SPAN = 10**4


def slope(totals, gaps):
    """
    Least-squares log-log slope of a gap against block size.
    """
    xs = [math.log10(t) for t in totals]

    return fitslope(xs, [math.log10(g) for g in gaps])


def arms(km):
    """
    Per-arm transmittance with the relay at the midpoint of a total span of km.
    """

    return DET * 10.0 ** (-ALPHA * (km / 2.0) / 10.0)


def relay(eta_a, eta_b, dark, mis):
    """
    Ma & Razavi Eqs. (21) and (23) at arbitrary arm transmittances, written here rather than called
    so the estimators are checked against arithmetic.
    """
    core = eta_a * eta_b / 2.0
    one = (2.0 * eta_a + 2.0 * eta_b - 3.0 * eta_a * eta_b) * dark
    two = 4.0 * (1.0 - eta_a) * (1.0 - eta_b) * dark * dark
    total = core + one + two
    if total <= 0.0:
        return 0.0, 0.5

    live = (1.0 - dark) ** 2

    return live * total, min((mis * core + 0.5 * (one + two)) / total, 0.5)


@memo
def model(km):
    """
    The cached photon-number-resolved forward model at one span; n photons through one arm are read
    as one mode of transmittance 1 - (1 - eta)^n, exact at n = 1 and a stand-in above it.
    """
    eta = arms(km)
    hit = [1.0 - (1.0 - eta) ** n for n in range(CUT + 1)]
    span = range(CUT + 1)
    ys = [[relay(hit[n], hit[m], DARK, MISALIGN) for m in span] for n in span]
    taus = [[0.0] * (CUT + 1) for _ in range(CUT + 1)]
    probs, gain, errs = [], [], []
    for i, a in enumerate(INTS):
        for j, b in enumerate(INTS):
            p = BASIS * BASIS * SETS[i] * SETS[j]
            q, e = 0.0, 0.0
            for n in range(CUT + 1):
                for m in range(CUT + 1):
                    w = poisson(a, n) * poisson(b, m)
                    taus[n][m] += p * w
                    q += w * ys[n][m][0]
                    e += w * ys[n][m][0] * ys[n][m][1]

            probs.append(p)
            gain.append(p * q)
            errs.append(p * e)

    return ys, taus, probs, gain, errs


def truth(km):
    """
    (single-photon-pair share per emitted pulse pair, its yield, its error rate, the signal-setting
    announcement probability, the signal-setting error rate).
    """
    ys, taus, probs, gain, errs = model(km)
    y11, e11 = ys[1][1]

    return taus[1][1] * y11, y11, e11, gain[0] / probs[0], errs[0] / gain[0]


def bounds(total, km, eps=EPS_MDI):
    """
    (n0, n1, phi, n_key) at one block of emitted pulse pairs and one span.
    """
    _, _, probs, gain, errs = model(km)
    share = _core.mdi_eps(eps)
    key = [total * g for g in gain]
    faults = [total * e for e in errs]
    n_key = CODE * key[0]
    n0 = _core.mdi_vacuum(INTS, INTS, probs, key, n_key, share)
    n1 = _core.mdi_single(INTS, INTS, probs, key, n_key, share)
    nbar = _core.mdi_pairs(INTS, INTS, probs, key, share)
    ebar = _core.mdi_faults(INTS, INTS, probs, faults, share)

    return n0, n1, _core.mdi_phase(n1, nbar, ebar, share), n_key


def length(total, km, eps=EPS_MDI):
    """
    Curty's key length in bits for one announced Bell state.
    """
    n0, n1, phi, n_key = bounds(total, km, eps)

    return _core.mdi_length(n0, n1, phi, n_key, truth(km)[4], FEC, eps, EPS_COR)


def sifted(n_em):
    """
    (sifted bits, bit error rate) of the RRDPS receiver for a block of emitted packets.
    """
    q, e_bit = _core.rrdps_counts(L_TRAIN, MU_TRAIN, ETA_TRAIN, D_TRAIN, E_TRAIN)

    return q * n_em, e_bit


def train(n_em, e_bit=None, doubles=0.0):
    """
    The RRDPS finite-key chain for a block of emitted packets.
    """
    count, seen = sifted(n_em)

    return _core.rrdps_finite(
        n_em,
        count,
        doubles,
        L_TRAIN,
        MU_TRAIN,
        NU_TH,
        seen if e_bit is None else e_bit,
        F_TRAIN,
        D_SEC,
        0.125,
    )


def station(km):
    """
    (pairs per round, sifted Z-pair share, single-pair fraction, phase error, Z-basis bit error) of
    the mode-pairing station at intensity one half.
    """
    eta = _core.decoy_eta(A_PAIR, km / 2.0, DET_PAIR)
    click = _core.pairing_click(eta, 0.5, DARK_PAIR)
    pairs = _core.pairing_pairs(click, SPAN)
    sift, e_z = _core.pairing_sift(eta, 0.5, DARK_PAIR)
    q11 = _core.pairing_single(eta, 0.5, DARK_PAIR)
    e_x = _core.mdi_yield(eta, eta, DARK_PAIR, 0.0244)[1]

    return pairs, sift, q11, e_x, e_z


def sender(bias):
    """
    One keyed weak-coherent sender; `bias` is the key-basis probability the finite chain reads, None
    what the asymptotic path takes.
    """

    return q.Sender(
        modulation=q.BasisKeying(
            decoy=q.Decoy(intensities=tuple(INTS), probs=SETS),
            sift=1.0,
            bias=bias,
        )
    )


def bell_swap(km, bias, security, **bell):
    """
    Two senders at `bias`, two fibre arms and a four-detector Bell analyser at the middle of a span
    of km.
    """

    return q.Swap(
        alice=sender(bias),
        bob=sender(bias),
        relay=q.Relay(
            bell=q.BellAnalyser(
                eta=DET,
                dark=DARK,
                misalign=MISALIGN,
                misalign_test=MISALIGN,
                **bell,
            )
        ),
        channels=(q.Fiber(length=km / 2.0, alpha=ALPHA),) * 2,
        security=security,
    )


def midpoint(km, total=None, states=2, fer=None):
    """
    The q.Swap the finite branch runs through; `total` is emitted pulse PAIRS, None the asymptotic
    rate.
    """
    block = (
        None
        if total is None
        else q.RelayBlock(
            n=total,
            eps_sec=EPS_MDI,
            eps_cor=EPS_COR,
            code=CODE,
            fer=fer,
        )
    )
    bias = None if total is None else BASIS

    return bell_swap(km, bias, q.TestBasisBound(f=FEC, block=block), states=states)


def covariance():
    """
    The continuous-variable relay, the other branch q.Swap labels an attack row on.
    """

    return q.Swap(
        alice=q.Sender(modulation=q.GaussianModulation(v_a=5.0)),
        bob=q.Sender(modulation=q.GaussianModulation(v_a=5.0)),
        relay=q.Relay(bell=q.BellDetector(eta=0.98, v_el=0.01)),
        channels=(q.Channel(T=0.9, xi=0.002, ref="input"),) * 2,
    )


def wired(total, km, states=2):
    """
    (n0, n1, phi, code string, length per announced Bell state) rebuilt from _core over the same two
    grids, which are DIFFERENT BASES: the vacuum and single-pair counts read the rectilinear one,
    the pair and error counts the diagonal one.
    """
    eta = arms(km)
    each = total / 2.0
    zed = [_core.mdi_rect(x, y, eta, eta, DARK, MISALIGN) for x in INTS for y in INTS]
    diag = [_core.mdi_diag(x, y, eta, eta, DARK, MISALIGN) for x in INTS for y in INTS]
    p_key = [BASIS * BASIS * SETS[i] * SETS[j] for i in range(3) for j in range(3)]
    p_test = [(1.0 - BASIS) ** 2 * SETS[i] * SETS[j] for i in range(3) for j in range(3)]
    keys = [each * p * g for p, (g, _) in zip(p_key, zed)]
    tests = [each * p * g for p, (g, _) in zip(p_test, diag)]
    faults = [each * p * g * e for p, (g, e) in zip(p_test, diag)]
    eps = EPS_MDI / states
    share = _core.mdi_eps(eps)
    kept = CODE * keys[0]
    n0 = _core.mdi_vacuum(INTS, INTS, p_key, keys, kept, share)
    n1 = _core.mdi_single(INTS, INTS, p_key, keys, kept, share)
    nbar = _core.mdi_pairs(INTS, INTS, p_test, tests, share)
    ebar = _core.mdi_faults(INTS, INTS, p_test, faults, share)
    phi = _core.mdi_phase(n1, nbar, ebar, share)

    return (
        n0,
        n1,
        phi,
        kept,
        _core.mdi_length(n0, n1, phi, kept, zed[0][1], FEC, eps, EPS_COR),
    )


class MdiChernoff(Question):
    """
    Curty's Claim 3, the multiplicative Chernoff deviation of an observed count.
    """

    def test_branch_ladder(self):
        """
        The reported branch falls from the Hoeffding fallback at a count of 1e2 to the fully
        multiplicative form at 1e6.
        """
        share = _core.mdi_eps(EPS_MDI)
        seen = [_core.mdi_deviate(x, 1e8, share)[2] for x in (1e2, 1e3, 1e4, 1e6)]
        self.assertMonotone(seen, rising=False, strict=False, msg=f"the branch must fall: {seen}")
        self.assertEqual(seen[-1], 1, msg="a large count takes both multiplicative tails")
        self.assertEqual(seen[0], 6, msg="a small count falls back to Hoeffding twice")

    def test_tail_asymmetry(self):
        """
        On the multiplicative branch the lower deviation exceeds the upper by the ratio of the two
        logarithms, being taken at eps^4/16 and eps^(3/2).
        """
        share = _core.mdi_eps(EPS_MDI)
        lo, hi, branch = _core.mdi_deviate(1e6, 1e8, share)
        self.assertEqual(branch, 1, msg="this count is on the multiplicative branch")
        self.assertGreater(lo, hi, msg=f"the lower tail must be wider: {lo} vs {hi}")

        beta = math.log(1.0 / share)
        want = math.sqrt((math.log(16.0) + 4.0 * beta) / (1.5 * beta))
        self.assertClose(lo / hi, want, atol=1e-12, msg="the ratio is the two logarithms")

    def test_width_scaling(self):
        """
        The deviation's share of the observed count falls with log-log slope -0.5.
        """
        share = _core.mdi_eps(EPS_MDI)
        counts = (1e6, 1e8, 1e10, 1e12)
        rel = [_core.mdi_deviate(x, 10.0 * x, share)[0] / x for x in counts]
        self.assertMonotone(rel, rising=False, msg=f"the share must fall: {rel}")
        self.assertClose(
            slope(counts, rel),
            -0.5,
            atol=1e-9,
            msg="the width closes as one over sqrt(x)",
        )

    def test_count_guard(self):
        """
        An observed count above the number of trials it was drawn from is refused.
        """

        self.assertFails(
            ValueError,
            "x must not exceed n",
            _core.mdi_deviate,
            10.0,
            5.0,
            1e-12,
            msg="more successes than trials is refused",
        )


class MdiEstimator(Question):
    """
    The two-dimensional decoy inversion Curty's key length is built from.
    """

    def test_model_agrees(self):
        """
        CROSS-ENGINE. The exam's own forward model reproduces mdi_yield at one photon from each
        party.
        """
        eta = arms(50.0)
        mine = relay(eta, eta, DARK, MISALIGN)
        theirs = _core.mdi_yield(eta, eta, DARK, MISALIGN)
        self.assertClose(mine[0], theirs[0], msg="the yields must agree")
        self.assertClose(mine[1], theirs[1], msg="the error rates must agree")

    def test_pair_bound(self):
        """
        The certified single-photon-pair count rises with the block, stays below the
        infinite-decoy ceiling tau_11 times Y_11, and closes on its zero-fluctuation limit as
        one over the square root of the block: exponent -0.5004 over four decades from 1e16.
        Below about 1e15 the clamp at zero and the switching maximum both bite and the exponent
        is not a power law.
        """
        want = truth(50.0)[0]
        _, _, probs, gain, _ = model(50.0)
        share = _core.mdi_eps(EPS_MDI)
        totals = (1e12, 1e14, 1e16)
        seen = [_core.mdi_pairs(INTS, INTS, probs, [t * g for g in gain], share) / t for t in totals]
        self.assertMonotone(seen, rising=True, msg=f"the share must rise: {seen}")

        for got in seen:
            self.assertLess(got, want, msg=f"the pair share {got} is above {want}")

        big = 1e22
        limit = _core.mdi_pairs(INTS, INTS, probs, [big * g for g in gain], share) / big
        wide = (1e16, 1e17, 1e18, 1e19)
        gaps = [limit - _core.mdi_pairs(INTS, INTS, probs, [t * g for g in gain], share) / t for t in wide]

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"the limit must sit above: {gap}")
        self.assertClose(slope(wide, gaps), -0.5, atol=0.01, msg=f"exponent {slope(wide, gaps)}")

    def test_fault_ceiling(self):
        """
        The certified single-pair error count falls with the block and stays above the true count.
        """
        ys, taus, probs, _, errs = model(50.0)
        want = taus[1][1] * ys[1][1][0] * ys[1][1][1]
        totals = (1e12, 1e14, 1e16)
        seen = [_core.mdi_faults(INTS, INTS, probs, [t * e for e in errs], _core.mdi_eps(EPS_MDI)) / t for t in totals]
        self.assertMonotone(seen, rising=False, msg=f"the count must fall: {seen}")

        for got in seen:
            self.assertGreater(got, want, msg=f"the error share {got} is below {want}")

    def test_phase_ceiling(self):
        """
        The phase-error bound falls with the block and never sits below the single-pair error rate
        in the channel.
        """
        want = truth(50.0)[2]
        seen = [bounds(t, 50.0)[2] for t in (1e12, 1e14, 1e16)]
        self.assertMonotone(seen, rising=False, msg=f"phi must fall: {seen}")

        for phi in seen:
            self.assertGreater(phi, want, msg=f"phi {phi} is below e_11 {want}")

    def test_empty_test(self):
        """
        With no certified test sample the phase bound reads one half rather than dividing by zero.
        """

        self.assertClose(
            _core.mdi_phase(1e6, 0.0, 0.0, 1e-12),
            0.5,
            msg="an empty test basis certifies nothing",
        )


class MdiLength(Question):
    """
    Curty's key length per announced Bell state.
    """

    def test_length_units(self):
        """
        The length is whole bits, floored rather than rounded so no uncertified bit is claimed,
        and zero on a block too short.
        """
        out = length(1e14, 50.0)
        self.assertGreater(out, 0.0, msg=f"this block must certify a key: {out}")
        self.assertClose(out, math.floor(out), msg="a key length is whole bits")
        self.assertClose(length(1e8, 50.0), 0.0, msg="a short block certifies none")

    def test_block_monotone(self):
        """
        The length and the length per emitted pulse pair both rise with the block size and fall
        with the span, to zero at 250 km. MidpointBlock.test_block_reach is the same pair
        through q.Swap.
        """
        totals = (1e13, 1e14, 1e15, 1e16, 1e17)
        lens = [length(t, 50.0) for t in totals]
        rates = [lens[i] / totals[i] for i in range(len(totals))]
        self.assertMonotone(lens, rising=True, msg=f"length must rise: {lens}")
        self.assertMonotone(rates, rising=True, msg=f"rate must rise: {rates}")

        spans = [length(1e15, km) for km in (25.0, 50.0, 100.0, 150.0, 250.0)]
        self.assertMonotone(spans, rising=False, strict=False, msg=f"length must fall: {spans}")
        self.assertClose(spans[-1], 0.0, msg="250 km certifies nothing at this block")

    def test_epsilon_price(self):
        """
        A weaker secrecy parameter buys a longer key.
        """
        lens = [length(1e13, 50.0, eps=e) for e in (1e-15, 1e-10, 1e-5)]
        self.assertMonotone(lens, rising=True, msg=f"a weaker eps pays: {lens}")

    def test_rate_ceiling(self):
        """
        CROSS-ENGINE. The finite length per code bit, less its vacuum term, rises toward mdi_rate
        over the signal-setting gain without exceeding it, at exponent -0.5001 over four decades
        from 1e16.
        """
        _, y11, e11, q_z, e_z = truth(50.0)
        q11 = _core.mdi_gain(y11, INTS[0], INTS[0])
        want = _core.mdi_rate(q11, e11, q_z, e_z, FEC) / q_z
        totals = (1e13, 1e14, 1e15, 1e16)
        seen = []
        for total in totals:
            n0, n1, phi, n_key = bounds(total, 50.0)
            seen.append((length(total, 50.0) - n0) / n_key)
        self.assertMonotone(seen, rising=True, msg=f"the rate must rise: {seen}")

        for got in seen:
            self.assertLess(got, want, msg=f"the finite rate {got} is above {want}")

        big = 1e22
        limit = length(big, 50.0) / (CODE * big * model(50.0)[3][0])
        wide = (1e16, 1e17, 1e18, 1e19)
        gaps = []
        for total in wide:
            gaps.append(limit - length(total, 50.0) / bounds(total, 50.0)[3])

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"the limit must sit above: {gap}")
        self.assertClose(slope(wide, gaps), -0.5, atol=0.01, msg=f"exponent {slope(wide, gaps)}")


class MdiBudget(Guarded):
    """
    The epsilon composition and what the estimators refuse.
    """

    def test_eps_share(self):
        """
        The per-bound failure probability is the composed per-state secrecy parameter over 266.
        """

        self.assertClose(_core.mdi_eps(266e-12), 1e-12, atol=1e-24, msg="266 shares of one epsilon")
        self.assertLess(_core.mdi_eps(EPS_MDI), EPS_MDI, msg="the share is below the composed value")

    def test_grid_guard(self):
        """
        A grid that is not three by three, and intensities that do not decrease strictly, are
        refused.
        """
        ok = (INTS, INTS, [0.1] * 9, [1e3] * 9, 1e-12)
        self.assertSlots(
            _core.mdi_pairs,
            ok,
            (
                (0, "decrease strictly", ([0.1, 0.3, 0.0],)),
                (2, "3x3 grid", ([0.1] * 8,)),
                (3, "3x3 grid", ([1e3] * 4,)),
            ),
            msg="mdi_pairs",
        )

    def test_count_guard(self):
        """
        Certified vacuum and single-pair counts above the code string, and a code string longer
        than the announcements it is drawn from, are both refused.
        """

        self.assertFails(
            ValueError,
            "n0 + n1 must not exceed n_key",
            _core.mdi_length,
            600.0,
            600.0,
            0.02,
            1000.0,
            0.02,
            FEC,
            EPS_MDI,
            EPS_COR,
            msg="n0 + n1 above n_key accepted",
        )
        self.assertFails(
            ValueError,
            "n_key must not exceed",
            _core.mdi_vacuum,
            INTS,
            INTS,
            [0.1] * 9,
            [1e3] * 9,
            1e4,
            1e-12,
            msg="n_key above the announcements accepted",
        )


class RrdpsTails(Question):
    """
    The three binomial tails RRDPS's analysis is made of, and its epsilon split.
    """

    def test_split_anchor(self):
        """
        ANCHOR. The epsilon assignment at d = 2^-50 returns Takesue's own five values.
        """
        eps1, eps2, eps3, s_x, s_z = _core.rrdps_split(D_SEC)
        got = (eps1, eps2, eps3, 2.0**-s_x, 2.0**-s_z)

        for mine, want in zip(got, TAKESUE):
            self.assertClose(mine, want, atol=1e-40, msg=f"{mine} against {want}")

    def test_secpar_target(self):
        """
        Those five values compose back to the target exactly, both branches of the maximum landing
        on it.
        """
        eps1, eps2, eps3, s_x, s_z = _core.rrdps_split(D_SEC)
        self.assertClose(
            _core.rrdps_secpar(eps1, eps2, eps3, s_x, s_z),
            D_SEC,
            atol=1e-30,
            msg="the composition must land on d",
        )
        self.assertClose(
            _core.rrdps_secpar(*TAKESUE[:3], -math.log2(TAKESUE[3]), -math.log2(TAKESUE[4])),
            D_SEC,
            atol=1e-30,
            msg="Takesue's own five values compose to 2^-50",
        )

    def test_tagged_constant(self):
        """
        With no double clicks observed the multiphoton threshold is the constant
        log2(1/eps)/log2(8/7).
        """
        eps = _core.rrdps_split(D_SEC)[0]
        want = math.log2(1.0 / eps) / math.log2(1.0 / (1.0 - 0.125))
        got = _core.rrdps_tagged(0.0, 0.125, eps)
        self.assertClose(got, math.ceil(want), atol=1e-9, msg=f"{got} against {want}")
        self.assertGreater(
            _core.rrdps_tagged(1e4, 0.125, eps),
            8e4,
            msg="an observed double-click count costs at least eight times itself",
        )

    def test_above_monotone(self):
        """
        The upper-tail threshold rises with the population and with a tighter eps, and its excess
        over the mean closes as one over the square root of the population.
        """
        seen = [_core.rrdps_above(n, 0.01, 1e-10) for n in (1e6, 1e8, 1e10)]
        self.assertMonotone(seen, rising=True, msg=f"the threshold must rise: {seen}")

        for n, got in zip((1e6, 1e8, 1e10), seen):
            self.assertGreater(got, 0.01 * n, msg=f"{got} is below the mean {0.01 * n}")

        tighter = [_core.rrdps_above(1e8, 0.01, e) for e in (1e-5, 1e-10, 1e-20)]
        self.assertMonotone(tighter, rising=True, msg=f"a tighter eps costs: {tighter}")

        totals = (1e8, 1e10, 1e12, 1e14)
        gaps = [_core.rrdps_above(n, 0.01, 1e-10) / n - 0.01 for n in totals]
        self.assertClose(slope(totals, gaps), -0.5, atol=0.02, msg=f"exponent {slope(totals, gaps)}")

    def test_vacuous_tail(self):
        """
        A population too small for any threshold to satisfy the tail returns the whole population.
        """

        self.assertClose(_core.rrdps_above(10.0, 0.4, 1e-30), 10.0, msg="every packet may be tagged")


class RrdpsLength(Question):
    """
    Takesue's key length.
    """

    def test_length_units(self):
        """
        The length is whole bits, never negative, and zero on a block too short to certify.
        """
        out = train(1e9)[0]
        self.assertGreater(out, 0.0, msg=f"this block must certify a key: {out}")
        self.assertClose(out, math.floor(out), msg="a key length is whole bits")
        self.assertClose(train(1e4)[0], 0.0, msg="a short block certifies none")

    def test_block_monotone(self):
        """
        The length and the length per emitted packet both rise with the block, the two hash
        terms and the three thresholds all being amortised.
        """
        totals = (1e7, 1e8, 1e9, 1e10, 1e11)
        lens = [train(t)[0] for t in totals]
        rates = [lens[i] / totals[i] for i in range(len(totals))]
        self.assertMonotone(lens, rising=True, msg=f"length must rise: {lens}")
        self.assertMonotone(rates, rising=True, msg=f"rate must rise: {rates}")

    def test_leak_independent(self):
        """
        Changing the bit error rate moves no threshold, so the whole change in length is the leakage
        f*n_sift*(h2(0.05) - h2(0.01)).
        """
        low = train(1e10, e_bit=0.01)
        high = train(1e10, e_bit=0.05)

        for i in (2, 3, 4):
            self.assertClose(low[i], high[i], msg=f"threshold {i} must not move")

        count = sifted(1e10)[0]
        want = F_TRAIN * count * (h2(0.05) - h2(0.01))
        self.assertClose(low[0] - high[0], want, atol=1.5, msg=f"{low[0] - high[0]} against {want}")

    def test_rate_converge(self):
        """
        CROSS-ENGINE. The length per sifted bit closes from below on 1 - f h2(e_bit) - h2(e_ph) at
        exponent -0.5006 over three decades from 1e9.
        """
        count, e_bit = sifted(1e12)
        q = count / 1e12
        e_ph = _core.rrdps_phase(q, L_TRAIN, MU_TRAIN, NU_TH)
        want = 1.0 - F_TRAIN * h2(e_bit) - h2(e_ph)
        totals = (1e9, 1e10, 1e11, 1e12)
        gaps = [want - train(t)[0] / sifted(t)[0] for t in totals]

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"the finite length must sit below: {gap}")
        self.assertClose(slope(totals, gaps), -0.5, atol=0.05, msg=f"exponent {slope(totals, gaps)}")

    def test_double_price(self):
        """
        Observed double clicks raise the multiphoton threshold and shorten the length.
        """
        clean = train(1e10)
        dirty = train(1e10, doubles=1e3)
        self.assertGreater(dirty[2], clean[2], msg="the threshold must rise")
        self.assertLess(dirty[0], clean[0], msg="the length must fall")

    def test_sift_guard(self):
        """
        More sifted bits than emitted packets is refused.
        """

        self.assertFails(
            ValueError,
            "n_sift must not exceed n_em",
            _core.rrdps_finite,
            1e6,
            2e6,
            0.0,
            L_TRAIN,
            MU_TRAIN,
            NU_TH,
            0.02,
            F_TRAIN,
            D_SEC,
            0.125,
            msg="n_sift above n_em accepted",
        )


class PairingFinite(Question):
    """
    Xie's statistical layer and key length for mode pairing.
    """

    def test_eps_budget(self):
        """
        ANCHOR. eps_sec = 24 eps in the symmetric case, the paper's printed 2.4e-9 at eps = 1e-10.
        """

        self.assertClose(_core.pairing_eps(2.4e-9), 1e-10, atol=1e-22, msg="24 shares of one epsilon")
        self.assertClose(24.0 * 1e-10, 2.4e-9, atol=1e-22, msg="the printed symmetric bound")

    def test_chernoff_pair(self):
        """
        The two Chernoff directions bracket a count the right way round and are near inverses, the
        variant interval being the wider at the same count.
        """
        lo, hi = _core.pairing_observe(1e6, 1e-10)
        self.assertLess(lo, 1e6, msg=f"the lower observation must sit below: {lo}")
        self.assertGreater(hi, 1e6, msg=f"the upper observation must sit above: {hi}")
        self.assertClose(
            _core.pairing_expect(lo, 1e-10)[1],
            1e6,
            atol=1.0,
            msg="round trip off the mean",
        )
        self.assertLess(_core.pairing_expect(hi, 1e-10)[0], 1e6, msg="the round trip must contain")

        variant = _core.pairing_expect(1e6, 1e-10)
        self.assertGreater(
            variant[1] - variant[0],
            hi - lo,
            msg=f"{variant} must be wider than {(lo, hi)}",
        )

    def test_gamma_shrink(self):
        """
        The random-sampling penalty falls as the two certified counts grow and is positive at every
        finite pair of them.
        """
        seen = [_core.pairing_gamma(n, n, 0.03, 1e-10) for n in (1e4, 1e6, 1e8, 1e10)]
        self.assertMonotone(seen, rising=False, msg=f"the penalty must fall: {seen}")

        for got in seen:
            self.assertGreater(got, 0.0, msg=f"the penalty must be positive: {got}")

    def test_gamma_empty(self):
        """
        A test sample showing no errors certifies nothing, the penalty saturating at one half where
        Xie's transfer differs from Lim's.
        """

        self.assertClose(
            _core.pairing_gamma(1e8, 1e8, 0.0, 1e-10),
            0.5,
            msg="a sample with no errors certifies nothing",
        )
        self.assertClose(
            _core.pairing_phase(1e8, 1e8, 0.0, 1e-10),
            0.5,
            msg="and the phase bound follows it",
        )

    def test_yield_ceiling(self):
        """
        The one-sided decoy inversion certifies a positive yield below the single-photon yield it
        bounds.
        """
        eta = _core.decoy_eta(A_PAIR, 200.0 / 2.0, DET_PAIR)
        nu, mu = 0.05, 0.4
        rates = [_core.pairing_click(eta, x, DARK_PAIR) if x > 0.0 else 2.0 * DARK_PAIR for x in (nu, mu, 0.0)]
        got = _core.pairing_yield(nu, mu, rates[0], rates[1], rates[2])
        self.assertGreater(got, 0.0, msg=f"the bound must certify something: {got}")
        self.assertLess(got, eta + 2.0 * DARK_PAIR, msg=f"{got} is above the true yield")

    def test_length_converge(self):
        """
        CROSS-ENGINE. The length per sifted Z-pair closes from below on pairing_rate at exponent
        -1.0001 over three decades from 1e8, the gap being the epsilon budget alone.
        """
        pairs, sift, q11, e_x, e_z = station(300.0)
        want = _core.pairing_rate(pairs, sift, q11, e_x, e_z, F_PAIR) / (pairs * sift)
        totals = (1e8, 1e9, 1e10, 1e11)
        gaps = []
        for n_z in totals:
            got = _core.pairing_length(0.0, q11 * n_z, e_x, n_z, e_z, F_PAIR, 2.4e-9, 1e-15)
            gaps.append(want - got / n_z)

        for gap in gaps:
            self.assertGreater(gap, 0.0, msg=f"the finite length must sit below: {gap}")
        self.assertClose(slope(totals, gaps), -1.0, atol=0.01, msg=f"exponent {slope(totals, gaps)}")


class MidpointBlock(Question):
    """
    Curty's length reached through q.Swap, summed over announced Bell states.
    """

    def test_chain_agrees(self):
        """
        CROSS-ENGINE. The counts, phase bound, code string and per-state length the layer reports
        are the ones _core returns over the same grids.
        """
        got = midpoint(50.0, 1e15).run()
        n0, n1, phi, kept, each = wired(1e15, 50.0)
        self.assertClose(got.n0, n0, msg=f"vacuum count {got.n0} against {n0}")
        self.assertClose(got.n1, n1, msg=f"single-pair count {got.n1} against {n1}")
        self.assertClose(got.phi, phi, msg=f"phase bound {got.phi} against {phi}")
        self.assertClose(got.n_key, kept, msg=f"code string {got.n_key} against {kept}")
        self.assertClose(
            got.explain["length_state"]["value"],
            each,
            msg="length_state",
        )

    def test_state_sum(self):
        """
        The length is summed over announced Bell states, two of them running at half the secrecy
        parameter.
        """
        got = midpoint(50.0, 1e15).run()
        info = got.explain
        self.assertEqual(info["states"]["value"], 2, msg="the announcement count moved")
        self.assertClose(
            info["eps_state"]["value"],
            EPS_MDI / 2.0,
            msg="eps_state",
        )
        self.assertClose(
            info["eps_bound"]["value"],
            _core.mdi_eps(EPS_MDI / 2.0),
            msg="eps_bound",
        )
        self.assertClose(
            got.key_length,
            2.0 * info["length_state"]["value"],
            msg=f"key_length {got.key_length}",
        )

    def test_states_kept(self):
        """
        q.BellAnalyser(states=...) says how many announcement classes the relay KEEPS, not how the
        counts are split.
        """
        both = midpoint(50.0, 1e15, states=2).run()
        alone = midpoint(50.0, 1e15, states=1).run()
        self.assertClose(
            alone.explain["n_class"]["value"],
            both.explain["n_class"]["value"],
            msg="n_class",
        )
        self.assertClose(
            alone.explain["length_state"]["value"],
            wired(1e15, 50.0, states=1)[4],
            msg="length_state at states=1",
        )

        ratio = both.key_length / alone.key_length
        self.assertGreater(ratio, 1.9, msg=f"keeping both bought only {ratio}")
        self.assertLess(ratio, 2.0, msg=f"keeping both bought more than double: {ratio}")

    def test_rate_over_block(self):
        """
        key_rate is the summed length in whole bits over the emitted pulse PAIRS, and sits below the
        asymptotic rate.
        """
        got = midpoint(50.0, 1e15).run()
        self.assertClose(got.key_rate, got.key_length / 1e15, msg="rate is length over the block")
        self.assertLess(
            got.key_rate,
            midpoint(50.0).run().key_rate,
            msg=f"finite rate {got.key_rate}",
        )
        self.assertClose(got.key_length, math.floor(got.key_length), msg="a length is whole bits")

    def test_block_reach(self):
        """
        The length rises with the block and falls with the span through the layer, to zero at 200
        km.
        """
        lens = [midpoint(50.0, n).run().key_length for n in (1e14, 1e15, 1e16, 1e17)]
        self.assertMonotone(lens, rising=True, msg=f"length must rise: {lens}")

        spans = [midpoint(km, 1e16).run().key_length for km in (25.0, 50.0, 100.0, 200.0)]
        self.assertMonotone(spans, rising=False, strict=False, msg=f"length must fall: {spans}")
        self.assertClose(spans[-1], 0.0, msg="200 km certifies nothing at this block")

    def test_branch_shown(self):
        """
        The report names Claim 3's branch in all nine cells, the vacuum corner staying on the
        Hoeffding fallback at any block size.
        """
        seen = [midpoint(50.0, n).explain()["branch_key"]["value"] for n in (1e11, 1e15)]
        fell = [sum(1 for b in row if b == 6) for row in seen]
        self.assertEqual(len(seen[0]), 9, msg="nine cells, row-major")
        self.assertEqual(seen[0][0], 1, msg="the signal cell is never the short one")
        self.assertEqual(seen[0][-1], 6, msg="the vacuum corner must fall back")
        self.assertEqual(seen[-1][-1], 6, msg="and never leaves the fallback")
        self.assertMonotone(fell, rising=False, msg=f"fallbacks must fall: {fell}")

    def test_frame_error(self):
        """
        A declared frame error rate scales the whole summed length.
        """
        base = midpoint(50.0, 1e16).run().key_length
        cut = midpoint(50.0, 1e16, fer=0.1).run().key_length
        self.assertClose(cut, 0.9 * base, msg=f"{cut} is not 0.9 of {base}")

    def test_forward_apart(self):
        """
        mdi_yield is the forward model and is CALLED nowhere in the layer, though its name is
        written there to say so.
        """
        text = inspect.getsource(topology)
        self.assertIn("_core.mdi_pairs(", text, msg="the pair bound left the chain")
        self.assertIn("mdi_yield", text, msg="the warning against it was deleted")
        self.assertNotIn("mdi_yield(", text, msg="the forward model is in the chain")


class MidpointGuards(Guarded):
    """
    What the midpoint's two branches refuse.
    """

    def test_bias_required(self):
        """
        The finite chain refuses a block without a key-basis probability, naming the field.
        """
        swap = bell_swap(50.0, None, q.TestBasisBound(f=FEC, block=q.RelayBlock()), states=2)
        self.assertFails(
            NotImplementedError,
            "declares no key-basis probability",
            swap.run,
            msg="a block was split without a bias",
        )

    def test_bias_dropped(self):
        """
        The asymptotic midpoint rate refuses a declared basis bias rather than accepting and
        dropping it.
        """
        swap = bell_swap(50.0, BASIS, q.TestBasisBound(f=FEC))
        self.assertFails(
            ValueError,
            "would be accepted and dropped",
            swap.run,
            msg="a bias reached the asymptotic rate",
        )

    def test_states_declared(self):
        """
        An unstated announcement count is refused in q.RelayBlock's own words.
        """
        swap = midpoint(50.0, 1e15)
        self.assertFails(
            ValueError,
            "part of the claim",
            midpoint(50.0, 1e15, states=None).run,
            msg="states=None accepted",
        )
        self.assertGreater(swap.run().key_length, 0.0, msg="the stated one still runs")

    def test_block_typed(self):
        """
        A midpoint's block counts pulse PAIRS, so the slot refuses a single-sender q.KeyBlock by
        name.
        """

        self.assertFails(
            ValueError,
            "counts pulse PAIRS",
            lambda: q.TestBasisBound(block=q.KeyBlock(n=1e12)),
            msg="q.KeyBlock accepted",
        )

    def test_security_named(self):
        """
        A midpoint takes q.TestBasisBound alone, the refusal naming the slot the finite form sits
        in.
        """
        swap = bell_swap(50.0, None, q.Asymptotic())
        self.assertFails(
            NotImplementedError,
            "q.TestBasisBound(block=q.RelayBlock(...))",
            swap.run,
            msg="refusal does not name the slot",
        )


class RelayAttack(Question):
    """
    The attack-class row both midpoint branches carry.
    """

    def test_attack_stated(self):
        """
        Both branches report an attack class as a bare string after the security row, and both say
        unstated.
        """
        for swap in (midpoint(50.0), midpoint(50.0, 1e15), covariance()):
            info = swap.explain()
            keys = list(info)
            self.assertIsInstance(info["attack"], str, msg="the row is not a bare string")
            self.assertEqual(
                keys[keys.index("security") + 1],
                "attack",
                msg="the row does not follow security",
            )
            self.assertIn("unstated", info["attack"], msg=f"{info['attack']!r}")

    def test_attack_split(self):
        """
        The counting branch's attack row names mdi_rate and the covariance branch's names
        cvmdi_rate.
        """
        counted = midpoint(50.0).explain()["attack"]
        covar = covariance().explain()["attack"]
        self.assertIn("mdi_rate", counted, msg=f"the counting row names {counted!r}")
        self.assertIn("cvmdi_rate", covar, msg=f"the covariance row names {covar!r}")
        self.assertNotEqual(counted, covar, msg="one row was copied to both branches")


if __name__ == "__main__":
    rc = Exam(
        "RelayChernoff",
        "Curty's Claim 3, the generalised multiplicative Chernoff deviation",
        "finiterelay_chernoff.md",
    ).run(load(MdiChernoff))
    rc |= Exam(
        "RelayEstimator",
        "The two-dimensional MDI-BB84 decoy inversion, Curty et al. (2014)",
        "finiterelay_estimator.md",
    ).run(load(MdiEstimator))
    rc |= Exam(
        "RelayLength",
        "The MDI-BB84 finite key length per announced Bell state",
        "finiterelay_length.md",
    ).run(load(MdiLength))
    rc |= Exam(
        "RelayBudget",
        "The MDI-BB84 epsilon composition and what the estimators refuse",
        "finiterelay_budget.md",
    ).run(load(MdiBudget))
    rc |= Exam(
        "MidpointBlock",
        "Curty's MDI-BB84 key length wired to q.Swap, summed over Bell states",
        "finiterelay_midpoint.md",
    ).run(load(MidpointBlock))
    rc |= Exam(
        "MidpointGuards",
        "What q.Swap's finite and asymptotic midpoint branches refuse",
        "finiterelay_guards.md",
    ).run(load(MidpointGuards))
    rc |= Exam(
        "RelayAttack",
        "The attack-class row q.Swap carries on both of its branches",
        "finiterelay_attack.md",
    ).run(load(RelayAttack))
    rc |= Exam(
        "TrainTails",
        "RRDPS finite key: three binomial tails at rates no observable moves",
        "finiterelay_tails.md",
    ).run(load(RrdpsTails))
    rc |= Exam(
        "TrainLength",
        "The RRDPS key length, Takesue, Sasaki, Tamaki & Koashi (2015), Methods",
        "finiterelay_train.md",
    ).run(load(RrdpsLength))
    rc |= Exam(
        "PairingFinite",
        "Mode-pairing finite key: Xie et al., PRX Quantum 3, 020315 (2022)",
        "finiterelay_pairing.md",
    ).run(load(PairingFinite))
    sys.exit(rc)
