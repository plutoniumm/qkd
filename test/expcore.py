import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.cache import memo
from kit.forms import bisect
import qkd as q
from qkd import _core, std

# Tier B: fielded runs for four families.
#
# Cao, Sun, Li, Lu, Yin & Chen, "Experimental
# Coherent One-Way Quantum Key Distribution with Simplicity and Practical
# Security", Science Advances 12, eaec2776 (2026), arXiv:2601.06772: the
# FOUR-sequence vacuum-decoy variant src/sdp.rs is written for, not q.IntensityKeying's three-sequence COW.
# Their Eqs. (6) and (7) are _core.sdp_analytic, cited to their Ref. [60], Gao, Xie, Gu, Liu, Weng, Li,
# Yin & Chen, Opt. Express 30, 23783 (2022), arXiv:2107.09329. Plain COW anchors are in test/expreach.py.
#
# Eq. (6) is wrong in print: its prefactors are N^-/N^+ of Eq. (7)'s (test_cow_prefactor). Every number
# below carries Eq. (7)'s prefactor on both bounds.

# Takesue, Sasaki, Tamaki & Koashi, "Experimental quantum key distribution
# without monitoring signal disturbance", Nature Photonics 9, 827 (2015),
# arXiv:1505.07914. NOT Wang, Yin, Chen et al., Nature Photonics 9, 832 (2015),
# whose title is "Experimental demonstration of a quantum key distribution
# without signal disturbance monitoring", in the same issue.
#
# Pinned: L = 5 at 2 GHz, system loss 12.7 dB, baseline error 1.5%, f = 1.1. rrdps_counts is Yin et al.'s
# receiver model, not their Eqs. (8)-(10).
T_L = 5
T_SYS = 12.7
T_EMIS = 0.015
T_FEC = 1.1

# FITTED to the 30 km error rate, not the cutoff: 10.4 times their 2 counts/s at 2 GHz, absorbing four detectors,
# the 3 dB coupler and the 200 ps window.
T_DARK = 1.038972e-8

# Fig. 3 read off the plot at 83.9 px/dB: fibre-run squares at 4.720 and 7.533 dB, not the 7.05 dB a 20 km slope
# gives. (loss dB, error rate) from panel (c), optimised intensity from (b), key rate per pulse from (a). T_CUTOFF:
# "we can generate secure keys up to 8.7 dB channel loss, and secure key
# distribution over a 30-km fiber was realized".
T_POINTS = ((0.066, 0.015320), (4.720, 0.016318), (7.533, 0.018201))
T_MODE = (3.0025e-4, 1.0825e-4, 5.625e-5)
T_RATE = (2.565e-7, 2.2895e-8, 3.0375e-9)
T_CUTOFF = 8.7

# Zhu, Huang, Liu, Zeng, Zou, Dai, Tang, Li, You, Wang, Chen, Ma, Chen & Pan,
# "Experimental mode-pairing measurement-device-independent quantum key
# distribution without global phase-locking", Phys. Rev. Lett. 130, 030801
# (2023), arXiv:2208.05649. Table VI (Appendix E): (name, mu, nu, eta, rounds, L_max, key bits per pair).
# eta is ONE SIDE's transmittance with the 62.46% detector inside ("single link"). A POSSIBLE PAIR is half a round.
Z_ROWS = (
    ("101 km", 0.309, 0.032, 4.32e-2, 5.07e11, 500, 7.75e-5),
    ("202 km", 0.338, 0.035, 6.80e-3, 2.10e12, 1000, 9.34e-6),
    ("304 km", 0.531, 0.053, 7.43e-4, 6.33e12, 2000, 8.65e-8),
    ("407 km", 0.429, 0.038, 2.18e-4, 7.66e13, 2000, 3.46e-9),
)

# Table I caption: dark "2.72e-8/pulse", f = 1.1.
Z_DARK = 2.72e-8
Z_FEC = 1.1

# pairing_rate's e_x: Table VI's single-photon-pair e_ph_11, NOT Fig. 3's legend 31% and 33% X-basis error rates.
Z_PHASE = (0.2474, 0.2571, 0.3370, 0.3468)

# Rest of Table VI: click probability, pairing probability, Z error rate, single-photon Z-pairs, and signal
# probability (pairing_click assumes Zeng's symmetric 1/2).
Z_CLICK = (6.35e-3, 1.13e-3, 1.89e-4, 5.00e-5)
Z_PAIRS = (2.46e-3, 4.29e-4, 4.42e-5, 4.21e-6)
Z_QBER = (3.10e-4, 1.89e-4, 3.75e-4, 1.46e-3)
Z_SINGLE = (1.07e8, 5.03e7, 4.51e6, 6.02e6)
Z_PMU = (0.22, 0.25, 0.20, 0.23)

# Xu, Wei, Sajeed, Kaiser, Sun, Tang, Qian, Makarov & Lo, "Experimental quantum
# key distribution with source flaws", Phys. Rev. A 92, 032305 (2015),
# arXiv:1408.3667. Table I ID-500: misalignment 2.35%, Bob efficiency 5.05%, dark 4.01e-5 per pulse per detector.
# X_DELTA is Table II's largest ID-500 modulation error, not the Clavis2 column's 0.145.
X_DELTA = 0.134
X_ETAB = 0.0505
X_DARK = 4.01e-5
X_FEC = 1.16
X_EDET = 0.0235

# Table III: (km, attenuation dB, single-photon phase error upper bound at eps_tot = 1e-10, measured Z error rate).
X_ROWS = ((5, 1.4, 0.0628, 0.0267), (20, 4.5, 0.0867, 0.0274), (50, 10.5, 0.0846, 0.0298))

# Line through the 5 and 50 km attenuations, for sweeps; misses the printed 4.5 dB at 20 km by 0.07 dB.
X_FIT = (0.389, 0.2022)

# Cao et al. Table 4. C_D1 is the CONSTRUCTIVE monitoring port, C_D2 destructive, counts in |0>|0>, |a>|a>, |0>|a>,
# |a>|0> order. C_VAC: the refined record's |0>|0> counts, 35.2% of the Z-basis count; nothing else changes.
C_KM = (25, 50, 75, 100)
C_MU = (3.50e-3, 1.40e-3, 5.65e-4, 2.43e-4)
C_NZ = (25278913, 4007708, 6432214, 1077297)
C_EZ = (0.0030, 0.0020, 0.0034, 0.0076)
C_D1 = (
    (22843, 4413507, 4668353, 4202853),
    (2194, 692141, 742102, 661568),
    (4942, 1121394, 1221403, 1067269),
    (840, 190418, 192729, 182993),
)
C_D2 = (
    (3751, 9980, 4209875, 4654407),
    (566, 1190, 666824, 742418),
    (1857, 4228, 1067140, 1221586),
    (1058, 1571, 172140, 204693),
)
C_VAC = ((1904, 1904), (81, 81), (554, 554), (493, 493))

# Sec. IV data sizes in two-pulse ROUNDS, Sec. III preparation probabilities, Table 2 bit/s (direct, refined), clock.
C_ROUNDS = (1e11, 1e11, 1e12, 1e12)
C_PREP = (0.10, 0.10, 0.40, 0.40)
C_RATE = ((1.37e4, 2.53e4), (2.47e3, 4.21e3), (2.82e2, 5.31e2), (12.8, 29.0))
C_CLOCK = 5.0e8

# Sec. III: Z share of the 30:70 splitter, D0 and D1/D2 efficiencies, darker monitor dark rate (Hz), fibre dB/km.
# Sec. IV 100 km run: "final key length ... approximately 58 kb" and 74,145 bits of Leak_EC.
C_SPLIT = 0.30
C_DATA = 0.762
C_MON = 0.460
C_DARK = 1.0
C_LOSS = 0.161
C_KEY = 58000.0
C_LEAK = 74145.0

# Their Sec. II security parameters, and the constant Eq. (1) charges for them.
C_EPS = (1e-15, 1e-10)
C_SLACK = math.log2(2.0 / C_EPS[0]) + 2.0 * math.log2(5.0 / C_EPS[1])

# FITTED: 1.0683 from the one printed Leak_EC = f n_z h(E_z) at 100 km; the 100 km rows use Leak_EC and carry none.
C_FEC = C_LEAK / (C_NZ[3] * std.entropy(C_EZ[3]))


def train(loss, dark=T_DARK):
    """
    (key rate per pulse, mu, nu_th, sifted rate, bit error rate) at ``loss`` dB on Takesue's
    receiver, maximised over swept mu.
    """
    eta = 10.0 ** (-(T_SYS + loss) / 10.0)
    top, arg = -1.0, None
    for step in range(1, 2400):
        mu = step * 1e-5
        q, e_bit = _core.rrdps_counts(T_L, mu, eta, dark, T_EMIS)
        for nu in range(1, T_L):
            got = _core.rrdps_rate(q, e_bit, _core.rrdps_gllp(q, T_L, mu, nu), T_FEC, T_L)
            if got > top:
                top, arg = got, (mu, nu, q, e_bit)

    return max(top, 0.0) / T_L, arg[0], arg[1], arg[2], arg[3]


# Hand-rolled, not kit.forms.bisect: 24 halvings against its fixed 200 is ~8x faster at ~9,600 core calls a probe.
@memo
def cutoff(dark=T_DARK):
    """
    Channel loss in dB where Takesue's asymptotic rate reaches zero.
    """
    lo, hi = 0.0, 25.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if train(mid, dark)[0] > 0.0:
            lo = mid
        else:
            hi = mid

    return lo


def paired(i, pinned=False):
    """
    (key rate per possible pair, sifted share, pairs per round, single-pair share, Z error) at
    Zhu's i-th row; ``pinned`` uses Table VI's pairing rate.
    """
    _name, mu, _nu, eta, _rounds, span, _meas = Z_ROWS[i]
    sift, e_z = _core.pairing_sift(eta, mu, Z_DARK)
    pairs = Z_PAIRS[i] if pinned else _core.pairing_pairs(_core.pairing_click(eta, mu, Z_DARK), span)
    q11 = _core.pairing_single(eta, mu, Z_DARK)
    rate = _core.pairing_rate(pairs, sift, q11, Z_PHASE[i], e_z, Z_FEC)

    return 2.0 * rate, sift, pairs, q11, e_z


def flawed(km, delta=X_DELTA, att=None):
    """
    (loss-tolerant, standard) tuples on Xu's ID-500 at ``km`` and tilt = -delta; ``att``
    defaults to the X_FIT line.
    """
    eta = X_ETAB * 10.0 ** (-(X_FIT[0] + X_FIT[1] * km if att is None else att) / 10.0)
    tol = _core.flaw_tolerant(delta, eta, X_DARK, -delta, X_FEC, 1.0)
    std = _core.flaw_standard(delta, eta, X_DARK, -delta, X_FEC, 1.0)

    return tol, std


def reach(fn):
    """
    Fibre length in km where ``fn`` reaches zero.
    """

    return bisect(fn, 0.1, 400.0)


def record(i, refined=False):
    """
    (constructive, destructive) gains per two-pulse sequence SENT at Cao's i-th distance, in
    src/sdp.rs order.
    """
    top = list(C_D1[i])
    bot = list(C_D2[i])

    if refined:
        top[0], bot[0] = C_VAC[i]

    return (
        [n / (C_ROUNDS[i] * p) for n, p in zip(top, C_PREP)],
        [n / (C_ROUNDS[i] * p) for n, p in zip(bot, C_PREP)],
    )


@memo
def bounded(i, refined=False):
    """
    (Gao's Cauchy-Schwarz closed form, certified SDP value) on Cao's i-th record.
    """
    q0, q1 = record(i, refined)

    return _core.sdp_analytic(C_MU[i], q0, q1), _core.sdp_phase(C_MU[i], q0, q1)[0]


def leaked(i):
    """
    Leak_EC in bits at Cao's i-th distance: the printed 74,145 at 100 km, f n_z h(E_z)
    elsewhere.
    """

    return C_LEAK if C_KM[i] == 100 else C_FEC * C_NZ[i] * std.entropy(C_EZ[i])


def length(i, phase):
    """
    Cao's Eq. (1) key length in bits over the i-th block at ``phase``.
    """

    return C_NZ[i] * (1.0 - std.entropy(phase)) - leaked(i) - C_SLACK


def implied(i, refined=False):
    """
    Phase error rate Cao's published key rate implies: Eq. (1) inverted on their n_z, E_z and
    Leak_EC.
    """
    span = C_ROUNDS[i] / C_CLOCK
    want = 1.0 - (C_RATE[i][1 if refined else 0] * span + leaked(i) + C_SLACK) / C_NZ[i]

    return bisect(std.entropy, 0.0, 0.5, want)


def cauchy(i, printed=False):
    """
    Gao's Cauchy-Schwarz phase error on Cao's i-th record from Eqs. (5)-(7); ``printed`` uses
    Eq. (6)'s printed prefactor, N^-/N^+ times Eq. (7)'s.
    """
    mu = C_MU[i]
    q0, q1 = record(i)
    plus = 2.0 * (1.0 + math.exp(-mu))
    minus = 2.0 * (1.0 - math.exp(-mu))
    over = math.exp(0.5 * mu) * math.sqrt(q1[1]) + math.exp(-0.5 * mu) * math.sqrt(q1[0])
    under = math.exp(0.5 * mu) * math.sqrt(q0[1]) - math.exp(-0.5 * mu) * math.sqrt(q0[0])
    top = over * over + minus * (0.25 * minus * math.exp(mu) + math.exp(mu) * math.sqrt(q1[1]) + math.sqrt(q1[0]))
    bot = under * under - minus * (math.exp(mu) * math.sqrt(q0[1]) + math.sqrt(q0[0]))
    top = min(top / plus, 1.0)
    bot = max(bot / plus, 0.0) * (minus / plus if printed else 1.0)

    return (plus * top + 2.0 * (q0[2] + q0[3]) - plus * bot) / (2.0 * (q0[2] + q1[2] + q0[3] + q1[3]))


def forward(i, dark=C_DARK):
    """
    Analytic phase error at Cao's i-th distance on stated hardware, through q.rates.
    """
    eta = C_MON * std.from_db(C_LOSS * C_KM[i])
    got = q.rates.keyrate(
        "cow-vacuum",
        mu=C_MU[i],
        t_b=C_SPLIT,
        eta=eta,
        dark=dark / C_CLOCK,
        f_ec=C_FEC,
        bound="analytic",
    )

    return got.values["e_phase"]


class TrainRig(Question):
    """
    Tier B: Takesue et al. 2015, RRDPS over 30 km. Pinned: packet length, system loss, baseline
    error, f, Fig. 3. Fitted: one per-pulse dark rate, from the 30 km error rate.
    """

    def test_train_error(self):
        """
        One dark rate fitted at the 30 km Fig. 3(c) point predicts the 20 km error rate to 0.044
        points and zero loss to 0.022, both from below.
        """
        got = [train(loss)[4] for loss, _want in T_POINTS]

        for (loss, want), seen in zip(T_POINTS, got):
            self.assertLess(seen - want, 1e-6, msg=f"{loss} dB: error {seen * 100:.4f}%")
            self.assertGreater(seen - want, -5e-4, msg=f"{loss} dB: error {seen * 100:.4f}%")

        self.assertMonotone(got, msg=f"error rates {got}")

    def test_train_cutoff(self):
        """
        The asymptotic rate reaches zero at 8.56 dB against their stated 8.7 dB, 0.14 dB short.
        """
        got = cutoff()

        self.assertClose(got, T_CUTOFF, atol=0.25, msg=f"cutoff {got:.3f} dB")
        self.assertLess(got, T_CUTOFF, msg=f"cutoff {got:.3f} dB")

    def test_train_darkness(self):
        """
        The fitted dark rate is 10.4 times their stated 2 counts/s per detector, and reach falls
        as it rises.
        """
        naive = 2.0 / 2e9

        self.assertClose(T_DARK / naive, 10.39, atol=0.1, msg=f"dark ratio {T_DARK / naive:.2f}")

        got = [cutoff(d) for d in (4e-10, 1e-9, 1e-8, 3e-8)]
        self.assertMonotone(got, rising=False, msg=f"cutoffs {got}")

    def test_train_intensity(self):
        """
        Declared miss: the sweep selects 5.8-6.1 times Fig. 3(b)'s intensity for 1.21-1.33 times
        Fig. 3(a)'s rate, rrdps_counts lacking Eq. (10)'s double-click term p_d = (L mu eta)^2 /
        16.
        """
        for i, (loss, _want) in enumerate(T_POINTS):
            rate, mu, nu, _q, _e = train(loss)
            seen = mu / T_MODE[i]

            self.assertGreater(seen, 5.5, msg=f"{loss} dB: mu ratio {seen:.3f}")
            self.assertLess(seen, 6.5, msg=f"{loss} dB: mu ratio {seen:.3f}")
            self.assertEqual(nu, 1, msg=f"{loss} dB: nu_th {nu}")

            seen = rate / T_RATE[i]
            self.assertGreater(seen, 1.15, msg=f"{loss} dB: rate ratio {seen:.4f}")
            self.assertLess(seen, 1.40, msg=f"{loss} dB: rate ratio {seen:.4f}")


class PairedRig(Question):
    """
    Tier B: Zhu et al. 2023, mode-pairing MDI-QKD over 101-407 km. Pinned: Table VI, Table I
    caption dark rate and f; e_ph_11 is a pinned input. Nothing fitted.
    """

    def test_paired_rate(self):
        """
        On Table VI's pairing rate and e_ph_11 the key rate per pair is 0.84 and 1.03 of
        measured at 101 and 202 km, and 2.9 and 4.5 times at 304 and 407 km, where eps = 1e-10
        bites on 4.51e6 and 6.02e6 single-photon Z-pairs.
        """
        got = [paired(i, pinned=True)[0] / row[6] for i, row in enumerate(Z_ROWS)]

        for row, seen in zip(Z_ROWS, got):
            self.assertGreater(seen, 0.8, msg=f"{row[0]}: rate ratio {seen:.4f}")

        for i in (0, 1):
            self.assertLess(got[i], 1.1, msg=f"{Z_ROWS[i][0]}: rate ratio {got[i]:.4f}")

        for i in (2, 3):
            self.assertGreater(got[i], 2.5, msg=f"{Z_ROWS[i][0]}: rate ratio {got[i]:.4f}")
            self.assertLess(got[i], 5.0, msg=f"{Z_ROWS[i][0]}: rate ratio {got[i]:.4f}")

        for i in (2, 3):
            seen = Z_SINGLE[0] / Z_SINGLE[i]
            self.assertGreater(seen, 15.0, msg=f"{Z_ROWS[i][0]}: single-pair ratio {seen:.1f}")

    def test_paired_pairs(self):
        """
        Declared miss: pairing_click is 1.87-2.09 times Table VI's click probability and
        pairing_pairs 2.5-3.2 times its pairing rate, Zeng's symmetric source against Zhu's p_mu
        of 0.20-0.25, putting the end-to-end rate 2.3-14.7 times high.
        """
        for i, row in enumerate(Z_ROWS):
            click = _core.pairing_click(row[3], row[1], Z_DARK)
            seen = click / Z_CLICK[i]
            self.assertGreater(seen, 1.8, msg=f"{row[0]}: click ratio {seen:.4f}")
            self.assertLess(seen, 2.1, msg=f"{row[0]}: click ratio {seen:.4f}")

            seen = paired(i)[2] / Z_PAIRS[i]
            self.assertGreater(seen, 2.5, msg=f"{row[0]}: pairs ratio {seen:.4f}")
            self.assertLess(seen, 3.3, msg=f"{row[0]}: pairs ratio {seen:.4f}")

            self.assertLess(Z_PMU[i], 0.26, msg=f"{row[0]}: p_mu {Z_PMU[i]}")

            seen = paired(i)[0] / row[6]
            self.assertGreater(seen, 2.2, msg=f"{row[0]}: end-to-end ratio {seen:.4f}")
            self.assertLess(seen, 15.0, msg=f"{row[0]}: end-to-end ratio {seen:.4f}")

    def test_paired_square(self):
        """
        Table VI's eta is one side's, so the rate scales as eta to the 1.144 measured and 1.033
        modelled over the two shortest rows and 1.894 and 1.576 over all four, all under the
        two-mode 2.
        """
        rates = [paired(i, pinned=True)[0] for i in range(len(Z_ROWS))]

        for near, far, want, mine in ((0, 1, 1.144, 1.033), (0, 3, 1.894, 1.576)):
            span = math.log(Z_ROWS[far][3] / Z_ROWS[near][3])
            seen = math.log(Z_ROWS[far][6] / Z_ROWS[near][6]) / span
            self.assertClose(seen, want, atol=0.02, msg=f"measured exponent {seen:.4f}")

            model = math.log(rates[far] / rates[near]) / span
            self.assertClose(model, mine, atol=0.02, msg=f"model exponent {model:.4f}")
            self.assertLess(model, 2.0, msg=f"model exponent {model:.4f}")
            self.assertLess(seen, 2.0, msg=f"measured exponent {seen:.4f}")

    def test_paired_observables(self):
        """
        At all four distances the sifted share is 1/8 to 1e-3 and the single-pair share exp(-2
        mu) to 0.008, while the modelled Z error sits under their QBER, pairing_sift taking no
        misalignment.
        """
        got = []
        for i, row in enumerate(Z_ROWS):
            _rate, sift, _pairs, q11, e_z = paired(i)
            self.assertClose(sift, 0.125, atol=1e-3, msg=f"{row[0]}: sifted share {sift:.5f}")
            self.assertClose(q11, math.exp(-2.0 * row[1]), atol=8e-3, msg=f"{row[0]}: q11 {q11:.5f}")
            self.assertLess(e_z, Z_QBER[i], msg=f"{row[0]}: e_z {e_z:.3e}")
            got.append(e_z / Z_QBER[i])

        self.assertLess(got[0], 0.03, msg=f"101 km: share {got[0]:.4f}")
        self.assertGreater(got[3], 0.75, msg=f"407 km: share {got[3]:.4f}")
        self.assertMonotone(got, msg=f"shares {got}")


class FlawedRig(Question):
    """
    Tier B: Xu et al. 2015, decoy-state QKD with a measured source flaw over 5, 20 and 50 km.
    Pinned: Tables I-III. Nothing fitted.
    """

    def test_flawed_bounded(self):
        """
        The asymptotic loss-tolerant phase error sits under Table III's finite-key upper bound
        at 5, 20 and 50 km, and the bit error under their e_z by less than their 2.35%
        misalignment.
        """
        for km, att, want, e_z in X_ROWS:
            tol = flawed(km, att=att)[0]
            self.assertLess(tol[3], want, msg=f"{km} km: phase error {tol[3] * 100:.4f}%")
            self.assertGreater(tol[3], 0.0, msg=f"{km} km: phase error {tol[3]}")

            # flaw_channel takes no misalignment: e_z is under-predicted by at most their 2.35%.
            short = e_z - tol[2]
            self.assertGreater(short, 0.0, msg=f"{km} km: shortfall {short * 100:.3f}%")
            self.assertLess(short, X_EDET, msg=f"{km} km: shortfall {short * 100:.3f}%")

    def test_flawed_costs_little(self):
        """
        The measured 0.134 rad flaw costs 2.0% of loss-tolerant reach, 91.3 to 89.5 km, against
        their "almost the same as the case without source flaws".
        """
        with_flaw = reach(lambda km: flawed(km)[0][0])
        without = reach(
            lambda km: _core.flaw_tolerant(
                0.0, X_ETAB * 10.0 ** (-(X_FIT[0] + X_FIT[1] * km) / 10.0), X_DARK, 0.0, X_FEC, 1.0
            )[0]
        )
        cost = 1.0 - with_flaw / without

        self.assertLess(cost, 0.05, msg=f"reach cost {cost * 100:.2f}%")
        self.assertGreater(cost, 0.0, msg=f"reach cost {cost * 100:.2f}%")

    def test_flawed_beats_coin(self):
        """
        The quantum-coin bound reaches 26.9 km against the loss-tolerant 89.5 km, 3.3 times,
        beside their Fig. 3 caption's finite-key 9 and 60 km.
        """
        tol = reach(lambda km: flawed(km)[0][0])
        std = reach(lambda km: flawed(km)[1][0])

        self.assertGreater(tol / std, 2.0, msg=f"reach ratio {tol / std:.2f}")
        self.assertGreater(tol, 60.0, msg=f"loss-tolerant reach {tol:.2f} km")
        self.assertGreater(std, 9.0, msg=f"coin reach {std:.2f} km")

    def test_flawed_needs_tilt(self):
        """
        At tilt = 0 flaw_phase returns the same value to 1e-15 for delta = 0, 0.134 and 0.3; at
        tilt = -delta it rises, by over 1.5 times at 0.134.
        """
        eta = X_ETAB * 10.0 ** (-4.5 / 10.0)
        flat = [_core.flaw_phase(d, eta, X_DARK, 0.0) for d in (0.0, X_DELTA, 0.3)]

        self.assertClose(flat[1], flat[0], atol=1e-15, msg=f"tilt 0 phase errors {flat}")
        self.assertClose(flat[2], flat[0], atol=1e-15, msg=f"tilt 0 phase errors {flat}")

        opposed = [_core.flaw_phase(d, eta, X_DARK, -d) for d in (0.0, X_DELTA, 0.3)]

        self.assertMonotone(opposed, msg=f"tilt -delta phase errors {opposed}")
        self.assertGreater(opposed[1] / opposed[0], 1.5, msg=f"ratio {opposed[1] / opposed[0]:.4f}")


class CowRig(Question):
    """
    Tier B: Cao et al. 2026, vacuum-decoy COW' over 25-100 km. Pinned: Table 4 in both records
    and the Sec. II-IV hardware. Fitted: f from the one printed Leak_EC; the 100 km rows carry
    none.
    """

    def test_cow_closure(self):
        """
        Z-basis totals follow from their hardware to 1.3% at all four distances with data size
        in two-pulse rounds, and 58 kb over 29.0 bit/s closes at 2000 s.
        """
        for i, km in enumerate(C_KM):
            eta = C_SPLIT * C_DATA * std.from_db(C_LOSS * km)
            seen = C_ROUNDS[i] * 0.8 * C_MU[i] * eta / C_NZ[i]
            self.assertClose(seen, 1.0, atol=0.015, msg=f"{km} km: n_z ratio {seen:.5f}")

        span = C_KEY / C_RATE[3][1]

        self.assertClose(span, C_ROUNDS[3] / C_CLOCK, atol=0.5, msg=f"run {span:.1f} s")

    def test_cow_ports(self):
        """
        The |a>|a> gain is 0.98-0.99 of mu and the four Z gains within 11% of mu/4, fixing
        src/sdp.rs's ``ports`` convention against a doubled flux's 0.5, with 99% visibility.
        """
        for i, km in enumerate(C_KM):
            q0, q1 = record(i)
            full = C_MU[i] * (1.0 - C_SPLIT) * C_MON * std.from_db(C_LOSS * km)
            seen = q0[1] / full
            self.assertClose(seen, 1.0, atol=0.025, msg=f"{km} km: |a>|a> gain over mu {seen:.5f}")
            self.assertGreater(seen, 0.75, msg=f"{km} km: |a>|a> gain over mu {seen:.5f}")

            for tag, got in (("q0[2]", q0[2]), ("q1[2]", q1[2]), ("q0[3]", q0[3]), ("q1[3]", q1[3])):
                seen = got / (0.25 * full)
                self.assertClose(seen, 1.0, atol=0.11, msg=f"{km} km {tag}: {seen:.5f} of a quarter")

            seen = 1.0 - 2.0 * q1[1] / (q0[1] + q1[1])
            self.assertClose(seen, 0.99, atol=0.01, msg=f"{km} km: visibility {seen:.5f}")

    def test_cow_record(self):
        """
        Gao's closed form on their eight gains gives 0.929-0.978 of the phase error their key
        rate implies through Eq. (1), always below, and 4.8% and 5.1% low at 100 km with nothing
        fitted.
        """
        for i, km in enumerate(C_KM):
            for tag, ref in (("direct", False), ("refined", True)):
                theirs = implied(i, ref)
                seen = bounded(i, ref)[0] / theirs
                self.assertLess(seen, 0.985, msg=f"{km} km {tag}: ratio {seen:.6f}")
                self.assertGreater(seen, 0.92, msg=f"{km} km {tag}: ratio {seen:.6f}")

        for tag, ref, want in (("direct", False, 0.951854), ("refined", True, 0.949453)):
            seen = bounded(3, ref)[0] / implied(3, ref)
            self.assertClose(seen, want, atol=5e-4, msg=f"100 km {tag}: ratio {seen:.6f}")

    def test_cow_leak(self):
        """
        The two records differ only in the |0>|0> counts, so the difference of their key lengths
        cancels every reconciliation and epsilon term; the model reproduces it to 3.3%-7.3%.
        """
        for i, km in enumerate(C_KM):
            want = (C_RATE[i][1] - C_RATE[i][0]) * C_ROUNDS[i] / C_CLOCK
            got = C_NZ[i] * (std.entropy(bounded(i)[0]) - std.entropy(bounded(i, True)[0]))
            self.assertGreater(got / want, 1.0, msg=f"{km} km: gap ratio {got / want:.6f}")
            self.assertLess(got / want, 1.09, msg=f"{km} km: gap ratio {got / want:.6f}")

    def test_cow_certified(self):
        """
        On all eight records the certified bound sits 19.7%-26.2% under the closed form,
        1.36-3.85 times the asymptotic key length, 102.3 against 39.5 bit/s on the refined 100
        km record; neither compares with their finite-size 29.0 bit/s.
        """
        for i, km in enumerate(C_KM):
            for tag, ref in (("direct", False), ("refined", True)):
                plain, sure = bounded(i, ref)
                self.assertLess(sure, plain, msg=f"{km} km {tag}: SDP {sure:.6f}, closed form {plain:.6f}")
                seen = 1.0 - sure / plain
                self.assertGreater(seen, 0.19, msg=f"{km} km {tag}: tightening {seen * 100:.2f}%")
                self.assertLess(seen, 0.27, msg=f"{km} km {tag}: tightening {seen * 100:.2f}%")
                seen = length(i, sure) / length(i, plain)
                self.assertGreater(seen, 1.35, msg=f"{km} km {tag}: length ratio {seen:.4f}")

        plain, sure = bounded(3, True)
        span = C_ROUNDS[3] / C_CLOCK

        self.assertClose(length(3, plain) / span, 39.48, atol=0.05, msg="100 km bit/s, closed form")
        self.assertClose(length(3, sure) / span, 102.34, atol=0.05, msg="100 km bit/s, certified")

    def test_cow_vacuum(self):
        """
        The two records share six identical counts, and the two |0>|0> gains move the phase
        error by 8.3%-21.3%, most at 25 km.
        """
        for i, km in enumerate(C_KM):
            q0, q1 = record(i)
            r0, r1 = record(i, True)
            self.assertEqual(q0[1:], r0[1:], msg=f"{km} km: q0")
            self.assertEqual(q1[1:], r1[1:], msg=f"{km} km: q1")

            seen = 1.0 - bounded(i, True)[0] / bounded(i)[0]
            self.assertGreater(seen, 0.08, msg=f"{km} km: shift {seen * 100:.2f}%")
            self.assertLess(seen, 0.22, msg=f"{km} km: shift {seen * 100:.2f}%")

        self.assertGreater(
            1.0 - bounded(0, True)[0] / bounded(0)[0],
            1.0 - bounded(3, True)[0] / bounded(3)[0],
            msg="25 km shift against 100 km",
        )

    def test_cow_prefactor(self):
        """
        Eqs. (5)-(7) with Eq. (7)'s prefactor equal _core.sdp_analytic to 1e-12, while Eq. (6)
        as printed reads 0.529-0.558 and aborts at all four published distances.
        """
        for i, km in enumerate(C_KM):
            seen = cauchy(i)
            self.assertClose(seen, bounded(i)[0], atol=1e-12, msg=f"{km} km: Eqs. (5)-(7) {seen:.6f}")

            seen = cauchy(i, printed=True)
            self.assertGreater(seen, 0.5, msg=f"{km} km: printed Eq. (6) {seen:.6f}")
            self.assertLess(seen / bounded(i)[0], 2.1, msg=f"{km} km: ratio {seen / bounded(i)[0]:.4f}")
            self.assertGreater(seen / bounded(i)[0], 1.8, msg=f"{km} km: ratio {seen / bounded(i)[0]:.4f}")
            self.assertLess(bounded(i)[0], 0.5, msg=f"{km} km: Eq. (7) {bounded(i)[0]:.6f}")

        for i, want in ((0, 571.4), (3, 8230.5)):
            seen = (1.0 + math.exp(-C_MU[i])) / (1.0 - math.exp(-C_MU[i]))
            self.assertClose(seen, want, atol=0.1, msg=f"{C_KM[i]} km: factor {seen:.1f}")

    def test_cow_reflection(self):
        """
        Declared miss: the measured |0>|0> gain is 1142 times the 25 km dark floor, attributed
        to Michelson reflections, so the forward model reads 21%-35% under the recorded phase
        error.
        """
        for i, km in enumerate(C_KM):
            seen = forward(i) / bounded(i)[0]
            self.assertLess(seen, 0.80, msg=f"{km} km: forward ratio {seen:.5f}")
            self.assertGreater(seen, 0.64, msg=f"{km} km: forward ratio {seen:.5f}")

        q0, q1 = record(0)
        eta = C_MON * std.from_db(C_LOSS * C_KM[0])
        floor = _core.sdp_gains(C_MU[0], C_SPLIT, eta, C_DARK / C_CLOCK)
        seen = q0[0] / floor[0][0]

        self.assertGreater(seen, 500.0, msg=f"25 km: |0>|0> over floor {seen:.1f}")

        seen = forward(3, dark=11.0)
        self.assertGreater(seen, bounded(3)[0], msg=f"100 km: 11 Hz forward {seen:.5f}")


if __name__ == "__main__":
    rc = Exam(
        "ExpTakesueTrain",
        "Tier B: Takesue et al. 2015, RRDPS measured over 30 km of fibre",
        "exp_rrdps.md",
    ).run(load(TrainRig))
    rc |= Exam(
        "ExpZhuPairing",
        "Tier B: Zhu et al. 2023, mode pairing measured from 101 to 407 km",
        "exp_pairing.md",
    ).run(load(PairedRig))
    rc |= Exam(
        "ExpXuFlaws",
        "Tier B: Xu et al. 2015, a measured source flaw and its key rate",
        "exp_flaws.md",
    ).run(load(FlawedRig))
    rc |= Exam(
        "ExpCaoVacuum",
        "Tier B: Cao et al. 2026, the vacuum-decoy COW' measured from 25 to 100 km",
        "exp_cow_vacuum.md",
    ).run(load(CowRig))
    sys.exit(rc)
