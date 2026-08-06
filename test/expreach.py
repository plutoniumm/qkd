import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.forms import bisect, h2
from qkd import _core

# Korzh, Lim, Houlmann, Gisin, Li, Nolan, Sanguinetti, Thew & Zbinden,
# "Provably Secure and Practical Quantum Key Distribution over 307 km of
# Optical Fibre", Nature Photonics 9, 163 (2015), arXiv:1407.7427. COW on free-running InGaAs/InP NFADs.
# Supplementary Table I, "Overview of experimental parameters at different distances", plus the main text:
#   K_ALPHA   "an average attenuation of 0.160 dB/km without splices and
#             connectors"
#   K_CLOCK   their 625 MHz state preparation frequency
#   K_BETA    their chosen security level, beta = 1e-9
#   the receiver-factor check, "a DCR of a
#   few counts per second can be achieved at detection efficiencies of more
#   than 20%": stated below 150 K, where Table I runs 223 to 153 K
#   K_QBER    their Q_tot column. Their Eq. (1) does not consume it as printed: "the
#             dark count contribution to V and Q in equation (1) are subtracted"
# One bit occupies two 625 MHz slots.
K_DIST = (104.0, 153.0, 203.0, 256.0, 307.0)
K_LOSS = (16.9, 25.7, 34.1, 42.6, 51.9)
K_DARK = (548.0, 117.0, 16.2, 3.26, 1.33)
K_DEAD = (8.6e-6, 25.4e-6, 42.3e-6, 84.5e-6, 114.0e-6)
K_MU = (0.06, 0.09, 0.10, 0.09, 0.07)
K_BLOCK = (1.97e7, 1.97e7, 1.05e7, 1.3e6, 6.6e5)
K_TIME = (536.8, 1450.0, 3108.0, 8586.0, 17245.0)
K_FEC = (1.217, 1.272, 1.271, 1.228, 1.287)
K_FSEC = (0.347, 0.310, 0.301, 0.230, 0.084)
K_VIS = (0.983, 0.981, 0.982, 0.984, 0.970)
K_QBER = (0.024, 0.015, 0.015, 0.020, 0.035)
K_RATE = (1.27e4, 5.20e3, 1.02e3, 78.0, 3.18)
K_CLOCK = 625e6
K_ALPHA = 0.160
K_BETA = 1e-9

# FITTED together at 203 and 307 km, then held. K_FACTOR: detector efficiency times Bob's data-line tap share.
# K_BASE: fraction of signal detections in the wrong slot.
K_FACTOR = 0.1866276
K_BASE = 0.01153716

# Takesue, Nam, Zhang, Hadfield, Honjo, Tamaki & Yamamoto, "Quantum key
# distribution over 40 dB channel loss using superconducting single photon
# detectors", Nature Photonics 1, 343 (2007), arXiv:0706.0397. DPS at 10 GHz into NbN SSPDs. Their Eq. (4) is
# ``dps_rate``: Waks, Takesue & Yamamoto, Phys. Rev. A 73, 012344 (2006) Eq. (37).
#
# Quoted from their text:
#   T_MU, T_CLOCK  their 10 GHz clock and mean photon number 0.2 per pulse
#   T_MZI          "The excess loss of the interferometer is 2.5 dB"
#   T_ETA, T_DARK, T_GATE, T_WIN
#                  "we set the quantum efficiency, dark count rate and time
#                  window width at 1.4%, 50 Hz and 50 ps, respectively. The use
#                  of the time window reduced the effective quantum efficiency
#                  by 36%"
#   T_BASE         their 2.3% baseline system error, stated for the same
#                  theoretical curve
#   T_ALPHA        Fig. 5 caption, the rate drawn against 0.2 dB/km fibre
#   T_NEAR         their 17 kbit/s at 105 km
#   T_FAR          their 12 bit/s over 200 km, 12.1 bit/s in the abstract
#   T_MAX          "The maximum channel loss for secure key generation was
#                  42.1 dB"
#   T_LIMIT        their "approximately 4.1%" error threshold at mu = 0.2,
#                  against general collective attack for individual photons
T_MU = 0.2
T_CLOCK = 10e9
T_ETA = 0.014
T_WIN = 0.64
T_DARK = 50.0
T_GATE = 50e-12
T_MZI = 2.5
T_BASE = 0.023
T_ALPHA = 0.2
T_NEAR = 17000.0
T_FAR = 12.1
T_MAX = 42.1
T_LIMIT = 0.041

# FITTED: f(e) is unstated; their 4.1% threshold selects 1.16 (test_takesue_threshold).
T_FEC = 1.16

# A READING, not published: "50 Hz" is the detector pair's rate, halved before two-gate composition (test_takesue_dark).
T_SPLIT = 0.5

# Their note 23, the 1 GHz companion run on the same rig: "the quantum
# efficiency, dark count rate and time window width of the SSPDs were set at
# 0.6%, 6 Hz, and 100 ps respectively. The use of the time window reduced the
# effective quantum efficiency by 45%. The baseline system error was estimated
# to be 1.5%. The error threshold of a sequential USD attack for a 41.7 dB
# transmission with this detector condition was 4.1%".
S_CLOCK = 1e9
S_ETA = 0.006
S_WIN = 0.55
S_DARK = 6.0
S_GATE = 100e-12
S_BASE = 0.015
S_LOSS = 41.7

# Lucamarini, Patel, Dynes, Froehlich, Sharpe, Dixon, Yuan, Penty & Shields,
# "Efficient decoy-state quantum key distribution with quantified security",
# Opt. Express 21, 24550 (2013), arXiv:1310.0240. T12: decoy-state BB84 with a biased basis choice.
#
# Quoted from their text:
#   L_PX, L_PROB  their intensity probabilities p_w = 2^-8, p_v = 2^-7,
#                 p_u = 1 - p_w - p_v; the T12 basis bias is p_X = 1/16
#   L_MU          their average photon numbers u = 0.425, v = 0.044, w = 0.001
#   L_ETA, L_DARK, L_AFTER
#                 "The APDs are cooled thermoelectrically to -30 degrees C and
#                 operated at a detection efficiency of 20.5%, dark count
#                 probability per gate of 2.1e-5 and after pulse probability
#                 5.25%"
#   L_CLOCK       their 1550 nm laser diode, pulsed at 1 GHz
#   L_SECS        "key sessions of 20 minutes"
#   L_ZCNT, L_XCNT
#                 their per-session sifted counts C_uZZ, C_vZZ, C_wZZ and
#                 C_uXX, C_vXX, C_wXX at 50 km
#   L_QBER        "Q_uZZ = (4.26 +/- 0.20)% and Q_uXX = (3.64 +/- 0.65)%"
#   L_DIST, L_T12, L_BB84
#                 "The T12 secure key rates are, for increasing distances, 2.20,
#                 1.09, 0.40 and 0.12 Mbps"; the p_X = 1/2 comparison row of the
#                 same figure reads 1.18, 0.63, 0.26 and 0.06 Mbps
L_CLOCK = 1e9
L_SECS = 1200.0
L_PX = 1.0 / 16.0
L_MU = (0.425, 0.044, 0.001)
L_PROB = (1.0 - 2.0**-8 - 2.0**-7, 2.0**-7, 2.0**-8)
L_ZCNT = (5.016e9, 6.21e6, 1.259e6)
L_XCNT = (2.231e7, 2.843e4, 5.79e3)
L_QBER = (0.0426, 0.0364)
L_ETA = 0.205
L_DARK = 2.1e-5
L_AFTER = 0.0525
L_DIST = (35.0, 50.0, 65.0, 80.0)
L_T12 = (2.20e6, 1.09e6, 0.40e6, 0.12e6)
L_BB84 = (1.18e6, 0.63e6, 0.26e6, 0.06e6)

# FITTED: L_FEC in [1.0, 1.22] against the 50 km T12 rate alone; L_FIBRE in [0.19, 0.21], distance ladder only.
L_FEC = 1.0995239
L_FIBRE = 0.20

# A background click errs half the time; crate::decoy and crate::discrete write the same 1/2.
E_VAC = 0.5


def cow_slot(i, factor=K_FACTOR):
    """
    (signal detection probability per bit, dark probability per slot) at Korzh's i-th distance;
    the signal term is ``cow_ceiling`` at no decoy fraction.
    """
    trans = 10.0 ** (-K_LOSS[i] / 10.0) * factor

    return _core.cow_ceiling(0.0, trans, K_MU[i]), K_DARK[i] / K_CLOCK


def cow_qber(i, factor=K_FACTOR, base=K_BASE, dark=None):
    """
    Data-line QBER at Korzh's i-th distance: two slots per bit put the background twice in the
    denominator and once in the numerator.
    """
    sig, seen = cow_slot(i, factor)
    if dark is not None:
        seen = dark

    return (base * sig + seen) / (sig + 2.0 * seen)


def cow_fit(i, j):
    """
    (receiver factor, baseline slot error) reproducing the measured QBER at distances i and j.
    """

    def base(k, at):
        sig, seen = cow_slot(at, k)

        return (K_QBER[at] * (sig + 2.0 * seen) - seen) / sig

    factor = bisect(lambda k: base(k, i) - base(k, j), 0.02, 1.0)

    return factor, base(factor, i)


def cow_extra(i):
    """
    Extra background per slot needed for the measured QBER at distance i, as a fraction of the
    detected count rate: an afterpulse probability.
    """
    sig, seen = cow_slot(i)
    want = sig * (K_QBER[i] - K_BASE) / (1.0 - 2.0 * K_QBER[i])

    return (want - seen) * 2.0 / (sig + 2.0 * seen)


def cow_phase(i):
    """
    Branciard-Gisin-Scarani collective-attack COW phase error as Korzh's supplementary Eq. (3)
    prints it: xi = (2V-1)e^-mu - 2 sqrt((1 - e^-2mu) V (1-V)), returned as (1-xi)/2.
    """
    vis, mu = K_VIS[i], K_MU[i]
    xi = (2.0 * vis - 1.0) * math.exp(-mu) - 2.0 * math.sqrt((1.0 - math.exp(-2.0 * mu)) * vis * (1.0 - vis))

    return 0.5 * (1.0 - xi)


def cow_frac(i):
    """
    (qkd's asymptotic secret fraction per sifted bit, Korzh's Branciard-Gisin-Scarani form) at
    distance i, both on the measured Q_tot and K_VIS without their dark-count subtraction.
    """
    phase = cow_phase(i)
    ours = _core.cow_rate(1.0, K_QBER[i], phase, K_FEC[i])
    theirs = (1.0 - K_QBER[i]) * (1.0 - h2(phase)) - K_FEC[i] * h2(K_QBER[i])

    return ours, theirs


def cow_penalty(i):
    """
    Korzh's Eq. (3) deviation term 7 sqrt(n_cpp log2 1/beta) per sifted bit; Eq. (2)'s
    visibility deviation is not modelled.
    """

    return 7.0 * math.sqrt(math.log2(1.0 / K_BETA) / K_BLOCK[i])


def dps_slot(loss, eta=None, dark=None, base=T_BASE):
    """
    (click probability per pulse, QBER) behind ``loss`` dB and Takesue's 2.5 dB interferometer,
    written out to arbitrate the engine's p_click (4e-12 relative); the overrides carry note
    23's run.
    """
    if eta is None:
        eta = T_ETA * T_WIN
    if dark is None:
        dark = T_SPLIT * T_DARK * T_GATE
    trans = 10.0 ** (-(loss + T_MZI) / 10.0) * eta
    sig = 1.0 - math.exp(-trans * T_MU)
    back = 1.0 - (1.0 - dark) ** 2
    click = sig + back - sig * back

    return click, (base * sig + E_VAC * back) / click


def dps_rate(loss, eta=None, dark=None):
    """
    Secure bit/s at Takesue's 10 GHz clock, Waks-Takesue-Yamamoto Eq. (37).
    """
    click, err = dps_slot(loss, eta, dark)

    return T_CLOCK * _core.dps_rate(click, err, T_MU, T_FEC)


def dps_reach(eta=None, dark=None):
    """
    Channel loss in dB at which the secure rate crosses zero.
    """

    return bisect(lambda a: dps_rate(a, eta, dark), 20.0, 60.0)


def slow_rate(loss):
    """
    (secure bit/s, QBER) for the note 23 run at its 1 GHz clock.
    """
    click, err = dps_slot(loss, S_ETA * S_WIN, S_DARK * S_GATE, S_BASE)

    return S_CLOCK * _core.dps_rate(click, err, T_MU, T_FEC), err


def t12_gain(i, wide=True):
    """
    Gain per emitted pulse of Lucamarini's i-th intensity from one basis's sifted count;
    ``wide`` picks Z, X being an independent reading.
    """
    share = (1.0 - L_PX) if wide else L_PX
    sent = L_CLOCK * L_SECS * L_PROB[i] * share * share

    return (L_ZCNT[i] if wide else L_XCNT[i]) / sent


def t12_back():
    """
    Background per gate from the published detector constants: two dark gates plus afterpulsing
    on the u-line count rate.
    """

    return 1.0 - (1.0 - L_DARK) ** 2 + L_AFTER * t12_gain(0)


def t12_line(trans, back, err, mu):
    """
    (gain, QBER) of one intensity line from ``decoy_gain``.
    """

    return _core.decoy_gain(mu, trans, back, err)


def t12_model():
    """
    (transmittance, background yield, Z misalignment, X misalignment), all derived from
    published counts, QBERs and detector constants.
    """
    back = t12_back()
    wide = t12_gain(0)
    trans = bisect(lambda t: t12_line(t, back, 0.0, L_MU[0])[0] - wide, 1e-5, 1.0)
    narrow = t12_gain(0, wide=False)
    errs = tuple(
        (q * g - E_VAC * back) * (1.0 - back) / (g - back) for q, g in ((L_QBER[0], wide), (L_QBER[1], narrow))
    )

    return trans, back, errs[0], errs[1]


def t12_decoy(trans, back, err):
    """
    ``decoy_bounds`` on that model; the paper withholds the v and w error rates, so the forward
    model calibrated at u stands in (test_t12_lines).
    """
    lines = [t12_line(trans, back, err, mu) for mu in L_MU]

    return _core.decoy_bounds(
        L_MU[0],
        L_MU[1],
        L_MU[2],
        lines[0][0],
        lines[0][1],
        lines[1][0],
        lines[1][1],
        lines[2][0],
        lines[2][1],
    )


def t12_far(trans, dist):
    """
    (transmittance, background) at ``dist`` km along L_FIBRE, the afterpulse background
    re-solved at the new count rate.
    """
    moved = trans * 10.0 ** (-L_FIBRE * (dist - L_DIST[1]) / 10.0)
    floor = 1.0 - (1.0 - L_DARK) ** 2
    back = floor
    for _ in range(64):
        back = floor + L_AFTER * t12_line(moved, back, 0.0, L_MU[0])[0]

    return moved, back


def t12_rate(bias, f_ec=None, dist=None):
    """
    Asymptotic bit/s summed over both bases at X-basis probability ``bias`` (1/16 for T12, 1/2
    for BB84); ``dist`` moves the link off 50 km.
    """
    trans, back, wide, narrow = t12_model()
    if f_ec is None:
        f_ec = L_FEC
    if dist is not None:
        trans, back = t12_far(trans, dist)
    lines = (
        t12_line(trans, back, wide, L_MU[0]),
        t12_line(trans, back, narrow, L_MU[0]),
    )
    q1 = t12_decoy(trans, back, wide)[2]
    phase = t12_decoy(trans, back, narrow)[1]
    rates = (
        _core.bb84_rate((1.0 - bias) ** 2, lines[0][0], lines[0][1], q1, phase, f_ec),
        _core.bb84_rate(bias * bias, lines[1][0], lines[1][1], q1, phase, f_ec),
    )

    return L_CLOCK * (rates[0] + rates[1])


class KorzhCow(Question):
    """
    Tier B: Korzh et al. 2015 (Nature Photonics 9, 163; arXiv:1407.7427), COW over 104-307 km.
    Pinned: supplementary Table I, 0.160 dB/km, 625 MHz. Fitted: receiver factor and baseline
    slot error at 203 and 307 km. Finite-key rates are bounded, not reproduced.
    """

    def test_korzh_span(self):
        """
        The loss column exceeds bare 0.160 dB/km fibre by 0.26 dB at 104 km rising to 2.78 dB at
        307 km, 0.1625 to 0.1691 dB/km end to end.
        """
        extra = [K_LOSS[i] - K_ALPHA * K_DIST[i] for i in range(5)]
        self.assertMonotone(extra, msg=f"excess {extra}")
        self.assertClose(extra[0], 0.26, atol=1e-9, msg=f"104 km {extra[0]:.3f} dB")
        self.assertClose(extra[4], 2.78, atol=1e-9, msg=f"307 km {extra[4]:.3f} dB")

        rate = [K_LOSS[i] / K_DIST[i] for i in range(5)]
        self.assertClose(min(rate), 0.1625, atol=1e-6, msg=f"lowest {min(rate):.5f}")
        self.assertClose(max(rate), 0.169055, atol=1e-6, msg=f"highest {max(rate):.6f}")

        for got in rate:
            self.assertGreater(got, K_ALPHA, msg=f"{got:.5f} dB/km")

    def test_korzh_qber(self):
        """
        Fitted at 203 and 307 km, the model gives 1.555% at 153 km against their 1.5% and 1.700%
        at 256 km against their 2.0%.
        """
        factor, base = cow_fit(2, 4)

        self.assertClose(factor, K_FACTOR, atol=1e-6, msg=f"factor {factor:.7f}")
        self.assertClose(base, K_BASE, atol=1e-7, msg=f"baseline {base:.8f}")

        got = [cow_qber(i, factor, base) for i in range(5)]

        for i in (2, 4):
            self.assertClose(got[i], K_QBER[i], atol=1e-9, msg=f"fitted point {i} is {got[i]:.6f}")
        self.assertClose(got[1], 0.0155493, atol=1e-6, msg=f"153 km {got[1]:.6f}")
        self.assertClose(got[3], 0.0169960, atol=1e-6, msg=f"256 km {got[3]:.6f}")

        for i in (1, 3):
            self.assertLess(
                abs(got[i] - K_QBER[i]),
                0.0031,
                msg=f"{K_DIST[i]:.0f} km QBER {got[i]:.5f}",
            )

    def test_korzh_receiver(self):
        """
        The fitted factor over their 20% efficiency needs a 0.9331 data-line share, a lower
        bound since the 20% is stated below 150 K; refitting at 153 and 307 km moves the factor
        2.8% and the baseline 5.8%.
        """
        factor = cow_fit(2, 4)[0]
        share = factor / 0.20

        self.assertClose(share, 0.933138, atol=1e-5, msg=f"data share {share:.6f}")
        self.assertLess(share, 1.0, msg=f"data share {share:.6f}")
        self.assertGreater(share, 0.5, msg=f"data share {share:.6f}")

        again = cow_fit(1, 4)
        self.assertClose(again[0] / factor, 0.972307, atol=1e-5, msg=f"refit factor {again[0]:.7f}")
        self.assertClose(again[1] / K_BASE, 0.942078, atol=1e-5, msg=f"refit baseline {again[1]:.8f}")

    def test_korzh_near(self):
        """
        Declared miss: 104 km reads 1.526% against their 2.40%, needing an afterpulse-like
        background of 1.84% of the count rate, against 0.63% at 256 km and -0.11% at 153 km.
        """
        got = cow_qber(0)

        self.assertClose(got, 0.0152556, atol=1e-6, msg=f"104 km {got:.6f}")
        self.assertLess(got, K_QBER[0], msg=f"104 km {got:.6f}")
        self.assertClose(K_QBER[0] - got, 0.0087444, atol=1e-6, msg=f"short by {K_QBER[0] - got:.5f}")

        need = [cow_extra(i) for i in range(5)]
        self.assertClose(need[0], 0.018371, atol=1e-5, msg=f"104 km needs {need[0]:.6f}")
        self.assertClose(need[3], 0.006259, atol=1e-5, msg=f"256 km needs {need[3]:.6f}")
        self.assertClose(need[1], -0.001132, atol=1e-5, msg=f"153 km {need[1]:.6f}")

        for i in (0, 1, 3):
            self.assertLess(abs(need[i]), 0.05, msg=f"{K_DIST[i]:.0f} km needs {need[i]:.6f}")

    def test_korzh_dark(self):
        """
        Background-free the model is flat at 1.1537%; halving their dark column puts 307 km at
        2.36% and doubling it at 5.63%, both excluded by their 3.5%.
        """
        flat = [cow_qber(i, dark=0.0) for i in range(5)]

        self.assertLess(max(flat) / min(flat) - 1.0, 1e-12, msg=f"flat {flat}")

        for got in flat:
            self.assertClose(got, K_BASE, atol=1e-9, msg=f"floor {got:.9f}")

        band = [cow_qber(4, dark=K_DARK[4] / K_CLOCK * f) for f in (0.5, 1.0, 2.0)]
        self.assertMonotone(band, msg=f"QBER {band}")
        self.assertClose(band[0], 0.0235569, atol=1e-6, msg=f"half {band[0]:.6f}")
        self.assertClose(band[2], 0.0563117, atol=1e-6, msg=f"double {band[2]:.6f}")

        for got in (band[0], band[2]):
            self.assertGreater(
                abs(got - K_QBER[4]),
                0.01,
                msg=f"QBER {got:.5f}",
            )

    def test_korzh_secret(self):
        """
        qkd's asymptotic COW fraction on the Branciard-Gisin-Scarani phase error exceeds their
        finite-key f_sec at every distance, 1.065 times at 104 km and 2.133 at 307 km.
        """
        ours = [cow_frac(i)[0] for i in range(5)]

        for i in range(5):
            self.assertGreater(
                ours[i],
                K_FSEC[i],
                msg=f"{K_DIST[i]:.0f} km: {ours[i]:.5f}",
            )
        self.assertClose(ours[0], 0.369435, atol=1e-5, msg=f"104 km {ours[0]:.6f}")
        self.assertClose(ours[4], 0.179156, atol=1e-5, msg=f"307 km {ours[4]:.6f}")

        slack = [ours[i] / K_FSEC[i] for i in range(5)]
        self.assertClose(slack[0], 1.064655, atol=1e-4, msg=f"104 km {slack[0]:.5f}")
        self.assertClose(slack[4], 2.132814, atol=1e-4, msg=f"307 km {slack[4]:.5f}")
        self.assertGreater(slack[4], slack[3], msg=f"slack {slack}")

    def test_korzh_form(self):
        """
        ``cow_rate`` is $1 - h_2(e_{phase}) - f h_2(e_{bit})$, above Korzh's
        Branciard-Gisin-Scarani form by exactly $Q(1 - h_2(e_{phase}))$, 9.9% at 307 km.
        """
        for i in range(5):
            ours, theirs = cow_frac(i)
            gap = K_QBER[i] * (1.0 - h2(cow_phase(i)))
            self.assertClose(
                ours - theirs,
                gap,
                atol=1e-12,
                msg=f"{K_DIST[i]:.0f} km: {ours - theirs:.9f}",
            )
            self.assertGreater(ours, theirs, msg=f"{ours:.6f}, {theirs:.6f}")

        ours, theirs = cow_frac(4)
        self.assertClose(ours - theirs, 0.0161298, atol=1e-6, msg=f"{ours - theirs:.7f}")
        self.assertClose(ours / theirs - 1.0, 0.098940, atol=1e-5, msg=f"{ours / theirs - 1.0:.6f}")

    def test_korzh_penalty(self):
        """
        Their Eq. (3) fraction less the 7 sqrt(n_cpp log2 1/beta) term reproduces f_sec to 0.05%
        at 104 km (block 1.97e7) and misses by 38% at 307 km (6.6e5), the unmodelled Eq. (2)
        visibility deviation.
        """
        got = [cow_frac(i)[1] - cow_penalty(i) for i in range(5)]
        slack = [got[i] / K_FSEC[i] for i in range(5)]

        self.assertClose(
            cow_penalty(4),
            0.0471132,
            atol=1e-6,
            msg=f"307 km term {cow_penalty(4):.7f}",
        )
        self.assertClose(slack[0], 1.000503, atol=1e-5, msg=f"104 km {slack[0]:.6f}")
        self.assertClose(slack[4], 1.379921, atol=1e-5, msg=f"307 km {slack[4]:.6f}")

        for i in range(5):
            self.assertGreater(
                got[i],
                K_FSEC[i],
                msg=f"{K_DIST[i]:.0f} km: {got[i]:.5f}",
            )
        self.assertMonotone(slack[2:], msg=f"slack {slack}")

    def test_korzh_ceiling(self):
        """
        Their secret rates sit at 12.1% (307 km) to 45.0% (203 km) of the per-click ceiling;
        declared miss: against their sifted block the ceiling reads 0.672 and 0.689 at 203 and
        307 km.
        """
        caps = [cow_slot(i)[0] * K_CLOCK * 0.5 for i in range(5)]
        share = [K_RATE[i] / caps[i] for i in range(5)]

        for i in range(5):
            self.assertLess(share[i], 1.0, msg=f"{K_DIST[i]:.0f} km share {share[i]:.4f}")
            self.assertGreater(share[i], 0.05, msg=f"{K_DIST[i]:.0f} km share {share[i]:.4f}")
        self.assertClose(share[4], 0.120643, atol=1e-5, msg=f"307 km {share[4]:.6f}")
        self.assertClose(share[2], 0.449547, atol=1e-5, msg=f"203 km {share[2]:.6f}")

        # DECLARED MISS: against the sifted rate n_cpp/t_cpp the ceiling owes >= 1; it fails at 203 and 307 km.
        over = [caps[i] / (K_BLOCK[i] / K_TIME[i]) for i in range(5)]
        self.assertClose(over[0], 1.946584, atol=1e-5, msg=f"104 km sifted headroom {over[0]:.6f}")
        self.assertClose(over[2], 0.671603, atol=1e-5, msg=f"203 km sifted headroom {over[2]:.6f}")
        self.assertClose(over[4], 0.688712, atol=1e-5, msg=f"307 km sifted headroom {over[4]:.6f}")
        self.assertLess(over[2], 1.0, msg=f"203 km {over[2]:.6f}")
        self.assertLess(over[4], 1.0, msg=f"307 km {over[4]:.6f}")

    def test_korzh_dead(self):
        """
        Model click rate plus measured dark counts uses 61.9% of the 1/dead_time ceiling at 104
        km, falling to 0.32% at 307 km.
        """
        caps = [cow_slot(i)[0] * K_CLOCK * 0.5 for i in range(5)]
        share = [(caps[i] + K_DARK[i]) * K_DEAD[i] for i in range(5)]

        for i in range(5):
            self.assertLess(share[i], 1.0, msg=f"{K_DIST[i]:.0f} km share {share[i]:.4f}")
            self.assertGreater(share[i], 0.0, msg=f"{K_DIST[i]:.0f} km share {share[i]:.4f}")
        self.assertMonotone(share, rising=False, msg=f"share {share}")

        self.assertClose(share[0], 0.619077, atol=1e-5, msg=f"104 km {share[0]:.6f}")
        self.assertClose(share[4], 0.003157, atol=1e-5, msg=f"307 km {share[4]:.6f}")

    def test_korzh_table(self):
        """
        Source defect: r_sec = n_cpp f_sec / t_cpp holds to 1.2% at 104, 203 and 307 km but
        reads 0.810 at 153 km and 0.446 at 256 km.
        """
        ratio = [K_BLOCK[i] * K_FSEC[i] / K_TIME[i] / K_RATE[i] for i in range(5)]

        for i in (0, 2, 4):
            self.assertLess(
                abs(ratio[i] - 1.0),
                0.012,
                msg=f"{K_DIST[i]:.0f} km reconstructs to {ratio[i]:.4f}",
            )
        self.assertClose(ratio[1], 0.809947, atol=1e-5, msg=f"153 km {ratio[1]:.6f}")
        self.assertClose(ratio[3], 0.446463, atol=1e-5, msg=f"256 km {ratio[3]:.6f}")

        for i in (1, 3):
            self.assertLess(ratio[i], 0.9, msg=f"{K_DIST[i]:.0f} km {ratio[i]:.4f}")


class TakesueDps(Question):
    """
    Tier B: Takesue et al. 2007 (Nature Photonics 1, 343; arXiv:0706.0397), DPS at 10 GHz over
    42.1 dB. Fitted: f(e), selected by their 4.1% threshold. Declared reading: "50 Hz" is the
    detector pair's rate.
    """

    def test_takesue_threshold(self):
        """
        ``dps_rate`` crosses zero at 4.5619% at the Shannon limit, 4.1487% at f = 1.16 and
        4.0123% at f = 1.22, their "approximately 4.1%" at mu = 0.2 selecting f = 1.16.
        """
        zeros = [bisect(lambda e: _core.dps_rate(1.0, e, T_MU, f), 1e-3, 0.2) for f in (1.0, T_FEC, 1.22)]
        self.assertClose(zeros[0], 0.0456188, atol=1e-6, msg=f"Shannon {zeros[0]:.7f}")
        self.assertClose(zeros[1], 0.0414866, atol=1e-6, msg=f"f=1.16 {zeros[1]:.7f}")
        self.assertClose(zeros[2], 0.0401228, atol=1e-6, msg=f"f=1.22 {zeros[2]:.7f}")
        self.assertLess(
            abs(zeros[1] - T_LIMIT),
            0.0006,
            msg=f"f=1.16 {zeros[1]:.5f}",
        )
        self.assertMonotone(zeros, rising=False, msg=f"zeros {zeros}")

    def test_takesue_reach(self):
        """
        Fitting only f(e), the rate crosses zero at 42.109 dB against their 42.1 dB, and their
        12.1 bit/s at 200 km inverts to 41.700 dB, 0.2085 dB/km.
        """
        reach = dps_reach()

        self.assertClose(reach, 42.109023, atol=1e-5, msg=f"reach {reach:.6f} dB")
        self.assertLess(abs(reach - T_MAX), 0.05, msg=f"reach {reach:.4f} dB")
        self.assertGreater(dps_rate(T_MAX - 0.5), 0.0, msg="rate at 41.6 dB")
        self.assertEqual(dps_rate(T_MAX + 1.0), 0.0, msg="rate at 43.1 dB")

        loss = bisect(lambda x: dps_rate(x) - T_FAR, 20.0, 60.0)
        self.assertClose(loss, 41.700382, atol=1e-5, msg=f"200 km loss {loss:.6f} dB")
        self.assertClose(loss / 200.0, 0.208502, atol=1e-6, msg=f"{loss / 200.0:.6f}")
        self.assertLess(loss, dps_reach(), msg=f"200 km loss {loss:.4f} dB")
        self.assertLess(dps_reach() - loss, 0.5, msg=f"gap {dps_reach() - loss:.4f} dB")
        self.assertGreater(loss / 200.0, 0.19, msg=f"{loss / 200.0:.4f} dB/km")

    def test_takesue_near(self):
        """
        At 105 km (21.0 dB) the model gives 17.42 kbit/s against their 17 kbit/s, from 80.07
        kclick/s at 2.315% QBER with a 0.64% background share.
        """
        rate = dps_rate(T_ALPHA * 105.0)
        click, err = dps_slot(T_ALPHA * 105.0)

        self.assertClose(rate, 17418.52, atol=0.1, msg=f"105 km {rate:.2f} bit/s")
        self.assertLess(abs(rate / T_NEAR - 1.0), 0.03, msg=f"105 km {rate:.0f} bit/s")
        self.assertClose(T_CLOCK * click, 80070.38, atol=0.1, msg=f"sifted {T_CLOCK * click:.2f}/s")
        self.assertClose(err, 0.0231489, atol=1e-6, msg=f"QBER {err:.6f}")

        share = 1.0 - T_BASE / err
        self.assertClose(share, 0.0064336, atol=1e-6, msg=f"dark share {share:.6f}")

    def test_takesue_window(self):
        """
        Dropping their 36% window cost gives 27.3 kbit/s at 105 km against 17 and a 44.05 dB
        reach against 42.1, excluding that reading.
        """
        rate = dps_rate(T_ALPHA * 105.0, eta=T_ETA)

        self.assertClose(rate, 27302.63, atol=0.1, msg=f"uncosted {rate:.2f} bit/s")
        self.assertGreater(rate / T_NEAR - 1.0, 0.5, msg=f"excess {rate / T_NEAR - 1.0:.4f}")

        reach = dps_reach(eta=T_ETA)
        self.assertClose(reach, 44.047223, atol=1e-5, msg=f"uncosted reach {reach:.6f}")
        self.assertGreater(reach - T_MAX, 1.5, msg=f"uncosted reach {reach:.4f}")

    def test_takesue_dark(self):
        """
        Reading "50 Hz" per detector gives a 39.099 dB reach and 0.1944 dB/km at 200 km, against
        42.109 dB and 0.2085 dB/km for the pair reading, bracketing their 42.1 dB.
        """
        each = T_DARK * T_GATE
        reach = dps_reach(dark=each)

        self.assertClose(reach, 39.098723, atol=1e-5, msg=f"per-detector {reach:.6f}")
        self.assertLess(reach, T_MAX, msg=f"per-detector {reach:.4f}")
        self.assertGreater(dps_reach(), T_MAX, msg="pair reach")

        loss = bisect(lambda x: dps_rate(x, dark=each) - T_FAR, 20.0, 60.0)
        self.assertClose(loss / 200.0, 0.194444, atol=1e-5, msg=f"{loss / 200.0:.6f}")
        self.assertLess(
            loss / 200.0,
            T_ALPHA,
            msg=f"{loss / 200.0:.6f} dB/km",
        )

    def test_takesue_slow(self):
        """
        Note 23's 1 GHz run gives 3.714% QBER at 41.7 dB, under its 4.1% threshold, and reaches
        42.520 dB.
        """
        rate, err = slow_rate(S_LOSS)

        self.assertClose(err, 0.0371356, atol=1e-6, msg=f"1 GHz QBER {err:.6f}")
        self.assertLess(err, T_LIMIT, msg=f"1 GHz QBER {err:.5f}")
        self.assertGreater(rate, 0.0, msg=f"1 GHz rate {rate:.4f}")

        reach = bisect(lambda a: slow_rate(a)[0], 20.0, 60.0)
        self.assertClose(reach, 42.520363, atol=1e-5, msg=f"1 GHz reach {reach:.6f}")
        self.assertGreater(reach, S_LOSS, msg=f"1 GHz reach {reach:.4f}")


class LucamariniDecoy(Question):
    """
    Tier B: Lucamarini et al. 2013 (Opt. Express 21, 24550; arXiv:1310.0240), biased-basis decoy
    BB84 over 35-80 km. Fitted: f(e), and the fibre attenuation for the distance ladder only.
    """

    def test_t12_sifting(self):
        """
        $p_X = 1/16$ alone gives 11.72% discarded, 88.28% kept and a 76.56% gain over BB84's
        half, against their 11.7%, 88.3% and 76.6%.
        """
        kept = (1.0 - L_PX) ** 2 + L_PX * L_PX

        self.assertClose(kept, 0.8828125, atol=1e-12, msg=f"kept {kept:.7f}")
        self.assertClose(1.0 - kept, 0.1171875, atol=1e-12, msg=f"lost {1.0 - kept:.7f}")
        self.assertClose(kept / 0.5 - 1.0, 0.765625, atol=1e-12, msg=f"gain {kept / 0.5 - 1.0:.6f}")

        for got, want in (
            (kept, 0.883),
            (1.0 - kept, 0.117),
            (kept / 0.5 - 1.0, 0.766),
        ):
            self.assertLess(abs(got - want), 5e-4, msg=f"{got:.6f}")

    def test_t12_counts(self):
        """
        The Z and X readings of each gain agree to 0.075% at u, 2.9% at v and 3.4% at w, where
        the minority-basis samples are 2.8e4 and 5.8e3 counts.
        """
        pairs = [(t12_gain(i), t12_gain(i, wide=False)) for i in range(3)]

        self.assertClose(pairs[0][0], 4.812305e-3, atol=1e-9, msg=f"u gain {pairs[0][0]:.6e}")

        split = [abs(a / b - 1.0) for a, b in pairs]
        self.assertClose(split[0], 7.4705e-4, atol=1e-7, msg=f"u split {split[0]:.6e}")
        self.assertClose(split[1], 0.0291945, atol=1e-6, msg=f"v split {split[1]:.6f}")
        self.assertClose(split[2], 0.0335828, atol=1e-6, msg=f"w split {split[2]:.6f}")
        self.assertLess(split[0], 0.1 * split[1], msg=f"split {split}")

    def test_t12_background(self):
        """
        With nothing fitted, two dark gates at 2.1e-5 plus 5.25% afterpulsing on the u-line rate
        predict a 2.9465e-4 background against the 2.9493e-4 their w line measures.
        """
        back = t12_back()
        trans = t12_model()[0]
        sig = 1.0 - math.exp(-trans * L_MU[2])
        seen = (t12_gain(2) - sig) / (1.0 - sig)

        self.assertClose(back, 2.946456e-4, atol=1e-9, msg=f"predicted {back:.6e}")
        self.assertClose(seen, 2.949381e-4, atol=1e-9, msg=f"measured {seen:.6e}")
        self.assertLess(abs(back / seen - 1.0), 2e-3, msg=f"ratio {back / seen:.6f}")

        plain = 1.0 - (1.0 - L_DARK) ** 2
        self.assertClose(plain / back, 0.142543, atol=1e-5, msg=f"dark share {plain / back:.6f}")
        self.assertGreater(back / plain, 5.0, msg=f"background over dark {back / plain:.4f}")

    def test_t12_lines(self):
        """
        The forward model calibrated at u predicts 7.633e-4 against a measured 7.537e-4 at v and
        3.053e-4 against 3.056e-4 at w.
        """
        trans, back, wide, _narrow = t12_model()
        model = [t12_line(trans, back, wide, mu)[0] for mu in L_MU]
        seen = [t12_gain(i) for i in range(3)]

        self.assertClose(model[0] / seen[0], 1.0, atol=1e-9, msg=f"u {model[0] / seen[0]:.12f}")
        self.assertClose(model[1] / seen[1], 1.012794, atol=1e-5, msg=f"v {model[1] / seen[1]:.6f}")
        self.assertClose(model[2] / seen[2], 0.999043, atol=1e-5, msg=f"w {model[2] / seen[2]:.6f}")
        self.assertClose(trans, 1.065702e-2, atol=1e-8, msg=f"transmittance {trans:.6e}")

        loss = -10.0 * math.log10(trans / L_ETA)
        self.assertClose(loss, 12.841182, atol=1e-5, msg=f"channel+optics {loss:.6f} dB")

    def test_t12_misalign(self):
        """
        Background is 71.9% of the 4.26% Z QBER and 84.0% of the 3.64% X QBER, leaving
        misalignments of 1.276% and 0.619%, X the cleaner basis as they state.
        """
        back = t12_back()
        errs = t12_model()[2:]

        self.assertClose(errs[0], 0.0127642, atol=1e-6, msg=f"Z misalignment {errs[0]:.6f}")
        self.assertClose(errs[1], 0.0061859, atol=1e-6, msg=f"X misalignment {errs[1]:.6f}")
        self.assertLess(errs[1], errs[0], msg=f"misalignments {errs}")

        share = (
            E_VAC * back / (L_QBER[0] * t12_gain(0)),
            E_VAC * back / (L_QBER[1] * t12_gain(0, wide=False)),
        )
        self.assertClose(share[0], 0.718633, atol=1e-5, msg=f"Z share {share[0]:.6f}")
        self.assertClose(share[1], 0.840409, atol=1e-5, msg=f"X share {share[1]:.6f}")

    def test_t12_decoy(self):
        """
        ``decoy_bounds`` on their intensities gives a single-photon yield 1.09% under the
        infinite-decoy ceiling and phase error bounds above the true value in both bases.
        """
        trans, back, wide, narrow = t12_model()
        y1, e1, q1 = t12_decoy(trans, back, wide)
        far = t12_decoy(trans, back, narrow)[1]
        best = _core.decoy_ideal(trans, back, wide)
        clean = _core.decoy_ideal(trans, back, narrow)[1]

        self.assertClose(y1 / best[0], 0.989058, atol=1e-5, msg=f"y1 {y1 / best[0]:.6f}")
        self.assertLess(y1, best[0], msg=f"y1 {y1:.6e}")
        self.assertClose(e1, 0.0270520, atol=1e-6, msg=f"Z phase bound {e1:.6f}")
        self.assertClose(far, 0.0202818, atol=1e-6, msg=f"X phase bound {far:.6f}")
        self.assertGreater(e1, best[1], msg=f"Z phase bound {e1:.6f}")
        self.assertGreater(far, clean, msg=f"X phase bound {far:.6f}")
        self.assertClose(q1, 3.008785e-3, atol=1e-8, msg=f"q1 {q1:.6e}")

    def test_t12_rate(self):
        """
        With $f(e) = 1.0995$ fitted to the 50 km T12 rate, the $p_X = 1/2$ row is predicted at
        0.6546 Mbps against their 0.63.
        """
        got = t12_rate(L_PX)
        self.assertClose(got, L_T12[1], atol=1.0, msg=f"T12 {got / 1e6:.6f} Mbps")

        plain = t12_rate(0.5)
        self.assertClose(plain, 6.546413e5, atol=1e2, msg=f"BB84 {plain / 1e6:.6f} Mbps")
        self.assertLess(
            abs(plain / L_BB84[1] - 1.0),
            0.05,
            msg=f"BB84 {plain / 1e6:.4f} Mbps",
        )
        self.assertGreater(t12_rate(L_PX, f_ec=1.0), L_T12[1], msg="T12 at f = 1.0")
        self.assertLess(t12_rate(L_PX, f_ec=1.22), L_T12[1], msg="T12 at f = 1.22")

    def test_t12_ladder(self):
        """
        At 0.20 dB/km (2.84 dB on Bob's side) the asymptotic rate stays at or above their
        finite-key rate at all four distances, 1.560 times at 80 km.
        """
        got = [t12_rate(L_PX, dist=d) for d in L_DIST]
        slack = [got[i] / L_T12[i] for i in range(4)]

        for i in range(4):
            self.assertGreater(
                got[i],
                L_T12[i] - 1.0,
                msg=f"{L_DIST[i]:.0f} km: {got[i] / 1e6:.4f} Mbps",
            )
        self.assertClose(slack[0], 1.045658, atol=1e-4, msg=f"35 km {slack[0]:.6f}")
        self.assertClose(slack[3], 1.560106, atol=1e-4, msg=f"80 km {slack[3]:.6f}")
        self.assertMonotone(slack[1:], msg=f"slack {slack}")

        bob = -10.0 * math.log10(t12_model()[0] / L_ETA) - L_FIBRE * L_DIST[1]
        self.assertClose(bob, 2.841182, atol=1e-5, msg=f"Bob-side loss {bob:.6f} dB")

    def test_t12_enhance(self):
        """
        Declared miss: the modelled T12-over-BB84 gain at 50 km is 66.5% against their 73.0%,
        the model charging error correction on the noisier Z basis that carries 99.6% of T12's
        key.
        """
        got = t12_rate(L_PX) / t12_rate(0.5) - 1.0
        theirs = L_T12[1] / L_BB84[1] - 1.0

        self.assertClose(got, 0.665034, atol=1e-5, msg=f"modelled gain {got:.6f}")
        self.assertClose(theirs, 0.730159, atol=1e-6, msg=f"their pair {theirs:.6f}")
        self.assertLess(got, theirs, msg=f"gain {got:.6f}")
        self.assertClose(theirs - got, 0.065125, atol=1e-4, msg=f"short by {theirs - got:.5f}")

        kept = (1.0 - L_PX) ** 2 + L_PX * L_PX
        self.assertLess(got, kept / 0.5 - 1.0, msg=f"gain {got:.6f}")


if __name__ == "__main__":
    rc = Exam(
        "ExpKorzhCow",
        "Tier B: Korzh et al. 2015, COW over 307 km of ultra-low-loss fibre",
        "exp_cow_307.md",
    ).run(load(KorzhCow))
    rc |= Exam(
        "ExpTakesueDps",
        "Tier B: Takesue et al. 2007, DPS at 10 GHz over 42.1 dB of channel loss",
        "exp_dps_40db.md",
    ).run(load(TakesueDps))
    rc |= Exam(
        "ExpLucamariniDecoy",
        "Tier B: Lucamarini et al. 2013, decoy-state BB84 with a biased basis",
        "exp_decoy_t12.md",
    ).run(load(LucamariniDecoy))
    sys.exit(rc)
