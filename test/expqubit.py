import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.cache import memo
from kit.forms import bisect, h2
from qkd import _core, pairs

# Naik, Peterson, White, Berglund & Kwiat, "Entangled state quantum
# cryptography: eavesdropping on the Ekert protocol", Phys. Rev. Lett. 84, 4733
# (2000), arXiv:quant-ph/9912105. Tier B for ekert.rs.
#
# The key ladder is Phys. Rev. Lett. 84, 4733's; e-print v1 prints 17452 / 12215 / 5.1 s^-1. Never cite the arXiv
# id for FREE_NAIK, KEY_NAIK or 6.435. PRL text unread (APS 403); 15444/2400 s = 6.435 exactly supports it.
#
# Quoted from their text:
#   S_NAIK, SP_NAIK, D_NAIK  "for the 40 min of collected data, our combined
#                            values were S = -2.665 +/- 0.019, S' = -2.644 +/-
#                            0.019, each a 34-sigma violation of Bell's
#                            inequality". Both negative in Ekert's sign
#                            convention; ekert_rate reads the magnitude.
#   E_NAIK, D_ERR            "the corresponding bit error rate (BER) was
#                            3.06 +/- 0.11%"
#   RAW_NAIK                 "we obtained a total of 24 252 secret key bits ...
#                            corresponding to a raw bit rate of 10.1 s^-1". 24 252
#                            is the RAW block
#   FREE_NAIK                "18 298 error-free bits remained"
#   KEY_NAIK                 "this was further compressed to 15 444 useful
#                            secret bits (a net bit rate of 6.4 s^-1)"
#   PAIRS_NAIK               "Typically we collected 40 useful pairs per second"
#   SECS_NAIK                four independent runs of ~10 min each
#   SIFT_NAIK                "only 1/4 of the data actually contribute to the
#                            raw cryptographic key; half the data are used to
#                            test Bell's inequalities; and 1/4 are not used at
#                            all"
S_NAIK = 2.665
SP_NAIK = 2.644
D_NAIK = 0.019
E_NAIK = 0.0306
D_ERR = 0.0011
RAW_NAIK = 24252
FREE_NAIK = 18298
KEY_NAIK = 15444
PAIRS_NAIK = 40.0
SECS_NAIK = 2400.0
SIFT_NAIK = 0.25

# Eavesdropper predictions, not data: "if the eavesdropper measures one photon
# from every pair, then |S_eve| <= sqrt(2)"; "a potential eavesdropper, who
# introduces a minimum BER of 25% if she measures every photon"; "if she
# measures (in the optimal basis) less than 58.6% of the photons, S > 2 and the
# corresponding BER < 15%".
S_EVE = math.sqrt(2.0)
E_EVE = 0.25
P_EVE = 0.586

# Ling, Peloso, Marcikic, Scarani, Lamas-Linares & Kurtsiefer, "Experimental
# quantum key distribution based on a Bell test", Phys. Rev. A 78, 020301(R)
# (2008), arXiv:0805.3629. Privacy amplification priced by the CHSH violation, by Acin, Brunner,
# Gisin, Massar, Pironio & Scarani, Phys. Rev. Lett. 98, 230501 (2007) Eq. (1), as pair_holevo computes.
# S is estimated on the key blocks: source="measured".
#
# Quoted from their text:
#   S_LING     "stayed at around 2.5 over the whole measurement time"
#   COINC_LING "a photo coincidence rate of about 18000 s^-1 ... directly at the
#              source"
#   LOSS_LING  "introducing a link loss of about 3 dB"
#   SIFT_LING  "About a quarter of the pairs in detector combinations (1,2) with
#              (1',2') contributed to the raw key"
#   ACCID_LING "Accidental coincidences were about 0.5% of the coincidences from
#              down-converted photon pairs"
#   KEY_LING   "an average final key rate of around 300 bit s^-1"
#   BITS_LING  "or about 10^7 bit of error-free secret key", over "one 9.5 hour
#              run"
S_LING = 2.5
COINC_LING = 18000.0
LOSS_LING = 3.0
SIFT_LING = 0.25
ACCID_LING = 0.005
KEY_LING = 300.0
BITS_LING = 1e7
HOURS_LING = 9.5

# EXAM-SIDE CHOICE, unpublished: every rate below is a CEILING; one BELOW a measured rate falsifies the model.
F_SHANNON = 1.0

# A background coincidence errs half the time; crate::pairs writes the same E0.
E_VAC = 0.5

# Acin et al. 2007 zero-key error rates, recomputed: violation alone, and characterised qubits (Naik's note [30],
# Shor-Preskill's one-way BB84 zero, root of 1 = 2 h2(e)).
CHSH_ZERO = 0.07149175884448569
BBM92_ZERO = 0.11002786443835955

# Enzer, Hadley, Hughes, Peterson & Kwiat, "Entangled-photon six-state quantum
# cryptography", New J. Phys. 4, 45 (2002), doi:10.1088/1367-2630/4/1/345.
# Simulated eavesdropper in three orthogonal Poincare planes. Benchtop, no distance.
#
# Quoted from their text:
#   RATE_ENZER  "our measured 'raw' key generation rate was 33 s^-1"
#   BITS_ENZER  "a 94 min data collection period yielded a 'sifted' key of
#   MINS_ENZER  55 650 bits that was fairly unbiased (49% '1's), with a BER of
#   E_ENZER     1.7%"
#   TOT_ENZER   "the total BER is 34.0 +/- 1.4%, in agreement with the predicted
#   D_ENZER     value 33.3%, and in contrast with 25%, the corresponding result
#   BB_ENZER    for the BB84 protocol"
#   PEAK_ENZER  "The resulting BER curves were similar to those of figure 2(a)
#   AVG_ENZER   but peaked at only 11% and averaged to only 7%"
#   CROSS_ENZER "BB84 actually has a higher yield for BERs below ~8-10%"
#   DEAD_ENZER  "there is never any yield for BERs greater than 10-15%"
RATE_ENZER = 33.0
BITS_ENZER = 55650.0
MINS_ENZER = 94.0
E_ENZER = 0.017
TOT_ENZER = 0.340
D_ENZER = 0.014
BB_ENZER = 0.25
PEAK_ENZER = 0.11
AVG_ENZER = 0.07
CROSS_ENZER = (0.08, 0.10)
DEAD_ENZER = (0.10, 0.15)

# Jeong, Kim & Kim, "An experimental comparison of BB84 and SARG04 quantum key
# distribution protocols", Laser Phys. Lett. 11, 095201 (2014),
# doi:10.1088/1612-2011/11/9/095201. No arXiv version. BB84 and SARG04 on ONE rig, differing only in sifting
# and post-processing.
#
# Quoted from their text:
#   V_JEONG     "the same visibility (V = 0.954) of the quantum channel"
#   ETA_JEONG   "detection efficiency (eta_det) of ~0.6", "eta_opt = 0.72, so
#               overall Bob's detection efficiency is eta_Bob = eta_opt eta_det
#               ~ 0.40". The product is 0.432; 0.40 runs, and test_sarg_ladder's
#               miss reverses under 0.432
#   DARK_JEONG  "dark-count probability (3.3 +/- 0.6) x 10^-5", per detector
#   ALPHA_JEONG "attenuation coefficient is alpha = 3 dB km^-1" at 780 nm
#   DIST_JEONG  "1.27 km single-mode fiber"
#   CLOCK_JEONG "3 ns laser pulses at 1 MHz repetition rate"
#   SIFT_JEONG  "The measured protocol efficiency (ratio of sifted bits to the
#   D_JEONG     number of raw bits received) was 0.50 +/- 0.01 for BB84 and
#               0.25 +/- 0.01 for SARG04"
#   QBB_JEONG   "the QBER (BB84 ~3%; SARG04 ~5%) did not change much when the
#   QSG_JEONG   dark-count contributions were subtracted"
#   MU_JEONG    the eight average photon numbers on their figure 2 axis, and
#   SIFT_BB84   the sifted key rates in kbit/s printed on the bars. The last two
#   SIFT_SARG   columns were not checked and are not used.
V_JEONG = 0.954
ETA_JEONG = 0.40
DARK_JEONG = 3.3e-5
ALPHA_JEONG = 3.0
DIST_JEONG = 1.27
CLOCK_JEONG = 1e6
SIFT_JEONG = 0.25
D_JEONG = 0.01
QBB_JEONG = 0.03
QSG_JEONG = 0.05
MU_JEONG = (0.03, 0.06, 0.09, 0.13, 0.16, 0.19)
SIFT_BB84 = (2.7, 5.4, 8.2, 10.9, 13.6, 16.6)
SIFT_SARG = (1.4, 2.8, 4.3, 5.7, 7.1, 8.6)

# Four detectors composed into one per-pulse background; mirrors Link._background.
DARK_BOB = 1.0 - (1.0 - DARK_JEONG) ** 4


# Hughes, Morgan & Peterson, "Practical quantum key distribution over a 48-km
# optical fiber network", arXiv:quant-ph/9904038, published as J. Mod. Opt. 47,
# 533 (2000). Interferometric B92 over installed fibre at Los Alamos.
#
# PLAIN B92: figure 5's 1.55 um bright pulse is a TIMING pulse, not Koashi's strong reference. Neither Koashi's
# proof nor Tamaki & Lutkenhaus's (single photon) covers mu = 0.63. QBER reproduced; rates only bounded.
#
# Quoted from their text:
#   LOSS_HMP    "22.9 dB of attenuation owing to the fibre's 0.3-dB/km
#               attenuation and seven connections along the path", over 48 km
#   CLOCK_HMP   "a laser pulse rate of 100 kHz"
#   ETA_HMP     "a detection efficiency of 11%"
#   MU_HMP      "average central peak photon number of 0.63 leaving Alice's
#               interferometer ... assuming Poisson photon number statistics"
#   VIS_HMP     "high visibility single-photon interference (98.99+/-1.24% after
#   D_VIS       background subtraction)"
#   FRINGE      "the 730-ps wide central peak in the dphi = pi/2 graph would
#               correspond to 10,668 bits identified as '1's; the dphi = 3pi/2
#               graph would correspond to 10,856 bits identified as '0's; and
#               the dphi = pi graph corresponds to 1,102 errors"
#   SAMPLE      "A sample of 128 detected bits containing 6 errors"
#   BER_HMP     "The bit error rate (BER), defined as the ratio of errors to all
#   DARK_SHARE  events, in the entire data set is ~9.3%. (Approximately 90% of
#               the errors are attributable to detector dark counts.)"
#   KEY_HMP     "the ~10-Hz key generation rate of our QKD system"
#   EMPTY_HMP   "for 53% of the laser pulses no photon leaves Alice's
#               interferometer"
#   MULTI_HMP   "28% of the detectable laser pulses leaving Alice's
#               interferometer contain two or more photons because we operate at
#               an average photon number per pulse of 0.63"
#   MU_LOW      "We have also operated our system at a reduced average photon
#   MULTI_LOW   number per pulse of 0.39, for which only 18% of the detectable
#   KEY_LOW     pulses contain two or more photons. In this case, the key rate
#   BER_LOW     dropped to ~3.4 Hz and the BER increased to ~17.8%"
#   NOISE_HMP   "noise rates, R, of 10-100 kHz"
#   WIN_HMP     the 730 ps central peak, the window an error is counted in
#   SIFT_HMP    "The B92 QKD procedure itself has an intrinsic inefficiency of
#               only identifying one shared bit from four initial bits", and
#               their Eq. (7) puts 1/8 of the central-window photons in a port
LOSS_HMP = 22.9
CLOCK_HMP = 1e5
ETA_HMP = 0.11
MU_HMP = 0.63
MU_LOW = 0.39
VIS_HMP = 0.9899
D_VIS = 0.0124
FRINGE = (10668.0, 10856.0, 1102.0)
SAMPLE = (128.0, 6.0)
BER_HMP = 0.093
BER_LOW = 0.178
DARK_SHARE = 0.90
KEY_HMP = 10.0
KEY_LOW = 3.4
EMPTY_HMP = 0.53
MULTI_HMP = 0.28
MULTI_LOW = 0.18
NOISE_HMP = (1e4, 1e5)
WIN_HMP = 730e-12
SIFT_HMP = 0.25


# Gordon, Fernandez, Townsend & Buller, "A Short Wavelength GigaHertz Clocked
# Fiber-Optic Quantum Key Distribution System", IEEE J. Quantum Electron. 40,
# 900 (2004), arXiv:quant-ph/0605222. Polarisation-encoded plain B92, QBER at nine lengths and two clock rates.
#
# ONLY THE 100 MHZ ROWS: b92_detect does not model Eq. (7)'s temporal term R(dt)_nu, which splits the 1 GHz
# zero-length QBER 1.6% / 5.3%. The window pairs are the same bit-period fractions, not widths (10 ns, 1 ns).
#
# Quoted from their text and their tables I and II:
#   MU_GOR      "a mean number of 0.1 photons per laser pulse"
#   ALPHA_GOR   "transmission fiber losses of ~2.2 dBkm-1" at 850 nm
#   NS_GOR      "typical system insertion losses of ~17.0 dB. This loss includes
#               the 75% loss caused by the unambiguous bit filtering technique
#               at the PBS's and the ~40% detection efficiency of the Si SPAD's"
#   DARK_GOR    "a dark count rate of ~180 counts-1 each", per Si SPAD
#   CLOCK_GOR   the 100 MHz clock, so a 10 ns bit period
#   WIN_GOR     "For 100 MHz two values of dt were chosen 50% and 90%, 5 ns and
#               9 ns respectively, where the bit width was 10 ns"
#   DIST_GOR    the distance column of their table I. THE AXIS IS LOSS, NOT
#               FIBRE: "When a required fiber length was not readily available,
#               extra attenuation could be added using the attenuator in order
#               to simulate the losses induced by a specific length of standard
#               telecommunications fiber." The fit is against dB; 2.2 dB/km
#               only names the axis
#   SIFT_5, Q_5 the R_sift and QBER columns at dt = 5 ns
#   SIFT_9, Q_9 the same at dt = 9 ns
#   FAST_GOR    their table II at 1 GHz and zero fibre, 1.6% and 5.3%, whose
#               windows are "50%, and 98%, giving 0.5 ns, and 0.98 ns
#               respectively, where the bit width was 1 ns"
MU_GOR = 0.1
ALPHA_GOR = 2.2
NS_GOR = 17.0
DARK_GOR = 180.0
CLOCK_GOR = 1e8
WIN_GOR = (5e-9, 9e-9)
DIST_GOR = (0.0, 2.15, 3.75, 4.19, 6.16, 8.08, 9.96, 11.07, 11.85)
SIFT_5 = (61948.0, 20419.0, 8569.0, 7805.0, 2882.0, 1220.0, 554.0, 493.0, 291.0)
Q_5 = (0.004, 0.007, 0.014, 0.013, 0.030, 0.069, 0.156, 0.243, 0.318)
SIFT_9 = (117698.0, 38675.0, 15839.0, 14807.0, 5361.0, 2192.0, 917.0, 717.0, 425.0)
Q_9 = (0.004, 0.008, 0.015, 0.014, 0.030, 0.070, 0.157, 0.284, 0.323)
FAST_GOR = (0.016, 0.053)

# The last two lengths are a declared miss (test_gordon_tail).
NEAR_GOR = 7


def naik_mean():
    """
    Mean CHSH magnitude of S and S', from disjoint setting quadruples of one run.
    """

    return 0.5 * (S_NAIK + SP_NAIK)


def ling_raw():
    """
    Raw key bits per second, an UPPER bound: receiver optics beyond the link are not itemised.
    """

    return SIFT_LING * COINC_LING * 10.0 ** (-LOSS_LING / 10.0)


def ling_band():
    """
    Raw-key bit error rate S = 2.5 implies on a depolarised singlet.
    """

    return pairs.disturbance(S_LING)


def ling_rate(e_bit, f_ec=F_SHANNON):
    """
    Final key in bit/s at ``e_bit`` on the measured violation, through ekert_rate's
    source="measured".
    """
    frac = _core.ekert_rate(1.0, e_bit, S_LING, f_ec, 1.0, "measured")

    return frac * ling_raw()


def ling_qber(key=KEY_LING, f_ec=F_SHANNON):
    """
    Raw-key bit error rate the published throughput implies, inverted on the measured S.
    """

    return bisect(lambda e: -ling_rate(e, f_ec), 0.0, CHSH_ZERO, -key)


def enzer_attack(theta, strength=1.0):
    """
    Per-basis error rates on H/V, L/R, D/d of a projective attack in basis cos(theta)|H> +
    sin(theta)|V> at ``strength``.
    """
    half = 0.5 * strength

    return (
        half * math.sin(2.0 * theta) ** 2,
        half,
        half * math.cos(2.0 * theta) ** 2,
    )


def enzer_totals(theta, strength=1.0):
    """
    (six-state BER, BB84 BER) under that attack, read from the Bell-diagonal weights.
    """
    l1, l2, l3, _l4 = _core.sixstate_bell(*enzer_attack(theta, strength))
    total = 2.0 * (1.0 - l1)

    return total / 3.0, 0.5 * (total - l2 - l3)


def enzer_yields(qber, f_ec=F_SHANNON):
    """
    (six-state, BB84) secret bits per DETECTED pair at a depolarising error rate, each behind
    its own 1/3 or 1/2 sifting.
    """
    six = _core.sixstate_sift(1.0 / 3.0)[0] * _core.sixstate_secret(qber, qber, qber, f_ec)

    return six, 0.5 * max(0.0, 1.0 - f_ec * h2(qber) - h2(qber))


def jeong_eta(alpha=ALPHA_JEONG, eta=ETA_JEONG):
    """
    End-to-end detection probability of a photon leaving Alice over their spool.
    """

    return _core.decoy_eta(alpha, DIST_JEONG, eta)


def jeong_gains(mu, vis=V_JEONG, dark=DARK_BOB):
    """
    ((Q, E) BB84, (Q, E) SARG04) per emitted pulse at visibility ``vis``; sarg_gain includes
    SARG04's 1/4 sifting, decoy_gain excludes BB84's 1/2.
    """
    mis = 0.5 * (1.0 - vis)
    eta = jeong_eta()

    return (
        _core.decoy_gain(mu, eta, dark, mis),
        _core.sarg_gain(mu, eta, dark, mis),
    )


def jeong_ratio(vis=V_JEONG):
    """
    SARG04-to-BB84 sifted-rate ratio by Fung, Tamaki & Lo Eq. (40): (e_det/2 + 1/4) over 1/2.
    """

    return _core.sarg_sift(0.5 * (1.0 - vis)) / 0.5


def hughes_trans():
    """
    End-to-end transmittance, their "factor of two hundred".
    """

    return 10.0 ** (-LOSS_HMP / 10.0)


def hughes_detect(mu, dark, vis=VIS_HMP):
    """
    (conclusive probability, bit error rate) per pulse through b92_detect, in qkd's
    normalisation, not theirs (``hughes_chain``).
    """

    return _core.b92_detect(mu, hughes_trans(), ETA_HMP, vis, dark)


def hughes_chain(mu):
    """
    Conclusive probability per pulse from THEIR stated factors.
    """

    return mu * hughes_trans() * SIFT_HMP * ETA_HMP


def hughes_dark(ber=BER_HMP, mu=MU_HMP):
    """
    The one fitted parameter: per-gate background putting b92_detect's bit error rate on
    ``ber``.
    """

    return bisect(lambda d: hughes_detect(mu, d)[1], 0.0, 1e-2, ber)


@memo
def gordon_setup(window, sift, qber):
    """
    (background per SPAD per bit period, zero-length transmittance, visibility) for one window:
    background published, transmittance inverted from the zero-length sift, visibility the ONE
    fitted parameter.
    """
    dark = DARK_GOR * window
    trans = sift / CLOCK_GOR / (4.0 * MU_GOR)
    vis = bisect(lambda v: -_core.b92_detect(MU_GOR, trans, 1.0, v, dark)[1], 0.9, 1.0, -qber)

    return dark, trans, vis


def gordon_curve(window, sift, qber, dark=None):
    """
    Modelled QBER at their nine lengths on that window's zero-length fit; ``dark`` overrides the
    published background.
    """
    fitted, trans, vis = gordon_setup(window, sift, qber)
    used = fitted if dark is None else dark

    return [_core.b92_detect(MU_GOR, trans * 10.0 ** (-ALPHA_GOR * d / 10.0), 1.0, vis, used)[1] for d in DIST_GOR]


class NaikEkert(Question):
    """
    Tier B: Naik et al. 2000 (Phys. Rev. Lett. 84, 4733; arXiv:quant-ph/9912105), the Ekert
    protocol. Pinned: S, S', BER, raw block, pair rate, PRL key ladder. Nothing fitted.
    """

    def test_naik_bridge(self):
        """
        The bridge Q = (1 - |S|/2sqrt2)/2 maps the mean of S and S' to 3.0746% against the
        measured 3.06 +/- 0.11% (0.13 sigma), S and S' alone bracketing it at 1.55 sigma below
        and 1.82 above.
        """
        gap = abs(S_NAIK - SP_NAIK) / (math.sqrt(2.0) * D_NAIK)
        self.assertLess(gap, 1.0, msg=f"S, S' gap {gap:.4f} sigma")

        got = pairs.disturbance(naik_mean())
        self.assertClose(got, 0.0307459, atol=1e-6, msg=f"mean QBER {got:.6f}")
        self.assertLess(
            abs(got - E_NAIK) / D_ERR,
            0.5,
            msg=f"QBER {got:.6f}, {(got - E_NAIK) / D_ERR:+.3f} sigma",
        )

        lo = pairs.disturbance(S_NAIK)
        hi = pairs.disturbance(SP_NAIK)

        self.assertClose(lo, 0.028890, atol=1e-6, msg=f"S reads QBER {lo:.6f}")
        self.assertClose(hi, 0.032602, atol=1e-6, msg=f"S' reads QBER {hi:.6f}")
        self.assertLess(lo, E_NAIK, msg=f"S QBER {lo:.6f}")
        self.assertGreater(hi, E_NAIK, msg=f"S' QBER {hi:.6f}")

        for edge in (lo, hi):
            self.assertGreater(
                abs(edge - E_NAIK) / D_ERR,
                1.0,
                msg=f"edge {edge:.6f}",
            )

    def test_naik_eavesdropper(self):
        """
        The same bridge gives exactly 25% BER at Ekert's |S_eve| <= sqrt(2), and (2 - sqrt2)/4 =
        14.645% at S = 2, an intercepted share of 58.579% against their 58.6%.
        """
        got = pairs.disturbance(S_EVE)
        self.assertClose(got, E_EVE, atol=1e-15, msg=f"S_eve QBER {got:.6f}")

        lost = pairs.disturbance(2.0)
        self.assertClose(lost, 0.14644660940672627, atol=1e-15, msg=f"S = 2 QBER {lost:.6f}")
        self.assertLess(lost, 0.15, msg=f"S = 2 QBER {lost:.6f}")

        share = lost / E_EVE
        self.assertClose(share, 2.0 - math.sqrt(2.0), atol=1e-15, msg=f"fraction {share:.6f}")
        self.assertClose(share, P_EVE, atol=5e-4, msg=f"fraction {share:.6f}")

    def test_naik_sifting(self):
        """
        24 252 raw bits from 40 pairs/s over 2400 s is 25.263% of pairs against their table I's
        1/4, and 10.105 bit/s against their 10.1.
        """
        share = RAW_NAIK / (PAIRS_NAIK * SECS_NAIK)
        self.assertClose(share, 0.252625, atol=1e-6, msg=f"key share {share:.6f}")
        self.assertClose(share, SIFT_NAIK, atol=0.005, msg=f"key share {share:.6f}")

        rate = RAW_NAIK / SECS_NAIK
        self.assertClose(rate, 10.105, atol=1e-3, msg=f"raw rate {rate:.4f} bit/s")
        self.assertClose(KEY_NAIK / SECS_NAIK, 6.435, atol=1e-3, msg=f"net rate {KEY_NAIK / SECS_NAIK:.4f}")

    def test_naik_price(self):
        """
        The collective-attack CHSH price on their violation is 0.34178 bit per raw bit, 2.90
        times the 0.11768 their privacy amplification charged.
        """
        charged = (FREE_NAIK - KEY_NAIK) / RAW_NAIK
        priced = pairs.holevo(naik_mean())
        self.assertClose(charged, 0.117681, atol=1e-6, msg=f"PA charge {charged:.6f}")
        self.assertClose(priced, 0.341781, atol=1e-6, msg=f"CHSH price {priced:.6f}")
        self.assertClose(
            priced / charged,
            2.9043,
            atol=1e-3,
            msg=f"price ratio {priced / charged:.4f}",
        )

    def test_naik_modern(self):
        """
        Declared miss: their 24.551% EC leakage (1.244 times Shannon) plus the CHSH price leaves
        10 009 bits, 64.8% of their published 15 444.
        """
        leak = (RAW_NAIK - FREE_NAIK) / RAW_NAIK
        self.assertClose(leak, 0.245506, atol=1e-6, msg=f"EC leakage {leak:.6f}")
        self.assertClose(
            leak / h2(E_NAIK),
            1.2438,
            atol=1e-3,
            msg=f"EC over Shannon {leak / h2(E_NAIK):.4f}",
        )

        bits = RAW_NAIK * (1.0 - leak - pairs.holevo(naik_mean()))
        self.assertClose(bits, 10009.13, atol=0.05, msg=f"bits {bits:.2f}")
        self.assertLess(bits, KEY_NAIK, msg=f"bits {bits:.2f}")
        self.assertClose(
            bits / KEY_NAIK,
            0.6481,
            atol=1e-3,
            msg=f"kept {bits / KEY_NAIK:.4f}",
        )

    def test_naik_margin(self):
        """
        Their 3.06% sits 4.09 points inside the CHSH-priced zero of 7.149% and 7.94 inside the
        characterised-device 11.003% of their note [30].
        """
        self.assertClose(
            _core.ekert_threshold(F_SHANNON),
            CHSH_ZERO,
            atol=1e-12,
            msg="CHSH-priced zero",
        )
        self.assertLess(E_NAIK, CHSH_ZERO, msg=f"BER {E_NAIK}")
        self.assertClose(
            CHSH_ZERO - E_NAIK,
            0.040892,
            atol=1e-5,
            msg=f"CHSH margin {CHSH_ZERO - E_NAIK:.6f}",
        )
        self.assertLess(BBM92_ZERO, 0.11003, msg=f"BBM92 zero {BBM92_ZERO}")


class LingEkert(Question):
    """
    Tier B: Ling et al. 2008 (Phys. Rev. A 78, 020301(R); arXiv:0805.3629), modified Ekert91
    over 1.5 km free space. Free: reconciliation inefficiency, at the Shannon limit, so every
    prediction is a ceiling.
    """

    def test_ling_throughput(self):
        """
        10^7 bit over 9.5 hours is 292.40 bit/s against their "around 300 bit/s".
        """
        got = BITS_LING / (HOURS_LING * 3600.0)

        self.assertClose(got, 292.3977, atol=1e-3, msg=f"{got:.4f} bit/s")
        self.assertLess(got, KEY_LING, msg=f"{got:.4f} bit/s")
        self.assertLess(
            abs(got / KEY_LING - 1.0),
            0.03,
            msg=f"throughput gap {got / KEY_LING - 1.0:.4f}",
        )

    def test_ling_ceiling(self):
        """
        S = 2.5 on a depolarised singlet fixes the bit error at 5.806%, and the Shannon-limit
        rate is 308.41 bit/s against the measured 292.40-300.
        """
        e_bit = ling_band()
        self.assertClose(e_bit, 0.058058, atol=1e-6, msg=f"QBER {e_bit:.6f}")

        got = ling_rate(e_bit)
        self.assertClose(got, 308.41, atol=0.05, msg=f"rate {got:.2f} bit/s")
        self.assertGreater(got, KEY_LING, msg=f"rate {got:.2f} bit/s")
        self.assertLess(
            got / KEY_LING - 1.0,
            0.05,
            msg=f"headroom {got / KEY_LING - 1.0:.4f}",
        )

    def test_ling_qber(self):
        """
        Inverting the 300 bit/s throughput gives a 5.899% bit error against the 5.806% the
        violation implies, 0.093 points apart.
        """
        got = ling_qber()
        band = ling_band()

        self.assertClose(got, 0.058995, atol=1e-5, msg=f"throughput QBER {got:.6f}")
        self.assertClose(
            got,
            band,
            atol=1.5e-3,
            msg=f"throughput QBER {got:.6f}, violation {band:.6f}",
        )
        self.assertGreater(got, band, msg=f"throughput QBER {got:.6f}, violation {band:.6f}")

    def test_ling_anisotropy(self):
        """
        Declared caveat: below the depolarising 5.806% the ceiling only rises above 300 bit/s,
        their "residual distinguishability" putting the true QBER lower.
        """
        band = [ling_rate(e) for e in (0.058058, 0.04, 0.02, 0.0)]
        self.assertMonotone(band, msg=f"rates {band}")

        for got in band:
            self.assertGreater(
                got,
                KEY_LING,
                msg=f"rate {got:.2f} bit/s",
            )

    def test_ling_accidentals(self):
        """
        The 0.5% accidental coincidences contribute 0.250 points of error, 4.3% of the 5.806%
        the violation implies.
        """
        share = E_VAC * ACCID_LING
        band = ling_band()

        self.assertClose(share, 0.0025, atol=1e-12, msg=f"accidental error share {share:.6f}")
        self.assertLess(
            share / band,
            0.05,
            msg=f"accidental share {share / band:.4f}",
        )

    def test_ling_assumption(self):
        """
        On their numbers BBM92 pays 813.33 bit/s against the CHSH price's 308.41, 2.637 times.
        """
        e_bit = ling_band()
        chsh = ling_rate(e_bit)
        symmetry = pairs.rate(1.0, e_bit, e_bit, F_SHANNON, 1.0) * ling_raw()

        self.assertClose(symmetry, 813.33, atol=0.05, msg=f"BBM92 {symmetry:.2f} bit/s")
        self.assertGreater(symmetry, chsh, msg=f"BBM92 {symmetry:.2f}, CHSH {chsh:.2f} bit/s")
        self.assertClose(
            symmetry / chsh,
            2.6372,
            atol=5e-4,
            msg=f"ratio {symmetry / chsh:.4f}",
        )

    def test_ling_measured(self):
        """
        The depolarising bridge inverts to S = 2.5 exactly, and ekert_rate refuses
        source="modelled" for this fielded violation.
        """
        e_bit = ling_band()

        self.assertClose(
            pairs.chsh(1.0 - 2.0 * e_bit),
            S_LING,
            atol=1e-12,
            msg="S from bridge",
        )
        self.assertFails(
            ValueError,
            "modelled",
            _core.ekert_rate,
            1.0,
            e_bit,
            S_LING,
            F_SHANNON,
            1.0,
            "modelled",
            msg='source="modelled"',
        )


class EnzerSixState(Question):
    """
    Tier B: Enzer et al. 2002 (New J. Phys. 4, 45), six-state on entangled photons under a
    simulated eavesdropper. Fitted: the partial measurement's strength.
    """

    def test_enzer_intercept(self):
        """
        Intercept-resend in any linear basis gives l1 = 1/2, a six-state BER of 33.333% against
        their 34.0 +/- 1.4% (0.48 sigma), and BB84 25.000% against their 25%.
        """
        for deg in (0.0, 7.5, 15.0, 22.5, 30.0, 45.0):
            weights = _core.sixstate_bell(*enzer_attack(math.radians(deg)))
            six, four = enzer_totals(math.radians(deg))
            self.assertClose(weights[0], 0.5, atol=1e-15, msg=f"l1 = {weights[0]:.15f} at {deg} deg")
            self.assertClose(six, 1.0 / 3.0, atol=1e-15, msg=f"six-state BER {six:.15f} at {deg} deg")
            self.assertClose(four, 0.25, atol=1e-15, msg=f"BB84 BER {four:.15f} at {deg} deg")

        six, four = enzer_totals(math.radians(22.5))
        self.assertLess(
            abs(six - TOT_ENZER) / D_ENZER,
            0.6,
            msg=f"six-state {(six - TOT_ENZER) / D_ENZER:+.3f} sigma",
        )
        self.assertClose(four, BB_ENZER, atol=1e-15, msg=f"BB84 BER {four:.15f}")

    def test_enzer_spread(self):
        """
        Per-basis error rates span 0 to 50% against their "~0 to 50%", with a peak-to-average
        ratio of 3/2.
        """
        floor = min(min(enzer_attack(math.radians(d))) for d in (0.0, 22.5, 45.0))
        roof = max(max(enzer_attack(math.radians(d))) for d in (0.0, 22.5, 45.0))

        self.assertClose(floor, 0.0, atol=1e-15, msg=f"per-basis floor {floor:.15f}")
        self.assertClose(roof, 0.5, atol=1e-15, msg=f"per-basis peak {roof:.15f}")

        rates = enzer_attack(math.radians(22.5))
        self.assertClose(
            max(rates) / (sum(rates) / 3.0),
            1.5,
            atol=1e-15,
            msg="peak-to-average",
        )

    def test_enzer_partial(self):
        """
        Their partial attack's 11%/7% = 1.571 against 3/2 at any strength, the fitted strength
        reading 0.22 from the peak and 0.21 from the average, with l4 = 0.
        """
        ratio = PEAK_ENZER / AVG_ENZER
        self.assertClose(ratio, 1.5714, atol=1e-3, msg=f"measured ratio {ratio:.4f}")
        self.assertLess(
            abs(ratio / 1.5 - 1.0),
            0.05,
            msg=f"miss {ratio / 1.5 - 1.0:.4f}",
        )

        strengths = [2.0 * PEAK_ENZER, 3.0 * AVG_ENZER]
        self.assertClose(strengths[0], 0.22, atol=1e-12, msg=f"strength {strengths[0]:.4f} from peak")
        self.assertClose(strengths[1], 0.21, atol=1e-12, msg=f"strength {strengths[1]:.4f} from mean")

        for got in strengths:
            weights = _core.sixstate_bell(*enzer_attack(math.radians(22.5), got))
            self.assertClose(weights[3], 0.0, atol=1e-15, msg=f"l4 {weights[3]} at strength {got}")

    def test_enzer_detection(self):
        """
        The secret fraction is 0.7883 at their undisturbed 1.7%, zero under intercept-resend for
        every key basis, and 0.3901 under the partial attack at strength 0.22.
        """
        clean = _core.sixstate_secret(E_ENZER, E_ENZER, E_ENZER, F_SHANNON)
        self.assertClose(clean, 0.788287, atol=1e-6, msg=f"undisturbed secret fraction {clean:.6f}")

        rates = enzer_attack(math.radians(22.5))
        keyed = [(rates[i], rates[(i + 1) % 3], rates[(i + 2) % 3]) for i in range(3)]

        for got in keyed:
            dead = _core.sixstate_secret(got[0], got[1], got[2], F_SHANNON)
            self.assertEqual(dead, 0.0, msg=f"secret fraction {dead} at {got}")

        weak = _core.sixstate_secret(*enzer_attack(math.radians(22.5), 0.22), F_SHANNON)
        self.assertClose(weak, 0.390084, atol=1e-6, msg=f"partial attack {weak:.6f}")

    def test_enzer_crossover(self):
        """
        Six-state overtakes BB84 at 8.340% with nothing fitted, inside their "below ~8-10%".
        """
        # Perfect reconciliation (f_ec = 1): Enzer states no f_EC, and the crossing moves with it.
        got = bisect(lambda q: enzer_yields(q)[0] - enzer_yields(q)[1], 0.01, 0.11, 0.0)
        self.assertClose(got, 0.083395, atol=1e-6, msg=f"crossover at {got:.6f}")
        self.assertGreater(got, CROSS_ENZER[0], msg=f"crossover {got:.6f}")
        self.assertLess(got, CROSS_ENZER[1], msg=f"crossover {got:.6f}")

        near = enzer_yields(0.02)
        far = enzer_yields(0.11)
        self.assertGreater(near[1], near[0], msg=f"yields at 2% {near}")
        self.assertGreater(far[0], far[1], msg=f"yields at 11% {far}")

    def test_enzer_dead(self):
        """
        BB84 yield dies at 11.003% and six-state at 12.619%, both inside their "greater than
        10-15%".
        """
        # Perfect reconciliation (f_ec = 1): Enzer states no f_EC, and the zeros move with it.
        zeros = [
            bisect(lambda q: -enzer_yields(q)[1], 0.05, 0.20, -1e-12),
            bisect(lambda q: -enzer_yields(q)[0], 0.05, 0.20, -1e-12),
        ]
        self.assertClose(zeros[0], BBM92_ZERO, atol=1e-6, msg=f"BB84 zero {zeros[0]:.6f}")
        self.assertClose(zeros[1], 0.126193, atol=1e-6, msg=f"six-state zero {zeros[1]:.6f}")

        for got in zeros:
            self.assertGreater(got, DEAD_ENZER[0], msg=f"zero {got:.6f}")
            self.assertLess(got, DEAD_ENZER[1], msg=f"zero {got:.6f}")

    def test_enzer_sifting(self):
        """
        Declared miss: 55 650 sifted bits over 94 min at 33 s^-1 is 29.900% of events against
        sixstate_sift's 1/3, a 10.3% deficit.
        """
        share = BITS_ENZER / (MINS_ENZER * 60.0 * RATE_ENZER)
        self.assertClose(share, 0.299001, atol=1e-6, msg=f"measured share {share:.6f}")
        self.assertClose(
            _core.sixstate_sift(1.0 / 3.0)[0],
            1.0 / 3.0,
            atol=1e-15,
            msg="uniform sifting",
        )
        self.assertLess(share, 1.0 / 3.0, msg=f"measured share {share:.6f}")
        self.assertLess(
            1.0 - 3.0 * share,
            0.12,
            msg=f"deficit {1.0 - 3.0 * share:.4f}",
        )


class JeongSarg(Question):
    """
    Tier B: Jeong, Kim & Kim 2014 (Laser Phys. Lett. 11, 095201), BB84 and SARG04 on one rig
    over 1.27 km. The sifting leg fits nothing; the QBER legs fit one visibility.
    """

    def test_sarg_sifting(self):
        """
        sarg_sift at a perfect frame is exactly 1/4, their measured 0.25 +/- 0.01, half BB84's
        0.50.
        """
        got = _core.sarg_sift(0.0)
        self.assertClose(got, SIFT_JEONG, atol=1e-15, msg=f"ideal sifting {got:.15f}")
        self.assertClose(2.0 * got, 0.5, atol=1e-15, msg=f"twice {2.0 * got}")
        self.assertLess(abs(got - SIFT_JEONG), D_JEONG, msg=f"sifting {got:.6f}")

    def test_sarg_ratio(self):
        """
        At V = 0.954 the SARG04-to-BB84 sifted ratio (e_det/2 + 1/4) over 1/2 is 0.5230, against
        0.52075 averaged over their six figure 2 pairs, each above 1/2.
        """
        rats = [s / b for s, b in zip(SIFT_SARG, SIFT_BB84)]
        mean = sum(rats) / len(rats)

        self.assertClose(mean, 0.520749, atol=1e-6, msg=f"measured ratio {mean:.6f}")

        for got in rats:
            self.assertGreater(got, 0.5, msg=f"ratio {got:.4f}")

        got = jeong_ratio()
        self.assertClose(got, 0.523, atol=1e-12, msg=f"predicted ratio {got:.6f}")
        self.assertLess(
            abs(got / mean - 1.0),
            0.005,
            msg=f"miss {got / mean - 1.0:.5f}",
        )

    def test_sarg_three_ways(self):
        """
        V = 0.954, the sifted-rate ladder's 0.95850, the BB84 QBER's 0.94000 and the SARG04
        QBER's 0.94737 each fix the visibility, a 1.97% spread.
        """
        rats = [s / b for s, b in zip(SIFT_SARG, SIFT_BB84)]
        band = [
            V_JEONG,
            bisect(jeong_ratio, 0.85, 1.0, sum(rats) / len(rats)),
            1.0 - 2.0 * QBB_JEONG,
            bisect(lambda v: -(1.0 - v) / (2.0 - v), 0.85, 1.0, -QSG_JEONG),
        ]
        self.assertClose(band[1], 0.958502, atol=1e-5, msg=f"ladder V {band[1]:.6f}")
        self.assertClose(band[2], 0.94, atol=1e-12, msg=f"BB84 V {band[2]:.6f}")
        self.assertClose(band[3], 0.947368, atol=1e-6, msg=f"SARG04 V {band[3]:.6f}")

        wide = max(band) / min(band) - 1.0
        self.assertLess(wide, 0.025, msg=f"visibility spread {wide:.5f} across {band}")
        self.assertLess(
            abs(band[1] / V_JEONG - 1.0),
            0.006,
            msg=f"ladder miss {band[1] / V_JEONG - 1.0:.5f}",
        )

    def test_sarg_qber_law(self):
        """
        Weak-signal and background-free, the engine gives the paper's (1 - V)/2 = 2.300% for
        BB84 and (1 - V)/(2 - V) = 4.398% for SARG04 at V = 0.954.
        """
        mis = 0.5 * (1.0 - V_JEONG)
        bb, sg = jeong_gains(0.19, dark=0.0)

        self.assertClose(bb[1], mis, atol=1e-15, msg=f"BB84 QBER {bb[1]:.9f}")
        self.assertClose(
            sg[1],
            (1.0 - V_JEONG) / (2.0 - V_JEONG),
            atol=1e-15,
            msg=f"SARG04 QBER {sg[1]:.9f}",
        )
        self.assertClose(sg[1] / bb[1], 1.912, atol=2e-3, msg=f"printed ratio {sg[1] / bb[1]:.4f}")

    def test_sarg_qber_miss(self):
        """
        Declared miss: at mu = 0.19 the engine gives 2.502% and 4.753% against their ~3% and
        ~5%, a shortfall dark counts do not carry.
        """
        bb, sg = jeong_gains(0.19)
        self.assertClose(bb[1], 0.025018, atol=1e-6, msg=f"BB84 QBER {bb[1]:.6f}")
        self.assertClose(sg[1], 0.047533, atol=1e-6, msg=f"SARG04 QBER {sg[1]:.6f}")

        for got, want in ((bb[1], QBB_JEONG), (sg[1], QSG_JEONG)):
            self.assertLess(got, want, msg=f"QBER {got:.6f}")
            self.assertLess(
                want - got,
                0.006,
                msg=f"QBER {got:.6f}",
            )
        self.assertLess(
            bb[1] - jeong_gains(0.19, dark=0.0)[0][1],
            0.003,
            msg="dark-count share",
        )

    def test_sarg_ladder(self):
        """
        Declared miss: their figure 2 ladder needs efficiency 0.17770 against 0.16636 from
        eta_Bob = 0.40 and 3 dB/km, 6.8% over, reversing to 1.1% under at eta_opt eta_det =
        0.432.
        """
        got = bisect(
            lambda e: -0.5 * (1.0 - math.exp(-e * 0.19)) * CLOCK_JEONG,
            0.05,
            0.5,
            -SIFT_BB84[-1] * 1e3,
        )
        self.assertClose(got, 0.177703, atol=1e-6, msg=f"ladder needs eta {got:.6f}")
        self.assertGreater(got, jeong_eta(), msg=f"eta {got:.6f}")
        self.assertLess(
            got / jeong_eta() - 1.0,
            0.08,
            msg=f"excess {got / jeong_eta() - 1.0:.4f}",
        )

        facs = [0.5 * jeong_gains(mu)[0][0] * CLOCK_JEONG / (rate * 1e3) for mu, rate in zip(MU_JEONG, SIFT_BB84)]

        for got in facs:
            self.assertLess(got, 1.0, msg=f"collection factor {got:.4f}")
            self.assertGreater(got, 0.85, msg=f"collection factor {got:.4f}")


class GordonB92(Question):
    """
    Tier B: Gordon, Fernandez, Townsend & Buller 2004 (IEEE J. Quantum Electron. 40, 900;
    arXiv:quant-ph/0605222), B92 at nine lengths at 100 MHz. Fitted: one polarisation leakage
    per window. NS_GOR is consumed by nothing; the axis is loss, partly emulated by attenuator.
    """

    def test_gordon_curve(self):
        """
        One visibility per window fitted at zero length, with their 180 counts/s and 2.2 dB/km,
        reproduces the QBER from 0.4% to 15.6% out to 9.96 km, worst miss 0.73 points at 5 ns
        and 0.54 at 9 ns.
        """
        cols = (Q_5, Q_9)
        misses = (0.007327, 0.005409)
        rows = tuple(zip(WIN_GOR, (SIFT_5[0], SIFT_9[0]), (Q_5[0], Q_9[0])))

        for row, col, miss in zip(rows, cols, misses):
            band = gordon_curve(*row)[:NEAR_GOR]
            worst = max(abs(g - w) for g, w in zip(band, col[:NEAR_GOR]))
            self.assertClose(worst, miss, atol=1e-5, msg=f"worst miss {worst:.6f}")

        band = gordon_curve(*rows[0])[:NEAR_GOR]

        for got, want in zip(band, Q_5[:NEAR_GOR]):
            self.assertLess(
                abs(got - want),
                0.008,
                msg=f"QBER {got:.5f}",
            )
        self.assertClose(
            band[NEAR_GOR - 1],
            0.157239,
            atol=1e-6,
            msg=f"9.96 km QBER {band[NEAR_GOR - 1]:.6f}",
        )

        vis = [gordon_setup(*row)[2] for row in rows]
        self.assertClose(vis[1], 0.994735, atol=1e-6, msg=f"9 ns V {vis[1]:.6f}")
        self.assertLess(
            abs(vis[0] / vis[1] - 1.0),
            2e-4,
            msg=f"V {vis}",
        )

    def test_gordon_takeover(self):
        """
        Background-free the model holds at 0.256% at every length; halving their 180 counts/s
        puts 9.96 km at 9.41% and doubling it at 23.85%, both excluded by their 15.6%.
        """
        flat = gordon_curve(WIN_GOR[0], SIFT_5[0], Q_5[0], dark=0.0)
        spread = max(flat) / min(flat) - 1.0
        self.assertLess(spread, 1e-3, msg=f"spread {spread:.2e}")

        for got in flat:
            self.assertClose(got, 0.002558, atol=1e-6, msg=f"dark-free leakage floor {got:.7f}")

        base = DARK_GOR * WIN_GOR[0]
        band = [gordon_curve(WIN_GOR[0], SIFT_5[0], Q_5[0], dark=base * f)[NEAR_GOR - 1] for f in (0.5, 1.0, 2.0)]
        self.assertMonotone(band, msg=f"QBER {band}")
        self.assertClose(band[0], 0.094137, atol=1e-5, msg=f"half reads {band[0]:.6f}")
        self.assertClose(band[2], 0.238542, atol=1e-5, msg=f"double reads {band[2]:.6f}")

        for got in (band[0], band[2]):
            self.assertGreater(
                abs(got - Q_5[NEAR_GOR - 1]),
                0.05,
                msg=f"QBER {got:.6f}",
            )

    def test_gordon_leak(self):
        """
        The fitted leakage is 0.256% per beam splitter, a 25.9 dB extinction, and their
        background carries 36.1% of the 0.4% zero-length QBER.
        """
        dark, trans, vis = gordon_setup(WIN_GOR[0], SIFT_5[0], Q_5[0])
        self.assertClose(vis, 0.994885, atol=1e-6, msg=f"fitted visibility {vis:.6f}")

        leak = 0.5 * (1.0 - vis)
        self.assertClose(leak, 0.0025576, atol=1e-7, msg=f"leakage {leak:.7f}")

        ratio = -10.0 * math.log10(leak)
        self.assertGreater(ratio, 20.0, msg=f"extinction {ratio:.2f} dB")
        self.assertLess(ratio, 30.0, msg=f"extinction {ratio:.2f} dB")

        share = 1.0 - leak / Q_5[0]
        self.assertClose(share, 0.36061, atol=1e-4, msg=f"background share {share:.5f}")

    def test_gordon_tail(self):
        """
        Declared miss: 11.07 and 11.85 km read 22.24% and 27.14% against their 24.3% and 31.8%,
        their counts falling 10.7% from 9.96 to 11.07 km where 2.2 dB/km asks 43.0%.
        """
        band = gordon_curve(WIN_GOR[0], SIFT_5[0], Q_5[0])[NEAR_GOR:]
        self.assertClose(band[0], 0.222387, atol=1e-6, msg=f"11.07 km {band[0]:.6f}")
        self.assertClose(band[1], 0.271350, atol=1e-6, msg=f"11.85 km {band[1]:.6f}")

        for got, want in zip(band, Q_5[NEAR_GOR:]):
            self.assertLess(got, want, msg=f"QBER {got:.6f}")

        wide = gordon_curve(WIN_GOR[1], SIFT_9[0], Q_9[0])[NEAR_GOR:]
        self.assertClose(
            Q_9[NEAR_GOR] - wide[0],
            0.068178,
            atol=1e-5,
            msg=f"9 ns miss {Q_9[NEAR_GOR] - wide[0]:.6f}",
        )

    def test_gordon_window(self):
        """
        The two 100 MHz windows agree at zero length while the 1 GHz rows split 3.7 points, more
        than the whole 100 MHz error, a temporal term b92_detect does not model.
        """
        self.assertEqual(Q_5[0], Q_9[0], msg="100 MHz QBER at 0 km")

        split = FAST_GOR[1] - FAST_GOR[0]
        self.assertClose(split, 0.037, atol=1e-9, msg=f"1 GHz split {split:.4f}")
        self.assertGreater(split, Q_5[0], msg=f"1 GHz split {split:.4f}")

        near = gordon_curve(WIN_GOR[0], SIFT_5[0], Q_5[0])[0]
        self.assertLess(near, FAST_GOR[0], msg=f"floor {near:.5f}")


class HughesB92(Question):
    """
    Tier B: Hughes, Morgan & Peterson 1999 (arXiv:quant-ph/9904038), plain B92 over 48 km of
    installed fibre. Fitted: one background at mu = 0.63. Rates only bounded: no shipped proof
    covers plain B92 with weak coherent pulses.
    """

    def test_hughes_poisson(self):
        """
        Poisson mu = 0.63 leaves 53.26% of pulses empty against their 53% and 28.21% of
        non-empty ones multiphoton against 28%; mu = 0.39 gives 18.24% against 18%.
        """
        empty = math.exp(-MU_HMP)
        self.assertClose(empty, EMPTY_HMP, atol=5e-3, msg=f"empty {empty:.5f}")

        for mu, want in ((MU_HMP, MULTI_HMP), (MU_LOW, MULTI_LOW)):
            seen = 1.0 - math.exp(-mu)
            multi = (1.0 - math.exp(-mu) * (1.0 + mu)) / seen
            self.assertClose(multi, want, atol=3e-3, msg=f"multiphoton {multi:.5f}")

    def test_hughes_fringe(self):
        """
        Their fringe scan's 1102 errors in 21 524 bits (4.871%) and 128-bit sample's 6 (4.688%)
        agree to 0.18 points, both under the 9.3% whole-set BER.
        """
        scan = FRINGE[2] / sum(FRINGE)
        sample = SAMPLE[1] / SAMPLE[0]

        self.assertClose(scan, 0.048705, atol=1e-6, msg=f"fringe BER {scan:.6f}")
        self.assertClose(sample, 0.046875, atol=1e-6, msg=f"sample BER {sample:.6f}")
        self.assertLess(
            abs(scan - sample),
            0.003,
            msg=f"BER {scan}, {sample}",
        )
        self.assertLess(scan, BER_HMP, msg=f"fringe BER {scan:.6f}")

    def test_hughes_floor(self):
        """
        The measured visibility gives (1 - V)/2 = 0.505%, 5.43% of their 9.3% BER, and across
        the +/-1.24-point bar brackets the 10% their "approximately 90% of the errors are
        attributable to detector dark counts" leaves.
        """
        got = hughes_detect(MU_HMP, 0.0)[1]
        self.assertClose(got, 0.005054, atol=1e-6, msg=f"dark-free bit error {got:.6f}")
        self.assertClose(got, 0.5 * (1.0 - VIS_HMP), atol=5e-6, msg=f"bit error {got:.6f}")

        band = [0.5 * (1.0 - min(1.0, VIS_HMP + D_VIS)) / BER_HMP]
        band += [0.5 * (1.0 - VIS_HMP + D_VIS) / BER_HMP]
        self.assertLess(band[0], 1.0 - DARK_SHARE, msg=f"lower share {band[0]:.5f}")
        self.assertGreater(band[1], 1.0 - DARK_SHARE, msg=f"upper share {band[1]:.5f}")

    def test_hughes_dark(self):
        """
        The fitted background is 1.535e-4 per gate, 1.92e-5 per 730 ps window in their
        normalisation, a 26.3 kHz noise rate inside their 10-100 kHz.
        """
        dark = hughes_dark()
        self.assertClose(dark, 1.5347e-4, atol=1e-8, msg=f"fitted background {dark:.6e}")

        norm = hughes_detect(MU_HMP, 0.0)[0] / hughes_chain(MU_HMP)
        self.assertClose(norm, 8.0, atol=0.01, msg=f"normalisation ratio {norm:.4f}")

        rate = dark / norm / WIN_HMP
        self.assertGreater(rate, NOISE_HMP[0], msg=f"noise {rate:.1f} Hz")
        self.assertLess(rate, NOISE_HMP[1], msg=f"noise {rate:.1f} Hz")

    def test_hughes_predict(self):
        """
        Declared miss: the background fitted at mu = 0.63 predicts 13.31% at mu = 0.39 against
        their 17.8%, and the reverse fit 12.86% at 0.63 against 9.3%.
        """
        got = hughes_detect(MU_LOW, hughes_dark())[1]
        self.assertClose(got, 0.133091, atol=1e-6, msg=f"BER {got:.6f} at mu=0.39")
        self.assertGreater(got, BER_HMP, msg=f"BER {got:.6f} at mu=0.39")
        self.assertLess(got, BER_LOW, msg=f"BER {got:.6f} at mu=0.39")
        self.assertClose(BER_LOW - got, 0.04491, atol=1e-4, msg=f"shortfall {BER_LOW - got:.5f}")

        back = hughes_detect(MU_HMP, hughes_dark(BER_LOW, MU_LOW))[1]
        self.assertClose(
            back,
            0.128581,
            atol=1e-6,
            msg=f"BER {back:.6f} at mu=0.63",
        )
        self.assertGreater(back, BER_HMP, msg=f"BER {back:.6f} at mu=0.63")

    def test_hughes_bounded(self):
        """
        Their 10 Hz and 3.4 Hz are 11.6% and 5.7% of the conclusive ceiling, while their own
        stated factors put the first ceiling at 8.885 Hz, under the 10 they quote.
        """
        dark = hughes_dark()
        fracs = []
        for mu, key in ((MU_HMP, KEY_HMP), (MU_LOW, KEY_LOW)):
            ceiling = _core.b92_ceiling(mu, hughes_trans(), ETA_HMP, VIS_HMP, dark)
            fracs.append(key / (ceiling * CLOCK_HMP))
            self.assertLess(fracs[-1], 1.0, msg=f"{key} Hz fraction {fracs[-1]:.5f}")
        self.assertClose(fracs[0], 0.11579, atol=1e-4, msg=f"10 Hz fraction {fracs[0]:.5f}")
        self.assertClose(fracs[1], 0.05731, atol=1e-4, msg=f"3.4 Hz fraction {fracs[1]:.5f}")

        theirs = [hughes_chain(mu) * CLOCK_HMP for mu in (MU_HMP, MU_LOW)]
        self.assertClose(theirs[0], 8.885, atol=1e-3, msg=f"chain {theirs[0]:.3f} Hz")
        self.assertLess(
            abs(theirs[0] / KEY_HMP - 1.0),
            0.15,
            msg=f"chain {theirs[0]:.3f} Hz",
        )


if __name__ == "__main__":
    rc = Exam(
        "ExpNaikEkert",
        "Tier B: Naik et al. 2000, E91 with a measured CHSH value beside its QBER",
        "exp_ekert.md",
    ).run(load(NaikEkert))
    rc |= Exam(
        "ExpLingEkert",
        "Tier B: Ling et al. 2008, an E91 key rate priced by a measured violation",
        "exp_chsh.md",
    ).run(load(LingEkert))
    rc |= Exam(
        "ExpEnzerSixState",
        "Tier B: Enzer et al. 2002, six-state QBER under a simulated eavesdropper",
        "exp_sixstate.md",
    ).run(load(EnzerSixState))
    rc |= Exam(
        "ExpJeongSarg",
        "Tier B: Jeong et al. 2014, SARG04 against BB84 on one rig over 1.27 km",
        "exp_sarg.md",
    ).run(load(JeongSarg))
    rc |= Exam(
        "ExpGordonB92",
        "Tier B: Gordon et al. 2004, the B92 QBER curve to 11.85 km at 100 MHz",
        "exp_b92_curve.md",
    ).run(load(GordonB92))
    rc |= Exam(
        "ExpHughesB92",
        "Tier B: Hughes et al. 1999, plain B92 over 48 km of installed fibre",
        "exp_b92.md",
    ).run(load(HughesB92))
    sys.exit(rc)
