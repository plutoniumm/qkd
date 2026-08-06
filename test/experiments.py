import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.cache import memo
from kit.forms import bisect
from qkd import _core

# Never import kit/anchors.py here: its ETA_COW is Gao 2022's 0.8, not Stucki's 0.0265, and its ALPHA is tabulated.

# Gobby, Yuan & Shields, Appl. Phys. Lett. 84, 3762 (2004),
# arXiv:quant-ph/0412171. BB84-WCP over 122 km of Corning SMF-28. P_ERR is
# their error count per clock cycle, per interferometer output port.
MU_GYS = 0.1
ETA_BOB = 0.045
P_ERR = 8.5e-7
CLOCK_GYS = 2e6

# FITTED, the only free optical parameter: Eq. (1) is drawn at 0.2 dB/km but the sifted-rate ladder falls at
# ~0.21, as Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005) Table 1 tabulate. SMF-28 band is [0.20, 0.22].
ALPHA_GYS = 0.21

# FITTED band, not pinned: the 3.3% plateau read as pure modulation error at
# 4.4 km (0.0328) and as carrying a 0.43% dark share at 65 km (0.0289).
E_LOW = 0.028911
E_HIGH = 0.032781

# Diamanti, Takesue, Langrock, Fejer & Yamamoto, Opt. Express 14, 13073 (2006),
# arXiv:quant-ph/0608110. DPS over 100 km, 1 GHz, 66 ps pulses, 1-bit-delay PLC
# Mach-Zehnder at 2 dB; mu = 0.2 is their stated optimum. DARK_FAR/DARK_NEAR
# are dark counts per window (350 Hz / 100 ps far, 97.5 kHz / 200 ps near). Their
# measured 3.4% QBER splits 1% interferometric, 1.7% dark, 0.7% jitter.
MU_DPS = 0.2
CLOCK_DPS = 1e9
LOSS_MZI = 2.0
DARK_FAR = 3.5e-8
DARK_NEAR = 1.95e-5
QBER_FAR = 0.034
JITTER = 0.007
RATE_FAR = 166.0

# FITTED: Fig. 5 "eta=0.4%" read as pre-window, with the 54% (100 ps) and 40% (200 ps) window costs on top.
ETA_FAR = 0.004 * 0.46
ETA_NEAR = 0.06 * 0.60

# FITTED: the 1% interferometric share is (1 - V)/2. Whole-receiver contrast, not the coupler's ">20 dB".
V_DPS = 0.98

# FITTED: top of the stated "on the order of 50-80 ns".
DEAD_DPS = 80e-9

# FITTED: f(e) is unstated; 1.16 is the standard DPS value and IndividualAttack's default.
F_DPS = 1.16

# FITTED, the only free optical parameter: mean of 0.2067, 0.2076, 0.2080 dB/km (test_dps_three_ways).
ALPHA_DPS = 0.2074

# Arrival-time response modelled instead of folded into eta: RAW_FAR and RAW_NEAR are pre-window.
WIN_FAR = 100e-12
WIN_NEAR = 200e-12
RAW_FAR = 0.004
RAW_NEAR = 0.06
PULSE_DPS = 66e-12

# FITTED, exactly determined: 54% loss at 100 ps, 40% at 200 ps, 0.7% jitter share (one constraint with the 3.4%).
JIT_FWHM = 7.933476e-11
JIT_FRAC = 0.579157
JIT_TAU = 2.951590e-10

# Stucki, Walenta, Vannel, Thew, Gisin, Zbinden, Gray, Towery & Ten, New J.
# Phys. 11, 075003 (2009), arXiv:0903.3907. COW over Corning SMF-28 ULL, bits
# carried by pulse PAIRS so BITS_COW is half the clock. DARK_SLOT is their 5 Hz
# SSPD rate per slot; LOSS_250 is their measured 250 km total, not ALPHA_ULL*L.
MU_COW = 0.5
ETA_COW = 0.0265
CLOCK_COW = 625e6
BITS_COW = 312.5e6
DARK_SLOT = 5.0 / CLOCK_COW
ALPHA_ULL = 0.164
LOSS_250 = 42.6
QBER_100 = 0.0085
QBER_250 = 0.019


def gys_eta(dist, alpha=ALPHA_GYS):
    return _core.decoy_eta(alpha, dist, ETA_BOB)


def gys_gain(dist, e_mod, alpha=ALPHA_GYS):
    """
    (gain, QBER) per clock cycle on GYS's hardware; Y0 = 2*P_e, both ports carrying P_e.
    """

    return _core.decoy_gain(MU_GYS, gys_eta(dist, alpha), 2.0 * P_ERR, e_mod)


def gys_vis(dist, alpha=ALPHA_GYS):
    """
    Visibility by their Eq. (1): bright port mu*T*eta_Bob + P_e, dim port P_e.
    """
    sig = MU_GYS * gys_eta(dist, alpha)

    return _core.cow_visibility(sig + P_ERR, P_ERR)


def gys_sift(dist, alpha=ALPHA_GYS):
    """
    Sifted bits per second: half the gain at their 2 MHz clock.
    """

    return 0.5 * CLOCK_GYS * gys_gain(dist, E_LOW, alpha)[0]


def dps_slot(dist, eta, dark, vis, alpha=ALPHA_DPS, extra=0.0):
    """
    (sifted, QBER, dim-port click probability) in closed form; a double click keeps a random
    bit.
    """
    trans = 10.0 ** (-(alpha * dist + LOSS_MZI + extra) / 10.0)
    amp = math.sqrt(trans * MU_DPS)
    hi, lo = _core.interfere(amp, 0.0, amp, 0.0, vis)
    p_hi = _core.click_prob(math.sqrt(hi), 0.0, eta, dark)
    p_lo = _core.click_prob(math.sqrt(lo), 0.0, eta, dark)
    sift = 1.0 - (1.0 - p_hi) * (1.0 - p_lo)
    err = p_lo * (1.0 - p_hi) + 0.5 * p_hi * p_lo

    return sift, err / sift, p_lo


def dps_far(alpha=ALPHA_DPS):
    return dps_slot(100.0, ETA_FAR, DARK_FAR, V_DPS, alpha=alpha)


def dps_near(extra=0.0):
    """
    The 10 km point on short-distance detector settings; Fig. 5's "baseline system error rate of
    1.5%" is V = 0.97.
    """

    return dps_slot(10.0, ETA_NEAR, DARK_NEAR, 0.97, extra=extra)


def dps_trans(dist=10.0):
    return 10.0 ** (-(ALPHA_DPS * dist + LOSS_MZI) / 10.0)


@memo
def dps_clicks(dead=0.0):
    """
    The 10 km click stream: 4e6 pulses, seed 11, ``dead`` seconds of detector memory, memoised
    on ``dead``.
    """

    return _core.run_clicks(
        4_000_000,
        11,
        MU_DPS,
        dps_trans(),
        ETA_NEAR,
        DARK_NEAR,
        0.97,
        1,
        0.0,
        CLOCK_DPS,
        dead,
        0.0,
        4096,
        False,
    )


def dps_timed(dist, eta, dark, vis, window, alpha=ALPHA_DPS, resp=None):
    """
    ``dps_slot`` with the arrival-time response ahead of the threshold: (sifted, QBER, accepted
    fraction, bin leak). Contrast scales by (1 - leak).
    """
    fwhm, frac, tau = resp or (JIT_FWHM, JIT_FRAC, JIT_TAU)
    loss, leak = _core.jitter_split(CLOCK_DPS, fwhm, window, frac, tau)
    keep = 1.0 - loss
    amp = math.sqrt(keep * 10.0 ** (-(alpha * dist + LOSS_MZI) / 10.0) * MU_DPS)
    hi, lo = _core.interfere(amp, 0.0, amp, 0.0, (1.0 - leak) * vis)
    p_hi = _core.click_prob(math.sqrt(hi), 0.0, eta, dark)
    p_lo = _core.click_prob(math.sqrt(lo), 0.0, eta, dark)
    sift = 1.0 - (1.0 - p_hi) * (1.0 - p_lo)
    err = p_lo * (1.0 - p_hi) + 0.5 * p_hi * p_lo

    return sift, err / sift, keep, leak


def timed_far(alpha=ALPHA_DPS, resp=None):
    return dps_timed(100.0, RAW_FAR, DARK_FAR, V_DPS, WIN_FAR, alpha, resp)


def timed_near(vis=V_DPS, resp=None):
    return dps_timed(10.0, RAW_NEAR, DARK_NEAR, vis, WIN_NEAR, ALPHA_DPS, resp)


def dps_shares(dist, eta, dark, vis, window, resp=None):
    """
    QBER as (dark, interferometric, jitter), linearised in flux; matches the threshold form to 6
    figures at 2e-6 photons/slot.
    """
    sift, _qber, keep, leak = dps_timed(dist, eta, dark, vis, window, resp=resp)
    trans = 10.0 ** (-(ALPHA_DPS * dist + LOSS_MZI) / 10.0)
    flux = eta * 0.5 * trans * MU_DPS * keep

    return dark / sift, flux * (1.0 - vis) / sift, flux * leak * vis / sift


def dps_secure(alpha=ALPHA_DPS):
    """
    Secure bits/s from the Waks-Takesue-Yamamoto bound on the derived sift and the measured 3.4%
    QBER, as the paper did.
    """
    sift = dps_far(alpha)[0]

    return CLOCK_DPS * _core.dps_rate(sift, QBER_FAR, MU_DPS, F_DPS)


def cow_sig(loss):
    """
    Data-line detection probability per bit: mu = 0.5 through ``loss`` dB and the SSPD.
    """

    return _core.cow_ceiling(0.0, 10.0 ** (-loss / 10.0) * ETA_COW, MU_COW)


def cow_qber(loss, base, split=1.0):
    """
    Data-line QBER; ``base`` (wrong-slot fraction) and ``split`` (data-line share after Bob's
    tap) are unpublished.
    """
    sig = cow_sig(loss) * split

    return (base * sig + DARK_SLOT) / (sig + 2.0 * DARK_SLOT)


def cow_base():
    """
    Baseline slot error fitted to the 0.85% QBER at 100 km, where dark counts contribute
    nothing.
    """

    return bisect(lambda b: -cow_qber(ALPHA_ULL * 100.0, b), 0.0, 0.05, -QBER_100)


class GysBb84(Question):
    """
    Tier B: Gobby, Yuan & Shields 2004 (Appl. Phys. Lett. 84, 3762; arXiv:quant-ph/0412171),
    BB84-WCP over 122 km. Fitted: alpha = 0.21 dB/km, e_mod in [0.0289, 0.0328]. Their key rate
    rests on a superseded 2004 proof and is not asserted.
    """

    def test_gys_tabulation(self):
        """
        Ma, Qi, Zhao and Lo's GYS parameter set is this hardware: Y0 = 1.7e-6 is 2 P_e, and
        e_detector = 3.3% is the QBER plateau at 4.4 km.
        """

        self.assertEqual(2.0 * P_ERR, 1.7e-6, msg=f"2 P_e = {2.0 * P_ERR}")
        self.assertClose(gys_eta(0.0), 0.045, atol=1e-15, msg="eta_Bob at zero length")

        near = gys_gain(4.4, E_HIGH)[1]
        self.assertClose(
            near,
            0.033,
            atol=2e-4,
            msg=f"4.4 km QBER {near:.5f}",
        )

    def test_gys_visibility(self):
        """
        Visibility at 122 km is 87.89% against their measured 88.4%, and inverting it gives
        alpha = 0.2083 dB/km inside [0.20, 0.22].
        """
        got = gys_vis(122.0)

        # atol 0.01 admits alpha in [0.2047, 0.2116].
        self.assertClose(got, 0.884, atol=0.01, msg=f"visibility {got:.5f} vs measured 0.884")

        alpha = bisect(lambda a: gys_vis(122.0, a), 0.15, 0.30, 0.884)
        self.assertClose(alpha, 0.21, atol=5e-3, msg=f"alpha {alpha:.5f}")

    def test_gys_qber(self):
        """
        The fitted e_mod band gives 8.60%-8.94% QBER at 122 km, bracketing their measured 8.9%.
        """
        lo = gys_gain(122.0, E_LOW)[1]
        hi = gys_gain(122.0, E_HIGH)[1]

        self.assertLess(lo, 0.089, msg=f"low-end QBER {lo:.5f}")
        self.assertGreater(
            hi * 1.001,
            0.089,
            msg=f"high-end QBER {hi:.5f}",
        )
        self.assertClose(hi, 0.089, atol=1e-3, msg=f"122 km QBER {hi:.5f}")

    def test_gys_three_ways(self):
        """
        Visibility, sifted-rate ratio and QBER invert to alpha = 0.20826, 0.20941 and
        0.20974-0.21211 dB/km, a 1.9% spread inside [0.20, 0.22].
        """
        alphas = [
            bisect(lambda a: gys_vis(122.0, a), 0.15, 0.30, 0.884),
            bisect(
                lambda a: -(gys_sift(101.0, a) / gys_sift(122.0, a)),
                0.15,
                0.30,
                -(23.4 / 9.2),
            ),
        ]
        alphas += [bisect(lambda a: -gys_gain(122.0, e, a)[1], 0.15, 0.30, -0.089) for e in (E_LOW, E_HIGH)]
        wide = max(alphas) / min(alphas) - 1.0

        self.assertLess(wide, 0.025, msg=f"alpha spread {wide:.4f} across {alphas}")

        for got in alphas:
            self.assertGreater(got, 0.20, msg=f"alpha {got:.5f} below the SMF-28 band")
            self.assertLess(got, 0.22, msg=f"alpha {got:.5f} above the SMF-28 band")

    def test_gys_dark_floor(self):
        """
        Their "less than 0.4%" dark and stray share at 65 km is 0.374% at their alpha = 0.2; the
        fitted 0.21 gives 0.434%, under 0.5%.
        """
        spec = gys_gain(65.0, 0.0, 0.20)[1]
        fitted = gys_gain(65.0, 0.0)[1]

        self.assertLess(spec, 0.004, msg=f"dark+stray {spec:.5f} at alpha 0.20")
        self.assertLess(fitted, 0.005, msg=f"dark+stray {fitted:.5f} at alpha 0.21")
        self.assertGreater(fitted, spec, msg=f"dark+stray {fitted:.5f} against {spec:.5f}")

    def test_gys_sift_ladder(self):
        """
        Their 23.4 and 9.2 bit/s sifted rates give a ratio of 2.5435 against the model's 2.5473,
        and collection factors 0.6544 and 0.6554, both below unity.
        """
        ratio = gys_sift(101.0) / gys_sift(122.0)
        self.assertClose(ratio, 23.4 / 9.2, atol=0.02, msg=f"sifted ratio {ratio:.5f} vs 2.54348")

        far = [23.4 / gys_sift(101.0), 9.2 / gys_sift(122.0)]

        for got in far:
            self.assertGreater(got, 0.0, msg=f"collection factor {got:.5f}")
            self.assertLess(got, 1.0, msg=f"collection factor {got:.5f}")
        self.assertClose(far[0], far[1], atol=0.005, msg=f"collection factors {far}")

    def test_gys_short_deficit(self):
        """
        Declared miss: the collection factor is 0.936 at 4.4 km against 0.655 at 101 and 122 km,
        from polarisation drift the paper names and qkd does not model.
        """
        near = 3400.0 / gys_sift(4.4)
        far = 9.2 / gys_sift(122.0)

        self.assertLess(near, 1.0, msg=f"4.4 km factor {near:.5f}")
        self.assertGreater(
            near / far,
            1.3,
            msg=f"deficit {near / far:.4f}x",
        )

    def test_gys_pns_reading(self):
        """
        Their ~50 km photon-splitting limit under Brassard-Lutkenhaus-Mor-Sanders: untrusted
        fails everywhere, trusted gives 63.3 km, trusted with sifting 49.0 km at alpha = 0.21
        and 51.4 km at 0.20.
        """
        multi = 1.0 - math.exp(-MU_GYS) * (1.0 + MU_GYS)
        back = gys_gain(0.0, E_LOW)[0]

        self.assertClose(multi, 4.6788e-3, atol=1e-6, msg=f"multiphoton {multi:.5e}")
        self.assertLess(
            back,
            multi,
            msg=f"untrusted gain {back:.5e} below multiphoton {multi:.5e}",
        )

        for alpha, want in ((0.21, 49.0), (0.20, 51.4)):
            reach = 10.0 / alpha * math.log10(0.5 * MU_GYS / multi)
            self.assertClose(reach, want, atol=0.1, msg=f"sifted reach {reach:.2f} km at {alpha}")

        for alpha, want in ((0.21, 63.3), (0.20, 66.5)):
            reach = 10.0 / alpha * math.log10(MU_GYS / multi)
            self.assertClose(reach, want, atol=0.1, msg=f"unsifted reach {reach:.2f} km at {alpha}")


class DiamantiDps(Question):
    """
    Tier B: Diamanti, Takesue, Langrock, Fejer & Yamamoto 2006 (Opt. Express 14, 13073;
    arXiv:quant-ph/0608110), DPS over 100 km. Fitted: alpha = 0.2074, pre-window efficiency,
    f(e) = 1.16, 80 ns dead time. No jitter: QBER is compared to 3.4% - 0.7%, computed here.
    """

    def test_dps_qber(self):
        """
        The 100 km QBER is 2.691% against the 2.7% jitter-free residual, splitting into 1.726%
        dark against their 1.7% and 0.965% interferometric against their 1%.
        """
        sift, got, dim = dps_far()
        dark = DARK_FAR / sift
        intf = (dim - DARK_FAR) / sift

        self.assertClose(
            got,
            QBER_FAR - JITTER,
            atol=1e-3,
            msg=f"QBER {got:.5f}",
        )
        self.assertClose(dark, 0.017, atol=1e-3, msg=f"dark share {dark:.5f}")
        self.assertClose(intf, 0.010, atol=1e-3, msg=f"interferometric share {intf:.5f}")
        self.assertClose(
            dark + intf,
            QBER_FAR - JITTER,
            atol=1e-3,
            msg=f"shares sum {dark + intf:.5f}",
        )

    def test_dps_secure(self):
        """
        The Waks-Takesue-Yamamoto rate at 100 km is 168.2 bit/s against their 166, 1.3% high.
        """
        got = dps_secure()

        # atol 3.5 bit/s pins alpha to 0.0019 dB/km; the SMF-28 band spans 127-198 bit/s.
        self.assertClose(
            got,
            RATE_FAR,
            atol=3.5,
            msg=f"secure rate {got:.2f} bit/s",
        )
        self.assertGreater(got, 0.0, msg=f"secure rate {got:.2f} bit/s")

    def test_dps_three_ways(self):
        """
        QBER, dark share and rate invert to alpha = 0.20763, 0.20673 and 0.20799 dB/km, a 0.61%
        spread inside [0.20, 0.22]; only the rate leg is independent of the error decomposition.
        """
        alphas = [
            bisect(lambda a: -dps_far(a)[1], 0.15, 0.30, -(QBER_FAR - JITTER)),
            bisect(lambda a: -(DARK_FAR / dps_far(a)[0]), 0.15, 0.30, -0.017),
            bisect(dps_secure, 0.15, 0.30, RATE_FAR),
        ]
        wide = max(alphas) / min(alphas) - 1.0

        self.assertLess(wide, 0.01, msg=f"alpha spread {wide:.5f} across {alphas}")

        for got in alphas:
            self.assertGreater(got, 0.20, msg=f"alpha {got:.5f} below the SMF-28 band")
            self.assertLess(got, 0.22, msg=f"alpha {got:.5f} above the SMF-28 band")
        self.assertClose(
            sum(alphas) / 3.0,
            ALPHA_DPS,
            atol=1e-3,
            msg=f"mean of {alphas}",
        )

    def test_dps_window_reading(self):
        """
        Reading "eta=0.4%" as post-window is excluded: it needs alpha = 0.2404-0.2417 dB/km, and
        gives 1.79% QBER against 2.7% and 359 bit/s against 166.
        """
        wrong = dps_slot(100.0, 0.004, DARK_FAR, V_DPS)
        alphas = [
            bisect(
                lambda a: -dps_slot(100.0, 0.004, DARK_FAR, V_DPS, alpha=a)[1],
                0.15,
                0.60,
                -(QBER_FAR - JITTER),
            ),
            bisect(
                lambda a: -(DARK_FAR / dps_slot(100.0, 0.004, DARK_FAR, V_DPS, alpha=a)[0]),
                0.15,
                0.60,
                -0.017,
            ),
        ]

        for got in alphas:
            self.assertGreater(got, 0.23, msg=f"post-window alpha {got:.5f}")
        self.assertLess(
            wrong[1] / (QBER_FAR - JITTER),
            0.75,
            msg=f"post-window QBER {wrong[1]:.5f}",
        )
        self.assertGreater(
            CLOCK_DPS * _core.dps_rate(wrong[0], QBER_FAR, MU_DPS, F_DPS),
            2.0 * RATE_FAR,
            msg="post-window secure rate",
        )

    def test_dps_montecarlo(self):
        """
        At 10 km, 4e6 pulses of ``run_clicks`` reproduce the closed form to 3% on sift and 0.4
        points on QBER, about three standard errors.
        """
        sift, qber, _dim = dps_near()
        out = dps_clicks()

        self.assertClose(
            out.sift_rate / sift,
            1.0,
            atol=0.03,
            msg=f"sampled sift {out.sift_rate:.6e} vs closed form {sift:.6e}",
        )
        self.assertClose(out.qber, qber, atol=4e-3, msg=f"sampled QBER {out.qber:.5f} vs {qber:.5f}")

    def test_dps_deadtime(self):
        """
        Their Eq. (5) dead-time factor exp(-nu*mu*T*t_d/2) agrees with the sequential scan in
        ``run_clicks`` to 4% at 10 km and 80 ns.
        """
        flux = CLOCK_DPS * MU_DPS * dps_trans() * ETA_NEAR
        want = math.exp(-flux * DEAD_DPS / 2.0)
        free = dps_clicks()
        held = dps_clicks(DEAD_DPS)
        got = held.sift_rate / free.sift_rate

        self.assertClose(got, want, atol=0.04, msg=f"dead-time factor {got:.5f} vs their {want:.5f}")
        self.assertLess(got, 1.0, msg=f"dead-time factor {got:.5f}")

    def test_dps_near_deficit(self):
        """
        Declared miss at 10 km: QBER 2.16% against 2.2% but sift 2.55 Mbit/s against 2, needing
        ~1.1 dB of unpublished loss that moves the QBER to 2.35%.
        """
        sift, qber, _dim = dps_near()
        trans = 10.0 ** (-(ALPHA_DPS * 10.0 + LOSS_MZI) / 10.0)
        flux = CLOCK_DPS * MU_DPS * trans * ETA_NEAR
        rate = CLOCK_DPS * sift * math.exp(-flux * DEAD_DPS / 2.0)

        self.assertClose(qber, 0.022, atol=1.5e-3, msg=f"10 km QBER {qber:.5f}")
        self.assertGreater(
            rate / 2e6,
            1.2,
            msg=f"10 km rate {rate / 1e6:.4f} Mbit/s",
        )

        patched = dps_near(extra=1.1)

        # atol 2e-3 holds at 1.1 dB (1.496e-3 miss) and at the exact 1.1905 dB (1.671e-3).
        self.assertClose(
            patched[1],
            0.022,
            atol=2e-3,
            msg=f"QBER {patched[1]:.5f} at +1.1 dB",
        )


class DiamantiJitter(Question):
    """
    Tier B: Diamanti et al. 2006 with timing jitter modelled, reproducing 3.4% outright. Fitted:
    alpha and V carried over, plus three response parameters against three timing statements.
    """

    def test_window_costs(self):
        """
        One fitted response gives their 54% loss at 100 ps and 40% at 200 ps, 1.30x more light
        from doubling the window where a single Gaussian gives 1.44x.
        """
        far = _core.jitter_split(CLOCK_DPS, JIT_FWHM, WIN_FAR, JIT_FRAC, JIT_TAU)
        near = _core.jitter_split(CLOCK_DPS, JIT_FWHM, WIN_NEAR, JIT_FRAC, JIT_TAU)

        self.assertClose(far[0], 0.54, atol=1e-4, msg=f"100 ps loss {far[0]:.5f}")
        self.assertClose(near[0], 0.40, atol=1e-4, msg=f"200 ps loss {near[0]:.5f}")
        self.assertClose(
            (1.0 - near[0]) / (1.0 - far[0]),
            0.60 / 0.46,
            atol=1e-3,
            msg=f"window gain {(1.0 - near[0]) / (1.0 - far[0]):.4f}",
        )
        self.assertGreater(far[1], 0.01, msg=f"bin leak {far[1]:.5f}")

    def test_qber_total(self):
        """
        The response fit reproduces 3.4% and the 0.7% jitter share by construction, and predicts
        the rest splits 1.726% dark against 1.7% and 0.966% interferometric against 1%.
        """
        sift, qber, _keep, _leak = timed_far()
        dark, intf, jit = dps_shares(100.0, RAW_FAR, DARK_FAR, V_DPS, WIN_FAR)

        self.assertClose(qber, QBER_FAR, atol=1e-4, msg=f"QBER {qber:.5f}")
        self.assertClose(dark, 0.017, atol=1e-3, msg=f"dark share {dark:.5f}")
        self.assertClose(intf, 0.010, atol=1e-3, msg=f"optics share {intf:.5f}")
        self.assertClose(jit, JITTER, atol=1e-3, msg=f"jitter share {jit:.5f}")
        self.assertClose(
            dark + intf + jit,
            qber,
            atol=1e-6,
            msg=f"shares sum {dark + intf + jit:.6f}",
        )
        self.assertGreater(qber, dps_far()[1], msg=f"QBER {qber:.5f}")

    def test_gaussian_excluded(self):
        """
        A pure Gaussian fitted to the 100 ps loss needs 192.1 ps FWHM, predicts 22.0% loss at
        200 ps against 40%, leaks zero and gives 2.691% QBER instead of 3.4%.
        """
        wide = bisect(
            lambda f: -_core.jitter_split(CLOCK_DPS, f, WIN_FAR, 0.0, 0.0)[0],
            1e-12,
            1e-9,
            -0.54,
        )
        resp = (wide, 0.0, 0.0)
        near = _core.jitter_split(CLOCK_DPS, wide, WIN_NEAR, 0.0, 0.0)
        _sift, qber, _keep, leak = timed_far(resp=resp)

        self.assertClose(wide, 192.13e-12, atol=1e-13, msg=f"Gaussian FWHM {wide:.4e} s")
        self.assertEqual(leak, 0.0, msg=f"leak {leak} past the 9-sigma core cut")
        self.assertLess(near[0], 0.25, msg=f"200 ps loss {near[0]:.5f}")
        self.assertGreater(0.40 - near[0], 0.15, msg=f"200 ps loss {near[0]:.5f}")
        self.assertClose(
            qber,
            QBER_FAR - JITTER,
            atol=1e-3,
            msg=f"Gaussian QBER {qber:.5f}",
        )

    def test_response_plausible(self):
        """
        The 79.33 ps core leaves 44.0 ps prompt jitter beside the 66 ps pulse, the 295 ps tail
        sits in the 0.2-3 ns diffusion band, and a 0.1-point QBER shift moves the core 9.1 ps.
        """
        prompt = math.sqrt(JIT_FWHM * JIT_FWHM - PULSE_DPS * PULSE_DPS)

        self.assertGreater(JIT_FWHM, PULSE_DPS, msg=f"core {JIT_FWHM:.4e} s")
        self.assertClose(prompt, 44.0e-12, atol=2e-12, msg=f"detector prompt jitter {prompt:.4e} s")
        self.assertGreater(JIT_TAU, 200e-12, msg=f"tail {JIT_TAU:.3e} s")
        self.assertLess(JIT_TAU, 3e-9, msg=f"tail {JIT_TAU:.3e} s")
        self.assertLess(JIT_FRAC, 0.7, msg=f"tail fraction {JIT_FRAC}")

        moved = bisect(lambda f: -timed_far(resp=(f, JIT_FRAC, JIT_TAU))[1], 1e-12, 1e-9, -0.035)

        self.assertClose(
            moved - JIT_FWHM,
            9.1e-12,
            atol=1e-12,
            msg=f"core shift {moved - JIT_FWHM:.3e} s",
        )

    def test_rate_unmoved(self):
        """
        Deriving the 54% window cost leaves the 100 km sift at 2.02816e-6 and the 166 bit/s
        anchor at 168.19.
        """
        got = timed_far()[0]
        was = dps_far()[0]

        self.assertClose(got / was, 1.0, atol=1e-6, msg=f"sift moved: {got:.6e} vs {was:.6e}")
        self.assertClose(
            CLOCK_DPS * _core.dps_rate(got, QBER_FAR, MU_DPS, F_DPS),
            RATE_FAR,
            atol=8.0,
            msg="100 km secure rate",
        )

    def test_near_gap(self):
        """
        Declared miss: the response fitted at 100 km predicts 2.795% QBER at 10 km against their
        2.2%, while the jitter-free reading sits below 2.2%.
        """
        qber = timed_near()[1]
        gap = qber - 0.022

        self.assertGreater(qber, 0.022, msg=f"10 km QBER {qber:.5f}")
        self.assertClose(qber, 0.027952, atol=5e-4, msg=f"10 km QBER {qber:.5f}")
        self.assertLess(gap, 0.008, msg=f"gap {gap:.5f}")
        self.assertLess(
            dps_near()[1],
            0.022,
            msg="jitter-free 10 km QBER",
        )

    def test_timed_montecarlo(self):
        """
        At 10 km, 4e6 pulses of ``run_clicks`` with the response on reproduce the sift to 0.2%
        and QBER to 0.11 points, at 40.000% window loss and 2.33% bin leak.
        """
        trans = 10.0 ** (-(ALPHA_DPS * 10.0 + LOSS_MZI) / 10.0)
        sift, qber, keep, leak = timed_near()
        out = _core.run_clicks(
            4_000_000,
            11,
            MU_DPS,
            trans,
            RAW_NEAR,
            DARK_NEAR,
            V_DPS,
            1,
            0.0,
            CLOCK_DPS,
            0.0,
            0.0,
            4096,
            False,
            JIT_FWHM,
            WIN_NEAR,
            JIT_FRAC,
            JIT_TAU,
        )

        self.assertClose(
            out.window_loss,
            1.0 - keep,
            atol=1e-12,
            msg=f"sampled loss {out.window_loss:.6f} vs {1.0 - keep:.6f}",
        )
        self.assertClose(out.bin_leak, leak, atol=1e-12, msg=f"sampled leak {out.bin_leak:.6f}")
        self.assertClose(
            out.sift_rate / sift,
            1.0,
            atol=0.01,
            msg=f"sampled sift {out.sift_rate:.6e} vs closed form {sift:.6e}",
        )
        self.assertClose(out.qber, qber, atol=3e-3, msg=f"sampled QBER {out.qber:.5f} vs {qber:.5f}")


class StuckiCow(Question):
    """
    Tier B: Stucki et al. 2009 (New J. Phys. 11, 075003; arXiv:0903.3907), COW to 250 km of ULL
    fibre. Fitted: one baseline error at 100 km. Key rates only bounded: Gonzalez-Payo, Trenyi,
    Wang & Curty, PRL 125, 260510 (2020) broke their analysis.
    """

    def test_cow_qber_curve(self):
        """
        A baseline slot error fitted to 0.85% at 100 km predicts 1.904% at 250 km against their
        1.9%; their 250 km cryostat ran unusually cold, so the agreement is not a clean
        prediction.
        """
        base = cow_base()
        self.assertLess(base, 0.01, msg=f"baseline {base:.6f}")

        got = cow_qber(LOSS_250, base)
        self.assertClose(
            got,
            QBER_250,
            atol=5e-4,
            msg=f"250 km QBER {got:.5f}",
        )

    def test_cow_dark_takeover(self):
        """
        Dark counts are 0.0026% of QBER at 100 km and 1.08% at 250 km, so the published 5 Hz
        dark rate carries the whole rise.
        """
        near = cow_qber(ALPHA_ULL * 100.0, 0.0)
        far = cow_qber(LOSS_250, 0.0)

        self.assertLess(near, 1e-4, msg=f"100 km dark share {near:.7f}")
        self.assertGreater(far, 0.01, msg=f"250 km dark share {far:.5f}")
        self.assertGreater(
            QBER_250 - QBER_100,
            0.5 * far,
            msg=f"250 km dark share {far:.5f}",
        )

    def test_cow_split_sensitivity(self):
        """
        Diverting an unpublished 5% or 10% to Bob's monitoring tap moves the 250 km QBER to
        1.96% or 2.02%, within 7% of their 1.9%.
        """
        base = cow_base()
        band = [cow_qber(LOSS_250, base, split=s) for s in (1.0, 0.95, 0.9)]

        self.assertMonotone(band, msg=f"QBER {band}")

        for got in band:
            self.assertLess(
                abs(got / QBER_250 - 1.0),
                0.07,
                msg=f"250 km QBER {got:.5f}",
            )

    def test_cow_rate_bounded(self):
        """
        Their 6 kbit/s and 15 bit/s are 6.3% and 6.6% of the per-click ceiling (94.8 kbit/s,
        227.5 bit/s), asserted only as inequalities.
        """
        caps = [cow_sig(ALPHA_ULL * 100.0) * BITS_COW, cow_sig(LOSS_250) * BITS_COW]
        fracs = [6000.0 / caps[0], 15.0 / caps[1]]

        for i, got in enumerate(fracs):
            self.assertLess(got, 1.0, msg=f"rate {i} ceiling fraction {got:.4f}")
            self.assertGreater(got, 0.03, msg=f"rate {i} ceiling fraction {got:.4f}")

        # the ratio misses 1 by 0.040.
        self.assertClose(
            fracs[0] / fracs[1],
            1.0,
            atol=0.1,
            msg=f"ceiling fractions {fracs}",
        )


if __name__ == "__main__":
    rc = Exam(
        "ExpGysBb84",
        "Tier B: Gobby-Yuan-Shields 2004, BB84-WCP measured over 122 km",
        "exp_gys.md",
    ).run(load(GysBb84))
    rc |= Exam(
        "ExpDiamantiDps",
        "Tier B: Diamanti et al. 2006, DPS measured over 100 km",
        "exp_dps.md",
    ).run(load(DiamantiDps))
    rc |= Exam(
        "ExpDiamantiJitter",
        "Tier B: Diamanti et al. 2006, the 3.4% QBER with timing jitter modelled",
        "exp_jitter.md",
    ).run(load(DiamantiJitter))
    rc |= Exam(
        "ExpStuckiCow",
        "Tier B: Stucki et al. 2009, COW QBER curve to 250 km of ULL fibre",
        "exp_cow.md",
    ).run(load(StuckiCow))
    sys.exit(rc)
