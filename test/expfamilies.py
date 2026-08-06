import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.cache import memo
from kit.forms import bisect, fitslope
import qkd
from qkd import _core, budget, fock


@memo
def gauss_link(v_a, trans, xi, detector, beta):
    """
    Unclamped asymptotic rate in bit/symbol of one Gaussian-modulation q.Link: `key_raw`, as the
    untrusted Hajomer readings are negative.
    """
    link = qkd.Link(
        modulation=qkd.GaussianModulation(v_a=v_a),
        channel=qkd.Channel(T=trans, xi=xi, ref="input"),
        bob=qkd.Bob(detector=detector),
        security=qkd.Asymptotic(beta=beta),
    )

    return link.run().explain["key_raw"]["value"]


# Lvovsky, Hansen, Aichele, Benson, Mlynek & Schiller, "Quantum state
# reconstruction of the single-photon Fock state", Phys. Rev. Lett. 87, 050402
# (2001), arXiv:quant-ph/0101051. Phase-randomised pulsed homodyne tomography of a heralded |1>.
#
#   ETA_FIT   "The best fit efficiency value was eta = 0.55"
#   RHO_ONE   "We found rho_11 = 0.553 +/- 0.013"; D_RHO is that error bar
#   W_ABEL    their W(0,0) = -0.062, Abel transform of the measured marginal,
#             no state model
#   W_PATTERN "we find W(0,0) = -0.067 +/- 0.016", RHO_ONE through their
#             Eq. (5); D_W is that error bar
#   ETA_TOTAL "a total measurement efficiency of 55.3 +/- 1.3%"
#   VAC_ZERO  the simultaneous vacuum control's rho_00 = 0.9975 +/- 0.0029
#   VAC_ONE   the same control's rho_11 = 0.0021 +/- 0.0032
ETA_FIT = 0.55
RHO_ONE = 0.553
D_RHO = 0.013
W_ABEL = -0.062
W_PATTERN = -0.067
D_W = 0.016
ETA_TOTAL = 0.553
VAC_ZERO = 0.9975
VAC_ONE = 0.0021

# Efficiency decomposition:
#   V_MATCH their v = 83 +/- 1% visibility, entering the efficiency as v^2,
#           which they print as 0.69 +/- 0.02
#   F_MODE  "the DFG wave does not perfectly mimic the conditionally prepared
#           mode of the single photon causes an additional reduction by x0.95"
#   F_PATH  "A factor of 0.90 arises from losses in the signal beam path and
#           non-perfect quantum efficiency of the photodiodes"
#   F_TRIG  "a 2% reduction occurs due to false trigger count events"
#   ETA_CAP "the upper limit estimate of the quantum efficiency as (57 +/- 2)%"
V_MATCH = 0.83
F_MODE = 0.95
F_PATH = 0.90
F_TRIG = 0.98
ETA_CAP = 0.57

# Fig. 1 caption prints W(X,P) = (2/pi)*(4*(X^2+P^2) - 1)*exp(-2*(X^2+P^2)), vacuum peak 2/pi, but defines
# X = (a + a-dagger)/sqrt(2), peak 1/pi. Their published W(0,0) values follow the printed function.
SCALE = 2.0


def lvovsky(eta):
    """
    Their Eq. (5) state eta|1><1| + (1 - eta)|0><0|, built as pure loss on |1>
    (test_fock_model).
    """

    return fock.Number(1).loss(eta)


def origin(eta):
    """
    W(0, 0) of that state in their normalisation, on a grid with a node at the origin.
    """
    axis = [-0.5, 0.0, 0.5]

    return SCALE * float(lvovsky(eta).wigner(axis, axis)[1][1])


class LvovskyFock(Question):
    """
    Tier B: Lvovsky et al. 2001 (PRL 87, 050402; arXiv:quant-ph/0101051), homodyne tomography of
    a single-photon Fock state. Nothing fitted: eta is published twice and the state is their
    Eq. (5).
    """

    def test_fock_model(self):
        """
        qkd's pure-loss channel on $\\lvert 1\\rangle$ reproduces their Eq. (5) populations at
        their $\\rho_{11}$, and at $\\eta = 0$ their vacuum control.
        """
        pops = lvovsky(RHO_ONE).populations()
        self.assertClose(float(pops[1]), RHO_ONE, atol=1e-15, msg=f"rho_11 {pops[1]}")
        self.assertClose(
            float(pops[0]),
            1.0 - RHO_ONE,
            atol=1e-15,
            msg=f"rho_00 {pops[0]}",
        )
        self.assertClose(float(sum(pops[2:])), 0.0, atol=1e-15, msg=f"n >= 2 population {sum(pops[2:])}")

        vac = fock.Number(0).populations()
        self.assertClose(float(vac[0]), 1.0, atol=1e-15, msg=f"vacuum rho_00 {vac[0]}")
        self.assertClose(float(vac[1]), 0.0, atol=1e-15, msg=f"vacuum rho_11 {vac[1]}")
        self.assertClose(
            origin(0.0),
            SCALE * (float(vac[0]) - float(vac[1])) / math.pi,
            atol=1e-15,
            msg="eta = 0 W(0,0)",
        )
        self.assertLess(
            VAC_ONE / RHO_ONE,
            0.0078,
            msg=f"vacuum leakage {VAC_ONE / RHO_ONE:.5f}",
        )
        self.assertLess(1.0 - VAC_ZERO, 0.003, msg=f"vacuum rho_00 deficit {1.0 - VAC_ZERO:.5f}")

    def test_fock_origin(self):
        """
        Their quoted $W(0,0) = -0.067 \\pm 0.016$ recomputes from $\\rho_{11}$ through Eq. (5)
        at -0.0675, while the 1/pi reading of their Fig. 1 operator definition misses by 2.09
        error bars.
        """
        got = origin(RHO_ONE)
        self.assertClose(got, W_PATTERN, atol=D_W, msg=f"W(0,0) {got:.6f}")
        self.assertClose(
            got,
            -0.0674817,
            atol=1e-6,
            msg=f"W(0,0) {got:.7f}",
        )
        self.assertLess(got, 0.0, msg=f"W(0,0) {got:.7f}")

        halved = got / SCALE
        self.assertClose(halved, -0.0337408, atol=1e-6, msg=f"1/pi W(0,0) {halved:.7f}")

        gap = abs(halved - W_PATTERN) / D_W
        self.assertGreater(gap, 2.0, msg=f"1/pi gap {gap:.3f} error bars")
        self.assertClose(
            origin(0.0),
            SCALE / math.pi,
            atol=1e-15,
            msg="eta = 0 W(0,0)",
        )

    def test_fock_three_ways(self):
        """
        Their marginal fit, pattern-function sampling and Abel-reconstructed $W(0,0) = -0.062$
        inverted through qkd give three $\\eta$ spread by 0.78%, inside their $\\pm 0.013$ on
        $\\rho_{11}$.
        """
        back = bisect(origin, 0.30, 0.70, W_ABEL)
        self.assertClose(back, 0.548695, atol=1e-5, msg=f"Abel eta {back:.6f}")

        etas = [ETA_FIT, RHO_ONE, back]
        wide = max(etas) / min(etas) - 1.0
        self.assertLess(wide, 0.01, msg=f"efficiency spread {wide:.5f} across {etas}")

        for got in etas:
            self.assertLess(
                abs(got - RHO_ONE),
                D_RHO,
                msg=f"efficiency {got:.6f}",
            )

    def test_fock_threshold(self):
        """
        W(0,0) crosses zero at exactly $\\eta = 0.5$, their Fig. 3 caption's $\\eta \\gt 0.5$
        for negativity.
        """
        self.assertEqual(origin(0.5), 0.0, msg=f"W(0,0) at 0.5 {origin(0.5)}")
        self.assertGreater(origin(0.49), 0.0, msg="W(0,0) at 0.49")
        self.assertLess(origin(0.51), 0.0, msg="W(0,0) at 0.51")

        # Not published: 0.0098 against 0.42616 for the pure |1>.
        volume = float(lvovsky(RHO_ONE).negativity())
        self.assertClose(volume, 0.0098221, atol=1e-6, msg=f"negativity volume {volume:.7f}")
        self.assertGreater(volume, 0.0, msg=f"negativity volume {volume:.7f}")

    def test_fock_budget(self):
        """
        Their four measured efficiency factors multiply to 0.5772, matching the $(57 \\pm 2)$%
        upper limit they print and above their measured 55.3%.
        """
        built = V_MATCH * V_MATCH * F_MODE * F_PATH * F_TRIG
        self.assertClose(V_MATCH * V_MATCH, 0.69, atol=5e-3, msg=f"v^2 {V_MATCH**2:.4f}")
        self.assertClose(built, 0.577229, atol=1e-6, msg=f"budget {built:.6f}")
        self.assertClose(built, ETA_CAP, atol=0.02, msg=f"budget {built:.5f}")
        self.assertGreater(
            built,
            ETA_TOTAL,
            msg=f"budget {built:.5f}",
        )


# Zhang, Chen, Pirandola, Wang, Zhou, Chu, Zhao, Xu, Yu & Guo, "Long-Distance
# Continuous-Variable Quantum Key Distribution over 202.81 km of Fiber",
# Phys. Rev. Lett. 125, 010502 (2020),
# arXiv:2001.02555. Transmitted LO, homodyne. Table I from the arXiv LaTeX source; xi and xi' are
# "excess noise at the channel input" and "worst-case excess noise estimator at the channel input".
Z_LOSS = (4.36, 8.29, 11.68, 15.89, 23.46, 32.45)
Z_KM = (27.27, 49.30, 69.53, 99.31, 140.52, 202.81)
Z_SNR = (2.8035, 1.0715, 0.4619, 0.1806, 0.0308, 0.0023)
Z_BETA = (0.95, 0.95, 0.96, 0.96, 0.96, 0.98)
Z_FER = (0.50, 0.50, 0.10, 0.10, 0.10, 0.90)
Z_VA = (14.37, 14.14, 14.12, 14.53, 14.23, 7.65)
Z_XI = (0.0015, 0.0033, 0.0049, 0.0063, 0.0086, 0.0081)
Z_WORST = (0.0016, 0.0037, 0.0058, 0.0085, 0.0219, 0.0383)
Z_VEL = (0.1216, 0.1881, 0.2411, 0.1893, 0.2717, 0.1523)
Z_KEY = (2.78e5, 0.62e5, 4.28e4, 1.18e4, 318.85, 6.214)

#   Z_ETA    the eta row of Table I, 61.34% at every point
#   Z_CLOCK  their 5 MHz repetition rate
#   Z_OVER   the alpha row, 100 reference pulses every 1000 data pulses
#   Z_ALPHA  "The average fiber attenuation (without splices) is 0.16 dB/km"
#   Z_KAPPA  "we can keep (1 - k) to low values, which is about 7.6e-5", their
#            supplement, kappa = [E(cos theta)]^2, excess noise "is equal to (1 - kappa)*V_A"
# Their key rate: f*(1 - alpha)*(1 - FER)*[beta*I(A:B) - chi(B:E) - Delta(n)], parameters bounded at
# 6.5 standard deviations. "Bob's detector is assumed to be inaccessible to Eve".
Z_ETA = 0.6134
Z_CLOCK = 5e6
Z_OVER = 0.10
Z_ALPHA = 0.16
Z_KAPPA = 7.6e-5


def zhang(i, xi=None):
    """
    Asymptotic trusted-homodyne rate in bit/symbol at their point i, on that point's V_A, T, xi,
    v_el and beta.
    """

    return gauss_link(
        Z_VA[i],
        10.0 ** (-Z_LOSS[i] / 10.0),
        Z_XI[i] if xi is None else xi,
        qkd.Homodyne(eta=Z_ETA, v_el=Z_VEL[i], trusted=True),
        Z_BETA[i],
    )


def perbit(i, xi=None):
    """
    That rate in bit/s through their prefactor f*(1 - alpha)*(1 - FER).
    """

    return Z_CLOCK * (1.0 - Z_OVER) * (1.0 - Z_FER[i]) * zhang(i, xi)


def snr(i):
    """
    Homodyne SNR eta*T*V_A over shot, electronic and excess terms in SNU, from the other
    columns.
    """
    net = Z_ETA * 10.0 ** (-Z_LOSS[i] / 10.0)

    return net * Z_VA[i] / (1.0 + Z_VEL[i] + net * Z_XI[i])


def missing(i):
    """
    Loss in dB closing their measured SNR against the same columns.
    """
    want = Z_SNR[i] * (1.0 + Z_VEL[i]) / Z_VA[i]

    return 10.0 * math.log10(Z_ETA * 10.0 ** (-Z_LOSS[i] / 10.0) / want)


def slope(ys):
    """
    Log-log slope of a six-point curve against the end-to-end transmittance.
    """
    xs = [-loss / 10.0 for loss in Z_LOSS]

    return fitslope(xs, [math.log10(y) for y in ys])


class ZhangLongHaul(Question):
    """
    Tier B: Zhang et al. 2020 (PRL 125, 010502; arXiv:2001.02555), CV-QKD at six points to
    202.81 km. Pinned: Table I, 5 MHz clock, 10% reference overhead, (1 - kappa) = 7.6e-5.
    Nothing fitted; finite-size rates bounded, Delta(n) not modelled.
    """

    def test_zhang_curve(self):
        """
        Through their clock, overhead and FER the asymptotic rate upper-bounds K_finite at all
        six points, 1.0056 to 9.39 times, and 3.97 times at 202.81 km on the worst-case $\\xi'$,
        the Tier A anchor's 5.5e-5 bit/symbol.
        """
        loose = [perbit(i) / Z_KEY[i] for i in range(6)]
        tight = [perbit(i, Z_WORST[i]) / Z_KEY[i] for i in range(6)]

        for i, got in enumerate(loose):
            self.assertGreater(
                got,
                1.0,
                msg=f"{Z_KM[i]} km ratio {got:.4f}",
            )
            self.assertGreater(tight[i], 1.0, msg=f"point {i} worst-case ratio {tight[i]:.4f}")
            self.assertLess(tight[i], got + 1e-9, msg=f"point {i} worst-case ratio {tight[i]:.4f}")
        self.assertMonotone(loose, msg=f"ratios {loose}")
        self.assertClose(loose[0], 1.0056, atol=5e-4, msg=f"27.27 km ratio {loose[0]:.4f}")
        self.assertClose(loose[5], 9.3897, atol=5e-3, msg=f"202.81 km ratio {loose[5]:.4f}")
        self.assertClose(tight[5], 3.9678, atol=5e-3, msg=f"202.81 km worst-case {tight[5]:.4f}")
        self.assertClose(
            zhang(5, Z_WORST[5]),
            5.479e-5,
            atol=1e-7,
            msg=f"202.81 km worst-case rate {zhang(5, Z_WORST[5]):.6e} bit/symbol",
        )

    def test_zhang_scaling(self):
        """
        The log-log slope against end-to-end transmittance is 1.658 measured, the steepest,
        against 1.266 asymptotic and 1.419 worst-case.
        """
        published = slope(Z_KEY)
        loose = slope([perbit(i) for i in range(6)])
        tight = slope([perbit(i, Z_WORST[i]) for i in range(6)])

        self.assertClose(published, 1.65835, atol=1e-4, msg=f"measured slope {published:.5f}")
        self.assertClose(loose, 1.26607, atol=1e-4, msg=f"asymptotic {loose:.5f}")
        self.assertGreater(published, tight, msg=f"measured {published:.4f} vs worst-case {tight:.4f}")
        self.assertGreater(tight, loose, msg=f"worst-case {tight:.4f}, asymptotic {loose:.4f}")

    def test_zhang_snr(self):
        """
        Their SNR column recomputed from $V_A$, $T$, $v_{el}$ and $\\xi$ needs 0.020-0.290 dB of
        loss the \"without splices\" attenuation leaves unreported.
        """
        gaps = [missing(i) for i in range(6)]

        for i, got in enumerate(gaps):
            self.assertGreater(got, 0.0, msg=f"point {i} gap {got:.4f} dB")
            self.assertLess(got, 0.3, msg=f"point {i} gap {got:.4f} dB")
        self.assertClose(
            max(gaps),
            0.2900,
            atol=1e-3,
            msg=f"worst gap {max(gaps):.4f} dB",
        )
        self.assertClose(snr(5), 0.0023167, atol=1e-6, msg=f"202.81 km SNR {snr(5):.7f}")

    def test_zhang_forms(self):
        """
        Their $V_A(1 - e^{-v_{err}})$ phase form agrees with qkd's literature and estimator
        forms to 1.9e-5 and 1.8e-4 relative at $v_{err}$ = 7.600e-5 rad^2.
        """
        v_err = -math.log(1.0 - Z_KAPPA)
        self.assertClose(v_err, 7.600289e-5, atol=1e-10, msg=f"v_err {v_err:.7e}")

        theirs = Z_VA[0] * (1.0 - math.exp(-v_err))
        default = budget.phase(Z_VA[0], v_err, Z_XI[0])
        older = budget.phase(Z_VA[0], v_err, form="literature")

        self.assertClose(
            default / theirs - 1.0,
            1.804e-4,
            atol=1e-6,
            msg=f"estimator form {default / theirs - 1.0:.4e}",
        )
        self.assertClose(
            older / theirs - 1.0,
            1.900e-5,
            atol=1e-7,
            msg=f"literature form {older / theirs - 1.0:.4e}",
        )

    def test_zhang_deficit(self):
        """
        Declared miss: their $(1 - \\kappa) = 7.6$e-5 carries 72.8% of the measured $\\xi$ at
        27.27 km and 7.2% at 202.81 km, against their naming residual phase noise the main
        contribution.
        """
        v_err = -math.log(1.0 - Z_KAPPA)
        shares = [budget.phase(Z_VA[i], v_err, Z_XI[i]) / Z_XI[i] for i in range(6)]

        self.assertClose(shares[0], 0.7282, atol=1e-3, msg=f"27.27 km phase share {shares[0]:.4f}")
        self.assertClose(shares[5], 0.0719, atol=1e-3, msg=f"202.81 km phase share {shares[5]:.4f}")
        self.assertMonotone(shares, rising=False, msg=f"shares {shares}")

        need = (Z_XI[5] / Z_VA[5]) / v_err
        self.assertClose(
            need,
            13.93,
            atol=0.02,
            msg=f"202.81 km phase need {need:.2f}x",
        )

    def test_zhang_attenuation(self):
        """
        Their loss column over length gives 0.15988 to 0.16816 dB/km against a stated 0.16
        without splices, three of six on it exactly.
        """
        rates = [Z_LOSS[i] / Z_KM[i] for i in range(6)]

        for i, got in enumerate(rates):
            self.assertGreater(got, 0.1598, msg=f"point {i} {got:.5f} dB/km")
            self.assertLess(got, 0.169, msg=f"point {i} {got:.5f} dB/km")

        # 2e-4 dB/km: 0.005 dB rounding over the shortest link, 0.005/27.27 = 1.83e-4.
        exact = [got for got in rates if abs(got - Z_ALPHA) < 2e-4]
        self.assertEqual(len(exact), 3, msg=f"dB/km {rates}")
        self.assertClose(
            max(rates) - Z_ALPHA,
            0.00816,
            atol=1e-5,
            msg=f"worst excess {max(rates) - Z_ALPHA:.5f} dB/km",
        )


# Hajomer, Derkach, Jain, Chin, Andersen & Gehring, "Long-distance continuous-
# variable quantum key distribution over 100 km of optical fiber" is the other
# Hajomer paper; this row is the discrete-modulation one: "Continuous-variable
# quantum key distribution at 10 GBaud using an integrated photonic-electronic
# receiver", Optica 11, 1197 (2024), arXiv:2305.19642. Table 1 from the arXiv LaTeX source; V_el and eps are
# printed in percent of an SNU, carried here as SNU.
#
#   "we presume that the receiver station is fully trusted. Hence, the
#    heterodyne detector efficiency eta = 44% and electronic noise with variance
#    V_el were modeled as linear coupling of the signal to a thermal noise
#    source purified by trusted parties"
#   "the mean excess noise observed in our experiment eps = 0.035, 0.071, 0.032
#    SNU (at channel input)"
# Their bound is Denys, Brown & Leverrier's, as `dm_rate`, but on shaped QAM and a trusted receiver;
# `dm_rate` is M-PSK and untrusted-only.
H_ROWS = (
    (16, 10.0, 10.0, 0.87, 0.569, 0.0650, 0.02622, 0.048, 0.035, 0.351),
    (16, 8.0, 5.0, 1.01, 0.618, 0.0495, 0.05187, 0.035, 0.021, 0.171),
    (32, 10.0, 5.0, 0.93, 0.702, 0.0676, 0.07183, 0.033, 0.019, 0.194),
    (64, 8.0, 5.0, 1.03, 0.733, 0.0503, 0.01590, 0.115, 0.093, 0.746),
)
H_ETA = 0.44
H_BETA = 0.95


def shaped(row, trusted=True):
    """
    Gaussian-modulation rate at one row's parameters: their own M -> infinity limit at the same
    variance.
    """
    _m, _s, _km, v_m, trans, v_el, eps, _r, _f, _skr = row

    return gauss_link(
        v_m,
        trans,
        eps,
        qkd.Heterodyne(eta=H_ETA, v_el=v_el, trusted=trusted),
        H_BETA,
    )


@memo
def ringed(row):
    """
    The same row through q.PhaseShiftKeying, untrusted: a PSK ring where theirs is shaped QAM.
    """
    m, _s, _km, v_m, trans, v_el, eps, _r, _f, _skr = row
    link = qkd.Link(
        modulation=qkd.PhaseShiftKeying(states=m, alpha=math.sqrt(v_m / 2.0)),
        channel=qkd.Channel(T=trans, xi=eps, ref="input"),
        bob=qkd.Bob(detector=qkd.Heterodyne(eta=H_ETA, v_el=v_el, trusted=False)),
        security=qkd.Asymptotic(beta=H_BETA),
    )

    return link.run().explain["key_raw"]["value"]


class HajomerShaped(Question):
    """
    Tier B: Hajomer et al. 2024 (Optica 11, 1197; arXiv:2305.19642), discrete-modulation CV-QKD
    at 10 GBaud. Pinned: Table 1, beta = 0.95, eta = 0.44. Nothing fitted. `dm_rate` cannot
    reproduce their trusted rate; the Gaussian ceiling is asserted instead.
    """

    def test_shaped_ceiling(self):
        """
        Gaussian modulation at the same variance, the $M \\to \\infty$ limit, sits above their
        rate at M = 16 and 32 and is 0.988 of it at M = 64, as they claim.
        """
        ratios = [shaped(row) / row[7] for row in H_ROWS]

        for i, got in enumerate(ratios[:3]):
            self.assertGreater(got, 1.0, msg=f"M={H_ROWS[i][0]} ratio {got:.4f}")
        self.assertClose(ratios[3], 0.988, atol=5e-3, msg=f"M=64 ratio {ratios[3]:.4f}")
        self.assertLess(
            abs(ratios[3] - 1.0),
            min(abs(got - 1.0) for got in ratios[:3]),
            msg=f"ratios {ratios}",
        )
        self.assertClose(
            shaped(H_ROWS[3]),
            0.113630,
            atol=1e-5,
            msg=f"M=64 ceiling {shaped(H_ROWS[3]):.6f} bit/symbol",
        )

    def test_shaped_table(self):
        """
        Their SKR_finite column is the symbol rate times R_finite to 2.1% on all four rows, with
        R_finite under R_asymptotic.
        """
        for i, row in enumerate(H_ROWS):
            got = row[1] * row[8]
            self.assertClose(
                got / row[9],
                1.0,
                atol=0.022,
                msg=f"row {i}: {got:.4f} Gb/s",
            )
            self.assertLess(row[8], row[7], msg=f"row {i}: R_finite {row[8]}")

    def test_shaped_untrusted(self):
        """
        `dm_rate` folds $\\eta$ and $v_{el}$ in as untrusted, negative on all four rows, and the
        same swap in the Gaussian layer costs 2.91 times their asymptotic rate.
        """
        for i, row in enumerate(H_ROWS):
            self.assertLess(ringed(row), 0.0, msg=f"row {i} untrusted M-PSK {ringed(row):.5f}")
            self.assertLess(
                shaped(row, trusted=False),
                0.0,
                msg=f"row {i} untrusted Gaussian {shaped(row, False):.5f}",
            )

        gap = shaped(H_ROWS[3]) - shaped(H_ROWS[3], trusted=False)
        self.assertClose(gap, 0.335193, atol=1e-5, msg=f"trust gap {gap:.6f}")
        self.assertClose(
            gap / H_ROWS[3][7],
            2.9147,
            atol=1e-3,
            msg=f"trust gap ratio {gap / H_ROWS[3][7]:.4f}",
        )

    def test_shaped_refusal(self):
        """
        `q.Link` refuses a trusted detector under `q.PhaseShiftKeying` outright.
        """
        link = qkd.Link(
            modulation=qkd.PhaseShiftKeying(states=4, alpha=0.4),
            channel=qkd.Channel(T=0.733, xi=0.0159, ref="input"),
            bob=qkd.Bob(detector=qkd.Heterodyne(eta=H_ETA, v_el=0.0503, trusted=True)),
            security=qkd.Asymptotic(beta=H_BETA),
        )

        with self.assertRaises(NotImplementedError, msg="trusted M-PSK"):
            link.run()


# Rubenok, Slater, Chan, Lucio-Martinez & Tittel, "Real-World Two-Photon
# Interference and Proof-of-Principle Quantum Key Distribution Immune to
# Detector Attacks", Phys. Rev. Lett. 111, 130501 (2013), arXiv:1304.2463.
# Time-bin MDI-QKD at 2 MHz, three spools and one deployed link. Table I and supplementary gain table from the
# arXiv LaTeX source: (name, loss_a dB, loss_b dB, mu_signal, four measured gains, four measured error rates).
# "Alice and Bob both select the same
# mean photon numbers for the three intensities and use channels of equal
# transmission."
R_SETS = (
    ("1a", 4.6, 4.5, 0.396, 1.028e-4, 0.0311, 1.95e-4, 0.270, 1.89e-6, 3.40e-6),
    ("1b", 6.8, 6.9, 0.279, 1.67e-5, 0.041, 3.57e-5, 0.274, 6.0e-7, 1.192e-6),
    ("1c", 9.1, 9.1, 0.251, 5.57e-6, 0.053, 9.87e-6, 0.270, 2.66e-7, 4.49e-7),
    ("2", 4.5, 4.5, 0.402, 1.042e-4, 0.0323, 2.020e-4, 0.265, 1.82e-6, 3.35e-6),
)

# Their other published constants:
#   R_DECOY  "For our decoy intensity we generated attenuated laser pulses
#            containing on average mu = sigma = 0.05 +/- 5% photons"
#   R_VACUUM the third setting of the three-intensity protocol
#   R_QVV    the vacuum-vacuum gain, common to all four setups; no signal,
#            misalignment or efficiency in it
#   R_HOM    "we found V_HOM = 47 +/- 1%, which is close to the maximum value of
#            50% for attenuated laser pulses with a Poissonian photon number
#            distribution"
#   R_PULSE  500 ps FWHM temporal modes separated by 1.4 ns at 1552.910 nm
#   R_DRIFT  arrival-time difference held "under 30 ps"
#   R_FEC    f = 1.14, the value they take from the literature
R_DECOY = 0.05
R_VACUUM = 0.0
R_QVV = 7.1e-10
R_HOM = 0.47
R_PULSE = 500e-12
R_DRIFT = 30e-12
R_FEC = 1.14

# STRUCTURAL: Ma & Razavi's relay announces four patterns; Rubenok's gains are "the probability of a projection
# onto |psi-> per emitted pair of pulses". Cancels from every error rate.
R_HALF = 0.5

# PINNED INPUT from the same apparatus: Chan, Slater, Lucio-Martinez,
# Rubenok & Tittel, Opt. Express 22, 12716 (2014), arXiv:1204.0738, Table 1 --
# efficiency 0.145, dark (1.83 +/- 0.77)e-5 per time bin. R_DARK is derived from R_QVV and checked against R_PDARK.
R_ETA = 0.145
R_PDARK = 1.83e-5
R_DARK = math.sqrt(R_QVV / 2.0)

# FITTED, the only fitted quantity: misalignment from setup 2's rectilinear QBER.
R_MISALIGN = 0.02887035


# Diagonal-basis 3x3 gain grid, row-major, Alice's intensity as row, [signal, decoy, vacuum]. The unpublished
# (signal, decoy) and (decoy, signal) cells are unread by `mdi_y11`; 0.0 stands in (test_mdi_decoy).
R_GRIDS = (
    (1.95e-4, 0.0, 5.68e-5, 0.0, 3.40e-6, 8.76e-7, 5.77e-5, 8.59e-7, R_QVV),
    (3.57e-5, 0.0, 9.62e-6, 0.0, 1.192e-6, 3.08e-7, 9.32e-6, 3.03e-7, R_QVV),
    (9.87e-6, 0.0, 2.50e-6, 0.0, 4.49e-7, 1.25e-7, 2.95e-6, 1.22e-7, R_QVV),
    (2.020e-4, 0.0, 5.63e-5, 0.0, 3.35e-6, 8.5e-7, 5.10e-5, 8.5e-7, R_QVV),
)

# `mdi_e11` error rates (decoy-decoy, vacuum-vacuum, decoy-vacuum, vacuum-decoy), as measured, not rounded to 1/2.
# e^x_vv is their Fig. 3 caption's 0.49 +/- 0.021 for both bases and all distances.
R_XERRS = (
    (0.277, 0.49, 0.511, 0.503),
    (0.278, 0.49, 0.50, 0.50),
    (0.286, 0.49, 0.51, 0.51),
    (0.269, 0.49, 0.502, 0.501),
)

# Their key rates S, "in bits per detector gate"; not reproduced (test_mdi_rates).
R_RATES = (1.4e-6, 1.7e-7, 1.2e-7, 1.5e-6)


def elevens(row, grid, errs):
    """
    (Y11, e11) in the diagonal basis from their measured grid alone, by the Xu-Curty-Qi-Lo
    three-intensity bound.
    """
    sets = [row[3], R_DECOY, R_VACUUM]
    y11 = _core.mdi_y11(sets, sets, list(grid))
    cells = (grid[4] * errs[0], grid[8] * errs[1], grid[5] * errs[2], grid[7] * errs[3])

    return y11, _core.mdi_e11(R_DECOY, R_VACUUM, R_DECOY, R_VACUUM, *cells, y11)


def bothways(row):
    """
    (eta_a, eta_b) at one setup: published per-arm loss through the companion paper's detector
    efficiency.
    """

    return tuple(R_ETA * 10.0 ** (-db / 10.0) for db in (row[1], row[2]))


# (signal-z, signal-x, decoy-z, decoy-x) as (basis engine, is decoy).
R_LEGS = (
    (_core.mdi_rect, False),
    (_core.mdi_diag, False),
    (_core.mdi_rect, True),
    (_core.mdi_diag, True),
)


def forward(row, ed=R_MISALIGN, arms=None):
    """
    The four R_LEGS settings forward-modelled as (gain, error), gains halved onto their |psi->
    convention; ``arms`` overrides the efficiencies.
    """
    eta_a, eta_b = bothways(row) if arms is None else arms
    out = [fn(mu, mu, eta_a, eta_b, R_DARK, ed) for fn, low in R_LEGS for mu in (R_DECOY if low else row[3],)]

    return [(R_HALF * gain, err) for gain, err in out]


def measured(row):
    """
    Their four published gains for the same settings.
    """

    return [row[4], row[6], row[8], row[9]]


def inverted(row, which):
    """
    Detector efficiency their gain ``which`` implies alone.
    """

    def scan(eta):
        arms = tuple(eta * 10.0 ** (-db / 10.0) for db in (row[1], row[2]))

        return forward(row, arms=arms)[which][0]

    return bisect(scan, 0.02, 0.60, measured(row)[which])


class RubenokRelay(Question):
    """
    Tier B: Rubenok, Slater, Chan, Lucio-Martinez & Tittel 2013 (PRL 111, 130501;
    arXiv:1304.2463), MDI-QKD over three spools and a deployed link. Fitted: misalignment.
    Pinned input: 0.145 detector efficiency from the companion paper. Structural: 1/2. Key rates
    not reproduced; they disclaim a secure key.
    """

    def test_mdi_bell(self):
        """
        Reading Q_vv as a projection onto psi- alone gives p_d = 1.884e-5, 3.0% from their
        companion paper's 1.83e-5, against 27% for Ma and Razavi's four-pattern reading.
        """
        self.assertClose(R_DARK, 1.884144e-5, atol=1e-10, msg=f"p_d {R_DARK:.6e}")

        near = abs(R_DARK / R_PDARK - 1.0)
        far = abs(math.sqrt(R_QVV / 4.0) / R_PDARK - 1.0)

        self.assertLess(near, 0.04, msg=f"|psi-> reading off by {near:.4f}")
        self.assertGreater(far, 0.25, msg=f"four-pattern reading off by {far:.4f}")
        self.assertClose(
            R_HALF * _core.mdi_rect(0.0, 0.0, 0.05, 0.05, R_DARK, R_MISALIGN)[0],
            R_QVV,
            atol=1e-13,
            msg="halved vacuum gain",
        )

    def test_mdi_gains(self):
        """
        All sixteen measured gains across four setups, two bases and two intensities sit within
        0.876x to 1.051x of the model, nothing fitted.
        """
        ratios = [got / want for row in R_SETS for got, want in zip([g for g, _ in forward(row)], measured(row))]
        self.assertEqual(len(ratios), 16, msg=f"{len(ratios)} gains")

        for i, got in enumerate(ratios):
            self.assertGreater(got, 0.85, msg=f"gain {i} ratio {got:.4f}")
            self.assertLess(got, 1.10, msg=f"gain {i} ratio {got:.4f}")

        mean = sum(ratios) / len(ratios)
        self.assertClose(mean, 0.9813, atol=5e-3, msg=f"mean gain ratio {mean:.4f}")

    def test_mdi_efficiency(self):
        """
        Each gain inverted for detector efficiency alone gives a 10.2% spread straddling their
        companion paper's 0.145.
        """
        etas = [inverted(row, which) for row in R_SETS for which in range(4)]

        for got in etas:
            self.assertGreater(got, 0.135, msg=f"efficiency {got:.5f}")
            self.assertLess(got, 0.165, msg=f"efficiency {got:.5f}")

        wide = max(etas) / min(etas) - 1.0
        self.assertLess(wide, 0.12, msg=f"efficiency spread {wide:.4f} across {etas}")

        mean = sum(etas) / len(etas)
        self.assertClose(mean, R_ETA, atol=5e-3, msg=f"mean inverted efficiency {mean:.5f}")

    def test_mdi_qber(self):
        """
        One misalignment fitted to setup 2's rectilinear QBER puts all four diagonal QBERs
        within 0.8 points of measurement, near Ma and Razavi's $1/4 + e_d/2$.
        """
        fitted = bisect(lambda e: forward(R_SETS[3], e)[0][1], 0.0, 0.2, R_SETS[3][5])
        self.assertClose(fitted, R_MISALIGN, atol=1e-7, msg=f"misalignment {fitted:.7f}")

        for i, row in enumerate(R_SETS):
            got = forward(row)[1][1]
            self.assertClose(
                got,
                row[7],
                atol=8e-3,
                msg=f"setup {row[0]} diagonal QBER {got:.5f}",
            )

        limit = 0.25 + R_MISALIGN / 2.0
        self.assertClose(
            forward(R_SETS[3])[1][1],
            limit,
            atol=1e-3,
            msg=f"diagonal QBER against {limit:.5f}",
        )

    def test_mdi_visibility(self):
        """
        The Poissonian HOM ceiling is exactly their \"maximum value of 50%\", their $47 \\pm 1$%
        inverts to 0.940 mode overlap, and 500 ps pulses at 30 ps drift account for 0.50 of its
        6.0 missing points.
        """
        ceiling = _core.hom_visibility(1.0, R_SETS[3][3], R_SETS[3][3])
        self.assertClose(ceiling, 0.5, atol=1e-15, msg=f"HOM ceiling {ceiling}")

        overlap = bisect(
            lambda o: _core.hom_visibility(o, R_SETS[3][3], R_SETS[3][3]),
            0.0,
            1.0,
            R_HOM,
        )
        self.assertClose(overlap, 0.94, atol=1e-9, msg=f"overlap {overlap:.6f}")

        timed = _core.mode_overlap(R_PULSE, R_PULSE, R_DRIFT)
        self.assertGreater(timed, overlap, msg=f"timing overlap {timed:.5f}")
        self.assertClose(
            1.0 - timed,
            0.004978,
            atol=1e-5,
            msg=f"drift cost {1.0 - timed:.6f}",
        )

    def test_mdi_decoy(self):
        """
        Xu, Curty, Qi and Lo's single-photon-pair bound on their 3x3 diagonal grid sits within
        2.1% of the halved forward model at all four setups, above it on the error rate, and
        ignores the two unpublished cells.
        """
        for row, grid, errs in zip(R_SETS, R_GRIDS, R_XERRS):
            bound, want = elevens(row, grid, errs)
            eta_a, eta_b = bothways(row)
            true = R_HALF * _core.mdi_yield(eta_a, eta_b, R_DARK, R_MISALIGN)[0]
            moved = list(grid)
            moved[1], moved[3] = 1e-3, 1e-3

            self.assertEqual(
                _core.mdi_y11([row[3], R_DECOY, R_VACUUM], [row[3], R_DECOY, R_VACUUM], moved),
                bound,
                msg=f"setup {row[0]}: unpublished cells",
            )
            self.assertClose(
                bound / true,
                1.0,
                atol=0.07,
                msg=f"setup {row[0]}: bound {bound:.4e} vs forward {true:.4e}",
            )
            self.assertGreater(
                want,
                _core.mdi_yield(eta_a, eta_b, R_DARK, R_MISALIGN)[1],
                msg=f"setup {row[0]}: e11 bound {want:.5f}",
            )

    def test_mdi_rates(self):
        """
        Rebuilt from their grid the key rates are 2.94x to 7.36x their published figures,
        bounded not reproduced, their overlaps being "insufficient to securely distribute key".
        """
        ratios = []
        for row, grid, errs, published in zip(R_SETS, R_GRIDS, R_XERRS, R_RATES):
            y11, e11 = elevens(row, grid, errs)
            q11 = _core.mdi_gain(y11, row[3], row[3])
            ratios.append(_core.mdi_rate(q11, e11, row[4], row[5], R_FEC) / published)

        for i, got in enumerate(ratios):
            self.assertGreater(got, 1.0, msg=f"setup {i} ratio {got:.4f}")
            self.assertLess(got, 10.0, msg=f"setup {i} ratio {got:.4f}")
        self.assertGreater(
            max(ratios) / min(ratios),
            2.0,
            msg=f"ratios {ratios}",
        )

    def test_mdi_deficit(self):
        """
        Declared miss: the rectilinear QBER is exact at fitted setup 2 but 0.04429 against a
        measured 0.053 at 18.2 dB, a channel background the Ma-Razavi model has no term for.
        """
        near = forward(R_SETS[3])[0][1] - R_SETS[3][5]
        far = forward(R_SETS[2])[0][1] - R_SETS[2][5]

        self.assertClose(near, 0.0, atol=1e-5, msg=f"setup 2 gap {near:.6f}")
        self.assertClose(far, -0.008708, atol=5e-5, msg=f"18.2 dB gap {far:.6f}")
        self.assertLess(far, near, msg=f"gaps {near:.6f}, {far:.6f}")

        for row, measure in zip(R_SETS, (0.070, 0.082, 0.129, 0.071)):
            gap = measure - forward(row)[2][1]
            self.assertGreater(
                gap,
                0.0,
                msg=f"setup {row[0]}: decoy gap {gap:.5f}",
            )


# Honjo, Nam, Takesue, Zhang, Kamada, Nishida, Tadanaga, Asobe, Baek, Hadfield,
# Miki, Fujiwara, Sasaki, Wang, Inoue & Yamamoto, "Long-distance
# entanglement-based quantum key distribution over optical fiber", Opt. Express
# 16, 19118 (2008). No arXiv version; read from the open-access text at doi 10.1364/OE.16.019118.
# Time-bin entanglement, source at the midpoint of two 50 km dispersion-shifted-fibre arms.
#
#   H_PULSE  "The pulse width and repetition frequency were 100 ps and 1 GHz"
#   H_SLOT   "A series of double pulses was generated by extinguishing one of
#            three sequential pulses", so the time-bin slot rate is 1 GHz / 3;
#            the paper never states this figure
#   H_DARK   "The dark count rate was around 250 cps for each detector"
#   H_HIGH   "The quantum efficiency of one of the SSPDs was 2.0 %, and those of
#            the others were all 0.7%"
#   H_LOW    the same sentence
#
# A READING: H_HIGH on both signal-arm detectors, which the paper does not say. The arm mean, 1.35%, widens
# test_pairs_deficit's miss from 0.773x to 0.522x.
H_PULSE = 1e9
H_SLOT = 1e9 / 3.0
H_DARK = 250.0
H_HIGH = 0.020
H_LOW = 0.007

# Receiver loss chain, signal / idler dB: PPLN plus coupling 4.45 / 3.95, 775 nm filter 0.05, band-pass
# 0.35 / 0.53, delay interferometer "about 2.0 dB". At 100 km: "~17.4 dB for
# the signal and ~17.0 dB for the idler", spool "~10.5 dB".
H_NEAR = (4.45 + 0.05 + 0.35 + 2.0, 3.95 + 0.05 + 0.53 + 2.0)
H_FAR = (17.4, 17.0)
H_SPOOL = 10.5

# (mean pairs per slot, losses, total QBER, sifted bit/s, secure bit/s, raw visibility, their accidentals-only
# QBER, energy-basis QBER). 100 km: 0.07 per pulse, "which corresponds to an average number of photon pairs of 0.14 per
# time-bin entanglement slot".
H_RUNS = (
    (0.04, H_NEAR, 0.0235, 29.4, 20.2, 0.938, 0.019, 0.031),
    (0.14, H_FAR, 0.0691, 0.57, 0.14, 0.850, 0.0614, 0.0759),
)

# Interference runs, one detector per site on the full 1 GHz train: (pairs per pulse, losses, singles).
# "the count rates at the signal and idler detectors were ~36 and ~11 kcps"; ~12.0 / ~3.1 kcps at 100 km.
H_SINGLES = ((0.016, H_NEAR, 36e3, 11e3), (0.07, H_FAR, 12.0e3, 3.1e3))

# EXAM-SIDE: sifting unstated; 1/4 is forced by equal per-basis key lengths (63 080 against 62 476 at 0 km).
H_SIFT = 0.25

# FITTED, the only fitted quantity: misalignment from the 0 km total QBER.
H_MISALIGN = 0.00424426


def paired(run):
    """
    (lam, eta_signal, eta_idler, y0) for one run in Ma-Fung-Lo units: lam is pairs per pulse per
    mode pair (half their per-slot mean), y0 two detectors' dark counts per slot.
    """
    _mu, losses, *_rest = run
    keep = [10.0 ** (-db / 10.0) for db in losses]

    return (
        run[0] / 2.0,
        H_HIGH * keep[0],
        H_LOW * keep[1],
        2.0 * H_DARK / H_SLOT,
    )


def coincide(run, e_d=H_MISALIGN):
    """
    (gain per slot, QBER, sifted bit/s) at one run's brightness, through q.pairs.
    """
    lam, eta_a, eta_b, y0 = paired(run)
    gain = qkd.pairs.gain(lam, eta_a, eta_b, y0, y0)
    qber = qkd.pairs.qber(lam, eta_a, eta_b, y0, y0, e_d)

    return gain, qber, H_SIFT * H_SLOT * gain


def singled(entry, swap=False):
    """
    Their Eq. (5)-(6) singles rates for one interference run: half the pair number through each
    arm on the 1 GHz train plus dark count; the 1/2 is the unread interferometer port.
    """
    mu, losses, _s, _i = entry
    pair = (H_LOW, H_HIGH) if swap else (H_HIGH, H_LOW)

    return tuple(H_PULSE * 0.5 * mu * eta * 10.0 ** (-db / 10.0) + H_DARK for eta, db in zip(pair, losses))


class HonjoPairs(Question):
    """
    Tier B: Honjo et al. 2008 (Opt. Express 16, 19118; no arXiv version), BBM92 over 100 km.
    Fitted: misalignment, from the 0 km QBER. Key rates only bounded: theirs is Waks, Zeevi &
    Yamamoto's individual-attack bound, qkd's Shor-Preskill through Koashi-Preskill.
    """

    def test_pairs_assignment(self):
        """
        Their singles put the 2.0% detector on the signal arm: 33.3 and 12.7 kcps against ~36
        and ~11, the swap 14.5x worse.
        """
        got = singled(H_SINGLES[0])
        back = singled(H_SINGLES[0], swap=True)

        self.assertClose(got[0], 33296.0, atol=1.0, msg=f"signal singles {got[0]:.0f}")
        self.assertClose(got[1], 12700.5, atol=2.0, msg=f"idler singles {got[1]:.0f}")

        for i, want in enumerate((36e3, 11e3)):
            self.assertLess(
                abs(got[i] / want - 1.0),
                0.16,
                msg=f"arm {i} singles {got[i]:.0f}",
            )
            self.assertGreater(
                abs(back[i] / want - 1.0),
                0.65,
                msg=f"arm {i} swapped singles {back[i]:.0f}",
            )

        wrong = max(abs(back[i] / w - 1.0) for i, w in enumerate((36e3, 11e3)))
        right = max(abs(got[i] / w - 1.0) for i, w in enumerate((36e3, 11e3)))
        self.assertGreater(
            wrong / right,
            10.0,
            msg=f"swap ratio {wrong / right:.2f}",
        )

    def test_pairs_losses(self):
        """
        Their itemised receiver chain plus the "~10.5 dB" spool, exactly 0.21 dB/km over 50 km,
        rebuilds their "~17.4" and "~17.0" dB totals.
        """
        for i, want in enumerate(H_FAR):
            got = H_NEAR[i] + H_SPOOL
            self.assertClose(got, want, atol=0.06, msg=f"arm {i} loss {got:.2f} dB")

        extra = H_SPOOL - 0.21 * 50.0
        self.assertClose(
            extra,
            0.0,
            atol=1e-12,
            msg=f"spool excess {extra:.2f} dB",
        )

    def test_pairs_accidentals(self):
        """
        Their Eq. (11) accidental QBER $\\mu_t/(2(1+\\mu_t))$, 1.923% and 6.140%, sits 2.2% and
        5.8% above Ma, Fung and Lo's Eq. (10) at zero misalignment and background.
        """
        for run, want in zip(H_RUNS, (0.019, 0.0614)):
            lam, eta_a, eta_b, _y0 = paired(run)
            bare = qkd.pairs.qber(lam, eta_a, eta_b, 0.0, 0.0, 0.0)
            theirs = run[0] / (2.0 * (1.0 + run[0]))
            self.assertClose(theirs, want, atol=6e-4, msg=f"Eq. (11) {theirs:.5f}")
            self.assertLess(bare, theirs, msg=f"exact {bare:.5f}, Eq. (11) {theirs:.5f}")
            self.assertGreater(
                bare / theirs,
                0.93,
                msg=f"exact {bare:.5f}, Eq. (11) {theirs:.5f}",
            )
            self.assertLess(bare, run[2], msg=f"accidental QBER {bare:.5f}")

    def test_pairs_qber(self):
        """
        One misalignment fitted to their 2.35% back-to-back QBER predicts 6.702% at 100 km
        against their 6.91%, across 21 dB.
        """
        fitted = bisect(lambda e: coincide(H_RUNS[0], e)[1], 0.0, 0.3, H_RUNS[0][2])
        self.assertClose(fitted, H_MISALIGN, atol=1e-7, msg=f"misalignment {fitted:.7f}")
        self.assertLess(fitted, 0.01, msg=f"misalignment {fitted:.5f}")

        got = coincide(H_RUNS[1])[1]
        self.assertClose(got, 0.06702, atol=1e-4, msg=f"100 km QBER {got:.5f}")
        self.assertClose(got, H_RUNS[1][2], atol=3e-3, msg=f"100 km QBER {got:.5f}")

    def test_pairs_visibility(self):
        """
        Their raw two-photon visibilities predict the energy-basis QBERs through $e = (1 -
        V)/2$: 3.100% against 3.1% at 0 km and 7.500% against 7.59% at 100 km.
        """
        for run in H_RUNS:
            got = 0.5 * (1.0 - run[5])
            self.assertClose(
                got,
                run[7],
                atol=1e-3,
                msg=f"V = {run[5]}: e {got:.5f}",
            )
        self.assertLess(H_RUNS[1][5], H_RUNS[0][5], msg="visibilities")
        self.assertGreater(
            H_RUNS[0][7],
            0.5 * (1.0 - 1.0),
            msg="0 km energy-basis QBER",
        )

    def test_pairs_deficit(self):
        """
        Declared miss: their sifted rate falls 51.6x between runs against the model's 31.3x,
        0.773x their 29.4 bit/s at 0 km and 1.275x their 0.57 at 100 km.
        """
        rates = [coincide(run)[2] for run in H_RUNS]
        ratios = [got / run[3] for got, run in zip(rates, H_RUNS)]

        self.assertClose(ratios[0], 0.7733, atol=5e-3, msg=f"0 km rate {rates[0]:.4f} bit/s")
        self.assertClose(ratios[1], 1.2745, atol=5e-3, msg=f"100 km rate {rates[1]:.4f} bit/s")
        self.assertLess(ratios[0], 1.0, msg=f"0 km ratio {ratios[0]:.4f}")
        self.assertGreater(ratios[1], 1.0, msg=f"100 km ratio {ratios[1]:.4f}")

        fall = (H_RUNS[0][3] / H_RUNS[1][3]) / (rates[0] / rates[1])
        self.assertClose(
            fall,
            1.648,
            atol=5e-3,
            msg=f"steepness ratio {fall:.4f}",
        )

    def test_pairs_bounded(self):
        """
        qkd's Shor-Preskill `pair_rate` keeps 94.8% of their Waks-Zeevi-Yamamoto
        individual-attack fraction at 0 km and 86.8% at 100 km.
        """
        for run in H_RUNS:
            share = qkd.pairs.rate(1.0, run[2], run[2], 1.17, 1.0)
            theirs = run[4] / run[3]
            self.assertLess(
                share,
                theirs,
                msg=f"collective {share:.5f}, individual {theirs:.5f}",
            )
            self.assertGreater(
                share / theirs,
                0.8,
                msg=f"ratio {share / theirs:.4f}",
            )

        shares = [qkd.pairs.rate(1.0, run[2], run[2], 1.17, 1.0) / (run[4] / run[3]) for run in H_RUNS]
        self.assertClose(shares[0], 0.948015, atol=1e-5, msg=f"0 km ratio {shares[0]:.6f}")
        self.assertClose(shares[1], 0.868246, atol=1e-5, msg=f"100 km ratio {shares[1]:.6f}")
        self.assertLess(shares[1], shares[0], msg=f"ratios {shares}")


# Hajomer, Andersen & Gehring, "Continuous-variable measurement-device-
# independent quantum key distribution with a locally generated local
# oscillator", Quantum Sci. Technol. 10, 025032 (2025), arXiv:2303.01611. 10 km of fibre, relay at Alice.
#
# Table 1: V_A = V_B = 6.5 SNU, 20 MBaud, N = 4e6,
# xi = 39.5 mSNU, tau_A = 1, tau_B = 0.56, relay efficiency 0.94, beta = 97%.
# "After CV BSM, an excess noise of 39.5 mSNU was measured at the relay";
# "the worst-case estimator consider finite-size correction was ~45 mSNU";
# "we achieved a positive expected secrete key rate of 2.6 Mbit/s" at a block
# size of 4e6 and a failure probability of 1e-10.
C_VA = 6.5
C_TAUA = 1.0
C_TAUB = 0.56
C_RELAY = 0.94
C_BETA = 0.97
C_XI = 0.0395
C_WORST = 0.045
C_BAUD = 20e6
C_SKR = 2.6e6

# Their phase-noise form and value:
# "the excess noise due to the residual phase noise can be calculated as
# xi_phase = 2*T*V*(1 - exp(-sigma_theta^2/2)), where T is the transmittance,
# including the quantum channel and the relay efficiency", giving
# "xi_phase ~ 12.6 mSNU, considering sigma_theta = 0.06 rad".
C_SIGMA = 0.06
C_PHASE = 0.0126


def swapped(ta, xi_a, tb, xi_b):
    """
    Their two arms as a q.Swap with an ideal Bell detector, all loss and noise in the channels.
    """

    return qkd.Swap(
        alice=qkd.Sender(modulation=qkd.GaussianModulation(v_a=C_VA)),
        bob=qkd.Sender(modulation=qkd.GaussianModulation(v_a=C_VA)),
        channels=(
            qkd.Channel(T=ta, xi=xi_a, ref="input"),
            qkd.Channel(T=tb, xi=xi_b, ref="input"),
        ),
        relay=qkd.Relay(bell=qkd.BellDetector(eta=1.0, v_el=0.0)),
        security=qkd.Asymptotic(beta=C_BETA),
    ).run()


def relayed(share=1.0, xi=None):
    """
    (chi, security bound, per-attack point) at Table 1's arms, relay efficiency calibrated out;
    ``xi`` overrides the relay-plane noise, ``share`` puts that share on Bob's arm. Reads
    `key_bound`, not key_rate, which carries q.Swap's ONE clamp.
    """
    xi_b = share * (C_XI if xi is None else xi) / C_TAUB
    out = swapped(C_TAUA, 0.0, C_TAUB, xi_b)

    return out.chi, out.explain["key_bound"]["value"], out.attack.key_rate


class HajomerRelay(Question):
    """
    Tier B: Hajomer, Andersen & Gehring 2025 (Quantum Sci. Technol. 10, 025032;
    arXiv:2303.01611), CV-MDI-QKD over 10 km. Nothing fitted. Declared readings: relay-plane
    noise referral (bracketed) and relay efficiency charged to Eve (flips the sign).
    """

    def test_cvmdi_phase(self):
        """
        Their phase-noise form is `form="literature"` term for term, 12.307 mSNU at 0.06 rad
        against their 12.6, inside `PHASE_LIMIT` with no `PhaseDomainWarning`.
        """
        got = C_TAUB * C_RELAY * budget.phase(C_VA, C_SIGMA**2, form="literature")
        self.assertClose(got, 0.01230668, atol=1e-7, msg=f"xi_phase {got:.8f} SNU")
        self.assertClose(got, C_PHASE, atol=4e-4, msg=f"xi_phase {got:.6f} SNU")

        back = bisect(
            lambda s: C_TAUB * C_RELAY * budget.phase(C_VA, s * s, form="literature"),
            0.0,
            0.5,
            C_PHASE,
        )
        self.assertClose(back, 0.0607115, atol=1e-6, msg=f"sigma {back:.6f} rad")

        split = budget.phase(C_VA, C_SIGMA**2, C_XI) / budget.phase(C_VA, C_SIGMA**2, form="literature")
        self.assertClose(split, 1.008797, atol=1e-5, msg=f"form ratio {split:.6f}")
        self.assertLess(
            C_SIGMA**2,
            budget.PHASE_LIMIT,
            msg=f"v_err {C_SIGMA**2}",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", budget.PhaseDomainWarning)
            quiet = budget.phase(C_VA, C_SIGMA**2, form="literature")
        self.assertGreater(quiet, 0.0, msg=f"xi_phase {quiet}")

        share = C_TAUB * C_RELAY * quiet / C_XI
        self.assertClose(share, 0.31156, atol=1e-4, msg=f"phase share {share:.5f}")

    def test_cvmdi_rate(self):
        """
        Their 39.5 mSNU referred to Bob's arm through $\\tau_B$ gives a 0.180 bit/symbol
        per-attack rate at their $\\beta$, above the 0.130 their 2.6 Mbit/s at 20 MBaud comes
        to.
        """
        chi, bound, point = relayed()
        theirs = C_SKR / C_BAUD

        self.assertClose(theirs, 0.13, atol=1e-9, msg=f"their rate {theirs} bit/symbol")
        self.assertClose(chi, 5.681464, atol=1e-5, msg=f"equivalent noise {chi:.6f}")
        self.assertClose(point, 0.180425, atol=1e-5, msg=f"per-attack rate {point:.6f} bit/symbol")
        self.assertGreater(
            point,
            theirs,
            msg=f"per-attack {point:.5f}",
        )
        self.assertGreater(bound, point, msg=f"bound {bound:.5f}, point {point:.5f}")

    def test_cvmdi_floor(self):
        """
        The pure-loss equivalent noise at $\\tau_A = 1$, $\\tau_B = 0.56$ is 5.5714, against
        4.3457 for the most favourable correlated environment and their measured 5.6815.
        """
        out = swapped(C_TAUA, 0.0, C_TAUB, C_XI / C_TAUB)
        chi, floor, least = out.chi, out.floor, out.least

        self.assertClose(floor, 5.571429, atol=1e-5, msg=f"pure-loss floor {floor:.6f}")
        self.assertClose(least, 4.345714, atol=1e-5, msg=f"least noise {least:.6f}")
        self.assertGreater(chi, floor, msg=f"chi {chi:.6f}")
        self.assertClose(
            chi / floor - 1.0,
            0.019756,
            atol=1e-5,
            msg=f"excess over floor {chi / floor - 1.0:.6f}",
        )

    def test_cvmdi_referral(self):
        """
        Every referral of their relay-plane 39.5 mSNU, from all on Bob to an even split, and
        their worst-case estimator, stays above their 0.130 bit/symbol.
        """
        band = [relayed(share)[2] for share in (1.0, 0.75, 0.5)]
        self.assertMonotone(band, msg=f"rates {band}")

        for got in band:
            self.assertGreater(got, C_SKR / C_BAUD, msg=f"rate {got:.5f}")
        self.assertClose(band[2], 0.248448, atol=1e-5, msg=f"even split {band[2]:.6f}")

        worst = relayed(xi=C_WORST)[2]
        self.assertLess(worst, band[0], msg=f"worst-case {worst:.5f}")
        self.assertGreater(
            worst,
            C_SKR / C_BAUD,
            msg=f"worst-case {worst:.5f}",
        )

    def test_cvmdi_relay(self):
        """
        Charging the relay's 94% efficiency to Eve as arm loss, instead of calibrating it out as
        their $\\tau_A = 1$ does, takes the rate to -0.134 bit/symbol.
        """
        ta, tb = C_TAUA * C_RELAY, C_TAUB * C_RELAY
        out = swapped(ta, 0.0, tb, C_XI / tb)
        chi, point = out.chi, out.attack.key_rate

        self.assertClose(chi, 6.044111, atol=1e-5, msg=f"chi {chi:.6f}")
        self.assertClose(point, -0.134048, atol=1e-5, msg=f"rate {point:.6f} bit/symbol")
        self.assertLess(point, 0.0, msg=f"rate {point:.6f}")
        self.assertGreater(relayed()[2], 0.0, msg="calibrated-out rate")


if __name__ == "__main__":
    rc = Exam(
        "ExpLvovskyFock",
        "Tier B: Lvovsky et al. 2001, the measured single-photon Wigner dip",
        "exp_fock.md",
    ).run(load(LvovskyFock))
    rc |= Exam(
        "ExpZhangLongHaul",
        "Tier B: Zhang et al. 2020, six measured CV points to 202.81 km",
        "exp_longhaul.md",
    ).run(load(ZhangLongHaul))
    rc |= Exam(
        "ExpRubenokRelay",
        "Tier B: Rubenok et al. 2013, MDI-QKD measured over deployed fibre",
        "exp_mdi.md",
    ).run(load(RubenokRelay))
    rc |= Exam(
        "ExpHonjoPairs",
        "Tier B: Honjo et al. 2008, BBM92 measured over 100 km of fibre",
        "exp_pairs.md",
    ).run(load(HonjoPairs))
    rc |= Exam(
        "ExpHajomerRelay",
        "Tier B: Hajomer et al. 2025, CV-MDI measured over 10 km of fibre",
        "exp_cvmdi.md",
    ).run(load(HajomerRelay))
    rc |= Exam(
        "ExpHajomerShaped",
        "Tier B: Hajomer et al. 2024, discrete modulation measured at 10 GBaud",
        "exp_shaped.md",
    ).run(load(HajomerShaped))
    sys.exit(rc)
