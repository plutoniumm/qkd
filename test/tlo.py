import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.cache import memo
from kit.checks import Guarded
from qkd import _core
from qkd import attacks, budget

# src/tlo.rs and src/pipeline.rs's run_tlo branch. q.TransmittedLO itself is exammed in
# test/surface.py.
#
# Two published TLO experiments, both read from the e-prints named beside them:
#
# Lodewyck, Bloch, Garcia-Patron, Fossier, Karpov, Diamanti, Debuisschert,
# Cerf, Tualle-Brouri, McLaughlin & Grangier, Phys. Rev. A 76, 042305 (2007),
# arXiv:0706.4255. Time-division multiplexed, 400 ns apart down an 80 m delay line, LO
# about 1e9 photons per pulse, shot noise normalised from the monitored LO level.
#
# Zhang, Chen, Pirandola, Wang, Zhou, Chu, Zhao, Xu, Yu & Guo, Phys. Rev.
# Lett. 125, 010502 (2020), arXiv:2001.02555. Time AND polarisation multiplexed, NKT BasiK
# E15 at 100 Hz, 5 MHz clock, and -- unlike Lodewyck -- the signal path is randomly switched
# off for a real-time shot-noise calibration.
LOD_VA = 18.5
LOD_T = 0.302
LOD_XI = 0.005
LOD_ETA = 0.606
LOD_VEL = 0.041
LOD_BETA = 0.898
LOD_LO = 1e9
LOD_DELAY = 400e-9

# Lodewyck's three pinned outputs, as test/anchors.py and ci/smoke.py carry them.
LOD_IAB = 1.0436
LOD_CHI = 0.9020
LOD_KEY = 0.0352

ZH_VA = 7.65
ZH_T = 10**-3.245
ZH_XI = 0.0081
ZH_ETA = 0.6134
ZH_VEL = 0.1523
ZH_KAPPA = 7.6e-5
ZH_WIDTH = 100.0
ZH_CLOCK = 5e6

# Jouguet arXiv:1304.7024's worked point; qkd/attacks.py pins the 1.525 that zeroes the
# report.
JG_T = 0.5
JG_ETA = 0.5
JG_XI = 0.1
JG_RESEND = 2.0

# The arbiter's operating point. run_symbols' oracle estimator at 2^22 symbols scatters
# with sd 0.00142 SNU about a mean offset of 0.00077; ARB_TOL is applied to a four-seed
# mean, so the bar sits ~6 sigma out.
ARB_VA = 5.0
ARB_T = 0.8
ARB_XI = 0.2
ARB_ETA = 0.9
ARB_VEL = 0.05
ARB_N = 1 << 22
ARB_TOL = 5e-3

# The ladder sweeps the residual variance over two decades, 7.6e-3 to 0.76 rad^2, at
# Zhang's implied delay and 5 MHz clock.
TLO_LADDER = (1e4, 1e5, 3e5, 1e6)

# Per-seed sampling error over eight seeds: v_err 1.4e-3 and the transmittance 1.5e-3 at
# the worst ladder point, so this bar is ~3x it.
TLO_TOL = 4e-3

# The phase row is a DIFFERENCE of two estimates: 1.0e-2 at the bottom of the ladder,
# 2.7e-3 at the top.
TLO_ROW = 3e-2

# The two halves together are tighter than either alone: 5.5e-4 at the worst ladder point.
TLO_PLANE = 2e-3


def zhang_verr():
    """
    Zhang's residual phase noise as a variance, from 1 - kappa in the small-angle limit.
    """
    return -math.log(1.0 - ZH_KAPPA)


def zhang_deficit():
    """
    The part of Zhang's 202.81 km excess noise the phase row misses, channel-input, in SNU.
    """
    return ZH_XI - budget.phase(ZH_VA, zhang_verr())


def zhang_delay():
    """
    The multiplexing delay Zhang's residual phase noise implies at their published
    linewidth, in seconds. FITTED to their kappa, not read off their paper.
    """

    return zhang_verr() / (math.tau * ZH_WIDTH)


@memo
def tlo_run(lw, delay, seed=1):
    """
    One run_tlo at the arbiter's operating point and Zhang's clock, memoised.
    """

    return _core.run_tlo(ARB_N, seed, ARB_VA, ARB_T, ARB_XI, ARB_ETA, ARB_VEL, lw, delay, ZH_CLOCK, False)


def oracle_t(out):
    """
    The oracle branch's channel transmittance, 2*t^2/eta on its slope -- SimOut.t_chan's map.
    """

    return 2.0 * out.t_ideal * out.t_ideal / ARB_ETA


class LoArm(Question):
    """
    The oscillator's propagation, and the shot-noise unit it sets.
    """

    def test_lo_arrives(self):
        """
        tlo_shot returns an LO arm of T*mux and an LO at Bob of n_LO*T*mux.
        """
        n_bob, t_lo, _unit, _vel = _core.tlo_shot(LOD_LO, LOD_T, 0.5, LOD_LO, LOD_VEL)
        self.assertClose(t_lo, LOD_T * 0.5, atol=1e-15, msg="t_lo != T*mux")
        self.assertClose(n_bob, LOD_LO * LOD_T * 0.5, atol=1e-3, msg="n_bob != n_lo*T*mux")

    def test_unit_tracks(self):
        """
        The shot-noise unit is 1 at the reference LO and halves exactly when the LO arm does.
        """
        full = _core.tlo_shot(LOD_LO, 1.0, 1.0, LOD_LO, LOD_VEL)[2]
        half = _core.tlo_shot(LOD_LO, 0.5, 1.0, LOD_LO, LOD_VEL)[2]
        self.assertClose(full, 1.0, atol=0.0, msg="unit != 1 at the reference LO")
        self.assertClose(half / full, 0.5, atol=1e-15, msg="unit ratio != 0.5")

    def test_vel_grows(self):
        """
        $v_{el}$ in shot-noise units is $v_{el}^{ref}/T_{LO}$, and exceeds 1 after 20 dB of LO
        loss: the noise is fixed in volts while the unit shrinks with the oscillator.
        """
        vel = _core.tlo_shot(LOD_LO, 0.01, 1.0, LOD_LO, LOD_VEL)[3]
        self.assertClose(vel, LOD_VEL / 0.01, atol=1e-12, msg="v_el over 20 dB")
        self.assertGreater(vel, 1.0, msg="v_el under 1 after 20 dB")

    def test_untrusted_square(self):
        """
        With the LO on the signal's span the untrusted penalty 2*v_el/(eta*T) is quadratic
        in the loss: a factor 4 from T = 0.4 to 0.2.
        """
        pen = []
        for t in (0.4, 0.2):
            vel = _core.tlo_shot(LOD_LO, t, 1.0, LOD_LO, LOD_VEL)[3]
            pen.append(2.0 * vel / (LOD_ETA * t))
        self.assertClose(pen[1] / pen[0], 4.0, atol=1e-12, msg="penalty ratio != 4")

    def test_leak_displaces(self):
        """
        tlo_leak at 60 dB extinction on Lodewyck's $10^9$-photon oscillator is
        2*sqrt(1e3) = 63 SNU, above the modulation variance 18.5.
        """
        shift = _core.tlo_leak(LOD_LO, 1e-6)
        self.assertClose(shift, 2.0 * math.sqrt(1e3), atol=1e-9, msg="tlo_leak != 2 sqrt(1e3)")
        self.assertGreater(shift, LOD_VA, msg="leak below V_A")

    def test_leak_novariance(self):
        """
        _core exposes no tlo_leak_xi, tlo_leak_noise or tlo_leak_variance.
        """
        for name in ("tlo_leak_xi", "tlo_leak_noise", "tlo_leak_variance"):
            self.assertFalse(hasattr(_core, name), msg=f"_core has {name}")

    def test_phase_wiener(self):
        """
        tlo_phase is 2*pi*linewidth*delay, equalling run_symbols' per-symbol walk increment
        at delay = 1/symbol_rate.
        """
        step = _core.tlo_phase(1.5e3 + 2.5e3, 1.0 / 1e8)
        self.assertClose(step, math.tau * 4.0e3 / 1e8, atol=1e-18, msg="tlo_phase != run_symbols walk step")
        self.assertClose(
            _core.tlo_phase(ZH_WIDTH, LOD_DELAY),
            math.tau * ZH_WIDTH * LOD_DELAY,
            atol=1e-18,
            msg="tlo_phase != 2*pi*lw*delay",
        )

    def test_phase_zero(self):
        """
        tlo_phase at zero delay is exactly 0.
        """

        self.assertClose(_core.tlo_phase(1e6, 0.0), 0.0, atol=0.0, msg="tlo_phase != 0 at zero delay")


class ShotCalib(Question):
    """
    The calibration line, and the estimator bias a wrong unit makes.
    """

    def test_line_reads(self):
        """
        tlo_calib returns the assumed and actual shot noise, their ratio, and the electronic
        floor in the assumed unit.
        """
        cal, real, ratio, vel, _obs = _core.tlo_calib(1.0, 0.8, 2.0, 0.1)
        self.assertClose(cal, 2.0, atol=1e-15, msg="assumed shot noise != 2.0")
        self.assertClose(real, 1.6, atol=1e-15, msg="actual shot noise != 1.6")
        self.assertClose(ratio, 1.25, atol=1e-15, msg="ratio != 1.25")
        self.assertClose(vel, 0.05, atol=1e-15, msg="floor != 0.05")

    def test_monitor_reads(self):
        """
        The normalised zero-signal variance is 1.05 when honest and 0.85 at ratio 1.25, the
        electronic floor unmoved.
        """
        honest = _core.tlo_calib(1.0, 1.0, 2.0, 0.1)
        attacked = _core.tlo_calib(1.0, 0.8, 2.0, 0.1)
        self.assertClose(honest[4], 1.05, atol=1e-15, msg="honest monitor != 1.05")
        self.assertClose(attacked[4], 0.85, atol=1e-15, msg="attacked monitor != 0.85")
        self.assertClose(attacked[3], honest[3], atol=0.0, msg="floor moved with the oscillator")

    def test_bias_signs(self):
        """
        An assumed unit above the actual one divides the fitted transmittance by the ratio and
        lowers the estimated excess noise by (ratio - 1)/(eta*T).
        """
        t_est, xi_est, _flat, gap, _db = _core.tlo_estimate(JG_T, JG_XI, JG_ETA, 1.2)
        self.assertClose(t_est, JG_T / 1.2, atol=1e-15, msg="t_est != T/1.2")
        self.assertLess(xi_est, JG_XI, msg="xi_est not below xi")
        self.assertClose(gap, 0.2 / (JG_ETA * JG_T), atol=1e-15, msg="gap != 0.2/(eta*T)")

    def test_jouguet_hides(self):
        """
        At Jouguet's T = eta = 0.5 an intercept-resend raising the excess noise to 2.1 SNU
        is erased by attacks.calib_ratio's shot-noise ratio of 1.525.
        """
        real = JG_XI + JG_RESEND
        ratio = attacks.calib_ratio(real, JG_T, JG_ETA)
        self.assertClose(ratio, 1.525, atol=1e-12, msg="calib_ratio != 1.525")

        xi_est = _core.tlo_estimate(JG_T, real, JG_ETA, ratio)[1]
        self.assertClose(xi_est, 0.0, atol=1e-14, msg="xi_est != 0")

    def test_flat_matches(self):
        """
        tlo_estimate's unbiased-slope row equals attacks.calib_xi bit for bit at five ratios.
        """
        for ratio in (0.7, 1.0, 1.2, 1.525, 2.0):
            flat = _core.tlo_estimate(JG_T, JG_XI + JG_RESEND, JG_ETA, ratio)[2]
            want = attacks.calib_xi(JG_XI + JG_RESEND, ratio, JG_T, JG_ETA)
            self.assertClose(flat, want, atol=0.0, msg=f"calib_xi at ratio {ratio}")

    def test_conventions_differ(self):
        """
        The self-consistent estimate is Jouguet Eq. (8)'s unbiased-slope form times the
        shot-noise ratio.
        """
        for ratio in (0.7, 1.2, 2.0):
            _t, xi_est, flat, _gap, _db = _core.tlo_estimate(JG_T, 0.4, JG_ETA, ratio)
            self.assertClose(xi_est, flat * ratio, atol=1e-15, msg="xi_est != flat * ratio")

    def test_loss_signature(self):
        """
        The fitted transmittance carries an apparent extra loss of 10*log10(ratio) dB,
        3.0103 dB at a factor two.
        """
        db = _core.tlo_estimate(JG_T, JG_XI, JG_ETA, 2.0)[4]
        self.assertClose(db, 10.0 * math.log10(2.0), atol=1e-15, msg="apparent loss != 10 log10 ratio")
        self.assertClose(db, 3.0102999566398120, atol=1e-13, msg="apparent loss != 3.0103 dB")

    def test_gap_hidden(self):
        """
        The excess noise a fixed ratio hides rises as the channel darkens, as
        (ratio - 1)/(eta*T).
        """
        gaps = [_core.tlo_estimate(t, 0.05, JG_ETA, 1.001)[3] for t in (0.5, 0.1, 0.01)]
        self.assertMonotone(gaps, rising=True, msg="hidden noise not rising")
        self.assertClose(gaps[2], 0.001 / (JG_ETA * 0.01), atol=1e-12, msg="hidden noise != 0.001/(eta*T)")

    def test_range_moves(self):
        """
        tlo_range is alpha*sqrt(ratio): a half-range fixed in photocurrent moves in
        shot-noise units.
        """

        self.assertClose(
            _core.tlo_range(20.0, 1.525),
            20.0 * math.sqrt(1.525),
            atol=1e-13,
            msg="tlo_range != alpha sqrt(ratio)",
        )

        far = _core.tlo_range(20.0, 100.0)
        self.assertClose(far, 200.0, atol=1e-12, msg="tlo_range != 200")


class LocalLimit(Question):
    """
    As the oscillator is made local, every TLO quantity reduces to the LLO path.
    """

    def test_limit_shot(self):
        """
        tlo_shot at a transparent LO arm returns the characterisation unit and $v_{el}$ to the
        last bit.
        """
        n_bob, t_lo, unit, vel = _core.tlo_shot(LOD_LO, 1.0, 1.0, LOD_LO, LOD_VEL)
        self.assertClose(n_bob, LOD_LO, atol=0.0, msg="n_bob != n_lo")
        self.assertClose(t_lo, 1.0, atol=0.0, msg="t_lo != 1")
        self.assertClose(unit, 1.0, atol=0.0, msg="unit != 1")
        self.assertClose(vel, LOD_VEL, atol=0.0, msg="v_el moved")

    def test_limit_calib(self):
        """
        tlo_calib read at the power actually sampled returns ratio exactly 1 and a monitor
        reading of exactly $1 + v_{el}$.
        """
        _cal, _real, ratio, vel, obs = _core.tlo_calib(3.7, 3.7, 2.0, 0.1)
        self.assertClose(ratio, 1.0, atol=0.0, msg="ratio != 1")
        self.assertClose(obs, 1.0 + vel, atol=0.0, msg="monitor != 1 + v_el")

    def test_limit_estimate(self):
        """
        At ratio 1 tlo_estimate returns T and xi unchanged, both conventions coincident and
        both signatures exactly zero, over nine channels.
        """
        for t in (0.9, 0.302, 1e-3):
            for xi in (0.0, 0.005, 0.2):
                t_est, xi_est, flat, gap, db = _core.tlo_estimate(t, xi, LOD_ETA, 1.0)
                self.assertClose(t_est, t, atol=0.0, msg=f"T at ({t}, {xi})")
                self.assertClose(xi_est, xi, atol=0.0, msg=f"xi at ({t}, {xi})")
                self.assertClose(flat, xi, atol=0.0, msg=f"flat xi at ({t}, {xi})")
                self.assertClose(gap, 0.0, atol=0.0, msg=f"gap at ({t}, {xi})")
                self.assertClose(db, 0.0, atol=0.0, msg=f"loss at ({t}, {xi})")

    def test_limit_rate(self):
        """
        cv_rate on the ratio-1 estimate equals cv_rate on the true channel exactly, in both
        detectors and both trust models.
        """
        for hom in (True, False):
            for trusted in (True, False):
                t_est, xi_est = _core.tlo_estimate(LOD_T, LOD_XI, LOD_ETA, 1.0)[:2]
                got = _core.cv_rate(LOD_VA, t_est, xi_est, LOD_ETA, LOD_VEL, LOD_BETA, hom, trusted)
                want = _core.cv_rate(LOD_VA, LOD_T, LOD_XI, LOD_ETA, LOD_VEL, LOD_BETA, hom, trusted)
                self.assertClose(got[2] - want[2], 0.0, atol=0.0, msg=f"key at ({hom}, {trusted})")

    def test_limit_continuous(self):
        """
        The key-rate deficit falls linearly with the shot-noise ratio's departure from 1, to
        under 1e-11 bit/symbol at a part in $10^{12}$.
        """
        want = _core.cv_rate(LOD_VA, LOD_T, LOD_XI, LOD_ETA, LOD_VEL, LOD_BETA, True, True)[2]
        gaps = []
        for step in (1e-2, 1e-4, 1e-6, 1e-9, 1e-12):
            t_est, xi_est = _core.tlo_estimate(LOD_T, LOD_XI, LOD_ETA, 1.0 + step)[:2]
            key = _core.cv_rate(LOD_VA, t_est, max(xi_est, 0.0), LOD_ETA, LOD_VEL, LOD_BETA, True, True)[2]
            gaps.append(abs(want - key))
        self.assertMonotone(gaps, rising=False, msg="deficit not falling")
        self.assertLess(gaps[4], 1e-11, msg="deficit above 1e-11 bit")
        self.assertClose(gaps[2] / gaps[3], 1e3, atol=1.0, msg="deficit ratio != 1e3")

    def test_limit_epr(self):
        """
        tlo_cov at T = 1, xi = 0, ratio 1 equals GaussianState.epr's covariance to 1e-14.
        """
        got = _core.tlo_cov(LOD_VA, 1.0, 0.0, LOD_ETA, 1.0)
        want = _core.GaussianState.epr(math.acosh(LOD_VA + 1.0) / 2.0)

        self.gridClose(got.cov(), want.cov(), atol=1e-14, msg="tlo_cov != epr covariance")
        self.assertPhysical(
            [got.cov()[4 * i : 4 * i + 4] for i in range(4)],
            msg="tlo_cov not bona fide",
        )

    def test_limit_physical(self):
        """
        tlo_cov satisfies V + i*Omega/2 >= 0 over nine channels and ratios inside the
        estimator's physical range.
        """
        for t in (0.9, 0.302, 1e-3):
            for ratio in (0.6, 1.0, 1.0 + LOD_ETA * t * LOD_XI):
                cov = _core.tlo_cov(LOD_VA, t, LOD_XI, LOD_ETA, ratio).cov()
                self.assertPhysical(
                    [cov[4 * i : 4 * i + 4] for i in range(4)],
                    msg=f"bona fide at ({t}, {ratio})",
                )

    def test_limit_pipeline(self):
        """
        tlo_estimate at ratio 1 agrees with run_symbols' oracle xi, four-seed mean, within
        ARB_TOL.
        """
        want = _core.tlo_estimate(ARB_T, ARB_XI, ARB_ETA, 1.0)[1]
        runs = []
        for seed in (1, 2, 3, 4):
            out = _core.run_symbols(
                ARB_N,
                seed,
                ARB_VA,
                ARB_T,
                ARB_XI,
                ARB_ETA,
                ARB_VEL,
                0.0,
                0.0,
                0.0,
                1e8,
                20.0,
                0.0625,
                1024,
                False,
            )
            runs.append(out.xi_ideal)

        got = sum(runs) / len(runs)
        self.assertClose(got, want, atol=ARB_TOL, msg="run_symbols xi vs tlo_estimate")

    def test_limit_range(self):
        """
        tlo_range at ratio 1 returns the quoted half-range unchanged.
        """

        self.assertClose(_core.tlo_range(20.0, 1.0), 20.0, atol=0.0, msg="tlo_range moved at ratio 1")


class RealTime(Question):
    """
    Real-time shot-noise monitoring, and what it leaves hideable.
    """

    def test_root_falls(self):
        """
        tlo_monitor's resolution is sqrt(2/n), falling tenfold from n = 1e4 to 1e6.
        """
        one = _core.tlo_monitor(10**4 + 1, 1.0, 1.0, 0.0, JG_T, JG_ETA)[0]
        two = _core.tlo_monitor(10**6 + 1, 1.0, 1.0, 0.0, JG_T, JG_ETA)[0]
        self.assertClose(one, math.sqrt(2.0 / 1e4), atol=1e-15, msg="resolution != sqrt(2/1e4)")
        self.assertClose(one / two, 10.0, atol=1e-12, msg="resolution ratio != 10")

    def test_detects_jouguet(self):
        """
        tlo_monitor flags Jouguet's ratio 1.525 at $10^6$ vacuum samples and not at 100.
        """
        ratio = attacks.calib_ratio(JG_XI + JG_RESEND, JG_T, JG_ETA)
        many = _core.tlo_monitor(10**6, 3.0, ratio, 0.0, JG_T, JG_ETA)
        few = _core.tlo_monitor(100, 3.0, ratio, 0.0, JG_T, JG_ETA)
        self.assertTrue(many[2], msg="1e6 samples missed it")
        self.assertFalse(few[2], msg="100 samples flagged it")

    def test_residual_falls(self):
        """
        The excess noise still hideable beneath the detection threshold falls as 1/sqrt(n),
        tenfold per four decades of samples.
        """
        left = [_core.tlo_monitor(n, 1.0, 1.0, 0.0, JG_T, JG_ETA)[3] for n in (10**4 + 1, 10**6 + 1, 10**8 + 1)]
        self.assertMonotone(left, rising=False, msg="residual not falling")
        self.assertClose(left[1] / left[2], 10.0, atol=0.05, msg="residual ratio != 10")

    def test_honest_quiet(self):
        """
        At ratio 1 the monitored gap is exactly 0 and nothing is flagged at any n.
        """
        for n in (10, 10**3, 10**7):
            sigma, gap, hit, _left = _core.tlo_monitor(n, 1.0, 1.0, 0.15, JG_T, JG_ETA)
            self.assertClose(gap, 0.0, atol=0.0, msg=f"gap != 0 at n = {n}")
            self.assertFalse(hit, msg=f"flagged at n = {n}")


class TloPipeline(Question):
    """
    The transmitted-oscillator branch of the symbol pipeline.
    """

    def test_delay_sets_the_variance(self):
        """
        run_tlo's v_err equals tlo_phase's Wiener increment across the multiplexing delay to
        TLO_TOL, over a two-decade linewidth ladder.
        """
        for lw in TLO_LADDER:
            out = tlo_run(lw, zhang_delay())
            want = _core.tlo_phase(lw, zhang_delay())
            self.assertClose(out.v_err / want, 1.0, atol=TLO_TOL, msg=f"v_err at {lw} Hz")

    def test_charges_both_halves(self):
        """
        The branch charges BOTH halves: the recovered channel is the oracle one through
        budget.infer at the measured v_err, the recovered excess noise the oracle one plus
        budget.phase at it.
        """
        for lw in TLO_LADDER:
            out = tlo_run(lw, zhang_delay())
            span = budget.infer(oracle_t(out), out.v_err)
            row = budget.phase(ARB_VA, out.v_err, out.xi_ideal)
            self.assertClose(out.t_chan / span, 1.0, atol=TLO_TOL, msg=f"transmittance half at {lw} Hz")
            self.assertClose((out.xi_hat - out.xi_ideal) / row, 1.0, atol=TLO_ROW, msg=f"noise half at {lw} Hz")

    def test_conserves_the_plane(self):
        """
        The two halves conserve T*(V_A + xi) across the rotation to two parts in ten
        thousand over the ladder.
        """
        for lw in TLO_LADDER:
            out = tlo_run(lw, zhang_delay())
            got = out.t_chan * (ARB_VA + out.xi_hat)
            want = oracle_t(out) * (ARB_VA + out.xi_ideal)
            self.assertClose(got / want, 1.0, atol=TLO_PLANE, msg=f"Bob-plane signal at {lw} Hz")

    def test_oracle_stands_still(self):
        """
        The oracle excess noise moves under 1e-4 SNU while the charged phase row rises across
        two decades.
        """
        runs = [tlo_run(lw, zhang_delay()) for lw in TLO_LADDER]
        oracle = [out.xi_ideal for out in runs]
        rows = [out.xi_hat - out.xi_ideal for out in runs]
        self.assertLess(max(oracle) - min(oracle), 1e-4, msg="oracle xi spread above 1e-4")
        self.assertMonotone(rows, msg="charged row not rising")
        self.assertGreater(rows[-1] / rows[0], 100.0, msg="charged row spans under 100x")

    def test_zero_delay_is_free(self):
        """
        At zero multiplexing delay run_tlo's v_err and both branch differences are exactly 0,
        where run_symbols leaves a positive residual.
        """
        out = tlo_run(TLO_LADDER[-1], 0.0)
        self.assertClose(out.v_err, 0.0, atol=0.0, msg="v_err != 0 at zero delay")
        self.assertClose(out.xi_hat - out.xi_ideal, 0.0, atol=0.0, msg="xi branches differ at zero delay")
        self.assertClose(out.t_hat - out.t_ideal, 0.0, atol=0.0, msg="slopes differ at zero delay")

        llo = _core.run_symbols(
            ARB_N, 1, ARB_VA, ARB_T, ARB_XI, ARB_ETA, ARB_VEL, 0.0, 0.0, 0.0, ZH_CLOCK, 20.0, 0.0625, 4096, False
        )
        self.assertGreater(llo.v_err, 0.0, msg="run_symbols v_err not positive")

    def test_branch_is_the_llo_oracle(self):
        """
        At zero delay run_tlo and run_symbols return bit-identical oracle slope and excess
        noise.
        """
        for seed in (1, 7):
            got = _core.run_tlo(ARB_N, seed, ARB_VA, ARB_T, ARB_XI, ARB_ETA, ARB_VEL, 0.0, 0.0, ZH_CLOCK, False)
            want = _core.run_symbols(
                ARB_N, seed, ARB_VA, ARB_T, ARB_XI, ARB_ETA, ARB_VEL, 0.0, 0.0, 0.0, ZH_CLOCK, 20.0, 0.0625, 4096, False
            )
            self.assertEqual(got.t_ideal, want.t_ideal, msg=f"oracle slope at seed {seed}")
            self.assertEqual(got.xi_ideal, want.xi_ideal, msg=f"oracle excess noise at seed {seed}")

    def test_zhang_round_trip(self):
        """
        At Zhang's 100 Hz linewidth and their implied delay the branch measures that phase
        noise back to TLO_TOL. The delay is FITTED, so this closes the engine against the
        pipeline rather than reproducing a measurement.
        """
        out = tlo_run(ZH_WIDTH, zhang_delay())
        self.assertClose(out.v_err / zhang_verr(), 1.0, atol=TLO_TOL, msg="Zhang's residual phase noise")

    def test_no_pilot_no_beat(self):
        """
        run_tlo returns a NaN pilot SNR -- a zero would read as a tone sent and lost -- a CFO
        of exactly 0, and the full symbol count.
        """
        out = tlo_run(TLO_LADDER[1], zhang_delay())
        self.assertTrue(math.isnan(out.pilot_snr), msg="pilot_snr is not NaN")
        self.assertClose(out.cfo, 0.0, atol=0.0, msg="cfo != 0")
        self.assertEqual(out.n_used, ARB_N, msg="n_used != ARB_N")


class TloAnchors(Question):
    """
    Lodewyck 2007 and Zhang 2020, read through the engine.
    """

    def test_lodewyck_delay(self):
        """
        The linewidth whose phase row saturates Lodewyck's 0.005 SNU across their 400 ns
        delay is 107.52 Hz.
        """
        want = math.log1p(LOD_XI / LOD_VA)
        width = want / (math.tau * LOD_DELAY)
        self.assertClose(width, 107.52, atol=0.01, msg="saturating linewidth != 107.52 Hz")
        self.assertClose(
            budget.phase(LOD_VA, _core.tlo_phase(width, LOD_DELAY)),
            LOD_XI,
            atol=1e-15,
            msg="phase row != LOD_XI",
        )

    def test_zhang_delay(self):
        """
        Zhang's residual phase noise at 100 Hz implies a 120.96 ns signal-to-LO delay, inside
        their 200 ns symbol period.
        """
        delay = zhang_delay()
        self.assertClose(delay, 1.2096e-7, atol=1e-11, msg="implied delay != 1.2096e-7 s")
        self.assertLess(delay, 1.0 / ZH_CLOCK, msg="delay past the symbol period")
        self.assertClose(
            _core.tlo_phase(ZH_WIDTH, delay),
            zhang_verr(),
            atol=1e-18,
            msg="tlo_phase != zhang_verr",
        )

    def test_zhang_share(self):
        """
        Zhang's phase row is 7.18% of their measured excess noise at 202.81 km, leaving
        0.00752 SNU unexplained.
        """
        share = budget.phase(ZH_VA, zhang_verr()) / ZH_XI
        self.assertClose(share, 0.0718, atol=5e-4, msg="phase share != 0.0718")
        self.assertClose(zhang_deficit(), 0.00752, atol=1e-5, msg="deficit != 0.00752 SNU")

    def test_zhang_ppm(self):
        """
        That deficit is 2.623 parts per million of shot-noise unit, and tlo_estimate at that
        ratio opens exactly it.
        """
        ratio = 1.0 - zhang_deficit() * ZH_ETA * ZH_T
        self.assertClose((1.0 - ratio) * 1e6, 2.623, atol=1e-3, msg="ppm != 2.623")

        gap = _core.tlo_estimate(ZH_T, ZH_XI, ZH_ETA, ratio)[3]
        self.assertClose(gap, -zhang_deficit(), atol=1e-12, msg="gap != -deficit")

    def test_zhang_samples(self):
        """
        Certifying that part in 2.6e6 to one sigma takes 3.858e11 vacuum samples, 21.44 hours
        at Zhang's 5 MHz clock.
        """
        need = 1.0 + 2.0 * ((1.0 + ZH_VEL) / (zhang_deficit() * ZH_ETA * ZH_T)) ** 2
        self.assertClose(need / 1e11, 3.858, atol=1e-3, msg="samples != 3.858e11")
        self.assertClose(need / ZH_CLOCK / 3600.0, 21.44, atol=0.01, msg="hours != 21.44")

        left = _core.tlo_monitor(int(need), 1.0, 1.0, ZH_VEL, ZH_T, ZH_ETA)[3]
        self.assertClose(left / zhang_deficit(), 1.0, atol=1e-5, msg="residual != deficit")

    def test_anchor_invariant(self):
        """
        The LLO limit of tlo_estimate reproduces Lodewyck's I_AB 1.0436, chi_BE 0.9020 and
        key 0.0352.
        """
        t_est, xi_est = _core.tlo_estimate(LOD_T, LOD_XI, LOD_ETA, 1.0)[:2]
        i_ab, chi, key = _core.cv_rate(LOD_VA, t_est, xi_est, LOD_ETA, LOD_VEL, LOD_BETA, True, True)
        self.assertClose(i_ab, LOD_IAB, atol=1e-4, msg="I_AB != 1.0436")
        self.assertClose(chi, LOD_CHI, atol=1e-4, msg="chi_BE != 0.9020")
        self.assertClose(key, LOD_KEY, atol=1e-4, msg="key != 0.0352")


class TloGuards(Guarded):
    """
    Argument domains of the transmitted-LO engine, and its refusals.
    """

    def test_shot_slots(self):
        """
        tlo_shot refuses a non-positive LO, a transmittance outside (0, 1] on
        either arm, and a negative electronic noise.
        """
        ok = (1e9, 0.5, 0.5, 1e9, 0.04)
        self.assertSlots(
            _core.tlo_shot,
            ok,
            (
                (0, "n_lo", (0.0, -1.0, math.nan)),
                (1, "t", (0.0, 1.5, math.inf)),
                (2, "mux", (0.0, 1.5)),
                (3, "n_ref", (0.0, -1.0)),
                (4, "vel_ref", (-1e-9, math.nan)),
            ),
            msg="tlo_shot",
        )

    def test_calib_slots(self):
        """
        tlo_calib refuses a non-positive power on either read, a non-positive
        calibration slope and a negative electronic floor.
        """
        ok = (1.0, 1.0, 2.0, 0.1)
        self.assertSlots(
            _core.tlo_calib,
            ok,
            (
                (0, "p_mon", (0.0, -1.0)),
                (1, "p_eff", (0.0, math.nan)),
                (2, "slope", (0.0, -2.0)),
                (3, "floor", (-1e-9, math.inf)),
            ),
            msg="tlo_calib",
        )

    def test_estimate_slots(self):
        """
        tlo_estimate refuses a transmittance or efficiency outside (0, 1], a
        negative excess noise and a non-positive shot-noise ratio.
        """
        ok = (0.5, 0.1, 0.5, 1.0)
        self.assertSlots(
            _core.tlo_estimate,
            ok,
            (
                (0, "t", (0.0, 1.5)),
                (1, "xi", (-1e-9, math.nan)),
                (2, "eta", (0.0, 2.0)),
                (3, "ratio", (0.0, -1.0, math.inf)),
            ),
            msg="tlo_estimate",
        )

    def test_range_slots(self):
        """
        tlo_range, tlo_phase and tlo_leak refuse their own domains: a
        non-positive half-range or unit ratio, a negative linewidth or delay, and
        a leak fraction outside [0, 1].
        """

        self.assertSlots(
            _core.tlo_range,
            (20.0, 1.0),
            ((0, "alpha", (0.0, -1.0)), (1, "ratio", (0.0, math.nan))),
            msg="tlo_range",
        )
        self.assertSlots(
            _core.tlo_phase,
            (1e3, 1e-7),
            ((0, "linewidth", (-1.0, math.inf)), (1, "delay", (-1e-12, math.nan))),
            msg="tlo_phase",
        )
        self.assertSlots(
            _core.tlo_leak,
            (1e9, 1e-6),
            ((0, "n_bob", (0.0, -1.0)), (1, "leak", (-1e-9, 1.5))),
            msg="tlo_leak",
        )

    def test_pipeline_slots(self):
        """
        run_tlo refuses a negative modulation, excess noise, electronic noise,
        linewidth, delay or jitter, a transmittance or efficiency outside
        (0, 1], and a non-positive symbol rate.
        """
        ok = (1 << 13, 1, 5.0, 0.5, 0.02, 0.6, 0.05, 1e4, 1e-8, 5e6, False)
        self.assertSlots(
            _core.run_tlo,
            ok,
            (
                (2, "va", (-1e-9, math.nan)),
                (3, "t", (0.0, 1.5)),
                (4, "xi", (-1e-9, math.inf)),
                (5, "eta", (0.0, 1.5)),
                (6, "vel", (-1e-9, math.nan)),
                (7, "linewidth", (-1.0, math.nan)),
                (8, "delay", (-1e-12, math.nan)),
                (9, "symbol_rate", (0.0, -1.0)),
            ),
            msg="run_tlo",
        )

    def test_delay_past_the_symbol(self):
        """
        run_tlo refuses a multiplexing delay longer than one symbol period, where consecutive
        Wiener increments overlap.
        """

        self.assertFails(
            ValueError,
            "only while the delay fits inside one symbol period",
            _core.run_tlo,
            1 << 13,
            1,
            5.0,
            0.5,
            0.02,
            0.6,
            0.05,
            1e4,
            2.0 / ZH_CLOCK,
            ZH_CLOCK,
            False,
            msg="run_tlo past one symbol period",
        )

    def test_pipeline_needs_a_group(self):
        """
        run_tlo refuses a run shorter than the pipeline's reduction group.
        """

        self.assertFails(
            ValueError,
            "at least one",
            _core.run_tlo,
            4095,
            1,
            5.0,
            0.5,
            0.02,
            0.6,
            0.05,
            1e4,
            1e-8,
            5e6,
            False,
            msg="run_tlo under one group",
        )

    def test_cov_refuses(self):
        """
        tlo_cov refuses a ratio past the zero crossing, while tlo_estimate still hands the
        negative excess noise back as the attack signature.
        """
        ratio = 1.0 + LOD_ETA * LOD_T * LOD_XI
        self.assertFails(
            ValueError,
            "no thermal-loss channel is quieter than the vacuum",
            _core.tlo_cov,
            LOD_VA,
            LOD_T,
            LOD_XI,
            LOD_ETA,
            2.0 * ratio,
            msg="tlo_cov past the crossing",
        )

        still = _core.tlo_estimate(LOD_T, LOD_XI, LOD_ETA, 2.0 * ratio)[1]
        self.assertLess(still, 0.0, msg="tlo_estimate xi not negative")

    def test_monitor_refuses(self):
        """
        tlo_monitor refuses a sample count whose resolution reaches the whole shot-noise
        unit, and refuses n = 1.
        """

        self.assertFails(
            ValueError,
            "certifies nothing",
            _core.tlo_monitor,
            4,
            3.0,
            1.0,
            0.15,
            JG_T,
            JG_ETA,
            msg="tlo_monitor with too few samples",
        )
        self.assertFails(
            ValueError,
            "n must be >= 2",
            _core.tlo_monitor,
            1,
            1.0,
            1.0,
            0.0,
            JG_T,
            JG_ETA,
            msg="tlo_monitor with one sample",
        )


if __name__ == "__main__":
    rc = Exam(
        "Oscillator",
        "The transmitted oscillator's propagation and the shot-noise unit it sets",
        "tlo_arm.md",
    ).run(load(LoArm))
    rc |= Exam(
        "ShotCalibration",
        "The calibration line, and the estimator bias a wrong shot-noise unit makes",
        "tlo_calib.md",
    ).run(load(ShotCalib))
    rc |= Exam(
        "LocalLimit",
        "As the oscillator is made local, every TLO quantity reduces to the LLO path",
        "tlo_limit.md",
    ).run(load(LocalLimit))
    rc |= Exam(
        "RealTime",
        "Real-time shot-noise monitoring, and what it still leaves hideable",
        "tlo_monitor.md",
    ).run(load(RealTime))
    rc |= Exam(
        "TloPipeline",
        "The transmitted-oscillator branch of the symbol pipeline, and the two halves it derives",
        "tlo_pipeline.md",
    ).run(load(TloPipeline))
    rc |= Exam(
        "TloAnchors",
        "Lodewyck 2007 and Zhang 2020, the two transmitted-LO experiments qkd pins",
        "tlo_anchors.md",
    ).run(load(TloAnchors))
    rc |= Exam(
        "TloGuards",
        "Argument domains of the transmitted-LO engine, and its two refusals",
        "tlo_guards.md",
    ).run(load(TloGuards))
    sys.exit(rc)
