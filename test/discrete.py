import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, load

import qkd as q
from kit.anchors import ALPHA, D_COW, E_ALIGN, ETA_BOB, E_DET, E_VAC
from kit.checks import Guarded
from kit.anchors import ETA_COW as ETA_DET
from kit.anchors import F_COW, F_REC, P_DARK, Y_DARK
from kit.forms import background, band
from kit.links import basis as basis_link
from qkd import _core

# Gao et al., Opt. Express 30, 23783 (2022) Sec. IV; T_SPLIT, F_DEC, MU_REF and E_PHASE are this exam's, not theirs.
T_SPLIT = 0.9
F_DEC = 0.1
MU_REF = 0.5
E_PHASE = 0.20
SIX_SIFT = _core.sixstate_sift(1.0 / 3.0)[0]

# `counted`'s plan, positional so a guard can substitute one trailing slot.
PLAN = (1000, 0, (0.5, 0.1, 0.0), (0.8, 0.15, 0.05), 0.5, 1.0, 0.0, 0.02, 0.5, 0)


def gys_gains(dist, mu):
    """
    Infinite-decoy gains and error rates for the GYS channel at ``dist`` km.
    """
    eta = ETA_BOB * 10 ** (-ALPHA * dist / 10.0)
    y1 = Y_DARK + eta - Y_DARK * eta
    q1 = y1 * mu * math.exp(-mu)
    e1 = (E_VAC * Y_DARK + E_DET * eta) / y1
    gain = Y_DARK + 1.0 - math.exp(-eta * mu)
    qber = (E_VAC * Y_DARK + E_DET * (1.0 - math.exp(-eta * mu))) / gain

    return gain, qber, q1, e1


def gys_rate(dist, mu=0.48, sift=0.5):
    gain, qber, q1, e1 = gys_gains(dist, mu)

    return _core.bb84_rate(sift, gain, qber, q1, e1, F_REC)


def gys_best(dist, sift=0.5):
    rates = [(gys_rate(dist, i * 2e-3, sift), i * 2e-3) for i in range(1, 501)]

    return max(rates)


def cow_point(dist):
    """
    COW rate at ``dist`` km with the intensity forced down linearly in loss.
    """
    chan = 10 ** (-0.02 * dist)
    trans = ETA_DET * T_SPLIT * chan
    gain = _core.cow_ceiling(F_DEC, trans, MU_REF * chan)

    return _core.cow_rate(gain, E_ALIGN, E_PHASE, F_COW)


def basis_rates(out, arm=0):
    """
    ``(e_x, e_y, e_z)`` at one intensity arm from a run's own per-basis counts, key basis LAST as
    sixstate_bell reads it.
    """
    return (
        out.x_errors[arm] / out.x_sifted[arm],
        out.y_errors[arm] / out.y_sifted[arm],
        out.key_errors[arm] / out.key_sifted[arm],
    )


def counted(pulses, seed, miss, probs=(0.8, 0.15, 0.05), sift=0.5, chunk=0, **kw):
    """
    One counted run on a lossless-analyser, dark-free receiver at T*eta = 0.5.
    """

    return _core.run_basis(pulses, seed, (0.5, 0.1, 0.0), probs, 0.5, 1.0, 0.0, miss, sift, chunk, **kw)


def six_run(pulses, seed, miss, chunk=0):
    """
    One three-basis run at the uniform bias on that receiver, ``miss`` being ``(e_x, e_y, e_z)``.
    """

    return counted(
        pulses,
        seed,
        miss[2],
        (1.0, 0.0, 0.0),
        SIX_SIFT,
        chunk,
        bases=3,
        misalign_x=miss[0],
        misalign_y=miss[1],
    )


def log_slope(fn, near, far, alpha):
    ratio = fn(near) / fn(far)

    return math.log(ratio) / math.log(10 ** (alpha * (far - near) / 10.0))


def nodecoy(dist, mu):
    """
    GLLP rate at ``dist`` km with no decoys: Q1 >= Q_mu - (1 - e^-mu (1 + mu)), all error on the
    singles.
    """
    eta = _core.decoy_eta(ALPHA, dist, ETA_BOB)
    gain, qber = _core.decoy_gain(mu, eta, Y_DARK, E_DET)
    multi = 1.0 - math.exp(-mu) * (1.0 + mu)
    q1 = max(0.0, gain - multi)
    if q1 <= 0.0:
        return 0.0

    return _core.bb84_rate(0.5, gain, qber, q1, min(0.5, qber * gain / q1), F_REC)


def nodecoy_best(dist):
    return max(nodecoy(dist, i * 1e-3) for i in range(1, 1001))


def dark_link(dark, alice=None):
    """
    Basis-keyed q.Link at T*eta = 0.5; dark = 1e-2, four orders above InGaAs, makes the 2x between
    the two dark-count planes visible.
    """

    return q.Link(
        modulation=q.BasisKeying(decoy=q.Decoy(intensities=(0.5, 0.1, 0.0), probs=(0.9, 0.07, 0.03))),
        channel=q.Channel(T=0.5),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=dark),
            receiver=q.BasisAnalyser(misalign=E_DET),
        ),
        security=q.SplittingAttack(f=F_REC),
        alice=alice,
    )


def polar(drift=0.0, **kw):
    return q.PolarisationKeying(
        decoy=q.Decoy(intensities=(0.48, 0.1, 0.0)),
        frame=q.ReferenceFrame(drift=drift, **kw),
    )


def cow_link(dist=None, ext=None, frac=F_DEC, miss=E_ALIGN):
    """
    Intensity-keyed q.Link at ``dist`` km, or over a transparent span on the closed forms' plane,
    with an optional extinction ratio in dB.
    """

    return q.Link(
        modulation=q.IntensityKeying(mu=MU_REF, decoy_frac=frac, extinction=ext),
        channel=q.Fiber(length=dist, alpha=0.2) if dist else q.Channel(T=1.0),
        bob=q.Bob(
            detector=q.ClickDetector(eta=ETA_DET, dark=D_COW),
            receiver=q.CoherenceMonitor(split=1.0 - T_SPLIT, misalign=miss),
        ),
        security=q.PhaseBound(e_phase=E_PHASE, f=F_COW),
    )


class Bb84Wcp(Guarded):
    """
    Decoy BB84-WCP under GLLP as Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005), Eq. (1), with (Q1, e1) at
    the infinite-decoy limit.
    """

    def test_gys_secure_distance(self):
        """
        Ma, Qi, Zhao & Lo 2005's "over 140 km of Telecom fibers" on the GYS parameters: the rate is
        positive at 140 km and zero at 145 km.
        """
        near = gys_best(140.0)[0]
        far = gys_best(145.0)[0]

        self.assertGreater(near, 0.0, msg=f"rate at 140 km: {near}")
        self.assertEqual(far, 0.0, msg=f"rate at 145 km: {far}")

    def test_recon_floor(self):
        """
        bb84_rate and cow_rate refuse an inefficiency f_ec < 1, which beats Shannon, and the rate
        falls from f_ec = 1 to 1.22.
        """
        for bad in (0.0, 0.5, 0.95, 0.999999):
            self.assertBad(
                "f_ec must be >= 1",
                _core.bb84_rate,
                (0.5, 4e-3, 0.01, 1.6e-3, 0.01, bad),
                msg=f"bb84_rate at f_ec = {bad}",
            )
            self.assertBad(
                "f_ec must be >= 1",
                _core.cow_rate,
                (0.01, 0.02, 0.2, bad),
                msg=f"cow_rate at f_ec = {bad}",
            )

        shannon = _core.bb84_rate(0.5, 4e-3, 0.01, 1.6e-3, 0.01, 1.0)

        self.assertGreater(
            shannon,
            _core.bb84_rate(0.5, 4e-3, 0.01, 1.6e-3, 0.01, 1.22),
            msg=f"rate at f_ec 1: {shannon}",
        )

    def test_gys_optimal_mu(self):
        """
        The 25 km optimum intensity is within 0.02 of Ma et al.'s mu_optimal = 0.48 at f(e) = 1.22.
        """
        best = gys_best(25.0)[1]

        self.assertClose(best, 0.48, atol=0.02, msg=f"optimal mu {best}")

    def test_gys_rate_scale(self):
        """
        At mu = 0.48 the GYS rate is 2.55e-3 bit/pulse at 0 km and 1.72e-5 at 100 km.
        """
        self.assertClose(gys_rate(0.0), 2.55e-3, atol=1e-5, msg="0 km rate")
        self.assertClose(gys_rate(100.0), 1.72e-5, atol=1e-7, msg="100 km rate")

    def test_gys_linear_scaling(self):
        """
        The GYS decoy BB84 rate's log-log slope against eta over 25-75 km is 1.0 within 0.05.
        """
        slope = log_slope(gys_rate, 25.0, 75.0, ALPHA)

        self.assertClose(slope, 1.0, atol=0.05, msg=f"slope {slope}")

    def test_zero_error_closed_form(self):
        """
        At zero error bb84_rate is q*Q1 exactly.
        """
        got = _core.bb84_rate(0.5, 4e-3, 0.0, 1.6e-3, 0.0, 1.22)

        self.assertClose(got, 0.5 * 1.6e-3, atol=1e-15, msg="e = 0 closed form")

    def test_rate_falls_distance(self):
        """
        The rate falls monotonically with fibre length and is exactly 0.0 from 150 to 400 km.
        """
        rates = [gys_rate(d) for d in (0.0, 25.0, 50.0, 75.0, 100.0, 125.0)]

        self.assertMonotone(rates, rising=False, msg=f"rates {rates}")

        for dist in (150.0, 175.0, 200.0, 400.0):
            self.assertEqual(gys_rate(dist), 0.0, msg=f"rate at {dist} km")

    def test_bb84_guards(self):
        """
        bb84_rate raises a ValueError naming the slot for an error rate outside [0, 1/2], a gain
        outside [0, 1], q_sift outside (0, 1], NaN anywhere, and q1 > q_mu.
        """
        base = (0.5, 4e-3, 0.01, 1.6e-3, 0.01, 1.22)
        slots = ("q_sift", "q_mu", "e_mu", "q1", "e1", "f_ec")
        rows = ((0.0, -4e-3, 0.6, -1.6e-3, -0.01, -1.0), (float("nan"),) * 6)

        self.assertGreater(_core.bb84_rate(*base), 0.0, msg="baseline rate")

        for row in rows:
            for i, name in enumerate(slots):
                args = list(base)
                args[i] = row[i]
                self.assertFails(ValueError, name, _core.bb84_rate, *args, msg=f"{name} unguarded")

        self.assertBad(
            "q1 must not exceed q_mu",
            _core.bb84_rate,
            (0.5, 1e-3, 0.01, 2e-3, 0.01, 1.22),
            msg="q1 > q_mu unguarded",
        )

    def test_gys_nondecoy_deficit(self):
        """
        GLLP with no decoys reaches 40 km optimised and no key at GYS's mu = 0.1, a different
        theorem from the 49.0 km Brassard-Lutkenhaus-Mor-Sanders condition of test_gys_pns_reading.
        """
        cutoff = max(d for d in range(0, 60) if nodecoy_best(float(d)) > 0.0)

        self.assertEqual(cutoff, 40, msg=f"no-decoy cutoff {cutoff} km")

        for dist in (0.0, 10.0, 25.0, 50.0):
            self.assertEqual(
                nodecoy(dist, 0.1),
                0.0,
                msg=f"no-decoy rate at mu 0.1, {dist} km",
            )

        self.assertGreater(
            gys_best(40.0)[0],
            nodecoy_best(40.0),
            msg="decoy vs no-decoy at 40 km",
        )


class CowRevised(Guarded):
    """
    The linear-in-eta bounds (Stucki et al. 2005; Branciard, Gisin & Scarani, New J. Phys. 10,
    013031 (2008)) fall to the zero-error attack of Gonzalez-Payo, Trenyi, Wang & Curty, PRL 125,
    260510 (2020), capping scaling at eta^2; qkd takes the phase-error form of Gao et al., Opt.
    Express 30, 23783 (2022), and no visibility-based rate.
    """

    def test_cow_quadratic_scaling(self):
        """
        With the intensity forced down linearly in loss the COW rate scales as eta^2 within 0.05 and
        at 100 km, Gao et al. 2022's "within 100 kilometers", is positive but under 1e-5 bit/pulse
        and three decades below 0 km.
        """
        near = cow_point(0.0)
        far = cow_point(100.0)

        self.assertClose(
            log_slope(cow_point, 20.0, 60.0, 0.2),
            2.0,
            atol=0.05,
            msg="COW log-log slope",
        )
        self.assertGreater(far, 0.0, msg=f"rate at 100 km: {far}")
        self.assertLess(far, 1e-5, msg=f"rate at 100 km: {far}")
        self.assertLess(far, near / 1000.0, msg=f"rate at 100 km {far} vs 0 km {near}")

    def test_cow_loses_to_bb84(self):
        """
        COW loses ground monotonically to GYS decoy BB84 over 0-80 km and stays below it to 120 km
        despite its 0.2 against 0.21 dB/km fibre.
        """
        ratios = [cow_point(d) / gys_rate(d) for d in (0.0, 20.0, 40.0, 60.0, 80.0)]

        self.assertMonotone(ratios, rising=False, msg=f"COW/BB84 {ratios}")

        for dist in (80.0, 100.0, 120.0):
            self.assertLess(
                cow_point(dist),
                gys_rate(dist),
                msg=f"COW vs BB84 at {dist} km",
            )

    def test_cow_phase_pinned(self):
        """
        Unverified exam-side pin: across the 0.17-0.24 e_phase band read off Gao et al. 2022 Fig. 3,
        whose Eq. (7) needs unmodelled monitoring-line decoy gains, COW stays below BB84 at 100 km.
        """
        rates = [cow_point(100.0)]
        for phase in (0.17, 0.20, 0.24):
            chan = 10 ** (-2.0)
            gain = _core.cow_ceiling(F_DEC, ETA_DET * T_SPLIT * chan, MU_REF * chan)
            rate = _core.cow_rate(gain, E_ALIGN, phase, F_COW)
            rates.append(rate)
            self.assertLess(rate, gys_rate(100.0), msg=f"rate at e_phase {phase}: {rate}")

        self.assertFinite(rates, msg=f"rates {rates}")

    def test_cow_falls_phase_error(self):
        """
        cow_rate falls monotonically in e_ph and is exactly zero at e_ph = 1/2.
        """
        rates = [_core.cow_rate(0.01, 0.02, e, 1.1) for e in (0.0, 0.05, 0.1, 0.2, 0.3)]

        self.assertMonotone(rates, rising=False, msg=f"rates {rates}")
        self.assertEqual(_core.cow_rate(0.01, 0.02, 0.5, 1.1), 0.0, msg="rate at e_ph 1/2")

    def test_visibility_is_not_security(self):
        """
        Visibility is an observable, not a bound: cow_visibility reads 1 one-sided and 0 balanced,
        and V = 1 at e_ph = 1/2 gives no key.
        """
        perfect = _core.cow_visibility(1.0, 0.0)
        blind = _core.cow_visibility(0.5, 0.5)

        self.assertClose(perfect, 1.0, atol=1e-15, msg="V one-sided")
        self.assertClose(blind, 0.0, atol=1e-15, msg="V balanced")
        self.assertEqual(_core.cow_rate(0.01, 0.0, 0.5, 1.1), 0.0, msg="rate at e_bit 0, e_ph 1/2")

    def test_cow_guards(self):
        """
        cow_rate, cow_ceiling and cow_visibility refuse out-of-domain gains and error rates, a
        non-positive intensity or transmittance, and a dark monitoring line, naming the argument.
        """
        cases = (
            (_core.cow_rate, (0.01, 0.02, 0.6, 1.1), "e_phase", "e_ph > 1/2"),
            (_core.cow_rate, (0.01, -0.01, 0.2, 1.1), "e_bit", "e_bit < 0"),
            (_core.cow_rate, (-0.01, 0.02, 0.2, 1.1), "q_z", "negative gain"),
            (_core.cow_ceiling, (0.1, 0.5, 0.0), "mu", "mu = 0"),
            (_core.cow_ceiling, (0.1, 0.0, 0.5), "t", "t = 0"),
            (_core.cow_visibility, (0.0, 0.0), "never clicked", "dark monitoring line"),
        )
        for fn, args, needle, why in cases:
            self.assertBad(needle, fn, args, msg=why)

    def test_cow_finite_refused(self):
        """
        cow_finite checks n_total, then raises NotImplementedError naming the uncovered phase error,
        nothing to estimate it from, and the vacuum-decoy route's sdp_* lengths.
        """
        for needle in (
            "THE EPSILON WOULD COVER EVERYTHING BUT THE THING IT IS FOR",
            "THERE IS NOTHING TO ESTIMATE",
            "THE ROUTE IS A PROTOCOL CHANGE, NOT A POST-PROCESSING ONE",
        ):
            self.assertFails(
                NotImplementedError,
                needle,
                _core.cow_finite,
                1e9,
                msg=needle,
            )

        self.assertBad("n_total", _core.cow_finite, (0.0,), msg="n_total = 0")
        self.assertFails(
            NotImplementedError,
            "sdp_interval, sdp_sample and sdp_length",
            _core.cow_finite,
            1e9,
            msg="refusal lacks the sdp_* names",
        )


class CowExtinction(Guarded):
    """
    A 20-30 dB modulator leaves 1e-2 to 1e-3 of the signal in the vacuum slot, raising the bit error
    rate and escaping the coherence check while the fringe never moves.
    """

    def test_extinction_default(self):
        """
        An undescribed modulator has residual 0.0, cow_data returns its inputs bit for bit, and the
        link rate equals the closed form.
        """
        mod = q.IntensityKeying(mu=MU_REF, decoy_frac=F_DEC)
        gain, e_bit = _core.decoy_gain(MU_REF, 0.72, background(D_COW), E_ALIGN)
        again, eagain = _core.cow_data(gain, e_bit, 0.0, 0.72)

        self.assertEqual(mod.residual, 0.0, msg=f"residual {mod.residual}")
        self.assertEqual(again, gain, msg=f"gain {again} vs {gain}")
        self.assertEqual(eagain, e_bit, msg=f"e_bit {eagain} vs {e_bit}")

        plain = cow_link(20.0).run()
        data = ETA_DET * T_SPLIT * 10 ** (-0.4)
        direct = _core.decoy_gain(MU_REF, data, background(D_COW), E_ALIGN)
        pinned = _core.cow_rate((1.0 - F_DEC) * direct[0], direct[1], E_PHASE, F_COW)

        self.assertEqual(plain.key_rate, pinned, msg=f"rate {plain.key_rate} vs {pinned}")

    def test_no_cancellation(self):
        """
        Over 40 to 15 dB extinction the data-line gain rises while the rate falls, and at 20 dB the
        rate is 54% of perfect on a gain 0.83% higher.
        """
        plain = cow_link().run()
        runs = [cow_link(ext=db).run() for db in (40.0, 30.0, 25.0, 20.0, 15.0)]
        # explain's gain is the data-line gain cow_data returns; p_click carries (1 - decoy_frac).
        gains = [r.explain["gain"]["value"] for r in runs]

        self.assertMonotone(gains, rising=True, msg=f"gains {gains}")
        self.assertMonotone(
            [r.key_rate for r in runs],
            rising=False,
            msg="rates over extinction",
        )
        self.assertClose(
            runs[3].key_rate / plain.key_rate,
            0.54,
            atol=0.01,
            msg="rate ratio at 20 dB",
        )
        self.assertClose(
            gains[3] / plain.explain["gain"]["value"],
            1.0083,
            atol=5e-4,
            msg="gain ratio at 20 dB",
        )

    def test_error_caps(self):
        """
        cow_data's error rate saturates at 1/2 as the extinction ratio falls to 0.001 dB and never
        passes it.
        """
        gain, e_bit = _core.decoy_gain(0.5, 0.5, 1e-3, 0.1)
        worst = [
            _core.cow_data(gain, e_bit, _core.cow_residual(0.5, db), 0.5)[1] for db in (0.001, 0.01, 0.1, 1.0, 5.0)
        ]

        for got in worst:
            self.assertLessEqual(got, 0.5, msg=f"error rate {got} passed 1/2")

        self.assertClose(worst[0], 0.5, atol=1e-3, msg=f"error rate at 0.001 dB: {worst[0]}")

    def test_monitor_coverage(self):
        """
        Coverage is 1 at 60 dB, 0.9919 at 20 dB and rises with monitoring, while the fringe stays
        the optic's contrast and the certified value is contrast times coverage.
        """
        contrast = 1.0 - 2.0 * E_ALIGN
        ports = ((1.0 + contrast) / 2.0, (1.0 - contrast) / 2.0)
        seen = _core.cow_visibility(*ports)
        info = [cow_link(ext=db).run().explain for db in (60.0, 30.0, 20.0, 10.0)]
        covers = [row["monitored"]["value"] for row in info]
        certs = [row["certified"]["value"] for row in info]

        self.assertClose(seen, contrast, atol=1e-12, msg=f"visibility {seen}")
        self.assertClose(covers[0], 1.0, atol=1e-5, msg="coverage at 60 dB")
        self.assertMonotone(covers, rising=False, msg=f"coverage {covers}")
        self.assertClose(covers[2], 0.9919, atol=1e-4, msg="coverage at 20 dB")
        self.assertClose(certs[0], seen, atol=1e-4, msg="certified at 60 dB")
        self.assertMonotone(certs, rising=False, msg=f"certified {certs}")
        self.assertClose(
            certs[2],
            contrast * covers[2],
            atol=1e-12,
            msg="certified at 20 dB",
        )
        self.assertGreater(
            cow_link(ext=20.0, frac=0.3).run().explain["monitored"]["value"],
            covers[2],
            msg="coverage at decoy_frac 0.3",
        )

    def test_extinction_link(self):
        """
        Through q.Link a 20 dB modulator raises qber and gain, lowers the rate under the ceiling,
        and reports extinction, residual, monitored and certified rows a perfect one omits.
        """
        plain = cow_link(20.0).run()
        real = cow_link(20.0, ext=20.0).run()
        info = real.explain

        self.assertGreater(real.qber, plain.qber, msg=f"qber {real.qber} vs {plain.qber}")
        self.assertGreater(real.p_click, plain.p_click, msg=f"p_click {real.p_click} vs {plain.p_click}")
        self.assertLess(real.key_rate, plain.key_rate, msg=f"rate {real.key_rate} vs {plain.key_rate}")

        for key in ("extinction", "residual", "monitored", "certified"):
            self.assertIn(key, info, msg=f"explain lacks {key}")

        self.assertEqual(info["residual"]["value"], MU_REF / 100.0, msg="20 dB residual row")
        self.assertLessEqual(real.key_rate, info["ceiling"]["value"], msg="rate vs ceiling")
        self.assertNotIn("extinction", plain.explain, msg="extinction row at no extinction")

    def test_extinction_guards(self):
        """
        cow_residual, cow_monitor and q.IntensityKeying refuse a non-positive extinction ratio, and
        cow_data a signal error rate past 1/2.
        """
        cases = (
            (_core.cow_residual, (0.5, 0.0), "ext_db", "0 dB"),
            (_core.cow_monitor, (1.0, -3.0, 0.1), "ext_db", "negative dB"),
            (_core.cow_data, (0.1, 0.7, 0.0, 0.5), "e_sig", "e_sig > 1/2"),
            (
                q.IntensityKeying,
                (0.5, 0.1, 0.0),
                "extinction must be a positive",
                "IntensityKeying at 0 dB",
            ),
        )
        for fn, args, needle, why in cases:
            self.assertBad(needle, fn, args, msg=why)


class BasisSymbols(Guarded):
    """
    Counted against closed-form BB84, exact at zero dark count and apart at O(dark); and the
    per-basis error account, informative only off a channel that tells the bases apart.
    """

    def test_dark_free_agreement(self):
        """
        At zero dark count over 2e6 pulses at T*eta = 0.5 the qber lands in a four-sigma binomial
        band of the 0.033 misalignment and the gains in theirs, with no double clicks.
        """
        out = counted(2_000_000, 4, 0.033)
        for i, mu in enumerate((0.5, 0.1)):
            want = _core.decoy_gain(mu, 0.5, 0.0, 0.033)
            self.assertClose(
                out.gain[i],
                want[0],
                atol=band(want[0], out.sent[i]),
                msg=f"gain at mu = {mu}",
            )

            # Half-widths 1.70e-3 and 8.39e-3; seeds 1..100 use 0.734 and 0.588 of them.
            self.assertClose(
                out.qber[i],
                0.033,
                atol=band(0.033, out.sifted[i]),
                msg=f"qber at mu = {mu}",
            )

        self.assertEqual(out.doubles, 0, msg=f"doubles {out.doubles}")

    def test_dark_convention(self):
        """
        ClickDetector.dark is one detector's per-gate probability and the report derives Y0 = 1 - (1
        - dark)^2 from it; at 1% the sample sits below the closed form by the coincidence cross
        term, over eight sigma, and above the detector-dark reading, 0.0678 against 0.0537.
        """
        dark = 0.01
        closed = dark_link(dark).run()
        sampled = dark_link(dark, alice=q.Alice()).run(symbols=2_000_000, seed=4)
        y0 = background(dark)
        hit = 1.0 - math.exp(-0.25)
        drop = hit * E_DET * dark + 0.5 * hit * dark * (1.0 - dark)
        want = closed.qber - drop / closed.p_click
        halved = _core.decoy_gain(0.5, 0.5, dark, E_DET)[1]
        edge = band(want, sampled.sifted, 1.0)

        self.assertEqual(closed.explain["y0"]["value"], y0, msg="explain y0")
        self.assertEqual(closed.explain["dark"]["value"], dark, msg="explain dark")
        self.assertClose(
            sampled.p_click,
            closed.p_click,
            atol=band(closed.p_click, sampled.slots),
            msg=f"sampled gain {sampled.p_click:.6f} vs closed {closed.p_click:.6f}",
        )
        self.assertClose(
            sampled.qber,
            want,
            atol=4.0 * edge,
            msg=f"sampled qber {sampled.qber:.6f} vs predicted {want:.6f}",
        )
        self.assertGreater(
            (closed.qber - sampled.qber) / edge,
            8.0,
            msg=f"closed qber {closed.qber:.6f} vs sampled {sampled.qber:.6f}",
        )
        self.assertGreater(
            (sampled.qber - halved) / edge,
            8.0,
            msg=f"sampled qber {sampled.qber:.6f} vs Y0 reading {halved:.6f}",
        )

    def test_counts_consistent(self):
        """
        run_basis counts nest errors in sifted in clicks in sent, sent sums to the pulses, the three
        bases partition sifted and errors, and a two-basis run puts every test count in X and none
        in Y.
        """
        out = _core.run_basis(400_000, 9, (0.5, 0.1, 0.02), (0.5, 0.3, 0.2), 0.4, 0.6, 1e-3, 0.02, 0.5, 0)

        self.assertEqual(sum(out.sent), 400_000, msg=f"sent {sum(out.sent)}")
        self.assertEqual(out.bases, 2, msg=f"bases {out.bases}")
        self.assertEqual(out.x_sifted, out.test_sifted, msg="x_sifted vs test_sifted")
        self.assertEqual(out.x_errors, out.test_errors, msg="x_errors vs test_errors")
        self.assertEqual(out.y_sifted, [0, 0, 0], msg=f"y_sifted {out.y_sifted}")
        self.assertEqual(out.y_errors, [0, 0, 0], msg=f"y_errors {out.y_errors}")

        for i in range(3):
            self.assertLessEqual(out.errors[i], out.sifted[i], msg=f"errors exceed sifted at arm {i}")
            self.assertLessEqual(out.sifted[i], out.clicks[i], msg=f"sifted exceeds clicks at arm {i}")
            self.assertLessEqual(out.clicks[i], out.sent[i], msg=f"clicks exceed sent at arm {i}")

            kept = out.key_sifted[i] + out.x_sifted[i] + out.y_sifted[i]
            self.assertEqual(kept, out.sifted[i], msg=f"basis sifted sum at arm {i}")

            wrong = out.key_errors[i] + out.x_errors[i] + out.y_errors[i]
            self.assertEqual(wrong, out.errors[i], msg=f"basis error sum at arm {i}")

        self.assertLessEqual(out.doubles, sum(out.sifted), msg=f"doubles {out.doubles}")

    def test_sift_thinning(self):
        """
        The sifted-per-click rate converges on the sift factor at 0.5 and 0.9 within a binomial
        band.
        """
        for want in (0.5, 0.9):
            out = counted(400_000, 2, 0.02, sift=want)
            self.assertClose(
                out.sift_rate,
                want,
                atol=band(want, sum(out.clicks)),
                msg=f"sift rate {out.sift_rate:.5f} against {want}",
            )

    def test_seed_and_grain(self):
        """
        run_basis reproduces every count at one seed and at grains 1, 997 and 65536, and differs at
        another seed.
        """
        args = ((0.5, 0.1, 0.0), (0.6, 0.25, 0.15), 0.5, 0.8, 1e-4, 0.03, 0.5)
        base = _core.run_basis(200_000, 17, *args, 0)
        same = _core.run_basis(200_000, 17, *args, 0)
        other = _core.run_basis(200_000, 18, *args, 0)
        grains = [_core.run_basis(200_000, 17, *args, cz) for cz in (1, 997, 65536)]

        self.assertEqual(base.errors, same.errors, msg="errors at seed 17")
        self.assertEqual(base.sifted, same.sifted, msg="sifted at seed 17")
        self.assertNotEqual(base.errors, other.errors, msg="errors at seeds 17, 18")

        for i, got in enumerate(grains):
            self.assertEqual(got.errors, base.errors, msg=f"grain {i} moved an error count")
            self.assertEqual(got.clicks, base.clicks, msg=f"grain {i} moved a click count")

    def test_sampled_link(self):
        """
        A sampled q.Link on GYS hardware at 25 km and 1.5e8 pulses matches the closed-form gain and
        qber in their bands and the rate within 12%, and labels its stages.
        """
        closed = basis_link(25.0).run()
        sampled = basis_link(25.0, alice=q.Alice()).run(symbols=150_000_000, seed=5)
        info = sampled.explain

        self.assertClose(
            sampled.p_click,
            closed.p_click,
            atol=band(closed.p_click, sampled.slots // 3),
            msg=f"sampled gain {sampled.p_click} vs {closed.p_click}",
        )
        self.assertClose(
            sampled.qber,
            closed.qber,
            atol=band(closed.qber, sampled.sifted),
            msg=f"sampled qber {sampled.qber} vs {closed.qber}",
        )

        # 12% is 5.5 sigma at 1.5e8 pulses (seeds 1..24: 2.1%, max 0.0526); a shorter block needs a wider band.
        self.assertClose(
            sampled.key_rate / closed.key_rate,
            1.0,
            atol=0.12,
            msg=f"sampled rate {sampled.key_rate:.4e} vs {closed.key_rate:.4e}",
        )
        self.assertEqual(info["stages"], "pulse train + sifting", msg="stages")
        self.assertEqual(info["pulses"]["value"], 150_000_000, msg="explain pulses")
        self.assertEqual(
            basis_link(25.0, alice=q.Alice()).explain()["stages"],
            "pulse train (unrun)",
            msg="dry-run stages",
        )
        self.assertEqual(closed.explain["stages"], "closed-form only", msg="closed-form stages")

    def test_sampled_wiring(self):
        """
        The sampled link's y1, e1, q1 and rate equal decoy_bounds on run_basis's (gain, error) pairs
        interleaved per intensity, then bb84_rate, bit for bit.
        """
        link = basis_link(25.0, alice=q.Alice())
        res = link.run(symbols=400_000, seed=11)
        info = res.explain
        out = _core.run_basis(
            400_000,
            11,
            (0.48, 0.1, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            _core.decoy_eta(ALPHA, 25.0, 1.0),
            ETA_BOB,
            P_DARK,
            E_DET,
            0.5,
            0,
        )
        y1, e1, q1 = _core.decoy_bounds(
            0.48,
            0.1,
            0.0,
            out.gain[0],
            out.qber[0],
            out.gain[1],
            out.qber[1],
            out.gain[2],
            out.qber[2],
        )
        want = _core.bb84_rate(0.5, out.gain[0], out.qber[0], q1, e1, F_REC)

        self.assertEqual(res.p_click, out.gain[0], msg="p_click vs signal arm")
        self.assertEqual(res.qber, out.qber[0], msg="qber vs signal arm")
        self.assertEqual(info["y1"]["value"], y1, msg="y1")
        self.assertEqual(info["e1"]["value"], e1, msg="e1")
        self.assertEqual(info["q1"]["value"], q1, msg="q1")
        self.assertEqual(res.key_rate, want, msg="key_rate")

    def test_sampled_guards(self):
        """
        The sampled path refuses a laser, and q.Decoy and run_basis refuse probabilities not summing
        to 1 and intensities violating nu1 + nu2 < mu.
        """
        self.assertFails(
            ValueError,
            "intensity set",
            basis_link(25.0, alice=q.Alice(laser=q.Laser())).run,
            msg="q.Laser",
        )
        self.assertBad(
            "probs must sum to 1",
            q.Decoy,
            ((0.5, 0.1, 0.0), (0.5, 0.2, 0.2)),
            msg="q.Decoy probs sum 0.9",
        )

        plans = (
            (
                (0.5, 0.1, 0.0),
                (0.5, 0.2, 0.2),
                "sum to 1",
                "run_basis probs sum 0.9",
            ),
            (
                (0.2, 0.15, 0.1),
                (0.4, 0.3, 0.3),
                "nu1 + nu2 < mu",
                "run_basis nu1 + nu2 = 0.25 > mu",
            ),
        )
        for intens, probs, needle, why in plans:
            args = (1000, 0, intens, probs, 0.5, 1.0, 0.0, 0.02, 0.5, 0)
            self.assertBad(needle, _core.run_basis, args, msg=why)

    def test_basis_defaulted(self):
        """
        Naming bases = 2 and the key basis's own monitor rate reproduces every count of the unnamed
        call at three grains.
        """
        args = ((0.5, 0.1, 0.0), (0.6, 0.25, 0.15), 0.5, 0.8, 1e-4, 0.03, 0.5)
        base = _core.run_basis(200_000, 17, *args, 0)
        named = [_core.run_basis(200_000, 17, *args, cz, None, None, 2, 0.03) for cz in (0, 997, 65536)]

        for i, got in enumerate(named):
            self.assertEqual(got.errors, base.errors, msg=f"grain {i} moved an error count")
            self.assertEqual(got.clicks, base.clicks, msg=f"grain {i} moved a click count")
            self.assertEqual(got.key_errors, base.key_errors, msg=f"grain {i} moved the key basis")
            self.assertEqual(
                [got.clicks_d0, got.clicks_d1],
                [base.clicks_d0, base.clicks_d1],
                msg=f"grain {i} moved a detector tally",
            )

    def test_three_shares(self):
        """
        run_basis's three-basis key share inverts sixstate_sift to 3e-16 over six biases, and at the
        uniform bias X and Y counts sit within 0.6% of the key basis against a 2.5% four-sigma band.
        """
        for bias in (1.0 / 3.0, 0.4, 0.5, 0.7, 0.9, 0.99):
            sift, key = _core.sixstate_sift(bias)
            out = counted(2000, 1, 0.02, (1.0, 0.0, 0.0), sift, bases=3)
            self.assertClose(
                out.key_share,
                key,
                atol=3e-16,
                msg=f"key share at bias {bias}",
            )

        out = six_run(2_000_000, 5, (0.02, 0.02, 0.02))
        edge = 4.0 * math.sqrt(2.0 / out.key_sifted[0])

        for got in (out.x_sifted[0], out.y_sifted[0]):
            self.assertClose(
                got / out.key_sifted[0],
                1.0,
                atol=edge,
                msg=f"{got} against {out.key_sifted[0]} in the key basis",
            )

    def test_spectrum_recovered(self):
        """
        At (e_x, e_y, e_z) = (0.05, 0.07, 0.03) over 6e6 pulses the rates read back in their
        binomial bands, sixstate_bell resolves weights 0.0448, 0.0243, 0.0051 beyond four propagated
        sigma, and the tomographic Holevo term is 0.2814 against the depolarising 0.2148.
        """
        want = (0.05, 0.07, 0.03)
        out = six_run(6_000_000, 5, want)
        got = basis_rates(out)
        seen = (out.x_sifted[0], out.y_sifted[0], out.key_sifted[0])
        edge = [band(p, m) for p, m in zip(want, seen)]

        for i in range(3):
            self.assertClose(
                got[i],
                want[i],
                atol=edge[i],
                msg=f"basis {i} read {got[i]:.5f} against {want[i]}",
            )

        weights = _core.sixstate_bell(*got)
        spread = 0.5 * math.sqrt(sum(x * x for x in edge))

        self.assertGreater(min(weights), 0.0, msg=f"weights {weights}")

        for i in (1, 2):
            self.assertGreater(
                (weights[i] - weights[i + 1]) / spread,
                4.0,
                msg=f"weights {i}, {i + 1} separation",
            )

        eve = _core.sixstate_eve(*got)
        flat = _core.sixstate_holevo(sum(got) / 3.0)

        self.assertClose(eve, 0.2814, atol=5e-4, msg=f"sixstate_eve {eve}")
        self.assertClose(flat, 0.2148, atol=5e-4, msg=f"sixstate_holevo {flat}")
        self.assertGreater(eve / flat, 1.3, msg=f"eve/flat {eve / flat}")

    def test_symmetric_noise(self):
        """
        One rate in every basis samples one number: sixstate_bell refuses seed 7's 5, 6 and 14
        errors at 4e5 pulses, and at 1e6 pulses the three depolarising weights Q/2 = 5.00e-4 spread
        over more than a fifth of it, each inside its four-sigma band.
        """
        flat = (0.001, 0.001, 0.001)
        thin = six_run(400_000, 7, flat)
        out = six_run(1_000_000, 5, flat)
        seen = (out.x_sifted[0], out.y_sifted[0], out.key_sifted[0])
        edge = [band(0.001, m) for m in seen]
        weights = _core.sixstate_bell(*basis_rates(out))[1:]
        spread = 0.5 * math.sqrt(sum(x * x for x in edge))

        self.assertEqual(
            [thin.key_errors[0], thin.x_errors[0], thin.y_errors[0]],
            [5, 6, 14],
            msg="seed 7 error counts",
        )
        self.assertFails(
            ValueError,
            "no Bell-diagonal state",
            _core.sixstate_bell,
            *basis_rates(thin),
            msg="sixstate_bell at seed 7",
        )
        self.assertClose(
            _core.sixstate_bell(*flat)[1],
            5e-4,
            msg="sixstate_bell(flat)[1]",
        )
        self.assertGreater(
            (max(weights) - min(weights)) / 5e-4,
            0.2,
            msg=f"weight spread {max(weights) - min(weights):.3e}",
        )

        for i, got in enumerate(weights):
            self.assertClose(
                got,
                5e-4,
                atol=spread,
                msg=f"weight {i} at {got:.3e}",
            )

    def test_basis_guards(self):
        """
        run_basis refuses a basis count other than 2 or 3, a second monitor rate in a two-basis run,
        a monitor rate past 1/2, and a rate triple no Pauli channel has.
        """
        cases = (
            ((None, None, 4), "bases must be 2", "bases = 4"),
            (
                (None, None, 2, None, 0.02),
                "second monitor basis",
                "misalign_y at bases = 2",
            ),
            (
                (None, None, 3, 0.7),
                "misalign_x must be in [0, 1/2]",
                "misalign_x = 0.7",
            ),
            (
                (None, None, 3, 0.005, 0.005),
                "no Pauli channel",
                "0.005 + 0.005 < 0.02 leaves a Bell weight negative",
            ),
        )
        for tail, needle, why in cases:
            self.assertBad(needle, _core.run_basis, PLAN + tail, msg=why)


class PolarisationCarrier(Guarded):
    """
    Assumed: both bases linear and turned by one angle; real SU(2) birefringence makes them
    elliptical and is strictly worse for one basis.
    """

    def test_carrier_reduces(self):
        """
        An undescribed frame has contrast exactly 1.0, pol_error returns the misalignment, and the
        polarisation link equals the plain one in rate, qber and gain.
        """
        self.assertEqual(_core.pol_contrast(0.0, 0.0, 0.0), 1.0, msg="pol_contrast(0, 0, 0)")
        self.assertEqual(_core.pol_error(E_DET, 1.0), E_DET, msg="pol_error at contrast 1")

        plain = basis_link(25.0).run()
        keyed = basis_link(25.0, mod=polar()).run()

        self.assertEqual(keyed.key_rate, plain.key_rate, msg="key_rate")
        self.assertEqual(keyed.qber, plain.qber, msg="qber")
        self.assertEqual(keyed.p_click, plain.p_click, msg="p_click")
        self.assertEqual(
            keyed.explain["carrier"]["value"],
            "polarisation",
            msg="explain carrier",
        )

    def test_tracking_matters(self):
        """
        A 5 mrad tracked frame and a 0.1 rad/sqrt(s) free walk over 1 s differ over 100x in error
        rate, and free-walk contrast decays from 1 to 0 with the interval.
        """
        tracked = _core.pol_error(0.0, _core.pol_contrast(0.005, 0.0, 0.0))
        free = _core.pol_error(0.0, _core.pol_walk(0.1, 1.0))

        self.assertGreater(free / tracked, 100.0, msg=f"free/tracked {free / tracked}")

        walks = [_core.pol_walk(0.1, t) for t in (0.0, 1.0, 10.0, 1e3, 1e5)]

        self.assertEqual(walks[0], 1.0, msg="pol_walk at t = 0")
        self.assertMonotone(walks, rising=False, msg=f"walks {walks}")
        self.assertClose(walks[-1], 0.0, atol=1e-3, msg="pol_walk at t = 1e5")

    def test_contrast_composes(self):
        """
        pol_error composes misalignment and frame rotation as contrasts, 1 - 2e = (1 - 2e_miss) C to
        1e-14, not as summed error rates.
        """
        for miss in (0.0, 0.02, 0.1, 0.3):
            for contrast in (1.0, 0.9, 0.5, 0.0):
                got = _core.pol_error(miss, contrast)
                self.assertClose(
                    1.0 - 2.0 * got,
                    (1.0 - 2.0 * miss) * contrast,
                    atol=1e-14,
                    msg=f"pol_error at ({miss}, {contrast})",
                )

    def test_drift_link(self):
        """
        Through q.Link a 50 mrad drift lifts the qber from 3.31% to 3.545% and keeps 0.898 of the
        key rate, reporting tracking, optics misalignment, derived misalignment and contrast as
        separate rows.
        """
        plain = basis_link(25.0).run()
        drift = basis_link(25.0, mod=polar(drift=0.05)).run()
        info = drift.explain

        self.assertClose(drift.qber, 0.03545, atol=1e-4, msg="qber under 50 mrad drift")
        self.assertClose(drift.key_rate / plain.key_rate, 0.898, atol=0.005, msg="rate ratio at 50 mrad")
        self.assertEqual(info["tracking"]["value"], "tracked", msg="explain tracking")
        self.assertEqual(info["misalign_optics"]["value"], E_DET, msg="explain misalign_optics")
        self.assertEqual(info["misalign"]["label"], "derived", msg="explain misalign label")
        self.assertClose(
            info["pol_contrast"]["value"],
            math.exp(-2.0 * 0.05 * 0.05),
            atol=1e-12,
            msg="explain pol_contrast",
        )

    def test_pmd_link(self):
        """
        PMD contrast is the field overlap sqrt(mode_overlap), not the intensity overlap, and the
        link scales DGD by sqrt(length) to 0.5 ps at 25 km and refuses PMD with no span.
        """
        width = 10e-12
        scale = 2.0 * math.sqrt(2.0 * math.log(2.0))
        for dgd in (2e-12, 5e-12, 10e-12):
            both = _core.mode_overlap(width * scale, width * scale, dgd)
            self.assertClose(
                _core.pol_contrast(0.0, dgd, width),
                math.sqrt(both),
                atol=1e-12,
                msg=f"PMD contrast at dgd = {dgd}",
            )

        self.assertClose(
            _core.pol_contrast(0.0, 0.0, width),
            1.0,
            atol=1e-15,
            msg="pol_contrast at dgd 0",
        )
        self.assertBad(
            "needs a pulse width",
            _core.pol_contrast,
            (0.0, 1e-12, 0.0),
            msg="width = 0",
        )

        frame = polar(dispersion=0.1e-12, width=10e-12)
        res = basis_link(25.0, mod=frame).run()
        info = res.explain

        self.assertClose(info["dgd"]["value"], 0.5e-12, atol=1e-15, msg="explain dgd")
        self.assertClose(
            info["pol_contrast"]["value"],
            math.exp(-(0.05**2) / 8.0),
            atol=1e-12,
            msg="explain pol_contrast",
        )
        self.assertLess(res.key_rate, basis_link(25.0).run().key_rate, msg="rate with PMD")

        bare = q.Link(
            modulation=frame,
            channel=q.Channel(T=0.3),
            bob=q.Bob(
                detector=q.ClickDetector(eta=ETA_BOB, dark=P_DARK),
                receiver=q.BasisAnalyser(misalign=E_DET),
            ),
            security=q.SplittingAttack(f=F_REC),
        )

        self.assertFails(ValueError, "fibre span", bare.run, msg="PMD on q.Channel")

    def test_frame_guards(self):
        """
        q.ReferenceFrame refuses an unknown model, a field the chosen tracking model does not
        consume, a free frame with no rate, and dispersion without a width.
        """
        cases = (
            (("auto",), "tracking must be", "unknown model"),
            (
                ("tracked", 0.01, 0.1, 1.0),
                "belong to tracking='free'",
                "rate on tracked",
            ),
            (
                ("free", 0.01, 0.1, 1.0),
                "belongs to tracking='tracked'",
                "drift on free",
            ),
            (
                ("free",),
                "positive rate and interval",
                "free with no rate",
            ),
            (
                ("tracked", 0.0, 0.0, 0.0, 0.1e-12),
                "read together",
                "dispersion with no pulse width",
            ),
        )
        for args, needle, why in cases:
            self.assertBad(needle, q.ReferenceFrame, args, msg=why)


if __name__ == "__main__":
    rc = Exam(
        "Bb84Wcp",
        "Tier A: GLLP + decoy BB84 with weak coherent pulses vs the GYS parameter set",
        "discrete_bb84.md",
    ).run(load(Bb84Wcp))
    rc |= Exam(
        "CowRevised",
        "COW under the post-zero-error-attack analysis: quadratic, not linear",
        "discrete_cow.md",
    ).run(load(CowRevised))
    rc |= Exam(
        "CowExtinction",
        "Finite modulator extinction: light in the slot the protocol calls empty",
        "discrete_extinction.md",
    ).run(load(CowExtinction))
    rc |= Exam(
        "BasisSymbols",
        "Symbol-level BB84: gains and error rates counted rather than integrated",
        "discrete_symbols.md",
    ).run(load(BasisSymbols))
    rc |= Exam(
        "PolarisationCarrier",
        "Polarisation encoding: misalignment from a frame rotation, not a knob",
        "discrete_polarisation.md",
    ).run(load(PolarisationCarrier))
    sys.exit(rc)
