import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

import qkd as q
from qkd import _core

# The physics of each engine is pinned by test/pnr.py, test/tlo.py,
# test/finitequbit.py, test/mdi.py and test/flaws.py against closed forms and
# published numbers; nothing is re-derived here. What is checked is that the
# component reaches the same engine with the same arguments, that the refusals
# refuse through q. rather than only through qkd._core, and that every name is
# in __all__.

# Curty's and Ma & Razavi's shared MDI-BB84 setting, as test/mdi.py carries it:
# relay detection efficiency 14.5%, misalignment 1.5%, 0.2 dB/km, f_ec = 1.16.
# DARK is PER DETECTOR, half the 6.02e-6 the literature quotes as a receiver
# total, which is the convention mdi_yield and friends take.
ALPHA = 0.2
DET = 0.145
MISALIGN = 0.015
FEC = 1.16
DARK = 6.02e-6 / 2.0

# The intensity ladder and the half-span both arms cross.
SIGNAL = 0.3
NU = 0.005
HALF = 25.0

# The four cells of the 3x3 grid mdi_e11 reads, in its own argument order.
CELLS = (4, 8, 5, 7)

# Lodewyck's transmitted-oscillator point, from test/tlo.py: 1e9 oscillator
# photons per pulse, a 400 ns multiplexing delay, T = 0.302, eta = 0.606 and
# v_el = 0.041 SNU at characterisation.
LOD_LO = 1e9
LOD_DELAY = 400e-9
LOD_T = 0.302
LOD_VEL = 0.041

# Fock cutoffs. 80 is high enough that a coherent state of mu <= 3 loses nothing
# to the truncation; 12 is the small one the confusion matrix is read on.
CUT_BIG = 80
CUT_LOW = 12

# The source-flaw point: a 0.3 rad modulation deviation read through half
# transmittance overall into a 1e-6 per-gate dark probability. FLAWS is the
# spread the tilt = 0 cancellation is measured across, all below pi/2 so that
# sin^2(delta/2) stays under the 1/2 the engine caps at.
FLAW = 0.3
FLAWS = (0.05, 0.2, 1.0)
ETA = 0.5
GATE = 1e-6


def arms(km):
    """
    Per-arm transmittance with the midpoint at the centre of a total span of km.
    """

    return 10.0 ** (-ALPHA * (km / 2.0) / 10.0)


def grid(fn, eta, miss):
    """
    One basis's 3x3 forward model, row-major with Alice's intensity as the row, written
    out so the surface is checked against arithmetic rather than against itself.
    """
    levels = (SIGNAL, NU, 0.0)

    return [fn(x, y, eta, eta, DARK, miss) for x in levels for y in levels]


def engine(km):
    """
    The asymptotic MDI-BB84 rate at a symmetric midpoint, composed through qkd._core alone.
    """
    eta = DET * arms(km)
    zed = grid(_core.mdi_rect, eta, MISALIGN)
    test = grid(_core.mdi_diag, eta, MISALIGN)
    sets = [SIGNAL, NU, 0.0]
    y_key = _core.mdi_y11(sets, sets, [g for g, _ in zed])
    y_test = _core.mdi_y11(sets, sets, [g for g, _ in test])
    e1 = _core.mdi_e11(
        NU,
        0.0,
        NU,
        0.0,
        *[test[k][0] * test[k][1] for k in CELLS],
        y_test,
    )
    gain, qber = zed[0]

    return _core.mdi_rate(_core.mdi_gain(y_key, SIGNAL, SIGNAL), e1, gain, qber, FEC)


def sender():
    """
    One keyed weak-coherent sender in the biased-basis limit the midpoint bound is in.
    """

    return q.Sender(modulation=q.BasisKeying(decoy=q.Decoy((SIGNAL, NU, 0.0)), sift=1.0))


def analyser(**kw):
    """
    The midpoint's four-detector Bell-state measurement.
    """
    slots = dict(eta=DET, dark=DARK, misalign=MISALIGN, misalign_test=MISALIGN)
    slots.update(kw)

    return q.Relay(bell=q.BellAnalyser(**slots))


def midpoint(alice=None, bob=None, relay=None, channels=None, security=None, env=None):
    """
    A swap through the qubit midpoint, every part overridable.
    """

    return q.Swap(
        alice=sender() if alice is None else alice,
        bob=sender() if bob is None else bob,
        relay=analyser() if relay is None else relay,
        channels=((q.Fiber(length=HALF, alpha=ALPHA),) * 2 if channels is None else channels),
        security=q.TestBasisBound(f=FEC) if security is None else security,
        environment=env,
    )


def keyed(security, **kw):
    """
    A basis-keyed weak coherent link, the variant selected by kw.
    """

    # GYS hardware but NOT test/api.py's basis_link: the decoy set there is
    # (0.48, 0.1, 0.0) against (0.2, 0.08, 0.0) here, so the two carry different
    # single-photon yields and neither file's pins survive the other's fixture.
    return q.Link(
        modulation=q.BasisKeying(decoy=q.Decoy((0.2, 0.08, 0.0)), **kw),
        channel=q.Fiber(length=25.0, alpha=0.21),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
            receiver=q.BasisAnalyser(misalign=0.033),
        ),
        security=security,
    )


def two_state(security, ref=False):
    """
    A two-state link and Bob's nulling receiver.
    """

    return q.Link(
        modulation=q.TwoStateKeying(mu=0.23, reference=ref),
        channel=q.Channel(T=0.9),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=1e-6),
            receiver=q.NullingReceiver(visibility=0.98),
        ),
        security=security,
    )


class Counting(Question):
    def test_names_shipped(self):
        """
        `PnrDetector`, `ThresholdArray`, `TransmittedLO` and `ShotNoise` are in `__all__`
        and present on the package.
        """
        want = ("PnrDetector", "ThresholdArray", "TransmittedLO", "ShotNoise")
        gone = [name for name in want if name not in q.__all__]
        self.assertEqual(gone, [], msg=f"absent from __all__: {gone}")

        absent = [name for name in want if not hasattr(q, name)]
        self.assertEqual(absent, [], msg=f"promised and missing: {absent}")

    def test_array_complete(self):
        """
        A multiplexed array's POVM sums to the identity on its truncated space
        and carries no negative diagonal entry.
        """
        povm = q.ThresholdArray(elements=6, eta=0.7, dark=1e-3, cutoff=CUT_LOW).povm()
        self.assertEqual(povm.outcomes, 7, msg="one outcome per element, plus none")
        self.assertEqual(povm.cutoff, CUT_LOW, msg="defined on the declared levels")
        self.assertClose(povm.defect(), 0.0, atol=1e-14, msg="sum_k Pi_k is not I")
        self.assertGreaterEqual(povm.floor(), 0.0, msg="a negative POVM entry")

    def test_array_click(self):
        """
        At one element the array's click outcome is `click_prob`, the threshold law, to 1e-15.
        """
        worst = 0.0
        for eta in (0.1, 0.4, 1.0):
            for dark in (0.0, 1e-3):
                one = q.ThresholdArray(elements=1, eta=eta, dark=dark, cutoff=CUT_BIG)
                for mu in (0.01, 0.5, 3.0):
                    got = float(one.outcomes(mu)[1])
                    want = _core.click_prob(math.sqrt(mu), 0.0, eta, dark)
                    worst = max(worst, abs(got - want))
        self.assertClose(worst, 0.0, atol=1e-15, msg=f"one element off click_prob by {worst:.3e}")

    def test_array_folds(self):
        """
        The array's closed-form outcome distribution agrees to 1e-14 with folding its own
        POVM against a Poisson photon-number distribution.
        """
        arr = q.ThresholdArray(elements=8, eta=0.5, dark=1e-3, cutoff=CUT_BIG)
        worst = 0.0
        for mu in (0.2, 1.0, 3.0):
            got = np.asarray(arr.outcomes(mu))
            want = np.asarray(arr.povm().coherent(mu))
            worst = max(worst, float(np.max(np.abs(got - want))))
        self.assertClose(worst, 0.0, atol=1e-14, msg=f"closed form off the fold by {worst:.3e}")

    def test_array_distinct(self):
        """
        `distinct(n)`, the array's fidelity to true number resolution, is 1 at one photon,
        falls with $n$, and is exactly 0 once $n$ passes the element count.
        """
        arr = q.ThresholdArray(elements=4)
        self.assertClose(arr.distinct(1), 1.0, msg="one photon always lands alone")
        self.assertClose(arr.distinct(5), 0.0, msg="five photons into four bins")

        seq = [arr.distinct(n) for n in range(1, 5)]
        self.assertMonotone(seq, rising=False, msg="collisions only accumulate")

    def test_number_absorbs(self):
        """
        The number-resolving readout's POVM sums to the identity, absorbing at the top,
        and a zero readout width gives the identity confusion matrix.
        """
        det = q.PnrDetector(eta=1.0, background=0.0, sigma=0.0, cutoff=CUT_LOW)
        povm = det.povm()
        self.assertClose(povm.defect(), 0.0, atol=1e-14, msg="sum_k Pi_k is not I")

        conf = np.asarray(det.confusion(4))
        self.assertClose(
            float(np.max(np.abs(conf - np.eye(5)))),
            0.0,
            atol=1e-14,
            msg="a zero readout width is not the identity assignment",
        )

    def test_number_widens(self):
        """
        A finite readout width moves mass off the diagonal of the confusion
        matrix while every column still sums to one.
        """
        wide = np.asarray(q.PnrDetector(sigma=0.4).confusion(6))
        sums = wide.sum(axis=0)
        self.assertClose(
            float(np.max(np.abs(sums - 1.0))),
            0.0,
            atol=1e-12,
            msg="a confusion column is not a distribution",
        )
        self.assertLess(float(wide[3][3]), 1.0, msg="a wide readout is still exact")

    def test_counting_refused(self):
        """
        `q.Link` refuses both counting receivers by name: every bound it ships over a
        counting receiver is written on a binary click.
        """
        for kind in (q.PnrDetector(), q.ThresholdArray()):
            link = q.Link(
                modulation=q.GaussianModulation(),
                channel=q.Channel(T=0.5),
                bob=q.Bob(detector=kind),
            )
            self.assertFails(
                NotImplementedError,
                "photon-number outcome",
                link.run,
                msg=f"{type(kind).__name__} reached a rate",
            )

    def test_counting_domains(self):
        """
        Both counting receivers refuse an out-of-range efficiency, dark rate, element
        count, background and readout width, naming the parameter refused.
        """
        self.assertFails(ValueError, "eta", lambda: q.ThresholdArray(eta=1.5), msg="eta above 1")
        self.assertFails(ValueError, "dark", lambda: q.ThresholdArray(dark=1.0), msg="dark at 1")
        self.assertFails(
            ValueError,
            "elements",
            lambda: q.ThresholdArray(elements=0),
            msg="an array of no elements measures nothing",
        )
        self.assertFails(
            ValueError,
            "background",
            lambda: q.PnrDetector(background=-1e-9),
            msg="a negative spurious count",
        )
        self.assertFails(ValueError, "sigma", lambda: q.PnrDetector(sigma=-0.1), msg="negative width")


class Oscillator(Question):
    def test_local_limit(self):
        """
        A lossless span with a back-to-back characterisation returns the locally generated
        limit: unit and ratio both 1, electronic noise unchanged, no light lost.
        """
        out = q.TransmittedLO(photons=LOD_LO).shot(1.0, LOD_VEL)
        self.assertClose(out.unit, 1.0, msg="the operating unit moved")
        self.assertClose(out.ratio, 1.0, msg="assumed and actual differ")
        self.assertClose(out.v_el, LOD_VEL, msg="electronic noise was re-referred")
        self.assertClose(out.arriving, LOD_LO, msg="the oscillator lost light")

    def test_unit_tracks(self):
        """
        The operating shot-noise unit is the oscillator arm's transmittance, span times
        multiplexer, and the ratio to the characterisation level is its reciprocal.
        """
        lo = q.TransmittedLO(photons=LOD_LO, mux=0.5)
        out = lo.shot(LOD_T, LOD_VEL)
        self.assertClose(out.transmittance, 0.5 * LOD_T, msg="the arm is span x mux")
        self.assertClose(out.unit, 0.5 * LOD_T, msg="the unit is not the arm")
        self.assertClose(out.ratio, 1.0 / (0.5 * LOD_T), msg="assumed over actual")

    def test_vel_grows(self):
        """
        Electronic noise referred to the operating unit grows as the oscillator arm loses
        light: 0.01 SNU quoted back to back reads 1.0 once the oscillator has crossed 20 dB.
        """
        lo = q.TransmittedLO(photons=LOD_LO)
        out = lo.shot(0.01, 0.01)
        self.assertClose(out.v_el, 1.0, msg="v_el did not follow the oscillator")

        seq = [lo.shot(t, 0.01).v_el for t in (1.0, 0.5, 0.1, 0.01)]
        self.assertMonotone(seq, msg="a dimmer oscillator must cost more")

    def test_phase_common(self):
        """
        The residual phase variance is the Wiener 2*pi*linewidth*delay over the
        multiplexing delay alone, and a zero delay carries exactly none.
        """
        lo = q.TransmittedLO(delay=LOD_DELAY)
        want = 2.0 * math.pi * 10e3 * LOD_DELAY
        self.assertClose(lo.phase(10e3), want, msg="not the Wiener variance")
        self.assertClose(
            q.TransmittedLO(delay=0.0).phase(10e3),
            0.0,
            msg="a zero delay accumulated phase",
        )

    def test_phase_absent(self):
        """
        `q.TransmittedLO` has no linewidth field and refuses one: both modes leave one
        laser, so the linewidth that multiplies the delay is Alice's own.
        """
        lo = q.TransmittedLO()
        self.assertFails(
            TypeError,
            "linewidth",
            lambda: q.TransmittedLO(linewidth=10e3),
            msg="a linewidth field was accepted",
        )
        self.assertFalse(hasattr(lo, "linewidth"), msg="no field appeared")

    def test_leak_displaces(self):
        """
        A leaking multiplexer displaces the measured quadrature rather than adding a
        variance, and an undeclared extinction returns None rather than zero.
        """
        lo = q.TransmittedLO(photons=LOD_LO, extinction=60.0)
        want = 2.0 * math.sqrt(1e-6 * LOD_LO * LOD_T)
        self.assertClose(lo.leak(LOD_T), want, msg="not the coherent displacement")
        self.assertIsNone(q.TransmittedLO().leak(LOD_T), msg="an undeclared leak became a zero")

    def test_bias_signs(self):
        """
        `estimate()` at a unit ratio of 1 returns the true channel untouched; an inflated
        ratio hides excess noise while it fakes extra channel loss.
        """
        lo = q.TransmittedLO()
        flat = lo.estimate(0.5, 0.1, 0.5, 1.0)
        self.assertClose(flat[0], 0.5, msg="the transmittance moved at ratio 1")
        self.assertClose(flat[3], 0.0, msg="a gap opened at ratio 1")

        bias = lo.estimate(0.5, 0.1, 0.5, 1.2)
        self.assertLess(bias[1], 0.1, msg="the estimate did not fall")
        self.assertGreater(bias[3], 0.0, msg="the hidden noise is not positive")
        self.assertGreater(bias[4], 0.0, msg="the apparent loss is not positive")

    def test_state_refused(self):
        """
        `covariance()` returns a bona fide Gaussian state at a unit ratio of 1 and refuses
        one a misnormalisation has pushed below the vacuum.
        """
        lo = q.TransmittedLO()
        state = lo.covariance(5.0, 0.5, 0.1, 0.5, 1.0)
        self.assertPhysical(state.cov, msg="the honest reconstruction is unphysical")
        self.assertFails(
            ValueError,
            "estimated excess noise",
            lambda: lo.covariance(5.0, 0.5, 0.1, 0.5, 2.0),
            msg="a channel quieter than the vacuum was built",
        )

    def test_monitor_resolves(self):
        """
        The monitor flags a 20% misnormalisation and not a 1e-6 one, and the noise it still
        hides falls with the sample count rather than with the attack.
        """
        lo = q.TransmittedLO()
        seen = lo.monitor(1_000_000, 0.5, 0.5, 0.1, ratio=1.2)
        self.assertTrue(seen[2], msg="a 20% misnormalisation went unseen")

        near = lo.monitor(1_000_000, 0.5, 0.5, 0.1, ratio=1.0 + 1e-6)
        self.assertFalse(near[2], msg="the monitor claims a resolution it has not")

        seq = [lo.monitor(n, 0.5, 0.5, 0.1)[3] for n in (1e4, 1e6, 1e8)]
        self.assertMonotone(seq, rising=False, msg="more samples must hide less")

    def test_local_refused(self):
        """
        `q.Link` refuses a transmitted oscillator by name on the quadrature path and the
        click one alike.
        """
        for det in (q.Homodyne(), q.ClickDetector()):
            mod = q.GaussianModulation() if isinstance(det, q.Homodyne) else None
            link = q.Link(
                modulation=q.DifferentialPhase() if mod is None else mod,
                channel=q.Channel(T=0.5),
                bob=q.Bob(detector=det, lo=q.TransmittedLO()),
            )
            self.assertFails(
                NotImplementedError,
                "locally generated oscillator",
                link.run,
                msg=f"{type(det).__name__} accepted a transmitted oscillator",
            )

    def test_oscillator_domains(self):
        """
        `q.TransmittedLO` refuses an out-of-range photon number, multiplexer transmittance,
        extinction and delay, naming the parameter refused.
        """
        self.assertFails(ValueError, "photons", lambda: q.TransmittedLO(photons=0.0), msg="no light")
        self.assertFails(ValueError, "mux", lambda: q.TransmittedLO(mux=1.5), msg="gain in an arm")
        self.assertFails(
            ValueError,
            "extinction",
            lambda: q.TransmittedLO(extinction=0.0),
            msg="0 dB is not a suppression",
        )
        self.assertFails(
            ValueError,
            "delay",
            lambda: q.TransmittedLO(delay=-1e-9),
            msg="a past pulse",
        )


class QubitRelay(Question):
    def test_matches_engine(self):
        """
        The swap's rate is `qkd._core`'s from the same forward model, to the last bit.
        """
        got = midpoint().run()
        self.assertClose(got.key_rate, engine(2.0 * HALF), atol=0.0, msg="the surface drifted")
        self.assertGreater(got.key_rate, 0.0, msg="a metro span distils nothing")

    def test_reports_bounds(self):
        """
        The result carries the announcement gain, its error rate and the three
        single-photon-pair quantities the decoy grid bounds, all finite.
        """
        got = midpoint().run()
        self.assertFinite((got.p_click, got.qber, got.y1, got.e1, got.q1), msg="a row is not finite")
        self.assertLess(got.q1, got.p_click, msg="the pair gain is part of the total")
        self.assertGreater(got.e1, got.qber, msg="a bound below the observed rate")

    def test_bound_under(self):
        """
        `y1` from the decoy grid is a BOUND: it sits under `mdi_yield`'s forward-model
        value, and above half of it.
        """
        eta = DET * arms(2.0 * HALF)
        truth = _core.mdi_yield(eta, eta, DARK, MISALIGN)[0]
        got = midpoint().run()
        self.assertLess(got.y1, truth, msg="the bound passed the forward model")
        self.assertGreater(got.y1, 0.5 * truth, msg="the bound is vacuous")

    def test_reach_falls(self):
        """
        The rate falls with the span and is zero at an 800 km total.
        """
        seq = [midpoint(channels=(q.Fiber(length=km, alpha=ALPHA),) * 2).run().key_rate for km in (10.0, 40.0, 80.0)]
        self.assertMonotone(seq, rising=False, msg="distance bought rate")

        far = midpoint(channels=(q.Fiber(length=400.0, alpha=ALPHA),) * 2).run()
        self.assertClose(far.key_rate, 0.0, msg="an 800 km span still distils")

    def test_explain_labels(self):
        """
        `explain()` labels every quantity, names the measurement, and reports the sifting
        factor as absent rather than as a number.
        """
        info = midpoint().explain()
        self.assertEqual(
            info["measurement"],
            "bell (four threshold detectors)",
            msg="the midpoint is not named",
        )
        self.assertIsNone(info["sift"]["value"], msg="a sifting factor was charged")
        self.assertEqual(info["sift"]["label"], "absent", msg="mislabelled sifting")

        bad = [k for k, v in info.items() if isinstance(v, dict) and "label" not in v]
        self.assertEqual(bad, [], msg=f"rows with no provenance: {bad}")

    def test_two_misalignments(self):
        """
        Key and test basis carry separate misalignments: worsening only the test one lowers
        the rate and improving only the key one raises it.
        """
        base = midpoint().run().key_rate
        worse = midpoint(relay=analyser(misalign_test=0.04)).run().key_rate
        self.assertLess(worse, base, msg="the test basis reached no term")
        self.assertGreater(
            midpoint(relay=analyser(misalign=0.001)).run().key_rate,
            base,
            msg="the key basis reached no term",
        )

    def test_covariance_kept(self):
        """
        A `q.BellDetector` still runs the covariance path, reporting `chi` and no
        counting row.
        """
        swap = q.Swap(
            alice=q.Sender(modulation=q.GaussianModulation(v_a=5.0)),
            bob=q.Sender(modulation=q.GaussianModulation(v_a=5.0)),
            relay=q.Relay(bell=q.BellDetector(eta=0.98, v_el=0.01)),
            channels=(q.Channel(T=0.9, xi=0.002, ref="input"),) * 2,
        )
        got = swap.run()
        self.assertIsNotNone(got.chi, msg="the covariance branch lost its noise")
        self.assertIsNone(got.p_click, msg="a counting row leaked into it")


class Midpoint(Question):
    def test_gaussian_refused(self):
        """
        A Gaussian-modulated sender into a counting midpoint is refused, and a keyed sender
        into a covariance one is refused by the sibling message.
        """
        self.assertFails(
            NotImplementedError,
            "q.BasisKeying",
            midpoint(alice=q.Sender(modulation=q.GaussianModulation())).run,
            msg="a Gaussian sender reached the coincidence bound",
        )

        swap = q.Swap(
            alice=sender(),
            bob=sender(),
            relay=q.Relay(bell=q.BellDetector(eta=0.9, v_el=0.05)),
            channels=(q.Fiber(length=HALF),) * 2,
        )
        self.assertFails(
            NotImplementedError,
            "q.BellAnalyser",
            swap.run,
            msg="the covariance refusal does not name the other branch",
        )

    def test_carrier_refused(self):
        """
        `q.PolarisationKeying` is refused: it derives a misalignment from its own frame
        while `q.BellAnalyser` already carries two, so the same error would be charged twice.
        """
        self.assertFails(
            NotImplementedError,
            "charged twice",
            midpoint(alice=q.Sender(modulation=q.PolarisationKeying())).run,
            msg="two misalignment models composed silently",
        )

    def test_variant_refused(self):
        """
        Three bases and the pair announcement are both refused: the midpoint bound reads
        one key basis against one test basis.
        """
        for kw in (dict(bases=3, sift=1.0), dict(announce="pair")):
            party = q.Sender(modulation=q.BasisKeying(decoy=q.Decoy(), **kw))
            self.assertFails(
                NotImplementedError,
                "two conjugate bases",
                midpoint(alice=party).run,
                msg=f"{kw} reached the midpoint bound",
            )

    def test_sifting_stated(self):
        """
        A sender carrying any sifting factor but 1 is refused: the key-basis gain already
        counts key-basis rounds only.
        """
        party = q.Sender(modulation=q.BasisKeying(decoy=q.Decoy(), sift=0.5))
        self.assertFails(
            ValueError,
            "charge the same sifting twice",
            midpoint(alice=party).run,
            msg="a sifting factor was applied on top of the gain",
        )

    def test_misalign_required(self):
        """
        Neither misalignment defaults and neither is derived, so a `q.BellAnalyser`
        carrying none, or one number for both, is refused before it runs.
        """
        bare = q.Relay(bell=q.BellAnalyser(eta=DET, dark=DARK))
        self.assertFails(
            NotImplementedError,
            "Hong-Ou-Mandel",
            midpoint(relay=bare).run,
            msg="an underived misalignment was invented",
        )

        half = q.Relay(bell=q.BellAnalyser(eta=DET, dark=DARK, misalign=MISALIGN))
        self.assertFails(
            NotImplementedError,
            "misalign_test",
            midpoint(relay=half).run,
            msg="one number served for both bases",
        )

    def test_extras_refused(self):
        """
        A pulse width, a correlated environment, an excess noise and a security model from
        another family each reach no term of this bound and are refused by name.
        """
        wide = q.Sender(modulation=q.BasisKeying(decoy=q.Decoy(), sift=1.0), pulse=1e-9)
        self.assertFails(
            NotImplementedError,
            "mode model",
            midpoint(alice=wide).run,
            msg="a pulse width was accepted and dropped",
        )
        self.assertFails(
            NotImplementedError,
            "q.CorrelatedEnvironment",
            midpoint(env=q.CorrelatedEnvironment(x=0.1, p=0.1)).run,
            msg="a Gaussian environment reached a counting bound",
        )
        self.assertFails(
            ValueError,
            "excess noise",
            midpoint(channels=(q.Channel(T=0.3, xi=0.01, ref="input"),) * 2).run,
            msg="an excess noise reached a threshold detector",
        )
        self.assertFails(
            NotImplementedError,
            "q.TestBasisBound",
            midpoint(security=q.Asymptotic()).run,
            msg="the wrong security model ran",
        )


class Refusals(Question):
    def test_sixstate_decoy(self):
        """
        Six-state accepts a key block and the engine refuses the combination: the decoy
        layer carries the photon-number inversion across but not the statistical transfer
        onto one key-basis population.
        """
        link = keyed(q.SplittingAttack(block=q.KeyBlock(n=1e10)), bases=3)
        self.assertFails(
            NotImplementedError,
            "STATISTICAL TRANSFER",
            link.run,
            msg="a weak-coherent six-state length was certified",
        )
        self.assertFails(
            NotImplementedError,
            "Scarani & Renner",
            link.explain,
            msg="explain() reported an asymptotic plan for a finite question",
        )

    def test_sixstate_asymptotic(self):
        """
        Without a block the same six-state link still runs and still reports three bases.
        """
        got = keyed(q.SplittingAttack(f=1.22), bases=3).run()
        self.assertGreater(got.key_rate, 0.0, msg="the asymptotic rate was lost")
        self.assertEqual(got.explain["bases"]["value"], 3, msg="the third basis went missing")

    def test_sarg_refused(self):
        """
        SARG04 accepts a key block and the engine refuses it, naming three missing pieces.
        """
        link = keyed(q.SplittingAttack(block=q.KeyBlock(n=1e10)), announce="pair")
        self.assertFails(
            NotImplementedError,
            "no published analysis states a key length",
            link.run,
            msg="a SARG04 key length was reported",
        )

    def test_b92_branches(self):
        """
        Both two-state branches refuse a key block and the two messages differ: the plain
        one needs a new statistical statement, the strong-reference one is malformed.
        """
        block = q.KeyBlock(n=1e10)
        self.assertFails(
            NotImplementedError,
            "joint confidence region",
            two_state(q.DiscriminationBound(block=block)).run,
            msg="a plain B92 length was reported",
        )
        self.assertFails(
            NotImplementedError,
            "malformed rather than unimplemented",
            two_state(q.DiscriminationBound(e_phase=0.05, block=block), ref=True).run,
            msg="a strong-reference B92 length was reported",
        )

    def test_b92_asymptotic(self):
        """
        Without a block the two-state link runs, and `q.DiscriminationBound` refuses
        anything but a `q.KeyBlock` in the slot.
        """
        got = two_state(q.DiscriminationBound(f=1.16)).run()
        self.assertGreater(got.key_rate, 0.0, msg="the asymptotic rate was lost")
        self.assertFails(
            ValueError,
            "q.KeyBlock",
            lambda: q.DiscriminationBound(block=1e10),
            msg="a bare block size was accepted",
        )

    def test_bb84_untouched(self):
        """
        BB84's own finite-key length still runs, and its rate is that length over the block.
        """
        got = keyed(q.SplittingAttack(f=1.22, block=q.KeyBlock(n=1e10))).run()
        self.assertGreater(got.key_length, 0.0, msg="the BB84 length was lost")
        self.assertClose(got.key_rate, got.key_length / 1e10, msg="rate is length over the block")


# Every exam below is the COMPONENT's own -- its validation, its readers and the
# engine it reaches. What q.Link and q.Swap do with each is their files' to pin.

# Tamaki & Lutkenhaus's own optimum at zero loss: amplitude overlap 0.68125,
# which mu = -ln(c)/2 spells, and the depolarising rate their Fig. 2 quotes.
PLAIN_MU = 0.19187
PLAIN_P = 0.02

# One certified run small enough to be quick: four states, cutoff 6, six
# Frank-Wolfe steps, a 20 km span at 0.02 dB/km and 0.01 excess noise.
DM_NC = 6
DM_STEPS = 6
DM_ETA = 10.0 ** (-0.02 * 20.0)
DM_XI = 0.01


class Receiver(Question):
    def test_symmetric_unchanged(self):
        """
        A `q.ClickDetector` with no partner is the same detector twice, and its background
        yield is the $1 - (1 - d)^2$ `q.Link` already derives.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6)
        one, two = det.both()
        self.assertClose(
            det.background(),
            1.0 - (1.0 - 1e-6) ** 2,
            msg="the symmetric two-gate yield moved",
        )
        self.assertIs(one, two, msg="a partnerless detector was not itself twice")
        self.assertEqual(det.pair(), (0.2, 0.2), msg="a symmetric pair split")

    def test_partner_read(self):
        """
        `pair()` hands `q.attacks` the two efficiencies larger first, and `background()`
        reads both dark counts.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6, partner=q.ClickDetector(eta=0.1, dark=2e-6))
        self.assertEqual(det.pair(), (0.2, 0.1), msg="pair() lost the ordering")
        self.assertClose(
            det.background(),
            1.0 - (1.0 - 1e-6) * (1.0 - 2e-6),
            msg="the two dark counts were not both read",
        )
        self.assertClose(
            q.attacks.mismatch_rate(*det.pair(), 0.01, 0.02),
            q.attacks.mismatch_rate(0.2, 0.1, 0.01, 0.02),
            msg="pair() into attacks.mismatch_rate",
        )

    def test_partner_shallow(self):
        """
        A partner carries no partner of its own, and anything but a `q.ClickDetector` in
        the slot is refused.
        """

        self.assertFails(
            ValueError,
            "there is no third",
            lambda: q.ClickDetector(partner=q.ClickDetector(partner=q.ClickDetector())),
            msg="a chain of detectors was accepted",
        )
        self.assertFails(
            ValueError,
            "q.ClickDetector",
            lambda: q.ClickDetector(partner=0.1),
            msg="a bare efficiency stood in for a detector",
        )

    def test_range_declared(self):
        """
        The quadrature receivers default to no linear range and no bandwidth, and `clip`
        is a HALF-range: `full_scale()` is twice it.
        """
        bare = q.Homodyne()
        self.assertIsNone(bare.clip, msg="a linear range was invented")
        self.assertIsNone(bare.full_scale(), msg="an undeclared range scaled")
        self.assertIsNone(bare.bandwidth, msg="a bandwidth was invented")
        self.assertClose(
            q.Heterodyne(clip=5.0).full_scale(),
            10.0,
            msg="full_scale() is not twice the clip",
        )

    def test_range_charged(self):
        """
        `full_scale()` is `budget.adc`'s ratio and `bandwidth` is the RIN rows', while an
        undeclared intensity noise is neither charged nor refused.
        """
        det = q.Heterodyne(clip=5.0, bandwidth=100e6)
        self.assertClose(
            q.budget.adc(12, 0.5, ratio=det.full_scale()),
            q.budget.adc(12, 0.5),
            msg="full_scale() against budget.adc's default ratio",
        )

        laser = q.Laser(rin=-155.0)
        rows = q.budget.assemble(v_a=5.0, t=0.5, rin=laser.rin, bandwidth=det.bandwidth).entries
        names = [e.source for e in rows]
        self.assertEqual(names, ["rin_sig", "rin_lo"], msg="the bandwidth reached no RIN row")
        self.assertIsNone(
            q.Laser().rin,
            msg="q.Laser().rin",
        )

    def test_range_positive(self):
        """
        `clip` and `bandwidth` refuse zero and negatives: a receiver linear over nothing
        is not an idealisation of one that has a range.
        """
        for kw in (dict(clip=0.0), dict(bandwidth=0.0), dict(clip=-1.0)):
            self.assertFails(
                ValueError,
                "must be positive",
                lambda kw=kw: q.Homodyne(**kw),
                msg=f"{kw} was accepted",
            )


class Carrier(Question):
    def test_beat_signed(self):
        """
        The offset is Alice's carrier less Bob's, signed, and reverses when the two lasers
        swap places.
        """
        lo = q.LocalLO(carrier=193.4e12)
        above = q.Laser(carrier=193.4e12 + 4e6)
        below = q.Laser(carrier=193.4e12 - 4e6)
        self.assertClose(lo.beat(above), 4e6, msg="the beat lost its magnitude")
        self.assertClose(lo.beat(below), -4e6, msg="the beat lost its sign")

    def test_beat_unclamped(self):
        """
        A beat past the Nyquist frequency comes back unclamped, and the pipeline refuses
        it rather than aliasing it.
        """
        lo = q.LocalLO(carrier=193.4e12)
        far = q.Laser(carrier=193.4e12 + 90e6)
        self.assertClose(lo.beat(far), 90e6, msg="the offset was clamped")
        self.assertFails(
            ValueError,
            "2*|cfo| < symbol_rate",
            _core.run_symbols,
            1000,
            0,
            5.0,
            0.5,
            0.01,
            0.6,
            0.1,
            10e3,
            10e3,
            lo.beat(far),
            100e6,
            12.0,
            1.8,
            32,
            False,
            msg="the pipeline aliased an offset past Nyquist",
        )

    def test_beat_paired(self):
        """
        An offset is a difference, so either end missing a carrier is refused by name.
        """

        self.assertFails(
            ValueError,
            "q.LocalLO carries no carrier",
            lambda: q.LocalLO().beat(q.Laser(carrier=193.4e12)),
            msg="a one-sided carrier produced an offset",
        )
        self.assertFails(
            ValueError,
            "q.Laser carries no carrier",
            lambda: q.LocalLO(carrier=193.4e12).beat(q.Laser()),
            msg="a one-sided carrier produced an offset",
        )

    def test_carrier_optional(self):
        """
        Neither `q.Laser` nor `q.LocalLO` carries a default centre frequency.
        """

        self.assertIsNone(q.Laser().carrier, msg="a carrier was invented")
        self.assertIsNone(q.LocalLO().carrier, msg="a carrier was invented")


class Certified(Question):
    def test_steps_apart(self):
        """
        `certify()` returns the step-2 proof and the step-1 Frank-Wolfe value as separate
        fields, the second above the first, and `key` is `bound` less error correction.
        """
        got = q.CertifiedBound(cutoff=DM_NC, steps=DM_STEPS).certify(
            q.PhaseShiftKeying(states=4, alpha=0.7), DM_ETA, DM_XI
        )
        self.assertGreater(got.key, 0.0, msg="the certified rate died")
        self.assertGreater(got.upper, got.key, msg="step 1 did not bound step 2 from above")
        self.assertClose(
            got.key,
            got.bound - got.p_pass * got.delta_ec,
            msg="key vs bound less p_pass*delta_ec",
        )

    def test_upper_barred(self):
        """
        A Certificate has no `key_rate` or `rate`: each raises rather than resolving to
        either of the two steps.
        """
        got = q.CertifiedBound(cutoff=DM_NC, steps=DM_STEPS).certify(
            q.PhaseShiftKeying(states=4, alpha=0.7), DM_ETA, DM_XI
        )

        for name in ("key_rate", "rate"):
            self.assertFails(
                AttributeError,
                "proves nothing",
                lambda name=name: getattr(got, name),
                msg=f"{name} resolved to one of the two steps",
            )

    def test_matches_engine(self):
        """
        The component reaches `_core.dm_secure` with the same arguments in the same order.
        """
        sec = q.CertifiedBound(cutoff=DM_NC, cut=0.1, phase=0.05, steps=DM_STEPS)
        mod = q.PhaseShiftKeying(states=4, alpha=0.7)
        raw = _core.dm_secure(4, 0.7, DM_ETA, DM_XI, DM_NC, 0.1, 0.05, 0.95, 1e-10, DM_STEPS, 1e-8)
        got = sec.certify(mod, DM_ETA, DM_XI)
        self.assertEqual(
            (got.key, got.bound, got.upper, got.p_pass, got.delta_ec),
            raw[:5],
            msg="certify() argument order",
        )

    def test_cutoff_declared(self):
        """
        The cutoff has no default -- the bound is rigorous given it -- and the engine's own
        range refuses one below 2.
        """

        self.assertFails(
            TypeError,
            "cutoff",
            q.CertifiedBound,
            msg="a photon-number cutoff was taken silently",
        )
        self.assertFails(
            ValueError,
            "must be at least 2",
            lambda: q.CertifiedBound(cutoff=1),
            msg="a one-level Fock space was accepted",
        )

    def test_pure_loss_refused(self):
        """
        Pure loss is refused by the engine: at xi = 0 the certificate is valid and
        worthless, so it raises rather than returning it.
        """

        self.assertFails(
            ValueError,
            "raise xi off zero",
            lambda: q.CertifiedBound(cutoff=DM_NC, steps=DM_STEPS).certify(
                q.PhaseShiftKeying(states=4, alpha=0.7), DM_ETA, 0.0
            ),
            msg="a rank-deficient certificate was returned as a rate",
        )


class Sources(Question):
    def test_single_reaches_plain(self):
        """
        `source='single'` reaches `b92_plain`, with `mu` spelling the amplitude overlap
        e^-2mu that engine is written in.
        """
        mod = q.TwoStateKeying(mu=PLAIN_MU, source="single", depol=PLAIN_P)
        self.assertClose(mod.overlap, math.exp(-2.0 * PLAIN_MU), msg="the overlap moved")

        got = _core.b92_plain(mod.overlap, 0.0, mod.depol, 1.0)
        self.assertGreater(got[3], 0.0, msg="Tamaki's own operating point died")

    def test_source_paired(self):
        """
        The depolarising channel belongs to the single-photon branch alone, and each branch
        refuses the other's description by name.
        """

        self.assertFails(
            ValueError,
            "needs depol",
            lambda: q.TwoStateKeying(source="single"),
            msg="a single-photon branch ran with no channel",
        )
        self.assertFails(
            ValueError,
            "reaches no term of the coherent branch",
            lambda: q.TwoStateKeying(depol=0.02),
            msg="a depolarising rate was accepted and dropped",
        )
        self.assertFails(
            NotImplementedError,
            "different proofs, not two sources",
            lambda: q.TwoStateKeying(source="single", depol=0.02, reference=True),
            msg="two proofs were composed as one",
        )

    def test_coherent_unchanged(self):
        """
        The default source is the coherent pair, with no depolarising rate.
        """
        mod = q.TwoStateKeying()
        self.assertEqual(mod.source, "coherent", msg="the default source moved")
        self.assertIsNone(mod.depol, msg="a depolarising rate was invented")

    def test_block_per_state(self):
        """
        `per_state(n)` divides `eps_sec` by the announced Bell-state count, and that
        divided budget is what reaches `_core.mdi_length`.
        """
        blk = q.RelayBlock(eps_sec=1e-10)
        self.assertClose(blk.per_state(1), 1e-10, msg="one announcement was split")
        self.assertClose(blk.per_state(2), 5e-11, msg="two announcements were not")
        self.assertClose(
            _core.mdi_length(1e5, 3e5, 0.03, 1e6, 0.015, 1.16, blk.per_state(2), blk.eps_cor),
            _core.mdi_length(1e5, 3e5, 0.03, 1e6, 0.015, 1.16, 5e-11, 1e-15),
            msg="the divided budget did not reach the engine",
        )

    def test_block_counted(self):
        """
        The announced Bell-state count has no default, and a linear-optics analyser
        refuses four.
        """

        self.assertFails(
            ValueError,
            "part of the claim",
            lambda: q.RelayBlock().per_state(None),
            msg="an unstated count was taken for one",
        )
        self.assertIsNone(
            q.BellAnalyser(eta=DET, dark=DARK).states,
            msg="an announcement count was invented",
        )
        self.assertFails(
            ValueError,
            "no third announcement",
            lambda: q.BellAnalyser(eta=DET, dark=DARK, states=4),
            msg="a linear-optics analyser announced four Bell states",
        )

    def test_bias_apart(self):
        """
        A sender's key-basis probability is its own field, undefaulted and refused outside
        (0, 1): folding it into `sift` would charge a sifting the midpoint rate excludes.
        """
        keyed = q.BasisKeying(decoy=q.Decoy(), sift=1.0, bias=0.9)
        self.assertClose(keyed.sift, 1.0, msg="the biased-basis limit moved")
        self.assertClose(keyed.bias, 0.9, msg="the bias did not survive")
        self.assertIsNone(q.BasisKeying(decoy=q.Decoy()).bias, msg="a bias was invented")

        for bad in (0.0, 1.0):
            self.assertFails(
                ValueError,
                "bias must be in",
                lambda bad=bad: q.BasisKeying(decoy=q.Decoy(), bias=bad),
                msg=f"bias={bad} left one of the two bases empty",
            )

    def test_block_typed(self):
        """
        A midpoint's block is not a `q.KeyBlock`, and the slot refuses one by name rather
        than reading the fields they share.
        """

        self.assertFails(
            ValueError,
            "counts pulse PAIRS",
            lambda: q.TestBasisBound(block=q.KeyBlock()),
            msg="a single-sender block described a midpoint",
        )
        self.assertIsNone(q.TestBasisBound().block, msg="the asymptotic default moved")
        self.assertClose(q.RelayBlock().code, 0.9, msg="the code share moved")


class Attenuation(Question):
    def test_round_trip(self):
        """
        The forward no-click probabilities fed back through the inversion bracket the
        photon-number distribution they came from, with the vacuum weight exact.
        """
        att = q.DecoyAttenuator()
        det = q.ClickDetector(eta=1.0, dark=1e-6)
        probs = [0.6, 0.3, 0.08, 0.02]
        got = att.bounds(det, att.noclick(det, probs))
        self.assertClose(got.p0, 0.6, atol=1e-6, msg="the vacuum weight is exact")
        self.assertLess(got.p1_lo, 0.3, msg="the one-photon bracket lost its floor")
        self.assertGreater(got.p1_hi, 0.3, msg="the one-photon bracket lost its cap")
        self.assertLess(got.p2_lo, 0.08, msg="the two-photon bracket lost its floor")
        self.assertGreater(got.p2_hi, 0.08, msg="the two-photon bracket lost its cap")

    def test_transparent_first(self):
        """
        The first setting anchors $x_0$ on $f(0)$, so an attenuating first setting is
        refused here and an imperfect detector is refused by the engine.
        """

        self.assertFails(
            ValueError,
            "must be exactly 1.0",
            lambda: q.DecoyAttenuator(settings=(0.9, 0.8, 0.7)),
            msg="an attenuating first setting was accepted",
        )

        att = q.DecoyAttenuator()
        dim = q.ClickDetector(eta=0.2, dark=1e-6)
        self.assertFails(
            ValueError,
            "Hand the detector efficiency to the channel",
            lambda: att.bounds(dim, (0.9, 0.9, 0.9)),
            msg="an imperfect detector was absorbed silently",
        )

    def test_probes_ordered(self):
        """
        The two probing settings are the proposition's c1 below c2, written as
        transmittances, so they must decrease strictly below the transparent one.
        """
        for bad in ((1.0, 0.5, 0.9), (1.0, 0.9, 0.9), (1.0, 0.9)):
            self.assertFails(
                ValueError,
                "settings",
                lambda bad=bad: q.DecoyAttenuator(settings=bad),
                msg=f"{bad} was accepted as a probing ladder",
            )


class Flaws(Question):
    def test_flaw_shipped(self):
        """
        `q.SourceFlaw` is in `__all__` and present on the package.
        """

        self.assertIn("SourceFlaw", q.__all__, msg="absent from __all__")
        self.assertTrue(hasattr(q, "SourceFlaw"), msg="promised and missing")

    def test_tilt_stated(self):
        """
        Neither angle defaults, so the configuration in which Bob's modulator cancels
        Alice's cannot be fallen into by leaving a field out; `opposed()` is tilt = -delta.
        """

        self.assertFails(
            TypeError,
            "tilt",
            lambda: q.SourceFlaw(delta=FLAW),
            msg="tilt defaulted",
        )
        self.assertFails(
            TypeError,
            "delta",
            lambda: q.SourceFlaw(),
            msg="both angles defaulted",
        )

        pair = q.SourceFlaw.opposed(FLAW)
        self.assertClose(pair.tilt, -FLAW, msg="opposed() is not tilt = -delta")

    def test_flaws_cancel(self):
        """
        At tilt = 0 the phase error does not move with the flaw at all, and at
        tilt = -delta it is sin^2(delta/2).
        """
        flat = [q.SourceFlaw(delta=d, tilt=0.0).phase(ETA, GATE) for d in FLAWS]

        for got in flat[1:]:
            self.assertClose(got, flat[0], msg="tilt = 0 stopped being flat in the flaw")

        for delta in FLAWS:
            want = math.sin(delta / 2.0) ** 2
            self.assertClose(
                q.SourceFlaw.opposed(delta).phase(ETA, 0.0),
                want,
                msg=f"the published tilt lost sin^2(delta/2) at delta = {delta}",
            )

    def test_engine_match(self):
        """
        Every method reaches its engine entry point with the two angles in the engine's
        own argument order.
        """
        pair = q.SourceFlaw.opposed(FLAW)
        args = (pair.delta, ETA, GATE, pair.tilt)
        self.assertEqual(pair.angles(), _core.flaw_angles(pair.delta), msg="angles() drifted")
        self.assertEqual(pair.triangle(), _core.flaw_triangle(pair.delta), msg="triangle() drifted")
        self.assertEqual(pair.fidelity(), _core.flaw_fidelity(pair.delta), msg="fidelity() drifted")
        self.assertEqual(pair.channel(ETA, GATE), _core.flaw_channel(*args), msg="channel()")
        self.assertEqual(pair.phase(ETA, GATE), _core.flaw_phase(*args), msg="phase() drifted")
        self.assertEqual(
            pair.tolerant(ETA, GATE, FEC, 0.5),
            _core.flaw_tolerant(*args, FEC, 0.5),
            msg="tolerant() drifted",
        )
        self.assertEqual(
            pair.standard(ETA, GATE, FEC, 0.5),
            _core.flaw_standard(*args, FEC, 0.5),
            msg="standard() drifted",
        )

    def test_analyses_apart(self):
        """
        The two four-tuples are not slot-aligned: the fourth of `tolerant` is a phase error
        rate and the fourth of `standard` is a coin imbalance, whose GLLP bound sits above
        the exact value.
        """
        pair = q.SourceFlaw.opposed(FLAW)
        exact = pair.tolerant(ETA, GATE, FEC, 0.5)
        worst = pair.standard(ETA, GATE, FEC, 0.5)
        self.assertEqual(exact[1:3], worst[1:3], msg="the two read one channel")
        self.assertLess(worst[3], exact[3], msg="the coin was returned as a phase error")
        self.assertGreater(
            _core.flaw_gllp(worst[2], worst[3]),
            exact[3],
            msg="the coin's GLLP bound vs the exact value",
        )
        self.assertGreater(exact[0], worst[0], msg="the worst case bought the longer key")

    def test_decoy_seam(self):
        """
        `coin()` is the only entry point taking a supplied `y1`; the two rate methods build
        their own and expose no seam a decoy bound could enter.
        """
        taken = inspect.signature(q.SourceFlaw.coin).parameters
        self.assertIn("y1", taken, msg="the one y1 seam closed")

        for name in ("tolerant", "standard", "phase", "channel"):
            names = inspect.signature(getattr(q.SourceFlaw, name)).parameters
            self.assertNotIn(
                "y1",
                names,
                msg=f"{name}() takes a y1",
            )

    def test_tilt_signed(self):
        """
        The published configuration is a SIGNED tilt = -delta, while the only angle a
        `q.Link` receiver carries is an unsigned misalignment refused below zero.
        """

        self.assertClose(q.SourceFlaw.opposed(FLAW).tilt, -FLAW, msg="the published frame stopped being signed")
        self.assertFails(
            ValueError,
            "misalign must be in",
            lambda: q.BasisAnalyser(misalign=-FLAW),
            msg="a receiver took a signed frame after all",
        )


# The three asymptotic rates with no component tree, at one operating point
# each. PAIR_* is Xie's symmetric asynchronous-MDI station, as test/pairing.py
# carries it, 150 km end to end; RR_* is Yin's variable-delay receiver at a
# metro loss, as test/rrdps.py carries it; COW_* is the peak test/sdp.py
# measures at zero distance under the certified bound. RATE_FEC is the 1.1 all
# three of those exams reconcile at, which is not this file's 1.16.
RATE_FEC = 1.1
PAIR_ETA = _core.decoy_eta(0.165, 75.0, 0.70)
PAIR_MU = 0.4
PAIR_DARK = 1e-8
PAIR_MISS = (1.0 - math.cos(math.pi / 10.0)) / 2.0
SPAN = 10**4

# Zeng's own span estimate, and the drift figure Xie measures over 402 km of
# fibre, in rad/s at a 625 MHz clock.
CLOCK = 625e6
COHERE = 5e-6
DRIFT = 8000.0

RR_TRAIN = 32
RR_MU = 0.005
RR_ETA = 0.1
RR_DARK = 1e-6
RR_MISS = 0.015
RR_NU = 3

COW_MU = 0.062503
COW_SPLIT = 0.4389
COW_ETA = 0.8
COW_DARK = 1e-8

# Where the analytic bound has long since aborted: 100 km of standard fibre into
# the same detector, at an intensity the monitor cannot certify.
GONE_ETA = 0.8 * 10.0 ** (-0.2 * 100.0 / 10.0)
GONE_MU = 0.01


def pairing(span=SPAN, **kw):
    """
    One mode-pairing rate at the station above, span free.
    """

    return q.rates.keyrate(
        "pairing",
        eta=PAIR_ETA,
        mu=PAIR_MU,
        dark=PAIR_DARK,
        span=span,
        misalign=PAIR_MISS,
        f_ec=RATE_FEC,
        **kw,
    )


def train(cost="tagged", l=RR_TRAIN, nu=RR_NU):
    """
    One RRDPS rate at the receiver above, packet length, photon-number threshold and cost
    assembly free.
    """

    return q.rates.keyrate(
        "rrdps",
        l=l,
        mu=RR_MU,
        eta=RR_ETA,
        dark=RR_DARK,
        e_mis=RR_MISS,
        nu_th=nu,
        f_ec=RATE_FEC,
        cost=cost,
    )


def vacuum(bound="certified", eta=COW_ETA, mu=COW_MU):
    """
    One COW' rate at the operating point above, phase-error bound free.
    """

    return q.rates.keyrate(
        "cow-vacuum",
        mu=mu,
        t_b=COW_SPLIT,
        eta=eta,
        dark=COW_DARK,
        f_ec=RATE_FEC,
        bound=bound,
    )


class Rates(Question):
    def test_rates_shipped(self):
        """
        `q.rates` is in `__all__` and registers exactly cow-vacuum, pairing and rrdps, each
        catalogued.
        """

        self.assertIn("rates", q.__all__, msg="absent from __all__")
        self.assertEqual(
            q.rates.families(),
            ("cow-vacuum", "pairing", "rrdps"),
            msg="the register moved",
        )
        for name in q.rates.families():
            self.assertIn(name, dict((r[0], r) for r in q.rates.catalogue()), msg=f"{name} is uncatalogued")

    def test_units_stated(self):
        """
        Each family names its own per-shot unit on every rate: a round for pairing, a pulse
        for RRDPS, and a pulse plus its two-pulse sequence for COW'.
        """

        self.assertIn("ROUND", pairing().units, msg="mode pairing stopped naming the round")
        self.assertIn("PULSE", train().units, msg="RRDPS stopped naming the pulse")
        self.assertIn("PULSE", vacuum().units, msg="COW' stopped naming the pulse")
        self.assertIn("sequence", vacuum().units, msg="the two-pulse sequence went unstated")

    def test_layer_apart(self):
        """
        No `explain()` row in this layer is an epsilon, and the finite-key length for the
        same family comes from `q.security.keylength`, in bits.
        """
        rows = list(pairing().explain()) + list(train().explain()) + list(vacuum().explain())
        charged = [name for name in rows if name.startswith("eps")]
        self.assertEqual(charged, [], msg=f"an epsilon reached the rate layer: {charged}")

        led = q.security.compose("pairing", eps_sec=1e-10, eps_cor=1e-15)
        long = q.security.keylength(led, s0=1e6, s11=4e7, phi=0.03, n_z=1e8, e_z=0.02, f_ec=1.1)
        self.assertGreater(long.bits, 0.0, msg="the length layer stopped reaching this family")
        self.assertIn("bits", long.units, msg="a length stopped being bits")

    def test_working_shown(self):
        """
        Every rate carries the engine's intermediate quantities under the engine's own
        names, each row labelled pinned or derived.
        """
        got = pairing()
        self.assertEqual(set(got.values), {"click", "pairs", "sift", "q11", "e_x", "e_z"}, msg="pairing's working")
        self.assertEqual(got.explain()["rate"]["value"], got.rate, msg="the rate row is the rate")
        self.assertEqual(got.explain()["mu"]["label"], "pinned", msg="an argument is not pinned")
        self.assertEqual(got.explain()["q11"]["label"], "derived", msg="a derived row is not derived")


class PairingRate(Question):
    def test_pairing_route(self):
        """
        The route is the rate Zeng's five engine calls compose by hand, bit for bit, with
        the phase error from Ma and Razavi through `mdi_yield`.
        """
        got = pairing()
        sift, e_z = _core.pairing_sift(PAIR_ETA, PAIR_MU, PAIR_DARK)
        pairs = _core.pairing_pairs(_core.pairing_click(PAIR_ETA, PAIR_MU, PAIR_DARK), SPAN)
        single = _core.pairing_single(PAIR_ETA, PAIR_MU, PAIR_DARK)
        e_x = _core.mdi_yield(PAIR_ETA, PAIR_ETA, PAIR_DARK, PAIR_MISS)[1]
        self.assertEqual(
            got.rate,
            _core.pairing_rate(pairs, sift, single, e_x, e_z, RATE_FEC),
            msg="the route stopped being the engine chain",
        )
        self.assertEqual(got.values["e_x"], e_x, msg="the phase error stopped coming from mdi_yield")
        self.assertEqual(got.values["e_z"], e_z, msg="the Z-basis error drifted")

    def test_window_span(self):
        """
        `q.rates.window("pairing", ...)` is `pairing_span`: 3125 rounds at 625 MHz and
        5 us, and a reference not surviving one round is refused.
        """

        self.assertEqual(q.rates.window("pairing", CLOCK, COHERE), 3125, msg="Zeng's own span estimate moved")
        self.assertEqual(
            q.rates.window("pairing", CLOCK, COHERE),
            _core.pairing_span(CLOCK, COHERE),
            msg="window() stopped being pairing_span",
        )
        self.assertFails(
            ValueError,
            "no two rounds can be paired",
            lambda: q.rates.window("pairing", CLOCK, 1e-12),
            msg="a reference that does not survive one round still paired",
        )

    def test_span_buys(self):
        """
        The rate rises with the pairing interval, a span of $10^4$ buying more than 25
        times a span of one round.
        """
        got = [pairing(span=s).rate for s in (1, 10, 100, 1000, SPAN)]
        self.assertMonotone(got, strict=False, msg="the rate stopped rising with the window")
        self.assertGreater(got[-1], 25.0 * got[0], msg=f"the window bought only {got[-1] / got[0]:.3g}x")

    def test_drift_shape(self):
        """
        The drift shape reads `pairing_drift` instead, reporting a mean phase excursion
        and a larger `e_x`, and the yield shape reports no excursion at all.
        """
        got = pairing(drift=DRIFT, clock=CLOCK)
        excursion, e_x = _core.pairing_drift(CLOCK, SPAN, DRIFT, PAIR_MISS)
        self.assertEqual(got.values["excursion"], excursion, msg="the excursion stopped being reported")
        self.assertEqual(got.values["e_x"], e_x, msg="the drift shape stopped reading pairing_drift")
        self.assertGreater(e_x, pairing().values["e_x"], msg="a drifting reference cost nothing")
        self.assertNotIn("excursion", pairing().values, msg="the yield shape reported a window it never read")

    def test_pairing_decoy(self):
        """
        The pair intensity ladder has six ascending rungs summing to one where a round
        carries three intensities, and the four basis labels sum to one, both slots empty
        at the vacuum draw squared.
        """
        probs = (0.5, 0.1, 0.4)
        rungs = q.rates.ladder("pairing", 0.01, PAIR_MU, probs)
        labels = q.rates.bases("pairing", probs)
        self.assertEqual(len(rungs), 6, msg="the pair ladder stopped having six rungs")
        self.assertEqual(rungs, _core.pairing_ladder(0.01, PAIR_MU, probs), msg="ladder() drifted")
        self.assertEqual(labels, _core.pairing_bases(probs), msg="bases() drifted")
        self.assertClose(sum(p for _v, p in rungs), 1.0, msg="the ladder stopped being a distribution")
        self.assertClose(sum(labels), 1.0, msg="the four labels stopped summing to one")
        self.assertClose(labels[2], probs[0] ** 2, msg="both slots empty is not the vacuum draw squared")
        self.assertMonotone([v for v, _p in rungs], strict=False, msg="the ladder stopped ascending")


class RrdpsRate(Question):
    def test_rrdps_route(self):
        """
        The route is Takesue's Eq. (4) over Yin's receiver model, the rate the three engine
        calls compose by hand, with the sifted rate and the bit error rate derived.
        """
        got = train()
        sifted, e_bit = _core.rrdps_counts(RR_TRAIN, RR_MU, RR_ETA, RR_DARK, RR_MISS)
        src = _core.rrdps_src(RR_TRAIN, RR_MU, RR_NU)
        cost = _core.rrdps_tag(sifted, src, _core.rrdps_leak(RR_TRAIN, RR_NU))
        self.assertEqual(
            got.rate,
            _core.rrdps_rate(sifted, e_bit, cost, RATE_FEC, RR_TRAIN),
            msg="the route stopped being the engine chain",
        )
        self.assertEqual(got.values["q"], sifted, msg="the sifted rate drifted")
        self.assertEqual(got.values["e_bit"], e_bit, msg="the bit error rate stopped being an output")

    def test_costs_ordered(self):
        """
        The four assemblies of one privacy-amplification cost are ordered resolved, tagged,
        entropy at this operating point, the collective bound tightens the tagged form, and
        no two return the same number.
        """
        seen = {form: train(cost=form).values["cost"] for form in q.rates.COSTS}
        self.assertLess(seen["resolved"], seen["tagged"], msg="per-photon-number tagging stopped being tighter")
        self.assertLess(seen["tagged"], seen["entropy"], msg="h2 of the mixed error rate stopped being loosest")
        self.assertLess(seen["collective"], seen["tagged"], msg="the collective bound stopped tightening")
        self.assertEqual(len(set(seen.values())), len(q.rates.COSTS), msg="two assemblies returned one number")

    def test_refined_tightens(self):
        """
        Matsuura's refined bound sits between its witness and its tangent, below the closed
        form it is reported beside, and keys where that form reports exactly zero.
        """
        got = q.rates.keyrate("rrdps", l=RR_TRAIN, p_src=0.2, q=0.5, e_bit=0.02, f_ec=RATE_FEC)
        self.assertLess(got.values["cost"], got.values["plain"], msg="the refined bound stopped tightening")
        self.assertLessEqual(got.values["witness"], got.values["cost"] + 1e-12, msg="the witness passed the bound")
        self.assertLessEqual(got.values["cost"], got.values["tangent"] + 1e-12, msg="the bound passed the tangent")
        self.assertEqual(
            _core.rrdps_rate(0.5, 0.02, got.values["plain"], RATE_FEC, RR_TRAIN),
            0.0,
            msg="the closed form still keys here",
        )
        self.assertGreater(got.rate, 0.0, msg="the refined bound stopped certifying key")

    def test_optimum_route(self):
        """
        The optimum's rate is the rate at the packet length and threshold it names, and no
        other length at the same hardware beats it.
        """
        best, nu, rate = q.rates.optimum("rrdps", 64, RR_MU, RR_ETA, RR_DARK, RR_MISS, RATE_FEC)
        self.assertEqual(train(l=best, nu=nu).rate, rate, msg=f"optimum {rate} vs the route at l = {best}")
        for l in (8, 16, 32, 64):
            self.assertGreaterEqual(rate, train(l=l).rate, msg=f"a packet of {l} beat the optimum")


class VacuumRate(Question):
    def test_vacuum_route(self):
        """
        The route is the COW' data line priced at the certified phase error and halved, the
        gain being per two-pulse sequence where the rate is per pulse.
        """
        got = vacuum()
        q0, q1 = _core.sdp_gains(COW_MU, COW_SPLIT, COW_ETA, COW_DARK)
        e_phase = _core.sdp_phase(COW_MU, q0, q1, 1e-9)[0]
        gain, e_bit = _core.sdp_data(COW_MU, COW_SPLIT, COW_ETA, COW_DARK)
        self.assertEqual(got.values["e_phase"], e_phase, msg="the certified bound drifted")
        self.assertEqual(
            got.rate,
            0.5 * _core.cow_rate(gain, e_bit, e_phase, RATE_FEC),
            msg="the halving that makes this per pulse went missing",
        )
        self.assertEqual(got.values["honest"], _core.sdp_honest(COW_MU, COW_SPLIT, COW_ETA, COW_DARK), msg="floor")

    def test_bounds_ordered(self):
        """
        The certified worst case sits below Gao's Cauchy-Schwarz closed form at the same
        record and above the honest value, and the shorter phase error buys the longer key.
        """
        fast = vacuum(bound="certified")
        slow = vacuum(bound="analytic")
        self.assertLess(fast.values["e_phase"], slow.values["e_phase"], msg="the certificate stopped tightening")
        self.assertGreater(fast.rate, slow.rate, msg="the tighter bound bought no key")
        self.assertGreaterEqual(fast.values["e_phase"], fast.values["honest"], msg="the bound fell below the honest")

    def test_abort_raw(self):
        """
        Past the abort the rate is exactly zero and the phase error is reported above one
        half rather than clamped to it.
        """
        got = vacuum(bound="analytic", eta=GONE_ETA, mu=GONE_MU)
        self.assertEqual(got.rate, 0.0, msg="a phase error above 1/2 still keyed")
        self.assertGreater(got.values["e_phase"], 0.5, msg="the abort was clamped away")

    def test_prime_apart(self):
        """
        'cow' names the THREE-sequence protocol `q.Link` runs, so this layer refuses it
        rather than answering with the four-sequence variant, as it refuses 'cow-prime'.
        """

        self.assertFails(
            ValueError,
            "THREE-sequence",
            lambda: q.rates.keyrate("cow", mu=COW_MU, t_b=COW_SPLIT, eta=COW_ETA, dark=COW_DARK, f_ec=RATE_FEC),
            msg="COW was answered by COW'",
        )
        self.assertFails(
            ValueError,
            "q.IntensityKeying",
            lambda: q.rates.describe("cow-prime"),
            msg="the prime spelling did not name the registered one",
        )

    def test_four_sequences(self):
        """
        The Gram matrix is 4x4, symmetric, unit on its diagonal, with the empty sequence
        overlapping at e^-mu and e^-mu/2; a paired protocol has no such alphabet.
        """
        gram = q.rates.overlaps("cow-vacuum", COW_MU)
        self.assertEqual(len(gram), 16, msg="the source stopped having four sequences")
        self.assertEqual(gram, _core.sdp_gram(COW_MU), msg="overlaps() drifted")
        for k in range(4):
            self.assertClose(gram[5 * k], 1.0, msg=f"sequence {k} stopped being normalised")

        for i in range(4):
            for j in range(4):
                self.assertClose(gram[4 * i + j], gram[4 * j + i], msg=f"the Gram matrix is not symmetric at {i},{j}")

        self.assertClose(gram[1], math.exp(-COW_MU), msg="the empty row's overlap with |a>|a> moved")
        self.assertClose(gram[2], math.exp(-COW_MU / 2.0), msg="the empty sequence lost its overlap with a bit")
        self.assertFails(
            NotImplementedError,
            "a fourth source sequence",
            lambda: q.rates.overlaps("pairing", COW_MU),
            msg="a paired protocol was given a sequence alphabet",
        )


class RateGuards(Question):
    def test_shape_whole(self):
        """
        A shape is taken whole: a partial one is refused naming every shape the family
        reads, no argument here having a safe default.
        """

        self.assertFails(
            ValueError,
            "reads (eta, mu, dark, span, misalign, f_ec)",
            lambda: q.rates.keyrate("pairing", eta=PAIR_ETA, mu=PAIR_MU, dark=PAIR_DARK, span=SPAN),
            msg="a pairing rate ran without a reconciliation efficiency",
        )
        self.assertFails(
            ValueError,
            "shape is taken whole",
            lambda: q.rates.keyrate("rrdps", l=RR_TRAIN, mu=RR_MU, eta=RR_ETA),
            msg="an RRDPS rate ran on half a receiver",
        )

    def test_named_stated(self):
        """
        The cost assembly and the phase-error bound are the caller's to state, and an
        unregistered name for either is refused rather than resolved.
        """

        self.assertFails(
            ValueError,
            "cost must be one of",
            lambda: train(cost="gllp"),
            msg="an unregistered cost assembly was assembled",
        )
        self.assertFails(
            ValueError,
            "bound must be one of",
            lambda: vacuum(bound="sdp"),
            msg="an unregistered bound was priced",
        )

    def test_reltol_scoped(self):
        """
        A central-path tolerance is refused by a shape that reaches no solver rather than
        accepted and ignored, and accepted by the branch that does read one.
        """

        self.assertFails(
            ValueError,
            "reaches no solver",
            lambda: pairing(reltol=1e-9),
            msg="a closed form accepted a solver tolerance",
        )
        self.assertGreater(
            q.rates.keyrate(
                "cow-vacuum",
                reltol=1e-7,
                mu=COW_MU,
                t_b=COW_SPLIT,
                eta=COW_ETA,
                dark=COW_DARK,
                f_ec=RATE_FEC,
                bound="certified",
            ).rate,
            0.0,
            msg="the branch that does read one refused it",
        )

    def test_counts_whole(self):
        """
        A span and a packet length are counts of rounds and of pulses, so a fractional one
        is refused rather than truncated.
        """

        self.assertFails(
            ValueError,
            "whole number",
            lambda: pairing(span=1000.5),
            msg="a fractional pairing interval was truncated",
        )
        self.assertFails(
            ValueError,
            "whole number",
            lambda: train(l=32.5),
            msg="a fractional packet length was truncated",
        )

    def test_families_apart(self):
        """
        The pairing window and the packet-length sweep run for one family each and refuse
        the other by name.
        """

        self.assertFails(
            NotImplementedError,
            "a pairing interval",
            lambda: q.rates.window("rrdps", CLOCK, COHERE),
            msg="a packet protocol was given a coherence window",
        )
        self.assertFails(
            NotImplementedError,
            "a free packet length",
            lambda: q.rates.optimum("pairing", 64, RR_MU, RR_ETA, RR_DARK, RR_MISS, RATE_FEC),
            msg="a paired protocol was swept over packet lengths",
        )

    def test_register_named(self):
        """
        A family with a component tree is not registered here, and the refusal names
        `q.Link`, `q.Swap` or `q.PairLink` instead of offering a second route.
        """

        self.assertFails(
            ValueError,
            "no asymptotic rate is registered for 'bb84'",
            lambda: q.rates.describe("bb84"),
            msg="a component-tree family was registered",
        )
        self.assertFails(
            ValueError,
            "q.Link, q.Swap or q.PairLink",
            lambda: q.rates.describe("cv"),
            msg="the refusal stopped naming where the rate does live",
        )


if __name__ == "__main__":
    rc = Exam(
        "SurfaceCounting",
        "Photon-number-resolving receivers, and the rate none of them reaches",
        "surface_counting.md",
    ).run(load(Counting))
    rc |= Exam(
        "SurfaceOscillator",
        "The transmitted oscillator, its shot-noise unit and its refusal",
        "surface_oscillator.md",
    ).run(load(Oscillator))
    rc |= Exam(
        "SurfaceRelay",
        "MDI-BB84 through q.Swap, against the engine composition it must match",
        "surface_relay.md",
    ).run(load(QubitRelay))
    rc |= Exam(
        "SurfaceMidpoint",
        "What the qubit midpoint refuses, and why each is not a gap in the code",
        "surface_midpoint.md",
    ).run(load(Midpoint))
    rc |= Exam(
        "SurfaceRefusals",
        "Finite-key refusals reaching a caller through q. rather than the core",
        "surface_refusals.md",
    ).run(load(Refusals))
    rc |= Exam(
        "SurfaceReceiver",
        "Bob's two detectors, his linear range and his detection bandwidth",
        "surface_receiver.md",
    ).run(load(Receiver))
    rc |= Exam(
        "SurfaceCarrier",
        "The laser carrier, and the beat that stops being structurally zero",
        "surface_carrier.md",
    ).run(load(Carrier))
    rc |= Exam(
        "SurfaceCertified",
        "The certified discrete-modulation bound, and its two steps kept apart",
        "surface_certified.md",
    ).run(load(Certified))
    rc |= Exam(
        "SurfaceSources",
        "B92's single-photon branch and the midpoint's finite block",
        "surface_sources.md",
    ).run(load(Sources))
    rc |= Exam(
        "SurfaceAttenuation",
        "Detector decoy: three receiver settings inverted for photon weights",
        "surface_attenuation.md",
    ).run(load(Attenuation))
    rc |= Exam(
        "SurfaceFlaws",
        "A flawed source's two angles, and the tilt that cancels the flaw",
        "surface_flaws.md",
    ).run(load(Flaws))
    rc |= Exam(
        "SurfaceRates",
        "The asymptotic-rate layer for the three families with no component tree",
        "surface_rates.md",
    ).run(load(Rates))
    rc |= Exam(
        "SurfacePairing",
        "Mode pairing from q., and the span that is the whole of its advantage",
        "surface_pairing.md",
    ).run(load(PairingRate))
    rc |= Exam(
        "SurfaceTrain",
        "Round-robin DPS from q., and the four assemblies of one cost",
        "surface_train.md",
    ).run(load(RrdpsRate))
    rc |= Exam(
        "SurfaceVacuum",
        "COW' from q., its two phase-error bounds and the protocol it is not",
        "surface_vacuum.md",
    ).run(load(VacuumRate))
    rc |= Exam(
        "SurfaceRateGuards",
        "What the rate layer refuses, and what each refusal states",
        "surface_rateguards.md",
    ).run(load(RateGuards))
    sys.exit(rc)
