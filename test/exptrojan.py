import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import THA_CLOCK, THA_N, THA_TABLE, THA_TARGET
from kit.forms import bisect
from qkd import _core, attacks

# Tier A, nothing fitted. q.attacks.Injection has no forward observable; it anchors on the transmitter.
#
# Lucamarini, Choi, Ward, Dynes, Yuan & Shields, "Practical security bounds
# against the Trojan-horse attack in quantum key distribution", Phys. Rev. X 5,
# 031030 (2015), arXiv:1506.01989, DOI 10.1103/PhysRevX.5.031030. Eqs. (4), (7), (15), (16).
#
# Sec. IV.B measures the fielded prototype of their Ref. [66]:
# Dixon, Dynes, Lucamarini, Frohlich, Sharpe, Plews, Tam, Yuan, Tanizawa,
# Sato, Kawamura, Fujiwara, Sasaki & Shields, "High speed prototype quantum key
# distribution system and long term field trial", Opt. Express 23, 7583 (2015),
# DOI 10.1364/OE.23.007583.
#
# Fig. 3's "dark count probability per gate 1e-5" is per-pulse background, not composed over two gates.
# Figs. 3 and 4 are rasters: zero crossings are read, the rate scale is not.

# Sec. IV.B, measured, dB: summed nu-OTDR reflectivity (Fig. 6 shaded region), attenuator, Fig. 7 isolator
# backward isolation and forward insertion loss, C-band filter suppression, isolator minimum across S, C, L.
THA_REFLECT = 42.87
THA_ATTEN = 35.0
THA_TESTED = 65.0
THA_INSERT = 0.36
THA_FILTER = 80.0
THA_FLOOR = 40.0

# Sec. III.B damage chain. Inputs: fused-silica LIDT (their Ref. [55]), 50 um^2 core in cm^2, dissipation time,
# diffusivity. Printed outputs: power (W), photons/s, photons/s at 2 W, fuse range of their Ref. [62] (W).
THA_FLUENCE = 1.1e7
THA_AREA = 50e-8
THA_DWELL = 100e-6
THA_DIFF = 0.75
THA_WATTS = 5.5e4
THA_HARD = 4.3e23
THA_SOFT = 1.6e19
THA_FUSE = (1.2, 5.3)
THA_LAMBDA = 1.55e-6

# SI defining constants, exact; qkd/budget.py carries the same pair as _H and _C.
THA_PLANCK = 6.62607015e-34
THA_LIGHT = 299792458.0

# Fig. 3 caption's five simulation parameters, which Fig. 4 reuses, plus Fig. 4's signal intensity.
THA_LOSS = 0.2
THA_DETECT = 0.125
THA_ERROR = 0.01
THA_DARK = 1e-5
THA_FEC = 1.2
THA_SIGNAL = 0.5

# (reach with no THA, reach at mu_out = 1e-2 / 1e-6, largest mu_out with positive rate): Secs. II.B / II.C.
# 170 and 140 are read off a plot on a 10 km grid; 146 and both mu_out are printed.
THA_SINGLE = (170.0, 9.0, 0.015)
THA_PAIRED = (146.0, 140.0, 0.012)

# Zero crossings in km read off the rasters against 25 km gridlines, to a stroke width (~1 km), on the legend's
# leak ladder. Fig. 4 has no 1e-8 curve.
THA_LADDER = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)
THA_FIG3 = (8.76, 58.28, 104.09, 137.76, 157.23, 165.59, 168.81, 170.46)
THA_FIG4 = (4.62, 53.22, 96.63, 125.19, 138.49, 143.28, 145.88)

# Jain, Anisimova, Khan, Makarov, Marquardt & Leuchs, "Trojan-horse attacks
# threaten the security of practical quantum cryptography", New J. Phys. 16,
# 123030 (2014), arXiv:1406.5813, DOI 10.1088/1367-2630/16/12/123030.
# Clavis2 runs plug-and-play SARG04: the probe goes into Bob, who has no isolator, not into Alice.
# Round trip at 1550 nm (dB), injected and returned photons per pulse, Bob's gate rate, Sec. 2 success probability.
JAIN_ROUND = 57.0
JAIN_PROBE = (2e6, 1.5e6)
JAIN_RETURN = (4.0, 3.0)
JAIN_CLOCK = 5e6
JAIN_READ = 0.982


def energy():
    """
    Photon energy at 1550 nm in J.
    """

    return THA_PLANCK * THA_LIGHT / THA_LAMBDA


def photons(watts):
    """
    Photons per second in ``watts`` of CW light at 1550 nm.
    """

    return watts / energy()


def spent(row):
    """
    Eq. (15) over one Table I row's component split, in positive dB; the filter term is 0 dB in
    every row and omitted.
    """

    return attacks.probe_isolation(row[2], row[4], row[5], row[3])


def yields(dist, dark=THA_DARK):
    """
    (Y_1, e_1) at ``dist`` km on the Fig. 3 hardware: the infinite-decoy limit of Ma, Qi, Zhao &
    Lo, Phys. Rev. A 72, 012326 (2005), their Ref. [41].
    """
    eta = _core.decoy_eta(THA_LOSS, dist, THA_DETECT)

    return _core.decoy_ideal(eta, dark, THA_ERROR)


def single(dist, spill, dark=THA_DARK):
    """
    Eq. (6), single-photon efficient BB84 rate at ``dist`` km under a leak of ``spill`` photons
    per setting; the X-basis probability cancels from every crossing.
    """
    y1, e1 = yields(dist, dark)
    seen = attacks.probe_phase(e1, spill, y1)

    return _core.bb84_rate(1.0, y1, e1, y1, seen, THA_FEC)


def paired(dist, spill, dark=THA_DARK):
    """
    Eq. (8), decoy-state efficient BB84 rate at ``dist`` km under a leak of ``spill``,
    single-photon gain Y_1 s exp(-s).
    """
    eta = _core.decoy_eta(THA_LOSS, dist, THA_DETECT)
    y1, e1 = _core.decoy_ideal(eta, dark, THA_ERROR)
    q_mu, e_mu = _core.decoy_gain(THA_SIGNAL, eta, dark, THA_ERROR)
    seen = attacks.probe_phase(e1, spill, y1)
    q1 = y1 * THA_SIGNAL * math.exp(-THA_SIGNAL)

    return _core.bb84_rate(1.0, q_mu, e_mu, q1, seen, THA_FEC)


def reach(rate, spill, dark=THA_DARK):
    """
    Distance in km where ``rate`` reaches zero, bisected over [0, 600].
    """

    return bisect(lambda km: rate(km, spill, dark), 0.0, 600.0)


def ceiling(rate):
    """
    Largest leak per setting with ``rate`` positive at zero distance, bisected over [0, 1].
    """

    return bisect(lambda mu: rate(0.0, mu), 0.0, 1.0)


def leaked(isolator):
    """
    Eq. (4) at the Sec. IV.B transmitter: photons per modulator setting at ``isolator`` dB of
    backward isolation.
    """
    spread = attacks.probe_isolation(THA_REFLECT, isolator, 1, THA_ATTEN)

    return attacks.probe_photons(THA_N, THA_CLOCK, spread)


class TrojanRig(Question):
    """
    Tier A: Lucamarini et al. 2015 Table I, Sec. III.B damage chain, and the Sec. IV.B
    transmitter of Dixon et al. 2015. Nothing fitted.
    """

    def test_lucamarini_table(self):
        """
        All six Table I rows agree three ways: component split by Eq. (15), total from the 1e-6
        target by Eq. (16), and that total back to the target by Eq. (4).
        """
        for row in THA_TABLE:
            self.assertClose(spent(row), row[1], msg=f"{row[0]:.0e} Hz: Eq. (15) split")
            self.assertClose(
                attacks.probe_budget(THA_TARGET, THA_N, row[0]),
                row[1],
                atol=1e-12,
                msg=f"{row[0]:.0e} Hz: Eq. (16) budget",
            )
            self.assertClose(
                attacks.probe_photons(THA_N, row[0], spent(row)),
                THA_TARGET,
                atol=1e-20,
                msg=f"{row[0]:.0e} Hz: Eq. (4) leak",
            )

        self.assertMonotone(
            [row[1] for row in THA_TABLE],
            strict=False,
            msg="isolation against clock",
        )

    def test_lucamarini_damage(self):
        """
        Sec. III.B's printed damage chain follows from one photon energy: 5.5e4 W, 4.3e23
        photons/s at 1550 nm, the adopted 1e20 at 12.8 W and 4.3e3 below, and 1.6e19 at 2 W.
        """
        watts = THA_FLUENCE * THA_AREA / THA_DWELL

        self.assertClose(watts, THA_WATTS, msg=f"the fluence comes to {watts:.4g} W")
        self.assertClose(
            photons(watts) / THA_HARD,
            1.0,
            atol=5e-3,
            msg=f"{photons(watts):.4g} photons/s",
        )
        self.assertClose(
            THA_N * energy(),
            12.8,
            atol=0.05,
            msg=f"threshold {THA_N * energy():.4g} W",
        )
        self.assertClose(
            photons(watts) / THA_N / 4.3e3,
            1.0,
            atol=5e-3,
            msg=f"ratio {photons(watts) / THA_N:.4g}",
        )
        self.assertClose(
            photons(2.0) / THA_SOFT,
            1.0,
            atol=0.03,
            msg=f"2 W is {photons(2.0):.4g} photons/s",
        )
        self.assertGreater(2.0, THA_FUSE[0], msg="2 W against fuse range low")
        self.assertLess(2.0, THA_FUSE[1], msg="2 W against fuse range high")

    def test_lucamarini_dwell(self):
        """
        The 50 um^2 core gives a 66.67 us dissipation time against the footnote's 100 us, and
        one area throughout gives 8.25e4 W and 6.44e23 photons/s against the printed 5.5e4 and
        4.3e23.
        """
        got = THA_AREA * 1e2 / THA_DIFF

        self.assertClose(got * 1e6, 66.667, atol=0.01, msg=f"dwell {got * 1e6:.2f} us")
        self.assertClose(THA_DWELL / got, 1.5, atol=0.01, msg=f"ratio {THA_DWELL / got:.4f}")
        self.assertClose(
            THA_FLUENCE * THA_DIFF * 1e-2 / 8.25e4,
            1.0,
            atol=1e-9,
            msg="one-area power",
        )
        self.assertClose(
            photons(THA_FLUENCE * THA_DIFF * 1e-2) / 6.4373e23,
            1.0,
            atol=1e-4,
            msg=f"{photons(THA_FLUENCE * THA_DIFF * 1e-2):.4e} photons/s",
        )
        self.assertGreater(
            photons(THA_FLUENCE * THA_DIFF * 1e-2) / THA_HARD,
            1.0,
            msg="one-area photons/s against printed",
        )

    def test_lucamarini_reflect(self):
        """
        The Sec. IV.B nu-OTDR reflectivity of 42.87 dB with the 35 dB attenuator leaves 57.13 dB
        for the isolator by Eq. (15), under Table I's 60 dB and the measured 65 dB.
        """
        self.assertGreater(THA_REFLECT, THA_TABLE[0][2], msg=f"reflectivity {THA_REFLECT} dB")

        need = THA_TABLE[0][1] - 2.0 * THA_ATTEN - THA_REFLECT
        self.assertClose(need, 57.13, atol=1e-9, msg=f"isolator need {need:.2f} dB")
        self.assertLess(need, THA_TABLE[0][4], msg=f"need {need:.2f} dB against 60")
        self.assertLess(need, THA_TESTED, msg=f"need {need:.2f} dB against 65")

    def test_lucamarini_leak(self):
        """
        The Sec. IV.B transmitter leaks 5.164e-7 photons per setting at the specified 60 dB
        isolator and 1.633e-7 at the measured 65 dB, 1.94 and 6.12 times under the 1e-6 target;
        the paper prints neither.
        """
        got = leaked(THA_TABLE[0][4])
        self.assertClose(got * 1e7, 5.1642, atol=1e-3, msg=f"60 dB leak {got:.4e}")
        self.assertClose(THA_TARGET / got, 1.9364, atol=1e-3, msg=f"{THA_TARGET / got:.4f} times under target")

        got = leaked(THA_TESTED)
        self.assertClose(got * 1e7, 1.6331, atol=1e-3, msg=f"65 dB leak {got:.4e}")
        self.assertClose(THA_TARGET / got, 6.1235, atol=1e-3, msg=f"{THA_TARGET / got:.4f} times under target")
        self.assertLess(got, leaked(THA_TABLE[0][4]), msg=f"65 dB leak {got:.4e}")

    def test_lucamarini_filter(self):
        """
        Eq. (15) counts the 80 dB C-band filter twice, giving 312.87 dB at the isolator's 40 dB
        band edge against 177.87 dB at 1550 nm.
        """
        self.assertClose(
            attacks.probe_isolation(0.0, bandpass=THA_FILTER),
            160.0,
            msg="filter isolation",
        )

        edge = attacks.probe_isolation(THA_REFLECT, THA_FLOOR, 1, THA_ATTEN, THA_FILTER)
        line = attacks.probe_isolation(THA_REFLECT, THA_TESTED, 1, THA_ATTEN)
        self.assertClose(edge, 312.87, atol=1e-9, msg=f"band edge {edge:.2f} dB")
        self.assertClose(line, 177.87, atol=1e-9, msg=f"1550 nm {line:.2f} dB")
        self.assertGreater(edge, line, msg=f"band edge {edge:.2f} dB")

    def test_lucamarini_forward(self):
        """
        Reading Fig. 7's 0.36 dB forward insertion loss as the 65 dB backward isolation
        understates Eq. (15) by 64.64 dB, a leak 2.9e6 times larger.
        """
        self.assertLess(THA_INSERT, 1.0, msg=f"insertion loss {THA_INSERT} dB")

        gap = attacks.probe_isolation(THA_REFLECT, THA_TESTED, 1, THA_ATTEN) - attacks.probe_isolation(
            THA_REFLECT, THA_INSERT, 1, THA_ATTEN
        )
        self.assertClose(gap, 64.64, atol=1e-9, msg=f"gap {gap:.2f} dB")
        self.assertClose(
            leaked(THA_INSERT) / leaked(THA_TESTED) / 1e6,
            2.9107,
            atol=1e-3,
            msg=f"leak ratio {leaked(THA_INSERT) / leaked(THA_TESTED):.4e}",
        )


class TrojanReach(Question):
    """
    Tier A: Lucamarini et al. 2015 Figs. 3 and 4, seven printed numbers and fifteen raster
    crossings. Nothing fitted.
    """

    def test_reach_single(self):
        """
        Fig. 3 single-photon curve against Sec. II.B: reach 171.10 km against 170, 9.16 km
        against 9 at a 1e-2 leak, largest leak 0.015262 against 0.015.
        """
        got = reach(single, 0.0)

        self.assertClose(got, THA_SINGLE[0], atol=1.5, msg=f"reach {got:.3f} km")

        got = reach(single, 1e-2)
        self.assertClose(got, THA_SINGLE[1], atol=0.5, msg=f"reach at 1e-2 {got:.3f} km")

        got = ceiling(single)
        self.assertClose(got, THA_SINGLE[2], atol=5e-4, msg=f"largest leak {got:.6f}")

    def test_reach_decoy(self):
        """
        Fig. 4 decoy curve at signal intensity 0.5 against Sec. II.C: reach 146.200 km against
        146, 138.86 km against 140 at a 1e-6 leak, largest leak 0.012339 against 0.012.
        """
        got = reach(paired, 0.0)
        self.assertClose(got, THA_PAIRED[0], atol=0.5, msg=f"reach {got:.3f} km")

        got = reach(paired, THA_TARGET)
        self.assertClose(got, THA_PAIRED[1], atol=2.0, msg=f"reach at 1e-6 {got:.3f} km")

        got = ceiling(paired)
        self.assertClose(got, THA_PAIRED[2], atol=5e-4, msg=f"largest leak {got:.6f}")

    def test_reach_fractions(self):
        """
        100 km is 58.4% of the single-photon reach against their "about 60%" and 68.4% of the
        decoy reach against 70%, and a 1e-6 leak keeps 96% of the decoy reach.
        """
        got = 100.0 / reach(single, 0.0)
        self.assertClose(got, 0.60, atol=0.02, msg=f"single-photon share {got * 100:.1f}%")

        got = 100.0 / reach(paired, 0.0)
        self.assertClose(got, 0.70, atol=0.02, msg=f"decoy share {got * 100:.1f}%")

        got = reach(paired, THA_TARGET) / reach(paired, 0.0)
        self.assertClose(got, 0.96, atol=0.015, msg=f"1e-6 leak keeps {got * 100:.1f}%")

    def test_reach_curve(self):
        """
        Over leaks from 1e-8 to 1e-2 both reaches fall, the decoy reach sits under the
        single-photon one at every rung, and the rate share a 1e-6 leak leaves falls with
        distance.
        """
        ladder = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
        thin = [reach(single, mu) for mu in ladder]
        wide = [reach(paired, mu) for mu in ladder]

        self.assertMonotone(thin, rising=False, msg=f"single-photon reach {thin}")
        self.assertMonotone(wide, rising=False, msg=f"decoy reach {wide}")
        for mu, one, two in zip(ladder, thin, wide):
            self.assertLess(two, one, msg=f"mu={mu:g}: decoy {two:.2f} km, single {one:.2f} km")

        held = [paired(km, THA_TARGET) / paired(km, 0.0) for km in (0.0, 25.0, 50.0, 75.0, 100.0, 125.0)]

        self.assertMonotone(held, rising=False, msg=f"rate share {held}")
        self.assertGreater(held[4], 0.9, msg=f"share at 100 km {held[4] * 100:.2f}%")

    def test_reach_darkness(self):
        """
        Composing the 1e-5 per-gate dark count over two gates, as q.Link does for
        q.ClickDetector, puts the reaches at 156.04 and 131.15 km, over 10 km short of the
        published 170 and 146.
        """
        pair = 1.0 - (1.0 - THA_DARK) ** 2

        self.assertClose(pair, 2e-5, atol=1e-9, msg=f"two-gate dark {pair:.6e}")

        got = (reach(single, 0.0, dark=pair), reach(paired, 0.0, dark=pair))
        self.assertClose(got[0], 156.04, atol=0.05, msg=f"single reach {got[0]:.2f} km")
        self.assertClose(got[1], 131.15, atol=0.05, msg=f"decoy reach {got[1]:.2f} km")
        for seen, want in zip(got, (THA_SINGLE[0], THA_PAIRED[0])):
            self.assertGreater(want - seen, 10.0, msg=f"{seen:.2f} km against {want:.0f}")

    def test_reach_family(self):
        """
        All fifteen raster crossings sit inside 1.00 km of the model: rms 0.706 km on Fig. 3,
        every read left of the model, and 0.286 km on Fig. 4.
        """
        rungs = THA_LADDER[:6] + (0.0,)
        thin = [reach(single, mu) for mu in THA_LADDER]
        wide = [reach(paired, mu) for mu in rungs]

        for read, got, mu in zip(THA_FIG3, thin, THA_LADDER):
            self.assertClose(got, read, atol=1.05, msg=f"Fig. 3 mu={mu:g}: {got:.2f} km against {read} read")
        for read, got, mu in zip(THA_FIG4, wide, rungs):
            self.assertClose(got, read, atol=0.6, msg=f"Fig. 4 mu={mu:g}: {got:.2f} km against {read} read")

        gap = [got - read for read, got in zip(THA_FIG3, thin)]
        self.assertGreater(min(gap), 0.0, msg=f"Fig. 3 min gap {min(gap):.3f} km")

        rms = math.sqrt(sum(one * one for one in gap) / len(gap))
        self.assertClose(rms, 0.706, atol=0.02, msg=f"rms {rms:.3f} km")

    def test_reach_price(self):
        """
        The Sec. IV.B leak moves the decoy reach from 146.20 km to 143.17 km at the measured
        isolator, 140.86 km at Table I's 60 dB and 138.86 km at the 1e-6 target.
        """
        clean = reach(paired, 0.0)
        best = reach(paired, leaked(THA_TESTED))
        spec = reach(paired, leaked(THA_TABLE[0][4]))
        aim = reach(paired, THA_TARGET)

        self.assertClose(best, 143.166, atol=0.05, msg=f"65 dB reach {best:.3f} km")
        self.assertClose(spec, 140.864, atol=0.05, msg=f"60 dB reach {spec:.3f} km")
        self.assertClose(clean - best, 3.034, atol=0.05, msg=f"price {clean - best:.3f} km")
        self.assertMonotone([clean, best, spec, aim], rising=False, msg=f"reaches {[clean, best, spec, aim]}")


class ProbedRig(Question):
    """
    Tier A: Jain et al. 2014, Eq. (4) on a Clavis2 receiver's measured round trip.
    """

    def test_clavis_return(self):
        """
        The measured 57 dB round trip returns 3.99 photons from 2e6 injected against their 4.0,
        and 2.99 from 1.5e6 against their 3.
        """
        for sent, want in zip(JAIN_PROBE, JAIN_RETURN):
            got = attacks.probe_photons(sent * JAIN_CLOCK, JAIN_CLOCK, JAIN_ROUND)
            self.assertClose(got, want, atol=0.02, msg=f"{sent:.1e} in: return {got:.4f}")

        self.assertMonotone(
            [attacks.probe_photons(sent * JAIN_CLOCK, JAIN_CLOCK, JAIN_ROUND) for sent in JAIN_PROBE],
            rising=False,
            msg="return against probe",
        )

    def test_clavis_reading(self):
        """
        Sec. 2's reading probability 1 - exp(-mu) at mu = 3.99 is 98.15% against their 98.2%,
        under the exp(-2mu) unambiguous-discrimination ceiling of Lucamarini's Eq. (45), 99.97%.
        """
        got = attacks.probe_photons(JAIN_PROBE[0] * JAIN_CLOCK, JAIN_CLOCK, JAIN_ROUND)

        self.assertClose(
            1.0 - math.exp(-got),
            JAIN_READ,
            atol=1e-3,
            msg=f"non-empty {1.0 - math.exp(-got):.5f}",
        )
        self.assertGreater(
            1.0 - math.exp(-2.0 * got),
            1.0 - math.exp(-got),
            msg=f"exp(-2mu) ceiling {1.0 - math.exp(-2.0 * got):.5f}",
        )
        self.assertClose(
            1.0 - math.exp(-2.0 * got),
            0.99966,
            atol=1e-4,
            msg=f"ceiling {1.0 - math.exp(-2.0 * got):.5f}",
        )

    def test_clavis_vacuous(self):
        """
        Four returned photons put Lucamarini's coin imbalance at 0.5061, past one half, so the
        phase error rate saturates at 1/2 at every yield and the two papers' leaks do not
        compose.
        """
        got = attacks.probe_photons(JAIN_PROBE[0] * JAIN_CLOCK, JAIN_CLOCK, JAIN_ROUND)

        self.assertGreater(attacks.probe_delta(got), 0.5, msg=f"imbalance {attacks.probe_delta(got):.4f}")
        self.assertClose(attacks.probe_delta(got), 0.50611, atol=1e-4, msg=f"imbalance {attacks.probe_delta(got):.5f}")
        for y1 in (1.0, 0.5, 0.1, 1e-3):
            self.assertClose(
                attacks.probe_phase(0.0252, got, y1),
                0.5,
                msg=f"y1={y1:g}: phase error",
            )

        self.assertGreater(
            got / leaked(THA_TESTED),
            1e7,
            msg=f"leak ratio {got / leaked(THA_TESTED):.3e}",
        )


if __name__ == "__main__":
    rc = Exam(
        "ExpLucamariniTrojan",
        "Tier A: Lucamarini et al. 2015, a fielded transmitter's measured Trojan-horse isolation",
        "exp_trojan.md",
    ).run(load(TrojanRig))
    rc |= Exam(
        "ExpTrojanReach",
        "Tier A: Lucamarini et al. 2015, both Trojan-horse key-rate curves at their published crossings",
        "exp_trojan_reach.md",
    ).run(load(TrojanReach))
    rc |= Exam(
        "ExpJainClavis",
        "Tier A: Jain et al. 2014, a commercial Clavis2 read through its own back-reflection",
        "exp_trojan_clavis.md",
    ).run(load(ProbedRig))
    sys.exit(rc)
