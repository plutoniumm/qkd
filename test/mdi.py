import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd
from kit.checks import Guarded
from kit.forms import bisect, fitslope, h2, plob
from qkd import _core

# The MDI-QKD parameter set, not GYS: Lo, Curty & Qi, Phys. Rev. Lett. 108, 130503 (2012), Fig. 2,
# restated by Ma & Razavi, Phys. Rev. A 86, 062319 (2012), Table 1 and by Xu, Curty, Qi & Lo,
# New J. Phys. 15, 113007 (2013), Table 1.
ALPHA = 0.2
DET = 0.145
MISALIGN = 0.015
FEC = 1.16

# Y0 = 6.02e-6 is the total background; `mdi_yield` takes Ma & Razavi Table 1's per-detector p_d = 3.0e-6.
BACKGROUND = 6.02e-6
DARK = BACKGROUND / 2.0

# Vacuum-plus-weak decoys (Ma, Qi, Zhao & Lo); `best` re-optimises the signal per distance.
DECOY = 0.005
VACUUM = 0.0

# Curty's weakest setting (imperfect extinction); the programme's gain over Table 2 is set by it, not the span.
EXTINCT = 5e-4
LEVELS = [0.3, 0.1, EXTINCT]

# Photon-number truncation n + m <= CUT; below about ten the engine refuses at long spans.
CUT = 10


def grids(sets, km):
    """
    The rectilinear gain grid, the diagonal gain grid and the diagonal error-gain grid at
    one span.
    """
    eta = arms(km)
    zed = [_core.mdi_rect(x, y, eta, eta, DARK, MISALIGN) for x in sets for y in sets]
    ex = [_core.mdi_diag(x, y, eta, eta, DARK, MISALIGN) for x in sets for y in sets]

    return [g for g, _ in zed], [g for g, _ in ex], [g * e for g, e in ex]


def disturb(nu, vac, faults, y_x):
    """
    Table 2's four-cell closed form for e11 off a row-major diagonal error-gain grid, both
    senders on one ladder.
    """

    return _core.mdi_e11(nu, vac, nu, vac, faults[4], faults[8], faults[5], faults[7], y_x)


def better(fn, args, fallback):
    """
    The programme's bound, or the closed form when the engine refuses.
    """
    try:
        return fn(*args)[0]
    except ValueError:
        return fallback


def paid(sets, km, lp):
    """
    Bits per pulse pair at one span, single-photon quantities from the closed forms or, with
    lp, the nine-cell programmes.
    """
    gz, gx, faults = grids(sets, km)
    y_z = _core.mdi_y11(sets, sets, gz)
    y_x = _core.mdi_y11(sets, sets, gx)

    if lp:
        y_z = better(_core.mdi_program, (sets, sets, gz, CUT), y_z)
        y_x = better(_core.mdi_program, (sets, sets, gx, CUT), y_x)

    e11 = disturb(sets[1], sets[2], faults, y_x)

    if lp:
        e11 = better(_core.mdi_disturb, (sets, sets, faults, y_x, CUT), e11)

    eta = arms(km)
    gain, qber = _core.mdi_rect(sets[0], sets[0], eta, eta, DARK, MISALIGN)

    return _core.mdi_rate(_core.mdi_gain(y_z, sets[0], sets[0]), e11, gain, qber, FEC)


def arms(km):
    """
    Per-arm efficiency with the relay at the midpoint of a TOTAL span of km.
    """
    return DET * 10.0 ** (-ALPHA * (km / 2.0) / 10.0)


def total(km):
    """
    End-to-end channel transmittance over the whole span, detectors excluded.
    """
    return 10.0 ** (-ALPHA * km / 10.0)


def observe(mu, nu, eta, fn):
    """
    The 3x3 grid one basis's forward model produces at intensities (mu, nu, 0).
    """
    levels = (mu, nu, VACUUM)

    return [fn(x, y, eta, eta, DARK, MISALIGN) for x in levels for y in levels]


def rate(km, mu, nu=DECOY):
    """
    Bits per pulse pair at a symmetric relay: observables forward, single-photon quantities bounded.
    """
    eta = arms(km)
    zed = observe(mu, nu, eta, _core.mdi_rect)
    ex = observe(mu, nu, eta, _core.mdi_diag)
    sets = [mu, nu, VACUUM]
    y_z = _core.mdi_y11(sets, sets, [g for g, _ in zed])
    y_x = _core.mdi_y11(sets, sets, [g for g, _ in ex])
    e11 = disturb(nu, VACUUM, [g * e for g, e in ex], y_x)
    gain, qber = zed[0]

    return _core.mdi_rate(_core.mdi_gain(y_z, mu, mu), e11, gain, qber, FEC)


def best(km):
    """
    The best rate over signal intensities 0.05 to 1.2 in steps of 0.05.
    """
    return max(rate(km, 0.05 * i) for i in range(1, 25))


class Forward(Question):
    def test_bell_ceiling(self):
        """
        Without dark counts Y11 is eta_a*eta_b/2, two of the four Bell states identified.
        """
        for ea, eb in ((1.0, 1.0), (0.5, 0.5), (0.1, 0.3), (0.02, 0.9)):
            y11, _ = _core.mdi_yield(ea, eb, 0.0, MISALIGN)
            self.assertClose(y11, ea * eb / 2.0, msg=f"Y11 at eta {ea}, {eb}")

    def test_pair_error(self):
        """
        Without dark counts the single-photon-pair error rate is the misalignment itself, in
        both bases and at every arm efficiency.
        """
        for e_d in (0.0, 0.005, 0.015, 0.05):
            _, e11 = _core.mdi_yield(0.1, 0.4, 0.0, e_d)
            self.assertClose(e11, e_d, msg=f"e11 at e_d = {e_d}")

    def test_rect_error(self):
        """
        Without dark counts the rectilinear error rate is the misalignment at any intensity.
        """
        for mu in (0.05, 0.5, 2.0):
            _, err = _core.mdi_rect(mu, mu, 0.1, 0.1, 0.0, MISALIGN)
            self.assertClose(err, MISALIGN, msg=f"E_rect at mu = {mu}")

    def test_diag_quarter(self):
        """
        The diagonal-basis QBER of two independent coherent sources tends to 1/4 + e_d/2 as
        the signal weakens, not to the misalignment.
        """
        for e_d in (0.0, 0.015, 0.05):
            _, err = _core.mdi_diag(1e-8, 1e-8, 0.1, 0.1, 0.0, e_d)
            self.assertClose(err, 0.25 + e_d / 2.0, atol=1e-9, msg=f"limit at {e_d}")

    def test_diag_interferes(self):
        """
        The diagonal gain over the rectilinear one is 2.000001, 2.000050 and 2.010069 at
        mu = 1e-5, 0.001 and 0.2.
        """
        for mu, want in ((1e-5, 2.000001), (0.001, 2.000050), (0.2, 2.010069)):
            zed = _core.mdi_rect(mu, mu, 0.1, 0.1, 0.0, MISALIGN)[0]
            ex = _core.mdi_diag(mu, mu, 0.1, 0.1, 0.0, MISALIGN)[0]
            self.assertClose(ex / zed, want, atol=1e-6, msg=f"ratio at mu = {mu}")

    def test_gain_scaling(self):
        """
        Halving one arm efficiency halves the single-photon-pair yield, halving both quarters it.
        """
        base = _core.mdi_yield(0.1, 0.1, 0.0, MISALIGN)[0]
        half = _core.mdi_yield(0.05, 0.1, 0.0, MISALIGN)[0]
        self.assertClose(half / base, 0.5, msg="one arm halved")

        both = _core.mdi_yield(0.05, 0.05, 0.0, MISALIGN)[0]
        self.assertClose(both / base, 0.25, msg="both arms halved")


class Decoy(Question):
    """
    mdi_y11 is a bound and mdi_yield the forward model: the bound stays under the truth.
    """

    def test_yield_sandwich(self):
        """
        The three-intensity Y11 bound sits under the forward yield at every decoy strength,
        tightening monotonically as the decoy weakens to past 99% of it at nu = 0.005.
        """
        eta = arms(100.0)
        truth = _core.mdi_yield(eta, eta, DARK, MISALIGN)[0]
        got = []
        for nu in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
            sets = [0.5, nu, VACUUM]
            grid = [g for g, _ in observe(0.5, nu, eta, _core.mdi_rect)]
            low = _core.mdi_y11(sets, sets, grid)
            got.append(low / truth)
            self.assertLess(low, truth, msg=f"Y11 bound at nu = {nu}")
        self.assertMonotone(got, msg="bound against decoy strength")
        self.assertLess(0.99, got[-1], msg=f"share at nu = 0.005 is {got[-1]:.4f}")

    def test_asymmetric_arms(self):
        """
        With the relay at 20 km against 80 the bound stays under the forward yield for equal
        intensity sets and both orders of unequal ones.
        """
        ea, eb = DET * 10.0 ** (-ALPHA * 20 / 10), DET * 10.0 ** (-ALPHA * 80 / 10)
        truth = _core.mdi_yield(ea, eb, DARK, MISALIGN)[0]
        one = [0.5, 0.05, VACUUM]
        two = [0.3, 0.02, VACUUM]
        grid = [_core.mdi_rect(x, y, ea, eb, DARK, MISALIGN)[0] for x in one for y in one]
        ahead = [_core.mdi_rect(x, y, ea, eb, DARK, MISALIGN)[0] for x in one for y in two]
        after = [_core.mdi_rect(x, y, ea, eb, DARK, MISALIGN)[0] for x in two for y in one]
        self.assertLess(_core.mdi_y11(one, one, grid), truth, msg="equal sets")
        self.assertLess(_core.mdi_y11(one, two, ahead), truth, msg="first branch")
        self.assertLess(_core.mdi_y11(two, one, after), truth, msg="second branch")

    def test_error_bound(self):
        """
        The e11 bound is under 3% where the raw diagonal QBER is past 24%, and above the 1.5%
        misalignment.
        """
        eta = arms(50.0)
        ex = observe(0.5, DECOY, eta, _core.mdi_diag)
        sets = [0.5, DECOY, VACUUM]
        y_x = _core.mdi_y11(sets, sets, [g for g, _ in ex])
        e11 = disturb(DECOY, VACUUM, [g * e for g, e in ex], y_x)
        self.assertLess(0.24, ex[0][1], msg=f"raw diagonal QBER {ex[0][1]:.4f}")
        self.assertLess(e11, 0.03, msg=f"e11 bound {e11:.4f}")
        self.assertLess(MISALIGN, e11, msg=f"e11 bound {e11:.4f} under the misalignment")

    def test_infeasible(self):
        """
        Data no single-photon component explains returns e11 = 1/2, not zero.
        """
        got = _core.mdi_e11(0.05, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertClose(got, 0.5, msg="a vanished yield reads as the cap")


class Anchors(Question):
    def test_single_photon(self):
        """
        `mdi_rate` at (y11, e11, y11, e11) reproduces Ma & Razavi Eq. (7), the single-photon
        source rate.
        """
        eta = arms(100.0)
        y11, e11 = _core.mdi_yield(eta, eta, DARK, MISALIGN)
        want = y11 * (1.0 - h2(e11) - FEC * h2(e11))
        got = _core.mdi_rate(y11, e11, y11, e11, FEC)
        self.assertClose(got, want, msg="Eq. (7) recovered from Eq. (1)")

    def test_forty_decibels(self):
        """
        Lo, Curty and Qi's stated tolerance of over 40 dB (about 200 km) holds on their
        parameter set: positive at 200 km, zero by 260.
        """
        self.assertLess(0.0, best(200.0), msg="rate at 200 km")
        self.assertClose(best(260.0), 0.0, msg="rate at 260 km")

    def test_secure_distance(self):
        """
        The three-intensity secure distance on that parameter set is 230.5379 km, 46.1076 dB.
        """
        reach = bisect(lambda km: best(km), 200.0, 280.0)
        self.assertClose(reach, 230.5379, atol=1e-3, msg="maximum secure distance, km")
        self.assertClose(ALPHA * reach, 46.1076, atol=1e-3, msg="the same in dB")

    def test_linear_scaling(self):
        """
        The rate against end-to-end transmittance has log-log slope 1.03695, linear like
        weak-coherent BB84.
        """
        spans = (40.0, 60.0, 80.0, 100.0, 120.0)
        xs = [math.log10(total(km)) for km in spans]
        ys = [math.log10(best(km)) for km in spans]
        self.assertClose(fitslope(xs, ys), 1.03695, atol=1e-4, msg="log-log slope against T")

    def test_repeaterless(self):
        """
        Every rate from 50 to 230 km sits under Pirandola, Laurenza, Ottaviani and Banchi's
        repeaterless bound at its end-to-end transmittance.
        """
        for km in (50.0, 100.0, 150.0, 200.0, 230.0):
            cap = plob(total(km))
            self.assertLess(best(km), cap, msg=f"rate against capacity at {km} km")

    def test_hundred_kilometres(self):
        """
        The rate at 100 km is 4.679772e-06 bits per pulse pair, at the mu = 0.55 the sweep
        selects.
        """
        self.assertClose(best(100.0), 4.679772e-06, atol=1e-12, msg="bits per pulse pair")
        self.assertClose(rate(100.0, 0.55), best(100.0), msg="the sweep picks mu = 0.55")


class Guards(Guarded):
    def test_yield_slots(self):
        """
        `mdi_yield` refuses each efficiency, the per-gate dark probability and the
        misalignment outside its own domain.
        """
        ok = (0.1, 0.1, DARK, MISALIGN)
        self.assertSlots(
            _core.mdi_yield,
            ok,
            [
                (0, "eta_a", (0.0, -0.1, 1.5, float("nan"))),
                (1, "eta_b", (0.0, 1.5)),
                (2, "dark", (1.0, -1e-9)),
                (3, "e_d", (0.6, -1e-9)),
            ],
            msg="mdi_yield",
        )

    def test_intensity_order(self):
        """
        The three intensities must decrease strictly.
        """
        grid = [1e-3] * 9
        self.assertBad(
            "decrease strictly",
            _core.mdi_y11,
            ([0.5, 0.05, 0.05], [0.5, 0.05, 0.0], grid),
            msg="flat ladder",
        )
        self.assertBad(
            "decrease strictly",
            _core.mdi_y11,
            ([0.05, 0.5, 0.0], [0.5, 0.05, 0.0], grid),
            msg="inverted ladder",
        )

    def test_grid_shape(self):
        """
        A gain grid other than nine cells (row-major, Alice's intensity as row) is refused.
        """
        sets = [0.5, 0.05, 0.0]
        self.assertBad(
            "3x3 gain grid",
            _core.mdi_y11,
            (sets, sets, [1e-3] * 6),
            msg="six cells",
        )

    def test_rate_ordering(self):
        """
        A single-photon-pair gain above the total gain is an inverted decoy bound and is
        refused, never rescaled.
        """
        self.assertBad(
            "must not exceed",
            _core.mdi_rate,
            (0.2, 0.01, 0.1, 0.01, FEC),
            msg="q11 above q_z",
        )

    def test_error_slots(self):
        """
        `mdi_e11` refuses a decoy at or below its vacuum and a y11 outside [0, 1].
        """
        ok = (0.05, 0.0, 0.05, 0.0, 1e-4, 1e-6, 1e-5, 1e-5, 1e-3)
        self.assertBad(
            "stronger than the vacuum",
            _core.mdi_e11,
            (0.0, 0.0, 0.05, 0.0, 1e-4, 1e-6, 1e-5, 1e-5, 1e-3),
            msg="decoy at the vacuum",
        )
        self.assertSlots(
            _core.mdi_e11,
            ok,
            [(8, "y11", (1.5, -0.1))],
            msg="mdi_e11",
        )


class Parts(Guarded):
    def test_analyser_domain(self):
        """
        `q.BellAnalyser` keeps its efficiency and takes no `trusted` flag.
        """
        good = qkd.BellAnalyser(eta=DET, dark=DARK, misalign=MISALIGN)
        self.assertClose(good.eta, DET, msg="efficiency kept")
        self.assertFails(
            TypeError,
            "trusted",
            lambda: qkd.BellAnalyser(eta=DET, dark=DARK, trusted=True),
            msg="trusted=True on a midpoint measurement",
        )

    def test_two_misalignments(self):
        """
        Key and test basis carry separate misalignments, neither defaulted and each refused
        past a half.
        """
        an = qkd.BellAnalyser(eta=DET, dark=DARK)
        self.assertClose(float(an.misalign is None), 1.0, msg="misalign default")
        self.assertClose(float(an.misalign_test is None), 1.0, msg="misalign_test default")

        for slot in ("misalign", "misalign_test"):
            self.assertFails(
                ValueError,
                slot,
                lambda s=slot: qkd.BellAnalyser(eta=DET, dark=DARK, **{s: 0.9}),
                msg=f"{slot} = 0.9",
            )

    def test_bound_has_no_phase(self):
        """
        `q.TestBasisBound` refuses `e_phase`, the test basis measuring it, and defaults f to 1.16.
        """
        sec = qkd.TestBasisBound()
        self.assertClose(sec.f, FEC, msg="f_ec")
        self.assertFails(
            TypeError,
            "e_phase",
            lambda: qkd.TestBasisBound(e_phase=0.05),
            msg="e_phase on a test-basis bound",
        )

    def test_distinct_names(self):
        """
        `q.TestBasisBound`, `q.PhaseBound` and `q.SymmetryBound` are three classes.
        """
        names = (
            qkd.TestBasisBound,
            qkd.PhaseBound,
            qkd.SymmetryBound,
        )
        self.assertClose(float(len(set(names))), 3.0, msg="distinct classes")


class Programme(Guarded):
    """
    Table 2's elimination reads seven of the nine cells in closed form; the programme reads
    all nine and keeps the terms that one drops as variables.
    """

    def test_yield_bracket(self):
        """
        At five spans the programme's Y11 sits under the forward yield and above the closed form
        returned beside it, with a positive central-path gap and non-zero iterations.
        """
        for km in (0.0, 50.0, 100.0, 150.0, 200.0):
            eta = arms(km)
            truth = _core.mdi_yield(eta, eta, DARK, MISALIGN)[0]
            gz = grids(LEVELS, km)[0]
            bound, plain, gap, iters = _core.mdi_program(LEVELS, LEVELS, gz, CUT)
            self.assertLess(bound, truth, msg=f"Y11 under the forward model at {km} km")
            self.assertClose(plain, _core.mdi_y11(LEVELS, LEVELS, gz), msg=f"mdi_y11 at {km} km")
            self.assertLess(plain, bound, msg=f"programme against closed form at {km} km")
            self.assertLess(0.0, gap, msg=f"central-path gap at {km} km")
            self.assertLess(0, iters, msg=f"iterations at {km} km")

    def test_vacuum_decides(self):
        """
        At 50 and 150 km the programme tightens Y11 by under 1% at an exact-zero vacuum and
        by over 3% at 5e-4.
        """
        for km in (50.0, 150.0):
            ideal = [0.55, DECOY, VACUUM]
            one = _core.mdi_program(ideal, ideal, grids(ideal, km)[0], CUT)
            two = _core.mdi_program(LEVELS, LEVELS, grids(LEVELS, km)[0], CUT)
            self.assertLess(one[0] / one[1], 1.01, msg=f"gain at a true vacuum, {km} km")
            self.assertLess(1.03, two[0] / two[1], msg=f"gain at 5e-4, {km} km")

    def test_error_sandwich(self):
        """
        At five spans the error programme sits above the forward e11 and below the four-cell
        closed form.
        """
        for km in (0.0, 50.0, 100.0, 150.0, 200.0):
            eta = arms(km)
            truth = _core.mdi_yield(eta, eta, DARK, MISALIGN)[1]
            gx, faults = grids(LEVELS, km)[1:]
            y_x = _core.mdi_program(LEVELS, LEVELS, gx, CUT)[0]
            bound, plain = _core.mdi_disturb(LEVELS, LEVELS, faults, y_x, CUT)[:2]
            self.assertLess(truth, bound, msg=f"above the forward e11 at {km} km")
            self.assertLess(bound, plain, msg=f"below mdi_e11 at {km} km")

    def test_random_sandwich(self):
        """
        Over twenty random spans, ladders and off-centre relays every accepted certificate
        stays under the forward yield, and more than fourteen certify.
        """
        dice = random.Random(20260904)
        seen = 0

        for _ in range(20):
            km = dice.uniform(0.0, 240.0)
            one = [
                dice.uniform(0.2, 0.8),
                dice.uniform(0.01, 0.15),
                dice.uniform(0.0, 1e-3),
            ]
            two = [
                dice.uniform(0.2, 0.8),
                dice.uniform(0.01, 0.15),
                dice.uniform(0.0, 1e-3),
            ]
            share = dice.uniform(0.2, 0.8)
            ea = DET * 10.0 ** (-ALPHA * share * km / 10.0)
            eb = DET * 10.0 ** (-ALPHA * (1.0 - share) * km / 10.0)
            grid = [_core.mdi_rect(x, y, ea, eb, DARK, MISALIGN)[0] for x in one for y in two]
            truth = _core.mdi_yield(ea, eb, DARK, MISALIGN)[0]

            try:
                bound = _core.mdi_program(one, two, grid, CUT)[0]
            except ValueError:
                continue

            seen += 1
            self.assertLess(bound, truth, msg=f"draw under the forward yield at {km} km")
        self.assertLess(14, seen, msg=f"{seen} of 20 certify")

    def test_pinned_gain(self):
        """
        At Curty's intensities and 100 km the nine cells tighten Y11 by 3.6% and e11 by 1.9%
        over the seven.
        """
        gz, gx, faults = grids(LEVELS, 100.0)
        bound, plain = _core.mdi_program(LEVELS, LEVELS, gz, CUT)[:2]
        y_x = _core.mdi_program(LEVELS, LEVELS, gx, CUT)[0]
        got, want = _core.mdi_disturb(LEVELS, LEVELS, faults, y_x, CUT)[:2]
        self.assertClose(bound / plain, 1.035890, atol=1e-4, msg="Y11 tightens by 3.6%")
        self.assertClose(got / want, 0.980580, atol=1e-4, msg="e11 tightens by 1.9%")

    def test_rate_improves(self):
        """
        At 0 and 100 km the programmes raise the rate by 16% at Curty's intensities and by
        under 2% against an exact vacuum.
        """
        for km in (0.0, 100.0):
            ideal = [0.55, DECOY, VACUUM]
            one = paid(LEVELS, km, True) / paid(LEVELS, km, False)
            two = paid(ideal, km, True) / paid(ideal, km, False)
            self.assertClose(one, 1.16, atol=0.01, msg=f"a sixth at 5e-4, {km} km")
            self.assertLess(two, 1.02, msg=f"a percent at a true vacuum, {km} km")
            self.assertLess(1.0, two, msg=f"still an improvement, {km} km")

    def test_small_cut(self):
        """
        Cutoff 4 at 200 km, valid but no tighter than the closed form, is refused.
        """
        self.assertBad(
            "nothing tighter than the closed form",
            _core.mdi_program,
            (LEVELS, LEVELS, grids(LEVELS, 200.0)[0], 4),
            msg="cut = 4 at 200 km",
        )

    def test_exact_vacuum(self):
        """
        Against a true vacuum the four-cell e11 is already exact and the error programme refuses.
        """
        ideal = [0.55, DECOY, VACUUM]
        gx, faults = grids(ideal, 100.0)[1:]
        y_x = _core.mdi_program(ideal, ideal, gx, CUT)[0]
        self.assertBad(
            "four-cell elimination exact",
            _core.mdi_disturb,
            (ideal, ideal, faults, y_x, CUT),
            msg="vacuum of zero",
        )

    def test_empty_cell(self):
        """
        An empty grid cell, leaving no strictly feasible primal point, is refused.
        """
        grid = list(grids(LEVELS, 50.0)[0])
        grid[8] = 0.0
        self.assertBad(
            "no strictly feasible point",
            _core.mdi_program,
            (LEVELS, LEVELS, grid, CUT),
            msg="empty vacuum-vacuum cell",
        )

    def test_cut_domain(self):
        """
        A truncation below 2 (no (1,1) term) or above 16 (past the solver's row cap) is refused.
        """
        gz = grids(LEVELS, 50.0)[0]

        for cut in (0, 1, 17, 64):
            self.assertBad(
                "photon-number truncation",
                _core.mdi_program,
                (LEVELS, LEVELS, gz, cut),
                msg=f"cut = {cut}",
            )

    def test_yield_needed(self):
        """
        The error programme refuses y11 = 0.
        """
        self.assertBad(
            "strictly positive",
            _core.mdi_disturb,
            (LEVELS, LEVELS, grids(LEVELS, 50.0)[2], 0.0, CUT),
            msg="y11 = 0",
        )


if __name__ == "__main__":
    rc = 0
    rc |= Exam(
        "MdiForward",
        "Ma and Razavi's MDI-BB84 observables at the limits that pin them",
        "mdi_forward.md",
    ).run(load(Forward))
    rc |= Exam(
        "MdiDecoy",
        "The joint two-sender decoy inversion, bounded against the forward model",
        "mdi_decoy.md",
    ).run(load(Decoy))
    rc |= Exam(
        "MdiAnchors",
        "Published MDI-BB84 reach, scaling and rates on the standard parameter set",
        "mdi_anchors.md",
    ).run(load(Anchors))
    rc |= Exam(
        "MdiProgramme",
        "The nine-cell decoy bound as a linear programme, against the seven-cell form",
        "mdi_programme.md",
    ).run(load(Programme))
    rc |= Exam(
        "MdiParts",
        "The components the family adds, and the three security models it is not",
        "mdi_parts.md",
    ).run(load(Parts))
    rc |= Exam(
        "MdiGuards",
        "Out-of-domain MDI-BB84 inputs raise rather than returning a wrong number",
        "mdi_guards.md",
    ).run(load(Guards))
    sys.exit(rc)
