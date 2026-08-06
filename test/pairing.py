import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.forms import bisect, fitslope, plob, poisson
from qkd import _core

# The asynchronous-MDI parameter set: Xie, Lu, Weng, Cao et al., arXiv:2112.11635, Table 1.
# Zeng, Zhou, Wu and Ma (arXiv:2201.04300) run a comparable set, printed only in a figure
# inset, so nothing here is anchored to it.
ALPHA = 0.165
DET = 0.70
DARK = 1e-8
FEC = 1.1

# Xie's X-basis misalignment ANGLE, not a rate; 2.447% through the interferometric law.
SIGMA = math.pi / 10.0
MISALIGN = (1.0 - math.cos(SIGMA)) / 2.0

# Pairing intervals in rounds: Zeng's neighbour-only, and what his 625 MHz run supports.
SHORT = 1
LONG = 10**4

# Yield-ladder truncation: the tail past it is under 1e-30 here, and truncating relaxes it.
NCUT = 25

# Pinned, not counted: no name here resolves a gain by intensity pair or by phase slice.
SURFACE = [
    "pairing_bases",
    "pairing_click",
    "pairing_drift",
    "pairing_eps",
    "pairing_expect",
    "pairing_gamma",
    "pairing_ladder",
    "pairing_length",
    "pairing_observe",
    "pairing_pairs",
    "pairing_phase",
    "pairing_rate",
    "pairing_sift",
    "pairing_single",
    "pairing_span",
    "pairing_yield",
]


def arms(km):
    """
    Single-arm transmittance to a midpoint station over a TOTAL span of km, detectors in.
    """
    return _core.decoy_eta(ALPHA, km / 2.0, DET)


def total(km):
    """
    End-to-end channel transmittance over the whole span, detectors excluded.
    """
    return 10.0 ** (-ALPHA * km / 10.0)


def rate(km, mu, span):
    """
    Bits per ROUND at a symmetric station, phase error from Ma and Razavi through mdi_yield.
    """
    eta = arms(km)

    sift, e_z = _core.pairing_sift(eta, mu, DARK)
    pairs = _core.pairing_pairs(_core.pairing_click(eta, mu, DARK), span)
    q11 = _core.pairing_single(eta, mu, DARK)
    e_x = _core.mdi_yield(eta, eta, DARK, MISALIGN)[1]

    return _core.pairing_rate(pairs, sift, q11, e_x, e_z, FEC)


def best(km, span):
    """
    The rate at the best signal intensity on a fixed grid.
    """
    return max(rate(km, 0.05 * i, span) for i in range(1, 25))


def peak(km, span):
    """
    The intensity the sweep selects.
    """
    return max((rate(km, 0.05 * i, span), round(0.05 * i, 2)) for i in range(1, 25))[1]


def slope(span, spans=(200.0, 220.0, 240.0, 260.0, 280.0)):
    """
    Least-squares log-log slope of the best rate against end-to-end transmittance.
    """
    xs = [math.log10(total(km)) for km in spans]
    ys = [math.log10(best(km, span)) for km in spans]

    return fitslope(xs, ys)


def gains(km, nu, mu):
    """
    The three per-round gains Xie's one-sided inversion takes: decoy, signal, vacuum.
    """
    eta = arms(km)

    return [_core.pairing_click(eta, x, DARK) for x in (nu, mu)] + [2.0 * DARK]


def program(nu, mu, rates):
    """
    pairing_yield as an lp.rs standard-form programme, as (c, A, b, dual start).
    """
    q_nu, q_mu, q_vac = rates
    wide = NCUT + 1

    rows = [[0.0] * (wide + 3) for _ in range(3)]
    rows[0][0] = 1.0
    rows[0][wide] = 1.0
    rows[1][wide + 1] = -1.0
    rows[2][wide + 2] = 1.0

    for k in range(wide):
        rows[1][k] = poisson(nu, k)
        rows[2][k] = poisson(mu, k)

    # Truncated rungs go to the adversary on the decoy row, dropped on the signal: both relax.
    lost = max(0.0, 1.0 - sum(poisson(nu, k) for k in range(wide)))

    cost = [0.0] * (wide + 3)
    cost[1] = 1.0

    # z = c - A^T y is strictly positive here: the decoy multiplier stays under exp(nu - mu).
    start = [-1.0 + 0.5 * math.exp(-mu), 0.5 * math.exp(nu - mu), -1.0]

    return cost, [x for row in rows for x in row], [q_vac, q_nu - lost, q_mu], start


class Strategy(Question):
    def test_long_limit(self):
        """
        `pairing_pairs(p, 10^12)` equals $p/2$ at every $p$: an unbounded interval pairs
        every click.
        """
        for p in (1e-6, 1e-4, 1e-2, 0.5):
            got = _core.pairing_pairs(p, 10**12)

            self.assertClose(got, p / 2.0, msg=f"r_p at p = {p}")

    def test_short_limit(self):
        """
        `pairing_pairs(p, 1)` equals $p^2/(1+p)$, the neighbour-only limit.
        """
        for p in (1e-6, 1e-4, 1e-2):
            got = _core.pairing_pairs(p, 1)

            self.assertClose(got, p * p / (1.0 + p), msg=f"r_p at p = {p}")

    def test_span_linear(self):
        """
        `pairing_pairs(1e-5, l)/l` is flat to 2e-3 at l = 2, 10 and 100.
        """
        base = _core.pairing_pairs(1e-5, 1)

        for span in (2, 10, 100):
            got = _core.pairing_pairs(1e-5, span) / span

            self.assertClose(got / base, 1.0, atol=2e-3, msg=f"per-round at l = {span}")

    def test_span_saturates(self):
        """
        `pairing_pairs(1e-4, l)` never falls with l and reaches 0.774614 of its limit at
        $l = 1/p$, 0.999977 at $l = 10/p$.
        """
        got = [_core.pairing_pairs(1e-4, s) for s in (10**4, 10**5, 10**6, 10**12)]

        self.assertMonotone(got, strict=False, msg="pairs against l")
        self.assertClose(got[0] / got[-1], 0.774614, atol=1e-6, msg="share at l = 1/p")
        self.assertClose(got[1] / got[-1], 0.999977, atol=1e-6, msg="share at l = 10/p")

    def test_pair_ceiling(self):
        """
        `pairing_pairs` is exactly 0.5 when every round announces and under it at p = 0.9,
        0.99 and 1.0.
        """
        self.assertClose(_core.pairing_pairs(1.0, 10**6), 0.5, msg="pairs at p = 1")

        for p in (0.9, 0.99, 1.0):
            self.assertLess(_core.pairing_pairs(p, 10**6), 0.5000000001, msg=f"pairs at p = {p}")

    def test_span_from_laser(self):
        """
        `pairing_span` returns 3125 at 625 MHz and 5 us, against Zeng's quoted three to four
        thousand, and 20000 at 4 GHz.
        """
        self.assertClose(float(_core.pairing_span(625e6, 5e-6)), 3125.0, msg="span at 625 MHz")
        self.assertClose(float(_core.pairing_span(4e9, 5e-6)), 20000.0, msg="span at 4 GHz")

    def test_order_dependent(self):
        """
        The unbounded-to-neighbour pairing ratio is $(1+p)/(2p)$, 5000.5 at $p = 10^{-4}$,
        and the drift excursion rises with the interval: this family is not permutation
        invariant.
        """
        for p in (1e-6, 1e-4, 1e-2):
            wide = _core.pairing_pairs(p, 10**12)
            near = _core.pairing_pairs(p, 1)
            spread = wide / near

            self.assertClose(spread * 2.0 * p / (1.0 + p), 1.0, msg=f"spread at p = {p}")

        self.assertClose(
            _core.pairing_pairs(1e-4, 10**12) / _core.pairing_pairs(1e-4, 1),
            5000.5,
            atol=1e-6,
            msg="spread at p = 1e-4",
        )

        excursion = [_core.pairing_drift(1e9, s, 8e3, 0.0)[0] for s in (1, 10**3, 10**5)]

        self.assertMonotone(excursion, msg="excursion against l")


class Forward(Question):
    def test_sift_eighth(self):
        """
        `pairing_sift` returns 0.125 to 5e-4 at 300, 400 and 500 km.
        """
        for km in (300.0, 400.0, 500.0):
            got = _core.pairing_sift(arms(km), 0.5, DARK)[0]

            self.assertClose(got, 0.125, atol=5e-4, msg=f"r_s at {km} km")

    def test_sift_from_bases(self):
        """
        `pairing_bases` gives each party a Z share of 0.5, half of whose square is the 0.125
        sifting rate.
        """
        share = _core.pairing_bases((0.5, 0.0, 0.5))[0]

        self.assertClose(share, 0.5, msg="Z share")
        self.assertClose(0.5 * share * share, 0.125, msg="sift rate")

    def test_single_fraction(self):
        """
        `pairing_single` tends to exp(-2*mu) as the arm weakens, and is a fraction of sifted
        Z-pairs rather than a gain: a further Poisson prefactor charges the source twice.
        """
        for mu in (0.2, 0.5, 1.0):
            got = _core.pairing_single(1e-6, mu, 0.0)

            self.assertClose(got, math.exp(-2.0 * mu), atol=1e-6, msg=f"limit at mu = {mu}")

        for mu in (0.2, 0.5, 1.0):
            near = _core.pairing_single(arms(500.0), mu, DARK)

            self.assertClose(near, math.exp(-2.0 * mu), atol=5e-3, msg=f"500 km at mu = {mu}")

    def test_error_dark_only(self):
        """
        `pairing_sift` gives a Z-basis error of exactly zero without dark counts, at every
        intensity and transmittance.
        """
        for eta, mu in ((1e-3, 0.5), (0.5, 0.1), (1.0, 1.0)):
            got = _core.pairing_sift(eta, mu, 0.0)[1]

            self.assertClose(got, 0.0, msg=f"E^Z at eta = {eta}, mu = {mu}")

    def test_error_grows(self):
        """
        With dark counts the Z-basis error rate rises with distance, under 1e-4 at 200 km and
        past a percent by 600 km.
        """
        got = [_core.pairing_sift(arms(km), 0.5, DARK)[1] for km in (200.0, 400.0, 600.0)]

        self.assertMonotone(got, msg="E^Z against distance")
        self.assertLess(got[0], 1e-4, msg="E^Z at 200 km")
        self.assertLess(0.01, got[-1], msg="E^Z at 600 km")

    def test_click_linear(self):
        """
        `pairing_click` is linear in the single-arm transmittance, hence in the square root
        of the end-to-end one.
        """
        base = _core.pairing_click(1e-4, 0.5, 0.0)
        half = _core.pairing_click(0.5e-4, 0.5, 0.0)

        self.assertClose(half / base, 0.5, atol=1e-4, msg="p ratio at half arm")


class Window(Question):
    def test_fibre_drift(self):
        """
        Xie's phase-tracking-free case, 8 rad/ms at 402 km over 50 us on a 1 GHz system:
        0.2 rad of excursion and 0.0099667 of error, against his about 1%.
        """
        phase, err = _core.pairing_drift(1e9, 50000, 8e3, 0.0)

        self.assertClose(phase, 0.2, msg="phase, rad")
        self.assertClose(err, 0.0099667, atol=1e-6, msg="e_d against Xie's 1%")

    def test_laser_drift(self):
        """
        Xie's case with neither tracking nor locking, 100 kHz over 1 us on a 10 GHz system:
        pi/10 and 2.447% of error, against the 2.4% he prints.
        """
        phase, err = _core.pairing_drift(1e10, 10000, 2.0 * math.pi * 1e5, 0.0)

        self.assertClose(phase, 0.1 * math.pi, msg="phase, rad")
        self.assertClose(err, 0.0244717, atol=1e-6, msg="e_d against Xie's 2.4%")

    def test_drift_endpoints(self):
        """
        A still reference returns the optical misalignment itself, a perfect interferometer
        the drift alone, and the two add. Neither paper prints the composition; both
        endpoints are theirs.
        """
        for opt in (0.0, 0.01, MISALIGN, 0.2):
            got = _core.pairing_drift(1e9, 1, 0.0, opt)[1]

            self.assertClose(got, opt, msg=f"e_d at opt = {opt}")

        alone = _core.pairing_drift(1e10, 10000, 2.0 * math.pi * 1e5, 0.0)[1]
        both = _core.pairing_drift(1e10, 10000, 2.0 * math.pi * 1e5, MISALIGN)[1]

        self.assertLess(alone, both, msg="drift alone against both")

    def test_drift_wraps(self):
        """
        Past pi the error rate saturates at a half while the raw phase keeps rising past
        1000 rad.
        """
        got = [_core.pairing_drift(1.0, 1, d, 0.0)[1] for d in (6.0, 20.0, 200.0, 2e4)]

        for k, err in enumerate(got):
            self.assertClose(err, 0.5, msg=f"e_d at case {k}")

        self.assertLess(1000.0, _core.pairing_drift(1.0, 1, 2e4, 0.0)[0], msg="raw phase")

    def test_interval_costs(self):
        """
        The interference error is 0.0002467, 0.0244717 and 0.2061074 over 1, 10 and 30 us on
        a 4 GHz system at 10 kHz.
        """
        got = [_core.pairing_drift(4e9, s, 2.0 * math.pi * 1e4, 0.0)[1] for s in (4 * 10**3, 4 * 10**4, 12 * 10**4)]

        self.assertMonotone(got, msg="e_d against window")
        self.assertClose(got[0], 0.0002467, atol=1e-7, msg="e_d at 1 us")
        self.assertClose(got[1], 0.0244717, atol=1e-7, msg="e_d at 10 us")
        self.assertClose(got[2], 0.2061074, atol=1e-7, msg="e_d at 30 us")


class Anchors(Question):
    def test_root_scaling(self):
        """
        The log-log slope against end-to-end transmittance is 0.501848 at a wide interval and
        has saturated by $l = 10^6$.
        """
        got = slope(LONG)

        self.assertClose(got, 0.501848, atol=1e-5, msg="slope against T")
        self.assertClose(slope(10**6), got, atol=1e-5, msg="slope at l = 1e6")

    def test_linear_scaling(self):
        """
        Neighbour-only pairing gives a slope of 0.996627.
        """
        self.assertClose(slope(SHORT), 0.996627, atol=1e-5, msg="slope at l = 1")

    def test_slope_between(self):
        """
        The slope rises monotonically as the interval narrows, 0.762288 at l = 100.
        """
        got = [slope(s) for s in (10**6, 10**4, 10**3, 10**2, 10, 1)]

        self.assertMonotone(got, msg="slope against l")
        self.assertClose(got[3], 0.762288, atol=1e-5, msg="slope at l = 100")

    def test_optimal_intensity(self):
        """
        The sweep selects mu = 0.5 at a wide interval and mu = 1 at a narrow one over 200 to
        350 km, Zeng's two closed-form optima.
        """
        for km in (200.0, 250.0, 300.0, 350.0):
            self.assertClose(peak(km, LONG), 0.5, msg=f"mu at {km} km, l = LONG")
            self.assertClose(peak(km, SHORT), 1.0, msg=f"mu at {km} km, l = 1")

    def test_beats_repeaterless(self):
        """
        The rate crosses the repeaterless capacity at 282.7181 km, 46.6485 dB, and falls back
        under it at 673.3811 km. Capacity as `-log1p(-T)/ln2`: `-log2(1 - T)` is 11% wrong by
        160 dB and exactly zero by 176 dB.
        """
        up = bisect(lambda km: best(km, LONG) - plob(total(km)), 200.0, 500.0)

        self.assertClose(up, 282.7181, atol=1e-3, msg="crossing, km")
        self.assertClose(ALPHA * up, 46.6485, atol=1e-3, msg="crossing, dB")

        down = bisect(lambda km: best(km, LONG) - plob(total(km)), 554.0, 800.0)

        self.assertClose(down, 673.3811, atol=1e-3, msg="return crossing, km")

    def test_needs_a_window(self):
        """
        The peak rate-to-capacity ratio over 200 to 600 km is 0.00476746 at l = 1 and
        0.45254514 at l = 100, both under one, and 4.06498441 at l = 1000 and 30.38392590 at
        $l = 10^4$, both over.
        """
        grid = [200.0 + 20.0 * i for i in range(21)]

        for span, want in ((SHORT, 0.00476746), (10**2, 0.45254514)):
            top = max(best(km, span) / plob(total(km)) for km in grid)

            self.assertClose(top, want, atol=1e-7, msg=f"peak ratio at l = {span}")
            self.assertLess(top, 1.0, msg=f"ratio against 1 at l = {span}")

        for span, want in ((10**3, 4.06498441), (LONG, 30.38392590)):
            top = max(best(km, span) / plob(total(km)) for km in grid)

            self.assertClose(top, want, atol=1e-6, msg=f"peak ratio at l = {span}")
            self.assertLess(1.0, top, msg=f"ratio against 1 at l = {span}")

    def test_window_buys_rate(self):
        """
        The zero-key distance is 676.0949 km at both intervals while the wide one buys 291x
        more rate at 300 km.
        """
        got = [bisect(lambda km: best(km, s), 400.0, 900.0) for s in (SHORT, LONG)]

        self.assertClose(got[0], 676.0949, atol=1e-3, msg="zero-key distance at l = 1, km")
        self.assertClose(got[1], got[0], msg="zero-key distance at l = LONG, km")
        self.assertLess(
            290.0,
            best(300.0, LONG) / best(300.0, SHORT),
            msg="rate ratio at 300 km",
        )

    def test_three_hundred(self):
        """
        The rate at 300 km is 2.245794808e-05 bits per round, at the mu = 0.5 the sweep
        selects.
        """
        self.assertClose(best(300.0, LONG), 2.245794808e-05, atol=1e-13, msg="bits per round")
        self.assertClose(rate(300.0, 0.5, LONG), best(300.0, LONG), msg="rate at mu = 0.5")


class Decoy(Question):
    def test_six_rungs(self):
        """
        `pairing_ladder` returns six distinct pair intensities in ascending order from three
        per-round settings, four more than `decoy_bounds` is written over.
        """
        got = _core.pairing_ladder(0.1, 0.5, (0.5, 0.2, 0.3))

        self.assertClose(float(len(got)), 6.0, msg="rung count")
        self.assertClose(float(len({x for x, _ in got})), 6.0, msg="distinct rungs")

        for want, (x, _) in zip((0.0, 0.1, 0.2, 0.5, 0.6, 1.0), got):
            self.assertClose(x, want, msg=f"rung {want}")

    def test_ladder_normalised(self):
        """
        The six rung probabilities sum to one at three settings triples.
        """
        for probs in ((0.5, 0.2, 0.3), (0.8, 0.1, 0.1), (0.34, 0.33, 0.33)):
            got = sum(w for _, w in _core.pairing_ladder(0.05, 0.4, probs))

            self.assertClose(got, 1.0, msg=f"rung sum at {probs}")

    def test_ladder_collapses(self):
        """
        A signal at twice the decoy merges two of the six rungs -- six entries, five values
        -- and is not refused.
        """
        got = _core.pairing_ladder(0.25, 0.5, (0.5, 0.25, 0.25))

        self.assertClose(float(len(got)), 6.0, msg="rung count")
        self.assertClose(float(len({x for x, _ in got})), 5.0, msg="distinct rungs")

    def test_z_pairs_transfer(self):
        """
        A Z-pair has one empty slot, so the six rungs carry the [signal, decoy, vacuum] ladder
        `mdi_y11` takes, returning 9.617759641e-05 on it unchanged.
        """
        rungs = {x for x, _ in _core.pairing_ladder(0.1, 0.5, (0.5, 0.2, 0.3))}
        ladder = [0.5, 0.1, 0.0]

        for x in ladder:
            self.assertClose(float(x in rungs), 1.0, msg=f"rung {x} present")

        self.assertClose(
            _core.mdi_y11(ladder, ladder, [1e-4] * 9),
            9.617759641e-05,
            atol=1e-13,
            msg="y11 on the sub-ladder",
        )

    def test_basis_table(self):
        """
        Zeng's basis assignment gives 0.5 Z, 0.13 X, 0.25 reserved and 0.12 dropped,
        partitioning one round pair.
        """
        probs = (0.5, 0.2, 0.3)
        z, x, zero, drop = _core.pairing_bases(probs)

        self.assertClose(z + x + zero + drop, 1.0, msg="share sum")
        self.assertClose(z, 0.5, msg="Z share")
        self.assertClose(x, 0.13, msg="X share")
        self.assertClose(zero, 0.25, msg="reserved share")
        self.assertClose(drop, 0.12, msg="dropped share")

    def test_drop_is_real(self):
        """
        The dropped share is zero only when the decoy is never sent.
        """
        self.assertClose(
            _core.pairing_bases((0.5, 0.0, 0.5))[3],
            0.0,
            msg="dropped share at no decoy",
        )
        self.assertLess(0.0, _core.pairing_bases((0.4, 0.3, 0.3))[3], msg="dropped share at 0.4/0.3/0.3")

    def test_fibre_transfers(self):
        """
        The single arm at 300 km is eta_d * 10^(-alpha*150/10).
        """
        self.assertClose(
            arms(300.0),
            DET * 10.0 ** (-ALPHA * 150.0 / 10.0),
            msg="single arm at 300 km",
        )

    def test_lp_reaches_closed(self):
        """
        `lp_dual` climbs to `pairing_yield`'s closed form to 1e-7 and stops under it at four
        spans, so a solver tightens nothing here.
        """
        grid = (
            (200.0, 0.05, 0.4),
            (400.0, 0.1, 0.5),
            (100.0, 0.02, 0.3),
            (800.0, 0.01, 0.1),
        )

        for km, nu, mu in grid:
            rates = gains(km, nu, mu)
            closed = _core.pairing_yield(nu, mu, *rates)
            got = _core.lp_dual(*program(nu, mu, rates), 1e-14)[0]

            self.assertLess(got, closed * (1.0 + 1e-9), msg=f"certified over closed, {km}")
            self.assertClose(got / closed, 1.0, atol=1e-7, msg=f"ratio to closed, {km}")

    def test_lp_has_no_grid(self):
        """
        `mdi_rect` names both senders' intensities, `pairing_click` names one, and no
        `pairing_` name resolves a gain by intensity pair or by phase slice.
        """
        for name in ("mu_a", "mu_b"):
            self.assertIn(
                name,
                _core.mdi_rect.__text_signature__,
                msg=f"mdi_rect names {name}",
            )

        self.assertEqual(
            _core.pairing_click.__text_signature__,
            "(eta_s, mu, dark)",
            msg="pairing_click signature",
        )
        self.assertEqual(
            sorted(n for n in dir(_core) if n.startswith("pairing_")),
            SURFACE,
            msg="the pairing_ surface",
        )


class Guards(Guarded):
    def test_forward_slots(self):
        """
        `pairing_click`, `pairing_sift` and `pairing_single` each refuse an arm transmittance,
        intensity or dark probability outside its domain.
        """
        ok = (0.1, 0.5, DARK)
        cases = [
            (0, "eta_s", (0.0, -0.1, 1.5, float("nan"))),
            (1, "mu", (0.0, -0.1)),
            (2, "dark", (0.6, -1e-9)),
        ]

        self.assertSlots(_core.pairing_click, ok, cases, msg="pairing_click")
        self.assertSlots(_core.pairing_sift, ok, cases, msg="pairing_sift")
        self.assertSlots(_core.pairing_single, ok, cases, msg="pairing_single")

    def test_dark_ceiling(self):
        """
        The refusal of a dark probability past a half names the background announcement floor.
        """
        self.assertBad(
            "background announcement floor",
            _core.pairing_click,
            (0.1, 0.5, 0.7),
            msg="dark = 0.7",
        )

    def test_zero_span(self):
        """
        `pairing_pairs` and `pairing_drift` both refuse a pairing interval of zero rounds.
        """
        self.assertBad(
            "at least 1 round",
            _core.pairing_pairs,
            (1e-4, 0),
            msg="span = 0",
        )
        self.assertBad(
            "at least 1 round",
            _core.pairing_drift,
            (1e9, 0, 8e3, 0.0),
            msg="span = 0, drift model",
        )

    def test_short_coherence(self):
        """
        `pairing_span` refuses a coherence time shorter than one round, and an absurd window
        at the other end.
        """
        self.assertBad(
            "does not survive one round",
            _core.pairing_span,
            (1e6, 1e-9),
            msg="1 kHz laser on a 1 MHz system",
        )
        self.assertBad(
            "nobody meant to type",
            _core.pairing_span,
            (1e12, 1e6),
            msg="1 s coherence at 1 THz",
        )

    def test_rate_ceiling(self):
        """
        `pairing_rate` refuses a pairing rate above a half, which is what an announcement
        probability in that slot produces.
        """
        self.assertBad(
            "at most 0.5",
            _core.pairing_rate,
            (0.8, 0.125, 0.37, 0.02, 1e-5, FEC),
            msg="pairs = 0.8",
        )

    def test_rate_slots(self):
        """
        `pairing_rate` refuses each of its six arguments outside its domain.
        """
        ok = (0.05, 0.125, 0.37, 0.02, 1e-5, FEC)

        self.assertSlots(
            _core.pairing_rate,
            ok,
            [
                (1, "sift", (1.5, -0.1)),
                (2, "q11", (1.5, -0.1)),
                (3, "e_x", (0.6, -1e-9)),
                (4, "e_z", (0.6, -1e-9)),
                (5, "f_ec", (0.9, float("nan"))),
            ],
            msg="pairing_rate",
        )

    def test_ladder_order(self):
        """
        `pairing_ladder` refuses a decoy at or above the signal, and it and `pairing_bases`
        refuse intensity probabilities that do not sum to one.
        """
        self.assertBad(
            "weaker than the signal",
            _core.pairing_ladder,
            (0.5, 0.5, (0.5, 0.2, 0.3)),
            msg="decoy at the signal",
        )
        self.assertBad(
            "must sum to 1",
            _core.pairing_ladder,
            (0.1, 0.5, (0.5, 0.2, 0.2)),
            msg="settings sum to 0.9",
        )
        self.assertBad(
            "must sum to 1",
            _core.pairing_bases,
            ((0.5, 0.5, 0.5),),
            msg="settings sum to 1.5",
        )


if __name__ == "__main__":
    rc = 0
    rc |= Exam(
        "PairingStrategy",
        "Zeng's postmatching combinatorics and the two exponents it interpolates",
        "pairing_strategy.md",
    ).run(load(Strategy))
    rc |= Exam(
        "PairingForward",
        "The mode-pairing observables at the limits Zeng writes closed forms for",
        "pairing_forward.md",
    ).run(load(Forward))
    rc |= Exam(
        "PairingWindow",
        "The pairing interval, the phase drift it must survive, and the trade between",
        "pairing_window.md",
    ).run(load(Window))
    rc |= Exam(
        "PairingAnchors",
        "Square-root scaling, the intensity optimum and the repeaterless crossing",
        "pairing_anchors.md",
    ).run(load(Anchors))
    rc |= Exam(
        "PairingDecoy",
        "What the pair intensity ladder inherits from the one-dimensional layer",
        "pairing_decoy.md",
    ).run(load(Decoy))
    rc |= Exam(
        "PairingGuards",
        "Out-of-domain mode-pairing inputs raise rather than returning a wrong number",
        "pairing_guards.md",
    ).run(load(Guards))
    sys.exit(rc)
