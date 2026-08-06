import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

from kit.cache import memo
from kit.checks import Guarded
from kit.forms import band
from kit.procs import threads
from qkd import _core, impairments

RATE = 1e9

# Eraerds, Walenta, Legre, Gisin & Zbinden, New J. Phys. 12, 063027 (2010), arXiv:0912.1798,
# Sec. III.1: a 45 pm (fwhm) filter and 1.5 ns gates, read at their eta = 0.07.
FILTER = 45e-12

GATE = 1.5e-9

DETECT = 0.07

# impairments.raman_photons' wavelength: both sides of the occupancy-power identity read one photon energy.
WAVE = 1531.12e-9

PLANCK = 6.62607015e-34

LIGHT = 299792458.0

# 0 dBm, 25 km, 0.2 dB/km, Kumar, Qin & Alleaume's worst C-band Raman coefficient; co-propagating span L*exp(-alpha*L).
LAUNCH = 1e-3

SPAN = 25.0

# Fibre nonlinear coefficient per W per m; not kit.anchors.BETA, a reconciliation efficiency.
BETA = 3.0e-9

# rayon reads RAYON_NUM_THREADS once, so this probe needs a fresh process.
CHILD = (
    "import hashlib\n"
    "import qkd._core as c\n"
    "o = c.run_clicks(400000, 7, 0.3, 0.6, 0.4, 1e-5, 0.96, 1, 2e7, "
    f"{RATE!r}, 20e-9, 0.03, 4096, True)\n"
    "b = bytes(o.record_d0) + bytes(o.record_d1) + bytes(o.record_bit)\n"
    "print(hashlib.sha256(b).hexdigest(), o.sifted, o.errors, repr(o.qber), "
    "repr(o.visibility), repr(o.v_phase))\n"
)


@memo
def run(
    n=1_000_000,
    seed=1,
    mu=0.25,
    t=1.0,
    eta=0.4,
    dark=0.0,
    vis=1.0,
    delay=1,
    lw=0.0,
    dead=0.0,
    after=0.0,
    chunk=4096,
    record=False,
    jitter=0.0,
    window=0.0,
    frac=0.0,
    tau=0.0,
    eta1=None,
    dark1=None,
    dead1=None,
    after1=None,
    shape=None,
    shape1=None,
):
    """
    One memoised DPS run at 1 GHz, ideal unless overridden; eta1/dark1/dead1/after1 are the
    pi-phase detector's, None keeping its partner's, and shape/shape1 are measured afterpulse
    curves.
    """

    return _core.run_clicks(
        n,
        seed,
        mu,
        t,
        eta,
        dark,
        vis,
        delay,
        lw,
        RATE,
        dead,
        after,
        chunk,
        record,
        jitter,
        window,
        frac,
        tau,
        eta1,
        dark1,
        dead1,
        after1,
        shape,
        shape1,
    )


def after_curve(total, tau, span):
    """
    A normalised exponential afterpulse curve, `total` over `span` gates at 1/e length `tau` --
    a shape to feed, not one the engine assumes, Ziarkash, Joshi, Stipcevic and Ursin, Sci. Rep.
    8, 5076 (2018) finding no universal one.
    """
    w = [math.exp(-m / tau) for m in range(span)]
    s = sum(w)

    return tuple(total * x / s for x in w)


def gate_draws(seed, stage, n):
    """
    The engine's own Threefry uniforms for one stage as an (n, 4) array, (block(k) | 1) *
    2**-32.
    """
    raw = np.asarray(_core.cpu_words(seed, stage, 0, n, 0), dtype=np.uint64)

    return ((raw | 1).astype(np.float64) * 2.0**-32).reshape(n, 4)


def memory_model(n, seed, mu, eta, dark, vis, dead, after, shape):
    """
    Independent serial reference for the detector-memory scan on the engine's own stages,
    assuming mono walk, unit transmittance, delay 1 and no arrival-time response.
    """
    key = gate_draws(seed, 20, n)
    gate = gate_draws(seed, 21, n)
    coin = gate_draws(seed, 22, n)
    bits = (key[:, 0] < 0.5).astype(np.uint8)
    half = 0.5 * mu

    def port(j, v):
        return 1.0 - (1.0 - dark[j]) * math.exp(-eta[j] * half * (1.0 + v))

    d0 = (port(0, vis), port(0, -vis))
    d1 = (port(1, vis), port(1, -vis))
    curved = any(len(x) for x in shape)
    wheel = max(len(shape[0]), len(shape[1])) + 1
    ring = [[1.0] * wheel, [1.0] * wheel]
    blind = [min(int(math.ceil(x * RATE)), n) for x in dead]
    live = [0, 0]
    armed = [False, False]
    hits = [0, 0]
    wrong = [0, 0]
    sift = 0
    errs = 0
    dbl = 0
    for k in range(1, n):
        bit = int(bits[k] ^ bits[k - 1])
        pair = (d0[0], d1[1]) if bit == 0 else (d0[1], d1[0])
        u = gate[k]
        pend = [0.0, 0.0]
        if curved:
            at = k % wheel
            for j in (0, 1):
                pend[j] = 1.0 - ring[j][at]
                ring[j][at] = 1.0

        fired = [False, False]
        for j in (0, 1):
            if k < live[j]:
                continue

            spur = u[2 + j] < (pend[j] if curved else armed[j] * after[j])
            armed[j] = False
            if u[j] < pair[j] or spur:
                fired[j] = True
                live[j] = k + 1 + blind[j]
                armed[j] = True
                hits[j] += 1
                for m, q in enumerate(shape[j]):
                    ring[j][(k + 1 + m) % wheel] *= 1.0 - q

        if fired[0] and fired[1]:
            dbl += 1
            sift += 1
            errs += int((coin[k][0] < 0.5) != bit)
        elif fired[0]:
            sift += 1
            wrong[0] += int(bit != 0)
            errs += int(bit != 0)
        elif fired[1]:
            sift += 1
            wrong[1] += int(bit != 1)
            errs += int(bit != 1)

    return (hits[0], hits[1], sift, errs, dbl, wrong[0], wrong[1])


def counters(out):
    """
    Every integer the memory scan produces, in memory_model's order.
    """

    return (
        out.clicks_d0,
        out.clicks_d1,
        out.sifted,
        out.errors,
        out.doubles,
        out.errors_d0,
        out.errors_d1,
    )


def weights(jitter, window, frac=0.0, tau=0.0, span=64):
    """
    Acceptance weights on the 1 GHz slot grid, Gaussian core plus one-sided exponential tail,
    from math.erf rather than the engine's Gauss-Legendre panels.
    """
    ts = 1.0 / RATE
    sd = jitter / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = 0.5 * (window if window > 0.0 else ts)
    out = []
    for m in range(-span, span + 1):
        lo = m * ts - half
        hi = m * ts + half
        core = float(lo <= 0.0 < hi)
        drag = core
        if sd > 0.0:
            root = sd * np.sqrt(2.0)
            core = 0.5 * (math.erf(hi / root) - math.erf(lo / root))

        if tau > 0.0:
            drag = np.exp(-max(lo, 0.0) / tau) - np.exp(-max(hi, 0.0) / tau)

        out.append((1.0 - frac) * core + frac * drag)

    return out


def split_model(jitter, window, frac=0.0, tau=0.0):
    """
    Detections landing in no window at all, and accepted ones in a neighbour's.
    """
    w = weights(jitter, window, frac, tau)
    total = sum(w)
    own = w[len(w) // 2]

    return max(0.0, 1.0 - total), (total - own) / total


def sigma_phi(lw, delay=1):
    """
    Lorentzian phase walk over ``delay`` periods: sqrt(2*pi*lw*delay/R).
    """

    return float(np.sqrt(2.0 * np.pi * lw * delay / RATE))


def digest(out):
    """
    Every reported quantity as one tuple: record hashed, scalars by exact repr.
    """

    body = bytes(out.record_d0) + bytes(out.record_d1) + bytes(out.record_bit)

    return (
        hashlib.sha256(body).hexdigest(),
        out.sifted,
        out.errors,
        out.clicks,
        repr(out.qber),
        repr(out.visibility),
        repr(out.v_phase),
    )


def qber_model(mu, eta, dark, vis, sigma):
    """
    Closed-form one-slot QBER, exact in flux and Gauss-Hermite phase-averaged, under squashing:
    a slot counts when either detector fires and doubles keep a random bit.
    """

    x, w = np.polynomial.hermite.hermgauss(96)
    delta = np.sqrt(2.0) * sigma * x
    fringe = vis * np.cos(delta)
    ip = 0.5 * mu * (1.0 + fringe)
    im = 0.5 * mu * (1.0 - fringe)
    qp = (1.0 - dark) * np.exp(-eta * ip)
    qm = (1.0 - dark) * np.exp(-eta * im)
    both = (1.0 - qp) * (1.0 - qm)
    err = (1.0 - qm) * qp + 0.5 * both
    sift = 1.0 - qp * qm
    wgt = w / np.sqrt(np.pi)

    return float((wgt * err).sum() / (wgt * sift).sum())


def raman_point():
    """
    (occupancy per detection mode, passband in Hz): Kumar, Qin and Alleaume Eq. (6) at 0 dBm
    over 25 km, behind Eraerds' 45 pm filter as c dl/l^2.
    """
    occ = impairments.raman_photons(LAUNCH, SPAN, beta=BETA, wavelength=WAVE)

    return occ, LIGHT * FILTER / (WAVE * WAVE)


def raman_power():
    """
    Photons in one gate by Eraerds' route: Eq. (2) forward power P_out*L*rho*dlambda through Eq.
    (11), P_out read at the far end so the span factor matches Kumar's.
    """
    out = LAUNCH * math.exp(-SPAN * 0.2 / (10.0 / math.log(10.0)))
    watts = out * SPAN * BETA * (FILTER * 1e9)

    return watts / (PLANCK * LIGHT / WAVE) * GATE


def slot_model(mu, eta, dark, vis, sigma):
    """
    One slot in closed form with Bob's detectors allowed to differ, averaged over Alice's
    differential bit, eta and dark as (d0, d1) pairs.
    """
    x, w = np.polynomial.hermite.hermgauss(96)
    fringe = vis * np.cos(np.sqrt(2.0) * sigma * x)
    wgt = w / np.sqrt(np.pi)
    fires = [0.0, 0.0]
    errs = [0.0, 0.0]
    both = 0.0
    sift = 0.0
    for bit in (0, 1):
        sign = 1.0 - 2.0 * bit
        ip = 0.5 * mu * (1.0 + sign * fringe)
        im = 0.5 * mu * (1.0 - sign * fringe)
        q0 = (1.0 - dark[0]) * np.exp(-eta[0] * ip)
        q1 = (1.0 - dark[1]) * np.exp(-eta[1] * im)
        wrong = (1.0 - q1) * q0 if bit == 0 else (1.0 - q0) * q1
        errs[1 - bit] += 0.5 * (wgt * wrong).sum()
        both += 0.5 * (wgt * (1.0 - q0) * (1.0 - q1)).sum()
        sift += 0.5 * (wgt * (1.0 - q0 * q1)).sum()
        fires[0] += 0.5 * (wgt * (1.0 - q0)).sum()
        fires[1] += 0.5 * (wgt * (1.0 - q1)).sum()

    return {
        "d0": float(fires[0]),
        "d1": float(fires[1]),
        "err0": float(errs[0]),
        "err1": float(errs[1]),
        "both": float(both),
        "sift": float(sift),
    }


def pair_model(mu, eta, dark, vis, sigma):
    """
    qber_model with the pair split: lone errors from either detector plus half the doubles, over
    the sift.
    """
    m = slot_model(mu, eta, dark, vis, sigma)

    return (m["err0"] + m["err1"] + 0.5 * m["both"]) / m["sift"]


def basis_model(mu, probs, t, eta, dark, misalign):
    """
    One basis-keyed pulse in closed form at sifting 1, averaged over Alice's bit and the
    misalignment coin, eta and dark as (d0, d1) pairs.
    """
    fires = [0.0, 0.0]
    gain = []
    errs = 0.0
    kept = 0.0
    for i, weight in enumerate(probs):
        hit = [1.0 - math.exp(-t * eta[j] * mu[i]) for j in (0, 1)]
        seen = 0.0
        for bit in (0, 1):
            for flip, odds in ((0, 1.0 - misalign), (1, misalign)):
                lands = bit ^ flip
                p0 = 1.0 - (1.0 - dark[0]) * (1.0 - (hit[0] if lands == 0 else 0.0))
                p1 = 1.0 - (1.0 - dark[1]) * (1.0 - (hit[1] if lands == 1 else 0.0))
                share = 0.5 * odds
                click = 1.0 - (1.0 - p0) * (1.0 - p1)
                wrong = p0 * (1.0 - p1) * float(bit == 1)
                wrong += (1.0 - p0) * p1 * float(bit == 0)
                wrong += 0.5 * p0 * p1
                seen += share * click
                errs += weight * share * wrong
                kept += weight * share * click
                fires[0] += weight * share * p0
                fires[1] += weight * share * p1

        gain.append(seen)

    return {
        "d0": fires[0],
        "d1": fires[1],
        "gain": gain,
        "qber": errs / kept,
    }


def counts(out):
    """
    Every count a basis-keyed run reports as one tuple, floats by exact repr.
    """

    return (
        tuple(out.sent),
        tuple(out.clicks),
        tuple(out.sifted),
        tuple(out.errors),
        tuple(out.key_sifted),
        tuple(out.key_errors),
        out.doubles,
        out.clicks_d0,
        out.clicks_d1,
        tuple(repr(g) for g in out.gain),
        tuple(repr(q) for q in out.qber),
    )


class Optics(Guarded):
    """
    Threshold-detector vacuum-overlap click probability and the delay-line stencil, in photon
    number, never quadratures.
    """

    def test_stencil_ports(self):
        """
        In-phase pulses fill the constructive port and anti-phase the destructive one.
        """
        same = _core.interfere(1.0, 0.0, 1.0, 0.0, 1.0)
        flip = _core.interfere(1.0, 0.0, -1.0, 0.0, 1.0)

        self.assertClose(same[0], 1.0, msg="in-phase must fill the bright port")
        self.assertClose(same[1], 0.0, msg="in-phase must empty the dim port")
        self.assertClose(flip[0], 0.0, msg="anti-phase must empty the bright port")
        self.assertClose(flip[1], 1.0, msg="anti-phase must fill the dim port")

    def test_stencil_flux(self):
        """
        One slot's ports sum to half the two pulses' photons at any visibility or phase, where
        the /sqrt(2) two-mode form would double every count rate.
        """
        for vis in (0.0, 0.5, 1.0):
            for phi in (0.0, 0.7, 2.1, np.pi):
                ports = _core.interfere(1.0, 0.0, np.cos(phi), np.sin(phi), vis)
                self.assertClose(ports[0] + ports[1], 1.0, msg=f"flux leak at vis={vis} phi={phi}")

    def test_stencil_visibility(self):
        """
        Imperfect contrast scales the interference term alone, so an in-phase slot reproduces $V
        = (I_{max} - I_{min})/(I_{max} + I_{min})$.
        """
        for vis in (0.2, 0.5, 0.9, 1.0):
            hi, lo = _core.interfere(1.0, 0.0, 1.0, 0.0, vis)
            self.assertClose((hi - lo) / (hi + lo), vis, msg=f"contrast at V={vis}")

    def test_stencil_overflow(self):
        """
        `interfere` raises once $\\lvert\\alpha\\rvert^2/4$ leaves f64 range rather than return
        an (inf, nan) that NaN comparisons would hide, while `click_prob` saturates.
        """
        for mag in (1.0, 1e6, 1e150, 1e154):
            hi, lo = _core.interfere(mag, 0.0, mag, 0.0, 1.0)
            self.assertClose(
                hi / (mag * mag),
                1.0,
                msg=f"in-phase flux at |alpha| = {mag:g}",
            )
            self.assertClose(lo, 0.0, msg=f"perfect contrast nulls at {mag:g}")

        for mag in (1e160, 1e300):
            self.assertBad(
                "overflows f64",
                _core.interfere,
                (mag, 0.0, mag, 0.0, 1.0),
                msg=f"|alpha| = {mag:g}",
            )
        self.assertClose(
            _core.click_prob(1e300, 0.0, 1.0, 0.0),
            1.0,
            msg="click_prob at |alpha| = 1e300",
        )

    def test_rejects_nonsense(self):
        """
        Efficiency, dark probability, visibility and interferometer delay are each refused by
        name.
        """
        self.assertBad(
            "eta",
            _core.click_prob,
            (1.0, 0.0, 1.5, 0.0),
            msg="eta = 1.5",
        )
        self.assertBad(
            "dark",
            _core.click_prob,
            (1.0, 0.0, 0.5, 1.0),
            msg="dark = 1",
        )
        self.assertBad(
            "vis",
            _core.interfere,
            (1.0, 0.0, 1.0, 0.0, 1.2),
            msg="vis = 1.2",
        )
        self.assertBad(
            "delay",
            _core.run_clicks,
            (1000, 1, 0.1, 1.0, 0.2, 0.0, 1.0, 0, 0.0, RATE, 0.0, 0.0, 4096, False),
            msg="delay = 0",
        )


class Dps(Question):
    """
    QBER as an output of the hardware, every value checked against ``qber_model``.
    """

    def test_zero_noise(self):
        """
        With no dark counts, linewidth or dead time the dim port is empty: QBER is exactly 0.0
        at sift rate $1 - e^{-\\eta\\mu}$.
        """
        out = run(n=1_000_000, mu=0.4, eta=0.5)

        self.assertEqual(out.errors, 0, msg=f"{out.errors} errors with no noise")
        self.assertClose(out.visibility, 1.0, msg=f"visibility {out.visibility}")
        self.assertClose(
            out.sift_rate,
            1.0 - np.exp(-0.2),
            atol=0.002,
            msg=f"sift rate {out.sift_rate:.5f} != 1 - exp(-eta*mu)",
        )

    def test_visibility_law(self):
        """
        With no dark counts and a noiseless laser the QBER over a visibility sweep lands on $e =
        (1 - V)/2$ and on the finite-flux form.
        """
        for vis in (1.0, 0.98, 0.94, 0.88, 0.8):
            out = run(vis=vis)
            want = qber_model(0.25, 0.4, 0.0, vis, 0.0)
            self.assertClose(out.visibility, vis, atol=1e-9, msg=f"visibility {out.visibility}")
            self.assertClose(out.qber, want, atol=0.004, msg=f"V={vis}: {out.qber:.5f} vs {want:.5f}")

    def test_linewidth_raises_qber(self):
        """
        QBER rises with linewidth and tracks the finite-flux form from 0 to 50 MHz, a walk
        $\\sigma^2 = 2\\pi \\Delta\\nu \\, d/R$ from 0 to 0.31 rad^2.
        """
        errs = []
        for lw in (0.0, 1e6, 1e7, 3e7, 5e7):
            out = run(lw=lw)
            sigma = sigma_phi(lw)
            want = qber_model(0.25, 0.4, 0.0, 1.0, sigma)
            errs.append(out.qber)
            self.assertClose(
                out.qber,
                want,
                atol=0.004,
                msg=f"lw={lw:.0e}: {out.qber:.5f} vs {want:.5f}",
            )
        self.assertMonotone(errs, msg=f"QBER: {errs}")

    def test_phase_walk_variance(self):
        """
        The differential phase variance is the quadrature pipeline's Wiener walk $2\\pi
        \\Delta\\nu \\, d/R$, leaving visibility $V e^{-\\sigma^2/2}$.
        """
        for lw in (1e6, 1e7, 5e7):
            out = run(n=500_000, lw=lw)
            sigma = sigma_phi(lw)
            self.assertClose(
                out.v_phase,
                sigma * sigma,
                atol=0.1 * sigma * sigma,
                msg=f"lw={lw:.0e}: v_phase {out.v_phase:.5f} vs {sigma * sigma:.5f}",
            )
            self.assertClose(
                out.visibility,
                np.exp(-0.5 * sigma * sigma),
                atol=0.01,
                msg=f"lw={lw:.0e}: visibility {out.visibility:.5f}",
            )

    def test_delay_scales_walk(self):
        """
        A two-period delay line doubles the integrated phase variance, and the QBER follows the
        closed form with the longer arm.
        """
        one = run(n=800_000, lw=2e7, delay=1)
        two = run(n=800_000, lw=2e7, delay=2)

        self.assertClose(
            two.v_phase,
            2.0 * one.v_phase,
            atol=0.1 * one.v_phase,
            msg=f"delay=2 variance {two.v_phase:.5f} vs {one.v_phase:.5f}",
        )

        want = qber_model(0.25, 0.4, 0.0, 1.0, sigma_phi(2e7, delay=2))

        self.assertClose(two.qber, want, atol=0.005, msg=f"delay=2: {two.qber:.5f} vs {want:.5f}")
        self.assertGreater(two.qber, one.qber, msg=f"QBER {two.qber} vs {one.qber}")

    def test_dark_count_floor(self):
        """
        As $\\eta\\mu$ falls at nonzero dark probability the QBER climbs to 1/2, the 1e-2 dark
        probability exaggerated to reach the floor, so the shape is the anchor, not the number.
        """
        errs = []
        for mu in (0.2, 0.02, 2e-3, 2e-4, 2e-5):
            out = run(mu=mu, eta=0.5, dark=1e-2)
            want = qber_model(mu, 0.5, 1e-2, 1.0, 0.0)
            errs.append(out.qber)
            self.assertClose(
                out.qber,
                want,
                atol=0.02,
                msg=f"mu={mu:.0e}: {out.qber:.5f} vs {want:.5f}",
            )
        self.assertMonotone(errs, msg=f"QBER: {errs}")
        self.assertClose(errs[-1], 0.5, atol=0.02, msg=f"floor is {errs[-1]:.5f}")

    def test_dead_time(self):
        """
        Raising dead time strictly lowers the click rate and cannot lower the QBER.
        """
        rates = []
        errs = []
        for dead in (0.0, 5e-9, 20e-9, 100e-9):
            out = run(n=600_000, mu=0.4, eta=0.5, vis=0.94, dead=dead)
            rates.append(out.click_rate)
            errs.append(out.qber)
        self.assertMonotone(rates, rising=False, msg=f"click rates: {rates}")

        for i, e in enumerate(errs):
            self.assertGreaterEqual(
                e,
                errs[0] - 0.002,
                msg=f"dead time {i} lowered QBER: {e:.5f} < {errs[0]:.5f}",
            )

    def test_afterpulsing(self):
        """
        A 5% afterpulse, landing on either detector equally, lifts the click rate and drags the
        QBER toward 1/2 without destroying the key.
        """
        off = run(n=600_000, mu=0.4, eta=0.5, vis=0.94)
        on = run(n=600_000, mu=0.4, eta=0.5, vis=0.94, after=0.05)

        self.assertGreater(on.click_rate, off.click_rate, msg=f"click rate {on.click_rate} vs {off.click_rate}")
        self.assertGreater(on.qber, off.qber, msg=f"QBER {on.qber} vs {off.qber}")
        self.assertLess(on.qber, 0.5, msg=f"QBER {on.qber}")

    def test_curve_reduces(self):
        """
        A one-entry afterpulse curve at zero hold-off reproduces the flat coin's record digest,
        counts and QBER.
        """
        args = dict(n=400_000, seed=7, mu=0.3, t=0.6, eta=0.4, dark=1e-5)
        flat = run(vis=0.96, lw=2e7, after=0.03, record=True, **args)
        curve = run(vis=0.96, lw=2e7, after=0.03, record=True, shape=(0.03,), **args)

        self.assertEqual(digest(curve), digest(flat), msg="digest, one-entry curve vs flat coin")

    def test_curve_discards(self):
        """
        A carrier released during hold-off is lost, not deferred: a two-gate curve under a 50 ns
        hold-off reproduces the afterpulse-free digest, while the flat coin charges its whole
        probability at any hold-off.
        """
        args = dict(n=200_000, seed=7, mu=0.3, t=0.6, eta=0.4, dark=1e-5)
        off = run(vis=0.96, lw=2e7, dead=50e-9, record=True, **args)
        curve = run(
            vis=0.96,
            lw=2e7,
            dead=50e-9,
            after=0.03,
            record=True,
            shape=(0.02, 0.01),
            **args,
        )
        flat = run(vis=0.96, lw=2e7, dead=50e-9, after=0.03, record=True, **args)

        self.assertEqual(digest(curve), digest(off), msg="digest, held-off curve vs no afterpulse")
        self.assertNotEqual(digest(flat), digest(off), msg="digest, flat coin vs no afterpulse")

    def test_holdoff_reaches(self):
        """
        The flat coin's QBER ignores hold-off where a 20-gate curve of the same 3% decays to the
        bare floor, so the shipped QBER is pessimistic, 57% relative at 160 ns.
        """
        args = dict(n=600_000, mu=0.4, eta=0.5, vis=0.94)
        curve = after_curve(0.03, 20.0, 120)
        near = [run(dead=20e-9, **args), run(dead=20e-9, after=0.03, **args)]
        near.append(run(dead=20e-9, shape=curve, **args))

        self.assertClose(near[0].qber, 0.031811, atol=5e-6, msg="bare QBER moved")
        self.assertClose(near[1].qber, 0.045587, atol=5e-6, msg="flat coin QBER moved")
        self.assertClose(near[2].qber, 0.034133, atol=5e-6, msg="curve QBER moved")

        excess = [near[1].qber - near[0].qber, near[2].qber - near[0].qber]

        self.assertClose(
            excess[0] / excess[1],
            5.93,
            atol=0.01,
            msg=f"excess: {excess}",
        )

        far = [
            run(dead=160e-9, after=0.03, **args),
            run(dead=160e-9, shape=curve, **args),
        ]

        self.assertClose(far[0].qber, 0.050071, atol=5e-6, msg="flat coin at 160 ns")
        self.assertClose(far[1].qber, 0.031913, atol=5e-6, msg="curve at 160 ns")
        self.assertGreater(far[0].clicks, far[1].clicks, msg=f"clicks {far[0].clicks} vs {far[1].clicks}")

        zero = [run(after=0.03, **args), run(shape=curve, **args)]

        self.assertClose(
            zero[0].qber / zero[1].qber,
            1.0,
            atol=0.02,
            msg=f"QBER {zero[0].qber} vs {zero[1].qber}",
        )

    def test_memory_exact(self):
        """
        An independent serial reference on the engine's Threefry stages agrees on all seven
        counters to the integer across five hold-off, afterpulse and curve configurations.
        """
        cases = (
            ((0.0, 0.0), (0.03, 0.03), ((), ())),
            ((2e-9, 7e-9), (0.03, 0.05), ((), ())),
            ((0.0, 0.0), (0.03, 0.05), ((0.03,), (0.05,))),
            ((6e-9, 6e-9), (0.0, 0.0), (after_curve(0.05, 3.0, 9),) * 2),
            ((11e-9, 3e-9), (0.01, 0.02), (after_curve(0.06, 4.0, 7), (0.03, 0.02))),
        )
        for dead, after, shape in cases:
            out = run(
                n=30_000,
                seed=5,
                mu=0.4,
                eta=0.5,
                dark=1e-3,
                vis=0.9,
                dead=dead[0],
                after=after[0],
                eta1=0.3,
                dark1=2e-3,
                dead1=dead[1],
                after1=after[1],
                shape=shape[0] or None,
                shape1=shape[1] or None,
            )
            want = memory_model(30_000, 5, 0.4, (0.5, 0.3), (1e-3, 2e-3), 0.9, dead, after, shape)
            self.assertEqual(counters(out), want, msg=f"memory scan drifted at {dead} {after}")

    def test_split_memory(self):
        """
        Hold-off and afterpulse rate split per detector as eta and dark do: equal values
        reproduce the pooled run bit for bit, and either on the pi-phase detector alone moves
        only its count.
        """
        args = dict(n=400_000, seed=3, mu=0.4, eta=0.5, dark=1e-4, vis=0.94)
        same = run(dead=20e-9, after=0.05, record=True, **args)
        split = run(dead=20e-9, after=0.05, dead1=20e-9, after1=0.05, record=True, **args)

        self.assertEqual(digest(split), digest(same), msg="digest, equal pair vs pooled")

        free = run(record=True, **args)
        slow = run(dead1=20e-9, record=True, **args)
        noisy = run(after1=0.20, record=True, **args)

        self.assertEqual(
            (slow.clicks_d0, noisy.clicks_d0),
            (free.clicks_d0, free.clicks_d0),
            msg="clicks_d0 under dead1, after1",
        )
        self.assertLess(slow.clicks_d1, free.clicks_d1, msg=f"clicks_d1 {slow.clicks_d1} vs {free.clicks_d1}")
        self.assertGreater(noisy.clicks_d1, free.clicks_d1, msg=f"clicks_d1 {noisy.clicks_d1} vs {free.clicks_d1}")

    def test_recovery_refused(self):
        """
        A partial-efficiency recovery ramp is refused naming bias current, kinetic inductance,
        the eta/dark/response split and latching, while zero and None run as the gated detector.
        """
        args = (1000, 1, 0.2, 1.0, 0.5, 0.01, 0.98, 1, 0.0, RATE, 0.0, 0.0, 4096, False)
        tail = (0.0, 0.0, 0.0, 0.0, None, None, None, None, None, None)
        for needle in ("bias current", "kinetic inductance", "LATCHES", "gated"):
            self.assertFails(
                NotImplementedError,
                needle,
                _core.run_clicks,
                *(args + tail + (5e-9,)),
                msg=f"recovery refusal {needle!r}",
            )

        for good in (None, 0.0):
            self.assertFinite(
                _core.run_clicks(*(args + tail + (good,))).qber,
                msg=f"recovery={good}",
            )

    def test_curve_rejects(self):
        """
        An empty curve, an entry outside [0, 1) and a curve summing to 1 or more are refused by
        name on either detector.
        """
        args = (1000, 1, 0.2, 1.0, 0.5, 0.01, 0.98, 1, 0.0, RATE, 0.0, 0.0, 4096, False)
        head = (0.0, 0.0, 0.0, 0.0, None, None, None, None)
        for slot, name in ((0, "after_shape"), (1, "after_shape_d1")):
            for bad in ((), (1.0,), (-0.1,), (0.4, 0.7)):
                tail = [None, None]
                tail[slot] = bad
                self.assertFails(
                    ValueError,
                    name,
                    _core.run_clicks,
                    *(args + head + tuple(tail)),
                    msg=f"{name} = {bad}",
                )

    def test_doubles_squashed(self):
        """
        Under squashing, doubles are a strict subset of the sifted key, the click total counts
        each twice, and the QBER lands on the squashing closed form.
        """
        out = run(n=400_000, mu=0.45, eta=0.9, dark=1e-3, vis=0.99)

        self.assertGreater(out.doubles, 0, msg=f"doubles {out.doubles}")
        self.assertLess(out.doubles, out.sifted, msg=f"doubles {out.doubles}, sifted {out.sifted}")
        self.assertEqual(
            out.clicks,
            out.sifted + out.doubles,
            msg="clicks = sifted + doubles",
        )
        self.assertClose(
            out.qber,
            qber_model(0.45, 0.9, 1e-3, 0.99, 0.0),
            atol=5e-4,
            msg=f"QBER {out.qber:.6f}",
        )

    def test_doubles_carry_blinding(self):
        """
        Dark counts push the receiver toward doubles, as blinding does, and under squashing the
        QBER climbs with them.
        """
        fracs = []
        errs = []
        for dark in (1e-5, 1e-3, 1e-2, 5e-2):
            out = run(n=200_000, mu=0.05, eta=0.5, dark=dark)
            fracs.append(out.doubles / out.sifted)
            errs.append(out.qber)
        self.assertMonotone(fracs, msg=f"double fractions: {fracs}")
        self.assertMonotone(errs, msg=f"QBER: {errs}")
        self.assertLess(errs[-1], 0.5, msg=f"QBER {errs[-1]}")

    def test_chunk_invariance(self):
        """
        Every draw is a pure function of (seed, stage, slot index), so chunk sizes two orders
        apart give identical records and scalars.
        """
        cfg = {
            "n": 400_000,
            "seed": 3,
            "lw": 1e7,
            "dark": 1e-5,
            "dead": 20e-9,
            "after": 0.03,
            "record": True,
        }
        base = digest(run(chunk=512, **cfg))
        for chunk in (65536, 0):
            got = digest(run(chunk=chunk, **cfg))
            self.assertEqual(got, base, msg=f"chunk={chunk} changed the result")

    def test_thread_invariance(self):
        """
        The stateful dead-time and afterpulse scan runs in index order, so 1 and 8 workers agree
        bitwise.
        """
        one = threads(CHILD, 1)
        many = threads(CHILD, 8)

        self.assertEqual(one[0], many[0], msg="record hash, 1 vs 8 threads")
        self.assertEqual(one[1:], many[1:], msg=f"scalars: {one[1:]} vs {many[1:]}")

    def test_seed_decorrelates(self):
        """
        A second seed reproduces the same physics while differing bitwise.
        """
        a = run(n=400_000, seed=3, lw=1e7, record=True, share=False)
        b = run(n=400_000, seed=3, lw=1e7, record=True, share=False)
        c = run(n=400_000, seed=4, lw=1e7, record=True, share=False)

        self.assertEqual(digest(a), digest(b), msg="digest, seed 3 twice")
        self.assertNotEqual(digest(a), digest(c), msg="digest, seed 3 vs 4")
        self.assertClose(a.qber, c.qber, atol=0.01, msg=f"QBER {a.qber} vs {c.qber}")


class Jitter(Guarded):
    """
    Light in no window is loss and light in a neighbour's an error at half weight; Gaussian core
    plus exponential tail.
    """

    def test_jitter_off(self):
        """
        Window and tail time set with zero jitter and tail weight reproduce a jitter-free run
        bit for bit, at zero loss and leak.
        """
        cfg = {
            "n": 400_000,
            "seed": 7,
            "mu": 0.3,
            "t": 0.6,
            "vis": 0.96,
            "lw": 2e7,
            "dark": 1e-5,
            "dead": 20e-9,
            "after": 0.03,
            "record": True,
            "share": False,
        }
        base = digest(run(**cfg))
        for window, tau in ((0.0, 0.0), (100e-12, 0.0), (0.0, 3e-10), (5e-10, 1e-9)):
            got = digest(run(window=window, tau=tau, **cfg))
            self.assertEqual(got, base, msg=f"window={window} tau={tau}")

        out = run(n=1000, window=100e-12)

        self.assertEqual(out.window_loss, 0.0, msg=f"window_loss {out.window_loss}")
        self.assertEqual(out.bin_leak, 0.0, msg=f"bin_leak {out.bin_leak}")

    def test_gaussian_has_no_tail(self):
        """
        A Gaussian core alone gives window loss and no bin error, the leak at a 79 ps core in a
        100 ps window being exactly 0.0 by the 9-sigma cut, the true 29-sigma tail 7.2e-175.
        """
        losses = []
        for fwhm in (50e-12, 100e-12, 200e-12, 300e-12):
            loss, leak = _core.jitter_split(RATE, fwhm, 100e-12, 0.0, 0.0)
            losses.append(loss)
            self.assertLess(leak, 1e-12, msg=f"leak at {fwhm:.0e} s: {leak}")
        self.assertMonotone(losses, msg=f"losses: {losses}")
        self.assertClose(losses[0], 0.018532, atol=1e-5, msg="50 ps FWHM window loss")
        self.assertClose(losses[-1], 0.694711, atol=1e-5, msg="300 ps FWHM window loss")
        self.assertEqual(
            _core.jitter_split(RATE, 79.3e-12, 100e-12, 0.0, 0.0)[1],
            0.0,
            msg="leak at a 79.3 ps core",
        )

        loss, leak = _core.jitter_split(RATE, 79.3e-12, 100e-12, 0.5792, 295e-12)

        self.assertGreater(leak, 0.01, msg=f"leak {leak}")
        self.assertClose(loss, 0.539922, atol=1e-5, msg="the same core, now tailed")

    def test_split_closed_form(self):
        """
        The engine's Gauss-Legendre panels agree to 1e-12 with weights from math.erf and the
        exponential's antiderivative, out to slots of order $e^{-100}$ where a rational erf's
        1e-7 floor would read as leak.
        """
        cases = (
            (60e-12, 100e-12, 0.0, 0.0),
            (200e-12, 250e-12, 0.0, 0.0),
            (79.3e-12, 100e-12, 0.5792, 295e-12),
            (79.3e-12, 200e-12, 0.5792, 295e-12),
            (0.0, 400e-12, 0.4, 800e-12),
            (150e-12, 0.0, 0.2, 500e-12),
        )
        for jitter, window, frac, tau in cases:
            got = _core.jitter_split(RATE, jitter, window, frac, tau)
            want = split_model(jitter, window, frac, tau)
            self.assertClose(got[0], want[0], atol=1e-12, msg=f"loss {got[0]} vs {want[0]}")
            self.assertClose(got[1], want[1], atol=1e-12, msg=f"leak {got[1]} vs {want[1]}")

    def test_window_is_a_loss(self):
        """
        A gated Gaussian core only loses counts: the rate falls by the reported window loss to
        an eta-scaled run's, QBER unmoved.
        """
        loss = _core.jitter_split(RATE, 150e-12, 100e-12, 0.0, 0.0)[0]
        off = run(n=600_000, mu=0.4, eta=0.5, vis=0.94, dark=1e-5)
        on = run(
            n=600_000,
            mu=0.4,
            eta=0.5,
            vis=0.94,
            dark=1e-5,
            jitter=150e-12,
            window=100e-12,
        )
        cut = run(n=600_000, mu=0.4, eta=0.5 * (1.0 - loss), vis=0.94, dark=1e-5)

        self.assertClose(on.window_loss, loss, atol=1e-12, msg="the reported loss")
        self.assertEqual(on.bin_leak, 0.0, msg=f"bin_leak {on.bin_leak}")
        self.assertLess(on.click_rate, off.click_rate, msg=f"click rate {on.click_rate} vs {off.click_rate}")
        self.assertClose(
            on.click_rate,
            cut.click_rate,
            atol=0.002,
            msg=f"gated {on.click_rate:.5f} vs eta-scaled {cut.click_rate:.5f}",
        )
        self.assertClose(
            on.qber,
            off.qber,
            atol=0.002,
            msg=f"QBER {on.qber:.5f} vs {off.qber:.5f}",
        )

    def test_bin_error_law(self):
        """
        Contiguous bins lose nothing, so bin leak is an error: QBER is half the leak and
        contrast falls to 1 - leak.
        """
        for jitter in (400e-12, 800e-12):
            leak = _core.jitter_split(RATE, jitter, 0.0, 0.0, 0.0)[1]
            out = run(n=2_000_000, mu=0.05, eta=0.5, jitter=jitter)
            self.assertClose(out.window_loss, 0.0, atol=1e-12, msg=f"window_loss {out.window_loss}")
            self.assertClose(out.bin_leak, leak, atol=1e-12, msg="the reported leak")

            # Contrast residual is Smear's truncated end slots (src/click.rs), 1/n: 6.06e-8 at 2e6.
            self.assertClose(
                out.visibility,
                1.0 - leak,
                atol=1e-7,
                msg=f"contrast {out.visibility:.6f} vs 1 - leak {1.0 - leak:.6f}",
            )

            # A counted QBER: the band is the four-sigma binomial one, not a literal sigma.
            self.assertClose(
                out.qber,
                0.5 * leak,
                atol=band(0.5 * leak, out.sifted),
                msg=f"QBER {out.qber:.5f} vs half the leak {0.5 * leak:.5f}",
            )

    def test_tail_raises_qber(self):
        """
        At a realistic core width only the diffusion tail reaches the next slot, and growing it
        raises QBER and window loss separately.
        """
        errs = []
        losses = []
        for frac in (0.0, 0.2, 0.4, 0.6):
            out = run(
                n=1_500_000,
                mu=0.1,
                eta=0.5,
                jitter=80e-12,
                window=100e-12,
                frac=frac,
                tau=300e-12,
            )
            errs.append(out.qber)
            losses.append(out.window_loss)
            self.assertClose(
                out.qber,
                0.5 * out.bin_leak,
                atol=0.002,
                msg=f"frac={frac}: QBER {out.qber:.5f} vs half-leak",
            )
        self.assertMonotone(errs, msg=f"QBER: {errs}")
        self.assertMonotone(losses, msg=f"losses: {losses}")
        self.assertEqual(errs[0], 0.0, msg=f"QBER {errs[0]}")

    def test_grain_invariance(self):
        """
        The convolution reads neighbours across chunk boundaries by absolute index, so the
        record is chunk-invariant.
        """
        cfg = {
            "n": 400_000,
            "seed": 3,
            "lw": 1e7,
            "dark": 1e-5,
            "jitter": 90e-12,
            "window": 120e-12,
            "frac": 0.5,
            "tau": 400e-12,
            "record": True,
        }
        base = digest(run(chunk=512, **cfg))
        for chunk in (65536, 0):
            got = digest(run(chunk=chunk, **cfg))
            self.assertEqual(got, base, msg=f"chunk={chunk} changed the result")

    def test_rejects_nonsense(self):
        """
        Refused by name: a window wider than the symbol period, a tail weight with no time
        constant or above 1, a negative jitter, and a tail spanning too many slots.
        """
        self.assertBad(
            "overlap",
            _core.jitter_split,
            (RATE, 50e-12, 2e-9, 0.0, 0.0),
            msg="window 2 ns",
        )
        self.assertBad(
            "tail_time",
            _core.jitter_split,
            (RATE, 50e-12, 100e-12, 0.3, 0.0),
            msg="tail_frac 0.3, tail_time 0",
        )
        self.assertBad(
            "tail_frac",
            _core.jitter_split,
            (RATE, 50e-12, 100e-12, 1.5, 1e-9),
            msg="tail_frac 1.5",
        )
        self.assertBad(
            "jitter",
            _core.jitter_split,
            (RATE, -1e-12, 100e-12, 0.0, 0.0),
            msg="jitter -1 ps",
        )
        self.assertBad(
            "symbol periods",
            _core.jitter_split,
            (RATE, 50e-12, 100e-12, 0.5, 1e-7),
            msg="tail_time 100 ns",
        )


class Pair(Guarded):
    """
    Bob's two detectors as two: the mismatch a time-shift or faked-state attack exploits is
    described, not corrected, and an equal pair is the old run bit for bit.
    """

    def assertShares(self, out, want, rows, msg=None):
        """
        Each named counter's per-slot rate lands on slot_model's entry inside a four-sigma
        binomial band.
        """
        for name, got in rows:
            self.assertClose(
                got / out.n_slots,
                want[name],
                atol=band(want[name], out.n_slots),
                msg=f"{msg}: {name} rate {got / out.n_slots:.6f} vs {want[name]:.6f}",
            )

    def test_model_reduces(self):
        """
        At equal detectors the bit-averaged pair_model returns qber_model to 1e-15 over flux,
        contrast, dark probability and phase spread.
        """
        cases = (
            (0.25, 0.4, 0.0, 1.0, 0.0),
            (0.45, 0.9, 1e-3, 0.99, 0.0),
            (0.05, 0.5, 1e-2, 0.94, 0.3),
            (2e-4, 0.5, 1e-2, 1.0, 0.0),
        )
        for mu, eta, dark, vis, sigma in cases:
            want = qber_model(mu, eta, dark, vis, sigma)
            got = pair_model(mu, (eta, eta), (dark, dark), vis, sigma)
            self.assertClose(got, want, atol=1e-15, msg=f"mu={mu} eta={eta} dark={dark}")

    def test_symmetric_unchanged(self):
        """
        Naming detector 1 its partner's efficiency and dark probability reproduces records,
        counts and floats over the flat path, phase walk, dead-time scan and timing convolution.
        """
        cases = (
            {
                "n": 300_000,
                "seed": 5,
                "mu": 0.4,
                "eta": 0.5,
            },
            {
                "n": 300_000,
                "seed": 5,
                "lw": 2e7,
                "dark": 1e-4,
                "vis": 0.93,
            },
            {
                "n": 300_000,
                "seed": 6,
                "dark": 1e-3,
                "dead": 20e-9,
                "after": 0.03,
            },
            {
                "n": 300_000,
                "seed": 7,
                "lw": 1e7,
                "dark": 1e-5,
                "jitter": 90e-12,
                "window": 120e-12,
                "frac": 0.5,
                "tau": 400e-12,
            },
        )
        for cfg in cases:
            base = digest(run(record=True, share=False, **cfg))
            split = run(
                record=True,
                share=False,
                eta1=cfg.get("eta", 0.4),
                dark1=cfg.get("dark", 0.0),
                **cfg,
            )
            self.assertEqual(digest(split), base, msg=f"a pair moved {cfg}")

    def test_split_is_invisible(self):
        """
        At perfect contrast with no dark counts an efficiency mismatch makes no errors, so the
        QBER reads 0 and only detector 1 answering a third of the gates shows it.
        """
        want = slot_model(0.4, (0.5, 0.15), (0.0, 0.0), 1.0, 0.0)
        out = run(n=1_000_000, mu=0.4, eta=0.5, vis=1.0, eta1=0.15)

        self.assertEqual(out.errors, 0, msg=f"{out.errors} errors at V = 1")
        self.assertEqual(out.doubles, 0, msg=f"doubles {out.doubles}")
        self.assertEqual(out.qber, 0.0, msg=f"QBER {out.qber}")

        rows = (("d0", out.clicks_d0), ("d1", out.clicks_d1))

        self.assertShares(out, want, rows, msg="an invisible efficiency split")

    def test_efficiency_mismatch(self):
        """
        With contrast and a dark floor each detector's click rate and lone errors land on
        slot_model in a four-sigma binomial band, and the pooled QBER on pair_model, not the
        symmetric form.
        """
        eta = (0.5, 0.1)
        dark = (1e-3, 1e-3)
        want = slot_model(0.4, eta, dark, 0.94, 0.0)
        out = run(n=1_000_000, mu=0.4, eta=0.5, dark=1e-3, vis=0.94, eta1=0.1)

        self.assertLess(
            out.clicks_d1,
            out.clicks_d0,
            msg=f"clicks_d1 {out.clicks_d1}, clicks_d0 {out.clicks_d0}",
        )

        rows = (
            ("d0", out.clicks_d0),
            ("d1", out.clicks_d1),
            ("err0", out.errors_d0),
            ("err1", out.errors_d1),
        )

        self.assertShares(out, want, rows, msg="an efficiency mismatch")

        pair = pair_model(0.4, eta, dark, 0.94, 0.0)

        self.assertClose(
            out.qber,
            pair,
            atol=band(pair, out.sifted),
            msg=f"QBER {out.qber:.6f} vs the mismatched form {pair:.6f}",
        )

    def test_dark_mismatch(self):
        """
        Dark counts on one detector lift its click rate past its partner's, give it over twice
        the errors, and move the QBER from 0.0301 to 0.0526, eleven binomial widths off the
        symmetric form.
        """
        eta = (0.5, 0.5)
        dark = (0.0, 1e-2)
        want = slot_model(0.4, eta, dark, 0.94, 0.0)
        pair = pair_model(0.4, eta, dark, 0.94, 0.0)
        even = qber_model(0.4, 0.5, 0.0, 0.94, 0.0)
        out = run(n=1_000_000, mu=0.4, eta=0.5, dark=0.0, vis=0.94, dark1=1e-2)

        self.assertGreater(out.clicks_d1, out.clicks_d0, msg=f"clicks_d1 {out.clicks_d1}, clicks_d0 {out.clicks_d0}")
        self.assertGreater(
            out.errors_d1,
            2 * out.errors_d0,
            msg=f"errors split {out.errors_d0} to {out.errors_d1}",
        )

        rows = (("err0", out.errors_d0), ("err1", out.errors_d1))

        self.assertShares(out, want, rows, msg="a dark-count mismatch")
        self.assertClose(
            out.qber,
            pair,
            atol=band(pair, out.sifted),
            msg=f"QBER {out.qber:.6f} vs the split form {pair:.6f}",
        )
        self.assertGreater(
            abs(pair - even),
            8.0 * band(pair, out.sifted),
            msg=f"pair {pair:.6f}, even {even:.6f}",
        )

    def test_error_split(self):
        """
        errors_d0 and errors_d1 are each detector's lone errors, so what they omit is the
        squashing coin's share, within [0, doubles] and half of them.
        """
        want = slot_model(0.4, (0.5, 0.5), (1e-3, 1e-3), 0.94, 0.0)
        out = run(n=600_000, seed=3, mu=0.4, eta=0.5, dark=1e-3, vis=0.94)
        rest = out.errors - out.errors_d0 - out.errors_d1

        self.assertGreaterEqual(rest, 0, msg=f"coin errors {rest}")
        self.assertLessEqual(rest, out.doubles, msg=f"{rest} coin errors among {out.doubles} doubles")
        self.assertClose(
            rest / out.doubles,
            0.5,
            atol=band(0.5, out.doubles),
            msg=f"the squash coin lost {rest} of {out.doubles}",
        )

        rows = (
            ("err0", out.errors_d0),
            ("err1", out.errors_d1),
            ("both", out.doubles),
        )

        self.assertShares(out, want, rows, msg="an equal pair")

    def test_basis_symmetric(self):
        """
        run_basis with detector 1 given detector 0's numbers reproduces every count and float
        row, and at sifting 1 the two detectors sum to the clicks plus the doubles.
        """
        args = ((0.5, 0.1, 0.02), (0.7, 0.2, 0.1), 0.5, 0.6, 1e-5, 0.02, 1.0)
        base = _core.run_basis(1_000_000, 17, *args, 0)
        same = _core.run_basis(1_000_000, 17, *args, 0, 0.6, 1e-5)

        self.assertEqual(counts(same), counts(base), msg="counts, symmetric split")
        self.assertEqual(
            base.clicks_d0 + base.clicks_d1,
            sum(base.clicks) + base.doubles,
            msg="clicks_d0 + clicks_d1 = clicks + doubles",
        )

    def test_basis_mismatch(self):
        """
        With no dark floor or misalignment the basis-keyed detectors partition the clicks with
        no bit read wrong, the mismatch showing only as imbalance on basis_model, and with both
        restored the pooled QBER sits on the matched model.
        """
        mu = (0.5, 0.1, 0.02)
        probs = (0.7, 0.2, 0.1)
        clean = _core.run_basis(2_000_000, 21, mu, probs, 0.5, 0.6, 0.0, 0.0, 1.0, 0, 0.15, 0.0)
        want = basis_model(mu, probs, 0.5, (0.6, 0.15), (0.0, 0.0), 0.0)
        pulses = clean.n_pulses

        self.assertEqual(clean.doubles, 0, msg=f"doubles {clean.doubles}")
        self.assertEqual(sum(clean.errors), 0, msg=f"errors {sum(clean.errors)}")
        self.assertEqual(
            clean.clicks_d0 + clean.clicks_d1,
            sum(clean.clicks),
            msg="clicks_d0 + clicks_d1 = clicks",
        )

        for name, got in (("d0", clean.clicks_d0), ("d1", clean.clicks_d1)):
            self.assertClose(
                got / pulses,
                want[name],
                atol=band(want[name], pulses),
                msg=f"{name} share {got / pulses:.6f} vs {want[name]:.6f}",
            )

        out = _core.run_basis(2_000_000, 21, mu, probs, 0.5, 0.6, 1e-5, 0.02, 1.0, 0, 0.15, 1e-5)
        even = basis_model(mu, probs, 0.5, (0.6, 0.6), (1e-5, 1e-5), 0.02)
        odd = basis_model(mu, probs, 0.5, (0.6, 0.15), (1e-5, 1e-5), 0.02)
        kept = sum(out.sifted)
        qber = sum(out.errors) / kept

        self.assertLess(out.clicks_d1, out.clicks_d0, msg=f"clicks_d1 {out.clicks_d1}, clicks_d0 {out.clicks_d0}")

        # Alice's uniform bit weights both error directions equally: 5.4e-5 apart before sampling.
        self.assertClose(
            odd["qber"],
            even["qber"],
            atol=1e-4,
            msg=f"the two models differ by {abs(odd['qber'] - even['qber']):.2e}",
        )
        self.assertClose(
            qber,
            even["qber"],
            atol=band(even["qber"], kept),
            msg=f"pooled QBER {qber:.6f}",
        )

    def test_rejects_nonsense(self):
        """
        A zero or above-unit efficiency and an always-firing gate are refused as eta_d1 and
        dark_d1, not under their partner's names, by the DPS and basis-keyed runs alike.
        """
        dps = (1000, 1, 0.2, 0.5, 0.9, 0.01, 0.98, 1, 0.0, RATE, 0.0, 0.0, 4096, False)
        keyed = ((0.5, 0.1, 0.02), (0.7, 0.2, 0.1), 0.5, 0.6, 1e-5, 0.02, 1.0)
        for bad in (0.0, 1.5, -1.0):
            self.assertBad(
                "eta_d1",
                _core.run_clicks,
                dps + (0.0, 0.0, 0.0, 0.0, bad),
                msg=f"eta_d1 = {bad}",
            )
            self.assertBad(
                "eta_d1",
                _core.run_basis,
                (1000, 1) + keyed + (0, bad),
                msg=f"run_basis eta_d1 = {bad}",
            )

        for bad in (1.0, 1.5, -5e-324):
            self.assertBad(
                "dark_d1",
                _core.run_clicks,
                dps + (0.0, 0.0, 0.0, 0.0, 0.9, bad),
                msg=f"dark_d1 = {bad}",
            )

        for bad in (1.5, -5e-324):
            self.assertBad(
                "dark_d1",
                _core.run_basis,
                (1000, 1) + keyed + (0, 0.6, bad),
                msg=f"run_basis dark_d1 = {bad}",
            )


class Background(Guarded):
    """
    A Raman occupancy and a Rayleigh return as photons in one gate, and the single floor a
    threshold detector reads both through.
    """

    def test_modes_count_the_gate(self):
        """
        The mode count is passband*gate*pol, so doubling the filter width, the gate or the
        polarisations doubles the photons one occupancy delivers.
        """
        occ, passband = raman_point()
        flux, modes = _core.bg_raman(occ, passband, GATE, 2, 1.0)

        self.assertClose(modes, 2.0 * passband * GATE, msg=f"modes {modes}")
        self.assertClose(flux, occ * modes, msg=f"flux {flux}")
        for scale in (0.5, 2.0, 10.0):
            wide = _core.bg_raman(occ, scale * passband, GATE, 2, 1.0)[0]
            long = _core.bg_raman(occ, passband, scale * GATE, 2, 1.0)[0]
            self.assertClose(wide, scale * flux, msg=f"passband x{scale}")
            self.assertClose(long, scale * flux, msg=f"gate x{scale}")

        one = _core.bg_raman(occ, passband, GATE, 1, 1.0)[0]

        self.assertClose(2.0 * one, flux, msg=f"pol=1 flux {one}")

    def test_raman_matches_power(self):
        """
        Kumar's Eq. (6) occupancy over passband*gate*2 reproduces Eraerds' Eq. (11)
        $P_{ram}/(h\\nu) \\cdot dt_{gate}$, Eq. (6)'s 1/2 being the oscillator's and the 45 pm
        filter carrying no polarisation factor.
        """
        occ, passband = raman_point()
        flux = _core.bg_raman(occ, passband, GATE, 2, 1.0)[0]
        want = raman_power()

        self.assertClose(flux / want, 1.0, atol=1e-12, msg=f"{flux:.12e} against {want:.12e}")
        self.assertClose(
            _core.bg_raman(occ, passband, GATE, 1, 1.0)[0] / want,
            0.5,
            msg="pol=1",
        )
        self.assertClose(
            _core.bg_raman(occ, passband, GATE, 2, 0.5)[0] / want,
            0.5,
            msg="split=0.5",
        )
        self.assertClose(
            _core.bg_raman(occ, passband, GATE, 1, 0.5)[0] / want,
            0.25,
            msg="pol=1, split=0.5",
        )

    def test_poisson_beats_linear(self):
        """
        bg_floor exponentiates where Eraerds' Eq. (11) is linear, a form their note under Eq.
        (15) says carries no Poisson statistics, so the linear form reads high at every flux.
        """
        occ, passband = raman_point()
        flux = _core.bg_raman(occ, passband, GATE, 2, 1.0)[0]
        got = _core.bg_floor(0.0, [flux], DETECT)

        self.assertClose(got / (DETECT * flux) - 1.0, -4.31758e-4, atol=1e-8, msg=f"gap {got:.6e}")
        seen = []
        for scale in (1e-3, 1.0, 10.0, 100.0):
            p = _core.bg_floor(0.0, [scale * flux], DETECT)
            self.assertTrue(DETECT * scale * flux >= p, msg=f"linear vs Poisson at x{scale}")
            seen.append(DETECT * scale * flux / p)

        self.assertMonotone(seen, msg=f"linear excess: {seen}")

    def test_thermal_bounds_poisson(self):
        """
        Spontaneous Raman is chaotic: the M-mode law $1 - (1 - d)(1 + \\eta n)^{-M}$ reads below
        the Poisson one at equal flux, the gap closing as the mode count rises at fixed flux.
        """
        occ, passband = raman_point()
        flux = _core.bg_raman(occ, passband, GATE, 2, 1.0)[0]
        pois = _core.bg_floor(0.0, [flux], DETECT)
        hot = _core.bg_thermal(occ, passband, GATE, 2, 1.0, DETECT, 0.0)

        self.assertClose(hot / pois - 1.0, -2.50050e-5, atol=1e-9, msg=f"thermal gap {hot:.6e}")
        seen = []
        for scale in (1.0, 10.0, 100.0, 1000.0):
            spread = _core.bg_thermal(occ / scale, scale * passband, GATE, 2, 1.0, DETECT, 0.0)
            seen.append(abs(spread / pois - 1.0))

        self.assertMonotone(seen, rising=False, msg=f"gaps: {seen}")

    def test_rayleigh_takes_the_gate(self):
        """
        An elastic return is continuous: a gate takes gate/period of one period's photons and
        `split` this detector's share, a full-period gate at split=1 returning its input.
        """
        photons = impairments.rayleigh_photons(1e-9, 50.0, 1.0 / RATE)
        whole = _core.bg_rayleigh(photons, 1.0 / RATE, 1.0 / RATE, 1.0)

        self.assertClose(whole, photons, msg=f"whole {whole}")
        for share in (0.1, 0.5, 0.9):
            got = _core.bg_rayleigh(photons, 1.0 / RATE, 0.25 / RATE, share)
            self.assertClose(got, 0.25 * share * photons, msg=f"gate and split at split={share}")

    def test_rayleigh_split_is_optics(self):
        """
        The 1/2 in rayleigh_photons, the oscillator's polarisation selectivity, and the 1/2 in
        `split`, two detectors dividing an unpolarised return, cancel, so the engine takes the
        unpolarised number.
        """
        photons = impairments.rayleigh_photons(1e-9, 50.0, 1.0 / RATE)
        got = _core.bg_rayleigh(2.0 * photons, 1.0 / RATE, 1.0 / RATE, 0.5)

        self.assertClose(got, photons, msg=f"got {got}")
        self.assertClose(
            _core.bg_rayleigh(2.0 * photons, 1.0 / RATE, 1.0 / RATE, 1.0),
            2.0 * photons,
            msg="split=1",
        )

    def test_floor_composes(self):
        """
        bg_floor is $1 - (1 - d)e^{-\\eta \\sum \\Phi}$: an empty list is the bare dark, fluxes
        sum, and one flux matches click_prob at that photon number.
        """
        occ, passband = raman_point()
        flux = _core.bg_raman(occ, passband, GATE, 2, 1.0)[0]

        self.assertClose(_core.bg_floor(1e-6, [], 0.2), 1e-6, msg="empty flux list")
        self.assertClose(
            _core.bg_floor(1e-6, [flux, 3.0 * flux], 0.2),
            _core.bg_floor(1e-6, [4.0 * flux], 0.2),
            msg="flux sum",
        )
        self.assertClose(
            _core.bg_floor(1e-6, [flux], 0.2),
            _core.click_prob(math.sqrt(flux), 0.0, 0.2, 1e-6),
            msg="bg_floor vs click_prob",
        )
        rows = [_core.bg_floor(1e-6, [scale * flux], 0.2) for scale in (0.0, 0.5, 1.0, 2.0)]

        self.assertMonotone(rows, msg=f"floors: {rows}")

    def test_background_reads_as_dark(self):
        """
        A Raman flux through bg_floor drives the DPS QBER exactly as an equal dark probability
        does, `q.ClickDetector(dark=...)` being the composition point with no separate QBER
        term.
        """
        occ, passband = raman_point()
        flux = _core.bg_raman(occ, passband, GATE, 2, 1.0)[0]
        floor = _core.bg_floor(1e-6, [flux], 0.5)

        self.assertTrue(floor > 1e-3, msg=f"floor {floor}")
        out = run(mu=0.02, eta=0.5, dark=floor)
        want = qber_model(0.02, 0.5, floor, 1.0, 0.0)

        self.assertClose(
            out.qber,
            want,
            atol=4.0 * band(want, out.sifted),
            msg=f"QBER {out.qber:.5f}, closed {want:.5f}, floor {floor:.3e}",
        )
        quiet = run(mu=0.02, eta=0.5, dark=1e-6)

        self.assertTrue(out.qber > quiet.qber, msg=f"QBER {out.qber:.5f} vs {quiet.qber:.5f}")

    def test_rejects_nonsense(self):
        """
        The passband, gate, polarisation count, split and a saturating flux are each refused by
        name.
        """
        occ, passband = raman_point()

        self.assertBad(
            "passband",
            _core.bg_raman,
            (occ, 0.0, GATE, 2, 1.0),
            msg="passband = 0",
        )
        self.assertBad(
            "gate",
            _core.bg_raman,
            (occ, passband, 0.0, 2, 1.0),
            msg="gate = 0",
        )
        self.assertBad(
            "pol",
            _core.bg_raman,
            (occ, passband, GATE, 3, 1.0),
            msg="pol = 3",
        )
        self.assertBad(
            "pol",
            _core.bg_thermal,
            (occ, passband, GATE, 0, 1.0, 0.5, 0.0),
            msg="bg_thermal pol = 0",
        )
        self.assertBad(
            "must not exceed the symbol period",
            _core.bg_rayleigh,
            (1e-3, 1.0 / RATE, 2.0 / RATE, 0.5),
            msg="gate = 2 periods",
        )
        self.assertBad(
            "split",
            _core.bg_rayleigh,
            (1e-3, 1.0 / RATE, 1.0 / RATE, 1.5),
            msg="split = 1.5",
        )
        self.assertBad(
            "flux[1]",
            _core.bg_floor,
            (1e-6, [0.1, -1e-30], 0.2),
            msg="flux[1] = -1e-30",
        )
        self.assertBad(
            "fires the gate with probability 1",
            _core.bg_floor,
            (1e-6, [1e4], 0.2),
            msg="flux = 1e4",
        )
        self.assertBad(
            "dark",
            _core.bg_floor,
            (1.0, [0.1], 0.2),
            msg="dark = 1, bg_floor",
        )
        self.assertBad(
            "overflows f64",
            _core.bg_raman,
            (1e300, 1e300, 1e300, 2, 1.0),
            msg="occupancy, passband, gate = 1e300",
        )


if __name__ == "__main__":
    rc = Exam(
        "ClickOptics",
        "Phase 5b: threshold-detector and delay-interferometer closed forms",
        "click_optics.md",
    ).run(load(Optics))
    rc |= Exam(
        "ClickDps",
        "Phase 5b: DPS QBER derived from hardware, not pinned",
        "click_dps.md",
    ).run(load(Dps))
    rc |= Exam(
        "ClickJitter",
        "Phase 5b: detector timing jitter, window loss and bin error apart",
        "click_jitter.md",
    ).run(load(Jitter))
    rc |= Exam(
        "ClickPair",
        "Phase 5b: Bob's two detectors, split in efficiency and dark rate",
        "click_pair.md",
    ).run(load(Pair))
    rc |= Exam(
        "ClickBackground",
        "Phase 5b: Raman and Rayleigh flux as one per-gate floor beside the dark",
        "click_background.md",
    ).run(load(Background))
    sys.exit(rc)
