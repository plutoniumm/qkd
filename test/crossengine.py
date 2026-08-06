import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import BETA, HARDWARE
from kit.forms import bisect, untrusted
from qkd import _core, fock, gaussian

# T stops at 0.99; T = 1 is test_transparent_seam.
VAS = (1e-3, 0.02, 0.2, 2.0, 5.0, 20.0, 100.0, 1e3)
TS = (0.99, 0.9, 10**-0.5, 0.1, 0.01, 1e-3, 1e-4)
XIS = (0.0, 1e-3, 0.01, 0.05, 0.3, 1.0)

# Phase-space window: holds a squeezed vacuum at r = 1 (sigma_p = 1.92) to 2 sigma.
XS = np.linspace(-4.0, 4.0, 31)


def gauss_z(va, t):
    """
    Alice-Bob correlation sqrt(t*va*(va + 2)) at the output plane, not t*((va+1)^2 - 1).
    """

    return math.sqrt(t * va * (va + 2.0))


def merged(t, xi, eta, vel):
    """
    Bob's heterodyne handed to Eve: (T, xi) -> (eta*T, xi + 2*v_el/(eta*T)), input-referred.
    """

    return untrusted(t, xi, eta, vel, 2.0)


def moment_err(state, ref):
    """
    Largest disagreement between a Fock state's first two moments and a Gaussian's.
    """
    mean, cov = state.moments()
    dmean = np.max(np.abs(mean - ref.mean))
    dcov = np.max(np.abs(cov - ref.cov))

    return float(max(dmean, dcov))


def wigner_err(state, ref):
    """
    Largest pointwise disagreement between the two layers' Wigner grids.
    """

    return float(np.max(np.abs(state.wigner(XS, XS) - ref.wigner(0, XS, XS))))


def channel(mu, nu1, nu2, eta, y0, edet):
    """
    Decoy bounds from self-consistent forward-model gains.
    """
    q_mu, e_mu = _core.decoy_gain(mu, eta, y0, edet)
    q_one, e_one = _core.decoy_gain(nu1, eta, y0, edet)
    q_two, e_two = (y0, 0.5) if nu2 <= 0.0 else _core.decoy_gain(nu2, eta, y0, edet)

    return _core.decoy_bounds(mu, nu1, nu2, q_mu, e_mu, q_one, e_one, q_two, e_two)


TWINS = (
    (
        "coherent(1.4, 0.9)",
        lambda c: fock.Coherent(1.4, 0.9, cutoff=c),
        gaussian.Coherent(1.4, 0.9),
        (6, 12, 24),
    ),
    (
        "squeezed(0.8)",
        lambda c: fock.Squeezed(0.8, cutoff=c),
        gaussian.Squeezed(0.8),
        (12, 24, 48),
    ),
    (
        "squeezed(1.0)",
        lambda c: fock.Squeezed(1.0, cutoff=c),
        gaussian.Squeezed(1.0),
        (15, 30, 60),
    ),
    (
        "thermal(1.0)",
        lambda c: fock.Thermal(1.0, cutoff=c),
        gaussian.Thermal(1.0),
        (10, 20, 40),
    ),
    (
        "thermal(2.0)",
        lambda c: fock.Thermal(2.0, cutoff=c),
        gaussian.Thermal(2.0),
        (15, 30, 60),
    ),
)


class Modulation(Question):
    """
    dm_holevo knows no constellation, detector or trust model; cv_rate carries all three.
    """

    def test_gaussian_limit(self):
        """
        dm_holevo at $z = \\sqrt{T V_A (V_A + 2)}$ with an unbounded alphabet equals cv_rate at
        ideal heterodyne to 1e-10 over 336 points, $V_A$ 1e-3 to 1e3, $T$ 1e-4 to 0.99, $\\xi$ 0
        to 1.
        """
        worst = 0.0
        count = 0
        for va in VAS:
            for t in TS:
                for xi in XIS:
                    count += 1
                    z = gauss_z(va, t)
                    got = _core.dm_holevo(va, t, xi, z, float("inf"), BETA)
                    want = _core.cv_rate(va, t, xi, 1.0, 0.0, BETA, False, True)
                    worst = max(worst, max(abs(got[i] - want[i]) for i in range(3)))
        self.assertEqual(count, 336, msg=f"points {count}")
        self.assertLessEqual(worst, 1e-10, msg=f"worst {worst:.3e}")

    def test_transparent_seam(self):
        """
        At $T = 1$, where both eigenvalues sit within sqrt(eps) of vacuum, dm_holevo and cv_rate
        agree to 1e-12 on I_AB and 2e-6 on chi_BE, twice an offline 1.2e5-point worst of 9.6e-7,
        cv_rate reading Eve low.
        """
        random.seed(20260808)
        worst = [0.0, 0.0]
        count = 0
        for _ in range(4000):
            va = 10 ** random.uniform(-4.0, 3.0)
            xi = 10 ** random.uniform(-7.0, 0.5)
            try:
                got = _core.dm_holevo(va, 1.0, xi, gauss_z(va, 1.0), float("inf"), BETA)
            except ValueError:
                continue

            count += 1
            want = _core.cv_rate(va, 1.0, xi, 1.0, 0.0, BETA, False, True)
            worst[0] = max(worst[0], abs(got[0] - want[0]))
            worst[1] = max(worst[1], abs(got[1] - want[1]))
            self.assertGreaterEqual(min(got[1], want[1]), 0.0, msg=f"chi_BE at V_A={va}, xi={xi}")

        # A floor: the physicality gate rejects 0 to 3 of 4000 on the vacuum boundary, per seed.
        self.assertGreaterEqual(count, 3900, msg=f"draws {count}")
        self.assertLessEqual(worst[0], 1e-12, msg=f"I_AB {worst[0]:.3e}")
        self.assertLessEqual(worst[1], 2e-6, msg=f"chi_BE {worst[1]:.3e}")

    def test_untrusted_detector(self):
        """
        With $\\eta$ and $v_{el}$ folded into the channel by hand, dm_holevo reproduces
        cv_rate's untrusted mode to 1e-13.
        """
        worst = 0.0
        for t, eta, vel, va in HARDWARE:
            for xi in (0.0, 0.01, 0.05):
                total, folded = merged(t, xi, eta, vel)
                z = gauss_z(va, total)
                got = _core.dm_holevo(va, total, folded, z, float("inf"), BETA)
                want = _core.cv_rate(va, t, xi, eta, vel, BETA, False, False)
                worst = max(worst, max(abs(got[i] - want[i]) for i in range(3)))
        self.assertLessEqual(worst, 1e-13, msg=f"worst {worst:.3e}")

    def test_alice_marginal(self):
        """
        M-PSK of amplitude $\\alpha$ has $V_A = 2\\alpha^2$ for every M, so the discrete penalty
        lives in z alone.
        """
        for alpha in (0.01, 0.35, 1.0, 3.0):
            for m in (3, 4, 6, 8, 16, 64):
                va, z_lin, w, z_gauss = _core.dm_moments(m, alpha)
                self.assertClose(va, 2.0 * alpha * alpha, atol=1e-14, msg=f"V_A at M={m}")
                self.assertClose(
                    z_gauss,
                    gauss_z(va, 1.0),
                    atol=1e-14,
                    msg=f"z_gauss at M={m}",
                )
                self.assertLessEqual(z_lin, z_gauss + 1e-15, msg=f"z_lin at M={m}: {z_lin}")
                self.assertGreaterEqual(w, 0.0, msg=f"noise price at M={m}")

    def test_dm_receiver(self):
        """
        dm_rate on the true channel with a real receiver equals dm_rate on the merged channel
        with an ideal one, pinning heterodyne's mu = 2.
        """
        for t, eta, vel, va in HARDWARE:
            alpha = math.sqrt(va / 2.0)
            total, folded = merged(t, 0.01, eta, vel)
            got = _core.dm_rate(8, alpha, t, 0.01, eta, vel, BETA)
            want = _core.dm_rate(8, alpha, total, folded, 1.0, 0.0, BETA)

            for i in range(4):
                self.assertClose(got[i], want[i], atol=1e-13, msg=f"component {i} at T={t:.4f}")


class Relay(Question):
    """
    cvmdi_rate is the large-modulation limit of a topology with a lossless encoder arm.
    """

    def _direct(self, t, va):
        return _core.cv_rate(va, t, 0.0, 1.0, 0.0, 1.0, False, True)[2]

    def test_relay_limit(self):
        """
        With a lossless encoder arm cvmdi_rate(1, T) equals the infinite-modulation trusted
        heterodyne rate at T, extrapolated as (10*K(1e8) - K(1e7))/9.
        """
        for t in (0.9, 0.5, 0.2, 0.1, 0.01, 0.002):
            relay = _core.cvmdi_rate(1.0, t, _core.cvmdi_floor(1.0, t))
            near = self._direct(t, 1e7)
            far = self._direct(t, 1e8)
            limit = (10.0 * far - near) / 9.0
            self.assertClose(
                relay,
                limit,
                atol=1e-12,
                msg=f"T={t}: relay {relay:.14f} vs extrapolated link {limit:.14f}",
            )

    def test_relay_scaling(self):
        """
        A finite-modulation link falls short of the relay's asymptotic value by a gap dividing
        by ten per decade of $V_A$.
        """
        for t in (0.9, 0.5, 0.1, 0.01):
            relay = _core.cvmdi_rate(1.0, t, _core.cvmdi_floor(1.0, t))
            gaps = [relay - self._direct(t, va) for va in (1e5, 1e6, 1e7)]

            for gap in gaps:
                self.assertGreater(gap, 0.0, msg=f"T={t}: gap {gap}")

            for i in range(len(gaps) - 1):
                ratio = gaps[i] / gaps[i + 1]

                # atol 0.05: closed form, worst miss 7.0e-4 (second-order 1/V_A).
                self.assertClose(ratio, 10.0, atol=0.05, msg=f"T={t}: decade {ratio:.3f}")

    def test_relay_covariance(self):
        """
        cvmdi_point's conditioned four-mode spectrum converges onto cvmdi_rate's closed form as
        $1/V_A$, the two sharing no code past the arguments.
        """
        for ta, tb in ((1.0, 0.5), (0.9, 0.3), (0.7, 0.7), (1.0, 0.1)):
            closed = _core.cvmdi_rate(ta, tb, _core.cvmdi_floor(ta, tb))
            gaps = [abs(closed - _core.cvmdi_point(va, ta, tb, 1.0, 1.0, 0.0, 0.0, 1.0)[2]) for va in (1e3, 1e4)]
            ratio = gaps[0] / gaps[1]

            # atol 0.1: a deterministic 1/V_A term moves the ratio off 10 by <= 0.036.
            self.assertClose(ratio, 10.0, atol=0.1, msg=f"({ta}, {tb}): decade ratio {ratio:.3f}")
            self.assertLessEqual(gaps[1], 1e-3, msg=f"({ta}, {tb}): residue {gaps[1]:.3e} at V_A = 1e4")


class Truncation(Question):
    """
    Gaussian states in the number basis against the closed forms, converged at a second cutoff.
    """

    def test_fock_converges(self):
        """
        The number basis reproduces the Gaussian layer's mean, covariance and Wigner function on
        a 31x31 window, each error falling at least 8x per cutoff doubling, a factor an hbar
        slip or a missing sqrt(2) fails.
        """
        for tag, err_of, tol in (("", moment_err, 1e-5), ("W ", wigner_err, 1e-4)):
            for name, make, ref, cutoffs in TWINS:
                errs = [err_of(make(c), ref) for c in cutoffs]
                self.assertMonotone(errs, rising=False, msg=f"{name}: {tag}{errs}")

                for i in range(len(errs) - 1):
                    floor = max(errs[i + 1], 1e-15)
                    self.assertGreaterEqual(
                        errs[i] / floor,
                        8.0,
                        msg=f"{name}: {tag}{cutoffs[i]} -> {cutoffs[i + 1]}: {errs[i] / floor:.1f}x",
                    )
                self.assertLessEqual(
                    errs[-1],
                    tol,
                    msg=f"{name}: {tag}off by {errs[-1]:.3e} at cutoff {cutoffs[-1]}",
                )

    def test_trunc_underreports(self):
        """
        trunc_error scores the lost population, not its effect on the moments: a squeezed
        vacuum's covariance error runs 15x to 140x above it, growing with the cutoff.
        """
        ratios = []
        for c in (15, 30, 60):
            state = fock.Squeezed(1.0, cutoff=c)
            ratios.append(moment_err(state, gaussian.Squeezed(1.0)) / state.trunc_error())
        self.assertMonotone(ratios, msg=f"ratios: {ratios}")
        self.assertGreater(ratios[0], 5.0, msg=f"ratio at cutoff 15: {ratios[0]:.1f}")
        self.assertGreater(ratios[-1], 50.0, msg=f"ratio at cutoff 60: {ratios[-1]:.1f}")

    def test_fock_channel(self):
        """
        A number-basis beamsplitter with vacuum equals the thermal-loss map at $\\xi = 0$ to
        1e-13 on displaced and thermal states.
        """
        for eta in (0.9, 0.5, 0.2):
            got = fock.Coherent(1.2, -0.4, cutoff=40).loss(eta)
            want = gaussian.Coherent(1.2, -0.4).thermal_loss(0, eta, 0.0, ref="input")
            self.assertLessEqual(moment_err(got, want), 1e-13, msg=f"coherent through eta={eta}")

            self.gridClose(
                got.wigner(XS, XS),
                want.wigner(0, XS, XS),
                atol=1e-13,
                msg=f"coherent W through eta={eta}",
            )

            got = fock.Thermal(1.5, cutoff=80).loss(eta)
            want = gaussian.Thermal(1.5).thermal_loss(0, eta, 0.0, ref="input")
            self.assertLessEqual(moment_err(got, want), 1e-13, msg=f"thermal through eta={eta}")

    def test_fock_entropy(self):
        """
        A thermal state's von Neumann entropy and purity in the number basis equal the
        symplectic ones.
        """
        for nbar in (0.0, 0.5, 2.0):
            got = fock.Thermal(nbar, cutoff=100)
            want = gaussian.Thermal(nbar)
            self.assertClose(got.entropy(), want.entropy(), atol=1e-11, msg=f"S at nbar={nbar}")
            self.assertClose(float(got.purity()), want.purity(), atol=1e-11, msg=f"P at nbar={nbar}")

    def test_fock_physical(self):
        """
        At every cutoff, including ones where the moments are 100% wrong, the number basis's
        second moments are bona fide and satisfy Heisenberg, and its Husimi grid stays inside
        $[0, 1/(2\\pi)]$.
        """
        for name, make, _ref, cutoffs in TWINS:
            for c in (4,) + cutoffs:
                _mean, cov = make(c).moments()

                # atol 1e-12: eigensolve round-off on the boundary, worst measured -1.2e-15.
                self.assertPhysical(cov, atol=1e-12, msg=f"{name} at cutoff {c}: bona fide")
                self.assertUncertainty(cov, atol=1e-12, msg=f"{name} at cutoff {c}: uncertainty")
                self.assertHusimi(
                    make(c).husimi(XS, XS),
                    atol=1e-12,
                    msg=f"{name} at cutoff {c}: Husimi",
                )

    def test_fock_purity(self):
        """
        A truncated pure state's density matrix reports purity 1 at every cutoff while its
        moments read $1/(2\\sqrt{\\det V})$ = 0.650 at cutoff 6 and 0.99996 at 48, the
        pessimistic reading for a pure state.
        """
        moments = []
        for c in (6, 12, 24, 48):
            state = fock.Squeezed(1.0, cutoff=c)
            _mean, cov = state.moments()
            got = 0.5 / math.sqrt(float(np.linalg.det(np.asarray(cov))))
            moments.append(got)

            # atol 1e-14 is exactness: a renormalised pure truncation stays rank one, measured 2.2e-16.
            self.assertClose(
                float(state.purity()),
                1.0,
                atol=1e-14,
                msg=f"purity at cutoff {c}",
            )
            self.assertLessEqual(
                got,
                1.0 + 1e-12,
                msg=f"moment purity at cutoff {c}: {got}",
            )
        self.assertMonotone(moments, msg=f"moment purity: {moments}")
        self.assertLess(moments[0], 0.7, msg=f"moment purity at cutoff 6: {moments[0]}")
        self.assertGreater(moments[-1], 0.9999, msg=f"moment purity at cutoff 48: {moments[-1]}")

        # A mixed state reverses the ordering: the two purities approach 1/3 from opposite sides.
        near = fock.Thermal(1.0, cutoff=6)
        _mean, cov = near.moments()

        self.assertGreater(
            0.5 / math.sqrt(float(np.linalg.det(np.asarray(cov)))),
            float(near.purity()),
            msg="thermal at cutoff 6: moment vs matrix purity",
        )

        # atol 1e-13: 48-level trace round-off, measured 2.3e-15.
        self.assertClose(
            float(fock.Thermal(1.0, cutoff=48).purity()),
            gaussian.Thermal(1.0).purity(),
            atol=1e-13,
            msg="thermal purity at cutoff 48",
        )

    def test_fock_gaussianity(self):
        """
        Relative-entropy non-Gaussianity and Wigner negativity vanish on every Gaussian twin at
        cutoff 96, converged 10x against 48, where a squeezed vacuum still reads 3e-4;
        negativity is bounded above only.
        """
        for name, make, _ref, _cutoffs in TWINS:
            near = abs(float(make(48).non_gaussianity()))
            far = abs(float(make(96).non_gaussianity()))
            self.assertLessEqual(
                far,
                max(near / 10.0, 1e-12),
                msg=f"{name}: non-Gaussianity {far:.3e}, cutoff 48 {near:.3e}",
            )
            self.assertLessEqual(far, 1e-8, msg=f"{name}: non-Gaussianity {far:.3e}")
            self.assertLessEqual(
                float(make(96).negativity(XS, XS)),
                1e-9,
                msg=f"{name}: negativity",
            )


class Threshold(Question):
    """
    Click response, decoy gain and relay port counts are one function of (mu, eta, dark).
    """

    def test_click_gain(self):
        """
        The click engine at a coherent amplitude and the decoy layer at a phase-randomised
        intensity both return $1 - (1 - d)e^{-\\eta\\mu}$, at four phases.
        """
        for mu in (1e-4, 0.01, 0.1, 0.5, 1.0, 4.0):
            for eta in (1.0, 0.5, 0.1, 1e-3):
                for dark in (0.0, 1e-6, 1e-3):
                    want = _core.decoy_gain(mu, eta, dark, 0.0)[0]
                    for phase in (0.0, 0.7, 2.9, -1.3):
                        amp = math.sqrt(mu)
                        got = _core.click_prob(amp * math.cos(phase), amp * math.sin(phase), eta, dark)
                        self.assertClose(
                            got,
                            want,
                            atol=1e-15,
                            msg=f"mu={mu} eta={eta} dark={dark} phase={phase}",
                        )

    def test_relay_ports(self):
        """
        With no light and no dark counts on port d, the relay's exactly-one-fired count reduces
        to port c's click probability.
        """
        for mu in (0.01, 0.2, 1.0):
            for eta in (1.0, 0.5, 0.1):
                single, double = _core.relay_counts(mu, 0.0, eta, 0.0)
                self.assertClose(
                    single,
                    _core.click_prob(math.sqrt(mu), 0.0, eta, 0.0),
                    atol=1e-15,
                    msg=f"single-fire at mu={mu} eta={eta}",
                )
                self.assertEqual(double, 0.0, msg=f"double {double}")


class Decoy(Question):
    """
    Finite-decoy bounds sit on the safe side of the ideal and walk onto it as the decoys weaken.
    """

    def test_decoy_sandwich(self):
        """
        Over 4000 random self-consistent channels -- four decades of transmittance and
        background, misalignment to 5%, decoys to nu1 + nu2 = 1.71 mu -- the Y1 lower bound
        never exceeds the infinite-decoy Y1 and the e1 upper bound never falls below it, at zero
        slack.
        """
        random.seed(20260808)
        count = 0
        # Ma's ordering filter binds on 22.0% of draws (sd 0.7%, 13 seeds): 6000 hold 4000 at 21 sigma.
        for _ in range(6000):
            eta = 10 ** random.uniform(-4.0, -0.2)
            y0 = 10 ** random.uniform(-7.0, -3.0)
            edet = random.uniform(0.0, 0.05)
            mu = random.uniform(0.2, 0.8)
            nu1 = random.uniform(0.02, 0.9) * mu
            nu2 = random.uniform(0.0, 0.9) * nu1
            if nu1 + nu2 >= mu:
                continue

            count += 1
            y1_lo, e1_hi, _q1 = channel(mu, nu1, nu2, eta, y0, edet)
            y_ideal, e_ideal = _core.decoy_ideal(eta, y0, edet)
            tag = f"eta={eta:.3e} y0={y0:.2e} mu={mu:.3f} nu1={nu1:.3f} nu2={nu2:.3f}"
            self.assertLessEqual(y1_lo, y_ideal, msg=f"Y1 {y1_lo} > {y_ideal}: {tag}")
            self.assertGreaterEqual(e1_hi, e_ideal, msg=f"e1 {e1_hi} < {e_ideal}: {tag}")

            if count == 4000:
                break

        # Exhausting the attempt budget fails here.
        self.assertGreaterEqual(count, 4000, msg=f"channels {count}")

    def test_decoy_limit(self):
        """
        Halving nu1 and nu2 halves both deviations from the ideal to atol 0.05, second order
        leaving 2.023 at nu1 = 0.024, nu2 a fixed fraction of nu1 so the full two-decoy
        expression runs.
        """
        eta, y0, edet, mu = 1e-2, 1e-6, 0.033, 0.48
        y_ideal, e_ideal = _core.decoy_ideal(eta, y0, edet)
        devs = []
        for step in range(5):
            scale = 0.05 * 0.5**step
            y1_lo, e1_hi, _q1 = channel(mu, scale * mu, 0.3 * scale * mu, eta, y0, edet)
            devs.append(((y_ideal - y1_lo) / y_ideal, e1_hi - e_ideal))
        self.assertMonotone([d[0] for d in devs], rising=False, msg="Y1 deviations")
        self.assertMonotone([d[1] for d in devs], rising=False, msg="e1 deviations")

        excess = []
        for i in range(len(devs) - 1):
            for side in (0, 1):
                ratio = devs[i][side] / devs[i + 1][side]
                self.assertClose(ratio, 2.0, atol=0.05, msg=f"halving ratio {ratio:.4f}")

            excess.append(devs[i][0] / devs[i + 1][0] - 2.0)
        self.assertMonotone(excess, rising=False, msg=f"excess: {excess}")
        self.assertLessEqual(devs[-1][0], 1e-3, msg=f"Y1 deviation {devs[-1][0]:.3e} at nu1 = 0.0015")

    def test_decoy_vacuum(self):
        """
        At a fixed signal and first decoy nu2 = 0 dominates every positive nu2 on both sides
        over 2000 random channels at zero slack: weaker decoys, not more intensities, converge
        onto the ideal.
        """
        rng = random.Random(20260903)
        count = 0
        for _ in range(4000):
            eta = 10 ** rng.uniform(-4.0, -0.5)
            y0 = 10 ** rng.uniform(-7.0, -4.0)
            edet = rng.uniform(0.0, 0.05)
            mu = rng.uniform(0.2, 0.8)
            nu1 = rng.uniform(0.05, 0.9) * mu
            nu2 = rng.uniform(0.01, 0.9) * nu1
            if nu1 + nu2 >= mu:
                continue

            count += 1
            bare = channel(mu, nu1, 0.0, eta, y0, edet)
            both = channel(mu, nu1, nu2, eta, y0, edet)
            tag = f"eta={eta:.3e} mu={mu:.3f} nu1={nu1:.3f} nu2={nu2:.3f}"
            self.assertGreaterEqual(bare[0], both[0], msg=f"Y1 loosened by nu2: {tag}")
            self.assertLessEqual(bare[1], both[1], msg=f"e1 loosened by nu2: {tag}")

            if count == 2000:
                break
        self.assertGreaterEqual(count, 2000, msg=f"channels {count}")

        ladder = [channel(0.48, 0.1, f * 0.1, 1e-2, 1e-6, 0.033) for f in (0.0, 0.2, 0.6, 0.95)]

        self.assertMonotone(
            [row[0] for row in ladder],
            rising=False,
            msg="Y1 over the nu2 ladder",
        )
        self.assertMonotone(
            [row[1] for row in ladder],
            msg="e1 over the nu2 ladder",
        )


class Qubit(Question):
    """
    Photon pairs, Ekert's construction, and six-state against BB84, on one two-level physics.
    """

    def test_one_violation(self):
        """
        pairs.rs's $2\\sqrt{2}v$ and ekert.rs's four-term correlation sum at Ekert's angles
        agree in magnitude under opposite sign conventions, and ekert_qber inverts back to $Q =
        (1 - v)/2$.
        """
        angles = _core.ekert_settings()
        worst = 0.0
        for k in range(0, 41):
            v = k / 40.0
            got = _core.ekert_chsh(angles[0], angles[2], angles[3], angles[5], v)
            want = _core.pair_chsh(v)
            back = _core.ekert_qber(want)
            worst = max(worst, abs(abs(got) - want))
            self.assertLessEqual(got, 0.0, msg=f"v={v}: sign of {got}")

            # atol 1e-15: four-term cosine sum against one product, worst 4.4e-16.
            self.assertClose(abs(got), want, atol=1e-15, msg=f"v={v}: {abs(got)} vs {want}")

            # atol 1e-15: f64 residue on two composed closed forms, worst 5.6e-17.
            self.assertClose(back, 0.5 * (1.0 - v), atol=1e-15, msg=f"v={v}: Q {back}")
        self.assertLessEqual(worst, 1e-15, msg=f"worst {worst:.3e}")

    def test_third_basis_pays(self):
        """
        On the same observables six-state never returns less key than BB84, and at least 0.24%
        more wherever BB84 distils, so the ordering is no shared clamp at zero.
        """
        least = 1.0
        count = 0
        for k in range(1, 240):
            qber = 0.0005 * k
            for gain in (1e-3, 0.02, 0.5):
                one = 0.9 * gain
                bb84 = _core.bb84_rate(1.0, gain, qber, one, qber, 1.0)
                six = _core.sixstate_rate(1.0, gain, qber, one, qber, 1.0)
                self.assertGreaterEqual(six, bb84, msg=f"Q={qber:.4f} gain={gain}: {six} < {bb84}")

                if bb84 > 0.0:
                    count += 1
                    least = min(least, (six - bb84) / bb84)
        self.assertGreater(count, 100, msg=f"BB84-alive points {count}")
        self.assertGreater(least, 2e-3, msg=f"least excess {least:.3e}")


class Quantile(Question):
    """
    z_pe's rational fit against libm's erfc inverted by bisection.
    """

    def test_quantile_matches(self):
        """
        From failure probability 0.5 to 1e-30 z_pe and the erfc inversion agree to 1.3e-8
        absolute and 1.1e-9 relative, the stated accuracy of the rational fit.
        """
        worst = 0.0
        for k in range(0, 61):
            eps = 10.0 ** (-0.3 - 0.5 * k)
            got = _core.z_pe(eps)
            want = bisect(lambda z: math.erfc(z / math.sqrt(2.0)), 0.0, 60.0, eps)
            worst = max(worst, abs(got - want) / want)

            # atol 2e-9: Acklam specifies 1.15e-9, measured 1.12e-9; headroom for libm erfc per wheel target.
            self.assertClose(
                got / want,
                1.0,
                atol=2e-9,
                msg=f"eps={eps:.3e}: z_pe {got} vs erfc inversion {want}",
            )
        self.assertLessEqual(worst, 2e-9, msg=f"worst {worst:.3e}")
        self.assertGreater(worst, 1e-10, msg=f"worst {worst:.3e}")


if __name__ == "__main__":
    rc = Exam(
        "CrossModulation",
        "The discrete-modulation engine reduced to the Gaussian key-rate layer",
        "cross_modulation.md",
    ).run(load(Modulation))
    rc |= Exam(
        "CrossRelay",
        "The CV relay reduced to a point-to-point link, and to its own covariance",
        "cross_relay.md",
    ).run(load(Relay))
    rc |= Exam(
        "CrossTruncation",
        "The Fock engine reduced to the Gaussian one, with convergence demonstrated",
        "cross_truncation.md",
    ).run(load(Truncation))
    rc |= Exam(
        "CrossThreshold",
        "Click, decoy and relay engines on one threshold detector",
        "cross_threshold.md",
    ).run(load(Threshold))
    rc |= Exam(
        "CrossDecoy",
        "The finite-decoy estimator reduced to the infinite-decoy ideal",
        "cross_decoy.md",
    ).run(load(Decoy))
    rc |= Exam(
        "CrossQubit",
        "Three qubit-layer engines on one violation, one disturbance, one ordering",
        "cross_qubit.md",
    ).run(load(Qubit))
    rc |= Exam(
        "CrossQuantile",
        "The finite-key quantile against libm's erfc, into the far tail",
        "cross_quantile.md",
    ).run(load(Quantile))
    sys.exit(rc)
