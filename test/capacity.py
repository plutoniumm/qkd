import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import ETA_COW, F_COW
from kit.forms import LN2, bisect, plob
from qkd import _core

# kit.forms.plob: Pirandola, Laurenza, Ottaviani & Banchi, Nature Communications 8, 15043 (2017),
# bits per channel use.

# 0.02 to 2.0 keeps every weak-coherent optimum interior.
MUS = tuple(0.02 * k for k in range(1, 101))

ETAS = (0.9, 0.5, 0.1, 1e-2, 1e-3, 1e-4)

# COW' at test/sdp.py's operating point, 0.2 dB/km behind ETA_COW.
COW_DARK = 1e-8
COW_SPANS = (0.0, 10.0, 20.0, 40.0, 60.0, 80.0)


def vacuum(dist):
    """
    (best COW' rate over a (mu, t_B) grid, capacity) at dist km, Gao's derived phase error; bits per
    pulse.
    """
    eta = ETA_COW * 10.0 ** (-0.2 * dist / 10.0)
    best = 0.0
    for i in range(30):
        mu = 10.0 ** (-6.0 + 5.4 * i / 29.0)
        for t_b in (0.15, 0.35, 0.5, 0.65, 0.85):
            try:
                q0, q1 = _core.sdp_gains(mu, t_b, eta, COW_DARK)
                phase = _core.sdp_analytic(mu, q0, q1)
            except ValueError:
                continue

            if phase >= 0.5:
                continue

            gain, e_bit = _core.sdp_data(mu, t_b, eta, COW_DARK)
            best = max(best, 0.5 * _core.cow_rate(gain, e_bit, phase, F_COW))

    return best, plob(eta)


def bb84(eta):
    """
    BB84-WCP at infinite decoy, dark-free and aligned.
    """
    best = 0.0
    y1, e1 = _core.decoy_ideal(eta, 0.0, 0.0)
    for mu in MUS:
        q_mu, e_mu = _core.decoy_gain(mu, eta, 0.0, 0.0)
        one = mu * math.exp(-mu) * y1
        best = max(best, _core.bb84_rate(1.0, q_mu, e_mu, one, e1, 1.0))

    return best


def sixstate(eta):
    """
    The same observables priced by the six-state Holevo bound.
    """
    best = 0.0
    y1, e1 = _core.decoy_ideal(eta, 0.0, 0.0)
    for mu in MUS:
        q_mu, e_mu = _core.decoy_gain(mu, eta, 0.0, 0.0)
        one = mu * math.exp(-mu) * y1
        best = max(best, _core.sixstate_rate(1.0, q_mu, e_mu, one, e1, 1.0))

    return best


def sarg(eta):
    """
    SARG04, paid on the one- and two-photon yields at the infinite-decoy limit.
    """
    best = 0.0
    y1, e1 = _core.sarg_yield(eta, 0.0, 0.0, 1)
    y2, e2 = _core.sarg_yield(eta, 0.0, 0.0, 2)
    for mu in MUS:
        q_mu, e_mu = _core.sarg_gain(mu, eta, 0.0, 0.0)
        one = mu * math.exp(-mu) * y1
        two = 0.5 * mu * mu * math.exp(-mu) * y2
        best = max(best, _core.sarg_rate(q_mu, e_mu, one, e1, two, e2, 1.0))

    return best


def dps(eta):
    return max(_core.dps_rate(1.0 - math.exp(-eta * mu), 0.0, mu, 1.0) for mu in MUS)


def rrdps(eta):
    return max(_core.rrdps_optimum(16, mu, eta, 0.0, 0.0, 1.0)[2] for mu in MUS[:40])


def b92(eta):
    """
    Plain B92, single-photon source, no depolarisation.
    """

    return max(_core.b92_plain(math.exp(-2.0 * mu), 1.0 - eta, 0.0, 1.0)[3] for mu in MUS)


def bbm92(eta):
    """
    BBM92 off a central pair source, each arm sqrt(eta).
    """
    arm = math.sqrt(eta)
    best = 0.0
    for lam in MUS:
        gain = _core.pair_gain(lam, arm, arm, 0.0, 0.0)
        err = _core.pair_error(lam, arm, arm, 0.0, 0.0, 0.0)
        best = max(best, _core.pair_rate(gain, err, err, 1.0, 1.0))

    return best


def ekert(eta):
    """
    E91 on that same source, S read off the model that produced it.
    """
    arm = math.sqrt(eta)
    best = 0.0
    for lam in MUS:
        gain = _core.pair_gain(lam, arm, arm, 0.0, 0.0)
        err = _core.pair_error(lam, arm, arm, 0.0, 0.0, 0.0)
        s = 2.0 * math.sqrt(2.0) * (1.0 - 2.0 * err)
        best = max(best, _core.ekert_rate(gain, err, s, 1.0, 1.0, "measured"))

    return best


def mpsk(eta):
    """
    M-PSK into an ideal heterodyne receiver.
    """

    return max(_core.dm_rate(m, 0.05 * k, eta, 0.0, 1.0, 0.0, 1.0)[2] for m in (4, 8, 16) for k in range(1, 60))


def gauss(eta):
    """
    Gaussian modulation, both detectors, ideal receiver at the Shannon limit.
    """

    return max(
        _core.cv_rate(10.0 ** (-1.0 + 0.2 * k), eta, 0.0, 1.0, 0.0, 1.0, hom, True)[2]
        for k in range(1, 60)
        for hom in (True, False)
    )


FAMILIES = (
    ("bb84-wcp", bb84),
    ("six-state", sixstate),
    ("sarg04", sarg),
    ("dps", dps),
    ("rrdps", rrdps),
    ("b92-plain", b92),
    ("bbm92", bbm92),
    ("e91", ekert),
    ("m-psk", mpsk),
    ("gaussian", gauss),
)


def relay(ta, tb, wa, wb, g, gp):
    """
    (cvmdi_rate, cvmdi_point) at one configuration, None where the engine refuses; va = 1e5 is on
    the asymptote.
    """
    try:
        chi = _core.cvmdi_noise(ta, tb, wa, wb, g, gp)
        bound = _core.cvmdi_rate(ta, tb, chi)
        point = _core.cvmdi_point(1e5, ta, tb, wa, wb, g, gp, 1.0)[2]
    except ValueError:
        return None

    return (bound, point)


class Capacity(Question):
    """
    No rate on observables derived from a channel exceeds its repeaterless capacity.
    """

    def test_capacity_form(self):
        """
        The capacity goes through log1p, not -log2(1 - eta), which reads 8.3e-8 off at 100 dB, 11.0%
        high at 160 dB and 0.0 at 176 dB.
        """
        naive = []
        for db in (100.0, 140.0, 160.0, 176.0):
            eta = 10.0 ** (-db / 10.0)
            naive.append((db, plob(eta), -math.log2(1.0 - eta)))

        # atol 2e-7: f64 residue of 1 - eta at 100 dB.
        self.assertClose(
            naive[0][2] / naive[0][1],
            1.0,
            atol=2e-7,
            msg=f"100 dB: {naive[0]}",
        )

        # atol 5e-5: rounding of 1 - 1e-16.
        self.assertClose(
            naive[2][2] / naive[2][1],
            1.11022,
            atol=5e-5,
            msg=f"160 dB: {naive[2]}",
        )
        self.assertEqual(naive[3][2], 0.0, msg=f"176 dB: {naive[3]}")
        self.assertGreater(naive[3][1], 0.0, msg=f"176 dB log1p: {naive[3][1]}")

    def test_vacuum_bounded(self):
        """
        COW' with a derived phase error peaks at 3.41e-4 of the repeaterless capacity from 0 to 80 km.
        """
        worst = 0.0
        for dist in COW_SPANS:
            best, cap = vacuum(dist)
            share = best / cap
            worst = max(worst, share)
            self.assertLess(share, 1e-3, msg=f"{dist} km: share {share:.6f}")

        self.assertClose(worst, 3.41e-4, atol=5e-6, msg=f"worst {worst:.6f}")

    def test_families_bounded(self):
        """
        Ten families at optimal intensity, dark-free, aligned and at the Shannon limit, stay under
        the repeaterless capacity for eta = 0.9 to 1e-4, worst 0.677 at Gaussian modulation.
        """
        worst = 0.0
        where = ""
        for name, fn in FAMILIES:
            for eta in ETAS:
                ratio = fn(eta) / plob(eta)
                if ratio > worst:
                    worst = ratio
                    where = f"{name} at eta={eta:.0e}"
                self.assertLess(ratio, 1.0, msg=f"{name} at eta={eta:.0e}: ratio {ratio:.4f}")

        self.assertGreater(worst, 0.6, msg=f"worst {worst:.4f}")
        self.assertLess(worst, 0.7, msg=f"worst {worst:.4f} at {where}")

    def test_gaussian_halves(self):
        """
        An ideal reverse-reconciled homodyne link at zero excess noise reaches half the repeaterless
        capacity to 1e-10 once the 1/V_A deficit is extrapolated out by (10 K(1e8) - K(1e7))/9.
        """
        worst = 0.0
        for eta in (0.99, 0.9, 0.5, 0.2, 0.1, 0.01, 1e-3):
            near = _core.cv_rate(1e7, eta, 0.0, 1.0, 0.0, 1.0, True, True)[2]
            far = _core.cv_rate(1e8, eta, 0.0, 1.0, 0.0, 1.0, True, True)[2]
            limit = (10.0 * far - near) / 9.0
            worst = max(worst, abs(limit - 0.5 * plob(eta)))

            # atol 1e-10: f64 residue, amplified 10x by the extrapolation.
            self.assertClose(
                limit,
                0.5 * plob(eta),
                atol=1e-10,
                msg=f"eta={eta}: {limit:.15f} vs {0.5 * plob(eta):.15f}",
            )

        self.assertLess(worst, 1e-10, msg=f"worst {worst:.3e}")

    def test_heterodyne_band(self):
        """
        Heterodyne sits between half the repeaterless capacity and the capacity, its ratio falling
        from 0.50084 at eta = 1e-2 to 0.5000008 at 1e-5, the excess tenfold per decade.
        """
        excess = []
        for eta in (1e-2, 1e-3, 1e-4, 1e-5):
            best = max(
                _core.cv_rate(10.0 ** (-1.0 + 0.2 * k), eta, 0.0, 1.0, 0.0, 1.0, False, True)[2] for k in range(1, 60)
            )
            ratio = best / plob(eta)
            excess.append(ratio - 0.5)
            self.assertGreater(ratio, 0.5, msg=f"eta={eta:.0e}: ratio {ratio:.7f}")
            self.assertLess(ratio, 1.0, msg=f"eta={eta:.0e}: ratio {ratio:.7f}")

        for i in range(len(excess) - 1):
            step = excess[i] / excess[i + 1]

            # atol 0.08: the second-order term in eta leaves the ratio at 10.045 worst.
            self.assertClose(step, 10.0, atol=0.08, msg=f"decade {i}: {step:.4f}")

    def test_relay_bounded(self):
        """
        Neither cvmdi_rate nor cvmdi_point reaches the repeaterless capacity of its arms in the 4518
        of 40000 random configurations the engine accepts, closest approach 0.41: cvmdi_point's
        above-capacity claim is unreproduced.
        """
        rng = random.Random(20260903)
        worst = 0.0
        count = 0
        for _ in range(40000):
            ta = rng.uniform(0.02, 1.0)
            tb = rng.uniform(0.02, 1.0)
            wa = 1.0 + 10.0 ** rng.uniform(-4.0, 2.0)
            wb = 1.0 + 10.0 ** rng.uniform(-4.0, 2.0)
            g = rng.uniform(-8.0, 8.0)
            out = relay(ta, tb, wa, wb, g, rng.uniform(-8.0, 8.0))
            if out is None:
                continue

            count += 1
            for total in (ta * tb, ta, tb):
                worst = max(worst, max(out) / plob(total))
                self.assertLess(
                    max(out),
                    plob(total),
                    msg=f"({ta:.4f}, {tb:.4f}, {wa:.4f}, {wb:.4f}): {max(out):.6f} vs {plob(total):.6f}",
                )

        self.assertGreater(count, 3500, msg=f"accepted {count}")
        self.assertLess(worst, 0.6, msg=f"worst {worst:.4f}")


class Finite(Question):
    """
    Papanastasiou, Ottaviani & Pirandola, Phys. Rev. A 96, 042332 (2017), Eq. (23), against the
    repeaterless capacity.
    """

    def test_finite_bounded(self):
        """
        Over 4000 random relay configurations at N = 1e8 and 1e12, cvmdi_finite stays under the
        repeaterless capacity of its arms and under half the asymptotic cvmdi_gauss rate.
        """
        rng = random.Random(20260905)
        count = 0
        worst = 0.0
        for _ in range(4000):
            ta = rng.uniform(0.05, 0.999)
            tb = rng.uniform(0.05, 0.999)
            omega = 1.0 + 10.0 ** rng.uniform(-4.0, 0.0)
            edge = math.sqrt(omega * omega - 1.0) * rng.uniform(-1.0, 1.0)
            try:
                vq, vp = _core.cvmdi_excess(ta, tb, omega, omega, edge, -edge)
                limit = _core.cvmdi_gauss(1e4, ta, tb, vq, vp, 1.0)[2]
            except ValueError:
                continue

            count += 1
            for n in (1e8, 1e12):
                key = _core.cvmdi_finite(1e4, ta, tb, vq, vp, 1.0, n, 0.5, 6.5, 1e-10, 1e-10, "gaussian")[0]
                self.assertLessEqual(
                    key,
                    0.5 * limit + 1e-12,
                    msg=f"N={n} ({ta:.4f}, {tb:.4f}): {key:.6f} vs {0.5 * limit:.6f}",
                )
                for total in (ta * tb, ta, tb):
                    worst = max(worst, key / plob(total))
                    self.assertLess(
                        key,
                        plob(total),
                        msg=f"N={n} ({ta:.4f}, {tb:.4f}): {key:.6f} vs {plob(total):.6f}",
                    )

        self.assertGreater(count, 3500, msg=f"accepted {count}")
        self.assertLess(worst, 0.6, msg=f"worst {worst:.4f}")


class Escapes(Question):
    """
    Rates above the repeaterless capacity: a ceiling need not be tight, and a supplied error rate
    describes no channel.
    """

    def test_ceiling_escapes(self):
        """
        b92_ceiling, pricing beam splitting alone, crosses the repeaterless capacity at
        transmittance 0.120257 and is 4.87 times it at 1e-4, while b92_plain stays under it.
        """
        cross = bisect(lambda t: _core.b92_optimum(t, 1.0, 1.0, 0.0)[2] - plob(t), 0.05, 0.3)

        # atol 1e-5: b92_optimum's golden-section search; the bisect bracket is past its last bit.
        self.assertClose(cross, 0.120257, atol=1e-5, msg=f"crossing {cross:.6f}")
        self.assertGreater(
            _core.b92_optimum(1e-4, 1.0, 1.0, 0.0)[2] / plob(1e-4),
            4.0,
            msg="ceiling/capacity at eta=1e-4",
        )
        self.assertLess(
            _core.b92_optimum(0.5, 1.0, 1.0, 0.0)[2] / plob(0.5),
            1.0,
            msg="ceiling/capacity at eta=0.5",
        )

        for eta in (1e-4, 1e-3, 1e-2, 0.1):
            self.assertLess(
                b92(eta),
                plob(eta),
                msg=f"b92_plain at eta={eta:.0e}",
            )

    def test_dialled_escapes(self):
        """
        cow_rate at a supplied zero phase error crosses the repeaterless capacity at mu = 1/ln 2
        = 1.4427 and sits at 0.693 of it at mu = 1.
        """
        for eta in (1e-2, 1e-3, 1e-4):
            cross = bisect(
                lambda mu, e=eta: _core.cow_rate(1.0 - math.exp(-e * mu), 0.0, 0.0, 1.0) - plob(e),
                0.5,
                5.0,
            )
            self.assertGreater(cross, 1.44, msg=f"eta={eta:.0e}: crossing mu={cross:.6f}")
            self.assertLess(
                _core.cow_rate(1.0 - math.exp(-eta), 0.0, 0.0, 1.0) / plob(eta),
                0.7,
                msg=f"eta={eta:.0e}: ratio at mu=1",
            )

        deep = bisect(
            lambda mu: _core.cow_rate(1.0 - math.exp(-1e-4 * mu), 0.0, 0.0, 1.0) - plob(1e-4),
            0.5,
            5.0,
        )

        # atol 3e-4: the crossing's leading correction is first order in eta.
        self.assertClose(
            deep,
            1.0 / LN2,
            atol=3e-4,
            msg=f"crossing at eta=1e-4: {deep:.8f}",
        )


if __name__ == "__main__":
    rc = Exam(
        "Repeaterless",
        "Every family whose observables are derived sits under the PLOB capacity",
        "capacity_plob.md",
    ).run(load(Capacity))
    rc |= Exam(
        "CapacityFinite",
        "The finite-size CV relay number against the same repeaterless ceiling",
        "capacity_finite.md",
    ).run(load(Finite))
    rc |= Exam(
        "CapacityEscapes",
        "The two mechanisms that report above the capacity",
        "capacity_escapes.md",
    ).run(load(Escapes))
    sys.exit(rc)
