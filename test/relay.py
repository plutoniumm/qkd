import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.checks import Guarded
from kit.forms import bisect
from qkd import _core

# Fibre attenuation, dB/km. NOT kit.anchors.ALPHA, the GYS receiver's 0.21 dB/km.
ALPHA = 0.2


def loss(km):
    return 10.0 ** (-ALPHA * km / 10.0)


def span(tau):
    return -10.0 * math.log10(tau) / ALPHA


def pure(ta, tb):
    """
    Rate of a relay whose two arms are pure loss.
    """
    return _core.cvmdi_rate(ta, tb, _core.cvmdi_floor(ta, tb))


def p2p(t, va=1e6):
    return _core.cv_rate(va, t, 0.0, 1.0, 0.0, 1.0, False, True)[2]


def matrix(flat, n=4):
    return [list(flat[i * n : (i + 1) * n]) for i in range(n)]


def closed(va, ta, tb, wa, wb, g=0.0, gp=0.0):
    """
    The published post-relay covariance, written out independently here.
    """
    mu = va + 1.0
    kappa = (1.0 - ta) * wa + (1.0 - tb) * wb
    u = 2.0 * math.sqrt((1.0 - ta) * (1.0 - tb))
    lam = kappa - u * g
    lamp = kappa + u * gp
    th = (ta + tb) * mu + lam
    thp = (ta + tb) * mu + lamp
    off = math.sqrt(ta * tb)
    sq = mu * mu - 1.0

    return [
        [mu - sq * ta / th, 0.0, sq * off / th, 0.0],
        [0.0, mu - sq * ta / thp, 0.0, -sq * off / thp],
        [sq * off / th, 0.0, mu - sq * tb / th, 0.0],
        [0.0, -sq * off / thp, 0.0, mu - sq * tb / thp],
    ]


def layered(va, ta, tb, wa, wb):
    """
    The same optics driven through the public Gaussian layer.
    """
    r = math.acosh(va + 1.0) / 2.0
    st = _core.GaussianState.vacuum(4)
    st = st.squeeze(0, r).squeeze(1, -r).bs(0, 1, 0.5)
    st = st.squeeze(2, r).squeeze(3, -r).bs(2, 3, 0.5)
    st = st.thermal_loss(1, ta, (1.0 - ta) * (wa - 1.0) / ta, True)
    st = st.thermal_loss(2, tb, (1.0 - tb) * (wb - 1.0) / tb, True)
    st = st.bs(1, 2, 0.5)
    st = st.condition(1, math.pi / 2.0, 0.0).condition(1, 0.0, 0.0)

    return [2.0 * x for x in st.cov()]


def entropy(x):
    """
    The engine's own von Neumann entropy grouping, which crate::std does not share.
    """
    if not x > 1.0:
        return 0.0

    b = (x - 1.0) / 2.0

    return b * math.log1p(2.0 / (x - 1.0)) / math.log(2.0) + math.log2((x + 1.0) / 2.0)


def rebuild(cov, beta):
    """
    (i_ab, chi_e, key) off the post-relay covariance alone, its determinant assumed
    to factorise over the quadratures.
    """
    dx = cov[0] * cov[10] - cov[2] * cov[8]
    dp = cov[5] * cov[15] - cov[7] * cov[13]
    delta = (
        cov[0] * cov[5]
        - cov[1] * cov[4]
        + cov[10] * cov[15]
        - cov[11] * cov[14]
        + 2.0 * (cov[2] * cov[7] - cov[3] * cov[6])
    )
    disc = math.sqrt(max(0.0, delta * delta - 4.0 * dx * dp))
    top = math.sqrt(max(0.0, (delta + disc) / 2.0))
    low = math.sqrt(max(0.0, (delta - disc) / 2.0))
    ai = [cov[0] + 1.0, cov[1], cov[4], cov[5] + 1.0]
    dai = ai[0] * ai[3] - ai[1] * ai[2]
    inv = [ai[3] / dai, -ai[1] / dai, -ai[2] / dai, ai[0] / dai]
    bc = [0.0, 0.0, 0.0, 0.0]
    for i in range(2):
        for j in range(2):
            acc = 0.0
            for k in range(2):
                for m in range(2):
                    acc += cov[k * 4 + 2 + i] * inv[k * 2 + m] * cov[m * 4 + 2 + j]

            bc[i * 2 + j] = cov[(2 + i) * 4 + 2 + j] - acc

    db = (cov[10] + 1.0) * (cov[15] + 1.0) - cov[11] * cov[14]
    dbc = (bc[0] + 1.0) * (bc[3] + 1.0) - bc[1] * bc[2]
    i_ab = 0.5 * math.log2(db / dbc)
    chi_e = entropy(top) + entropy(low) - entropy(math.sqrt(max(0.0, bc[0] * bc[3] - bc[1] * bc[2])))

    return i_ab, chi_e, beta * i_ab - chi_e


def estimated(va, ta, tb, vq, vp):
    """
    The published post-relay covariance in POP17's estimated coordinates, Eqs. (24)-(26).
    """
    phi = (ta + tb) * va + 2.0 + 2.0 * vq
    phip = (ta + tb) * va + 2.0 + 2.0 * vp
    mu = va + 1.0
    sq = va * (va + 2.0)
    off = math.sqrt(ta * tb)

    return [
        mu - sq * ta / phi,
        0.0,
        sq * off / phi,
        0.0,
        0.0,
        mu - sq * ta / phip,
        0.0,
        -sq * off / phip,
        sq * off / phi,
        0.0,
        mu - sq * tb / phi,
        0.0,
        0.0,
        -sq * off / phip,
        0.0,
        mu - sq * tb / phip,
    ]


def sigmas(m, va, ta, tb, vq, vp):
    """
    POP17's four estimator standard deviations, Eqs. (14)-(16), (18) and (20), written out.
    """
    noises = (1.0 + vq, 1.0 + vp)
    out = []
    for near, far in ((ta, tb), (tb, ta)):
        mix = near + 0.5 * far
        pair = [8.0 * near / m * mix * (1.0 + vn / (mix * va)) for vn in noises]
        out.append(math.sqrt(pair[0] * pair[1] / (pair[0] + pair[1])))

    return tuple(out) + tuple(math.sqrt(2.0 / m) * vn for vn in noises)


def penalty(n, eps_s, eps_pa):
    """
    The two-term smooth-min-entropy correction at dim H_X = 2, which cv_finite carries too.
    """
    return 7.0 * math.sqrt(math.log2(2.0 / eps_s) / n) + 2.0 / n * math.log2(1.0 / eps_pa)


def dip(mu_a, mu_b, contrast, shots=1024):
    """
    Phase-averaged coincidence dip from the relay's own port statistics.
    """
    flat = 0.0
    deep = 0.0
    for k in range(shots):
        delta = 2.0 * math.pi * k / shots
        wide = _core.relay_ports(mu_a, mu_b, delta, 0.0)
        tight = _core.relay_ports(mu_a, mu_b, delta, contrast)
        flat += _core.relay_counts(wide[0], wide[1], 1.0, 0.0)[1]
        deep += _core.relay_counts(tight[0], tight[1], 1.0, 0.0)[1]

    return (flat - deep) / flat


class Conditioned(Question):
    """
    Pirandola et al., Nature Photonics 9, 397 (2015), arXiv:1312.4104 --
    covariance Eq. (VabGamma), entropies Eqs. (HolINFOapp) and (mutuaINab).
    """

    def test_cov_physical(self):
        """
        `cvmdi_cov` satisfies V + i*Omega/2 >= 0 over 27 combinations of arm loss,
        environment variance and Eve correlation.
        """
        count = 0
        for ta in (0.05, 0.5, 1.0):
            for tb in (0.05, 0.5, 1.0):
                for wa, wb, g in ((1.0, 1.0, 0.0), (1.8, 1.2, 0.0), (2.5, 1.8, 0.4)):
                    cov = _core.cvmdi_cov(6.0, ta, tb, wa, wb, g, -g)
                    count += 1

                    self.assertPhysical(
                        [[0.5 * x for x in row] for row in matrix(cov)],
                        msg=f"tau=({ta}, {tb}) omega=({wa}, {wb}) g={g}",
                    )

        self.assertEqual(count, 27, msg=f"swept {count} of 27")

    def test_cov_matches(self):
        """
        `cvmdi_cov` reproduces the published closed form to 1e-12, correlated ancillas
        included.
        """
        for va, ta, tb, wa, wb, g, gp in (
            (10.0, 0.9, 0.5, 1.3, 1.7, 0.0, 0.0),
            (10.0, 0.9, 0.5, 1.3, 1.7, 0.2, -0.2),
            (10.0, 0.9, 0.5, 1.6, 1.6, 0.5, 0.0),
            (6.0, 0.7, 0.7, 1.05, 1.2, 0.0, 0.0),
            (3.0, 1.0, 0.1, 1.0, 1.0, 0.0, 0.0),
        ):
            got = matrix(_core.cvmdi_cov(va, ta, tb, wa, wb, g, gp))
            want = closed(va, ta, tb, wa, wb, g, gp)

            self.gridClose(got, want, atol=1e-12, msg=f"tau=({ta}, {tb}) g={g}")

    def test_cov_conditions(self):
        """
        `cvmdi_cov` matches the same optics driven through the public Gaussian
        layer to 1e-12.
        """
        for va, ta, tb, wa, wb in (
            (4.0, 0.9, 0.5, 1.3, 1.7),
            (9.0, 0.6, 0.95, 1.0, 1.4),
            (2.0, 0.3, 0.3, 2.0, 1.1),
        ):
            got = matrix(_core.cvmdi_cov(va, ta, tb, wa, wb, 0.0, 0.0))
            want = matrix(layered(va, ta, tb, wa, wb))

            self.gridClose(got, want, atol=1e-12, msg=f"va={va} tau=({ta}, {tb})")

    def test_cov_structure(self):
        """
        The projection onto q_A = q_B, p_A = -p_B leaves a positive x-x and a negative
        p-p correlation, x-p even unless the attack is correlated.
        """
        even = matrix(_core.cvmdi_cov(8.0, 0.8, 0.4, 1.5, 1.5, 0.0, 0.0))
        odd = matrix(_core.cvmdi_cov(8.0, 0.8, 0.4, 1.5, 1.5, 0.5, 0.0))

        self.assertGreater(even[0][2], 0.0, msg=f"x-x correlation {even[0][2]}")
        self.assertLess(even[1][3], 0.0, msg=f"p-p correlation {even[1][3]}")
        self.assertClose(even[0][0], even[1][1], msg="uncorrelated attack not x-p even")
        self.assertGreater(abs(odd[0][0] - odd[1][1]), 1e-3, msg="correlated attack stayed even")

    def test_cov_vacuum(self):
        """
        The 4x4 is in SHOT-NOISE UNITS, vacuum 1 against the Gaussian layer's 1/2, so an
        unmodulated sender leaves the identity whatever the arms did.
        """
        eye = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        clean = _core.cvmdi_cov(1e-12, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)

        self.gridClose(matrix(clean), eye, atol=1e-11, msg="vacuum is not the identity")

        murky = _core.cvmdi_cov(1e-9, 0.5, 0.5, 2.0, 2.0, 0.0, 0.0)

        self.gridClose(matrix(murky), eye, atol=1e-8, msg="thermal arms moved the vacuum")

    def test_cov_rebuilds(self):
        """
        The (i_ab, chi_e, key) triple rebuilt from the 4x4 and beta alone agrees with
        `cvmdi_point` to 1e-12.
        """
        for args in (
            (6.0, 0.9, 0.5, 1.3, 1.7, 0.0, 0.0),
            (1e4, 0.95, 0.95, 1.2, 1.2, 0.1, -0.1),
            (50.0, 0.7, 0.7, 1.05, 1.2, 0.3, -0.2),
            (1e3, 0.4, 0.9, 2.0, 1.5, 0.5, -0.5),
        ):
            cov = _core.cvmdi_cov(*args)
            big = max(abs(x) for x in cov)
            odd = max(abs(cov[k]) for k in (1, 3, 4, 6, 9, 11, 12, 14))

            self.assertLess(odd, 1e-12 * big, msg=f"va = {args[0]}: x-p entry {odd:g}")

            got = rebuild(cov, 0.95)
            want = _core.cvmdi_point(*args, 0.95)

            for a, b in zip(got, want):
                self.assertClose(a, b, atol=1e-12, msg=f"va = {args[0]}")

    def test_cov_apart(self):
        """
        Two attacks sharing one chi share `cvmdi_rate` to the last bit while their
        covariances and per-attack rates differ by more than 1e-3.
        """
        ta, tb, wa, wb = 0.95, 0.95, 1.2, 1.2
        u = 2.0 * math.sqrt((1.0 - ta) * (1.0 - tb))
        held = (ta + tb) + (1.0 - ta) * wa + (1.0 - tb) * wb
        # The partner of (0.1, -0.1) at the same (sum + lambda)(sum + lambda').
        prod = (held - u * 0.1) * (held - u * 0.1)
        gp = 0.05
        g = (held - prod / (held + u * gp)) / u
        one = _core.cvmdi_noise(ta, tb, wa, wb, 0.1, -0.1)
        two = _core.cvmdi_noise(ta, tb, wa, wb, g, gp)

        self.assertEqual(one, two, msg=f"chi split the attacks: {one} vs {two}")
        self.assertEqual(
            _core.cvmdi_rate(ta, tb, one),
            _core.cvmdi_rate(ta, tb, two),
            msg="cvmdi_rate split them",
        )

        left = _core.cvmdi_point(1e4, ta, tb, wa, wb, 0.1, -0.1, 0.95)[2]
        right = _core.cvmdi_point(1e4, ta, tb, wa, wb, g, gp, 0.95)[2]

        self.assertGreater(abs(left - right), 1e-3, msg=f"per-attack gap {abs(left - right):g}")

        first = _core.cvmdi_cov(1e4, ta, tb, wa, wb, 0.1, -0.1)
        second = _core.cvmdi_cov(1e4, ta, tb, wa, wb, g, gp)
        gaps = [abs(a - b) for a, b in zip(first, second)]

        self.assertGreater(max(gaps), 1e-3, msg=f"covariance gap {max(gaps):g}")

    def test_cov_lossless(self):
        """
        Lossless arms leave a pure two-mode state: chi_e under 1e-5, key the whole of
        I_AB.
        """
        i_ab, chi_e, key = _core.cvmdi_point(4.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

        self.assertLess(chi_e, 1e-5, msg=f"chi_e = {chi_e:g}")
        self.assertClose(key, i_ab, atol=1e-5, msg=f"key {key:g} vs I_AB {i_ab:g}")
        self.assertGreater(i_ab, 0.0, msg=f"I_AB = {i_ab:g}")

    def test_cov_asymptote(self):
        """
        `cvmdi_point` approaches `cvmdi_rate` monotonically in va, to 3e-5 at va = 1e5.
        """
        for ta, tb, wa, wb, g in (
            (0.9, 0.5, 1.3, 1.7, 0.0),
            (0.99, 0.2, 1.0, 1.0, 0.0),
            (0.7, 0.7, 1.05, 1.2, 0.0),
            (1.0, 0.1, 1.0, 1.0, 0.0),
            (0.9, 0.5, 2.0, 2.0, 1.2),
            (0.95, 0.9, 3.0, 3.0, -1.5),
        ):
            chi = _core.cvmdi_noise(ta, tb, wa, wb, g, -g)
            want = _core.cvmdi_rate(ta, tb, chi)
            keys = [_core.cvmdi_point(va, ta, tb, wa, wb, g, -g, 1.0)[2] for va in (1e3, 1e4, 1e5)]
            gaps = [abs(k - want) for k in keys]

            self.assertMonotone(gaps, rising=False, msg=f"tau=({ta}, {tb})")
            self.assertClose(keys[-1], want, atol=3e-5, msg=f"tau=({ta}, {tb})")

    def test_cov_correlated(self):
        """
        A harming correlation costs key monotonically and beats two independent cloners,
        the opposite sign aiding detection.
        """
        args = (0.9, 0.4, 1.6, 1.6)
        plain = _core.cvmdi_point(1e4, *args, 0.0, 0.0, 1.0)[2]
        keys = [_core.cvmdi_point(1e4, *args, -g, g, 1.0)[2] for g in (0.0, 0.2, 0.4, 0.6)]
        helps = _core.cvmdi_point(1e4, *args, 0.6, -0.6, 1.0)[2]

        self.assertMonotone(keys, rising=False, msg=f"harming keys {keys}")
        self.assertLess(keys[-1], plain, msg=f"correlated {keys[-1]:g} vs plain {plain:g}")
        self.assertGreater(helps, plain, msg=f"helping {helps:g} vs plain {plain:g}")
        self.assertFinite(keys, msg="a correlated point is not finite")


class Position(Question):
    """
    All numbers from arXiv:1312.4104: rate Eq. (RateASYMM), symmetric branch
    Eq. (RateSYMM), chi_loss = 2(tau_a + tau_b)/(tau_a*tau_b), and its two
    pure-loss zeroes.
    """

    def test_relay_symmetric(self):
        """
        The pure-loss symmetric rate vanishes at tau = 0.84, the published 3.8 km
        per arm at 0.2 dB/km.
        """
        cross = bisect(lambda t: pure(t, t), 0.5, 0.99)

        self.assertClose(cross, 0.84, atol=5e-3, msg=f"symmetric zero at {cross:g}")
        self.assertClose(span(cross), 3.8, atol=0.05, msg=f"{span(cross):g} km per arm")
        self.assertGreater(pure(0.9, 0.9), 0.0, msg=f"tau = 0.9 gives {pure(0.9, 0.9):g}")
        self.assertLess(pure(0.8, 0.8), 0.0, msg=f"tau = 0.8 gives {pure(0.8, 0.8):g}")

    def test_relay_decoder(self):
        """
        A relay beside the DECODER gives log2(tau/((1-tau)e)), zero at e/(1+e) = 0.7311,
        or 6.8 km.
        """
        cross = bisect(lambda t: pure(t, 1.0), 0.5, 0.99)
        want = math.e / (1.0 + math.e)

        self.assertClose(cross, want, atol=1e-6, msg=f"zero at {cross:g}, want e/(1+e)")
        self.assertClose(span(cross), 6.8, atol=0.05, msg=f"{span(cross):g} km")

        for ta in (0.75, 0.85, 0.95):
            self.assertClose(
                pure(ta, 1.0),
                math.log2(ta / ((1.0 - ta) * math.e)),
                atol=1e-9,
                msg=f"closed form at tau_a={ta}",
            )

    def test_relay_encoder(self):
        """
        A relay beside the ENCODER keeps the rate positive out to 300 km.
        """
        keys = [pure(1.0, loss(d)) for d in (25.0, 50.0, 100.0, 200.0, 300.0)]

        self.assertMonotone(keys, rising=False, msg=f"keys {keys}")
        self.assertFinite(keys, msg="a distance is not finite")
        self.assertGreater(keys[-1], 1e-9, msg=f"300 km gives {keys[-1]:g}")

    def test_relay_asymmetry(self):
        """
        At a total transmittance of 0.1 the rate climbs from below -2 bit at the
        symmetric point to positive at the encoder.
        """
        total = 0.1
        mid = math.sqrt(total)
        keys = [pure(ta, total / ta) for ta in (0.11, 0.2, mid, 0.5, 0.9, 1.0)]

        self.assertMonotone(keys, msg=f"keys {keys}")
        self.assertLess(pure(mid, mid), -2.0, msg=f"central relay at {pure(mid, mid):g}")
        self.assertGreater(pure(1.0, total), 0.0, msg=f"extreme relay at {pure(1.0, total):g}")

    def test_relay_noise(self):
        """
        Arm noise raises chi above the loss floor, pure loss is omega = 1, and
        `relay_omega(0.5, 0.04)` reads 1.04 from an input-referred xi.
        """
        base = _core.cvmdi_noise(0.9, 0.3, 1.0, 1.0, 0.0, 0.0)

        self.assertClose(base, _core.cvmdi_floor(0.9, 0.3), msg=f"omega = 1 gives chi {base:g}")

        keys = [
            _core.cvmdi_rate(0.9, 0.3, _core.cvmdi_noise(0.9, 0.3, w, 1.0, 0.0, 0.0)) for w in (1.0, 1.02, 1.05, 1.2)
        ]

        self.assertMonotone(keys, rising=False, msg=f"keys {keys}")
        self.assertClose(_core.relay_omega(0.5, 0.04), 1.04, msg=f"omega {_core.relay_omega(0.5, 0.04):g}")

    def test_relay_seam(self):
        """
        Straddling tau_a = tau_b by one part in 1e7 moves the rate by under 1e-6.
        """
        chi = 6.0
        exact = _core.cvmdi_rate(0.7, 0.7, chi)
        near = _core.cvmdi_rate(0.7, 0.7 * (1.0 + 1e-7), chi)
        far = _core.cvmdi_rate(0.7, 0.7 * (1.0 + 1e-4), chi)

        self.assertClose(near, exact, atol=1e-6, msg=f"1e-7 apart: {near:g} vs {exact:g}")
        self.assertClose(far, exact, atol=1e-3, msg=f"1e-4 apart: {far:g} vs {exact:g}")


class Untrusted(Question):
    """
    Repeaterless ceiling from Pirandola, Laurenza, Ottaviani & Banchi, Nature
    Communications 8, 15043 (2017).
    """

    def test_relay_ceiling(self):
        """
        Every PURE-LOSS relay configuration sits under -log2(1 - T); a helping
        correlation donates entanglement and is not repeaterless-bounded.
        """
        # Not kit.forms.plob, here or at test_audit_helps: 11% high at 160 dB, and
        # equal to 1.7e-15 only above tau = 0.02.
        for total in (0.5, 0.2, 0.1, 0.02):
            cap = -math.log2(1.0 - total)
            for ta in (total, 0.4, 0.7, 0.95, 1.0):
                if ta < total:
                    continue

                self.assertLess(pure(ta, total / ta), cap, msg=f"T={total} tau_a={ta} vs PLOB")

    def test_relay_matches(self):
        """
        With the encoder's arm lossless the relay rate equals the trusted-heterodyne rate
        over the surviving arm to 1e-4.
        """
        for total in (0.5, 0.1, 0.01):
            best = pure(1.0, total)
            direct = p2p(total)

            self.assertClose(best / direct, 1.0, atol=1e-4, msg=f"T={total} ratio {best / direct:g}")

    def test_relay_hardware(self):
        """
        Worse efficiency, more dark counts and worse mode match each raise `relay_error`.
        """
        for eta in (0.1, 0.5, 1.0):
            errs = [_core.relay_error(1e-3, 1e-3, con, eta, 1e-6) for con in (1.0, 0.9, 0.7, 0.4, 0.0)]

            self.assertMonotone(errs, msg=f"eta={eta}: {errs}")

        darks = [_core.relay_error(1e-3, 1e-3, 0.98, 0.2, d) for d in (0.0, 1e-7, 1e-5)]

        self.assertMonotone(darks, msg=f"dark sweep {darks}")

        etas = [_core.relay_error(1e-3, 1e-3, 0.98, e, 1e-6) for e in (0.05, 0.2, 0.8)]

        self.assertMonotone(etas, rising=False, msg=f"eta sweep {etas}")


class Interference(Question):
    """
    The V < 0.37 zero-key threshold quoted beside these figures is polarization-based
    MDI-QKD's, and is deliberately not wired in.
    """

    def test_overlap_width(self):
        """
        `mode_overlap` falls with width mismatch and offset, is symmetric in the widths
        and even in the offset, and reads 0.8 at a 2:1 ratio.
        """
        overs = [_core.mode_overlap(30e-12, w * 1e-12, 0.0) for w in (30, 45, 60, 120)]

        self.assertMonotone(overs, rising=False, msg=f"width sweep {overs}")
        self.assertClose(overs[2], 0.8, msg=f"2:1 ratio gives {overs[2]:g}")
        self.assertClose(
            _core.mode_overlap(60e-12, 30e-12, 0.0),
            _core.mode_overlap(30e-12, 60e-12, 0.0),
            msg="not symmetric in the two widths",
        )

        slid = [_core.mode_overlap(30e-12, 30e-12, t * 1e-12) for t in (0, 10, 30, 60)]

        self.assertMonotone(slid, rising=False, msg=f"offset sweep {slid}")
        self.assertClose(
            _core.mode_overlap(30e-12, 30e-12, -20e-12),
            _core.mode_overlap(30e-12, 30e-12, 20e-12),
            msg="not even in the offset",
        )
        self.assertLess(_core.mode_overlap(30e-12, 30e-12, 300e-12), 1e-9, msg="10 widths apart is nonzero")

    def test_hom_orders(self):
        """
        The phase-averaged dip from `relay_ports` equals `hom_visibility` at intensity
        overlap = contrast^2, to 1e-7.
        """
        for mu_a, mu_b, con in (
            (1e-4, 1e-4, 1.0),
            (1e-4, 1e-4, 0.8),
            (1e-4, 3e-5, 0.6),
        ):
            want = _core.hom_visibility(con * con, mu_a, mu_b)

            self.assertClose(dip(mu_a, mu_b, con), want, atol=1e-7, msg=f"mu={mu_a} c={con}")

        self.assertClose(dip(1e-4, 1e-4, 0.0), 0.0, atol=1e-12, msg="zero contrast left a dip")

    def test_error_monotone(self):
        """
        `relay_error` rises as the mode match falls, reads (1 - contrast)/2 in the
        weak-flux limit with no fitted constant, and is floored by dark counts.
        """
        errs = [_core.relay_error(1e-4, 1e-4, con, 0.2, 0.0) for con in (1.0, 0.95, 0.8, 0.5, 0.2, 0.0)]

        self.assertMonotone(errs, msg=f"contrast sweep {errs}")

        for con in (0.9, 0.7, 0.5, 0.2):
            self.assertClose(
                _core.relay_error(1e-6, 1e-6, con, 0.2, 0.0),
                (1.0 - con) / 2.0,
                atol=1e-6,
                msg=f"weak-flux limit at contrast={con}",
            )

        floors = [_core.relay_error(1e-3, 1e-3, 1.0, 0.2, d) for d in (1e-8, 1e-6, 1e-4)]

        self.assertMonotone(floors, msg=f"dark floors {floors}")

        for got in floors:
            self.assertGreater(got, 0.0, msg=f"floor {got:g} at perfect contrast")

        weak = _core.relay_error(1e-5, 1e-5, 1.0, 0.2, 1e-6)
        bright = _core.relay_error(1e-2, 1e-2, 1.0, 0.2, 1e-6)

        self.assertLess(bright, weak, msg=f"bright {bright:g} vs weak {weak:g}")


class Adversarial(Guarded):
    """
    The domain ends where lambda vanishes, at (tau_a + tau_b)^2/(tau_a*tau_b).
    """

    def test_audit_modulation(self):
        """
        `cvmdi_point` converges as 1/va to 2.6e-6 at the guard's va = 1e5 ceiling and
        raises past it, where the va^2 blocks put it above the bound.
        """
        bound = _core.cvmdi_rate(0.9, 0.3, _core.cvmdi_noise(0.9, 0.3, 1.0, 1.0, 0.0, 0.0))
        walk = []
        for va in (1e3, 1e4, 1e5):
            point = _core.cvmdi_point(va, 0.9, 0.3, 1.0, 1.0, 0.0, 0.0, 1.0)
            walk.append(abs(point[2] - bound))

        self.assertMonotone(walk, rising=False, msg=f"gaps to the bound {walk}")
        self.assertLess(walk[-1], 5e-6, msg=f"gap at va = 1e5 is {walk[-1]:g}")

        for va in (1e5 + 1.0, 1e6, 3e6, 1e8):
            self.assertBad(
                "va must be <=",
                _core.cvmdi_point,
                (va, 0.9, 0.3, 1.0, 1.0, 0.0, 0.0, 1.0),
                msg=f"va = {va} is past the reduction's useful range",
            )

        self.assertBad(
            "va must be <=",
            _core.cvmdi_cov,
            (3e6, 0.9, 0.9, 1.0, 1.0, 0.0, 0.0),
            msg="cvmdi_cov at va = 3e6",
        )

    def test_audit_domain(self):
        """
        At tau = (0.9, 0.5), omega = 2, g = 1.72 puts chi at 5.696, between the bound
        4.356 and the pure-loss 6.222, and `cvmdi_point` converges onto `cvmdi_rate`.
        """
        args = (0.9, 0.5, 2.0, 2.0)
        chi = _core.cvmdi_noise(*args, 1.72, -1.72)

        self.assertLess(chi, _core.cvmdi_floor(0.9, 0.5), msg=f"chi {chi:g} vs pure loss")
        self.assertGreater(chi, _core.cvmdi_least(0.9, 0.5), msg=f"chi {chi:g} vs the bound")

        keys = [_core.cvmdi_point(va, *args, 1.72, -1.72, 1.0)[2] for va in (1e3, 1e4, 1e5)]
        gaps = [abs(k - _core.cvmdi_rate(0.9, 0.5, chi)) for k in keys]

        self.assertMonotone(gaps, rising=False, msg=f"gaps {gaps}")

        # atol 3e-5 is 1.28x the smallest reachable residual: the gap falls as 2.34/va
        # and va = 1e5 is the guard's ceiling.
        self.assertClose(
            keys[-1],
            _core.cvmdi_rate(0.9, 0.5, chi),
            atol=3e-5,
            msg=f"point {keys[-1]:g} vs rate {_core.cvmdi_rate(0.9, 0.5, chi):g}",
        )

    def test_audit_helps(self):
        """
        A HELPING correlation raises `cvmdi_rate` above pure loss, which is legal: Alice
        and Bob never get to assume the attack was kind.
        """
        args = (0.9, 0.5, 2.0, 2.0)
        plain = _core.cvmdi_rate(0.9, 0.5, _core.cvmdi_noise(*args, 0.0, 0.0))
        helped = _core.cvmdi_rate(0.9, 0.5, _core.cvmdi_noise(*args, 1.72, -1.72))
        clean = pure(0.9, 0.5)

        self.assertGreater(helped, plain, msg=f"helped {helped:g} vs plain {plain:g}")
        self.assertGreater(helped, clean, msg=f"helped {helped:g} vs pure loss {clean:g}")
        self.assertLess(
            clean,
            -math.log2(1.0 - 0.45),
            msg=f"pure loss {clean:g} vs the repeaterless ceiling",
        )

    def test_audit_nan(self):
        """
        chi at 3.9, 3.9999999996 and 4.0 on lossless equal arms raises, a NaN comparing
        false against every bound.
        """
        for chi in (3.9, 3.9999999996, 4.0):
            self.assertBad(
                "noise term",
                _core.cvmdi_rate,
                (1.0, 1.0, chi),
                msg=f"chi = {chi} at lossless equal arms",
            )

        self.assertClose(
            _core.cvmdi_noise(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            4.0,
            msg="the natural composition is off the bound",
        )

        for chi in (4.0000001, 4.5, 6.0, 20.0):
            self.assertFinite(_core.cvmdi_rate(1.0, 1.0, chi), msg=f"chi = {chi} not finite")

        for tb in (1.0, 1.0 - 1e-12, 1.0 - 1e-10):
            self.assertBad(
                "noise term",
                _core.cvmdi_rate,
                (1.0, tb, 4.0),
                msg=f"near-equal arms at tau_b = {tb}",
            )

    def test_audit_quadratures(self):
        """
        Off the bisector chi does not determine the rate: `cvmdi_rate` reports -0.155
        where the per-attack rate is +0.067, never the other way.
        """
        args = (0.95, 0.90, 3.0, 3.0)
        chi = _core.cvmdi_noise(*args, 1.8, 1.8)

        self.assertLess(
            _core.cvmdi_rate(0.95, 0.90, chi),
            0.0,
            msg=f"bound {_core.cvmdi_rate(0.95, 0.90, chi):g}",
        )
        self.assertGreater(
            _core.cvmdi_point(1e5, *args, 1.8, 1.8, 1.0)[2],
            0.0,
            msg=f"point {_core.cvmdi_point(1e5, *args, 1.8, 1.8, 1.0)[2]:g}",
        )

        for g, gp in ((1.8, 1.8), (1.0, -0.5), (0.0, 1.5), (-1.2, 0.3), (0.0, 0.0)):
            worst = _core.cvmdi_rate(0.95, 0.90, _core.cvmdi_noise(*args, g, gp))
            point = _core.cvmdi_point(1e5, *args, g, gp, 1.0)[2]

            self.assertLessEqual(worst, point + 1e-5, msg=f"({g}, {gp}): bound {worst:g} over point {point:g}")


class Finite(Question):
    """
    Papanastasiou, Ottaviani & Pirandola, Phys. Rev. A 96, 042332 (2017).
    """

    def test_excess_bridges(self):
        """
        The (v_q, v_p) of `cvmdi_excess` rebuild chi to 1e-12, and pure loss puts both at
        zero.
        """
        for ta, tb, wa, wb, g, gp in (
            (0.98, 0.5, 1.01, 1.01, 0.0, 0.0),
            (0.6, 0.6, 1.2, 1.5, 0.3, -0.3),
            (0.3, 0.9, 1.05, 1.05, -0.2, 0.2),
        ):
            vq, vp = _core.cvmdi_excess(ta, tb, wa, wb, g, gp)
            want = 2.0 * (ta + tb) / (ta * tb) * math.sqrt((1.0 + vq) * (1.0 + vp))

            self.assertClose(
                _core.cvmdi_noise(ta, tb, wa, wb, g, gp),
                want,
                atol=1e-12,
                msg=f"chi rebuilt from (v_q, v_p) at tau = ({ta}, {tb})",
            )
        clean = _core.cvmdi_excess(0.4, 0.7, 1.0, 1.0, 0.0, 0.0)

        self.assertClose(clean[0], 0.0, atol=1e-15, msg=f"pure loss v_q {clean[0]:g}")
        self.assertClose(clean[1], 0.0, atol=1e-15, msg=f"pure loss v_p {clean[1]:g}")
        self.assertClose(
            _core.cvmdi_noise(0.4, 0.7, 1.0, 1.0, 0.0, 0.0),
            _core.cvmdi_floor(0.4, 0.7),
            atol=1e-12,
            msg="chi is off the pure-loss floor",
        )

    def test_gauss_matches(self):
        """
        `cvmdi_gauss` reproduces `cvmdi_point` to 2e-9 across helping and harming
        correlations.
        """
        for va in (20.0, 1e3):
            for ta, tb, wa, wb, g, gp in (
                (0.98, 0.5, 1.0, 1.0, 0.0, 0.0),
                (0.98, 0.5, 1.01, 1.01, 0.0, 0.0),
                (0.7, 0.7, 1.2, 1.5, 0.3, -0.3),
                (0.7, 0.7, 1.2, 1.5, -0.3, 0.3),
            ):
                vq, vp = _core.cvmdi_excess(ta, tb, wa, wb, g, gp)
                want = _core.cvmdi_point(va, ta, tb, wa, wb, g, gp, 0.98)
                got = _core.cvmdi_gauss(va, ta, tb, vq, vp, 0.98)
                for a, b, tag in zip(got, want, ("i_ab", "chi_e", "key")):
                    self.assertClose(a, b, atol=2e-9, msg=f"{tag} at va = {va}, tau = ({ta}, {tb})")

    def test_gauss_closed(self):
        """
        `cvmdi_gauss` matches the published Eqs. (24) to (26), conditioned and written
        out here, to 1e-11.
        """
        for va, ta, tb, vq, vp in (
            (30.0, 0.9, 0.4, 0.02, 0.02),
            (30.0, 0.9, 0.4, 0.05, -0.01),
            (200.0, 0.5, 0.5, 0.0, 0.0),
        ):
            i_ab, chi_e, key = rebuild(estimated(va, ta, tb, vq, vp), 0.95)
            got = _core.cvmdi_gauss(va, ta, tb, vq, vp, 0.95)

            self.assertClose(got[0], i_ab, atol=1e-11, msg=f"I_AB at va = {va}")
            self.assertClose(got[1], chi_e, atol=1e-11, msg=f"chi_E at va = {va}")
            self.assertClose(got[2], key, atol=1e-11, msg=f"key at va = {va}")

    def test_width_matches(self):
        """
        `cvmdi_width` matches the published Eqs. (14) to (20) to 1e-13 and falls as
        1/sqrt(m).
        """
        args = (60.0, 0.98, 0.5, 0.01, 0.02)
        for m in (1e5, 1e7, 1e9):
            got = _core.cvmdi_width(m, *args)
            want = sigmas(m, *args)
            for a, b, tag in zip(got, want, ("sigma_a", "sigma_b", "s_q", "s_p")):
                self.assertClose(a / b, 1.0, atol=1e-13, msg=f"{tag} at m = {m}")
        near = _core.cvmdi_width(1e6, *args)
        far = _core.cvmdi_width(1e8, *args)
        for a, b, tag in zip(near, far, ("sigma_a", "sigma_b", "s_q", "s_p")):
            self.assertClose(10.0 * b / a, 1.0, atol=1e-12, msg=f"{tag} ratio over two decades {a / b:g}")

    def test_finite_converges(self):
        """
        `cvmdi_finite` climbs with the block size to within 2e-3 of the key rounds' share
        of the asymptotic rate at N = 1e14.
        """
        va, ta, tb = 60.0, 0.98, 0.5
        vq, vp = _core.cvmdi_excess(ta, tb, 1.01, 1.01, -0.1, 0.1)
        limit = 0.5 * _core.cvmdi_gauss(va, ta, tb, vq, vp, 0.95)[2]
        keys = [
            _core.cvmdi_finite(va, ta, tb, vq, vp, 0.95, n, 0.5, 6.5, 1e-10, 1e-10, "gaussian")[0]
            for n in (1e6, 1e8, 1e10, 1e12, 1e14)
        ]

        self.assertMonotone(keys, rising=True, msg=f"keys {keys}")
        self.assertLess(keys[-1], limit, msg=f"key {keys[-1]:g} at or past the asymptote {limit:g}")
        self.assertClose(keys[-1] / limit, 1.0, atol=2e-3, msg=f"ratio at N = 1e14 is {keys[-1] / limit:g}")

    def test_finite_pessimistic(self):
        """
        Over 243 combinations of modulation, arm split and correlation sign the key never
        exceeds the asymptotic rate times the key share.
        """
        seen = 0
        for va in (10.0, 60.0, 1e3):
            for ta, tb in ((0.99, 0.9), (0.8, 0.2), (0.5, 0.05)):
                for w in (1.0, 1.01, 1.1):
                    for sign in (0.0, 0.5, -0.5):
                        g = sign * math.sqrt(max(0.0, w * w - 1.0))
                        vq, vp = _core.cvmdi_excess(ta, tb, w, w, -g, g)
                        limit = 0.5 * _core.cvmdi_gauss(va, ta, tb, vq, vp, 0.95)[2]
                        for n in (1e6, 1e9, 1e12):
                            got = _core.cvmdi_finite(va, ta, tb, vq, vp, 0.95, n, 0.5, 6.5, 1e-10, 1e-10, "gaussian")

                            self.assertLessEqual(got[0], limit, msg=f"va = {va}, tau = ({ta}, {tb}), N = {n}")
                            seen += 1

        self.assertEqual(seen, 243, msg=f"walked {seen} of 243")

    def test_finite_worst(self):
        """
        Eq. (23) takes tau_a, tau_b 6.5 sigma low and v_q, v_p 6.5 sigma high, less the
        two-term Delta(n) over the key rounds.
        """
        va, ta, tb, vq, vp = 60.0, 0.98, 0.5, 0.01, 0.02
        n = 1e9
        out = _core.cvmdi_finite(va, ta, tb, vq, vp, 0.95, n, 0.4, 6.5, 1e-10, 1e-11, "gaussian")
        sa, sb, sq, sp = _core.cvmdi_width(0.4 * n, va, ta, tb, vq, vp)

        self.assertClose(out[1], ta - 6.5 * sa, atol=1e-15, msg="tau_a is taken 6.5 sigma low")
        self.assertClose(out[2], tb - 6.5 * sb, atol=1e-15, msg="tau_b is taken 6.5 sigma low")
        self.assertClose(out[3], vq + 6.5 * sq, atol=1e-15, msg="v_q is taken 6.5 sigma high")
        self.assertClose(out[4], vp + 6.5 * sp, atol=1e-15, msg="v_p is taken 6.5 sigma high")
        self.assertClose(out[6] / n, 0.6, atol=1e-14, msg=f"key-round share {out[6] / n:g}")
        self.assertClose(out[5], penalty(0.6 * n, 1e-10, 1e-11), atol=1e-15, msg=f"Delta(n) = {out[5]:g}")
        inner = _core.cvmdi_gauss(va, out[1], out[2], out[3], out[4], 0.95)[2]

        self.assertClose(out[0], 0.6 * (inner - out[5]), atol=1e-13, msg=f"Eq. (23) key {out[0]:g}")

    def test_finite_unclamped(self):
        """
        A block that cannot distil returns a finite negative number, not zero.
        """
        va, ta, tb = 60.0, 0.5, 0.02
        vq, vp = _core.cvmdi_excess(ta, tb, 1.2, 1.2, 0.0, 0.0)
        got = _core.cvmdi_finite(va, ta, tb, vq, vp, 0.95, 1e6, 0.5, 6.5, 1e-10, 1e-10, "gaussian")

        self.assertLess(got[0], 0.0, msg=f"dead relay reports {got[0]:g}")
        self.assertFinite(got[0], msg="the deficit is not a number")

    def test_finite_extremal(self):
        """
        The paper's two-mode optimum lowers both v_q and v_p, and a thousandth past it is
        not a state.
        """
        for wa, wb in ((1.01, 1.01), (1.05, 1.2), (1.5, 1.05), (2.0, 1.1)):
            g = min(math.sqrt((wa - 1.0) * (wb + 1.0)), math.sqrt((wb - 1.0) * (wa + 1.0)))
            plain = _core.cvmdi_excess(0.9, 0.4, wa, wb, 0.0, 0.0)
            edge = _core.cvmdi_excess(0.9, 0.4, wa, wb, g, -g)

            self.assertLess(edge[0], plain[0], msg=f"v_q at omega = ({wa}, {wb}): {edge[0]:g} vs {plain[0]:g}")
            self.assertLess(edge[1], plain[1], msg=f"v_p at omega = ({wa}, {wb}): {edge[1]:g} vs {plain[1]:g}")
            self.assertFails(
                ValueError,
                "uncertainty principle",
                _core.cvmdi_excess,
                0.9,
                0.4,
                wa,
                wb,
                g * 1.001,
                -g * 1.001,
                msg=f"a thousandth past g at omega = ({wa}, {wb})",
            )

    def test_finite_headline(self):
        """
        At the paper's stated parameters the block sizes it names each give above 1e-2
        bit, a comparison against a sentence and not against a published number.
        """
        w, arm, beta = 1.01, 0.98, 0.98
        g = math.sqrt(w * w - 1.0)
        for db, va, share, n in ((1.0, 100.0, 0.4, 1e6), (3.0, 40.0, 0.65, 1e6), (5.0, 30.0, 0.1, 1e9)):
            tb = 10.0 ** (-db / 10.0)
            vq, vp = _core.cvmdi_excess(arm, tb, w, w, g, -g)
            key = _core.cvmdi_finite(va, arm, tb, vq, vp, beta, n, share, 6.5, 1e-10, 1e-10, "gaussian")[0]

            self.assertGreater(key, 1e-2, msg=f"{db} dB on Bob's arm at N = {n:.0e} gives {key:.4g}")

    def test_finite_classes(self):
        """
        The coherent and collective classes are refused by name and an unrecognised one
        is a ValueError.
        """
        args = (60.0, 0.98, 0.5, 0.01, 0.02, 0.95, 1e9, 0.5, 6.5, 1e-10, 1e-10)

        self.assertFails(
            NotImplementedError,
            "restricted to the case of Gaussian attacks",
            _core.cvmdi_finite,
            *args,
            "coherent",
            msg="coherent accepted",
        )
        self.assertFails(
            NotImplementedError,
            "sits in the ESTIMATORS rather than in the rate",
            _core.cvmdi_finite,
            *args,
            "collective",
            msg="collective accepted",
        )
        self.assertFails(
            ValueError,
            "unknown attack class",
            _core.cvmdi_finite,
            *args,
            "two-mode",
            msg="two-mode accepted",
        )


class Guards(Guarded):
    """
    Out-of-domain relay inputs raise.
    """

    def test_guard_arms(self):
        """
        Transmissivities outside (0, 1] and a chi where the noise term vanishes raise,
        while chi below the PURE-LOSS value 8 is legal and better.
        """
        for bad in (0.0, -0.1, 1.5, float("nan")):
            self.assertBad(
                "tau_a must be",
                _core.cvmdi_rate,
                (bad, 0.5, 20.0),
                msg=f"tau_a = {bad}",
            )
            self.assertBad(
                "tau_b must be",
                _core.cvmdi_rate,
                (0.5, bad, 20.0),
                msg=f"tau_b = {bad}",
            )

        self.assertBad(
            "where these arms' noise term",
            _core.cvmdi_rate,
            (0.5, 0.5, 4.0),
            msg="chi = 4 at equal arms",
        )
        self.assertClose(_core.cvmdi_least(0.5, 0.5), 4.0, msg=f"least {_core.cvmdi_least(0.5, 0.5):g}")
        self.assertClose(_core.cvmdi_floor(0.5, 0.5), 8.0, msg=f"floor {_core.cvmdi_floor(0.5, 0.5):g}")
        self.assertGreater(
            _core.cvmdi_rate(0.5, 0.5, 7.0),
            _core.cvmdi_rate(0.5, 0.5, 8.0),
            msg="chi = 7 not above chi = 8",
        )

    def test_guard_environment(self):
        """
        Sub-vacuum environment variances and correlations past the positivity cap or the
        symplectic bound raise.
        """
        for bad in (0.0, 0.99, -1.0):
            self.assertBad(
                "omega_a is an environment variance",
                _core.cvmdi_cov,
                (5.0, 0.5, 0.5, bad, 1.0, 0.0, 0.0),
                msg=f"omega_a = {bad}",
            )

        self.assertBad(
            "must be < sqrt",
            _core.cvmdi_cov,
            (5.0, 0.5, 0.5, 1.5, 1.5, 2.0, 0.0),
            msg="correlation past the positivity cap",
        )
        self.assertBad(
            "uncertainty principle",
            _core.cvmdi_cov,
            (5.0, 0.5, 0.5, 1.2, 1.2, 1.0, -1.0),
            msg="correlation past the symplectic bound",
        )

    def test_guard_estimated(self):
        """
        A v_q or v_p past the per-quadrature edge, a NaN, a non-positive sample, a zero
        confidence coefficient and a block with no key rounds each raise.
        """
        for bad in (-0.35, -2.0):
            self.assertBad(
                "where these arms' noise term vanishes",
                _core.cvmdi_gauss,
                (60.0, 0.9, 0.4, bad, 0.0, 0.95),
                msg=f"v_q = {bad}",
            )
            self.assertBad(
                "cvmdi_least's edge per quadrature",
                _core.cvmdi_width,
                (1e6, 60.0, 0.9, 0.4, 0.0, bad),
                msg=f"v_p = {bad}",
            )

        self.assertBad(
            "v_q must be finite",
            _core.cvmdi_gauss,
            (60.0, 0.9, 0.4, float("nan"), 0.0, 0.95),
            msg="v_q = nan",
        )
        self.assertBad(
            "m must be > 0",
            _core.cvmdi_width,
            (0.0, 60.0, 0.9, 0.4, 0.0, 0.0),
            msg="m = 0",
        )
        self.assertBad(
            "POP17's split needs both sides positive",
            _core.cvmdi_finite,
            (60.0, 0.9, 0.4, 0.01, 0.01, 0.95, 1e-300, 1e-300, 6.5, 1e-10, 1e-10, "gaussian"),
            msg="underflowed block size",
        )
        self.assertBad(
            "conf must be > 0",
            _core.cvmdi_finite,
            (60.0, 0.9, 0.4, 0.01, 0.01, 0.95, 1e6, 0.5, 0.0, 1e-10, 1e-10, "gaussian"),
            msg="conf = 0",
        )
        self.assertBad(
            "no two-mode Gaussian environment drives",
            _core.cvmdi_finite,
            (60.0, 0.9, 0.4, -0.9, -0.9, 0.95, 1e6, 0.5, 6.5, 1e-10, 1e-10, "gaussian"),
            msg="v pair past the edge",
        )

    def test_guard_omega(self):
        """
        `relay_omega(1.0, 0.01)` raises on a lossless arm, `relay_omega(1.0, 0.0)` is 1.0,
        and a negative xi raises.
        """

        self.assertFails(ValueError, "no environment", _core.relay_omega, 1.0, 0.01, msg="tau = 1")
        self.assertClose(_core.relay_omega(1.0, 0.0), 1.0, msg=f"omega {_core.relay_omega(1.0, 0.0):g}")
        self.assertFails(ValueError, "xi must be", _core.relay_omega, 0.5, -0.01, msg="negative xi")

    def test_guard_optics(self):
        """
        Overlaps, contrasts, efficiencies, pulse widths and a dark rate of 1 each raise
        naming the offender.
        """
        for bad in (-0.1, 1.1, float("inf")):
            self.assertBad(
                "contrast must be",
                _core.relay_ports,
                (0.1, 0.1, 0.0, bad),
                msg=f"contrast = {bad}",
            )
            self.assertBad(
                "overlap must be",
                _core.hom_visibility,
                (bad, 0.1, 0.1),
                msg=f"overlap = {bad}",
            )

        for bad in (0.0, -1.0):
            self.assertBad(
                "fwhm_a must be",
                _core.mode_overlap,
                (bad, 1.0, 0.0),
                msg=f"fwhm_a = {bad}",
            )

        self.assertBad(
            "eta must be",
            _core.relay_counts,
            (0.1, 0.1, 1.5, 0.0),
            msg="eta above 1",
        )
        self.assertBad(
            "dark must be",
            _core.relay_counts,
            (0.1, 0.1, 0.5, 1.0),
            msg="dark rate 1",
        )

    def test_guard_limits(self):
        """
        A nearly dead arm and a large modulation stay finite, and a saturated relay
        detector reads 0.5.
        """
        for tb in (1e-9, 1e-6, 1e-3):
            self.assertFinite(pure(1.0, tb), msg=f"tau_b={tb}")

        self.assertFinite(
            _core.cvmdi_point(1e5, 0.9, 0.4, 1.2, 1.2, 0.0, 0.0, 0.95),
            msg="va = 1e5 is not finite",
        )
        self.assertClose(
            _core.relay_error(1e6, 1e6, 0.5, 1.0, 0.0),
            0.5,
            msg=f"saturated relay reads {_core.relay_error(1e6, 1e6, 0.5, 1.0, 0.0):g}",
        )


if __name__ == "__main__":
    rc = Exam(
        "RelayConditioned",
        "CV-MDI: the two-party state the untrusted Bell relay leaves behind",
        "relay_conditioned.md",
    ).run(load(Conditioned))
    rc |= Exam(
        "RelayPosition",
        "Where the relay sits: the asymmetry anchors and the pure-loss zeroes",
        "relay_position.md",
    ).run(load(Position))
    rc |= Exam(
        "RelayUntrusted",
        "An untrusted relay never beats a trusted link with the same hardware",
        "relay_untrusted.md",
    ).run(load(Untrusted))
    rc |= Exam(
        "RelayInterference",
        "Two-source interference: mode overlap, HOM ceiling, error from contrast",
        "relay_interference.md",
    ).run(load(Interference))
    rc |= Exam(
        "RelayFinite",
        "Finite-size CV-MDI: estimated coordinates, the four widths, and Eq. (23)",
        "relay_finite.md",
    ).run(load(Finite))
    rc |= Exam(
        "RelayAdversarial",
        "Regressions from the adversarial audit, on its own reproducers",
        "relay_adversarial.md",
    ).run(load(Adversarial))
    rc |= Exam(
        "RelayGuards",
        "Out-of-domain relay inputs raise rather than returning a wrong number",
        "relay_guards.md",
    ).run(load(Guards))
    sys.exit(rc)
