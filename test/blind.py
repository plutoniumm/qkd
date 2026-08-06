import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.forms import bisect
from qkd import _core

# THIS FILE REIMPLEMENTS THE ENGINES FROM THE PAPERS, INDEPENDENTLY: h2,
# tamaki_coef, the GLLP and Koashi forms and the Pironio rate below are typed
# from the sources so that a drift in src/ fails here instead of agreeing with
# itself. Nothing here may be folded onto kit/, and tamaki_coef stays
# byte-identical to test/flaws.py's rather than shared -- no test/*.py imports
# another, so the only landing site is kit/, which defeats the charter.
# kit.forms.bisect is imported for death_db, whose objective is continuous; the
# two inline bisections below bisect a BOOLEAN predicate, which is not that
# function's contract.

PD = 1.0e-7
F_EC = 1.16
Q_SIFT = 0.5
ROOT2 = math.sqrt(2.0)
TSIRELSON = 2.0 * ROOT2


def h2(x):
    """
    Binary Shannon entropy in bits, zero at the closed endpoints.
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0

    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def eta_of(db):
    return 10.0 ** (-db / 10.0)


def my_angles(d):
    """
    Alice's four Bloch polar angles, derived from Tamaki App. C. The single-photon part of |e^(i
    xi) sqrt(a)>_r |e^(i(xi+p)) sqrt(a)>_s is (|10> + e^(ip)|01>)/sqrt2 with p = theta +
    d*theta/pi. On App. C's own |0z> = (|0y>+|1y>)/sqrt2, |1z> = (-i|0y>+i|1y>)/sqrt2 that is
    e^(ip/2)[cos(p/2)|0z> + sin(p/2)|1z>], so the Bloch polar angle IS the modulated phase p.
    Order (0Z, 1Z, 0X, 1X) <- theta in {0, pi, pi/2, 3pi/2}.
    """
    return tuple(t * (1.0 + d / math.pi) for t in (0.0, math.pi, math.pi / 2.0, 3.0 * math.pi / 2.0))


def bloch(t):
    return (math.sin(t), math.cos(t))


def my_triangle(d):
    """
    Twice the signed area of the triangle on the three states Alice sends. det of the rows (1,
    px, pz) for 0Z, 0X, 1Z is twice the signed area, which for three points on the unit circle
    at angles a, b, c is sin(b-a) + sin(c-b) + sin(a-c). With a = 0, b = pi/2 + d/2, c = pi + d
    that is 2 cos(d/2) + sin(d).
    """
    a, b, e = 0.0, math.pi / 2.0 + d / 2.0, math.pi + d

    return abs(math.sin(b - a) + math.sin(e - b) + math.sin(a - e))


def my_virtual(d):
    """
    Priors and Bloch angles of the two virtual X-basis states. Alice's X measurement on
    (|0z>|phi_0z> + |1z>|phi_1z>)/sqrt2 leaves B in |phi_0z> +/- |phi_1z>, unnormalised, whose
    half-norms are the priors. With |phi_1z> at pi + d, i.e. -sin(d/2)|0z> + cos(d/2)|1z>, the
    two are exactly orthogonal, since (u+v).(u-v) = |u|^2 - |v|^2 = 0 for unit u, v.
    """
    s, k = math.sin(d / 2.0), math.cos(d / 2.0)
    plus, minus = (1.0 - s, k), (1.0 + s, -k)

    def prior(v):
        return (v[0] ** 2 + v[1] ** 2) / 4.0

    def polar(v):
        n = v[0] ** 2 + v[1] ** 2

        return math.atan2(2.0 * v[0] * v[1] / n, (v[0] ** 2 - v[1] ** 2) / n) % (2.0 * math.pi)

    return (prior(plus), polar(plus), prior(minus), polar(minus))


def my_fidelity(d):
    """
    Uhlmann fidelity Tr|sqrt(rho_Z) sqrt(rho_X)| between Alice's two bases. For qubits with
    Bloch vectors a and b this is sqrt([1 + a.b + sqrt((1-|a|^2)(1-|b|^2))]/2).
    """
    a0, a1, x0, x1 = my_angles(d)
    rz = tuple((u + v) / 2.0 for u, v in zip(bloch(a0), bloch(a1)))
    rx = tuple((u + v) / 2.0 for u, v in zip(bloch(x0), bloch(x1)))
    dot = rz[0] * rx[0] + rz[1] * rx[1]
    nz = rz[0] ** 2 + rz[1] ** 2
    nx = rx[0] ** 2 + rx[1] ** 2

    return math.sqrt((1.0 + dot + math.sqrt(max(0.0, (1.0 - nz) * (1.0 - nx)))) / 2.0)


def my_intrinsic(d, t):
    """
    Z-basis error with no dark counts: |phi_0z> at 0, |phi_1z> at pi + d, Bob's analyser rotated
    by t, so the two errors are sin^2(t/2) and sin^2((d-t)/2), averaged over the two equally
    likely bits.
    """
    return (math.sin(t / 2.0) ** 2 + math.sin((d - t) / 2.0) ** 2) / 2.0


def my_yield(eta, pd):
    """
    Single-photon yield, Pereira, Curty & Tamaki, Sec. V: Y_Z = 4(1-eta/2)p_d + eta.
    """
    return eta + 4.0 * (1.0 - eta / 2.0) * pd


def pereira_mix(ebar, eta, pd):
    """
    Pereira Eq. (biterror) with (cos 2d + cos d)/2 carried as 1 - 2*ebar.
    """
    num = 2.0 * (1.0 - eta / 2.0) * pd + eta / 2.0 + (eta / 2.0) * (1.0 - 2.0 * ebar) * (pd - 1.0)

    return num / my_yield(eta, pd)


def my_channel(d, eta, pd, t):
    """
    The (yield, bit error) pair the two analyses are both fed.
    """
    return my_yield(eta, pd), pereira_mix(my_intrinsic(d, t), eta, pd)


def my_coin(d, y1):
    """
    Quantum coin imbalance, Tamaki, Lo, Fung & Qi Eqs. (fDelta), (fidDelta): Delta_ini = (1-F)/2
    for a single flawed sender, enhanced by loss to Delta_ini/Y_1, Eve being free to attribute
    every loss event to the coin.
    """
    return min(0.5, (1.0 - my_fidelity(d)) / (2.0 * y1))


def my_gllp(e_bit, coin):
    """
    Phase error from the Bloch-sphere bound sqrt(e e') + sqrt((1-e)(1-e')) >= 1-2D. e = sin^2(b)
    and e' = sin^2(a) turn the left side into cos(a-b), so the bound is |a-b| <= arccos(1-2D) =
    2 arcsin(sqrt(D)) and the largest admissible phase error is sin^2(arcsin(sqrt(e)) + 2
    arcsin(sqrt(D))).
    """
    return min(
        0.5,
        math.sin(math.asin(math.sqrt(e_bit)) + 2.0 * math.asin(math.sqrt(coin))) ** 2,
    )


def gllp_quartic(e_bit, coin):
    """
    The same bound in the form GLLP print, expanded from the sin^2 above.
    """
    root = math.sqrt(coin * (1.0 - coin) * e_bit * (1.0 - e_bit))
    val = e_bit + 4.0 * coin * (1.0 - coin) * (1.0 - 2.0 * e_bit) + 4.0 * (1.0 - 2.0 * coin) * root

    return min(0.5, val)


def tamaki_coef(g):
    """
    Tamaki App. C's virtual-state coefficients C_00, C_10, C_01, C_11 at flaw g.
    """
    s, k = math.sin(g / 2.0), math.cos(g / 2.0)

    return (
        (1.0 + s + k) / (2.0 * math.sqrt(1.0 + s)),
        (1.0 + s - k) / (2.0 * math.sqrt(1.0 + s)),
        (1.0 - s - k) / (2.0 * math.sqrt(1.0 - s)),
        (1.0 - s + k) / (2.0 * math.sqrt(1.0 - s)),
    )


def tamaki_phase(d):
    """
    The loss-tolerant phase error Tamaki App. C's own Eq. (channel) produces at zero dark count
    and zero loss: [C_10(3d/2)^2 + C_01(3d/2)^2]/2, the 3d/2 being their substitution for Bob's
    own modulator. Equals sin^2(3d/8).
    """
    _, c10, c01, _ = tamaki_coef(1.5 * d)

    return (c10**2 + c01**2) / 2.0


def my_tolerant(d, eta, pd, t, f_ec, q):
    """
    Loss-tolerant rate: the flaw rotates Alice's virtual X frame and Bob's X analyser together,
    so only the tilt survives into the phase error.
    """
    y1, e_bit = my_channel(d, eta, pd, t)
    phase = pereira_mix(math.sin(t / 2.0) ** 2, eta, pd)

    return q * y1 * (1.0 - h2(phase) - f_ec * h2(e_bit)), y1, e_bit, phase


def my_standard(d, eta, pd, t, f_ec, q):
    """
    Standard (GLLP) rate: the same channel, but the phase error is bought from the bit error
    through the loss-enhanced quantum coin.
    """
    y1, e_bit = my_channel(d, eta, pd, t)
    coin = my_coin(d, y1)

    return q * y1 * (1.0 - h2(my_gllp(e_bit, coin)) - f_ec * h2(e_bit)), y1, e_bit, coin


def death_db(fn, d, pd=PD, f_ec=F_EC, t=0.0, lo=1.0, hi=90.0):
    """
    The loss in dB at which a rate stops being positive, by bisection.
    """

    return bisect(lambda db: fn(d, eta_of(db), pd, t, f_ec, Q_SIFT)[0], lo, hi)


def pironio(q, s, f_ec=1.0):
    """
    Pironio, Acin, Brunner, Gisin, Massar & Scarani Eq. (keyrate): r >= 1 - h(Q) - h((1 +
    sqrt((S/2)^2 - 1))/2).
    """
    return 1.0 - f_ec * h2(q) - h2((1.0 + math.sqrt((s / 2.0) ** 2 - 1.0)) / 2.0)


def s_eff(eta):
    """
    CHSH under Pironio Fig. 3's detector model, a missing click read as -1. Both detectors fire
    with probability eta^2, giving the ideal correlator; a lone click contributes nothing, the
    ideal conditional marginal being zero; a double miss contributes +1 with weight (1-eta)^2,
    and the four CHSH coefficients sum to 2.
    """
    return TSIRELSON * eta**2 + 2.0 * (1.0 - eta) ** 2


def q_eff(eta):
    """
    QBER under that same model: the key pair is perfectly correlated when both fire and when
    neither does, and coin-flip wrong when exactly one does.
    """
    return eta * (1.0 - eta)


class Geometry(Question):
    def test_angles(self):
        """
        Alice's four Bloch angles are the modulated phases theta*(1 + delta/pi).
        """
        for d in (0.0, 0.063, 0.2, 1.0, math.pi / 3.0):
            got, want = _core.flaw_angles(d), my_angles(d)

            for a, b in zip(got, want):
                self.assertClose(a, b, atol=1e-14, msg=f"angle at delta={d}")

    def test_triangle(self):
        """
        The triangle determinant equals 2 cos(delta/2) + sin(delta) in closed form.
        """
        for d in (0.0, 0.063, 0.5, 1.0, 2.0, 3.0):
            closed = 2.0 * math.cos(d / 2.0) + math.sin(d)
            self.assertClose(_core.flaw_triangle(d), closed, atol=1e-14, msg=f"delta={d}")
            self.assertClose(my_triangle(d), closed, atol=1e-14, msg=f"det vs closed at {d}")

    def test_extremes(self):
        """
        The triangle is 2 at zero flaw, peaks at 3 sqrt(3)/2 where the three states are
        equilateral, and vanishes as the flaw reaches pi.
        """
        self.assertClose(_core.flaw_triangle(0.0), 2.0, atol=1e-14, msg="zero flaw")
        self.assertClose(
            _core.flaw_triangle(math.pi / 3.0),
            3.0 * math.sqrt(3.0) / 2.0,
            atol=1e-14,
            msg="equilateral at pi/3",
        )

        peak = max((_core.flaw_triangle(3.0 * i / 2000.0), i) for i in range(2000))
        self.assertClose(peak[1] * 3.0 / 2000.0, math.pi / 3.0, atol=2e-3, msg="argmax is pi/3")
        self.assertClose(_core.flaw_triangle(math.pi - 1e-9), 0.0, atol=1e-8, msg="limit at pi")
        self.assertFails(ValueError, "pi", _core.flaw_triangle, math.pi, msg="degenerate at pi")

    def test_virtual(self):
        """
        The two virtual X-basis states carry priors (1 -+ sin(delta/2))/2 and are exactly
        antipodal.
        """
        for d in (0.0, 0.063, 0.4, 1.0):
            got, want = _core.flaw_virtual(d), my_virtual(d)

            for a, b in zip(got, want):
                self.assertClose(a, b, atol=1e-14, msg=f"virtual at delta={d}")
            self.assertClose(
                (got[3] - got[1]) % (2.0 * math.pi),
                math.pi,
                atol=1e-14,
                msg=f"antipodal at delta={d}",
            )

    def test_fidelity(self):
        """
        The two-basis fidelity is the Uhlmann fidelity of the Bloch vectors, and is exactly one
        at zero flaw.
        """
        self.assertClose(_core.flaw_fidelity(0.0), 1.0, atol=1e-15, msg="F = 1 at zero flaw")

        for d in (0.0, 0.063, 0.2, 0.5, 1.0):
            self.assertClose(_core.flaw_fidelity(d), my_fidelity(d), atol=1e-14, msg=f"delta={d}")


class Channel(Question):
    def test_yield(self):
        """
        The single-photon yield is Pereira's Y_Z = 4(1 - eta/2) p_d + eta.
        """
        for eta in (1.0, 0.8, 0.5, 0.1, 1e-3):
            for pd in (1e-7, 1e-5, 1e-3, 1e-2):
                y1, _ = _core.flaw_channel(0.0, eta, pd, 0.0)
                self.assertClose(y1, my_yield(eta, pd), atol=1e-15, msg=f"eta={eta} pd={pd}")

    def test_excess(self):
        """
        At unit efficiency the published yield exceeds one by exactly twice the dark count: a
        defect quoted from the paper rather than repaired.
        """
        for pd in (1e-7, 1e-5, 1e-3, 1e-2, 0.05):
            y1, _ = _core.flaw_channel(0.0, 1.0, pd, 0.0)
            self.assertClose(y1 - 1.0, 2.0 * pd, atol=1e-15, msg=f"excess at pd={pd}")
        self.assertFails(
            ValueError,
            "q_mu",
            _core.bb84_rate,
            Q_SIFT,
            1.0 + 2e-7,
            0.0,
            1.0 + 2e-7,
            0.0,
            F_EC,
            msg="the repo's own BB84 entry point rejects that yield",
        )

    def test_biterror(self):
        """
        At zero flaw the bit error is Pereira's printed e_Z verbatim.
        """
        for eta in (1.0, 0.5, 0.1, 0.01):
            for pd in (1e-7, 1e-3, 1e-2):
                num = 2.0 * (1.0 - eta / 2.0) * pd + eta / 2.0 + (eta / 2.0) * (pd - 1.0)
                _, e_bit = _core.flaw_channel(0.0, eta, pd, 0.0)
                self.assertClose(e_bit, num / my_yield(eta, pd), atol=1e-15, msg=f"eta={eta} pd={pd}")

    def test_intrinsic(self):
        """
        With dark counts off, the bit error is the averaged misalignment of Alice's two Z states
        against Bob's rotated analyser.
        """
        for d in (0.0, 0.063, 0.4, 0.8, 1.5):
            for t in (0.0, 0.01, 0.05, 0.2):
                _, e_bit = _core.flaw_channel(d, 1.0, 0.0, t)
                self.assertClose(e_bit, my_intrinsic(d, t), atol=1e-15, msg=f"delta={d} tilt={t}")

    def test_dark(self):
        """
        Flaw and dark counts combine through Pereira's expression with the flaw carried as one
        minus twice the intrinsic error.
        """
        for d in (0.0, 0.063, 0.4):
            for eta in (1.0, 0.3):
                for pd in (1e-7, 1e-3):
                    _, e_bit = _core.flaw_channel(d, eta, pd, 0.0)
                    want = pereira_mix(my_intrinsic(d, 0.0), eta, pd)
                    self.assertClose(e_bit, want, atol=1e-15, msg=f"d={d} eta={eta} pd={pd}")


class Estimator(Question):
    def test_coin(self):
        """
        The coin imbalance is (1 - F)/(2 Y1), capped at one half.
        """
        for d in (0.0, 0.063, 0.4):
            for y1 in (1.0, 0.5, 0.01, 1e-4, 1e-6):
                self.assertClose(
                    _core.flaw_coin(d, y1),
                    my_coin(d, y1),
                    atol=1e-15,
                    msg=f"d={d} y1={y1}",
                )

    def test_gllp(self):
        """
        The standard phase error is the Bloch-sphere bound, in both of its equivalent closed
        forms.
        """
        for e_bit in (0.0, 0.01, 0.05, 0.2):
            for coin in (0.0, 1e-4, 0.01, 0.05, 0.2, 0.5):
                got = _core.flaw_gllp(e_bit, coin)
                self.assertClose(got, my_gllp(e_bit, coin), atol=1e-14, msg="sin^2 form")
                self.assertClose(got, gllp_quartic(e_bit, coin), atol=1e-14, msg="GLLP form")

    def test_direct(self):
        """
        The loss-tolerant phase error agrees with measuring the two virtual states directly,
        which is what makes it exact rather than a bound.
        """
        for d in (0.0, 0.063, 0.4, 1.0, 2.0):
            for eta in (1.0, 0.1, 1e-3):
                self.assertClose(
                    _core.flaw_phase(d, eta, PD, 0.0),
                    _core.flaw_direct(d, eta, PD, 0.0),
                    atol=1e-15,
                    msg=f"d={d} eta={eta}",
                )

    def test_flat(self):
        """
        The loss-tolerant phase error carries no flaw dependence, only the tilt: the flaw turns
        Alice's virtual frame and Bob's analyser together.
        """
        base = _core.flaw_phase(0.0, 1.0, PD, 0.0)

        for d in (0.0, 1e-6, 0.063, 0.5, 1.0, 2.0, 3.0):
            self.assertClose(_core.flaw_phase(d, 1.0, PD, 0.0), base, atol=1e-15, msg=f"d={d}")

        for t in (0.0, 0.01, 0.05, 0.2):
            self.assertClose(
                _core.flaw_phase(0.4, 1.0, 0.0, t),
                math.sin(t / 2.0) ** 2,
                atol=1e-15,
                msg=f"tilt={t}",
            )

    def test_divergence(self):
        """
        Tamaki's own simulated channel gives a flaw-dependent loss-tolerant phase error of
        sin^2(3 delta/8), which this engine does not reproduce.
        """
        for d in (0.0, 0.063, 0.126, 0.4, 1.0):
            self.assertClose(
                tamaki_phase(d),
                math.sin(3.0 * d / 8.0) ** 2,
                atol=1e-14,
                msg=f"closed form {d}",
            )
            self.assertClose(_core.flaw_phase(d, 1.0, 0.0, 0.0), 0.0, atol=1e-15, msg=f"core {d}")
        self.assertGreater(tamaki_phase(0.063), 5e-4, msg="the paper's model is not flaw-free")
        self.assertGreater(
            tamaki_phase(0.063),
            _core.flaw_channel(0.063, 1.0, 0.0, 0.0)[1],
            msg="and is not negligible beside the bit error",
        )

    def test_zerobit(self):
        """
        At zero flaw and zero tilt the phase error equals the bit error exactly.
        """
        for eta in (1.0, 0.5, 0.01, 1e-4):
            for pd in (1e-7, 1e-3):
                _, e_bit = _core.flaw_channel(0.0, eta, pd, 0.0)
                self.assertClose(
                    _core.flaw_phase(0.0, eta, pd, 0.0),
                    e_bit,
                    atol=2e-16,
                    msg=f"eta={eta} pd={pd}",
                )


class Rates(Question):
    def test_tolerant(self):
        """
        The loss-tolerant rate reproduces an independently assembled sift * yield * (1 - the two
        entropies).
        """
        for d in (0.0, 0.063, 0.4):
            for db in (5.0, 20.0, 40.0):
                got = _core.flaw_tolerant(d, eta_of(db), PD, 0.0, F_EC, Q_SIFT)
                want = my_tolerant(d, eta_of(db), PD, 0.0, F_EC, Q_SIFT)
                self.assertClose(got[0], max(0.0, want[0]), atol=1e-15, msg="rate")

                for a, b in zip(got[1:], want[1:]):
                    self.assertClose(a, b, atol=1e-15, msg="slot")

    def test_standard(self):
        """
        The standard rate reproduces the same assembly with the GLLP phase error in place of the
        loss-tolerant one.
        """
        for d in (0.0, 0.063, 0.4):
            for db in (5.0, 20.0):
                got = _core.flaw_standard(d, eta_of(db), PD, 0.0, F_EC, Q_SIFT)
                want = my_standard(d, eta_of(db), PD, 0.0, F_EC, Q_SIFT)
                self.assertClose(got[0], max(0.0, want[0]), atol=1e-15, msg="rate")

                for a, b in zip(got[1:], want[1:]):
                    self.assertClose(a, b, atol=1e-15, msg="slot")

    def test_bb84(self):
        """
        At zero flaw both analyses collapse onto the plain single-photon BB84 rate.
        """
        for db in (1.0, 5.0, 20.0, 40.0, 55.0):
            y1, e_bit = _core.flaw_channel(0.0, eta_of(db), PD, 0.0)
            want = _core.bb84_rate(Q_SIFT, y1, e_bit, y1, e_bit, F_EC)
            tol = _core.flaw_tolerant(0.0, eta_of(db), PD, 0.0, F_EC, Q_SIFT)[0]
            std = _core.flaw_standard(0.0, eta_of(db), PD, 0.0, F_EC, Q_SIFT)[0]
            self.assertLessEqual(abs(tol - want), 2e-15 * max(want, 1e-300), msg="tolerant")
            self.assertLessEqual(abs(std - want), 2e-15 * max(want, 1e-300), msg="standard")

    def test_slots(self):
        """
        The fourth return slot is a phase error for one analysis and a coin imbalance for the
        other, and neither is silently the other.
        """
        for d in (0.0, 0.063, 0.3):
            for db in (10.0, 30.0):
                eta = eta_of(db)
                tol = _core.flaw_tolerant(d, eta, PD, 0.0, F_EC, Q_SIFT)
                std = _core.flaw_standard(d, eta, PD, 0.0, F_EC, Q_SIFT)
                self.assertClose(
                    tol[3],
                    _core.flaw_phase(d, eta, PD, 0.0),
                    atol=1e-15,
                    msg="tolerant slot",
                )
                self.assertClose(std[3], _core.flaw_coin(d, tol[1]), atol=1e-15, msg="standard slot")

        far = _core.flaw_standard(0.3, eta_of(30.0), PD, 0.0, F_EC, Q_SIFT)
        self.assertGreater(
            far[3],
            100.0 * _core.flaw_phase(0.3, eta_of(30.0), PD, 0.0),
            msg="the coin dwarfs the loss-tolerant phase error",
        )

    def test_gap(self):
        """
        The loss-tolerant rate never falls below the standard one, and is strictly above it once
        the flaw is non-zero.
        """
        for d in (0.01, 0.063, 0.2):
            for db in (10.0, 25.0, 40.0):
                eta = eta_of(db)
                tol = _core.flaw_tolerant(d, eta, PD, 0.0, F_EC, Q_SIFT)[0]
                std = _core.flaw_standard(d, eta, PD, 0.0, F_EC, Q_SIFT)[0]
                self.assertGreater(tol, std, msg=f"d={d} db={db}")

    def test_death(self):
        """
        Both analyses die at the same loss at zero flaw; with a flaw the loss-tolerant one
        survives far past the standard one.
        """
        zero_t = death_db(_core.flaw_tolerant, 0.0)
        zero_s = death_db(_core.flaw_standard, 0.0)
        self.assertClose(zero_t, death_db(my_tolerant, 0.0), atol=1e-9, msg="tolerant vs derived")
        self.assertClose(zero_s, death_db(my_standard, 0.0), atol=1e-9, msg="standard vs derived")
        self.assertClose(zero_t, zero_s, atol=1e-9, msg="the two agree at zero flaw")
        self.assertClose(zero_t, 57.855244, atol=1e-5, msg="zero-flaw death in dB")

        flaw_t = death_db(_core.flaw_tolerant, 0.063)
        flaw_s = death_db(_core.flaw_standard, 0.063)
        self.assertClose(flaw_t, death_db(my_tolerant, 0.063), atol=1e-9, msg="tolerant at 0.063")
        self.assertClose(flaw_s, death_db(my_standard, 0.063), atol=1e-9, msg="standard at 0.063")
        self.assertClose(flaw_t, 57.843443, atol=1e-5, msg="tolerant death in dB")
        self.assertClose(flaw_s, 29.659930, atol=1e-5, msg="standard death in dB")
        self.assertClose(flaw_t - flaw_s, 28.183513, atol=1e-5, msg="the gap in dB")

    def test_alive(self):
        """
        Both rates are strictly positive at the two losses an earlier report named as their
        death points, which rules those two figures out.
        """
        self.assertGreater(
            _core.flaw_tolerant(0.063, eta_of(57.775328), PD, 0.0, F_EC, Q_SIFT)[0],
            0.0,
            msg="tolerant is alive at the reported death",
        )
        self.assertGreater(
            _core.flaw_standard(0.063, eta_of(28.766343), PD, 0.0, F_EC, Q_SIFT)[0],
            1e-6,
            msg="standard is alive at the reported death",
        )


class Ekert(Question):
    def test_rate(self):
        """
        The rate is Pironio's bound: 1 - h2(Q) - the Holevo term in the CHSH value.
        """
        for s in (TSIRELSON, 2.7, 2.4, 2.1, 2.01):
            for q in (0.0, 0.02, 0.05):
                want = pironio(q, s, 1.0)
                got = _core.ekert_rate(1.0, q, s, 1.0, 1.0, "measured")
                self.assertClose(got, max(0.0, want), atol=1e-15, msg=f"S={s} Q={q}")

    def test_qber(self):
        """
        The depolarised-singlet error map inverts S = 2 sqrt(2) (1 - 2Q).
        """
        for s in (TSIRELSON, 2.7, 2.4, 2.0):
            self.assertClose(
                _core.ekert_qber(s),
                (1.0 - s / TSIRELSON) / 2.0,
                atol=1e-15,
                msg=f"S={s}",
            )

    def test_threshold(self):
        """
        The error rate at which the priced rate vanishes is Pironio's quoted 7.1% at a perfect
        reconciliation.
        """
        for f_ec in (1.0, 1.1, 1.16, 1.22):
            lo, hi = 0.0, 0.5
            for _ in range(200):
                mid = (lo + hi) / 2.0
                s = TSIRELSON * (1.0 - 2.0 * mid)
                if s > 2.0 and pironio(mid, s, f_ec) > 0.0:
                    lo = mid
                else:
                    hi = mid
            self.assertClose(
                _core.ekert_threshold(f_ec),
                (lo + hi) / 2.0,
                atol=1e-12,
                msg=f"f_ec={f_ec}",
            )
        self.assertClose(_core.ekert_threshold(1.0), 0.0714917588, atol=1e-9, msg="derived")
        self.assertClose(_core.ekert_threshold(1.0), 0.071, atol=5e-4, msg="paper's 7.1 percent")

    def test_domain(self):
        """
        No violation buys no key, past Tsirelson raises, and neither a modelled nor a
        device-independent provenance is accepted.
        """
        for s in (2.0, 1.999, 1.0, 0.0):
            self.assertClose(
                _core.ekert_rate(1.0, 0.0, s, 1.0, 1.0, "measured"),
                0.0,
                atol=0.0,
                msg=f"S={s}",
            )
        self.assertGreater(
            _core.ekert_rate(1.0, 0.0, 2.0000001, 1.0, 1.0, "measured"),
            0.0,
            msg="any violation buys some key",
        )
        self.assertFails(
            ValueError,
            "Tsirelson",
            _core.ekert_rate,
            1.0,
            0.0,
            2.9,
            1.0,
            1.0,
            "measured",
        )
        self.assertFails(
            ValueError,
            "circular",
            _core.ekert_rate,
            1.0,
            0.0,
            2.5,
            1.0,
            1.0,
            "modelled",
        )
        self.assertFails(
            NotImplementedError,
            "0.924",
            _core.ekert_rate,
            1.0,
            0.0,
            2.5,
            1.0,
            1.0,
            "device-independent",
        )

    def test_chsh(self):
        """
        Ekert's own analyser orientations drive the signed CHSH to -2 sqrt(2) times the
        visibility.
        """
        a1, a2, a3, b1, b2, b3 = _core.ekert_settings()

        for want, got in zip((0.0, math.pi / 4.0, math.pi / 2.0), (a1, a2, a3)):
            self.assertClose(got, want, atol=1e-15, msg="Alice's angles")

        for want, got in zip((math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0), (b1, b2, b3)):
            self.assertClose(got, want, atol=1e-15, msg="Bob's angles")

        for v in (1.0, 0.9, 0.5):
            self.assertClose(
                _core.ekert_chsh(a1, a3, b1, b3, v),
                -TSIRELSON * v,
                atol=1e-15,
                msg=f"v={v}",
            )
            self.assertClose(
                _core.ekert_correlate(a1, b1, v),
                -v * math.cos(a1 - b1),
                atol=1e-15,
                msg="correlator",
            )

    def test_detection(self):
        """
        Reading a missing click as -1 puts the efficiency threshold at Pironio's Fig. 3 value of
        0.924 to three figures, and it is a threshold, not a range.
        """
        lo, hi = 0.5, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            s = s_eff(mid)
            if s > 2.0 and pironio(q_eff(mid), s, 1.0) > 0.0:
                hi = mid
            else:
                lo = mid
        star = (lo + hi) / 2.0
        self.assertClose(star, 0.923137, atol=1e-5, msg="my derived threshold")
        self.assertClose(star, 0.924, atol=1e-3, msg="Pironio's Fig. 3 figure")

        for eta in (0.93, 0.95, 0.98, 1.0):
            self.assertGreater(
                _core.ekert_rate(1.0, q_eff(eta), s_eff(eta), 1.0, 1.0, "measured"),
                0.0,
                msg=f"eta={eta}",
            )

        rates = [
            _core.ekert_rate(
                1.0,
                q_eff(0.90 + 0.01 * i),
                s_eff(0.90 + 0.01 * i),
                1.0,
                1.0,
                "measured",
            )
            for i in range(11)
        ]
        self.assertMonotone(rates, rising=True, strict=False, msg="a threshold, not a range")


if __name__ == "__main__":
    rc = Exam(
        "BlindGeometry",
        "Flawed-source Bloch geometry derived from Tamaki et al. and checked blind",
        "blind_geometry.md",
    ).run(load(Geometry))
    rc |= Exam(
        "BlindChannel",
        "The simulated channel against Pereira, Curty & Tamaki's printed expressions",
        "blind_channel.md",
    ).run(load(Channel))
    rc |= Exam(
        "BlindEstimator",
        "Loss-tolerant and GLLP phase errors, kept apart",
        "blind_estimator.md",
    ).run(load(Estimator))
    rc |= Exam(
        "BlindRates",
        "Both key rates, their zero-flaw agreement and their separation",
        "blind_rates.md",
    ).run(load(Rates))
    rc |= Exam(
        "BlindEkert",
        "The CHSH-priced rate against Pironio et al.'s collective-attack bound",
        "blind_ekert.md",
    ).run(load(Ekert))
    sys.exit(rc)
