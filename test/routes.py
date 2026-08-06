import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.anchors import DMCS_EPS
from kit.links import cow_link
from qkd import _core, security

# MTT23 Sec. V: zeta' = 28, zeta = 58, eps1 = eps2 = 2^-58/6, composing to 2^-27 under Eq. (50).
ZETA = 58.0

ZETA_EC = 28.0

DPS_EPS = 2.0**-58 / 6.0

# MTT23 Sec. V forward model: mu = 2e-4 per pulse, eta = 0.1, half the detected rounds kept as code.
DPS_MU = 2e-4

DPS_ETA = 0.1

DPS_T = 0.5

DPS_N = 1e13

DPS_EBIT = 0.01

DPS_FEC = 1.16

# One cutoff sizes both the energy test and the certificate.
PSK = q.PhaseShiftKeying(states=4, alpha=0.7)

CUTOFF = 8

# Kanitschar's dimension-reduction charge over a four-symbol key map, per src/dmcs.rs::dm_price.
PRICE_4 = 0.101

PRICE_6 = 1.34e-2

# Relaxed-set operating point; W_MIN is the floor Kanitschar's three-term rule puts under a weight.
ETA = 0.5

XI = 0.01

W_MIN = 1e-6


def dps_length(mu=DPS_MU, eta=DPS_ETA, t=DPS_T, kato=True, detector="pnr"):
    """
    MTT23 Eq. (24) through `q.dps`, at the Eq. (52) forward model.
    """
    tail = q.dps.source(mu)
    counts = q.dps.counts(DPS_N, eta, mu, t)

    return q.dps.finite(
        DPS_N,
        counts[1],
        counts[2],
        DPS_EBIT,
        tail[0],
        tail[1],
        tail[2],
        t,
        DPS_EPS,
        DPS_EPS,
        ZETA,
        ZETA_EC,
        DPS_FEC,
        kato,
        detector=detector,
    )


def dmcs_ledger():
    """
    Kanitschar's own five-term demonstration budget.
    """

    return security.compose("dmcs", **DMCS_EPS)


class Differential(Question):
    def test_module_length(self):
        """
        `qkd.dps.finite` returns a positive MTT23 key length in bits over the run, its source
        tail, forward counts and Kato pair each equal to `_core`'s.
        """
        tail = q.dps.source(DPS_MU)
        counts = q.dps.counts(DPS_N, DPS_ETA, DPS_MU, DPS_T)

        self.assertEqual(tail, _core.dps_source(DPS_MU), msg="source tail")
        self.assertEqual(counts, _core.dps_counts(DPS_N, DPS_ETA, DPS_MU, DPS_T), msg="forward counts")

        bound = q.dps.phase(
            DPS_N,
            counts[1],
            counts[2],
            DPS_EBIT,
            tail[0],
            tail[1],
            tail[2],
            DPS_T,
            DPS_EPS,
            DPS_EPS,
            True,
        )
        got = dps_length()

        self.assertGreater(got, 0.0, msg=f"dps length {got}")
        self.assertLess(bound, counts[1], msg=f"phase bound {bound} vs code rounds {counts[1]}")
        self.assertEqual(q.dps.optimum(1e6, 50.0, DPS_EPS), _core.dps_kato(1e6, 50.0, DPS_EPS), msg="Kato's pair")

    def test_kato_beats_azuma(self):
        """
        `kato=True` gives a longer key than MTT23 Eq. (53)'s Azuma branch at this operating
        point.
        """
        tight = dps_length(kato=True)
        loose = dps_length(kato=False)

        self.assertGreater(tight, loose, msg=f"kato {tight} vs azuma {loose}")

    def test_receiver_stated(self):
        """
        `detector` is keyword-only with no default, the engine refusing `threshold` and the
        argument check any other string.
        """
        with self.assertRaises(TypeError):
            q.dps.finite(DPS_N, 1e6, 1e6, 0.01, 0.5, 0.1, 0.01, 0.5, DPS_EPS, DPS_EPS, ZETA, ZETA_EC, 1.16, True)

        self.assertFails(
            NotImplementedError,
            "PHOTON-NUMBER-RESOLVING receiver",
            lambda: dps_length(detector="threshold"),
            msg="detector=threshold",
        )
        self.assertFails(
            ValueError,
            'detector must be "pnr" or "threshold"',
            lambda: dps_length(detector="pnr "),
            msg="detector='pnr '",
        )

    def test_train_is_other(self):
        """
        A `q.DifferentialPhase` link, the continuous threshold-detector train of Inoue, Waks &
        Yamamoto, reports an asymptotic rate and no key length.
        """
        got = q.Link(
            modulation=q.DifferentialPhase(mu=0.2),
            channel=q.Channel(T=0.1),
            bob=q.Bob(detector=q.ClickDetector(eta=0.2, dark=1e-6)),
            security=q.IndividualAttack(qber=0.01, f=1.16),
        ).run()

        self.assertGreater(got.key_rate, 0.0, msg=f"train rate {got.key_rate}")
        self.assertIsNone(got.key_length, msg=f"train key_length {got.key_length}")

    def test_budget_declared(self):
        """
        Family `dps` composes MTT23 Eq. (50), not a sum, to 2^-27 at the paper's parameters, and
        `qkd.security.keylength` refuses it by name.
        """
        led = security.compose("dps", eps_1=DPS_EPS, eps_2=DPS_EPS, zeta=ZETA, zeta_ec=ZETA_EC)

        self.assertEqual(led.secrecy, 2.0**-27, msg=f"Eq. (50) secrecy {led.secrecy}")
        self.assertFails(
            NotImplementedError,
            "qkd.dps.finite",
            security.keylength,
            led,
            msg="keylength(dps)",
        )
        self.assertIn("NO DEFAULT", str(security.ELSEWHERE["dps"]), msg="ELSEWHERE dps")


class Vacuum(Question):
    def test_block_refused(self):
        """
        `q.PhaseBound` accepts a key block and the engine refuses both `run()` and `explain()`,
        a supplied phase error being the dominant term outside any finite-size budget.
        """
        link = cow_link(security=q.PhaseBound(e_phase=0.2, f=1.1, block=q.KeyBlock(n=1e10)))

        self.assertFails(
            NotImplementedError,
            "malformed rather than unimplemented",
            link.run,
            msg="run() with block",
        )
        self.assertFails(
            NotImplementedError,
            "THERE IS NOTHING TO ESTIMATE",
            link.explain,
            msg="explain() with block",
        )

    def test_refusal_routes(self):
        """
        The refusal names the COW' entry points that carry a length, registered as `cow-vacuum`
        at Li's weights 2/1/6/1.
        """
        link = cow_link(security=q.PhaseBound(e_phase=0.2, f=1.1, block=q.KeyBlock(n=1e10)))
        try:
            link.run()
            text = ""
        except NotImplementedError as exc:
            text = str(exc)
        for needle in ("sdp_interval", "sdp_sample", "sdp_length", "COW'"):
            self.assertIn(needle, text, msg=f"{needle} in refusal")

        spec = security.describe("cow-vacuum")

        self.assertIn("2/1/6/1", spec.rule, msg=f"cow-vacuum rule: {spec.rule}")

    def test_block_typed(self):
        """
        A bare block size in place of a `q.KeyBlock` is refused at construction.
        """
        self.assertFails(
            ValueError,
            "q.KeyBlock",
            lambda: q.PhaseBound(e_phase=0.2, block=1e10),
            msg="block=1e10",
        )

    def test_asymptotic_kept(self):
        """
        Without a block the same COW link reports a positive asymptotic rate.
        """
        got = cow_link(security=q.PhaseBound(e_phase=0.2, f=1.1)).run()

        self.assertGreater(got.key_rate, 0.0, msg=f"asymptotic COW rate {got.key_rate}")


class Pricing(Question):
    def test_price_charged(self):
        """
        The dimension-reduction charge over a four-symbol key map is 0.101 bit per pulse at
        weight 1e-4 and 1.34e-2 at 1e-6, falling with the weight.
        """
        sec = q.CertifiedBound(cutoff=CUTOFF)

        self.assertClose(sec.price(PSK, 1e-4), PRICE_4, atol=5e-4, msg="charge at w = 1e-4")
        self.assertClose(sec.price(PSK, 1e-6), PRICE_6, atol=5e-5, msg="charge at w = 1e-6")

        got = [sec.price(PSK, w) for w in (1e-4, 1e-5, 1e-6, 1e-7)]

        self.assertMonotone(got, rising=False, msg=f"charge over weights: {got}")
        self.assertEqual(sec.price(PSK, 1e-4), _core.dm_price(1e-4, 4), msg="price() vs dm_price")

    def test_energy_logged(self):
        """
        The energy test returns the base-two LOGARITHM of its failure probability,
        unrepresentable as a double at a 1e10 block, with Theorem 3's r at or above 1 and a
        positive divergence, and refuses a tolerance at or above the test rounds.
        """
        sec = q.CertifiedBound(cutoff=CUTOFF)
        logeps, ratio, div = sec.energy(2.0, 1e10, 1e2, 1e-4)

        self.assertLess(logeps, 0.0, msg=f"log2 eps {logeps}")
        self.assertGreaterEqual(ratio, 1.0, msg=f"Theorem 3 r = {ratio}")
        self.assertGreater(div, 0.0, msg=f"divergence {div}")
        self.assertEqual((logeps, ratio, div), _core.dm_energy(CUTOFF, 2.0, 1e10, 1e2, 1e-4), msg="energy() vs engine")

        self.assertFails(
            ValueError,
            "l_t must be below k_t",
            lambda: sec.energy(2.0, 1e4, 1e4, 1e-4),
            msg="l_t = k_t = 1e4",
        )

    def test_weight_inverts(self):
        """
        Kanitschar Eq. (5)'s two directions invert one another at fixed test rounds, on one
        shared r.
        """
        sec = q.CertifiedBound(cutoff=CUTOFF)
        want = -1.0e5
        weight, ratio = sec.weight(2.0, 1e10, 1e2, want)

        self.assertGreater(weight, 0.0, msg=f"weight {weight}")
        self.assertEqual(ratio, sec.energy(2.0, 1e10, 1e2, weight)[1], msg="r across the two directions")
        self.assertClose(sec.energy(2.0, 1e10, 1e2, weight)[0], want, atol=1.0, msg="round-tripped log2 eps")

    def test_accept_bounded(self):
        """
        Theorem 4's acceptance half-width falls with rounds, rises as the failure probability
        tightens, and halves for an operator declared positive semidefinite, range [0, x] not
        [-x, x].
        """
        sec = q.CertifiedBound(cutoff=CUTOFF)
        got = [sec.accept(10.0, m, 1e-11) for m in (1e7, 1e8, 1e9)]

        self.assertMonotone(got, rising=False, msg=f"width over rounds: {got}")
        self.assertGreater(sec.accept(10.0, 1e9, 1e-15), sec.accept(10.0, 1e9, 1e-9), msg="width vs eps")
        self.assertClose(
            sec.accept(10.0, 1e9, 1e-11, True) * 2.0,
            sec.accept(10.0, 1e9, 1e-11),
            msg="psd width against half the full one",
        )

    def test_relaxed_route(self):
        """
        The finite-key length runs in the engine's order -- three-term weight, clipped norms,
        two acceptance half-widths, certified minimum over the RELAXED set -- and is positive,
        charged at `price()`, and under $\\log_2 4$ bits per kept round.
        """
        sec = q.CertifiedBound(cutoff=6, steps=4)
        w_exp = sec.expected(PSK, ETA, XI)
        w_eps = sec.weight(2.0, 1e10, 1e2, -1.0e4)[0]
        w = max(w_exp, w_eps, W_MIN)
        norms = sec.clip(5.0, 0.7)
        widths = (sec.accept(norms[0], 1e9, 1e-11), sec.accept(norms[1], 1e9, 1e-11))
        got = sec.relaxed(PSK, ETA, XI, w, widths, norms)

        self.assertGreater(got.entropy, 0.0, msg=f"relaxed entropy {got.entropy}")
        self.assertGreater(got.upper, got.bound, msg=f"upper {got.upper} vs bound {got.bound}")

        led = dmcs_ledger()
        out = security.keylength(
            led,
            n_total=1e10,
            n_key=9e9,
            entropy=got.entropy,
            alphabet=4,
            w=w,
            leak=1e9,
            source="relaxed",
        )

        self.assertGreater(out.bits, 0.0, msg=f"length {out.bits}")
        self.assertClose(out.values["price"], sec.price(PSK, w), msg="charged price")
        self.assertLess(out.bits, 9e9 * math.log2(4), msg=f"length {out.bits}")

    def test_relaxed_apart(self):
        """
        An `EntropyBound` has no key, rate or key_rate, and `keylength` refuses a minimum over
        the CONTAINED equality set, which would overstate the length.
        """
        sec = q.CertifiedBound(cutoff=6, steps=4)
        norms = sec.clip(5.0, 0.7)
        widths = (sec.accept(norms[0], 1e9, 1e-11), sec.accept(norms[1], 1e9, 1e-11))
        got = sec.relaxed(PSK, ETA, XI, W_MIN, widths, norms)
        for name in ("key", "key_rate", "rate"):
            self.assertFails(
                AttributeError,
                "has no key",
                lambda n=name: getattr(got, n),
                msg=f"EntropyBound.{name} resolved",
            )

        led = dmcs_ledger()
        for source in ("certified", "asymptotic", "dm_secure", "dm_trusted"):
            self.assertFails(
                NotImplementedError,
                "CONTAINED in the relaxed set",
                lambda x=source: security.keylength(
                    led,
                    n_total=1e10,
                    n_key=9e9,
                    entropy=1.0,
                    alphabet=4,
                    w=W_MIN,
                    leak=1e9,
                    source=x,
                ),
                msg=f"length at source={source}",
            )


if __name__ == "__main__":
    rc = Exam(
        "RoutesDifferential",
        "The finite-key DPS length, and the receiver it will not default",
        "routes_dps.md",
    ).run(load(Differential))
    rc |= Exam(
        "RoutesVacuum",
        "What a key block asks of three-sequence COW, and where it is sent",
        "routes_cow.md",
    ).run(load(Vacuum))
    rc |= Exam(
        "RoutesPricing",
        "The four discrete-modulation finite-size pieces, and the two feasible sets kept apart",
        "routes_dmcs.md",
    ).run(load(Pricing))
    sys.exit(rc)
