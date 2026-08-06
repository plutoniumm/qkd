import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import numpy as np

import qkd as q
from kit.anchors import ALPHA, D_COW, E_ALIGN, ETA_BOB, ETA_COW, E_DET, F_COW, F_REC, P_DARK, Y_DARK
from kit.cache import memo
from kit.checks import Guarded
from kit.forms import background
from kit.links import basis
from qkd import _core
from qkd._core import GaussianState

RATE = 1e9


@memo
def clicks(mu=0.2, t=0.1, det=None, optics=None, lw=1e6, n=200_000, seed=3):
    """
    One phase-keyed run with the error rate simulated rather than pinned.
    """

    return q.Link(
        modulation=q.DifferentialPhase(mu=mu),
        channel=q.Channel(T=t),
        alice=q.Alice(laser=q.Laser(linewidth=lw), symbol_rate=RATE),
        bob=q.Bob(
            detector=det or q.ClickDetector(eta=0.2, dark=1e-6),
            receiver=optics or q.DelayInterferometer(delay=1, visibility=0.98),
        ),
        security=q.IndividualAttack(f=1.16),
    ).run(symbols=n, seed=seed)


def cow(dist=25.0, phase=0.2, split=0.1, mu=0.5):
    """
    An intensity-keyed link with its coherence-monitoring tap.
    """

    return q.Link(
        modulation=q.IntensityKeying(mu=mu, decoy_frac=0.1),
        channel=q.Fiber(length=dist),
        bob=q.Bob(
            detector=q.ClickDetector(eta=ETA_COW, dark=D_COW),
            receiver=q.CoherenceMonitor(split=split, misalign=E_ALIGN),
        ),
        security=q.PhaseBound(e_phase=phase, f=F_COW),
    )


def shown(res):
    """
    The value of every labelled entry of a report, keyed by name.
    """

    return {k: v["value"] for k, v in res.items() if isinstance(v, dict)}


def bell(la=0.1, lb=5.0, eta=1.0, vel=0.0, va=1e5, env=None, arms=None):
    """
    A CV relay: two senders, two arms, one Bell measurement that belongs to Eve.
    """

    span = arms or (q.Fiber(length=la), q.Fiber(length=lb))

    return q.Swap(
        alice=q.Sender(modulation=q.GaussianModulation(v_a=va)),
        bob=q.Sender(modulation=q.GaussianModulation(v_a=va)),
        relay=q.Relay(bell=q.BellDetector(eta=eta, v_el=vel)),
        channels=span,
        security=q.Asymptotic(beta=0.98),
        environment=env,
    )


def midpoint(va, arms, corr, eta, vel):
    """
    Post-relay 4x4 in SNU (cvmdi_cov's convention), ``arms`` and ``corr`` at the channel plane; detector loss
    and v_el sit on the output ports, homodynes last.
    """
    (ta, wa), (tb, wb) = arms
    g, gp = corr
    mu = va + 1.0
    s = math.sqrt(mu * mu - 1.0)
    u = math.sqrt((1.0 - ta) * (1.0 - tb))
    cov = np.zeros((8, 8))
    for i, d in enumerate(
        (
            mu,
            mu,
            ta * mu + (1.0 - ta) * wa,
            ta * mu + (1.0 - ta) * wa,
            tb * mu + (1.0 - tb) * wb,
            tb * mu + (1.0 - tb) * wb,
            mu,
            mu,
        )
    ):
        cov[i, i] = d

    for i, j, v in (
        (0, 2, math.sqrt(ta) * s),
        (1, 3, -math.sqrt(ta) * s),
        (4, 6, math.sqrt(tb) * s),
        (5, 7, -math.sqrt(tb) * s),
        (2, 4, u * g),
        (3, 5, u * gp),
    ):
        cov[i, j] = v
        cov[j, i] = v

    st = GaussianState.from_moments([0.0] * 8, (0.5 * cov).reshape(-1).tolist())
    st = st.bs(1, 2, 0.5)
    for m in (1, 2):
        st = st.thermal_loss(m, eta, vel, False)

    st = st.condition(1, math.pi / 2, 0.0)
    st = st.condition(1, 0.0, 0.0)

    return 2.0 * np.array(st.cov()).reshape(4, 4)


def variant(dist=25.0, mu=0.48, **kw):
    """
    A GYS basis-keyed link with a third basis or a pair announcement, everything else BB84's.
    """

    return basis(dist, mod=q.BasisKeying(decoy=q.Decoy((mu, 0.1, 0.0)), **kw))


def bounds(dist=25.0, mu=0.48, pair=False):
    """
    (q_mu, e_mu, y1, e1, q1) assembled here; ``pair`` swaps BB84's yield law for the conclusive-exclusion one.
    """
    eta = _core.decoy_eta(ALPHA, dist, ETA_BOB)
    y0 = background(P_DARK)
    if pair:
        q_mu, e_mu = _core.sarg_gain(mu, eta, y0, E_DET)
        weak = _core.sarg_gain(0.1, eta, y0, E_DET)
        vac = _core.sarg_yield(eta, y0, E_DET, 0)
    else:
        q_mu, e_mu = _core.decoy_gain(mu, eta, y0, E_DET)
        weak = _core.decoy_gain(0.1, eta, y0, E_DET)
        vac = (y0, 0.5)

    y1, e1, q1 = _core.decoy_bounds(mu, 0.1, 0.0, q_mu, e_mu, weak[0], weak[1], vac[0], vac[1])

    return q_mu, e_mu, y1, e1, q1


def two_state(t=0.6, mu=0.23, eph=None, ref=False, vis=1.0, dark=0.0):
    """
    A two-state link: one weak pulse per bit, Bob's nulling receiver.
    """

    return q.Link(
        modulation=q.TwoStateKeying(mu=mu, reference=ref),
        channel=q.Channel(T=t),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=dark),
            receiver=q.NullingReceiver(visibility=vis),
        ),
        security=q.DiscriminationBound(e_phase=eph, f=1.0),
    )


def violated(s=2.7, source="measured", f=1.22, dist=10.0):
    """
    A photon-pair link priced by a Bell violation rather than by a phase error.
    """

    return q.PairLink(
        source=q.PairSource(brightness=0.01),
        detectors=(
            q.ClickDetector(eta=0.2, dark=1e-6),
            q.ClickDetector(eta=0.2, dark=1e-6),
        ),
        channels=(q.Fiber(length=dist), q.Fiber(length=dist)),
        security=q.ViolationBound(s=s, source=source, f=f),
    )


def keyed(states=4, alpha=0.4, t=0.5, xi=0.01, eta=1.0, vel=0.0, beta=0.95):
    """
    A phase-shift-keyed link: a finite constellation on the two-party shape.
    """

    return q.Link(
        modulation=q.PhaseShiftKeying(states=states, alpha=alpha),
        channel=q.Channel(T=t, xi=xi, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=eta, v_el=vel, trusted=False)),
        security=q.Asymptotic(beta=beta),
    )


def single(t=0.6, mu=0.19, depol=0.005, dark=0.0, vis=1.0, eph=None):
    """
    A two-state link on a single-photon source over Tamaki and Lutkenhaus's depolarising channel.
    """

    return q.Link(
        modulation=q.TwoStateKeying(source="single", mu=mu, depol=depol),
        channel=q.Channel(T=t),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=dark),
            receiver=q.NullingReceiver(visibility=vis),
        ),
        security=q.DiscriminationBound(e_phase=eph, f=1.0),
    )


def certified(alpha=0.7, t=0.5, xi=0.01, nc=6, steps=6):
    """
    A phase-shift-keyed link bounded by the two-step numerical proof rather than the closed form.
    """

    return q.Link(
        modulation=q.PhaseShiftKeying(states=4, alpha=alpha),
        channel=q.Channel(T=t, xi=xi, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=1.0, v_el=0.0, trusted=False)),
        security=q.CertifiedBound(cutoff=nc, steps=steps, beta=0.95),
    )


def receiver(trust, t=0.5, xi=0.01, nc=6, steps=6, eta=0.5, vel=0.05):
    """
    The certified link with Bob's receiver trusted or not; certified() above pins an ideal one.
    """

    return q.Link(
        modulation=q.PhaseShiftKeying(states=4, alpha=0.7),
        channel=q.Channel(T=t, xi=xi, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=eta, v_el=vel, trusted=trust)),
        security=q.CertifiedBound(cutoff=nc, steps=steps, beta=0.95),
    )


@memo
def reports():
    """
    One explain() per family, and per branch proved against a different class of attack.
    """
    gauss = q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=0.01, ref="input"),
        bob=q.Bob(detector=q.Homodyne(eta=0.6, v_el=0.1, trusted=True)),
        security=q.Asymptotic(),
    )
    phase = q.Link(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Channel(T=0.1),
        bob=q.Bob(detector=q.ClickDetector(eta=0.2, dark=1e-6)),
        security=q.IndividualAttack(qber=0.03, f=1.16),
    )

    flaw = q.Link(
        modulation=q.FlawedKeying(flaw=q.SourceFlaw.opposed(0.063), sift=0.25),
        channel=q.Channel(T=0.5),
        bob=q.Bob(detector=q.ClickDetector(eta=0.9, dark=1e-7)),
        security=q.FlawBound(analysis="tolerant", f=1.16),
    )

    return {
        "gaussian": gauss.explain(),
        "dps": phase.explain(),
        "flaw": flaw.explain(),
        "flaw_coin": q.Link(
            modulation=flaw.modulation,
            channel=flaw.channel,
            bob=flaw.bob,
            security=q.FlawBound(analysis="standard", f=1.16),
        ).explain(),
        "bb84": basis(25.0).explain(),
        "sixstate": variant(bases=3).explain(),
        "sarg": variant(announce="pair").explain(),
        "cow": cow().explain(),
        "b92": two_state().explain(),
        "b92_ref": two_state(eph=0.05, ref=True).explain(),
        "b92_single": single().explain(),
        "psk": keyed().explain(),
        "psk_certified": certified().explain(),
    }


class AttackClass(Question):
    """
    Quoted from each engine's own module header, never classified from outside it.
    """

    def test_every_family(self):
        """
        Every family's report carries an `attack` string beside `security`, not in place of it.
        """
        for name, info in reports().items():
            self.assertIn("attack", info, msg=f"{name} names no attack class")
            self.assertIn("security", info, msg=f"{name} dropped its component")
            self.assertTrue(info["attack"], msg=f"{name}'s class reads empty")
            self.assertIsInstance(info["attack"], str, msg=f"{name} is not text")

    def test_stated_classes(self):
        """
        The attack row is the engine header's class verbatim: Gaussian collective, general individual for
        phase keying, collective with one-way post-processing for six-state.
        """
        seen = reports()

        self.assertEqual(seen["gaussian"]["attack"], "Gaussian collective", msg="src/keyrate.rs:203")
        self.assertEqual(seen["dps"]["attack"], "general individual", msg="src/keyrate.rs:317")
        self.assertEqual(
            seen["sixstate"]["attack"],
            "collective, one-way post-processing",
            msg="src/sixstate.rs:157",
        )

    def test_qualified_classes(self):
        """
        A qualified proof carries its qualifier and an unqualified one does not: bb84 and flaw unstated, cow
        and b92_ref undefined, b92 coherent hardware, psk_certified cutoff.
        """
        seen = reports()
        carried = (
            ("bb84", "unstated"),
            ("cow", "undefined"),
            ("b92_ref", "undefined"),
            ("b92", "coherent hardware"),
            ("psk_certified", "cutoff"),
            ("flaw", "unstated"),
        )
        absent = (
            ("sarg", "unstated"),
            ("b92", "undefined"),
            ("b92_single", "coherent hardware"),
            ("psk", "cutoff"),
        )
        for name, needle in carried:
            self.assertIn(needle, seen[name]["attack"], msg=f"{name} dropped '{needle}'")

        for name, needle in absent:
            self.assertNotIn(needle, seen[name]["attack"], msg=f"{name} carries '{needle}'")

    def test_classes_distinct(self):
        """
        No two families read the same attack class except b92_ref/cow and flaw/flaw_coin, one engine's two
        branches each.
        """
        # bb84 and sarg are not a pair.
        shared = {("b92_ref", "cow"), ("flaw", "flaw_coin")}
        rows = {}
        for name, info in sorted(reports().items()):
            rows.setdefault(info["attack"], []).append(name)

        for names in rows.values():
            if len(names) > 1:
                self.assertIn(tuple(names), shared, msg=f"{names} read one attack class")


class PhaseKeyed(Question):

    def test_engine_agreement(self):
        """
        Every reported number is bit for bit what `_core.run_clicks` and `_core.dps_rate` return for the same
        hardware.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6, dead_time=20e-9, afterpulse=0.02)
        res = clicks(det=det)
        out = _core.run_clicks(200_000, 3, 0.2, 0.1, 0.2, 1e-6, 0.98, 1, 1e6, RATE, 20e-9, 0.02, 0, False)
        key = _core.dps_rate(out.sift_rate, out.qber, 0.2, 1.16)

        self.assertEqual(res.qber, out.qber, msg=f"qber {res.qber} vs {out.qber}")
        self.assertEqual(res.sift_rate, out.sift_rate, msg=f"sift_rate {res.sift_rate}")
        self.assertEqual(res.visibility, out.visibility, msg=f"visibility {res.visibility}")
        self.assertEqual(res.clicks, out.clicks, msg=f"clicks {res.clicks}")
        self.assertEqual(res.mu_bob, out.mu_bob, msg=f"mu_bob {res.mu_bob}")
        self.assertEqual(res.key_rate, key, msg=f"key_rate {res.key_rate} is not dps_rate's")

    def test_alice_plane(self):
        """
        The rate is built on Alice's mu, not mu_bob = T*mu, and sits below the Bob-plane rate.
        """
        res = clicks(mu=0.3, t=0.1)
        alice = _core.dps_rate(res.sift_rate, res.qber, 0.3, 1.16)
        bob = _core.dps_rate(res.sift_rate, res.qber, res.mu_bob, 1.16)

        self.assertClose(res.mu_bob, 0.03, msg=f"mu_bob {res.mu_bob}")
        self.assertEqual(res.key_rate, alice, msg="key_rate is not built on Alice's mu")
        self.assertGreater(bob, res.key_rate, msg=f"Bob-plane rate {bob} not above {res.key_rate}")

    def test_one_definition(self):
        """
        The simulated p_click matches p_ideal = 1 - (1 - dark)^2*exp(-mu*T*eta) to 5 sigma, and dead time
        only lowers it.
        """
        shots = 1_000_000
        free = clicks(n=shots)
        ideal = free.explain["p_ideal"]["value"]
        held = clicks(det=q.ClickDetector(eta=0.2, dark=1e-6, dead_time=1e-6))
        # 5 sigma of the binomial se 1.6e-2 at p = 4.0e-3; worst of seeds 1..20 is 2.1 sigma.
        sigma = math.sqrt((1.0 - ideal) / (shots * ideal))

        self.assertClose(
            free.p_click / ideal,
            1.0,
            atol=5.0 * sigma,
            msg=f"{free.p_click:.6f} vs {ideal:.6f} (5 sigma = {5.0 * sigma:.4f})",
        )
        self.assertClose(
            ideal,
            1.0 - (1.0 - 1e-6) ** 2 * math.exp(-0.2 * 0.1 * 0.2),
            msg=f"p_ideal {ideal}",
        )
        self.assertLess(
            held.p_click,
            held.explain["p_ideal"]["value"],
            msg=f"held p_click {held.p_click}",
        )

    def test_dead_time_reaches(self):
        """
        A microsecond of dead time at a 1 GHz clock costs clicks and key, while afterpulsing adds counts and
        raises the QBER until no key is left.
        """
        free = clicks()
        held = clicks(det=q.ClickDetector(eta=0.2, dark=1e-6, dead_time=1e-6))
        noisy = clicks(det=q.ClickDetector(eta=0.2, dark=1e-6, afterpulse=0.5))

        self.assertLess(held.clicks, free.clicks, msg=f"held clicks {held.clicks} vs free {free.clicks}")
        self.assertLess(held.key_rate, free.key_rate, msg=f"held key {held.key_rate} vs free {free.key_rate}")
        self.assertGreater(held.clicks, 0, msg=f"held clicks {held.clicks}")
        self.assertGreater(noisy.clicks, free.clicks, msg=f"noisy clicks {noisy.clicks} vs free {free.clicks}")
        self.assertGreater(noisy.qber, free.qber, msg=f"noisy qber {noisy.qber} vs free {free.qber}")
        self.assertEqual(noisy.key_rate, 0.0, msg=f"noisy key_rate {noisy.key_rate}")

    def test_linewidth_costs(self):
        """
        The QBER rises monotonically with laser linewidth and the key falls, reaching zero at 10 MHz.
        """
        runs = [clicks(lw=lw) for lw in (0.0, 1e6, 1e7)]

        self.assertMonotone([r.qber for r in runs], rising=True, msg=f"qber {[r.qber for r in runs]}")
        self.assertMonotone([r.key_rate for r in runs], rising=False, msg=f"key {[r.key_rate for r in runs]}")
        self.assertEqual(runs[2].key_rate, 0.0, msg=f"key at 10 MHz {runs[2].key_rate}")

    def test_doubles_squashed(self):
        """
        Double clicks are kept with a random bit (squashing model): clicks = sifted + doubles, and p_click
        equals the sift rate.
        """
        res = clicks(mu=0.4, t=1.0, optics=q.DelayInterferometer(visibility=0.9), n=500_000)

        self.assertGreater(res.doubles, 0, msg=f"doubles {res.doubles}")
        self.assertEqual(res.clicks, res.sifted + res.doubles, msg=f"clicks {res.clicks}, sifted {res.sifted}")
        self.assertLess(res.doubles, res.sifted, msg=f"doubles {res.doubles} vs sifted {res.sifted}")
        self.assertClose(res.sift_rate, res.sifted / res.slots, msg=f"sift_rate {res.sift_rate}")
        self.assertEqual(
            res.p_click,
            res.sift_rate,
            msg=f"p_click {res.p_click} vs sift_rate {res.sift_rate}",
        )

    def test_pinned_closed_form(self):
        """
        Pinning the error rate gives the Waks-Takesue-Yamamoto closed form (PRA 73, 012344): at e = 0 the rate
        is p_click*(1 - 2*mu) with p_click = 1 - (1 - dark)^2*exp(-mu*T*eta), and no clicks counted.
        """
        res = q.Link(
            modulation=q.DifferentialPhase(mu=0.2),
            channel=q.Channel(T=0.1),
            bob=q.Bob(detector=q.ClickDetector(eta=0.2, dark=1e-6)),
            security=q.IndividualAttack(qber=0.0, f=1.16),
        ).run()
        p_click = 1.0 - (1.0 - 1e-6) ** 2 * math.exp(-0.2 * 0.1 * 0.2)

        self.assertClose(res.key_rate, p_click * 0.6, atol=1e-12, msg=f"key_rate {res.key_rate}")
        self.assertIsNone(res.clicks, msg=f"clicks {res.clicks}")

    def test_click_provenance(self):
        """
        Before a run every simulated quantity is labelled still to compute, after one it is derived.
        """
        link = q.Link(
            modulation=q.DifferentialPhase(mu=0.2),
            channel=q.Channel(T=0.1),
            alice=q.Alice(laser=q.Laser(linewidth=1e6), symbol_rate=RATE),
            bob=q.Bob(
                detector=q.ClickDetector(eta=0.2, dark=1e-6, dead_time=20e-9),
                receiver=q.DelayInterferometer(delay=2, visibility=0.98),
            ),
            security=q.IndividualAttack(f=1.16),
        )
        dry = link.explain()
        res = link.run(symbols=200_000, seed=3)
        for key in ("qber", "sift_rate", "visibility", "mu_bob", "p_click"):
            self.assertIsNone(dry[key]["value"], msg=f"{key} known dry")
            self.assertEqual(
                dry[key]["label"],
                "derived (run to compute)",
                msg=f"{key} dry label",
            )
            self.assertEqual(res.explain[key]["label"], "derived", msg=f"{key} run label")

        self.assertEqual(res.explain["qber"]["value"], res.qber, msg=f"explain qber vs {res.qber}")
        self.assertEqual(shown(res.explain)["delay"], 2, msg="delay not pinned at 2")
        self.assertEqual(shown(res.explain)["dead_time"], 20e-9, msg="dead_time not pinned at 20e-9")


class BasisDecoy(Question):
    """
    Privacy amplification is paid on the decoy-bounded single-photon component alone, error correction against
    the whole sifted gain.
    """

    def test_engine_agreement(self):
        """
        The rate is `decoy_gain` into `decoy_bounds` into `bb84_rate`, on Y0 twice the per-gate dark
        probability over two ports.
        """
        res = basis(25.0).run()
        info = shown(res.explain)
        eta = _core.decoy_eta(ALPHA, 25.0, ETA_BOB)
        y0 = background(P_DARK)
        q_mu, e_mu = _core.decoy_gain(0.48, eta, y0, E_DET)
        q_nu1, e_nu1 = _core.decoy_gain(0.1, eta, y0, E_DET)
        y1, e1, q1 = _core.decoy_bounds(0.48, 0.1, 0.0, q_mu, e_mu, q_nu1, e_nu1, y0, 0.5)

        # Ma, Qi, Zhao & Lo (PRA 72, 012326) tabulate Y0 = 1.7e-6, twice GYS's printed P_e = 8.5e-7.
        # atol 1e-12: coincidence term P_DARK^2 = 7.2243e-13 plus 38%.
        self.assertClose(y0, Y_DARK, atol=1e-12, msg=f"y0 {y0} vs Y_DARK {Y_DARK}")
        self.assertEqual(info["y0"], y0, msg=f"reported y0 {info['y0']}")
        self.assertClose(info["eta_ab"], eta, msg=f"eta_ab {info['eta_ab']}")
        self.assertEqual(res.p_click, q_mu, msg=f"p_click {res.p_click} vs q_mu {q_mu}")
        self.assertEqual(res.qber, e_mu, msg=f"qber {res.qber} vs e_mu {e_mu}")
        self.assertEqual(info["y1"], y1, msg=f"y1 {info['y1']}")
        self.assertEqual(info["e1"], e1, msg=f"e1 {info['e1']}")
        self.assertEqual(
            res.key_rate,
            _core.bb84_rate(0.5, q_mu, e_mu, q1, e1, F_REC),
            msg=f"key_rate {res.key_rate} is not bb84_rate's",
        )

    def test_bounds_sandwich(self):
        """
        y1 sits below the infinite-decoy yield and e1 above its error rate at every distance, and a weakest
        intensity of 0 is the tighter bound.
        """
        exact = basis(25.0, intens=(0.48, 0.1, 0.0)).run()
        weak = basis(25.0, intens=(0.48, 0.1, 0.02)).run()

        self.assertGreater(exact.key_rate, 0.0, msg=f"vacuum-decoy key {exact.key_rate}")
        self.assertGreater(
            shown(exact.explain)["y1"],
            shown(weak.explain)["y1"],
            msg="vacuum+weak y1 is not the tighter",
        )

        for dist in (0.0, 25.0, 50.0, 100.0):
            info = shown(basis(max(dist, 1e-9)).run().explain)
            y1, e1 = _core.decoy_ideal(info["eta_ab"], background(P_DARK), E_DET)
            self.assertLess(info["y1"], y1, msg=f"{dist} km: y1 {info['y1']} not below {y1}")
            self.assertGreater(info["e1"], e1, msg=f"{dist} km: e1 {info['e1']} not above {e1}")

    def test_sifting_is_a_factor(self):
        """
        The basis-reconciliation factor multiplies the whole rate: sift = 1.0 is exactly twice sift = 0.5.
        """
        half = basis(25.0, sift=0.5).run().key_rate
        full = basis(25.0, sift=1.0).run().key_rate

        self.assertClose(full, 2.0 * half, msg=f"full {full} vs twice half {2.0 * half}")


class BasisVariants(Guarded):
    """
    Both run BB84's decoy layer unchanged and part company at the privacy amplification alone.
    """

    def test_engine_agreement(self):
        """
        Three bases: the rate is `decoy_gain` into `decoy_bounds` into `sixstate_rate`, at `sixstate_sift`'s
        share, not a literal 1/3.
        """
        res = variant(bases=3).run()
        info = shown(res.explain)
        q_mu, e_mu, _, e1, q1 = bounds()
        sift = _core.sixstate_sift(1.0 / 3.0)[0]

        self.assertEqual(info["sift"], sift, msg=f"sift {info['sift']}")
        self.assertEqual(res.p_click, q_mu, msg=f"p_click {res.p_click} vs q_mu {q_mu}")
        self.assertEqual(info["e1"], e1, msg=f"e1 {info['e1']}")
        self.assertEqual(
            res.key_rate,
            _core.sixstate_rate(sift, q_mu, e_mu, q1, e1, F_REC),
            msg=f"key_rate {res.key_rate} is not sixstate_rate's",
        )
        self.assertEqual(
            info["chi_e1"],
            _core.sixstate_holevo(e1),
            msg=f"chi_e1 {info['chi_e1']}",
        )

    def test_pair_agreement(self):
        """
        The pair announcement: `sarg_gain`'s and `sarg_yield`'s conclusive rates into the same protocol-blind
        inversion, and the rate is `sarg_rate` with no sifting factor outside it.
        """
        res = variant(mu=0.16, announce="pair").run()
        info = shown(res.explain)
        q_mu, e_mu, _, e1, q1 = bounds(mu=0.16, pair=True)

        self.assertEqual(res.p_click, q_mu, msg=f"p_click {res.p_click} vs q_mu {q_mu}")
        self.assertEqual(res.qber, e_mu, msg=f"qber {res.qber} vs e_mu {e_mu}")
        self.assertEqual(info["q1"], q1, msg=f"q1 {info['q1']}")
        self.assertEqual(
            res.key_rate,
            _core.sarg_rate(q_mu, e_mu, q1, e1, 0.0, 0.0, F_REC),
            msg=f"key_rate {res.key_rate} is not sarg_rate's",
        )
        self.assertEqual(
            info["sift"],
            _core.sarg_sift(E_DET),
            msg=f"sift {info['sift']}",
        )

    def test_third_basis_pays(self):
        """
        At equal sifting the third basis buys a larger rate on the same decoy bounds, and at the uniform 1/3
        against 1/2 it costs more on this span than it buys.
        """
        q_mu, e_mu, _, e1, q1 = bounds()
        six = _core.sixstate_rate(0.5, q_mu, e_mu, q1, e1, F_REC)
        bb84 = _core.bb84_rate(0.5, q_mu, e_mu, q1, e1, F_REC)

        self.assertGreater(six, bb84, msg=f"six {six} vs bb84 {bb84}")
        self.assertLess(
            variant(bases=3).run().key_rate,
            variant().run().key_rate,
            msg="three-basis rate not below BB84's at uniform sifting",
        )
        self.assertLess(
            _core.sixstate_holevo(e1),
            _core.sixstate_eve(e1, e1, e1) + 1e-15,
            msg="sixstate_holevo above sixstate_eve at equal error rates",
        )

    def test_two_photon_absent(self):
        """
        The two-photon term is reported absent (the Ma-Qi-Zhao-Lo inversion bounds Y1, nothing bounds Y2), so
        the rate sits below the infinite-decoy one.
        """
        res = variant(mu=0.16, announce="pair").run()
        info = shown(res.explain)
        eta = _core.decoy_eta(ALPHA, 25.0, ETA_BOB)
        y2, e2 = _core.sarg_yield(eta, background(P_DARK), E_DET, 2)
        q2 = y2 * 0.16 * 0.16 * math.exp(-0.16) / 2.0
        full = _core.sarg_rate(res.p_click, res.qber, info["q1"], info["e1"], q2, e2, F_REC)

        self.assertEqual(res.explain["q2"]["label"], "absent", msg="q2 is not labelled absent")
        self.assertEqual(info["q2"], 0.0, msg=f"q2 {info['q2']}")
        self.assertGreater(q2, 0.0, msg=f"infinite-decoy q2 {q2}")
        self.assertGreater(full, res.key_rate, msg=f"full {full} vs {res.key_rate}")

    def test_sift_defaults(self):
        """
        `q.BasisKeying.sift` is 1/bases unless declared, and None under a pair announcement.
        """
        self.assertEqual(q.BasisKeying().sift, 0.5, msg=f"sift {q.BasisKeying().sift}")
        self.assertClose(
            q.BasisKeying(bases=3).sift,
            1.0 / 3.0,
            atol=1e-15,
            msg="sift at bases=3",
        )
        self.assertIsNone(
            q.BasisKeying(announce="pair").sift,
            msg="the pair announcement resolved a sift",
        )
        self.assertEqual(
            q.BasisKeying(bases=3, sift=0.81).sift,
            0.81,
            msg="a declared sift was dropped",
        )

    def test_guards(self):
        """
        A sampled pulse train, a finite block and detector afterpulsing are each refused by the two variants.
        """
        cases = (
            (
                dict(bases=3),
                "alice",
                q.Alice(),
                NotImplementedError,
                "cannot emit six-state",
                "six-state pulse train",
            ),
            (
                dict(announce="pair", mu=0.16),
                "security",
                q.SplittingAttack(f=F_REC, block=q.KeyBlock(n=1e10)),
                NotImplementedError,
                "no published analysis states a key length",
                "SARG04 finite block",
            ),
            (
                dict(bases=3),
                "impairments",
                (q.DeadTime(dead=0.0, afterpulse=0.05),),
                ValueError,
                "BB84's yield law",
                "six-state afterpulsing",
            ),
        )
        for kw, field, part, exc, needle, why in cases:
            link = variant(**kw)
            setattr(link, field, part)
            self.assertFails(exc, needle, link.run, msg=why)

    def test_bias_refused(self):
        """
        All four basis-keyed branches refuse a declared bias by name rather than dropping it.
        """
        finite = variant(bias=0.8)
        finite.security = q.SplittingAttack(f=F_REC, block=q.KeyBlock(n=1e10))
        cases = (
            ("BB84 asymptotic", variant(bias=0.8)),
            ("BB84 finite", finite),
            ("six-state", variant(bases=3, bias=0.8)),
            ("SARG04", variant(mu=0.16, announce="pair", bias=0.8)),
        )
        for name, link in cases:
            self.assertFails(ValueError, "bias", link.run, msg=f"{name} took a bias")

        self.assertGreater(variant().run().key_rate, 0.0, msg="no bias declared, no key")


class TwoState(Guarded):

    def test_engine_agreement(self):
        """
        Every reported number is bit for bit `_core.b92_point`'s for the same receiver, the gain and error
        rate `b92_detect`'s.
        """
        res = two_state().run()
        info = shown(res.explain)
        out = _core.b92_point(0.23, 0.6, 1.0, 1.0, 0.0, 0.0, 1.0, False)

        self.assertEqual(res.p_click, out[0], msg=f"p_click {res.p_click} vs {out[0]}")
        self.assertEqual(res.qber, out[1], msg=f"qber {res.qber} vs {out[1]}")
        self.assertEqual(info["ceiling"], out[2], msg=f"ceiling {info['ceiling']} vs {out[2]}")
        self.assertEqual(res.key_rate, out[3], msg=f"key_rate {res.key_rate} vs {out[3]}")
        self.assertEqual(
            info["gain"],
            _core.b92_detect(0.23, 0.6, 1.0, 1.0, 0.0)[0],
            msg=f"gain {info['gain']}",
        )
        self.assertEqual(info["overlap"], _core.b92_overlap(0.23), msg=f"overlap {info['overlap']}")

    def test_derived_phase(self):
        """
        `b92_rate` at the reported gain, bit error and phase error, capped by the reported ceiling, rebuilds
        the rate exactly.
        """
        for t in (0.4, 0.6, 0.9):
            res = two_state(t=t, dark=1e-6, vis=0.98).run()
            info = shown(res.explain)
            want = _core.b92_rate(res.p_click, res.qber, info["e_phase"], 1.0)
            self.assertEqual(
                res.key_rate,
                min(want, info["ceiling"]),
                msg=f"T={t}: key_rate {res.key_rate} not rebuilt",
            )
            self.assertEqual(
                res.explain["e_phase"]["label"],
                "derived",
                msg="e_phase is not labelled derived",
            )

    def test_usd_boundary(self):
        """
        At a detected transmittance equal to one less the state overlap the derived bound is one half and the
        rate zero, and a better span carries key.
        """
        edge = _core.b92_usd(q.TwoStateKeying(mu=0.23).overlap)
        dead = two_state(t=edge).run()
        live = two_state(t=min(1.0, 1.5 * edge)).run()

        self.assertClose(
            shown(dead.explain)["discrimination"],
            edge,
            msg=f"discrimination row vs edge {edge}",
        )
        self.assertClose(
            shown(dead.explain)["e_phase"],
            0.5,
            atol=1e-9,
            msg="e_phase is not 0.5 at the edge",
        )
        self.assertEqual(dead.key_rate, 0.0, msg=f"key_rate at the edge {dead.key_rate}")
        self.assertGreater(live.key_rate, 0.0, msg=f"key_rate beyond it {live.key_rate}")

    def test_supplied_floor(self):
        """
        A supplied bound is a floor on the plain branch: a looser one is honoured and costs key, a tighter one
        is not taken.
        """
        free = two_state().run()
        tight = two_state(eph=1e-6).run()
        loose = two_state(eph=0.4).run()

        self.assertEqual(tight.key_rate, free.key_rate, msg=f"tight {tight.key_rate} vs free {free.key_rate}")
        self.assertLess(loose.key_rate, free.key_rate, msg=f"loose {loose.key_rate} vs free {free.key_rate}")
        self.assertEqual(
            free.explain["e_phase_floor"]["label"],
            "absent",
            msg="e_phase_floor is not absent",
        )
        self.assertEqual(
            loose.explain["e_phase_floor"]["label"],
            "pinned",
            msg="a supplied e_phase_floor is not pinned",
        )

    def test_reference_supplied(self):
        """
        The strong-reference branch derives no phase error, pins a supplied one, and prices it cheaper than
        the derived branch at this loss.
        """
        self.assertFails(
            NotImplementedError,
            "SUPPLIED e_phase",
            two_state(ref=True).run,
            msg="the reference branch derived a phase error",
        )

        held = two_state(ref=True, eph=0.05).run()
        self.assertEqual(
            held.explain["e_phase"]["label"],
            "pinned",
            msg="e_phase is not pinned",
        )
        self.assertGreater(
            held.key_rate,
            two_state(eph=0.05).run().key_rate,
            msg=f"reference key {held.key_rate}",
        )

    def test_single_engine(self):
        """
        The seam-free branch's numbers are `b92_plain`'s at the overlap, loss and depolarising rate reported,
        and `overlap_best` is the amplitude overlap `b92_limit` returns, not its square.
        """
        res = single(t=0.6, mu=0.19, depol=0.005).run()
        info = shown(res.explain)
        want = _core.b92_plain(info["overlap"], info["loss"], info["depol"], info["f"])

        for name, got in zip(("gain", "qber", "e_phase"), want):
            self.assertClose(info[name], got, msg=f"{name} {info[name]} vs {got}")

        self.assertClose(res.key_rate, want[3], msg=f"key_rate {res.key_rate} vs {want[3]}")
        self.assertNotIn("ceiling", info, msg="a ceiling row was reported")

        plan = shown(single().explain())
        limit, best = _core.b92_limit(plan["loss"], plan["f"])
        self.assertClose(plan["depol_limit"], limit, msg=f"depol_limit {plan['depol_limit']}")
        self.assertClose(plan["overlap_best"], best, msg=f"overlap_best {plan['overlap_best']} vs {best}")
        self.assertGreater(best, best * best, msg=f"best {best} equals its square")

    def test_single_refusals(self):
        """
        A dark count, a fringe visibility and a supplied phase floor reach no term of the single-photon branch
        and are refused.
        """
        cases = (
            (dict(dark=1e-6), "dark count rate", "dark count accepted"),
            (dict(vis=0.98), "fringe visibility", "visibility accepted"),
            (dict(eph=0.05), "DERIVES its phase error", "floor accepted under a derived bound"),
        )
        for kw, needle, why in cases:
            self.assertFails(ValueError, needle, single(**kw).run, msg=why)

    def test_guards(self):
        """
        The family refuses the receivers, security models, detector memory and pulse train its closed form
        does not consume.
        """
        cases = (
            (
                "bob",
                q.Bob(detector=q.ClickDetector(), receiver=q.DelayInterferometer()),
                ValueError,
                "NullingReceiver",
                "delay interferometer accepted",
            ),
            (
                "security",
                q.PhaseBound(e_phase=0.05),
                NotImplementedError,
                "DiscriminationBound",
                "PhaseBound accepted",
            ),
            (
                "bob",
                q.Bob(
                    detector=q.ClickDetector(dead_time=50e-9),
                    receiver=q.NullingReceiver(),
                ),
                ValueError,
                "closed form",
                "dead time accepted by a closed form",
            ),
            (
                "alice",
                q.Alice(laser=q.Laser()),
                NotImplementedError,
                "emits no pulse train",
                "q.Alice accepted",
            ),
        )
        for field, part, exc, needle, why in cases:
            link = two_state()
            setattr(link, field, part)
            self.assertFails(exc, needle, link.run, msg=why)


class PairViolation(Guarded):
    """
    Eve is priced with no assumption about the source state, and never more cheaply.
    """

    def test_engine_agreement(self):
        """
        The rate is `_core.ekert_rate` on the pair model's coincidence gain and error rate, and the two model
        legs beside it are `ekert_point`'s.
        """
        res = violated().run()
        info = shown(res.explain)
        point = _core.ekert_point(res.gain, res.qber, 1.22, 2.0 / 9.0)

        self.assertEqual(
            res.key_rate,
            _core.ekert_rate(res.gain, res.qber, 2.7, 1.22, 2.0 / 9.0, "measured"),
            msg=f"key_rate {res.key_rate} is not ekert_rate's",
        )
        self.assertEqual(info["s"], 2.7, msg=f"s {info['s']}")
        self.assertEqual(info["key_model"], point[1], msg=f"key_model {info['key_model']}")
        self.assertEqual(info["key_symmetry"], point[2], msg=f"key_symmetry {info['key_symmetry']}")
        self.assertEqual(info["chsh"], point[0], msg=f"chsh {info['chsh']}")
        self.assertClose(info["sift"], 2.0 / 9.0, msg=f"sift {info['sift']}")

    def test_violation_stands_alone(self):
        """
        `e_phase` is None and labelled absent, the claimed S round-trips, and the plan denies device
        independence.
        """
        res = violated().run()

        self.assertIsNone(res.e_phase, msg=f"e_phase {res.e_phase}")
        self.assertEqual(
            res.explain["e_phase"]["label"],
            "absent",
            msg="e_phase is not labelled absent",
        )
        self.assertEqual(res.s, 2.7, msg=f"s {res.s}")
        self.assertEqual(
            res.explain["device_independent"]["value"],
            False,
            msg="device_independent is not False",
        )

    def test_price_is_worse(self):
        """
        The CHSH-priced rate sits below the basis-symmetry rate on the same coincidences, and its zero-key
        error rate is 0.0714918 against 11.003%.
        """
        res = violated(f=1.0).run()
        info = shown(res.explain)

        self.assertLess(
            info["key_model"],
            info["key_symmetry"],
            msg="CHSH price undercut the symmetry one",
        )
        self.assertClose(
            info["e_threshold"],
            0.0714918,
            atol=1e-6,
            msg=f"e_threshold {info['e_threshold']}",
        )
        self.assertGreater(
            _core.ekert_threshold(1.0),
            _core.ekert_threshold(1.22),
            msg="a worse code did not lower the threshold",
        )

    def test_source_refused(self):
        """
        A modelled violation, a device-independent claim, an unknown origin and an S past Tsirelson's bound
        are each refused with the engine's own message, while S = 2 is priced at zero key.
        """
        register = (
            (
                "modelled",
                ValueError,
                ("modelled", "circular", "ekert_point", "pair_rate"),
            ),
            (
                "device-independent",
                NotImplementedError,
                ("semidefinite", "entropy accumulation", "fair sampling", "0.924"),
            ),
            ("observed", ValueError, ("device-independent",)),
        )
        for source, exc, needles in register:
            for needle in needles:
                self.assertFails(
                    exc,
                    needle,
                    violated(source=source).run,
                    msg=f"{source} refusal did not name {needle}",
                )

        self.assertFails(
            ValueError,
            "Tsirelson",
            violated(s=2.9).run,
            msg="s = 2.9 was priced",
        )
        self.assertEqual(
            violated(s=2.0).run().key_rate,
            0.0,
            msg="s = 2.0 carried key",
        )


class IntensityMonitor(Question):
    """
    The monitoring visibility is never a security input.
    """

    def test_engine_agreement(self):
        """
        The data line's gain is the Poisson forward model at T*(1 - split)*eta thinned by the monitoring
        sequences, and the rate is `cow_rate` on that pair with the supplied bound.
        """
        res = cow().run()
        info = shown(res.explain)
        data = 10.0 ** (-0.2 * 25.0 / 10.0) * 0.9 * ETA_COW
        gain, e_bit = _core.decoy_gain(0.5, data, background(D_COW), E_ALIGN)

        self.assertClose(info["t_data"], data, msg=f"t_data {info['t_data']}")
        self.assertEqual(res.qber, e_bit, msg=f"qber {res.qber} vs {e_bit}")
        self.assertClose(res.p_click, 0.9 * gain, msg=f"p_click {res.p_click}")
        self.assertEqual(
            res.key_rate,
            _core.cow_rate(0.9 * gain, e_bit, 0.2, F_COW),
            msg=f"key_rate {res.key_rate} is not cow_rate's",
        )

    def test_phase_bound_floor(self):
        """
        A bound below the multiphoton fraction 1 - exp(-mu)*(1 + mu), 9.02% at mu = 0.5, is refused at run and
        a bound of zero at construction.
        """
        self.assertFails(
            ValueError,
            "e_phase",
            lambda: q.PhaseBound(e_phase=0.0),
            msg="e_phase = 0 constructed",
        )
        self.assertFails(
            ValueError,
            "multiphoton fraction",
            cow(phase=0.05).run,
            msg="e_phase = 0.05 ran below the 9.02% fraction",
        )
        self.assertGreater(
            cow(phase=0.1).run().key_rate,
            0.0,
            msg="e_phase = 0.1 carried no key",
        )

    def test_visibility_is_not_input(self):
        """
        q.PhaseBound requires an explicit bound and no visibility row enters the rate.
        """
        with self.assertRaises(TypeError):
            q.PhaseBound()

        info = shown(cow().run().explain)
        self.assertNotIn("visibility", info, msg="a visibility row reached the rate")
        self.assertClose(info["e_phase"], 0.2, msg=f"e_phase {info['e_phase']}")

    def test_split_costs(self):
        """
        A bigger monitoring tap lowers the gain and the rate with it, and the rate stays under the ceiling of
        one bit per detected signal at every distance.
        """
        runs = [cow(split=s).run() for s in (0.05, 0.1, 0.3)]

        self.assertMonotone([r.p_click for r in runs], rising=False, msg=f"gain {[r.p_click for r in runs]}")
        self.assertMonotone([r.key_rate for r in runs], rising=False, msg=f"rate {[r.key_rate for r in runs]}")

        for dist in (1.0, 25.0, 100.0):
            res = cow(dist=dist).run()
            self.assertLess(
                res.key_rate,
                shown(res.explain)["ceiling"],
                msg=f"{dist} km: rate {res.key_rate} above the ceiling",
            )


class RelayGaussian(Guarded):
    """
    Trusted/untrusted is the topology here rather than a flag on a detector.
    """

    def test_engine_agreement(self):
        """
        Every reported number is bit for bit what `_core.cvmdi_noise`, `cvmdi_floor`, `cvmdi_least`,
        `cvmdi_rate` and `cvmdi_point` return for the two arms the report names.
        """
        res = bell(la=0.5, lb=8.0, eta=0.9, vel=0.02).run()
        info = shown(res.explain)
        ta, tb = info["tau_a"], info["tau_b"]
        wa, wb = info["omega_a"], info["omega_b"]
        chi = _core.cvmdi_noise(ta, tb, wa, wb, 0.0, 0.0)
        point = _core.cvmdi_point(1e5, ta, tb, wa, wb, 0.0, 0.0, 0.98)

        self.assertEqual(res.chi, chi, msg=f"chi {res.chi} vs {chi}")
        self.assertEqual(res.floor, _core.cvmdi_floor(ta, tb), msg=f"floor {res.floor}")
        self.assertEqual(res.least, _core.cvmdi_least(ta, tb), msg=f"least {res.least}")
        self.assertEqual(
            res.key_rate,
            max(0.0, _core.cvmdi_rate(ta, tb, chi)),
            msg=f"key_rate {res.key_rate} is not cvmdi_rate's at the observed chi",
        )
        self.assertEqual(
            (res.attack.i_ab, res.attack.chi_e, res.attack.key_rate),
            point,
            msg="the per-attack triple is not cvmdi_point's",
        )

    def test_relay_belongs_to_eve(self):
        """
        A relay detector has no trusted variant: eta folds into the arm as loss and v_el as noise, and an
        input variance through arm-then-detector lands on what the reported (tau, omega) predicts.
        """
        ideal = shown(bell(eta=1.0, vel=0.0).run().explain)
        real = shown(bell(eta=0.7, vel=0.05).run().explain)

        with self.assertRaises(TypeError):
            q.BellDetector(eta=0.9, v_el=0.01, trusted=True)
        self.assertEqual(ideal["tau_a"], ideal["T_a"], msg=f"ideal tau_a {ideal['tau_a']}")
        self.assertEqual(
            real["tau_a"],
            0.7 * real["T_a"],
            msg=f"tau_a {real['tau_a']} is not eta*T_a",
        )

        for v in (1.0, 7.3):
            through = real["T_a"] * v + (1.0 - real["T_a"]) * 1.0
            seen = 0.7 * through + 0.3 + 0.05
            self.assertClose(
                real["tau_a"] * v + (1.0 - real["tau_a"]) * real["omega_a"],
                seen,
                atol=1e-12,
                msg=f"folded arm at V={v} missed {seen}",
            )

    def test_floor_is_not_least(self):
        """
        Pure loss lands chi on `cvmdi_floor`, `cvmdi_least` sits strictly below it (exactly 4 for equal arms
        whatever the loss), and excess noise pushes chi above the floor.
        """
        pure = bell(la=2.0, lb=2.0).run()
        noisy = bell(
            arms=(
                q.Channel(T=0.9, xi=0.02, ref="input"),
                q.Channel(T=0.9, xi=0.02, ref="input"),
            )
        ).run()

        self.assertClose(pure.chi, pure.floor, msg=f"pure chi {pure.chi} vs floor {pure.floor}")
        self.assertClose(pure.least, 4.0, msg=f"least {pure.least}")
        self.assertLess(pure.least, pure.floor, msg=f"least {pure.least} not below floor {pure.floor}")
        self.assertGreater(noisy.chi, noisy.floor, msg=f"noisy chi {noisy.chi} not above floor")
        self.assertGreater(noisy.chi, noisy.least, msg=f"noisy chi {noisy.chi} below least")

    def test_correlated_eve(self):
        """
        A correlated environment of the helping sign drives chi below its pure-loss floor and raises the rate;
        one the uncertainty principle cannot support is refused.
        """
        arms = (
            q.Channel(T=0.9, xi=0.05, ref="input"),
            q.Channel(T=0.5, xi=0.05, ref="input"),
        )
        alone = bell(arms=arms).run()
        helped = bell(arms=arms, env=q.CorrelatedEnvironment(x=0.3, p=-0.3)).run()

        self.assertGreater(alone.chi, alone.floor, msg=f"uncorrelated chi {alone.chi} not above floor")
        self.assertLess(helped.chi, helped.floor, msg=f"helped chi {helped.chi} not under floor")
        self.assertGreater(helped.key_rate, alone.key_rate, msg=f"helped rate {helped.key_rate}")
        self.assertFails(
            ValueError,
            "uncertainty principle",
            bell(arms=arms, env=q.CorrelatedEnvironment(x=0.9, p=-0.9)).run,
            msg="x = 0.9 was priced",
        )

    def test_corr_refers(self):
        """
        The CorrelatedEnvironment is rescaled by 3/7 and the folded relay reproduces the four-mode model to
        1e-11, where unreferred chi reads 9.6 against 12.8 and the rate rises by 0.5514 bit per use.
        """
        arms = (
            q.Channel(T=0.5, xi=0.3, ref="input"),
            q.Channel(T=0.5, xi=0.3, ref="input"),
        )
        env = q.CorrelatedEnvironment(x=0.6, p=-0.6)
        res = bell(arms=arms, eta=0.6, vel=0.05, va=4.0, env=env).run()
        info = shown(res.explain)
        ta, tb = info["tau_a"], info["tau_b"]
        wa, wb = info["omega_a"], info["omega_b"]
        want = midpoint(4.0, ((0.5, 1.3), (0.5, 1.3)), (0.6, -0.6), 0.6, 0.05)
        got = _core.cvmdi_cov(4.0, ta, tb, wa, wb, info["g"], info["gp"])
        naive = _core.cvmdi_cov(4.0, ta, tb, wa, wb, 0.6, -0.6)
        loose = _core.cvmdi_noise(ta, tb, wa, wb, 0.6, -0.6)

        self.assertClose(
            info["g_scale"],
            3.0 / 7.0,
            msg=f"g_scale {info['g_scale']}, eta*sqrt((1-T_a)(1-T_b)/((1-tau_a)(1-tau_b))) is 3/7 here",
        )
        self.assertClose(info["g"], 0.6 * 3.0 / 7.0, msg=f"g {info['g']}")

        self.gridClose(
            np.array(got).reshape(4, 4),
            want,
            atol=1e-11,
            msg="the folded relay is not the physical one",
        )
        self.assertGreater(
            float(np.max(np.abs(np.array(naive).reshape(4, 4) - want))),
            0.1,
            msg="the unreferred correlation matched the physical model",
        )
        self.assertClose(res.chi, 12.8, msg=f"chi {res.chi}")
        self.assertClose(loose, 9.6, msg=f"unreferred chi {loose}")
        self.assertClose(
            _core.cvmdi_rate(ta, tb, res.chi),
            -2.8336431665,
            atol=1e-9,
            msg="referred rate",
        )
        self.assertClose(
            _core.cvmdi_rate(ta, tb, loose) - _core.cvmdi_rate(ta, tb, res.chi),
            0.5514,
            atol=1e-3,
            msg="unreferred minus referred rate",
        )

    def test_corr_not_refused(self):
        """
        Unequal arms fold to the four-mode model to 1e-11, and the same correlation unreferred is refused by the
        uncertainty principle.
        """
        arms = (
            q.Channel(T=0.9, xi=1.0 / 30.0, ref="input"),
            q.Channel(T=0.2, xi=1.2, ref="input"),
        )
        env = q.CorrelatedEnvironment(x=0.6, p=-0.6)
        info = shown(bell(arms=arms, eta=0.6, vel=0.05, va=4.0, env=env).run().explain)
        ta, tb = info["tau_a"], info["tau_b"]
        wa, wb = info["omega_a"], info["omega_b"]
        want = midpoint(4.0, ((0.9, 1.3), (0.2, 1.3)), (0.6, -0.6), 0.6, 0.05)
        got = _core.cvmdi_cov(4.0, ta, tb, wa, wb, info["g"], info["gp"])

        self.gridClose(
            np.array(got).reshape(4, 4),
            want,
            atol=1e-11,
            msg="unequal arms folded differently",
        )
        self.assertBad(
            "uncertainty principle",
            _core.cvmdi_noise,
            (ta, tb, wa, wb, 0.6, -0.6),
            msg="the unreferred correlation was accepted",
        )

    def test_position_matters(self):
        """
        chi is symmetric under swapping the arms and the rate is not: away from Alice costs more than away
        from Bob, and the rate falls monotonically with reach.
        """
        near = bell(la=0.1, lb=20.0).run()
        far = bell(la=20.0, lb=0.1).run()
        walk = [bell(la=0.1, lb=d).run().key_rate for d in (1.0, 5.0, 20.0, 40.0)]

        self.assertClose(near.chi, far.chi, msg=f"chi {near.chi} vs {far.chi}")
        self.assertGreater(near.key_rate, far.key_rate, msg=f"near {near.key_rate}, far {far.key_rate}")
        self.assertMonotone(walk, rising=False, msg=f"rate over reach: {walk}")

    def test_guards(self):
        """
        Asymmetric modulation, a finite-size reduction and a pulse width no formula consumes are each refused
        by this topology.
        """
        cases = (
            (
                "alice",
                q.Sender(modulation=q.GaussianModulation(v_a=2e5)),
                NotImplementedError,
                "one modulation variance",
                "asymmetric modulation averaged",
            ),
            (
                "security",
                q.FiniteSize(),
                NotImplementedError,
                "q.TwoModeBound(block=q.GaussianBlock(...))",
                "q.FiniteSize accepted",
            ),
            (
                "bob",
                q.Sender(modulation=q.GaussianModulation(v_a=1e5), pulse=100e-12),
                ValueError,
                "accepted and dropped",
                "pulse width accepted",
            ),
        )
        for field, part, exc, needle, why in cases:
            link = bell()
            setattr(link, field, part)
            self.assertFails(exc, needle, link.run, msg=why)


class DiscretePhase(Question):

    def test_engine_agreement(self):
        """
        The rate is exactly `_core.dm_rate` and the reported mutual information `dm_info`'s at the effective
        channel, pinning the untrusted substitution T -> eta*T, xi -> xi + 2*v_el/(eta*T).
        """
        res = keyed(states=4, alpha=0.4, t=0.5, xi=0.02, eta=0.6, vel=0.1).run()
        info = shown(res.explain)
        i_ab, chi_be, key, z_star = _core.dm_rate(4, 0.4, 0.5, 0.02, 0.6, 0.1, 0.95)
        exact, gauss = _core.dm_info(4, 0.4, info["T_eff"], info["xi_eff"])

        self.assertClose(info["T_eff"], 0.3, msg=f"T_eff {info['T_eff']}")
        self.assertClose(info["xi_eff"], 0.02 + 2.0 * 0.1 / 0.3, msg=f"xi_eff {info['xi_eff']}")
        self.assertEqual(res.key_rate, max(0.0, key), msg=f"key_rate {res.key_rate} is not dm_rate's")
        self.assertEqual(res.i_ab, i_ab, msg=f"i_ab {res.i_ab}")
        self.assertEqual(res.chi_be, chi_be, msg=f"chi_be {res.chi_be}")
        self.assertEqual(info["z_star"], z_star, msg=f"z_star {info['z_star']}")
        self.assertEqual(res.i_ab, exact, msg=f"i_ab {res.i_ab} vs dm_info exact {exact}")
        self.assertEqual(info["i_ab_gauss"], gauss, msg=f"i_ab_gauss {info['i_ab_gauss']}")

    def test_both_informations(self):
        """
        Both informations are reported and beta*gauss - chi_BE rebuilds the published rate, the gap under 1%
        at the operating point and over 10% at a bright eight-state ring.
        """
        run = shown(keyed().run().explain)
        bright = shown(keyed(states=8, alpha=1.0, t=0.95, xi=0.005).run().explain)

        self.assertGreater(run["i_ab_gauss"], run["i_ab"], msg=f"i_ab_gauss {run['i_ab_gauss']} vs i_ab {run['i_ab']}")
        self.assertClose(
            run["key_gauss"],
            0.95 * run["i_ab_gauss"] - run["chi_be"],
            msg=f"key_gauss {run['key_gauss']}",
        )
        self.assertLess(
            run["key_gauss"] / run["key_raw"] - 1.0,
            0.01,
            msg="gap above 1% at the operating point",
        )
        self.assertGreater(
            bright["key_gauss"] / bright["key_raw"] - 1.0,
            0.1,
            msg="gap below 10% at the bright ring",
        )

    def test_bits_is_required(self):
        """
        `dm_holevo` requires the alphabet entropy, `holevo()` passes log2(m), and the capped value is at most
        the unbounded one.
        """
        mod = q.PhaseShiftKeying(states=4, alpha=0.4)
        capped = mod.holevo(0.5, 0.01, 0.5, 0.95)
        free = _core.dm_holevo(mod.v_a, 0.5, 0.01, 0.5, float("inf"), 0.95)

        with self.assertRaises(TypeError):
            _core.dm_holevo(mod.v_a, 0.5, 0.01, 0.5)
        self.assertClose(mod.bits, 2.0, msg=f"bits {mod.bits}")
        self.assertEqual(
            capped,
            _core.dm_holevo(mod.v_a, 0.5, 0.01, 0.5, 2.0, 0.95),
            msg="holevo() did not pass log2(m)",
        )
        self.assertLessEqual(capped[0], free[0], msg=f"capped {capped[0]} vs free {free[0]}")

    def test_discrete_costs(self):
        """
        z_star sits below the Gaussian value, the rate below the Gaussian-modulation rate at the same
        variance, and the gap closes as states are added.
        """
        res = keyed()
        info = shown(res.run().explain)
        gauss = _core.cv_rate(info["v_a"], 0.5, 0.01, 1.0, 0.0, 0.95, False, False)
        more = [keyed(states=m).run().key_rate for m in (4, 5, 6)]

        self.assertLess(info["z_star"], info["z_gauss"], msg=f"z_star {info['z_star']} vs z_gauss {info['z_gauss']}")
        self.assertLess(info["key_raw"], gauss[2], msg=f"key_raw {info['key_raw']} vs gaussian {gauss[2]}")
        self.assertMonotone(more, rising=True, msg=f"rate over states: {more}")
        self.assertClose(info["v_a"], 2.0 * 0.4**2, msg=f"v_a {info['v_a']}")

    def test_effective_plane(self):
        """
        z_star, i_ab and chi_be rebuild bit for bit from the reported T_eff and xi_eff alone.
        """
        link = keyed(states=6, alpha=0.5, t=0.302, xi=0.02, eta=0.6, vel=0.1, beta=0.9)
        mod = link.modulation
        info = shown(link.run().explain)
        t_eff = info["T_eff"]
        xi_eff = info["xi_eff"]
        _va, z_lin, w, _zg = _core.dm_moments(mod.states, mod.alpha)
        z = max(0.0, math.sqrt(t_eff) * (z_lin - math.sqrt(2.0 * xi_eff * w)))
        exact = _core.dm_info(mod.states, mod.alpha, t_eff, xi_eff)[0]
        holevo = _core.dm_holevo(mod.v_a, t_eff, xi_eff, z, mod.bits, 0.9)

        self.assertEqual(z, info["z_star"], msg=f"z_star {info['z_star']} vs rebuilt {z}")
        self.assertLess(exact, mod.bits, msg=f"exact {exact} at the cap {mod.bits}")
        self.assertEqual(exact, info["i_ab"], msg=f"i_ab {info['i_ab']} vs rebuilt {exact}")
        self.assertEqual(holevo[1], info["chi_be"], msg=f"chi_be {info['chi_be']} vs rebuilt {holevo[1]}")

    def test_constellation(self):
        """
        `constellation()` is `states` amplitudes of modulus alpha equally spaced from the real axis, as
        `_core.dm_states` returns them.
        """
        mod = q.PhaseShiftKeying(states=6, alpha=0.5)
        ring = mod.constellation()

        self.assertEqual(len(ring), 6, msg=f"ring size {len(ring)}")
        self.assertClose(ring[0].real, 0.5, msg=f"ring[0] {ring[0]}")

        for point in ring:
            self.assertClose(abs(point), 0.5, msg=f"modulus {abs(point)}")
        self.assertClose(
            ring[1].imag,
            _core.dm_states(6, 0.5)[1][1],
            msg=f"ring[1].imag {ring[1].imag}",
        )

    def test_information_wrapper(self):
        """
        `information(T, xi)` returns (exact, Gaussian) = (0.1107575, 0.1107648), bit for bit
        `_core.dm_info(4, 0.4, 0.5, 0.01)`, transposed slots give 0.0023007, and domain errors reach the
        caller; tree values, not a published anchor.
        """
        mod = q.PhaseShiftKeying(states=4, alpha=0.4)
        exact, gauss = mod.information(T=0.5, xi=0.01)

        self.assertClose(exact, 0.1107575, atol=5e-8, msg=f"exact {exact}")
        self.assertClose(gauss, 0.1107648, atol=5e-8, msg=f"gauss {gauss}")
        self.assertEqual(
            (exact, gauss),
            _core.dm_info(4, 0.4, 0.5, 0.01),
            msg="information() is not dm_info at the same constellation",
        )
        self.assertEqual(mod.information(0.5, 0.01), (exact, gauss), msg="positional and keyword disagree")

        swapped = mod.information(0.01, 0.5)
        self.assertEqual(
            swapped,
            _core.dm_info(4, 0.4, 0.01, 0.5),
            msg=f"swapped {swapped}",
        )
        self.assertLess(swapped[0], 0.5 * exact, msg=f"swapped exact {swapped[0]}")

        wide = q.PhaseShiftKeying(states=6, alpha=0.5)
        self.assertEqual(
            wide.information(0.5, 0.01),
            _core.dm_info(6, 0.5, 0.5, 0.01),
            msg="the wide ring was defaulted",
        )
        self.assertGreater(
            wide.information(0.5, 0.01)[0],
            exact,
            msg=f"the wide ring carries no more than {exact}",
        )
        self.assertGreaterEqual(gauss, exact, msg=f"gauss {gauss} below exact {exact}")

        for args, needle in (((0.0, 0.01), "t must be in (0, 1]"), ((0.5, -0.01), "xi must be >= 0")):
            self.assertFails(
                ValueError,
                needle,
                lambda a=args: mod.information(*a),
                msg=f"domain at {args}",
            )

    def test_certified_reach(self):
        """
        The numerical proof certifies key where the closed form is negative, and the report carries both
        numbers side by side.
        """
        res = certified(t=0.5, xi=0.01, alpha=0.7).run()
        info = shown(res.explain)

        self.assertLess(info["key_analytic"], 0.0, msg=f"key_analytic {info['key_analytic']}")
        self.assertGreater(res.key_rate, 0.0, msg=f"key_rate {res.key_rate}")
        self.assertClose(res.key_rate, res.certificate.key, msg=f"key_rate {res.key_rate} vs {res.certificate.key}")
        self.assertClose(info["key_raw"], res.certificate.key, msg=f"key_raw {info['key_raw']}")

    def test_steps_apart(self):
        """
        `bound` is step 2 and is the proof, `upper` step 1's Frank-Wolfe value labelled a diagnostic, and the
        certificate refuses a key_rate; before a run those rows and the solver read unrun.
        """
        res = certified().run()
        info = res.explain
        plan = certified().explain()

        self.assertEqual(info["upper"]["label"], "diagnostic", msg="upper is not labelled diagnostic")
        self.assertGreater(
            info["bound"]["value"],
            info["upper"]["value"],
            msg="bound equals upper",
        )

        with self.assertRaises(AttributeError):
            res.certificate.key_rate
        self.assertEqual(plan["solver"], "unrun", msg=f"plan solver {plan['solver']}")
        self.assertEqual(
            plan["bound"]["label"],
            "derived (run to compute)",
            msg="the plan filled bound",
        )
        self.assertEqual(plan["cutoff"]["label"], "pinned", msg="cutoff is not pinned")

    def test_trusted_plane(self):
        """
        A trusted receiver reaches `_core.dm_trusted` at the channel plane with efficiency and electronic
        noise held apart, never folded into the transmittance.
        """
        res = receiver(True).run()
        want = _core.dm_trusted(4, 0.7, 0.5, 0.01, 6, 0.5, 0.05, 0.0, 0.0, 0.95, 1e-10, 6)

        self.assertClose(res.key_rate, want[0], msg=f"key_rate {res.key_rate} vs dm_trusted {want[0]}")
        self.assertClose(res.certificate.bound, want[1], msg=f"bound {res.certificate.bound} vs {want[1]}")
        self.assertClose(res.certificate.upper, want[2], msg=f"upper {res.certificate.upper} vs {want[2]}")

    def test_trusted_reach(self):
        """
        The untrusted branch refers v_el back as xi + 2*v_el/(eta*T) and certifies nothing at T = 0.5, 0.2 or
        0.05, where the trusted one certifies key at all three.
        """
        for t in (0.5, 0.2, 0.05):
            hot = receiver(True, t=t).run()
            cold = receiver(False, t=t).run()
            self.assertGreater(hot.key_rate, 0.0, msg=f"T = {t}: trusted key {hot.key_rate}")
            self.assertEqual(cold.key_rate, 0.0, msg=f"T = {t}: untrusted key {cold.key_rate}")
            self.assertLess(cold.certificate.key, 0.0, msg=f"T = {t}: untrusted certificate {cold.certificate.key}")

    def test_trusted_rows(self):
        """
        The trusted report drops T_eff, xi_eff, key_analytic and key_gauss and returns i_ab and chi_be as
        None, keeping the claimed plane, the trusted flag and the certificate's rows.
        """
        res = receiver(True).run()
        info = res.explain
        for row in ("T_eff", "xi_eff", "key_analytic", "key_gauss", "i_ab", "chi_be"):
            self.assertNotIn(row, info, msg=f"{row} on the trusted report")

        self.assertIsNone(res.i_ab, msg=f"i_ab {res.i_ab}")
        self.assertIsNone(res.chi_be, msg=f"chi_be {res.chi_be}")
        self.assertTrue(info["trusted"]["value"], msg="the trusted row is not set")
        self.assertIn("T_claimed", info, msg="T_claimed dropped")
        self.assertEqual(info["solver"], res.certificate.status, msg=f"solver {info['solver']}")
        self.assertEqual(receiver(True).explain()["solver"], "unrun", msg="the plan solved")

    def test_guards(self):
        """
        The closed form refuses a trusted receiver, a homodyne and a finite block, and a two-state
        constellation is refused at construction.
        """
        cases = (
            (
                q.Heterodyne(trusted=True),
                q.Asymptotic(),
                "trusted-detector form",
                "trusted=True accepted by the closed form",
            ),
            (
                q.Homodyne(trusted=False),
                q.Asymptotic(),
                "heterodyne",
                "homodyne accepted",
            ),
            (
                q.Heterodyne(trusted=False),
                q.FiniteSize(),
                "asymptotic",
                "q.FiniteSize accepted",
            ),
        )
        for det, sec, needle, why in cases:
            link = q.Link(
                modulation=q.PhaseShiftKeying(),
                channel=q.Channel(T=0.5),
                bob=q.Bob(detector=det),
                security=sec,
            )
            self.assertFails(NotImplementedError, needle, link.run, msg=why)

        self.assertFails(
            ValueError,
            "at least 3",
            lambda: q.PhaseShiftKeying(states=2),
            msg="states=2 constructed",
        )


if __name__ == "__main__":
    rc = Exam(
        "ProtocolPhase",
        "Phase 5b API: the phase-keyed family with its QBER simulated, not pinned",
        "protocols_phase.md",
    ).run(load(PhaseKeyed))
    rc |= Exam(
        "ProtocolBasis",
        "Phase 5b API: basis-keyed weak coherent pulses with decoy-state bounds",
        "protocols_basis.md",
    ).run(load(BasisDecoy))
    rc |= Exam(
        "ProtocolVariants",
        "Basis keying with a third basis, and with a pair announced in place of one",
        "protocols_variants.md",
    ).run(load(BasisVariants))
    rc |= Exam(
        "ProtocolTwoState",
        "Two non-orthogonal states, and a phase error derived rather than dialled",
        "protocols_two_state.md",
    ).run(load(TwoState))
    rc |= Exam(
        "ProtocolViolation",
        "A photon-pair link priced by the Bell violation, and what that costs",
        "protocols_violation.md",
    ).run(load(PairViolation))
    rc |= Exam(
        "ProtocolIntensity",
        "Phase 5b API: intensity keying, its monitoring tap and its phase bound",
        "protocols_cow.md",
    ).run(load(IntensityMonitor))
    rc |= Exam(
        "ProtocolRelayGaussian",
        "Phase 5c API: the continuous-variable relay, where the detector is Eve's",
        "protocols_relay_cv.md",
    ).run(load(RelayGaussian))
    rc |= Exam(
        "ProtocolDiscrete",
        "Phase 5c API: phase-shift keying and the correlation a finite alphabet cedes",
        "protocols_discrete.md",
    ).run(load(DiscretePhase))
    rc |= Exam(
        "ProtocolAttack",
        "The class of attack each bound is proved against, quoted per family",
        "protocols_attack.md",
    ).run(load(AttackClass))
    sys.exit(rc)
