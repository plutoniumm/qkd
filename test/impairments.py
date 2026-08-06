import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.cache import memo
from kit.checks import Guarded
from kit.fuzzing import rng
from qkd import _core, budget, impairments

RAMAN = q.Coexistence(channels=4, launch=1e-3)

# One descriptor per Gaussian-modulation excess-noise form q.Link consumes.
ROWS = (
    RAMAN,
    q.Modulator(ratio=0.97, angle=0.05),
    q.Polarisation(drift=0.12),
    q.Timing(jitter=4e-11, width=5e-10),
)


def metro(**kw):
    return impairments.assemble_extra(v_a=5.0, t=10**-0.5, length=25.0, **kw)


def span(imp=(), dsp=False, alice=None):
    """
    A 25 km metro link; ``dsp`` puts the pipeline in the loop, ``alice`` supplies a symbol period alone.
    """
    lo = None
    chain = None
    if dsp:
        alice = q.Alice(
            laser=q.Laser(linewidth=1e4),
            pilots=q.Pilots(power_db=12.0),
            symbol_rate=100e6,
        )
        lo = q.LocalLO(linewidth=1e4)
        chain = q.DSP()

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Fiber(length=25.0),
        alice=alice,
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True), lo=lo),
        dsp=chain,
        security=q.Asymptotic(beta=0.95),
        impairments=imp,
    )


@memo
def frames(symbols=100_000, seed=1):
    """
    The pipeline half of that link at (symbols, seed), computed once.
    """

    return span(dsp=True).measure(symbols, seed)


def front(rin=None, band=None, carrier=None, local=None):
    """
    The metro link with a laser RIN in dBc/Hz, a detection bandwidth in Hz and two optical carriers in Hz.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Fiber(length=25.0),
        alice=q.Alice(
            laser=q.Laser(linewidth=1e4, rin=rin, carrier=carrier),
            pilots=q.Pilots(power_db=12.0),
            symbol_rate=100e6,
        ),
        bob=q.Bob(
            detector=q.Heterodyne(eta=0.6, v_el=0.1, bandwidth=band),
            lo=q.LocalLO(linewidth=1e4, carrier=local),
        ),
        dsp=q.DSP(),
        security=q.Asymptotic(beta=0.95),
    )


def pinned(imp=()):
    """
    The metro link pinned at T = 0.5 and xi = 0.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=0.0, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True)),
        security=q.Asymptotic(beta=0.95),
        impairments=imp,
    )


# Large enough that a 1e-4 per-gate floor moves the observed QBER.
SYMBOLS = 200000


def lit(passband, launch=-20.0, imp=None, channel=None):
    """
    A phase-keyed click link beside one coexisting classical channel.
    """
    if imp is None:
        imp = (q.Coexistence(channels=1, launch=launch, wavelength=1531.12e-9),)

    return q.Link(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=channel or q.Fiber(length=25.0),
        alice=q.Alice(laser=q.Laser(linewidth=1e6), symbol_rate=1e9),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.2, dark=1e-6, passband=passband),
            receiver=q.DelayInterferometer(delay=1, visibility=0.98),
        ),
        security=q.IndividualAttack(f=1.16),
        impairments=imp,
    )


def keyed(det, alice=None):
    """
    A basis-keyed link over ``det``, sampled only when ``alice`` is given.
    """

    return q.Link(
        modulation=q.BasisKeying(decoy=q.Decoy(intensities=(0.5, 0.1, 0.0))),
        channel=q.Fiber(length=30.0),
        alice=alice,
        bob=q.Bob(detector=det, receiver=q.BasisAnalyser(misalign=0.015)),
        security=q.SplittingAttack(f=1.16),
    )


def phased(det, qber=None):
    """
    A phase-keyed link over ``det``, sampled unless ``qber`` is pinned.
    """
    pinned = qber is not None
    optics = None if pinned else q.DelayInterferometer(delay=1, visibility=0.98)
    source = None if pinned else q.Alice(laser=q.Laser(linewidth=1e4), symbol_rate=1e9)

    return q.Link(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Fiber(length=25.0),
        alice=source,
        bob=q.Bob(detector=det, receiver=optics),
        security=q.IndividualAttack(qber=qber, f=1.16),
    )


def fringe(lw, delay=1, contrast=0.98):
    """
    A sampled phase-keyed run on a 1 GHz clock; ``lw`` must be wide enough for the laser walk to set the contrast.
    """

    return q.Link(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Fiber(length=25.0),
        alice=q.Alice(laser=q.Laser(linewidth=lw), symbol_rate=1e9),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.2, dark=1e-6),
            receiver=q.DelayInterferometer(delay=delay, visibility=contrast),
        ),
        security=q.IndividualAttack(f=1.16),
    ).run(symbols=400_000, seed=11)


class Optical(Question):
    """
    Impairments with an excess-noise form.
    """

    def test_raman_anchor(self):
        """
        Published anchor: Kumar, Qin and Alleaume (arXiv:1412.1403) Eq. (6) at 0 dBm, 25 km,
        beta = 3e-9 /(km nm), 1531.12 nm and 0.5 dB add/drop reproduces their 1.3e-3 SNU forward and 1.6e-3
        backward at Bob to 3%.
        """
        fwd = 2.0 * impairments.raman_photons(1e-3, 25.0, beta=3e-9, wavelength=1531.12e-9, demux=0.891)
        bwd = 2.0 * impairments.raman_photons(1e-3, 25.0, beta=3e-9, wavelength=1531.12e-9, demux=0.891, backward=True)
        self.assertClose(fwd, 1.3e-3, atol=4e-5, msg=f"forward {fwd:.4e} vs 1.3e-3")
        self.assertClose(bwd, 1.6e-3, atol=4e-5, msg=f"backward {bwd:.4e} vs 1.6e-3")

    def test_raman_growth(self):
        """
        Raman xi grows monotonically with launch power and linearly with channel count.
        """
        powers = [metro(coexist=impairments.Coexistence(1, p))[0].xi for p in range(-6, 13, 3)]
        self.assertEqual(powers, sorted(powers), msg=f"raman xi {powers}")

        counts = [metro(coexist=impairments.Coexistence(n, 0.0))[0].xi for n in (1, 2, 4, 8)]
        self.assertEqual(counts, sorted(counts), msg=f"raman xi {counts}")
        self.assertClose(counts[1], 2.0 * counts[0], atol=1e-12, msg="raman xi not linear in n")

    def test_raman_plane(self):
        """
        The Raman entry reads xi_input = xi_Bob / T, and the demux cancels between Kumar Eqs. (6) and (7).
        """
        t = 10**-0.5
        entries = metro(coexist=impairments.Coexistence(1, 0.0, demux=0.891))
        bud = budget.Budget(entries=entries, T=t)
        bob = 2.0 * impairments.raman_photons(1e-3, 25.0)
        self.assertClose(bud.at("bob")["raman"], bob, atol=1e-12, msg="Bob-plane raman")
        self.assertClose(bud.at("input")["raman"], bob / t, atol=1e-12, msg="input plane is /T")

        plain = metro(coexist=impairments.Coexistence(1, 0.0))
        self.assertClose(plain[0].xi, entries[0].xi, atol=1e-15, msg="eta_D did not cancel")

    def test_rayleigh_saturates(self):
        """
        The Rayleigh return fraction rises with length to coeff/(alpha v_g), about -31 dB for SMF-28 at 1550 nm
        and less for ULL (arXiv:2407.08009), and q.Backscatter charges assemble_extra's row alone, input-referred
        noise unmoved by a receive coupling and raised by exactly the decibels of a launch coupling.
        """
        lengths = [impairments.rayleigh_fraction(x) for x in (1.0, 10.0, 50.0, 200.0)]
        self.assertEqual(lengths, sorted(lengths), msg=f"return fractions {lengths}")

        far = impairments.rayleigh_fraction(1e4)
        self.assertLess(far, 1.5e-3, msg=f"saturated return {far:.3e} above -28 dB")
        self.assertGreater(far, 5e-4, msg=f"saturated return {far:.3e} below -33 dB")

        ull = impairments.rayleigh_fraction(50.0, coeff=6.54)
        self.assertLess(ull, impairments.rayleigh_fraction(50.0), msg=f"ULL return {ull}")

        clock = q.Alice(symbol_rate=100e6)
        base = span(alice=clock).run()
        rows = [span(imp=(q.Backscatter(power=p),), alice=clock).run() for p in (1e-13, 1e-12, 1e-11)]
        seen = [r.budget.at("input")["rayleigh"] for r in rows]
        self.assertClose(
            seen[1],
            impairments.assemble_extra(
                v_a=5.0,
                t=10**-0.5,
                length=25.0,
                symbol=1e-8,
                probe=impairments.Backscatter(1e-12),
            )[0].xi,
            atol=0.0,
            msg="rayleigh row != assemble_extra",
        )
        self.assertMonotone(seen, msg=f"rayleigh rows {seen}")
        self.assertMonotone(
            [base.key_rate] + [r.key_rate for r in rows],
            rising=False,
            msg="rate did not fall",
        )

        step = 10.0 ** (-1.5 / 10.0)
        moved = []
        for site in ("launch", "receive"):
            link = span(imp=(q.Backscatter(power=1e-12),), alice=clock)
            link.losses = (q.Coupling(loss=1.5, site=site),)
            moved.append(link.run().budget.at("input")["rayleigh"])
        self.assertClose(
            moved[1],
            seen[1],
            atol=0.0,
            msg="receive coupling moved the row",
        )
        self.assertClose(
            moved[0],
            seen[1] / step,
            atol=1e-18,
            msg="launch coupling did not raise it",
        )

    def test_dephasing_form(self):
        """
        impairments.dephasing is budget.phase's estimator form, 6.2871e-3 SNU at 20 kHz over 10 ns
        (V_phi = 1.257e-3 rad^2) against the literature form's smaller 6.2812e-3, and q.Dephasing charges that row.
        """
        args = (5.0, 20e3, 1e-8)
        est = impairments.dephasing(*args)
        lit = impairments.dephasing(*args, form="literature")
        v_phi = impairments.phase_variance(20e3, 1e-8)
        self.assertEqual(est, budget.phase(5.0, v_phi), msg="default form != estimator")
        self.assertEqual(
            lit,
            budget.phase(5.0, v_phi, form="literature"),
            msg="literature form != budget.phase",
        )
        self.assertLess(lit, est, msg=f"literature {lit} vs estimator {est}")

        base = span().run()
        walk = span(imp=(q.Dephasing(linewidth=2e4, delay=1e-8),)).run()
        self.assertClose(
            walk.budget.at("input")["dephasing"],
            est,
            atol=0.0,
            msg="dephasing row != impairments.dephasing",
        )
        self.assertLess(walk.key_rate, base.key_rate, msg="dephasing cost no key")

    def test_coherence_form(self):
        """
        coherence(lw, tau) equals exp(-pi*lw*tau) and exp(-V/2) to 1e-15 (lw in Hz, tau in s), the mean cosine of
        200000 phase draws to 5e-3, squares when tau doubles, and reads 0.9690724263048106 at 10 kHz over 1 us.
        """
        spans = ((10e3, 1e-6), (5e7, 1e-9), (2e4, 1e-5), (1e6, 4e-7))
        for lw, tau in spans:
            got = impairments.coherence(lw, tau)
            self.assertClose(got, math.exp(-math.pi * lw * tau), atol=1e-15, msg=f"lw={lw:.0e} tau={tau:.0e}")
            self.assertClose(
                got,
                math.exp(-0.5 * impairments.phase_variance(lw, tau)),
                atol=1e-15,
                msg="coherence != exp(-V/2)",
            )
            self.assertClose(got, impairments.coherence(tau, lw), atol=1e-15, msg="only the product enters")

        self.assertClose(
            impairments.coherence(10e3, 1e-6),
            0.9690724263048106,
            atol=1e-15,
            msg="coherence(10e3, 1e-6)",
        )
        self.assertEqual(impairments.coherence(10e3, 0.0), 1.0, msg="coherence at zero delay")
        self.assertEqual(impairments.coherence(0.0, 1e-6), 1.0, msg="coherence at zero linewidth")
        self.assertClose(
            impairments.coherence(10e3, 2e-6),
            impairments.coherence(10e3, 1e-6) ** 2,
            atol=1e-15,
            msg="doubling the delay did not square it",
        )

        fell = [impairments.coherence(10e3, t) for t in (1e-7, 1e-6, 1e-5, 1e-4)]
        self.assertMonotone(fell, rising=False, msg=f"coherence {fell}")

        for tag, (lw, tau) in enumerate(spans[:3]):
            draw = rng(tag)
            sd = math.sqrt(impairments.phase_variance(lw, tau))
            mean = sum(math.cos(draw.gauss(0.0, sd)) for _ in range(200_000)) / 200_000
            self.assertClose(
                mean,
                impairments.coherence(lw, tau),
                atol=5e-3,
                msg=f"lw={lw:.0e}: mean cosine {mean:.6f} vs {impairments.coherence(lw, tau):.6f}",
            )

    def test_fading_noise(self):
        """
        Polarisation noise is zero when aligned and rises as V_A*s^4/2 (Usenko's fading identity), and
        sampling-jitter noise is zero at zero jitter, depends on j/w alone and rises as V_A*u^2/8 with u = (j/w)^2.
        """

        self.assertEqual(impairments.polarisation(5.0, 0.0), 0.0, msg="polarisation(0) != 0")

        vals = [impairments.polarisation(5.0, s) for s in (0.01, 0.03, 0.1, 0.3)]
        self.assertEqual(vals, sorted(vals), msg=f"polarisation xi {vals}")
        self.assertClose(
            impairments.polarisation(5.0, 0.02),
            2.5 * 0.02**4,
            atol=1e-12,
            msg="small-drift limit != V_A s^4/2",
        )

        eff, var = impairments.pol_fading(0.1)
        self.assertLessEqual(eff, 1.0, msg=f"pol_fading eta {eff}")
        self.assertGreaterEqual(var, 0.0, msg=f"pol_fading var {var}")
        self.assertEqual(impairments.timing(5.0, 0.0, 5e-10), 0.0, msg="timing(0) != 0")

        vals = [impairments.timing(5.0, j, 5e-10) for j in (1e-12, 1e-11, 1e-10)]
        self.assertEqual(vals, sorted(vals), msg=f"timing xi {vals}")
        self.assertClose(
            impairments.timing(5.0, 1e-11, 5e-10),
            impairments.timing(5.0, 2e-11, 1e-9),
            atol=1e-14,
            msg="timing xi moved at fixed j/w",
        )
        self.assertClose(
            impairments.timing(5.0, 5e-12, 5e-10),
            5.0 * 1e-4**2 / 8.0,
            atol=1e-11,
            msg="small-jitter limit != V_A u^2/8",
        )

    def test_fading_identity(self):
        """
        fading_factor is the transmittance half of polarisation() and timing() (1.0 undescribed, <sqrt(eta)>^2 per
        row, their product for both), and q.Link closes T_claimed*(V_A + xi) = T*V_A*<eta> to 1e-14, where the
        noise half alone reads 2.50003094 against 2.48759298 at j/w = 0.1.
        """
        drift = impairments.Polarisation(0.1)
        clock = impairments.Timing(5e-11, 5e-10)
        self.assertEqual(impairments.fading_factor(), 1.0, msg="fading_factor() != 1")
        self.assertEqual(
            impairments.fading_factor(pol=drift),
            impairments.pol_fading(0.1)[0],
            msg="pol half != pol_fading[0]",
        )
        self.assertEqual(
            impairments.fading_factor(clock=clock),
            impairments.jitter_fading(5e-11, 5e-10)[0],
            msg="timing half != jitter_fading[0]",
        )
        self.assertClose(
            impairments.fading_factor(pol=drift, clock=clock),
            impairments.pol_fading(0.1)[0] * impairments.jitter_fading(5e-11, 5e-10)[0],
            atol=0.0,
            msg="two rows did not compose",
        )
        self.assertClose(
            impairments.polarisation(5.0, 0.1),
            0.0002500020833401647,
            atol=1e-18,
            msg="polarisation noise half moved",
        )
        self.assertClose(
            impairments.timing(5.0, 5e-11, 5e-10),
            6.188080519593797e-05,
            atol=1e-18,
            msg="timing noise half moved",
        )

        rows = (
            (
                q.Timing(jitter=5e-11, width=5e-10),
                impairments.jitter_fading(5e-11, 5e-10),
            ),
            (
                q.Timing(jitter=2.5e-10, width=5e-10),
                impairments.jitter_fading(2.5e-10, 5e-10),
            ),
            (q.Polarisation(drift=0.1), impairments.pol_fading(0.1)),
            (q.Polarisation(drift=0.3), impairments.pol_fading(0.3)),
        )
        for row, half in rows:
            res = pinned((row,)).run()
            got = res.explain["T_claimed"]["value"]
            name = type(row).__name__
            self.assertClose(got, 0.5 * half[0], atol=0.0, msg=f"{name}: T_claimed {got}")
            self.assertClose(
                got * (5.0 + res.explain["xi_total"]["value"]),
                0.5 * 5.0 * (half[0] + half[1]),
                atol=1e-14,
                msg=f"{name}: fading identity open",
            )

        res = pinned((q.Timing(jitter=5e-11, width=5e-10),)).run()
        self.assertClose(
            0.5 * (5.0 + res.explain["xi_total"]["value"]),
            2.5000309404025978,
            atol=1e-12,
            msg="charged-only signal",
        )

    def test_fading_costs_rate(self):
        """
        On a noiseless T = 0.5, leaving the transmittance unattenuated over-claims the rate by 0.83%, 3.34% and
        22.16% at j/w = 0.1, 0.2 and 0.5, and 0.07%, 0.42% and 1.68% at a polarisation drift of 0.02, 0.05 and 0.1 rad.
        """
        det = (0.6, 0.1, 0.95, False, True)
        rows = (
            (q.Timing(jitter=5e-11, width=5e-10), 0.8341),
            (q.Timing(jitter=1e-10, width=5e-10), 3.3418),
            (q.Timing(jitter=2.5e-10, width=5e-10), 22.1554),
            (q.Polarisation(drift=0.02), 0.0667),
            (q.Polarisation(drift=0.05), 0.4175),
            (q.Polarisation(drift=0.1), 1.6772),
        )
        for row, want in rows:
            res = pinned((row,)).run()
            loose = _core.cv_rate(5.0, 0.5, res.explain["xi_total"]["value"], *det)[2]
            name = f"{type(row).__name__}{row}"
            self.assertGreater(loose, res.key_rate, msg=f"{name}: loose {loose} <= {res.key_rate}")
            self.assertClose(
                100.0 * (loose / res.key_rate - 1.0),
                want,
                atol=1e-3,
                msg=f"{name}: over-claim {100.0 * (loose / res.key_rate - 1.0):.4f}%",
            )

    def test_imbalance_anchor(self):
        """
        Published anchor, fitted V_A = 5.25 SNU: arXiv:2503.10168 Eq. (E1) at d = 1 reproduces their 0.02 SNU
        imbalance noise "within a deviation angle range of +/- 5 degrees", zero for a balanced modulator, rising
        with imbalance and bias, and under 1e-4 SNU at their measured d = 0.9937 and V_A = 5.
        """
        got = impairments.imbalance(5.25, 1.0, math.radians(5.0))
        self.assertClose(got, 0.02, atol=5e-4, msg=f"MIN {got:.4e} vs 0.02 SNU (V_A=5.25 fitted)")
        self.assertLess(
            impairments.imbalance(5.25, 1.0, math.radians(3.0)),
            0.02,
            msg="imbalance at 3 degrees above 0.02",
        )
        self.assertEqual(impairments.imbalance(5.0), 0.0, msg="perfect modulator, no noise")

        angles = [impairments.imbalance(5.0, 1.0, a) for a in (0.01, 0.05, 0.2)]
        self.assertEqual(angles, sorted(angles), msg=f"imbalance xi {angles}")

        ratios = [impairments.imbalance(5.0, 1.0 - d, 0.0) for d in (0.001, 0.01, 0.1)]
        self.assertEqual(ratios, sorted(ratios), msg=f"imbalance xi {ratios}")
        self.assertLess(
            impairments.imbalance(5.0, 0.9937, 0.0),
            1e-4,
            msg="d = 0.9937 above 1e-4 SNU",
        )

    def test_omitted_hardware(self):
        """
        An undescribed row emits no entry, a perfect modulator emits 0.0, entries keep catalogue order, and V_A or
        T out of range or a Raman entry without a fibre length is refused.
        """

        self.assertEqual(len(metro()), 0, msg="metro() emitted an entry")

        entries = metro(modulator=impairments.Modulator())
        self.assertEqual(len(entries), 1, msg=f"entries {entries}")
        self.assertEqual(entries[0].xi, 0.0, msg="perfect modulator xi")

        named = metro(
            coexist=impairments.Coexistence(1, 0.0),
            dephase=impairments.Dephasing(2e4, 1e-8),
            pol=impairments.Polarisation(0.05),
            modulator=impairments.Modulator(0.99, 0.01),
            clock=impairments.Timing(1e-11, 5e-10),
        )
        self.assertEqual(
            [e.source for e in named],
            ["raman", "dephasing", "polarisation", "imbalance", "timing"],
            msg="catalogue order",
        )

        for call in (
            lambda: impairments.assemble_extra(v_a=0.0, t=0.5),
            lambda: impairments.assemble_extra(v_a=5.0, t=1.5),
            lambda: impairments.assemble_extra(v_a=5.0, t=0.5, coexist=impairments.Coexistence(1, 0.0)),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_charged_both_paths(self):
        """
        Every Gaussian-modulation impairment raises the budget and costs key on both paths, less sampled than
        closed (1.33e-2 against 1.89e-2 bit/symbol at 25 km, four channels), and 1, 2, 4, 8 channels give a rising
        total and a falling rate.
        """
        base = span().run()
        clean = span(dsp=True).claim(frames())
        for row in ROWS:
            closed = span(imp=(row,)).run()
            sampled = span(imp=(row,), dsp=True).claim(frames())
            name = type(row).__name__
            shut = base.key_rate - closed.key_rate
            drop = clean.key_rate - sampled.key_rate
            self.assertGreater(closed.budget.total, base.budget.total, msg=f"{name} closed budget")
            self.assertGreater(sampled.budget.total, clean.budget.total, msg=f"{name} sampled budget")
            self.assertGreater(shut, 0.0, msg=f"{name}: closed form not charged")
            self.assertGreater(drop, 0.0, msg=f"{name}: sampled path not charged")
            self.assertLess(sampled.key_rate, clean.key_rate, msg=f"{name} sampled key rate")
            self.assertLess(drop, shut, msg=f"{name}: sampled drop {drop} >= closed {shut}")
            self.assertGreater(4.0 * drop, shut, msg=f"{name}: drop {drop} under a quarter of {shut}")

        runs = [span(imp=(q.Coexistence(channels=n, launch=1e-3),), dsp=True).claim(frames()) for n in (1, 2, 4, 8)]
        totals = [r.budget.total for r in runs]
        keys = [r.key_rate for r in runs]
        self.assertEqual(totals, sorted(totals), msg=f"totals {totals}")
        self.assertEqual(keys, sorted(keys, reverse=True), msg=f"keys {keys}")

    def test_charged_off_pipeline(self):
        """
        One measurement claims with and without impairment rows on the same recovered channel, xi_claimed sitting
        above est.xi when charged and equal to it when not.
        """
        clean = span(dsp=True).claim(frames())
        dirty = span(imp=(RAMAN,), dsp=True).claim(frames())
        self.assertEqual(dirty.est.xi, clean.est.xi, msg="one measurement, two claims")
        self.assertLess(dirty.key_rate, clean.key_rate, msg="impairment not charged on the claim")
        self.assertGreater(
            dirty.explain["xi_claimed"]["value"],
            dirty.est.xi,
            msg="xi_claimed at or below est.xi",
        )

        self.assertClose(
            clean.explain["xi_claimed"]["value"],
            clean.est.xi,
            atol=0.0,
            msg="xi_claimed != est.xi with no impairment",
        )

    def test_rin_charged(self):
        """
        A declared RIN and detection bandwidth emit rin_sig and rin_lo rows matching explain's, xi_total rises with
        RIN, and -100 dBc/Hz leaves no key.
        """
        plain = front().run(symbols=20_000, seed=4)
        quiet = front(rin=-155.0, band=1e9).run(symbols=20_000, seed=4)
        loud = front(rin=-100.0, band=1e9).run(symbols=20_000, seed=4)
        rows = [e.source for e in quiet.budget.entries]
        self.assertIn("rin_sig", rows, msg="no rin_sig row")
        self.assertIn("rin_lo", rows, msg="no rin_lo row")
        self.assertEqual(
            len([e for e in plain.budget.entries if e.source.startswith("rin")]),
            0,
            msg="undeclared RIN emitted a row",
        )

        for res in (quiet, loud):
            self.assertClose(
                res.explain["xi_rin_sig"]["value"] + res.explain["xi_rin_lo"]["value"],
                sum(e.xi for e in res.budget.entries if e.source.startswith("rin")),
                atol=1e-15,
                msg="reported RIN != charged RIN",
            )
        self.assertMonotone(
            [
                plain.explain["xi_total"]["value"],
                quiet.explain["xi_total"]["value"],
                loud.explain["xi_total"]["value"],
            ],
            msg="xi_total not rising with RIN",
        )
        self.assertEqual(loud.key_rate, 0.0, msg=f"key at -100 dBc/Hz is {loud.key_rate}")

    def test_rin_half(self):
        """
        A RIN without a detection bandwidth, or a bandwidth without a RIN, is refused naming the missing end.
        """
        cases = (
            ("reaches nothing on its own", front(rin=-155.0).run),
            ("q.Heterodyne(bandwidth=...)", front(rin=-155.0).run),
            ("q.Laser(rin=...)", front(band=1e9).run),
        )
        for needle, call in cases:
            self.assertFails(ValueError, needle, call, msg=f"{needle!r} not named")

    def test_carrier_beat(self):
        """
        Two declared carriers move v_err through q.LocalLO.beat(laser), one carrier alone is refused naming the
        other end, and an offset past Nyquist is refused.
        """
        base = front().run(symbols=20_000, seed=4)
        offset = front(carrier=193.4e12 + 5e6, local=193.4e12).run(symbols=20_000, seed=4)
        self.assertNotEqual(base.dsp.v_err, offset.dsp.v_err, msg="v_err unmoved by beat")

        fast = front(carrier=193.4e12 + 2e8, local=193.4e12)
        cases = (
            ("q.LocalLO carries no carrier frequency", front(carrier=193.4e12).run),
            ("q.Laser carries no carrier frequency", front(local=193.4e12).run),
            ("2*|cfo| < symbol_rate", fast.run),
        )
        for needle, call in cases:
            self.assertFails(ValueError, needle, call, msg=f"{needle!r} not named")

    def test_demux_cancels(self):
        """
        raman_photons is linear in the demux (Kumar Eq. (6)'s eta_D), raman divides it back out to 1e-15, and q.Link
        refuses a demux other than 1.0.
        """
        counts = [impairments.raman_photons(1e-3, 25.0, demux=d) for d in (1.0, 0.1)]
        noises = [impairments.raman(1e-3, 25.0, 0.3, demux=d) for d in (1.0, 0.1)]
        self.assertClose(counts[1], 0.1 * counts[0], atol=1e-18, msg="raman_photons not linear in demux")
        self.assertLess(
            abs(noises[1] - noises[0]) / noises[0],
            1e-15,
            msg=f"raman xi {noises}",
        )
        self.assertFails(
            ValueError,
            "demux",
            span(imp=(q.Coexistence(channels=4, launch=0.0, demux=0.1),)).run,
            msg="quadrature demux accepted",
        )

    def test_dropped_rows(self):
        """
        A PMD coefficient and a modulator extinction ratio are refused by name on a quadrature link and still
        reported by security() as dgd and extinction.
        """
        cases = (
            ("dispersion", q.Polarisation(drift=0.01, dispersion=1e-12)),
            ("extinction", q.Modulator(ratio=0.99, angle=0.01, extinction=100.0)),
        )
        for needle, item in cases:
            self.assertFails(
                ValueError,
                needle,
                span(imp=(item,)).run,
                msg=f"{needle!r} not named",
            )

        held = impairments.security(
            pol=q.Polarisation(drift=0.01, dispersion=1e-12),
            modulator=q.Modulator(ratio=0.99, angle=0.01, extinction=100.0),
            length=25.0,
            qber=0.0,
        )
        self.assertIn("dgd", held, msg="security() dropped the DGD")
        self.assertIn("extinction", held, msg="security() dropped the extinction QBER")

    def test_drift_still_runs(self):
        """
        q.Polarisation and q.Modulator carrying only charged fields reach the budget as polarisation and imbalance.
        """
        rows = span(imp=(q.Polarisation(drift=0.01), q.Modulator(ratio=0.99, angle=0.01))).run()
        named = {e.source for e in rows.budget.entries}
        self.assertIn("polarisation", named, msg="no polarisation row")
        self.assertIn("imbalance", named, msg="no imbalance row")


class Detector(Question):
    """
    No excess-noise form: backflash, extinction, dead time, afterpulsing, visibility.
    """

    def test_backflash_anchor(self):
        """
        Published anchor, fitted e = 1.29%: Singh, Sharma, Singh and Prabhakar (arXiv:2502.04081) count 1598
        backflashes in 18000 sifted, which backflash_leak returns as their P_b = 0.0888, and
        P_sec = P_sift(1 - P_b - f h(e)) at f = 1.16 reproduces their "approximately 10%" rate loss.
        """
        # atol: rounding half-width of the source's 4-decimal 0.0888.
        prob = 1598.0 / 18000.0
        leak = impairments.backflash_leak(prob)
        self.assertClose(leak, 0.0888, atol=5e-5, msg=f"P_b {leak:.5f} vs 0.0888")

        clean = impairments.backflash_rate(1.0, 0.0, 0.0129)
        under = impairments.backflash_rate(1.0, prob, 0.0129)
        self.assertClose(
            1.0 - under / clean,
            0.10,
            atol=0.01,
            msg=f"reduction {1.0 - under / clean:.4f} vs 10% (e=0.0129 fitted, f=1.16)",
        )
        self.assertClose(
            impairments.backflash_leak(prob, 0.5),
            0.5 * prob,
            atol=1e-15,
            msg="leak != P_b * sift",
        )

    def test_backflash_inefficiency(self):
        """
        The house f = 1.16 (Lutkenhaus PRA 61, 052304 (2000), Table I) differs from Singh et al.'s 1.15 by exactly
        0.01*h(e) and security() carries f through, a probability outside [0, 1] is refused, and at Meda et al.'s
        measured 6% and 9.8% (arXiv:1605.05562) the leak stays under the sift and the rate non-negative.
        """
        for e in (0.01, 0.0129, 0.05, 0.10):
            house = impairments.backflash_rate(1.0, 0.0, e)
            source = impairments.backflash_rate(1.0, 0.0, e, 1.15)
            self.assertClose(
                source - house,
                0.01 * impairments._entropy(e),
                msg=f"gap at e = {e} is not 0.01*h(e)",
            )

        spec = impairments.security(backflash=impairments.Backflash(0.0888), sift=1.0, qber=0.0129, f=1.15)
        self.assertClose(
            spec["backflash_rate"],
            impairments.backflash_rate(1.0, 0.0888, 0.0129, 1.15),
            msg="security() dropped f",
        )

        house = impairments.security(backflash=impairments.Backflash(0.0888), sift=1.0, qber=0.0129)
        self.assertClose(
            house["backflash_rate"],
            impairments.backflash_rate(1.0, 0.0888, 0.0129, 1.16),
            msg="security() did not default to 1.16",
        )

        for bad in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                impairments.backflash_leak(bad)

        for prob in (0.0, 0.06, 0.098, 1.0):
            leak = impairments.backflash_leak(prob, 0.5)
            self.assertLessEqual(leak, 0.5, msg=f"leak {leak} exceeds the sift rate")
        self.assertEqual(
            impairments.backflash_rate(1.0, 0.9, 0.2),
            0.0,
            msg="backflash_rate went negative",
        )

    def test_extinction_limit(self):
        """
        The extinction QBER floor 2/(r+3) (Huang et al. arXiv:1206.6591, Eq. (5)) falls below 1e-5 by r = 1e6,
        reads 1.9% at 20 dB, only raises a given QBER, and refuses r = 0.
        """
        floors = [impairments.extinction(r) for r in (10.0, 100.0, 1e4, 1e6)]
        self.assertEqual(floors, sorted(floors, reverse=True), msg=f"floors {floors}")
        self.assertLess(floors[-1], 1e-5, msg=f"floor at r = 1e6 is {floors[-1]}")
        self.assertClose(
            impairments.extinction(100.0),
            2.0 / 103.0,
            atol=1e-15,
            msg="floor is 2/(r+3)",
        )
        self.assertGreater(
            impairments.extinction(100.0, 0.01),
            0.01,
            msg="extinction did not raise the QBER",
        )

        with self.assertRaises(ValueError):
            impairments.extinction(0.0)

    def test_dead_saturation(self):
        """
        A non-paralysable detector saturates below 1/tau_d and a paralysable one falls past 1/tau_d, and at R tau_d =
        5e-3 both lose under R^2 tau_d, the non-paralysable shedding R^3 tau_d^2 at second order and the paralysable
        half of it.
        """
        dead = 50e-9
        rates = [impairments.saturate(r, dead) for r in (1e6, 1e7, 1e8, 1e10)]
        self.assertEqual(rates, sorted(rates), msg=f"rates {rates}")
        self.assertLess(rates[-1], 1.0 / dead, msg=f"rate {rates[-1]} at or above 1/tau")

        peak = impairments.saturate(1.0 / dead, dead, True)
        self.assertGreater(
            peak,
            impairments.saturate(4.0 / dead, dead, True),
            msg="paralysable rate did not fall past R*",
        )

        # Knoll, Radiation Detection and Measurement, 4th ed., Ch. 4.7: m = n/(1 + n tau), m = n exp(-n tau).
        low = 1e5
        frac = low * dead
        first = low * frac
        self.assertLess(
            low - impairments.saturate(low, dead),
            first,
            msg="non-paralysable loss above R^2 tau_d",
        )
        self.assertLess(
            low - impairments.saturate(low, dead, True),
            first,
            msg="paralysable loss above R^2 tau_d",
        )

        # atol: 4x the third term of each alternating series.
        self.assertClose(
            low - impairments.saturate(low, dead),
            first * (1.0 - frac),
            atol=4.0 * low * frac**3,
            msg="non-paralysable second order",
        )
        self.assertClose(
            low - impairments.saturate(low, dead, True),
            first * (1.0 - 0.5 * frac),
            atol=4.0 * low * frac**3 / 6.0,
            msg="paralysable second order",
        )

    def test_paralysable_paired(self):
        """
        At zero dead time saturate returns one number on both branches, and q.Link refuses paralysable=True beside
        dead=0.0 by name.
        """
        rate = 1e7
        self.assertEqual(
            impairments.saturate(rate, 0.0, True),
            impairments.saturate(rate, 0.0, False),
            msg="branches differ at zero dead time",
        )
        held = keyed(q.ClickDetector(eta=0.1, dark=1e-6))
        held.impairments = (q.DeadTime(dead=0.0, afterpulse=0.02, paralysable=True),)
        self.assertFails(
            ValueError,
            "paralysable",
            held.run,
            msg="paralysable at dead = 0 accepted",
        )
        self.assertLess(
            impairments.saturate(rate, 20e-9, True),
            impairments.saturate(rate, 20e-9, False),
            msg="branches agree at a real dead time",
        )

    def test_afterpulse_qber(self):
        """
        Afterpulsing raises both gain and QBER (Papapanos et al. arXiv:2010.03358,
        Eqs. (4), (7), (8)) and at p_AP = 0 collapses onto _core.decoy_gain itself.
        """
        gain, err = impairments.afterpulse(0.5, 0.2, 1e-6, 0.0, 0.01)
        plain = _core.decoy_gain(0.5, 0.2, 1e-6, 0.01)
        self.assertClose(gain, plain[0], atol=1e-15, msg="p_AP=0 gain")
        self.assertClose(err, plain[1], atol=1e-15, msg="p_AP=0 QBER")
        self.assertClose(err, 0.010005, atol=1e-5, msg="p_AP=0 QBER is the detector floor")

        errs = [impairments.afterpulse(0.5, 0.2, 1e-6, p, 0.01)[1] for p in (0.0, 0.01, 0.05)]
        self.assertEqual(errs, sorted(errs), msg=f"QBERs {errs}")

        gains = [impairments.afterpulse(0.5, 0.2, 1e-6, p, 0.01)[0] for p in (0.0, 0.01, 0.05)]
        self.assertEqual(gains, sorted(gains), msg=f"gains {gains}")

        with self.assertRaises(ValueError):
            impairments.afterpulse(0.5, 0.2, 1e-6, 1.5, 0.01)

    def test_visibility_drift(self):
        """
        Interference visibility is 1 at zero polarisation mismatch and falls through the 0.37 two-source floor
        between 11 and 69 degrees, and mean DGD goes as D sqrt(L), 1 ps over 100 km at 0.1 ps/sqrt(km).
        """

        self.assertEqual(impairments.visibility(0.0), 1.0, msg="visibility(0) != 1")

        vals = [impairments.visibility(a) for a in (0.05, 0.2, 0.5, 1.0)]
        self.assertEqual(vals, sorted(vals, reverse=True), msg=f"visibilities {vals}")
        self.assertGreater(
            impairments.visibility(math.radians(11.0)),
            0.37,
            msg="11 degrees at or below the 0.37 floor",
        )
        self.assertLess(
            impairments.visibility(math.radians(69.0)),
            0.37,
            msg="69 degrees at or above the 0.37 floor",
        )

        far = impairments.dgd(0.1e-12, 100.0)
        self.assertClose(far, 1e-12, atol=1e-15, msg=f"DGD {far:.3e} s over 100 km")
        self.assertClose(
            impairments.dgd(0.1e-12, 400.0),
            2.0 * far,
            atol=1e-18,
            msg="DGD not as sqrt(L)",
        )

    def test_coherence_fringe(self):
        """
        Cross-check, not a reproduction: at 50 MHz on a 1 GHz clock and d = 1, 2 periods, the simulated fringe is
        0.98*coherence(lw, d/R) to 3e-3 and the simulated v_phase is phase_variance(lw, d/R) to 0.01.
        """
        seen = []
        for delay in (1, 2):
            res = fringe(5e7, delay=delay)
            span = delay / 1e9
            seen.append(res.explain["visibility"]["value"])
            self.assertClose(
                seen[-1],
                0.98 * impairments.coherence(5e7, span),
                atol=3e-3,
                msg=f"delay {delay}: fringe {seen[-1]:.5f}",
            )
            self.assertClose(
                res.explain["v_phase"]["value"],
                impairments.phase_variance(5e7, span),
                atol=0.01,
                msg=f"delay {delay}: simulated v_phase",
            )
        self.assertLess(seen[1], seen[0], msg=f"fringes {seen}")
        self.assertClose(
            impairments.coherence(5e7, 2e-9),
            impairments.coherence(5e7, 1e-9) ** 2,
            atol=1e-15,
            msg="doubling the delay did not square it",
        )

    def test_partner_split(self):
        """
        Bob's second threshold detector reaches both sampled click engines and moves their
        outputs, while an equal pair is the one-detector run bit for bit.
        """
        sym = q.ClickDetector(eta=0.2, dark=1e-6)
        twin = q.ClickDetector(eta=0.2, dark=1e-6, partner=sym)
        slow = q.ClickDetector(
            eta=0.2,
            dark=1e-6,
            partner=q.ClickDetector(eta=0.1, dark=4e-6),
        )
        self.assertClose(
            twin.background(),
            1.0 - (1.0 - 1e-6) * (1.0 - 1e-6),
            atol=0.0,
            msg="equal pair Y0",
        )

        one = keyed(sym, q.Alice(symbol_rate=1e9)).run(symbols=40_000, seed=1)
        same = keyed(twin, q.Alice(symbol_rate=1e9)).run(symbols=40_000, seed=1)
        split = keyed(slow, q.Alice(symbol_rate=1e9)).run(symbols=40_000, seed=1)
        self.assertEqual(
            (one.key_rate, one.qber, one.clicks, one.sifted, one.doubles),
            (same.key_rate, same.qber, same.clicks, same.sifted, same.doubles),
            msg="run_basis moved at an equal pair",
        )
        self.assertNotEqual(split.key_rate, one.key_rate, msg="split key_rate unmoved")

        plain = phased(sym).run(symbols=20_000, seed=2)
        pair = phased(twin).run(symbols=20_000, seed=2)
        uneven = phased(slow).run(symbols=20_000, seed=2)
        self.assertEqual(
            (plain.qber, plain.clicks, plain.visibility),
            (pair.qber, pair.clicks, pair.visibility),
            msg="run_clicks moved at an equal pair",
        )
        self.assertNotEqual(uneven.qber, plain.qber, msg="uneven qber unmoved")

    def test_partner_closed(self):
        """
        Every closed-form click link refuses a partner detector, decoy_gain, sarg_gain, sarg_yield and dps_rate each
        taking one eta and one dark.
        """
        pair = q.ClickDetector(
            eta=0.2,
            dark=1e-6,
            partner=q.ClickDetector(eta=0.1, dark=4e-6),
        )
        cow = q.Link(
            modulation=q.IntensityKeying(mu=0.5, decoy_frac=0.1),
            channel=q.Fiber(length=25.0),
            bob=q.Bob(
                detector=pair,
                receiver=q.CoherenceMonitor(split=0.1, misalign=0.02),
            ),
            security=q.PhaseBound(e_phase=0.3, f=1.16),
        )
        b92 = q.Link(
            modulation=q.TwoStateKeying(mu=0.23),
            channel=q.Fiber(length=25.0),
            bob=q.Bob(detector=pair, receiver=q.NullingReceiver(visibility=0.99)),
            security=q.DiscriminationBound(f=1.16),
        )
        for link in (keyed(pair), phased(pair, qber=0.03), cow, b92):
            self.assertFails(
                NotImplementedError,
                "no closed form here reads a second detector",
                link.run,
                msg=f"{type(link.modulation).__name__}",
            )

    def test_partner_fields(self):
        """
        The partner split reads eta and dark alone, so a partner differing in dead_time or jitter is refused by
        field name on the sampled paths.
        """
        late = q.ClickDetector(
            eta=0.2,
            dark=1e-6,
            partner=q.ClickDetector(eta=0.2, dark=1e-6, dead_time=1e-8),
        )
        timed = q.ClickDetector(
            eta=0.2,
            dark=1e-6,
            jitter=5e-11,
            partner=q.ClickDetector(eta=0.2, dark=1e-6, jitter=9e-11),
        )
        self.assertFails(
            NotImplementedError,
            "dead_time",
            keyed(late, q.Alice(symbol_rate=1e9)).run,
            msg="dead_time not named",
        )
        self.assertFails(
            NotImplementedError,
            "jitter",
            phased(timed).run,
            msg="jitter not named",
        )

    def test_clip_refused(self):
        """
        full_scale is 2*clip or None, and a declared clip is refused on homodyne and heterodyne alike, naming
        q.attacks.Saturation.
        """

        self.assertClose(
            q.Homodyne(clip=5.0).full_scale(),
            10.0,
            atol=0.0,
            msg="full_scale != 2*clip",
        )
        self.assertIsNone(q.Heterodyne().full_scale(), msg="full_scale set with no clip")

        for det in (q.Homodyne(clip=5.0), q.Heterodyne(clip=4.0)):
            link = span()
            link.bob = q.Bob(detector=det)
            self.assertFails(
                NotImplementedError,
                "nothing on the quadrature path models clipping",
                link.run,
                msg=f"q.{type(det).__name__}(clip=...) accepted",
            )
            self.assertFails(
                NotImplementedError,
                "q.attacks.Saturation(alpha=det.clip",
                link.run,
                msg="q.attacks.Saturation not named",
            )

    def test_security_dict(self):
        """
        security() mirrors assemble_extra's omit-when-unknown rule, never emitting xi.
        """

        self.assertEqual(impairments.security(), {}, msg="security() emitted a row")

        out = impairments.security(
            backflash=impairments.Backflash(0.0888),
            dead=impairments.DeadTime(50e-9, 0.02),
            modulator=impairments.Modulator(extinction=100.0),
            pol=impairments.Polarisation(0.05, 0.1e-12),
            sift=0.5,
            qber=0.0129,
            rate=1e7,
            mu=0.5,
            eta=0.2,
            dark=1e-6,
            length=100.0,
        )
        self.assertEqual(
            sorted(out),
            [
                "backflash_leak",
                "backflash_rate",
                "click_rate",
                "dgd",
                "extinction",
                "gain",
                "qber",
                "visibility",
            ],
            msg=f"keys {sorted(out)}",
        )
        self.assertNotIn("xi", out, msg="security() emitted an xi")


class ClickPath(Question):
    """
    Impairment descriptors on a threshold-detector link.
    """

    def test_background_floors(self):
        """
        A coexisting channel and a Rayleigh return each raise the click floor and so the QBER, less through a
        narrower passband and more at a stronger launch.
        """
        wide = lit(passband=100e9).run(SYMBOLS, 1)
        narrow = lit(passband=5.62e9).run(SYMBOLS, 1)
        clean = lit(passband=100e9, imp=()).run(SYMBOLS, 1)
        self.assertGreater(wide.qber, narrow.qber, msg="wide passband did not cost QBER")
        self.assertGreater(narrow.qber, clean.qber, msg="background QBER unmoved")
        self.assertGreater(clean.key_rate, narrow.key_rate, msg="clean key at or below narrow")
        self.assertGreater(narrow.key_rate, wide.key_rate, msg="narrow key at or below wide")

        louder = lit(passband=5.62e9, launch=-10.0).run(SYMBOLS, 1)
        self.assertGreater(louder.qber, narrow.qber, msg="launch QBER unmoved")

        probe = lit(passband=0.0, imp=(q.Backscatter(power=1e-9),)).run(SYMBOLS, 1)
        self.assertGreater(probe.qber, clean.qber, msg="backscatter QBER unmoved")

    def test_background_stated(self):
        """
        A coexisting channel without q.ClickDetector(passband=...) is refused naming the field, and a background
        over a bare transmittance is refused for want of a fibre length.
        """
        self.assertFails(
            ValueError,
            "q.ClickDetector(passband=...)",
            lit(passband=0.0).run,
            msg="no passband accepted",
        )
        self.assertFails(
            ValueError,
            "DETECTION bandwidth and is not it",
            lit(passband=0.0).run,
            msg="detection bandwidth not named",
        )
        bare = lit(passband=100e9, channel=q.Channel(T=0.1))
        self.assertFails(
            ValueError,
            "needs q.Channel(length=...)",
            bare.run,
            msg="a bare transmittance accepted",
        )

    def test_demux_apart(self):
        """
        The click route refuses a demultiplexer share as a loss on Bob's receive chain.
        """
        self.assertFails(
            ValueError,
            "loss on Bob's receive chain",
            lit(passband=100e9, imp=(q.Coexistence(channels=1, launch=-20.0, demux=0.1),)).run,
            msg="a click-path demux was read",
        )

    def test_extinction_charged(self):
        """
        An extinction ratio dilutes a basis-keyed link's misalignment by 4/(r+3) exactly as impairments.extinction
        does, keeps misalign_optics apart, and costs QBER and rate.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6)
        clean = keyed(det).run()
        leaky = keyed(det)
        leaky.impairments = (q.Modulator(extinction=100.0),)
        got = leaky.run()
        want = impairments.extinction(100.0, 0.015)
        self.assertClose(got.explain["misalign"]["value"], want, atol=0.0, msg="misalign != impairments.extinction")
        self.assertEqual(got.explain["misalign_optics"]["value"], 0.015, msg="misalign_optics moved")
        self.assertEqual(got.explain["extinction"]["value"], 100.0, msg="extinction not reported")
        self.assertGreater(got.qber, clean.qber, msg="extinction did not raise the QBER")
        self.assertLess(got.key_rate, clean.key_rate, msg="extinction cost no rate")

    def test_extinction_sampled(self):
        """
        The same extinction dilution costs QBER and rate on the sampled pulse train.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6)
        alice = q.Alice(symbol_rate=1e9)
        clean = keyed(det, alice=alice).run(40_000, 3)
        leaky = keyed(det, alice=alice)
        leaky.impairments = (q.Modulator(extinction=20.0),)
        got = leaky.run(40_000, 3)
        self.assertGreater(got.qber, clean.qber, msg="sampled QBER did not move")
        self.assertLess(got.key_rate, clean.key_rate, msg="sampled rate did not move")

    def test_fields_apart(self):
        """
        A quadrature link refuses a q.Modulator extinction ratio and a click link its ratio and angle, each by name,
        and a click link refuses a q.Modulator with no extinction.
        """
        both = q.Modulator(ratio=0.99, angle=0.01, extinction=100.0)
        self.assertFails(
            ValueError,
            "modulator extinction ratio",
            span(imp=(both,)).run,
            msg="extinction accepted on a quadrature link",
        )

        link = keyed(q.ClickDetector(eta=0.2, dark=1e-6))
        link.impairments = (both,)
        self.assertFails(
            ValueError,
            "IQ modulator's quadrature imbalance",
            link.run,
            msg="ratio and angle accepted on a click link",
        )

        link.impairments = (q.Modulator(),)
        self.assertFails(
            ValueError,
            "read for its `extinction` alone and none was declared",
            link.run,
            msg="a bare q.Modulator accepted",
        )

    def test_four_paths(self):
        """
        The 4 of 4/(r+3) counts four transmitter paths, so six-state and DPS refuse an extinction ratio by name and
        COW refuses it as declared twice.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6)
        six = keyed(det)
        six.modulation = q.BasisKeying(decoy=q.Decoy(intensities=(0.5, 0.1, 0.0)), bases=3)
        six.impairments = (q.Modulator(extinction=100.0),)
        self.assertFails(ValueError, "refused at bases = 3", six.run, msg="bases = 3 accepted")

        dps = phased(det, qber=0.01)
        dps.impairments = (q.Modulator(extinction=100.0),)
        self.assertFails(ValueError, "FOUR-path transmitter", dps.run, msg="a phase-keyed train accepted it")

        cow = q.Link(
            modulation=q.IntensityKeying(mu=0.5, decoy_frac=0.1),
            channel=q.Fiber(length=25.0),
            bob=q.Bob(detector=q.ClickDetector(eta=0.8, dark=1e-6), receiver=q.CoherenceMonitor()),
            security=q.PhaseBound(e_phase=0.2),
            impairments=(q.Modulator(extinction=100.0),),
        )
        self.assertFails(ValueError, "declared twice on this link", cow.run, msg="a doubly declared modulator accepted")

    def test_reasons_apart(self):
        """
        A click link refuses q.Dephasing, q.Polarisation and q.Timing each with its own reason; the q.Dephasing
        message is the one the guide prints.
        """
        det = q.ClickDetector(eta=0.2, dark=1e-6)
        rows = (
            (q.Dephasing(linewidth=1e4, delay=1e-6), "it yields a Gaussian excess noise"),
            (q.Polarisation(drift=0.02), "q.ReferenceFrame(drift=...)"),
            (q.Timing(jitter=1e-12, width=5e-9), "q.ClickDetector carries it"),
        )
        for item, needle in rows:
            link = keyed(det)
            link.impairments = (item,)
            self.assertFails(
                ValueError,
                needle,
                link.run,
                msg=f"{type(item).__name__} refusal wording",
            )

    def test_quadrature_reasons(self):
        """
        A quadrature link refuses q.Backflash and q.DeadTime as having no Gaussian-modulation term, in the message
        the guide prints.
        """
        for item in (q.Backflash(prob=0.05), q.DeadTime(dead=0.0, afterpulse=0.01)):
            self.assertFails(
                ValueError,
                f"no Gaussian-modulation term consumes {type(item).__name__}",
                span(imp=(item,)).run,
                msg=f"{type(item).__name__} accepted",
            )


class Domain(Guarded):
    """
    The argument domain of the phase-walk pair.
    """

    def test_walk_slots(self):
        """
        phase_variance and coherence refuse a negative linewidth or delay by argument name, coherence stays at or
        below 1, and zero on either slot returns 1.0.
        """
        cases = (
            (0, "linewidth must be nonnegative", (-1e3, -1e-30, -math.inf)),
            (1, "delay must be nonnegative", (-1e-6, -1e-30, -math.inf)),
        )
        for fn in (impairments.phase_variance, impairments.coherence):
            self.assertSlots(fn, (10e3, 1e-6), cases, msg=fn.__name__)

        self.assertEqual(impairments.phase_variance(0.0, 0.0), 0.0, msg="phase_variance(0, 0)")
        for lw, tau in ((10e3, 0.0), (0.0, 1e-6), (0.0, 0.0)):
            self.assertEqual(
                impairments.coherence(lw, tau),
                1.0,
                msg=f"coherence({lw:g}, {tau:g}) != 1",
            )
        for lw, tau in ((1e-30, 1e-30), (10e3, 1e-6), (5e7, 1e-9), (1e9, 1.0)):
            self.assertLessEqual(
                impairments.coherence(lw, tau),
                1.0,
                msg=f"coherence({lw:g}, {tau:g}) above 1",
            )

    def test_walk_consumers(self):
        """
        impairments.dephasing and a q.Link declaring a negative q.Dephasing refuse in the same words on both the
        closed-form and the sampled path.
        """
        self.assertBad(
            "linewidth must be nonnegative",
            impairments.dephasing,
            (5.0, -1e3, 1e-6),
            msg="dephasing took a negative linewidth",
        )
        self.assertBad(
            "delay must be nonnegative",
            impairments.dephasing,
            (5.0, 1e4, -1e-8),
            msg="dephasing took a negative delay",
        )
        self.assertFails(
            ValueError,
            "linewidth must be nonnegative",
            span(imp=(q.Dephasing(linewidth=-1e3, delay=1e-8),)).run,
            msg="closed form took a negative linewidth",
        )
        self.assertFails(
            ValueError,
            "delay must be nonnegative",
            lambda: span(imp=(q.Dephasing(linewidth=2e4, delay=-1e-8),), dsp=True).claim(frames()),
            msg="sampled path took a negative delay",
        )


if __name__ == "__main__":
    rc = Exam(
        "ImpairmentsOptical",
        "Fibre and transmitter impairments with an excess-noise form",
        "impairments_optical.md",
    ).run(load(Optical))
    rc |= Exam(
        "ImpairmentsDetector",
        "Click-protocol impairments and security flags",
        "impairments_detector.md",
    ).run(load(Detector))
    rc |= Exam(
        "ImpairmentsClick",
        "Which descriptors a threshold detector reads, and one reason per mechanism for each it does not",
        "impairments_click.md",
    ).run(load(ClickPath))
    rc |= Exam(
        "ImpairmentsDomain",
        "The phase-walk pair's argument domain",
        "impairments_domain.md",
    ).run(load(Domain))
    sys.exit(rc)
