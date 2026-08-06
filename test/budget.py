import inspect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.noise import assembled
from qkd import budget, impairments


def metro():
    """
    The reference metro budget: V_A = 5 SNU through a 5 dB channel.
    """

    return assembled(5.0, 10**-0.5)


def optics(launch=0.5, receive=1.5):
    """
    A metro budget over an itemised loss chain.
    """

    return budget.assemble(
        v_a=5.0,
        t=10**-0.5,
        v_err=2e-3,
        adc_bits=12,
        losses=(
            q.Connector(loss=0.5 * launch, count=2, site="launch"),
            q.Splice(count=6),
            q.Coupling(loss=receive, site="receive"),
        ),
    )


def raman(losses):
    """
    A 25 km link carrying four co-propagating classical channels.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Fiber(length=25.0),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
        impairments=(q.Coexistence(channels=4, launch=0.0),),
        losses=losses,
    ).run()


def quantised(bits):
    """
    The metro link with a pinned phase residual and a described DAC.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=0.01, ref="input"),
        alice=q.Alice(iq=q.IQModulator(bits=bits)),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
        dsp=q.DSP(phase=q.PilotPhase(v_err=2e-3)),
        security=q.Asymptotic(beta=0.95),
    )


def plain():
    """
    The same pinned channel with no converter and no phase row to form.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=0.01, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
        security=q.Asymptotic(beta=0.95),
    )


# Tier B fixture. Hajomer et al., Sci. Adv. 10, eadi9474 (2024) = arXiv:2305.08156,
# one paper under two labels: V_mod = 8.41 SNU (Sec. 4), 15.4 dB channel and 16-bit
# DAC/ADC (Sec. 3.1), so T = 10^-1.54.
# v_err is fitted inside [1e-4, 1e-2] rad^2: lower 7.6e-5 from Zhang et al., PRL
# 125, 010502 (2020), sigma^2 = 1 - kappa in the small-angle limit (transmitted
# LO); upper 0.040 rad^2 from Qi et al., PRX 5, 041009 (2015), Sec. III.3, LLO.
def hajomer(v_err):
    """
    Their receiver at their channel, on the fitted phase-error variance.
    """

    return assembled(8.41, 10**-1.54, v_err, adc_bits=16)


class Engine(Question):
    """
    Phase 3 noise-budget engine: planes, closed forms, assembly semantics.
    """

    def test_plane_rule(self):
        """
        Every entry obeys at("bob") == T * at("input").
        """
        t = 10**-0.5
        bud = metro()
        outp = bud.at("bob")
        for src, xi in bud.at("input").items():
            self.assertEqual(outp[src], t * xi, msg=f"{src} plane conversion")
        self.assertEqual(len(outp), 5, msg="all five sources must be present")

    def test_phase_exact(self):
        """
        Input-referred, the literature form is exactly 2*V_A*(1 - exp(-V_err/2)) and ignores T
        -- Marie & Alleaume, Phys. Rev. A 95, 012316 (2017), Eq. (10).
        """
        want = 2.0 * 5.0 * (1.0 - math.exp(-2e-3 / 2.0))
        got = budget.phase(5.0, 2e-3, form="literature")
        self.assertEqual(got, want, msg="literature closed form")

        near = budget.assemble(v_a=5.0, t=0.9, v_err=2e-3, phase_form="literature")
        far = budget.assemble(v_a=5.0, t=0.05, v_err=2e-3, phase_form="literature")
        self.assertEqual(
            near.at("input")["phase"],
            far.at("input")["phase"],
            msg="phase is unaffected by T",
        )

    def test_phase_default_form(self):
        """
        assemble() emits the estimator form (V_A + xi)*(exp(V_err) - 1) and a plane_note naming
        Kish, Quantum 8, 1382 (2024), App. E (76)-(82), where it follows from Eqs. (80)-(82) by
        rearrangement, and Shen, Opt. Express 29, 30978 (2021),
        Eqs. (18)-(19).
        """
        want = (5.0 + 0.02) * math.expm1(2e-3)
        self.assertEqual(budget.phase(5.0, 2e-3, 0.02), want, msg="estimator form")

        bud = budget.assemble(v_a=5.0, t=0.5, v_err=2e-3, xi=0.02)
        self.assertEqual(bud.at("input")["phase"], want, msg="assemble default form")
        self.assertEqual(
            bud.entries[0].plane_note,
            budget.PHASE_NOTES["estimator"],
            msg=f"plane_note = {bud.entries[0].plane_note}",
        )
        self.assertIn(
            "Kish",
            bud.entries[0].plane_note,
            msg=f"plane_note = {bud.entries[0].plane_note}",
        )

    def test_phase_transmittance(self):
        """
        infer() is the transmittance half to phase()'s noise half, t*exp(-v_err) of Kish App. E
        Eq. (78), and the pair conserves the Bob-plane variance.
        """
        for v_err in (0.0, 2e-3, 0.27, 1.15, 3.48):
            got = budget.infer(0.4, v_err)
            self.assertClose(got, 0.4 * math.exp(-v_err), atol=0.0, msg=f"t e^-v at {v_err}")

            step = budget.phase(5.0, v_err, 0.01)
            self.assertClose(
                got * (5.0 + 0.01 + step),
                0.4 * (5.0 + 0.01),
                atol=1e-15,
                msg=f"Bob-plane variance at v_err = {v_err}",
            )
        self.assertEqual(budget.infer(0.4, 0.0), 0.4, msg="infer(0.4, 0)")
        self.assertEqual(
            budget.infer(0.4, float("inf")),
            0.0,
            msg="infer(0.4, inf)",
        )

    def test_phase_refuses(self):
        """
        Each refusal names its argument: the literature form refuses a channel xi and a
        transmittance half, and a negative variance or t above 1 is refused.
        """
        cases = (
            ("estimator", lambda: budget.phase(5.0, 2e-3, form="marie")),
            ("literature", lambda: budget.phase(5.0, 2e-3, 0.02, form="literature")),
            ("v_err", lambda: budget.assemble(v_a=5.0, t=0.5, xi=0.02)),
            ("nonnegative", lambda: budget.phase(5.0, 2e-3, -0.01)),
            ("transmittance half", lambda: budget.infer(0.4, 2e-3, form="literature")),
            ("estimator", lambda: budget.infer(0.4, 2e-3, form="marie")),
            ("nonnegative", lambda: budget.infer(0.4, -1e-3)),
            ("v_err", lambda: budget.infer(0.4, float("nan"))),
            ("t", lambda: budget.infer(1.5, 2e-3)),
        )
        for needle, call in cases:
            self.assertFails(ValueError, needle, call, msg=f"the refusal must name {needle!r}")

    def test_budget_inferred(self):
        """
        Budget.inferred is infer(T, v_err) where a phase row was formed, T itself where none
        was, explain's T_claimed on a run, and a raise on a literature budget.
        """
        bud = budget.assemble(v_a=5.0, t=0.5, v_err=0.27, xi=0.01)
        self.assertEqual(bud.T, 0.5, msg=f"T = {bud.T}")
        self.assertEqual(bud.inferred, budget.infer(0.5, 0.27), msg="the transmittance half")
        self.assertLess(bud.inferred, bud.T, msg=f"inferred = {bud.inferred}")
        self.assertClose(
            bud.inferred * (5.0 + 0.01 + bud.total),
            bud.T * (5.0 + 0.01),
            atol=1e-15,
            msg="Bob-plane variance",
        )

        bare = budget.assemble(v_a=5.0, t=0.5, adc_bits=12)
        self.assertEqual(bare.inferred, bare.T, msg="bare inferred")
        self.assertIsNone(bare.v_err, msg="bare v_err")
        self.assertIsNone(bare.phase_form, msg="bare phase_form")

        old = budget.assemble(v_a=5.0, t=0.5, v_err=2e-3, phase_form="literature")
        self.assertFails(
            ValueError,
            "transmittance half",
            lambda: old.inferred,
            msg="literature inferred",
        )
        self.assertEqual(old.T, 0.5, msg="literature T")

        run = quantised(16).run()
        self.assertEqual(run.budget.v_err, 2e-3, msg=f"run v_err = {run.budget.v_err}")
        self.assertEqual(
            run.budget.inferred,
            budget.infer(run.budget.T, 2e-3),
            msg="run inferred",
        )
        self.assertEqual(
            run.budget.inferred,
            run.explain["T_claimed"]["value"],
            msg="run inferred vs T_claimed",
        )

        flat = plain().run()
        self.assertIsNone(flat.budget.v_err, msg="plain v_err")
        self.assertEqual(flat.budget.inferred, flat.budget.T, msg="plain inferred")

    def test_rin_rows(self):
        """
        assemble() emits budget.rin_rows verbatim, values and plane notes alike.
        """
        bud = metro()
        rows = {e.source: e for e in bud.entries}
        want = budget.rin_rows(5.0, -155.0, 100e6, bud.T)
        self.assertEqual([e.source for e in want], ["rin_sig", "rin_lo"], msg="two rows, in order")

        for entry in want:
            self.assertEqual(rows[entry.source].xi, entry.xi, msg=f"{entry.source} bit-for-bit")
            self.assertEqual(
                rows[entry.source].plane_note,
                entry.plane_note,
                msg=f"{entry.source} plane note",
            )
        self.assertIn("9.21", want[1].plane_note, msg="rin_lo plane_note")

    def test_dac_reaches_link(self):
        """
        DAC quantisation noise falls with bit depth, under 1e-3 SNU input-referred at 16 bits,
        and a q.Link charges budget.dac's own value.
        """
        vals = [budget.dac(5.0, b, 0.3) for b in (8, 10, 12, 14, 16)]
        self.assertEqual(vals, sorted(vals, reverse=True), msg=f"dac xi {vals}")
        self.assertLess(vals[-1], 1e-3, msg=f"16-bit dac xi {vals[-1]:.2e}")

        runs = [quantised(b).run() for b in (16, 12, 8, 4)]
        keys = [r.key_rate for r in runs]
        self.assertIn("dac", runs[0].budget.at("input"), msg="dac row")
        self.assertClose(
            runs[-1].budget.at("input")["dac"],
            budget.dac(5.0, 4, runs[-1].budget.T),
            atol=0.0,
            msg="4-bit dac row",
        )
        self.assertEqual(keys, sorted(keys, reverse=True), msg=f"keys {keys}")
        self.assertLess(keys[-1], keys[0] - 1e-3, msg=f"keys {keys}")

    def test_total_ungated(self):
        """
        xi_total sums every assembled row, charged or not.
        """
        res = quantised(16).run()
        bare = plain().run()
        self.assertClose(
            res.explain["xi_total"]["value"],
            res.budget.total,
            atol=0.0,
            msg="xi_total vs budget.total",
        )
        self.assertClose(
            bare.explain["xi_total"]["value"],
            0.01,
            atol=0.0,
            msg="xi_total without a phase row",
        )

    def test_omitted_hardware(self):
        """
        Hardware left None omits its row rather than zeroing it, and assemble() takes no vel,
        trusted or detector term.
        """
        bud = budget.assemble(v_a=5.0, t=0.316, v_err=2e-3)
        self.assertEqual(set(bud.at("input")), {"phase"}, msg="rows with v_err alone")

        full = metro()
        self.assertEqual(
            set(full.at("input")),
            {"phase", "dac", "adc", "rin_sig", "rin_lo"},
            msg="metro rows",
        )

        names = set(inspect.signature(budget.assemble).parameters)
        self.assertEqual(names & {"vel", "trusted"}, set(), msg="no v_el knobs")
        self.assertFalse(hasattr(budget, "detector"), msg="no detector() term")

    def test_raman_density(self):
        """
        Laudenbach Eq. (9.63) on a measured dBm/nm density: linear in the density, the filter
        width and the integration time, zero through a shut filter.
        """
        width = budget.raman_width(1e9)
        base = budget.raman(-90.0, width, 1e-9, 1.0)
        # 2 dlambda 10^(N/10) tau/(hf) x 1e6, hand-evaluated at N_Ram = -90.
        hand = 2.0 * width * 1.0e-9 * 1.0e6 * 1.0e-9 * 1550.12e-9
        hand = hand / (6.62607015e-34 * 299792458.0)
        self.assertClose(base, hand, atol=1e-18, msg=f"xi_Ram {base:.6e} vs Eq. (9.63) by hand")
        self.assertClose(
            budget.raman(-80.0, width, 1e-9, 1.0),
            10.0 * base,
            atol=1e-16,
            msg="raman at -80 dBm/nm",
        )
        self.assertClose(
            budget.raman(-90.0, 2.0 * width, 2e-9, 1.0),
            4.0 * base,
            atol=1e-16,
            msg="raman at twice width and time",
        )
        self.assertEqual(
            budget.raman(-90.0, 0.0, 1e-9, 1.0),
            0.0,
            msg="raman at zero width",
        )

    def test_raman_width(self):
        """
        The matched filter width is Laudenbach's 8 pm at 1 GHz on 193.4 THz, and B cancels out
        of the assembled Raman row.
        """
        width = budget.raman_width(1e9)
        self.assertClose(width, 8.0e-12, atol=2e-14, msg=f"{width:.3e} m vs 8 pm")
        self.assertClose(
            299792458.0 / 1550.12e-9 / 1e12,
            193.4,
            atol=1e-3,
            msg="default carrier, THz",
        )

        rows = [budget.assemble(v_a=5.0, t=0.5, raman_db=-80.0, bandwidth=b).total for b in (1e6, 1e8, 1e9, 1e10)]
        self.assertClose(
            max(rows),
            min(rows),
            atol=1e-18,
            msg=f"Raman totals over B: {rows}",
        )

    def test_raman_convention(self):
        """
        Laudenbach Eq. (9.63) is twice Kumar 2015 (impairments.raman) on one density, Kumar
        keeping the polarising beamsplitter's half.
        """
        lam = 1531.12e-9
        # The W/nm four 0 dBm channels lay on a 25 km span, Kumar's own way.
        dens = 3.0e-9 * 4.0e-3 * 25.0 * math.exp(-0.2 * 25.0 / (10.0 / math.log(10.0)))
        n_ram = 10.0 * math.log10(dens * 1e9 / 1e6)
        mine = budget.raman(n_ram, budget.raman_width(1e9, lam), 1e-9, 0.5, wavelength=lam)
        self.assertClose(
            mine,
            2.0 * impairments.raman(4.0e-3, 25.0, 0.5, wavelength=lam),
            atol=1e-15,
            msg=f"one density {n_ram:.2f} dBm/nm, two stated conventions",
        )

    def test_raman_refuses(self):
        """
        A negative width or integration time, an infinite density, and a density or a rin with
        no bandwidth are all refused.
        """
        cases = (
            ("width", lambda: budget.raman(-80.0, -1e-12, 1e-9, 0.5)),
            ("symbol", lambda: budget.raman(-80.0, 8e-12, -1e-9, 0.5)),
            ("n_ram", lambda: budget.raman(float("inf"), 8e-12, 1e-9, 0.5)),
            ("given together", lambda: budget.assemble(v_a=5.0, t=0.5, raman_db=-80.0)),
            ("rin and bandwidth", lambda: budget.assemble(v_a=5.0, t=0.5, rin=-155.0)),
        )
        for needle, call in cases:
            self.assertFails(ValueError, needle, call, msg=f"the refusal must name {needle!r}")


class TierB(Question):
    """
    Pin what the papers publish, fit what they do not, and name every fit.
    """

    def test_metro_envelope(self):
        """
        The default 25 km metro link (V_A=5, T=10^-0.5, v_err 2e-3 rad^2, 16-bit DAC, 12-bit
        ADC, RIN -155 dBc/Hz, B=100 MHz) assembles an input-referred xi inside the measured
        envelope [0.005, 0.03] SNU, phase dominating.
        """
        bud = metro()
        total = bud.total
        self.assertGreater(total, 0.005, msg=f"xi = {total:.5f}")
        self.assertLess(total, 0.03, msg=f"xi = {total:.5f}")

        parts = bud.at("input")
        self.assertEqual(
            max(parts, key=parts.get),
            "phase",
            msg=f"rows {parts}",
        )

    def test_hajomer_xi(self):
        """
        Hajomer 2024 100 km LLO (arXiv:2305.08156), V_mod=8.41 SNU, T=10^-1.54, 16-bit DAC/ADC:
        the Bob-plane xi brackets their 2.12e-4 SNU within a factor of 3, on a fit against the
        literature form at v_err=7e-4 rad^2, RIN -155 dBc/Hz, B=100 MHz, ADC ratio 10.
        """
        bud = hajomer(7e-4)
        got = bud.T * bud.total
        self.assertGreater(
            got,
            2.12e-4 / 3.0,
            msg=f"xi_bob {got:.3e} vs 2.12e-4 (v_err=7e-4, rin=-155, B=100e6, ratio=10)",
        )
        self.assertLess(
            got,
            2.12e-4 * 3.0,
            msg=f"xi_bob {got:.3e} vs 2.12e-4 (v_err=7e-4, rin=-155, B=100e6, ratio=10)",
        )

    def test_hajomer_key(self):
        """
        Hajomer's trusted receiver (heterodyne eta=0.68, v_el=0.06272 SNU, beta=0.925) on the
        assembled budget gives ~2.7e-3 bit/symbol, above their finite-size 2.54e-4 and within a
        factor of 5 (not 2: chi_het/T ~ 74 SNU at T=10^-1.54) of 0.0117, derived not printed
        from Pietri et al., Quantum 8, 1575 (2024), arXiv:2404.18637, 1.17 Mbit/s asymptotic
        over 25 km at 100 MBaud.
        """
        bud = hajomer(7e-4)
        res = q.Link(
            modulation=q.GaussianModulation(v_a=8.41),
            channel=q.Channel(T=bud.T, xi=bud.total, ref="input"),
            bob=q.Bob(detector=q.Heterodyne(eta=0.68, v_el=0.06272, trusted=True)),
            security=q.Asymptotic(beta=0.925),
        ).run()
        key = res.key_rate
        self.assertGreater(
            key,
            2.54e-4,
            msg=f"asymptotic {key:.4e} vs their finite-size floor 2.54e-4 (xi_in={bud.total:.4e})",
        )
        self.assertGreater(
            key,
            0.0117 / 5.0,
            msg=f"achieved {key:.4e} vs 0.0117/5 (xi_in={bud.total:.4e})",
        )

        ceiling = q.Link(
            modulation=q.GaussianModulation(v_a=8.41),
            channel=q.Channel(T=10**-1.54),
            bob=q.Bob(detector=q.Heterodyne(eta=0.68, v_el=0.06272, trusted=True)),
            security=q.Asymptotic(beta=0.925),
        ).run()
        self.assertLess(
            ceiling.key_rate,
            0.0117 / 2.0,
            msg=f"xi=0 ceiling {ceiling.key_rate:.4e}",
        )
        self.assertGreater(
            ceiling.key_rate,
            0.0117 / 5.0,
            msg=f"xi=0 ceiling {ceiling.key_rate:.4e}",
        )


class Planes(Question):
    """
    Composition, order, and the one factor a budget refuses to invent.
    """

    def test_plane_names(self):
        """
        The four planes in optical order, "input" and "bob" aliasing the channel and detector
        inputs, with referral factors 1, launch*span, T, T*eta.
        """
        bud = optics()
        self.assertEqual(
            budget.PLANES,
            (
                "channel_input",
                "channel_output",
                "detector_input",
                "post_detection",
            ),
            msg="the four planes, in optical order",
        )
        self.assertEqual(bud.factor("input"), bud.factor("channel_input"), msg="input alias")
        self.assertEqual(bud.factor("bob"), bud.factor("detector_input"), msg="bob")

        with self.assertRaises(ValueError):
            bud.factor("alice")

        want = [bud.factor(p, eta=0.6) for p in budget.PLANES]
        self.assertEqual(want, sorted(want, reverse=True), msg="factors must fall down the path")
        self.assertEqual(want[0], 1.0, msg="the channel input is the identity")
        self.assertClose(want[1], bud.launch * bud.span, atol=1e-15, msg="output is launch*span")
        self.assertClose(want[2], bud.T, atol=1e-15, msg="detector input is T")

    def test_plane_compose(self):
        """
        Referring through an intermediate plane equals referring straight there, and at(plane)
        is refer() row by row.
        """
        bud = optics()
        for start in budget.PLANES:
            for mid in budget.PLANES:
                for end in budget.PLANES:
                    step = bud.refer(0.037, start, mid, eta=0.6)
                    both = bud.refer(step, mid, end, eta=0.6)
                    once = bud.refer(0.037, start, end, eta=0.6)
                    self.assertClose(
                        both,
                        once,
                        atol=1e-15,
                        msg=f"{start} -> {mid} -> {end}",
                    )

        for plane in budget.PLANES:
            got = bud.at(plane, eta=0.6)
            for src, xi in bud.at("channel_input").items():
                want = bud.refer(xi, "channel_input", plane, eta=0.6)
                self.assertEqual(got[src], want, msg=f"{src} at {plane}")

    def test_plane_detector(self):
        """
        post_detection is the one plane needing eta, one factor of it past detector_input.
        """
        bud = optics()

        with self.assertRaises(ValueError):
            bud.at("post_detection")

        with self.assertRaises(ValueError):
            bud.at("post_detection", eta=1.5)

        names = set(f.name for f in budget.Budget.__dataclass_fields__.values())
        self.assertEqual(names & {"eta", "v_el"}, set(), msg="no receiver fields")
        self.assertClose(
            bud.at("post_detection", eta=0.6)["adc"],
            0.6 * bud.at("detector_input")["adc"],
            atol=1e-18,
            msg="post_detection adc",
        )


class Optics(Question):
    """
    Multiplicative in T, named line by line, asymmetric for a span-born noise.
    """

    def test_loss_optional(self):
        """
        A budget declaring no optics keeps its T, rows and totals, prints no loss section, and
        collapses the channel-output and detector planes.
        """
        bud = metro()
        self.assertEqual(bud.T, 10**-0.5, msg=f"T = {bud.T}")
        self.assertEqual(bud.launch, 1.0, msg="no launch optics")
        self.assertEqual(bud.receive, 1.0, msg="no receive optics")
        self.assertEqual(bud.span, bud.T, msg=f"span = {bud.span}")
        self.assertNotIn("loss", bud.table(), msg="loss section in table()")
        self.assertEqual(
            bud.factor("channel_output"),
            bud.factor("detector_input"),
            msg="channel_output vs detector_input factor",
        )

    def test_loss_multiplies(self):
        """
        Every declared optic multiplies into T and the decibels add: two 0.25 dB connectors, six
        0.02 dB splices and a 1.5 dB coupling take 5.0 dB to 7.12.
        """
        bud = optics()
        rows = {row.source: row for row in bud.losses}
        self.assertEqual(
            set(rows),
            {"connector_launch", "fiber_span", "splice_span", "coupling_receive"},
            msg=f"loss rows {set(rows)}",
        )
        self.assertClose(rows["connector_launch"].db, 0.5, atol=1e-12, msg="2 x 0.25")
        self.assertClose(rows["splice_span"].db, 0.12, atol=1e-12, msg="6 x 0.02")
        self.assertClose(-10.0 * math.log10(bud.T), 7.12, atol=1e-12, msg="total dB")

        want = 1.0
        for row in bud.losses:
            want = want * row.transmittance
        self.assertClose(bud.T, want, atol=1e-15, msg="T vs row product")

    def test_loss_refuses(self):
        """
        A duplicate descriptor at one site, a non-loss component, a negative loss and an unknown
        site are refused, and splices at two sites keep two lines.
        """

        with self.assertRaises(ValueError):
            budget.chain(0.5, (q.Splice(), q.Splice(count=2)))

        with self.assertRaises(ValueError):
            budget.chain(0.5, (q.Fiber(length=10.0),))

        with self.assertRaises(ValueError):
            q.Connector(loss=-0.1)

        with self.assertRaises(ValueError):
            q.Coupling(loss=1.0, site="middle")

        rows = budget.chain(0.5, (q.Splice(), q.Splice(count=2, site="receive")))[3]
        self.assertEqual(len(rows), 3, msg="two splice lines plus the fibre")

    def test_loss_asymmetry(self):
        """
        1.5 dB at Alice's output and at Bob's input give one T and one detector-plane referral
        but different channel-output and span-born input referrals.
        """
        near = optics(launch=1.5, receive=0.0)
        far = optics(launch=0.0, receive=1.5)
        step = 10.0 ** (-1.5 / 10.0)
        self.assertClose(near.T, far.T, atol=1e-15, msg="T")
        self.assertClose(
            near.at("detector_input")["adc"],
            far.at("detector_input")["adc"],
            atol=1e-18,
            msg="detector_input adc",
        )
        self.assertClose(
            near.factor("channel_output"),
            step * far.factor("channel_output"),
            atol=1e-15,
            msg="channel_output factor",
        )

        born = 1e-3
        self.assertClose(
            near.refer(born, "channel_output", "channel_input"),
            far.refer(born, "channel_output", "channel_input") / step,
            atol=1e-15,
            msg="span-born input referral",
        )

    def test_loss_pinned(self):
        """
        Itemised losses on a pinned q.Channel are refused by run() and explain().
        """
        link = q.Link(
            modulation=q.GaussianModulation(v_a=5.0),
            channel=q.Channel(T=0.5, xi=0.01, ref="output"),
            bob=q.Bob(detector=q.Heterodyne()),
            losses=(q.Connector(),),
        )

        with self.assertRaises(ValueError):
            link.run()

        with self.assertRaises(ValueError):
            link.explain()

    def test_link_chain(self):
        """
        A Link folds the declared chain into T: 7.12 dB of link costs key against 5.0 dB of
        fibre, and res.budget names every optic.
        """

        def link(losses):
            return q.Link(
                modulation=q.GaussianModulation(v_a=5.0),
                channel=q.Fiber(length=25.0),
                bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
                losses=losses,
            ).run()

        bare = link(())
        full = link(
            (
                q.Connector(loss=0.25, count=2, site="launch"),
                q.Splice(count=6),
                q.Coupling(loss=1.5, site="receive"),
            )
        )
        self.assertClose(bare.budget.T, 10**-0.5, atol=1e-15, msg="fibre alone")
        self.assertClose(-10.0 * math.log10(full.budget.T), 7.12, atol=1e-12, msg="7.12 dB of link")
        self.assertLess(full.key_rate, bare.key_rate, msg=f"key {full.key_rate} vs {bare.key_rate}")
        self.assertIn(
            "coupling_receive",
            [row.source for row in full.budget.losses],
            msg="coupling_receive row",
        )

    def test_link_budget(self):
        """
        res.budget totals the xi the rate used, one row per source, and answers .at() over the
        run's chain; a click family reports None.
        """
        res = raman((q.Coupling(loss=1.5, site="receive"),))
        rows = res.budget.at("channel_input")
        self.assertEqual(set(rows), {"channel", "raman"}, msg="one row per source")
        self.assertClose(
            res.budget.total,
            res.explain["xi_total"]["value"],
            atol=1e-15,
            msg="budget total vs xi_total",
        )
        self.assertClose(
            res.budget.at("detector_input")["raman"],
            res.budget.T * rows["raman"],
            atol=1e-18,
            msg="detector_input raman",
        )

        clicks = q.Link(
            modulation=q.DifferentialPhase(mu=0.2),
            channel=q.Fiber(length=25.0),
            bob=q.Bob(detector=q.ClickDetector()),
            security=q.IndividualAttack(qber=0.03),
        ).run()
        self.assertIsNone(clicks.budget, msg="click budget")

    def test_born_in_span(self):
        """
        Raman photons are born inside the span: Bob's 1.5 dB leaves their input-referred noise
        unchanged, while Alice's inflates it and costs key.
        """
        plain = raman(())
        near = raman((q.Coupling(loss=1.5, site="launch"),))
        far = raman((q.Coupling(loss=1.5, site="receive"),))
        step = 10.0 ** (-1.5 / 10.0)
        self.assertClose(near.budget.T, far.budget.T, atol=1e-15, msg="one total T")
        self.assertClose(
            far.budget.at("channel_input")["raman"],
            plain.budget.at("channel_input")["raman"],
            atol=1e-15,
            msg="receive-side input raman",
        )
        self.assertClose(
            near.budget.at("channel_input")["raman"],
            plain.budget.at("channel_input")["raman"] / step,
            atol=1e-15,
            msg="launch-side input raman",
        )
        self.assertLess(
            near.key_rate,
            far.key_rate,
            msg=f"key {near.key_rate} vs {far.key_rate}",
        )

    def test_raman_span(self):
        """
        assemble()'s Raman row refers in by launch*span where every other row uses the total T,
        as a q.Link run does.
        """
        kw = {
            "v_a": 5.0,
            "t": 10**-0.5,
            "raman_db": -80.0,
            "bandwidth": 100e6,
            "adc_bits": 12,
        }
        plain = budget.assemble(**kw)
        far = budget.assemble(losses=(q.Coupling(loss=1.5, site="receive"),), **kw)
        near = budget.assemble(losses=(q.Coupling(loss=1.5, site="launch"),), **kw)
        step = 10.0 ** (-1.5 / 10.0)
        self.assertClose(far.T, near.T, atol=1e-15, msg="T")
        self.assertClose(
            far.at("channel_input")["raman"],
            plain.at("channel_input")["raman"],
            atol=1e-15,
            msg="receive-side input raman",
        )
        self.assertClose(
            near.at("channel_input")["raman"],
            plain.at("channel_input")["raman"] / step,
            atol=1e-15,
            msg="launch-side input raman",
        )
        self.assertClose(
            far.at("channel_input")["adc"],
            plain.at("channel_input")["adc"] / far.receive,
            atol=1e-15,
            msg="receive-side input adc",
        )
        self.assertClose(
            far.at("detector_input")["raman"],
            far.T * far.at("channel_input")["raman"],
            atol=1e-18,
            msg="detector_input raman",
        )


if __name__ == "__main__":
    rc = Exam(
        "BudgetEngine",
        "Phase 3: xi assembled from hardware parameters",
        "budget_engine.md",
    ).run(load(Engine))
    rc |= Exam(
        "BudgetPlanes",
        "The four named planes and the algebra of referring a noise between them",
        "budget_planes.md",
    ).run(load(Planes))
    rc |= Exam(
        "BudgetOptics",
        "Itemised coupling, splice and connector losses, and where they sit",
        "budget_optics.md",
    ).run(load(Optics))
    rc |= Exam(
        "BudgetTierB",
        "Tier B: metro-envelope and Hajomer 2024 experimental reproduction",
        "budget_tierb.md",
    ).run(load(TierB))
    sys.exit(rc)
