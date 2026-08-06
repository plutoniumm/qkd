import ast
import dataclasses
import inspect
import math
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q
from kit.cache import memo
from kit.links import cow_link, frames, sim_link
from kit.procs import ROOT
from qkd import _core

LABELS = ("pinned", "derived", "default", "derived (run to compute)")

# The shipped package, listed once: three portability exams walk it.
SHIPPED = sorted(pathlib.Path(ROOT, "qkd").glob("*.py"))


def shadowed(node, name):
    bound, out = [], []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None:
            for sub in ast.walk(stmt.annotation):
                if isinstance(sub, ast.Name) and sub.id in bound:
                    out.append(f"{name}:{stmt.lineno} {node.name}.{sub.id}")
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                bound.append(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            bound += [t.id for t in stmt.targets if isinstance(t, ast.Name)]

    return out


MARKUP = (
    re.compile(r"\[[^\]\n]+\]\([^)\n]*\)"),
    re.compile(r"(?<![\w.])/(guide|tutorials|tests|usage|roadmap|architecture)\b"),
    re.compile(r":::|\[\[toc\]\]"),
)


def linked(path):
    """
    Docstring fragments in one shipped module that read as VitePress markup: a markdown link, a
    site-relative docs path, a container directive.
    """
    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, kinds):
            continue

        text = ast.get_docstring(node, clean=False)
        if not text:
            continue

        for pat in MARKUP:
            out += [f"{path.name}:{getattr(node, 'lineno', 1)} {hit.group(0)}" for hit in pat.finditer(text)]

    return out


def bare(mod=None, channel=None, bob=None, **kw):
    """
    The minimal Gaussian link the scope gates are written against.
    """

    return q.Link(
        modulation=q.GaussianModulation() if mod is None else mod,
        channel=q.Channel(T=0.5) if channel is None else channel,
        bob=q.Bob(detector=q.Homodyne()) if bob is None else bob,
        **kw,
    )


def phased(bob=None, **kw):
    """
    The same minimal link on the differential-phase path.
    """

    return q.Link(
        modulation=q.DifferentialPhase(),
        channel=q.Channel(T=0.1),
        bob=q.Bob(detector=q.ClickDetector()) if bob is None else bob,
        **kw,
    )


def cv_link(channel, detector=None, security=None):
    return q.Link(
        modulation=q.GaussianModulation(v_a=18.5),
        channel=channel,
        bob=q.Bob(detector=detector or q.Homodyne(eta=0.606, v_el=0.041)),
        security=security or q.Asymptotic(beta=0.898),
    )


def pinned_dsp():
    return q.Link(
        modulation=q.GaussianModulation(v_a=4.0),
        channel=q.Channel(T=0.4, xi=0.01, ref="input"),
        bob=q.Bob(detector=q.Heterodyne()),
        dsp=q.DSP(phase=q.PilotPhase(v_err=0.01)),
    )


def dps_link(detector=None, security=None):
    return q.Link(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Channel(T=0.1),
        bob=q.Bob(detector=detector or q.ClickDetector(eta=0.2, dark=1e-6)),
        security=security or q.IndividualAttack(qber=0.01, f=1.16),
    )


def click_link(detector=None, receiver=None, **kw):
    """
    The phase-keyed link with a simulated error rate; `kw` replaces any slot whole.
    """
    parts = dict(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Channel(T=0.1),
        alice=q.Alice(laser=q.Laser(linewidth=1e6), symbol_rate=1e9),
        bob=q.Bob(
            detector=detector or q.ClickDetector(eta=0.2, dark=1e-6),
            receiver=receiver or q.DelayInterferometer(delay=1, visibility=0.98),
        ),
        security=q.IndividualAttack(f=1.16),
    )
    parts.update(kw)

    return q.Link(**parts)


def basis_link(security=None):
    """
    Basis-keyed weak coherent pulses at the decoy set (0.48, 0.1, 0.0), not test/surface.py's `keyed` set.
    """

    return q.Link(
        modulation=q.BasisKeying(decoy=q.Decoy((0.48, 0.1, 0.0)), sift=0.5),
        channel=q.Fiber(length=25.0, alpha=0.21),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
            receiver=q.BasisAnalyser(misalign=0.033),
        ),
        security=security or q.SplittingAttack(f=1.22),
    )


def variant_link(**kw):
    """
    Basis keying with a third basis or a pair announcement selected.
    """

    return q.Link(
        modulation=q.BasisKeying(decoy=q.Decoy((0.2, 0.08, 0.0)), **kw),
        channel=q.Fiber(length=25.0, alpha=0.21),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
            receiver=q.BasisAnalyser(misalign=0.033),
        ),
        security=q.SplittingAttack(f=1.22),
    )


def state_link(eph=None, ref=False):
    """
    A two-state link and Bob's nulling receiver.
    """

    return q.Link(
        modulation=q.TwoStateKeying(mu=0.23, reference=ref),
        channel=q.Channel(T=0.6),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=1e-6),
            receiver=q.NullingReceiver(visibility=0.98),
        ),
        security=q.DiscriminationBound(e_phase=eph, f=1.16),
    )


def violated_link(s=2.7, source="measured"):
    """
    A photon-pair link priced by a Bell violation.
    """

    return q.PairLink(
        source=q.PairSource(brightness=0.01),
        detectors=(q.ClickDetector(eta=0.2, dark=1e-6),) * 2,
        channels=(q.Fiber(length=10.0),) * 2,
        security=q.ViolationBound(s=s, source=source),
    )


@memo
def metro(security, symbols=200_000, imp=()):
    """
    Shared so eleven configurations come from two pipeline runs.
    """

    return sim_link(security, imp=imp).claim(frames(symbols))


class Components(Question):
    """
    Component validation paths and frozen-dataclass immutability.
    """

    def test_modulation_bounds(self):
        """
        `GaussianModulation` rejects $V_A \\le 0$ and `DifferentialPhase` rejects $\\mu \\le 0$,
        each with a `ValueError` naming the parameter.
        """
        self.assertFails(ValueError, "v_a", lambda: q.GaussianModulation(v_a=0.0), msg="v_a = 0")
        self.assertFails(
            ValueError,
            "v_a",
            lambda: q.GaussianModulation(v_a=-1.0),
            msg="negative v_a",
        )
        self.assertFails(ValueError, "mu", lambda: q.DifferentialPhase(mu=0.0), msg="mu = 0")
        self.assertFails(ValueError, "mu", lambda: q.DifferentialPhase(mu=-0.1), msg="negative mu")

    def test_detector_bounds(self):
        """
        All three detectors reject $\\eta$ outside (0, 1]; the quadrature ones reject negative
        $v_{el}$, the click one a dark rate outside [0, 1).
        """
        for cls in (q.Homodyne, q.Heterodyne, q.ClickDetector):
            self.assertFails(ValueError, "eta", lambda k=cls: k(eta=0.0), msg="eta = 0")
            self.assertFails(ValueError, "eta", lambda k=cls: k(eta=1.5), msg="eta above 1")
            self.assertFails(ValueError, "eta", lambda k=cls: k(eta=-0.2), msg="negative eta")

        for cls in (q.Homodyne, q.Heterodyne):
            self.assertFails(ValueError, "v_el", lambda k=cls: k(v_el=-1e-9), msg="negative v_el")
        self.assertFails(ValueError, "dark", lambda: q.ClickDetector(dark=1.0), msg="dark = 1")
        self.assertFails(ValueError, "dark", lambda: q.ClickDetector(dark=-1e-9), msg="negative dark")

    def test_channel_bounds(self):
        """
        `Channel` rejects $T$ outside (0, 1] and a negative $\\xi$.
        """
        for bad in (0.0, -0.1, 1.5):
            self.assertFails(ValueError, "T", lambda t=bad: q.Channel(T=t), msg=f"T = {bad}")
        self.assertFails(
            ValueError,
            "xi",
            lambda: q.Channel(T=0.5, xi=-1e-9, ref="input"),
            msg="negative xi",
        )

    def test_channel_ref(self):
        """
        A nonzero $\\xi$ with no `ref` is refused, and `ref` must name one of the two planes,
        case-sensitively.
        """
        self.assertFails(ValueError, "ref", lambda: q.Channel(T=0.5, xi=0.01), msg="xi without ref")
        self.assertFails(
            ValueError,
            "ref",
            lambda: q.Channel(T=0.5, xi=0.01, ref="bob"),
            msg="unknown ref string",
        )
        self.assertFails(
            ValueError,
            "ref",
            lambda: q.Channel(T=0.5, xi=0.01, ref="INPUT"),
            msg="ref is case sensitive",
        )
        self.assertEqual(q.Channel(T=0.5).xi, 0.0, msg="a lossy channel needs no ref")

    def test_fiber_exclusive(self):
        """
        `Fiber` takes the hardware description or the derived $T$, never both and never neither,
        and validates each.
        """
        self.assertFails(
            ValueError,
            "both",
            lambda: q.Fiber(length=25.0, T=0.3),
            msg="length and T together",
        )
        self.assertFails(ValueError, "either", q.Fiber, msg="neither length nor T")
        self.assertFails(ValueError, "length", lambda: q.Fiber(length=0.0), msg="length = 0")
        self.assertFails(ValueError, "alpha", lambda: q.Fiber(length=1.0, alpha=-0.1), msg="alpha")
        self.assertFails(ValueError, "T", lambda: q.Fiber(T=1.5), msg="T above 1")

    def test_security_bounds(self):
        """
        `Asymptotic` and `FiniteSize` reject $\\beta$ outside (0, 1], and `FiniteSize` also
        bounds $\\epsilon$, $n$ and `pe_fraction`.
        """
        for cls in (q.Asymptotic, q.FiniteSize):
            self.assertFails(ValueError, "beta", lambda k=cls: k(beta=0.0), msg="beta = 0")
            self.assertFails(ValueError, "beta", lambda k=cls: k(beta=1.2), msg="beta above 1")
        self.assertFails(ValueError, "eps", lambda: q.FiniteSize(eps=0.0), msg="eps = 0")
        self.assertFails(ValueError, "eps", lambda: q.FiniteSize(eps=1.0), msg="eps = 1")
        self.assertFails(ValueError, "n", lambda: q.FiniteSize(n=0.0), msg="n = 0")
        self.assertFails(
            ValueError,
            "pe_fraction",
            lambda: q.FiniteSize(pe_fraction=1.0),
            msg="pe_fraction = 1",
        )

    def test_decoy_ordering(self):
        """
        `Decoy` refuses anything but three positive decreasing intensities with the decoys
        together weaker than the signal.
        """
        self.assertFails(
            ValueError,
            "nu2 < nu1",
            lambda: q.Decoy((0.5, 0.1, 0.2)),
            msg="decoys must decrease",
        )
        self.assertFails(
            ValueError,
            "weaker than the signal",
            lambda: q.Decoy((0.3, 0.2, 0.15)),
            msg="nu1 + nu2 must stay under mu",
        )
        self.assertFails(
            ValueError,
            "intensities",
            lambda: q.Decoy((0.5, 0.1)),
            msg="a pair is not an intensity set",
        )
        self.assertFails(
            ValueError,
            "positive",
            lambda: q.Decoy((0.0, 0.1, 0.0)),
            msg="an empty signal state",
        )
        self.assertClose(q.Decoy().signal, 0.5, msg="signal is the first intensity")

    def test_click_optics(self):
        """
        The click-family receivers validate on construction: a delay of at least one symbol, a
        contrast in [0, 1], a misalignment no worse than chance, a split that leaves a data
        line.
        """
        self.assertFails(
            ValueError,
            "delay",
            lambda: q.DelayInterferometer(delay=0),
            msg="delay = 0",
        )
        self.assertFails(
            ValueError,
            "visibility",
            lambda: q.DelayInterferometer(visibility=1.5),
            msg="contrast above 1",
        )

        for cls in (q.BasisAnalyser, q.CoherenceMonitor):
            self.assertFails(ValueError, "misalign", lambda k=cls: k(misalign=0.6), msg="misalign")
        self.assertFails(
            ValueError,
            "split",
            lambda: q.CoherenceMonitor(split=1.0),
            msg="the whole beam tapped away",
        )
        self.assertFails(ValueError, "sift", lambda: q.BasisKeying(sift=0.0), msg="sift = 0")
        self.assertFails(ValueError, "mu", lambda: q.IntensityKeying(mu=0.0), msg="empty slots")
        self.assertFails(
            ValueError,
            "decoy_frac",
            lambda: q.IntensityKeying(decoy_frac=1.0),
            msg="every sequence a monitor",
        )

    def test_click_security(self):
        """
        The per-family security components bound their own inputs: a phase error in [0, 1/2] and
        every `f` at or above the Shannon limit.
        """
        self.assertFails(
            ValueError,
            "e_phase",
            lambda: q.PhaseBound(e_phase=0.6),
            msg="a phase error past chance",
        )

        for make in (
            lambda: q.PhaseBound(e_phase=0.1, f=0.9),
            lambda: q.SplittingAttack(f=0.9),
        ):
            self.assertFails(ValueError, "f", make, msg="f below the Shannon limit")
        self.assertIsNone(q.IndividualAttack().qber, msg="an unpinned qber is derived, not defaulted")

    def test_qber_domain(self):
        """
        `q.IndividualAttack` refuses a QBER outside [0, 1/2) and an `f` below the Shannon limit.
        """
        self.assertFails(ValueError, "qber", lambda: q.IndividualAttack(qber=0.5), msg="qber = 0.5")
        self.assertFails(ValueError, "qber", lambda: q.IndividualAttack(qber=0.9), msg="qber = 0.9")
        self.assertFails(
            ValueError,
            "qber",
            lambda: q.IndividualAttack(qber=-1e-9),
            msg="negative qber",
        )
        self.assertFails(
            ValueError,
            "f",
            lambda: q.IndividualAttack(qber=0.01, f=0.99),
            msg="f below 1",
        )

    def test_nonfinite_f(self):
        """
        A non-finite `f` is refused on every public path that charges one, by the Rust `f_ec`
        guard, the components' comparison against 1.0 admitting NaN and `+inf`.
        """
        builds = (
            lambda v: dps_link(security=q.IndividualAttack(qber=0.01, f=v)),
            lambda v: basis_link(security=q.SplittingAttack(f=v)),
            lambda v: cow_link(security=q.PhaseBound(e_phase=0.2, f=v)),
        )
        for bad in (float("nan"), float("inf")):
            for build in builds:
                self.assertFails(
                    ValueError,
                    "Shannon limit",
                    lambda b=build, v=bad: b(v).run(),
                    msg=f"f = {bad} must be refused before it becomes a rate",
                )

    def test_hardware_bounds(self):
        """
        The front-end hardware components validate on construction: bit depths, tones,
        linewidths, symbol rate and Bob's mandatory detector.
        """
        for cls in (q.ADC, q.IQModulator):
            self.assertFails(ValueError, "bits", lambda k=cls: k(bits=0), msg="bits = 0")
        self.assertFails(ValueError, "tones", lambda: q.Pilots(tones=0), msg="tones")

        for cls in (q.Laser, q.LocalLO):
            self.assertFails(
                ValueError,
                "linewidth",
                lambda k=cls: k(linewidth=-1.0),
                msg="negative linewidth",
            )
        self.assertFails(ValueError, "symbol_rate", lambda: q.Alice(symbol_rate=0.0), msg="rate = 0")
        self.assertFails(ValueError, "detector", lambda: q.Bob(detector=None), msg="Bob needs one")

    def test_frozen_components(self):
        """
        Components are frozen dataclasses: rebinding a field raises.
        """
        parts = [
            q.Channel(T=0.5),
            q.Homodyne(),
            q.GaussianModulation(),
            q.Asymptotic(),
            q.Fiber(length=25.0),
        ]
        for part in parts:
            field = next(iter(part.__dataclass_fields__))

            with self.assertRaises(AttributeError):
                setattr(part, field, 0.123)

    def test_dark_rate(self):
        """
        `ClickDetector.gated` takes the dark floor as a free-running count rate over a gate
        width and converts it to the per-gate probability the field holds, $1 - e^{-rg}$,
        passing every other field through.
        """
        det = q.ClickDetector.gated(rate=1e3, gate=1e-9, eta=0.15, afterpulse=0.02)
        self.assertClose(det.dark, -math.expm1(-1e-6), atol=1e-21, msg="gated() is not the Poisson complement")
        self.assertClose(det.dark, 1e-6, atol=1e-12, msg="a gated figure must be rate*gate to parts in 1e6")
        self.assertEqual(det.eta, 0.15, msg="gated() dropped a passed-through field")
        self.assertEqual(det.afterpulse, 0.02, msg="gated() dropped a passed-through field")

        # Gate width is not the clock period: a 10% duty cycle is a floor ten times smaller.
        period = q.ClickDetector.gated(rate=1e3, gate=1e-9)
        window = q.ClickDetector.gated(rate=1e3, gate=1e-10)
        self.assertClose(period.dark / window.dark, 10.0, atol=1e-5, msg="duty cycle must scale the floor")

    def test_dark_refusals(self):
        """
        `gated()` refuses a dark floor stated twice and a gate of zero width, and leaves the
        per-gate constructor and its published default untouched.
        """
        self.assertFails(
            ValueError,
            "gated() derives dark from a count rate and a gate width",
            lambda: q.ClickDetector.gated(1e3, 1e-9, dark=1e-6),
            msg="a floor stated twice must be refused",
        )
        self.assertFails(ValueError, "gate must be positive", lambda: q.ClickDetector.gated(1e3, 0.0), msg="zero gate")
        self.assertFails(ValueError, "rate must be nonnegative", lambda: q.ClickDetector.gated(-1.0, 1e-9), msg="rate")
        self.assertEqual(q.ClickDetector().dark, 1e-6, msg="the per-gate default moved")
        self.assertEqual(q.ClickDetector(dark=8.5e-7).dark, 8.5e-7, msg="GYS's own per-gate figure moved")

    def test_frozen_result(self):
        """
        `LinkResult` is frozen, and on a pinned channel its oracle is the result itself.
        """
        res = cv_link(q.Channel(T=0.302, xi=0.005, ref="input")).run()

        with self.assertRaises(AttributeError):
            res.key_rate = 1.0
        self.assertIs(res.oracle, res, msg="a run that estimates nothing is its oracle")


class Explain(Question):
    """
    The explain() contract: labelled provenance matching what run() used.
    """

    def test_explain_labels(self):
        """
        Every entry carrying a value labels its provenance pinned, derived or default, and
        carries nothing else.
        """
        links = [
            cv_link(q.Channel(T=0.302, xi=0.005, ref="input")),
            cv_link(q.Fiber(length=25.0), security=q.FiniteSize(n=1e9)),
            pinned_dsp(),
            dps_link(),
            click_link(),
            basis_link(),
            cow_link(),
        ]
        for link in links:
            for key, entry in link.explain().items():
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                self.assertEqual(set(entry), {"value", "label"}, msg=f"{key} entry shape")
                self.assertIn(label, LABELS, msg=f"{key} has label {label!r}")

    def test_explain_stages(self):
        """
        The pinned CV path reports closed-form-only stages and names its protocol, detection and
        security, no DSP chain having run.
        """
        info = cv_link(q.Channel(T=0.302, xi=0.005, ref="input")).explain()
        self.assertEqual(info["stages"], "closed-form only", msg="no DSP stages yet")
        self.assertEqual(info["protocol"], "gaussian modulation", msg="protocol must be named")
        self.assertEqual(info["detection"], "homodyne", msg="detection must be named")
        self.assertEqual(info["security"], "Asymptotic", msg="security must be named")

    def test_explain_detection(self):
        """
        `explain()` reports the detection dispatched on, so a `Heterodyne` receiver never reads
        back as homodyne.
        """
        info = cv_link(q.Channel(T=0.5), detector=q.Heterodyne()).explain()
        dps = dps_link().explain()
        self.assertEqual(info["detection"], "heterodyne", msg="heterodyne detection")
        self.assertEqual(dps["detection"], "click", msg="DPS detection is a click")
        self.assertEqual(dps["stages"], "closed-form only", msg="DPS has no stages")

    def test_explain_matches(self):
        """
        `explain()` reports the numbers `run()` computes with: the run adds `key_raw` and the
        assembled `xi_total` and nothing else, and dropping those two leaves the dicts
        identical.
        """
        link = cv_link(q.Channel(T=0.302, xi=0.005, ref="input"))
        info = link.explain()
        res = link.run()
        used = dict(res.explain)

        # xi_total is run-side: explain() has no impairment rows to total.
        self.assertEqual(
            set(used) - set(info),
            {"key_raw", "xi_total"},
            msg="a run adds the rate and the total, and no other row",
        )
        self.assertClose(
            used["xi_total"]["value"],
            0.005,
            atol=0.0,
            msg="with no phase row the total is the channel alone",
        )

        used.pop("key_raw")
        used.pop("xi_total")
        self.assertEqual(used, info, msg="explain() must match the run() inputs")
        self.assertClose(res.explain["key_raw"]["value"], res.key_rate, msg="key_raw is the rate")

    def test_explain_finite(self):
        """
        `FiniteSize` adds $n$, $\\epsilon$ and `pe_fraction` to the report as pinned inputs.
        """
        sec = q.FiniteSize(beta=0.898, eps=1e-10, n=1e9, pe_fraction=0.4)
        info = cv_link(q.Channel(T=0.302, xi=0.005, ref="input"), security=sec).explain()
        self.assertEqual(info["security"], "FiniteSize", msg="security name")

        for key in ("n", "eps", "pe_fraction"):
            self.assertEqual(info[key]["label"], "pinned", msg=f"{key} is pinned")

        self.assertClose(info["n"]["value"], 1e9, atol=0.0, msg="block size must round-trip")
        self.assertClose(info["pe_fraction"]["value"], 0.4, msg="pe_fraction must round-trip")

    def test_explain_dps(self):
        """
        The DPS report derives `p_click` from $\\mu$, $T$ and $\\eta$.
        """
        link = dps_link()
        info = link.explain()
        res = link.run()
        self.assertEqual(info["p_click"]["label"], "derived", msg="p_click is derived")
        self.assertClose(info["p_click"]["value"], res.p_click, msg="p_click must match the run")
        self.assertEqual(info["qber"]["label"], "pinned", msg="pinned qber")
        self.assertClose(res.qber, 0.01, msg="qber must round-trip")

    def test_explain_checks(self):
        """
        `explain()` runs the same scope check as `run()`.
        """
        link = bare(alice=q.Alice(laser=q.Laser()))
        self.assertFails(
            NotImplementedError,
            "budget.assemble",
            link.explain,
            msg="explain must gate too",
        )

        bad = bare(channel=0.5)
        self.assertFails(ValueError, "channel", bad.explain, msg="a bare float is not a channel")


class Pinning(Question):
    """
    Noise-plane conversion, fibre derivation and bit-for-bit determinism.
    """

    def test_plane_labels(self):
        """
        An output-referred $\\xi$ is reported as derived at its input-plane value $\\xi/T$; an
        input-referred one as pinned, unchanged.
        """
        out = cv_link(q.Channel(T=0.25, xi=0.01, ref="output")).explain()
        inp = cv_link(q.Channel(T=0.25, xi=0.01, ref="input")).explain()
        self.assertEqual(out["xi_input"]["label"], "derived", msg="output is converted")
        self.assertClose(out["xi_input"]["value"], 0.04, msg="xi_in = xi_out / T")
        self.assertEqual(inp["xi_input"]["label"], "pinned", msg="input is verbatim")
        self.assertClose(inp["xi_input"]["value"], 0.01, msg="input xi is untouched")

    def test_plane_direction(self):
        """
        An output-referred $\\xi$ divides by $T$ and costs key rate against the same number
        input-referred.
        """
        out = cv_link(q.Channel(T=0.302, xi=0.01, ref="output")).run()
        inp = cv_link(q.Channel(T=0.302, xi=0.01, ref="input")).run()
        self.assertLess(out.key_rate, inp.key_rate, msg="output plane is noisier")
        self.assertGreater(out.chi_be, inp.chi_be, msg="Eve must gain by the division")

    def test_fiber_derivation(self):
        """
        `Fiber(length, alpha)` derives $T = 10^{-\\alpha L/10}$ and labels it derived; a pinned
        $T$ reports pinned; $\\xi$ is default either way, a `Fiber` carrying no excess noise.
        """
        info = cv_link(q.Fiber(length=25.0, alpha=0.2)).explain()
        pinned = cv_link(q.Fiber(T=0.302)).explain()
        self.assertEqual(info["T"]["label"], "derived", msg="length gives derived T")
        self.assertClose(info["T"]["value"], 10.0**-0.5, msg="T = 10^(-alpha L / 10)")
        self.assertEqual(pinned["T"]["label"], "pinned", msg="a given T is pinned")
        self.assertClose(pinned["T"]["value"], 0.302, msg="pinned T is verbatim")
        self.assertEqual(
            info["xi_input"]["label"],
            "default",
            msg="a Fiber carries no excess noise yet",
        )

    def test_determinism(self):
        """
        Two identically constructed `q.Link`s return bit-identical floats: nothing in the
        closed-form path is sampled.
        """
        chan = q.Channel(T=0.302, xi=0.005, ref="input")
        one = cv_link(chan).run()
        two = cv_link(chan).run()
        self.assertEqual(one.key_rate, two.key_rate, msg="key rate must be identical")
        self.assertEqual(one.i_ab, two.i_ab, msg="I_AB must be identical")
        self.assertEqual(one.chi_be, two.chi_be, msg="Holevo must be identical")
        self.assertEqual(one.explain, two.explain, msg="reports must be identical")

    def test_dac_off_pipeline(self):
        """
        A described DAC is charged onto the claim, not sampled: one set of frames gives the same
        estimates at 16 and 6 bits, and 6 bits costs key.
        """
        kept = frames(200_000)
        fine = sim_link(q.Asymptotic(), iq=q.IQModulator(bits=16)).claim(kept)
        coarse = sim_link(q.Asymptotic(), iq=q.IQModulator(bits=6)).claim(kept)
        self.assertEqual(fine.est, coarse.est, msg="one measurement, two claims")
        self.assertLess(coarse.key_rate, fine.key_rate, msg="6 bits must cost key")
        self.assertGreater(
            coarse.explain["xi_claimed"]["value"],
            fine.explain["xi_claimed"]["value"],
            msg="the coarser DAC is claimed at the higher plane",
        )

    def test_measure_matches_run(self):
        """
        One measurement claimed under two security models reproduces two independent runs bit
        for bit.
        """
        for sec in (q.Asymptotic(), q.FiniteSize(n=1e8)):
            fresh = sim_link(sec).run(200_000, seed=1)
            kept = sim_link(sec).claim(frames(200_000))
            name = type(sec).__name__
            self.assertEqual(kept.key_rate, fresh.key_rate, msg=f"{name}: rate")
            self.assertEqual(kept.t_min, fresh.t_min, msg=f"{name}: interval")
            self.assertEqual(kept.dsp, fresh.dsp, msg=f"{name}: DSP chain")
            self.assertEqual(kept.est, fresh.est, msg=f"{name}: estimates")
            self.assertEqual(kept.budget, fresh.budget, msg=f"{name}: budget")
            self.assertEqual(kept.explain, fresh.explain, msg=f"{name}: report")


class Scope(Question):
    """
    Phase gating: an out-of-scope configuration raises rather than return a number.
    """

    def test_dsp_component(self):
        """
        The `dsp` slot takes only a `q.DSP`, and the pilot chain is heterodyne-only.
        """
        bad = bare(bob=q.Bob(detector=q.Heterodyne()), dsp="pilot-assisted")
        self.assertFails(NotImplementedError, "DSP", bad.run, msg="a string is not a DSP chain")

        hom = bare(dsp=q.DSP(phase=q.PilotPhase(v_err=0.01)))
        self.assertFails(NotImplementedError, "heterodyne", hom.run, msg="the pilot chain needs I/Q")

    def test_budget_gated(self):
        """
        A laser, IQ modulator, pilots, ADC or local oscillator on the CV path with no DSP chain is
        refused, naming `qkd.budget.assemble`.
        """
        alice = [
            q.Alice(laser=q.Laser()),
            q.Alice(iq=q.IQModulator()),
            q.Alice(pilots=q.Pilots()),
        ]
        for hw in alice:
            link = bare(alice=hw)
            self.assertFails(
                NotImplementedError,
                "budget.assemble",
                link.run,
                msg="the refusal must name the budget layer",
            )

        for bob in (
            q.Bob(detector=q.Homodyne(), adc=q.ADC()),
            q.Bob(detector=q.Homodyne(), lo=q.LocalLO()),
        ):
            link = bare(bob=bob)
            self.assertFails(
                NotImplementedError,
                "budget.assemble",
                link.run,
                msg="Bob hardware is gated",
            )

    def test_dps_gated(self):
        """
        The phase-keyed path refuses the CV DSP chain, an LO behind a threshold detector, and
        any security analysis past the individual attack.
        """
        chain = phased(
            dsp=q.DSP(phase=q.PilotPhase(v_err=0.01)),
            security=q.IndividualAttack(qber=0.01),
        )
        self.assertFails(
            NotImplementedError,
            "Gaussian-modulation only",
            chain.run,
            msg="the pilot chain is not a DPS receiver",
        )

        hw = phased(
            bob=q.Bob(detector=q.ClickDetector(), lo=q.LocalLO()),
            security=q.IndividualAttack(qber=0.01),
        )
        self.assertFails(ValueError, "local oscillator", hw.run, msg="a click receiver has no LO")
        self.assertFails(
            NotImplementedError,
            "IndividualAttack",
            dps_link(security=q.Asymptotic()).run,
            msg="other DPS bounds are gated",
        )

    def test_ignored_fields(self):
        """
        An unconsumed field is refused: detector memory beside a pinned qber, and an
        interferometer beside a pinned error rate.
        """
        for det in (
            q.ClickDetector(dead_time=50e-9),
            q.ClickDetector(afterpulse=0.02),
        ):
            self.assertFails(
                ValueError,
                "leave qber unpinned",
                dps_link(detector=det).run,
                msg="detector memory needs the simulation",
            )

        pinned = phased(
            bob=q.Bob(detector=q.ClickDetector(), receiver=q.DelayInterferometer(delay=2)),
            security=q.IndividualAttack(qber=0.01),
        )
        self.assertFails(
            ValueError,
            "closed form",
            pinned.run,
            msg="a pinned qber consumes no interferometer",
        )

    def test_window_gated(self):
        """
        The three closed forms refuse an acceptance window by name, the sampled path refuses one
        with no jitter to gate, and beside a jitter it reaches `window_loss`.
        """
        held = basis_link()
        held.bob = q.Bob(
            detector=q.ClickDetector(eta=0.045, dark=8.5e-7, window=100e-12),
            receiver=q.BasisAnalyser(misalign=0.033),
        )
        two = q.Link(
            modulation=q.TwoStateKeying(mu=0.23),
            channel=q.Channel(T=0.6),
            bob=q.Bob(
                detector=q.ClickDetector(eta=1.0, dark=1e-6, window=100e-12),
                receiver=q.NullingReceiver(visibility=0.98),
            ),
            security=q.DiscriminationBound(f=1.16),
        )
        closed = (
            ("a pinned qber", dps_link(detector=q.ClickDetector(eta=0.2, dark=1e-6, window=100e-12))),
            ("the decoy gains", held),
            ("the two-state gains", two),
        )
        for why, link in closed:
            self.assertFails(ValueError, "window", link.run, msg=f"{why} reads no acceptance window")

        alone = click_link(detector=q.ClickDetector(eta=0.2, dark=1e-6, window=100e-12))
        self.assertFails(
            ValueError,
            "window",
            lambda: alone.run(2000, 1),
            msg="a window with no response to gate is refused on the sampled path too",
        )

        gated = click_link(detector=q.ClickDetector(eta=0.2, dark=1e-6, jitter=400e-12, window=100e-12))
        res = gated.run(2000, 1)
        self.assertGreater(
            res.explain["window_loss"]["value"],
            0.0,
            msg="declared beside a jitter it is read and reported",
        )

    def test_derived_qber(self):
        """
        A derived error rate needs Bob's delay interferometer and Alice's laser.
        """
        blind = phased(alice=q.Alice(laser=q.Laser()), security=q.IndividualAttack())
        self.assertFails(
            ValueError,
            "DelayInterferometer",
            blind.run,
            msg="no receiver to derive from",
        )

        dark = phased(
            bob=q.Bob(detector=q.ClickDetector(), receiver=q.DelayInterferometer()),
            security=q.IndividualAttack(),
        )
        self.assertFails(ValueError, "laser", dark.run, msg="no source to derive from")

    def test_family_pairing(self):
        """
        Neither receiver nor security model is cross-applied: basis keying needs a
        `q.BasisAnalyser` and the splitting bound, intensity keying a `q.PhaseBound`.
        """
        swapped = q.Link(
            modulation=q.BasisKeying(),
            channel=q.Channel(T=0.3),
            bob=q.Bob(detector=q.ClickDetector(), receiver=q.CoherenceMonitor()),
            security=q.SplittingAttack(),
        )
        self.assertFails(
            ValueError,
            "BasisAnalyser",
            swapped.run,
            msg="the wrong receiver for the family",
        )

        crossed = q.Link(
            modulation=q.IntensityKeying(),
            channel=q.Channel(T=0.3),
            bob=q.Bob(detector=q.ClickDetector(), receiver=q.CoherenceMonitor()),
            security=q.SplittingAttack(),
        )
        self.assertFails(
            NotImplementedError,
            "PhaseBound",
            crossed.run,
            msg="security is per family, never cross-applied",
        )

    def test_detector_pairing(self):
        """
        Gaussian modulation needs a quadrature receiver and differential phase a click detector;
        anything else is refused.
        """
        cv = bare(bob=q.Bob(detector=q.ClickDetector()))
        self.assertFails(
            NotImplementedError,
            "Homodyne",
            cv.run,
            msg="CV needs a quadrature receiver",
        )

        dps = phased(bob=q.Bob(detector=q.Homodyne()), security=q.IndividualAttack(qber=0.01))
        self.assertFails(NotImplementedError, "ClickDetector", dps.run, msg="DPS needs clicks")

    def test_security_pairing(self):
        """
        Gaussian modulation accepts only `q.Asymptotic` and `q.FiniteSize`; an individual-attack
        bound belongs to the DPS path.
        """
        link = bare(security=q.IndividualAttack(qber=0.01))
        self.assertFails(NotImplementedError, "Asymptotic", link.run, msg="wrong security model")

    def test_unknown_parts(self):
        """
        An unrecognised modulation or channel object is refused, not coerced.
        """
        mod = bare(mod="gaussian")
        self.assertFails(
            NotImplementedError,
            "modulation",
            mod.run,
            msg="a string is not a component",
        )

        chan = bare(channel=0.5)
        self.assertFails(ValueError, "channel", chan.run, msg="a bare float is not a channel")

    def test_seam_gated(self):
        """
        `measure()` refuses a click family, and `claim()` refuses a measurement from a changed
        channel rather than re-run it.
        """
        moved = sim_link(q.Asymptotic())
        moved.channel = q.Channel(T=0.4, xi=0.02, ref="input")
        self.assertFails(
            NotImplementedError,
            "Gaussian-modulation",
            dps_link().measure,
            msg="a click family has no separable pipeline",
        )
        self.assertFails(
            ValueError,
            "different pipeline",
            lambda: moved.claim(frames(200_000)),
            msg="a changed channel invalidates the measurement",
        )


class Estimation(Question):
    """
    The interval must come from the smaller of the declared block and the frames the simulation
    produced.
    """

    def test_pairs_reported(self):
        """
        `pe_pairs` reports two pairs per heterodyne symbol, derived, far below the declared block.
        """
        res = metro(q.FiniteSize(n=1e8))
        info = res.explain["pe_pairs"]
        self.assertEqual(info["label"], "derived", msg="the count is derived")
        self.assertClose(
            info["value"],
            2.0 * res.dsp.symbols,
            atol=0.0,
            msg="two pairs per heterodyne symbol",
        )
        self.assertLess(info["value"], 0.5e8, msg="far below the declared block")

    def test_block_cannot_buy(self):
        """
        Raising $n$ from 1e8 to 1e9 at fixed frames leaves the confidence interval unchanged and
        improves only $\\Delta$.
        """
        one = metro(q.FiniteSize(n=1e8))
        two = metro(q.FiniteSize(n=1e9))
        self.assertEqual(one.t_min, two.t_min, msg="the interval is the sample's")
        self.assertEqual(one.xi_max, two.xi_max, msg="the interval is the sample's")
        self.assertGreater(two.key_rate, one.key_rate, msg="Delta may still improve")
        self.assertLess(two.delta, one.delta, msg="and it is Delta that moved")

    def test_declared_governs(self):
        """
        Where the simulation outruns the declared block the declared count governs, and the
        result is exactly the closed-form finite-size path.
        """
        res = metro(q.FiniteSize(n=1e5))
        out = _core.cv_finite(
            5.0,
            min(1.0, max(1e-12, res.est.T)),
            max(0.0, res.est.xi),
            0.6,
            0.1,
            0.95,
            False,
            True,
            1e5,
            0.5,
            1e-10,
            1e-10,
            1e-10,
        )
        self.assertEqual(res.t_min, out[3], msg="the declared interval is untouched")
        self.assertEqual(res.xi_max, out[4], msg="and so is its excess noise")
        self.assertEqual(res.key_rate, out[2], msg="and so is the rate")


class Reconciliation(Question):
    """
    Frame error rate: a reconciliation frame that fails to converge is discarded with its key,
    and the rate scales with the share that survive.
    """

    def test_fer_scales(self):
        """
        A 40% frame error rate leaves 60% of the rate on both estimation paths, with $I_{AB}$,
        the Holevo bound and the interval untouched.
        """
        for n in (1e5, 1e8):
            free = metro(q.FiniteSize(n=n))
            lossy = metro(q.FiniteSize(n=n, fer=0.4))
            self.assertClose(lossy.key_rate, 0.6 * free.key_rate, msg=f"n={n:.0e}: 60% survives")
            self.assertEqual(lossy.i_ab, free.i_ab, msg="I_AB is not reconciliation")
            self.assertEqual(lossy.chi_be, free.chi_be, msg="nor is the Holevo bound")
            self.assertEqual(lossy.t_min, free.t_min, msg="nor is the interval")

    def test_fer_not_modelled(self):
        """
        An unset FER reports as default with no value, not a measured zero, a published FER
        pairing with its $\\beta$ and code rate (Hajomer arXiv:2305.08156 reports 0.59 beside
        $\\beta = 0.925$ on a rate-0.05 MET-LDPC).
        """
        free = metro(q.FiniteSize(n=1e8))
        zero = metro(q.FiniteSize(n=1e8, fer=0.0))
        row = free.explain["fer"]
        self.assertIsNone(row["value"], msg="an unset fer has no value to report")
        self.assertEqual(row["label"], "default", msg="and is labelled as unset")
        self.assertEqual(zero.explain["fer"]["label"], "pinned", msg="a measured zero is pinned")
        self.assertEqual(zero.key_rate, free.key_rate, msg="zero is the same rate")

    def test_variance_sign(self):
        """
        A negative phase-error variance is refused, not clamped: $v_{err} = -1$ would give
        -6.49 SNU at $V_A = 5$.
        """
        self.assertFails(
            ValueError,
            "nonnegative",
            lambda: q.budget.phase(5.0, -1.0),
            msg="a negative variance is the caller's error",
        )
        self.assertFails(
            ValueError,
            "nonnegative",
            lambda: q.budget.assemble(v_a=5.0, t=0.5, v_err=-1e-9),
            msg="and the budget must not assemble one",
        )
        self.assertEqual(q.budget.phase(5.0, 0.0), 0.0, msg="a perfect estimator costs nothing")

    def test_fer_domain(self):
        """
        The FER is refused outside [0, 1], and at 1 the rate is zero while the oracle rate stays
        positive.
        """
        self.assertFails(ValueError, "fer", lambda: q.FiniteSize(fer=1.5), msg="fer above 1")
        self.assertFails(ValueError, "fer", lambda: q.FiniteSize(fer=-1e-9), msg="negative fer")

        dead = metro(q.FiniteSize(n=1e8, fer=1.0))
        self.assertEqual(dead.key_rate, 0.0, msg="every frame lost is no key")
        self.assertGreater(dead.oracle.key_rate, 0.0, msg="while the channel itself still carries one")


class Variants(Question):
    """
    The four protocol variants sitting on existing shapes: two as parameters on `q.BasisKeying`,
    a two-state modulation with its own receiver, and a violation standing where a phase error
    stands.
    """

    def test_basis_domains(self):
        """
        `q.BasisKeying` refuses a basis count other than 2 or 3, an unknown announcement, and a
        pair announcement in three bases.
        """
        self.assertFails(
            ValueError,
            "bases must be 2",
            lambda: q.BasisKeying(bases=4),
            msg="no four-basis bound ships",
        )
        self.assertFails(
            ValueError,
            "bases must be 2",
            lambda: q.BasisKeying(bases=1),
            msg="one basis is not a protocol",
        )
        self.assertFails(
            ValueError,
            "announce must be",
            lambda: q.BasisKeying(announce="state"),
            msg="an unknown announcement must name the two that are read",
        )
        self.assertFails(
            ValueError,
            "no three-basis form",
            lambda: q.BasisKeying(bases=3, announce="pair"),
            msg="the pair announcement is written for four states in two bases",
        )

    def test_sift_is_not_doubled(self):
        """
        The pair announcement refuses `sift`, its conclusive fraction already inside the gains.
        """
        self.assertFails(
            ValueError,
            "takes no sift",
            lambda: q.BasisKeying(announce="pair", sift=0.25),
            msg="charging the conclusive fraction twice must be refused",
        )
        self.assertFails(
            ValueError,
            "sift",
            lambda: q.BasisKeying(bases=3, sift=0.0),
            msg="a vanishing factor is still refused where one is read",
        )
        self.assertEqual(
            q.BasisKeying(announce="pair").mu,
            q.Decoy().signal,
            msg="the source intensity is still the decoy set's signal",
        )

    def test_two_state_parts(self):
        """
        The two-state modulation and its receiver validate on construction, and the state
        overlap is derived from the signal strength rather than set.
        """
        self.assertFails(
            ValueError,
            "mu",
            lambda: q.TwoStateKeying(mu=0.0),
            msg="two coinciding states carry no bit",
        )
        self.assertFails(
            ValueError,
            "visibility",
            lambda: q.NullingReceiver(visibility=1.5),
            msg="a contrast above one",
        )
        self.assertClose(
            q.TwoStateKeying(mu=0.23).overlap,
            _core.b92_overlap(0.23),
            atol=0.0,
            msg="the overlap is exp(-2 mu), and it is the engine's",
        )
        self.assertFalse(q.TwoStateKeying().reference, msg="the plain branch is the default")

    def test_bound_domains(self):
        """
        The two security components bound their own inputs, and `q.ViolationBound` takes no
        default `source`, the origin of $S$ deciding whether it is a bound.
        """
        self.assertFails(
            ValueError,
            "e_phase",
            lambda: q.DiscriminationBound(e_phase=0.6),
            msg="a phase error past chance",
        )
        self.assertFails(
            ValueError,
            "f",
            lambda: q.DiscriminationBound(f=0.9),
            msg="f below the Shannon limit",
        )
        self.assertFails(
            ValueError,
            "f",
            lambda: q.ViolationBound(s=2.7, source="measured", f=0.9),
            msg="and on the violation too",
        )
        self.assertFails(
            ValueError,
            "sift",
            lambda: q.ViolationBound(s=2.7, source="measured", sift=1.5),
            msg="a sifting factor above one",
        )

        with self.assertRaises(TypeError):
            q.ViolationBound(s=2.7)
        self.assertIsNone(
            q.DiscriminationBound().e_phase,
            msg="an unsupplied bound is derived, not defaulted",
        )

    def test_variant_labels(self):
        """
        Every entry the new reports carry labels its provenance, adding `absent` for a quantity
        nothing certifies and `structural` for one the analysis assumes rather than measures.
        """
        vocab = set(LABELS) | {"absent", "structural", "diagnostic"}
        plans = [
            variant_link(bases=3).explain(),
            variant_link(announce="pair").explain(),
            state_link().explain(),
            state_link(eph=0.05, ref=True).explain(),
            violated_link().explain(),
        ]
        for info in plans:
            for key, entry in info.items():
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                self.assertEqual(set(entry), {"value", "label"}, msg=f"{key} entry shape")
                self.assertIn(label, vocab, msg=f"{key} has label {label!r}")

    def test_variant_purity(self):
        """
        `explain()` equals `run().explain` less `key_raw` for both basis-keyed variants, the
        two-state link and the violation link.
        """
        for link in (variant_link(bases=3), variant_link(announce="pair")):
            info = link.explain()
            used = dict(link.run().explain)
            used.pop("key_raw")
            self.assertEqual(used, info, msg="the basis-keyed plan must match the run")

        for link in (state_link(), violated_link()):
            res = link.run()
            info = dict(res.explain)
            info.pop("key_raw", None)
            plan = dict(link.explain())
            plan.pop("key_raw", None)
            self.assertEqual(plan, info, msg="and so must the closed-form ones")


class Portability(Question):
    """
    Import hazards that bite in a built wheel or on the Python it targets.
    """

    def test_shipped_surface(self):
        """
        Every module in `qkd/` is reachable from the package and every name `__all__` promises
        exists, making `ci/smoke.py`'s sweep a complete gate.
        """
        names = [p.stem for p in SHIPPED]
        gone = [n for n in names if n != "__init__" and not hasattr(q, n)]
        self.assertEqual(gone, [], msg=f"qkd/ modules never imported: {gone}")

        absent = [n for n in q.__all__ if not hasattr(q, n)]
        self.assertEqual(absent, [], msg=f"__all__ promises what is absent: {absent}")

    def test_docstring_links(self):
        """
        No shipped docstring carries a markdown link, a site-relative docs path or a VitePress
        directive, which `help()` prints verbatim.
        """
        found = []
        for path in SHIPPED:
            found += linked(path)
        self.assertEqual(
            found,
            [],
            msg=f"docstrings carrying site markup: {found}",
        )

    def test_annotation_shadowing(self):
        """
        No dataclass field annotation refers to a name an earlier field already bound, which
        raises `AttributeError` on Python 3.11-3.13 and is silently hidden by 3.14's deferred
        annotations.
        """
        found = []
        for path in SHIPPED:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                found += shadowed(node, path.name)
        self.assertEqual(
            found,
            [],
            msg=f"annotation evaluated after its name is shadowed: {found}",
        )


class Composition(Question):
    """
    Which impairment each path consumes, and the pair that describes one thing twice.
    """

    def test_dephasing_once(self):
        """
        Where the DSP produced a residual only `budget.phase(v_err)` is charged: a declared
        `q.Dephasing` changes no rate and reports `v_floor` below the residual.
        """
        plain = metro(q.Asymptotic())
        named = metro(q.Asymptotic(), imp=(q.Dephasing(linewidth=2e4, delay=1e-8),))
        self.assertEqual(named.key_rate, plain.key_rate, msg="the laser is charged once")
        self.assertNotIn("xi_dephasing", named.explain, msg="the second term must not be summed")
        self.assertLess(
            named.explain["v_floor"]["value"],
            named.dsp.v_err,
            msg="the floor is reported, and the residual sits above it",
        )

    def test_dephasing_ordering(self):
        """
        A declared linewidth that diffuses further over its delay than the whole estimator
        residual is a contradiction and raises, naming both variances.
        """
        bad = sim_link(q.Asymptotic(), imp=(q.Dephasing(linewidth=1e8, delay=1e-3),))
        self.assertFails(
            ValueError,
            "already diffused",
            lambda: bad.claim(frames(200_000)),
            msg="a residual below the floor is impossible",
        )

    def test_dephasing_alone(self):
        """
        With no DSP chain `q.Dephasing` is charged as its own row.
        """
        link = q.Link(
            modulation=q.GaussianModulation(v_a=5.0),
            channel=q.Channel(T=0.5, xi=0.02, ref="input"),
            bob=q.Bob(detector=q.Heterodyne()),
            impairments=(q.Dephasing(linewidth=2e4, delay=1e-8),),
        )
        res = link.run()
        self.assertGreater(res.explain["xi_dephasing"]["value"], 0.0, msg="the floor is charged here")
        self.assertNotIn("v_floor", res.explain, msg="nothing to compare it against")

    def test_family_impairments(self):
        """
        A descriptor no path consumes is refused: timing and polarisation on the pinned click
        path, backflash and detector memory on a quadrature one.
        """
        for item in (
            q.Timing(jitter=1e-12, width=5e-9),
            q.Polarisation(drift=0.02),
        ):
            link = dps_link()
            link.impairments = (item,)
            self.assertFails(
                ValueError,
                "no click bound consumes",
                link.run,
                msg=f"{type(item).__name__} is not a click impairment",
            )

        for item in (q.Backflash(prob=0.05), q.DeadTime(dead=0.0, afterpulse=0.01)):
            link = cv_link(q.Channel(T=0.5))
            link.impairments = (item,)
            self.assertFails(
                ValueError,
                "no Gaussian-modulation term consumes",
                link.run,
                msg=f"{type(item).__name__} is not a quadrature impairment",
            )

    def test_backflash_reported(self):
        """
        Backflash leaves the rate unchanged and reports `leak` as $P_b$ times the sifted rate.
        """
        clean = basis_link().run()
        leaky = basis_link()
        leaky.impairments = (q.Backflash(prob=0.0888),)
        res = leaky.run()
        self.assertEqual(res.key_rate, clean.key_rate, msg="a flag corrects no rate")
        self.assertClose(
            res.leak,
            0.0888 * 0.5 * res.p_click,
            msg="leak is P_b times the sifted rate",
        )
        self.assertEqual(res.explain["backflash_leak"]["value"], res.leak, msg="and it is reported")
        self.assertIsNone(clean.leak, msg="no descriptor, no claim about leakage")

    def test_afterpulse_reaches(self):
        """
        Afterpulsing on the decoy family raises the error rate and costs key, and $p_{AP} = 0$
        is the plain model.
        """
        plain = basis_link().run()
        noisy = basis_link()
        noisy.impairments = (q.DeadTime(dead=0.0, afterpulse=0.05),)
        res = noisy.run()
        null = basis_link()
        null.impairments = (q.DeadTime(dead=0.0, afterpulse=0.0),)
        self.assertGreater(res.qber, plain.qber, msg="afterpulses are errors")
        self.assertLess(res.key_rate, plain.key_rate, msg="and they cost key")
        self.assertClose(null.run().key_rate, plain.key_rate, msg="p_AP = 0 is the plain model")

    def test_dead_time_placed(self):
        """
        Dead time is refused on the decoy family, whose interleaved intensities have no single
        rate, and as an impairment on the phase-keyed path, which already models it.
        """
        held = basis_link()
        held.impairments = (q.DeadTime(dead=50e-9, afterpulse=0.01),)
        self.assertFails(
            ValueError,
            "one click stream at one rate",
            held.run,
            msg="interleaved intensities have no single rate",
        )

        twice = click_link()
        twice.impairments = (q.DeadTime(dead=50e-9),)
        self.assertFails(
            ValueError,
            "twice",
            lambda: twice.run(symbols=10_000, seed=1),
            msg="the simulated path already models it",
        )


# The field audit: every settable field is read by some path (two values, two answers) or refused by name.
# A probe compares two values at one point: q.ClickDetector.window needs a jitter and
# q.PnrDetector.saturation a readout width in `base`, or they read dead. A probe proves a value reaches an
# output, not that the term is right. SPARED: no probe can speak to it. OPEN: a probe says it is dropped.
# Impairments are probed per consumer (noise_of, flux_of), but verdicts key on (component, field): split the
# key before filing one consumer's field in OPEN.
FIELD_N = 4000
CLICK_N = 20000
NU = 193.4e12


def gauss(**kw):
    """
    The full-hardware Gaussian link, pilot DSP in the loop.
    """
    parts = dict(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=0.02, ref="input"),
        alice=q.Alice(
            laser=q.Laser(linewidth=1e4),
            pilots=q.Pilots(power_db=12.0),
            symbol_rate=100e6,
        ),
        bob=q.Bob(
            detector=q.Heterodyne(eta=0.6, v_el=0.1),
            lo=q.LocalLO(linewidth=1e4),
        ),
        dsp=q.DSP(),
        security=q.Asymptotic(beta=0.95),
    )
    parts.update(kw)

    return q.Link(**parts).run(FIELD_N, 1)


def closed(**kw):
    """
    The same link with no front end or DSP chain: the closed form.
    """
    parts = dict(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=0.02, ref="input"),
        bob=q.Bob(detector=q.Homodyne(eta=0.6, v_el=0.1)),
        security=q.Asymptotic(beta=0.95),
    )
    parts.update(kw)

    return q.Link(**parts).run()


def lit(**kw):
    """
    The click link beside a classical neighbour, over a fibre with a length and a clock; the only
    path `q.ClickDetector.passband` reaches.
    """
    parts = dict(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Fiber(length=25.0),
        alice=q.Alice(laser=q.Laser(linewidth=1e6), symbol_rate=1e9),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.2, dark=1e-6, passband=100e9),
            receiver=q.DelayInterferometer(delay=1, visibility=0.98),
        ),
        security=q.IndividualAttack(f=1.16),
        impairments=(q.Coexistence(channels=1, launch=-20.0, wavelength=1531.12e-9),),
    )
    parts.update(kw)

    return q.Link(**parts).run(CLICK_N, 1)


def clicks(**kw):
    """
    The phase-keyed link whose error rate comes out of the click simulation.
    """

    return click_link(**kw).run(CLICK_N, 1)


def wcp(**kw):
    """
    Basis-keyed weak coherent pulses over a fibre span.
    """
    parts = dict(
        modulation=q.BasisKeying(decoy=q.Decoy((0.48, 0.1, 0.0))),
        channel=q.Fiber(length=25.0, alpha=0.21),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.045, dark=8.5e-7),
            receiver=q.BasisAnalyser(misalign=0.033),
        ),
        security=q.SplittingAttack(f=1.22),
    )
    parts.update(kw)

    return q.Link(**parts).run(CLICK_N, 1)


def cow(**kw):
    """
    The same intensity-keyed link, run over the click simulation.
    """

    return cow_link(**kw).run(CLICK_N, 1)


def psk(**kw):
    """
    An M-PSK link on the closed-form bound.
    """
    parts = dict(
        modulation=q.PhaseShiftKeying(states=4, alpha=0.4),
        channel=q.Channel(T=0.5, xi=0.01, ref="input"),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=False)),
        security=q.Asymptotic(beta=0.95),
    )
    parts.update(kw)

    return q.Link(**parts).run()


def duo(**kw):
    """
    A photon-pair link, source in the middle of two fibre arms.
    """
    parts = dict(
        source=q.PairSource(brightness=0.01),
        detectors=(q.ClickDetector(eta=0.2, dark=1e-6),) * 2,
        channels=(q.Fiber(length=10.0),) * 2,
        security=q.SymmetryBound(f=1.22, sift=0.5),
    )
    parts.update(kw)

    return q.PairLink(**parts).run()


def qubit(**kw):
    """
    The MDI-BB84 midpoint on its finite-key branch, both senders biased.
    """
    mod = q.BasisKeying(
        decoy=q.Decoy(intensities=(0.3, 0.1, 5e-4), probs=(0.7, 0.2, 0.1)),
        sift=1.0,
        bias=0.5,
    )
    parts = dict(
        alice=q.Sender(modulation=mod),
        bob=q.Sender(modulation=mod),
        relay=q.Relay(
            bell=q.BellAnalyser(
                eta=0.145,
                dark=3.01e-6,
                misalign=0.015,
                misalign_test=0.015,
                states=2,
            )
        ),
        channels=(q.Fiber(length=25.0, alpha=0.2),) * 2,
        security=q.TestBasisBound(f=1.16, block=q.RelayBlock(n=1e12, code=0.9)),
    )
    parts.update(kw)

    return q.Swap(**parts).run()


def midpoint(**kw):
    """
    The continuous-variable relay, one Bell detector between two spans.
    """
    parts = dict(
        alice=q.Sender(modulation=q.GaussianModulation(v_a=5.0)),
        bob=q.Sender(modulation=q.GaussianModulation(v_a=5.0)),
        relay=q.Relay(bell=q.BellDetector(eta=0.98, v_el=0.01)),
        channels=(q.Channel(T=0.9, xi=0.002, ref="input"),) * 2,
    )
    parts.update(kw)

    return q.Swap(**parts).run()


def laser_of(x):
    """
    The link a `q.Laser` is described on: the detector declares a bandwidth exactly when the laser
    declares rin, and Bob's oscillator a carrier exactly when Alice's laser does.
    """
    band = None if x.rin is None else 1e9
    tone = None if x.carrier is None else NU

    return gauss(
        alice=q.Alice(laser=x, pilots=q.Pilots(power_db=12.0), symbol_rate=100e6),
        bob=q.Bob(
            detector=q.Heterodyne(eta=0.6, v_el=0.1, bandwidth=band),
            lo=q.LocalLO(linewidth=1e4, carrier=tone),
        ),
    )


def lo_of(x):
    """
    The same link with Bob's oscillator under test, Alice's laser naming a carrier exactly when
    this one does.
    """
    tone = None if x.carrier is None else NU + 1e6

    return gauss(
        alice=q.Alice(
            laser=q.Laser(linewidth=1e4, carrier=tone),
            pilots=q.Pilots(power_db=12.0),
            symbol_rate=100e6,
        ),
        bob=q.Bob(detector=q.Heterodyne(eta=0.6, v_el=0.1), lo=x),
    )


def two_of(x):
    """
    The two-state link a `q.TwoStateKeying` is described on: the single-photon branch takes no
    dark rate or visibility, and the strong-reference branch a supplied phase error.
    """
    dark = 0.0 if x.source == "single" else 1e-6
    vis = 1.0 if x.source == "single" else 0.98
    eph = 0.02 if x.reference else None

    return q.Link(
        modulation=x,
        channel=q.Channel(T=0.6),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=dark),
            receiver=q.NullingReceiver(visibility=vis),
        ),
        security=q.DiscriminationBound(e_phase=eph, f=1.16),
    ).run(CLICK_N, 1)


def bound_of(x):
    """
    The same link with the discrimination bound under test, the strong reference selected
    exactly when a phase error is supplied to it.
    """
    return q.Link(
        modulation=q.TwoStateKeying(mu=0.23, reference=x.e_phase is not None),
        channel=q.Channel(T=0.6),
        bob=q.Bob(
            detector=q.ClickDetector(eta=1.0, dark=1e-6),
            receiver=q.NullingReceiver(visibility=0.98),
        ),
        security=x,
    ).run(CLICK_N, 1)


def attack_of(x):
    """
    The phase-keyed link an individual-attack bound is described on: a pinned error rate takes no
    interferometer or laser, an unpinned one needs both.
    """
    if x.qber is None:
        return clicks(security=x)

    return q.Link(
        modulation=q.DifferentialPhase(mu=0.2),
        channel=q.Channel(T=0.1),
        bob=q.Bob(detector=q.ClickDetector(eta=0.2, dark=1e-6)),
        security=x,
    ).run(CLICK_N, 1)


def relay_of(x):
    """
    The midpoint a `q.TestBasisBound` is described on: bias is set exactly when a block is, the
    asymptotic branch refusing one.
    """
    bias = None if x.block is None else 0.5
    mod = q.BasisKeying(
        decoy=q.Decoy(intensities=(0.3, 0.1, 5e-4), probs=(0.7, 0.2, 0.1)),
        sift=1.0,
        bias=bias,
    )

    return qubit(alice=q.Sender(modulation=mod), bob=q.Sender(modulation=mod), security=x)


def noise_of(x):
    """
    The Gaussian link one excess-noise impairment descriptor is charged on.
    """
    return closed(
        channel=q.Fiber(length=25.0, alpha=0.2),
        alice=q.Alice(symbol_rate=1e9),
        impairments=(x,),
    )


def flux_of(x):
    """
    The simulated click link one impairment is charged on, a different consumer from `noise_of`:
    a flux joins the per-gate floor, and launch powers stay low enough that both ends of every
    probe distil rather than refuse on saturation.
    """

    return lit(impairments=(x,))


def array_of(x):
    """
    A multiplexed threshold array's truncated POVM beside its untruncated closed form.
    """
    return (x.povm().coherent(2.0), x.outcomes(2.0), x.distinct(3))


def atten_of(x):
    """
    The detector-decoy round trip: the no-click probabilities these settings produce, and the
    photon-number interval they are inverted back to.
    """
    det = q.ClickDetector(eta=1.0, dark=0.0)
    vacs = x.noclick(det, (0.4, 0.3, 0.2, 0.1))

    return (vacs, x.bounds(det, vacs))


def tlo_of(x):
    """
    Every quantity a transmitted local oscillator reports at one operating point, none of which
    `q.Link` reaches.
    """
    return (
        x.shot(0.5, 0.1),
        x.phase(1e4),
        x.leak(0.5),
        x.estimate(0.5, 0.01, 0.6, 1.0),
        x.covariance(5.0, 0.5, 0.01, 0.6, 1.0),
        x.monitor(10_000, 0.5, 0.6, 0.1),
    )


def flawed(**kw):
    """
    The flawed qubit-source link, the analysis named by the security component.
    """
    parts = dict(
        modulation=q.FlawedKeying(flaw=q.SourceFlaw.opposed(0.063), sift=0.25),
        channel=q.Channel(T=0.5),
        bob=q.Bob(detector=q.ClickDetector(eta=0.9, dark=1e-7)),
        security=q.FlawBound(analysis="tolerant", f=1.16),
    )
    parts.update(kw)

    return q.Link(**parts).run()


def flaw_of(x):
    """
    Both analyses of a flawed qubit source at one operating point.
    """
    return (
        x.angles(),
        x.channel(0.3, 1e-6),
        x.phase(0.3, 1e-6),
        x.coin(0.3),
        x.fidelity(),
        x.triangle(),
        x.tolerant(0.3, 1e-6, 1.16, 0.5),
        x.standard(0.3, 1e-6, 1.16, 0.5),
    )


FIELDS = (
    (
        q.GaussianModulation,
        dict(v_a=5.0),
        (("v_a", 5.0, 9.0),),
        lambda x: closed(modulation=x),
    ),
    (
        q.Channel,
        dict(T=0.5, xi=0.02, ref="input"),
        (
            ("T", 0.5, 0.3),
            ("xi", 0.02, 0.05),
            ("ref", "input", "output"),
        ),
        lambda x: closed(channel=x),
    ),
    (
        q.Fiber,
        dict(length=25.0),
        (
            ("length", 25.0, 40.0),
            ("alpha", 0.2, 0.35),
        ),
        lambda x: closed(channel=x),
    ),
    (
        q.Fiber,
        dict(T=0.5),
        (("T", 0.5, 0.4),),
        lambda x: closed(channel=x),
    ),
    (
        q.Connector,
        dict(loss=0.25),
        (
            ("loss", 0.25, 0.75),
            ("count", 1, 4),
            ("site", "launch", "receive"),
        ),
        lambda x: closed(channel=q.Fiber(length=25.0, alpha=0.2), losses=(x,)),
    ),
    (
        q.Splice,
        dict(loss=0.02),
        (
            ("loss", 0.02, 0.2),
            ("count", 1, 6),
            ("site", "span", "receive"),
        ),
        lambda x: closed(channel=q.Fiber(length=25.0, alpha=0.2), losses=(x,)),
    ),
    (
        q.Coupling,
        dict(loss=0.5),
        (
            ("loss", 0.5, 1.5),
            ("count", 1, 3),
            ("site", "launch", "receive"),
        ),
        lambda x: closed(channel=q.Fiber(length=25.0, alpha=0.2), losses=(x,)),
    ),
    (
        q.Homodyne,
        dict(eta=0.6, v_el=0.1),
        (
            ("eta", 0.6, 0.9),
            ("v_el", 0.1, 0.4),
            ("trusted", True, False),
            ("clip", None, 3.0),
            ("bandwidth", None, 2e8),
        ),
        lambda x: closed(bob=q.Bob(detector=x)),
    ),
    (
        q.Heterodyne,
        dict(eta=0.6, v_el=0.1),
        (
            ("eta", 0.6, 0.9),
            ("v_el", 0.1, 0.4),
            ("trusted", True, False),
            ("clip", None, 3.0),
            ("bandwidth", None, 2e8),
        ),
        lambda x: closed(bob=q.Bob(detector=x)),
    ),
    (
        q.Laser,
        dict(linewidth=1e4),
        (
            ("linewidth", 1e4, 1e5),
            ("rin", -155.0, -100.0),
            ("carrier", NU + 1e6, NU + 3e6),
        ),
        laser_of,
    ),
    (
        q.LocalLO,
        dict(linewidth=1e4),
        (
            ("linewidth", 1e4, 1e5),
            ("carrier", NU, NU - 3e6),
        ),
        lo_of,
    ),
    (
        q.Pilots,
        dict(power_db=12.0),
        (
            ("power_db", 12.0, 3.0),
            ("tones", 1, 2),
            ("freq", 180e6, 90e6),
        ),
        lambda x: gauss(alice=q.Alice(laser=q.Laser(linewidth=1e4), pilots=x, symbol_rate=100e6)),
    ),
    (
        q.IQModulator,
        dict(bits=16),
        (("bits", 16, 6),),
        lambda x: gauss(
            alice=q.Alice(
                laser=q.Laser(linewidth=1e4),
                pilots=q.Pilots(power_db=12.0),
                iq=x,
                symbol_rate=100e6,
            )
        ),
    ),
    (
        q.ADC,
        dict(bits=12),
        (("bits", 12, 6),),
        lambda x: gauss(
            bob=q.Bob(
                detector=q.Heterodyne(eta=0.6, v_el=0.1),
                lo=q.LocalLO(linewidth=1e4),
                adc=x,
            )
        ),
    ),
    (
        q.PilotPhase,
        dict(),
        (("v_err", None, 0.05),),
        lambda x: gauss(dsp=q.DSP(phase=x)),
    ),
    (
        q.DSP,
        dict(),
        (
            ("block", 32, 128),
            ("phase", q.PilotPhase(), q.PilotPhase(v_err=0.05)),
        ),
        lambda x: gauss(dsp=x),
    ),
    (
        q.Alice,
        dict(
            laser=q.Laser(linewidth=1e4),
            pilots=q.Pilots(power_db=12.0),
            symbol_rate=100e6,
        ),
        (
            ("laser", q.Laser(linewidth=1e4), q.Laser(linewidth=1e6)),
            ("iq", None, q.IQModulator(bits=6)),
            ("pilots", q.Pilots(power_db=12.0), q.Pilots(power_db=3.0)),
            ("symbol_rate", 100e6, 25e6),
        ),
        lambda x: gauss(alice=x),
    ),
    (
        q.Bob,
        dict(detector=q.Heterodyne(eta=0.6, v_el=0.1)),
        (
            (
                "detector",
                q.Heterodyne(eta=0.6, v_el=0.1),
                q.Homodyne(eta=0.6, v_el=0.1),
            ),
        ),
        lambda x: closed(bob=x),
    ),
    (
        q.Bob,
        dict(detector=q.Heterodyne(eta=0.6, v_el=0.1), lo=q.LocalLO(linewidth=1e4)),
        (
            ("lo", q.LocalLO(linewidth=1e4), q.LocalLO(linewidth=1e6)),
            ("adc", None, q.ADC(bits=6)),
        ),
        lambda x: gauss(bob=x),
    ),
    (
        q.Bob,
        dict(detector=q.ClickDetector(eta=0.2, dark=1e-6)),
        (
            (
                "receiver",
                q.DelayInterferometer(delay=1, visibility=0.98),
                q.DelayInterferometer(delay=2, visibility=0.98),
            ),
        ),
        lambda x: clicks(bob=x),
    ),
    (
        q.Asymptotic,
        dict(),
        (("beta", 0.95, 0.8),),
        lambda x: closed(security=x),
    ),
    (
        q.FiniteSize,
        dict(),
        (
            ("beta", 0.95, 0.8),
            ("eps", 1e-10, 1e-6),
            ("n", 1e9, 1e6),
            ("pe_fraction", 0.5, 0.2),
            ("fer", None, 0.1),
        ),
        lambda x: closed(security=x),
    ),
    (
        q.PhaseShiftKeying,
        dict(states=4, alpha=0.4),
        (
            ("states", 4, 8),
            ("alpha", 0.4, 0.7),
        ),
        lambda x: psk(modulation=x),
    ),
    (
        q.CertifiedBound,
        dict(cutoff=4, steps=2),
        (
            ("cutoff", 4, 5),
            ("cut", 0.0, 0.1),
            ("phase", 0.0, 0.05),
            ("beta", 0.95, 0.8),
            ("eps", 1e-10, 1e-13),
            ("steps", 2, 4),
            ("gaptol", 1e-8, 1e-1),
        ),
        lambda x: psk(
            modulation=q.PhaseShiftKeying(states=4, alpha=0.7),
            bob=q.Bob(detector=q.Heterodyne(eta=1.0, v_el=0.0, trusted=False)),
            security=x,
        ),
    ),
    (
        q.DifferentialPhase,
        dict(mu=0.2),
        (("mu", 0.2, 0.4),),
        lambda x: clicks(modulation=x),
    ),
    (
        q.ClickDetector,
        dict(
            eta=0.2,
            dark=1e-6,
            jitter=30e-12,
            window=400e-12,
            tail_frac=0.05,
            tail_time=1e-9,
        ),
        (
            ("eta", 0.2, 0.5),
            ("dark", 1e-6, 1e-4),
            ("dead_time", 0.0, 50e-9),
            ("afterpulse", 0.0, 0.05),
            ("jitter", 30e-12, 90e-12),
            ("window", 400e-12, 100e-12),
            ("tail_frac", 0.05, 0.2),
            ("tail_time", 1e-9, 5e-9),
            (
                "partner",
                None,
                q.ClickDetector(
                    eta=0.9,
                    dark=1e-6,
                    jitter=30e-12,
                    window=400e-12,
                    tail_frac=0.05,
                    tail_time=1e-9,
                ),
            ),
        ),
        lambda x: clicks(bob=q.Bob(detector=x, receiver=q.DelayInterferometer(delay=1, visibility=0.98))),
    ),
    (
        q.ClickDetector,
        dict(eta=0.2, dark=1e-6, passband=100e9),
        (("passband", 100e9, 5.62e9),),
        lambda x: lit(bob=q.Bob(detector=x, receiver=q.DelayInterferometer(delay=1, visibility=0.98))),
    ),
    (
        q.DelayInterferometer,
        dict(delay=1, visibility=0.98),
        (
            ("delay", 1, 2),
            ("visibility", 0.98, 0.9),
        ),
        lambda x: clicks(bob=q.Bob(detector=q.ClickDetector(eta=0.2, dark=1e-6), receiver=x)),
    ),
    (
        q.IndividualAttack,
        dict(f=1.16),
        (
            ("f", 1.16, 1.3),
            ("qber", dict(qber=0.01), dict(qber=0.02)),
        ),
        attack_of,
    ),
    (
        q.BasisKeying,
        dict(decoy=q.Decoy((0.48, 0.1, 0.0))),
        (
            ("decoy", q.Decoy((0.48, 0.1, 0.0)), q.Decoy((0.6, 0.2, 0.0))),
            ("sift", None, 0.9),
            ("bases", 2, 3),
            ("announce", "basis", dict(announce="pair", sift=None)),
            ("bias", None, 0.8),
        ),
        lambda x: wcp(modulation=x),
    ),
    (
        q.Decoy,
        dict(intensities=(0.48, 0.1, 0.0)),
        (
            ("intensities", (0.48, 0.1, 0.0), (0.6, 0.2, 0.0)),
            ("probs", None, (0.5, 0.3, 0.2)),
        ),
        lambda x: wcp(
            modulation=q.BasisKeying(decoy=x),
            security=q.SplittingAttack(f=1.22, block=q.KeyBlock(n=1e10)),
        ),
    ),
    (
        q.BasisAnalyser,
        dict(misalign=0.033),
        (("misalign", 0.033, 0.08),),
        lambda x: wcp(bob=q.Bob(detector=q.ClickDetector(eta=0.045, dark=8.5e-7), receiver=x)),
    ),
    (
        q.SplittingAttack,
        dict(f=1.22),
        (
            ("f", 1.22, 1.4),
            ("block", None, q.KeyBlock(n=1e10)),
        ),
        lambda x: wcp(security=x),
    ),
    (
        q.KeyBlock,
        dict(),
        (
            ("n", 1e10, 1e8),
            ("eps_sec", 1e-10, 1e-6),
            ("eps_cor", 1e-15, 1e-9),
            ("fer", None, 0.1),
        ),
        lambda x: wcp(security=q.SplittingAttack(f=1.22, block=x)),
    ),
    (
        q.PolarisationKeying,
        dict(decoy=q.Decoy((0.48, 0.1, 0.0)), frame=q.ReferenceFrame(drift=0.05)),
        (
            ("decoy", q.Decoy((0.48, 0.1, 0.0)), q.Decoy((0.6, 0.2, 0.0))),
            ("sift", 0.5, 0.9),
            (
                "frame",
                q.ReferenceFrame(drift=0.05),
                q.ReferenceFrame(drift=0.15),
            ),
        ),
        lambda x: wcp(modulation=x),
    ),
    (
        q.ReferenceFrame,
        dict(drift=0.05, dispersion=1e-13, width=50e-12),
        (
            ("drift", 0.05, 0.15),
            ("dispersion", 1e-13, 1e-12),
            ("width", 50e-12, 10e-12),
            (
                "tracking",
                dict(tracking="tracked"),
                dict(tracking="free", drift=0.0, rate=0.02, interval=1.0),
            ),
            (
                "rate",
                dict(tracking="free", drift=0.0, rate=0.02, interval=1.0),
                dict(tracking="free", drift=0.0, rate=0.2, interval=1.0),
            ),
            (
                "interval",
                dict(tracking="free", drift=0.0, rate=0.02, interval=1.0),
                dict(tracking="free", drift=0.0, rate=0.02, interval=10.0),
            ),
        ),
        lambda x: wcp(modulation=q.PolarisationKeying(decoy=q.Decoy((0.48, 0.1, 0.0)), frame=x)),
    ),
    (
        q.IntensityKeying,
        dict(mu=0.5, decoy_frac=0.1),
        (
            ("mu", 0.5, 0.8),
            ("decoy_frac", 0.1, 0.3),
            ("extinction", None, 20.0),
        ),
        lambda x: cow(modulation=x),
    ),
    (
        q.CoherenceMonitor,
        dict(split=0.1, misalign=0.02),
        (
            ("split", 0.1, 0.3),
            ("misalign", 0.02, 0.08),
        ),
        lambda x: cow(bob=q.Bob(detector=q.ClickDetector(eta=0.8, dark=1e-6), receiver=x)),
    ),
    (
        q.PhaseBound,
        dict(e_phase=0.2, f=1.1),
        (
            ("e_phase", 0.2, 0.3),
            ("f", 1.1, 1.3),
            ("block", None, q.KeyBlock(n=1e10)),
        ),
        lambda x: cow(security=x),
    ),
    (
        q.TwoStateKeying,
        dict(mu=0.23),
        (
            ("mu", 0.23, 0.4),
            ("reference", False, True),
            (
                "source",
                dict(source="coherent"),
                dict(mu=0.19, source="single", depol=0.005),
            ),
            (
                "depol",
                dict(mu=0.19, source="single", depol=0.005),
                dict(mu=0.19, source="single", depol=0.02),
            ),
        ),
        two_of,
    ),
    (
        q.NullingReceiver,
        dict(visibility=0.98),
        (("visibility", 0.98, 0.9),),
        lambda x: q.Link(
            modulation=q.TwoStateKeying(mu=0.23),
            channel=q.Channel(T=0.6),
            bob=q.Bob(detector=q.ClickDetector(eta=1.0, dark=1e-6), receiver=x),
            security=q.DiscriminationBound(f=1.16),
        ).run(CLICK_N, 1),
    ),
    (
        q.DiscriminationBound,
        dict(f=1.16),
        (
            ("f", 1.16, 1.3),
            ("e_phase", 0.02, 0.05),
            ("block", None, q.KeyBlock(n=1e9)),
        ),
        bound_of,
    ),
    (
        q.PairSource,
        dict(brightness=0.01),
        (
            ("brightness", 0.01, 0.05),
            ("rate", 100e6, 1e9),
            ("pumping", "pulsed", "cw"),
        ),
        lambda x: duo(source=x),
    ),
    (
        q.SymmetryBound,
        dict(),
        (
            ("e_phase", None, 0.05),
            ("f", 1.22, 1.4),
            ("sift", 0.5, 0.4),
        ),
        lambda x: duo(security=x),
    ),
    (
        q.ViolationBound,
        dict(s=2.7, source="measured"),
        (
            ("s", 2.7, 2.6),
            ("source", "measured", "derived"),
            ("f", 1.22, 1.4),
            ("sift", 2.0 / 9.0, 0.4),
        ),
        lambda x: duo(security=x),
    ),
    (
        q.BellAnalyser,
        dict(eta=0.145, dark=3.01e-6, misalign=0.015, misalign_test=0.015, states=2),
        (
            ("eta", 0.145, 0.4),
            ("dark", 3.01e-6, 1e-5),
            ("misalign", 0.015, 0.05),
            ("misalign_test", 0.015, 0.05),
            ("states", 2, 1),
        ),
        lambda x: qubit(relay=q.Relay(bell=x)),
    ),
    (
        q.TestBasisBound,
        dict(f=1.16, block=q.RelayBlock(n=1e12, code=0.9)),
        (
            ("f", 1.16, 1.3),
            ("block", q.RelayBlock(n=1e12, code=0.9), None),
        ),
        relay_of,
    ),
    (
        q.RelayBlock,
        dict(),
        (
            ("n", 1e12, 1e11),
            ("eps_sec", 1e-10, 1e-6),
            ("eps_cor", 1e-15, 1e-9),
            ("code", 0.9, 0.5),
            ("fer", None, 0.1),
        ),
        lambda x: qubit(security=q.TestBasisBound(f=1.16, block=x)),
    ),
    (
        q.TwoModeBound,
        dict(block=q.GaussianBlock(n=1e9, sigma=6.5)),
        (
            ("beta", 0.95, 0.8),
            ("block", q.GaussianBlock(n=1e9, sigma=6.5), None),
        ),
        lambda x: midpoint(security=x),
    ),
    (
        q.GaussianBlock,
        dict(sigma=6.5),
        (
            ("n", 1e9, 1e7),
            ("pe_fraction", 0.5, 0.2),
            ("sigma", 6.5, 3.0),
            ("eps_smooth", 1e-10, 1e-6),
            ("eps_pa", 1e-10, 1e-6),
            ("attack", "gaussian", "collective"),
        ),
        lambda x: midpoint(security=q.TwoModeBound(block=x)),
    ),
    (
        q.BellDetector,
        dict(eta=0.98, v_el=0.01),
        (
            ("eta", 0.98, 0.5),
            ("v_el", 0.01, 0.2),
        ),
        lambda x: midpoint(relay=q.Relay(bell=x)),
    ),
    (
        q.Relay,
        dict(bell=q.BellDetector(eta=0.98, v_el=0.01)),
        (
            (
                "bell",
                q.BellDetector(eta=0.98, v_el=0.01),
                q.BellDetector(eta=0.5, v_el=0.01),
            ),
        ),
        lambda x: midpoint(relay=x),
    ),
    (
        q.CorrelatedEnvironment,
        dict(x=0.001, p=0.0),
        (
            ("x", 0.001, 0.02),
            ("p", 0.0, -0.02),
        ),
        lambda x: midpoint(environment=x),
    ),
    (
        q.Sender,
        dict(modulation=q.GaussianModulation(v_a=5.0)),
        (
            (
                "modulation",
                q.GaussianModulation(v_a=5.0),
                q.GaussianModulation(v_a=9.0),
            ),
            ("pulse", None, 1e-9),
        ),
        lambda x: midpoint(alice=x),
    ),
    (
        q.ThresholdArray,
        dict(),
        (
            ("elements", 8, 4),
            ("eta", 0.2, 0.5),
            ("dark", 1e-6, 1e-3),
            ("cutoff", 40, 20),
        ),
        array_of,
    ),
    (
        q.PnrDetector,
        dict(sigma=0.3),
        (
            ("eta", 0.9, 0.5),
            ("background", 0.0, 0.05),
            ("sigma", 0.3, 0.6),
            ("saturation", 0.0, 5.0),
            ("cutoff", 40, 20),
        ),
        lambda x: x.povm().coherent(3.0),
    ),
    (
        q.DecoyAttenuator,
        dict(),
        (
            ("settings", (1.0, 0.99, 0.9), (1.0, 0.95, 0.5)),
            ("cap", 1.0, 0.8),
        ),
        atten_of,
    ),
    (
        q.TransmittedLO,
        dict(),
        (
            ("photons", 1e9, 1e8),
            ("mux", 1.0, 0.5),
            ("calibrated", None, 0.9),
            ("delay", 0.0, 1e-9),
            ("extinction", None, 40.0),
        ),
        tlo_of,
    ),
    (
        q.SourceFlaw,
        dict(delta=0.1, tilt=-0.1),
        (
            ("delta", 0.1, 0.3),
            ("tilt", -0.1, -0.3),
        ),
        flaw_of,
    ),
    (
        q.FlawedKeying,
        dict(flaw=q.SourceFlaw.opposed(0.063)),
        (
            ("flaw", q.SourceFlaw.opposed(0.063), q.SourceFlaw.opposed(0.2)),
            ("sift", 0.25, 0.6),
        ),
        lambda x: flawed(modulation=x),
    ),
    (
        q.FlawBound,
        dict(analysis="tolerant"),
        (
            ("analysis", "tolerant", "standard"),
            ("f", 1.16, 1.3),
            ("block", None, q.KeyBlock(n=1e10)),
        ),
        lambda x: flawed(security=x),
    ),
    (
        q.Coexistence,
        dict(channels=4, launch=0.0),
        (
            ("channels", 4, 8),
            ("launch", 0.0, 6.0),
            ("beta", 3e-9, 6e-9),
            ("wavelength", 1531.12e-9, 1310e-9),
            ("demux", 1.0, 0.1),
            ("backward", False, True),
        ),
        noise_of,
    ),
    (
        q.Backscatter,
        dict(power=1e-3),
        (
            ("power", 1e-3, 5e-3),
            ("coeff", 8.0, 12.0),
            ("index", 1.468, 1.5),
            ("wavelength", 1550.12e-9, 1310e-9),
        ),
        noise_of,
    ),
    (
        q.Dephasing,
        dict(linewidth=1e4, delay=1e-6),
        (
            ("linewidth", 1e4, 1e5),
            ("delay", 1e-6, 1e-5),
            ("form", "estimator", "literature"),
        ),
        noise_of,
    ),
    (
        q.Polarisation,
        dict(drift=0.01),
        (
            ("drift", 0.01, 0.05),
            ("dispersion", 0.0, 1e-12),
        ),
        noise_of,
    ),
    (
        q.Modulator,
        dict(ratio=0.99, angle=0.01),
        (
            ("ratio", 0.99, 0.9),
            ("angle", 0.01, 0.05),
            ("extinction", None, 100.0),
        ),
        noise_of,
    ),
    (
        q.Timing,
        dict(jitter=5e-12, width=50e-12),
        (
            ("jitter", 5e-12, 20e-12),
            ("width", 50e-12, 20e-12),
        ),
        noise_of,
    ),
    (
        q.Backflash,
        dict(prob=0.01),
        (("prob", 0.01, 0.05),),
        lambda x: cow(impairments=(x,)),
    ),
    (
        q.DeadTime,
        dict(dead=0.0, afterpulse=0.02),
        (
            ("dead", 0.0, 20e-9),
            ("afterpulse", 0.02, 0.1),
            ("paralysable", False, True),
        ),
        lambda x: cow(impairments=(x,)),
    ),
    (
        q.Coexistence,
        dict(channels=4, launch=-26.0),
        (
            ("channels", 4, 8),
            ("launch", -26.0, -20.0),
            ("beta", 3e-9, 6e-9),
            ("wavelength", 1531.12e-9, 1310e-9),
            ("demux", 1.0, 0.1),
            ("backward", False, True),
        ),
        flux_of,
    ),
    (
        q.Backscatter,
        dict(power=3e-10),
        (
            ("power", 3e-10, 1.5e-9),
            ("coeff", 8.0, 12.0),
            ("index", 1.468, 1.5),
            ("wavelength", 1550.12e-9, 1310e-9),
        ),
        flux_of,
    ),
    (
        q.Dephasing,
        dict(linewidth=1e4, delay=1e-6),
        (
            ("linewidth", 1e4, 1e5),
            ("delay", 1e-6, 1e-5),
            ("form", "estimator", "literature"),
        ),
        flux_of,
    ),
    (
        q.Polarisation,
        dict(drift=0.01),
        (
            ("drift", 0.01, 0.05),
            ("dispersion", 0.0, 1e-12),
        ),
        flux_of,
    ),
    (
        q.Modulator,
        dict(ratio=0.99, angle=0.01),
        (
            ("ratio", 0.99, 0.9),
            ("angle", 0.01, 0.05),
            ("extinction", None, 100.0),
        ),
        flux_of,
    ),
    (
        q.Timing,
        dict(jitter=5e-12, width=50e-12),
        (
            ("jitter", 5e-12, 20e-12),
            ("width", 50e-12, 20e-12),
        ),
        flux_of,
    ),
    (
        q.Backflash,
        dict(prob=0.01),
        (("prob", 0.01, 0.05),),
        flux_of,
    ),
    (
        q.DeadTime,
        dict(dead=0.0, afterpulse=0.02),
        (
            ("dead", 0.0, 20e-9),
            ("afterpulse", 0.02, 0.1),
            ("paralysable", False, True),
        ),
        flux_of,
    ),
)


ALIASES = {
    ("Pilots", "tones"): ("tone",),
    ("DiscriminationBound", "block"): ("finite-key",),
    ("PhaseBound", "block"): ("finite-key",),
}

SPARED = {
    ("PhotonBounds", "p0"): "engine return value",
    ("PhotonBounds", "p1_lo"): "engine return value",
    ("PhotonBounds", "p1_hi"): "engine return value",
    ("PhotonBounds", "p2_lo"): "engine return value",
    ("PhotonBounds", "p2_hi"): "engine return value",
    ("ShotNoise", "arriving"): "engine return value",
    ("ShotNoise", "calibrated"): "engine return value",
    ("ShotNoise", "unit"): "engine return value",
    ("ShotNoise", "ratio"): "engine return value",
    ("ShotNoise", "transmittance"): "engine return value",
    ("ShotNoise", "v_el"): "engine return value",
    ("EntropyBound", "entropy"): "engine return value",
    ("EntropyBound", "bound"): "engine return value",
    ("EntropyBound", "upper"): "engine return value",
    ("EntropyBound", "p_pass"): "engine return value",
    ("EntropyBound", "delta_ec"): "engine return value",
    ("EntropyBound", "viol"): "engine return value",
    ("EntropyBound", "zeta"): "engine return value",
    ("EntropyBound", "steps"): "engine return value",
    ("EntropyBound", "status"): "engine return value",
    ("Certificate", "key"): "engine return value",
    ("Certificate", "bound"): "engine return value",
    ("Certificate", "upper"): "engine return value",
    ("Certificate", "p_pass"): "engine return value",
    ("Certificate", "delta_ec"): "engine return value",
    ("Certificate", "viol"): "engine return value",
    ("Certificate", "zeta"): "engine return value",
    ("Certificate", "steps"): "engine return value",
    ("Certificate", "status"): "engine return value",
}

OPEN = set()


def merged(base, name, value):
    """
    One probe's keywords: a scalar replaces `name`, a dict merges whole to carry a partner field.
    """
    out = dict(base)
    if isinstance(value, dict):
        out.update(value)
    else:
        out[name] = value

    return out


def trial(run, cls, kw):
    """
    One configuration's report as a comparable string, a raise kept as an observation.
    """
    try:
        return repr(run(cls(**kw)))
    except Exception as exc:
        return f"raise {type(exc).__name__}: {exc}"


def graded(cls, name, low, high):
    """
    read, refused, dead or unnamed for one pair of signatures.
    """
    if low == high:
        if low.startswith("raise") and cls.lower() in low.lower():
            return "refused"

        return "dead"

    said = [t for t in (low, high) if t.startswith("raise")]
    if not said:
        return "read"

    words = (name,) + ALIASES.get((cls, name), ())
    named = [t for t in said if any(w in t.lower() for w in words)]

    return "refused" if named else "unnamed"


def sweep():
    """
    Every probe in `FIELDS` as (component, field, verdict).
    """
    out = []
    for cls, base, rows, run in FIELDS:
        for name, low, high in rows:
            one = trial(run, cls, merged(base, name, low))
            two = trial(run, cls, merged(base, name, high))
            out.append((cls.__name__, name, graded(cls.__name__, name, one, two)))

    return out


def settable():
    """
    Every (component, field) on a dataclass in `components.py` or `impairments.py`.
    """
    out = []
    for mod in (q.components, q.impairments):
        for name, obj in vars(mod).items():
            if not (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
                continue
            if obj.__module__ != mod.__name__:
                continue

            out += [(name, f.name) for f in dataclasses.fields(obj)]

    return out


@memo
def audited():
    """
    The whole sweep, computed once: about 450 short runs.
    """

    return tuple(sweep())


class Fields(Question):
    """
    Every settable component field, held against the paths that read it.
    """

    def test_field_coverage(self):
        """
        Every field of every dataclass in `components.py` and `impairments.py` is probed, spared
        with a reason, or filed open.
        """
        rows = audited()
        known = {(c, n) for c, n, _ in rows} | set(SPARED) | OPEN
        gone = sorted(f"{c}.{n}" for c, n in settable() if (c, n) not in known)
        self.assertEqual(gone, [], msg=f"fields nothing here classifies: {gone}")

    def test_impairment_paths(self):
        """
        Every impairment is probed on the simulated click path as well as on the closed form.
        """
        kinds = {
            name
            for name, obj in vars(q.impairments).items()
            if inspect.isclass(obj) and dataclasses.is_dataclass(obj) and obj.__module__ == q.impairments.__name__
        }
        onclick = {cls.__name__ for cls, _, _, run in FIELDS if run is flux_of}
        gone = sorted(kinds - onclick)
        self.assertEqual(gone, [], msg=f"no click-branch probe: {gone}")

        runs = [cls.__name__ for cls, _, _, _ in FIELDS]
        lone = sorted(k for k in kinds if runs.count(k) < 2)
        self.assertEqual(lone, [], msg=f"probed on one consumer only: {lone}")

    def test_fields_charged(self):
        """
        Every probed field outside the filed ones changes what its path reports, or is refused
        by a message naming it or its component.
        """
        rows = [(c, n, v) for c, n, v in audited() if (c, n) not in OPEN]
        bad = sorted(f"{c}.{n} is {v}" for c, n, v in rows if v not in ("read", "refused"))
        self.assertEqual(bad, [], msg=f"accepted and dropped, file it: {bad}")

    def test_open_fields(self):
        """
        Every field filed in `OPEN` is still dropped.
        """
        live = {(c, n): v for c, n, v in audited()}
        fixed = sorted(f"{c}.{n}" for c, n in OPEN if live.get((c, n)) != "dead")
        self.assertEqual(fixed, [], msg=f"no longer dropped, retire from OPEN: {fixed}")

    def test_spared_fields(self):
        """
        Every spared field still exists, carries the reason no probe here can speak to it, and
        is not also probed or filed open.
        """
        held = set(settable())
        stale = sorted(f"{c}.{n}" for c, n in set(SPARED) | OPEN if (c, n) not in held)
        self.assertEqual(stale, [], msg=f"spared or filed, but no such field: {stale}")

        mute = sorted(k for k, v in SPARED.items() if not v.strip())
        self.assertEqual(mute, [], msg=f"spared with no reason given: {mute}")

        both = sorted(set(SPARED) & OPEN)
        self.assertEqual(both, [], msg=f"spared and filed open at once: {both}")

    def test_probe_aliases(self):
        """
        Every refusal alias names an existing field that some probe refuses.
        """
        held = set(settable())
        stale = sorted(f"{c}.{n}" for c, n in ALIASES if (c, n) not in held)
        self.assertEqual(stale, [], msg=f"alias for no such field: {stale}")

        used = {(c, n) for c, n, v in audited() if v == "refused"}
        idle = sorted(f"{c}.{n}" for c, n in ALIASES if (c, n) not in used)
        self.assertEqual(idle, [], msg=f"alias on a field nothing refuses: {idle}")


if __name__ == "__main__":
    rc = Exam(
        "ApiComponents",
        "Component validation paths and frozen-dataclass immutability",
        "api_contract.md",
    ).run(load(Components))
    rc |= Exam(
        "ApiExplain",
        "explain(): labelled provenance, purity, and agreement with run()",
        "api_explain.md",
    ).run(load(Explain))
    rc |= Exam(
        "ApiPinning",
        "Noise-plane conversion, fibre derivation and bit-for-bit determinism",
        "api_pinning.md",
    ).run(load(Pinning))
    rc |= Exam(
        "ApiScope",
        "Phase gating: out-of-scope configurations raise instead of guessing",
        "api_scope.md",
    ).run(load(Scope))
    rc |= Exam(
        "ApiEstimation",
        "The sample count the confidence interval is claimed at, and its cap",
        "api_estimation.md",
    ).run(load(Estimation))
    rc |= Exam(
        "ApiReconciliation",
        "Frame error rate: the share of frames that converge, and the rate it leaves",
        "api_reconciliation.md",
    ).run(load(Reconciliation))
    rc |= Exam(
        "ApiVariants",
        "The components of the four protocol variants, and the reports they label",
        "api_variants.md",
    ).run(load(Variants))
    rc |= Exam(
        "ApiPortability",
        "Import hazards that bite in a built wheel or on the Python it targets",
        "api_portability.md",
    ).run(load(Portability))
    rc |= Exam(
        "ApiComposition",
        "Which path consumes which impairment, and the pair that describes one thing",
        "api_composition.md",
    ).run(load(Composition))
    rc |= Exam(
        "ApiFields",
        "Every settable component field: read on some path, or refused by name",
        "api_fields.md",
    ).run(load(Fields))
    sys.exit(rc)
