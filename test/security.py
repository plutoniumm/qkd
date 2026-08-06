import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.anchors import DMCS_EPS
from qkd import _core, security

# The security target every reading below is taken at.
TARGET = 1e-10

# src/rrdps.rs::rrdps_split: eps_1 = d, eta_z = d/2, eps_2 = eps_3 = eta_x = d^2/24.
SMALL = TARGET * TARGET / 24.0

# d/5 in all five slots: FLAT as they stand, s_x and s_z being HASH LENGTHS in bits, PROBS
# with those two converted.
FLAT = 2.414213562377714
PROBS = 1.0954471150103322e-05
RATIO = 1.0954471150103322e5

# Summing Takesue's five terms instead of taking his maximum, hash lengths converted and not.
SUM_RATIO = 1.5
RAW_SUM = 100.0

# Grouped by the arm Ledger.secrecy takes.
FORMS = {
    "sum": ["bb84", "cv", "cvmdi", "pair", "pairing", "sarg"],
    "per-state": ["mdi"],
    "nested": ["sixstate"],
    "weighted": ["cow-vacuum"],
    "max": ["rrdps"],
    "outer-max": ["dmcs"],
    "root": ["dps"],
}

# The divisors the engines publish, read off src/.
DIVISORS = {
    "bb84": 21,
    "sixstate": 3,
    "mdi": 266,
    "pairing": 24,
    "sarg": 18,
    "cow-vacuum": 10,
}


def rrdps_flat(each):
    """
    The composed RRDPS parameter with all five terms set to `each` as they stand.
    """

    return _core.rrdps_secpar(each, each, each, each, each)


def rrdps_probs(each):
    """
    The same, with the two hash lengths converted to the bit lengths rrdps_secpar wants.
    """
    bits = -math.log2(each)

    return _core.rrdps_secpar(each, each, each, bits, bits)


def cv_ledger():
    """
    The Gaussian budget as qkd/link.py declares it: one q.FiniteSize.eps to all three events.
    """

    return security.compose(
        "cv",
        eps_pe=TARGET,
        eps_smooth=TARGET,
        eps_pa=TARGET,
    )


# The RRDPS block every length below is read at, at Takesue's own security target.
TRAIN_D = 2.0**-50
TRAIN = {
    "n_em": 1e11,
    "n_sift": 1e11 * _core.rrdps_counts(32, 0.005, 0.1, 1e-6, 0.015)[0],
    "n_double": 0.0,
    "l": 32,
    "mu": 0.005,
    "nu_th": 2,
    "e_bit": _core.rrdps_counts(32, 0.005, 0.1, 1e-6, 0.015)[1],
    "f_ec": 1.1,
    "p_two": 0.125,
}


# sarg_gain carries SARG's 1/4 sifting INSIDE the gain: a count below is a conclusive round.
SARG_ETA = 0.2

SARG_DARK = 1e-6

SARG_EDET = 0.005

SARG_N = 1e11


def sarg_record(e_det=SARG_EDET, mu=0.5, nu=0.1, probs=(0.9, 0.05)):
    """
    The observation set keylength() reads for family 'sarg' at one misalignment.
    """
    q_mu, e_mu = _core.sarg_gain(mu, SARG_ETA, SARG_DARK, e_det)
    q_nu, e_nu = _core.sarg_gain(nu, SARG_ETA, SARG_DARK, e_det)
    counts = (SARG_N * probs[0] * q_mu, SARG_N * probs[1] * q_nu)
    errors = (counts[0] * e_mu, counts[1] * e_nu)
    seen = counts[0] + counts[1]

    return {
        "mu": mu,
        "nu": nu,
        "probs": probs,
        "counts": counts,
        "errors": errors,
        "n_key": seen,
        "e_key": (errors[0] + errors[1]) / seen,
        "f_ec": 1.16,
    }


# The COW' record: the two DECOY gains at each port widen to Kato bounds, the BIT entries
# stay observed, and that box is what sdp_interval maximises over.
COW_MU = 0.00214

COW_TB = 0.35

COW_ETA = 0.1

COW_DARK = 1e-8

COW_N = 1e12

COW_DECOY = 0.05


def cow_ledger():
    """
    Li's budget at one common share: eps_sec = 2*eps + eps_0 + 6*eps_1 + eps_2.
    """

    return security.compose(
        "cow-vacuum",
        eps=TARGET / 10.0,
        eps_0=TARGET / 10.0,
        eps_1=TARGET / 10.0,
        eps_2=TARGET / 10.0,
        eps_cor=1e-15,
    )


def cow_record(led):
    """
    The observation set keylength() reads for 'cow-vacuum', through its three estimators in order.
    """
    each = led.shares()[0]
    q0, q1 = _core.sdp_gains(COW_MU, COW_TB, COW_ETA, COW_DARK)
    hi0, hi1, lo = list(q0), list(q1), [0.0, 0.0]
    sent = COW_N * COW_DECOY
    for j in (0, 1):
        wide = security.deviation("cow-vacuum", sent, sent * q0[j], each, True)
        hi0[j] = (sent * q0[j] + wide) / sent
        hi1[j] = (sent * q1[j] + security.deviation("cow-vacuum", sent, sent * q1[j], each, True)) / sent
        lo[j] = max(0.0, (sent * q0[j] - security.deviation("cow-vacuum", sent, sent * q0[j], each, False)) / sent)

    gain, e_z = _core.sdp_data(COW_MU, COW_TB, COW_ETA, COW_DARK)
    n_z = COW_N * (1.0 - 2.0 * COW_DECOY) * gain
    expect = security.interval("cow-vacuum", COW_MU, hi0, hi1, lo)

    return {
        "n_z": n_z,
        "e_z": e_z,
        "phase": security.conversion("cow-vacuum", expect, n_z, each),
        "f_ec": 1.1,
    }


# One complete budget per registered family.
TERMS = {
    "bb84": dict(eps_sec=TARGET, eps_cor=1e-15),
    "sarg": dict(eps_sec=TARGET, eps_cor=1e-15),
    "sixstate": dict(eps=TARGET, eps_ec=1e-15),
    "mdi": dict(eps_sec=TARGET, eps_cor=1e-15, states=2),
    "pairing": dict(eps_sec=TARGET, eps_cor=1e-15),
    "cv": dict(eps_pe=TARGET, eps_smooth=TARGET, eps_pa=TARGET),
    "cvmdi": dict(eps_smooth=TARGET, eps_pa=TARGET),
    "pair": dict(eps_pe=TARGET, eps_pa=TARGET, eps_ec=1e-15, t=50.0),
    "dmcs": DMCS_EPS,
    "dps": dict(eps_1=2.0**-58 / 6.0, eps_2=2.0**-58 / 6.0, zeta=58.0, zeta_ec=28.0),
    "cow-vacuum": dict(eps=TARGET / 10.0, eps_0=TARGET / 10.0, eps_1=TARGET / 10.0, eps_2=TARGET / 10.0, eps_cor=1e-15),
    "rrdps": {},
}


class Register(Question):
    """
    What the epsilon register holds, against the engines its rows were read from.
    """

    def test_families(self):
        """
        Twelve families are registered, each describing to a record naming a rule, terms, a src/
        source and a paper that catalogue() repeats.
        """
        names = security.families()
        self.assertEqual(len(names), 12, msg=f"registered families: {names}")

        for name in names:
            spec = security.describe(name)
            self.assertEqual(spec.name, name, msg=f"{name} describes to itself")
            self.assertTrue(spec.rule, msg=f"{name} states no composition rule")
            self.assertTrue(spec.terms, msg=f"{name} lists no terms")
            self.assertIn("src/", spec.source, msg=f"{name} names no engine")
            self.assertTrue(spec.paper, msg=f"{name} names no paper")

        rows = security.catalogue()
        self.assertEqual([row[0] for row in rows], list(names), msg="catalogue complete")

        for name, rule, terms, shares in rows:
            spec = security.describe(name)
            self.assertEqual(rule, spec.rule, msg=f"{name} rule")
            self.assertEqual(terms, tuple(t.name for t in spec.terms), msg=f"{name}")
            self.assertEqual(shares, spec.shares, msg=f"{name} shares")

    def test_kinds(self):
        """
        Every term declares one of three kinds and one of four roles, and only RRDPS, the pair link
        and DPS carry a bit-valued term.
        """
        bits = []
        for name in security.families():
            for term in security.describe(name).terms:
                self.assertIn(term.kind, security.KINDS, msg=f"{name}.{term.name}")
                self.assertIn(term.role, security.ROLES, msg=f"{name}.{term.name}")

                if term.kind == "bits":
                    bits.append(name)
        self.assertEqual(sorted(set(bits)), ["dps", "pair", "rrdps"], msg=f"bit terms: {bits}")

    def test_divisors(self):
        """
        Each registered share count is the divisor its engine applies.
        """
        for name, want in DIVISORS.items():
            self.assertEqual(security.describe(name).shares, want, msg=f"{name} share count")
        self.assertClose(_core.mdi_eps(TARGET) * 266.0, TARGET, atol=1e-24, msg="mdi divisor")
        self.assertClose(_core.pairing_eps(TARGET) * 24.0, TARGET, atol=1e-24, msg="pairing divisor")
        self.assertClose(_core.sarg_eps(TARGET) * 18.0, TARGET, atol=1e-24, msg="sarg divisor")
        self.assertClose(_core.sdp_eps(TARGET) * 10.0, TARGET, atol=1e-24, msg="cow-vacuum divisor")

    def test_forms(self):
        """
        Every family declares a form from security.FORMS, six composing as a plain sum and six each
        having an arm of their own.
        """
        seen = {}
        for name in security.families():
            form = security.describe(name).form
            self.assertIn(form, security.FORMS, msg=f"{name} declares form {form!r}")

            seen.setdefault(form, []).append(name)

        self.assertEqual(seen, FORMS, msg=f"form assignments: {seen}")

    def test_form_guards(self):
        """
        Six-state, MDI, RRDPS and DPS re-registered as a plain sum raise rather than compose by
        another family's rule.
        """
        for name in ("sixstate", "mdi", "rrdps", "dps"):
            led = security.allocate("rrdps", TARGET) if name == "rrdps" else security.compose(name, **TERMS[name])
            held = security.REGISTER[name]
            security.REGISTER[name] = dataclasses.replace(held, form="sum")

            try:
                self.assertFails(
                    NotImplementedError,
                    "plain sum adds probabilities",
                    lambda: led.secrecy,
                    msg=f"{name} composed as a plain sum",
                )
            finally:
                security.REGISTER[name] = held

    def test_forms_partial(self):
        """
        COW' and discrete modulation declare probabilities alone, so a plain sum over either returns
        a wrong number rather than raising.
        """
        for name, plain, want in (("cow-vacuum", 4e-11, 1e-10), ("dmcs", 1.7e-10, 8e-11)):
            led = security.compose(name, **TERMS[name])
            held = security.REGISTER[name]
            security.REGISTER[name] = dataclasses.replace(held, form="sum")

            try:
                self.assertClose(led.secrecy, plain, atol=1e-24, msg=f"{name} as a plain sum")
            finally:
                security.REGISTER[name] = held

            self.assertClose(led.secrecy, want, atol=1e-24, msg=f"{name} recomposes")


class Compose(Question):
    """
    Declared terms in, one composed number out.
    """

    def test_cv_triple(self):
        """
        q.FiniteSize.eps is per term over three failure events, so a declared 1e-10 composes to
        3e-10.
        """
        led = cv_ledger()
        self.assertClose(led.secrecy, 3.0 * TARGET, atol=1e-24, msg="cv secrecy")
        self.assertIn("union bound", led.rule, msg=f"cv rule: {led.rule}")

    def test_bb84_sum(self):
        """
        BB84-WCP's total is eps_sec + eps_cor, and its secrecy parameter divides 21 ways.
        """
        led = security.compose("bb84", eps_sec=TARGET, eps_cor=1e-15)
        each, count = led.shares()
        self.assertClose(led.total, TARGET + 1e-15, atol=1e-24, msg="bb84 total")
        self.assertEqual(count, 21, msg="Lim's collapsed term count")
        self.assertClose(each * 21.0, TARGET, atol=1e-24, msg="bb84 share")
        self.assertTrue(led.meets(1e-9), msg="1.00001e-10 does not meet 1e-9")
        self.assertFalse(led.meets(1e-11), msg="1.00001e-10 meets 1e-11")

    def test_sixstate_nesting(self):
        """
        Six-state's secrecy is eps - eps_ec over three shares, and sixstate_length refuses
        eps_ec >= eps.
        """
        led = security.compose("sixstate", eps=1e-5, eps_ec=TARGET)
        self.assertClose(led.total, 1e-5, atol=1e-24, msg="six total")
        self.assertClose(led.secrecy, 1e-5 - TARGET, atol=1e-24, msg="six secrecy")
        self.assertClose(led.correctness, TARGET, atol=1e-24, msg="six correctness")
        self.assertClose(led.shares()[0], (1e-5 - TARGET) / 3.0, atol=1e-20, msg="six share")
        self.assertFails(
            ValueError,
            "nothing left to split",
            _core.sixstate_length,
            1.0e6,
            1.0e5,
            0.01,
            0.01,
            1.1,
            TARGET,
            1e-5,
            msg="sixstate_length accepted eps_ec >= eps",
        )

    def test_mdi_states(self):
        """
        MDI-BB84's parameter is per announced Bell state, so two states compose to twice one.
        """
        one = security.compose("mdi", eps_sec=TARGET, eps_cor=1e-15, states=1)
        two = security.compose("mdi", eps_sec=TARGET, eps_cor=1e-15, states=2)
        self.assertClose(two.secrecy, 2.0 * one.secrecy, atol=1e-24, msg="mdi secrecy")
        self.assertClose(two.correctness, 2.0 * one.correctness, atol=1e-28, msg="mdi correctness")
        self.assertClose(two.shares()[0], _core.mdi_eps(TARGET), atol=1e-24, msg="mdi share")

    def test_pairing_sum(self):
        """
        Mode pairing's total is eps_sec + eps_cor, and its secrecy parameter divides 24 ways.
        """
        led = security.compose("pairing", eps_sec=TARGET, eps_cor=1e-15)
        self.assertClose(led.total, TARGET + 1e-15, atol=1e-24, msg="pairing total")
        self.assertClose(led.shares()[0] * 24.0, TARGET, atol=1e-24, msg="pairing share")

    def test_explain(self):
        """
        explain() reports each term with its kind and role, the composed halves, and the refusal
        text where a half does not compose.
        """
        rows = cv_ledger().explain()
        self.assertEqual(rows["eps_pe"]["label"], "pinned (eps, secrecy)", msg="label")
        self.assertEqual(rows["secrecy"]["value"], 3.0 * TARGET, msg="composed secrecy")
        self.assertEqual(rows["total"]["label"], "refused", msg="cv composes no total")
        self.assertEqual(rows["eps_auth"]["label"], "not declared", msg="auth absent")


class Takesue(Question):
    """
    The family whose budget is not a sum, and what an equal share costs it.
    """

    def test_split_lands(self):
        """
        allocate() at target d composes back to d exactly, at eps_1 = d, eps_2 = d^2/24 and 2^-s_z =
        d/2.
        """
        led = security.allocate("rrdps", TARGET)
        vals = led.values
        self.assertClose(led.secrecy, TARGET, atol=1e-24, msg="split lands on d")
        self.assertClose(vals["eps_1"], TARGET, atol=1e-24, msg="eps_1 = d")
        self.assertClose(vals["eps_2"], SMALL, atol=1e-30, msg="eps_2 = d^2/24")
        self.assertClose(2.0 ** -vals["s_z"], TARGET / 2.0, atol=1e-24, msg="eta_z = d/2")

    def test_equal_shares(self):
        """
        Five equal shares of the target compose to 2.41, above one, and to 1.1e5 times the target
        read as probabilities.
        """
        got = rrdps_flat(TARGET / 5.0)
        self.assertClose(got, FLAT, atol=1e-12, msg=f"flat share {got:.6g}")
        self.assertGreater(got, 1.0, msg=f"a flat share composes to {got:g}")

        got = rrdps_probs(TARGET / 5.0)
        self.assertClose(got, PROBS, atol=1e-15, msg=f"probability share {got:.6g}")
        self.assertClose(got / TARGET, RATIO, atol=1.0, msg="overshoot of the target")

    def test_sum_misreports(self):
        """
        Summing Takesue's five terms gives 1.5 times the target with the hash lengths converted and
        above 100 without.
        """
        led = security.allocate("rrdps", TARGET)
        spec = security.describe("rrdps")
        total = sum(spec.spec(n).eps(v) for n, v in led.terms)
        self.assertClose(total / TARGET, SUM_RATIO, atol=1e-6, msg="sum over max")

        raw = sum(v for _, v in led.terms)
        self.assertGreater(raw, RAW_SUM, msg=f"raw sum of the tuple is {raw:g}")

    def test_bit_terms(self):
        """
        A bit-typed term reports the eta it stands for, and a bit length handed to a probability
        slot is refused.
        """
        spec = security.describe("rrdps")
        got = spec.spec("s_x").eps(71.0235243984684)
        self.assertClose(got, SMALL, atol=1e-30, msg=f"eta_x from s_x is {got:.6g}")
        self.assertFails(
            ValueError,
            "eps_1 must be in (0, 1)",
            lambda: security.compose(
                "rrdps",
                eps_1=71.0,
                eps_2=SMALL,
                eps_3=SMALL,
                s_x=71.0,
                s_z=34.0,
            ),
            msg="a bit length was accepted as a probability",
        )


class Classical(Question):
    """
    The join between the bit-counting layer and the failure-probability layer.
    """

    def test_authenticate(self):
        """
        authenticate() adds forgery probability times message count, leaves its source ledger alone,
        and replaces rather than stacks.
        """
        led = security.compose("bb84", eps_sec=TARGET, eps_cor=1e-15)
        auth = security.authenticate(led, 12, 1e-12)
        self.assertClose(auth.authentication, 12e-12, atol=1e-24, msg="union bound")
        self.assertClose(auth.total, TARGET + 1e-15 + 12e-12, atol=1e-24, msg="authed total")
        self.assertClose(led.total, TARGET + 1e-15, atol=1e-24, msg="source unchanged")

        twice = security.authenticate(auth, 4, 1e-12)
        self.assertClose(twice.authentication, 4e-12, atol=1e-24, msg="restated")
        self.assertEqual(len(twice.terms), len(auth.terms), msg="term count grew")

    def test_saturated(self):
        """
        A message count driving the union bound to one is refused.
        """
        led = security.compose("bb84", eps_sec=TARGET, eps_cor=1e-15)
        self.assertFails(
            ValueError,
            "union bound has saturated",
            lambda: security.authenticate(led, int(1e13), 1e-12),
            msg="saturated union bound reported",
        )

    def test_hashing_barred(self):
        """
        An eps_pa offered to a family whose secrecy parameter already covers the hashing is refused.
        """

        self.assertFails(
            ValueError,
            "double-counts the hashing",
            lambda: security.compose(
                "bb84",
                eps_sec=TARGET,
                eps_cor=1e-15,
                eps_pa=1e-12,
            ),
            msg="privacy amplification was charged twice",
        )


class Lengths(Question):
    """
    The finite-key lengths this layer exports.
    """

    def test_rrdps_route(self):
        """
        keylength for RRDPS is the engine call, reporting back the target the ledger was allocated
        from.
        """
        led = security.allocate("rrdps", TRAIN_D)
        got = security.keylength(led, **TRAIN)
        direct = _core.rrdps_finite(
            TRAIN["n_em"],
            TRAIN["n_sift"],
            TRAIN["n_double"],
            TRAIN["l"],
            TRAIN["mu"],
            TRAIN["nu_th"],
            TRAIN["e_bit"],
            TRAIN["f_ec"],
            led.secrecy,
            TRAIN["p_two"],
        )
        self.assertEqual(tuple(got.values.values()), direct, msg="the route is the engine call")
        self.assertEqual(got.values["secpar"], TRAIN_D, msg="the target recomposes")
        self.assertGreater(got.bits, 0.0, msg=f"a block of {TRAIN['n_em']:g} packets keys")

    def test_pairing_route(self):
        """
        keylength for mode pairing is the engine call at the undivided eps_sec, in bits per sifted
        Z-pair.
        """
        led = security.compose("pairing", eps_sec=TARGET, eps_cor=1e-15)
        got = security.keylength(led, s0=1e6, s11=4e7, phi=0.03, n_z=1e8, e_z=0.02, f_ec=1.1)
        self.assertEqual(
            got.bits,
            _core.pairing_length(1e6, 4e7, 0.03, 1e8, 0.02, 1.1, TARGET, 1e-15),
            msg="the route is the engine call",
        )
        self.assertGreater(got.bits, 0.0, msg="the block keys")
        self.assertIn("per sifted Z-PAIR", got.explain()["units"]["value"], msg="the unit is stated")

    def test_pair_route(self):
        """
        BBM92 reads eps_pa alone, Eq. (58) entering the length as 2*log2(2*eps_pa), so the composed
        secrecy parameter gives a different length.
        """
        led = security.compose("pair", eps_pe=1e-10, eps_pa=1e-10, eps_ec=1e-10, t=40)
        seen = dict(m=1e9, k=1e8, nu=0.02, delta=0.02, cbar=0.5, leak=0.11 * 9e8)
        got = security.keylength(led, **seen)
        self.assertEqual(
            got.bits,
            _core.pair_length(1e9, 1e8, 0.02, 0.02, 0.5, 0.11 * 9e8, 40.0, 1e-10),
            msg="the route is the engine call",
        )
        self.assertGreater(got.bits, 0.0, msg="a block of 1e9 sifted rounds keys")
        self.assertEqual(led.secrecy, 2e-10, msg="secrecy is eps_pe + eps_pa")
        wrong = _core.pair_length(1e9, 1e8, 0.02, 0.02, 0.5, 0.11 * 9e8, 40.0, led.secrecy)
        self.assertNotEqual(got.bits, wrong, msg="reading the ledger wrong moves the length")
        self.assertIn("SIFTED rounds", got.explain()["units"]["value"], msg="the unit is stated")

    def test_sixstate_nesting(self):
        """
        Six-state nests eps_ec INSIDE eps, so the route hands the engine the outer term and not
        Ledger.secrecy.
        """
        # eps_ec is a fraction of eps here, not orders below it.
        led = security.compose("sixstate", eps=1e-9, eps_ec=2e-10)
        got = security.keylength(led, n=1e8, m_test=1e7, e_key=0.02, e_test=0.02, f_ec=1.1)
        self.assertEqual(
            got.bits,
            _core.sixstate_length(1e8, 1e7, 0.02, 0.02, 1.1, 1e-9, 2e-10)[0],
            msg="the route is the engine call",
        )
        self.assertEqual(led.secrecy, 8e-10, msg="secrecy is eps - eps_ec")
        wrong = _core.sixstate_length(1e8, 1e7, 0.02, 0.02, 1.1, led.secrecy, 2e-10)[0]
        self.assertNotEqual(got.bits, wrong, msg="reading the ledger wrong moves the length")

    def test_estimators_single(self):
        """
        The entropy the six-state length is taken at sits below sixstate_bound, and the sample width
        falls with the sample and rises as eps tightens.
        """
        bare = security.residual_entropy("sixstate", 0.02, 0.02)
        led = security.compose("sixstate", eps=TARGET, eps_ec=1e-15)
        used = security.keylength(led, n=1e8, m_test=1e7, e_key=0.02, e_test=0.02, f_ec=1.1)
        self.assertEqual(bare, _core.sixstate_bound(0.02, 0.02), msg="the bare entropy")
        self.assertLess(used.values["hxe"], bare, msg="the widened entropy is the smaller")

        widths = [security.sample_width("sixstate", m, 1e-11) for m in (1e6, 1e7, 1e8)]
        self.assertMonotone(widths, rising=False, msg="the width falls with the sample")
        self.assertGreater(
            security.sample_width("sixstate", 1e7, 1e-13),
            security.sample_width("sixstate", 1e7, 1e-9),
            msg="a tighter failure probability costs width",
        )

    def test_component_refused(self):
        """
        BB84, MDI and CV refuse keylength, each naming the component route that already exports a
        length.
        """
        pairs = (
            ("bb84", "q.SplittingAttack"),
            ("mdi", "q.TestBasisBound"),
            ("cv", "q.FiniteSize"),
        )
        for family, route in pairs:
            led = security.compose(family, **TERMS[family])
            self.assertFails(
                NotImplementedError,
                route,
                security.keylength,
                led,
                msg=f"{family} is already exported",
            )

    def test_observations_named(self):
        """
        An incomplete observation set and one carrying another family's count are both refused,
        naming the shape the family reads.
        """
        led = security.compose("sixstate", eps=TARGET, eps_ec=1e-15)
        self.assertFails(
            ValueError,
            "reads (n, m_test, e_key, e_test, f_ec)",
            lambda: security.keylength(led, n=1e8, m_test=1e7, e_key=0.02),
            msg="an incomplete shape",
        )
        self.assertFails(
            ValueError,
            "was given (e_key, e_test, f_ec, m_test, n, n_double)",
            lambda: security.keylength(led, n=1e8, m_test=1e7, e_key=0.02, e_test=0.02, f_ec=1.1, n_double=3.0),
            msg="another family's count",
        )

    def test_rrdps_shapes(self):
        """
        RRDPS's block shape and its certified-threshold shape return one length on one block,
        stating different units.
        """
        led = security.allocate("rrdps", TRAIN_D)
        whole = security.keylength(led, **TRAIN)
        rows = whole.values
        part = security.keylength(
            led,
            n_sift=TRAIN["n_sift"],
            tagged=rows["tagged"],
            faults=rows["faults"],
            phase=rows["phase"],
            e_bit=TRAIN["e_bit"],
            f_ec=TRAIN["f_ec"],
        )
        self.assertEqual(part.bits, whole.bits, msg="the two shapes agree on one block")
        self.assertEqual(
            part.bits,
            _core.rrdps_length(
                TRAIN["n_sift"],
                rows["tagged"],
                rows["faults"],
                rows["phase"],
                TRAIN["e_bit"],
                TRAIN["f_ec"],
                led.values["s_x"],
                led.values["s_z"],
            ),
            msg="the threshold shape is the engine call",
        )
        self.assertNotEqual(part.units, whole.units, msg="the two shapes state different units")

    def test_sarg_route(self):
        """
        SARG04's length is the sarg_eps, sarg_counts, sarg_errors and sarg_length chain, and the
        certified shape fed its own counts agrees.
        """
        led = security.compose("sarg", eps_sec=TARGET, eps_cor=1e-15)
        seen = sarg_record()
        whole = security.keylength(led, **seen)
        self.assertGreater(whole.bits, 0.0, msg="no SARG04 length was certified")

        each = _core.sarg_eps(TARGET)
        self.assertEqual(whole.values["eps_each"], each, msg="the estimators run at eps_sec/18")
        counts = _core.sarg_counts(seen["mu"], seen["nu"], seen["probs"], seen["counts"], seen["errors"], each)
        v1 = _core.sarg_errors(seen["mu"], seen["nu"], seen["probs"], seen["errors"], each)
        rows = whole.values
        self.assertEqual((rows["s0"], rows["s1"], rows["s0_hi"]), counts, msg="the counts are the engine's")
        self.assertEqual(rows["v1"], v1, msg="the error count is the engine's")
        self.assertEqual(
            whole.bits,
            _core.sarg_length(
                counts[0],
                counts[1],
                v1,
                seen["n_key"],
                seen["e_key"],
                seen["f_ec"],
                TARGET,
                1e-15,
            ),
            msg="the length is the engine call",
        )

        part = security.keylength(
            led,
            s0=counts[0],
            s1=counts[1],
            v1=v1,
            n_key=seen["n_key"],
            e_key=seen["e_key"],
            f_ec=seen["f_ec"],
        )
        self.assertEqual(part.bits, whole.bits, msg="the two shapes agree on one record")
        self.assertNotEqual(part.units, whole.units, msg="the two shapes state different units")

    def test_sarg_misalignment(self):
        """
        The SARG04 length falls with misalignment over e_det 0.002 to 0.01 and clamps to zero at
        0.015.
        """
        led = security.compose("sarg", eps_sec=TARGET, eps_cor=1e-15)
        got = [security.keylength(led, **sarg_record(e_det=e)).bits for e in (0.002, 0.005, 0.008, 0.01)]
        self.assertMonotone(got, rising=False, msg=f"the length must fall with misalignment: {got}")
        self.assertGreater(got[-1], 0.0, msg="the last point still certifies key")
        self.assertEqual(security.keylength(led, **sarg_record(e_det=0.015)).bits, 0.0, msg="zero, not negative")

    def test_vacuum_route(self):
        """
        COW' reaches a length through six Kato deviations, the interval and conversion estimators,
        and sdp_length at the recomposed total.
        """
        led = cow_ledger()
        seen = cow_record(led)
        self.assertLess(seen["phase"], 0.5, msg="the record aborts before it is priced")
        out = security.keylength(led, **seen)
        self.assertGreater(out.bits, 0.0, msg="no COW' length was certified")
        self.assertEqual(
            out.bits,
            _core.sdp_length(
                seen["n_z"],
                seen["e_z"],
                seen["phase"],
                seen["f_ec"],
                led.secrecy,
                led.correctness,
            ),
            msg="the length is the engine call",
        )
        self.assertEqual(led.secrecy, TARGET, msg="the engine takes the TOTAL, not a share")

        each = led.shares()[0]
        widened = security.conversion("cow-vacuum", 0.2, seen["n_z"], each)
        self.assertGreater(widened, 0.2, msg="the conversion step widens an expected rate")
        self.assertLess(
            security.conversion("cow-vacuum", 0.2, 10.0 * seen["n_z"], each),
            widened,
            msg="a longer block narrows it",
        )

    def test_vacuum_refused(self):
        """
        deviation and conversion refuse every family but COW' by name.
        """
        for family in ("bb84", "sixstate", "sarg", "rrdps", "cv", "dps"):
            self.assertFails(
                NotImplementedError,
                "does not reach one here",
                security.deviation,
                family,
                1e9,
                1e3,
                1e-11,
                True,
                msg=f"a Kato deviation for {family}",
            )
            self.assertFails(
                NotImplementedError,
                "does not reach one here",
                security.conversion,
                family,
                0.05,
                1e7,
                1e-11,
                msg=f"a conversion for {family}",
            )

    def test_estimators_refused(self):
        """
        sample_width and residual_entropy refuse every family but six-state by name.
        """
        for family in ("bb84", "mdi", "pairing", "rrdps", "cv"):
            self.assertFails(
                NotImplementedError,
                "does not reach one here",
                security.sample_width,
                family,
                1e7,
                1e-11,
                msg=f"a width for {family}",
            )
            self.assertFails(
                NotImplementedError,
                "does not reach one here",
                security.residual_entropy,
                family,
                0.02,
                0.02,
                msg=f"an entropy for {family}",
            )


class Contract(Question):
    """
    What the module refuses rather than approximating.
    """

    def test_stated_wrong(self):
        """
        An unregistered family and a budget short of a term are both refused, naming the registered
        names and the missing term.
        """

        self.assertFails(
            ValueError,
            "no epsilon budget is registered for 'cow'",
            security.describe,
            "cow",
            msg="an unregistered family composed",
        )
        self.assertFails(
            ValueError,
            "missing eps_cor",
            lambda: security.compose("bb84", eps_sec=TARGET),
            msg="a short budget composed",
        )

    def test_cv_total(self):
        """
        The Gaussian path composes neither a total nor a correctness parameter, naming cv_finite's
        three arguments and its frame error rate.
        """
        led = cv_ledger()
        self.assertFails(
            ValueError,
            "cv_finite takes eps_pe, eps_smooth and eps_pa",
            lambda: led.total,
            msg="cv reported a protocol parameter",
        )
        self.assertFails(
            ValueError,
            "FRAME ERROR RATE",
            lambda: led.correctness,
            msg="cv reported a correctness parameter",
        )

    def test_rrdps_total(self):
        """
        RRDPS composes no total, the refusal naming the confidence interval its engine carries none
        of.
        """
        led = security.allocate("rrdps", TARGET)
        self.assertFails(
            ValueError,
            "carries no confidence interval",
            lambda: led.total,
            msg="rrdps reported a protocol parameter",
        )

    def test_no_shares(self):
        """
        CV and RRDPS refuse an equal share, naming three independent arguments and a maximum over
        two branches.
        """
        cv = cv_ledger()
        rr = security.allocate("rrdps", TARGET)
        self.assertFails(
            ValueError,
            "three independent arguments",
            cv.shares,
            msg="cv reported a share count",
        )
        self.assertFails(
            ValueError,
            "a maximum over two branches",
            rr.shares,
            msg="rrdps reported a share count",
        )

    def test_no_allocation(self):
        """
        allocate() runs for RRDPS alone, every other family naming the absent published assignment.
        """
        for name in security.families():
            if name == "rrdps":
                continue
            self.assertFails(
                NotImplementedError,
                "no published assignment",
                security.allocate,
                name,
                TARGET,
                msg=f"{name} allocated",
            )

    def test_bad_target(self):
        """
        A target of 0.0 or 1.0 is refused, matching src/std.rs::check_eps.
        """
        for bad in (0.0, 1.0):
            self.assertFails(
                ValueError,
                "target must be in (0, 1)",
                security.allocate,
                "rrdps",
                bad,
                msg=f"a target of {bad} allocated",
            )


class Shapes(Question):
    """
    The three compositions no other family shares.
    """

    def test_vacuum_weights(self):
        """
        Li's 2*eps + eps_0 + 6*eps_1 + eps_2 recomposes the target over ten shares, where a plain
        sum reports two fifths of it.
        """
        led = cow_ledger()
        self.assertClose(led.secrecy, TARGET, atol=1e-24, msg="Li's weights recompose the target")

        each, count = led.shares()
        self.assertEqual(count, 10, msg="ten shares, six of them Kato estimations")
        self.assertClose(each, TARGET / 10.0, atol=1e-24, msg="the share is the term declared")

        plain = sum(led.values[name] for name in ("eps", "eps_0", "eps_1", "eps_2"))
        self.assertClose(plain / led.secrecy, 0.4, atol=1e-12, msg="a plain sum reports two fifths")
        self.assertClose(led.total, TARGET + 1e-15, atol=1e-24, msg="eps_cor is separate and additive")

    def test_dmcs_theorem(self):
        """
        Kanitschar Theorem 6 adds eps_ec outside a maximum over two branches, composing to the
        paper's 1e-10 where a plain sum reports 1.9e-10.
        """
        led = security.compose("dmcs", **DMCS_EPS)
        vals = led.values
        whole = _core.dm_secpar(vals["eps_ec"], vals["eps_pa"], vals["eps_bar"], vals["eps_et"], vals["eps_at"])
        self.assertClose(led.total, whole, atol=1e-24, msg="halves against dm_secpar")
        self.assertClose(led.total, 1e-10, atol=1e-24, msg="Kanitschar's own demonstration value")
        self.assertClose(led.secrecy, 0.8e-10, atol=1e-24, msg="the branches tie at 8e-11")
        self.assertClose(sum(vals.values()), 1.9e-10, atol=1e-24, msg="a plain sum reports 1.9e-10")

    def test_dps_root(self):
        """
        MTT23 Eq. (50) composes to exactly 2^-27 at the paper's parameters, with no correctness
        parameter and no equal shares.
        """
        led = security.compose("dps", eps_1=2.0**-58 / 6.0, eps_2=2.0**-58 / 6.0, zeta=58.0, zeta_ec=28.0)
        self.assertEqual(led.secrecy, 2.0**-27, msg="Eq. (50) to the last bit")
        self.assertFails(
            ValueError,
            "charges the error-verification hash INSIDE it",
            lambda: led.correctness,
            msg="dps reported a correctness parameter",
        )
        self.assertFails(
            ValueError,
            "no equal shares",
            led.shares,
            msg="dps reported a share count",
        )

    def test_every_ledger_explains(self):
        """
        Every registered family explains to its own name, each half, the total and per_bound being a
        number or a refusal text.
        """
        for name in security.families():
            led = security.allocate("rrdps", TARGET) if name == "rrdps" else security.compose(name, **TERMS[name])
            rows = led.explain()
            self.assertEqual(rows["family"]["value"], name, msg=f"{name} explains to itself")

            for half in ("secrecy", "correctness", "total", "per_bound"):
                self.assertIn(half, rows, msg=f"{name} omits {half}")
                self.assertTrue(str(rows[half]["value"]), msg=f"{name}.{half} is empty")


class Paired(Question):
    """
    The four mode-pairing estimators beside the length.
    """

    def test_chernoff_apart(self):
        """
        Both Chernoff directions bracket the count at different widths, so neither is the other's
        inverse.
        """
        lo, hi = security.confidence("pairing", 1e6, 1e-11)
        seen, most = security.confidence("pairing", 1e6, 1e-11, kind="observed")
        self.assertLess(lo, 1e6, msg=f"expected range {lo}, {hi}")
        self.assertGreater(hi, 1e6, msg=f"expected range {lo}, {hi}")
        self.assertLess(seen, 1e6, msg=f"observed range {seen}, {most}")
        self.assertGreater(most, 1e6, msg=f"observed range {seen}, {most}")
        self.assertNotAlmostEqual(hi - lo, most - seen, places=6, msg="the two directions became inverses")

        self.assertFails(
            ValueError,
            "not inverses",
            lambda: security.confidence("pairing", 1e6, 1e-11, kind="both"),
            msg="unnamed direction chosen",
        )

    def test_estimators_route(self):
        """
        Each pairing estimator equals its _core call in the engine's argument order.
        """

        self.assertEqual(
            security.confidence("pairing", 1e6, 1e-11),
            _core.pairing_expect(1e6, 1e-11),
            msg="confidence() drifted from pairing_expect",
        )
        self.assertEqual(
            security.confidence("pairing", 1e6, 1e-11, kind="observed"),
            _core.pairing_observe(1e6, 1e-11),
            msg="confidence() drifted from pairing_observe",
        )
        self.assertEqual(
            security.single_yield("pairing", 0.01, 0.4, 1e-4, 2e-3, 1e-8),
            _core.pairing_yield(0.01, 0.4, 1e-4, 2e-3, 1e-8),
            msg="single_yield() drifted",
        )
        self.assertEqual(
            security.sampling("pairing", 1e6, 1e5, 0.02, 1e-11),
            _core.pairing_gamma(1e6, 1e5, 0.02, 1e-11),
            msg="sampling() drifted",
        )
        self.assertEqual(
            security.phase_bound("pairing", 1e6, 1e5, 0.02, 1e-11),
            _core.pairing_phase(1e6, 1e5, 0.02, 1e-11),
            msg="phase_bound() drifted",
        )

    def test_sampling_floor(self):
        """
        The pairing sampling penalty does not vanish with the observed rate, and the phase bound is
        the monitored error plus exactly it.
        """
        width = security.sampling("pairing", 1e6, 1e5, 0.02, 1e-11)
        self.assertEqual(
            security.phase_bound("pairing", 1e6, 1e5, 0.02, 1e-11),
            0.02 + width,
            msg="phase bound is not error + penalty",
        )
        self.assertEqual(security.sampling("pairing", 1e6, 1e5, 0.0, 1e-11), 0.5, msg="a clean sample certified")
        self.assertGreater(
            security.sampling("pairing", 1e6, 1e5, 1e-6, 1e-11),
            1e-4,
            msg="the penalty vanished with the observed rate",
        )

    def test_paired_refused(self):
        """
        The four pairing estimators refuse every other family by name.
        """
        rows = (
            (security.confidence, (1e6, 1e-11)),
            (security.single_yield, (0.01, 0.4, 1e-4, 2e-3, 1e-8)),
            (security.sampling, (1e6, 1e5, 0.02, 1e-11)),
            (security.phase_bound, (1e6, 1e5, 0.02, 1e-11)),
        )
        for family in ("bb84", "mdi", "sixstate", "rrdps", "cv"):
            for fn, args in rows:
                self.assertFails(
                    NotImplementedError,
                    "does not reach one here",
                    fn,
                    family,
                    *args,
                    msg=f"{fn.__name__} answered for {family}",
                )


if __name__ == "__main__":
    rc = Exam(
        "SecurityRegister",
        "What the epsilon register holds and how it agrees with the engines",
        "security_register.md",
    ).run(load(Register))
    rc |= Exam(
        "SecurityCompose",
        "Declared terms composed by each family's own rule",
        "security_compose.md",
    ).run(load(Compose))
    rc |= Exam(
        "SecurityTakesue",
        "The budget that is not a sum, and what an equal share costs it",
        "security_takesue.md",
    ).run(load(Takesue))
    rc |= Exam(
        "SecurityClassical",
        "Joining the authentication bill to the security parameter",
        "security_classical.md",
    ).run(load(Classical))
    rc |= Exam(
        "SecurityLengths",
        "The five finite-key lengths this layer exports, and the shapes that differ",
        "security_lengths.md",
    ).run(load(Lengths))
    rc |= Exam(
        "SecurityShapes",
        "The three compositions no other family shares, each against its own engine",
        "security_shapes.md",
    ).run(load(Shapes))
    rc |= Exam(
        "SecurityContract",
        "What the module refuses rather than approximating",
        "security_contract.md",
    ).run(load(Contract))
    rc |= Exam(
        "SecurityPaired",
        "The four mode-pairing estimators beside the length, and what each refuses",
        "security_paired.md",
    ).run(load(Paired))
    sys.exit(rc)
