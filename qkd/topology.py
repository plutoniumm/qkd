import math
from dataclasses import dataclass

from . import _core, std
from .components import (
    Asymptotic,
    BasisKeying,
    BellAnalyser,
    BellDetector,
    Channel,
    CorrelatedEnvironment,
    Fiber,
    GaussianModulation,
    Relay,
    Sender,
    TestBasisBound,
    TwoModeBound,
)

_q = std.q

# mdi_e11's cell order over the row-major 3x3 grid, cell (i, j) at 3*i + j:
# (decoy, decoy), (vacuum, vacuum), (decoy, vacuum), (vacuum, decoy).
_CELLS = (4, 8, 5, 7)

# Announcement classes the coupler optics produce, and the divisor on every
# per-class count. NOT q.BellAnalyser(states=...), which is how many of the two
# the relay KEEPS; mdi_rect and mdi_diag report the gain over both classes.
_CLASSES = 2


@dataclass(frozen=True)
class Attack:
    """
    One named Eve against a continuous-variable relay, bits per use at the
    configured v_a and beta (cvmdi_point). NOT the bound SwapResult.key_rate
    reports: the two do not sandwich each other, and this key_rate is NEVER
    clamped.
    """

    i_ab: float
    chi_e: float
    key_rate: float


@dataclass(frozen=True)
class SwapResult:
    """
    key_rate is a bound, clamped at 0 HERE and nowhere else: explain()'s
    key_bound and key_raw are unclamped, attack.key_rate never. A field the
    running measurement does not produce is None.

    BITS PER USE of the swap on the continuous-variable branch, BITS PER PULSE
    PAIR on the qubit one. chi is what the arms carry, floor pure loss, least
    where no covariance exists; the qubit rows below it are the announcement
    gain, its error rate, and the three single-photon-pair quantities the decoy
    grid bounds.

    hom is the Hong-Ou-Mandel dip visibility, present only where both senders
    carried a pulse width. A DIAGNOSTIC: what reaches the rate is the
    test-basis misalignment derived beside it, not the dip.
    """

    key_rate: float
    chi: float | None = None
    floor: float | None = None
    least: float | None = None
    attack: Attack | None = None
    explain: dict | None = None
    p_click: float | None = None
    qber: float | None = None
    y1: float | None = None
    e1: float | None = None
    q1: float | None = None
    hom: float | None = None
    # Counting path with a q.RelayBlock only. key_length is BITS FOR THE WHOLE
    # BLOCK summed over the announced Bell states and key_rate is length / n;
    # n0, n1, phi and n_key are PER ANNOUNCED BELL STATE.
    key_length: float | None = None
    n0: float | None = None
    n1: float | None = None
    phi: float | None = None
    n_key: float | None = None
    states: int | None = None


class Swap:
    """
    Two senders, two channels and one untrusted node between them: the
    topology is the security model, so nothing takes a trusted flag and the
    relay's imperfections fold into the arms. It is the time reversal of BBM92
    (Lo, Curty & Qi, PRL 108, 130503 (2012)).

    THE RELAY'S MEASUREMENT SELECTS THE FAMILY, and the two share no formula.
    q.BellDetector is two conjugate homodynes across a balanced coupler, takes
    q.GaussianModulation senders, and reports bits per use of the swap
    (Pirandola, Nat. Photonics 9, 397 (2015)). q.BellAnalyser is four threshold
    detectors behind one, takes q.BasisKeying senders with a q.Decoy intensity
    set each, and reports bits per PULSE PAIR off a two-dimensional decoy grid.

    ONLY THE COUNTING PATH HAS A FINITE BRANCH.
    q.TestBasisBound(block=q.RelayBlock(...)) asks for a key LENGTH in bits
    over a stated number of emitted pulse PAIRS, and reads two numbers the
    asymptotic rate does not: each sender's key-basis probability,
    q.BasisKeying(bias=...), and how many Bell states the analyser announces,
    q.BellAnalyser(states=...), the length being stated PER ANNOUNCED STATE and
    summed over them at a summed budget.

    With weak coherent pulses there is no entanglement anywhere: the swapping
    picture is the VIRTUAL protocol of the security proof, not the hardware,
    which does two-photon interference at a coupler and coincidence detection.
    """

    def __init__(self, alice, bob, relay, channels, security=None, environment=None):
        # channels is (Alice's arm, Bob's arm), not symmetric under a swap; an
        # arm's xi is in SNU at its own CHANNEL INPUT.
        self.alice = alice
        self.bob = bob
        self.relay = relay
        self.channels = tuple(channels)
        self.security = security if security is not None else Asymptotic()
        self.environment = environment

    def _check(self):
        for name, party in (("alice", self.alice), ("bob", self.bob)):
            if not isinstance(party, Sender):
                raise ValueError(f"{name} must be a q.Sender")

        if not isinstance(self.relay, Relay):
            raise ValueError("relay must be a q.Relay")

        if len(self.channels) != 2:
            raise ValueError("channels must be (alice's arm, bob's arm)")

        for ch in self.channels:
            if not isinstance(ch, (Channel, Fiber)):
                raise ValueError("each channel must be a q.Channel or q.Fiber")

        if isinstance(self.relay.bell, BellAnalyser):
            return self._check_qubit()

        if not isinstance(self.relay.bell, BellDetector):
            raise ValueError(
                "the relay's measurement must be a q.BellDetector (two "
                "conjugate homodynes across a balanced coupler) or a "
                "q.BellAnalyser (four threshold detectors behind one)"
            )

        return self._check_bell()

    def _check_qubit(self):
        """
        The counting path: decoy weak-coherent senders into a qubit Bell-state
        measurement.
        """
        for name, party in (("alice", self.alice), ("bob", self.bob)):
            self._check_sender(name, party)

        self._check_modes()

        if not isinstance(self.security, TestBasisBound):
            raise NotImplementedError(
                "a Bell-analysed midpoint takes q.TestBasisBound security: the "
                "phase error rides on the single-photon-pair error rate of the "
                "TEST basis, measured through the decoy grid. The finite-key "
                "form is a slot on that same class: "
                "q.TestBasisBound(block=q.RelayBlock(...))"
            )

        self._check_bias()

        if self.security.block is not None:
            # Called for its refusal alone, before anything is computed.
            self.security.block.per_state(self.relay.bell.states)

        if self.environment is not None:
            raise NotImplementedError(
                "q.CorrelatedEnvironment names a Gaussian cross-covariance and "
                "the counting path has no term for one: the decoy grid bounds "
                "the single-photon-pair yield from observed gains alone. Drop it"
            )

        for tag, ch in zip("ab", self.channels):
            if isinstance(ch, Channel) and ch.xi != 0.0:
                raise ValueError(
                    f"arm {tag} carries an excess noise, a Gaussian quantity no "
                    "term of this bound consumes: a threshold detector sees a "
                    "click or nothing. Give the arm as a bare transmittance and "
                    "put the receiver's imperfection on q.BellAnalyser"
                )

    def _check_sender(self, name, party):
        mod = party.modulation
        if not isinstance(mod, BasisKeying):
            raise NotImplementedError(
                "a qubit Bell-state measurement reads coincidences between two "
                f"keyed sources: {name} needs q.BasisKeying with a q.Decoy "
                "intensity set. q.PolarisationKeying DERIVES a misalignment "
                "from its own reference frame while q.BellAnalyser already "
                "carries two, so the same error would be charged twice"
            )

        if mod.bases != 2 or mod.announce != "basis":
            raise NotImplementedError(
                "the midpoint bound is written for two conjugate bases with the "
                "basis announced: neither the three-basis nor the pair-announced "
                "variant has a midpoint form here"
            )

        if mod.sift != 1.0:
            raise ValueError(
                "the midpoint rate is asymptotic in the biased-basis limit, "
                "where the key-basis gain already counts key-basis rounds only, "
                "so a sifting factor here would charge the same sifting twice. "
                "Pass q.BasisKeying(..., sift=1.0)"
            )

    def _check_bias(self):
        """
        Each sender's key-basis probability: read by the finite chain alone,
        and NOT `sift`.
        """
        wanted = self.security.block is not None
        for name, party in (("alice", self.alice), ("bob", self.bob)):
            bias = party.modulation.bias
            if wanted and bias is None:
                raise NotImplementedError(
                    f"{name} declares no key-basis probability and Curty's "
                    "length reads one per sender: the two 3x3 grids are the "
                    "outer product of the senders' intensity ladders weighted "
                    "by bias_a*bias_b in the key basis and by "
                    "(1 - bias_a)*(1 - bias_b) in the test basis. Pass "
                    "q.BasisKeying(..., sift=1.0, bias=...), a separate field "
                    "from sift"
                )

            if not wanted and bias is not None:
                raise ValueError(
                    f"{name} declares a key-basis probability and the "
                    "asymptotic midpoint rate reads no term of one, so the "
                    "number would be accepted and dropped. Ask for the finite "
                    "length, q.TestBasisBound(block=q.RelayBlock(...)), or drop "
                    "the bias"
                )

    def _check_modes(self):
        """
        The two misalignments, and the two-source mode model that may supply
        the test basis's. THE KEY BASIS IS NEVER DERIVED: separating two
        polarisations at a beamsplitter needs no interference.
        """
        bell = self.relay.bell
        pair = self._widths()
        if pair is None and (self.alice.pulse, self.bob.pulse) != (None, None):
            raise NotImplementedError(
                "the two-source mode model reads a PAIR of temporal modes, so "
                "one q.Sender(pulse=...) width alone would be accepted and "
                "dropped. Give both widths, or neither and the test basis's "
                "error directly, q.BellAnalyser(misalign_test=...)"
            )

        if bell.misalign is None:
            raise NotImplementedError(
                "q.BellAnalyser(misalign=...) has no default and no derived "
                "branch here: the key basis needs no Hong-Ou-Mandel "
                "indistinguishability and the test basis does, so one value for "
                "both is the error that makes a midpoint rate look good. Pin "
                "the key basis's own number"
            )

        if bell.misalign_test is None and pair is None:
            raise NotImplementedError(
                "q.BellAnalyser(misalign_test=...) has no default: the key "
                "basis needs no Hong-Ou-Mandel indistinguishability and the "
                "test basis does, so one value for both is the error that makes "
                "a midpoint rate look good. Give it directly, or give BOTH "
                "senders a q.Sender(pulse=...) width and the two-source mode "
                "model derives it"
            )

        if bell.misalign_test is not None and pair is not None:
            raise ValueError(
                "both senders carry a q.Sender(pulse=...) width AND "
                "q.BellAnalyser(misalign_test=...) is pinned, so one of the two "
                "would be accepted and dropped. Drop the widths to pin the "
                "number, or drop the number to derive it"
            )

    def _widths(self):
        """
        The two senders' temporal-mode widths, or None where the pair is
        incomplete: an overlap is a property of TWO modes.
        """
        pair = (self.alice.pulse, self.bob.pulse)
        if None in pair:
            return None

        return pair

    def _modes(self):
        """
        (intensity overlap, dip visibility, test-basis misalignment) of the two
        sources, or None where no pair of widths was given. Both senders are
        taken SYNCHRONISED, so a width mismatch is the only mismatch modelled.
        """
        pair = self._widths()
        if pair is None:
            return None

        bell = self.relay.bell
        mu_a = self.alice.modulation.decoy.signal
        mu_b = self.bob.modulation.decoy.signal
        overlap = _core.mode_overlap(pair[0], pair[1], 0.0)
        dip = _core.hom_visibility(overlap, mu_a, mu_b)
        # Against its own ceiling at these intensities, the 1/2 being a SOURCE
        # property: mdi_diag already carries the arriving imbalance, and pricing
        # it again would charge one imperfection twice.
        ceiling = _core.hom_visibility(1.0, mu_a, mu_b)
        share = dip / ceiling
        # Never below bell.misalign, and share <= 1 by monotonicity in
        # `overlap`, so no clamp is needed.
        miss = bell.misalign + (0.5 - bell.misalign) * (1.0 - share)

        return overlap, dip, miss

    def _check_bell(self):
        """
        The covariance path: ONE modulation variance for both senders.
        """
        for party in (self.alice, self.bob):
            if not isinstance(party.modulation, GaussianModulation):
                raise NotImplementedError(
                    "a q.BellDetector conditions a Gaussian state: both senders "
                    "need q.GaussianModulation. Two keyed weak-coherent senders "
                    "take q.Relay(bell=q.BellAnalyser(...)) instead, which "
                    "shares no formula with this"
                )

            if party.pulse is not None:
                raise ValueError(
                    "the covariance path carries no two-source mode model, so a "
                    "pulse width would be accepted and dropped"
                )

        if self.alice.modulation.v_a != self.bob.modulation.v_a:
            raise NotImplementedError(
                "the relay covariance is built at one modulation variance for "
                "both senders; asymmetric modulation is not implemented"
            )

        if not isinstance(self.security, (Asymptotic, TwoModeBound)):
            raise NotImplementedError(
                "the continuous-variable relay reduction takes q.Asymptotic("
                "beta=...), or no security at all, for the asymptotic bound, or "
                "q.TwoModeBound(block=q.GaussianBlock(...)) for "
                "the finite-size RATE of Papanastasiou, Ottaviani & Pirandola, "
                "Phys. Rev. A 96, 042332 (2017), which is a rate per use and "
                "not a length"
            )

        if self.environment is not None and not isinstance(self.environment, CorrelatedEnvironment):
            raise ValueError("environment must be a q.CorrelatedEnvironment")

    def _fiber(self, i):
        """
        One arm's channel as (T, xi), with xi at the CHANNEL INPUT plane.
        """
        ch = self.channels[i]
        if isinstance(ch, Channel):
            return ch.T, std.input_xi(ch)

        return ch.transmittance, 0.0

    def _arm(self, i):
        """
        One arm as the covariance engine sees it: (tau, omega), the channel's
        thermal-loss map composed with the relay detector's eta and v_el.
        """
        bell = self.relay.bell
        t, xi = self._fiber(i)
        w = _core.relay_omega(t, xi)
        tau = bell.eta * t
        if tau >= 1.0:
            if bell.v_el > 0.0:
                raise ValueError(
                    "a lossless arm with a perfect detector has no environment " f"to carry v_el = {bell.v_el}"
                )

            return 1.0, 1.0

        noise = bell.eta * (1.0 - t) * w + (1.0 - bell.eta) + bell.v_el

        return tau, noise / (1.0 - tau)

    def _refer(self):
        """
        The factor carrying a correlation from the CHANNEL INPUT, where
        CorrelatedEnvironment names it, to the folded arms. At most 1.
        """
        bell = self.relay.bell
        ta = self._fiber(0)[0]
        tb = self._fiber(1)[0]
        room = (1.0 - bell.eta * ta) * (1.0 - bell.eta * tb)
        if not room > 0.0:
            # A lossless arm behind a perfect detector admits no environment.
            return 0.0

        return bell.eta * math.sqrt((1.0 - ta) * (1.0 - tb) / room)

    def _corr(self, scale=None):
        """
        Eve's correlations between the two environments, (g, gp), referred to
        the folded arms by _refer's factor, which the caller may pass in.
        """
        env = self.environment
        if env is None:
            return 0.0, 0.0

        if scale is None:
            scale = self._refer()

        return env.x * scale, env.p * scale

    def _check_room(self, arms):
        """
        That the two folded environments have covariance left to correlate. A
        pure mode, omega = 1, has none: room comes from an arm's excess noise
        or the relay detector's v_el, never from efficiency.
        """
        if self.environment is None:
            return

        if self.environment.x == 0.0 and self.environment.p == 0.0:
            return

        # Arithmetic tolerance: the fold cancels to omega = 1 on a pure-loss arm
        # and lands a few ulp either side of it.
        pure = [tag for (_, w), tag in zip(arms, "ab") if w <= 1.0 + 1e-9]
        if not pure:
            return

        where = "arms a and b" if len(pure) == 2 else f"arm {pure[0]}"

        raise ValueError(
            f"the environment of {where} is the vacuum, which a "
            "q.CorrelatedEnvironment has no room in: a pure mode holds no "
            "cross-covariance. Give the arm excess noise, q.Channel(T=..., "
            "xi=..., ref=...), or the relay detector electronic noise, "
            "q.BellDetector(v_el=...); detector efficiency alone only mixes in "
            "more vacuum"
        )

    def run(self):
        """
        The rate this topology supports. Every path is a closed form, so there
        is nothing to seed.
        """
        self._check()
        if isinstance(self.relay.bell, BellAnalyser):
            return self._run_qubit()

        return self._run_bell()

    def _arms(self):
        """
        (eta_a, eta_b) on the counting path: each arm's transmittance carrying
        the midpoint detector's own efficiency, since to an untrusted node a
        photon lost in the fibre and one lost at the detector are the same loss.
        """
        eta = self.relay.bell.eta

        return self._fiber(0)[0] * eta, self._fiber(1)[0] * eta

    def _grid(self, fn, sets, arms, miss):
        """
        One basis's 3x3 forward model at the two arms, row-major with Alice's
        intensity as the row, the layout the decoy bounds read.
        """
        dark = self.relay.bell.dark

        return [fn(x, y, arms[0], arms[1], dark, miss) for x in sets[0] for y in sets[1]]

    def _run_qubit(self):
        # Bits per PULSE PAIR. Every observable is the forward model at the two
        # arms; every single-photon-pair quantity is the joint decoy bound, and
        # the two are never composed the other way round.
        bell = self.relay.bell
        arms = self._arms()
        sets = (
            list(self.alice.modulation.decoy.intensities),
            list(self.bob.modulation.decoy.intensities),
        )
        # The TEST basis alone may take a derived misalignment.
        modes = self._modes()
        miss = bell.misalign_test if modes is None else modes[2]
        zed = self._grid(_core.mdi_rect, sets, arms, bell.misalign)
        test = self._grid(_core.mdi_diag, sets, arms, miss)
        y_key = _core.mdi_y11(sets[0], sets[1], [g for g, _ in zed])
        y_test = _core.mdi_y11(sets[0], sets[1], [g for g, _ in test])
        e1 = _core.mdi_e11(
            sets[0][1],
            sets[0][2],
            sets[1][1],
            sets[1][2],
            *[test[k][0] * test[k][1] for k in _CELLS],
            y_test,
        )
        gain, qber = zed[0]
        q1 = _core.mdi_gain(y_key, sets[0][0], sets[1][0])
        key = _core.mdi_rate(q1, e1, gain, qber, self.security.f)
        out = (gain, qber, y_key, y_test, e1, q1, key, miss)
        info = self._explain_qubit(arms, sets, modes, out)
        if self.security.block is not None:
            return self._run_length(zed, test, info, out, modes)

        # mdi_rate clamps in Rust, as cow_rate and b92_point do, so this clamp
        # and the key_raw row beside it agree rather than bracketing anything.
        return SwapResult(
            key_rate=max(0.0, key),
            p_click=gain,
            qber=qber,
            y1=y_key,
            e1=e1,
            q1=q1,
            hom=None if modes is None else modes[1],
            explain=info,
        )

    def _counted(self, zed, test):
        """
        (key probabilities, key counts, test probabilities, test counts, test
        error counts) over the 3x3 intensity grid, every count PER ANNOUNCEMENT
        CLASS.

        THE PROBABILITY GRID IS AN OUTER PRODUCT and a marginal of it is not
        it: each cell is the two senders' intensity weights multiplied,
        weighted again by bias_a*bias_b in the key basis and
        (1 - bias_a)*(1 - bias_b) in the test one, so the two grids sum to less
        than 1 apiece.
        """
        block = self.security.block
        one, two = self.alice.modulation, self.bob.modulation
        share = block.n / _CLASSES
        wa, wb = one.decoy.weights, two.decoy.weights
        keyed, tested = [], []
        counts, tests, faults = [], [], []
        for i in range(3):
            for j in range(3):
                cell = 3 * i + j
                joint = wa[i] * wb[j]
                p_key = one.bias * two.bias * joint
                p_test = (1.0 - one.bias) * (1.0 - two.bias) * joint
                sample = share * p_test * test[cell][0]
                keyed.append(p_key)
                tested.append(p_test)
                counts.append(share * p_key * zed[cell][0])
                tests.append(sample)
                faults.append(sample * test[cell][1])

        return keyed, counts, tested, tests, faults

    def _branch(self, counts, eps):
        """
        Which Chernoff branch mdi_deviate takes in each of the nine cells,
        row-major: 1 is both tails multiplicative, 6 is both back on Hoeffding.

        The crossing needs the Hoeffding confidence bound
        mu_L = x - sqrt(n ln(1/eps) / 2) to clear about 105 at eps_sec/266, and
        `n` there is the TOTAL over the grid, so a cell six decades below the
        signal cell pays the whole grid's width.
        """
        total = sum(counts)

        return tuple(_core.mdi_deviate(x, total, eps)[2] for x in counts)

    def _run_length(self, zed, test, info, out, modes):
        """
        Curty's finite-key length, bits over the whole block. `out` is the
        asymptotic tuple _run_qubit already computed.

        Every estimator here is a BOUND read off the two forward grids, and
        `mdi_yield` -- the forward single-photon-pair model -- appears nowhere
        in the chain: composing the two the wrong way round is the error that
        makes a midpoint rate look great.
        """
        block = self.security.block
        states = self.relay.bell.states
        eps = block.per_state(states)
        # One share of Curty's 266-way split, which every estimator is taken at.
        share = _core.mdi_eps(eps)
        keyed, counts, tested, tests, faults = self._counted(zed, test)
        aset = list(self.alice.modulation.decoy.intensities)
        bset = list(self.bob.modulation.decoy.intensities)
        # The code string is the signal-setting KEY-basis announcements.
        kept = block.code * counts[0]
        n0 = _core.mdi_vacuum(aset, bset, keyed, counts, kept, share)
        n1 = _core.mdi_single(aset, bset, keyed, counts, kept, share)
        # The pair and error counts read the TEST basis, n0 and n1 the KEY
        # basis: one grid for both is the same slip as one misalignment for both.
        nbar = _core.mdi_pairs(aset, bset, tested, tests, share)
        ebar = _core.mdi_faults(aset, bset, tested, faults, share)
        phi = _core.mdi_phase(n1, nbar, ebar, share)
        each = _core.mdi_length(n0, n1, phi, kept, zed[0][1], self.security.f, eps, block.eps_cor)
        # l = sum_k l_k at eps_sec = sum_k eps_{k,sec}, Curty's own composition.
        # Behind a balanced coupler the announcement classes are symmetric, so
        # every term of the sum is this one.
        total = states * each
        if block.fer is not None:
            # A failed frame is discarded before amplification, yielding no key
            # and leaking none, so the whole length scales, as on q.KeyBlock.
            total *= 1.0 - block.fer

        rows = (n0, n1, phi, kept, each, total)
        self._explain_block(info, rows, (counts, tests), share)

        # mdi_length clamps at zero and floors in Rust, so this is the same
        # agreeing clamp the asymptotic branch carries and not a second policy.
        return SwapResult(
            key_rate=max(0.0, total) / block.n,
            p_click=out[0],
            qber=out[1],
            y1=out[2],
            e1=out[4],
            q1=out[5],
            hom=None if modes is None else modes[1],
            key_length=total,
            n0=n0,
            n1=n1,
            phi=phi,
            n_key=kept,
            states=states,
            explain=info,
        )

    def _run_bell(self):
        va = self.alice.modulation.v_a
        beta = self.security.beta
        arms = (self._arm(0), self._arm(1))
        self._check_room(arms)
        (ta, wa), (tb, wb) = arms
        scale = self._refer()
        g, gp = self._corr(scale)
        # g, gp reach the bound through chi, not only the attack.
        chi = _core.cvmdi_noise(ta, tb, wa, wb, g, gp)
        floor = _core.cvmdi_floor(ta, tb)
        least = _core.cvmdi_least(ta, tb)
        key = _core.cvmdi_rate(ta, tb, chi)
        i_ab, chi_e, point = _core.cvmdi_point(va, ta, tb, wa, wb, g, gp, beta)
        attack = Attack(i_ab=i_ab, chi_e=chi_e, key_rate=point)
        info = self._explain_bell(arms, (chi, floor, least, key), attack, (g, gp, scale))
        block = getattr(self.security, "block", None)
        if block is not None:
            key = self._finite_bell(info, va, arms, (g, gp), block, beta)

        # The ONE clamp: explain()'s key_bound/key_raw and attack stay raw.
        return SwapResult(
            key_rate=max(0.0, key),
            chi=chi,
            floor=floor,
            least=least,
            attack=attack,
            explain=info,
        )

    def explain(self):
        """
        The resolved plan, every quantity labelled pinned or derived.
        """
        self._check()
        if isinstance(self.relay.bell, BellAnalyser):
            return self._run_qubit().explain

        return self._run_bell().explain

    def _attack(self):
        """
        The class of attack this topology's engine is proved against, quoted
        from src/mdi.rs's and src/relay.rs's own module headers. Neither states
        one -- relay.rs says what cvmdi_rate is a bound IN, the large-v_a limit
        at the Shannon limit, not what it is a bound AGAINST -- so both rows
        read unstated rather than guessing.
        """
        if isinstance(self.relay.bell, BellAnalyser):
            return "unstated: neither mdi_rate nor mdi_length names one"

        return "unstated: neither cvmdi_rate nor cvmdi_point names one"

    def _plan(self, kind):
        info = {
            "topology": "swap",
            "measurement": kind,
            "trust": "structural: an untrusted relay has no trusted variant",
            "security": type(self.security).__name__,
            "attack": self._attack(),
        }
        for i, tag in enumerate(("a", "b")):
            ch = self.channels[i]
            t, xi = self._fiber(i)
            label = "derived" if isinstance(ch, Fiber) and ch.T is None else "pinned"
            info[f"T_{tag}"] = _q(t, label)
            info[f"xi_{tag}"] = _q(xi, "pinned" if xi != 0.0 else "default")

        return info

    def _explain_qubit(self, arms, sets, modes, out):
        """
        The counting report, over the numbers _run_qubit already computed
        rather than a second evaluation. `modes` is _modes(), or None.
        """
        bell = self.relay.bell
        gain, qber, y_key, y_test, e1, q1, key, miss = out
        info = self._plan("bell (four threshold detectors)")
        info["protocol"] = "basis keying"
        info["detection"] = "click"
        info["stages"] = "closed-form only"
        for tag, one in zip("ab", sets):
            info[f"mu_{tag}"] = _q(one[0], "pinned")
            info[f"nu1_{tag}"] = _q(one[1], "pinned")
            info[f"nu2_{tag}"] = _q(one[2], "pinned")

        info["eta_relay"] = _q(bell.eta, "pinned")
        # PER DETECTOR AND PER GATE. The literature usually quotes the
        # receiver total, about twice this because four gates are read.
        info["dark"] = _q(bell.dark, "pinned")
        shown = "pinned" if modes is None else "derived"
        info["misalign"] = _q(bell.misalign, "pinned")
        info["misalign_test"] = _q(miss, shown)
        widths = "absent" if modes is None else "pinned"
        found = "absent" if modes is None else "derived"
        info["pulse_a"] = _q(self.alice.pulse, widths)
        info["pulse_b"] = _q(self.bob.pulse, widths)
        # The INTENSITY overlap of the two temporal modes, taken synchronised.
        info["overlap"] = _q(None if modes is None else modes[0], found)
        info["hom"] = _q(None if modes is None else modes[1], found)
        info["f"] = _q(self.security.f, "pinned")
        info["eta_a"] = _q(arms[0], "derived")
        info["eta_b"] = _q(arms[1], "derived")
        # No sifting factor multiplies this rate: the key-basis gain already
        # counts key-basis rounds only.
        info["sift"] = _q(None, "absent")
        info["gain"] = _q(gain, "derived")
        info["qber"] = _q(qber, "derived")
        # The BOUNDS, one per basis, never the forward model.
        info["y11_key"] = _q(y_key, "derived")
        info["y11_test"] = _q(y_test, "derived")
        info["e1"] = _q(e1, "derived")
        info["q1"] = _q(q1, "derived")
        # Clamped in the engine already, so it agrees with key_rate rather
        # than bracketing it, unlike the covariance branch's row of this name.
        info["key_raw"] = _q(key, "derived")

        return info

    def _explain_block(self, info, rows, grids, share):
        """
        The finite rows, appended to the counting report. `rows` is
        _run_length's (n0, n1, phi, n_key, per-state length, total); `grids` its
        two count grids; `share` the per-bound failure probability.
        """
        block = self.security.block
        states = self.relay.bell.states
        n0, n1, phi, kept, each, total = rows
        counts, tests = grids
        one, two = self.alice.modulation, self.bob.modulation
        # Renamed so the asymptotic rate cannot be read as the length beside it.
        info["key_asymptotic"] = info.pop("key_raw")
        info["n_block"] = _q(block.n, "pinned")
        # states is HOW MANY OF THE TWO CLASSES THE RELAY KEEPS and is what the
        # length is summed over; n_class divides by _CLASSES, which the optics
        # fix whatever the relay keeps. Never each other.
        info["states"] = _q(states, "pinned")
        info["n_class"] = _q(block.n / _CLASSES, "derived")
        info["bias_a"] = _q(one.bias, "pinned")
        info["bias_b"] = _q(two.bias, "pinned")
        for tag, mod in zip("ab", (one, two)):
            shown = "pinned" if mod.decoy.probs is not None else "default"
            info[f"weights_{tag}"] = _q(mod.decoy.weights, shown)

        info["eps_sec"] = _q(block.eps_sec, "pinned")
        # eps_sec = sum_k eps_{k,sec}, divided once in q.RelayBlock.per_state,
        # then split 266 ways inside the engine.
        info["eps_state"] = _q(block.per_state(states), "derived")
        info["eps_bound"] = _q(share, "derived")
        info["eps_cor"] = _q(block.eps_cor, "pinned")
        info["code"] = _q(block.code, "pinned")
        info["branch_key"] = _q(self._branch(counts, share), "derived")
        info["branch_test"] = _q(self._branch(tests, share), "derived")
        # PER ANNOUNCED BELL STATE, every one of them.
        info["n0"] = _q(n0, "derived")
        info["n1"] = _q(n1, "derived")
        info["phi"] = _q(phi, "derived")
        info["n_key"] = _q(kept, "derived")
        info["e_key"] = _q(info["qber"]["value"], "derived")
        info["fer"] = _q(block.fer, "pinned" if block.fer is not None else "absent")
        info["length_state"] = _q(each, "derived")
        info["key_length"] = _q(total, "derived")
        info["length_rule"] = (
            "sum over announced Bell states: l = states * length_state at "
            "eps_sec = states * eps_state, scaled by (1 - fer) where one is "
            "declared"
        )

        return info

    def _finite_bell(self, info, va, arms, corr, block, beta):
        """
        POP17's finite-size number: a RATE PER USE, not a key length, so
        nothing here divides by block.n as the qubit branch's mdi_length does.
        Returned unclamped.
        """
        (ta, wa), (tb, wb) = arms
        g, gp = corr
        vq, vp = _core.cvmdi_excess(ta, tb, wa, wb, g, gp)
        key, ta_low, tb_low, vq_up, vp_up, delta, n_key = _core.cvmdi_finite(
            va,
            ta,
            tb,
            vq,
            vp,
            beta,
            block.n,
            block.pe_fraction,
            block.sigma,
            block.eps_smooth,
            block.eps_pa,
            block.attack,
        )
        # sigma is a confidence COEFFICIENT in standard deviations, not a
        # probability: labelled pinned beside the two epsilons and never added
        # to them.
        for name, value in (
            ("n_block", block.n),
            ("pe_fraction", block.pe_fraction),
            ("sigma", block.sigma),
            ("eps_smooth", block.eps_smooth),
            ("eps_pa", block.eps_pa),
        ):
            info[name] = _q(value, "pinned")

        info["block_attack"] = block.attack
        for name, value in (
            ("v_q", vq),
            ("v_p", vp),
            ("tau_a_low", ta_low),
            ("tau_b_low", tb_low),
            ("v_q_up", vq_up),
            ("v_p_up", vp_up),
            ("delta", delta),
            ("n_key", n_key),
            ("key_finite", key),
        ):
            info[name] = _q(value, "derived")

        return key

    def _explain_bell(self, arms, noise, attack, corr):
        """
        The covariance report, over the numbers the engine was handed rather
        than a second evaluation. `corr` is _run_bell's (g, gp, scale).
        """
        bell = self.relay.bell
        g, gp, scale = corr
        (ta, wa), (tb, wb) = arms
        chi, floor, least, key = noise
        info = self._plan("bell (conjugate homodynes)")
        info["protocol"] = "gaussian modulation"
        info["v_a"] = _q(self.alice.modulation.v_a, "pinned")
        info["eta_relay"] = _q(bell.eta, "pinned")
        info["v_el"] = _q(bell.v_el, "pinned")
        info["beta"] = _q(self.security.beta, "pinned")
        # The arms AFTER the relay detector has been folded in.
        info["tau_a"] = _q(ta, "derived")
        info["tau_b"] = _q(tb, "derived")
        info["omega_a"] = _q(wa, "derived")
        info["omega_b"] = _q(wb, "derived")
        # Two planes: g_env is what the caller named, at the CHANNEL INPUT; g
        # is the same correlation on the folded arms, g_scale <= 1 apart.
        env = self.environment
        label = "pinned" if env is not None else "default"
        info["g_env"] = _q(env.x if env is not None else 0.0, label)
        info["gp_env"] = _q(env.p if env is not None else 0.0, label)
        info["g_scale"] = _q(scale, "derived")
        info["g"] = _q(g, "derived")
        info["gp"] = _q(gp, "derived")
        info["chi"] = _q(chi, "derived")
        # THREE DIFFERENT NUMBERS: the observed chi may legally sit below
        # chi_floor and never at or below chi_least.
        info["chi_floor"] = _q(floor, "derived")
        info["chi_least"] = _q(least, "derived")
        # cvmdi_rate is written at the Shannon limit, so the configured beta
        # reaches the attack rows below and never key_bound.
        info["reconciliation"] = _q("Shannon limit", "derived")
        # Both UNCLAMPED, unlike SwapResult.key_rate.
        info["key_bound"] = _q(key, "derived")
        info["key_raw"] = _q(key, "derived")
        info["i_ab"] = _q(attack.i_ab, "derived")
        info["chi_e"] = _q(attack.chi_e, "derived")
        info["key_attack"] = _q(attack.key_rate, "derived")

        return info


@dataclass(frozen=True)
class Node:
    """
    A vertex of a q.Network: a station that holds key IN THE CLEAR. There is no
    untrusted variant and no trusted= flag. It carries no hardware; every
    physical description lives on the edge.
    """


@dataclass(frozen=True)
class Hop:
    """
    One edge of a q.Network: a single segment of quantum security between two
    key-holding nodes, and the clock it runs at.
    """

    # Two DISTINCT node names. For a q.Link payload the order is
    # (transmitter, receiver); for a q.Swap it is (alice, bob), which is not
    # symmetric under a swap of the two.
    ends: tuple
    # A q.Link or a q.Swap, verbatim: run and multiplied by the clock, never
    # taken apart.
    link: object
    # Symbols per second on the quadrature families, emitted pulses per second
    # on the click ones, emitted PULSE PAIRS per second on a counting midpoint.
    # REQUIRED: bits per second is the only unit a path can be measured in.
    clock: float

    def __post_init__(self):
        if not isinstance(self.ends, tuple) or len(self.ends) != 2:
            raise ValueError("ends must be a tuple of two node names")

        if not all(isinstance(name, str) and name for name in self.ends):
            raise ValueError("node names must be non-empty strings")

        if self.ends[0] == self.ends[1]:
            raise ValueError(f"a hop joins two distinct nodes, not {self.ends[0]!r} to itself")

        std.finite("clock", self.clock)

        std.positive("clock", self.clock)


@dataclass(frozen=True)
class NodeReport:
    """
    One vertex as a completed run saw it. `total` sums the incident segments'
    bits per second: what the node could source or sink with no contention.
    """

    degree: int
    segments: tuple
    total: float


class Pairs(dict):
    """
    A mapping keyed by an unordered pair of node names: written in whichever
    order the hop declared, read either way round.
    """

    def __missing__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            flipped = (key[1], key[0])
            if dict.__contains__(self, flipped):
                return dict.__getitem__(self, flipped)

        raise KeyError(key)

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True

        return isinstance(key, tuple) and len(key) == 2 and (key[1], key[0]) in self.keys()


_NO_RATE = (
    "a network has no key rate. Per-segment rates are in res.segments and "
    "res.rates; a path number is res.route(a, b).bottleneck, which is a "
    "KEY-MANAGEMENT THROUGHPUT and not a security bound, because no quantum "
    "bound spans a trusted node"
)

_NO_PATH = (
    "a route has no key rate. route.bottleneck is the bits per second that "
    "hop-by-hop key relay can deliver over it, a KEY-MANAGEMENT THROUGHPUT "
    "and not a security bound: every node in route.trusts held the key in the "
    "clear, so no security proof spans this path"
)

# This module's own: attacks.py's and pairs.py's tuples differ and must, each
# barring the names its own result type invites.
_BARRED = ("key_rate", "key_raw", "key", "rate", "secure_rate")


@dataclass(frozen=True)
class Route(std.Barred):
    """
    One path across the key graph. bottleneck is the only number here and it is
    NOT a bound: see route.security, which is always populated.
    """

    # std.Barred's __getattr__ reads these two. Same tuple as NetworkResult's,
    # a DIFFERENT message: a route refuses because trusted nodes held the key,
    # a network because it has no single number at all.
    _barred = _BARRED
    _refusal = _NO_PATH

    # ((a, b), (b, c), ...) in traversal order.
    hops: tuple
    # The intermediate nodes, EVERY ONE of which holds the end-to-end key in
    # the clear. Empty only on a single-hop route, the one case where
    # bottleneck is also a quantum claim.
    trusts: tuple
    # Bits per second: the minimum over the hops, each hop having to supply one
    # local key bit per relayed bit.
    bottleneck: float
    # A sentence naming the nodes that must be trusted. Never None.
    security: str


@dataclass(frozen=True)
class NetworkResult(std.Barred):
    """
    What one q.Network run reports. There is deliberately NO key_rate here:
    a network does not have one, and asking raises with the reason.
    """

    # std.Barred's __getattr__ reads these two; the message differs from
    # Route's.
    _barred = _BARRED
    _refusal = _NO_RATE

    # {(a, b): LinkResult | SwapResult}, per-shot rates in each payload's own
    # unit. Read either way round.
    segments: Pairs
    # {(a, b): bits per second}, the only unit that compares across families.
    rates: Pairs
    nodes: dict
    explain: dict
    # {name: ((other, bits per second), ...)}, what route() walks.
    graph: dict

    def route(self, source, target):
        """
        The widest path from source to target: the one whose bottleneck is
        largest, ties broken by fewest hops and therefore fewest trusted nodes.
        """
        for name in (source, target):
            if name not in self.graph:
                known = ", ".join(sorted(self.graph))
                raise ValueError(f"no node named {name!r}; the network has {known}")

        if source == target:
            raise ValueError(f"a route joins two distinct nodes, not {source!r} to itself")

        best, prev = self._widest(source)
        if target not in prev:
            raise ValueError(
                f"no path from {source!r} to {target!r}: the key graph is "
                "disconnected between them. An untrusted swap station is never "
                "a vertex, so no route reaches one or passes through one"
            )

        walk = [target]
        while walk[-1] != source:
            walk.append(prev[walk[-1]])

        walk.reverse()
        hops = tuple(zip(walk, walk[1:]))
        trusts = tuple(walk[1:-1])
        note = "single hop: no intermediate node holds this key"
        if trusts:
            who = ", ".join(trusts)
            note = f"trusted-node key relay through {who}; a key-management " "throughput, not a quantum bound"

        return Route(
            hops=hops,
            trusts=trusts,
            bottleneck=best[target][0],
            security=note,
        )

    def _widest(self, source):
        # Label-setting on (bottleneck, -hops) rather than on a summed cost:
        # key relay is limited by its narrowest hop.
        best = {source: (math.inf, 0)}
        prev = {}
        seen = set()
        while True:
            live = [name for name in best if name not in seen]
            if not live:
                return best, prev

            # max() keeps the FIRST maximal label, which is the tie-break the
            # relaxation below assumes: it replaces only on a strict gain.
            pick = max(live, key=lambda name: _rank(best[name]))
            seen.add(pick)
            cap, depth = best[pick]
            for other, rate in self.graph[pick]:
                step = (min(cap, rate), depth + 1)
                if other not in best or _rank(step) > _rank(best[other]):
                    best[other] = step
                    prev[other] = pick


def _rank(step):
    """
    How one (bottleneck, hops) label orders against another: more key first,
    then fewer trusted nodes.
    """

    return step[0], -step[1]


class Network:
    """
    A graph of key-holding nodes. Every vertex is a q.Node and holds key in
    the clear; every edge is one q.Hop carrying one segment of quantum
    security, either a q.Link or a q.Swap.

    AN UNTRUSTED SWAP STATION IS NOT A VERTEX. It is interior to the edge that
    carries it, so no path can pass through one and untrusted stations cannot
    be composed -- a shape the data structure has no way to express rather than
    a rule checked at run(). Chaining two swaps into end-to-end security needs
    quantum memory or a repeater, and neither is implemented.

    So any multi-hop number this class produces is a KEY-MANAGEMENT THROUGHPUT
    over a path on which every intermediate node held the key in the clear,
    never a quantum bound. There is no key_rate attribute on a NetworkResult or
    on a Route, and asking for one raises with that sentence.
    """

    def __init__(self, nodes, edges):
        # nodes is {name: q.Node}; the key IS the name, so q.Node carries none.
        self.nodes = dict(nodes)
        # Not a dict keyed by pair: a parallel clocks= mapping would be two
        # structures that have to agree.
        self.edges = tuple(edges)

    def _check(self):
        if not self.nodes:
            raise ValueError("a network needs at least one q.Node")

        for name, node in self.nodes.items():
            if isinstance(node, Relay):
                raise ValueError(
                    f"{name!r} is a q.Relay, which is not a vertex: an "
                    "untrusted station holds no key, so nothing can be routed "
                    "through it. Put it inside a q.Swap and give that swap to "
                    "a q.Hop"
                )

            if not isinstance(node, Node):
                raise ValueError(f"{name!r} must be a q.Node")

        if not self.edges:
            raise ValueError("a network needs at least one q.Hop")

        self._check_edges()
        self._check_bare()

    def _check_edges(self):
        seen = set()
        for hop in self.edges:
            if not isinstance(hop, Hop):
                raise ValueError("each edge must be a q.Hop")

            for name in hop.ends:
                if name not in self.nodes:
                    known = ", ".join(sorted(self.nodes))
                    raise ValueError(f"hop {hop.ends} names {name!r}, which is not a node; " f"the network has {known}")

            key = frozenset(hop.ends)
            if key in seen:
                raise ValueError(
                    f"two hops join {hop.ends[0]!r} and {hop.ends[1]!r}: "
                    "parallel segments between one pair are not modelled, their "
                    "key pools adding and nothing here tracking that"
                )

            seen.add(key)
            if not isinstance(hop.link, (Swap, _link_type())):
                raise ValueError("a hop carries a q.Link or a q.Swap, not " f"{type(hop.link).__name__}")

    def _check_bare(self):
        joined = set()
        for hop in self.edges:
            joined.update(hop.ends)

        alone = sorted(set(self.nodes) - joined)
        if alone:
            names = ", ".join(repr(name) for name in alone)
            raise ValueError(
                f"{names} sit on no hop: an isolated node distributes no key, " "so it is a typo rather than a topology"
            )

    def run(self, symbols=1_000_000, seed=0):
        """
        Run every segment and report them in bits per second. symbols and seed
        reach the q.Link payloads alone; a q.Swap is a closed form and takes
        neither.
        """
        self._check()
        segments = Pairs()
        rates = Pairs()
        for hop in self.edges:
            out = self._segment(hop, symbols, seed)
            segments[hop.ends] = out
            rates[hop.ends] = hop.clock * out.key_rate

        graph = self._graph(rates)

        return NetworkResult(
            segments=segments,
            rates=rates,
            nodes=self._report(rates),
            explain=self._plan(rates),
            graph=graph,
        )

    def _segment(self, hop, symbols, seed):
        # Computed by the payload alone: a q.Network never reports a number a
        # standalone q.Link or q.Swap would not have reported identically.
        try:
            if isinstance(hop.link, Swap):
                return hop.link.run()

            return hop.link.run(symbols=symbols, seed=seed)
        except (ValueError, NotImplementedError) as exc:
            raise type(exc)(f"hop {hop.ends}: {exc}") from exc

    def _graph(self, rates):
        out = {name: [] for name in self.nodes}
        for hop in self.edges:
            first, second = hop.ends
            rate = rates[hop.ends]
            out[first].append((second, rate))
            out[second].append((first, rate))

        return {name: tuple(edges) for name, edges in out.items()}

    def _report(self, rates):
        held = {name: [] for name in self.nodes}
        for hop in self.edges:
            for name in hop.ends:
                held[name].append(hop.ends)

        return {
            name: NodeReport(
                degree=len(pairs),
                segments=tuple(pairs),
                total=sum(rates[pair] for pair in pairs),
            )
            for name, pairs in held.items()
        }

    def _plan(self, rates=None):
        shown = "derived" if rates is not None else "derived (run to compute)"
        info = {
            "topology": "network",
            "nodes": _q(len(self.nodes), "pinned"),
            "hops": _q(len(self.edges), "pinned"),
            "trust": _q(
                "every q.Node holds key in the clear; an untrusted swap "
                "station is interior to a hop and is never a vertex",
                "default",
            ),
            "units": _q(
                "bits per second on every hop; per-shot rates, in each " "family's own unit, stay in res.segments",
                "default",
            ),
            "end_to_end": _q(
                "none. A multi-hop number is res.route(a, b).bottleneck, a "
                "key-management throughput over trusted nodes, never a bound",
                "default",
            ),
            "contention": _q(
                "not modelled: two routes sharing a node cannot both reach " "their bottleneck at once",
                "default",
            ),
        }
        for hop in self.edges:
            tag = f"{hop.ends[0]}_{hop.ends[1]}"
            info[f"kind_{tag}"] = _q(type(hop.link).__name__, "pinned")
            info[f"clock_{tag}"] = _q(hop.clock, "pinned")
            info[f"rate_{tag}"] = _q(None if rates is None else rates[hop.ends], shown)

        return info

    def explain(self):
        """
        The resolved plan, emitting no symbols; a rate only a run could fill
        reads "derived (run to compute)".
        """
        self._check()

        return self._plan()


def _link_type():
    # Imported late so qkd.topology does not drag link.py in for callers who
    # only want a swap.
    from .link import Link

    return Link
