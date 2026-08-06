import heapq
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

from kit.forms import h2
from qkd import _core, impairments, reconcile

# Jouguet, Kunz-Jacques & Leverrier, PRA 84, 062317 (2011), arXiv:1110.0100,
# Sec. III: the rate-1/2 multi-edge code of Richardson & Urbanke's Table VI.
# "The SNR threshold given by Discretized Density Evolution is s* = 1.074. The
# corresponding efficiency on the BIAWGNC is 98.2%."
MET_HALF = 1.074
MET_BETA = 0.982

# Leverrier, Alleaume, Boutros, Zemor & Grangier, PRA 77, 042325 (2008),
# arXiv:0712.3823, Sec. VI: "The reconciliation based on rotations in R^8 uses
# a LDPC code of rate 0.26", at a SNR "typically around 0.5", reaching
# efficiencies "around 90%". A consistency check, not a pinned anchor.
SO8_RATE = 0.26
SO8_SNR = 0.5

# FROZEN MAQAN Tier B point from test/network.py's test_secure_rate_miss, 2026-09-03, before Fig. 4 was
# re-read 20.2 -> 20.75 kbit/s; the classical share is a ratio the endpoint cancels from. 2.0 dB ERNET-SETS
# route; MAQAN_BEST is MAQAN Table 1. Not imported, so a COW engine change moves network.py alone.
MAQAN_GROSS = 39642.7
MAQAN_SIFT = 71075.6
MAQAN_PULSE = 3.51609e6
MAQAN_QBER = 0.0200428
MAQAN_BEST = 1.0e3

# FEC_TOL = 3 standard errors of the mean f_EC, 0.7% each, over 32 frames of 2^14 bits at Q = 2%.
FRAMES = 32
FEC_TOL = 0.021

# q.KeyBlock defaults; the exams pin two accountings agreeing, at any budget.
EPS_SEC = 1e-10
EPS_COR = 1e-15


def simulated(qber, frames=FRAMES, seed=11, cfg=None):
    """
    Measured f_EC from run_cascade: disclosed bits over n*h(Q).
    """
    cfg = cfg or reconcile.Cascade()
    leak, failed = reconcile.run_cascade(qber, frames=frames, seed=seed, cfg=cfg)

    # kit.forms.h2 is 0.0 at and outside the endpoints: a qber of 0 or 1 divides by zero here.
    return leak / (cfg.frame * h2(qber)), failed


def frame_draws(qber, frame, passes, rng):
    """
    One frame's draws: the error pattern, then one permutation per pass, the first unshuffled.
    """
    err = [1 if rng.random() < qber else 0 for _ in range(frame)]
    order = list(range(frame))
    out = []
    for step in range(passes):
        if step:
            rng.shuffle(order)

        out.append(list(order))

    return err, out


def cascade_ref(err, orders, sizes, reuse):
    """
    Reference Cascade on supplied draws: (disclosed parities, residual errors), as
    `_core.cascade_replay`.
    """
    err = list(err)
    frame = len(err)
    held = []
    member = [[] for _ in range(frame)]
    odd = []
    leak = 0

    def parity(idx):
        bit = 0
        for i in idx:
            bit ^= err[i]

        return bit

    def register(idx, bit):
        held.append([idx, bit])
        for i in idx:
            member[i].append(len(held) - 1)
        if bit:
            heapq.heappush(odd, (len(idx), len(held) - 1))

    def correct(bid):
        nonlocal leak
        idx = held[bid][0]
        while len(idx) > 1:
            cut = len(idx) // 2
            left, right = idx[:cut], idx[cut:]
            bit = parity(left)
            leak += 1
            if reuse:
                register(left, bit)
                register(right, bit ^ held[bid][1])
            idx = left if bit else right

        spot = idx[0]
        err[spot] ^= 1
        for b in member[spot]:
            held[b][1] ^= 1
            if held[b][1]:
                heapq.heappush(odd, (len(held[b][0]), b))

    def drain():
        while odd:
            size, bid = heapq.heappop(odd)
            if held[bid][1] and size == len(held[bid][0]):
                correct(bid)

    for step, size in enumerate(sizes):
        order = orders[step]
        chunks = [tuple(order[i : i + size]) for i in range(0, frame, size)]
        leak += max(0, len(chunks) - (1 if step else 0))
        for chunk in chunks:
            register(chunk, parity(chunk))

        drain()

    return leak, sum(err)


class Efficiency(Question):
    """
    Capacities, the three efficiency quantities, and the published code tables.
    """

    def test_awgn_capacity(self):
        """
        `awgn_capacity` is $\\frac{1}{2}\\log_2(1 + \\mathrm{SNR})$, the I(A:B) of Leverrier
        arXiv:0712.3823 Sec. VI.
        """
        for snr in (0.02, 0.5, 1.0, 3.0, 100.0):
            want = 0.5 * math.log2(1.0 + snr)
            self.assertClose(
                reconcile.awgn_capacity(snr),
                want,
                atol=1e-15,
                msg=f"capacity at SNR {snr}",
            )

    def test_biawgn_bounds(self):
        """
        The binary-input capacity sits under the Gaussian-input one, tending to 1 bit at high
        SNR and to the Gaussian-input value at low SNR.
        """
        for snr in (0.02, 0.161, 1.074, 5.0, 20.0):
            binary = reconcile.biawgn_capacity(snr)
            self.assertLess(
                binary,
                reconcile.awgn_capacity(snr),
                msg=f"SNR {snr}: binary {binary}",
            )
            self.assertGreater(binary, 0.0, msg=f"SNR {snr}: binary {binary}")
        self.assertClose(reconcile.biawgn_capacity(1000.0), 1.0, atol=1e-9, msg="capacity at SNR 1000")

        low = reconcile.biawgn_capacity(0.001) / reconcile.awgn_capacity(0.001)

        self.assertClose(low, 1.0, atol=1e-3, msg=f"low-SNR ratio {low:.6f}")

    def test_binary_anchor(self):
        """
        Published anchor: the rate-1/2 multi-edge code's 98.2% at s* = 1.074 is 0.9817 of the
        binary-input capacity and 0.9502 of the Gaussian-input one, so every published beta here
        is against the BIAWGNC.
        """
        binary = reconcile.efficiency(0.5, MET_HALF)
        gauss = reconcile.efficiency(0.5, MET_HALF, binary=False)

        self.assertClose(binary, MET_BETA, atol=6e-4, msg=f"binary-input beta {binary:.5f}")
        self.assertGreater(MET_BETA - gauss, 0.03, msg=f"Gaussian-input beta {gauss:.5f}")

    def test_asymptotic_table(self):
        """
        Published anchor: Jouguet Table I's 95.9%, 97.2% and 98.1% for the rate 0.1, 0.05 and
        0.02 multi-edge codes, as R over the binary-input capacity at the quoted threshold.
        """
        for code in reconcile.codes(dim=None):
            if code.rate == 0.5:
                continue

            got = reconcile.code_beta(code, code.snr)
            self.assertClose(
                got,
                code.beta,
                atol=0.012,
                msg=f"{code.name}: {got:.4f} against {code.beta}",
            )

    def test_rotated_table(self):
        """
        Published anchor: all fifteen rows of Jouguet Tables II and III, block 2^20 at d = 0, 1,
        2, 4 and 8, worst 0.0070 at rate 0.02, whose threshold is printed to two decimals.
        """
        worst = 0.0
        for code in reconcile.CODES:
            if code.dim is None:
                continue

            got = reconcile.code_beta(code, code.snr)
            worst = max(worst, abs(got - code.beta))
            self.assertClose(
                got,
                code.beta,
                atol=0.008,
                msg=f"{code.name}: {got:.4f} against {code.beta}",
            )
        self.assertClose(worst, 0.0070, atol=5e-4, msg=f"worst row {worst:.5f}")

    def test_dimension_gain(self):
        """
        Efficiency rises with rotation dimension at every code rate, over the dimensions 1, 2,
        4, 8 Leverrier's Theorem 1 allows.
        """
        dims = sorted({c.dim for c in reconcile.CODES if c.dim})

        self.assertEqual(dims, [1, 2, 4, 8], msg=f"dims {dims}")

        for rate in (0.1, 0.05, 0.02):
            betas = [c.beta for c in reconcile.codes(rate=rate) if c.dim is not None]
            self.assertMonotone(betas, strict=False, msg=f"rate {rate}: {betas}")

    def test_slice_comparison(self):
        """
        Consistency check, not an anchor: Leverrier's rate-0.26 code at SNR about 0.5 lands
        within 0.02 of the 90% he quotes, both given with "around".
        """
        got = reconcile.efficiency(SO8_RATE, SO8_SNR)

        self.assertClose(got, 0.90, atol=0.02, msg=f"SO(8) efficiency {got:.4f}")

    def test_three_quantities(self):
        """
        At rate 0.8 over a BSC at 2%, f_EC is 1.41402 and beta 0.93180, on opposite sides of
        one.
        """
        rate, qber = 0.8, 0.02
        f = reconcile.inefficiency(rate, qber)
        beta = reconcile.capacity_ratio(rate, qber)

        self.assertClose(f, 1.41402, atol=1e-5, msg=f"f_EC {f:.5f}")
        self.assertClose(beta, 0.93180, atol=1e-5, msg=f"beta {beta:.5f}")
        self.assertGreater(f, 1.0, msg=f"f_EC {f}")
        self.assertLess(beta, 1.0, msg=f"beta {beta}")

    def test_bridge_identity(self):
        """
        Published anchor: Martinez-Mateo Eq. (4) turns Table 3's f_EC column into its beta
        column in all twelve rows to 5e-5, the rounding of four printed decimals.
        """
        for row in reconcile.CASCADE_TABLE:
            qber, f, beta = row[0], row[5], row[7]
            got = reconcile.bridge(qber, f_ec=f)
            self.assertClose(got, beta, atol=5e-5, msg=f"Q={qber}: {got:.6f} vs {beta}")

            back = reconcile.bridge(qber, beta=got)
            self.assertClose(back, f, atol=1e-9, msg=f"Q={qber}: round trip {back:.6f}")

    def test_charged_table(self):
        """
        Published anchor: Martinez-Mateo Eq. (6) turns Table 3's f_EC and frame error rate
        columns into its eta_EC column in all twelve rows to 3e-5.
        """
        for row in reconcile.CASCADE_TABLE:
            qber, eta, f, fer = row[0], row[4], row[5], row[6]
            got = reconcile.charged(f, qber, fer)
            self.assertClose(got, eta, atol=3e-5, msg=f"Q={qber}: {got:.6f} vs {eta}")
            self.assertGreater(got, f, msg=f"Q={qber}: eta {got}, f {f}")

    def test_leak_ratio(self):
        """
        Eq. (5)'s leakage runs from 1 - R at no frame errors to the whole frame when every frame
        fails.
        """
        self.assertClose(reconcile.leak_ratio(0.7, 0.0), 0.3, atol=1e-15, msg="no failures")
        self.assertClose(reconcile.leak_ratio(0.7, 1.0), 1.0, atol=1e-15, msg="every frame fails")

    def test_code_penalty(self):
        """
        A fixed-rate code's efficiency falls as SNR rises past its threshold, so the published
        beta is the largest it can claim.
        """
        code = reconcile.pick(0.161, dim=8)
        seen = [reconcile.code_beta(code, s) for s in (0.161, 0.2, 0.4, 1.0)]

        self.assertMonotone(seen, rising=False, msg=f"beta: {seen}")
        self.assertClose(seen[0], code.beta, atol=0.008, msg=f"beta at threshold {seen[0]}")

    def test_pick_ladder(self):
        """
        `pick()` returns higher-rate codes as SNR rises, each decoding at that SNR.
        """
        rates = [reconcile.pick(s, dim=8).rate for s in (0.03, 0.08, 0.2)]

        self.assertMonotone(rates, strict=False, msg=f"rates: {rates}")

        for snr in (0.03, 0.08, 0.2):
            code = reconcile.pick(snr, dim=8)
            self.assertLessEqual(code.snr, snr, msg=f"{code.name}: threshold {code.snr}, SNR {snr}")


class CascadeRun(Question):
    """
    Cascade block-size rules, the published parameter table and the simulation.
    """

    def test_block_rules(self):
        """
        Martinez-Mateo's near-optimal rule reproduces the first three block sizes of nine of
        Table 3's twelve rows.
        """
        agree = 0
        for row in reconcile.CASCADE_TABLE:
            got = reconcile.blocks(row[0], 1 << 14, "nearoptimal")[:3]
            if tuple(got) == row[1:4]:
                agree += 1
        self.assertEqual(agree, 9, msg=f"agree {agree} of 12")

    def test_cascade_misfits(self):
        """
        In the three rows it misses, the closed form is one power-of-two step too large: k2 at
        1% and 4%, k1 at 8%.
        """
        for qber, pair in reconcile.CASCADE_MISFITS.items():
            got = reconcile.blocks(qber, 1 << 14, "nearoptimal")[:2]
            self.assertNotEqual(tuple(got), pair, msg=f"Q={qber}: {got}")
            self.assertEqual(
                got[0] * got[1] % (pair[0] * pair[1]),
                0,
                msg=f"Q={qber}: {got} vs {pair}",
            )

    def test_variant_sizes(self):
        """
        Each variant's first block size follows its own rule: 0.73/Q doubling for the original,
        0.8/Q with a fivefold second block for Yan, a power of two for the two optimised sets.
        """
        orig = reconcile.blocks(0.02, 10**4, "original")

        self.assertEqual(orig[0], math.ceil(0.73 / 0.02), msg=f"original k1 {orig[0]}")
        self.assertEqual(orig[1], 2 * orig[0], msg=f"original: {orig}")
        self.assertEqual(len(orig), 4, msg=f"original passes {len(orig)}")

        yan = reconcile.blocks(0.02, 10**4, "yan")

        self.assertEqual(yan[1], 5 * yan[0], msg=f"yan: {yan[:2]}")

        two = reconcile.blocks(0.02, 10**4, "poweroftwo")[0]

        self.assertEqual(two & (two - 1), 0, msg=f"k1 {two}")

    def test_table_anchor(self):
        """
        `cascade_point` returns Martinez-Mateo Table 3 verbatim, 1.04006 inefficiency and 9.3e-5
        frame error rate at 2%.
        """
        k1, k2, k3, eta, f, fer, beta, rounds = reconcile.cascade_point(0.02)

        self.assertEqual((k1, k2, k3), (64, 512, 4096), msg="block sizes at 2%")
        self.assertClose(f, 1.04006, atol=1e-9, msg=f"f_EC {f}")
        self.assertClose(fer, 9.3e-5, atol=1e-12, msg=f"frame error rate {fer}")
        self.assertClose(rounds, 407.6, atol=1e-9, msg=f"channel uses {rounds}")

    def test_replay_identity(self):
        """
        On one error pattern and pass permutations the Rust Cascade discloses the same parities
        and leaves the same residual errors as `cascade_ref`, over four variants and both
        subblock-reuse settings.
        """
        cases = (
            (0.02, 1 << 10, "nearoptimal"),
            (0.05, 1 << 9, "poweroftwo"),
            (0.03, 1 << 9, "yan"),
            (0.02, 1000, "original"),
        )
        for qber, frame, variant in cases:
            sizes = reconcile.blocks(qber, frame, variant)
            for reuse in (True, False):
                err, orders = frame_draws(qber, frame, len(sizes), random.Random(7))
                flat = [i for order in orders for i in order]
                want = cascade_ref(err, orders, sizes, reuse)
                got = _core.cascade_replay(err, flat, list(sizes), reuse)

                self.assertEqual(got, want, msg=f"Q={qber} {variant} reuse={reuse}: {got} against {want}")

    def test_simulated_fec(self):
        """
        Published anchor: counted Cascade parities over 32 frames of 2^14 bits reproduce Table
        3's measured f_EC, 1.04006 at 2% and 1.04313 at 5%.
        """
        for qber in (0.02, 0.05):
            got, failed = simulated(qber)
            want = reconcile.cascade_point(qber)[4]
            self.assertClose(got, want, atol=FEC_TOL, msg=f"Q={qber}: {got:.5f} against {want}")
            self.assertEqual(failed, 0, msg=f"Q={qber}: {failed} frames unreconciled")

    def test_frame_bound(self):
        """
        DECLARED GAP. Zero failures in 32 frames bound the frame error rate at 0.0894 with 95%
        confidence, 961 times Martinez-Mateo's 9.3e-5, so the simulation's count cannot stand in
        for a correctness parameter.
        """
        limit = reconcile.frame_bound(FRAMES, 0)

        self.assertClose(limit, 0.089368, atol=1e-6, msg=f"limit {limit:.6f}")
        self.assertClose(
            limit,
            1.0 - 0.05 ** (1.0 / FRAMES),
            atol=1e-12,
            msg=f"limit {limit}",
        )

        ratio = limit / reconcile.cascade_point(0.02)[5]

        self.assertClose(ratio, 961.0, atol=1.0, msg=f"{ratio:.1f} times the measured rate")
        self.assertMonotone(
            [reconcile.frame_bound(n, 0) for n in (8, 32, 1024)],
            rising=False,
            msg="frame_bound at 8, 32, 1024 frames",
        )

    def test_reuse_gain(self):
        """
        Without subblock reuse the same block sizes and fourteen passes leak a tenth more,
        taking Cascade from 1.04 to about 1.13.
        """
        plain = reconcile.Cascade(reuse=False)
        kept, _ = simulated(0.02, frames=8)
        dropped, _ = simulated(0.02, frames=8, cfg=plain)

        self.assertGreater(dropped, kept, msg=f"dropped {dropped:.4f}, kept {kept:.4f}")
        self.assertGreater(dropped, 1.08, msg=f"dropped {dropped:.4f}")

    def test_leak_floor(self):
        """
        Cascade's disclosed parities exceed the Slepian-Wolf minimum $n h(Q)$, the measured f_EC
        staying above one.
        """
        for qber in (0.01, 0.05):
            got, _ = simulated(qber, frames=4)
            self.assertGreater(got, 1.0, msg=f"Q={qber}: measured f_EC {got:.5f}")


class Hashing(Question):
    """
    Privacy amplification, Toeplitz seeds and the authentication key bill.
    """

    def test_hash_lemma(self):
        """
        The leftover hash lemma extracts $\\lfloor H_{min} - 2\\log_2(1/(2\\epsilon))\\rfloor$
        bits, within $\\epsilon$ of uniform at that length.
        """
        ell = reconcile.hash_length(1000.0, 1e-10)

        self.assertEqual(ell, 935, msg=f"extracted length {ell}")

        gap = reconcile.hash_distance(ell, 1000.0)

        self.assertLess(gap, 1e-10, msg=f"distance from uniform {gap:.4g}")

        exact = reconcile.hash_distance(1000.0 - reconcile.amplify_cost(1e-10), 1000.0)

        self.assertClose(exact, 1e-10, atol=1e-22, msg=f"distance at the toll {exact:.6g}")

    def test_amplify_toll(self):
        """
        Privacy amplification costs 64.44 bits at eps = 1e-10 at every block length, so block
        cadence alone sets its key-rate share.
        """
        toll = reconcile.amplify_cost(1e-10)

        self.assertClose(toll, 64.4386, atol=1e-4, msg=f"toll {toll:.4f}")

        for h_min in (1e3, 1e6, 1e9):
            spent = h_min - reconcile.hash_length(h_min, 1e-10)
            self.assertLess(spent, 66.0, msg=f"H_min {h_min}: spent {spent:.3f}")

    def test_engine_toll(self):
        """
        `mdi_length`, `pairing_length` and the delta of `sixstate_length` and `cv_finite`
        rebuild bit for bit from `hash_charge` at each engine's own share, the Gaussian one two
        bits higher.
        """
        s0, s1, phi = 900.0, 4.0e4, 0.03
        n_key, e_key, f_ec = 1.0e5, 0.02, 1.16
        head = s0 + s1 * (1.0 - h2(phi)) - f_ec * n_key * h2(e_key)
        eps = _core.mdi_eps(EPS_SEC)
        budget = math.log2(8.0 / EPS_COR) + 2.0 * math.log2(2.0 / eps**2) + reconcile.hash_charge("mdi", eps)
        got = _core.mdi_length(s0, s1, phi, n_key, e_key, f_ec, EPS_SEC, EPS_COR)

        self.assertEqual(got, math.floor(head - budget), msg=f"mdi_length {got} rebuilt")

        eps = _core.pairing_eps(EPS_SEC)
        budget = math.log2(2.0 / EPS_COR) + 2.0 * math.log2(2.0 / eps**2) + reconcile.hash_charge("pairing", eps)
        got = _core.pairing_length(s0, s1, phi, n_key, e_key, f_ec, EPS_SEC, EPS_COR)

        self.assertEqual(got, math.floor(head - budget), msg=f"pairing_length {got} rebuilt")

        slot = _core.sixstate_eps(EPS_SEC, EPS_COR)[0]
        delta = _core.sixstate_length(1.0e6, 1.0e5, 0.02, 0.02, 1.2, EPS_SEC, EPS_COR)[4]
        want = 7.0 * math.sqrt(1.0e6 * math.log2(2.0 / slot)) + reconcile.hash_charge("sixstate", slot)

        self.assertEqual(delta, want, msg=f"sixstate delta {delta} rebuilt")

        delta = _core.cv_finite(
            4.0,
            0.5,
            0.01,
            0.6,
            0.05,
            0.95,
            True,
            False,
            1.0e10,
            0.1,
            EPS_SEC,
            EPS_SEC,
            EPS_SEC,
        )[5]
        want = 7.0 * math.sqrt(math.log2(2.0 / EPS_SEC) / 9.0e9) + reconcile.hash_charge("cv", EPS_SEC) / 9.0e9

        self.assertEqual(delta, want, msg=f"cv_finite delta {delta} rebuilt")

    def test_auth_epsilon(self):
        """
        The 154 bits `auth_cost` bills carry a forgery probability of 1.79856e-10, under the
        3e-10 of three messages at 1e-10, `tag_length` rounding each tag up to a whole bit.
        """
        lengths = (1e6, 2e4, 3e4)
        got = reconcile.auth_failure(lengths, 1e-10)
        tags = [reconcile.tag_length(n, 1e-10) for n in lengths]
        want = sum(reconcile.tag_failure(n, k) for n, k in zip(lengths, tags))

        self.assertClose(got, 1.79856e-10, atol=1e-15, msg=f"round eps {got:.6g}")
        self.assertEqual(got, want, msg=f"auth_failure {got}, tags {want}")
        self.assertLess(got, 3.0e-10, msg=f"auth_failure {got:.6g}")

    def test_verify_link(self):
        """
        eps_cor is frame error rate times hash collision probability: Martinez-Mateo's 9.3e-5 at
        2% over a 64-bit tag gives 5.0416e-24, four decades under the collision probability
        alone.
        """
        fer = reconcile.cascade_point(0.02)[5]
        got = reconcile.verify_failure(fer, 2.0**-64)

        self.assertClose(got, 5.0416e-24, atol=1e-28, msg=f"eps_cor {got:.6g}")
        self.assertLess(got, 2.0**-64, msg=f"eps_cor {got:.6g}")

    def test_toeplitz_seed(self):
        """
        A plain Toeplitz matrix needs n + l - 1 seed bits and the modified $(T \\mid I)$ family
        n - 1.
        """
        self.assertClose(reconcile.toeplitz_seed(1e4, 5e3), 14999.0, atol=1e-9, msg="plain seed")
        self.assertClose(
            reconcile.toeplitz_seed(1e4, 5e3, True),
            9999.0,
            atol=1e-9,
            msg="modified seed",
        )

    def test_poly_collision(self):
        """
        The polynomial family is $\\delta$-almost two-universal, $\\delta = (r - 1)/\\lvert
        F\\rvert$, so not two-universal, and growing with the message chunks.
        """
        got = reconcile.poly_collision(1024, 64)

        self.assertClose(got, 1023.0 / 2.0**64, atol=1e-30, msg=f"delta {got:.6g}")
        self.assertGreater(got, 2.0**-64, msg=f"delta {got:.6g}")

    def test_tag_round(self):
        """
        `tag_length` inverts Krawczyk's $n 2^{-k+1}$: the tag meets the forgery probability and
        one bit shorter does not.
        """
        for message, eps in ((1e4, 1e-9), (1e6, 1e-10), (1e9, 1e-12)):
            tag = reconcile.tag_length(message, eps)
            self.assertLessEqual(
                reconcile.tag_failure(message, tag),
                eps,
                msg=f"tag {tag} misses eps {eps}",
            )
            self.assertGreater(
                reconcile.tag_failure(message, tag - 1),
                eps,
                msg=f"tag {tag - 1} meets eps {eps}",
            )

    def test_lfsr_compare(self):
        """
        DECLARED CORRECTION. At a seed twice the tag length Krawczyk's later bound sits a factor
        $2n/(n + k)$ BELOW his Theorem 9 one, not above as Fung, Ma & Chau state, and the costed
        tag uses the larger.
        """
        later = reconcile.lfsr_failure(1e6, 50, 100)
        costed = reconcile.tag_failure(1e6, 50)

        self.assertLess(later, costed, msg=f"{later:.4g} against {costed:.4g}")
        self.assertClose(
            costed / later,
            2.0e6 / (1.0e6 + 50.0),
            atol=1e-6,
            msg=f"ratio {costed / later:.6f}",
        )

    def test_auth_recycle(self):
        """
        A one-time-padded tag reuses the 2k-bit hash seed, so a round costs k bits per message,
        not 3k.
        """
        lengths = (1e6, 2e4, 3e4)
        thrifty = reconcile.auth_cost(lengths, 1e-10)
        fresh = reconcile.auth_cost(lengths, 1e-10, recycle=False)

        self.assertClose(fresh, 3.0 * thrifty, atol=1e-9, msg=f"{fresh} vs {thrifty}")
        self.assertClose(thrifty, 154.0, atol=1e-9, msg=f"recycled cost {thrifty}")

    def test_net_length(self):
        """
        Fung, Ma & Chau's net key length subtracts every secret bit the round spent and clamps
        at zero.
        """
        costs = {
            "amplify": 64.44,
            "authenticate": 154.0,
        }

        self.assertClose(
            reconcile.net_length(1000.0, costs),
            781.56,
            atol=1e-9,
            msg="net key length",
        )
        self.assertClose(reconcile.net_length(100.0, costs), 0.0, atol=1e-15, msg="clamped at zero")


class Delivery(Question):
    """
    What the classical layer costs a deployed link, at the MAQAN operating point.
    """

    def test_cadence_toll(self):
        """
        At the MAQAN operating point hashing and authentication cost 1.84% of gross key on 300
        ms blocks and 10.6% on 50 ms blocks, being per block, not per bit.
        """
        seen = []
        for tau in (0.05, 0.3, 1.0):
            costs = reconcile.round_cost(MAQAN_SIFT * tau, MAQAN_GROSS * tau, MAQAN_PULSE * tau)
            seen.append(sum(costs.values()) / (MAQAN_GROSS * tau))
        self.assertClose(seen[0], 0.10617, atol=1e-4, msg=f"50 ms toll {seen[0]:.5f}")
        self.assertClose(seen[1], 0.01837, atol=1e-4, msg=f"300 ms toll {seen[1]:.5f}")
        self.assertMonotone(seen, rising=False, msg=f"toll: {seen}")

    def test_block_floor(self):
        """
        Below 370 sifted bits a block at this yield cannot pay its own hashing and
        authentication and returns no key.
        """
        per = MAQAN_GROSS / MAQAN_SIFT
        floor = reconcile.block_floor(per, MAQAN_PULSE * 0.3)

        self.assertClose(floor, 370.1, atol=0.5, msg=f"floor {floor:.2f} sifted bits")

        below = reconcile.net_rate(MAQAN_GROSS, floor * 0.9, MAQAN_SIFT / (floor * 0.9), MAQAN_PULSE * 0.3)

        self.assertClose(below, 0.0, atol=1e-12, msg=f"below the floor: {below:.4g}")

    def test_maqan_share(self):
        """
        DECLARED MISS. The classical layer closes almost none of test_secure_rate_miss's 40.72x
        overshoot, 39.64x at this frozen point: Cascade at 2% beats the dialled f = 1.1, moving
        the prediction UP, and hashing plus authentication take back 1.8%.
        """
        f = reconcile.cascade_point(0.02)[4]

        self.assertLess(f, 1.1, msg=f"Cascade at 2% is {f:.5f}, not the dialled 1.1")

        gain = (1.0 - h2(0.05) - f * h2(MAQAN_QBER)) / (1.0 - h2(0.05) - 1.1 * h2(MAQAN_QBER))

        self.assertClose(gain, 1.01523, atol=1e-4, msg=f"reconciliation gain {gain:.5f}")

        net = reconcile.net_rate(MAQAN_GROSS * gain, MAQAN_SIFT * 0.3, 1.0 / 0.3, MAQAN_PULSE * 0.3)

        self.assertClose(
            net / MAQAN_BEST,
            39.51,
            atol=0.05,
            msg=f"overshoot is still {net / 1e3:.2f}x",
        )
        self.assertGreater(net / MAQAN_BEST, 30.0, msg=f"overshoot {net / MAQAN_BEST:.2f}x")


class Contract(Question):
    """
    What the module refuses rather than approximating.
    """

    def test_above_capacity(self):
        """
        A code rate above channel capacity is refused naming the Shannon limit, not returned as
        a beta above one.
        """
        self.assertFails(
            ValueError,
            "Shannon limit",
            reconcile.efficiency,
            0.5,
            0.05,
            msg="rate 0.5 at SNR 0.05",
        )

    def test_below_threshold(self):
        """
        A code below its measured threshold is refused as not decoding, and `pick()` refuses an
        SNR under every threshold on file.
        """
        code = reconcile.pick(0.161, dim=8)

        self.assertFails(
            ValueError,
            "does not decode",
            reconcile.code_beta,
            code,
            0.05,
            msg="SNR 0.05",
        )
        self.assertFails(
            ValueError,
            "below every d=8 threshold",
            reconcile.pick,
            0.001,
            msg="SNR 0.001",
        )

    def test_no_interpolation(self):
        """
        Cascade efficiency is refused at an error rate Table 3 does not carry, f_EC not being
        monotone across its grid.
        """
        self.assertFails(
            ValueError,
            "not monotone",
            reconcile.cascade_point,
            0.025,
            msg="Q=0.025",
        )

    def test_double_charge(self):
        """
        Charging reconciliation leakage inside the key length and again as a padded parity cost
        is refused.
        """
        costs = {
            "amplify": 64.0,
            "reconcile": 1000.0,
        }

        self.assertFails(
            ValueError,
            "charged twice",
            reconcile.net_length,
            1e5,
            costs,
            msg="costs with reconcile",
        )

    def test_biconf_refused(self):
        """
        The Sugimoto-Yamazaki variant, BICONF after two passes, is refused rather than run as
        more parity passes.
        """
        cfg = reconcile.Cascade(variant="sugimoto")

        self.assertFails(
            ValueError,
            "BICONF",
            reconcile.run_cascade,
            0.02,
            1,
            1,
            cfg,
            msg="variant sugimoto",
        )

    def test_short_block(self):
        """
        `hash_length` refuses a min-entropy under the leftover hash toll rather than return a
        negative length.
        """
        self.assertFails(
            ValueError,
            "no key comes out",
            reconcile.hash_length,
            40.0,
            1e-10,
            msg="H_min 40",
        )

    def test_subunit_fec(self):
        """
        An inefficiency below one, under the Slepian-Wolf bound, is refused, as is f_EC at a
        zero error rate.
        """
        self.assertFails(
            ValueError,
            "Slepian-Wolf",
            reconcile.inefficiency,
            0.95,
            0.02,
            msg="rate 0.95 at Q=0.02",
        )
        self.assertFails(
            ValueError,
            "divides by zero",
            reconcile.inefficiency,
            0.9,
            0.0,
            msg="Q=0",
        )

    def test_hash_absent(self):
        """
        `hash_charge` refuses `bb84`, whose six log terms are one composite, `rrdps`, whose
        terms are bit counts with no epsilon, and the unrecorded `cow`.
        """
        self.assertFails(
            ValueError,
            "not separable",
            reconcile.hash_charge,
            "bb84",
            1e-10,
            msg="bb84",
        )
        self.assertFails(
            ValueError,
            "declared bit count",
            reconcile.hash_charge,
            "rrdps",
            1e-10,
            msg="rrdps",
        )
        self.assertFails(
            ValueError,
            "refused by name",
            reconcile.hash_charge,
            "cow",
            1e-10,
            msg="cow",
        )

    def test_zero_fer(self):
        """
        `verify_failure` refuses a zero frame error rate, which would certify the keys never
        differ, naming what zero counted failures bound.
        """
        self.assertFails(
            ValueError,
            "not a count of zero",
            reconcile.verify_failure,
            0.0,
            2.0**-64,
            msg="fer 0",
        )

    def test_auth_saturated(self):
        """
        `auth_failure` refuses a round whose forgery probabilities union-bound past one.
        """
        self.assertFails(
            ValueError,
            "saturated",
            reconcile.auth_failure,
            (1e6,) * 20,
            0.4,
            msg="20 messages at 0.4",
        )

    def test_entropy_shared(self):
        """
        `reconcile._entropy` equals `impairments._entropy` bit for bit at nine points from 0 to
        1.
        """
        for e in (0.0, 1e-6, 0.01, 0.0568, 0.11, 0.25, 0.5, 0.9, 1.0):
            self.assertEqual(
                reconcile._entropy(e),
                impairments._entropy(e),
                msg=f"entropy at {e}",
            )

    def test_catalogue(self):
        """
        Every catalogue entry is an exported callable listing parameters, under a unique name.
        """
        names = [row[0] for row in reconcile.catalogue()]

        self.assertEqual(len(names), len(set(names)), msg=f"names: {names}")

        for name, fn, params in reconcile.catalogue():
            self.assertIs(getattr(reconcile, name), fn, msg=f"{name}")
            self.assertTrue(params, msg=f"{name}: params {params}")


if __name__ == "__main__":
    rc = Exam(
        "ReconcileEfficiency",
        "Capacities, the three efficiency quantities and the published code tables",
        "reconcile_efficiency.md",
    ).run(load(Efficiency))
    rc |= Exam(
        "ReconcileCascade",
        "Cascade block sizes, the published parameter table and a counted simulation",
        "reconcile_cascade.md",
    ).run(load(CascadeRun))
    rc |= Exam(
        "ReconcileHashing",
        "Privacy amplification, Toeplitz seeds and the authentication key bill",
        "reconcile_hashing.md",
    ).run(load(Hashing))
    rc |= Exam(
        "ReconcileDelivery",
        "What the classical layer costs a deployed link",
        "reconcile_delivery.md",
    ).run(load(Delivery))
    rc |= Exam(
        "ReconcileContract",
        "What the module refuses rather than approximating",
        "reconcile_contract.md",
    ).run(load(Contract))
    sys.exit(rc)
