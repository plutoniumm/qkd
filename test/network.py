import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q

# MAQAN, the Metro Area Quantum Access Network at Chennai: N. Sharma, S. Vilashini,
# G. Krishnan, A. Gayathri, A. K. Singh, V. P. Singh, V. Ramanathan, P. Mandayam &
# A. Prabhakar, "Developing quantum networks: some key innovations", Quantum Inf.
# Process. 25, 19 (2026), doi:10.1007/s11128-025-05031-x.
#
# Sect. 2, verbatim: "The testbed is set up with five nodes, with one node each
# at IIT Madras, ERNET (IITM Research Park), National Information Centre (NIC)
# and two nodes located at SETS Chennai." Sect. 1, verbatim: the network
# "exclusively integrates the Coherent One-Way (COW) and Differential Phase
# Shift (DPS) QKD protocols". Its BB84 (Sect. 2.3.1) is a spool experiment and
# its DPS-MDI (Sect. 3) a tabletop.
#
# Not the QuILA tender STF/TD/002/2025 Re:01 dt. 27/11/2025 (11 nodes, four
# cities), which secondary articles conflate with it; only MU_CAP and QBER_CAP
# come from the tender.
#
# Unpublished beyond the INVENTED parameters below: splice and connector counts,
# the quantum-channel wavelength, any Raman or scattered power, and any model of
# the deployed network, so nothing here is checked against one.
#
# SETS-1 and SETS-2 are this file's labels; the paper names neither.
IITM = "IITM"
ERNET = "ERNET"
SETS_A = "SETS-1"
SETS_B = "SETS-2"
NIC = "NIC"

# Fig. 2 map labels, tildes the paper's own: "4.8 km, ~7.5 dB" IIT Madras to IIT
# Research Park, "1.1 km, ~2 dB" IIT Research Park to SETS Chennai, "5.6 km, ~3.2
# dB" SETS Chennai to NIC India. Table 1: "5.6 km (3.5dB) 4.8 km (7.5 dB)".
#
# EXAM-SIDE FIGURE READ: Fig. 2 draws ERNET-SETS as two parallel cables (two of
# four yellow-pixel blobs, overlapping in x and y), read as one fibre per SETS
# node against Sect. 2.1's "Our setup only requires one fibre connection between
# Alice and Bob". Which SETS node carries the NIC span is unreadable; SETS-2 is arbitrary.
KM_ERNET = 4.8
KM_CAMPUS = 1.1
KM_NIC = 5.6
DB_ERNET = 7.5
DB_CAMPUS = 2.0

# SETS-NIC: Table 1's 3.5 dB is read; FIG_NIC keeps Fig. 2's ~3.2 dB.
DB_NIC = 3.5
FIG_NIC = 3.2

# Sect. 2.1: "The encoded coherent light is further attenuated to a mean photon number, mu <= 0.2".
MU_MAQAN = 0.2

# Sect. 2.1: "Alice uses a tunable laser source of about 100 KHz instantaneous linewidth".
LINEWIDTH = 100e3

# INVENTED, all three. Table 2's 0.146 efficiency and 3e-6 dark counts are the
# Sect. 3 DPS-MDI "Simulation parameter" at 0.2 dB/km, a different system.
ETA_SPAD = 0.2
DARK_SPAD = 1e-6
VIS_DLI = 0.98

# Fig. 3, Bob's chassis: a 90:10 data/monitoring coupler.
SPLIT_COW = 0.1

# INVENTED: no MAQAN QBER is published, so the QBER derived from this is an output, not an anchor.
MISALIGN = 0.02

# INVENTED: the bottom of Sect. 2.3.2's "Our QKD systems typically operate at gigahertz repetition rates";
# the group's COW runs 1 ns time bins (A. K. Singh, N. Sharma, V. P. Singh & Prabhakar, arXiv:2502.04081).
CLOCK = 1e9

# Installed ITU-T G.652 cable, category B, for the SYNTHETIC hops only; MAQAN's spans are end-to-end losses.
ALPHA = 0.35

# SYNTHETIC: Sect. 3's DPS-MDI is a tabletop proof of principle on a protocol this project no longer implements.
# Curty's relay setting per test/finiterelay.py: 14.5% efficiency, 6.02e-6 background halved per detector, 1.5%.
DET_MDI = 0.145
DARK_MDI = 6.02e-6 / 2.0
MIS_MDI = 0.015

# THIS FILE'S choice: 50 ps intensity FWHM, arms short enough that a 40% width mismatch keeps a positive rate.
FWHM = 50e-12
ARM_MDI = 25.0

# Tender STF/TD/002/2025 Re:01 (IITM CDoT Samgnya Technologies Foundation), Table 1 acceptance
# thresholds: a procurement spec, not a testbed measurement.
MU_CAP = 0.25
QBER_CAP = 0.25

# What the deployed SDN controller reroutes on (aoctestbed.in work package 15):
# "QBER (>20%)" and "SKR falls below acceptable threshold (<10Kbps)".
QBER_ALARM = 0.20
RATE_ALARM = 10e3

# Table 1, "Secure bit rate (highest)": 1 kbps for MAQAN (2 kbps CDoT, 237 bps IITD-DRDO), bounding every link.
SKR_BEST = 1e3

# EXAM-SIDE FIGURE READ of Fig. 4, "Raw Key Rate Time Series Plot": daily-average raw key on IITM-ERNET,
# measured in pixels on the 1181x576 image `pdfimages -f 7` extracts; the figure prints no value.
#
# Calibration: y gridlines at 67, 153, 327, 414 px for 24000, 22000, 18000, 16000 bps, 23.05 bps/px;
# x gridlines 110 px per 25 days; 227 daily markers 2024-11-26 to 2025-07-09 (2024-11-23 and 2025-07-06
# are tick labels). A marker centre locates to half a pixel, RAW_READ; the 0.10 kbps sd and 0.59 kbps
# peak-to-peak about a 15-day running median are data, not read error.
#
# The post-outage record CLIMBS, monthly means 20.12, 20.00, 20.16, 20.43, 20.77 kbps March to July
# (+4.7 bps/day), the paper's "thereafter, further optimisation has marginally improved the performance of
# the link", so a band is pinned. RAW_EARLY is the pre-outage record, flat at 19.32 kbps; RAW_MARCH the
# first monthly mean; RAW_LATE the last daily average; RAW_MEAN the whole post-outage mean, NOT an endpoint.
# The outage is the shaded interoperability-testing band at 2025-02-18.
RAW_EARLY = 19.3e3
RAW_MARCH = 20.12e3
RAW_MEAN = 20.2e3
RAW_LATE = 20.75e3
RAW_READ = 0.05e3

# The record floor and ceiling: the 2024-12-19 minimum and 2025-07-03 maximum.
RAW_FLOOR = 18.96e3
RAW_CEIL = 20.97e3

# Ceiling on the fitted detected-pulse rate: the group's MPD PDM-IR SPAD, 4 ns gate, 32 ns period (arXiv:2502.04081).
GATE_RATE = 1.0 / 32e-9


def sites(*names):
    """
    A node dict from bare names.
    """

    return {name: q.Node() for name in names}


def span(db):
    """
    The channel at a published end-to-end loss in dB; no single alpha fits MAQAN's routes.
    """

    return q.Fiber(T=10.0 ** (-db / 10.0))


def cow(db, eta=ETA_SPAD):
    """
    A COW hop across ``db`` decibels.
    """

    return q.Link(
        modulation=q.IntensityKeying(mu=MU_MAQAN, decoy_frac=0.1),
        channel=span(db),
        bob=q.Bob(
            detector=q.ClickDetector(eta=eta, dark=DARK_SPAD),
            receiver=q.CoherenceMonitor(split=SPLIT_COW, misalign=MISALIGN),
        ),
        security=q.PhaseBound(e_phase=0.05, f=1.1),
    )


def dps(db, eta=ETA_SPAD):
    """
    A DPS hop across ``db`` decibels, its QBER derived from the interferometer and the 100 kHz linewidth.
    """

    return q.Link(
        modulation=q.DifferentialPhase(mu=MU_MAQAN),
        channel=span(db),
        alice=q.Alice(laser=q.Laser(linewidth=LINEWIDTH), symbol_rate=CLOCK),
        bob=q.Bob(
            detector=q.ClickDetector(eta=eta, dark=DARK_SPAD),
            receiver=q.DelayInterferometer(delay=1, visibility=VIS_DLI),
        ),
        security=q.IndividualAttack(f=1.16),
    )


def swap(arm):
    """
    A SYNTHETIC CV swap over two ``arm`` km spans; its Bell detector must be near ideal, eta = 0.9 and
    v_el = 0.01 already giving chi = 4.99 against a floor of 4 and no key.
    """

    return q.Swap(
        alice=q.Sender(modulation=q.GaussianModulation(v_a=4.0)),
        bob=q.Sender(modulation=q.GaussianModulation(v_a=4.0)),
        relay=q.Relay(bell=q.BellDetector(eta=0.98, v_el=0.005)),
        channels=(
            q.Fiber(length=arm, alpha=ALPHA),
            q.Fiber(length=arm, alpha=ALPHA),
        ),
        security=q.Asymptotic(beta=0.95),
    )


def keyed(pulse):
    """
    A decoy weak-coherent sender in the biased-basis limit; ``pulse`` is its temporal-mode width or None.
    """

    return q.Sender(
        modulation=q.BasisKeying(decoy=q.Decoy((0.3, 0.1, 0.0)), sift=1.0),
        pulse=pulse,
    )


def midpoint(widths=(None, None), test=None, arm=ARM_MDI):
    """
    A SYNTHETIC qubit midpoint; two ``widths`` derive the test-basis misalignment, ``test`` pins it, both is refused.
    """

    return q.Swap(
        alice=keyed(widths[0]),
        bob=keyed(widths[1]),
        relay=q.Relay(
            bell=q.BellAnalyser(
                eta=DET_MDI,
                dark=DARK_MDI,
                misalign=MIS_MDI,
                misalign_test=test,
            )
        ),
        channels=(q.Fiber(length=arm, alpha=ALPHA),) * 2,
        security=q.TestBasisBound(),
    )


def maqan():
    """
    MAQAN as Fig. 2 draws it, five nodes and four hops; the protocol per span is INVENTED.
    """

    return q.Network(
        nodes=sites(IITM, ERNET, SETS_A, SETS_B, NIC),
        edges=[
            q.Hop(ends=(IITM, ERNET), link=cow(DB_ERNET), clock=CLOCK),
            q.Hop(ends=(ERNET, SETS_A), link=cow(DB_CAMPUS), clock=CLOCK),
            q.Hop(ends=(ERNET, SETS_B), link=dps(DB_CAMPUS), clock=CLOCK),
            q.Hop(ends=(SETS_B, NIC), link=dps(DB_NIC), clock=CLOCK),
        ],
    )


def swapped():
    """
    A SYNTHETIC two-edge network, one q.Link and one q.Swap.
    """

    return q.Network(
        nodes=sites(IITM, ERNET, SETS_A),
        edges=[
            q.Hop(ends=(IITM, ERNET), link=cow(DB_ERNET), clock=CLOCK),
            q.Hop(ends=(IITM, SETS_A), link=swap(1.3), clock=CLOCK),
        ],
    )


def pair(a, b, first, second):
    """
    A two-hop chain a -- x -- b out of two ready-made links.
    """

    return q.Network(
        nodes=sites(a, "MIDDLE", b),
        edges=[
            q.Hop(ends=(a, "MIDDLE"), link=first, clock=CLOCK),
            q.Hop(ends=("MIDDLE", b), link=second, clock=CLOCK),
        ],
    )


def rates(db):
    """
    (bits per emitted pulse, click probability per pulse) for a COW hop across ``db`` decibels.
    """
    out = cow(db).run()

    return out.key_rate, out.p_click


class Graph(Question):
    """
    The shape of a q.Network, and every configuration it refuses.
    """

    def test_swap_edge(self):
        """
        A swap is one edge between its two senders, its midpoint in no segment key and no node degree.
        """
        res = swapped().run(symbols=50_000)
        seen = set()
        for ends in res.segments:
            seen.update(ends)
        self.assertEqual(seen, {IITM, ERNET, SETS_A}, msg=f"segment endpoints {sorted(seen)}")
        self.assertIn((IITM, SETS_A), res.segments, msg="no IITM--SETS-1 segment")
        self.assertEqual(res.nodes[IITM].degree, 2, msg=f"IITM degree {res.nodes[IITM].degree}")

    def test_pair_order(self):
        """
        A segment rate reads the same with the hop's ends in either order.
        """
        res = maqan().run(symbols=50_000)
        ahead = res.rates[(IITM, ERNET)]
        behind = res.rates[(ERNET, IITM)]
        self.assertClose(ahead, behind, atol=0.0, msg=f"{ahead} != {behind}")
        self.assertIn((SETS_B, ERNET), res.rates, msg="reversed key not found")

    def test_clock_units(self):
        """
        Every hop's rate in bits per second is its per-shot key rate times the declared clock.
        """
        res = maqan().run(symbols=50_000)
        for ends, out in res.segments.items():
            want = CLOCK * out.key_rate
            self.assertClose(res.rates[ends], want, atol=0.0, msg=f"{ends}: {res.rates[ends]}")

    def test_node_total(self):
        """
        ERNET's total is the sum of its three incident segment rates in bits per second.
        """
        res = maqan().run(symbols=50_000)
        want = res.rates[(IITM, ERNET)] + res.rates[(ERNET, SETS_A)] + res.rates[(ERNET, SETS_B)]
        self.assertEqual(res.nodes[ERNET].degree, 3, msg=f"{res.nodes[ERNET]}")
        self.assertClose(res.nodes[ERNET].total, want, atol=1e-9, msg=f"{res.nodes[ERNET]}")

    def test_relay_vertex(self):
        """
        A q.Relay offered as a vertex is refused.
        """
        nodes = {
            IITM: q.Node(),
            SETS_A: q.Node(),
            "MID": q.Relay(bell=q.BellDetector(eta=0.98, v_el=0.005)),
        }
        net = q.Network(
            nodes=nodes,
            edges=[q.Hop(ends=(IITM, SETS_A), link=cow(DB_ERNET), clock=CLOCK)],
        )
        self.assertFails(
            ValueError,
            "not a vertex",
            net.run,
            msg="a q.Relay was accepted as a node",
        )

    def test_swap_isolated(self):
        """
        Every hop of a route beside a swap is a pair of q.Node vertices, and the midpoint is refused as an endpoint.
        """
        res = swapped().run(symbols=50_000)
        route = res.route(ERNET, SETS_A)

        for hop in route.hops:
            self.assertEqual(len(hop), 2, msg=f"hop {hop} is not a node pair")

            for name in hop:
                self.assertIn(name, res.nodes, msg=f"{name} is not a q.Node")
        self.assertFails(
            ValueError,
            "no node named",
            res.route,
            IITM,
            "midpoint",
            msg="a midpoint was routable",
        )

    def test_bad_graph(self):
        """
        Isolated nodes, parallel hops, unknown endpoints, self-loops and a payload neither q.Link nor q.Swap raise.
        """
        good = q.Hop(ends=(IITM, SETS_A), link=cow(DB_ERNET), clock=CLOCK)
        cases = (
            ("sit on no hop", sites(IITM, SETS_A, NIC), [good]),
            (
                "parallel segments",
                sites(IITM, SETS_A),
                [good, q.Hop(ends=(SETS_A, IITM), link=dps(DB_ERNET), clock=CLOCK)],
            ),
            ("which is not a node", sites(IITM, NIC), [good]),
            (
                "q.Link or a q.Swap",
                sites(IITM, SETS_A),
                [q.Hop(ends=(IITM, SETS_A), link=q.Node(), clock=CLOCK)],
            ),
        )
        for needle, nodes, edges in cases:
            net = q.Network(nodes=nodes, edges=edges)
            self.assertFails(ValueError, needle, net.run, msg=f"missed: {needle}")
        self.assertFails(
            ValueError,
            "distinct nodes",
            q.Hop,
            (IITM, IITM),
            cow(DB_ERNET),
            CLOCK,
            msg="a self-loop was constructed",
        )

    def test_payload_blame(self):
        """
        A segment's own refusal is re-raised naming the hop.
        """
        broken = q.Link(
            modulation=q.GaussianModulation(v_a=5.0),
            channel=span(DB_ERNET),
            bob=q.Bob(detector=q.ClickDetector(eta=ETA_SPAD, dark=DARK_SPAD)),
        )
        net = q.Network(
            nodes=sites(IITM, SETS_A),
            edges=[q.Hop(ends=(IITM, SETS_A), link=broken, clock=CLOCK)],
        )
        self.assertFails(
            NotImplementedError,
            f"hop ('{IITM}', '{SETS_A}')",
            net.run,
            msg="the failing hop was not named",
        )

    def test_dry_plan(self):
        """
        explain() runs nothing, labels a rate "derived (run to compute)" and carries the trust and end-to-end rows.
        """
        info = maqan().explain()
        self.assertEqual(
            info[f"rate_{IITM}_{ERNET}"]["label"],
            "derived (run to compute)",
            msg=f"{info[f'rate_{IITM}_{ERNET}']}",
        )

        for key in ("trust", "units", "end_to_end", "contention"):
            self.assertIn(key, info, msg=f"explain() lost the {key} row")
        self.assertIn("never a vertex", info["trust"]["value"], msg=f"{info['trust']}")


class Routing(Question):
    """
    A path across a q.Network is key management, and the API says so.
    """

    def test_no_rate(self):
        """
        Neither a network result nor a route answers key_rate, key, rate or secure_rate, naming no security bound.
        """
        res = maqan().run(symbols=50_000)
        route = res.route(NIC, IITM)
        for holder, needle in ((res, "no key rate"), (route, "no key rate")):
            for name in ("key_rate", "key", "rate", "secure_rate"):
                self.assertFails(
                    AttributeError,
                    needle,
                    getattr,
                    holder,
                    name,
                    msg=f"{type(holder).__name__}.{name} answered",
                )
        self.assertFails(
            AttributeError,
            "not a security bound",
            getattr,
            res,
            "key_rate",
            msg="no 'not a security bound'",
        )

    def test_bottleneck(self):
        """
        A route's bottleneck is the minimum rate over its hops.
        """
        res = maqan().run(symbols=50_000)
        route = res.route(NIC, IITM)
        want = min(res.rates[ends] for ends in route.hops)
        self.assertClose(route.bottleneck, want, atol=0.0, msg=f"{route.bottleneck} != {want}")

    def test_trusts_named(self):
        """
        route.trusts names every intermediate node, ERNET and SETS-2 from NIC to IITM, and none on a single hop.
        """
        res = maqan().run(symbols=50_000)
        long = res.route(NIC, IITM)
        self.assertEqual(len(long.trusts), len(long.hops) - 1, msg=f"{long.hops} vs {long.trusts}")
        self.assertEqual(set(long.trusts), {ERNET, SETS_B}, msg=f"{long.trusts}")
        self.assertIn("trusted-node key relay", long.security, msg=long.security)

        short = res.route(IITM, ERNET)
        self.assertEqual(short.trusts, (), msg=f"single hop trusts {short.trusts}")
        self.assertIn("no intermediate node", short.security, msg=short.security)

    def test_widest_path(self):
        """
        Routing maximises the bottleneck, taking a wider two-hop detour over a starved direct hop.
        """
        starved = q.Link(
            modulation=q.DifferentialPhase(mu=MU_MAQAN),
            channel=q.Fiber(length=120.0, alpha=0.2),
            bob=q.Bob(detector=q.ClickDetector(eta=ETA_SPAD, dark=DARK_SPAD)),
            security=q.IndividualAttack(qber=0.03, f=1.16),
        )
        net = q.Network(
            nodes=sites(IITM, ERNET, SETS_A),
            edges=[
                q.Hop(ends=(IITM, SETS_A), link=starved, clock=CLOCK),
                q.Hop(ends=(IITM, ERNET), link=cow(DB_ERNET), clock=CLOCK),
                q.Hop(ends=(ERNET, SETS_A), link=cow(DB_ERNET), clock=CLOCK),
            ],
        )
        res = net.run()
        route = res.route(IITM, SETS_A)
        self.assertEqual(len(route.hops), 2, msg=f"took {route.hops}")
        self.assertGreater(
            route.bottleneck,
            res.rates[(IITM, SETS_A)],
            msg="detour bottleneck <= direct rate",
        )

    def test_route_guards(self):
        """
        A route to an unknown node, to itself, or across a disconnected graph is refused.
        """
        res = maqan().run(symbols=50_000)
        self.assertFails(ValueError, "no node named", res.route, IITM, "MID", msg="unknown accepted")
        self.assertFails(ValueError, "distinct nodes", res.route, IITM, IITM, msg="self accepted")

        split = q.Network(
            nodes=sites(IITM, ERNET, SETS_A, NIC),
            edges=[
                q.Hop(ends=(IITM, ERNET), link=cow(DB_ERNET), clock=CLOCK),
                q.Hop(ends=(SETS_A, NIC), link=cow(DB_ERNET), clock=CLOCK),
            ],
        ).run()
        self.assertFails(
            ValueError,
            "disconnected",
            split.route,
            IITM,
            NIC,
            msg="a disconnected pair was routed",
        )

    def test_segments_agree(self):
        """
        Each segment's key rate equals its standalone payload's exactly.
        """
        link = cow(DB_ERNET)
        edge = swap(1.3)
        res = pair(IITM, SETS_A, link, edge).run()
        alone = (link.run().key_rate, edge.run().key_rate)
        joint = (
            res.segments[(IITM, "MIDDLE")].key_rate,
            res.segments[("MIDDLE", SETS_A)].key_rate,
        )
        self.assertClose(joint[0], alone[0], atol=0.0, msg=f"{joint[0]} != {alone[0]}")
        self.assertClose(joint[1], alone[1], atol=0.0, msg=f"{joint[1]} != {alone[1]}")


class Modes(Question):
    """
    The two-source mode model: pulse widths derive the TEST-basis misalignment, never the KEY basis's.
    """

    def test_perfect_match(self):
        """
        Two equal widths give overlap 1, the 1/2 two-source dip, a derived test-basis misalignment equal to the
        key basis's, and a rate bit-identical to pinning that value.
        """
        res = midpoint(widths=(FWHM, FWHM)).run()
        info = res.explain
        self.assertClose(info["overlap"]["value"], 1.0, msg="overlap at equal widths")
        self.assertClose(res.hom, 0.5, msg=f"hom {res.hom}")
        self.assertClose(
            info["misalign_test"]["value"],
            MIS_MDI,
            msg="misalign_test != MIS_MDI",
        )
        self.assertEqual(info["misalign_test"]["label"], "derived", msg="misalign_test not derived")
        self.assertClose(
            res.key_rate,
            midpoint(test=MIS_MDI).run().key_rate,
            atol=0.0,
            msg="rate != pinned rate",
        )

    def test_mismatch_costs(self):
        """
        Widening one sender's pulse lowers the overlap, the dip and the rate, and raises the derived test-basis
        misalignment from the key basis's.
        """
        runs = [midpoint(widths=(FWHM, FWHM * k)).run() for k in (1.0, 1.1, 1.2, 1.3, 1.4)]
        seen = [one.explain["overlap"]["value"] for one in runs]
        miss = [one.explain["misalign_test"]["value"] for one in runs]
        self.assertMonotone(seen, rising=False, msg=f"overlap {seen}")
        self.assertMonotone([one.hom for one in runs], rising=False, msg="hom not falling")
        self.assertMonotone(miss, msg=f"misalign_test {miss}")
        self.assertMonotone(
            [one.key_rate for one in runs],
            rising=False,
            msg="key_rate not falling",
        )

        for got in miss:
            self.assertGreaterEqual(got, MIS_MDI, msg=f"misalign_test {got} < MIS_MDI")

    def test_key_basis_pinned(self):
        """
        Two widths never derive the key-basis misalignment, and a pinned test-basis error beside two widths is refused.
        """
        bare = q.Swap(
            alice=keyed(FWHM),
            bob=keyed(FWHM),
            relay=q.Relay(bell=q.BellAnalyser(eta=DET_MDI, dark=DARK_MDI)),
            channels=(q.Fiber(length=ARM_MDI, alpha=ALPHA),) * 2,
            security=q.TestBasisBound(),
        )
        self.assertFails(
            NotImplementedError,
            "Hong-Ou-Mandel",
            bare.run,
            msg="key-basis misalign derived",
        )
        self.assertFails(
            ValueError,
            "accepted and dropped",
            midpoint(widths=(FWHM, FWHM), test=MIS_MDI).run,
            msg="test beside two widths accepted",
        )

    def test_one_width(self):
        """
        One width alone is refused, and with no width the test-basis error must be given directly.
        """
        self.assertFails(
            NotImplementedError,
            "PAIR of temporal modes",
            midpoint(widths=(FWHM, None), test=MIS_MDI).run,
            msg="one width accepted",
        )
        self.assertFails(
            NotImplementedError,
            "misalign_test",
            midpoint().run,
            msg="no misalign_test accepted",
        )

    def test_hop_carries(self):
        """
        A derived midpoint composes as one q.Hop at the clock times its own rate, its station no vertex and its dip
        on the reported segment.
        """
        edge = midpoint(widths=(FWHM, FWHM))
        net = q.Network(
            nodes=sites(IITM, SETS_A),
            edges=[q.Hop(ends=(IITM, SETS_A), link=edge, clock=CLOCK)],
        )
        res = net.run()
        self.assertClose(
            res.rates[(SETS_A, IITM)],
            CLOCK * edge.run().key_rate,
            atol=0.0,
            msg="hop rate != CLOCK * key_rate",
        )
        self.assertEqual(set(res.graph), {IITM, SETS_A}, msg=f"vertices {sorted(res.graph)}")
        self.assertIsNotNone(res.segments[(IITM, SETS_A)].hom, msg="segment hom is None")


class Maqan(Question):
    """
    The Chennai testbed as its own group's paper gives it: five nodes, two
    protocols, three published spans (QIP 25, 19 (2026)).
    """

    def test_five_nodes(self):
        """
        MAQAN's vertex set is its five deployed key-holding q.Node sites, two at SETS Chennai, a trusted-node
        network (TSDSI TR 60XX v1.0.8) with the IITM-ERNET span present.
        """
        net = maqan()
        self.assertEqual(
            set(net.nodes),
            {IITM, ERNET, SETS_A, SETS_B, NIC},
            msg=f"{sorted(net.nodes)}",
        )

        for name, node in net.nodes.items():
            self.assertIsInstance(node, q.Node, msg=f"{name} is {type(node).__name__}")

        sets = [name for name in net.nodes if name.startswith("SETS")]
        self.assertEqual(len(sets), 2, msg=f"SETS Chennai holds {sets}")
        self.assertIn(
            (IITM, ERNET),
            [hop.ends for hop in net.edges],
            msg="no IITM-ERNET hop",
        )

    def test_phase_reference_only(self):
        """
        Every MAQAN hop is a q.Link running COW or DPS, the two protocols the paper says it exclusively integrates.
        """
        net = maqan()
        kinds = set()
        for hop in net.edges:
            self.assertIsInstance(hop.link, q.Link, msg=f"{hop.ends} is not a q.Link")

            kinds.add(type(hop.link.modulation).__name__)
        self.assertEqual(
            kinds,
            {"IntensityKeying", "DifferentialPhase"},
            msg=f"MAQAN carries {sorted(kinds)}",
        )

        res = net.run(symbols=50_000)
        seen = {res.explain[f"kind_{a}_{b}"]["value"] for a, b in res.segments}
        self.assertEqual(seen, {"Link"}, msg=f"{seen}")

    def test_installed_loss(self):
        """
        The three published spans give 1.5625, 1.8182 and 0.6250 dB/km, a 2.9x spread, each above the 0.35 dB/km
        ITU-T category B figure for installed G.652 cable.
        """
        pairs = (
            (KM_ERNET, DB_ERNET),
            (KM_CAMPUS, DB_CAMPUS),
            (KM_NIC, DB_NIC),
        )
        seen = [db / km for km, db in pairs]
        self.assertClose(seen[0], 1.5625, atol=1e-4, msg=f"IITM-ERNET {seen[0]:.4f}")
        self.assertClose(seen[1], 1.81818, atol=1e-4, msg=f"ERNET-SETS {seen[1]:.4f}")
        self.assertClose(seen[2], 0.625, atol=1e-4, msg=f"SETS-NIC {seen[2]:.4f}")
        self.assertGreater(max(seen) / min(seen), 2.5, msg=f"spread {max(seen) / min(seen):.3f}")

        for got in seen:
            self.assertGreater(got, ALPHA, msg=f"{got:.4f} dB/km <= ALPHA")

    def test_loss_sources_differ(self):
        """
        Table 1's 3.5 dB and Fig. 2's ~3.2 dB for SETS-NIC differ by 0.3 dB, moving that hop's secure key rate by 7.1%.
        """
        gap = DB_NIC - FIG_NIC
        self.assertClose(gap, 0.3, atol=1e-9, msg=f"gap {gap}")

        table = cow(DB_NIC).run().key_rate
        figure = cow(FIG_NIC).run().key_rate
        self.assertGreater(figure, table, msg="Fig. 2 rate <= Table 1 rate")
        self.assertClose(
            figure / table - 1.0,
            0.0710,
            atol=5e-4,
            msg=f"rate shift {figure / table - 1.0:.4f}",
        )

    def test_tender_caps(self):
        """
        Every hop's derived QBER is under tender STF/TD/002/2025 Table 1's 25%, and the paper's mu <= 0.2 under 0.25.
        """
        res = maqan().run(symbols=200_000)
        for ends, out in res.segments.items():
            seen = getattr(out, "qber", None)
            self.assertLess(seen, QBER_CAP, msg=f"{ends}: error rate {seen:.4f}")
        self.assertLess(MU_MAQAN, MU_CAP + 1e-12, msg=f"MAQAN carries mu {MU_MAQAN}")

    def test_reroute_alarms(self):
        """
        Every hop and the NIC to IIT Madras widest path clear the deployed controller's 10 kbps reroute threshold,
        and SETS-2 to NIC its 20% QBER threshold.
        """
        res = maqan().run(symbols=200_000)
        for ends in res.segments:
            self.assertGreater(res.rates[ends], RATE_ALARM, msg=f"{ends}: {res.rates[ends]:.1f} bit/s")

        seen = res.segments[(SETS_B, NIC)].qber
        self.assertLess(seen, QBER_ALARM, msg=f"derived qber {seen:.4f}")

        route = res.route(NIC, IITM)
        self.assertGreater(
            route.bottleneck,
            RATE_ALARM,
            msg=f"bottleneck {route.bottleneck:.1f} bit/s",
        )


class MaqanTierB(Question):
    """
    Tier B on MAQAN's two measured numbers. Pinned: 7.5 dB IITM-ERNET, 2.0 dB ERNET-SETS, mu = 0.2. Fitted: ONE
    parameter, the detected-pulse rate, from Fig. 4's figure-read band. The 1 GHz clock is a ceiling, not a MAQAN
    number. The predicted best secure rate misses Table 1's 1 kbps by 41x, declared.
    """

    def test_late_band(self):
        """
        Fig. 4's post-outage record climbs 0.63 kbps, 12.6x the 0.05 kbps read uncertainty, from a 20.12 kbps March
        mean to a 20.75 kbps last daily average, its 20.2 kbps whole-band mean between them.
        """
        climb = RAW_LATE - RAW_MARCH
        self.assertGreater(climb, 10.0 * RAW_READ, msg=f"climb {climb:.0f} bit/s")
        self.assertLess(RAW_MARCH, RAW_MEAN, msg=f"March {RAW_MARCH:.0f} above the mean")
        self.assertLess(RAW_MEAN, RAW_LATE, msg=f"mean {RAW_MEAN:.0f} above the endpoint")
        self.assertGreater(RAW_MARCH - RAW_EARLY, climb, msg="outage step <= climb")

        for got in (RAW_EARLY, RAW_MARCH, RAW_MEAN, RAW_LATE):
            self.assertGreater(got, RAW_FLOOR, msg=f"{got:.0f} < RAW_FLOOR")
            self.assertLess(got, RAW_CEIL, msg=f"{got:.0f} > RAW_CEIL")

    def test_raw_rate_fit(self):
        """
        20.75 kbps of raw key over the 7.5 dB span's 5.745e-3 click probability fits a 3.61 MHz detected-pulse rate,
        0.36% of the 1 GHz clock and 11.6% of the 31.25 MHz SPAD gate rate, the pre-outage record fitting 6.99% lower.
        """
        _, click = rates(DB_ERNET)
        self.assertClose(click, 5.745e-3, atol=1e-6, msg=f"click {click:.6g}")

        pulses = RAW_LATE / click
        self.assertClose(pulses, 3.612e6, atol=2e3, msg=f"fitted {pulses:.5g} Hz")
        self.assertLess(pulses, CLOCK, msg=f"{pulses:.4g} Hz exceeds the clock")
        self.assertLess(pulses, GATE_RATE, msg=f"{pulses:.4g} Hz exceeds the SPAD gate rate")

        early = RAW_EARLY / click
        self.assertGreater(pulses, early, msg=f"early fit {early:.5g}")
        self.assertClose(
            (pulses - early) / pulses,
            0.0699,
            atol=2e-3,
            msg=f"early/late gap {(pulses - early) / pulses:.4f}",
        )

    def test_secure_rate_miss(self):
        """
        DECLARED MISS: the fitted rate on the 2.0 dB ERNET-SETS span predicts 40.7 kbps of asymptotic secure key
        against Table 1's delivered finite-size 1 kbps, a 40.7x unsafe overshoot the Fig. 4 band moves by 3.0% (39.5x).
        """
        _, click = rates(DB_ERNET)
        pulses = RAW_LATE / click
        best = max(cow(db).run().key_rate for db in (DB_ERNET, DB_CAMPUS, DB_NIC))
        got = pulses * best
        self.assertClose(got, 40.72e3, atol=1e2, msg=f"predicted {got:.5g} bit/s")
        self.assertGreater(got / SKR_BEST, 30.0, msg=f"overshoot {got / SKR_BEST:.2f}x")
        self.assertLess(got / SKR_BEST, 50.0, msg=f"overshoot {got / SKR_BEST:.2f}x")

        floor = RAW_MARCH / click * best
        self.assertClose(floor, 39.49e3, atol=1e2, msg=f"March end {floor:.5g} bit/s")
        self.assertClose(
            (got - floor) / got,
            0.0303,
            atol=2e-3,
            msg=f"band shift {(got - floor) / got:.4f}",
        )
        self.assertLess(got, CLOCK * best, msg="prediction above CLOCK * best")

    def test_fit_survives_eta(self):
        """
        Sweeping the INVENTED detector efficiency from 0.1 to 0.3 moves the overshoot only from 40.86x to 40.57x,
        a 0.72% spread.
        """
        seen = []
        for eta in (0.1, 0.146, 0.2, 0.3):
            near = cow(DB_ERNET, eta).run()
            far = cow(DB_CAMPUS, eta).run()
            seen.append(RAW_LATE / near.p_click * far.key_rate / SKR_BEST)
        self.assertMonotone(seen, rising=False, msg=f"overshoot {seen}")
        self.assertClose(seen[0], 40.860, atol=0.01, msg=f"eta 0.1 gives {seen[0]:.3f}")
        self.assertClose(seen[-1], 40.567, atol=0.01, msg=f"eta 0.3 gives {seen[-1]:.3f}")

        spread = max(seen) / min(seen) - 1.0
        self.assertClose(spread, 0.00723, atol=1e-4, msg=f"spread {spread:.5f}")


if __name__ == "__main__":
    rc = Exam(
        "Network · Graph",
        "The graph layer: vertices hold key, a swap is one edge, refusals",
        "network_graph.md",
    ).run(load(Graph))
    rc |= Exam(
        "Network · Routing",
        "Paths are key management: bottleneck, trusted nodes, and no key_rate",
        "network_routing.md",
    ).run(load(Routing))
    rc |= Exam(
        "Network · Mode model",
        "Two pulse widths derive the test basis's misalignment; the key basis's is pinned",
        "network_modes.md",
    ).run(load(Modes))
    rc |= Exam(
        "Network · MAQAN",
        "The Chennai testbed as its own group publishes it: five nodes, COW and DPS",
        "network_maqan.md",
    ).run(load(Maqan))
    rc |= Exam(
        "Network · MAQAN Tier B",
        "Tier B: Sharma et al. 2026, MAQAN's measured raw and secure key rates",
        "network_tierb.md",
    ).run(load(MaqanTierB))
    sys.exit(rc)
