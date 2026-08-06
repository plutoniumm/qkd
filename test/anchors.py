import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MDR import Exam, Question, load

import qkd as q


def lodewyck(security=None, trusted=True):
    """
    Lodewyck 2007 (arXiv:0706.4255) 25 km homodyne link, asymptotic and trusted by default.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=18.5),
        channel=q.Channel(T=0.302, xi=0.005, ref="input"),
        bob=q.Bob(detector=q.Homodyne(eta=0.606, v_el=0.041, trusted=trusted)),
        security=security or q.Asymptotic(beta=0.898),
    )


def sized(n):
    """
    The same link under Leverrier's finite-size security at block size n.
    """

    return lodewyck(q.FiniteSize(beta=0.898, eps=1e-10, n=n))


class CvTheory(Question):
    """
    Tier A CV anchors, atol per the source's quoted precision.
    """

    def test_lodewyck_2007_mutual_information(self):
        """
        Lodewyck 2007 (arXiv:0706.4255, 25 km, homodyne, trusted): $I_{AB} = 1.0436$ bit/symbol,
        their "365 kb/s" over the 350 kHz effective rate. Not their printed 1.045, which that
        same 365 kb/s does not reproduce.
        """
        res = lodewyck().run()
        self.assertClose(res.i_ab, 1.0436, atol=1e-4, msg="I_AB mismatch")

    def test_lodewyck_2007_holevo(self):
        """
        Lodewyck 2007: Holevo bound $\\chi_{BE} = 0.9020$, their printed "316 kb/s" over the
        same 350 kHz rate.
        """
        res = lodewyck().run()
        self.assertClose(res.chi_be, 0.9020, atol=1e-4, msg="chi_BE mismatch")

    def test_lodewyck_2007_key_rate(self):
        """
        Lodewyck 2007: $K = \\beta I_{AB} - \\chi_{BE} = 0.0352$ bit/symbol, their printed
        collective-attack "12.3 kb/s". Their 2 kb/s headline is a net rate after reconciliation
        throughput and is not this quantity.
        """
        res = lodewyck().run()
        self.assertClose(res.key_rate, 0.0352, atol=1e-4, msg="key rate mismatch")

    def test_zhang_2020_worst_case(self):
        """
        Zhang 2020 (arXiv:2001.02555, 202.81 km): asymptotic $K$ at worst-case $\\xi = 0.0383$
        is 5.5e-5 bit/symbol, above their finite-size value.
        """
        res = q.Link(
            modulation=q.GaussianModulation(v_a=7.65),
            channel=q.Channel(T=10**-3.245, xi=0.0383, ref="input"),
            bob=q.Bob(detector=q.Homodyne(eta=0.6134, v_el=0.1523, trusted=True)),
            security=q.Asymptotic(beta=0.98),
        ).run()
        self.assertClose(res.key_rate, 5.5e-5, atol=1e-6, msg="Zhang worst-case K")

    def test_jain_2022_heterodyne(self):
        """
        Jain 2022 (arXiv:2110.09262, 20 km LLO, heterodyne): asymptotic $K \\approx 0.067$
        bit/symbol, above their composable 0.027.
        """
        res = q.Link(
            modulation=q.GaussianModulation(v_a=2.90),
            channel=q.Channel(T=0.35, xi=0.0126 / 0.35, ref="input"),
            bob=q.Bob(detector=q.Heterodyne(eta=0.69, v_el=0.0514, trusted=True)),
            security=q.Asymptotic(beta=0.916),
        ).run()

        # 0.067 is qkd's own point estimate to 2 sf, not a Jain figure: a
        # regression pin, not an anchor. atol is that quote's rounding half-width.
        self.assertClose(res.key_rate, 0.067, atol=5e-4, msg="Jain point estimate")
        self.assertGreater(
            res.key_rate,
            0.027,
            msg="asymptotic must exceed their composable worst case",
        )

    def test_untrusted_is_more_pessimistic(self):
        """
        Handing the detector to Eve never raises the key rate on the same hardware.
        """
        trusted = lodewyck().run()
        res = lodewyck(trusted=False).run()
        self.assertLess(
            res.key_rate,
            trusted.key_rate,
            msg="untrusted detector must cost key rate",
        )


class CvFiniteSize(Question):
    """
    Leverrier, Grosshans & Grangier, Phys. Rev. A 81, 062343 (2010), arXiv:1005.0339.
    """

    def test_finite_below_asymptotic(self):
        """
        Worst-case parameters plus $\\Delta(n)$ hold the rate below asymptotic at every $n$, and
        $n = 10^9$ still yields key.
        """
        finite = sized(1e9).run()
        asym = lodewyck().run()
        self.assertLess(finite.key_rate, asym.key_rate, msg="finite >= asymptotic")
        self.assertGreater(finite.key_rate, 0.0, msg="1e9 block should still yield key")

    def test_zero_key_at_small_block(self):
        """
        At $n = 10^5$ the finite-size penalty gives exactly 0, never negative.
        """
        res = sized(1e5).run()
        self.assertEqual(res.key_rate, 0.0, msg="small block must give zero key")

    def test_finite_approaches_asymptotic(self):
        """
        The rate rises with block size towards $(n/N)$ of asymptotic without reaching it: Eq.
        (10) of Leverrier, Grosshans & Grangier, Phys. Rev. A 81, 062343 (2010) keeps the $n/N$
        prefactor, so estimation's half is never recovered.
        """
        k9 = sized(1e9).run().key_rate
        k12 = sized(1e12).run().key_rate
        asym = lodewyck().run().key_rate
        self.assertLess(k9, k12, msg="rate must grow with block size")
        self.assertLess(k12, 0.5 * asym, msg="still below the n/N-scaled limit")
        self.assertClose(k12, 0.5 * asym, atol=1e-3, msg="1e12 should be near (n/N)*asymptotic")


class DpsTheory(Question):
    """
    The Waks-Takesue-Yamamoto individual-attack bound, quant-ph/0508112.
    """

    def _link(self, qber, mu=0.2):
        return q.Link(
            modulation=q.DifferentialPhase(mu=mu),
            channel=q.Channel(T=0.1),
            bob=q.Bob(detector=q.ClickDetector(eta=0.2, dark=1e-6)),
            security=q.IndividualAttack(qber=qber, f=1.16),
        )

    def test_wty_zero_error_limit(self):
        """
        WTY at $e = 0$: their Eq. (34) gives $P_{c0} = 1/2$, so $R = p_{click}(1 - 2\\mu)$ with
        $p_{click} = 1 - (1 - d)^2 e^{-\\mu T \\eta}$ for dark rate $d$.
        """
        res = self._link(qber=0.0).run()
        p_click = 1.0 - (1.0 - 1e-6) ** 2 * math.exp(-0.2 * 0.1 * 0.2)
        self.assertClose(res.key_rate, p_click * 0.6, atol=1e-12, msg="e=0 closed form")

    def test_wty_monotone_in_qber(self):
        """
        The WTY rate decreases monotonically with QBER.
        """
        rates = [self._link(qber=e).run().key_rate for e in (0.0, 0.01, 0.03, 0.05)]
        self.assertEqual(rates, sorted(rates, reverse=True), msg="rate must fall with QBER")

    def test_wty_domain_saturation(self):
        """
        Past $e = 6/38 \\approx 0.158$ the collision bound saturates (their Eq. 34) and the rate
        is 0, never negative.
        """
        res = self._link(qber=0.16).run()
        self.assertEqual(res.key_rate, 0.0, msg="rate must clamp to 0 past domain")

    def test_wty_multiphoton_penalty(self):
        """
        $\\mu \\ge 1/2$ gives zero rate: Eq. (35)'s $(1 - 2\\mu)$ photon-splitting concession.
        """
        res = self._link(qber=0.01, mu=0.5).run()
        self.assertEqual(res.key_rate, 0.0, msg="mu=0.5 must yield zero")


if __name__ == "__main__":
    rc = Exam(
        "CvAnchors",
        "Tier A: closed-form CV key rates vs published theory numbers",
        "anchors_cv.md",
    ).run(load(CvTheory))
    rc |= Exam(
        "CvFiniteSize",
        "Leverrier 2010 finite-size machinery on the pinned path",
        "anchors_finite.md",
    ).run(load(CvFiniteSize))
    rc |= Exam(
        "DpsAnchors",
        "Tier A: WTY individual-attack DPS bound with pinned QBER",
        "anchors_dps.md",
    ).run(load(DpsTheory))
    sys.exit(rc)
