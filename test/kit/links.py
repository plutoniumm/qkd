import qkd as q

from kit.anchors import ALPHA, ETA_BOB, E_DET, F_REC, P_DARK
from kit.cache import memo


def basis(dist, mod=None, alice=None, misalign=E_DET, intens=(0.48, 0.1, 0.0), probs=None, sift=0.5, block=None):
    """
    A basis-keyed q.Link on the GYS hardware at ``dist`` km.
    """

    return q.Link(
        modulation=mod or q.BasisKeying(decoy=q.Decoy(intens, probs), sift=sift),
        channel=q.Fiber(length=dist, alpha=ALPHA),
        bob=q.Bob(
            detector=q.ClickDetector(eta=ETA_BOB, dark=P_DARK),
            receiver=q.BasisAnalyser(misalign=misalign),
        ),
        security=q.SplittingAttack(f=F_REC, block=block),
        alice=alice,
    )


def cow_link(**kw):
    """
    An intensity-keyed link with its coherence-monitoring tap; any slot arrives whole
    through ``kw``.
    """
    parts = dict(
        modulation=q.IntensityKeying(mu=0.5, decoy_frac=0.1),
        channel=q.Fiber(length=25.0),
        bob=q.Bob(
            detector=q.ClickDetector(eta=0.8, dark=1e-6),
            receiver=q.CoherenceMonitor(split=0.1, misalign=0.02),
        ),
        security=q.PhaseBound(e_phase=0.2, f=1.1),
    )
    parts.update(kw)

    return q.Link(**parts)


def sim_link(security, lw=1e4, db=12.0, imp=(), xi=0.02, dsp=None, iq=None):
    """
    The metro reference link, driven end to end by the symbol pipeline: T = 0.5 and xi at
    the channel input. lw = 0 with a loud pilot leaves the estimator statistics-limited.
    iq is charged onto the claim rather than sampled, so it leaves the pipeline key alone.
    """

    return q.Link(
        modulation=q.GaussianModulation(v_a=5.0),
        channel=q.Channel(T=0.5, xi=xi, ref="input"),
        alice=q.Alice(
            laser=q.Laser(linewidth=lw),
            iq=iq,
            pilots=q.Pilots(power_db=db),
            symbol_rate=100e6,
        ),
        bob=q.Bob(
            detector=q.Heterodyne(eta=0.6, v_el=0.1, trusted=True),
            lo=q.LocalLO(linewidth=lw),
        ),
        dsp=q.DSP() if dsp is None else dsp,
        security=security,
        impairments=imp,
    )


@memo
def frames(symbols, seed=1):
    """
    The metro link's pipeline half at (symbols, seed), computed once. Neither the security
    model nor the block size reaches it; Link.claim() refuses mismatched frames.
    """

    return sim_link(q.Asymptotic(beta=0.95)).measure(symbols, seed)
