BETA = 0.95  # reconciliation efficiency; test/click.py's BETA is a fibre nonlinearity

# CONSOLIDATE BY PROVENANCE, NEVER BY VALUE. Two papers reaching the same number stay two
# declarations; one name at two values is a live collision. The live ones: ALPHA is 0.21
# here, 0.2, 0.165 and 0.35 dB/km elsewhere, a half-range in sqrt(N_0) in attacks.py and a
# third quantity in expqubit.py; ETA_COW is Gao's 0.8 against Stucki's 0.0265
# (experiments.py); ETA_BOB, DET, FEC, F_REC, E_DET, Y_DARK and P_DARK each differ from a
# same-named constant elsewhere; the two F_REC = 1.22 are this one and pairs.py's
# Ma-Fung-Lo Table 1 row. 1.16 lives under nine distinct names, so the value-keyed
# direction is worse. experiments.py reads its GYS block from the paper, not from here.

# GYS hardware: Gobby, Yuan & Shields, Appl. Phys. Lett. 84, 3762 (2004), as tabulated by
# Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005), Table 1 "GYS" row -- ALPHA (alpha), ETA_BOB
# (eta_Bob), Y_DARK (Y_0), E_DET (e_detector). F_REC is f(e) = 1.22 from Ma et al.'s text,
# not Table 1's "f". P_DARK is one detector per gate, Y_DARK the receiver per pulse:
# _core.decoy_gain takes Y_DARK, q.ClickDetector takes P_DARK, q.Link composes
# Y0 = 1 - (1 - P_DARK)^2.
# P_DARK IS NOT A MEASURED DARK-COUNT RATE: it is GYS's ERROR count per clock cycle per
# interferometer output port (experiments.py names it P_ERR), which their text charges to
# dark counts and stray light together at "less than 0.4%".
# E_VAC = 0.5: a background click is uncorrelated with the encoding.
ALPHA = 0.21
ETA_BOB = 0.045
P_DARK = 8.5e-7
Y_DARK = 1.7e-6
E_DET = 0.033
F_REC = 1.22
E_VAC = 0.5

# COW setting of Gao et al., Opt. Express 30, 23783 (2022) Sec. IV. D_COW, the monitoring
# split and the phase-error bound are the exams' own.
ETA_COW = 0.8
E_ALIGN = 0.02
F_COW = 1.1
D_COW = 1e-6

# Kanitschar, George, Lin, Upadhyaya & Lutkenhaus, PRX Quantum 4, 040306 (2023),
# arXiv:2301.08686, Theorem 6: eps_ec + max(eps_pa/2 + eps_bar, eps_et + eps_at). These five
# compose to 1e-10 where a plain sum reports 1.9e-10.
DMCS_EPS = dict(eps_ec=0.2e-10, eps_pa=0.2e-10, eps_bar=0.7e-10, eps_et=0.1e-10, eps_at=0.7e-10)

# (T, eta, v_el, V_A) shared by test/consistency.py and test/crossengine.py. Row three is
# the receiver of Hajomer et al., Sci. Adv. 10, eadi9474 (2024) -- tau = 0.68 and 62.72 mSNU
# (Table 1), V_mod = 8.41 SNU (Sec. 4) -- at the default metro T = 10^-0.5. Row four takes
# their 15.4 dB channel (Sec. 3.1) AND NOTHING ELSE OF THEIRS. Neither row is their
# operating point; that anchor is in test/budget.py.
HARDWARE = (
    (0.80, 0.98, 0.010, 2.00),
    (0.40, 0.60, 0.100, 4.00),
    (10**-0.5, 0.68, 0.06272, 8.41),
    (10**-1.54, 0.50, 0.200, 5.00),
    (0.05, 0.75, 0.050, 12.0),
)

# Trojan-horse attack, the design side: Lucamarini, Choi, Ward, Dynes, Yuan & Shields,
# "Practical security bounds against the Trojan-horse attack in quantum key distribution",
# Phys. Rev. X 5, 031030 (2015), arXiv:1506.01989. THA_N is Sec. III.B's laser-induced
# damage threshold in photons per second onto a 50 um^2 core, THA_CLOCK Alice's phase
# modulator rate, THA_TARGET the leak Sec. IV.A designs for, THA_TABLE their Table I as
# (f_A, |gamma|, |R|, |A|, |I|, n), every dB positive as printed, F = 0 dB in all six rows.
# The measured transmitter behind the first row is in test/exptrojan.py.
THA_N = 1e20
THA_CLOCK = 1e9
THA_TARGET = 1e-6
THA_TABLE = (
    (1e9, 170.0, 40.0, 35.0, 60.0, 1),
    (1e9, 170.0, 50.0, 0.0, 60.0, 2),
    (1e6, 200.0, 40.0, 30.0, 50.0, 2),
    (1e6, 200.0, 50.0, 0.0, 50.0, 3),
    (1e3, 230.0, 40.0, 35.0, 60.0, 2),
    (1e3, 230.0, 50.0, 0.0, 60.0, 3),
)
