# Validation

qkd is checked against published numbers in two tiers, each with its own success
criterion and failure meaning. Both run as [MDR exams](/tests/); one exam is a
declared [skip](#skips).

## The scoreboard

**Theory**: qkd reproduces what a paper computed. **Experiment**: qkd reproduces what
a paper measured, at a stated number of fitted parameters. A family can be green on
the first and empty on the second.

A green row can answer an easier question than its neighbour. Every anchor reachable
from `q.` outside Gaussian modulation and BB84-WCP is **asymptotic**; six-state on a
single-photon source, MDI-BB84, RRDPS and mode pairing have finite-key anchors in
`qkd._core`. Refused finite-key analyses, per family:
[Security](/guide/security#asymptotic).

### The two tiers are not one score {#two-tiers}

A **Tier A** green cell: qkd and the paper computed the *same quantity* and got the
*same number*, every input pinned. A **Tier B** green cell: *physical parameters
inside declared ranges reproduce a measurement*. Tier B is strictly weaker, and its
parameter count is part of the claim.

| | Tier A | Tier B |
| --- | --- | --- |
| Free parameters | none | named, counted, and bounded in the [Tier B table](#tier-b) |
| Target | the paper's own computed value | the paper's measured observables |
| Success | equality to the quoted precision, or a correctly-signed inequality | inside the stated uncertainty, from fitted values that are themselves plausible |
| A failure means | a bug, or a [convention mismatch](/guide/conventions#where-the-literature-disagrees) | the noise model is falsified; a fit needing an absurd parameter falsifies it louder than a miss |
| Cost | milliseconds on the debug build: pinning collapses `run()` to closed form | a fit, reported with its free parameters |

#### The theory axis has three levels, and they are not interchangeable {#levels}

"It matches the paper" is not an answer. Every Tier A cell carries a qualifier, and
the qualifier *is* the claim.

| Level | Qualifier | Means |
| --- | --- | --- |
| **1** | ✅ **exact** | the same quantity, agreeing to numerical precision. The tolerance is round-off, $10^{-12}$ to $10^{-16}$, not chosen here |
| **1**, at the source's precision | ✅ **values** | a published constant, table or figure reproduced to the digits the source prints. The source's last digit is the tolerance: the same number, a weaker agreement than **exact** |
| **2** | ✅ **bound** | a correctly-signed inequality; not the same number |
| **3** | ⚙ **structural** | no published number, but the behaviour the source requires |
| orthogonal to level | ⚠️ **pinned input** | one input read from the literature rather than derived, named in the exam |

In Tier B the qualifier is a **count of free parameters**. A prediction the fit did
not reach is a [declared miss](#misses), kept rather than tuned away.

### The board {#board}

| Protocol family | Reached through | Tier A — theory, nothing fitted | Tier B — experiment, parameters fitted |
| --- | --- | --- | --- |
| Gaussian modulation, homodyne | `q.Link` | ✅ **values** — Lodewyck 2007's three printed kb/s figures | ✅ **3 assumed** — default metro link inside the measured $\xi$ envelope |
| Gaussian modulation, heterodyne | `q.Link` | ✅ **bound** — Jain 2022 | ✅ **3 fitted** — Hajomer 2024, $\xi$ within 1%; key rate bounded, with a documented shortfall |
| Long-distance / worst-case CV | `q.Link` | ✅ **bound** — Zhang 2020 | ✅ **0 fitted** — Zhang's six measured points, each at its own published parameters |
| Finite-size machinery | `q.FiniteSize`, `q.KeyBlock` | ⚙ **structural** — Leverrier 2010 (continuous variable) and Lim et al. 2014 (decoy BB84) | — not applicable |
| Differential phase shift | `q.Link` | ⚙ **structural** — WTY 2006; QBER derived | ✅ **4 fitted** — Diamanti et al. 2006; ✅ **3 fitted** — their full 3.4% with timing jitter, exactly determined; ✅ **1 fitted** — Takesue et al. 2007's 42.1 dB reach, where qkd and the paper evaluate the *same* expression |
| Round-robin differential phase shift | `qkd._core` | ✅ **values** — Yin et al. Table 1 in both columns, and their reading of the $L = 65$ run | ✅ **1 fitted** — Takesue, Sasaki, Tamaki & Koashi, *Nat. Photon.* **9**, 827, [arXiv:1505.07914](https://arxiv.org/abs/1505.07914), 30 km (`ExpTakesueTrain`) |
| Coherent one way | `q.Link` | ⚙ **structural** ⚠️ **pinned input** — Gao et al. 2022 reach, phase-error bound read off their figure; ✅ **bound** — the certified COW′ bound, [SDP · Bound](/tests/sdp_bound) | ✅ **1 fitted** — Stucki et al. 2009 QBER curve to 250 km; ✅ **2 fitted** — Korzh et al. 2015 to 307 km. Both papers' rates *bounded*, not reproduced |
| Decoy states | `q.Decoy` | ✅ **values** — Ma, Qi, Zhao & Lo 2005 on the GYS set | ✅ **0 fitted** on the background anchor, **1 fitted** on the rate — Lucamarini et al. 2013's T12, the first decoy *observables* reproduced |
| BB84 with weak coherent pulses | `q.BasisKeying` | ✅ **values** — GLLP (Gottesman, Lo, Lütkenhaus & Preskill, *QIC* **5**, 325 (2004)) with decoy on the GYS set; ✅ **bound** — the finite-key length below Zhang, Zhao, Razavi & Ma 2017 | ✅ **2 fitted** — Gobby, Yuan & Shields 2004 over 122 km |
| Six-state | `q.BasisKeying(bases=3)` | ✅ **exact** — Rev. Mod. Phys. 81, 1301 App. A, Eqs. (A4)–(A6); ✅ **values** — the 12.6% threshold at 12.6193% | ✅ **0 fitted** on the anchor, **1 fitted** on the partial-attack leg — Enzer et al. 2002 |
| SARG04 | `q.BasisKeying(announce="pair")` | ✅ **exact** — Fung, Tamaki & Lo Eq. (34); ✅ **values** — 9.689% and 2.710% tolerable, 97.2 km with decoy | ✅ **1 fitted** — Jeong et al. 2014, BB84 and SARG04 on one rig |
| B92 | `q.TwoStateKeying` | ✅ **values** — Tamaki & Lütkenhaus's three depolarising thresholds at 0.03379, 0.02313, 0.01215 | ✅ **1 fitted** — Hughes et al. 1999 over 48 km; ✅ **1 fitted** — Gordon et al. 2004's QBER curve to 9.96 km |
| E91, priced by the violation | `q.ViolationBound` | ✅ **exact** — Acín et al.'s $S = 2\sqrt2(1-2Q)$ bridge, Pironio et al. Eq. (12); ✅ **values** — the 7.1% threshold at 7.149% | ✅ **0 fitted** — Naik et al. 2000; ✅ **0 fitted** — Ling et al. 2008 |
| Discrete modulation, $M$-PSK | `q.PhaseShiftKeying` | ✅ **values** — Denys, Brown & Leverrier 2021 from primary sources; Ghorai et al. 2019 QPSK curve to $<0.5\%$ | ✅ **0 fitted** — Hajomer et al. 2024's four 10 GBaud rows, as a ceiling |
| Continuous-variable relay (CV-MDI) | `q.Swap` | ✅ **values** — zero crossings and asymmetry against Pirandola et al. 2015; PLOB as a ceiling | ✅ **0 fitted** — Hajomer et al. 2025 over 10 km |
| Basis-keyed relay (MDI-BB84) | `q.Swap` | ✅ **values** — Ma & Razavi 2012's closed forms at their own limits, Xu et al. 2013's joint decoy bound sandwiched, Lo, Curty & Qi's stated 40 dB tolerance | ✅ **1 fitted** — Rubenok et al. 2013, sixteen measured gains and eight error rates |
| Mode-pairing / asynchronous MDI | `qkd._core` | ✅ **values** — Zeng et al. 2022's two closed-form intensity optima; ⚙ **structural** — the $\sqrt\eta$ exponent at 0.5018 and the repeaterless crossing | ✅ **0 fitted** — Zhu, Huang, Liu, Zeng, Zou, Dai, Tang, Li, You, Wang, Chen, Ma, Chen & Pan, Phys. Rev. Lett. **130**, 030801 (2023), [arXiv:2208.05649](https://arxiv.org/abs/2208.05649), 101–407 km (`ExpZhuPairing`) |
| Entanglement-based basis keying (BBM92) | `q.PairLink` | ✅ **exact** — Ma, Fung & Lo 2007 Eqs. (9)–(10) against an independent four-detector enumeration; Acín et al. 2007's $\chi(S)$ endpoints and 7.1% threshold | ✅ **1 fitted** — Honjo et al. 2008 over 100 km |
| Loss-tolerant source flaws | `qkd._core` | ✅ **exact** — Pereira, Curty & Tamaki's printed yield to $10^{-15}$ and their Eq. (33) bit error rate; ⚙ **structural** — the loss-tolerant theorem's own equality | ✅ **0 fitted** — Xu, Wei, Sajeed, Kaiser, Sun, Tang, Qian, Makarov & Lo, "Experimental quantum key distribution with source flaws", Phys. Rev. A **92**, 032305 (2015), [arXiv:1408.3667](https://arxiv.org/abs/1408.3667), three distances (`ExpXuFlaws`). ⚠️ Not `test/flaws.py`, which uses Xu's flaw magnitude only as `MIZ_FLAW`, Mizutani's *simulation setting*, and reproduces none of Mizutani's Figs. 2–7 |
| Certified COW′ phase error (SDP) | `qkd._core` | ✅ **bound** — at or below Gao et al.'s Cauchy–Schwarz estimator at every operating point, at or above the honest channel; ⚙ **structural** — the $\eta^2$ exponent, the doubled rate, the extended reach | ✅ **1 fitted** — Cao, Sun, Li, Lu, Yin & Chen, *Sci. Adv.* **12**, eaec2776, [arXiv:2601.06772](https://arxiv.org/abs/2601.06772), four distances and both records (`ExpCaoVacuum`) |
| Detector attacks | `q.attacks` | ✅ **values** — Qin et al.'s level-II displacement, 19.54 against their stated 19.5, and $\xi_{\text{lin}} = 2.1$ SNU under a full intercept-resend; ✅ **values** — Lydersen 2010's blinding powers, Zhao 2008's Table I, Qi's $r = 0.2$ crossing, Weier 2011's Table I | ✅ **0 fitted** — Zhao's time-shift counts, Weier's seven blinding intensities, Lydersen's threshold ratio |
| Reconciliation and post-processing | `q.reconcile` | ✅ **values** — Jouguet et al. Tables I–III; Martinez-Mateo et al. Eqs. (4) and (6) turning their Table 3's columns into one another; simulated Cascade's counted parities reproduce their $f_{EC}$ at 1.04006 and 1.04313 — a quantity they obtained by running the algorithm, hence Tier A | — not applicable: their $f_{EC}$ measures an **algorithm**, not hardware |
| Truncated-Fock / non-Gaussian | `q.fock` | ✅ **values** — closed-form Wigner values and the $\|1\rangle$ negativity volume | ✅ **0 fitted** — Lvovsky et al. 2001's measured Wigner dip |
| Discrete modulation, the **certified** proof | `qkd._core` (`dm_secure`) | ✅ **exact** — the reduced relative entropy against an explicit numpy construction of the Kraus image to $10^{-12}$, and the analytic gradient against central differences of that objective; ✅ **bound** — the step-2 certificate never above the step-1 Frank–Wolfe value nor the repeaterless capacity, and looser after one step than after fifteen; ⚙ **structural** — LUL19 Sec. IV E's postselection gain, and reach past where the analytic bound has died | ❌ none |
| Photon-number-resolving detection | `q.ThresholdArray`, `q.PnrDetector` | ✅ **exact** — the one-element array reproduces `click.rs`'s threshold law to 2 ulp over efficiency, dark rate and flux, and the occupancy recursion agrees with the closed inclusion–exclusion form in exact rational arithmetic; ⚙ **structural** — completeness and positivity on the truncated space, and a shortfall from true number resolution first order in $1/N$ with coefficient $n(n-1)/2$ | ❌ none |
| Transmitted local oscillator | `q.TransmittedLO` | ✅ **exact** — every quantity reduces to the locally generated path as the oscillator is made local, the key-rate difference exactly zero across trust modes and both detectors, and the reconstruction at a transparent channel is the Gaussian layer's two-mode squeezed vacuum; ✅ **values** — Jouguet, Kunz-Jacques & Diamanti's calibration point, a shot-noise ratio of 1.525 erasing a full intercept-resend at $T = \eta = 0.5$ | ❌ none — but it bears on two green rows; [below](#tlo-bears) |
| Finite key, six-state | `qkd._core` (`sixstate_finite`) | ⚙ **structural** — Scarani & Renner's Lemmas 1–3: never above the asymptotic rate at any block, bias or error rate, monotone in the block, exactly zero past the 12.6193% asymptotic crossing, an interior optimum in the basis bias; ✅ **values** — convergence onto `sixstate_secret` as $1/\sqrt n$, gap 1.5e-8 at $N = 10^{22}$ | — not applicable |
| Finite key, MDI-BB84 / RRDPS / mode pairing | `qkd._core` | ✅ **values** — `rrdps_split` returns Takesue et al.'s five hand-fixed values, and Xie et al.'s printed $2.4\times10^{-9}$ security bound comes back from a 24-term composition at $\varepsilon = 10^{-10}$; ⚙ **structural** — convergence onto the asymptotic rates at exponents $-0.5001$ (MDI-BB84), $-0.5006$ (RRDPS, exact rather than approached) and $-1.0001$ (mode pairing, counts supplied exactly so the whole gap is the epsilon budget) | ❌ none |
| The repeaterless-capacity envelope | cross-family | ✅ **exact** — an ideal reverse-reconciled homodyne link sits at **exactly** half the PLOB capacity at every transmittance, residual $7.7\times10^{-12}$ after removing a deficit first order in $1/V_A$; ⚙ **structural** — ten families, each maximised over its own intensity, under the capacity at every transmittance from 0.9 to $10^{-4}$, worst approach 0.677; the two legitimate escapes pinned **with** their crossings | — not applicable |
| Network layer (key management, not physics) | `q.Network` | — no bound to reproduce | ✅ **1 fitted** — Sharma et al. 2026's MAQAN testbed, with a **$40.72\times$** declared miss |

| Distinction inside Tier A | Row by row |
| --- | --- |
| **Exact versus inequality** | Lodewyck is the one Gaussian-modulation row computing the identical quantity, at the source's printed three significant figures: a **values** row, not **exact**. Zhang and Jain publish finite-size or composable rates, which an asymptotic point estimate must *exceed* |
| **Numeric versus structural** | the Leverrier and WTY rows reproduce no published number; they check the machinery behaves as the source requires. The DPS protocol under the bound is anchored by the click exams against $e = (1-V)/2$ and $e = \tfrac12(1 - e^{-\sigma^2/2})$, $\sigma^2 = 2\pi\,\Delta\nu\,d / f_{\text{sym}}$; the WTY bound meets a *measured* rate at Diamanti's 100 km point in Tier B |

The network row's Tier B cell is green on reachability and on its controller's two
reroute thresholds only. Its rate prediction is a [declared miss](#misses) at
$40.72\times$, not a reproduction.

| Open in Tier B | Why |
| --- | --- |
| The relative-entropy proof for discrete modulation | its published comparisons are *calculations*; a row needs a fielded discrete-modulation run at parameters the proof accepts. Nearest published-data check: [not pinned](#unpinned) |
| Photon-number-resolving detection | nothing here reaches a key rate. A row needs a published PNR characterisation stating element count, efficiency and background beside measured click statistics |
| Finite key for MDI-BB84, RRDPS and mode pairing | no fielded run publishes a block size and an epsilon budget beside its counts |
| Transmitted local oscillator | not for want of an experiment: the two closest are green rows under a *different* engine ([below](#tlo-bears)) |

## Tier A — theory anchors, exact

Pin everything the paper states — $T$, $\xi$, $V_A$, $\eta$, $v_{el}$, $\beta$ — and
the closed form must reproduce the published rate. A discrepancy is a bug or a
[convention mismatch](/guide/conventions#where-the-literature-disagrees): the SNU
factor, the $\xi$ plane, the heterodyne 3 dB term.

### The continuous-variable rows

| Anchor | Configuration | Reproduced |
| --- | --- | --- |
| Lodewyck et al. 2007, 25 km ([arXiv:0706.4255](https://arxiv.org/abs/0706.4255)) | homodyne, trusted, asymptotic | **values**: $I_{AB} = 1.0436$, $\chi_{BE} = 0.9020$, $K = 0.0352$ bit/symbol. The paper prints them as rates at 350 kHz — $I_{AB} = 365$, $\chi_{BE} = 316$, $\Delta I^{\text{eff}}_{\text{Holevo}} = 12.3$ kb/s — and qkd returns 365.28, 315.72 and 12.304, each the published figure to the three digits printed. The paper's *other* printing, "$I(X,Y) = 1.045$" bit/symbol, disagrees with its own 365 kb/s (365.75) and is not what this row reads |
| Zhang et al. 2020, 202.81 km ([arXiv:2001.02555](https://arxiv.org/abs/2001.02555)) | homodyne, trusted, worst-case $\xi = 0.0383$ | $K = 5.5\times10^{-5}$ bit/symbol, upper-bounding their finite-size $1.24\times10^{-6}$ |
| Jain et al. 2022, 20 km LLO ([arXiv:2110.09262](https://arxiv.org/abs/2110.09262)) | heterodyne, trusted receiver | asymptotic $K \approx 0.067$, exceeding their composable worst-case $0.027$ |
| Leverrier et al. 2010 ([arXiv:1005.0339](https://arxiv.org/abs/1005.0339)) | finite-size machinery | three properties: below asymptotic, convergence to $(1 - p_{\text{PE}})\times$asymptotic, exact zero at small $N$ |
| Denys, Brown & Leverrier 2021 ([arXiv:2103.13945](https://arxiv.org/abs/2103.13945)) | $M$-PSK, heterodyne, collective attacks | constellation moments and the certified correlation to $10^{-12}$, against the paper's LaTeX source and its reference implementation, not a plot; plus the Gaussian-limit lock — `dm_holevo` at an unbounded alphabet reproduces `cv_rate` to $5\times10^{-12}$ |
| Ghorai, Grangier, Diamanti & Leverrier 2019 ([PRX **9**, 021059](https://doi.org/10.1103/PhysRevX.9.021059)) | QPSK, linear-objective SDP, $\alpha = 0.35$ | rate-versus-distance curve to better than $0.5\%$ at every point, dying at 74.1 km against a published 73.6 — read from a **figure digitisation** (Fig. 9 of Lin, Upadhyaya & Lütkenhaus 2019), so the exam pins qkd's own values at a loose tolerance |
| Pirandola et al. 2015, *High-rate quantum cryptography in untrusted networks*, Nat. Photon. **9**, 397 ([arXiv:1312.4104](https://arxiv.org/abs/1312.4104)) | CV relay, pure-loss arms | symmetric zero crossing $\tau = 0.83879$, **3.82 km** per arm against their 0.84 / 3.8; decoder-side crossing $e/(1+e) = 0.7310586$, **6.80 km**, against 0.73 / 6.8, in closed form to $10^{-6}$. Against a **calculation**, not an experiment: [W4](#register) |
| Number-state and cat closed forms | truncated Fock | vacuum $W(0,0) = 1/\pi$ and $Q(0,0) = 1/(2\pi)$ exactly; the $\|1\rangle$ negativity volume **0.4261636** against the exact $4/\sqrt e - 2 = 0.4261226$; even/odd cat $W(0,0) = \pm 1/\pi$ to $10^{-9}$. The volume is on the exam's $301^2$ grid over $\pm6$; the [module default window](/guide/fock#grid) gives $0.4262150$. A negativity without its grid and cutoff is not a number |

### The threshold-detector and qubit rows

| Anchor | Configuration | Reproduced |
| --- | --- | --- |
| Waks–Takesue–Yamamoto 2006 ([quant-ph/0508112](https://arxiv.org/abs/quant-ph/0508112)) | DPS, individual attacks | four structural properties: $e = 0$ limit, monotonicity, domain saturation at $6/38$, $\mu \ge 1/2$. Their Eq. (37) as printed cannot be evaluated: [E1](#register) |
| Ma, Qi, Zhao & Lo 2005 ([quant-ph/0503005](https://arxiv.org/abs/quant-ph/0503005)) | vacuum+weak decoy, GYS hardware | Sec. 3.1 and Figs. 1–2: $\mu_{\text{opt}} = 0.48$ at $f(e) = 1.22$ and $0.54$ at the Shannon limit; $Y_1$ 3.5% below and $e_1$ 16.8% above the infinite-decoy values at 40 km; maximum secure distance 140.55 km against the infinite-decoy 142.05 km |
| BB84-WCP on the same GYS set | GLLP, fed the **infinite-decoy** $(Q_1, e_1)$ | positive at 140 km and zero by 145 km; $2.55\times10^{-3}$ bit/pulse back to back and $1.72\times10^{-5}$ at 100 km; rate linear in $\eta$ to a log–log slope of $1.00 \pm 0.05$ |
| Lim, Curty, Walenta, Xu & Zbinden 2014 ([arXiv:1311.7129](https://arxiv.org/abs/1311.7129)) | decoy BB84 finite key, GYS hardware | the counted-event bounds **reduce to the rate-based decoy layer**: $s_1/N \to \tau_1 y_1 q^2$ and $\phi \to e_1$, each from the conservative side, $s_0/N \to \tau_0 Y_0 q^2$; the length is whole bits, rises with the block, falls with distance, never exceeds the asymptotic rate. Two printing errors: [E6](#register) |
| Zhang, Zhao, Razavi & Ma 2017 ([arXiv:1611.02524](https://arxiv.org/abs/1611.02524)) | the same protocol under a **tighter** concentration inequality | their Table 3 BB84 row, $N = 10^{10}$ at 100 km, gives $3.04\times10^{-6}$ bit/pulse. Lim's Hoeffding bound must not exceed it and certifies **nothing** there at any basis bias: the additive deviation on the total detection count swamps a vacuum line six decades smaller. A hundredfold longer block clears their figure |
| Gao et al. 2022, COW ([arXiv:2107.09329](https://arxiv.org/abs/2107.09329)) | post-zero-error-attack analysis | positive within 100 km, below $10^{-5}$ bit/pulse there, **quadratic** in $\eta$ and so below decoy BB84 at range |
| Bruß 1998 and Scarani et al. 2009, Rev. Mod. Phys. **81**, 1301, App. A | six-state, one-way, Bell-diagonal | **exact**: Eq. (A4) against Eq. (A5) as printed to $5\times10^{-15}$, the depolarising Eq. (A6) to $10^{-15}$; Bell-diagonal weights from Renner, Gisin & Kraus, PRA **72**, 012332 (2005), Eq. (10). Secret-fraction zero crossing **12.6193%**, rounding to the review's 12.61% after Eq. (A6) and within $2\times10^{-4}$ of Lo's 12.6%, against BB84's 11.003% on the same arithmetic |
| Fung, Tamaki & Lo 2006, PRA **73**, 012337 ([quant-ph/0510025](https://arxiv.org/abs/quant-ph/0510025)) | SARG04, unconditional, one and two photons | **exact**: the engine's phase bound is their Eq. (34) at $a = 0$ regrouped, $10^{-15}$ from the printed expression. Tolerable bit error rates **9.689%** against 9.68% and **2.710%** against 2.71%; the two-photon floor at $\sin^2(\pi/8)$ exactly; decoy secure distance **97.2 km**; distance ceilings $e_1 = 1/3$ at 207.68 km and Theorem 2's $(3-\sqrt2)/7$ at 201.43 km. Eq. (39)'s entropy subscript cannot be read literally: [E5](#register); Tamaki & Lo's two-photon parenthetical sits at the wrong minimiser: [E4](#register) |
| Branciard et al. and Niederberger et al., photon-number splitting ceilings | SARG04 against BB84 | optimised ceilings scale as $t^{3/2}$ (SARG04) and $t^{2}$ (BB84), peaking at $\mu = 2\sqrt t$ (Branciard Eq. (107)) and $\mu = t$ (Niederberger Eq. (29)); at its optimum the ceiling equals Branciard's $(\eta/3)h_2(p_1)t^{3/2}$ with $h_2(p_1) = 0.60088$. The curves cross once, at 7.70 km of 0.25 dB/km fibre |
| Tamaki & Lütkenhaus 2004, PRA **69**, 032316 ([quant-ph/0308048](https://arxiv.org/abs/quant-ph/0308048)) | plain B92, lossy and noisy | tolerable depolarising rates **0.03379, 0.02313, 0.01215** against 0.034, 0.023, 0.012 at $L = 0$, 0.2, 0.5. The phase bound reaches exactly $1/2$ at loss equal to the state overlap, so the rate arrives at zero with nothing refused. The e-print's depolarising channel drops a $p/3$: [E9](#register) |
| Koashi 2004, PRL **93**, 120501 ([quant-ph/0403131](https://arxiv.org/abs/quant-ph/0403131)) | strong-reference B92 | the $O(t)$ branch, linear in the small-$\eta t\mu$ sense: chord log–log slope over $t \in [10^{-4}, 0.1]$ is 0.99968 at $\mu = 0.05$ and 0.99367 at $\mu = 1$ |
| Ekert 1991, PRL **67**, 661; Acín et al. 2007 ([quant-ph/0702152](https://arxiv.org/abs/quant-ph/0702152)); Pironio et al. 2009 | E91 priced by the CHSH violation | **exact**: the error bridge inverts Acín's $S = 2\sqrt2(1-2Q)$ to $10^{-15}$; the rate reproduces Pironio et al. Eq. (12) term for term to $10^{-14}$. The CHSH-priced rate vanishes at **7.149%** against the photon-pair engine's **11.003%**; the violation is lost at $(2-\sqrt2)/4 = 14.645\%$. **No key rate is built on a device-independent claim** ([why](/guide/protocols#not-di)) |
| Ma & Razavi 2012 ([arXiv:1204.4856](https://arxiv.org/abs/1204.4856)) | MDI-BB84 forward model, weak coherent pulses | four **parameter-free** limits of their Eqs. (21), (23), (35), (39), (51)–(54): $Y_{11} \to \eta_a\eta_b/2$ exactly, the $1/2$ the linear-optics Bell-measurement ceiling; $e_{11} \to e_d$ and $E^{Z} \to e_d$ exactly without dark counts; $E^{X} \to \tfrac14 + e_d/2$, to $10^{-9}$ at $\mu = 10^{-8}$; $Q^{X}/Q^{Z} \to 2$. Their Eq. (7) single-photon-source rate is recovered from Xu et al.'s Eq. (1), not coded twice |
| Xu, Curty, Qi & Lo 2013 ([arXiv:1305.6965](https://arxiv.org/abs/1305.6965)) | MDI-BB84 joint decoy bound, Eq. (1) and Table 2 | the three-intensity $Y_{11}$ lower bound **sandwiched against the forward model** over decoy strengths, off-centre relays and both case-split branches: strictly below the true yield, monotonically tightening, within 1% at $\nu = 0.005$. The $e_{11}^{X}$ upper bound returns the misalignment where the raw test-basis QBER reads a quarter |
| Lo, Curty & Qi 2012 ([arXiv:1109.1473](https://arxiv.org/abs/1109.1473)) | MDI-BB84 reach, standard parameter set | their **more than 40 dB** tolerance, about 200 km, holds: positive at 200 km, dead by 260. Three-intensity secure distance **230.5379 km / 46.1076 dB**; $4.679772\times10^{-6}$ bit per pulse pair at 100 km; log–log slope against **end-to-end** transmittance $1.03695$, BB84-WCP's scaling class (1.01) |
| Zeng, Zhou, Wu & Ma 2022, Nat. Commun. **13**, 3903 ([arXiv:2201.04300](https://arxiv.org/abs/2201.04300)) | mode-pairing / asynchronous MDI | their two closed-form intensity optima off a sweep: $1/2$ when the pairing window holds several clicks, 1 when at most one. Rate as the **square root** of end-to-end transmittance, log–log slope **0.5018** against BB84-WCP's 1.01 and MDI-BB84's 1.037; above the repeaterless capacity of its own transmittance from **282.7 km / 46.6 dB** to **673.4 km**; neighbour-only pairing returns the linear exponent. Their Eq. (122) repeaterless bound is 11% wrong by 160 dB; the comparison uses `-log1p(-eta)/ln 2`. The arXiv id is routinely given wrong: [W8, W9](#register) |
| Ma, Fung & Lo 2007 ([quant-ph/0703122](https://arxiv.org/abs/quant-ph/0703122)) | entanglement-PDC pair source, threshold detectors | Eqs. (9) and (10) against an **independent four-detector enumeration**: worst gain residual $3.1\times10^{-16}$, worst $E\!\cdot\!Q$ residual $3.8\times10^{-16}$ over $\lambda \in [10^{-4}, 3]$, $\eta \in [0.02, 1]$. The enumeration **settles Eq. (10)'s denominator exponent**: the squared reading misses by $7\%$ at $\lambda = 0.1$ while agreeing to $0.03\%$ at $\lambda = 10^{-3}$. Also their Table 1 hardware ($\eta = 14.5\%$, $e_d = 1.5\%$, $Y_0 = 6.02\times10^{-6}$, $f = 1.22$), optimal $\mu = 2\lambda$ staying $O(1)$ across 25 dB, and log–log slope $1.02$ in end-to-end transmittance |
| Tamaki, Curty, Kato, Lo & Azuma 2014, PRA **90**, 052314 ([arXiv:1312.3514](https://arxiv.org/abs/1312.3514)), with Pereira, Curty & Tamaki 2019, npj QI **5**, 62 ([arXiv:1902.02126](https://arxiv.org/abs/1902.02126)) | loss-tolerant analysis with state-preparation flaws | **exact**: Pereira's printed yield to $10^{-15}$ and their Eq. (33) bit error rate wherever their closed form stays below $1/2$. Structurally, the phase error from inverting the rejected data equals the one from measuring the virtual states Alice never sends, and barely moves across forty decibels of loss. At the smaller published flaw, tilt $-\delta$, the quantum-coin analysis stops distilling at **28.766343 dB** and the loss-tolerant one at **57.775328 dB**: a **29.008984 dB** gap that is the worst-case assumption, **not** a security margin |
| Seksaria & Prabhakar (2026), Sec. VI D, tightening Gao et al., Opt. Express **30**, 23783 (2022) ([arXiv:2107.09329](https://arxiv.org/abs/2107.09329)) | COW′ certified phase error, by semidefinite programme | certified worst case **at or below** the Cauchy–Schwarz estimator at every operating point and **at or above** the honest channel's phase error. The certified rate is about **twice** the analytic one at each bound's own optimum, falls as $\eta^2$, and survives 100 km where the analytic one has stopped. Reach is a [declared miss](#misses); feeding the whole pulse reproduces the source's zero-distance phase errors, 0.235 analytic and 0.155 certified, to the digit |
| Qin, Kumar & Alléaume 2016, PRA **94**, 012325 ([arXiv:1511.01007](https://arxiv.org/abs/1511.01007)) | homodyne saturation attack | **values**: Sec. VII.3.1's $\xi_{\text{lin}} = 2.1$ SNU under a full intercept-resend and the unbiased transmittance at $G = 2$, exactly; the level-II displacement at 31 km, 19.54 against a stated 19.5, **partly** published, $V_A$ coming from the exam's fixed-SNR rule. `sat_gain` holds $\hat T = T$ to $10^{-12}$ — a root-finder self-consistency residual, not a published number — solving the **corrected** Eq. (19). Eq. (17)'s factor 2 is a typesetting loss, confirmed against the authors' thesis: [E7, E7b, E7c](#register) |
| Jouguet, Kunz-Jacques & Diamanti 2013, PRA **87**, 062313 ([arXiv:1304.7024](https://arxiv.org/abs/1304.7024)) | local-oscillator calibration attack | Eqs. (7) and (8) at $T = \eta = 0.5$: a full intercept-resend's 2.1 SNU behind a shot-noise overestimate of 1.5, which the paper calls "close to zero". Parameters and phrase are theirs; the two numbers are derived and printed nowhere — exactly $1/15 = 0.06667$ SNU reported, and $61/40 = 1.525$ drives it to zero. Eq. (3) carries a sign error: [E10](#register) |
| Qi, Fung, Lo & Ma 2007, QIC **7**, 73 ([quant-ph/0512080](https://arxiv.org/abs/quant-ph/0512080)) and Fung, Tamaki, Qi, Lo & Ma 2009, QIC **9**, 131 ([arXiv:0802.3788](https://arxiv.org/abs/0802.3788)) | detection-efficiency mismatch | Eq. (2) gives 0 at $r = 0$, $1/2$ at $r = 1$, crossing the intercept-resend $1/4$ at their stated $r = 0.2$; a mismatch of 2 gives their $h(2/3) = 0.9183$ ceiling, symmetric under $r \to 1/r$. Fung et al. Eqs. (32)–(34): matched detectors make the two forms identical; a mismatch makes the data-discarding form larger by exactly $(1 - \text{share})\,h(e_{\text{bit}})$ |
| Jouguet, Kunz-Jacques & Leverrier 2011, PRA **84**, 062317 ([arXiv:1110.0100](https://arxiv.org/abs/1110.0100)) | multi-edge LDPC reconciliation | Table I efficiencies 95.9%, 97.2%, 98.1% as $R$ over the **binary-input** capacity at the quoted threshold — the rate-1/2 code's 98.2% returns 0.9817 against the Gaussian-input capacity's 0.9502, which identifies the channel the published $\beta$ is measured against. All fifteen rows of Tables II and III at $d = 0, 1, 2, 4, 8$, largest deviation 0.0070 at rate 0.02, where the threshold is printed to two decimals |
| Martinez-Mateo, Pacher, Peev, Ciurana & Martin 2015, QIC **15**, 453 ([arXiv:1407.3257](https://arxiv.org/abs/1407.3257)) | Cascade | Eq. (4) turns Table 3's $f_{EC}$ column into its $\beta$ column in all twelve rows to $5\times10^{-5}$, the rounding of four printed decimals, round-tripping to $10^{-9}$; Eq. (6) turns $f_{EC}$ and the frame error rate into the $\eta_{EC}$ column in all twelve rows to $3\times10^{-5}$. Their near-optimal block rule reproduces the first three block sizes of nine of twelve rows; the three it misses are named |
| Yin, Wang, Chen, Han, Wang, Guo & Han 2018, Nat. Commun. **9**, 457 ([arXiv:1702.01260](https://arxiv.org/abs/1702.01260)) | round-robin DPS | Table 1's tolerable bit error rate at every tabulated train length under the original bound and their collective one, including $L = 3$, where the original certifies nothing; their reading of the $L = 65$ run of Wang et al., Nat. Photon. **9**, 832 (2015) — $R_1 = 5\times10^{-8}$ to $5\times10^{-10}$, $R_2 = 1.44\times10^{-6}$ to $10^{-8}$, twenty-nine times the first. Three published assemblies of the cost are ordered `rrdps_gllp` $\le$ `rrdps_tag` $\le h_2($`rrdps_phase`$)$ at every argument. The $L = 65$ paper is routinely mis-cited, [W6](#register); the RRDPS source paper has no arXiv version, [W7](#register) |

| Row | Caveat |
| --- | --- |
| BB84-WCP | composes GLLP from an **infinite-decoy closed form** in the exam, not from `decoy_bounds`: it anchors the rate composition and the GYS channel, and the 140–145 km crossing is *insensitive to the finite-decoy estimator*. That estimator: [Decoy · Anchors](/tests/decoy_anchors), [Decoy · Sandwich](/tests/decoy_sandwich), reduced to its infinite-decoy limit by [Cross-engine · Decoy](/tests/cross_decoy) |
| Denys | checked against the paper's **source** and its authors' ancillary implementation, removing the plot-reading error. The Gaussian-limit lock is internal consistency, not a literature anchor: two independent layers agreeing where they describe the same physics, both in SNU with $\xi$ at the channel input |
| Decoy | Ma, Qi, Zhao and Lo's Eq. (7) yields carry a double-count correction — a pulse that arrives *and* draws a background click is one detection — which their Eq. (10) drops. **A stated approximation, not an error**: Eq. (10) is printed with $\cong$ and the smallness assumption follows ([W1](#register)). qkd keeps the correction; dropping it put the $Y_1$ lower bound **above** its infinite-decoy ceiling. The forward gain sits about $10^{-8}$ below the paper's closed form on GYS hardware; `test_gain_double_count` pins the gap |
| COW | the analytic phase-error bound is **pinned at 0.20**, read off the 0.17–0.24 band of Gao et al.'s figure over 0–100 km at 2% misalignment; their bound is a function of monitoring-line decoy gains `discrete.rs` does not model. `test_cow_phase_pinned` carries the word UNVERIFIED. The pin stands for plain COW; the vacuum-decoy variant COW′ derives its phase error from the record ([SDP · Bound](/tests/sdp_bound)) |
| SARG04 | the two privacy-amplification domains no published equation states ([Protocol coverage](/guide/protocols#sarg)). Read literally, the two-photon term credits 0.0347 bit per two-photon detection at $e_2 = 0.25$. Both cutoffs return the value crediting zero rather than refusing, so a distance sweep runs through them: on the GYS row $e_2$ passes $1/6$ at 188.0 km and $e_1$ passes $1/3$ at 207.7 km |

## Tier B — experimental anchors, statistical {#tier-b}

Pin what the paper publishes; **fit what it does not**, within plausible ranges; land
inside the paper's stated uncertainty. Where a paper is incomplete, the target is the
[measured envelope](/guide/budget#the-measured-envelope), $\xi \approx 0.005$–$0.03$
SNU at 25 km.

- **The fitted values must be plausible**, and are reported beside the match: a fit needing an absurd parameter has falsified the noise model.
- **Fit curves, not points**: a scalar anchor is underdetermined.

### Continuous variable {#tier-b-cv}

| Anchor | Pinned, from the paper | Free, and its range | Result | Report |
| --- | --- | --- | --- | --- |
| Default 25 km metro link (no single source; target the [measured envelope](/guide/budget#the-measured-envelope)) | $V_A = 5$, $T = 10^{-0.5}$, 16-bit DAC, 12-bit ADC, $v_{el} = 0.1$ trusted | **assumed, not fitted** — $v_{\text{err}} = 2\times10^{-3}$ rad$^2$ in $[10^{-4}, 10^{-2}]$; RIN $-155$ dBc/Hz, an ordinary fibre-laser figure; $B = 100$ MHz, set by the 100 MBaud symbol rate | $\xi = 1.09\times10^{-2}$ SNU at the channel input, inside $[0.005, 0.03]$, residual phase noise the dominant row | [Budget · Tier B](/tests/budget_tierb) |
| Hajomer et al. 2024, 100 km LLO ([arXiv:2305.08156](https://arxiv.org/abs/2305.08156)) | $V_{\text{mod}} = 8.41$, 15.4 dB, 16-bit DAC/ADC, $\eta = 0.68$, $v_{el} = 62.72$ mSNU, $\beta = 0.925$ | **3 fitted** — $v_{\text{err}} = 7\times10^{-4}$ rad$^2$ in $[10^{-4}, 10^{-2}]$, between Zhang's transmitted-LO $7.6\times10^{-5}$ and Qi 2015's first LLO demonstration, $0.04$; RIN $-155$ dBc/Hz; $B = 100$ MHz. ADC range ratio held at 10 | Bob-plane $\xi$ within 1% of their $2.12\times10^{-4}$ SNU; asymptotic rate $\approx 2.7\times10^{-3}$ bit/symbol upper-bounds their finite-size $2.54\times10^{-4}$ | [Budget · Tier B](/tests/budget_tierb) |
| Zhang et al. 2020, six points to 202.81 km (PRL **125**, 010502; [arXiv:2001.02555](https://arxiv.org/abs/2001.02555)) | Table I in full — $V_A$, $\xi$, $\xi'$, $v_{el}$, $\beta$, FER, loss and SNR at six distances, from the arXiv source | **0 fitted.** Attenuation published beside each length, inverting to 0.15988–0.16816 dB/km against their stated 0.16 | the asymptotic rate at each point's parameters upper-bounds their finite-size rate at all six, $1.006\times$ at 27.27 km rising monotonically to $9.390\times$ at 202.81 km, and $1.004$–$3.968\times$ on worst-case $\xi'$. Their SNR column recomputes from $V_A$, $T$, $v_{el}$, $\xi$ to 0.020–0.290 dB. Their measured log–log slope 1.658 is steeper than the asymptotic model's 1.266, as a compounding finite-size penalty requires | [Experiments · Zhang](/tests/exp_longhaul) |
| Hajomer, Andersen & Gehring 2025, CV-MDI over 10 km (Quantum Sci. Technol. **10**, 025032; [arXiv:2303.01611](https://arxiv.org/abs/2303.01611)) | Table 1 in full — $V_A = V_B = 6.5$ SNU, 20 MBaud, $N = 4\times10^6$, $\xi = 39.5$ mSNU at the relay, $\tau_A = 1$, $\tau_B = 0.56$, relay efficiency 0.94, $\beta = 97\%$ | **0 fitted.** The referral of relay-plane $\xi$ into the arms is **bracketed**: all on Bob, split evenly, and between | equivalent noise 5.6815 and a per-attack rate of 0.18042 bit/symbol, above the 0.130 bit/symbol their finite-size 2.6 Mbit/s at 20 MBaud gives; the even split gives 0.24845 and the conclusion holds across the bracket. Their phase-noise expression **is** `budget.phase(form="literature")` term for term: 12.307 mSNU at their 0.06 rad against the 12.6 printed; 12.6 inverts to 0.06071 rad | [Experiments · CV-MDI](/tests/exp_cvmdi) |
| Hajomer et al. 2024, discrete modulation at 10 GBaud (Optica **11**, 1197; [arXiv:2305.19642](https://arxiv.org/abs/2305.19642)) | Table 1's four rows in full — $M$, $V_M$, $T$, $V_{el}$, $\varepsilon$, $\beta$, $N$, both rate columns | **0 fitted** | Gaussian modulation at the same variance, the $M \to \infty$ limit of their protocol, is a ceiling at $1.171\times$, $1.413\times$, $1.135\times$ on the three lower rows and **saturated** at $M = 64$: 0.11363 against their 0.115. Their SKR column is symbol rate times $R_{\text{finite}}$, to 2.1% across all four rows | [Experiments · 10 GBaud](/tests/exp_shaped) |
| Lvovsky et al. 2001, the measured single-photon Wigner dip (PRL **87**, 050402; [quant-ph/0101051](https://arxiv.org/abs/quant-ph/0101051)) | $\rho_{11} = 0.553 \pm 0.013$, their four efficiency factors, their Abel-reconstructed $W(0,0) = -0.062$ and quoted $W(0,0) = -0.067 \pm 0.016$ | **0 fitted** | three published measurements of the one parameter agree to 0.78%, inside $\pm 0.013$ ([inversions](#three-ways)). Their Fig. 3 caption's "negative values require $\eta > 0.5$" returns exactly 0.5; reading the caption's *operator* definition instead of its printed function is excluded by their error bar at 2.09$\sigma$ | [Experiments · Fock](/tests/exp_fock) |

### Threshold detectors and qubits {#tier-b-qubit}

| Anchor | Pinned, from the paper | Free, and its range | Result | Report |
| --- | --- | --- | --- | --- |
| Gobby, Yuan & Shields 2004, BB84-WCP over 122 km (Appl. Phys. Lett. **84**, 3762; [quant-ph/0412171](https://arxiv.org/abs/quant-ph/0412171)) | $\mu = 0.1$ per clock, $\eta_{\text{Bob}} = 0.045$ including 5 dB apparatus loss, $P_e = 8.5\times10^{-7}$ error counts per clock, 2 MHz clock | **2 fitted** — $\alpha = 0.21$ dB/km in $[0.20, 0.22]$, spliced SMF-28 at 1550 nm; the phase-modulation floor $e_{\text{mod}}$ **bracketed** in $[0.0289, 0.0328]$, both endpoints carried through every prediction | visibility 87.89% against a measured 88.4%; QBER band 8.60–8.94% brackets the measured 8.9%; 101/122 km sifted-rate ratio 2.5473 against 2.5435; dark and stray light 0.374% at their $\alpha = 0.2$ against "less than 0.4%" | [Experiments · GYS](/tests/exp_gys) |
| Diamanti, Takesue, Langrock, Fejer & Yamamoto 2006, DPS over 100 km (Opt. Express **14**, 13073; [quant-ph/0608110](https://arxiv.org/abs/quant-ph/0608110)) | $\mu = 0.2$, 1 GHz clock, 2 dB interferometer, $d = 3.5\times10^{-8}$ per 100 ps window, their decomposition of the measured 3.4% QBER into 1% interferometric, 1.7% dark, 0.7% jitter | **4 fitted** — $\alpha = 0.2074$ dB/km in $[0.20, 0.22]$; the pre-window reading of their stated $\eta = 0.4\%$, the post-window reading *excluded* for needing $\alpha = 0.2404$–$0.2417$; $f(e) = 1.16$, qkd's default, which the paper never states; dead time 80 ns, the top of their 50–80 ns band | jitter-free QBER 2.691% against the 2.7% left after their 0.7% jitter share — **2.7% is nowhere in the paper**, which prints only 3.4% and its three parts — split 1.726% dark and 0.965% interferometric; secure rate 168.2 bit/s against 166; their Eq. (5) dead-time factor 0.8969 against 0.8934, from a stateful scan that never sees the formula; 10 km QBER 2.16% against 2.2% | [Experiments · Diamanti](/tests/exp_dps) |
| The same experiment, detector response modelled | the row above, plus three timing statements: 54% efficiency cost at a 100 ps window, 40% at 200 ps, the measured 3.4% QBER at 100 km | **3 fitted, exactly determined** — core FWHM $79.335$ ps, tail weight $f = 0.579$, tail constant $\tau = 295$ ps. Three parameters, three statements: **3.4% is pinned, not predicted**. A 79.3 ps core against their 66 ps optical pulse leaves 44.0 ps in quadrature for detector and electronics; $\tau = 295$ ps sits inside the 0.2–3 ns silicon diffusion-tail band; $f$ is the weakest of the three | the test is the **split** of the remaining 2.692%: 0.966% interferometric against 1%, 1.726% dark against 1.7%. The 0.709% is the fit target read back, not a prediction, nor is the 3.4% total. Secure rate unmoved at 168.19 bit/s | [Experiments · Diamanti jitter](/tests/exp_jitter) |
| Stucki et al. 2009, COW to 250 km (New J. Phys. **11**, 075003; [arXiv:0903.3907](https://arxiv.org/abs/0903.3907)) | $\mu = 0.5$, 625 MHz clock, SSPD efficiency 2.65% at a 5 Hz noise rate, 42.6 dB over 250 km, 0.164 dB/km | **1 fitted** — a baseline slot error, fitted at 100 km alone, bounded below 1%, landing at 0.847% | **1.904% at 250 km against a measured 1.9%**, nothing moved in the model; the cryostat ran "particularly low" for that run, lowering the true QBER, so the agreement is fortuitous in the safe direction ([misses](#misses)). Key rates *bounded*, not reproduced: 6.3% and 6.6% of the per-click ceiling | [Experiments · Stucki](/tests/exp_cow) |
| Korzh, Lim, Houlmann, Gisin, Li, Nolan, Sanguinetti, Thew & Zbinden 2015, COW over 307 km (Nat. Photonics **9**, 163; [arXiv:1407.7427](https://arxiv.org/abs/1407.7427)) | supplementary Table I column by column — five distances, block sizes, session times, secret fractions, rates — their 0.160 dB/km fibre, dark-count column and visibility | **2 fitted**, solved at 203 and 307 km and held — receiver factor 0.186628, baseline slot error 1.1537%. The factor is detector efficiency times the splitter's data-line share; their ">20% efficiency" caps the share at 0.9331, a **loose** cap (scoped below 150 K; all five rows run at 153–223 K). Refitting at 153 and 307 km moves the factor 2.8% and the baseline 5.8% | the two unfitted distances: **1.555% against 1.5%** at 153 km, **1.700% against 2.0%** at 256 km, across 26.5 dB. The climb from 1.5% to 3.5% is their published dark rate alone: halving it puts 307 km at 2.36%, doubling at 5.63%, both excluded. Their finite-key secret fraction is **bounded, not reproduced** — `cow_rate` is the plain Devetak–Winter difference where their Branciard–Gisin–Scarani expression carries an extra $(1-Q)$ on privacy amplification, so qkd sits above by exactly $Q(1 - h_2(e_{ph}))$ | [Experiments · Korzh](/tests/exp_cow_307) |
| Takesue, Nam, Zhang, Hadfield, Honjo, Tamaki & Yamamoto 2007, DPS at 10 GHz over 42.1 dB (Nat. Photonics **1**, 343; [arXiv:0706.0397](https://arxiv.org/abs/0706.0397)) | 10 GHz clock, SSPD efficiency and its 36% window cost, 50 Hz dark rate, 42.1 dB maximum loss, 105 km and 200 km points, their "approximately 4.1%" error threshold | **1 fitted** — $f(e)$, fixed by their threshold: `dps_rate` crosses zero at 4.5619% at the Shannon limit, 4.1487% at $f = 1.16$, 4.0123% at $f = 1.22$. Their figure selects $f = 1.16$; the neighbours sit 0.46 and 0.09 points away | **the one experimental row where qkd and the paper evaluate the same expression** — their Eq. (4) *is* the WTY Eq. (34). Zero crossing at **42.109 dB against 42.1 dB**. 105 km: 17.42 kbit/s against a measured 17, 2.5% high; dropping the 36% window cost gives 27.3 kbit/s, a 61% excess, so the cost belongs on the efficiency | [Experiments · Takesue](/tests/exp_dps_40db) |
| Lucamarini, Patel, Dynes, Fröhlich, Sharpe, Dixon, Yuan, Penty & Shields 2013, decoy-state BB84 with a biased basis (Opt. Express **21**, 24550; [arXiv:1310.0240](https://arxiv.org/abs/1310.0240)) | three intensities and six sifted counts, $p_X = 1/16$, $2.1\times10^{-5}$ dark probability per gate, 5.25% afterpulse probability, four distances and measured rates | **0 fitted on the background anchor.** **1 fitted** on the 50 km rate — $f(e) = 1.0995$ in $[1.0, 1.22]$ — plus 0.20 dB/km for the distance ladder | **the decoy family's first reproduced observables.** Dark probability over two gates plus afterpulsing on the measured $u$-line rate predicts a background yield of $2.9465\times10^{-4}$; their 0.001-photon $w$ line measures $2.9493\times10^{-4}$ — **0.099% apart, no free parameter**. `decoy_bounds` on their intensities returns a single-photon yield 1.09% under the infinite-decoy ceiling and phase errors above the true ones in both bases, 2.705% against 2.588% (Z), 2.028% against 1.948% (X). $p_X = 1/16$ alone gives 11.72% / 88.28% / 76.56% against their 11.7% / 88.3% / 76.6% | [Experiments · T12](/tests/exp_decoy_t12) |
| Enzer, Hadley, Hughes, Peterson & Kwiat 2002, six-state under a simulated eavesdropper (New J. Phys. **4**, 45) | measured $34.0 \pm 1.4\%$ total BER under full intercept-resend, 25% for BB84 on the same data, 1.7% undisturbed, 11%/7% partial-attack ratio, 33 s⁻¹ raw rate, 55 650 sifted bits over 94 min | **0 fitted on the anchor.** **1 fitted** on the partial-attack leg — the Brewster-slab measurement strength, never stated | 33.333% against $34.0 \pm 1.4\%$ at 0.48$\sigma$, and 25.000% for BB84, both from the Bell-diagonal weights, flat in the attack angle. "BB84 has the higher yield below ~8–10%" comes out at 8.340%; "never any yield above 10–15%" brackets 11.003% and 12.619%. The fitted strength reads 0.21 from the average and 0.22 from the peak; the peak-to-average ratio is 3/2 at *any* strength against a measured 1.571 | [Experiments · six-state](/tests/exp_sixstate) |
| Jeong, Kim & Kim 2014, SARG04 against BB84 on one rig over 1.27 km (Laser Phys. Lett. **11**, 095201) | $\eta_{\text{Bob}} = 0.40$, $\alpha = 3$ dB/km at 780 nm, dark probability $(3.3\pm0.6)\times10^{-5}$ per detector, $\mu = 0.19$, figure 2 sifted-rate ladder, measured QBERs | **1 fitted** — channel visibility, over-determined **four ways** at a spread of 1.97% ([inversions](#three-ways)) | misalignment *raises* the SARG04 conclusive rate to $e_{\text{det}}/2 + 1/4$, so at $V = 0.954$ the SARG04-to-BB84 sifted-rate ratio is 0.5230, and all six figure 2 pairs sit above $1/2$, averaging 0.52075. Their measured sifting $0.25 \pm 0.01$ against BB84's $0.50 \pm 0.01$ is the perfect-frame conclusive probability, exactly $1/4$ | [Experiments · SARG04](/tests/exp_sarg) |
| Hughes, Morgan & Peterson 1999, plain B92 over 48 km of installed fibre ([quant-ph/9904038](https://arxiv.org/abs/quant-ph/9904038); J. Mod. Opt. **47**, 533) | 22.9 dB attenuation, $\mu = 0.63$ and 0.39, measured visibility, 9.3% and 17.8% BERs, the fringe scan's 1102 errors in 21 524 conclusive bits | **1 fitted** — background $1.535\times10^{-4}$ per gate: $1.92\times10^{-5}$ per 730 ps window, a 26.3 kHz noise rate inside their stated 10–100 kHz | zero-parameter leg: visibility fixes the interferometric error at $(1-V)/2 = 0.505\%$, 5.43% of their 9.3%, against their "approximately 90% of the errors are attributable to detector dark counts". Photon statistics with nothing fitted: 53.26% empty pulses against 53%, 28.21% multiphoton against 28%, 18.24% against 18%. Rates **bounded, not reproduced**, at 11.6% and 5.7% of the conclusive ceiling — their own position: "we have not implemented a full privacy amplification stage" | [Experiments · B92](/tests/exp_b92) |
| Gordon, Fernandez, Townsend & Buller 2004, B92 QBER curve to 11.85 km at 100 MHz (IEEE J. Quantum Electron. **40**, 900; [quant-ph/0605222](https://arxiv.org/abs/quant-ph/0605222)) | 180 background counts/s, 2.2 dB/km at 850 nm, nine fibre lengths, 5 ns and 9 ns QBER columns | **1 fitted** — polarisation leakage 0.256% per beam splitter, a 25.9 dB extinction ratio, fitted at zero length alone. The published background already carries 36.1% of the 0.4% measured there | the 5 ns QBER at every length to 9.96 km, across 0.4% to 15.6%, worst miss 0.73 points. The 9 ns window shares only the dark rate and fibre loss: its zero-length fit gives 0.99474 against 0.99488 and tracks to 0.54 points. The whole climb is the published 180 counts/s ([why](#three-ways)): halving it gives 9.41% at 9.96 km, doubling 23.85%, against a measured 15.6% the published value reads as 15.72%. The 1 GHz rows are excluded, and the exam says why | [Experiments · B92 curve](/tests/exp_b92_curve) |
| Naik, Peterson, White, Berglund & Kwiat 2000, E91 with a measured CHSH value beside its QBER (PRL **84**, 4733; [quant-ph/9912105](https://arxiv.org/abs/quant-ph/9912105)) | $S = -2.665 \pm 0.019$, $S' = -2.644 \pm 0.019$, bit error rate $3.06 \pm 0.11\%$, 24 252 raw key bits at 10.1 s⁻¹, 15 444 published secret bits, measured 24.551% error-correction leakage. ⚠️ **The key ladder is the published PRL's; the e-print disagrees**: quant-ph/9912105, its only version, prints 17 452 error-free bits compressed to 12 215 at 5.1 s⁻¹ where the row reads 18 298 / 15 444 / 6.435. CHSH values, bit error rate and the 24 252 raw block agree | **0 fitted** | the Bell-diagonal bridge $Q = (1 - \lvert S\rvert/2\sqrt2)/2$ joins their independent CHSH and BER measurements: mean violation reads **3.0746%** against $3.06 \pm 0.11\%$, 0.13$\sigma$. $S$ and $S'$ differ by 0.78 of their combined error bar. Their three eavesdropper statements are one: the bridge turns $\lvert S_{\text{Eve}}\rvert \le \sqrt2$ into their 25% BER and $S = 2$ into $(2-\sqrt2)/4 = 14.645\%$ at an intercepted fraction $2-\sqrt2 = 58.579\%$ — their "58.6%" and "<15%". Count rate, sharing no parameter with the error model: 25.263% of pairs against Table I's $1/4$, 10.105 bit/s against 10.1 | [Experiments · E91](/tests/exp_ekert) |
| Ling, Peloso, Marcikic, Scarani, Lamas-Linares & Kurtsiefer 2008, an E91 rate priced by a measured violation (PRA **78**, 020301(R); [arXiv:0805.3629](https://arxiv.org/abs/0805.3629)) | $S \approx 2.5$ over the run, $10^7$ bit over 9.5 hours, "around 300 bit/s", 0.5% accidental coincidences | **0 fitted** | the violation as a depolarised singlet fixes the bit error at 5.806%, and the rate equation predicts **308.41 bit/s** at the Shannon limit against a measured 292.40–300 bit/s: 2.8% headroom for their real reconciliation. Inverting 300 bit/s gives 5.899%, 0.093 points from the violation's 5.806%. BBM92 on the same data pays 813.33 bit/s against the CHSH price's 308.41, a factor of 2.637 | [Experiments · CHSH](/tests/exp_chsh) |
| Honjo, Nam, Takesue, Zhang, Kamada, Nishida, Tadanaga, Asobe, Baek, Hadfield, Miki, Fujiwara, Sasaki, Wang, Inoue & Yamamoto 2008, BBM92 over 100 km (Opt. Express **16**, 19118) | itemised receiver chain, two brightnesses, measured back-to-back and 100 km QBERs, separately measured two-photon visibilities, sifted rates | **1 fitted** — misalignment, from the 2.35% back-to-back QBER alone, at 0.4244% | predicts **6.702% at 100 km against a measured 6.91%**, 0.21 points low across 21 dB and a $3.5\times$ brightness change. Two legs share nothing with the fit: measured visibilities give the energy-basis QBER through $e = (1-V)/2$ exactly at 0 km (3.100% against 3.1%) and to 0.09 points at 100 km; their Eq. (11) accidental QBER $\mu_t/(2(1+\mu_t))$ agrees with Ma, Fung & Lo's Eq. (10) to 2.2% and 5.8%, the difference their dropped finite-efficiency factor. **No arXiv version**: [W3](#register) | [Experiments · pairs](/tests/exp_pairs) |
| Rubenok, Slater, Chan, Lucio-Martinez & Tittel 2013, MDI-QKD over deployed fibre (PRL **111**, 130501; [arXiv:1304.2463](https://arxiv.org/abs/1304.2463)) | Table I and the supplementary gain table — four setups across 9.1 dB, two bases, signal and decoy intensities, sixteen measured gains, X-basis error rates. The supplement prints six intensity pairs per setup, not a $3\times3$ grid: signal-decoy and decoy-signal cells are **absent**, not substituted; the $\sim\!0.5$ decoy-vacuum values (0.511(5), 0.50(1), 0.51(1), 0.502(6)) are used as printed, two sitting up to $2.2\sigma$ from the structural half. $Q_{vv} = 7.1\times10^{-10}$, HOM visibility $47 \pm 1\%$ | **1 fitted** — misalignment, from setup 2's rectilinear QBER alone, 0.028870. **1 pinned input** — detector efficiency 0.145, from Chan et al., Opt. Express **22**, 12716 (2014) ([arXiv:1204.0738](https://arxiv.org/abs/1204.0738)) | all sixteen gains within $0.876$–$1.051\times$ of measurement, mean $0.981\times$, nothing fitted. The fitted misalignment puts all four diagonal-basis QBERs within 0.8 points. The pinned efficiency is **reproduced by this paper's data**, each gain solved for it alone ([inversions](#three-ways)). $Q_{vv}$ inverts to a dark probability $1.884\times10^{-5}$ against the companion's $1.83\times10^{-5}$, 3.0% apart; dropping the two-detector factor 2 gives $1.332\times10^{-5}$, 27% off. Xu et al.'s decoy bound on their diagonal gains agrees with the forward model to 2.1% at all four setups. Key rates **bounded, not reproduced**, at their request | [Experiments · MDI](/tests/exp_mdi) |
| Detector attacks on deployed hardware — Lydersen et al., Nat. Photonics **4**, 686 (2010) ([arXiv:1008.4593](https://arxiv.org/abs/1008.4593)); Zhao, Fung, Qi, Chen & Lo, PRA **78**, 042333 (2008); Weier et al., New J. Phys. **13**, 073024 (2011) ([arXiv:1101.5289](https://arxiv.org/abs/1101.5289)) | Lydersen's $P_{\text{never}} = 647$ and 697 µW, $P_{\text{always}} = 808$ and 932 µW on a blinded Clavis2; Zhao's Table I counts at two time shifts; Weier's Table I, seven blinding intensities against Eve's information and both QBER columns | **0 fitted** | Lydersen's ratio 1.4405, below the 2 their Eq. (1) requires; CW blinding powers 397 and 765 µW. Zhao's equalising mixture: **23.031% and 3479.07 detections** against 23.0% and 3479; over same-basis tables **5.681%** against their overall 5.68%; the per-shift QBERs 6.135% and 5.365% come back out of the same table. Weier's seven $I_{EB}$ bounded from above throughout, the top point to **0.023 bit**; their $I_{EB}$ and Bob–Eve QBER columns obey $I_{EB} = 1 - h(\text{QBER}_{BE})$ to $6\times10^{-4}$, asserted | [Attacks · Threshold](/tests/attacks_threshold) |

### The network layer {#tier-b-network}

| Anchor | Pinned, from the paper | Free | Result | Report |
| --- | --- | --- | --- | --- |
| Sharma, Vilashini, Krishnan, Gayathri, A. K. Singh, V. P. Singh, Ramanathan, Mandayam & Prabhakar 2026, the MAQAN testbed at Chennai (Quantum Inf. Process. **25**, 19; [doi:10.1007/s11128-025-05031-x](https://doi.org/10.1007/s11128-025-05031-x)) | five nodes, three spans, Table 1 losses and key rates, Fig. 4 raw-key time series, COW and DPS only | **1 fitted** — effective detected-pulse rate 3.61 MHz, from Fig. 4's last daily average of 20.75 kbps over the 7.5 dB span: 0.36% of the 1 GHz clock, 11.6% of the group's 31.25 MHz COW gate rate, inside both ceilings. **Plus five INVENTED**, labelled so in the exam: `ETA_SPAD`, `DARK_SPAD`, `VIS_DLI`, `MISALIGN`, `CLOCK`, none published. Sensitivity to the first: [below](#three-ways) | every hop clears the two thresholds their controller reroutes on. The fitted rate carried to their best span **overshoots by $40.72\times$** ([misses](#misses)) | [Network · MAQAN](/tests/network_maqan) · [Network · Tier B](/tests/network_tierb) |

### One free parameter, inverted several independent ways {#three-ways}

The strongest claim a one-parameter row can make: separately published quantities
each pin it.

| Anchor | The one free parameter | Independent inversions | Spread |
| --- | --- | --- | --- |
| Gobby, Yuan & Shields 2004 | fibre attenuation $\alpha$. Neither printed value measures this link: "the specified value of fiber attenuation $\alpha = 0.2$ dB/km" is the datasheet figure of their Fig. 2 curve; the sifted-rate ladder "decreases with increasing fiber length at a rate of $\sim$0.21 dB/km" is what is fitted | 0.20826 (122 km visibility), 0.20941 (101/122 km sifted-rate ratio), 0.20974–0.21211 (122 km QBER) | 1.9%, inside $[0.20, 0.22]$, bracketing their $\sim$0.21 |
| Diamanti et al. 2006 | the same | 0.20763 (jitter-free QBER), 0.20673 (dark-count share), 0.20799 (166 bit/s secure rate) | 0.61%, inside $[0.20, 0.22]$ |
| Jeong, Kim & Kim 2014 | channel visibility | 0.954 (their channel measurement), 0.95850 (sifted-rate ladder, a *count* rate, no error model), 0.94000 (BB84 QBER), 0.94737 (SARG04 QBER) | 1.97% |
| Lvovsky et al. 2001 | single-photon fraction $\rho_{11}$ | 0.55 (marginal-distribution fit), 0.553 (pattern-function sampling), 0.548695 (their Abel-reconstructed $W(0,0)$ through qkd) | 0.78%, inside their $\pm 0.013$ |
| Rubenok et al. 2013 | detector efficiency, imported from a companion paper | sixteen inversions, one per measured gain: 0.14140 to 0.15578 | 10.2%, straddling the companion's 0.145 |
| Naik et al. 2000 | none — over-determined with **no** free parameter | $S$ alone 2.889%, $S'$ alone 3.260%, the mean 3.0746%, against a measured $3.06 \pm 0.11\%$ | bracketing the measurement, 1.55$\sigma$ below and 1.82$\sigma$ above |

| Leg carrying the argument | Why the free parameter cannot absorb it |
| --- | --- |
| GYS's sifted-rate ratio | a *count rate*, sharing no parameter with the error model, and sharp: $\alpha = 0.20$ gives 2.4766, $\alpha = 0.22$ gives 2.6045, against a measured 2.5435 |
| Stucki's dark-count takeover | the fitted baseline is flat in distance and cannot rise from 0.85% to 1.9%; the published 5 Hz dark rate does. It is fitted at 100 km, where dark counts are 0.0026% of the QBER. Halving the dark rate moves 250 km to 1.38%, doubling to 2.91% |
| Gordon's dark-count takeover | without the background the model is flat at 0.256% at every length to one part in $10^4$: the climb from 0.4% to 15.6% is the published 180 counts/s |
| Honjo's visibilities | measured separately from the misalignment fit, they give the energy-basis QBER through $e = (1-V)/2$ at both distances |
| Rubenok's sixteen gains | nothing fitted in any; the fitted misalignment enters only the error rates |
| MAQAN's detector efficiency | enters fit and prediction alike: sweeping it 0.1 to 0.3, wider than any InGaAs SPAD band, moves the overshoot from $40.86\times$ to $40.57\times$, a 0.72% spread |

**The decoy row's Tier B cell is provenance, not reproduction.**
[`test_gys_tabulation`](/tests/exp_gys) traces every constant of the Tier A "GYS
parameter set" to Gobby, Yuan and Shields's own measurements.

| GYS constant | Provenance |
| --- | --- |
| $Y_0 = 1.7\times10^{-6}$ | exactly twice their per-clock error count $P_e = 8.5\times10^{-7}$, one per interferometer port |
| $e_{\text{det}} = 3.3\%$ | their short-distance QBER plateau |
| $\eta_{\text{Bob}} = 4.5\%$ | their stated receiver efficiency |
| $\alpha = 0.21$ dB/km | the slope of their measured sifted-rate ladder, not the 0.2 their curve is drawn at |

GYS sent no decoy states, so this is theory on measured hardware, not a decoy
observable. The nearest check is the MDI row: Xu, Curty, Qi and Lo's *joint* decoy
bound on Rubenok's measured gain grid, a two-sender inversion.

### GYS's $\sim$50 km limit is a model choice, not a missing number {#gys-pns}

GYS state that the Brassard–Lütkenhaus–Mor–Sanders condition — Bob's measured bit rate
must exceed Alice's multiphoton emission rate — "imposes a limit of $\sim$50 km for
the current system". At $\mu = 0.1$, $p_{\text{multi}} = 1 - e^{-\mu}(1+\mu) =
4.6788\times10^{-3}$:

| Reading | Condition | $\alpha = 0.20$ | $\alpha = 0.21$ |
| --- | --- | --- | --- |
| $\eta_{\text{Bob}}$ handed to Eve | $\mu\,\eta_{\text{Bob}}\,10^{-\alpha L/10} > p_{\text{multi}}$ | no distance — ratio 0.962 at $L = 0$ | no distance |
| $\eta_{\text{Bob}}$ trusted, BLMS Eq. (5) as written | $\mu\,10^{-\alpha L/10} > p_{\text{multi}}$ | 66.5 km | 63.3 km |
| the same, with BB84's $\tfrac12$ sifting | $\tfrac12\mu\,10^{-\alpha L/10} > p_{\text{multi}}$ | 51.4 km | **49.0 km** |

| Question | Answer |
| --- | --- |
| Which row reads GYS's words? | The third: "the bit rate *measured by Bob*" is the sifted raw rate they plot. It **brackets $\sim$50 km from both sides** across the two $\alpha$ the experiment supports. BLMS state Eq. (5) in the $\eta_B = 1$ form and call handing $\eta_B$ to Eve the "somewhat conservative" alternative — the [trusted-versus-untrusted split](/guide/security#trusted-vs-untrusted-is-a-security-model) |
| Does any row recover GYS's arithmetic? | No. Row three adds the exact $1 - e^{-\mu}(1+\mu)$ for BLMS's $\mu/2$ proxy and BB84's $\tfrac12$ sifting; BLMS's own convention lands in the sixties of kilometres. Two readings bracketing a one-digit figure reproduce the *conclusion*, not the calculation |
| What survives of the objection? | GYS's sentence omits $\eta_{\text{Bob}}$ from the multiphoton side |
| Is $\sim$50 km a key-rate bound? | No — a **multiphoton-rate condition**. The **GLLP key-rate bound** on the same parameters reaches **40.5 km** with $\mu$ optimised per distance and nothing at GYS's $\mu = 0.1$. `test_gys_nondecoy_deficit` pins it |

### Diamanti's full 3.4% QBER

A purely Gaussian jitter is **excluded by the data**. Fitted to the 100 ps window
statement alone it needs a **192.13 ps** FWHM core, predicts **22.03%** loss at
200 ps against a measured 40% — a **17.97-point** miss on an unfitted number — and
0.000% jitter error against 0.7%, returning the total to the jitter-free 2.691%. The
model carries a one-sided exponential diffusion tail instead. The fit:
[Tier B row](#tier-b-qubit); window loss and bin leak:
[Protocol coverage](/guide/protocols#jitter-shape); the 10 km QBER:
[declared miss](#misses).

## Declared misses {#misses}

A prediction the fit did not reach, **asserted tightly so it cannot be quietly
closed**.

| Row | The miss, and its size |
| --- | --- |
| **Hajomer 2024's key rate** | a documented tolerance relaxation: at $T = 10^{-1.54}$ the detection term $\chi_{\text{het}}/T \approx 74$ SNU swamps $\chi_{\text{line}}$, so no budget fit reaches the target, even at $\xi = 0$ |
| **GYS's sifted rate at short range** | lumped collection factor 0.936 at 4.4 km against 0.655 at 101 and 122 km: the model over-predicts the long-distance rate by $1.43\times$ relative to the short. The paper names polarisation drift, which "reduces the bit rate, but does not degrade the QBER"; qkd models no polarisation on this path, so every QBER-side prediction survives |
| **Diamanti's 10 km sifted rate** | 2.55 Mbit/s against a measured 2, a 28% excess needing about 1.1 dB of unpublished loss. Adding it moves the QBER from 2.16% to 2.35% against 2.2% |
| **Diamanti's 10 km QBER with jitter** | unfitted, **2.795% against 2.2%**. A 200 ps window accepts roughly twice the neighbour's tail while collecting $1.30\times$ more of its own, so modelled jitter error *rises* from 0.71% to 1.13% between the two detector settings while their Fig. 5 baseline *falls*, 1.7% to 1.5%. Either the settings carry different timing responses, or part of their 0.7% is a mechanism this model lacks. Jitter-free: 2.162%, bracketing 2.2% from below |
| **Stucki's 250 km detector state** | the cryostat ran colder for that run, so the true prediction sits slightly *below* 1.904%: fortuitous agreement, in the safe direction |
| **Enzer's sifting count** | 55 650 sifted bits over 94 min at 33 s⁻¹ raw is **29.900%** where three-basis sifting gives a third: a **10.3% deficit**. Their words are "roughly one-third"; the <0.5% double-pair events they bound do not cover it |
| **Jeong's absolute scale** | the figure 2 ladder needs end-to-end efficiency **0.17770** where $\eta_{\text{Bob}} = 0.40$ and $\alpha = 3$ dB/km give **0.16636**, a **6.8% excess**; both printed to two figures, the ratio leg immune. **The sign reads a hedged sentence**: their components $\eta_{\text{opt}}\eta_{\text{det}} = 0.72 \times 0.6$ give 0.432, not 0.40, and at 0.432 the ladder asks **1.1% less** than the optics supply. Only the printed 0.40 is asserted |
| **Jeong's two QBERs** | **2.502% and 4.753%** against a measured ~3% and ~5%, low by **0.50 and 0.25 points**. The shortfall is the paper's own — its printed closed forms carry it — and their dark-count subtraction rules background out |
| **Hughes's second intensity** | background fitted at $\mu = 0.63$ predicts **13.31% at $\mu = 0.39$ against 17.8%**, 4.49 points short; fitting the other way over-predicts the first point, 12.86% against 9.3%. Right direction, wrong size |
| **Gordon's last two lengths** | **22.24% and 27.14% against 24.3% and 31.8%**, short by 2.06 and 4.67 points. Their raw counts fall 10.7% between 9.96 and 11.07 km (1109 to 990 per second, then 611 at 11.85) where 2.2 dB/km over 1.11 km asks 43.0%: the ladder flattens exactly where the QBER runs away. Strictly decreasing throughout, so a slope anomaly, not a non-monotonicity. The 9 ns column misses the same two points by 6.82 and 5.83 |
| **Korzh's sifted count rate** | the fitted receiver supplies **0.672 and 0.689** of $n_{\text{cpp}}/t_{\text{cpp}}$ at 203 and 307 km, the fit distances, where a per-click ceiling owes at least 1, and over-supplies at 104 and 256 km (1.947, 1.905). Comparing against $r_{\text{sec}}$ hides it by $f_{\text{sec}} \approx 0.3$. `test_korzh_ceiling` pins it |
| **Korzh's 104 km QBER** | **1.526% against 2.40%**, 0.87 points short. Closing it needs extra background of 1.84% of the detected count rate — an afterpulse probability the paper names ("for a given dead-time, the after-pulse probability increases exponentially with reducing temperatures") but never tabulates. 0.63% at 256 km and $-0.11\%$ at 153 km: all three inside an NFAD band |
| **Lucamarini's T12 enhancement** | measured T12-over-BB84 at 50 km is 73.0% from the published pair, quoted 73.5%; the model gives **66.5%, 6.5 points short**, charging error correction on the basis the counts land in. T12 puts 99.6% of its key in the noisier Z basis while BB84 splits evenly with the cleaner X, eroding the 76.6% sifting advantage further in the model |
| **Naik's published key length** | their 24.551% leakage then the CHSH price leaves **10 009 secret bits against 15 444 published**, 64.8%. Their leakage is 1.244 times the Shannon limit at 3.06% |
| **Honjo's count rate** | sifted rate falls $51.6\times$ between the runs where the model falls $31.3\times$: 0 km at $0.773\times$ their 29.4 bit/s, 100 km at $1.275\times$ their 0.57. QBER-side predictions survive, as on the threshold-detector rows |
| **Rubenok's rectilinear QBER drift** | 0.0323 exactly at setup 2, where fitted, but **0.04429 against 0.053** at 18.2 dB, the deficit larger on the weaker decoy at every setup: the shape of a channel background absent from the signal, a term Ma & Razavi's model lacks |
| **Zhang's phase-noise budget hole** | their $(1-\kappa) = 7.6\times10^{-5}$ accounts for **72.8% of measured $\xi$ at 27.27 km but 7.2% at 202.81 km**, against their sentence naming residual phase noise the main contribution. The other 92.8% has no row; `src/tlo.rs` narrows the hole without closing it ([below](#tlo-bears)) |
| **The certified COW′ reach** | **93 and 103 km against the source's 103 and 121 km**, in the honest channel, not in either bound. ⚠️ **The proposed repair is refuted; do not re-attempt it as written**: it read `sdp_gains` as putting half a pulse's flux in the interference slot and claimed the whole pulse — 3 dB, about 15 km — closes the gap. Cao, Sun, Li, Lu, Yin & Chen's constructive $|\alpha\rangle|\alpha\rangle$ gain is **0.98–0.99 of $\mu\cdot\eta$ at all four distances**, and their "additional 3 dB loss" is exactly the discarded side bins: `ports` is right, the instrument puts $\mu$ and not $2\mu$ in that slot. `test_cow_ports` pins it |
| **MAQAN's secure key rate** | **40.7 kbps against the 1 kbps Table 1 publishes as the network's highest: a $40.72\times$ overshoot**, unsafe for a rate prediction. qkd bounds one asymptotic pulse; the published figure is delivered finite-size key after LDPC reconciliation, privacy amplification, authentication and a distillation engine on a few-hundred-millisecond cadence, none modelled. Fitting from the March end of Fig. 4 instead of July gives $39.5\times$: the figure read moves the overshoot 3.0% |

| Declared reading or engine limit | |
| --- | --- |
| Hajomer 2025's relay efficiency | Table 1 lists $\tau_A = 1$ with the relay's 94% in its own column: the calibrated-out reading the anchor uses. Charging the 94% to Eve as arm loss — conservative, the relay being untrusted — takes the equivalent noise to 6.0441 and the rate **negative**, $-0.13405$ bit/symbol |
| Hajomer 2024's 10 GBaud rows, untrusted | `dm_rate` folds $\eta$ and $v_{el}$ in as untrusted only, and at their parameters is **negative on all four rows**. The obstruction is the trust model, not the constellation: the same swap in the Gaussian layer costs 0.335 bit/symbol at the 64-QAM row, 2.91 times their whole asymptotic rate ([refusal](/roadmap#refusals)) |
| Ling's anisotropy | they attribute the low $S$ to residual distinguishability on the $\pm 45^\circ$ correlation, so their key basis is the clean one and the true QBER sits *below* the depolarising reading; the ceiling only loosens, so 2.8% is a ceiling, not an equality |
| Naik's privacy amplification | the collective-attack CHSH price on their violation is **0.34178 bit per raw bit against the 0.11768 their privacy amplification charged**, 2.90 times as much. Acín et al. is seven years later; the engine's key is the smaller, the safe direction |

## What the transmitted-oscillator module adds to two existing rows {#tlo-bears}

Lodewyck et al. 2007 and Zhang et al. 2020 were measured on transmitted-oscillator
hardware and are anchored through the locally generated engine, legitimately: both
pin $\xi$ and reach `cv_rate` directly. `test_anchor_invariant` asserts neither
moves. `src/tlo.rs` sizes the gap those rows do not carry.

| Reading | |
| --- | --- |
| Lodewyck's delay | a 108 Hz laser across their 400 ns multiplexing delay accounts for their **entire** measured excess noise of 0.005 SNU; anything narrower leaves the phase term a minority contributor |
| Zhang's delay | their residual phase noise at their 100 Hz linewidth is a 121 ns signal-to-oscillator delay, inside the 200 ns symbol period of their 5 MHz clock: self-consistent |
| Zhang's share | the phase row covers **7.2%** of measured excess noise at 202.81 km: the [declared miss](#misses) |
| What the rest is worth | 2.6 parts per million of shot-noise unit: at 32.45 dB a unit error is referred in through $1/(\eta T)$ |
| What closing it costs | a part in $2.6\times10^{6}$ to one sigma takes about $3.9\times10^{11}$ vacuum samples, 21 hours at 5 MHz. Real-time shot-noise monitoring narrows a **uniform** rescale as one over the root of the sample count and closes no loophole: its null is forgeable (Huang, Kunz-Jacques, Jouguet, Weedbrook, Yin, Wang, Chen, Guo & Han, Phys. Rev. A **89**, 032304 (2014): the attack biases the estimate *even if it is done in real time*) |

None of it is a Tier B green cell or re-grades either row.

## Recorded but not pinned {#unpinned}

Comparisons in the repository's record that are **not exams**.

| Comparison | Standing |
| --- | --- |
| `dm_secure` against Lin & Lütkenhaus's ideal-detector curve at $\xi = 0.01$ | 0.4214 against a published 0.4541 at 0 km, $5.979\times10^{-3}$ against $6.049\times10^{-3}$ at 60 km, both **below** — the right side for a lower bound run with fewer Frank–Wolfe iterations. Their figure data is a recovered array, which [the register](#register) bars from citation as a published constant |
| Wen, Tamaki & Yamamoto 2009's 6.09% DPS zero crossing | named in [E1](#register), computed in a scratch script only |
| Lim's Fig. 1 | a **digitisation**: reach **134.6 km** at $n_X = 10^4$ where the text says 135; $R(10^9)/R(10^7) = 1.760$ at 100 km where the text says ~1.75. Curves [below](#lim-fig1) |

### Lim's Fig. 1, digitised — the unclaimed Tier B target {#lim-fig1}

At the paper's $\eta_{Bob} = 0.10$, $p_{dc} = 6\times10^{-7}$,
$p_{ap} = 4\times10^{-2}$, $e_{mis} = 5\times10^{-3}$, $f_{EC} = 1.16$, 0.2 dB/km,
$\epsilon_{cor} = 10^{-15}$, $\epsilon_{\text{sec}} = \kappa\ell$. Bit/pulse:

| $n_X$ | 0 km | 20 km | 50 km | 100 km | 140 km | max reach |
| --- | --- | --- | --- | --- | --- | --- |
| $10^4$ | 4.47e-04 | 1.71e-04 | 4.13e-05 | 3.15e-06 | | 134.6 km |
| $10^5$ | 1.58e-03 | 6.11e-04 | 1.50e-04 | 1.29e-05 | 9.13e-07 | 156.1 km |
| $10^6$ | 3.32e-03 | 1.29e-03 | 3.17e-04 | 2.87e-05 | 2.79e-06 | 167.5 km |
| $10^7$ | 5.19e-03 | 2.02e-03 | 4.99e-04 | 4.64e-05 | 5.30e-06 | 173.2 km |
| $10^8$ | 7.21e-03 | 2.81e-03 | 6.94e-04 | 6.55e-05 | 7.88e-06 | 177.1 km |
| $10^9$ | 8.85e-03 | 3.45e-03 | 8.54e-04 | 8.17e-05 | 1.03e-05 | 179.6 km |

The $n_X = 10^4$ row is optimisation-limited, not a formula discrepancy.

| Between this and an exam | |
| --- | --- |
| Detection model | Lim's $p_{ap}$ enters as $R_k = D_k(1 + p_{ap})$ and $p_{ap}D_k/2$ inside $e_k$; qkd's gains come from `decoy_gain` or the Papapanos form under `q.DeadTime(afterpulse=…)`, which is why [E6](#register)'s detection-model typo never reached qkd |
| Optimisation | the curve optimises five parameters |
| $\epsilon_{\text{sec}}$ | Lim's is the fixed point $\kappa\ell$; `q.KeyBlock` takes it as an input |

## Errors in published work, and the warnings beside them {#register}

| Class | |
| --- | --- |
| **E** | an error in published work |
| **W** | a claim of qkd's that did not survive checking (W1, W2), or a trap that will otherwise be *read* as an error (W3–W11). No W entry names a defect in anyone's paper; W10 and W11 are traps this repository walked into |

**Recovered data is not a published constant.** Numbers digitised from a figure or
decoded from its embedded vector arrays may be checked against, never cited as an
anchor.

| # | Source | Verdict | Stake in qkd |
| --- | --- | --- | --- |
| **E1** | Waks, Takesue & Yamamoto, PRA **73**, 012344 (2006), Eq. (37) | an inverted sign on the privacy-amplification term forces $R \le 0$ everywhere, most negative where their Fig. 3 plots the highest rate. **Confirmed independently**: Wen, Tamaki & Yamamoto, PRL **103**, 170503 (2009) print the corrected structure; Ramanathan et al., arXiv:2305.11822 Eq. (9) inherit the defect while contradicting their own Eqs. (4) and (7) | **load-bearing.** qkd evaluates the corrected form, which crosses zero at $e = 6.09\%$ — Wen et al.'s number — where the printed one has no crossing |
| **E2** | Denys, Brown & Leverrier, Quantum **5**, 540 (2021), §10 | a factor 2 in the $\xi$ term of the AWGN noise-variance annotation, which should read $\mathcal N(0, 1 + T\xi/2)$, the heterodyne beamsplitter halving the excess noise too. **Their printed SNR is correct.** Confirmed against Laudenbach Eq. (6.3) | none; §10 is an aside, and the SDP bound and §11 rates use the correct SNR |
| **E3** | *Numerical Recipes 3rd ed.*, `gauher` | wrong nodes from $n \approx 184$, collapse at $n = 200$. Reproduced against `scipy.special.roots_hermite` | none. The rule ships at 64 nodes with a Sturm-bracketed fallback |
| **E4** | Tamaki & Lo, PRA **73**, 010302(R) (2006), the SARG04 two-photon parenthetical | quoted at the wrong minimiser; blind verification against the LaTeX source, single-sourced | none; **not a cross-check** |
| **E5** | Fung, Tamaki & Lo, PRA **73**, 012337 (2006), Eq. (39) | a binary-entropy subscript on a conditional entropy, self-inconsistent within the paper | **load-bearing.** `src/sarg.rs` implements Eq. (39) |
| **E6** | Lim, Curty, Walenta, Xu & Zbinden, PRA **89**, 022307 (2014) | two printing errors, found by implementing the paper | one load-bearing; qkd implements the corrected $\lambda_{EC}$ |
| **E7** | Qin, Kumar & Alléaume, PRA **94**, 012325 (2016), Eq. (17) | $\mathrm{Var}(X_{B,\text{lin}})$ printed at half the value their Appendix B Eq. (B14) derives; at zero displacement, where the paper states 2.1 SNU, the printed expression returns $-3.2955$. **Five independent lines of evidence**, including the authors' thesis printing it correctly | `qkd.attacks`'s saturation family ships the corrected form, agreeing with `sat_estimate` to $10^{-14}$ at every displacement |
| **E7b** | the same paper, Eq. (19) | a factor 2 lost under a root; the authors' commented-out LaTeX carries it | same |
| **E7c** | the same paper, the sentence before Eq. (19) | a spurious 1/2 on a covariance, in the thesis too. **Inert** | none |
| **E8** | Matsumoto, arXiv:1301.5083, introduction | a 3.5% B92 depolarising threshold attributed to a paper that does not carry it; blind verification, a mis-attribution | none; not a cross-check |
| **E9** | Tamaki & Lütkenhaus, arXiv:quant-ph/0308048 — the **e-print**'s depolarising channel | drops its $p/3$ and is not trace-preserving. ⚠️ the published PRA **69**, 032316 (2004) was not checked | `src/b92.rs`'s `b92_channel` implements the corrected reading |
| **E10** | Jouguet, Kunz-Jacques & Diamanti, PRA **87**, 062313 (2013), Eq. (3) | a sign error, self-inconsistent within the paper | none |
| **E11** | Takesue, Nam, Zhang, Hadfield, Honjo, Tamaki & Yamamoto, Nat. Photon. **1**, 343 (2007), Eq. (4) | confirmed in [arXiv:0706.0397](https://arxiv.org/abs/0706.0397)v1, the only version, and inconsistent with the paper's stated threshold. ⚠️ the published text was not accessible | none. **Sibling of E1**, corroborating its $\tau - fh$ reading |
| **E12** | Korzh et al., Nat. Photon. **9**, 163 (2015), supplementary Table I | $r_{\text{sec}} = n_{\text{cpp}} f_{\text{sec}} / t_{\text{cpp}}$ holds on three rows of five; the paper's Fig. 3b splits which column is at fault in each | none; columns read individually |
| **E13** | Curty, Xu, Cui, Lim, Tamaki & Lo, Nat. Commun. **5**, 3732 (2014), the $\Gamma_{k,v,v'}$ equation | confirmed in v1, v2 **and** the published Supplementary, and inconsistent with the paper's $\Gamma_{k,v}$ three equations later. Not the maximisation declared, erring in the **insecure** direction on most admissible rungs. No share is quoted: three scans of the same box read 39%, 57% and 89%, the criterion being antisymmetric under $v \leftrightarrow v'$ | **load-bearing.** `src/mdi.rs` runs the sign-correct maximisation |
| **E13b** | the same paper, the key-length equation | $h(e_{k,1})$ where the paper's estimator defines $e_{k,1}$ as a ceilinged, capped **count** | **load-bearing.** `mdi_phase` returns the ratio |
| **E13c** | the same paper, Claim 3 items 3–6 | a mismatch: `log` where the inversion above gives `ln`, bare `log` undefined in the paper. The "safer" half of the earlier reading is **withdrawn**: `ln` is *not* the wider form under base 2, the paper's convention wherever stated | `src/mdi.rs` runs `ln` because it is the **derived** form |
| **E14** | Xie et al., PRX Quantum **3**, 020315 (2022), Table 1 | confirmed in v1–v3: $\epsilon = \tfrac{36}{23}\times10^{-10}$ reproduces neither printed security bound under the paper's composition; $10^{-10}$ reproduces both exactly | none; no anchor taken from that table |
| W1 | Ma, Qi, Zhao & Lo — "internal inconsistency at $10^{-8}$" | **withdrawn.** The paper prints $\cong$ and states the smallness assumption in the next sentence | none |
| W2 | Gobby, Yuan & Shields — "~50 km unrecoverable" | **withdrawn.** It reproduces at 49.0 km under a trusted-detector reading with their sifting, and is a multiphoton-rate condition, not a key-rate bound ([above](#gys-pns)) | none |
| W3 | Honjo et al., Opt. Express **16**, 19118 (2008) — its "arXiv id" | **no arXiv version exists.** The ids commonly attached to it and to Dynes et al. 2009 are unrelated astro-ph and instrumentation papers; the field is correctly empty | none; a trap for whoever "fixes" a missing id |
| W4 | Pirandola et al., Nat. Photon. **9**, 397 (2015) — as a *channel experiment* | **it is not one.** No optical channel is applied; loss is dialled into Bob's modulation depth, and its authors call it proof-of-principle | its Tier A row compares against a **calculation**; the CV-MDI Tier B row is Hajomer et al. 2025 |
| W5 | Takesue, Sasaki, Tamaki & Koashi, Nat. Photon. **9**, 827 (2015) — its arXiv id | **arXiv:1505.07884 does not exist.** The paper is **arXiv:1505.07914** | none; a one-digit trap that propagates by copying |
| W6 | "Wang, Yin, Chen et al., PRL **114**, 180502" | **three papers conflated.** That PRL is Guan et al.; the Wang/Yin/Chen paper is Nat. Photon. **9**, 832 (2015), not a PRL | none, but it is the $L = 65$ run the RRDPS row reproduces |
| W7 | Sasaki, Yamamoto & Koashi, Nature **509**, 475 (2014) — its "arXiv id" | **no arXiv version exists.** W3's shape; W5 is what inventing one looks like | none; the bound is reachable through two papers restating it |
| W8 | "arXiv:2201.04956" as the mode-pairing paper | **an astrobiology paper**, a near-miss id resolving cleanly to unrelated work. Mode pairing is **arXiv:2201.04300** | none. Why the check is "does it resolve to the paper I mean, by title *and* authors" |
| W9 | the mode-pairing author list | **routinely wrong.** "Zeng, Zhou, Yin & Zhang" conflates two groups' concurrent papers | none |
| W10 | "Wang, Tamaki & Curty, npj QI **5**, 64 (2019)" for [arXiv:1902.02126](https://arxiv.org/abs/1902.02126) | **the authors are Pereira, Curty & Tamaki**, article **62**. A different Wang, Tamaki & Curty paper exists; the id was right throughout | **reached shipped `src/`, `test/` and `docs/`**; corrected |
| W11 | "arXiv:1201.6555" for the Sperling–Vogel–Agarwal click-counting POVM | **a polarization-optics matrix-classification paper**, W8's failure mode again the same day | none: the correct id could not be recovered, so **no citation was written** |

## Pinning doubles as calibration

"Validate against a published experiment" and "calibrate qkd to your lab" are one
operation: pin what you measured, fit the rest against multi-point data, check the
fitted values are physical, then predict. [`explain()`](/guide/link#pinning) labels
each quantity `pinned`, `derived` or `default`; the labels are the only difference
between a calibration run and a validation run.

## Invariants, not just anchors

Exams checking machinery rather than a paper are the rest of the
[report index](/tests/); each report opens with its own description.

### Verification of the verification {#blind}

`test/blind.py` is evidence **about** two rows, not an anchor. `src/flaws.rs` and
`src/ekert.rs` shipped with exams written by the pass that wrote the engines; the
exams in `blind.py` were derived from the papers by a pass that never opened either
source file and called `_core` as a black box.

| Found | Effect on the board |
| --- | --- |
| **A contradiction**, the committed anchors the wrong half | two crossover figures were pinned **without the configuration producing them**: both papers write their channel at $\text{tilt} = -\delta$, the blind pass ran $\text{tilt} = 0$. Both readings now ship, each naming its tilt in the docstring the report prints; the constants block states $\delta$, dark rate, $f_{EC}$, bracket **and** tilt |
| Confirmations at level 1 | the zero-flaw death at 57.855244 dB, agreeing to $0.0$; `flaw_triangle`'s maximum at $\delta = \pi/3$ in closed form; both rates collapsing onto `bb84_rate` at zero flaw to $9.8\times10^{-16}$; Pironio's 7.1% threshold at 7.1491759% |
| A misattribution in shipped code | "yield exceeds 1 by exactly $2p_d$" is **Pereira's**, not Tamaki's, whose App. C gives $1 + 2.5p_d - p_d^2$. Both now appear in the exams |

### The test that does not run {#skips}

[`gpu_fallback::test_absent_gpu_is_not_an_error`](/tests/gpu_fallback), ⏭️: it
asserts that with no adapter the GPU entry points raise an actionable reason. Where an
adapter answers there is nothing to assert, and faking the absence would test the
fake.

## Where the defaults come from

Every numeric default in the [component reference](/guide/link#component-reference)
is a literature value. The CV hardware defaults are sourced [below](#cited-defaults);
the rest:

| Default | Value | Basis |
| --- | --- | --- |
| `Heterodyne(eta=0.6, v_el=0.1)` | — | **a composite, not one system**: Jouguet's $\eta = 0.552$ came with $v_{el} = 0.015$ at 1 MHz; wideband systems with $v_{el} \approx 0.1$ have higher $\eta$. Conservative |
| `IndividualAttack(f=1.16)` | 1.16 | standard DPS error-correction inefficiency |
| `SplittingAttack(f=1.22)` | 1.22 | the value Ma, Qi, Zhao and Lo optimise $\mu$ against |
| `ClickDetector(eta=0.2, dark=1e-6)` | — | gated InGaAs single-photon detectors at telecom wavelength. `dark` is per gate **per detector**; `Link` derives $Y_0 = 1 - (1-p_d)^2$ |
| `Decoy(intensities=(0.5, 0.1, 0.0))` | — | near Ma–Qi–Zhao–Lo's $\mu_{\text{opt}} = 0.48$, with the vacuum-plus-weak second decoy their Sec. 3.3 shows optimal |
| `Connector(loss=0.25)` | 0.25 dB per mated pair | the mean an IEC 61753-1 grade C connector is allowed (grade B 0.12, grade D 0.50), and Thorlabs' typical mated FC/APC pair. QOSST charges inside that band: 0.23 dB per PM mating sleeve, 0.47 dB per spool connector |
| `Splice(loss=0.02)` | 0.02 dB per splice | measured mean for SMF-28 Ultra spliced to itself at 1550 nm on a core-aligning splicer (Corning/AFL AN0041). A cladding-aligning v-groove splicer gives 0.03–0.04 dB; Telcordia GR-20-CORE asks for a group mean at or under 0.10 |
| `Coupling(loss=…)` | **none** | [deliberately absent](#where-a-default-is-deliberately-absent) |
| `PhaseShiftKeying(states=4, alpha=0.4)` | — | QPSK at modulation variance $2\alpha^2 = 0.32$ SNU, inside the range Denys, Brown and Leverrier's analytic bound is tightest over |

### The evidence base {#evidence-base}

Fielded systems and one modelling paper. Each parameter is in the units and at the
plane its paper states. Those disagree — Laudenbach et al. define $\xi$ at the
channel **output**, Jouguet et al. and the 10 GBaud work at the **input**, QOSST
$\xi_B$ "at Bob" — the
[classic mismatch](/guide/conventions#where-the-literature-disagrees).

| System | Architecture | Parameters and results |
| --- | --- | --- |
| **QOSST** — Piétri et al., *QOSST: A Highly-Modular Open Source Platform for Experimental CV-QKD*, Quantum **8**, 1575 (2024), [arXiv:2404.18637](https://arxiv.org/abs/2404.18637) | Locally generated LO, RF-heterodyne, 100 MBaud with RRC roll-off 0.5, single-sideband shift $+100$ MHz, Zadoff–Chu synchronisation (length 3989, root 5), **two frequency-multiplexed CW pilot tones** at 180 and 200 MHz roughly 12 dB above the quantum band. Teledyne SDR14Tx DAC (14-bit, 2 GSa/s), Teledyne ADQ32 ADC (2.5 GSa/s), Thorlabs PDB480C-AC balanced detector behind a 700 MHz analogue filter | At $\beta = 0.95$, $\varepsilon = 10^{-10}$, $10^6$-symbol frames: $\xi_B = 0.0095 / 0.0091 / 0.0076 / 0.0062 / 0.0072$ SNU at 0 / 5 / 10 / 25 km by attenuator and a 25 km spool (5.22 dB); asymptotic $22.4 / 11.9 / 6.35 / 1.43 / 1.17$ Mbit/s; finite-size $17.7 / 5.82 / 2.55 / 0 / 0$ Mbit/s at $N = 10^6$, positive again at 25 km for $N = 10^7$–$10^{10}$. Calibrated $\eta$ and $v_{el}$ are per-setup and not tabulated, so QOSST is no source for those two |
| **Grosshans et al. 2003** — Nature **421**, 238 (2003), [arXiv:quant-ph/0312016](https://arxiv.org/abs/quant-ph/0312016) | The original Gaussian-modulation demonstration: table-top, 780 nm, 800 kHz pulses, transmitted LO, homodyne | $V_A$ up to 41.7 SNU; homodyne efficiency 0.81–0.84; electronic noise 0.33 SNU; 0–5.9 dB attenuation; reconciliation efficiency $\approx 0.78$–$0.80$; 1.69 Mbit/s at 0 dB to 75 kbit/s at 3.1 dB. Eight bits of modulation resolution stated to hide quantisation under shot noise (16-bit converter, 12-bit digitised) |
| **Jouguet et al. 2013, 80 km** — Nature Photonics **7**, 378 (2013), [arXiv:1210.6216](https://arxiv.org/abs/1210.6216) | 1550 nm pulsed diode at 1 MHz, transmitted LO, homodyne. $\beta = 0.95$ by multidimensional reconciliation, GPU-decoded multi-edge LDPC | $\eta = 0.552 \pm 0.025$, $v_{el} = 0.015 \pm 0.002$ SNU, $V_A$ tuned in real time within 1–10 SNU, $\xi$ measured on $10^8$-symbol blocks at 0.001–0.002 SNU, 0.007–0.008 SNU under the worst-case estimator — all **channel-input** referred. 25 / 53 / 80.5 km at 0.2 dB/km (5.0 / 10.6 / 16.1 dB) give above 10 kbit/s, a few kbit/s and a few hundred bit/s; finite-size at $\varepsilon = 10^{-10}$ on $10^8$–$10^9$ blocks |
| **Zhang et al. 2020, 202.81 km** — PRL **125**, 010502 (2020), [arXiv:2001.02555](https://arxiv.org/abs/2001.02555) | NKT BasiK E15 laser at **100 Hz linewidth**, 5 MHz pulsed, transmitted LO, homodyne, ultra-low-loss fibre at 0.16 dB/km. 10-bit DAC; the supplement names a 12-bit 1 GHz ADC (ADS5400) | Table I: $V_A = 14.37 / 14.14 / 14.12 / 14.53 / 14.23 / 7.65$ SNU and $\xi$ — **at the channel input** — $= 0.0015 / 0.0033 / 0.0049 / 0.0063 / 0.0086 / 0.0081$ SNU at 27.27 / 49.30 / 69.53 / 99.31 / 140.52 / 202.81 km, worst-case $\xi'$ up to 0.0383; $v_{el} = 0.1216$–$0.2717$ SNU; $\eta = 0.6134$; $\beta = 0.95$–$0.98$ (slice plus polar codes, MET-LDPC, Raptor at the longest distance); finite key from $2.78\times10^5$ down to 6.214 bit/s. Phase compensation: 100 reference pulses per 1000 data pulses (10% overhead) at 34 dB above signal, leaving $1 - \kappa \approx 7.6\times10^{-5}$, $\kappa = [\mathbb{E}(\cos\theta)]^2$; the small-angle $\sigma_\theta^2 \approx 7.6\times10^{-5}$ rad$^2$ is a conversion made here, not published |
| **Hajomer et al. 2024, 100 km LLO** — Sci. Adv. **10**, eadi9474 (2024), [arXiv:2305.08156](https://arxiv.org/abs/2305.08156). The closest published system to qkd's default configuration: the [Tier B heterodyne anchor](#tier-b) | CW source, locally generated LO, heterodyne, frequency-multiplexed pilot, finite-size. Linewidth $\approx 100$ Hz both ends, free-running, CFO $\approx 230$ MHz; 100 MBaud, RRC roll-off 0.2, single-sideband shift 100 MHz, **one pilot tone at 180 MHz**; 16-bit DAC and ADC at 1 GSa/s on a shared 10 MHz clock; home-made balanced detector, $\approx 365$ MHz bandwidth, $\approx 15$ dB vacuum-to-electronic clearance; 100 km ultra-low-loss fibre at 0.146 dB/km, 15.4 dB total. Unscented-Kalman-filter phase recovery | $V_{\text{mod}} = 8.41$ SNU (optimised, Fig. 3), trusted efficiency $\tau = 0.68$, trusted detection noise $t = 62.72$ mSNU, untrusted transmittance $\eta = 0.028$, **$\xi = 0.212$ mSNU at the channel output**, $\beta = 92.5\%$ at FER $= 0.59$ (rate-adaptive MET-LDPC, rate 0.05), a $10^9$-state block with $\approx 9.5\times10^8$ used, $\delta_{\text{fail}} = 10^{-10}$, finite-size SKR 25.4 kbit/s |
| **Hajomer et al. 2024, 10 GBaud** — Optica **11**, 1197 (2024), [arXiv:2305.19642](https://arxiv.org/abs/2305.19642) | Probabilistically shaped 16/32/64-QAM, 100 Hz CW, 8-bit AWG at 32 GSa/s with digital pre-emphasis, pilot at 8 GHz (10 GBaud) or 7 GHz (8 GBaud), integrated silicon-photonic phase-diverse receiver at **$\eta = 44\%$**, 8-bit ADC at 80 GSa/s | Table 1: $V_M = 0.87$–$1.03$ SNU, the optimum near 1 SNU at GBaud rates; $T = 0.569$–$0.733$; $V_{el} = 4.95$–$6.76$ %SNU; **$\xi = 1.59$–$7.18$ %SNU at the channel input**; $\beta = 0.95$; $N = 1.6\times10^7$; finite-size SKR up to 0.746 Gbit/s at 5 km, 0.351 Gbit/s at 10 km |
| **Qi et al. 2015** — PRX **5**, 041009 (2015), [arXiv:1503.00662](https://arxiv.org/abs/1503.00662) | First locally-generated-LO demonstration: pilot-aided feedforward phase recovery, two free-running commercial lasers, 25 km | residual phase-noise variance **0.04 rad$^2$**, the upper end of the [$v_{\text{err}}$ plausibility range](#tier-b) |
| **Chin et al. 2021** — npj Quantum Inf. **7**, 20 (2021), [arXiv:2002.09321](https://arxiv.org/abs/2002.09321) | Unscented-Kalman-filter phase tracking over 20 km at 50 MBaud, quantum band at 60 MHz, pilot at 130 MHz, LO offset $\approx 200$ MHz | thermal state $\langle n\rangle = 2.73$ ($V_A = 5.46$ SNU), $\eta = 0.84$ trusted, $t \approx 0.022$ photons. Excess noise held at $\approx 2\times10^{-3}$ photons (channel output) down to pilot SNR $\approx 4$ dB with a 100 Hz laser, and $\approx 7$ dB with a **10 kHz** laser at best $e < 0.01$ — the basis of `Laser(linewidth=10e3)` |
| **Laudenbach et al. 2018** (modelling reference) — *CV-QKD with Gaussian Modulation: The Theory of Practical Implementations*, Adv. Quantum Technol. **1**, 1800011 (2018), [arXiv:1703.09278](https://arxiv.org/abs/1703.09278) | The canonical statement that $\xi$ is assembled from hardware parameters; its Section 9 term list is what the [budget engine](/guide/budget#the-per-source-table) implements, each term at its own plane | Worked example (Fig. 10.2: $B = 250$ MHz, 10 kHz linewidths, CMRR 30 dB, NEP 4.5 pW/$\sqrt{\text{Hz}}$, 10-bit ADC, $\text{RIN}_{\text{LO}} = 1.4\times10^{-7}$ /Hz, $\text{RIN}_{\text{sig}} = 8\times10^{-11}$ /Hz, $P_{\text{LO}} = 8$ mW, $T = 0.1$, $V_{\text{mod}} = 6$ SNU, $\langle n_{PT}\rangle = 600$) yields $\xi_{\text{tot}} = 0.0653$ SNU at the channel output, **$\xi_{\text{det}} \approx 0.0396$, some 60%**. Two derived requirements used as defaults: $\ge 6$ ADC bits at $T = 0.1$ ($\ge 3$ at $T = 0.9$); at $T = 0.1$, $\xi = 0.01$ SNU, reconciliation efficiency above $\approx 0.96$ |

### Every hardware default, with its source {#cited-defaults}

"Range seen" is what the systems above report, in their units and planes.

| Parameter (component) | Symbol | Range seen in experiments | qkd default | Citation for the default |
| --- | --- | --- | --- | --- |
| Laser linewidth (`q.Laser`, `q.LocalLO`) | $\Delta\nu$ | 100 Hz (fibre lasers: Zhang, both Hajomer papers) to 10 kHz (ECL, workable with UKF: Chin) | **10 kHz** — commodity ECL; 100 Hz is the hero-experiment choice, offered as a preset | Chin 2021, 10 kHz viable with a UKF, [arXiv:2002.09321](https://arxiv.org/abs/2002.09321); Laudenbach's worked example, 10 kHz, [arXiv:1703.09278](https://arxiv.org/abs/1703.09278); 100 Hz: [arXiv:2001.02555](https://arxiv.org/abs/2001.02555), [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| Modulation variance (`q.GaussianModulation`) | $V_A$ (SNU) | 1–10 (Jouguet, tuned in real time); 8.41 (Hajomer 100 km, optimised); 7.65–14.5 (Zhang); 0.87–1.03 (10 GBaud); 41.7 (Grosshans 2003) | **5.0 SNU** — mid metro-link optimum; must stay exposed to an optimiser | Jouguet's 1–10 range, [arXiv:1210.6216](https://arxiv.org/abs/1210.6216); Hajomer's optimisation curve, Fig. 3, [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| Symbol rate (`q.Alice`) | $f_{\text{sym}}$ | 0.8–5 MHz (pulsed transmitted-LO era) → 50–100 MBaud (LLO: Chin, Hajomer, QOSST) → 8–10 GBaud (integrated) | **100 MBaud** | QOSST, [arXiv:2404.18637](https://arxiv.org/abs/2404.18637); Hajomer, [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| Pilot arrangement (`q.Pilots`) | — | 1–2 CW tones frequency-multiplexed into the same sideband; QOSST 2 tones at 180/200 MHz, $\approx 12$ dB above the quantum band; Hajomer 1 tone at 180 MHz; Zhang (transmitted LO) time-multiplexed, 10% of pulses, $+34$ dB | **1 tone at 180 MHz, $+12$ dB** — an order of magnitude, not a measured constant. A pilot-to-signal *power* ratio in dB, since a *fraction* describes only time-multiplexed reference pulses | QOSST's $\approx 12$ dB pilots, [arXiv:2404.18637](https://arxiv.org/abs/2404.18637); Zhang's reference-pulse scheme, [arXiv:2001.02555](https://arxiv.org/abs/2001.02555) |
| Detector efficiency (`q.Homodyne`, `q.Heterodyne`) | $\eta$ | 0.552 (Jouguet) / 0.6134 (Zhang) / 0.68 (Hajomer 100 km, trusted part) / 0.81–0.84 (Grosshans 2003) / 0.44 (integrated 10 GBaud) | **0.6** | Jouguet's $\eta = 0.552$, [arXiv:1210.6216](https://arxiv.org/abs/1210.6216); Zhang's $\eta = 0.6134$, [arXiv:2001.02555](https://arxiv.org/abs/2001.02555) |
| Electronic noise (`q.Homodyne`, `q.Heterodyne`) | $v_{el}$ (SNU) | 0.015 (Jouguet, 1 MHz pulsed) / 0.12–0.27 (Zhang) / 0.063 (Hajomer 100 km) / 0.05–0.068 (10 GBaud) / 0.33 (Grosshans 2003) | **0.1 SNU** — clearance $C = 1 + \mu/v_{el}$ of **11, 10.4 dB, homodyne** ($\mu = 1$) and **21, 13.2 dB, heterodyne** ($\mu = 2$). Receiver-dependent; quote which. Hajomer's $\approx 15$ dB is a cross-check only under a stated $\mu$ | Hajomer's $t = 62.72$ mSNU, [arXiv:2305.08156](https://arxiv.org/abs/2305.08156); Zhang's $v_{el}$ column, [arXiv:2001.02555](https://arxiv.org/abs/2001.02555); clearance defined at Eqs. (9.95)–(9.96) and inverted at (9.97), [arXiv:1703.09278v3](https://arxiv.org/abs/1703.09278v3), v3 only |
| DAC resolution (`q.IQModulator`) | bits | 8 (10 GBaud AWG) / 10 (Zhang) / 14 (QOSST) / 16 (Hajomer 100 km) | **16** | Hajomer's 16-bit 1 GSa/s DAC, [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| ADC resolution (`q.ADC`) | bits | 8 (10 GBaud) / 10 (Chin's oscilloscope) / 12 (Zhang, supplement) / 16 (Hajomer); $\ge 6$ required at $T = 0.1$ by the model | **12** | Zhang's 12-bit ADS5400, [arXiv:2001.02555](https://arxiv.org/abs/2001.02555); the bit-requirement curve, Fig. 10.4a, [arXiv:1703.09278](https://arxiv.org/abs/1703.09278) |
| Fibre attenuation (`q.Fiber`) | $\alpha$ (dB/km) | 0.2 (SMF-28: Jouguet's 16.1 dB over 80.5 km; assumed in the 10 GBaud work); 0.146–0.16 (ultra-low-loss: Hajomer, Zhang) | **0.2 dB/km**; ultra-low-loss as a preset | Jouguet, [arXiv:1210.6216](https://arxiv.org/abs/1210.6216); Hajomer's 0.146, [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| Reconciliation efficiency (`q.Asymptotic`, `q.FiniteSize`) | $\beta$ | 0.78–0.80 (Grosshans 2003) → 0.95 (Jouguet, QOSST, 10 GBaud) / 0.925 at FER 0.59 (Hajomer) / 0.95–0.98 (Zhang, SNR-dependent) | **0.95** — the throughput is $(1-\text{FER})\beta$; Hajomer's *best* $\beta$ arrived with FER 0.8 and a *lower* key | Jouguet's $\beta = 0.95$, [arXiv:1210.6216](https://arxiv.org/abs/1210.6216); Zhang's $\beta$-versus-SNR curve, Fig. 7, [arXiv:2001.02555](https://arxiv.org/abs/2001.02555); Hajomer's $\beta$/FER trade, [arXiv:2305.08156](https://arxiv.org/abs/2305.08156) |
| Excess noise — an **output** | $\xi$ | channel input: 0.001–0.008 (Jouguet, measured and worst-case), 0.0015–0.0086 (Zhang), 0.016–0.072 (10 GBaud); at Bob: 0.006–0.0095 (QOSST); channel output: 0.000212 over 100 km (Hajomer) | **a validation target, not a default:** a 25 km default run lands at $\xi \approx 0.005$–$0.03$ SNU, channel-input referred. An assembled $\xi$ outside the envelope falsifies the models, not the envelope | every row above and the QOSST benchmark table, [arXiv:2404.18637](https://arxiv.org/abs/2404.18637); tabulated by plane under [the measured envelope](/guide/budget#the-measured-envelope) |

| Finite-size default | Cross-check |
| --- | --- |
| $\varepsilon = 10^{-10}$ | uniform across Jouguet, QOSST and Hajomer ($\delta_{\text{fail}} = 10^{-10}$; a 6.5-sigma worst case in Zhang and the 10 GBaud paper): the literature value |
| $N = 10^9$ | key-producing blocks at metro distances are $10^7$–$10^9$. QOSST's zero-key rows at $N = 10^6$ over 25 km are the natural `res.key_rate == 0` regression, named [below](#roadmap) |

### Where a default is deliberately absent

Arguments where a plausible default was available and declined.

| Argument | Why no default |
| --- | --- |
| `Coupling.loss` | an order of magnitude across mode-crossing interfaces ([values](/guide/budget#what-each-one-is-and-what-it-costs)) |
| `PhaseBound.e_phase`, `FiniteSize.fer` | [component reference](/guide/link#security) |
| `dm_holevo(bits=…)` | [discrete modulation](/guide/protocols#discrete) |
| `BellDetector.eta`, `.v_el` | a relay detector defaulting to perfect is an idealisation nobody chose, and flatters the topology: relay imperfections fold into the arms as loss and noise Eve holds |
| `CorrelatedEnvironment.x`, `.p` | [no neutral correlation](/guide/relay#swap) |
| `State.thermal_loss(ref=)` | [the plane is always named](/guide/gaussian#thermal-loss-and-the-required-ref-plane) |
| a `qkd.fock` cutoff for a negativity volume | [the cutoff is part of the state](/guide/fock#constructors) |

## What is not anchored yet {#roadmap}

| Unbuilt anchor | State |
| --- | --- |
| Joint curve fits | Tier B exercises the budget engine at fixed configurations. Zhang's six points are each read at their own parameters, not fitted jointly with bounded free parameters. Jouguet's distances are untouched |
| QOSST zero-key regression at $N = 10^6$, 25 km | not built |
| Lim's Fig. 1, end to end | digitised and compared, never pinned ([the three missing pieces](#lim-fig1)) |
| A full unpinned Gaussian run | its derived $\xi$ must land in the measured envelope |
| A vacuum+weak decoy field experiment | Lucamarini's T12 is **three**-intensity biased-basis. The vacuum+weak set of Ma, Qi, Zhao and Lo's Sec. 3.3, which the Tier A anchor is written against, has no measured counterpart |
| COW and BB84-WCP Tier B rows end to end | *assembled from closed forms*, not run through the sampled paths: the sampled train reproducing GYS's measured QBER, a simulated COW data line reproducing Stucki's |
