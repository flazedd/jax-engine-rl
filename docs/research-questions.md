# Research questions

Full protocol, plot specifications, and result stories for RQ1/RQ2/RQ3. Read this when actively designing or interpreting experiments that answer one of the RQs. For the short summary, see the main `CLAUDE.md`.

See also:
- `docs/milestones.md` — which milestone produces which RQ's answer.
- `docs/methodology.md` — the statistical tests used to interpret the results.
- `docs/environment.md` — the env parameterization the RQs are run on.

---

## Research questions

Three questions, each motivating the next. Together they form a tight chain: **decompose the gap** → **measure how much meta-RL closes it** → **characterize how closure scales with difficulty and whether belief quality predicts it**.

Each RQ below specifies (a) the question, (b) the experimental protocol, (c) the plots that answer it, and (d) the story those plots tell when everything works.

---

### RQ1. How does the optimality gap in regime-switching market making decompose into shared-network, inference, and compromise-policy costs?

**Terminology.** The four points below are collectively **reference performance levels** (or just *reference levels*). "Ceiling" refers specifically to an upper bound — so Oracle-PPO, Belief-PPO, and per-regime PPO are *ceilings*; regime-agnostic PPO is the *floor*, not a ceiling. The document uses "ceiling" only for actual upper bounds from here on.

**Protocol.** Establish four reference performance levels on a single reference parameterization of the MM env:
- **Per-regime PPO** (ceiling) — N separate networks, each trained on a locked regime. Upper bound on regime-conditional performance with no shared-network constraint.
- **Oracle-PPO** (ceiling) — single network conditioned on true regime one-hot. Upper bound under the shared-network constraint with ground-truth regime.
- **Belief-PPO** (ceiling) — single network conditioned on analytical HMM posterior (computed via the forward algorithm). Upper bound under the shared-network constraint when regime must be inferred from history.
- **Regime-agnostic PPO** (floor) — single network, no regime information. Compromise-policy baseline — the bottom of the gap.

Each trained with matched compute (same iteration count, same parallel env count, same hyperparameter search). 5 seeds per method minimum.

**Decomposition:**
- **Shared-network cost** = Per-regime − Oracle
- **Inference cost** = Oracle − Belief
- **Total belief value** = Belief − Regime-agnostic

**Plots that answer RQ1.**

- `fig_rq1_ceilings_bar.png`
  - *What it shows.* Four vertical bars on one axis, one per reference level (per-regime PPO, Oracle-PPO, Belief-PPO, regime-agnostic PPO), with error bars showing seed-level confidence intervals. Brackets drawn between consecutive bar tops label each gap component (shared-network cost, inference cost, compromise-policy cost). Per-regime PPO is the highest (regime-conditional ceiling); regime-agnostic PPO is the lowest (compromise-policy floor).
  - *What it tells you.* The total optimality gap and how it splits into its three named components at a glance. The height of each bracket is the size of that component.
  - *Why it matters.* This is the numerical answer to RQ1. It makes the decomposition visually obvious: whether inference cost dominates, or the shared-network constraint is the real bottleneck, or the compromise-policy cost is most of the story. The rest of the thesis evaluates meta-RL methods against these specific gap components, so this figure is the reference frame readers return to.

- `fig_rq1_learning_curves.png`
  - *What it shows.* Four line plots on one set of axes, one per reference level. X-axis is iteration (0–100); y-axis is mean return across seeds; each line has a shaded CI band.
  - *What it tells you.* Whether each reference level has actually converged (flat region at the end), and whether the gap ordering is stable throughout training or only emerges late.
  - *Why it matters.* The bar chart compresses training into a single final number. If two methods are still climbing at iteration 100, the reported gap is undercooked. This plot is the diagnostic that rules that out — or flags that more iterations are needed before the bar chart is trustworthy.

- `fig_rq1_gap_fractions.png`
  - *What it shows.* A single stacked bar (or equivalent pie) whose total height is the full regime-agnostic-to-per-regime gap, split into three colored segments labeled by gap component and sized by their fractional contribution.
  - *What it tells you.* The relative importance of each gap component as a proportion, not an absolute.
  - *Why it matters.* Two parameterizations might have similar total gaps but very different decompositions. The fraction view makes the *structure* of the problem visible independent of its scale — useful when comparing across difficulty settings in RQ3.

**The story these plots tell.**
*"The total gap between regime-agnostic PPO and per-regime PPO is X. Of this, the shared-network cost contributes Y%, the intrinsic inference cost Z%, and the compromise-policy cost W%. Inference is [the / not the] dominant source of lost performance. This decomposition is the reference frame for evaluating meta-RL methods in RQ2."*

---

### RQ2. How much of each gap component can meta-RL methods close from interaction history, and where do they fall short?

**Protocol.** On the reference parameterization, run the 7-rung core method ladder (regime-agnostic PPO, stacked-obs PPO, RL², VariBAD, Belief-PPO, Oracle-PPO, per-regime PPO) at the default configuration: concat integration, no exploration bonus. This is the clean "method X vs method Y" comparison where the only thing that varies is the belief source.

Then, on the two belief-learning methods (RL², VariBAD), run a 2×2 factorial over two orthogonal ablation axes: integration mechanism (concat vs hypernet) and exploration bonus (off vs on). 8 cells total.

**Measurements per configuration.**
- Return (gap-closed fractions relative to Oracle-PPO and Belief-PPO).
- Posterior approximation error: symmetric KL divergence between inferred belief (mapped to simplex via the linear probe specified in Statistical methodology) and analytical HMM posterior.
- Regime classification accuracy from the inferred belief.

**Factorial ablation (orthogonal-axes subsection under RQ2).** The factorial tests whether integration mechanism and exploration bonus are separable axes of method design, or whether they interact with belief source. Specifically:
- **Main effect of integration**: does hypernet beat concat on average across belief sources? If yes, hypernet is a general integration win; future meta-RL methods should use it regardless of belief source.
- **Main effect of exploration**: does the bonus help on average? If yes, the exploration-augmented training signal is generally useful.
- **Interaction effects**: does hypernet help VariBAD more than RL² (or vice versa)? Does exploration help one belief source more than the other? An interaction means the axes are not truly orthogonal — some combinations work better than their individual effects predict.

The mu-only vs full-posterior ablation is separate (it's a question about what part of the belief to expose, not how to integrate it). Reported as a focused sub-ablation within VariBAD, since mu-only is only defined for methods with an explicit Gaussian belief.

**Plots that answer RQ2.**

- `fig_rq2_ladder_returns.png`
  - *What it shows.* Bar chart across the full method ladder (regime-agnostic PPO through Oracle-PPO), each bar showing final mean return with seed-level CI error bars. Horizontal dashed lines mark RQ1's ceilings (Belief-PPO, Oracle-PPO, per-regime PPO).
  - *What it tells you.* How each meta-RL method ranks relative to every other and relative to the ceilings. The ceiling lines turn absolute returns into interpretable positions — "VariBAD sits between Belief-PPO and Oracle-PPO" is meaningful; "VariBAD got 12.4 return" alone is not.
  - *Why it matters.* This is the main RQ2 comparison figure. Every later claim about method ranking cites this plot.

- `fig_rq2_gap_closed.png`
  - *What it shows.* Grouped bar chart. Two bars per meta-RL method, side-by-side. For each method:
    - **Oracle-gap fraction** = `(method_return − regime_agnostic_return) / (oracle_return − regime_agnostic_return)`. Measures how much of the total inference gap the method closes, with Oracle-PPO as the ceiling.
    - **Belief-gap fraction** = `(method_return − regime_agnostic_return) / (belief_ppo_return − regime_agnostic_return)`. Measures how close the method gets to what Bayes-optimal inference can achieve, with Belief-PPO as the ceiling.
    - Both fractions are 0 when the method matches regime-agnostic PPO and 1 when the method matches the respective ceiling. Values can exceed 1 (beating the ceiling — worth flagging as either a finding or a bug).
  - *What it tells you.* How each method uses regime information, relative to two different theoretical maxima. A method with high Oracle-gap fraction is closing most of the total regime-info gap. A method with high Belief-gap fraction is approaching Bayes-optimal inference. The difference between the two fractions — Belief-gap high but Oracle-gap lower — reveals how much of the unclosed gap is *intrinsic inference cost* (stuff no history-based method can recover) vs method-specific shortcoming.
  - *Why it matters.* Absolute returns don't travel across parameterizations — return of 12.4 means nothing without knowing the local ceilings. Gap-closed fractions do travel, which is what lets RQ3 compare methods across difficulty sweeps. Also the numbers that drop cleanly into a thesis abstract: "VariBAD closes 68% of the Oracle gap and 92% of the Belief gap on the reference parameterization."

- `fig_rq2_learning_curves.png`
  - *What it shows.* All ladder methods overlaid on one axis, x = iteration, y = mean return, shaded CI bands. Ceiling reference lines as in the bar chart.
  - *What it tells you.* Convergence status per method, training-time behavior (does method X plateau early? does method Y overshoot and decline?), and whether rankings are stable throughout training or flip near the end.
  - *Why it matters.* Same role as in RQ1: the bar chart is a snapshot, this plot is the diagnostic that confirms the snapshot is from a settled point in training. Also reveals qualitative differences that the final-return bar chart hides (e.g., "VariBAD converges twice as fast as RL²").

- `fig_rq2_posterior_error.png`
  - *What it shows.* Line plot, x = iteration, y = symmetric KL divergence between inferred belief (mapped to simplex via the linear probe specified in Statistical methodology) and analytical HMM posterior (averaged over a held-out trajectory set). One line per belief-based method (RL², VariBAD).
  - *What it tells you.* Whether methods are learning the posterior, how fast they get there, and the ranking of methods *by belief quality* (which may differ from ranking by return).
  - *Why it matters.* The analytical posterior is your setup's unique asset. This plot is the operationalization of that asset — a direct measurement of inference quality that most meta-RL benchmarks cannot produce. Also the input to RQ3's posterior-vs-performance analysis.

- `fig_rq2_factorial.png`
  - *What it shows.* Grouped bar chart or heatmap for the 2×2×2 factorial: belief source (RL², VariBAD) × integration (concat, hypernet) × exploration bonus (off, on). Eight bars total, grouped by belief source, sub-grouped by integration, colored by exploration. Error bars are seed-level CIs. Horizontal reference lines at Belief-PPO and Oracle-PPO.
  - *What it tells you.* Three separable effects:
    1. **Belief source effect**: is RL² systematically below VariBAD, or vice versa, within matched integration/exploration cells?
    2. **Integration effect (main)**: does hypernet beat concat on average? Consistent direction across belief sources = general integration win.
    3. **Exploration effect (main)**: does the bonus help on average? Consistent direction = general training-signal win.
    4. **Interactions**: does hypernet help VariBAD more than RL²? Does exploration help one more than the other? Non-parallel patterns in the bars reveal interactions.
  - *Why it matters.* This is the rigor-bearing RQ2 figure. Without the factorial, a naive comparison like "VariBAD+hypernet vs RL²" conflates the belief-source effect with the integration effect. The factorial disentangles them, which is what makes the thesis's integration/exploration findings attributable to the right source. Prior meta-RL work tends to bundle these axes into named methods (a variational approach with exploration bonuses published under one name, a recurrent approach under another), which makes it impossible to tell which component produced the advantage.

- `fig_rq2_mu_only_ablation.png`
  - *What it shows.* Bar chart comparing VariBAD with mu-only conditioning vs full-posterior conditioning, at both concat and hypernet integration. Four bars total with seed CIs.
  - *What it tells you.* Whether second-order belief information (posterior variance) is actionable by the policy in this setting. If mu-only matches full-posterior, the variance is not being used. If mu-only loses, the policy needs the uncertainty.
  - *Why it matters.* Targeted follow-up ablation driven by the mu-only finding on toy tasks. Separate from the main factorial because "what belief information to expose" is a different question than "how to integrate the belief" — conflating them in the same figure would muddy both.

- `fig_rq2_regime_accuracy.png`
  - *What it shows.* Bar chart, one bar per method, y-axis = regime classification accuracy (argmax of inferred belief vs true regime, averaged over evaluation trajectories).
  - *What it tells you.* A coarser but more intuitive belief-quality measure than posterior MSE — "does the method know which regime it's in?"
  - *Why it matters.* MSE is a rigorous metric but not interpretable for readers outside the field. Classification accuracy is a back-up check — if MSE and accuracy rank methods differently, that's a flag worth diagnosing. Also makes the belief-quality story legible to supervisors and reviewers in one glance.

**The story these plots tell.**
*"Meta-RL methods close [X%, Y%, Z%] of the Oracle-PPO gap respectively. VariBAD's posterior approximation error is [lowest / not lowest], and this [does / does not] track its task performance. Integration mechanism has a [small / substantial] effect: mu-only [matches / underperforms] full posterior, suggesting second-order belief information is [not / is] actionable by the policy in this setting."*

---

### RQ3. How does method performance scale with problem difficulty, and does posterior approximation quality predict task performance?

**Motivation.** RQ2 answers how methods rank on one parameterization. This is a single data point — the ranking could be specific to that configuration. RQ3 sweeps difficulty to turn the point into a map, and simultaneously tests whether belief quality is mechanistically what drives performance, or whether the two can decouple.

**Protocol.**
- **Difficulty axis 1 — regime persistence**: per-step transition probability, swept across 3 points (easy / medium / hard).
- **Difficulty axis 2 — regime distinguishability**: separation between regime-conditional fill probabilities, swept across 3 points (easy / medium / hard).
- **Methods swept**: reduced core set (RL², VariBAD, Belief-PPO, Oracle-PPO, regime-agnostic PPO). Both belief-learning methods run at concat integration with no exploration bonus — the configuration that appears in the M5 core ladder. Factorial ablations are not re-run at every sweep point.
- **Per difficulty point**: re-compute RQ1's four reference levels locally (Oracle, Belief-PPO, per-regime, and regime-agnostic all change with parameterization), run the method ladder, measure gap-closure and posterior error.

**Two sub-questions answered here.**

*(a) Which methods are robust to harder inference vs which collapse?*
Gap-closure curves per method as difficulty increases. Methods that stay high are robust; methods that collapse reveal their failure modes.

*(b) Does posterior approximation quality predict task performance?*
At each difficulty point and for each method, plot gap-closure (y) vs posterior error (x). If the scatter shows tight negative correlation, belief quality is what drives returns. If points cluster in ways that decouple the two — especially a method with high returns and high posterior error — that is a finding: the "belief" is not really a posterior approximation, it is a policy-useful hidden state that happens to decode partly as one.

**Plots that answer RQ3.**

- `fig_rq3_persistence_sweep.png`
  - *What it shows.* Line plot with one line per method, x-axis = regime persistence level (easy/medium/hard — longer to shorter regime durations), y-axis = fraction of Oracle-PPO gap closed. Each line has a shaded seed-level CI band.
  - *What it tells you.* How each method's performance degrades as regimes switch more quickly. A flat line = robust; a steep drop = collapses under fast switching.
  - *Why it matters.* Turns the single-point RQ2 ranking into a *map*. "Method X dominates when regimes persist but collapses under fast switching" is a portable, useful finding that informs when to reach for which method in future work. Most meta-RL benchmarks can't produce this because their task distributions are fixed.

- `fig_rq3_distinguishability_sweep.png`
  - *What it shows.* Same structure as the persistence sweep, but x-axis = regime distinguishability level (how different regime-conditional fill probabilities are). Each method is one line with CI band.
  - *What it tells you.* How each method degrades as regimes become harder to tell apart from observations, independent of how often they switch.
  - *Why it matters.* Persistence and distinguishability are two different sources of inference difficulty — a method might be robust to one and fragile to the other. Separating the axes surfaces that distinction, which a 2D heatmap would bury.

- `fig_rq3_persistence_posterior_error.png`
  - *What it shows.* Same axes as the gap-closure sweep, but y-axis = posterior approximation error (symmetric KL divergence vs analytical HMM posterior, via linear probe). One line per belief-based method.
  - *What it tells you.* How belief inference quality degrades with regime persistence. Useful paired with the gap-closure plot: if posterior error rises and gap-closure falls together, belief quality is driving performance; if they come apart, something else is going on.
  - *Why it matters.* This plot is the setup for the posterior-vs-performance scatter. It shows the marginal relationship between difficulty and belief quality, which you then condition on in the scatter.

- `fig_rq3_distinguishability_posterior_error.png`
  - *What it shows.* Distinguishability analog of the persistence posterior-error plot.
  - *What it tells you.* Same kind of analysis, second difficulty axis.
  - *Why it matters.* Symmetric complement — belief quality under overlap, not frequency.

- `fig_rq3_posterior_vs_performance.png` **— the decoupling plot.**
  - *What it shows.* Scatter plot, x-axis = posterior approximation error, y-axis = fraction of gap closed. Each point is a (method × difficulty configuration × seed) triple. Points colored by method. Per-method regression lines overlaid.
  - *What it tells you.* Whether posterior quality actually predicts task performance. Three distinguishable patterns are possible: **(a)** tight negative correlation across all methods — belief quality is what matters; **(b)** decoupling — some methods achieve high returns with high posterior error (a "belief" that isn't really a belief, just a useful hidden state), or high-quality posteriors that don't translate to returns (integration bottleneck); **(c)** method-specific clusters — each method lives in a different region, revealing that different methods are optimizing for different things.
  - *Why it matters.* This is the plot that makes a mechanistic contribution to the meta-RL literature, not just an MM-specific one. Most meta-RL papers cannot run this analysis because they lack the analytical posterior. Your setup can — and the finding is genuinely unknown ahead of running the experiment. A decoupling result would push back on a common assumption in the field.

- `fig_rq3_difficulty_heatmap.png` *(optional, scope permitting)*
  - *What it shows.* One 2D heatmap per method: rows = persistence level, columns = distinguishability level, cell color = gap-closure fraction.
  - *What it tells you.* Whether the two difficulty axes interact. A method might handle either axis fine on its own but collapse when both are hard.
  - *Why it matters.* Completeness. Worth including if compute budget allows; skippable if the 1D sweeps already tell a clean story.

**The story these plots tell.**
*"As inference becomes harder (faster switching, less distinguishable regimes), method ranking [stays stable / changes]. VariBAD [maintains / loses] its advantage over RL² in the hard regime. Across the sweep, gap-closure and posterior error [are strongly negatively correlated / decouple in method-specific ways]. If decoupled: method X achieves [high / low] returns with [low / high] posterior error, suggesting its representation is [a genuine belief approximation / a policy-useful statistic that is not a belief]."*

---

### Integration across RQs

The three RQs together produce a coherent story:
1. **RQ1** establishes the reference frame (four reference levels — three ceilings plus a floor — and three gap components).
2. **RQ2** measures meta-RL methods against that frame on one parameterization.
3. **RQ3** generalizes: how does the answer change with difficulty, and is belief quality mechanistically what matters?

Each RQ's plots stand alone (readable by a skimmer) but build on the previous (the ceiling lines in RQ2 plots come from RQ1; the method ranking in RQ3 plots is evaluated against the local reference levels).


---

## Reference levels (quick reference table)

Quick reference for the four reference levels defined under RQ1. Oracle-PPO, Belief-PPO, and per-regime PPO are ceilings (upper bounds). Regime-agnostic PPO is the floor. See RQ1 for full discussion.

| Reference | Role | Information | Constraint | Purpose |
|---|---|---|---|---|
| Per-regime PPO | Ceiling | True regime (locked) | N separate networks | Regime-conditional optimum, no shared-representation cost |
| Oracle-PPO | Ceiling | True regime (one-hot) | Single shared network | Upper bound under shared-network constraint |
| Belief-PPO | Ceiling | Analytical HMM posterior | Single shared network | Upper bound under inferred-belief constraint |
| Regime-agnostic PPO | Floor | None | Single shared network | Compromise-policy floor |

Gap components:
- **Shared-network cost** = Per-regime − Oracle
- **Inference cost** = Oracle − Belief
- **Total belief value** = Belief − Regime-agnostic

## Method ladder

The project treats meta-RL method design as three orthogonal axes: **belief source** (how the belief is obtained), **integration mechanism** (how the belief reaches the policy), and **exploration bonus** (whether belief-novelty is rewarded during training). Separating these axes is what makes the contribution rigorous — prior meta-RL work tends to bundle them into named methods, which makes it impossible to attribute observed gains to the right source.

### Core ladder (7 methods, each architecturally distinct)

Evaluated in order of increasing belief explicitness:

1. **Regime-agnostic PPO** (MLP, current obs only) — memoryless floor.
2. **Stacked-obs PPO** — fixed window of past K observations, no recurrence. Isolates whether naive memory is sufficient.
3. **RL²** — recurrent PPO, implicit belief in hidden state, input augmented with prev action and prev reward. Belief trained by policy loss only.
4. **VariBAD** — explicit variational Gaussian belief from RNN encoder, trained with ELBO (reconstruction + KL). Architecturally and training-signal distinct from RL².
5. **Belief-PPO** — analytical HMM posterior fed directly into PPO. No learned inference.
6. **Oracle-PPO** — true regime one-hot fed into PPO. No inference needed.
7. **Per-regime PPO** — N separate PPOs, each on a locked regime.

Each rung tests a clearly distinct representational claim. The ladder is *only* about belief source — integration and exploration are handled as ablation axes, not ladder rungs.

### Orthogonal ablation axes (factorial, not ladder)

**Integration mechanism** — how the belief is consumed by the policy:
- **Concat**: belief appended to observation, MLP policy.
- **Hypernet**: belief generates policy weights via a small hypernetwork.

**Exploration bonus** — whether belief-novelty is rewarded during training:
- **Off**: policy trained on env reward only.
- **On**: auxiliary intrinsic reward proportional to distance from recent beliefs in the method's native representation space. For RL², novelty is measured on the recurrent hidden state. For VariBAD, novelty is measured on the latent Gaussian mean. For Belief-PPO, novelty is measured on the analytical posterior. Design choice: "novelty on whatever representation the method maintains" — each method's exploration bonus operates on whatever that method natively represents, so the comparison "does exploration help?" isn't entangled with "does adding a probe-trained decoder help?" The exploration-bonus concept is taken from prior work on belief-space exploration in meta-RL (Zintgraf et al., 2021); this project applies it as an orthogonal ablation axis rather than as a property of a single named method.

  **Novelty computation (committed spec).** At each step `t`, novelty is the L2 distance from the current belief-representation vector to the mean of the last `K` belief-representation vectors in the same episode: `bonus_t = coef · ‖b_t − mean(b_{t-K:t-1})‖₂`. Default `K=16`, default coefficient tuned in M4. The same `K` and coefficient are used across all methods for axis comparability. Reported in `stats_M5_factorial.json` per run.

### Factorial design for RQ2

Core factorial: 2 belief sources × 2 integration × 2 exploration = 8 cells, on belief-learning methods RL² and VariBAD. Reference levels (PPO, stacked-obs PPO, Belief-PPO, Oracle-PPO, per-regime PPO) run only at concat/no-bonus for the ladder figures; optional confirmatory cells on Belief-PPO with hypernet and/or exploration if compute allows.

### Excluded methods and why

- **PEARL** — the distinguishing features (off-policy training, set-based context encoder) either disappear under the PPO adaptation needed for ladder consistency or are mismatched to regime-switching POMDPs (set-based aggregation discards the order information the HMM posterior depends on). Including PEARL would muddy the method-source comparison. The underlying scientific question — does order information matter? — is handled qualitatively in the discussion, with reference to the sequential structure of the HMM forward algorithm.
- **PPO + regime classifier head** — originally on the ladder as a "supervised inference" baseline, but adds a training signal that none of the other belief-based methods have, making comparison unclean. Dropped.
- **Gradient-based meta-RL (MAML and descendants)**: do not maintain a belief representation, which is the object of study.
- **Transformer-based in-context meta-RL (AD, DPT)**: excluded on compute grounds, acknowledged as future work.

### Scope pre-commitment

M4 CPU compute budget is the binding constraint. Main-body experiments cover:
- All 7 core ladder rungs (single configuration each: concat integration, no exploration bonus).
- 2×2×2 factorial (RL², VariBAD × concat, hypernet × off, on) = 8 additional runs.

That is 15 configurations total for the reference parameterization in M5, each with 5 seeds. If compute is tight, the exploration axis is the first to drop (run factorial at 2×2×1 = 4 cells, with exploration bonus as a supplementary single comparison).
- **Full LOB microstructure simulator**: destroys analytical posterior tractability, not the right abstraction for the object of study.

