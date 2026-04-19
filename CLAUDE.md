# CLAUDE.md

Project spec for a master's thesis on belief-conditioned meta-RL for regime-switching market making. This file is the single source of truth for research questions, methodology, repo layout, and conventions. Keep it up to date as the project evolves.

## Glossary of abbreviations

Defined once here, used freely throughout the document.

**Reinforcement learning / meta-RL**
- **RL** — reinforcement learning.
- **MDP** — Markov decision process.
- **POMDP** — partially observable Markov decision process.
- **PPO** — Proximal Policy Optimization (Schulman et al., 2017).
- **SAC** — Soft Actor-Critic.
- **VI** — value iteration.
- **MLP** — multilayer perceptron (feedforward network).
- **GRU / LSTM** — gated recurrent unit / long short-term memory (recurrent network types).
- **GAE** — generalized advantage estimation.
- **RL²** — recurrent meta-RL method; policy is a recurrent network that treats the episode as inner-loop learning (Duan et al., 2016).
- **VariBAD** — variational Bayes-adaptive deep RL; meta-RL with an explicit variational posterior over tasks (Zintgraf et al., 2020).
- **MAML** — Model-Agnostic Meta-Learning. Excluded from this thesis; mentioned only for scope boundaries.
- **HN** — hypernetwork; a network whose output is the parameters of another network.

**Probability / information theory**
- **HMM** — hidden Markov model.
- **KL** — Kullback-Leibler divergence.
- **ELBO** — evidence lower bound (variational inference objective).
- **MSE** — mean squared error.
- **CI** — confidence interval. Throughout this project, "CI" refers specifically to a **95% bootstrap confidence interval computed across training seeds** (not across parallel envs within a seed). Bootstrap rather than parametric because seed counts are low (n=5 default) and the t-distribution assumption is unsafe. Computed via 10,000 bootstrap resamples of the per-seed final-return values. Plotted as a shaded band around the mean line. See "Statistical methodology" section for full details.

**Finance / market making**
- **MM** — market making.
- **AS** — Avellaneda-Stoikov, the standard reference MM model (Avellaneda & Stoikov, 2008).
- **LOB** — limit order book.

**Project-internal**
- **R1–R4** — the four problem requirements (policy divergence, locked-regime optimality, mixed-regime suboptimality, regime inferability) that any valid env parameterization must satisfy. See "Problem requirements".
- **E0, E1, E2, ...** — iterations of the env design process, starting from vanilla AS (E0) and adding structural elements one at a time.
- **M1–M7** — the project's progress milestones (M1 = env plumbing, ..., M7 = supplementary ablations).
- **RQ1, RQ2, RQ3** — the three research questions.

## Thesis overview

### Problem

Regime-switching market making is a partially observable Markov decision process (POMDP) in which a latent HMM regime governs fill dynamics, while the agent observes only noisy fills and local state. The MM environment is constructed such that regime information is load-bearing for the policy — the regime-conditional optimal policy varies meaningfully across regimes, so any regime-agnostic stationary policy is necessarily suboptimal.

### Central argument

This gap — between regime-agnostic PPO and an agent that has access to or can infer the regime — is the object of study. The thesis investigates how much of the gap is recoverable from interaction history, which representational and integration choices close it most efficiently, and *why* the remaining gap exists.

### Distinctive methodological feature

The HMM posterior over regime given history is analytically computable via the forward algorithm. This makes posterior approximation quality a directly measurable quantity against ground truth, which is unusual for meta-RL benchmarks (MuJoCo meta-tasks, MetaWorld) and is what makes this problem a clean testbed for belief-based meta-RL.

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
- Posterior approximation error: bidirectional mapping error between inferred belief and analytical HMM posterior.
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
  - *What it shows.* Line plot, x = iteration, y = bidirectional mapping error between inferred belief and analytical HMM posterior (averaged over a held-out trajectory set). One line per belief-based method (RL², VariBAD).
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
  - *What it shows.* Same axes as the gap-closure sweep, but y-axis = posterior approximation error (bidirectional mapping error vs analytical HMM posterior). One line per belief-based method.
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

## Environment

### Reduced-form market making MDP

- **State**: bounded discrete inventory; optionally cash/PnL, optionally mid-price tick.
- **Action**: 3 discrete quote configurations — sym(1,1), ask(1,3), bid(3,1).
- **Fill model**: Bernoulli, conditional on action and regime.
- **Reward**: spread capture minus inventory penalty.
- **Regime**: 3-state HMM (noise, bull, bear). Transition matrix and regime-conditional fill probabilities are **hand-chosen, not calibrated to real data**.
- **Horizon**: infinite-horizon discounted, γ = 0.99.

### Uncalibrated by design

The environment is a **stylized regime-switching POMDP**, not a realistic market simulator. Parameters (transition matrix, regime-conditional fill probabilities, reward weights) are hand-chosen and frozen in the env config. This is deliberate:

- **Thesis is about belief-based meta-RL, not market microstructure.** MM is the motivating setting; regime-switching POMDPs are the object of study.
- **Calibration would introduce confounds.** If a method underperforms, calibration noise would become a competing explanation for the result.
- **Reproducibility.** Anyone can regenerate the environment from the config without access to proprietary or licensed market data.
- **Controllable difficulty.** Distinguishability of regimes and transition frequency become explicit experimental axes rather than properties inherited from data.

This framing should be stated explicitly in the thesis: *"The MM environment is a stylized regime-switching POMDP with hand-chosen parameters, not calibrated to historical market data. Calibration is orthogonal to the research questions and left to future work."*

### Design rationale

- **3 regimes** keep the belief simplex 2D — visualizable, analytically tractable, sufficient to produce measurable policy disagreement across regimes.
- **3 discrete actions** keep VI tractable while being sufficient to produce meaningful regime-conditional policy divergence (R1) under reasonable parameterizations. See "Action space: discrete, not continuous" below for full justification.
- **Reduced-form (not full LOB)** preserves HMM posterior tractability, enables VI-based oracles, keeps env step near-scalar for fast JAX throughput.
- **Uncalibrated**, see above.

### Problem requirements

The MM environment must satisfy the following properties for the thesis to be well-posed. These are the foundation that makes meta-RL a meaningful thing to study here — if any of them fail, the research questions collapse.

**R1. Regime-conditional policy divergence.** The optimal policy on each locked regime must differ meaningfully from the optimal policies on other regimes. Formally: the VI-derived optimal policies on the three locked regimes should disagree on a non-trivial fraction of the state space, and where they disagree, the value loss from playing the wrong regime's policy should be substantial.
*Why required:* if optimal policies are the same across regimes, regime information is useless and no method can benefit from it. There is nothing to study.

**R2. PPO achieves optimality on locked regimes.** A PPO agent trained on a single locked regime (per-regime PPO) should converge to the regime-conditional optimum (matching VI). This must hold for every regime.
*Why required:* if PPO cannot learn a locked regime's optimum, any failure of PPO on the mixed setting is attributable to optimization, not to the regime-switching structure. The ceiling decomposition loses meaning.

**R3. PPO settles on a strictly suboptimal compromise on mixed regimes.** A regime-agnostic PPO trained on the full HMM-generated trajectories must converge to a stationary policy whose return is meaningfully below the return achievable with regime information (Oracle-PPO) and below the regime-conditional optimum (per-regime PPO). The gap must be large enough that the variation across seeds does not swamp it.
*Why required:* this is the gap that meta-RL methods are asked to close. Without a measurable gap, there is no room for belief-conditioned methods to shine.

**R4. Regime inferability from history.** The HMM posterior over regime, conditioned on a reasonable finite history of observations, must sharpen meaningfully — i.e. it cannot remain close to the stationary prior indefinitely. Belief-PPO must close a non-trivial fraction of the Oracle-PPO gap.
*Why required:* if the regime is not inferable from observations, no belief-based method can succeed regardless of its machinery. RQ2 becomes unanswerable.

These requirements are verified empirically on the chosen parameterization before main experiments begin. See **Sanity checks** below.

### Action space: discrete, not continuous

Action space is 3 discrete quote configurations. Discrete is the right choice specifically for this thesis, not a generic preference:

- **VI tractability.** Oracle-PPO and per-regime PPO ceilings come from value iteration, which requires discrete actions. Continuous actions would force either discretization (implicitly discrete anyway, less principled) or actor-critic oracles (which are themselves approximations, breaking the ceiling decomposition).
- **Belief-PPO cleanliness.** Discrete actions + 2D belief simplex → tractable belief-to-action mapping with closed-form optimal policy from VI. Continuous actions complicate this and conflate policy-class expressivity with belief-integration quality.
- **Comparison cleanliness.** Published meta-RL on continuous-action tasks (MuJoCo) carries confounds (reward scaling, action repetition, observation normalization) that discrete setups short-circuit.
- **AS natural reduction.** The AS optimal quoting decision has a natural discrete reduction: which direction to skew (symmetric, ask-skew, bid-skew). Captures the structural decision without the fine-grained magnitude question.

When discrete would be wrong: if an RQ required measuring the *magnitude* of optimal quote response to belief (smooth scaling with posterior). None of RQ1–RQ3 require this, so this caveat does not apply.

**Scaling.** 3 actions is the default. If R1 holds only marginally, 5 actions (sym, skew-small-bid, skew-large-bid, skew-small-ask, skew-large-ask) adds resolution before adding structural complexity. Never go higher — VI cost inflates quadratically and actions become redundant.

**No action masking.** Inventory bounds enforced through reward penalties, not masked actions. Keeps MDP structure clean and comparable with published meta-RL methods.

### Environment design process

The final MM environment is constructed by **iterative extension from Avellaneda-Stoikov**, adding structural elements one at a time until R1–R4 all hold. Each added element is justified by which requirement it addresses. Documented in a thesis appendix as the audit trail showing the env was not cherry-picked.

**Progression template** (actual stopping point determined by which requirements hold):

- **E0 — Vanilla AS (single regime).** Bounded discrete inventory, Bernoulli fills, standard AS reward. *R1 fails* (one regime). Purpose: PPO sanity check — PPO should match AS analytical optimal quotes.
- **E1 — AS with regime-switched volatility or drift.** Same as E0, with mid-price dynamics governed by an HMM regime. Test whether R1 and R4 hold via VI and posterior rollout.
- **E2 — AS with regime-switched directional fill intensities.** Regime affects arrival rates on bid vs ask side (bull = buy pressure, bear = sell pressure). This typically makes R1 and R4 hold cleanly — own-fills carry direct regime evidence.
- **E3+ — further structural additions only if needed** to strengthen a marginally-holding requirement.

**Stopping criterion.** All four requirements hold with clear margin (not marginal), verified by the script below. Do not add complexity beyond this point. "Real markets do X" is not a justification — that is calibration territory, excluded by design.

**Verification script: `oracles/verify_requirements.py`**

CLI:
```
uv run python -m oracles.verify_requirements --env-config experiments/configs/envs/{env_version}.yaml
```

What it checks, per requirement:
- **R1**: runs VI on the full-info MDP, extracts per-regime optimal policies, reports fraction of state space where policies disagree and value loss from playing the wrong regime's policy.
- **R2**: runs short per-regime PPO (fewer iterations than M1's full run — this is a sanity check, not a final result) on each locked regime, compares final return to VI optimum, reports the ratio per regime.
- **R3**: runs short regime-agnostic PPO and short Oracle-PPO, reports the return gap with seed-level CIs.
- **R4**: rolls out sample trajectories through the env, computes analytical HMM posterior at each step via the forward algorithm, reports posterior entropy decay curve and final-step entropy. Also runs short Belief-PPO and compares to regime-agnostic PPO to report the Belief-PPO gap-closure fraction.

What it outputs:
- `results/milestones/M2/stats_M2_requirements.json` with the full pass/fail report for each R (schema specified under M2).
- All six M2 plots in `figures/milestones/M2/`.
- Final stdout line: `[verify_requirements] OK | env_version=e2 | R1=pass | R2=pass | R3=pass | R4=pass | all_pass=true | output=results/milestones/M2/`

Design notes:
- The verification runs use `--fast` budgets internally (fewer iterations, fewer seeds) because they're diagnostic, not final-result-generating. A full-budget version of each component is what M1 and M3 do.
- **Threshold adjustment for short-run context**: the M2 R2 check uses `min_ratio >= 0.85` rather than M1's `return_ratio >= 0.95`, because `--fast` budgets don't always reach the final 10% of performance. M2 is asking "does PPO approach VI on locked regimes?" not "does PPO fully converge to VI?" The stricter `>= 0.95` is enforced by M1 (full-budget) and re-verified with M3 per-regime PPO (full-budget).
- If the verification runs take too long to iterate env designs quickly, reduce their internal budgets further — the goal is *directional confidence* in whether an R holds, not tight CIs. Tight CIs come from M3.
- The script is idempotent: rerunning it on the same env config overwrites the JSON and plots.

Each env iteration runs this script. Env revision is justified by which requirement was failing; new revision is adopted when all four pass. Full trajectory (E0 → ... → final) recorded in appendix.

**Anti-pattern to avoid.** The env must be tuned to satisfy requirements, *not* to make specific methods succeed. If you find yourself tuning the env until VariBAD beats RL², the thesis becomes circular. R1–R4 are method-agnostic well-posedness conditions; the env passes or fails independently of which meta-RL method is being evaluated.

### Parameter-choice discipline

Within the space of parameterizations satisfying R1–R4, parameters are hand-chosen and frozen in the env config. Parameters most relevant to satisfying the requirements:

- **Regime transition matrix**. Diagonal-dominant; per-step switching probability small enough to give R4 (regimes persist long enough to be inferred from finite history) but not so small that the mixed-regime setting degenerates into "almost always in one regime."
- **Regime-conditional fill probabilities** (or intensities, or price dynamics, depending on final env form). Distinct enough to give R1 (policies disagree) and R4 (regimes distinguishable from observations).
- **Reward weights**. Standard MM objective (spread capture minus inventory penalty); tuned to make the inventory constraint bind, otherwise the problem trivializes to "always quote symmetrically."

Transition frequency and regime distinguishability are also natural ablation axes — running experiments across a sweep is an expected robustness check for RQ2 (how posterior approximation quality depends on task difficulty).

### Regime locking

Env accepts `lock_regime: int | None`. When set, regime transitions are disabled — used for per-regime PPO training and for targeted evaluation.

### Toy validation environment

`regime_bandit.py`: 2-armed bandit where the better arm switches according to a Markov chain. Cheap, stresses the same Markovian latent structure as MM, used as shared validation testbed that all methods should pass before committing to full MM experiments.

## Implementation validation suite

Separate from the thesis research experiments. Purpose: verify that each method's implementation is correct by reproducing qualitative published behavior on standard meta-RL benchmarks. Not part of the thesis contribution — results go in an appendix with a single sentence in the main text. Prevents the failure mode where a subtle implementation bug invalidates MM results months in.

### Validation tasks

**Two-armed Bernoulli bandit** (`envs/validation/bandit.py`). Classic meta-RL sanity check from the RL² and VariBAD papers. Arms have fixed Bernoulli parameters within an episode, resampled across episodes from a prior. Expected behavior:
- PPO: converges to ~50/50, no within-episode adaptation.
- RL², VariBAD: approach Bayes-optimal, exploring early then exploiting.
- VariBAD posterior should sharpen with observations — plot this.

**Random-goal gridworld** (`envs/validation/gridworld.py`). Goal location sampled per episode, agent must find and exploit. Standard meta-RL benchmark.
- PPO: blind search pattern averaged over goal distribution.
- Meta methods: learn explore-then-exploit strategy, substantial gap over PPO.

**Regime-switching bandit** (`envs/regime_bandit.py`, shared with main experiments). Markov-switching arms. Validation-for-this-setting rather than published-comparison, since this is your own toy.

### Pass criteria

Not quantitative. Each method must satisfy:
1. **Learns**: curve goes up, not flat.
2. **Ranks correctly**: method ordering matches published qualitative ordering (e.g. RL² > PPO on bandit, VariBAD ≥ RL² on goal-search).
3. **Qualitative behavior**: exploration patterns and (for belief-based methods) posterior sharpening match expectations.

Exact numerical match with published curves is not required — hyperparameters, implementations, and compute differ. The question is *"does this implementation exhibit the qualitative behavior described in the original paper?"*

### Scope

Validation is implementation insurance, not research. Compute budget:
- 1 seed per method per task for initial smoke-test.
- 3 seeds per method per task for reported learning curves.
- Short training runs; these tasks are designed to converge quickly on CPU.
- Validation results live in `results/validation/`, separate from main experiments.
- One appendix figure per task showing learning curves across methods.

### Gating

Validation must pass before the corresponding method is included in MM experiments. If RL² fails the bandit validation, its MM results are not trustworthy — fix the implementation first.

## Compute plan

### Hardware

Mac mini M4 CPU. JAX CPU backend. No GPU.

### Training loop structure

Training is structured as a small number of **iterations** (outer loop steps), each running many **parallel environments** in a batched rollout. This trades total environment steps for statistical stability within each iteration — short curves, low noise.

**Core shape (non-negotiable):**
- **Iterations (outer loop)**: capped at 100. This is the x-axis on learning curves.
- **Parallel envs per iteration**: large (hundreds to low thousands), `vmap`-batched. Exact count tuned per experiment to fit M4 memory.
- **Rollout length per env per iteration**: short horizon (e.g., 128–512 steps per env per iteration).
- **Total env interactions per iteration** = parallel_envs × rollout_length. This is what the PPO update sees.

**Why this shape.** A learning curve with 100 iterations and many parallel envs per iteration is a **short line with low variance per point**. Each iteration aggregates across hundreds of parallel envs, giving a stable per-iteration estimate of return. 100 iterations is enough to see convergence but few enough to keep wall-clock manageable. Keep iterations low and parallel envs high — not the reverse.

**Statistical stability discipline.** If per-iteration returns are noisy (jagged learning curve even after seed averaging), the fix is **more parallel envs**, not more iterations. Variance across envs within an iteration should be reported in metrics.json so you can diagnose whether a jagged curve is genuinely high gradient variance or just too few envs averaged.

**JIT amortization.** The compiled rollout + update step is called exactly 100 times per seed. Keep function signatures shape-stable (fixed parallel_envs, fixed rollout_length, fixed batch shapes) so there's one compilation per seed, not recompilations mid-run.

**Seeds.** 5 seeds per method per experiment minimum, run sequentially (parallelism is inside each seed via `vmap` over envs, not across seeds). Seed-level confidence intervals reported. Pre-registered in the config.

### Implications

- JIT compilation overhead is one-time per seed — compile the full rollout + update step once, then run 100 iterations. Keep function signatures shape-stable.
- Env throughput matters less than at long-horizon training; agent forward/backward pass dominates.
- Ladder scope pre-commitment remains important. Even with short curves, 11 methods × 5 seeds × M4 is non-trivial wall-clock — plan for overnight runs.

### Configuration

All iteration counts, parallel env counts, rollout lengths, seed counts, and hyperparameters live in **one file per experiment** under `experiments/configs/`. No magic constants in code. The config file is the single source of truth for a run and is copied into `results/{experiment_name}/config.json` at run start for reproducibility.

### Hyperparameter discipline

The default failure mode for RL projects is per-experiment hyperparameter tuning that doubles or triples the true compute cost and creates irreproducible numbers. Avoid this with the following discipline:

**Tune once, freeze.** PPO core hyperparameters (learning rate, clip ratio, entropy coefficient, GAE lambda, value loss coefficient, minibatch size, epochs per update) are tuned *once* on the AS baseline in M1. From that point on, these values are frozen for the entire project. Every method that uses PPO internally — Oracle-PPO, Belief-PPO, per-regime PPO, the PPO inside RL² and VariBAD — uses the same core values.

**Method-specific hyperparameters are tuned during M4 validation** and then frozen for M5/M6:
- VariBAD: KL weight, latent dim, encoder/decoder sizes, reconstruction loss weight.
- RL²: recurrent hidden dim, input augmentation choice ((obs, prev_action, prev_reward) by default).
- Hypernet integration: hypernet hidden dim, target layer selection.
- Exploration bonus: bonus coefficient, novelty window size.

Each tuned per method on the cheapest validation task where the effect is visible, documented in the method's config file with a comment noting where the value came from.

**No retuning per experiment.** If a method wants different hyperparameters on MM than on the bandit, that's a *finding* (and a concerning one — it suggests hyperparameter sensitivity), not a convenience. Report it; do not silently retune.

**Sweeps are pre-registered.** If a hyperparameter sweep is done, it goes in the config with explicit values tried and the selection criterion. No post-hoc picking of the best seed.

**Composable configs.** Configs are built from a shared base (e.g., `base_ppo.yaml` has the core PPO values) and experiment-specific overrides (e.g., `m5_varibad.yaml` extends the base and adds VariBAD-specific values). This prevents drift where "PPO learning rate" accidentally has a different value in different experiments.

### Run modes (`--super-fast`, `--fast`, full)

Every script that can take longer than ~30 seconds supports three run modes via CLI flags. The flag modifies the loaded config in memory before execution — it never requires a separate config file.

**Both `--super-fast` and `--fast` use 1 seed.** Seed-level statistics are only meaningful in full runs. Fast modes are for pipeline verification and single-seed directional signal, not for claims about variance or ranking.

**`--super-fast`** — smoke test. Purpose: verify the pipeline compiles and runs end-to-end. Target runtime: under 30 seconds.
- Iterations: 2 (just enough to confirm the outer loop executes).
- Parallel envs: small (e.g., 16).
- Rollout length: short (e.g., 32).
- Seeds: **1**.
- JIT compilation still happens (this catches shape bugs and import errors).
- Output artifacts: JSON produced, plot optionally skipped or rendered as a placeholder. JSON must still validate against the expected schema — this is the primary check.

**`--fast`** — smoke test with signal. Purpose: verify that *learning is happening*, not just that the code runs. Target runtime: a few minutes.
- Iterations: 20.
- Parallel envs: moderate (e.g., 64–128).
- Rollout length: moderate (e.g., 128).
- Seeds: **1**.
- Output artifacts: full JSON and plot. Learning curve should show an upward trend even if not converged. Final return won't match full-run numbers and milestone pass criteria do not apply, but the slope of the curve tells you the method is functional.

**No flag (full)** — production run. Purpose: generate milestone/thesis artifacts.
- Iterations: 100.
- Parallel envs: config default (tuned per experiment to fit M4 memory).
- Rollout length: config default.
- Seeds: 5 (minimum).
- All milestone pass criteria apply.

**Implementation.** The flag is handled centrally in `training/config.py`:
```python
def apply_run_mode(cfg, mode):
    if mode == "super_fast":
        cfg.iterations = 2
        cfg.parallel_envs = 16
        cfg.rollout_length = 32
        cfg.num_seeds = 1
    elif mode == "fast":
        cfg.iterations = 20
        cfg.parallel_envs = 128
        cfg.rollout_length = 128
        cfg.num_seeds = 1
    return cfg
```

Every experiment script imports and calls this. No script implements its own version.

**When to use which.**
- `--super-fast` on every commit / before every long run / after every refactor. Five seconds of wait is cheap insurance against a crash at hour three.
- `--fast` when tweaking hyperparameters or debugging a method that isn't converging. Fast enough to iterate, slow enough to see whether the fix helped. Single seed — if behaviour seems seed-dependent, confirm with a full run rather than interpreting fast-mode noise.
- No flag for anything that produces a milestone artifact or a thesis figure.

**Plot conventions for fast modes.** Learning curves from fast modes are rendered without CI bands (single seed has no CI). Plotting code should detect `num_seeds == 1` from the summary JSON and draw a single line rather than a band — so the plot still looks sensible and isn't mistaken for a full run.

### Script output discipline

Every script that runs — training, evaluation, plotting, tests, utilities — produces **discrete, verifiable output** that confirms whether it worked.

**JSON is for decisions, PNG is for the human.** This is a load-bearing rule:

- **Claude Code reads JSON.** Every numeric or boolean pass criterion must be a field in the script's output JSON. Automated verification, milestone pass/fail decisions, and tuning loops all operate on JSON.
- **Claude Code does not read PNGs.** PNGs exist for the human (thesis reader, supervisor, you) to verify things make sense visually. They are not input to automated decisions.
- **If a pass criterion currently reads like "curve should look X,"** it must be operationalized as a JSON field: a ratio, a boolean, a fitted slope, a computed entropy decay. Visual judgments are not pass criteria for Claude Code.
- **When Claude Code finds itself wanting to inspect a PNG to make a decision, the JSON schema is incomplete.** Add the missing field to the JSON instead.

This is also a token-cost rule — reading a PNG costs ~2000 tokens; reading the equivalent JSON costs ~100. Over a long agentic session, the difference compounds significantly.

**Required output from every script.**
1. **A summary JSON** written to a predictable path. Contains enough stats that Claude Code can make every downstream decision without touching a PNG.
2. **A final line to stdout** of the form:
   ```
   [SCRIPT_NAME] OK | key_stat_1=... | key_stat_2=... | output=path/to/artifact
   ```
   or on failure:
   ```
   [SCRIPT_NAME] FAIL | reason=...
   ```
   This line is greppable, parseable, and fits on one terminal line. Long runs can also print intermediate progress lines — but the final line is what makes the run verifiable without reading the full log.

**Exit code.** Non-zero on any unexpected failure. Zero only if the script completed and the output JSON was written.

**No silent success.** A script that completes without producing its expected JSON is a bug, even if it didn't crash. The output JSON is the contract.

**Examples of the output discipline in action.**

*Training script (`training/train.py`):*
- Writes `results/{experiment}/metrics.json` and `results/{experiment}/eval.json`.
- Final stdout: `[train] OK | method=varibad | final_return=12.4 | gap_closed=0.67 | output=results/varibad_mm/`

*Plotting script (`plotting/learning_curves.py`):*
- Writes a summary JSON noting which input files were read and what was plotted.
- Final stdout: `[plot_learning_curves] OK | input_runs=5 | methods=['ppo','rl2','varibad'] | output=figures/milestones/M5/fig_rq2_learning_curves.png`

*Test script (`tests/test_beliefs.py`):*
- Writes a test report JSON with per-test pass/fail.
- Final stdout: `[test_beliefs] OK | tests_run=12 | tests_passed=12 | output=results/tests/beliefs.json`

*VI oracle (`oracles/value_iteration.py`):*
- Writes convergence stats and final policy.
- Final stdout: `[vi] OK | converged_at_iter=847 | bellman_residual=1.2e-7 | output=results/oracles/vi_regime_0.json`

**Why this matters.** When running dozens of experiments in batches (e.g., `run_ladder.sh`), the only practical way to verify everything worked is `grep "OK\|FAIL" logs/*.log`. Without this discipline, failure diagnosis becomes archaeology through full logs.

**Script output schema.**

Every script's summary JSON conforms to a minimal shared schema:
```json
{
  "script": "train",
  "status": "OK",
  "started_at": "2026-...",
  "finished_at": "2026-...",
  "run_mode": "super_fast|fast|full",
  "config_used": { ... the effective config after run-mode overrides ... },
  "key_stats": { ... script-specific stats ... },
  "outputs_written": [ "path/to/artifact1", "path/to/artifact2" ],
  "error": null
}
```

Every script writes this alongside its main output. Enables automated milestone verification and lets `make_milestone.sh` check that every step produced its expected artifacts before regenerating plots.

## Repo layout

This is the complete set of files the project will contain when M0–M6 are finished. Optional files (M7 ablations, scope-permitting figures) are marked.

```
thesis/
├── agents/
│   ├── __init__.py
│   ├── base.py                      # abstract Agent class, shared interface
│   ├── dummy.py                     # random-policy agent (M0 pipeline validation only)
│   ├── ppo.py                       # vanilla PPO (MLP). Used as-is for the regime-agnostic
│   │                                #   floor AND, wrapped with stack_obs, for stacked-obs PPO.
│   ├── ppo_oracle.py                # PPO conditioned on true regime one-hot
│   ├── ppo_belief.py                # PPO conditioned on analytical HMM posterior
│   ├── ppo_per_regime.py            # PPO trained on locked single regime (wraps ppo.py N times)
│   ├── rl2.py                       # RL² (recurrent PPO with prev action/reward)
│   ├── varibad.py                   # VariBAD (variational belief + PPO + ELBO)
│   └── modules/                     # composable ablation components (not standalone agents)
│       ├── __init__.py
│       ├── hypernet.py              # hypernetwork integration: belief → policy weights.
│       │                            #   Composes with rl2.py or varibad.py via config flag.
│       └── exploration_bonus.py     # belief-novelty auxiliary reward. Novelty signal source
│                                    #   is method-specific (RL² hidden state, VariBAD mu).
│
├── envs/
│   ├── __init__.py
│   ├── base.py                      # abstract Env class, JAX-compatible interface
│   ├── mm_reduced.py                # main env: reduced-form MM. Accepts a regime_config.
│   │                                #   Supports E0 (no regimes), E1+ (regime-switching),
│   │                                #   and lock_regime: int | None for per-regime PPO.
│   ├── regime_bandit.py             # Markov-switching bandit (M4 validation + shared toy)
│   ├── validation/                  # implementation-validation envs (not thesis-relevant)
│   │   ├── __init__.py
│   │   ├── dummy.py                 # constant-reward env (M0 pipeline validation)
│   │   ├── bandit.py                # 2-armed Bernoulli bandit (M4 RL²/VariBAD sanity)
│   │   └── gridworld.py             # random-goal gridworld (M4)
│   └── wrappers/
│       ├── __init__.py
│       └── stack_obs.py             # observation-stacking wrapper; stacked-obs PPO = ppo.py + this
│
├── beliefs/
│   ├── __init__.py
│   ├── hmm_posterior.py             # analytical forward algorithm for HMM belief
│   └── oracle.py                    # true regime extraction from env_state.
│                                    #   Separated from env to enforce leak discipline: only
│                                    #   Oracle-PPO, Belief-PPO, per-regime PPO may import this.
│
├── oracles/
│   ├── __init__.py
│   ├── analytical_as.py             # closed-form AS optimal return (M1 target)
│   ├── value_iteration.py           # VI on full-info MDP (used by R1, R2, M3 sanity checks)
│   └── verify_requirements.py       # R1–R4 verification script (M2 main tool). CLI:
│                                    #   uv run python -m oracles.verify_requirements --env-config ...
│
├── training/
│   ├── __init__.py
│   ├── train.py                     # main entry point; dispatches by agent/env in config
│   ├── rollout.py                   # lax.scan rollout utilities
│   ├── ppo_update.py                # shared PPO loss/update (reused by every PPO-based agent)
│   ├── varibad_update.py            # VariBAD ELBO + PPO joint update (separate optimizers)
│   └── config.py                    # dataclass configs per experiment + apply_run_mode()
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # gap-closed fractions, regime classification accuracy,
│   │                                #   factorial marginal means
│   ├── comparisons.py               # paired Wilcoxon + Holm correction + bootstrap CI;
│   │                                #   produces the "supported / not supported" decisions
│   ├── posterior_compare.py         # bidirectional mapping error vs analytical HMM posterior.
│   │                                #   Uses a trained linear probe from inferred belief to
│   │                                #   simplex; probe choice documented and held fixed.
│   └── posterior_performance.py     # M6 decoupling analysis: scatter of gap-closed vs
│                                    #   posterior error, per-method Spearman correlations
│
├── utils/
│   ├── __init__.py
│   └── script_output.py             # write_summary() helper; every script calls at end.
│                                    #   Produces the shared-schema JSON and OK/FAIL stdout line.
│
├── plotting/
│   ├── __init__.py
│   ├── style.py                     # apply_style(): colors, fonts, sizes. Called by every plot.
│   ├── load_results.py              # parse .json result files, handle missing fields
│   ├── make_milestone.py            # regenerate all figures for a milestone. Invoke:
│   │                                #   uv run python -m plotting.make_milestone M{n}
│   ├── learning_curves.py           # training curves with CI bands (RQ1, RQ2, M1, M2, etc.)
│   ├── gap_decomposition.py         # fig_rq1_ceilings_bar, fig_rq1_gap_fractions
│   ├── posterior_quality.py         # fig_rq2_posterior_error, fig_M4_varibad_posterior_sharpening
│   ├── factorial.py                 # fig_rq2_factorial (2×2×2 grouped bars)
│   ├── difficulty_sweep.py          # fig_rq3_persistence_sweep, fig_rq3_distinguishability_sweep
│   └── posterior_vs_performance.py  # fig_rq3_posterior_vs_performance (decoupling scatter)
│
├── experiments/
│   └── configs/
│       ├── base/                    # composable base configs (hyperparameter discipline)
│       │   ├── base_ppo.yaml        # core PPO hyperparameters, tuned in M1, frozen thereafter
│       │   ├── base_rl2.yaml        # extends base_ppo with RL² specifics
│       │   ├── base_varibad.yaml    # extends base_ppo with VariBAD specifics
│       │   └── base_env_mm.yaml     # shared MM env parameters
│       ├── envs/                    # env-specific configs for M2 iteration
│       │   ├── e0_as_baseline.yaml
│       │   ├── e1_vol_switched.yaml
│       │   ├── e2_fill_switched.yaml
│       │   └── e_final.yaml         # symlink to whichever env version passed R1–R4
│       ├── m0_dummy.yaml            # M0 pipeline test
│       ├── m1_ppo_as.yaml           # M1: PPO on AS baseline
│       ├── m3_per_regime.yaml       # M3: reference levels
│       ├── m3_oracle.yaml
│       ├── m3_belief.yaml
│       ├── m3_regime_agnostic.yaml
│       ├── m4_rl2_bandit.yaml       # M4: validation, one file per (method, task) pair
│       ├── m4_rl2_gridworld.yaml
│       ├── m4_rl2_regime_bandit.yaml
│       ├── m4_varibad_bandit.yaml
│       ├── m4_varibad_gridworld.yaml
│       ├── m4_varibad_regime_bandit.yaml
│       ├── m4_ablation_hypernet.yaml       # hypernet vs concat on toys
│       ├── m4_ablation_exploration.yaml    # bonus vs no-bonus on toys
│       ├── m5_ladder_ppo.yaml       # M5 core ladder: one file per ladder rung
│       ├── m5_ladder_stacked_ppo.yaml
│       ├── m5_ladder_rl2.yaml
│       ├── m5_ladder_varibad.yaml
│       ├── m5_ladder_belief_ppo.yaml
│       ├── m5_ladder_oracle_ppo.yaml
│       ├── m5_ladder_per_regime_ppo.yaml
│       ├── m5_factorial_rl2_concat_bonus.yaml          # M5 factorial (6 extra cells;
│       ├── m5_factorial_rl2_hypernet_nobonus.yaml      #   rl2_concat_nobonus and
│       ├── m5_factorial_rl2_hypernet_bonus.yaml        #   varibad_concat_nobonus overlap
│       ├── m5_factorial_varibad_concat_bonus.yaml      #   with m5_ladder_rl2 and
│       ├── m5_factorial_varibad_hypernet_nobonus.yaml  #   m5_ladder_varibad, don't re-run)
│       ├── m5_factorial_varibad_hypernet_bonus.yaml
│       ├── m5_mu_only_varibad_hypernet.yaml  # M5 mu-only vs full-posterior ablation
│       ├── m6_persistence_sweep.yaml     # M6: 3 points × methods × seeds
│       ├── m6_distinguishability_sweep.yaml
│       └── m6_heatmap.yaml               # optional 3×3 grid
│
├── tests/
│   ├── __init__.py
│   ├── test_envs.py                 # env invariants (inventory bounds, rewards finite, etc.)
│   ├── test_beliefs.py              # HMM posterior matches brute-force on short sequences
│   ├── test_oracles.py              # VI convergence, policy stability
│   ├── test_leak.py                 # regression: regime doesn't leak into non-oracle agents
│   ├── test_run_modes.py            # --super-fast < 30s, --fast < 5min
│   └── test_script_output.py        # every write_summary JSON validates against schema
│
├── scripts/
│   ├── make_milestone.sh            # orchestrate a milestone: run training, dump JSON, plot,
│   │                                #   check all artifacts present. Called as:
│   │                                #   scripts/make_milestone.sh M{n}
│   ├── run_sweep.py                 # M6 orchestration: run a grid of (difficulty × method
│   │                                #   × seed) configs and aggregate into stats_M6_sweep.json
│   └── run_ladder.sh                # batch runner for M5 ladder (calls train.py per method)
│
├── results/                         # gitignored; generated outputs
│   ├── milestones/                  # milestone-level artifacts
│   │   └── M{n}/
│   │       ├── stats_*.json         # verification JSONs per milestone
│   │       ├── PASS.md | FAIL.md    # review note, committed
│   │       └── ...
│   └── {experiment_name}/           # per-experiment outputs
│       ├── config.json              # effective config (run-mode applied), with commit hash
│       ├── summary.json             # shared-schema script output
│       ├── metrics.json             # per-iteration training metrics
│       ├── eval.json                # final evaluation numbers
│       └── checkpoints/             # optional saved params (flax serialization)
│
├── figures/                         # committed; PNGs referenced by LaTeX
│   ├── milestones/                  # milestone-internal figures
│   │   └── M{n}/
│   │       └── fig_M{n}_*.png
│   └── thesis/                      # figures cited in the thesis LaTeX
│       ├── fig_rq1_*.png
│       ├── fig_rq2_*.png
│       └── fig_rq3_*.png
│
├── pyproject.toml                   # uv-managed, exact version pins
├── uv.lock                          # committed lockfile for reproducibility
├── .gitignore                       # at minimum: results/, __pycache__/, .venv/, *.pyc
├── README.md                        # project summary + how to reproduce M0
└── CLAUDE.md                        # this file — the project spec
```

### Optional / scope-permitting files

These appear in the layout only if the corresponding scope decision goes "yes":

- `agents/modules/sequential_vs_set_encoder.py` — if the sequential-vs-set-encoder question gets tested as a mini-ablation in the discussion chapter.
- `envs/mm_variants.py` — if RQ3 ablations need env variants beyond the persistence / distinguishability sweep (e.g., different regime counts).
- `oracles/belief_vi.py` — VI on the belief-state POMDP (Oracle B). The current thesis framing uses Belief-PPO (analytical posterior + PPO) as the inferred-belief ceiling; Oracle B would be a stronger ceiling if included, but adds implementation complexity. Skip unless Belief-PPO underperforms significantly and you need a tighter ceiling.
- `experiments/configs/m7_*.yaml` — M7 supplementary ablations (stop-gradient, episode length, hidden-state probe). One file per ablation.

### Key structural points

**No HyperX, PEARL, classifier-head, or `_hn` files.** Hypernet integration and exploration bonus are composable modules in `agents/modules/`, selected via config flags on `rl2.py` and `varibad.py`. No `rl2_hn.py`, `varibad_hn.py`, `hyperx.py`, or `pearl.py`.

**Stacked-obs PPO = PPO + wrapper.** No separate `ppo_stacked.py`. The ladder rung "stacked-obs PPO" is `ppo.py` running on an env wrapped by `envs/wrappers/stack_obs.py`, selected via config.

**Per-regime PPO = PPO × N.** `agents/ppo_per_regime.py` is a thin wrapper that trains N independent instances of `ppo.py` on locked regimes, then reports either the regime-conditional optimum (evaluation on the matching regime) or an oracle-switched composite.

**All composition is in configs, not code.** Whether an RL² run uses hypernet or concat, bonus or no bonus, is a config value. The same `rl2.py` file serves all factorial cells.

**Milestone artifacts have two homes.** `results/milestones/M{n}/` for JSONs and `figures/milestones/M{n}/` for PNGs. Thesis figures are additionally copied to `figures/thesis/` under the `fig_rqN_*.png` name.

## Statistical methodology

Every numeric comparison in the thesis follows the rules in this section. These are committed choices; no per-experiment variation.

### Seed protocol

- **Primary experiments (M3, M5, M6): 5 seeds per configuration.** Chosen as the minimum at which bootstrap CIs are informative at the effect sizes expected from the M2 ceiling gap. At smaller n the bootstrap degenerates; at larger n compute becomes infeasible on M4.
- **Validation experiments (M4): 3 seeds per configuration.** Lower because M4 checks qualitative behavior, not quantitative claims.
- **Env verification (M2): 3 seeds per locked run.** M2 uses short runs as diagnostics, not final numbers.
- **Fixed seed values across experiments.** Seeds `{0, 1, 2, 3, 4}` are used consistently across methods. This means two methods' results are *paired* by seed (same env trajectories, same initialization randomness) — which enables paired statistical tests (lower variance, more power) and is what justifies using only 5 seeds.

### Confidence intervals

- **95% bootstrap CI across seeds.** Computed via 10,000 bootstrap resamples of the per-seed final-return values.
- **NOT parametric ±1.96·std/√n.** Parametric intervals assume normality which is unsafe at n=5 with possibly-non-Gaussian seed-level returns.
- **Applied to any reported scalar.** Return, gap-closed fractions, posterior error, etc. Every top-level numeric field in the JSON with a corresponding CI uses bootstrap.
- **`CI` in figure labels and in this document's prose always means 95% bootstrap CI from seeds.** No other CI convention is used.

### Hypothesis testing for method comparisons

Every "method X vs method Y" comparison in the thesis follows this protocol:

1. **Paired test by seed.** Since methods share seed values, use a paired test (Wilcoxon signed-rank) on the per-seed differences, not an unpaired test on the raw returns. This controls for shared randomness.
2. **Report the p-value and the effect size** (median difference, not just mean). Both matter: a statistically-significant but tiny effect is different from a large effect that's underpowered.
3. **Multiple-comparison correction** via Holm-Bonferroni across the pre-registered primary comparisons (see below). Exploratory comparisons are explicitly labeled as such in the thesis and not corrected.
4. **Decision rule.** A comparison is "supported" if Holm-Bonferroni-corrected p < 0.05 AND the paired-difference CI excludes zero. Either condition alone is not sufficient.

Why Wilcoxon signed-rank rather than paired t-test: non-parametric, makes no distributional assumption, more robust at n=5.

### Factorial analysis (M5 factorial)

The 2×2×2 factorial has 3 main effects and 4 interactions. Main effects are the primary analyses; interactions are exploratory unless the main thesis narrative depends on them.

- **Main effects computed as marginal means.** E.g., hypernet main effect = (mean return averaged over all hypernet cells) − (mean return averaged over all concat cells).
- **Significance via paired comparisons across the factorial's paired cells.** E.g., hypernet vs concat uses the 4 pairs (rl2-concat-nobonus vs rl2-hypernet-nobonus, and 3 more).
- **Pre-registered as primary comparisons**: integration main effect, exploration main effect. Belief-source main effect and all interactions are exploratory.

### Pre-registered primary hypotheses

Before any M5/M6 experiment runs, these hypotheses are committed in writing. Results on these are the "primary" findings; everything else is exploratory and labeled as such in the thesis.

**RQ1 primary hypothesis.** The four reference levels order as regime-agnostic < belief-PPO ≤ oracle-PPO ≤ per-regime PPO, with non-zero gaps between the strict inequalities. Pre-committed: if belief-PPO ≈ oracle-PPO (CI overlap), inference cost is trivially small and the RQ2 motivation is weaker.

**RQ2 primary hypotheses (pre-committed before running M5):**
1. VariBAD's final return > regime-agnostic PPO's final return (paired Wilcoxon, Holm-corrected).
2. RL²'s final return > regime-agnostic PPO's final return.
3. VariBAD's final return > stacked-obs PPO's final return.
4. RL²'s final return > stacked-obs PPO's final return.
5. Integration main effect (hypernet vs concat): ≠ 0.
6. Exploration main effect (bonus vs no bonus): ≠ 0.

Six primary comparisons. Holm-Bonferroni correction applied to the p-values. Family-wise error rate controlled at 0.05.

**RQ3 primary hypotheses (pre-committed before M6):**
1. Gap-closure is monotonically non-increasing in difficulty (persistence axis).
2. Gap-closure is monotonically non-increasing in difficulty (distinguishability axis).
3. Posterior error and gap-closure are correlated across methods and difficulty levels (Spearman ρ ≠ 0).

All other comparisons (specific method rankings at specific difficulty points, interaction effects, mu-only ablation, etc.) are exploratory.

**Why this matters.** Without pre-registration, a reviewer (or your own subconscious) can find patterns in noise by peeking at results first. Committing the primary hypotheses in writing before running M5/M6 disciplines the analysis. Exploratory findings are still reportable — just labeled differently.

### Effect size reporting

Every pairwise comparison reports, alongside the p-value:

- **Absolute effect**: median paired difference.
- **Relative effect**: median paired difference as a fraction of the Oracle-PPO gap. (A 2-unit absolute difference means different things depending on whether the total gap is 10 or 100.)
- **Cliff's delta** or a similar rank-based effect size, since returns may not be normally distributed.

### Handling seed outliers

Seeds occasionally fail catastrophically (NaN gradients, collapsed policies). Protocol:
- **Never delete a seed silently.** If a seed produces a NaN or an implausible return, investigate.
- **If the failure is reproducible** (same seed fails consistently, bug diagnosed), fix the bug and rerun all methods with the same seed set.
- **If the failure is random** (different seeds fail in different reruns), this is training instability; increase seeds and report the failure rate alongside the main results.
- **Never use a "reject seeds more than 2σ from the mean" heuristic.** That's post-hoc selection.

### Correlated seeds caveat

Because seeds are shared across methods, any bias in a specific seed (e.g., seed 2 happens to produce an unusually hard env trajectory) affects all methods. This is the cost of shared-seed pairing. Partial mitigation: report whether primary findings hold when any single seed is removed (leave-one-out sensitivity check). If removing one seed flips a primary finding, that finding is not robust — report accordingly.

### Reported statistics per comparison

Every primary comparison in the thesis reports:
```
{method A} vs {method B}:
  Median paired difference: Δ (95% bootstrap CI: [Δ_lo, Δ_hi])
  Wilcoxon signed-rank p-value: p (Holm-corrected: p_holm)
  Cliff's delta: d
  n_seeds: 5
  Decision: supported / not supported
```

This exact template lives in `evaluation/comparisons.py` and is the unit of reporting.

### Scope of claims

The methodology supports claims of the form:
- "On this env family, with hyperparameters tuned on standard meta-RL benchmarks and frozen, method X is supported / not supported as better than method Y at the α=0.05 family-wise level."
- "Across the persistence/distinguishability sweep, method X's gap-closure is monotonically non-increasing in difficulty / varies non-monotonically."
- "Posterior approximation error and task performance are correlated with Spearman ρ = ... / are decoupled."

The methodology does **not** support claims of the form:
- "Method X is the best meta-RL method." (Only tested on one env family.)
- "These findings apply to real market making." (Env is stylized.)
- "Method X is always better than Y." (Pre-registered hypotheses are directional and specific.)

This scoping is stated explicitly in the thesis discussion chapter.

### Posterior-quality measurement

Posterior approximation error is not well-defined until the method's belief representation is mapped into the simplex over regimes. Different choices of mapping give different numbers. This project locks in the following protocol:

- **Belief-PPO** is already in simplex form; no mapping needed.
- **VariBAD**: the latent Gaussian is decoded via a linear probe fit to the analytical posterior on a held-out trajectory set (separate from training, 5000 trajectories). The probe is trained once after M5 completes and frozen.
- **RL²**: hidden state is decoded via a linear probe fit the same way.
- **Error metric**: symmetric KL divergence between the mapped posterior and the analytical posterior, averaged over all timesteps in held-out trajectories.

Why linear probe rather than MLP: the probe is a measurement tool, not part of the method. A nonlinear probe could recover a posterior that the policy itself is not using, inflating the apparent posterior quality. Linear probe is the minimum that asks "is the information present in a form the policy could easily use?"

The probe R² on its training set is reported alongside the error metric. If R² is low, the "error" is partly probe-fit failure rather than method failure, and this caveat is acknowledged.

Robustness: the main results are replicated with an MLP probe and reported in an appendix. If the method ranking is the same under both probe choices, the finding is robust to probe form. If it changes, the choice of probe matters and both numbers are reported with discussion.

## Interface contracts

### `agents/base.py`

```python
class Agent:
    def init(self, key, obs_space, action_space, config) -> AgentState: ...
    def act(self, state, obs, key) -> tuple[action, new_state]: ...
    def update(self, state, trajectory_batch) -> tuple[new_state, metrics]: ...

    # Optional, implemented by belief-based agents:
    def infer_belief(self, state, history) -> Belief: ...

    # Optional metadata used by the training loop:
    requires_regime_label: bool = False  # only Oracle-PPO, per-regime PPO
    requires_analytical_posterior: bool = False  # only Belief-PPO
    is_recurrent: bool = False  # RL², VariBAD
    produces_belief_for_eval: bool = False  # all belief-based methods
```

`AgentState` is a pytree containing params, optimizer state, and any recurrent hidden state. Must be JAX-compatible (static shapes).

**Recurrent hidden state handling.** For recurrent agents (`is_recurrent == True`):
- The hidden state is part of `AgentState`, threaded through `act` calls.
- At episode boundaries, the hidden state is reset to the `init_hidden` value. This is the training loop's responsibility — the loop detects `done=True` and calls `reset_hidden_for_done` before the next `act`.
- RL² specifically augments input with `(obs, prev_action, prev_reward)`; this concatenation happens inside the agent, not in the env.

**Belief exposure for evaluation.** For agents with `produces_belief_for_eval == True`:
- `infer_belief(state, history)` returns the agent's current belief as a probability distribution over regimes (length-3 array summing to 1).
- The evaluation loop calls this on held-out trajectories to compute posterior-approximation error vs the analytical HMM posterior.
- Non-belief agents do not implement this method. PPO and stacked-obs PPO have no belief; the evaluation just skips them.

**Separate metrics for composite objectives.** Agents with auxiliary losses (VariBAD's ELBO, exploration bonus when enabled) return metrics dict with clear keys:
```python
{
    "ppo/policy_loss": ...,
    "ppo/value_loss": ...,
    "ppo/entropy": ...,
    "varibad/reconstruction_loss": ...,
    "varibad/kl": ...,
    "exploration_bonus/mean_bonus": ...,   # when exploration axis is on
}
```
Namespacing by component makes failure diagnosis trivial ("VariBAD's KL is blowing up" vs "VariBAD's return isn't improving" are different debugging paths).

### `envs/base.py`

```python
class Env:
    def reset(self, key) -> tuple[obs, env_state]: ...
    def step(self, env_state, action, key) -> tuple[obs, env_state, reward, done, info]: ...
```

`env_state` is a pytree containing everything needed for dynamics, *including the latent regime*. This is how `oracles/` and `beliefs/` access ground truth for evaluation while agents only see `obs`.

### Ground-truth leak prevention

`info` dict from `env.step` contains fields for both agent-accessible and evaluation-only information. The training loop splits:
- `agent_info`: what the agent may use during training (e.g. done flags, action masks).
- `eval_info`: what evaluation code sees (true regime, analytical posterior).

Agents must not read `eval_info` except where explicitly allowed (Oracle-PPO reads true regime; Belief-PPO reads analytical posterior; per-regime PPO reads locked regime at env config time, not from step output).

A leak here would silently invalidate results. Enforce with test cases.

## Results format

### JSON schema

Each experiment produces one directory under `results/{experiment_name}/` containing:

**`config.json`** — full run config (agent, env, hyperparameters, seed, commit hash).

**`metrics.json`** — per-iteration metrics as arrays (one entry per iteration, typically 100 entries):
```json
{
  "iteration": [0, 1, 2, ..., 99],
  "return_mean": [...],              // mean return across parallel envs
  "return_std": [...],               // std across parallel envs
  "policy_loss": [...],
  "value_loss": [...],
  "entropy": [...],
  "posterior_mse": [...],            // vs analytical HMM posterior, where applicable
  "regime_accuracy": [...],          // where applicable
  "wall_time": [...]                 // cumulative seconds
}
```

Metrics are aggregated *across parallel envs within an iteration*, so each iteration produces one scalar per metric. Confidence intervals in plots come from aggregating across seeds, not across envs within a seed.

**`eval.json`** — final evaluation numbers (return CI, gap closed, per-regime breakdown).

JSON for metrics, not pickle: human-readable, diff-able, stable across Python versions, decouples plotting from training code. Checkpoints (if saved) go in `checkpoints/` subdir as flax serialization.

## Plotting

Plotting scripts take JSON paths as input, produce PNGs as output, have no dependency on training code. Regenerate all thesis figures via `scripts/make_figures.sh`.

This decoupling means:
- Figures can be iterated on during writing without re-running training.
- Re-running plots after config changes is instant.
- Plot code is not in the critical path for training correctness.

All figures committed to `figures/`. Final PNGs directly referenced in LaTeX.

### Learning curve conventions

All learning curves follow the same visual format so figures compare cleanly across milestones:

- **X-axis**: iteration number (0 to ~100). Labelled "Iteration" — not "Environment steps", not "Timesteps". The iteration is the unit of comparison because parallel-env count may differ across experiments.
- **Y-axis**: mean episode return. Labelled with units where meaningful.
- **Line**: mean across seeds, per iteration.
- **Band**: 95% bootstrap CI across seeds (see Statistical methodology section), shaded at alpha=0.2.
- **Reference lines**: where relevant (e.g. Oracle-PPO return, AS analytical optimum), drawn as horizontal dashed lines with labels.
- **Legend**: method names, consistent across figures in a milestone.
- **Color palette**: consistent across all figures in the thesis. Define once in `plotting/style.py`.

### Figure style (committed choices)

These are concrete, committed choices. No fighting matplotlib over them per figure.

**Size.** 5.5 × 3.5 inches for single-column figures; 7.0 × 4.0 for double-column figures that span the LaTeX page width. Bar-chart-with-many-bars figures may go to 7.0 × 3.5 for horizontal breathing room.

**DPI.** 300 for rasterized elements; vector-first where possible (matplotlib's default savefig produces PDF-quality PNG at 300 DPI).

**Font.** Matplotlib default sans-serif at 10pt for axis labels and tick labels, 9pt for legend, 10pt for annotations. Thesis body text is usually 11pt so 10pt in figures reads naturally. Larger than default matplotlib (7-8pt) because 8pt is unreadable at thesis print size.

**No figure titles.** LaTeX captions carry the title. The figure is just axes, labels, legend, and data.

**Gridlines.** Light grey, horizontal only, at major y-axis ticks. Matplotlib `alpha=0.3`. No vertical gridlines (iteration axis doesn't need them). No minor gridlines.

**Spines.** Top and right spines removed (seaborn "despine" style). Left and bottom only.

**Color palette (per-method, committed).** Colorblind-friendly choices, consistent across every plot in the thesis:
- `ppo`: `#7F7F7F` (grey)
- `stacked_ppo`: `#BCBD22` (olive)
- `rl2`: `#1F77B4` (blue)
- `varibad`: `#FF7F0E` (orange)
- `belief_ppo`: `#2CA02C` (green)
- `oracle_ppo`: `#000000` (black)
- `per_regime_ppo`: `#9467BD` (purple)

**Reference lines style.** Ceiling / floor reference lines as horizontal dashed lines, dash pattern `(5, 5)`, line width 1.0, color matching the method (so Oracle-PPO ceiling line is black, Belief-PPO ceiling line is green). Labeled at the right edge of the plot with the method name in the method's color.

**CI bands.** Shaded polygon with `alpha=0.2`, color matching the line. Mean as a solid line at `alpha=1.0`, line width 1.5.

**Error bars (bar charts).** Black, `capsize=3`, `linewidth=1.0`. Always seed-level CI, not within-seed std.

**Legend.** Inside the axes when space permits, outside (right side) otherwise. No frame (`frameon=False`). Font size 9pt.

**File naming.** Snake case, project-prefixed: `fig_rq{n}_*.png` for thesis figures, `fig_M{n}_*.png` for milestone-internal figures. Naming is stable so LaTeX references never break.

All of the above lives in `plotting/style.py` as a function `apply_style()` that's called at the top of every plotting script. No per-script customization; fight style choices once, not per-plot.

## Implementation pitfalls to avoid

Meta-RL and PPO have several well-known subtle bugs that silently produce wrong results. The spec prevents each of these by design; this section names them so anyone reviewing the code can verify prevention.

**Shared optimizer across ELBO and PPO losses (VariBAD).** The reconstruction loss, KL term, and PPO loss have different natural learning rates. A single optimizer averages these, training the encoder at a rate optimized for the policy (or vice versa). Design rule: `varibad.py` uses two `optax` optimizers — one for encoder/decoder, one for policy — updating their respective parameter groups independently. Verified via the training loop's `metrics` dict containing `varibad/encoder_lr` and `varibad/policy_lr` as separate values.

**Minibatch shuffling breaking recurrent temporal consistency.** A vanilla PPO update shuffles transitions across timesteps within a minibatch. For recurrent agents, this breaks the temporal order the recurrent state depends on. Design rule: in `training/ppo_update.py`, recurrent agents use trajectory-level minibatches (no within-trajectory shuffling). Agents with `is_recurrent=True` in their metadata trigger this code path automatically.

**Hidden state not reset at episode boundaries.** If the GRU hidden state carries over from one episode to the next, the agent effectively never resets its belief between "tasks," and published meta-RL behavior won't reproduce. Design rule: the training loop detects `done=True` and calls `reset_hidden_for_done` before the next `act` call. Verified by `tests/test_envs.py` (checks done flags trigger reset) combined with M4 validation on bandit (if hidden state leaked across episodes, RL² would be flat; M4 catches this).

**Learning rate shared between hypernet and base policy.** When using hypernet integration, the hypernet and the base policy have different gradient scales; sharing an LR means one is always training too fast or too slow. Design rule: `agents/modules/hypernet.py` takes its own LR from config (default tuned in M4); the base policy uses the PPO core LR from `base_ppo.yaml`.

**VariBAD encoder cold-start.** The encoder's prior at step 0 is untrained; early rollouts feed garbage-belief into the policy. Design rule: the encoder is pretrained on random-policy rollouts for a small warmup phase before joint training begins (warmup length in config, default 10 iterations). If this is skipped, early training is noisy but not catastrophic — this is an optimization not a correctness issue.

**Ground-truth regime leaking into non-Oracle observations.** Cosmetic field names can mislead. If the env's observation dict includes `regime` by accident, any agent that indexes into the obs dict picks it up. Design rule: `beliefs/oracle.py` is the single module that may extract the true regime from env state; only Oracle-PPO, Belief-PPO, and per-regime PPO may import it. Verified by `tests/test_leak.py`.

**Advantages computed across episode boundaries.** GAE with an episode that spans a reset produces wrong advantages. Design rule: advantages reset at episode boundaries; `training/ppo_update.py` masks `done` transitions when computing GAE. Checked by `tests/test_envs.py` which verifies advantage values are identical for standalone-episode rollouts vs multi-episode rollouts up to the first `done`.

**Value-network normalization drift.** Running reward/return normalization statistics computed during training can drift and produce misleading value estimates across seeds. Design rule: normalization statistics are per-seed (not shared across seeds in the same experiment), recomputed from scratch each run, and frozen before evaluation.

## Sanity checks

Before trusting any meta-RL result, verify:

1. **R1 — regime-conditional policy divergence**: Oracle A (full-info VI) converges to a stable policy on the MM MDP, and the per-regime optimal policies disagree on a non-trivial fraction of the state space with substantial value loss from playing the wrong regime's policy.
2. **R2 — PPO on locked regimes matches VI**: per-regime PPO trained on a locked regime matches the VI optimal policy on that regime. Must hold for every regime. If this fails, PPO implementation is broken and all downstream results are suspect.
3. **R3 — mixed-regime PPO is strictly suboptimal**: regime-agnostic PPO on the full HMM trajectories settles below both Oracle-PPO and per-regime PPO, with the gap exceeding seed-level variation.
4. **R4 — regime inferability from history**: Belief-PPO (analytical HMM posterior) closes a non-trivial fraction of the Oracle-PPO gap. If not, regimes are not inferable from history and no meta-RL method can succeed.
5. **Analytical HMM posterior is correct**: forward algorithm output matches closed-form posterior on synthetic test sequences with known regime ground truth.
6. **No ground-truth leaks**: agents that should not see regime (everything except Oracle-PPO, Belief-PPO, per-regime PPO) cannot achieve Oracle-PPO-level performance.
7. **Validation suite passes**: each method exhibits published qualitative behavior on the 2-armed bandit and random-goal gridworld before being included in MM experiments. See "Implementation validation suite" above.
8. **Toy validation**: all methods learn on the regime-switching bandit before being trusted on MM.

## Limitations and rigor disclosures

This section catalogs what was fixed during spec design vs what remains as an acknowledged limitation of the project vs what was already properly defended. The goal is to be explicit about the thesis's shortcomings so that (a) reviewers see that known-limitations have been considered rather than overlooked, and (b) claims are scoped to what the methodology actually supports.

### Rigor fixes incorporated into the spec

Real gaps caught during spec review that now have committed solutions:

1. **CI convention ambiguity.** Previously "95% CI or ±1 std" (which are different things). Now committed to 95% bootstrap CI across seeds with 10,000 resamples, consistently.
2. **No paired statistical testing.** Previously "CIs don't overlap" as the informal decision rule. Now paired Wilcoxon signed-rank tests on per-seed differences, exploiting the fact that seeds are shared across methods.
3. **No multiple-comparison correction.** Now Holm-Bonferroni across 6 pre-registered primary hypotheses (4 ladder + 2 factorial main effects).
4. **No pre-registration.** Primary hypotheses are now committed in writing before M5/M6 experiments run. Everything else is labeled exploratory in the thesis.
5. **Informal decision rule.** Now: a comparison is "supported" iff Holm-corrected p < 0.05 AND the paired-difference CI excludes zero. Both conditions together.
6. **Bootstrap vs parametric CIs unspecified.** Now committed to bootstrap (parametric CIs are unsafe at n=5 with possibly-non-Gaussian seed-level returns).
7. **Exploration-bonus novelty unspecified.** Now L2 distance from the rolling mean of the last K=16 belief-representation vectors, coefficient shared across methods for axis comparability.
8. **Posterior-quality probe unspecified.** Now linear probe trained on 5000 held-out trajectories, frozen after M5. MLP probe reported as appendix robustness check.
9. **Per-regime PPO training budget ambiguity.** Now explicit: matched in *compute terms* (same iterations per instance), not in aggregate trajectory count.
10. **No seed-outlier discipline.** Now: never silent deletion, investigate every failure, no 2σ heuristic cutoff.
11. **Fuzzy pass criteria.** Every pass criterion across M0–M6 is now a JSON boolean or numeric threshold. No visual-judgment criteria remain.
12. **Figure style underspecified.** Now committed: exact dimensions, DPI, fonts, spines, per-method color palette, CI band alpha.
13. **Hyperparameter drift risk.** Now tune-once-freeze with `extends:` config composition enforcing single-source-of-truth for shared values.

### Acknowledged limitations (disclosed, not fixed)

These are genuine limits of the project that cannot be eliminated within its scope. They are stated explicitly in the thesis and constrain the claims made.

1. **Low seed count (n=5).** Defensible for feasibility on M4 CPU and for bootstrap viability, but underpowered for small effects — especially for interaction effects in the factorial (interactions have lower effective sample size than main effects). Mitigation: leave-one-out sensitivity on every primary finding; if removing any single seed flips the decision, the finding is not robust and reported as such.

2. **Single machine, single researcher.** No independent replication of implementations. A subtle bug in VariBAD's ELBO, RL²'s recurrence, or the analytical HMM posterior computation could consistently bias results and go undetected. Mitigation: M4 validation against published qualitative behavior on standard tasks; test suite in `tests/test_beliefs.py` verifies the HMM posterior against brute-force marginalization on short sequences.

3. **Single stylized env family.** The env is reduced-form and uncalibrated by design. Findings apply to this family of regime-switching POMDPs, not to real market making. Scope-of-claims section in the thesis explicitly restricts the conclusions.

4. **Hyperparameters tuned on toys, frozen for MM.** Prevents p-hacking but means reported MM numbers are modest underestimates for methods whose bandit-optimal hyperparameters transfer imperfectly to the MM env. The asymmetry is disclosed; no method retuning happens during M5/M6.

5. **Shared seeds across methods introduce correlated bias.** Paired-seed design enables stronger statistical tests but means a seed that happens to produce an unusually hard env trajectory biases all methods in the same direction. Mitigation: leave-one-out sensitivity, as in (1).

6. **RQ3 sweep is sparse.** 3 points per axis on 2 axes = 9 data points total. Portability claims across difficulty are based on these 9 points and are not dense-grid evidence. Widening the sweep is scope creep; an optional 3×3 heatmap is listed as scope-permitting.

7. **PEARL and HyperX excluded by argument, not by empirical comparison.** The exclusion rationale is methodologically defensible (PEARL's distinguishing features disappear under PPO adaptation; HyperX is covered by the orthogonal exploration axis), but a reviewer could reasonably ask whether the arguments hold empirically. Not testable within compute budget.

8. **Discrete 3-action, 3-regime design.** Other action-space sizes and regime counts are unexplored. The 3×3 choice is justified for VI tractability and Belief-PPO cleanliness but is not a derivation from first principles.

9. **Posterior-quality probe is a measurement tool, not a theoretical object.** The choice between linear and MLP probe affects the numbers. We default to linear (decoupling "is the info there in an easy-to-use form?" from "can a nonlinear decoder extract it?") and report MLP in appendix. Still, an asymmetry exists: Belief-PPO's posterior is *natively* in simplex form, while VariBAD and RL²'s posteriors are *mapped* into simplex form. This asymmetry is disclosed when reporting posterior-error numbers.

10. **Env iteration process (E0 → E_final) is not adversarial.** The env is tuned to satisfy R1–R4, not to differentiate specific methods. Mitigation is the explicit anti-pattern warning in "Environment design process": *never* tune the env until a favored method wins. But there's no external check on this.

### Already-defended choices (no action needed)

These appeared in the audit but are justified by existing spec sections and don't need further treatment:

- **Reduced-form env over full LOB.** Defended in "Uncalibrated by design" — analytical tractability of the HMM posterior is the thesis's core scientific asset. A full LOB would destroy it.
- **Exclusion of PEARL and HyperX from the ladder.** Defended in "Excluded methods and why" — PEARL's distinguishing features disappear under PPO adaptation and its set-based encoder mismatches HMM order-dependence; HyperX is subsumed by the orthogonal exploration axis.
- **Exclusion of MAML and transformer-based in-context methods.** Defended by scope (MAML has no belief representation, which is the object of study) and compute (transformer methods are infeasible on M4).
- **Discrete action space.** Defended in "Action space: discrete, not continuous" — VI tractability, Belief-PPO cleanliness, continuous sym/skew naturally reduces to a small discrete set.
- **100-iteration cap with many parallel envs.** Defended in "Training loop structure" — stability-per-point from parallel-env batching is more important than total-step count for ranking comparisons.
- **Single-researcher / non-replication.** Defended as mitigated (not eliminated) by M4 validation against published qualitative behavior.

### Summary of what the thesis can and cannot claim

The methodology supports claims of the form:
- "On this env family, with frozen hyperparameters, method X is supported / not supported as better than method Y at the α=0.05 family-wise level."
- "Across the persistence / distinguishability sweep, method X's gap-closure is monotonically non-increasing in difficulty."
- "Posterior approximation error and task performance are correlated with Spearman ρ = ... / are decoupled."
- "The factorial main effect of integration / exploration is / is not supported."

The methodology does **not** support claims of the form:
- "Method X is the best meta-RL method."
- "These findings apply to real market making."
- "Method X is always better than Y."
- "Method X works because of reason R." (Mechanistic claims require additional causal evidence — probes, ablations — some of which are in M7 scope but optional.)

## Contingency plans

Every milestone can fail. For each likely failure mode, this section specifies the first diagnostic to run and the fallback path. The discipline: when something fails, consult this table rather than improvising. Improvisation under frustration is where projects go off the rails.

### M0 fails (pipeline won't run end-to-end)

- **JAX/jaxlib version mismatch on M4** → use the aarch64 CPU wheel; pin explicitly in `pyproject.toml`.
- **Config not JSON-serializable** → add a custom encoder for dataclasses; never store non-serializable objects in config.
- **Script exits zero but writes no JSON** → your `write_summary` call is conditional on something it shouldn't be; audit the happy-path exit.

### M1 fails (PPO can't match AS analytical)

*Symptom: `return_ratio` < 0.95 or learning curve not plateauing.*

1. **First: verify the analytical target.** Hand-compute AS expected return on a trivial parameterization (e.g., symmetric noise). Does `oracles/analytical_as.py` match? If no, the target is wrong — fix that first.
2. **Then: increase iterations** with `--fast` to see whether the curve is still climbing. If yes → 100 iterations is too few, raise the cap.
3. **Then: tune PPO hyperparameters** with `--fast`. Most impactful: learning rate (try 1e-4 / 3e-4 / 1e-3), entropy coefficient (0.0 / 0.01 / 0.05), clip ratio (0.1 / 0.2 / 0.3). Do this in `--fast` mode so tuning is minutes not hours.
4. **Then: increase network size** — PPO might be capacity-limited on the env's state space.
5. **Last resort: simplify the env.** Reduce inventory range, reduce action count, fix time-of-day or other orthogonal noise. A simpler E0 is acceptable; the goal is trustworthy PPO, not env fidelity.

If after all of the above PPO still fails, there is a bug in the PPO implementation or rollout code. Step through a single rollout by hand and verify advantages/returns match what you'd compute on paper.

### M2 fails (R1–R4 can't all be satisfied)

*Symptom: one or more R flags is false after 3+ env iterations.*

- **R1 fails** (policies don't diverge across regimes) → regimes differ too subtly. Increase regime-conditional fill probability separation (e.g., bull gives 0.7 buy-side fill prob, bear gives 0.3 — make it 0.9 vs 0.1 if needed). If still failing, the action space may not be expressive enough: add a 4th or 5th discrete action.
- **R2 fails** (per-regime PPO < VI optimum) → PPO problem, not env problem. Same tweaks as M1. Per-regime PPO's failure here is usually a training-budget issue on the easier locked env.
- **R3 fails** (mixed PPO ≈ Oracle PPO) → compromise policy is somehow finding Oracle-level return, which means regime info is not actually needed. R1 is marginal. Strengthen regime distinction as with R1.
- **R4 fails** (posterior doesn't sharpen; Belief-PPO barely above regime-agnostic) → regime not inferable from fills alone. Either increase distinguishability (makes inference easier), enrich observations (add more fill resolution, add a noisy price signal), or slow the transition rate so there's more evidence per regime.

**Stop-and-reconsider rule.** If after 3 iterations no R-pattern is converging, do not keep adding structural complexity. Step back and consider whether the env abstraction itself needs rethinking — e.g., maybe drift-switched price dynamics (E1) is the wrong direction and fill-intensity switching (E2) is what you want.

### M3 fails (reference-level ordering is wrong)

*Symptom: `ordering_valid` is false. Belief-PPO > Oracle-PPO, or regime-agnostic > Belief-PPO, etc.*

- **Belief-PPO ≥ Oracle-PPO** → probably a training-budget artifact (Belief-PPO's posterior is richer than a one-hot, so it can sometimes converge faster). Run both for longer. If it persists, check whether the one-hot regime encoding in Oracle-PPO is being processed correctly (is it passed through an embedding? is there a bug where it's always zero?).
- **Regime-agnostic > Belief-PPO** → Belief-PPO is broken. Check the posterior input is what you think (log the values, compare to analytical forward algorithm output). Check the policy network is actually consuming the posterior (inspect gradients).
- **Per-regime < Oracle-PPO** → surprising but possible if per-regime PPO got less training data per regime than Oracle-PPO got across all regimes. **Training budget must be matched across reference levels in compute terms, not trajectory terms**: each of the N per-regime PPO instances gets the same number of iterations as Oracle-PPO (not 1/N of them). Check that `per_regime_ppo.total_training_iterations` equals `oracle_ppo.total_training_iterations`. If yes and per-regime still underperforms, try more iterations. If it persists even with more iterations, the finding is real (possibly shared-representation in Oracle-PPO actually helps via regularization) and reportable.

### M4 fails (a method doesn't learn validation tasks)

*Symptom: method's `learns` flag is false on bandit or gridworld.*

- **RL² flat on bandit** → recurrent state not being reset at episode boundaries, or hidden state not being included in the input. Log `(obs, prev_action, prev_reward)` tuple being fed to the GRU and verify it's updating.
- **VariBAD flat on bandit** → KL term swamping reconstruction loss. Start KL weight at 0.01 and scale up. Check that encoder output dimension is reasonable (not collapsed to 0 variance).
- **Hypernet variant fails to beat concat on toys** → hypernet parameterization is wrong (e.g., too-small output dim, nonlinearities eating the belief signal). Log the generated policy-weight norms; if they're constant across beliefs, the hypernet is ignoring its input.
- **Exploration bonus hurts rather than helps on gridworld** → bonus coefficient too large, swamping task reward; or the novelty signal source is pathological (e.g., RL² hidden state changes every step regardless of regime, so "novelty" is always high). Reduce coefficient by 10× and retry; if still broken, log the novelty signal distribution over an episode to see whether it's carrying regime information at all.

**Bisecting method failures.** When a meta-RL method fails, isolate the component: (a) does the underlying PPO work without the meta-RL machinery? (b) does the belief encoder produce non-trivial outputs? (c) is the policy consuming the belief? Each question has an independent test.

### M5 fails (meta-RL methods don't close much of the gap on MM)

*Symptom: any of the M5 pass criteria fail — `key_comparisons.belief_methods_beat_stacked_ppo.all_pass == false`, `ranking_stable_across_seeds == false`, `posterior_mse_range_spans_threshold == false`, or `at_least_one_main_effect_significant == false`.*

- **All methods' `gap_closed_vs_oracle` near 0** (all methods near the floor) → meta-RL methods aren't learning from history on this env despite working on M4 validation tasks. First: verify they're actually receiving history in the input (not just current obs). Then: check episode length — if episodes are too short, there's not enough history to be useful. Then: check whether stacked-obs PPO also fails. If `stacked_ppo.gap_closed_vs_oracle ≈ rl2.gap_closed_vs_oracle ≈ 0`, inference is too hard on this env; revisit `belief_ppo_gap_closure_fraction` from M2's R4 check.
- **All methods' `gap_closed_vs_oracle` near 1** (all methods near the ceiling) → inference is too easy. Belief-PPO gap is small. The env is under-stressed — consider tightening to make the problem harder. This is still a reportable result ("on easy regime inference, all methods match Oracle") but not the most interesting finding.
- **`ranking_stable_across_seeds == false`** or CIs overlap for key comparisons → more parallel envs per iteration first (reduces per-point variance), then more seeds (reduces across-seed variance), then more iterations (if learning curves not converged — check `converged` field).
- **Rankings flip across seeds** → seeds are too few, or method differences are below the signal threshold. Increase seeds to 8 or 10; if rankings still flip, accept that methods are genuinely indistinguishable.

### M6 fails (sweep curves are noisy or uninterpretable)

*Symptom: `stats_M6_sweep.interpretable_outcome == false` OR `stats_M6_posterior_vs_performance.scatter_interpretable == false`.*

- **`all_methods_monotonic == false`** (curves not monotonic in difficulty) → not enough seeds per point. Increase seeds before expanding grid.
- **CIs overlap across the whole sweep** (method curves barely separated at any difficulty level) → difficulty range is too narrow. Widen the span (easy = very easy, hard = very hard) until the endpoints clearly differ.
- **`signal_pattern == "noise"`** on the posterior-vs-performance scatter → either genuinely no relationship (report as finding), or noise. Distinguish by checking: within a single method, does `posterior_error` and `gap_closed` vary across difficulty levels (use `correlation_per_method` in the JSON)? If both vary but scatter is noisy, seeds are too few. If neither varies, the sweep range is too narrow.

### General: when results contradict expectations

- **Do not** immediately retune until results match expectations. That's p-hacking with extra steps.
- **Do** write the unexpected finding into `FINDINGS.md` with a timestamp, the commit hash, and the `stats_*.json` path before investigating further. This creates an audit trail — you can't later convince yourself the finding was something different.
- **Do** check implementation first (bisect with probes, run ablations that isolate suspected components) before concluding "method X doesn't work on problem Y."
- **Do** pre-commit in writing to what you'd do if results went either way, *before* running the experiment. The pre-registered hypotheses in the Statistical methodology section are this for M5/M6; smaller experiments should follow the same pattern in their config file's header comment.
- **Do** report the finding in the thesis even if it's negative or unexpected. Negative findings are the most under-reported and most informative results in meta-RL.

---

## Progress milestones

### Milestone dependency graph

```
M0 (pipeline skeleton)
  │
  ▼
M1 (PPO + AS baseline) ──────────┐
  │                               │
  ▼                               │  (frozen PPO hyperparameters
M2 (regime env + R1–R4) ─────┐   │   propagate to all downstream)
  │                           │   │
  ▼                           │   │
M3 (reference levels) ────────┴───┤──▶  RQ1 answered
  │                               │
  ▼                               │
M4 (validation suite) ────────────┤  (frozen method-specific hyperparameters
  │                               │   propagate to M5, M6)
  ▼                               │
M5 (ladder + factorial) ──────────┤──▶  RQ2 answered
  │                               │
  ▼                               │
M6 (difficulty sweep) ────────────┴──▶  RQ3 answered
  │
  ▼
M7 (optional supplementary ablations)
```

Each milestone blocks all downstream milestones; a failing milestone halts the chain. Re-running any milestone invalidates all downstream results and requires re-running those too.

### RQ-to-milestone-to-figure mapping

A single table so everyone (you, supervisors, reviewers, Claude Code) knows what answers what.

| RQ  | Answered by | Key JSON(s) | Thesis figures |
|---|---|---|---|
| RQ1 (gap decomposition) | M3 | `stats_M3_reference_levels.json` | `fig_rq1_ceilings_bar.png`, `fig_rq1_gap_fractions.png` |
| RQ2 (method rankings + factorial) | M5 | `stats_M5_ladder.json`, `stats_M5_factorial.json`, `stats_M5_mu_only_ablation.json` | `fig_rq2_ladder_returns.png`, `fig_rq2_gap_closed.png`, `fig_rq2_factorial.png`, `fig_rq2_posterior_error.png`, `fig_rq2_mu_only_ablation.png` |
| RQ3 (difficulty sweep + decoupling) | M6 | `stats_M6_sweep.json`, `stats_M6_posterior_vs_performance.json` | `fig_rq3_persistence_sweep.png`, `fig_rq3_distinguishability_sweep.png`, `fig_rq3_posterior_vs_performance.png`, (optional `fig_rq3_difficulty_heatmap.png`) |

Supporting milestones (M0, M1, M2, M4) produce their own per-milestone figures that establish infrastructure correctness but don't directly appear in the RQ answers.

### Total project shape

Rough file count by the end of each milestone, for orientation:

- **End of M0**: ~15 files (scaffolding, dummy agent, dummy env, config loader, script-output utility, smoke-test plotting).
- **End of M1**: ~25 files (PPO, analytical AS, basic plotting, env invariant tests).
- **End of M2**: ~40 files (regime env, HMM posterior, VI, verification script, requirement-plotting).
- **End of M3**: ~50 files (reference-level agents, gap-decomposition plotting).
- **End of M4**: ~70 files (validation envs, RL², VariBAD, hypernet module, exploration-bonus module, validation configs).
- **End of M5**: ~85 files (full ladder configs, factorial configs, mu-only config, RQ2 figures).
- **End of M6**: ~95 files (sweep configs, sweep orchestration, RQ3 figures).

These estimates are approximate but give a sense of cumulative effort.

### When the thesis is done

The spec stops at M7 but the thesis is what's ultimately produced. The spec treats thesis-writing as out of scope — but "done" needs a definition:

1. **All primary hypotheses have a supported / not-supported decision** in `stats_M5_ladder.primary_hypotheses` (pre-registered before running) with `leave_one_out_sensitivity` robustness checked.
2. **RQ3 has `interpretable_outcome = true`** in `stats_M6_sweep.json`.
3. **Every thesis figure in `figures/thesis/`** is traceable to a committed config and a committed JSON via its filename.
4. **Limitations section in the thesis draft** explicitly covers everything in "Acknowledged limitations" above.
5. **No exploratory finding is presented as primary** in the thesis text; exploratory findings are labeled as such.

Once these five are true, the thesis is done in the sense that further experimentation would be scope creep, not additional rigor.

### Principles that apply to every milestone

1. **Every tweakable thing has a verification artifact.** Something that needs to be tuned (VI convergence, HMM posterior correctness, PPO learning, env parameters) has a JSON output with the relevant statistics and a plot generated from it.
2. **JSON is for Claude Code, PNG is for the human.** See "Script output discipline." All pass criteria are JSON fields.
3. **Reproducible by anyone.** A reader following this file should be able to run each milestone's scripts, see the outputs, and know whether that step is working or needs tweaking.
4. **Gated.** A milestone does not pass until its artifacts show the required properties. A failing milestone blocks downstream work.
5. **Re-running is expected.** Code evolves; milestones are re-run. Figures always reflect current state. Git-tag the last passing state of each milestone so you can always recover.

### Milestone template

Each milestone specifies:
- **Goal** — what is being established.
- **What to tweak** — the parameters / code / configs that get adjusted at this step.
- **Verification artifact (JSON)** — the stats that answer "did the tweak work?"
- **Verification artifact (plot)** — the figure that visualizes the JSON.
- **Pass criteria** — what those artifacts must show, quantitatively where possible.
- **Blocks** — which downstream milestones depend on this passing.

**Tweaking loop uses run modes.** Any time something is adjusted, the sequence is:
1. `--super-fast` to confirm the code still runs end-to-end and produces the expected JSON schema.
2. `--fast` to see whether the change has the desired directional effect on learning / metrics.
3. Full run only after `--fast` looks right.

This prevents burning hours on a full run only to find a typo.

All artifacts saved under `results/milestones/M{n}/` (JSON) and `figures/milestones/M{n}/` (plots). Regenerated via `scripts/make_milestone.sh M{n}`.

---

### M0 — Infrastructure skeleton

**Goal.** Stand up the minimum end-to-end pipeline with dummy components so every piece of infrastructure is validated before any research logic enters. When M0 passes, the config system, run-mode flags, JSON output discipline, plotting, and test harness are all working — from that point onward, "did my code break" is separable from "did my research idea not work."

**What to tweak.** Nothing research-relevant. This milestone is about plumbing.

**Build order.**
1. `pyproject.toml` with pinned versions of jax/jaxlib/flax/optax/chex compatible with M4 CPU. Create a uv lockfile.
2. `utils/script_output.py` — the `write_summary(script_name, status, key_stats, outputs_written, ...)` helper that every script calls at the end. Produces the shared-schema JSON.
3. `training/config.py` — dataclass configs and `apply_run_mode(cfg, mode)`.
4. `envs/base.py` — abstract `Env` interface.
5. `envs/validation/dummy.py` — dummy env that returns constant reward and terminates after `rollout_length` steps. Purpose: exercise the pipeline without any dynamics.
6. `agents/base.py` — abstract `Agent` interface.
7. `agents/dummy.py` — random-policy agent. No learning, just returns random actions.
8. `training/rollout.py` — `lax.scan`-based rollout collecting trajectories into a pytree.
9. `training/train.py` — training entrypoint: loads config, runs outer loop (just rollouts for the dummy agent), writes JSON, prints `OK` line.
10. `plotting/style.py` — colors, font sizes, figure dimensions (see Plotting conventions below).
11. `plotting/load_results.py` — JSON parser.
12. `plotting/learning_curves.py` — minimal learning-curve plotter.
13. `tests/test_run_modes.py` — asserts `--super-fast` completes in <30s, `--fast` in <5min.
14. `scripts/make_milestone.sh` — orchestration script.

**Verification artifacts.**

`stats_M0_pipeline.json`:
```json
{
  "pipeline_version": "0.1",
  "python_version": "...",
  "jax_version": "...",
  "run_modes_tested": ["super_fast", "fast", "full"],
  "super_fast_duration_seconds": ...,
  "fast_duration_seconds": ...,
  "full_duration_seconds": ...,
  "all_scripts_exit_zero": true/false,
  "all_scripts_wrote_expected_json": true/false
}
```

- `fig_M0_dummy_learning_curve.png`
  - *What it shows.* A flat line (random policy return is constant in expectation, modulo CI band from seed noise).
  - *What it tells you.* The pipeline produced a plot from JSON. It doesn't matter that the content is uninteresting.
  - *Why it matters.* If this plot doesn't render cleanly, something is wrong upstream of any research code.

**Pass criteria.** All four of these are JSON fields in `stats_M0_pipeline.json` — no visual inspection required:
- `all_scripts_exit_zero: true`
- `all_scripts_wrote_expected_json: true`
- `super_fast_duration_seconds < 30`
- `fast_duration_seconds < 300`
- `schema_validates: true` (every output JSON passes the shared-schema validator)
- `make_milestone_script_succeeded: true` (`scripts/make_milestone.sh M0` exits 0)

The PNG (`fig_M0_dummy_learning_curve.png`) exists so you can eyeball that plotting works, but it is not part of the automated pass criteria.

**Commands to run.**
```
uv sync
uv run python -m training.train --config experiments/configs/m0_dummy.yaml --super-fast
uv run python -m training.train --config experiments/configs/m0_dummy.yaml --fast
uv run python -m training.train --config experiments/configs/m0_dummy.yaml
uv run python -m plotting.make_milestone M0
```

**What to do if this fails.** This is infrastructure. Fix immediately; do not proceed. Common failures: jaxlib version mismatch on M4 (use the aarch64 wheel), config dataclass not serializable to JSON (add custom encoder), `lax.scan` shape inconsistencies (make env state shapes fully static).

**Blocks.** Everything.

**Estimated duration.** 1–2 days.

---

### M1 — Environment plumbing and AS baseline

**Goal.** Confirm the simulator runs, JAX `vmap`/`jit`/`lax.scan` compile cleanly, and PPO on vanilla Avellaneda-Stoikov (single regime, E0) converges to near-analytical-optimal behavior.

**What to tweak.**
- PPO hyperparameters (learning rate, clip ratio, entropy coefficient, GAE lambda, value loss coefficient).
- Iteration count, parallel env count, rollout length (in the experiment config).
- Network size (hidden dim, depth).
- AS env parameters (inventory bounds, fill rates, reward weights).

**Verification artifacts.**
- `stats_M1_ppo.json`:
  ```json
  {
    "method": "ppo",
    "env": "as_e0",
    "seeds": [0, 1, 2, 3, 4],
    "final_return_per_seed": [...],
    "final_return_mean": ...,
    "final_return_ci": [..., ...],
    "as_analytical_return": ...,
    "return_ratio": ...,
    "iterations": 100,
    "parallel_envs": ...,
    "wall_time_seconds": ...,

    // Convergence diagnostics (JSON-native shape checks):
    "slope_last_20_iterations": ...,        // |slope| should be near zero if plateaued
    "plateau_reached_at_iteration": ...,    // first iter where return > 0.99 * final_return_mean
    "monotonic_increase_fraction": ...,     // fraction of adjacent-iteration pairs where return increased
    "converged": true/false,                // slope_last_20 < threshold AND plateau_reached_at < 90

    // Policy-shape diagnostics vs AS analytical (JSON-native):
    "policy_skew_correlation": ...,         // Pearson correlation between PPO skew-by-inventory and AS optimal skew-by-inventory
    "policy_skew_direction_agreement": ..., // fraction of inventory levels where PPO's skew sign matches AS
    "policy_shape_matches": true/false      // correlation > 0.9 AND direction_agreement > 0.95
  }
  ```
- `fig_M1_ppo_learning_curve.png`
  - *What it shows.* Single line plot, x = iteration (0–100), y = mean PPO return across 5 seeds with shaded CI band. Horizontal dashed line at the AS analytical expected return.
  - *What it tells you (human).* Whether PPO converges and how close it gets to the analytical ceiling. The JSON fields `return_ratio`, `converged`, and `plateau_reached_at_iteration` carry the same information for Claude Code.
  - *Why it matters.* The entire ladder rests on PPO being trustworthy. This is the human's visual confirmation; automated pass/fail operates on the JSON.

- `fig_M1_ppo_policy_vs_as.png`
  - *What it shows.* X-axis = inventory level, y-axis = action probability or skew magnitude. Two curves overlaid: the final PPO policy and the AS analytical optimal policy.
  - *What it tells you (human).* Whether PPO learned the *right* policy, not just a high-return one. The JSON fields `policy_skew_correlation`, `policy_skew_direction_agreement`, and `policy_shape_matches` carry this for Claude Code.
  - *Why it matters.* Return alone can hide a wrong-shape policy that happens to score well. The JSON bool `policy_shape_matches` is the automated version of this check.

**Pass criteria.** All three are JSON fields:
- `return_ratio >= 0.95`
- `converged == true` (slope near zero in last 20 iterations, plateau reached before iteration 90)
- `policy_shape_matches == true` (skew correlation > 0.9 and direction agreement > 0.95 vs AS analytical)

**Build order.**
1. `envs/mm_reduced.py` with an E0 (single-regime AS) configuration path. Test invariants (inventory bounded, reward finite).
2. `oracles/analytical_as.py` — closed-form AS optimal return calculator. Test on known parameter settings.
3. `agents/ppo.py` — the first real agent. Share PPO update code into `training/ppo_update.py`.
4. Experiment config `experiments/configs/m1_ppo_as.yaml`.
5. Run `--super-fast` to confirm end-to-end; fix any shape bugs.
6. Run `--fast` to confirm learning is happening (curve sloping up).
7. Tune PPO hyperparameters until `--fast` converges on the short run. These are the PPO hyperparameters *for the whole project*.
8. Full run with 5 seeds.

**Commands to run.**
```
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml --super-fast
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml --fast
# tune hyperparameters, repeat --fast until curve looks good
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml  # full run
uv run python -m plotting.make_milestone M1
```

**Tweaking loop.** If return_ratio < 0.95, first increase iterations (maybe 100 is too few for this env size). If already plateaued, tune learning rate or network size in `--fast` mode before committing to a full run. Iterate until pass.

**What to do if this fails.** See Contingency plans below under "M1 fails."

**Blocks.** Everything downstream.

**Estimated duration.** 1–2 days.

---

### M2 — Regime-switching env and requirements R1–R4

**Goal.** Extend E0 to a regime-switching env (E1, E2, ...) and verify the four problem requirements hold with clear margin. Iterate env design until all four pass.

**What to tweak.**
- Env structural additions: regime-switched volatility (E1), regime-switched directional fill intensities (E2), etc.
- Regime transition matrix (per-step switching probability).
- Regime-conditional fill probabilities (or intensities, or price parameters, depending on env form).
- Reward weights (inventory penalty, spread capture).

**Verification artifacts.**

`stats_M2_requirements.json`:
```json
{
  "env_version": "e2",
  "parameters": { ... full env config ... },
  "R1_policy_disagreement": {
    "fraction_disagreeing_states": ...,
    "mean_wrong_regime_value_loss": ...,
    "wrong_regime_value_loss_as_fraction_of_optimal_return": ...,  // normalized version
    "pass": true/false   // fraction >= 0.15 AND loss fraction >= 0.10
  },
  "R2_per_regime_ppo_vs_vi": {
    "regime_0_ratio": ...,
    "regime_1_ratio": ...,
    "regime_2_ratio": ...,
    "min_ratio": ...,
    "pass": true/false   // min_ratio >= 0.85 (diagnostic threshold; stricter 0.95 enforced in M1/M3 full-budget runs)
  },
  "R3_mixed_gap": {
    "regime_agnostic_return_mean": ...,
    "regime_agnostic_return_ci": [..., ...],
    "oracle_ppo_return_mean": ...,
    "oracle_ppo_return_ci": [..., ...],
    "gap_absolute": ...,
    "gap_ci_width": ...,
    "gap_to_ci_ratio": ...,          // gap_absolute / gap_ci_width
    "pass": true/false               // gap_to_ci_ratio >= 3.0
  },
  "R4_inferability": {
    "mean_posterior_entropy_at_t1": ...,
    "mean_posterior_entropy_at_mid_episode": ...,
    "entropy_decay_fraction": ...,   // 1 - (mid / t1)
    "belief_ppo_return_mean": ...,
    "belief_ppo_gap_closure_fraction": ...,  // (belief - agnostic) / (oracle - agnostic)
    "pass": true/false               // entropy_decay >= 0.30 AND gap_closure >= 0.30
  },
  "all_pass": true/false
}
```

- `fig_M2_R1_policy_heatmap.png`
  - *What it shows.* Heatmap, x-axis = inventory state, y-axis = regime (3 rows: noise / bull / bear), cell color = VI-derived optimal action (one of 3 discrete actions).
  - *What it tells you.* Whether regimes have different optimal policies. Different color patterns across rows = regimes disagree. Same patterns across rows = regimes don't matter and R1 fails.
  - *Why it matters.* R1 (policy divergence across regimes) is a precondition for the whole thesis. No divergence = no reason to care about regime inference. This is the single clearest visualization of whether the env is well-posed.

- `fig_M2_R1_value_loss_distribution.png`
  - *What it shows.* Histogram of value loss (value under correct regime's policy minus value under another regime's policy), aggregated across states.
  - *What it tells you.* How *costly* playing the wrong regime's policy is. A histogram clustered near zero = regimes differ in policy but outcomes are similar. A long-tailed distribution = real value at stake from getting the regime wrong.
  - *Why it matters.* R1 isn't just "policies differ"; it's "policies differ *in a way that costs performance*". This plot shows the latter. If the heatmap shows divergence but the loss distribution is near zero, R1 is technically satisfied but meaninglessly — you'd need to tighten the env further.

- `fig_M2_R2_per_regime_ppo.png`
  - *What it shows.* Three subplots, one per regime. Each subplot: PPO learning curve (x = iteration, y = mean return with CI band) trained on that locked regime, with a horizontal dashed line at the VI-derived optimal return.
  - *What it tells you.* Whether PPO can learn each regime's optimum when it doesn't have to share a network or handle regime switching. All three subplots should show curves reaching the dashed line.
  - *Why it matters.* R2 isolates PPO's raw optimization capability from the partial-observability difficulty. If PPO fails here, any failure on the mixed-regime setting is attributable to optimization, not to regime-switching — and the compromise-policy story collapses.

- `fig_M2_R3_mixed_gap.png`
  - *What it shows.* Bar chart with three bars: regime-agnostic PPO, Oracle-PPO, per-regime PPO. Error bars are seed-level CIs.
  - *What it tells you.* The gap that meta-RL methods will be asked to close. If regime-agnostic PPO is meaningfully below Oracle-PPO (CI bands don't overlap) and the gap is substantial, there's room for meta-RL to shine.
  - *Why it matters.* R3 is what makes the thesis question non-trivial. No gap = nothing to close = no story. This plot is the headline result of env design.

- `fig_M2_R4_posterior_entropy.png`
  - *What it shows.* X-axis = timestep within episode, y-axis = entropy of the analytical HMM posterior, averaged over many sample trajectories. Vertical markers indicate regime transition times.
  - *What it tells you.* Whether the posterior sharpens as evidence accumulates (entropy drops after transitions) or stays stuck at the stationary prior (entropy flat). A sawtooth pattern — drop, rise at transition, drop again — is the desired shape.
  - *Why it matters.* R4 (inferability) is a mathematical property of the HMM posterior: if this plot is flat, regimes genuinely cannot be inferred from fills alone and no belief-based method can succeed. Diagnoses env design failures before you waste compute on training meta-RL methods.

- `fig_M2_R4_belief_ppo_gap.png`
  - *What it shows.* Bar chart with three bars: regime-agnostic PPO, Belief-PPO, Oracle-PPO. Error bars are seed-level CIs.
  - *What it tells you.* How much of the Oracle gap Belief-PPO closes. Belief-PPO meaningfully above regime-agnostic = R4 holds in practice, not just in theory.
  - *Why it matters.* The posterior-entropy plot shows belief *can* sharpen in principle; this plot shows a policy can *use* that belief to improve performance. Both are needed. If Belief-PPO fails to close a meaningful fraction of the Oracle gap despite a sharpening posterior, the bottleneck is integration — relevant to RQ3 but also possibly a sign the env is poorly tuned.

**Pass criteria.** All are JSON fields under `stats_M2_requirements.json`:
- **R1**: `fraction_disagreeing_states >= 0.15` AND `wrong_regime_value_loss_as_fraction_of_optimal_return >= 0.10`.
- **R2**: `min_ratio >= 0.85` (diagnostic threshold; `--fast` runs are not expected to reach the stricter 0.95, which is re-verified with full budgets in M1 and M3).
- **R3**: `gap_to_ci_ratio >= 3.0` (gap exceeds CI width by at least 3×, so it's clearly non-marginal).
- **R4**: `entropy_decay_fraction >= 0.30` AND `belief_ppo_gap_closure_fraction >= 0.30`.
- Top-level: `all_pass == true` before proceeding.

**Tweaking loop.** If any requirement fails, identify which and adjust:
- R1 fails → regimes aren't distinct enough. Increase regime-conditional fill probability separation.
- R2 fails → PPO optimization issue. Same tweaks as M1.
- R3 fails → either R1 is marginal or compromise policy is too good. Strengthen regime distinction.
- R4 fails → regimes aren't inferable from fills alone. Add observational signal (e.g., more fill resolution) or increase distinguishability.

Each env revision gets a new `env_version` tag. Full revision trajectory (E0 → E1 → ... → final) documented in an appendix of the thesis.

**Build order.**
1. Add regime-switching to `envs/mm_reduced.py` — initial attempt (E1): regime-switched fill intensities. The env now accepts a `regime_config` dict.
2. `oracles/value_iteration.py` — VI on the full-info MDP. Test on AS analytical case (should match).
3. `beliefs/hmm_posterior.py` — forward algorithm for HMM belief. Test against brute-force posterior on short sequences.
4. `agents/ppo_oracle.py` and `agents/ppo_belief.py` — PPO variants that consume regime / analytical posterior respectively.
5. `oracles/verify_requirements.py` — the R1–R4 verification script. Takes an env config, outputs `stats_M2_requirements.json` and all six M2 plots.
6. Run verification on E1. Probably fails. Read the JSON to see which R failed.
7. Iterate env → E2 → E3 ... → final env, running verification each time.

**Commands to run.**
```
# iterate env revisions:
uv run python -m oracles.verify_requirements --env-config experiments/configs/envs/e1.yaml
# see which R failed, edit env config, try again
uv run python -m oracles.verify_requirements --env-config experiments/configs/envs/e2.yaml
# ... etc until all_pass is true
uv run python -m plotting.make_milestone M2
```

**Stop-and-reconsider rule.** If after 3 env iterations you still cannot get all four R's to pass, do not keep piling on structural complexity. Stop, review what's not working, and consider whether the env abstraction itself needs rethinking. Piling features tends to produce envs where requirements barely pass but results are uninterpretable.

**What to do if this fails.** See Contingency plans below under "M2 fails."

**Blocks.** M3, M4, M5, M6.

**Estimated duration.** 3–7 days. Unpredictable — this is the hardest milestone to budget because env iteration is open-ended.

---

### M3 — Reference-level decomposition (answers RQ1)

**Goal.** Produce the RQ1 answer: the numerical decomposition of the optimality gap.

**What to tweak.** Normally nothing — this milestone runs the four reference-level methods on the frozen env from M2 with the already-tuned PPO config from M1. Tweaks only if convergence is unstable.

**Verification artifacts.**

`stats_M3_reference_levels.json`:
```json
{
  "env_version": "e2",
  "reference_levels": {
    "per_regime_ppo": {
      "role": "ceiling",
      "mean": ..., "ci": [..., ...], "seed_returns": [...],
      "converged": true/false,              // slope_last_20 < threshold AND plateau before iter 90
      "plateau_reached_at_iteration": ...,
      "slope_last_20_iterations": ...
    },
    "oracle_ppo": {"role": "ceiling", ...},   // same fields
    "belief_ppo": {"role": "ceiling", ...},
    "regime_agnostic_ppo": {"role": "floor", ...}
  },
  "gap_components": {
    "shared_network_cost": {"absolute": ..., "fraction_of_total": ...},
    "inference_cost": {"absolute": ..., "fraction_of_total": ...},
    "compromise_policy_cost": {"absolute": ..., "fraction_of_total": ...},
    "total_gap": ...
  },
  "ordering_valid": true/false,             // regime_agnostic <= belief <= oracle <= per_regime, CIs considered
  "ordering_details": "regime_agnostic < belief <= oracle <= per_regime",
  "all_converged": true/false,              // AND over the four converged booleans
  "shared_network_cost_is_measurable": true/false  // shared_network_cost.absolute > CI width
}
```

- `fig_rq1_ceilings_bar.png` — **thesis figure.** See full description under RQ1.
- `fig_rq1_learning_curves.png` — **thesis figure.** See full description under RQ1.
- `fig_rq1_gap_fractions.png` — see full description under RQ1.

**Pass criteria.** All are JSON fields:
- `ordering_valid == true` (regime-agnostic ≤ belief-PPO ≤ oracle-PPO ≤ per-regime PPO, CIs considered).
- `all_converged == true` (every reference level has plateaued).
- `shared_network_cost_is_measurable` is informational — if false, the decomposition has only two components instead of three. This is reportable, not a fail.

**Story the plots tell (fills in RQ1 answer).** See RQ1 section above.

**Build order.**
1. `agents/ppo_per_regime.py` — wrapper that trains N separate PPO instances on locked regimes. Reuses `ppo.py` internals.
2. Experiment configs for the four reference levels on the frozen env from M2.
3. Run all four methods end-to-end. Should largely just work using components built in M1 and M2.

**Commands to run.**
```
uv run python -m training.train --config experiments/configs/m3_per_regime.yaml
uv run python -m training.train --config experiments/configs/m3_oracle.yaml
uv run python -m training.train --config experiments/configs/m3_belief.yaml
uv run python -m training.train --config experiments/configs/m3_regime_agnostic.yaml
uv run python -m plotting.make_milestone M3
```

**What to do if this fails.** See Contingency plans below under "M3 fails."

**Blocks.** M5, M6.

**Estimated duration.** 1 day of active work plus compute time for the four full runs.

---

### M4 — Implementation validation suite

**Goal.** Every method in the core ladder and every ablation axis reproduces published qualitative behavior on standard tasks before being trusted on MM.

**What to tweak.** Per-method hyperparameters, network sizes, encoder dimensions, KL weights (VariBAD), exploration bonus coefficients, hypernet size.

**Scope.** Validate the two belief-learning methods (RL², VariBAD) on standard meta-RL tasks, and separately validate that the ablation axes (hypernet integration, exploration bonus) produce expected directional effects on toy tasks where ground truth is available.

**Verification artifacts (per method per validation task).**

`stats_M4_{method}_{task}.json`:
```json
{
  "method": "varibad",
  "task": "bernoulli_bandit",
  "seeds": [0, 1, 2],
  "final_return_per_seed": [...],
  "final_return_mean": ...,
  "expected_ranking_position": 1,     // per published ordering
  "observed_ranking_position": 1,
  "ranking_matches_expected": true/false,
  "learns": true/false,               // slope over last 20 iterations > 0 AND final_mean > initial_mean + 2*initial_ci
  "qualitative_checks": {
    "posterior_sharpens_with_obs": true/false,  // VariBAD-only: entropy at t=last < 0.7 * entropy at t=1
    "posterior_entropy_decay_fraction": ...,    // 1 - (entropy_end / entropy_start); VariBAD-only
    "explores_then_exploits": true/false,       // RL²/VariBAD: first-half action entropy > second-half action entropy by threshold
    "action_entropy_first_half": ...,
    "action_entropy_second_half": ...
  },
  "pass": true/false                  // learns AND ranking_matches_expected AND all applicable qualitative_checks true
}
```

Additional ablation validation:

`stats_M4_ablation_axes.json`:
```json
{
  "hypernet_vs_concat": {
    "rl2_bandit": {
      "concat_return": ..., "hypernet_return": ...,
      "hypernet_delta": ...,             // hypernet - concat
      "hypernet_delta_exceeds_ci": true/false,
      "hypernet_helps": true/false       // delta > 0 AND exceeds CI
    },
    "varibad_bandit": {...},
    "hypernet_helps_on_at_least_one_task": true/false
  },
  "exploration_vs_none": {
    "varibad_gridworld": {
      "no_bonus_return": ..., "with_bonus_return": ...,
      "bonus_delta": ..., "bonus_delta_exceeds_ci": true/false,
      "bonus_helps": true/false
    },
    "varibad_bandit": {...},
    "rl2_gridworld": {...},
    "rl2_bandit": {...},
    "bonus_helps_more_on_gridworld_than_bandit": true/false   // sanity: exploration axis matters where exploration matters
  }
}
```

`stats_M4_summary.json` aggregates across methods and tasks with an overall pass/fail per (method × task) pair.

- `fig_M4_bandit_curves.png`
  - *What it shows.* Methods' learning curves overlaid on the 2-armed Bernoulli bandit meta-task. X = iteration, y = mean return across 3 seeds with CI bands. Lines for PPO, RL², VariBAD.
  - *What it tells you.* Whether each method learns the classic meta-RL bandit — a task where the published expected ordering is PPO (flat, stuck near random) < RL² ≈ VariBAD (both approaching the Bayes-optimal return, with explore-then-exploit behavior).
  - *Why it matters.* Canonical sanity check. Every belief-based meta-RL paper uses this task. If your RL² curve looks like PPO's (flat), the recurrent machinery is broken; if VariBAD doesn't outperform PPO, its encoder isn't being trained properly. Failing this plot means the implementation is bugged, not that the method is bad.

- `fig_M4_gridworld_curves.png`
  - *What it shows.* Methods' learning curves on random-goal gridworld (goal location resampled per episode). Lines for PPO, RL², VariBAD.
  - *What it tells you.* Whether methods learn to explore for the goal and exploit once found. Expected ordering: PPO (blind search baseline) below RL² and VariBAD.
  - *Why it matters.* Complements the bandit — tests spatial exploration rather than Bernoulli-arm exploration. A method passing bandit but failing gridworld may have memory issues specific to continuous state.

- `fig_M4_regime_bandit_curves.png`
  - *What it shows.* Methods on the Markov-switching bandit (arms switch according to a Markov chain). Lines for PPO, RL², VariBAD.
  - *What it tells you.* Whether methods handle the *regime-switching* structure that's central to the MM problem, on the cheapest possible task that exercises it.
  - *Why it matters.* The bandit and gridworld tasks are about task-level latents fixed per episode. The regime-switching bandit is about *within-episode latent switching* — the exact structure of MM. This plot is the last gate before a method enters MM experiments.

- `fig_M4_varibad_posterior_sharpening.png`
  - *What it shows.* X-axis = number of observations collected within an episode, y-axis = VariBAD's posterior entropy (or variance of latent z). Averaged over many episodes on the bandit.
  - *What it tells you.* Whether VariBAD's variational encoder is actually functioning as a belief — the entropy should decrease as more observations arrive. A flat line means the encoder is ignoring observations.
  - *Why it matters.* Without this plot, a VariBAD that "works" on task return might be succeeding through the policy alone while the encoder is dead weight. Confirming sharpening confirms the encoder is contributing.

- `fig_M4_hypernet_vs_concat.png`
  - *What it shows.* Bar chart comparing, for each belief-learning method (RL², VariBAD) on the bandit and gridworld, the effect of hypernet vs concat integration. Four subplots: (method × task).
  - *What it tells you.* Whether the hypernet integration axis produces its expected directional effect (improvement) on toy tasks before we rely on it in MM. If hypernet doesn't help at all on the toys, it probably doesn't help on MM either and the RQ2 factorial would report a null result.
  - *Why it matters.* Validates the hypernet implementation before running it in M5's factorial. Cheap sanity check that prevents discovering in M5 that hypernet is broken.

- `fig_M4_exploration_bonus.png`
  - *What it shows.* Bar chart comparing, for each belief-learning method (RL², VariBAD) on gridworld, the effect of exploration bonus vs no bonus. The effect should be larger on exploration-heavy tasks (gridworld) than on bandit.
  - *What it tells you.* Whether the exploration-bonus axis is wired up correctly — the bonus should help most where exploration matters. If the bonus *hurts* on both tasks, something is wrong with the novelty computation or the bonus coefficient is too large.
  - *Why it matters.* Same role as the hypernet validation. The M5 factorial assumes both ablation axes are functional; this confirms that assumption before burning MM compute.

- `fig_M4_method_ranking.png`
  - *What it shows.* A small table or slopegraph: for each validation task, the expected (published) method ranking on the left and the observed ranking on the right, connected by lines.
  - *What it tells you.* At a glance, whether every task has the expected qualitative ordering. Crossed lines indicate a method that's out of place.
  - *Why it matters.* Summary figure for the entire validation milestone. A supervisor can look at this one plot and know whether M4 passed.

**Pass criteria.** All are JSON fields. Per `(method, task)` pair in `stats_M4_{method}_{task}.json`:
- `pass == true` (which requires `learns`, `ranking_matches_expected`, and all applicable `qualitative_checks` to be true).

Across ablation axes in `stats_M4_ablation_axes.json`:
- `hypernet_vs_concat.hypernet_helps_on_at_least_one_task == true`.
- `exploration_vs_none.bonus_helps_more_on_gridworld_than_bandit == true`.

These are composite booleans computed from the underlying numeric fields. Claude Code only needs to check the top-level booleans.

**Gating.** A method failing M4 on any validation task does not enter M5 until fixed. Tweak the method's hyperparameters / implementation, rerun validation, check JSON, check plot.

**Build order.**
1. `envs/validation/bandit.py` and `envs/validation/gridworld.py`. Test invariants.
2. `envs/regime_bandit.py` (the regime-switching toy — shared between M2 posterior validation and M4 method validation).
3. `agents/rl2.py` — recurrent PPO with concat integration, no bonus. Validate on bandit first.
4. `agents/varibad.py` — variational encoder + PPO with concat integration, no bonus. Validate on bandit. Check posterior sharpens.
5. Add hypernet integration as a configurable component — the same hypernet class is used by any agent that takes a belief. Validate hypernet vs concat on RL² and VariBAD.
6. Add exploration bonus as a configurable auxiliary loss. The novelty signal source is method-specific (RL² hidden state vs VariBAD latent mean), but the bonus wiring is shared. Validate bonus vs no-bonus on RL² and VariBAD.
7. Method ranking plot.

For each method/ablation combination: start with `--super-fast` to confirm the pipeline, then `--fast` to see learning curve shape, then full run with 3 seeds for validation reporting.

**Commands to run.**
```
# for each method × task:
uv run python -m training.train --config experiments/configs/m4_{method}_{task}.yaml --super-fast
uv run python -m training.train --config experiments/configs/m4_{method}_{task}.yaml --fast
uv run python -m training.train --config experiments/configs/m4_{method}_{task}.yaml
uv run python -m plotting.make_milestone M4
```

**What to do if this fails.** See Contingency plans below under "M4 fails."

**Blocks.** M5 (for the specific method that failed).

**Estimated duration.** 3–5 days. Per-method debugging on validation tasks is where meta-RL implementations typically break.

---

### M5 — Method ladder on MM (answers RQ2)

**Goal.** Produce the RQ2 answer: how much of each gap component meta-RL methods close on the reference parameterization, plus the integration ablation.

**What to tweak.** Per-method hyperparameters carried forward from M4. Config for the integration ablation (which belief source to fix, which integration mechanisms to compare).

**Verification artifacts.**

`stats_M5_ladder.json`:
```json
{
  "env_version": "e2",
  "reference_levels": { ... from M3 ... },
  "methods": {
    "ppo": {
      "final_return": {"mean": ..., "ci": [..., ...], "seed_returns": [...]},
      "gap_closed_vs_oracle": ...,
      "gap_closed_vs_belief_ppo": ...
    },
    "stacked_ppo": {...},
    "rl2": {
      "final_return": {...},
      "gap_closed_vs_oracle": ...,
      "gap_closed_vs_belief_ppo": ...,
      "posterior_mse_final": ...,
      "regime_classification_accuracy": ...
    },
    "varibad": {...},
    "belief_ppo": {...},
    "oracle_ppo": {...},
    "per_regime_ppo": {...}
  },
  "ranking_by_return": ["oracle_ppo", "per_regime_ppo", "belief_ppo", "varibad", "rl2", "stacked_ppo", "ppo"],
  "ranking_by_posterior_quality": ["belief_ppo", "varibad", "rl2"],

  "primary_hypotheses": {
    // Each primary comparison: paired Wilcoxon, Holm-Bonferroni across the 4 ladder hypotheses + 2 factorial main effects
    "varibad_beats_agnostic": {
      "median_paired_delta": ..., "delta_ci": [..., ...],
      "wilcoxon_p": ..., "holm_corrected_p": ...,
      "cliffs_delta": ..., "supported": true/false
    },
    "rl2_beats_agnostic": {...},
    "varibad_beats_stacked_ppo": {...},
    "rl2_beats_stacked_ppo": {...}
  },

  "leave_one_out_sensitivity": {
    // For each primary hypothesis: does removing any single seed flip the "supported" flag?
    "varibad_beats_agnostic": {"robust_to_loo": true/false, "flipping_seed": null},
    ...
  },

  "ranking_stable_across_seeds": true/false,   // same top-to-bottom ordering in ≥ 4 of 5 seeds
  "posterior_quality_discrimination": {
    "posterior_mse_range": [min, max],
    "posterior_mse_range_spans_threshold": true/false   // max/min ≥ 2.0
  }
}
```

`stats_M5_factorial.json`:
```json
{
  "factorial_cells": {
    "rl2_concat_nobonus": {"return": {...}, "gap_closed_oracle": ..., "gap_closed_belief": ..., "posterior_error": ...},
    "rl2_concat_bonus": {...},
    "rl2_hypernet_nobonus": {...},
    "rl2_hypernet_bonus": {...},
    "varibad_concat_nobonus": {...},
    "varibad_concat_bonus": {...},
    "varibad_hypernet_nobonus": {...},
    "varibad_hypernet_bonus": {...}
  },
  "main_effects": {
    "integration_hypernet_vs_concat": {
      "marginal_mean_delta": ...,          // averaged across 4 paired cells
      "median_paired_delta": ...,
      "delta_ci": [..., ...],
      "wilcoxon_p": ..., "holm_corrected_p": ...,
      "cliffs_delta": ...,
      "supported": true/false              // primary hypothesis
    },
    "exploration_bonus_vs_none": {...},    // primary hypothesis, same structure
    "belief_source_varibad_vs_rl2": {...}  // EXPLORATORY — Holm-corrected but within its own family
  },
  "interactions": {
    "integration_x_belief": {"delta": ..., "wilcoxon_p": ..., "exploratory": true},
    "exploration_x_belief": {"delta": ..., "wilcoxon_p": ..., "exploratory": true},
    "integration_x_exploration": {"delta": ..., "wilcoxon_p": ..., "exploratory": true}
  },
  "at_least_one_main_effect_supported": true/false
}
```

`stats_M5_mu_only_ablation.json`:
```json
{
  "belief_source_fixed": "varibad",
  "integration_fixed": "hypernet",  // most interesting case
  "variants": {
    "mu_only": {"return": {...}, "gap_closed": ...},
    "full_posterior": {...}
  },
  "mu_only_vs_full_posterior_significant": true/false
}
```

- `fig_rq2_ladder_returns.png` — **thesis figure.** See full description under RQ2.
- `fig_rq2_gap_closed.png` — **thesis figure.** See full description under RQ2.
- `fig_rq2_learning_curves.png` — see full description under RQ2.
- `fig_rq2_posterior_error.png` — see full description under RQ2.
- `fig_rq2_factorial.png` — **thesis figure.** See full description under RQ2.
- `fig_rq2_mu_only_ablation.png` — see full description under RQ2.
- `fig_rq2_regime_accuracy.png` — see full description under RQ2.

**Pass criteria.** All are JSON fields:
- `stats_M5_ladder.primary_hypotheses.varibad_beats_agnostic.supported == true`.
- `stats_M5_ladder.primary_hypotheses.rl2_beats_agnostic.supported == true`.
- `stats_M5_ladder.primary_hypotheses.varibad_beats_stacked_ppo.supported == true`.
- `stats_M5_ladder.primary_hypotheses.rl2_beats_stacked_ppo.supported == true`.
- `stats_M5_factorial.main_effects.integration_hypernet_vs_concat.supported == true` OR `stats_M5_factorial.main_effects.exploration_bonus_vs_none.supported == true` (at least one main effect is supported; the factorial produces interpretable signal).
- `stats_M5_ladder.posterior_quality_discrimination.posterior_mse_range_spans_threshold == true`.
- All primary hypotheses that pass also pass `leave_one_out_sensitivity` (robust to removing any single seed).

"Supported" here means: Holm-Bonferroni-corrected paired-Wilcoxon p < 0.05 AND the delta CI excludes zero. Both conditions together, not either alone. See Statistical methodology section.

**Story the plots tell (fills in RQ2 answer).** See RQ2 section above.

**Build order.**
1. `evaluation/posterior_compare.py` — computes bidirectional mapping error between a method's inferred belief and the analytical posterior.
2. `evaluation/metrics.py` — gap-closed fractions, regime classification accuracy, main-effects and interaction stats for the factorial.
3. Experiment configs for each core ladder method on MM, frozen env from M2, reference levels from M3.
4. Factorial configs — 2 belief sources × 2 integration × 2 exploration = 8 config files, composed from the shared base.
5. Mu-only ablation config — VariBAD + hypernet + mu-only vs full posterior.
6. Run ladder. Run factorial. Run mu-only ablation.

**Commands to run.**
```
# core ladder (7 methods, single config each):
uv run python -m training.train --config experiments/configs/m5_ladder_ppo.yaml
uv run python -m training.train --config experiments/configs/m5_ladder_stacked_ppo.yaml
uv run python -m training.train --config experiments/configs/m5_ladder_rl2.yaml
uv run python -m training.train --config experiments/configs/m5_ladder_varibad.yaml
uv run python -m training.train --config experiments/configs/m5_ladder_belief_ppo.yaml
uv run python -m training.train --config experiments/configs/m5_ladder_oracle_ppo.yaml
uv run python -m training.train --config experiments/configs/m5_ladder_per_regime_ppo.yaml

# factorial: 2 belief sources x 2 integration x 2 exploration = 8 runs
# (rl2_concat_nobonus overlaps with m5_ladder_rl2 — don't double-run)
uv run python -m training.train --config experiments/configs/m5_factorial_rl2_concat_bonus.yaml
uv run python -m training.train --config experiments/configs/m5_factorial_rl2_hypernet_nobonus.yaml
uv run python -m training.train --config experiments/configs/m5_factorial_rl2_hypernet_bonus.yaml
uv run python -m training.train --config experiments/configs/m5_factorial_varibad_concat_bonus.yaml
uv run python -m training.train --config experiments/configs/m5_factorial_varibad_hypernet_nobonus.yaml
uv run python -m training.train --config experiments/configs/m5_factorial_varibad_hypernet_bonus.yaml
# varibad_concat_nobonus overlaps with m5_ladder_varibad

# mu-only ablation:
uv run python -m training.train --config experiments/configs/m5_mu_only_varibad_hypernet.yaml

uv run python -m plotting.make_milestone M5
```

**What to do if this fails.** See Contingency plans below under "M5 fails."

**Blocks.** M6.

**Estimated duration.** 2–3 days, mostly compute time for the full ladder + ablation.

---

### M6 — Difficulty sweep and posterior-performance relationship (answers RQ3)

**Goal.** Produce the RQ3 answer. Characterize method performance across difficulty axes, and test whether posterior quality predicts task performance.

**What to tweak.** Sweep-grid granularity (starting point: 3 points per axis), method subset for the sweep (starting point: RL², VariBAD, Belief-PPO, Oracle-PPO, regime-agnostic PPO — a reduced 5-method set since the factorial is run at M5, not at every sweep point), whether to run the 2D heatmap.

**Protocol.**
- Define 3 persistence levels and 3 distinguishability levels (per_env config).
- At each difficulty point: re-run all four reference levels (locally), re-run method ladder (subset).
- Seeds and iteration count as in M5.
- 2D heatmap (full 3×3 grid) is optional; 1D sweeps are required.

**Verification artifacts.**

`stats_M6_sweep.json`:
```json
{
  "persistence_sweep": {
    "levels": [easy_p, medium_p, hard_p],
    "per_level": [
      {
        "persistence": ...,
        "reference_levels": {...},
        "methods": {
          "rl2": {"gap_closed": ..., "posterior_error": ...},
          "varibad": {...},
          ...
        }
      },
      ...
    ],
    "per_method_monotonicity": {
      "rl2": {"gap_closed_slope_sign": "negative|positive|flat", "monotonic": true/false},
      "varibad": {...}
    },
    "all_methods_monotonic": true/false,          // gap_closed decreases as difficulty increases, for every method
    "ranking_stable": true/false                  // same method ordering at all 3 levels
  },
  "distinguishability_sweep": { ...same structure... },
  "heatmap_grid": { ...if run... },
  "rankings_per_level": [...],  // for quick diagnosis of ranking stability
  "interpretable_outcome": true/false            // (all_methods_monotonic on both axes) OR (ranking varies but per-method curves are monotonic)
}
```

`stats_M6_posterior_vs_performance.json`:
```json
{
  "scatter_points": [
    {"method": "rl2", "difficulty_config": "...", "seed": 0, "posterior_error": ..., "gap_closed": ...},
    ...
  ],
  "correlation_overall": ...,
  "correlation_per_method": {"rl2": ..., "varibad": ..., ...},
  "decoupling_detected": {
    "description": "method X has rank Y by posterior but rank Z by return",
    "methods_with_decoupling": [...]
  },
  "scatter_interpretable": true/false,
  // scatter_interpretable is true if either:
  //   (a) correlation_overall magnitude >= 0.5 (tight relationship), OR
  //   (b) decoupling_detected.methods_with_decoupling is non-empty AND CIs support it
  // Noise swamping both conditions → scatter_interpretable = false → M6 fails.
  "signal_pattern": "tight_correlation | decoupling | noise"
}
```

- `fig_rq3_persistence_sweep.png` — **thesis figure.** See full description under RQ3.
- `fig_rq3_distinguishability_sweep.png` — **thesis figure.** See full description under RQ3.
- `fig_rq3_persistence_posterior_error.png` — see full description under RQ3.
- `fig_rq3_distinguishability_posterior_error.png` — see full description under RQ3.
- `fig_rq3_posterior_vs_performance.png` — **thesis figure** and the key decoupling-analysis plot. See full description under RQ3.
- `fig_rq3_difficulty_heatmap.png` — optional, scope permitting. See RQ3.

**Pass criteria.** All are JSON fields:
- `stats_M6_sweep.persistence_sweep.all_methods_monotonic == true` (sanity check — harder problems don't help methods).
- `stats_M6_sweep.distinguishability_sweep.all_methods_monotonic == true`.
- `stats_M6_sweep.interpretable_outcome == true` (ranking is either stable across difficulty, or varies with monotonic per-method curves).
- `stats_M6_posterior_vs_performance.scatter_interpretable == true` (`signal_pattern` is `tight_correlation` or `decoupling`, not `noise`).

If `signal_pattern == noise`, increase seeds per point and/or widen difficulty range.

**Story the plots tell (fills in RQ3 answer).** See RQ3 section above.

**Build order.**
1. Sweep configs — three persistence levels and three distinguishability levels, each as a separate env config.
2. For each difficulty point: config for the full reference-level re-run (reuse M3 infrastructure), and configs for the reduced method ladder.
3. Orchestration script that runs the grid and aggregates results into `stats_M6_sweep.json`.
4. Posterior-vs-performance scatter analysis from the aggregated JSON.

**Pre-commitment.** Before running the full sweep, commit to: 3 points per axis, reduced method set (RL², VariBAD, plus all four reference levels locally). Both methods run at a single configuration — concat integration, no exploration bonus — which is the configuration that appears in the M5 core ladder. Factorial ablations are not re-run at every sweep point; that would quadruple compute without enough added insight. 2D heatmap only if one-axis sweeps are clean and compute allows. Resist scope expansion until the minimum-viable sweep has run.

**Commands to run.**
```
# one-axis sweep first — cheaper and often sufficient:
uv run python -m scripts.run_sweep --axis persistence --config experiments/configs/m6_persistence_sweep.yaml
uv run python -m scripts.run_sweep --axis distinguishability --config experiments/configs/m6_distinguishability_sweep.yaml
uv run python -m evaluation.posterior_performance_analysis --inputs results/m6/
uv run python -m plotting.make_milestone M6
# if compute allows:
uv run python -m scripts.run_sweep --axis both --config experiments/configs/m6_heatmap.yaml
```

**What to do if this fails.** See Contingency plans below under "M6 fails."

**Blocks.** Thesis writing (conclusions).

**Estimated duration.** 4–7 days, mostly compute. Active work is aggregation + plotting.

---

### M7 — Supplementary ablations (optional)

**Goal.** Targeted mechanistic ablations raised by M5/M6 findings. Only included if specific hypotheses emerge.

**Candidates.**
- Stop-gradient ablation on VariBAD decoder.
- Episode-length effect on gap-closure.
- Linear probe of hidden states for regime decoding (is RL²'s hidden state "really" a belief?).
- Any ablation the main results specifically suggest.

**Verification artifacts.** Per ablation: one JSON with stats, one figure.

**Pass criteria.** None strict. Included only if they sharpen the main findings.

**Blocks.** Nothing.

---

### Milestone tracking discipline

- Every milestone produces (a) JSON stats in `results/milestones/M{n}/`, (b) figures in `figures/milestones/M{n}/`, (c) a `PASS.md` or `FAIL.md` note committed after review.
- Figures intended for the thesis are also copied to `figures/thesis/` with the `fig_rqN_*.png` naming (as noted above) so LaTeX references are stable.
- Regenerate everything for a milestone with `scripts/make_milestone.sh M{n}`. This script runs training, dumps JSON, generates plots, in that order.
- A failing milestone blocks downstream work. Do not proceed past a "probably fine" — fix the root cause.
- Milestones are expected to be rerun as code evolves. Figures always reflect the current state of the repo; no stale artifacts.

## Conventions

### Language and environment

- Python 3.11+, managed with **uv** (the package manager; see https://docs.astral.sh/uv/). The project uses uv exclusively for dependency management and script execution. No `pip install`, no manual venv activation.
- JAX-native throughout; no PyTorch or NumPy-only agents.
- Type hints on public interfaces.

### Running scripts with uv

**Every Python command in this document uses `uv run`.** Never `python -m ...` directly.

Why: `uv run` guarantees the script executes in the project's locked environment — the exact Python version and dependency versions pinned in `pyproject.toml` and `uv.lock`. Plain `python -m` picks up whatever environment happens to be active, which on a development machine is often wrong and silently produces results that won't reproduce on a fresh clone.

The pattern is:

```
# correct — uses the project's locked environment:
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml

# wrong — depends on shell state:
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml
```

The commands listed in each milestone assume `uv run` as the prefix. They're shown as `uv run python -m ...` throughout.

**First-time setup:**
```
git clone <repo>
cd thesis
uv sync          # installs all deps from pyproject.toml + uv.lock
# verify:
uv run python -c "import jax; print(jax.devices())"
```

**Adding a dependency** (if ever needed):
```
uv add <package>     # updates pyproject.toml and uv.lock
git commit pyproject.toml uv.lock
```

Do not edit `pyproject.toml` by hand for dependencies; let uv manage it.

### Configuration files (YAML)

Every experiment is defined by a single YAML file under `experiments/configs/`. YAML is the format for two reasons: (a) human-readable and diffable in git (unlike pickle or JSON with embedded code), and (b) composable — base configs can be extended by experiment-specific overrides, preventing drift across runs.

**How YAML files get used.**

1. The user invokes a training or verification script with `--config path/to/config.yaml`.
2. `training/config.py` reads the YAML file, resolves any `extends:` directives (loading the base and deep-merging overrides on top), and materializes a typed `ExperimentConfig` dataclass.
3. If a CLI flag like `--super-fast` or `--fast` is passed, `apply_run_mode(cfg, mode)` mutates the dataclass in memory (overrides iterations, parallel envs, rollout length, seed count).
4. The effective config (after all resolution and run-mode overrides) is serialized to JSON and written to `results/{experiment_name}/config.json` at the start of the run, alongside the git commit hash. This is the reproducibility record — anyone can reproduce a run from `{config.json, commit_hash, seed}`.
5. The script then proceeds with the typed dataclass; no YAML is read after startup. No magic strings at runtime.

**Composition via `extends`.**

A typical experiment config is short — maybe 10 lines — because it extends a shared base:

```yaml
# experiments/configs/m5_ladder_varibad.yaml
extends: base/base_varibad.yaml
experiment_name: m5_ladder_varibad
env:
  extends: envs/e_final.yaml
integration: concat
exploration_bonus: off
iterations: 100
num_seeds: 5
```

The base file `base/base_varibad.yaml` holds the tuned PPO core hyperparameters, the VariBAD-specific parameters (KL weight, latent dim), and defaults for everything else. The experiment file only specifies what's different.

This is what makes the hyperparameter discipline enforceable — if "PPO learning rate" appears in one place (`base_ppo.yaml`), it can't drift across runs. The alternative (one full config per experiment, hand-copied) is where silent parameter drift comes from.

**Schema validation.**

Every YAML file is loaded into a Python dataclass with type hints (`ExperimentConfig`, `AgentConfig`, `EnvConfig`). If a field is missing or wrong-typed, the script fails at startup with a clear error — not deep into training. This also means autocomplete works when editing the config loader, which catches typos before they become silent bugs.

**What lives in YAML vs what lives in code.**

In YAML: anything a reasonable experiment run might need to vary. Hyperparameters, architecture sizes, network widths, learning rates, env parameters, seed count, iteration count, which ablation axes are on.

In code: things that define a *different* experiment rather than a variant of the same one. Loss function shape, architecture class (MLP vs GRU vs variational encoder), training-loop structure. These go in agent files, not configs.

The rule of thumb: if changing a value should produce a meaningfully different experiment that belongs in a different config file, it's a YAML field. If changing it changes the *science*, it probably needs a new agent file or a code-level decision.

### Version pinning

`pyproject.toml` pins *exact* versions of the core dependencies: `jax`, `jaxlib`, `flax`, `optax`, `chex`, `numpy`. Pins are chosen in M0 based on what works on M4 CPU; any subsequent upgrade is a deliberate decision reflected in a git commit, not a casual `uv sync`. The `uv.lock` file is committed to the repo. A fresh clone + `uv sync` reproduces the exact environment.

The JAX version is particularly important: on M4, the aarch64 CPU wheel has to be chosen, and JAX semantics around `lax.scan` and `vmap` have occasionally shifted between versions. Pin and don't drift.

### Test discipline

Tests exist for load-bearing correctness properties — not full coverage, but the handful of things that, if wrong, silently invalidate research results. Each test file produces a script-output JSON and is run as part of `scripts/make_milestone.sh M0` (and optionally before every long training run).

**Test categories and what each verifies:**

- **Env invariants** (`tests/test_envs.py`): inventory stays within declared bounds across random trajectories; reward is always finite; regime index is always in {0, 1, 2}; done flag set correctly at episode boundary; reset produces the declared initial distribution over states.

- **Belief correctness** (`tests/test_beliefs.py`): output of `beliefs/hmm_posterior.py` matches a brute-force marginalization posterior on short (length ≤ 5) test sequences with known regime ground truth. Posterior always sums to 1 and is non-negative. Sharpens monotonically with consistent evidence.

- **Oracle convergence** (`tests/test_oracles.py`): VI converges (Bellman residual decays below threshold); optimal policy is stable across repeated runs with the same seed; Q-values are finite.

- **Ground-truth leak regression** (`tests/test_leak.py`): for every agent that is *not* supposed to see the regime (everything except Oracle-PPO, Belief-PPO, per-regime PPO), verify that the regime does not appear in the agent's observation input after env.step returns. A simple assertion: construct a rollout, check that the agent's observed tensor has no component that equals the true regime.

- **Run-mode behavior** (`tests/test_run_modes.py`): `--super-fast` on the dummy env completes in <30 seconds; `--fast` in <5 minutes; both produce valid JSON matching the script-output schema.

- **Script output schema** (`tests/test_script_output.py`): every `write_summary` call produces a JSON that validates against the shared schema defined in `utils/script_output.py`.

Each test file prints `[test_{name}] OK | tests_run=N | tests_passed=N` on success. CI-style discipline: tests must pass before a milestone is declared passed. They run fast (seconds) and catch real bugs.

### Reproducibility

Each experiment run is fully reproducible from `{config.json, commit hash, seed}`. The commit hash is recorded in `config.json` at run start. If a run produces surprising results, the first question is "what commit was this from?" — which is immediately answerable from the result directory alone.

### Git discipline

- **Main branch always passes M1 at minimum** (ideally whichever milestone is the most recent pass). Never commit to main when the known-good state is broken.
- **Tag each passing milestone.** `git tag m1-passed`, `m2-passed`, etc. Tags are the recovery points — if a later change breaks M2, reverting to `m2-passed` is one command.
- **Branch before risky refactors.** Work on a branch; merge only after the target milestone re-passes. Any change that touches more than one agent or the training loop qualifies as "risky."
- **Commit hash in config.json is the diagnostic tool.** When results change and you don't know why, `git log {old_commit}..{new_commit}` is where the answer is.
- **Never force-push main.** Branches are fine.

**Handling dead ends during env iteration (M2).** Trying parameterizations that fail R1–R4 is expected; the spec's env-design-process is explicitly iterative. Workflow:
- Each env iteration lives on a branch named `env-iter-e{N}`.
- If the env iteration passes R1–R4, merge to main and tag.
- If it fails, commit the attempt + its JSON (the `FAIL.md` documents why it failed), push the branch, then start the next iteration on a new branch. Don't delete failed attempts — they're thesis-appendix material ("we tried X, Y, Z before converging on the final env").
- `envs/e_final.yaml` is a symlink to whichever env config passed. When it changes, every downstream milestone must be re-run.

**Handling milestone re-runs.** Re-running an earlier milestone invalidates all downstream results. Workflow:
- Revert to the earlier milestone's tag (`git checkout m{n}-passed`) or branch from current if the re-run is a forward change.
- Re-run M{n} with `scripts/make_milestone.sh M{n}`.
- If it passes, re-tag (force-move the tag or create `m{n}-passed-v2`). If it fails, document in `FAIL.md`.
- Re-run every downstream milestone. This is expensive, which is why the milestone discipline aims to get each milestone right before moving on — but is unavoidable when fundamental changes are needed.
- Record the re-run in `results/milestones/M{n}/REVISION_LOG.md`: what changed, why, what JSONs were invalidated, what was regenerated.

**Handling experimental dead ends within a milestone.** If you try a parameter combination that doesn't work (e.g., a hypernet hidden-dim that gives flat learning curves), commit the attempt with a descriptive message and keep going. The failed attempt's `stats_*.json` is useful in the thesis appendix as "we tried configuration X and it didn't work for reason Y." Don't silently delete failed attempts; that loses negative-finding information.

### Milestone discipline (recap)

Every milestone produces:
- JSON stats under `results/milestones/M{n}/`
- Figures under `figures/milestones/M{n}/`
- A `PASS.md` or `FAIL.md` note committed after review
- A git tag on pass: `git tag m{n}-passed`

Regenerate everything for a milestone with `scripts/make_milestone.sh M{n}`. A failing milestone blocks downstream work. Milestones are rerun as code evolves; figures always reflect current state.