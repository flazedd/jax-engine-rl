# Environment

Full specification of the reduced-form MM environment, the four problem requirements R1–R4 that any valid parameterization must satisfy, and the env-design iteration process.

See also:
- `docs/milestones.md` — M2 is the milestone that verifies R1–R4.
- `docs/methodology.md` — R1–R4 thresholds are operationalized as JSON fields with committed pass criteria.

---

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

The MM environment must satisfy the following properties for the thesis to be well-posed. These are the foundation that makes meta-RL a meaningful thing to study here — if any of them fail, the research questions collapse. Concrete thresholds are committed in the M2 JSON schema; what follows is the conceptual statement.

**R1. Regime-conditional policy divergence.** The optimal policy on each locked regime must differ across regimes, and the value loss from playing the wrong regime's policy must be real.
*Threshold:* `fraction_disagreeing_states >= 0.15` AND `wrong_regime_value_loss_as_fraction_of_optimal_return >= 0.10`.
*Why required:* if optimal policies are the same across regimes, regime information is useless and no method can benefit from it. There is nothing to study.

**R2. PPO achieves optimality on locked regimes.** A PPO agent trained on a single locked regime must converge to the regime-conditional optimum (matching VI). This must hold for every regime.
*Threshold:* `min_ratio >= 0.85` in M2 diagnostic, `>= 0.95` in full-budget M1/M3 re-verification.
*Why required:* if PPO cannot learn a locked regime's optimum, any failure of PPO on the mixed setting is attributable to optimization, not to the regime-switching structure. The ceiling decomposition loses meaning.

**R3. PPO settles on a strictly suboptimal compromise on mixed regimes.** A regime-agnostic PPO trained on the full HMM-generated trajectories must converge to a return below Oracle-PPO and below per-regime PPO. The gap must exceed seed-level CI width by a clear margin.
*Threshold:* `gap_to_ci_ratio >= 3.0`.
*Why required:* this is the gap that meta-RL methods are asked to close. Without a measurable gap, there is no room for belief-conditioned methods to shine.

**R4. Regime inferability from history.** The HMM posterior must sharpen with evidence, and a policy conditioned on it must improve performance over regime-agnostic PPO.
*Threshold:* `entropy_decay_fraction >= 0.30` AND `belief_ppo_gap_closure_fraction >= 0.30`.
*Why required:* if the regime is not inferable from observations, no belief-based method can succeed regardless of its machinery. RQ2 becomes unanswerable.

These requirements are verified empirically on the chosen parameterization in M2. The `oracles/verify_requirements.py` script produces `stats_M2_requirements.json` containing the pass/fail decision for each R.

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

**Synthetic-testbed framing.** This env is a synthetic POMDP testbed for meta-RL, not a calibrated market simulator. Some parameter choices in `E_final` (E6e) are mathematically convenient rather than physically standard — most notably R0 (the wide-favoring regime) has `p_wide > p_tight`, meaning wide quotes fill more often than tight ones. In a real market the ordering is usually reversed (counterparties prefer tighter quotes). The choice is justified because what matters for RQ1/RQ2 is the *structural* property that the optimal action class (sym vs favor_X) differs across regimes while inventory direction does not leak regime information — physical realism of the per-regime fill distribution is orthogonal to the meta-RL well-posedness conditions in R1–R4. Calibration to real markets is excluded by design (consistent with the "real markets do X" anti-pattern above).

### Regime locking

Env accepts `lock_regime: int | None`. When set, regime transitions are disabled — used for per-regime PPO training and for targeted evaluation.

### Toy validation environment

`regime_bandit.py`: 2-armed bandit where the better arm switches according to a Markov chain. Cheap, stresses the same Markovian latent structure as MM, used as shared validation testbed that all methods should pass before committing to full MM experiments.


---

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

