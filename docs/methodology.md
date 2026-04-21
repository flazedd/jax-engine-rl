# Statistical methodology and rigor

Paired Wilcoxon + Holm correction + bootstrap CI; pre-registered primary hypotheses; posterior-quality measurement protocol; scope of claims the methodology supports; catalog of project limitations.

Read this when running or interpreting any comparison where the word "better" matters.

See also:
- `docs/research-questions.md` — the pre-registered primary hypotheses map to specific RQs.
- `docs/milestones.md` — M5 and M6 apply these tests.

---

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

**RQ1 primary hypothesis.** The three reference levels order as regime-agnostic < belief-PPO ≤ oracle-PPO, with a non-zero gap between regime-agnostic and oracle. Pre-committed: if belief-PPO ≈ oracle-PPO (CI overlap), inference cost is trivially small and the RQ2 motivation is weaker.

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


---

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

