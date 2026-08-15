# Contingency plans

One section per milestone. When a milestone fails, consult this file rather than improvising — every common failure has a known diagnostic sequence.

The general principle ("when results contradict expectations") is at the bottom.

---

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

