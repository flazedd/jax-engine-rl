# M2 — PASS

**Date:** 2026-04-20
**Commit:** tagged `m2-passed`
**Env:** `E_final = e2_fill_switched` (3-regime: noise / bull / bear with regime-conditioned directional fill intensities).

## Pass criteria (from `docs/milestones/m2.md`)

Verified via `results/milestones/M2/stats_M2_requirements.json`. All four requirements clear their thresholds:

| Requirement | Metric | Value | Threshold |
|---|---|---|---|
| **R1** | `fraction_disagreeing_states` | 0.909 | ≥ 0.15 |
| **R1** | `mean_relative_value_loss_at_disagreeing_states` | 0.384 | ≥ 0.10 |
| **R2** | `min_ratio` (PPO / VI per locked regime) | 0.901 | ≥ 0.85 (diagnostic budget) |
| **R3** | `gap_to_ci_ratio` (paired bootstrap) | 4.65 | ≥ 3.0 |
| **R4** | `entropy_decay_fraction` | 0.496 | ≥ 0.30 |
| **R4** | `belief_ppo_gap_closure_fraction` | 0.663 | ≥ 0.30 |

`all_pass == true`.

### Supporting numbers

| Quantity | Value |
|---|---|
| VI per-regime episode return | [152.41, 78.38, 78.38] |
| VI mixed (stationary-regime) return | 125.65 |
| Regime-agnostic PPO mean return | 100.38 (CI [95.95, 104.73]) |
| Belief-PPO mean return | 114.11 (CI [111.71, 116.10]) |
| Oracle-PPO mean return | 121.08 (CI [119.02, 123.34]) |
| Paired gap (oracle − agnostic) per seed | [18.61, 20.43, 23.06] |
| Mean posterior entropy at t=1 | 0.857 nats |
| Mean posterior entropy at mid-episode | 0.432 nats (≈61% below the `log(3)=1.099` flat prior) |

## Figures

Under `figures/milestones/M2/`:

- `fig_M2_R1_policy_heatmap.png` — VI-optimal action per (regime, inventory); regimes select different actions at most inventory levels.
- `fig_M2_R1_value_loss_distribution.png` — histogram of relative policy-commitment loss `(V^π_true − V^π_other) / V^π_true` over (r_true, r_other, inventory) buckets; trimodal (~0 / ~0.30 / ~0.75), mean 0.384.
- `fig_M2_R2_per_regime_ppo.png` — three subplots, PPO learning curve vs VI dashed line on each locked regime. Bull/bear show clear learning; noise is flat because random-symmetric play is already near-optimal there.
- `fig_M2_R4_posterior_entropy.png` — mean posterior entropy across random-policy rollouts; drops rapidly to ~0.42 then plateaus. Terminal step trimmed (the belief-obs wrapper resets on `done`).
- `fig_M2_R4_belief_ppo_gap.png` — bar chart: regime-agnostic / Belief-PPO / Oracle-PPO with seed CIs.

## Reproduction

```
uv run python -m scripts.make_milestone M2
```

Runs the verifier in `full` mode for `E_final`, then regenerates figures. Exit 0 and `[make_milestone] OK | all_pass=True` indicate a pass.

## What M2 established

- `envs/mm_reduced.py` — regime extension: 3-regime HMM with per-regime fill probabilities, `lock_regime` mode for R2 evaluation, diagonal-dominant transition matrix (0.98 diag, 0.01 off-diag).
- `envs/wrappers/belief_obs.py` — appends the analytical HMM posterior to the observation; used by Belief-PPO.
- `beliefs/hmm_posterior.py` — forward-algorithm filter with a fill-likelihood that respects the inventory-boundary masking (blocked sides can't fill).
- `oracles/value_iteration.py` — exact VI on the full-info (inventory × regime) MDP; per-regime and mixed expected returns, plus the forward-rolled policy-commitment loss used for R1.
- `oracles/verify_requirements.py` — end-to-end verifier. Trains the six short PPO runs (3 per-regime, regime-agnostic, oracle, belief), computes all four R's, writes the stats JSON, renders the six M2 figures.
- `agents/ppo_oracle.py`, `agents/ppo_belief.py`, `agents/ppo_per_regime.py` — thin wrappers over the M1 PPO agent with regime-input or regime-locking plumbing.
- `plotting/m2_plots.py` + `plotting/regenerate_figures.py` — the six figures, regenerable from the stats JSON alone.
- Tests: `tests/test_envs.py` (lock_regime, stationary match, inventory bounds), `tests/test_beliefs.py` (forward algorithm vs brute force on short sequences), `tests/test_leak.py` (regime not leaked into the base observation).

## Notes

- **R2 threshold is 0.85, not M1's 0.95.** M2 is a diagnostic milestone with a deliberately short PPO budget (40 iterations, 3 seeds). Full-budget per-regime convergence is re-verified in M3. `regime_0_ratio = 0.901` is the smallest and clears 0.85 with margin.
- **Paired bootstrap for R3.** Across the 3 seeds every oracle run beat every agnostic run, so the paired gap is tight (CI [18.61, 23.06]). Summing independent per-method CIs instead of bootstrapping the paired difference would overstate uncertainty by ~2× and would miss the pass — this is why the spec mandates fixed seeds across methods.
- **Posterior plateau at ~0.42 nats** is the informative middle: inference works (entropy drops 61% below the flat prior) but is imperfect (doesn't crash to 0), leaving an interesting gap for learned belief-conditioned methods to fail or succeed on.
