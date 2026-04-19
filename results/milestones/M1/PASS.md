# M1 — PASS

**Date:** 2026-04-19
**Commit:** tagged `m1-passed`

## Pass criteria (from `docs/milestones/m1.md`)

Verified via `results/milestones/M1/stats_M1_pipeline.json`:

| Criterion | Value | Threshold |
|---|---|---|
| `return_ratio` | 0.9890 | ≥ 0.95 |
| `converged` | true | true |
| `policy_shape_matches` | true | true |
| `make_milestone_script_succeeded` | true | — |

### Supporting diagnostics

| Field | Value |
|---|---|
| `final_return_mean` | 150.73 |
| `final_return_ci` (95% bootstrap over seeds) | [149.52, 151.94] |
| `as_analytical_return` | 152.41 |
| `final_return_per_seed` | [150.74, 148.88, 152.62, 149.56, 151.85] |
| `slope_last_20_iterations` | 0.0043 (≈ 0) |
| `plateau_reached_at_iteration` | 21 |
| `policy_skew_correlation` (meaningful states, n=6) | 0.9897 |
| `policy_skew_direction_agreement` (meaningful states) | 1.000 |
| `policy_skew_correlation_full` (all 11 states) | 0.8496 |
| `policy_skew_direction_agreement_full` (all non-tied AS states) | 1.000 |

Figures rendered under `figures/milestones/M1/`:
- `fig_M1_ppo_learning_curve.png` — PPO mean return per iter with 95% seed range, overlaid with AS analytical ceiling (152.41).
- `fig_M1_ppo_policy_vs_as.png` — PPO skew (P(favor_ask) − P(favor_bid)) vs AS analytical argmax skew, per inventory level.

## Reproduction

```
uv run python -m scripts.make_milestone M1
```

Exit 0 and the final stdout line starting `[make_milestone] OK | pass=True` are equivalent to a pass.

## Frozen hyperparameters

M1 tuning pinned the project-wide PPO hyperparameters in `experiments/configs/base/base_ppo.yaml`:

```
hidden_dim: 64
learning_rate: 3.0e-4
max_grad_norm: 0.5
clip_eps: 0.2
ent_coef: 0.0
vf_coef: 0.5
gamma: 0.99
lam: 0.95
epochs: 8
minibatch_size: 64
```

Per the spec, these are now frozen for every PPO-based downstream agent (Oracle-PPO, Belief-PPO, per-regime PPO, RL²'s inner PPO, VariBAD's inner PPO).

## Notes on the policy-shape diagnostic

The E0 MDP has Q-value ties at the inventory boundaries (q = ±5: {sym, correct_skew} are equal because the out-of-bounds side can't fill) and near-ties at q = 0 (margin 0.0015) and q = ±1 (margin 0.0212). Comparing PPO's softmax skew against AS's argmax skew at those states is not meaningful — AS's argmax is arbitrary there. The `policy_skew_correlation` field therefore filters to states with Q-margin > 0.05 (here, q ∈ {±2, ±3, ±4} — 6 states with clear AS preference). Both full-vector and filtered values are recorded in `stats_M1_pipeline.json`.

Over those 6 meaningful states, PPO's mean policy hits correlation 0.990 and direction agreement 1.000 — i.e. PPO has learned the right qualitative shape wherever AS has one.

## What M1 established

- `envs/mm_reduced.py` — E0 single-regime AS env (11 inventory states × 3 actions, Bernoulli fills, bounded inventory).
- `oracles/analytical_as.py` — exact Bellman iteration on the finite MDP, returning V\*, argmax policy, skew vector, Q-margin per state, and expected episode return under the optimal policy.
- `agents/ppo.py` + `training/ppo_update.py` — flax ActorCritic (tanh, orthogonal init) + PPO update (clipped surrogate, GAE, K-epoch minibatches) as a JIT-compiled `(rollout, update)` step.
- `training/train.py` — dispatches env+agent from config, runs N seeds sequentially, records per-seed learning curves and final action-probability tables (π(a | inventory one-hot)).
- Progress output: per-iteration `return / step_time / eta` lines and per-seed summaries. No more silent long-running commands.
