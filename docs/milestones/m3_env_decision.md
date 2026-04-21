# M3 env-choice decision (paused)

**Status:** open. Pick up here before continuing M3 / starting M4.

## Where we are

M3 was run on `E_final = e3_asymmetric`. Reference levels converged and the
ordering is valid, but the empirical compromise-policy cost is ~0 and the
belief signal adds nothing on top of a constant prior.

```
regime_agnostic_ppo : 105.23   [104.10, 106.27]
belief_ppo          : 105.85   [104.99, 106.70]
oracle_ppo          : 111.49   [110.55, 112.68]

inference_cost           : 5.64   CI [4.64, 6.64]      (90.2% of total gap)
compromise_policy_cost   : 0.61   CI [-0.33, 1.46]     (9.8%, crosses 0)
total_gap                : 6.25
```

Source: `results/milestones/M3/stats_M3_reference_levels.json`.

## Diagnostics run (`scripts/m3_belief_diagnostics.py` + D2 ablation)

- **D1 (structural):** PASS. obs_size=14 = 11 inv + 3 belief; belief portion
  of obs matches tracked belief exactly; belief varies across steps.
- **D3 (informativeness, random policy):** STRONG. uniform entropy 1.099,
  mean entropy 0.420 (62% decay); argmax accuracy 85.4%; avg probability
  assigned to true regime 77.6%; 70.8% of timesteps "confident" (entropy <
  0.5 × uniform). Output: `results/milestones/M3/belief_diagnostics.json`.
- **D2 (retraining ablation, `mm_reduced_belief_constant` env):** DECISIVE.
  Replacing the time-varying posterior with the stationary prior at every
  step leaves return *unchanged*.
  ```
  real belief-PPO      : 105.85
  constant-belief-PPO  : 105.88
  regime-agnostic-PPO  : 105.23

  real − const : mean −0.04   CI [−0.76, +0.68]   indistinguishable
  const − agn  : mean +0.65   CI [+0.19, +1.31]   significant
  real − agn   : mean +0.61   CI [−0.36, +1.46]   not significant
  ```
  The 0.6-unit lift over agnostic is a pure capacity effect from 3 extra
  constant input dims, **not** belief content.

## Diagnosis

- Not a wiring bug. `envs/wrappers/belief_obs.py` feeds real time-varying
  posteriors; D1 confirms layout; const vs real obs differ per-step.
- Not a posterior-quality bug. D3 shows the analytical HMM posterior is
  genuinely informative under a random rollout policy.
- **E_final's compromise policy is near-optimal.** Oracle beats it by 6.3
  units, but that gap comes from a crisp one-hot that PPO can gate actions
  on; an 85%-sharp belief distribution gets smeared through the tanh MLP
  and the policy cannot convert it into extra reward.

## Decision

### Option A — switch to E4 (recommended)

`experiments/configs/envs/e4_deep_wide.yaml` already exists. VI sweep
projects total gap 59.6 (vs E_final 20.96), compromise_cost 55.3 (vs
12.9). That is the only way to get a meaningful empirical RQ1
decomposition: if compromise_cost is ~0 there is nothing for M5's meta-RL
methods to close.

Sub-steps if A:
1. Run R1–R4 validation on E4 (~30 s) before spending 40 min on retrain.
2. Retrain all three M3 reference levels on E4 (~40 min: 3 methods × 5
   seeds × 200 iter @ 512 envs ≈ 14 min each, serial).
3. Update `E_final` symlink to `e4_deep_wide.yaml`.
4. Update CLAUDE.md "Current env version" block.
5. Keep E3 available as the easy end of M6's difficulty sweep.

### Option B — stay on E_final

M3 technically passes JSON criteria and can be tagged. But the thesis
narrative for M5 becomes "meta-RL methods close almost none of the gap,
because there is almost no compromise-policy cost to close on this env",
which is not what RQ2 wants to claim. RQ1's decomposition itself becomes
a footnote: 90% inference cost, ~0% compromise cost.

## Recommendation

A, validate E4 against R1–R4 first, keep E3 as the M6-easy datapoint.

## To resume

1. Read this file.
2. Read `results/milestones/M3/stats_M3_reference_levels.json` and
   `results/milestones/M3/belief_diagnostics.json` to confirm numbers
   haven't changed.
3. Ask the user A or B; if A, start with R1–R4 on E4.
