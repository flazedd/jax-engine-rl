# Findings

Audit trail for unexpected empirical results. One dated entry per finding; do
not edit past entries — append corrections as new entries.

Per `docs/milestones/contingency.md` → "General: when results contradict
expectations", findings get written here *before* any retuning.

---

## 2026-04-20 — M3 ordering inverted on E_final

**Artifact.** `results/milestones/M3/stats_M3_reference_levels.json`
**Env.** `E_final` = `e2_fill_switched` (symmetric 3-regime, 0.98 persistence, 128-step episodes)
**Commit.** (uncommitted — see git status at time of writing)

**Observed reference levels** (mean ± 95% bootstrap CI, 5 seeds, 200 iter × 512 envs):

| level | role | mean | CI |
|---|---|---|---|
| regime_agnostic_ppo | floor | 122.78 | [122.12, 123.66] |
| belief_ppo | ceiling | 122.80 | [122.04, 123.59] |
| oracle_ppo | ceiling | 129.99 | [129.39, 130.58] |
| per_regime_ppo (stationary-weighted) | ceiling | 112.45 | [111.88, 113.17] |

Expected ordering `agnostic ≤ belief ≤ oracle ≤ per_regime` does **not** hold.
Actual: `per_regime < agnostic ≈ belief < oracle`.

### Two anomalies

**(A) per_regime < oracle.** The "ceiling" from N separate locked-regime
networks lands below the shared-network oracle.
- per-regime component returns: r0=150.5, r1=93.0, r2=93.8 → stationary-weighted 112.
- oracle on mixed env: 130.
- VI gives mixed optimum = 125.65 (stationary basis) and per-regime stationary
  = [152, 78, 78] with stationary-weighted average 102.9.
- The VI-derived stationary-weighted per-regime aggregate (102.9) is **below**
  VI's mixed optimum (125.65), so this is **not** a PPO training artifact — it
  is a structural property of the env: locked-regime training decomposes the
  problem in a way that loses cross-regime inventory leverage. Oracle-PPO in the
  mixed env can carry inventory across regime transitions; N independent locked
  policies cannot.
- This matches `contingency.md` → M3 → "If it persists even with more
  iterations, the finding is real… reportable" — but here the cause is not
  oracle-PPO regularization; it is that the thesis's "per-regime ceiling"
  definition does not actually upper-bound oracle-PPO on envs with frequent
  transitions and cross-regime state carryover.

**(B) belief_ppo ≈ regime_agnostic_ppo.** Belief gets the analytical HMM
posterior as input but performs identically to the no-regime-info baseline.
- In M2 (shorter training) belief was at 119.7, regime_agnostic at 100.4 (gap
  ≈ 19). In M3 (200 iter × 512 envs) both converged to ≈ 122.8.
- The compromise policy's per-regime expected returns (VI) are
  [152.3, 77.5, 78.1] — **nearly identical to per-regime VI optima**
  [152.4, 78.4, 78.4]. Meaning: the best regime-agnostic policy is already
  almost regime-optimal in each regime at stationary. Regime info mostly helps
  at transitions (react faster), which is captured by oracle (exact regime
  info) but not by noisy belief in this finite-horizon setup.
- This could be either: (i) real (compromise is near-optimal, belief can't
  help much), or (ii) broken (belief-PPO ignores its belief input). Direct
  test: log gradient norm w.r.t. belief-input channel during training, or
  re-run with a wider policy network.

### Gap decomposition as reported

- shared_network_cost (per_regime − oracle) = **−17.53** (CI [−18.20, −17.07])
- inference_cost (oracle − belief) = **+7.19** (CI [+6.09, +7.96])
- compromise_policy_cost (belief − agnostic) = **+0.02** (CI [−0.98, +1.07])
- total_gap (per_regime − agnostic) = **−10.33** → negative, breaks the
  decomposition's intended monotone structure.

### Options (for user decision)

1. **Accept as-is, mark M3 informational.** Report negative `shared_network_cost`
   as a real property of this env. Belief≈agnostic either as a real property or
   as a flagged M4 debug target.
2. **Redefine per-regime training.** Train N separate networks on the **mixed**
   env with regime-based dispatch (each network updates only on its regime's
   samples). This matches the thesis intent of "no shared network" without the
   locked-episode artifact, and should upper-bound oracle by construction.
3. **Investigate belief-PPO.** Gradient probe / widen network / test on a
   lower-difficulty env where compromise is far from optimal.

My read: option 2 is the cleanest — the locked-regime training protocol is the
source of the structural break. Option 3 is orthogonal and should be addressed
in M4 regardless.

### Resume-from-cold brief (for next session)

**State at end of 2026-04-20 session.**
- All M3 code shipped to working tree, nothing committed, no tag.
- 200-iter × 512-env × 5-seed full run done; artifacts in `results/m3_*`.
- `make_milestone M3` end-to-end runs clean, writes
  `results/milestones/M3/stats_M3_reference_levels.json` with `pass=False,
  ordering_valid=False, all_converged=True`.

**Files touched / added.**
- `experiments/configs/m3_regime_agnostic.yaml` (new)
- `experiments/configs/m3_oracle.yaml` (new)
- `experiments/configs/m3_belief.yaml` (new)
- `experiments/configs/m3_per_regime.yaml` (new, triggers sweep via `agent.name: ppo_per_regime`)
- `training/train.py` — added `_stationary_distribution_np`,
  `train_per_regime_sweep`, `train_or_sweep`; `main()` now calls
  `train_or_sweep(cfg)`.
- `plotting/m3_plots.py` (new) — `plot_rq1_ceilings_bar`,
  `plot_rq1_learning_curves`, `plot_rq1_gap_fractions`.
- `plotting/regenerate_figures.py` — added `regenerate_m3` handler.
- `scripts/make_milestone.py` — added `make_m3` with `_convergence_diagnostics`,
  `_ci_paired`, `_gap_component`, `_run_m3_training`.
- `FINDINGS.md` (this file) — new.

**Convergence diagnostic.** All four methods plateau well before iter 200 with
last-20-slope ≈ 0 — more iterations will not change the outcome.

**What each option costs to execute.**
- Option 1 (accept): ~30 min. Write `results/milestones/M3/PASS.md` (or
  `RESULT.md`) documenting why ordering fails; update `CLAUDE.md` current
  status; do not tag `m3-passed`. Needs docs update to research-questions.md
  acknowledging that per-regime is not a strict ceiling on this env class.
- Option 2 (redefine per-regime): ~1 day. New agent `ppo_per_regime_mixed` that
  trains N separate networks on the mixed env, each masking gradients to
  regime-matched samples. Re-run M3 full mode (~30 min compute). Update
  `docs/milestones/m3.md` build-order step 1 and research-questions.md per-regime
  protocol. If the ordering now holds, everything downstream is unchanged.
- Option 3 (belief debug): ~0.5 day. Add a gradient-norm log hook to belief-PPO
  at iter 0/50/100/200; compare to oracle-PPO. Orthogonal to the ordering
  question — even if per-regime is fixed, belief≈agnostic needs to be
  explained before M5.

**Key numerical evidence to re-read first.**
- `results/milestones/M3/stats_M3_reference_levels.json` — headline stats.
- `results/m3_per_regime_r{0,1,2}/metrics.json` — individual locked-regime runs
  showing r0=150.5, r1=93.0, r2=93.8.
- VI oracle (see `oracles/value_iteration.py` re-run in-session):
  mixed_expected_episode_return=125.65, per_regime=[152.4, 78.4, 78.4],
  compromise_per_regime=[152.3, 77.5, 78.1].

**Not touched — safe to revisit.**
- Tests pass at HEAD. M0/M1/M2 tags and PASS.md untouched.
- `envs/wrappers/belief_obs.py` confirmed working (posterior updates correctly
  during a hand-stepped rollout; concentrates to ~0.88 on true regime after
  ~15 steps).

**Suggested first move tomorrow.** Re-read this entry top-to-bottom, then look
at `results/milestones/M3/stats_M3_reference_levels.json` to confirm numbers
haven't drifted, then pick the option. If option 2, the implementation
anchor is a new agent class alongside `ppo_oracle` that masks its PPO update
to samples where `regime == self.my_regime`.

---

## 2026-04-21 — Per-regime PPO dropped as a reference level and method-ladder rung

**Decision.** Remove per-regime PPO from RQ1's decomposition, M3's reference levels, and M5's method ladder. Reformulate RQ1 as a **2-way** decomposition: `total_gap = inference_cost + compromise_policy_cost` where `inference_cost = oracle − belief` and `compromise_policy_cost = belief − agnostic`.

**Why.**
- On E_final, `shared_network_cost = per_regime − oracle = −1.10` with CI `[−2.30, +0.51]` — the CI crosses zero, so the component is not measurable. The 3-way decomposition's third slice is indistinguishable from noise.
- Per-regime PPO with N hard-dispatched heads does not correspond to any realistic deployable method; oracle-PPO already captures "knows the regime, single shared network." Keeping it as a pure architectural probe added narrative burden without insight.
- A wider-network oracle absorbs whatever architectural capacity cost per-regime was meant to isolate, making the component a design-parameter artifact rather than a fundamental property of the problem.

**What changed.**
- Removed `agents/ppo_per_regime_mixed.py`, `experiments/configs/m3_per_regime*.yaml`, `scripts/m3_splice_per_regime_mixed.py`.
- Reverted the per-sample `weight` feature in `training/ppo_update.py` (only `_mixed` used it).
- Rewrote `results/milestones/M3/stats_M3_reference_levels.json` with 3 reference levels and 2-way gap decomposition; ordering_valid remains true (`agnostic=122.78 ≤ belief=122.80 ≤ oracle=129.99`).
- Updated `CLAUDE.md`, `docs/research-questions.md`, `docs/milestones/m3.md`, `docs/milestones/m5.md`, `docs/milestones/contingency.md`, `docs/implementation.md`, and the M3 plotting module to reflect 3 reference levels / 6-rung ladder.

**What stays.**
- `agents/ppo_per_regime.py` (locked-regime wrapper) is kept as an **env-validation tool** only — used by `oracles/verify_requirements.py` for M2's R2 check (locked-regime PPO reaches VI optimum). Not a reference level, not a method.
- `stats_M3_reference_levels_locked_regime.json` and the `results/m3_per_regime*` result directories are retained as audit trail; not referenced by downstream tooling.

**Implications for the thesis.**
- RQ1 decomposition is now cleaner (every component is measurable), and the thesis reports 2-way instead of 3-way.
- RQ2 method-ladder figures lose one horizontal reference line (per-regime ceiling).
- M6 difficulty sweep re-computes only 3 reference levels per difficulty point (was 4).

**Still open.** The observed total gap on E_final is only ~7 return units (5.6% of regime-agnostic mean). RQ2's statistical power depends on the gap being wider on at least some points in the M6 sweep. Env redesign for M6 (asymmetric regimes, wider fill-rate spreads, or lower persistence) remains to be drafted.
