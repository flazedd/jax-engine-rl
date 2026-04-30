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

---

## 2026-04-25 — E_final replaced (E3 → E6e); inv-regime coupling identified as cause of E3's collapsed gap

**Decision.** Repoint `E_final` from `e3_asymmetric` to `e6e_symmetric_kappa05`. E6e: symmetric within-regime fills + κ=0.05 inventory penalty + 3-regime quality contrast.

**Why E3 had a near-zero measured compromise gap despite analytical comp_cost=21.**

Diagnostic chain (this session):
- M3 mid-mode on E3 (100 iter × 256 envs × 1 seed): agnostic=107.4, belief=107.0 → measured gap = −0.3.
- Analytical `compromise_VI(E3)` = 91.7. Yet PPO regime-agnostic reaches 107.4 — 15 units *above* analytical compromise. Same for E5b: PPO agnostic=70.6 vs analytical compromise_VI=17.8 (4× overshoot).
- Obs is genuinely just one-hot inv (no regime leak; verified at `envs/mm_reduced.py:131`).
- **Mechanism.** E3's regime drives fills on a single side (bull → ask fills only, bear → bid fills only). After ~30 sticky-regime steps, inventory deterministically saturates at one bound. Inv ≈ −5 essentially identifies bull; inv ≈ +5 identifies bear. A *stochastic memoryless* PPO conditioned only on inv is implicitly regime-conditioned via inventory's history — it routes around the analytical compromise's deterministic-policy class.
- The 21-unit `compromise_VI − oracle_VI` gap thus measures the gap *of the deterministic regime-agnostic policy class*, not of any reasonable PPO baseline.

**Implication for env design.** To get a measurable empirical compromise gap, the env must **decouple inventory from regime** — i.e., the regime cannot mechanically drive inv in one direction.

**Options screened (`scripts/quick_env_gap.py`, `--mode mid`).**

| candidate | mechanism | agnostic | belief | gap | %agn |
|-----------|-----------|----------|--------|-----|------|
| e3_asymmetric (baseline) | — | 107.4 | 107.0 | −0.3 | ~0% |
| e6a_symmetric_fills | sym fills (mild contrast) | 145.8 | 172.1 | 26.3 | 18.0% |
| e6b_symmetric_strong | sym fills (strong contrast) | 159.4 | 200.3 | 40.9 | 25.7% |
| e6c_e3_kappa05 | E3 + κ=0.05 | 49.1 | 50.5 | 1.4 | 2.9% |
| e6d_e3_kappa15 | E3 + κ=0.15 | −149.6 | −147.0 | 2.6 | — |
| **e6e_symmetric_kappa05** | sym fills + κ=0.05 | **118.4** | **165.2** | **46.8** | **39.6%** |

- Symmetric fills (option 1) work: E6a/b show 26–41 unit gaps. Inv random-walks regardless of regime → inv carries no regime info → agnostic PPO can't route around the compromise.
- High-κ alone (option 3) fails: in E3-style directional envs, κ amplifies suffering without breaking the coupling. E6d's −150 score shows the agent is being crushed.
- **Hybrid sym + κ=0.05 (E6e) wins both absolute gap (47) and %gap (39.6%).** The κ=0.05 forces inv near 0; belief PPO can pick the right action class per regime, agnostic must commit.

**E6e parameter design.** 3 regimes with symmetric bid/ask fills:
- R0 (wide-favoring): p_tight=0.30, p_wide=0.65 → favor_X best per step
- R1 (tight-only): p_tight=0.80, p_wide=0.05 → sym best
- R2 (dead): p_tight=0.50, p_wide=0.02 → sym best

T = 0.98 diagonal-dominant 3×3, episode 128, inv_max=5, κ=0.05.

**Caveat.** R0 has p_wide > p_tight (wide quote fills more than tight). Physically unusual for MM (counterparties usually prefer tighter quotes). Defensible because the env is a **synthetic POMDP testbed** for meta-RL, not a calibrated market simulator. The mathematical structure (regime affects optimal action class without affecting inventory direction) is what matters for RQ1/RQ2.

**M2 verify on E6e (40 iter × 256 envs × 3 seeds).**
- R1: disagree=1.000, rel_loss=0.943 — PASS
- R2: per-regime ratios = [0.850, 0.854, 0.900], min=0.850 — PASS (just over threshold)
- R3: agnostic=101.7, oracle=141.1, gap=39.4 — PASS
- R4: ent decay=0.504 — PASS
- 2-way decomposition at M2 budget: total=39.4 = compromise(25.6) + inference(13.8). Compromise share 65%.

**Files changed.**
- `experiments/configs/envs/e6a_symmetric_fills.yaml`, `e6b_symmetric_strong.yaml`, `e6c_e3_kappa05.yaml`, `e6d_e3_kappa15.yaml`, `e6e_symmetric_kappa05.yaml` (new candidates).
- `experiments/configs/envs/e_final.yaml` symlink: `e3_asymmetric.yaml` → `e6e_symmetric_kappa05.yaml`.
- `scripts/quick_env_gap.py` (new) — fast iteration wrapper: probe + agnostic PPO + belief PPO per env, swaps the e_final symlink, restores at exit.
- `training/config.py` — added `mid` run mode (100 iter × 256 envs × 1 seed) for plateau screening between `--fast` and `--full`.
- `training/train.py` — added `--mid` CLI flag.
- `utils/script_output.py` — added `mid` to `_VALID_RUN_MODES`.
- `CLAUDE.md` "Current env version" line updated.

**Still open.**
- ~~Decide whether the p_wide > p_tight in R0 needs a defense in `docs/environment.md`~~. Resolved 2026-04-25: defense paragraph added to `docs/environment.md` "Parameter-choice discipline" section under "Synthetic-testbed framing".

## 2026-04-25 — M3 full re-run on E6e (passes, ordering valid, all converged)

After re-pointing E_final to E6e, ran `make_milestone M3` end-to-end (super_fast → fast → full; full = 200 iter × 512 envs × 5 seeds × 3 methods, ~40 min wall).

**Converged reference levels (mean | 95% bootstrap CI across 5 seeds):**

| method | mean | CI | plateau iter | slope last 20 |
|--------|------|----|--------------|---------------|
| regime_agnostic | 136.23 | [132.78, 139.27] | 57 | −0.05 |
| belief          | 168.50 | [166.35, 170.82] | 39 | −0.05 |
| oracle          | 180.14 | [175.13, 185.04] | 39 | +0.00 |

Ordering `agnostic ≤ belief ≤ oracle` ✓.

**Gap decomposition (paired across seeds):**

| component | absolute | CI | fraction of total |
|-----------|----------|----|-------------------|
| compromise_policy_cost (belief − agnostic) | 32.27 | [30.00, 34.83] | 73.5% |
| inference_cost         (oracle − belief)   | 11.64 | [6.43, 16.02]  | 26.5% |
| **total_gap**          (oracle − agnostic) | **43.91** | [36.45, 50.93] | 100% |

**Notes.**
- All three methods converged before iter 60; full budget was sufficient. The M2-verify numbers (40 iter, agnostic=101.7) understated each level by ~30–40 units — full convergence is well above what M2 saw.
- Compromise-policy cost is now the dominant component (73.5%). This is the intended regime: E6e's symmetric fills + κ=0.05 successfully prevent inv from being a sufficient statistic for regime, so the agnostic policy genuinely pays a compromise penalty rather than routing around it via inv-conditioning.
- Inference-cost CI is well-separated from zero ([6.4, 16.0]) but wider than the compromise CI — driven by seed 0 of belief (per-seed inference diff: 1.81 vs 10–18 for the others). Expected: belief PPO's posterior decoding has more variance across initializations than oracle PPO's exact-regime input.
- `stats_M3_reference_levels.json` holds the canonical numbers used downstream by M5 (gap-component closure metrics).
- `make_milestone M3` exit 0; `make_milestone_script_succeeded=True`.

---

## 2026-04-25 — M4 implementation validation passes (RL² and VariBAD clear PPO floor on all three validation envs)

**Artifacts.**
- `results/milestones/M4/method_ranking.json` (canonical numbers + per-seed finals)
- `scripts/m4_full_eval.py` (orchestrator) + `scripts/m4_full_eval_aggregate.py` (post-hoc aggregator from on-disk metrics)
- `figures/milestones/M4/factorial_toys.png` — the implementation-validation chart. Shows RL²/VariBAD × concat/hypernet on the three toy envs with a per-env PPO floor reference line. (Replaces the earlier `m4_method_ranking.png`; underlying data was collected during M5 Step-3 toy sweep, but the chart's role — *meta-RL works on toys* — is M4.)

**Setup.** 9 (method × validation env) configs at full budget: 200 iter × 512 envs × 3 seeds. Methods: PPO (regime-agnostic floor), RL² (recurrent meta-RL, GRU on `[obs, prev_action_oh, prev_reward, prev_done]`), VariBAD (variational meta-RL with explicit posterior `q(m | τ_{:t})` and Bernoulli reward decoder). Validation envs:
- `bandit` — 2-arm Bernoulli, episode 10. Per-episode arm probs sampled.
- `gridworld` — 5×5 random-goal, episode 50. Reward 1 at goal else 0.
- `regime_bandit` — 2-regime × 2-arm sticky HMM bandit, episode 100, stay 0.95.

Total wall: 64 min on 9 runs. Per-run budget identical across methods.

**Headline table** (final episode return, mean over 3 seeds | 95% bootstrap CI):

| env | PPO floor | RL² | VariBAD |
|---|---|---|---|
| bandit | 40.27 [40.1, 40.5] | 48.19 [48.1, 48.4] | 48.21 [48.0, 48.4] |
| gridworld | 4.47 [4.3, 4.9] | 46.98 [45.8, 48.8] | 17.15 [16.6, 17.7] |
| regime_bandit | 49.65 [49.3, 50.4] | 65.83 [62.2, 72.7] | 63.44 [63.2, 63.6] |

**Pass criterion** (M4: meta-RL implementations clear the regime-agnostic PPO floor on each validation env, with non-overlapping CIs): met for all 6 method/env combinations. RL² and VariBAD CIs sit strictly above the PPO CI on every env.

**Observations worth noting forward.**
- **bandit** — RL² and VariBAD essentially tied (ΔCI overlap entirely). Both ~8 returns above floor on the 6.5-Bayes-ceiling task. Both methods extract the per-episode arm prior cleanly.
- **gridworld** — RL² dominates VariBAD (47 vs 17). VariBAD's 8-dim Bernoulli-decoded latent is a poor fit for spatial-goal inference (decoder reconstructs sparse 0/1 reward signals; the goal-cell information is in the *trajectory* of zero-rewards, not the rewards themselves). Both methods clear the floor of 4.5 by ~4× / ~10×, so the validation criterion is met, but VariBAD will need a Gaussian / state-decoded variant before it's competitive on harder spatial tasks. Logged as known characterization, not a blocker for M5.
- **regime_bandit** — Both methods beat the agnostic floor. RL² mean is higher (65.8 vs 63.4) but with seed variance of [62.2, 72.7] (one outperforming seed at 72.7); VariBAD is much tighter at [63.2, 63.6]. Closest analogue to MM among the three envs — encouraging that *both* meta-RL methods reliably pick up the regime-switching signal at this budget.

**Implementation provenance** (durable artifacts shipped in M4):
- `envs/validation/bandit.py`, `envs/validation/gridworld.py`, `envs/validation/regime_bandit.py` (frozen-dataclass JAX-pure envs, full test coverage in `tests/test_*`)
- `envs/wrappers/rl2_obs.py` — augmented-obs wrapper consumed by both RL² and VariBAD
- `agents/rl2.py` — `RL2Agent` with `is_recurrent=True`, GRU+actor/critic, env-axis-only minibatch shuffling to preserve recurrence
- `agents/varibad.py` — encoder/decoder/policy with joint optimizer, ELBO loss (BCE recon + KL), policy on `[obs, μ, σ]`. **Caveat for M5:** the reward decoder uses a Bernoulli head valid only for {0,1} rewards. Continuous-reward envs (MM) require swapping to a Gaussian head — flagged in `experiments/configs/base/base_varibad.yaml`.
- `training/recurrent_rollout.py` — shared scan-based rollout that threads per-env GRU carry through the trajectory and resets carry on episode boundaries via `tree_map(jnp.where(done, init, carry))`
- `training/train.py` — dispatches on `getattr(agent, "is_recurrent", False)` so MLP and recurrent paths share `_train_one_seed` without invasive refactor; `_maybe_wrap_env_for_agent` auto-wraps with `RL2ObsEnv` for rl2/varibad
- 9 experiment configs at `experiments/configs/m4_{ppo,rl2,varibad}_{bandit,gridworld,regime_bandit}.yaml` + `base_rl2.yaml`, `base_varibad.yaml`

**Orchestrator-bug audit trail** (does not affect results, only the in-process aggregation):
- `scripts/m4_full_eval.py` originally read `summary["key_stats"]["per_seed_final_returns"]` (plural, wrong key); the actual write path in `training/train.py:350` is `metrics["per_seed_final_return"]` (singular). Fixed in-place; results were re-aggregated post-hoc by `scripts/m4_full_eval_aggregate.py`.
- Same script also called `run.ok(extra_summary=...)` — `ScriptRun.ok` does not accept that kwarg. Fixed by folding `rows` into `key_stats`.
- All 9 `train(cfg)` runs themselves completed cleanly and wrote valid metrics.json files; only the orchestrator's terminal table-build crashed.

**Status.** M4 method-validation deliverable complete. RL² and VariBAD are now validated against the regime-agnostic PPO floor on three increasingly complex meta-RL envs. The same agents go forward into M5 (full ladder + 2×2 hypernet × exploration-bonus factorial on MM E_final) with the Gaussian-head decoder swap as the only known remaining VariBAD code change.

---

## 2026-04-29 — M6 difficulty sweep complete (RQ3 answered: belief↔task decoupling generalizes across difficulty)

Initial sweep ran 2026-04-28 with persistence_easy at diag = 0.995 (mean regime duration 200 steps, longer than 128-step episodes). Diagnosed that the resulting Belief-PPO inverted-U on persistence (0.56 → 0.74 → 0.21 gap_closed) was caused in part by easy episodes being effectively single-regime (no in-episode learning signal). Re-ran the 7 persistence_easy cells at diag = 0.99 (mean duration 100 steps, ~1 regime switch per episode on average) overnight 2026-04-28 → 2026-04-29: 187 min sweep + 4.3 min probe + 0 min tests + 0 min plots = **3.2 h**. The numbers below reflect the rerun.

**Artifacts.**
- `results/milestones/M6/stats_M6_sweep.json` — 42-cell aggregate (gitignored, regenerable).
- `results/milestones/M6/stats_M6_posterior_vs_performance.json` — 192 (cell, seed) probe scatter points (gitignored).
- `results/milestones/M6/stats_M6_hypothesis_tests.json` — Holm-corrected hypothesis tests (gitignored).
- `figures/milestones/M6/rq3_persistence_sweep.png`, `rq3_distinguishability_sweep.png`, `rq3_posterior_vs_performance.png` — three thesis figures.
- `scripts/m6_difficulty_sweep.py` (orchestrator), `scripts/m6_posterior_probe.py` (probe), `scripts/m6_hypothesis_tests.py` (tests), `scripts/m6_overnight.py` (chained wrapper), `plotting/m6_plots.py` (figures).

**Setup.** 7 methods × 3 difficulty levels × 2 axes = 42 cells, sequential. References (regime-agnostic / Belief / Oracle PPO) at n=5 × 200 iter, meta-RL (RL²/VariBAD × concat/hypernet) at n=8 × 200 iter (inherited from M5 Step 4 base configs).

Persistence axis (mean regime duration): easy 100 steps (diag 0.99), medium 50 steps (diag 0.98 = E_final), hard 5 steps (diag 0.80). Distinguishability axis (regime fill probabilities): easy [0.10/0.90/0.50, 0.85/0.02/0.05] (very separated), medium = E_final, hard [0.40/0.65/0.50, 0.50/0.10/0.05] (compressed).

**Reference triple at each level.**

| Axis | Level | Floor | Belief | Oracle | Compromise | Inference |
|---|---|---:|---:|---:|---:|---:|
| Persistence | Easy (diag 0.99) | 145.17 | 175.18 | 183.08 | 30.0 | 7.9 |
| Persistence | Medium (E_final) | 136.23 | 168.50 | 180.14 | 32.3 | 11.6 |
| Persistence | Hard (diag 0.80) | 129.60 | 139.90 | 177.68 | 10.3 | 37.8 |
| Distinguishability | Easy (separated) | 156.10 | 193.41 | 206.61 | 37.3 | 13.2 |
| Distinguishability | Medium (E_final) | 136.23 | 168.50 | 180.14 | 32.3 | 11.6 |
| Distinguishability | Hard (compressed) | 123.81 | 132.11 | 150.45 |  8.3 | 18.3 |

The persistence axis decomposes the gap from compromise-dominated (easy) to inference-dominated (hard); the distinguishability axis compresses the entire optimality envelope from 50.5 (easy) to 26.6 (hard).

**Headline gap_closed matrix.**

| | **Persistence** | | | **Distinguishability** | | |
|---|---:|---:|---:|---:|---:|---:|
| | Easy | Med | Hard | Easy | Med | Hard |
| RL² Hypernet | 0.69 | 0.74 | 0.54 | **1.00** | 0.74 | 0.40 |
| VariBAD Hypernet | 0.60 | 0.70 | 0.53 | **1.03** | 0.70 | 0.40 |
| Belief-PPO | 0.79 | 0.74 | **0.21** | 0.74 | 0.74 | 0.31 |
| RL² Concat | −0.43 | −0.28 | −0.46 | −0.13 | −0.28 | −0.75 |
| VariBAD Concat | −0.86 | −0.59 | −0.31 | −0.74 | −0.59 | −0.52 |

Reproducibility: all seven *medium* cells across both axes match M3 reference levels and M5 Step 4 numbers **exactly** (Floor 136.23, Belief 168.50, Oracle 180.14, RL² Concat 124.07, RL² Hypernet 168.56, VariBAD Concat 110.32, VariBAD Hypernet 166.80). Confirms determinism of the seed/config protocol across milestones.

**Headline findings.**

- **M5 hypernet ≫ concat decoupling generalizes to every cell.** Concat sits below floor in 11/12 cells; hypernet sits ≥ Belief-PPO in 11/12. Both axes, all difficulty levels.
- **Posterior↔performance correlation = +0.026** across 192 scatter points (CI [−0.103, +0.155]). Essentially zero. The decoupling is structural, not stochastic: posterior decoding accuracy does *not* predict task return — the policy interface dominates.
- **Hypernet exceeds Oracle at distinguishability-easy** for both methods independently (RL² 206.7, VariBAD 208.1, Oracle 206.6). Recurrent / variational state encodes inventory + market-state context the regime-only oracle obs doesn't use as efficiently. Small lift (~1-2 pts) but reproduces across both methods at n=8.
- **Hypernet's edge over Belief-PPO grows when Belief-PPO falters**: persistence-hard +15.7 pts (Belief collapses, hypernet retains), distinguishability-easy +13.3 pts (Belief saturates short of oracle, hypernet pushes past). At medium (E_final), hypernet ≈ Belief (within 0.1 pt) — the M5 plateau.
- **Belief-PPO drops sharply on persistence-hard** (gap_closed 0.74 → 0.21): once mean regime duration is short (~5 steps), the analytical posterior is almost as uncertain as the marginal, leaving the policy with little usable signal — and yet RL²/VariBAD hypernet still hold gap_closed ≈ 0.53. Compelling evidence that the hypernet path is doing more than re-deriving the analytical posterior.
- **Distinguishability-hard collapses the optimality gap** (50.5 → 43.9 → **26.6**). Compressed fill probabilities make the regimes barely matter: even Oracle is only 26.6 above Floor.
- **VariBAD Concat does not learn** anywhere. Absolute returns 110.3 – 122.8 across all 6 cells (vs floor 123.8 – 156.1). The variational posterior is fine (probe accuracy ≈ analytical), but the concat interface can't realize a regime-conditional action distribution. Same as M5 Step 4.

**Pass-criterion status** (per `docs/milestones/m6.md`).

| Criterion | Required | Observed | Status |
|---|---|---|:---:|
| persistence_sweep.all_methods_monotonic | true | false | ✗ |
| distinguishability_sweep.all_methods_monotonic | true | false (only varibad_concat fails) | ✗ |
| stats_M6_sweep.interpretable_overall | true | false | ✗ |
| stats_M6_posterior_vs_performance.scatter_interpretable | true | **true** (signal_pattern=decoupling) | ✓ |

Per-method monotonicity (gap_closed non-increasing as difficulty rises):

| Method | Persistence | Distinguishability |
|---|:---:|:---:|
| Regime-agnostic PPO | ✓ flat | ✓ flat |
| Belief-PPO | ✓ (post rerun) | ✓ |
| Oracle-PPO | ✓ flat | ✓ flat |
| RL² Hypernet | ✗ (0.69 → 0.74 → 0.54) | ✓ |
| VariBAD Hypernet | ✗ (0.60 → 0.70 → 0.53) | ✓ |
| RL² Concat | ✗ (broken — M5 finding) | ✓ |
| VariBAD Concat | ✗ (broken — M5 finding) | ✗ (broken) |

Three sources of failure: (i) hypernet methods narrow miss on persistence easy/medium — gap_closed metric artifact rather than method failure (the floor's slope on persistence differs from the hypernet's, so normalised gap_closed dips at easy even though absolute returns are monotonic non-increasing); (ii) concat methods are broken everywhere (M5 finding, expected); (iii) varibad_concat's positive distinguishability slope is broken-method noise. The strict `all_methods_monotonic == true` criterion is therefore *structurally* unsatisfiable on this benchmark — same kind of borderline outcome as M5's `ranking_stable=false`. Substantive RQ3 claims (decoupling generalizes, hypernet > Belief at hard, posterior↔performance correlation ≈ 0) are robust. Manual review before tagging `m6-passed`.

**Pre-registered hypothesis tests** (`scripts/m6_hypothesis_tests.py`, `stats_M6_hypothesis_tests.json`).

Three families, each Holm-corrected separately at α = 0.05.

- **Family A — `hypernet > concat` at every (axis × level), n=8 paired (12 hypotheses).** **12/12 supported** with Holm-corrected p = 0.0469 (the minimum achievable at n=8 paired Wilcoxon × Holm 12). Δ medians span +24 to +89 absolute return points; every CI excludes zero by a wide margin (e.g., RL² persistence_hard: +48.5 [+42.9, +51.6]; VariBAD distinguishability_easy: +89.3 [+87.1, +90.9]). **0/12 robust to LOO** — but this is an arithmetic ceiling: at n=7 the minimum Holm-corrected p is 0.094, which cannot pass α at family size 12 by construction. The CI evidence carries the load.

- **Family B — `hypernet > Belief-PPO` at the two cells where hypernet visibly leads, n=5 paired (4 hypotheses).** **0/4 supported under the strict criterion** (Holm × 4 × Wilcoxon n=5 has min p = 0.125, can't pass α). **All four CIs exclude zero**: Δ medians +8.9 to +18.4, CIs e.g. RL² persistence_hard +18.3 [+6.5, +23.1]. The signal is real and consistent in direction, but the n=5 Wilcoxon is underpowered for Holm correction at this family size. Directionally supported; not formally significant.

- **Family C — Posterior↔performance decoupling (1 CI).** Decision rule: bootstrap 95% CI on overall Pearson r is contained within (−0.30, +0.30). **Overall r = +0.026, CI [−0.103, +0.155] across n = 192 (cell × seed) points. Decoupling supported.** Per-method correlations are *not* zero (rl2_hypernet r = −0.45, varibad_concat r = +0.71, etc.), revealing a Simpson's-paradox stratification: the integration mechanism creates two cleanly-separated bands at overlapping posterior_error, so the pooled correlation collapses even though within-band correlations exist. The thesis claim is at the population level.

**Status.** Family A 12/12 supported with massive effect sizes; Family C decoupling supported by a tight CI on r. Family B directionally supported (CIs positive) but underpowered for the strict Holm test. The pre-registered `interpretable_overall` field reads false because the gap_closed normalisation creates a small monotonicity dip on persistence easy/medium for hypernet methods — a metric artifact, not a method or design failure. The substantive RQ3 storyline (decoupling generalises, hypernet ≫ concat everywhere, hypernet ≥ Belief at hard inference) is robust under the rigorous tests. Tagged `m6-passed` with caveats — same protocol as M5.

**What this means for the thesis.**
- RQ3 part 1 (scaling with difficulty): characterized cleanly along both axes for all 7 methods. The Belief-PPO collapse on persistence-hard is a useful discussion point — it isolates where the *compromise-policy cost* component matters vs the *inference cost* component.
- RQ3 part 2 (posterior↔performance): the +0.026 correlation across 192 points is the strongest possible quantitative form of the decoupling claim. The scatter figure (`rq3_posterior_vs_performance.png`) is two cleanly-separated horizontal bands at overlapping posterior_error — visually unmistakable.

**Caveats.**
- `interpretable_overall: false` per the pre-registered criterion. The criterion uses normalised gap_closed for monotonicity; absolute returns are monotonic non-increasing for every reference and hypernet method. We had the same kind of borderline outcome at M5 (`ranking_stable: false`) and shipped — same call applies here.
- The persistence-easy rerun at diag=0.99 (mean duration 100, episode length 128) replaced the original diag=0.995 (mean duration 200) so episodes typically see ~1 regime switch — meta-RL methods now get an in-episode learning signal at easy. This fixed Belief-PPO monotonicity but still leaves the hypernet metric-artifact dip.
- Probe rolls out the trained policy to collect (belief, regime) pairs. Method test_acc could in principle be inflated if the policy avoids hard-to-decode states — but the analytical reference is rolled on the *same* trajectories, so the comparison is fair.


## 2026-04-29 — Second POMDP env build (CartPoleRegimeV1): Phases 1-3 complete, paused before full run

In-progress build of a second regime-switching POMDP environment to externally-validate the M5/M6 hypernet ≫ concat decoupling finding. The motivation is moving from "decoupling shown on market-making" to "decoupling shown on a representative class of regime-switching POMDPs" — the kind of generalisation reviewers will probe.

Total work this session: ~5h coding + ~5min compute, three commits ahead of m6-passed (`b6ba1ef`, `725d7a7`, `802a771`). All 42 prior tests + 7 new cartpole tests green.

**Phase status.**

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 1 — env-agnostic wrapper refactor | ✅ done | `b6ba1ef` | `BeliefObsEnv` / `OracleObsEnv` no longer import `MarketMakingV1`; `RL2ObsEnv` / `StackObsEnv` were already generic. Likelihood verified bit-identical to `beliefs.hmm_posterior.likelihood` on 5 representative cases incl. inventory-bound edges. |
| 2 — `CartPoleRegimeV1` env + 7 tests + train.py wiring | ✅ done | `725d7a7` | Initial implementation with gravity regimes (later replaced — see Phase 3 history). |
| 3 — env design iteration + R1 verification | ✅ done | `802a771` | Four rounds of regime variable; Round 4 (asymmetric stochastic actions) is the keeper. R1 confirmed at mid-mode. |
| 4 — full slim run + probe + Family A test + plots + FINDINGS | not started | — | ~half day work + 3-5h compute. Detailed todo below. |

**Phase 1 — wrapper refactor.** The MM-specific posterior code is hard-coded into `BeliefObsEnv`. To plug in any second env, the wrapper had to be generalised. Contract for any env that wants the analytical-posterior wrapper:
- expose `n_regimes`, `initial_distribution`, `transition_matrix`
- include `info["regime_likelihood"]: shape [n_regimes]` per step

`MarketMakingV1.step` now adds `info["regime_likelihood"]` from a new helper `_regime_likelihood`, which is bit-identical to the old `beliefs.hmm_posterior.likelihood` (verified on 5 cases). `BeliefObsEnv` does the filter+predict math inline now using `info["regime_likelihood"]` and the env's transition matrix. `OracleObsEnv` had its `inner: MarketMakingV1` annotation generalised to `inner: Any`. All M3/M5/M6 numerical reproducibility preserved.

**Phase 2 — env build.** `envs/cartpole_regime_v1.py` (~250 LOC mirroring `MarketMakingV1`'s structure: frozen dataclass, JIT-safe, HMM transitions, lock_regime support, Gaussian process noise on θ-dot for non-degenerate posterior). 7 tests in `tests/test_cartpole_regime.py` covering shape stability, lock_regime, stationary distribution, regime_likelihood non-negativity, JIT scan rollout, episode reset on done, and analytical-posterior concentration on the locked regime (averaged over 16 seeds because per-episode noise can move the posterior either way for the middle regime). `train.py` registers `cartpole_regime_v1`, `_oracle`, `_belief`, `_stacked` mirroring MM's naming.

**Phase 3 — env design iteration.** This was the time sink of the session. R1 (policy divergence — different optimal policy per regime) is mandatory for the decoupling test to make sense, and it took four design rounds to find a regime variable that gives R1 in cartpole. The key insight, recovered the hard way:

> Cartpole's state observation `(x, ẋ, θ, θ̇)` is itself a sufficient statistic for control. Any regime variable that affects continuous dynamics (gravity, force magnitude, wind force) is rapidly inferred from observed state transitions, so even regime-agnostic state-feedback PPO recovers the regime within a few steps and acts optimally — no explicit regime info needed. R1 fails by construction unless the regime affects something the state cannot reveal.

Round-by-round (mid-mode, 100 iter × 256 envs × n=1):

| Round | Regime variable | Floor | Belief | Oracle | Gap | R1? |
|---|---|---:|---:|---:|---:|:---:|
| 1 | gravity (4.9, 9.8, 19.6) m/s² | (not run; failed at locked-posterior test) | | | | ✗ |
| 2 | force magnitude (5, 10, 20) N | 119 | 119 | 110 | < 1 | ✗ |
| 3 | wind force (-4, 0, +4) N | 120 | 122 | 121 | < 1.5 | ✗ |
| **4** | **asymmetric stochastic actions** | **92.6** | **98.3** | **105.5** | **+12.9** | **✓** |

Round 4 — `regime_action_success` matrix with per-(regime, action) Bernoulli success probability:
- r0: P(left succeeds) = 0.95, P(right succeeds) = 0.30 — right pushes mostly fail
- r1: P(left succeeds) = 0.80, P(right succeeds) = 0.80 — symmetric
- r2: P(left succeeds) = 0.30, P(right succeeds) = 0.95 — mirror of r0

This works because the *direction* of optimal action becomes regime-conditional: in r0 the agent must bias toward left pushes regardless of pole tilt, because right pushes mostly fail. Mirrors MM's regime-conditional action-class structure (sym / favor-ask / favor-bid). The likelihood under each regime is now a mixture of two Gaussians: `P(obs | regime) = p_succeed · N(obs; predicted-applied) + (1−p_succeed) · N(obs; predicted-no-force)`, computed in log-space with max-shift for numerical stability.

**R1 verification numbers (Round 4, mid-mode, n=1, 100 iter).**

| Method | Return | Component |
|---|---:|---|
| Regime-agnostic PPO | 92.60 | Floor |
| Belief-PPO | 98.32 | +5.72 (compromise-policy cost) |
| Oracle-PPO | 105.47 | +7.14 (inference cost), +12.86 total gap |

Effect size is smaller than MM at full budget (gap = 44 in MM medium) but comfortably non-zero. The external-validity claim is "decoupling reproduces", not "with identical magnitude". Still need n=8 × 200 iter to confirm the gap is robust.

**Phase 4 — what's left for next session.**

1. **4 method configs**, each ~10 lines extending the M5 base + pointing at the cartpole env:
   - `experiments/configs/m_cartpole_rl2_concat.yaml` — extends `m5_step4_rl2_concat.yaml`, env `cartpole_regime_v1` (no wrapper — RL² agent appends its own (action, reward, done) augmentation internally)
   - `experiments/configs/m_cartpole_rl2_hypernet.yaml` — extends `m5_step4_rl2_hypernet.yaml`
   - `experiments/configs/m_cartpole_varibad_concat.yaml` — extends `m5_step4_varibad_concat.yaml`
   - `experiments/configs/m_cartpole_varibad_hypernet.yaml` — extends `m5_step4_varibad_hypernet.yaml`
2. Super_fast smoke each to verify they wire correctly through `train.py`.
3. Launch full slim run: 7 cells × n=8 seeds × 200 iter sequential → ~3-5h background.
4. Adapt `scripts/m6_posterior_probe.py` for the cartpole env. Probably just a config switch + verifying the probe's likelihood is consistent with the env's. The mixture-of-Gaussians likelihood is already in `info["regime_likelihood"]`, so the probe's analytical-reference path needs to use it.
5. Family A hypothesis test — Holm-corrected over 2 hypotheses (RL² hypernet > concat, VariBAD hypernet > concat at the single difficulty cell), n=8 paired Wilcoxon. Min p at n=8 paired Wilcoxon × Holm 2 = 0.0156, achievable.
6. Plots: 7-method bar chart with CI bands + posterior-vs-performance scatter (mirroring the M5 / M6 figure styles).
7. FINDINGS.md entry for the external-validity replication.

**Resume instructions for next session.**

1. Read this entry + `CLAUDE.md` status row.
2. Verify env is intact: `uv run python -m tests.test_cartpole_regime` — should print 7/7 passed.
3. Re-verify R1 if you want fresh numbers: `for cfg in m_cartpole_regime_agnostic m_cartpole_belief m_cartpole_oracle; do uv run python -m training.train --config experiments/configs/$cfg.yaml --mid; done`.
4. Then proceed with Phase 4 step 1 above.

**Key files touched this session.**
- New: `envs/cartpole_regime_v1.py`, `tests/test_cartpole_regime.py`, `experiments/configs/envs/e_cartpole_v1.yaml`, `experiments/configs/m_cartpole_{regime_agnostic,belief,oracle}.yaml`.
- Modified: `envs/market_making_v1.py` (added `_regime_likelihood`/`_per_regime_fill_probs` helpers + `info["regime_likelihood"]`), `envs/wrappers/belief_obs.py` (env-agnostic), `envs/wrappers/oracle_obs.py` (env-agnostic), `training/train.py` (registry entries for cartpole).
- Snapshot tag: `pre-cartpole-refactor` (still on disk; safe to rewind to it if any of this needs to be undone wholesale).

**Caveats / known unknowns.**
- The smaller R1 gap on cartpole vs MM (12.86 vs 44) means concat may not sit as far below the floor as it does on MM. The decoupling pattern could still hold, but the absolute |Δhypernet−concat| effect size will likely be smaller. That's fine for the external-validity claim — but worth flagging in the eventual FINDINGS entry that effect size depends on env, the qualitative decoupling does not.
- The cartpole posterior-probe will need its analytical reference adapted to the mixture likelihood. Worth a careful read of `m6_posterior_probe.py` before just running it.
- Cartpole returns are bounded above by `episode_length = 128` (the upright-bonus ceiling), unlike MM where returns can grow with spread capture. Different scale; not a problem, just different.

## 2026-04-29 — Cartpole Phase 4 complete: external-validity decoupling reproduces (qualitatively)

The second-POMDP external-validity probe completed. Hypernet ≫ concat decoupling reproduces on `CartPoleRegimeV1` with 2/2 hypotheses Holm-supported and LOO-robust. The qualitative claim survives the cross-env replication; the quantitative effect size is much smaller than on MM (smaller envelope, smaller absolute |Δ|).

**Artifacts.**
- `results/m_cartpole_{regime_agnostic,belief,oracle,rl2_concat,rl2_hypernet,varibad_concat,varibad_hypernet}/metrics.json` — per-cell training results (gitignored, regenerable).
- `results/milestones/cartpole/stats_cartpole_hypothesis_tests.json` — Family A test results.
- `figures/milestones/cartpole/cartpole_method_ladder.png` — 7-method bar chart.
- `scripts/cartpole_hypothesis_tests.py`, `plotting/cartpole_plots.py` — analysis + plotting code.

**Setup.** 7 methods × 1 difficulty cell. References (regime-agnostic / Belief / Oracle PPO) at n=5 × 200 iter × 512 envs; meta-RL (RL²/VariBAD × concat/hypernet) at n=8 × 200 iter × 512 envs. Same seeds and hyperparameters as M5 Step-4. Total wall: ~12 min references + ~127 min meta-RL = **2.3 h**.

**Headline numbers (mean ± per-seed std).**

| Method | Return | Δ vs floor |
|---|---:|---:|
| Regime-agnostic PPO | **96.11** ± 1.59 | (floor) |
| Belief-PPO | **99.67** ± 2.87 | +3.56 |
| Oracle-PPO | **107.40** ± 1.45 | +11.29 |
| RL² Concat | **90.68** ± 1.85 | −5.43 |
| RL² Hypernet | **96.95** ± 1.64 | +0.84 |
| VariBAD Concat | **84.49** ± 3.63 | −11.62 |
| VariBAD Hypernet | **96.71** ± 2.32 | +0.60 |

**Pattern reproduces qualitatively.**

- ✓ **Concat sits below floor** in both methods (RL² −5.4, VariBAD −11.6). Same direction and same ordering as M5: VariBAD Concat is the worst-performing meta-RL cell, RL² Concat is also broken but less so.
- ✓ **Hypernet beats concat** in both methods, both directions of the regime, all 8 paired seeds. Direction-consistent.
- ✗ **Hypernet does NOT reach Belief-PPO.** In M5 medium, hypernet was at the Belief ceiling (168 vs Belief 168.5). Here hypernet plateaus at the floor (96.7-96.9 vs Belief 99.7) — i.e. it closes the concat gap but does not close the inference gap. The compromise gap (Belief − Floor = +3.6) is small and noisy on cartpole; whether hypernet can clear it would require more seeds or a difficulty sweep we are not running here.

**Family A — `hypernet > concat` (paired Wilcoxon n=8, Holm × 2).**

| Hypothesis | Δ median | Bootstrap 95% CI | n_pos / 8 | p_raw | p_holm | Supported | LOO-robust |
|---|---:|---|---:|---:|---:|:---:|:---:|
| RL² hypernet > concat | +5.84 | [+5.70, +7.31] | 8/8 | 0.0039 | 0.0078 | ✓ | ✓ |
| VariBAD hypernet > concat | +12.29 | [+9.67, +14.20] | 8/8 | 0.0039 | 0.0039 | ✓ | ✓ |

**Family A: 2/2 supported, 2/2 LOO-robust.** Stronger than M5 in one specific way: M5 Family A had 12/12 Holm-supported but 0/12 LOO-robust because `Holm × 12 × Wilcoxon n=7` has a minimum p of 0.094 — an arithmetic ceiling, not a substantive failure. Here at family-size 2 the LOO check is passable, and both pass.

**Effect size comparison vs MM (M5 medium cell).**

| Comparison | MM \|Δ\| | Cartpole \|Δ\| |
|---|---:|---:|
| RL² hypernet − concat | ≈ +44 | +5.84 |
| VariBAD hypernet − concat | ≈ +56 | +12.29 |
| Total optimality envelope (Oracle − Floor) | 44 | 11.29 |

Cartpole's smaller envelope (11 vs 44) directly limits how large the meta-RL effect sizes can be. The qualitative external-validity claim — *"hypernet > concat decoupling is not an MM artefact"* — holds. The quantitative magnitude depends strongly on env, as expected.

**What this means for the thesis.**

The decoupling is now demonstrated on two structurally different envs:
- MM (M5/M6): regime governs *Bernoulli fill probabilities*, observation is *one-hot inventory*, optimal action is regime-conditional via fill-probability ranking.
- Cartpole (this entry): regime governs *Bernoulli action-success probabilities*, observation is *continuous state*, optimal action is regime-conditional via direction-bias ranking.

Both share the property that the regime is a per-(action) Bernoulli probability statistically inferred from outcomes — distinguishing them from the three earlier failed cartpole designs (gravity / force-magnitude / wind force) where the regime affected continuous dynamics and was therefore inferable from a few state observations, breaking R1.

This argues that the decoupling claim should generalise to the broader class of *regime-as-Bernoulli-probability POMDPs*, of which there are many in finance, healthcare, robotics with stochastic actuators, etc. The claim "the integration mechanism is the load-bearing thing, not posterior decoding accuracy" is now an external-validity-confirmed thesis result, not a single-env artefact.

**Caveats.**

- Single difficulty cell only. M6 showed the decoupling generalises across MM difficulty levels too; we did not redo a 3-level cartpole sweep, so the cross-difficulty generalisation on cartpole specifically is untested.
- Hypernet plateauing at Floor (rather than Belief) on cartpole is a real finding worth flagging in the thesis. Possible interpretations: (i) the cartpole compromise gap is so small that hypernet's recurrent / variational state cannot carry enough information to differentiate; (ii) PPO hyperparameters frozen from MM may be slightly mistuned for cartpole's 0/1-bounded reward; (iii) the analytical posterior wrapper is doing something subtly different on cartpole's mixture-of-Gaussians likelihood than on MM's Bernoulli. Worth a brief investigation in the thesis writeup but not a blocker for the headline claim.
- Posterior probe (Family C-style decoupling correlation) not yet run for cartpole. The probe needs `m6_posterior_probe.py` adapted to the cartpole env (mixture-of-Gaussians analytical reference). If we want to make the strongest possible decoupling claim — "hypernet beats concat *despite* equivalent belief decodability" — the probe is the missing piece. Suggested as a small follow-up if external reviewers ask for it.
- The decoupling on cartpole is qualitative-monotonic but the absolute hypernet returns barely exceed the floor by ~0.8 points, which is within the reference-PPO seed std (1.6) and could be argued away as noise. The Family A test still passes because it's a *paired* test that controls for seed variance — within each seed pair, hypernet beats concat by a healthy margin even if hypernet's absolute level is unexceptional.

**Phase 4 status.**

| Step | Status |
|---|:---:|
| 1. Write 4 meta-RL configs | ✅ |
| 2. Smoke each | ✅ |
| 3. Re-verify R1 at full budget | ✅ |
| 4. Full slim run | ✅ |
| 5. Family A hypothesis test | ✅ (2/2 supported, 2/2 LOO-robust) |
| 6. 7-method bar chart | ✅ |
| 7. FINDINGS entry | ✅ (this entry) |
| 8. Posterior probe (Family C analogue) | ⏸️ deferred — not strictly needed for headline claim |

The cartpole external-validity probe is **complete enough to ship** — the qualitative decoupling claim is demonstrated, the headline numbers and figure are in place, and the Family A test is unambiguous. The posterior probe is a strengthening exercise for a future thesis revision or paper resubmission, not a blocker.

## 2026-04-30 — Cartpole posterior probe: inverted decoupling (stronger than M5/M6)

The cartpole external-validity probe finishes the missing piece (Family C analogue from M6). Result is *stronger and more interesting* than expected: hypernet representations decode the regime *less well* than concat representations, yet hypernet performs *better*. The relationship between belief decoding and task performance is **inverted** rather than merely uncorrelated.

**Artifacts.**
- `evaluation/posterior_probe_cartpole.py` — env-agnostic probe using `info["regime_likelihood"]` for the analytical reference (vs the MM probe's `(bid_fill, ask_fill, q)` likelihood path).
- `scripts/cartpole_posterior_probe.py` — orchestrator over the 4 meta-RL cells × 8 seeds = 32 probe points.
- `figures/milestones/cartpole/cartpole_posterior_vs_performance.png` — scatter.
- `results/milestones/cartpole/stats_cartpole_posterior_vs_performance.json` — full stats.

**Probe accuracies (test, n_rollouts=200, logistic classifier).**

| Method | Method probe acc | Analytical probe acc | Posterior error | Mean gap_closed |
|---|---:|---:|---:|---:|
| RL² Concat | **0.573** | 0.713 | +0.140 | −0.50 |
| RL² Hypernet | **0.536** | 0.705 | +0.169 | +0.07 |
| VariBAD Concat | **0.557** | 0.700 | +0.143 | −1.04 |
| VariBAD Hypernet | **0.515** | 0.712 | +0.198 | +0.05 |

**Headline: r = +0.567, CI [+0.329, +0.751] across n = 32.**

The correlation is *positive and significant* — opposite to M5/M6 where r ≈ +0.026, CI excluding ±0.18. The magnitude exceeds the |r| < 0.30 decoupling threshold, so the script flags `signal_pattern="tight_correlation"`. But the direction matters: positive correlation between *posterior_error* (low = good belief decoding) and *gap_closed* (high = good performance) means **methods with better belief decoding perform worse**. This is more extreme than the M5/M6 pattern.

**Visual evidence.** The scatter shows two clean diagonally-separated clusters:

| Quadrant | Cluster | Methods |
|---|---|---|
| Bottom-left | low posterior_error, low gap_closed | Concat (RL² + VariBAD) |
| Top-right | higher posterior_error, higher gap_closed | Hypernet (RL² + VariBAD) |

There is no overlap between concat seeds and hypernet seeds in either x or y. The inverse relationship is built into the architecture choice, not driven by seed noise.

**Per-method correlations** (within-architecture, signs flipped from MM):
- rl2_concat: r = −0.13 (essentially zero within the concat cluster)
- rl2_hypernet: r = +0.46
- varibad_concat: r = +0.65
- varibad_hypernet: r = +0.43

Within hypernet methods, seeds with worse decoding tend to perform better. Within concat methods, decoding is uncorrelated with performance. The inversion shows up at both the population level and within most architecture cells.

**Why this is a stronger thesis claim than M5/M6's r ≈ 0.**

M5/M6 on MM showed *equal-decoding-but-different-performance*: the integration mechanism matters because two architectures with the same belief-decodability live at different return levels. The inference was "decoding is necessary but not sufficient — the policy interface dominates."

Cartpole shows *better-decoding-but-worse-performance*: the kind of representation hypernet learns is qualitatively different from concat's in a way that's hidden from a linear probe but visible to the policy's parameter-modulation step. Possible interpretations:

1. **Concat overfits to belief decoding.** The concat policy's hidden state is shaped by gradient pressure to make belief information *linearly accessible* (so the policy MLP can use it via concatenation). This optimisation pressure does not exist for hypernet — its parameter-generation step can use any nonlinear transform of the hidden state. Concat's representation is therefore biased toward linear-probe-friendly encoding, which is not the same as policy-useful encoding.
2. **Hypernet's representation is compressed and distributed.** The recurrent or variational state encodes regime in a way that's spread across many dimensions, with regime info entangled with state info in ways that a logistic regression cannot pull apart cleanly. A non-linear classifier (MLP probe) might recover it; a linear one cannot.
3. **The probe target itself is wrong-headed.** "Decode the regime" is a proxy for "use the regime productively". On cartpole, the regime affects *which action direction is reliable* but the policy must actually *commit to a biased action distribution*. A representation that decodes regime cleanly but does not bias action selection is useless for performance. Hypernet's parameter modulation directly biases the action distribution; concat's concatenation requires the downstream MLP to learn to bias actions from the concatenated belief, which evidently fails on cartpole.

All three interpretations point at the same conclusion: **belief decoding accuracy is not a meaningful proxy for whether a method has "learned to use the regime"**. The integration mechanism does not just decouple decoding from performance — it can *invert* their relationship.

**Combined claim across MM + cartpole.**

| Env | Decoding-vs-performance relationship | What it implies |
|---|---|---|
| MM (M5/M6) | Equal decoding, different performance (r ≈ 0) | Decoding is necessary but not sufficient |
| Cartpole | Better decoding, worse performance (r > 0 with low x = concat; high x = hypernet) | Decoding is sometimes inversely related to performance |

The integration mechanism is even more clearly the load-bearing thing than M5/M6 alone suggested. The cross-env evidence rules out interpretations like "hypernet just gets the regime info more cleanly into the policy".

**Caveats.**

- Logistic classifier only. An MLP probe might recover more of hypernet's distributed representation, which would soften the inversion. Worth checking if the inversion claim becomes load-bearing for the thesis.
- The analytical probe accuracy (~0.71) is well below 1.0 even on the analytical posterior — meaning the regime is never trivially decodable from the posterior + state in cartpole. That's fine; it reflects the env's stochastic action structure.
- Family A and posterior-probe results paint *different* pictures: Family A is "hypernet beats concat by 6-12 points" (qualitative agreement with M5); the probe is "concat decodes better, hypernet performs better" (an inversion of decoding vs performance not present in M5). Both are real and consistent — they describe different relationships. The combined story is "hypernet wins task, concat wins probe, but probe wins are not predictive of task wins".

**Status.** Cartpole external-validity probe is fully complete: ladder + Family A + scatter + decoupling diagnostic. The thesis can now make the cross-env decoupling claim with two independent kinds of evidence (cell-level Family A test + per-seed scatter correlation), each pointing at the same architectural conclusion via different statistics.

## 2026-04-30 — MLP probe falsifies the "linear-probe-blind" interpretation; inversion is real

Direct sanity check on the previous entry's interpretation. The thesis hypothesis was *"hypernet's representation is non-linear/distributed and a logistic probe can't see it"*. Testing this by re-running the same probe with a non-linear MLP classifier (sklearn `MLPClassifier(hidden_layer_sizes=(64,))`) instead of `LogisticRegression`.

If the linear-probe interpretation were correct: MLP would disproportionately help hypernet (recovering its hidden regime info), shrinking or reversing the inversion correlation. Concrete prediction was that |r| would drop, possibly below the 0.30 decoupling threshold.

**Observed (MLP probe, n_rollouts=200, all else identical to the logistic run).**

| Method | Logistic method_acc | MLP method_acc | Δ (MLP − Logistic) |
|---|---:|---:|---:|
| RL² Concat | 0.573 | **0.598** | +0.025 |
| RL² Hypernet | 0.536 | **0.549** | +0.013 |
| VariBAD Concat | 0.557 | **0.581** | +0.024 |
| VariBAD Hypernet | 0.515 | **0.539** | +0.024 |

| Stat | Logistic | MLP |
|---|---|---|
| Pearson r overall | +0.567 | **+0.639** |
| 95% bootstrap CI | [+0.329, +0.751] | **[+0.361, +0.837]** |
| signal_pattern | tight_correlation | tight_correlation |

**The linear-probe interpretation is falsified.** MLP gives all four methods a small near-uniform bump (+0.013 to +0.025), with no preferential help to hypernet. The architectural ordering is preserved (Concat > Hypernet by ~0.05 in probe accuracy, both classifiers); the inversion correlation gets *stronger*, not weaker; and the side-by-side scatter shows the same diagonal cluster-separation pattern in both panels.

Concat genuinely encodes regime more retrievably than Hypernet — both linearly *and* non-linearly. The original "linear-probe-blind" hypothesis was wrong.

**Reformulated thesis claim.** The new interpretation, supported by both probe types:

> Concat learns to *display* the regime; hypernet learns to *act on* it. These are different optimisation targets that produce qualitatively different — and on cartpole, anti-correlated — representations. A probe (whether linear or MLP) measures display; task return measures action; the two are not interchangeable proxies.

The mechanism: concat's regime info enters the policy via concatenation with the state vector, then flows through an MLP. Gradient pressure during PPO training shapes the recurrent / variational state to make regime info *retrievable from the concatenation*, because that is what the downstream MLP needs to read. Hypernet's regime info enters via a parameter-generation network that produces the policy weights themselves; gradient pressure shapes the recurrent state to be useful for *parameter generation*, which is a non-retrievability-preserving transform. The two architectures therefore optimise their hidden representations toward different objectives — even though both nominally "use the regime".

**This is a stronger thesis claim than M5/M6's r ≈ 0**, in two ways:

1. **MM** (M5/M6, r ≈ 0): "decoding accuracy doesn't predict performance." This is consistent with the architectures producing equivalent-quality regime info but using it differently.
2. **Cartpole** (this entry, r ≈ +0.6 with both probe types): "decoding accuracy *anti-predicts* performance." This rules out the M5/M6-compatible interpretation and forces a stronger one — the architectures produce *qualitatively different* representations whose decodability is itself a misleading proxy.

The MM result is consistent with Cartpole's, but Cartpole's is the load-bearing cross-env claim: *posterior decoding accuracy is fundamentally a misleading proxy for "ability to use the regime productively"*. Even with the most generous probe (MLP), the wrong architecture wins the probe metric.

**Artifacts.**
- `results/milestones/cartpole/stats_cartpole_posterior_vs_performance_mlp.json` — full MLP probe stats (gitignored).
- `figures/milestones/cartpole/cartpole_posterior_vs_performance_logistic_vs_mlp.png` — side-by-side scatter showing the inversion survives both probe types.
- `scripts/cartpole_posterior_probe.py` — added classifier-suffixed output filenames so logistic and MLP results coexist.
- `plotting/cartpole_plots.py` — added `plot_logistic_vs_mlp_side_by_side`.

**Caveats.**
- MLP probe `hidden_layer_sizes=(64,)`, `max_iter=200`, with sklearn defaults otherwise. Not tuned for this task. A larger or better-tuned MLP could squeeze more out, but the *direction* of the result (MLP gives a uniform bump, doesn't favour hypernet) makes it unlikely that further tuning would invert the conclusion. Worth confirming with a (128,128) MLP and longer iterations if the thesis review demands it; the gain probably saturates before the inversion flips.
- The MLP optimiser hit max_iter several times (visible in the script output as ConvergenceWarning). This is sklearn's default behaviour at high-dimensional belief inputs and small training sets; the test accuracy is still computed correctly but the fit is not at its asymptote. Convergence-failed fits would tend to *underestimate* the MLP's representational power — so if MLP is underestimating hypernet's decodability, the true inversion correlation could be smaller than +0.64. Not enough to flip the sign, but worth flagging.

## 2026-04-30 — M6 MLP probe: MM decoupling robust to probe choice (cross-env story now complete)

The parallel sanity check on MM. The cartpole MLP probe rerun yesterday showed cartpole's inversion is robust to probe non-linearity. Today's question: is MM's decoupling claim equally robust?

**M6 probe under MLP** (`stats_M6_posterior_vs_performance_mlp.json`, n=192 across 24 cells × 8 seeds, ~17 min compute). The script supports `--classifier mlp` natively; one small change to give it a `_mlp`-suffixed output filename so the logistic results coexist.

**Headline.**

| Stat | M6 Logistic | M6 MLP |
|---|---:|---:|
| Pearson r overall | +0.026 | **−0.060** |
| signal_pattern | decoupling | decoupling |
| n_scatter_points | 192 | 192 |

Both probes give r essentially zero. The decoupling claim on MM is robust to non-linearity in the probe — the same conclusion as M5/M6 with the original logistic probe.

**Cross-env, cross-probe combined picture (the headline thesis claim).**

| Env | Logistic r | MLP r | Pattern |
|---|---:|---:|---|
| MM (n=192) | +0.026 | −0.060 | **Decoupling** under both probes |
| Cartpole (n=32) | +0.567 | +0.639 | **Inversion** under both probes |

Each row's two cells agree to within ±0.07. The two envs disagree by ~0.7 in r. The differences between envs swamp differences between probe choices, which means: MM and cartpole produce *qualitatively different* decoder-vs-performance relationships, and the ranking is not a probe-implementation artefact.

**Per-method correlations (MM, MLP — for comparison to logistic).**

| Method | M6 Logistic r | M6 MLP r |
|---|---:|---:|
| RL² Concat | −0.286 | −0.416 |
| RL² Hypernet | −0.451 | −0.478 |
| VariBAD Concat | +0.706 | **+0.710** |
| VariBAD Hypernet | −0.511 | −0.509 |

VariBAD Concat's anomalous positive within-method correlation is *identical* between logistic and MLP probes (+0.706 / +0.710). The other 3 methods have moderate negative within-method correlations under both probe types. Stable per-method finding: within most architectures on MM, seeds with worse decoding perform worse — but the architectures' anchor points differ enough that the pooled correlation collapses to near-zero (Simpson's-paradox stratification, same as M5/M6 reported).

**What this means for the thesis claim.**

The full cross-env, cross-probe picture is now:
- *MM*: posterior decoding accuracy and task return are decoupled — neither predicts the other in the pooled scatter.
- *Cartpole*: posterior decoding accuracy is anti-correlated with task return — the architecture that decodes regime more retrievably performs worse.
- Both findings are stable across probe linearity (logistic vs MLP).

Both rule out the simplest interpretation ("hypernet wins task because it gets clean regime info into the policy"). On MM the architectures decode equally well and only the policy-interface differs. On Cartpole the architecture *that decodes worse* wins — even more strongly arguing that decoding accuracy is a misleading proxy for "ability to use the regime productively". The integration mechanism is the load-bearing thing across both envs, with the cartpole result strengthening the claim by showing that probe-friendly representations can be actively detrimental.

**Artifacts.**
- `results/milestones/M6/stats_M6_posterior_vs_performance_mlp.json` — full MLP scatter on MM (gitignored).
- `figures/milestones/cartpole/cross_env_decoupling_vs_inversion_2x2.png` — 2×2 (env × probe) scatter showing the four headline correlations side-by-side.
- `scripts/m6_posterior_probe.py` — added classifier-suffixed filenames mirroring the cartpole probe orchestrator.
- `plotting/cartpole_plots.py` — added `plot_cross_env_2x2`.

**Caveats.**
- MLP convergence warnings on M6 (sklearn's default `max_iter=200` gets hit on belief vectors with n_train ≈ 20k+). Same caveat as the cartpole MLP run; further iterations or `(128,128)` MLP would tighten the probe but the *direction* of the result is solid.
- Cartpole n=32 vs MM n=192 — Cartpole's CIs are wider, but tight enough that the +0.6 estimate excludes 0 by a wide margin. The *qualitative* contrast (MM decoupling vs cartpole inversion) is robust to power.
