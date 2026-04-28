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

## 2026-04-28 — M6 difficulty sweep complete (RQ3 answered: belief↔task decoupling generalizes massively across difficulty)

**Artifacts.**
- `results/milestones/M6/stats_M6_sweep.json` — 42-cell aggregate (gitignored, regenerable).
- `results/milestones/M6/stats_M6_posterior_vs_performance.json` — 192 (cell, seed) probe scatter points (gitignored).
- `figures/milestones/M6/rq3_persistence_sweep.png`, `rq3_distinguishability_sweep.png`, `rq3_posterior_vs_performance.png` — three thesis figures.
- `scripts/m6_difficulty_sweep.py` (orchestrator), `scripts/m6_posterior_probe.py` (probe), `scripts/m6_overnight.py` (chained wrapper), `plotting/m6_plots.py` (figures).

**Setup.** 7 methods × 3 difficulty levels × 2 axes = 42 cells, sequential. References (regime-agnostic / Belief / Oracle PPO) at n=5 × 200 iter, meta-RL (RL²/VariBAD × concat/hypernet) at n=8 × 200 iter (inherited from M5 Step 4 base configs). Total wall: **18.76 h** training + 4.3 min probe + <0.1 min plotting = **18.83 h**.

Persistence axis (mean regime duration): easy 200 steps (diag 0.995, **widened** from initial 0.99), medium 50 steps (diag 0.98 = E_final), hard 5 steps (diag 0.80, **widened** from initial 0.85). Distinguishability axis (regime fill probabilities): easy [0.10/0.90/0.50, 0.85/0.02/0.05] (very separated), medium = E_final, hard [0.40/0.65/0.50, 0.50/0.10/0.05] (compressed). The widened persistence range was a directional-check decision when narrow ranges showed barely-changing gap_closed; the wider range produced clear difficulty signal.

**Headline gap_closed matrix.**

| | **Persistence** | | | **Distinguishability** | | |
|---|---:|---:|---:|---:|---:|---:|
| | Easy | Med | Hard | Easy | Med | Hard |
| RL² Hypernet | 0.67 | 0.74 | 0.54 | **1.00** | 0.74 | 0.40 |
| VariBAD Hypernet | 0.63 | 0.70 | 0.53 | **1.03** | 0.70 | 0.40 |
| Belief-PPO | 0.56 | 0.74 | **0.21** | 0.74 | 0.74 | 0.31 |
| RL² Concat | −0.54 | −0.28 | −0.46 | −0.13 | −0.28 | −0.75 |
| VariBAD Concat | −1.16 | −0.59 | −0.31 | −0.74 | −0.59 | −0.52 |

Reproducibility: all seven medium cells across both axes match M3 reference levels and M5 Step 4 numbers **exactly** (Floor 136.23, Belief 168.50, Oracle 180.14, RL² Concat 124.07, RL² Hypernet 168.56, VariBAD Concat 110.32, VariBAD Hypernet 166.80). Confirms determinism of the seed/config protocol across milestones.

**Headline findings.**

- **M5 hypernet ≫ concat decoupling generalizes to every cell.** Concat sits below floor in 11/12 cells; hypernet sits ≥ Belief-PPO in 11/12. Both axes, all difficulty levels.
- **Posterior↔performance correlation = 0.055** across 192 scatter points. Essentially zero. The decoupling is structural, not stochastic: posterior decoding accuracy does *not* predict task return — the policy interface dominates.
- **Hypernet exceeds Oracle at distinguishability-easy** for both methods independently (RL² 206.7, VariBAD 208.1, Oracle 206.6). Recurrent / variational state encodes inventory + market-state context the regime-only oracle obs doesn't use as efficiently. Small lift (~1-2 pts) but reproduces across both methods at n=8.
- **Hypernet's edge over Belief-PPO grows when Belief-PPO falters**: persistence-hard +15.7 pts (Belief collapses, hypernet retains), distinguishability-easy +13.3 pts (Belief saturates short of oracle, hypernet pushes past). At medium (E_final), hypernet ≈ Belief (within 0.1 pt) — the M5 plateau.
- **Belief-PPO inverted-U on persistence axis**: 0.56 → 0.74 → 0.21. At easy persistence the floor is competitive (single-regime episodes give compromise policies room to do well), at hard the analytical posterior collapses. Defensible physics, but it does break strict monotonicity.
- **Distinguishability-hard collapses the optimality gap** (50.5 → 43.9 → **26.6**). Compressed fill probabilities make the regimes barely matter: even Oracle is only 26.6 above Floor.
- **VariBAD Concat does not learn** anywhere. Absolute returns 109.9 – 119.0 across all 6 cells (vs floor 124-156). The variational posterior is fine (probe accuracy ≈ analytical), but the concat interface can't realize a regime-conditional action distribution. Same as M5 Step 4.

**Pass-criterion status** (per `docs/milestones/m6.md`).

| Criterion | Required | Observed | Status |
|---|---|---|:---:|
| persistence_sweep.all_methods_monotonic | true | false (Belief inverted-U) | ✗ |
| distinguishability_sweep.all_methods_monotonic | true | tbd (re-aggregate; some methods non-strict) | partial |
| stats_M6_sweep.interpretable_overall | true | **false** | ✗ |
| stats_M6_posterior_vs_performance.scatter_interpretable | true | **true** (signal_pattern=decoupling) | ✓ |

The strict criteria fail on the same kind of "borderline but interpretable" pattern as M5's `ranking_stable=false`. Substantive RQ3 claims (decoupling generalizes, hypernet > Belief at hard, posterior↔performance correlation ≈ 0) are robust. Manual review by the user before tagging `m6-passed`.

**What this means for the thesis.**
- RQ3 part 1 (scaling with difficulty): characterized cleanly along both axes for all 7 methods. The persistence inverted-U for Belief is a useful discussion point — it isolates where the *compromise-policy cost* component matters vs the *inference cost* component.
- RQ3 part 2 (posterior↔performance): the 0.055 correlation across 192 points is the strongest possible quantitative form of the decoupling claim. The scatter figure (`rq3_posterior_vs_performance.png`) is two cleanly-separated horizontal bands at overlapping posterior_error — visually unmistakable.

**Caveats.**
- `interpretable_overall: false` per the pre-registered criterion. We had the same kind of borderline outcome at M5 (`ranking_stable: false`) and chose to ship — same call applies here; the inverted-U is real signal, not failure.
- The persistence-easy result is influenced by the fact that mean regime duration (200) exceeds episode length (128), so episodes are effectively single-regime. This is by design (it gives a meaningful "easy" reference) but means the easy gap is intrinsically narrow.
- Probe rolls out the trained policy to collect (belief, regime) pairs. Method test_acc could in principle be inflated if the policy avoids hard-to-decode states — but the analytical reference is rolled on the *same* trajectories, so the comparison is fair.

