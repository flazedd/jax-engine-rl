# Milestones overview

The dependency graph, milestone template, RQ mapping, project shape estimates, and "when is the thesis done" criteria.

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

All artifacts saved under `results/milestones/M{n}/` (JSON) and `figures/milestones/M{n}/` (plots). Regenerated via `uv run python -m scripts.make_milestone M{n}`.

---

