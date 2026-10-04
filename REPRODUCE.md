# Reproducing the thesis results

This guide accompanies *Trading in the Dark*.
It distinguishes verification of saved results, replay of frozen policies, and
full retraining. None is a substitute for the others.

## Companion archive and source identity

The delivered `thesis-reproducibility-2026-10-04.tar.gz` contains `code/`,
`thesis/`, `MANIFEST.json`, and `ENVIRONMENT.json`. Its adjacent `.sha256` file
verifies the download. After extraction, verify every archived file:

```bash
cd thesis-reproducibility-2026-10-04
python3 code/scripts/verify_reproduction_bundle.py .
cd code
```

The manifest pins the source snapshot by SHA-256. It also pins all YAML
configurations, the lock file, analysis inputs, and the 160 main checkpoints.
The source snapshot is complete without Git history. This is a local companion
archive delivered with the thesis; no public release URL or DOI is claimed.

Training was recorded at `220b319f367e69a8f466c87d5346b1acf60574d6`. The original
three-seed toy-validation baseline is explicitly pinned in
`reproduction/inputs/method_ranking.json`, including all nine sets of per-seed
final returns. The bundle additionally preserves its original training artifacts.
The current factorial validation is a separate experiment with eight seeds.

`ENVIRONMENT.json` describes the verification machine and installed libraries.
Original training hardware details that were not recorded are marked unknown.
`TRAINING_EVIDENCE.json` records the saved learning curves and checkpoint hashes
for the eight main conditions.

## Install the locked environment

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -c "import platform, jax; print(platform.platform()); print(jax.devices())"
uv run pytest -q
```

The normal pytest suite includes environment, belief-filter, and observation
invariants, including the final e9 environment, plus the regression tests for
cache identity, required artifacts, comparison schemas, and dependency tracking.
Training-based validation is separate and runs in the full programme.

## Verify the saved scientific results

These commands use the bundle's saved data and do not train agents or refit probes:

```bash
uv run python -m scripts.restore_validation_baselines
uv run python -m scripts.m5r_seed_block_sensitivity
uv run python -m scripts.thesis_contract --strict --thesis-root ../thesis
```

The exploratory reference and short-history intervals can be regenerated with
`uv run python -m scripts.m5r_exploratory_baseline_seed_pairs`. The supplemental
posterior checks use the saved policies and new simulated episodes:

```bash
uv run python -m scripts.m5r_supplemental_belief_checks --n-rollouts 500
uv run python -m scripts.plot_m5r_supplemental_belief_checks --output ../thesis/figures/appendix/m5r_direct_posterior_accuracy.png
```

Both analyses were added after reviewing the main results. They do not alter the
planned comparison families or saved checkpoints. The second command compares
VariBAD readouts of its posterior mean and its complete mean and standard deviation
input on the same held-out episodes; it also scores the analytical posterior
directly on each method's own trajectories.

The sensitivity script recomputes all 18 original mean comparisons (including
bootstrap intervals, permutation p-values and six Holm adjustments) and checks
them against the saved results before calculating the seed-block analysis.
Training uses shared seeds and rollout-key schedules across conditions. The
original working-independence analysis is retained explicitly. The added paired
bootstrap/sign-flip analysis preserves seed blocks; sign flips require symmetric
paired differences or within-pair exchangeability under the null. Correlation
resamples preserve all four variants within a seed and recenter each draw.
This post-analysis check changes none of the 18 significance decisions.

The strict contract requires evaluation, both probes, primary diagnostics,
confusion data, all six comparison families, all eight training budgets and
20-by-1500 learning curves. Missing or malformed required data fail. The optional
history-substitution analyses are checked if present. With `--thesis-root`, it
also checks all seven generated tables and the manually typeset result rows.

## Replay frozen checkpoints

```bash
uv run python -m scripts.verify_checkpoint_replay
uv run python -m scripts.verify_checkpoint_replay --all-seeds
```

The first command replays seed 0 from all eight conditions (4096 episodes); the
second replays all 160 checkpoints (81920 episodes). Each uses the original
512 episodes, 128 steps, method-specific evaluation keys, and compares with the
saved per-seed result. It writes a separate verification file. Numerical identity
across hardware is not promised; discrepancies are reported with their size.
The accuracy and KL figures in Chapter 5 pair each method's accuracy and KL with reference probes fitted on the same episodes.
Current probe outputs retain both analytical reference curves for every training seed. For older
outputs that retain only the reference mean curve, recover the individual curves before plotting:

```bash
uv run python -m scripts.m5r_accuracy_references
```

This replays the saved checkpoints, refits the analytical reference classifiers, checks their
accuracies and KL values against the saved results, and writes `results/analysis/m5r_accuracy_references.json`.
The figures subtract these reference curves within each run before smoothing and bootstrapping.

The probes and locked-regime diagnostics can be regenerated with their corresponding
`m5r_posterior_probe`, `m5r_action_distributions`, and `m5r_belief_swap` scripts.

## Preflight and full retraining

Inspect the dependency-complete plan and run the deterministic rendering smoke test:

```bash
uv run python -m scripts.run_matched_programme --dry-run
uv run python -m scripts.run_matched_programme --dummy
uv run python -m scripts.verify_clean_preflight
```

The synthetic preflight uses checked-in schema fixtures, requires no archived
results, and executes the real table and figure renderers. It does not validate
training or statistically coherent synthetic results. It unconditionally redirects
all output roots into a new temporary directory, prints that directory, marks
outputs synthetic, and never falls back to real JSON. Inherited thesis-publication
paths are ignored. The temporary output can be removed after inspection.

Run training in a **new directory**, preserving the delivered evidence:

```bash
export THESIS_RESULTS_ROOT="$PWD/results-retrained"
export THESIS_PROJECT_FIGS="$PWD/figures-retrained"
export THESIS_FIG_ROOT="$PWD/retrained-thesis/figures"
uv run python -m scripts.run_matched_programme
```

The driver gates main training on configuration and foundation validation, trains
the four references, checks reference ordering, trains the four meta-RL variants,
and then runs evaluation, both probes, confusion collection, diagnostics, all six
comparison families, seed-block sensitivity, tables and figures. The strict
scientific contract runs after real outputs exist. Failed prerequisites block
all dependants. Selecting a phase includes its transitive prerequisites.

New runs record the complete resolved configuration, relevant source/lock hashes,
and a checksum for each checkpoint. Resume validates these before changing any
metadata. An incompatible or legacy cache is rejected, not relabelled or silently
retrained in place. Analysis-stage receipts hash code/configuration, dependency
outputs and every required output; deleting a secondary output or changing an
input invalidates the receipt. `--force STAGE` reruns a named analysis stage.

Results live under `foundations/`, `medium/`, and `analysis/` beneath
`THESIS_RESULTS_ROOT`. Progress is in `matched_programme_status.json`. Setting
`THESIS_FIG_ROOT` publishes figures there and tables in its sibling `tables/`.
A fresh retraining is a new replication; do not automatically replace published
numbers without reviewing its outputs.

## Regenerate the corrected thesis presentation

From the extracted `code/` directory:

```bash
export THESIS_FIG_ROOT="$(cd ../thesis && pwd)/figures"
uv run python -m plotting.m5r_plots
uv run python -m plotting.reference_levels
uv run python -m plotting.m5r_action_inventory_heatmap
uv run python -m plotting.m4_plots
uv run python -m scripts.make_tables
uv run python -m scripts.thesis_contract --strict --thesis-root ../thesis
cd ../thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The five environment-validation figures are included in the bundle with their
saved validation inputs. A full recomputation of those figures is part of
`python -m scripts.env_validation_final` and includes its validation training.
The frozen evaluation and probe measurements are included in the bundle. The
statistical analyses retain their stated assumptions. Architectural matching does
not isolate VariBAD's conditioning route from changes in trunk sharing and hidden
widths.
