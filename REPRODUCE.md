# Reproducing the thesis results

This guide reproduces the experiments reported in *Trading in the Dark: Belief-Conditioned
Meta-Reinforcement Learning for Regime-Switching Market Making*.

## 1. Use the archived version

Check out the release or commit archived with the thesis. Record the commit identifier and the
hardware and software details of the reproduction run.

## 2. Create the software environment

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --locked
uv run python -c "import platform, jax; print(platform.platform()); print(jax.devices())"
```

The project requires Python 3.12. `uv sync --locked` installs the versions recorded in `uv.lock`
without updating them.

## 3. Check the code and pipeline

Run the automated tests:

```bash
uv run pytest -q
```

Inspect the complete plan without running it:

```bash
uv run python -m scripts.run_matched_programme --dry-run
```

Then run the synthetic version of the pipeline:

```bash
uv run python -m scripts.run_matched_programme --dummy
```

The synthetic run checks the execution order, analysis code, tables, and figures without training
the full agents. Its outputs are marked as synthetic and are ignored by Git.

## 4. Run the experiments

```bash
uv run python -m scripts.run_matched_programme
```

The driver performs the following steps:

1. Check the model inputs, parameter counts, optimiser settings, and training budget.
2. Validate the market-making environment and the RL² and VariBAD implementations.
3. Train regime-agnostic PPO, stacked-observation PPO, Belief-PPO, and Oracle-PPO.
4. Check the two reference differences defined in the thesis.
5. Train RL² and VariBAD with concatenation and hypernetwork conditioning.
6. Evaluate the frozen policies on fresh episodes.
7. Run the belief probes, behavioural diagnostics, statistical comparisons, and figure generation.

Completed stages are detected from their output files, so rerunning the command resumes an
interrupted experiment. To inspect progress in another terminal, run:

```bash
uv run python -m scripts.status
```

The status file is `results/matched_programme_status.json`.

## 5. Check the generated results

After the programme finishes, run:

```bash
uv run python -m scripts.thesis_contract --strict
```

This command checks that the required outputs exist and that their settings agree with
`evaluation/protocol.py`.

When a thesis checkout is available, also check its manually typeset result tables:

```bash
uv run python -m scripts.thesis_contract --strict --thesis-root /path/to/thesis-checkout
```

## Outputs

- `results/foundations/` contains environment and implementation validation.
- `results/medium/` contains checkpoints and training results for the reported experiment.
- `results/analysis/` contains fresh evaluation returns, probe results, diagnostics, and statistical
  comparisons.
- `figures/` contains the generated figures.
- `tables/` contains the generated LaTeX result tables.

By default, every output stays inside this repository. To also publish figures and tables into a
separate thesis checkout, set `THESIS_FIG_ROOT` to that checkout's `figures` directory before
running the programme.

Each training directory records the resolved configuration and results for every seed. Exact
floating-point values may differ across hardware, but the same archived code, configuration, and
random seeds reproduce the stated experimental procedure.

## Corrections after the September 2026 consistency audit

The original analysis revision `3af71feedf24` predates these corrections. Use the
updated code for the revised thesis. Training checkpoints and fresh evaluation
returns are unchanged. The audit changed the probe/return join, correlation
bootstrap, RL² substitution intervention, diagnostic presentation, and supporting
text. The exact corrected files and their hashes are recorded in
`results/analysis/thesis_correction_manifest.json`.

The evaluator now records `seeds` beside `per_seed_evaluation_return`. Probe scores
must be joined to returns by **experiment and seed**, never by array position.
The legacy evaluator sorted checkpoint filenames lexically, whereas archived
training-return arrays used numeric seed order. The explicit migration below
recovers the former ordering from the complete set of archived checkpoints:

```bash
uv run python -m scripts.m5r_refresh_probe_returns --migrate-legacy-seeds
```

Use `scripts.m5r_refresh_probe_returns` without the flag for subsequent refreshes.
This reuses the saved probe fits and joins them to fresh frozen-policy evaluation
returns. It also refreshes cached correlations. It does not retrain agents or
refit classifiers. Both pooled and within-variant correlation intervals resample
independent runs within each variant, preserving each run's error/return pair.
Within-variant means are recomputed inside each resample.

The RL² collector records the state after the current encoder update. The corrected
substitution diagnostic evaluates the actor directly from that state, without a
second GRU update. Rerun it at the full 64 episodes per regime and all 20 seeds:

```bash
uv run python -m scripts.m5r_belief_swap
uv run python -m scripts.m5r_diagnostic_tests
uv run python -m scripts.m5r_hypothesis_tests
```

If optional history-channel substitution files exist, regenerate them using
`--swap-history` and `--swap-history --hold-belief-fixed` before running the
diagnostic tests. Stale substitution files are rejected rather than mixed with
the corrected intervention. The primary score averages covered inventory levels
without weighting. The exact-policy line is omitted from this figure because it
used a different time/action aggregation.

Regenerate the plots and tables, setting `THESIS_FIG_ROOT` to the thesis figure
directory if publishing to a separate checkout:

```bash
uv run python -m plotting.m5r_plots
uv run python -m scripts.make_tables
uv run pytest -q
uv run python -m scripts.thesis_contract --strict --thesis-root /path/to/thesis
uv run python -m scripts.write_thesis_correction_manifest
```

The manually typeset diagnostic and correlation rows must be updated to the new
analysis outputs; the final command checks them. The revised correlations retain
a modest negative association within variants. The previous claim that the
association disappears is withdrawn. Probe comparisons concern the tested
representation (VariBAD's posterior mean, not its full mean/variance input), and
each method supplies its own trajectories.
