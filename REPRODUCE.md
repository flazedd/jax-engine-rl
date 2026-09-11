# Reproducing the thesis results

This guide reproduces the experiments reported in *Trading in the Dark: Belief-Conditioned
Meta-RL for Regime-Switching Market Making*.

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
