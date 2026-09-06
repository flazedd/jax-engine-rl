# Reproducing the thesis results

This file is the executable guide for reproducing the experiments in *Trading in the Dark:
Belief-Conditioned Meta-RL for Regime-Switching Market Making*.

## What you need

- The archived repository commit recorded with the thesis results.
- [uv](https://docs.astral.sh/uv/), which installs the required Python 3.12 environment.
- A machine with enough memory for JAX training. Record the operating system, accelerator, driver,
  and numerical-library versions used for the run.

The full programme is computationally expensive. A fresh run should reproduce the experimental
procedure and the qualitative conclusions, but exact floating-point values can vary across
hardware.

## Set up the environment

Run these commands from the repository root:

```bash
git checkout <archived-commit>
uv sync --locked
uv run python -c "import jax; print(jax.devices())"
```

`uv sync --locked` creates the environment from `uv.lock` without changing dependency versions.

## Check the pipeline before training

First run the synthetic version of the complete pipeline. It checks that the analysis and figure
generation steps work without spending the full training budget:

```bash
uv run python -m scripts.run_matched_programme --dummy
uv run python -m scripts.thesis_contract
```

To inspect the full training plan and its time estimate without running it:

```bash
uv run python -m scripts.run_matched_programme --dry-run
```

## Run the full programme

```bash
uv run python -m scripts.run_matched_programme
uv run python -m scripts.thesis_contract --strict
```

The programme runs the steps in this order:

1. Check that the methods use the specified inputs, network sizes, optimiser settings, and training budget.
2. Validate each market-making setting against R1--R4 and run the standard benchmark checks.
3. Train the four reference agents: Regime-agnostic PPO, Stacked-observation PPO, Belief-PPO, and Oracle-PPO.
4. Check that Belief-PPO exceeds Regime-agnostic PPO and Oracle-PPO exceeds Belief-PPO before training the four belief-conditioned variants.
5. Train the variants, evaluate every trained policy on fresh episodes, run the belief and behavioural diagnostics, apply the statistical tests, and regenerate the figures.

The driver writes progress to `results/analysis/programme_status.json`. Check it with:

```bash
uv run python -m scripts.status
```

## Outputs

- `results/foundations/` contains environment and implementation-validation results.
- `results/medium/` contains the trained reference and variant runs.
- `results/analysis/` contains evaluation returns, diagnostic results, and statistical tables.
- `figures/` contains the generated figures. The thesis uses the matching copies in the thesis repository.

Each run writes its resolved configuration, commit identifier, and per-seed results alongside its outputs.
