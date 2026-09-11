# Belief-Conditioned Meta-RL for Regime-Switching Market Making

Research code for the master's thesis *Trading in the Dark: Belief-Conditioned Meta-RL for
Regime-Switching Market Making*.

The experiments compare RL² and VariBAD under two ways of providing a learned belief to the
policy: concatenation and hypernetwork conditioning. They are evaluated in a regime-switching
market-making environment against agents with no regime belief, a short observation history, the
analytical posterior, or the true regime.

## Installation

The project requires Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run python -c "import jax; print(jax.devices())"
```

The lock file fixes the package versions used by the experiments.

## Check the installation

```bash
uv run pytest -q
uv run python -m scripts.run_matched_programme --dry-run
```

The first command runs the tests. The second prints the experimental plan without training any
agents. A synthetic run checks the complete analysis and plotting pipeline:

```bash
uv run python -m scripts.run_matched_programme --dummy
```

## Reproduce the thesis results

The complete experiment is started with one command and can be resumed after interruption:

```bash
uv run python -m scripts.run_matched_programme
```

This run is computationally expensive. See [REPRODUCE.md](REPRODUCE.md) for the execution order,
outputs, validation checks, and instructions for monitoring a run.

## Repository layout

| Path | Contents |
|---|---|
| `agents/` | PPO, RL², VariBAD, and reference agents |
| `beliefs/` | Analytical HMM posterior and oracle belief |
| `envs/` | Market-making environment, wrappers, and validation tasks |
| `experiments/configs/m5r_e9/` | Final configurations used for the thesis results |
| `training/` | Training loop, configuration loading, and PPO updates |
| `evaluation/` | Metrics, probes, diagnostics, and statistical procedures |
| `scripts/` | Reproduction driver and individual analysis commands |
| `plotting/` | Figure generation |
| `figures/` | Figures generated from the reported experiments |
| `tables/` | Generated LaTeX result tables |
| `tests/` | Automated tests |

The thesis contains the complete method definitions and experimental justification. This
repository focuses on executable code and reproduction instructions.
