# Trading in the Dark

**Research code, saved results, and trained agents for the master's thesis**

*Belief-Conditioned Meta-Reinforcement Learning for Regime-Switching Market Making*

The research asks whether an agent trades better because it learns more about a hidden market
regime, or because it uses that information more effectively. The experiments compare RL² and
VariBAD with concatenation and hypernetwork conditioning in a controlled market-making simulator.
Four reference agents provide different amounts of information about the regime.

## Start here

| What you want to do | Where to start |
| --- | --- |
| Check the reported results and regenerate charts and tables | Follow the quick start below |
| Evaluate the saved agents without training | [Replay saved agents](REPRODUCE.md#replay-saved-agents) |
| Train all agents again | [Repeat the experiments](REPRODUCE.md#repeat-the-experiments) |
| Find the configuration or data behind a result | [Map from thesis to repository](REPRODUCE.md#map-from-thesis-to-repository) |

**Everything needed for the saved-result checks is included in this repository.** No separate
companion archive, private directory, or external data account is required. The compressed data
include all 160 main checkpoints, results for 20 training seeds per condition, and supporting
validation experiments. [Data inventory and checksums](reproduction/data/README.md).

## Quick start: reproduce the saved results

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run these commands in a terminal:

```bash
git clone https://github.com/flazedd/jax-engine-rl.git
cd jax-engine-rl
uv sync --locked
uv run python -m scripts.reproduce_saved_results
```

This verifies and unpacks the saved data, recalculates the main statistical comparisons, and
regenerates the result figures and LaTeX tables. It does **not** train agents. It writes into a new
`reproduced/` directory and leaves the published inputs unchanged.

On success, the command prints `Saved result reproduction passed`. Open:

- `reproduced/verification.json` for the overall check and links to stage logs.
- `reproduced/figures/` for regenerated charts.
- `reproduced/tables/` for generated LaTeX tables.
- `reproduced/results/audits/thesis_contract.json` for the scientific consistency check.

Use `--output another-directory` to repeat the check without overwriting a previous run.
The [full reproduction guide](REPRODUCE.md) explains what is checked, what requires checkpoint
replay, and how to repeat training. Saved-data verification establishes consistency of the
reported measurements; replay and retraining provide additional checks.

## Check the installation

```bash
uv run python -c "import jax; print(jax.devices())"
uv run pytest -q
uv run python -m scripts.run_matched_programme --dry-run
```

The last command lists the experiment stages without running them. The locked installation
supports CPU execution. Full training is substantially more expensive than the saved-result
checks; do not start it just to view the results.

## Repository layout

| Path | Contents |
| --- | --- |
| [`REPRODUCE.md`](REPRODUCE.md) | Commands, expected outputs, and experimental scope |
| [`reproduction/data/`](reproduction/data/) | Compressed saved evidence, checkpoints, and SHA-256 manifest |
| [`experiments/configs/main/`](experiments/configs/main/) | Final configurations for the eight conditions |
| `agents/`, `training/` | Agent architectures and training code |
| `envs/`, `beliefs/`, `oracles/` | Simulator, analytical posterior, and dynamic programming references |
| `evaluation/` | Return measurements, probes, diagnostics, and statistics |
| `scripts/`, `plotting/` | Reproduction commands, analyses, tables, and figures |
| [`figures/`](figures/) | Charts available to browse without running code |
| `tests/` | Automated checks |

The training code was recorded at commit
[`220b319f367e`](https://github.com/flazedd/jax-engine-rl/commit/220b319f367e69a8f466c87d5346b1acf60574d6).
Use the current repository for reproduction: it includes the analysis and presentation used by
the final thesis. Record `git rev-parse HEAD` with any new reproduction so the exact code version
is identifiable.
