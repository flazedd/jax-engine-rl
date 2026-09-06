# Belief-Conditioned Meta-RL for Regime-Switching Market Making

This repository contains the code and experiment configurations for the accompanying master's
thesis. For the complete, executable reproduction instructions, see
[REPRODUCE.md](REPRODUCE.md).

Quick start:

```bash
git checkout <the archived commit recorded with the thesis results>
uv sync --locked
uv run python -m scripts.run_matched_programme --dry-run
```

The full run trains the agents, evaluates them on fresh episodes, runs the statistical analyses,
and regenerates the figures. It requires substantial compute; use the dummy run in
`REPRODUCE.md` to check the full pipeline first.
