# Analytical Market Making MDP

## Commands
- Install: `uv sync`
- Tests: `uv run pytest tests/ -v`
- Solve: `uv run python scripts/solve_bellman.py [--fast]`

## Overview
Analytical market making MDP with HMM regime switching (Avellaneda-Stoikov style).
Discrete state/action spaces → exact Bellman equation solutions via value iteration.
No simulation or Monte Carlo needed — all transition/reward tables are closed-form.

## Problem Structure
- **State**: (regime, inventory) — 3 regimes × 31 inventory levels = 93 states
- **Actions**: 3 discrete (bid_offset, ask_offset) pairs with equal total offset (6 ticks)
  - a0=(3,3): symmetric — spread capture
  - a1=(1,5): tight bid, wide ask — go long
  - a2=(5,1): wide bid, tight ask — go short
- **Fill model**: p_fill(δ) = arrival_rate(regime) × exp(-κ(δ-1)), κ=0.3
- **Reward**: spread_capture + inventory×drift − φ×inventory²
- **HMM**: Noise/Bull/Bear with persistent transitions (90-92% self-transition)
- **Key calibration**: Subtle fill asymmetry (~1.7x ratio) makes regime hard to observe from fills alone, while large drift (±0.40) makes playing the wrong action very costly → ~23% value-of-information gap

## Solvers
1. **Full-info VI**: Regime observed → 93-state MDP, solves in ~3ms
2. **POMDP belief-state VI**: Regime hidden → discretized simplex, ~8s

## Architecture
```
lob_sim/__init__.py          — re-exports
lob_sim/analytical_mdp.py    — MDP definition, fill model, VI solvers, helpers
scripts/solve_bellman.py      — solve + plot + JSON output
tests/test_analytical_mdp.py  — 29 tests
```

## Outputs
- `results/bellman_solution.json` — policy, Q-values, value function, optimality certificate, config
- `plots/bellman_solution.png` — 6-panel visualization

## Global Constraints
- Python 3.12, `uv`
- Pure numpy for the analytical MDP (no JAX dependency for the core solver)
- `typing.NamedTuple` for config/data structures
