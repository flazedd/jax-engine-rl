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
- **State**: (regime, inventory) — 3 regimes × 21 inventory levels = 63 states
- **Actions**: 3 discrete (bid_offset, ask_offset) pairs
  - a0=(2,2): tight symmetric — best spread capture (total offset 4)
  - a1=(1,7): aggressive long — strong directional (total offset 8)
  - a2=(7,1): aggressive short — strong directional (total offset 8)
- **Fill model**: p_fill(δ) = arrival_rate(regime) × exp(-κ(δ-1)), κ=0.3
- **Reward**: spread_capture + inventory×drift − φ×inventory²
- **HMM**: Noise/Bull/Bear with persistent transitions (90-92% self-transition)
- **Key calibration**: a0 has lower total offset → genuinely better spread capture, creating a wide "a0 zone" in Noise. Subtle fill asymmetry (~1.7x ratio) makes regime hard to observe, while large drift (±0.40) makes wrong actions costly → ~30% value-of-information gap

## Solvers
1. **Full-info VI**: Regime observed → 63-state MDP, solves in ~2ms
2. **POMDP belief-state VI**: Regime hidden → discretized simplex, ~3s

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
