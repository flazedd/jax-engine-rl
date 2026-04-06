# Market Making RL with JAX

## Commands
- Install: `uv sync`
- Tests: `uv run pytest tests/ -v`
- Solve MDP: `uv run python scripts/solve_bellman.py [--fast]`
- Train PPO: `uv run python scripts/train_ppo.py [--fast]`
- Plot baselines: `uv run python scripts/plot_baselines.py`

## Overview
Analytical market making MDP with HMM regime switching (Avellaneda-Stoikov style).
The agent is a market maker quoting bid/ask prices in a limit order book with three
hidden regimes (Noise/Bull/Bear). Discrete state/action spaces enable exact Bellman
solutions — the RL agent's job is to close the gap between regime-blind and POMDP-optimal.

## Problem Structure
- **State**: (regime, inventory) — 3 regimes × 21 inventory levels = 63 states
- **Obs**: (inventory_normalized, last_bid_filled, last_ask_filled) — regime is hidden
- **Actions**: 3 discrete (bid_offset, ask_offset) pairs
  - a0=(2,2): tight symmetric — best spread capture (total offset 4)
  - a1=(1,9): aggressive long — strong directional (total offset 10)
  - a2=(9,1): aggressive short — strong directional (total offset 10)
- **Fill model**: p_fill(δ) = arrival_rate(regime) × exp(-κ(δ-1)), κ=0.3
- **Reward**: spread_capture + inventory×drift − φ×inventory²
- **HMM**: 98% self-transition (~50 step regime duration), 3.3× fill asymmetry

## Performance Levels (per-step reward)
- **Full-info**: ~4.09 (regime observed — theoretical ceiling)
- **POMDP**: ~3.36 (optimal belief tracking — achievable ceiling)
- **Regime-blind**: ~2.96 (always play a0 — floor)
- **RL target**: POMDP − blind ≈ 0.41/step (14% improvement, 82 cumulative over 200 steps)

## Architecture
```
lob_sim/
  __init__.py               — re-exports
  analytical_mdp.py         — MDP definition, fill model, VI solvers, simulation
  jax_env.py                — JAX-native functional environment (jit/vmap)

scripts/
  solve_bellman.py           — exact solvers + plots + JSON output
  plot_baselines.py          — horizontal bar chart from JSON
  train_ppo.py               — PPO feedforward training (Equinox + Optax)

tests/
  test_analytical_mdp.py     — 29 tests for MDP/solvers
  test_jax_env.py            — 18 tests for JAX environment
```

## JAX Environment (`lob_sim/jax_env.py`)
Fully functional, no mutable state. All randomness via explicit PRNGKeys.
- `EnvParams` — immutable params built from MDPConfig (fill probs, edges, drift, CDFs)
- `EnvState` — (regime, inventory, last_bid, last_ask, step_count)
- `env_reset(key, params)` → (state, obs)
- `env_step(key, state, action, params)` → (state, obs, reward, done, info)
- `rollout_episode(key, policy_fn, params, episode_length)` — jax.lax.scan
- `batch_rollout(key, policy_fn, params, n_envs)` — vmap over episodes

## PPO Training (`scripts/train_ppo.py`)
- Equinox ActorCritic (shared 64-unit trunk, actor + critic heads)
- Optax Adam with cosine LR schedule and gradient clipping
- GAE computed per-env before flattening (respects episode boundaries)
- 256 parallel envs, 64-step rollouts, 3 PPO epochs per iteration
- Feedforward PPO converges near regime-blind baseline (expected — can't track regime)

## RL Roadmap
1. ✅ PPO feedforward (baseline — matches regime-blind)
2. PPO + LSTM (can learn to infer regime from fill history)
3. RL² / VariBAD / HyperNetwork (meta-learning approaches)

## Tech Stack
- Python 3.12, `uv` for deps
- JAX for environment + training (jit, vmap, lax.scan)
- Equinox for neural networks
- Optax for optimization
- NumPy for analytical MDP solver (no JAX dependency for core math)
- CPU by default (macOS), GPU-ready (set JAX_PLATFORM_NAME=cuda)

## Outputs
- `results/bellman_solution.json` — policy, Q-values, value function, config
- `results/ppo_metrics.json` — training curves (eval rewards, losses)
- `plots/bellman_solution.png` — 6-panel MDP visualization
- `plots/reward_trajectories.png` — cumulative reward comparison
