"""Inventory Bandit with Regime Switching + Active Sensing.

Bridge between contextual bandit and market making. Inventory accumulates
across steps, and the hidden task can SWITCH mid-episode — forcing
continuous regime inference, not just one-shot identification.

5 tasks with different risk/reward profiles:
  Task 0: Low risk, bullish drift   — accumulate long, gentle penalty
  Task 1: Low risk, bearish drift   — accumulate short, gentle penalty
  Task 2: High risk, bullish drift  — must identify quickly or get hurt
  Task 3: High risk, bearish drift  — must identify quickly or get hurt
  Task 4: High risk, no drift       — best to stay flat (trap task)

Regime switching: at each step, with probability `switch_prob`, the task
changes to a uniformly random new task. This means:
  - Agents must continuously track belief, not just identify once
  - After a switch, the agent's position might be wrong (e.g., long in
    a bear regime) — it must detect the change and unwind
  - This mirrors real markets where regimes shift without warning

Actions:
  0: Buy   → inventory += 1, earn small spread
  1: Sell  → inventory -= 1, earn small spread
  2: Hold  → inventory unchanged, no spread
  3: Abstain → inventory unchanged, earn ABSTAIN_PAYOFF

Active sensing: acting in the task's "correct" direction gives noisy
next obs, wrong direction gives clear obs.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .bandit_context import SIDE_MEANS


# Environment constants
N_TASKS = 5
Q_MAX = 3                    # inventory in [-3, 3], 7 levels
ABSTAIN_PAYOFF = 0.02        # small safe reward — below optimal, above random
SPREAD_PAYOFF = 0.01         # small reward for trading
INVENTORY_PENALTY = 0.02     # base inventory penalty coefficient
BOUNDARY_PENALTY = 0.2       # penalty at |q| == Q_MAX
SWITCH_PROB = 0.05           # P(regime switch) per step — ~1 switch per episode

# Per-task parameters
TASK_DRIFT = jnp.array([+0.10, -0.10, +0.15, -0.15, 0.0])
TASK_RISK = jnp.array([0.3, 0.3, 1.5, 1.5, 2.0])
TASK_BEST_DIR = jnp.array([+1, -1, +1, -1, 0])

# Action-dependent observation noise
SIGMA_EXPLOIT = 4.0
SIGMA_EXPLORE = 1.5
SIGMA_DEFAULT = 2.5


class InventoryBanditParams(NamedTuple):
    """Immutable parameters."""
    n_tasks: int = N_TASKS
    n_actions: int = 4       # buy, sell, hold, abstain
    t_episode: int = 20
    side_dim: int = 4
    q_max: int = Q_MAX
    sigma_exploit: float = SIGMA_EXPLOIT
    sigma_explore: float = SIGMA_EXPLORE
    sigma_default: float = SIGMA_DEFAULT
    spread_payoff: float = SPREAD_PAYOFF
    inventory_penalty: float = INVENTORY_PENALTY
    boundary_penalty: float = BOUNDARY_PENALTY
    abstain_payoff: float = ABSTAIN_PAYOFF
    switch_prob: float = SWITCH_PROB


class InventoryBanditState(NamedTuple):
    """Mutable state."""
    task_id: jnp.ndarray          # int32 scalar — can change mid-episode
    inventory: jnp.ndarray        # int32 scalar, in [-Q_MAX, Q_MAX]
    step: jnp.ndarray             # int32 scalar
    prev_action: jnp.ndarray      # int32 scalar


def _get_sigma(prev_action, task_id, params):
    """Action-dependent observation noise."""
    best_dir = TASK_BEST_DIR[task_id]
    action_dir = jnp.where(prev_action == 0, 1,
                 jnp.where(prev_action == 1, -1, 0))
    is_hold_or_abstain = (prev_action >= 2)
    is_correct_dir = (action_dir == best_dir) & (best_dir != 0)
    sigma = jnp.where(
        is_hold_or_abstain, params.sigma_default,
        jnp.where(is_correct_dir, params.sigma_exploit, params.sigma_explore))
    sigma = jnp.where(
        (best_dir == 0) & ~is_hold_or_abstain,
        params.sigma_explore, sigma)
    return sigma


def env_reset(key, params):
    """Sample initial task, reset inventory, emit initial observation."""
    k_task, k_side = jax.random.split(key)
    task_id = jax.random.randint(k_task, (), 0, params.n_tasks)
    state = InventoryBanditState(
        task_id=task_id,
        inventory=jnp.int32(0),
        step=jnp.int32(0),
        prev_action=jnp.int32(3))
    side = SIDE_MEANS[task_id] + params.sigma_default * jax.random.normal(
        k_side, (params.side_dim,))
    inv_norm = jnp.float32(0.0)
    obs = jnp.concatenate([jnp.zeros(1), inv_norm[None], side])
    return state, obs


def env_step(key, state, action, params):
    """Execute action, possibly switch regime, compute reward.

    Regime switch happens BEFORE reward computation — if the regime
    switches, the agent immediately feels the new drift/risk on its
    existing position. This is the "sudden regime change" that makes
    stale positions dangerous.
    """
    k_switch, k_newtask, k_side = jax.random.split(key, 3)

    # --- Regime switch ---
    do_switch = jax.random.bernoulli(k_switch, params.switch_prob)
    new_task = jax.random.randint(k_newtask, (), 0, params.n_tasks)
    task_id = jnp.where(do_switch, new_task, state.task_id)

    # --- Inventory change ---
    inv_change = jnp.where(action == 0, 1,
                 jnp.where(action == 1, -1, 0))
    new_inv = jnp.clip(state.inventory + inv_change,
                       -params.q_max, params.q_max)

    # --- Reward (under current/new task) ---
    is_trade = (action <= 1).astype(jnp.float32)
    is_abstain = (action == 3).astype(jnp.float32)

    spread = params.spread_payoff * is_trade
    drift = TASK_DRIFT[task_id]
    direction_pnl = drift * state.inventory.astype(jnp.float32)
    risk = TASK_RISK[task_id]
    inv_penalty = (risk * params.inventory_penalty
                   * new_inv.astype(jnp.float32) ** 2)
    at_boundary = (jnp.abs(new_inv) == params.q_max).astype(jnp.float32)
    bound_pen = params.boundary_penalty * at_boundary
    abstain_rew = params.abstain_payoff * is_abstain

    reward = spread + direction_pnl - inv_penalty - bound_pen + abstain_rew

    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = InventoryBanditState(
        task_id=task_id,
        inventory=new_inv,
        step=new_step,
        prev_action=action)

    # Observation from current (possibly switched) task
    sigma = _get_sigma(action, task_id, params)
    side = SIDE_MEANS[task_id] + sigma * jax.random.normal(
        k_side, (params.side_dim,))
    time_sig = new_step.astype(jnp.float32) / params.t_episode
    inv_norm = new_inv.astype(jnp.float32) / params.q_max
    obs = jnp.concatenate([time_sig[None], inv_norm[None], side])
    return new_state, obs, reward, done, jnp.int32(0)
