"""Per-(method, regime) action-distribution rollouts.

For each trained policy on MM E_final, roll out evaluation episodes and
record the (action, true_regime) pair at every step. Aggregating over
all steps gives `P(action | regime, method)` — the average action-class
distribution the trained policy uses in each regime.

Used by `scripts/m5_action_distributions.py` to produce the 7-method ×
3-regime action-distribution figure, which is the most direct
visualisation of "what does each method actually do per regime?".

Memoryless and recurrent agents are handled uniformly via the same
`agent.init_carry` / `agent.act` interface that posterior_probe uses;
the only difference is that the recorded outputs here drop the belief
columns since this analysis is policy-behavior-only.
"""
from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np


def _extract_inventory(state: Any) -> Any:
    """Walk through wrapper-nested env state to recover the underlying
    MarketMakingV1 inventory `q`. Returns None if no `q` key is found.

    Handles raw envs ({"q", "t", "regime"}) as well as the rl2_obs and
    stack_obs wrappers that nest the inner state under an "inner" key.
    """
    if not isinstance(state, dict):
        return None
    if "q" in state:
        return state["q"]
    if "inner" in state:
        return _extract_inventory(state["inner"])
    return None


def collect_action_regime_rollouts(
    env: Any,
    agent: Any,
    agent_state: chex.ArrayTree,
    n_rollouts: int,
    rollout_length: int,
    key: chex.PRNGKey,
) -> dict[str, np.ndarray]:
    """Run `n_rollouts` parallel evaluation rollouts, recording per-step:
      - action: the action the policy took, shape [T, N]
      - regime: the true regime active during that step, shape [T, N]
      - reward: per-step reward, shape [T, N] (for sanity checks)
      - done: episode boundary marker, shape [T, N]

    Handles both memoryless (PPO / Belief-PPO / Oracle-PPO — agent.act
    takes (state, obs, key)) and recurrent (RL² / VariBAD — agent.act
    takes (state, carry, obs, key)) agents by branching on the
    presence of `init_carry`.

    Greedy action sampling is *not* used — actions follow the agent's
    stochastic policy. This is what we want for measuring `P(action |
    regime)`: the trained policy's distribution, not its mode."""

    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)

    reset_keys = jax.random.split(key, n_rollouts + 1)
    keys_init, key = reset_keys[:n_rollouts], reset_keys[-1]
    env_states, obses = reset_v(keys_init)

    has_carry = hasattr(agent, "init_carry")
    carry = agent.init_carry(n_rollouts) if has_carry else None

    out_action = []
    out_regime = []
    out_inventory = []
    out_reward = []
    out_done = []

    if has_carry:
        @jax.jit
        def act_step(agent_state_, carry_, obs_, key_):
            keys = jax.random.split(key_, n_rollouts)
            return jax.vmap(
                lambda c, o, k: agent.act(agent_state_, c, o, k),
                in_axes=(0, 0, 0),
            )(carry_, obs_, keys)
    else:
        @jax.jit
        def act_step(agent_state_, obs_, key_):
            keys = jax.random.split(key_, n_rollouts)
            return jax.vmap(
                lambda o, k: agent.act(agent_state_, o, k),
                in_axes=(0, 0),
            )(obs_, keys)

    for _ in range(rollout_length):
        step_key, key = jax.random.split(key)
        if has_carry:
            action, _extras, new_carry = act_step(agent_state, carry, obses, step_key)
        else:
            action, _extras, _ = act_step(agent_state, obses, step_key)
            new_carry = None

        # Pre-action inventory: the inventory state the agent observed when
        # selecting `action`. Recorded before stepping so it pairs with the
        # action that was just chosen rather than the post-fill inventory.
        # Wrappers may nest the underlying env state one or more levels deep
        # under an "inner" key (rl2_obs, stack_obs); walk down to find "q".
        q_pre = _extract_inventory(env_states)

        step_keys = jax.random.split(key, n_rollouts + 1)
        env_keys, key = step_keys[:n_rollouts], step_keys[-1]
        env_states, obses_next, rewards, dones, info = step_v(
            env_states, action, env_keys,
        )

        out_action.append(np.asarray(action))
        out_regime.append(np.asarray(info["regime"]))
        if q_pre is not None:
            out_inventory.append(np.asarray(q_pre))
        out_reward.append(np.asarray(rewards))
        out_done.append(np.asarray(dones))

        if has_carry:
            zeros = jnp.zeros_like(new_carry)
            mask = dones[:, None].astype(new_carry.dtype)
            carry = mask * zeros + (1.0 - mask) * new_carry
        obses = obses_next

    out = {
        "action": np.stack(out_action).astype(np.int32),
        "regime": np.stack(out_regime).astype(np.int32),
        "reward": np.stack(out_reward),
        "done": np.stack(out_done).astype(np.int32),
    }
    if out_inventory:
        out["inventory"] = np.stack(out_inventory).astype(np.int32)
    return out


def compute_action_given_regime(
    actions: np.ndarray,  # [T, N] int
    regimes: np.ndarray,  # [T, N] int
    n_actions: int,
    n_regimes: int,
) -> np.ndarray:
    """Compute P(action | regime) — shape [n_regimes, n_actions].

    Each row sums to 1. Steps with regime not in [0, n_regimes) are
    excluded (defensive — should never happen on a valid env).
    """
    a_flat = actions.reshape(-1)
    r_flat = regimes.reshape(-1)
    mask = (r_flat >= 0) & (r_flat < n_regimes)
    a_flat = a_flat[mask]
    r_flat = r_flat[mask]
    counts = np.zeros((n_regimes, n_actions), dtype=np.float64)
    for r in range(n_regimes):
        sel = r_flat == r
        if sel.sum() == 0:
            continue
        actions_in_r = a_flat[sel]
        for a in range(n_actions):
            counts[r, a] = (actions_in_r == a).sum()
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero on empty regimes
    return counts / row_sums


def compute_action_given_regime_inventory(
    actions: np.ndarray,    # [T, N] int
    regimes: np.ndarray,    # [T, N] int
    inventories: np.ndarray,  # [T, N] int, signed pre-action inventory
    n_actions: int,
    n_regimes: int,
    inv_max: int,
) -> np.ndarray:
    """Compute P(action | regime, inventory) -> [n_regimes, 2*inv_max+1, n_actions].

    Inventories are signed integers in [-inv_max, +inv_max] and are mapped to
    bins 0..2*inv_max via `bin = q + inv_max`. Each (regime, inventory) row
    sums to 1 over actions; rows with zero observed steps are kept as zeros.
    """
    a_flat = actions.reshape(-1)
    r_flat = regimes.reshape(-1)
    q_flat = inventories.reshape(-1)
    n_inv = 2 * inv_max + 1
    bin_q = q_flat + inv_max
    mask = (
        (r_flat >= 0) & (r_flat < n_regimes)
        & (bin_q >= 0) & (bin_q < n_inv)
    )
    a_flat = a_flat[mask]
    r_flat = r_flat[mask]
    bin_q = bin_q[mask]
    counts = np.zeros((n_regimes, n_inv, n_actions), dtype=np.float64)
    flat_idx = (r_flat * n_inv + bin_q) * n_actions + a_flat
    np.add.at(counts.reshape(-1), flat_idx, 1.0)
    row_sums = counts.sum(axis=2, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return counts / safe
