"""JAX-native vectorized market making environment.

Fully functional — no mutable state. All randomness via explicit PRNGKeys.
Designed for jit compilation and vmap vectorization across parallel envs.

Usage:
    params = EnvParams.default()
    key = jax.random.PRNGKey(0)
    state = env_reset(key, params)
    state, obs, reward, done = env_step(key, state, action, params)

    # Vectorized across N envs:
    states = jax.vmap(env_reset, in_axes=(0, None))(keys, params)
    states, obs, rewards, dones = jax.vmap(env_step, in_axes=(0, 0, 0, None))(
        keys, states, actions, params)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Static environment parameters (not traced by JAX)
# ---------------------------------------------------------------------------

class EnvParams(NamedTuple):
    """Immutable environment parameters — passed to jitted functions."""
    # Fill probabilities: (n_regimes, n_actions)
    fill_bid: jnp.ndarray
    fill_ask: jnp.ndarray
    # Bid/ask edge per action: (n_actions,)
    bid_edge: jnp.ndarray
    ask_edge: jnp.ndarray
    # Per-regime drift: (n_regimes,)
    drift: jnp.ndarray
    # HMM transition CDF: (n_regimes, n_regimes) — cumulative for sampling
    trans_cdf: jnp.ndarray
    # Stationary distribution CDF: (n_regimes,)
    stationary_cdf: jnp.ndarray
    # Scalars
    max_inv: int = 10
    inv_penalty: float = 0.001
    episode_length: int = 200
    n_actions: int = 3
    n_regimes: int = 3
    locked_regime: int = -1  # -1 = mixed (HMM), 0/1/2 = locked regime

    @staticmethod
    def default():
        """Build params from default MDPConfig."""
        return EnvParams.from_config()

    @staticmethod
    def from_config(cfg=None):
        """Build params from an MDPConfig."""
        from lob_sim.analytical_mdp import (
            MDPConfig, build_mdp_tables, _get_action_table,
            stationary_distribution,
        )
        if cfg is None:
            cfg = MDPConfig()
        tables = build_mdp_tables(cfg)
        at = _get_action_table(cfg)
        pi = stationary_distribution(tables)

        tick = cfg.tick_size
        hs = cfg.half_spread_ticks

        return EnvParams(
            fill_bid=jnp.array(tables.fill_probs.bid),
            fill_ask=jnp.array(tables.fill_probs.ask),
            bid_edge=jnp.array((at[:, 0] + hs) * tick),
            ask_edge=jnp.array((at[:, 1] + hs) * tick),
            drift=jnp.array(cfg.drift),
            trans_cdf=jnp.cumsum(jnp.array(tables.trans_regime), axis=1),
            stationary_cdf=jnp.cumsum(jnp.array(pi)),
            max_inv=cfg.max_inv,
            inv_penalty=cfg.inv_penalty,
            episode_length=200,
            n_actions=len(at),
            n_regimes=len(cfg.drift),
        )


# ---------------------------------------------------------------------------
# Environment state (carried between steps, traced by JAX)
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Mutable environment state — updated each step."""
    regime: jnp.ndarray       # int32 scalar
    inventory: jnp.ndarray    # int32 scalar
    last_bid: jnp.ndarray     # float32 scalar (0 or 1)
    last_ask: jnp.ndarray     # float32 scalar (0 or 1)
    step_count: jnp.ndarray   # int32 scalar


# ---------------------------------------------------------------------------
# Core functions — all jit-compatible
# ---------------------------------------------------------------------------

def _sample_categorical(key: jnp.ndarray, cdf: jnp.ndarray) -> jnp.ndarray:
    """Sample from categorical distribution given CDF row."""
    u = jax.random.uniform(key)
    return jnp.searchsorted(cdf, u)


def get_obs(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Extract observation from state. Shape: (3,)."""
    return jnp.array([
        state.inventory / params.max_inv,
        state.last_bid,
        state.last_ask,
    ], dtype=jnp.float32)


def env_reset(key: jnp.ndarray, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
    """Reset environment. Returns (state, obs)."""
    sampled = _sample_categorical(key, params.stationary_cdf)
    regime = jnp.where(params.locked_regime >= 0, params.locked_regime, sampled)
    state = EnvState(
        regime=regime.astype(jnp.int32),
        inventory=jnp.int32(0),
        last_bid=jnp.float32(0),
        last_ask=jnp.float32(0),
        step_count=jnp.int32(0),
    )
    return state, get_obs(state, params)


def env_step(
    key: jnp.ndarray,
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Take one step. Returns (new_state, obs, reward, done, info)."""
    k1, k2, k3 = jax.random.split(key, 3)
    r = state.regime
    a = action

    # Sample fills
    pb = params.fill_bid[r, a]
    pa = params.fill_ask[r, a]
    bid_filled = (jax.random.uniform(k1) < pb).astype(jnp.float32)
    ask_filled = (jax.random.uniform(k2) < pa).astype(jnp.float32)

    # Reward
    reward = (bid_filled * params.bid_edge[a]
              + ask_filled * params.ask_edge[a]
              + state.inventory * params.drift[r]
              - params.inv_penalty * state.inventory * state.inventory)

    # Inventory update
    dq = bid_filled.astype(jnp.int32) - ask_filled.astype(jnp.int32)
    new_inv = jnp.clip(state.inventory + dq, -params.max_inv, params.max_inv)

    # Regime transition (locked = no transition)
    sampled_regime = _sample_categorical(k3, params.trans_cdf[r])
    new_regime = jnp.where(params.locked_regime >= 0, r, sampled_regime)

    new_state = EnvState(
        regime=new_regime.astype(jnp.int32),
        inventory=new_inv.astype(jnp.int32),
        last_bid=bid_filled,
        last_ask=ask_filled,
        step_count=state.step_count + 1,
    )
    obs = get_obs(new_state, params)
    done = (new_state.step_count >= params.episode_length)
    info = {"regime": r}

    return new_state, obs, reward, done, info


# ---------------------------------------------------------------------------
# Vectorized rollout — collect full episodes for training
# ---------------------------------------------------------------------------

def rollout_episode(
    key: jnp.ndarray,
    policy_fn,
    params: EnvParams,
    episode_length: int = 200,
) -> dict:
    """Roll out a single episode, collecting trajectory data.

    Args:
        policy_fn: callable(key, obs) -> action (int32 scalar)

    Returns dict with arrays of shape (episode_length,):
        obs, actions, rewards, dones, regimes
    """
    k_reset, k_steps = jax.random.split(key)
    state, obs = env_reset(k_reset, params)
    keys = jax.random.split(k_steps, episode_length)

    def scan_step(carry, key_t):
        state, obs = carry
        k_act, k_env = jax.random.split(key_t)
        action = policy_fn(k_act, obs)
        regime = state.regime
        new_state, new_obs, reward, done, _ = env_step(k_env, state, action, params)
        return (new_state, new_obs), {
            "obs": obs,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "regimes": regime,
        }

    _, trajectory = jax.lax.scan(scan_step, (state, obs), keys)
    return trajectory


def batch_rollout(
    key: jnp.ndarray,
    policy_fn,
    params: EnvParams,
    n_envs: int = 64,
    episode_length: int = 200,
) -> dict:
    """Roll out n_envs episodes in parallel via vmap.

    Returns dict with arrays of shape (n_envs, episode_length).
    """
    keys = jax.random.split(key, n_envs)
    return jax.vmap(rollout_episode, in_axes=(0, None, None, None))(
        keys, policy_fn, params, episode_length)
