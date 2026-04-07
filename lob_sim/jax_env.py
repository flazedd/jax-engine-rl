"""JAX-native POMDP market making environment with HMM regime switching.

Phase 0 implementation per CLAUDE.md specification.

State: (inventory, regime, mid_price, step)
Actions: 3 discrete spread configurations
  0: symmetric  (1, 1) — optimal in noise
  1: lean-ask   (1, 3) — optimal in bull
  2: lean-bid   (3, 1) — optimal in bear

Observations: (fill_bid, fill_ask, mid_change, inventory)

Fill model: P(fill_side) = exp(-κ[regime, side] · δ[action, side])
Reward: spread_pnl - inventory_risk - boundary_penalty

All state/config as NamedTuples. Designed for jit, vmap, lax.scan.

Usage:
    params = EnvParams.default()
    key = jax.random.PRNGKey(0)
    state, obs = env_reset(key, params)
    state, obs, reward, done, info = env_step(key, state, action, params)

    # Multi-episode trial (4 × 200 = 800 steps):
    traj = rollout_trial(key, policy_fn, params)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Environment parameters (frozen, passed to jitted functions)
# ---------------------------------------------------------------------------

class EnvParams(NamedTuple):
    """Immutable environment parameters matching CLAUDE.md spec."""
    kappa: jnp.ndarray            # (3, 2) fill intensity per [regime, side]
    delta: jnp.ndarray            # (3, 2) spread per [action, side]
    drift_probs: jnp.ndarray      # (3, 3) mid-price change probs per regime
    sigma_sq: jnp.ndarray         # (3,) per-regime volatility
    hmm_transition: jnp.ndarray   # (3, 3) regime transition matrix
    stationary_dist: jnp.ndarray  # (3,) precomputed stationary distribution
    gamma_disc: float = 0.99
    gamma_inventory: float = 0.1
    boundary_penalty: float = 5.0
    mtm_weight: float = 1.0
    inventory_max: int = 5
    t_episode: int = 200
    n_regimes: int = 3
    n_actions: int = 3
    locked_regime: int = -1       # -1 = mixed (HMM), 0/1/2 = locked

    @staticmethod
    def default():
        """Build default params from CLAUDE.md specification."""
        hmm = jnp.array([[0.90, 0.05, 0.05],
                          [0.10, 0.80, 0.10],
                          [0.10, 0.10, 0.80]])
        pi = _stationary_distribution(hmm)
        return EnvParams(
            kappa=jnp.array([[2.0, 2.0],    # noise — symmetric
                             [2.0, 0.8],    # bull  — ask fills easily
                             [0.8, 2.0]]),  # bear  — bid fills easily
            delta=jnp.array([[1.0, 1.0],    # symmetric
                             [1.0, 3.0],    # lean-ask
                             [3.0, 1.0]]),  # lean-bid
            drift_probs=jnp.array([[0.20, 0.60, 0.20],   # noise — zero mean
                                   [0.12, 0.50, 0.38],   # bull  — positive
                                   [0.38, 0.50, 0.12]]), # bear  — negative
            sigma_sq=jnp.array([0.5, 1.5, 1.5]),
            hmm_transition=hmm,
            stationary_dist=pi,
        )


# ---------------------------------------------------------------------------
# Environment state (carried between steps, traced by JAX)
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Mutable environment state — updated each step."""
    inventory: jnp.ndarray        # int32 scalar, [-5, 5]
    regime: jnp.ndarray           # int32 scalar, {0, 1, 2}
    mid_price: jnp.ndarray        # float32 scalar
    step: jnp.ndarray             # int32 scalar
    last_fill_bid: jnp.ndarray    # float32 scalar, {0, 1}
    last_fill_ask: jnp.ndarray    # float32 scalar, {0, 1}
    last_mid_change: jnp.ndarray  # float32 scalar, {-1, 0, +1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stationary_distribution(hmm: jnp.ndarray) -> jnp.ndarray:
    """Stationary distribution of HMM via power iteration."""
    pi = jnp.ones(hmm.shape[0]) / hmm.shape[0]
    def body(_, pi):
        pi = hmm.T @ pi
        return pi / pi.sum()
    return jax.lax.fori_loop(0, 100, body, pi)


# ---------------------------------------------------------------------------
# Core functions — all jit-compatible
# ---------------------------------------------------------------------------

def get_obs(state: EnvState, mid_change: jnp.ndarray) -> jnp.ndarray:
    """Extract 4D observation: (fill_bid, fill_ask, mid_change, inventory).

    Args:
        state: current environment state
        mid_change: mid-price change from current step (float, {-1, 0, +1})
    """
    return jnp.array([
        state.last_fill_bid,
        state.last_fill_ask,
        mid_change,
        state.inventory.astype(jnp.float32),
    ], dtype=jnp.float32)


def env_reset(
    key: jnp.ndarray, params: EnvParams,
) -> tuple[EnvState, jnp.ndarray]:
    """Reset environment. Returns (state, obs).

    Initial regime sampled from stationary distribution (or locked).
    Initial observation is all zeros.
    """
    sampled = jax.random.categorical(key, jnp.log(params.stationary_dist))
    regime = jnp.where(
        params.locked_regime >= 0, params.locked_regime, sampled,
    ).astype(jnp.int32)

    state = EnvState(
        inventory=jnp.int32(0),
        regime=regime,
        mid_price=jnp.float32(0.0),
        step=jnp.int32(0),
        last_fill_bid=jnp.float32(0.0),
        last_fill_ask=jnp.float32(0.0),
        last_mid_change=jnp.float32(0.0),
    )
    obs = get_obs(state, jnp.float32(0.0))
    return state, obs


def env_step(
    key: jnp.ndarray,
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """One environment step.

    Returns (new_state, obs, reward, done, info).
    """
    k_bid, k_ask, k_mid, k_regime = jax.random.split(key, 4)
    r = state.regime
    a = action

    # --- Fills: P(fill) = exp(-κ[regime, side] · δ[action, side]) ---
    p_bid = jnp.exp(-params.kappa[r, 0] * params.delta[a, 0])
    p_ask = jnp.exp(-params.kappa[r, 1] * params.delta[a, 1])
    fill_bid = jax.random.bernoulli(k_bid, p_bid).astype(jnp.float32)
    fill_ask = jax.random.bernoulli(k_ask, p_ask).astype(jnp.float32)

    # --- Inventory update: clip to [-5, 5] ---
    new_inv = jnp.clip(
        state.inventory + fill_bid.astype(jnp.int32) - fill_ask.astype(jnp.int32),
        -params.inventory_max, params.inventory_max,
    ).astype(jnp.int32)

    # --- Mid-price change: Categorical({-1, 0, +1}) from drift_probs ---
    mid_idx = jax.random.categorical(k_mid, jnp.log(params.drift_probs[r]))
    mid_change = mid_idx.astype(jnp.float32) - 1.0  # {0,1,2} → {-1,0,+1}
    new_mid = state.mid_price + mid_change

    # --- Reward ---
    # Spread PnL: both sides earn their delta on fill
    spread_pnl = fill_bid * params.delta[a, 0] + fill_ask * params.delta[a, 1]
    # Inventory risk penalty: 0.1 · q'² · σ²[regime]  (post-step q')
    q_f = new_inv.astype(jnp.float32)
    inv_penalty = params.gamma_inventory * q_f * q_f * params.sigma_sq[r]
    # Boundary penalty: 5.0 · |q'| when |q'| == inventory_max
    at_boundary = (jnp.abs(new_inv) == params.inventory_max).astype(jnp.float32)
    boundary = params.boundary_penalty * jnp.abs(q_f) * at_boundary
    # Mark-to-market: reward for holding inventory in the right direction
    q_pre = state.inventory.astype(jnp.float32)
    mtm = params.mtm_weight * q_pre * mid_change

    reward = spread_pnl - inv_penalty - boundary + mtm

    # --- Regime transition ---
    new_regime_sampled = jax.random.categorical(
        k_regime, jnp.log(params.hmm_transition[r]))
    new_regime = jnp.where(
        params.locked_regime >= 0, r, new_regime_sampled,
    ).astype(jnp.int32)

    # --- New state ---
    new_step = state.step + 1
    new_state = EnvState(
        inventory=new_inv,
        regime=new_regime,
        mid_price=new_mid,
        step=new_step,
        last_fill_bid=fill_bid,
        last_fill_ask=fill_ask,
        last_mid_change=mid_change,
    )

    obs = get_obs(new_state, mid_change)
    done = (new_step >= params.t_episode)
    info = {"regime": r}

    return new_state, obs, reward, done, info


# ---------------------------------------------------------------------------
# lax.scan rollout — single episode
# ---------------------------------------------------------------------------

def rollout_episode(
    key: jnp.ndarray,
    policy_fn,
    params: EnvParams,
) -> dict:
    """Roll out a single episode via lax.scan.

    Args:
        policy_fn: callable(key, obs) -> action (int32 scalar)

    Returns dict with arrays of shape (t_episode,):
        obs, actions, rewards, dones, true_regimes
    """
    k_reset, k_steps = jax.random.split(key)
    state, obs = env_reset(k_reset, params)
    keys = jax.random.split(k_steps, params.t_episode)

    def scan_step(carry, key_t):
        state, obs = carry
        k_act, k_env = jax.random.split(key_t)
        action = policy_fn(k_act, obs)
        regime = state.regime
        new_state, new_obs, reward, done, _ = env_step(
            k_env, state, action, params)
        return (new_state, new_obs), {
            "obs": obs,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "true_regimes": regime,
        }

    _, trajectory = jax.lax.scan(scan_step, (state, obs), keys)
    return trajectory


def batch_rollout(
    key: jnp.ndarray,
    policy_fn,
    params: EnvParams,
    n_envs: int = 64,
) -> dict:
    """Roll out n_envs episodes in parallel via vmap.

    Returns dict with arrays of shape (n_envs, t_episode, ...).
    """
    keys = jax.random.split(key, n_envs)
    return jax.vmap(rollout_episode, in_axes=(0, None, None))(
        keys, policy_fn, params)


# ---------------------------------------------------------------------------
# Multi-episode trial wrapper
# ---------------------------------------------------------------------------

def _reset_episode(regime: jnp.ndarray) -> EnvState:
    """Create a fresh episode state preserving the given regime."""
    return EnvState(
        inventory=jnp.int32(0),
        regime=regime.astype(jnp.int32),
        mid_price=jnp.float32(0.0),
        step=jnp.int32(0),
        last_fill_bid=jnp.float32(0.0),
        last_fill_ask=jnp.float32(0.0),
        last_mid_change=jnp.float32(0.0),
    )


def rollout_trial(
    key: jnp.ndarray,
    policy_fn,
    params: EnvParams,
    episodes_per_trial: int = 4,
) -> dict:
    """Roll out a multi-episode trial (e.g., 4 × 200 = 800 steps).

    HMM regime is continuous across episode boundaries.
    Inventory and mid_price reset to 0 at each episode boundary.

    Args:
        policy_fn: callable(key, obs) -> action (int32 scalar)
        episodes_per_trial: number of consecutive episodes

    Returns dict with arrays of shape (episodes_per_trial * t_episode,):
        obs, actions, rewards, dones, true_regimes
    """
    k_init, k_episodes = jax.random.split(key)

    # Sample initial regime from stationary distribution
    init_regime = jnp.where(
        params.locked_regime >= 0,
        params.locked_regime,
        jax.random.categorical(k_init, jnp.log(params.stationary_dist)),
    ).astype(jnp.int32)

    episode_keys = jax.random.split(k_episodes, episodes_per_trial)

    def episode_scan(regime, episode_key):
        state = _reset_episode(regime)
        obs = get_obs(state, jnp.float32(0.0))
        step_keys = jax.random.split(episode_key, params.t_episode)

        def step_scan(carry, key_t):
            state, obs = carry
            k_act, k_env = jax.random.split(key_t)
            action = policy_fn(k_act, obs)
            cur_regime = state.regime
            new_state, new_obs, reward, done, _ = env_step(
                k_env, state, action, params)
            return (new_state, new_obs), {
                "obs": obs,
                "actions": action,
                "rewards": reward,
                "dones": done,
                "true_regimes": cur_regime,
            }

        (final_state, _), episode_traj = jax.lax.scan(
            step_scan, (state, obs), step_keys)
        return final_state.regime, episode_traj

    final_regime, trial_traj = jax.lax.scan(
        episode_scan, init_regime, episode_keys)

    # Reshape from (episodes, steps, ...) to (total_steps, ...)
    return jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), trial_traj)
