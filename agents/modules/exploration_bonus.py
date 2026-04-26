"""Exploration-bonus module — composable axis for RL² and VariBAD.

Adds an intrinsic reward proportional to the L2 distance from the rolling
mean of the last K beliefs. Belief source is method-specific (GRU hidden
state for RL², latent μ for VariBAD), but the bonus function is
source-agnostic — it operates on whatever belief-like vector the agent
hands it.

Spec lives at docs/implementation.md → "agents/modules/exploration_bonus.py".
"""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp


def compute_exploration_bonus(
    belief_trajectory: chex.Array,  # [T, D]
    coef: float,
    window_K: int,
) -> chex.Array:
    """L2 distance from rolling-mean-of-last-K beliefs, scaled by coef.

    Returns per-step bonus, shape [T]. At step t, the window is
    [max(0, t-K), t); positions before the window's effective start are
    masked out so the rolling mean is well-defined even at small t. Bonus
    at t=0 is zero.

    Pure JAX, vmap-parallel over T — no Python loops.
    """
    T, D = belief_trajectory.shape

    def bonus_at(t):
        start = jnp.maximum(0, t - window_K)
        # Static window length so dynamic_slice has a fixed slice size.
        window = jax.lax.dynamic_slice(
            belief_trajectory,
            (start, 0),
            (window_K, D),
        )
        # Mask positions outside [start, t). When t < K, only the first
        # (t - start) entries are valid.
        valid = jnp.arange(window_K) < (t - start)
        n_valid = valid.sum()
        denom = jnp.maximum(n_valid, 1)
        weights = valid.astype(belief_trajectory.dtype) / denom
        window_mean = (window * weights[:, None]).sum(axis=0)
        # Spec: bonus at t=0 is zero — no prior beliefs to compare against.
        # Without this guard the empty-window mean collapses to zero and the
        # bonus would be ‖belief_t - 0‖ instead of 0.
        raw = coef * jnp.linalg.norm(belief_trajectory[t] - window_mean)
        return jnp.where(n_valid > 0, raw, 0.0)

    return jax.vmap(bonus_at)(jnp.arange(T))


def compute_exploration_bonus_batch(
    belief_trajectory_TND: chex.Array,  # [T, N, D] — env axis vmapped
    coef: float,
    window_K: int,
) -> chex.Array:
    """Vmap over the env axis. Returns bonus of shape [T, N]."""
    # [T, N, D] → vmap over axis 1 (envs), feed each env's [T, D] in.
    return jax.vmap(
        lambda traj: compute_exploration_bonus(traj, coef, window_K),
        in_axes=1,
        out_axes=1,
    )(belief_trajectory_TND)
