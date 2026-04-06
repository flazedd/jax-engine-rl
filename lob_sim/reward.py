"""Reward computation (legacy Phase 2 helper)."""
import jax.numpy as jnp


def compute_reward(prev_state, next_state, config, inventory_penalty=0.05):
    """Compute step reward.

    reward = value(next) - value(prev) - inventory_penalty * next.inventory²
    where value(s) = s.cash + s.inventory * s.mid_price

    Note: The main reward is computed inline in step.py using regime params.
    This function exists for backward compatibility with Phase 2 tests.
    """
    prev_val = prev_state.cash + prev_state.inventory * prev_state.mid_price
    next_val = next_state.cash + next_state.inventory * next_state.mid_price
    penalty = inventory_penalty * next_state.inventory ** 2
    return next_val - prev_val - penalty
