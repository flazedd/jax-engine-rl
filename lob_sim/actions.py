"""Discrete action table — (bid_ticks, ask_ticks) pairs."""
import jax.numpy as jnp

ACTION_TABLE = jnp.array([
    [1, 1],  # both sides tight — spread capture
    [1, 7],  # active bid, passive ask — go long
    [7, 1],  # passive bid, active ask — go short
])
N_ACTIONS = len(ACTION_TABLE)  # 3


def action_index_to_offsets(idx):
    """Convert action index to (bid_ticks, ask_ticks)."""
    return ACTION_TABLE[idx]


def offsets_to_action_index(bid, ask):
    """Convert (bid_ticks, ask_ticks) to action index.

    Returns the index of the closest matching action.
    """
    diffs = jnp.abs(ACTION_TABLE[:, 0] - bid) + jnp.abs(ACTION_TABLE[:, 1] - ask)
    return jnp.argmin(diffs)
