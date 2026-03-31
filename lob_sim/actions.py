"""Discrete action table — 5x5 grid of (bid_ticks, ask_ticks)."""
import itertools

import jax.numpy as jnp

BID_TICKS = [1, 2, 3, 4, 5]
ASK_TICKS = [1, 2, 3, 4, 5]

ACTION_TABLE = jnp.array(list(itertools.product(BID_TICKS, ASK_TICKS)))
N_ACTIONS = 25


def action_index_to_offsets(idx):
    """Convert action index (0-24) to (bid_ticks, ask_ticks)."""
    return ACTION_TABLE[idx]


def offsets_to_action_index(bid, ask):
    """Convert (bid_ticks, ask_ticks) to action index."""
    return (bid - 1) * 5 + (ask - 1)
