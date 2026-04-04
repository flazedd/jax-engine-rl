"""Discrete action table — grid of (bid_ticks, ask_ticks)."""
import itertools

import jax.numpy as jnp

BID_TICKS = [1, 5, 9]
ASK_TICKS = [1, 5, 9]

ACTION_TABLE = jnp.array(list(itertools.product(BID_TICKS, ASK_TICKS)))
N_ACTIONS = len(ACTION_TABLE)  # 9


def action_index_to_offsets(idx):
    """Convert action index to (bid_ticks, ask_ticks)."""
    return ACTION_TABLE[idx]


def offsets_to_action_index(bid, ask):
    """Convert (bid_ticks, ask_ticks) to action index."""
    bid_idx = jnp.array(BID_TICKS).searchsorted(bid)
    ask_idx = jnp.array(ASK_TICKS).searchsorted(ask)
    return bid_idx * len(ASK_TICKS) + ask_idx
