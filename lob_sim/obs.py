"""Observation vector — observe(state, config) → flat jnp array."""
import jax.numpy as jnp

OBS_DEPTH = 10


def obs_size():
    """Return the observation vector length: 3 * OBS_DEPTH + 3."""
    return 3 * OBS_DEPTH + 3


def observe(state, config):
    """Compute observation vector from state.

    Components (concatenated):
    - Top-K bid volumes (normalized by initial_volume_per_level)
    - Top-K ask volumes (normalized)
    - Cumulative bid-ask imbalance at each depth
    - Spread (normalized by mid_price)
    - Inventory (normalized by max_inventory)
    - PnL proxy: (cash + inventory * mid_price) / 1000
    """
    k = OBS_DEPTH
    eps = 1e-8

    norm = config.initial_volume_per_level + eps
    bid_top = state.bid_volumes[:k] / norm
    ask_top = state.ask_volumes[:k] / norm

    cum_bid = jnp.cumsum(state.bid_volumes[:k])
    cum_ask = jnp.cumsum(state.ask_volumes[:k])
    imbalance = (cum_bid - cum_ask) / (cum_bid + cum_ask + eps)

    spread = 2.0 * state.half_spread_ticks * config.tick_size / (state.mid_price + eps)
    inv_norm = state.inventory / (config.max_inventory + eps)
    pnl = (state.cash + state.inventory * state.mid_price) / 1000.0

    return jnp.concatenate([
        bid_top,
        ask_top,
        imbalance,
        jnp.stack([spread, inv_norm, pnl]),
    ])
