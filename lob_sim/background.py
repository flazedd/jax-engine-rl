"""Background order flow — cancellations, limit orders, market orders."""
import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig


def generate_background_flow(bid_volumes, ask_volumes, config: SimConfig, rng_key, regime_params=None):
    """Generate one step of background order flow.

    If regime_params is provided, use regime-specific parameters for varying
    quantities. Otherwise fall back to config values (Phase 1/2 behaviour).

    Returns (new_bid_volumes, new_ask_volumes, market_buy_qty, market_sell_qty).
    """
    n = bid_volumes.shape[0]
    levels = jnp.arange(n, dtype=jnp.float32)

    keys = jax.random.split(rng_key, 12)

    if regime_params is not None:
        cancel_prob = regime_params.cancel_prob
        limit_order_rate = regime_params.limit_order_rate
        market_buy_prob = regime_params.market_buy_prob
        market_sell_prob = regime_params.market_sell_prob
        volatility_scale = regime_params.volatility_scale
    else:
        cancel_prob = config.cancel_prob
        limit_order_rate = config.limit_order_rate
        market_buy_prob = config.market_buy_prob
        market_sell_prob = config.market_sell_prob
        volatility_scale = config.volatility_scale

    # --- Cancellations ---
    # Per-level: cancel with prob cancel_prob, remove random 0-50% fraction
    bid_cancel_mask = jax.random.bernoulli(keys[0], cancel_prob, shape=(n,))
    bid_cancel_frac = jax.random.uniform(keys[1], shape=(n,), minval=0.0, maxval=0.5)
    bid_cancelled = bid_volumes * bid_cancel_mask * bid_cancel_frac
    bid_after_cancel = bid_volumes - bid_cancelled

    ask_cancel_mask = jax.random.bernoulli(keys[2], cancel_prob, shape=(n,))
    ask_cancel_frac = jax.random.uniform(keys[3], shape=(n,), minval=0.0, maxval=0.5)
    ask_cancelled = ask_volumes * ask_cancel_mask * ask_cancel_frac
    ask_after_cancel = ask_volumes - ask_cancelled

    # --- New limit orders ---
    # Rate decays with depth: limit_order_rate * exp(-depth_decay * i)
    rates = limit_order_rate * jnp.exp(-config.depth_decay * levels)
    bid_arrivals = jax.random.bernoulli(keys[4], rates, shape=(n,))
    bid_new = bid_after_cancel + bid_arrivals * config.limit_order_size

    ask_arrivals = jax.random.bernoulli(keys[5], rates, shape=(n,))
    ask_new = ask_after_cancel + ask_arrivals * config.limit_order_size

    # --- Market orders ---
    do_market_buy = jax.random.bernoulli(keys[6], market_buy_prob)
    market_buy_size = jax.random.uniform(
        keys[7], minval=config.market_order_size_min, maxval=config.market_order_size_max
    )
    market_buy_qty = jnp.where(do_market_buy, market_buy_size * volatility_scale, 0.0)

    do_market_sell = jax.random.bernoulli(keys[8], market_sell_prob)
    market_sell_size = jax.random.uniform(
        keys[9], minval=config.market_order_size_min, maxval=config.market_order_size_max
    )
    market_sell_qty = jnp.where(do_market_sell, market_sell_size * volatility_scale, 0.0)

    return bid_new, ask_new, market_buy_qty, market_sell_qty
