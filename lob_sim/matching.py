"""Matching engine — fill_market_buy() and fill_market_sell()."""
import jax.numpy as jnp


def fill_market_buy(ask_volumes, qty, half_spread_ticks, mid_price, tick_size):
    """Fill a market buy order against the ask side of the book.

    Returns (new_ask_volumes, filled_qty, avg_fill_price, new_mid_price, new_half_spread).
    """
    n = ask_volumes.shape[0]
    levels = jnp.arange(n)

    # Pre-existing gap: first level with volume
    pre_has_vol = ask_volumes > 0
    pre_first = jnp.where(jnp.any(pre_has_vol), jnp.argmax(pre_has_vol).astype(jnp.int32), jnp.int32(0))

    # Ask prices: mid_price + (i + half_spread_ticks) * tick_size
    prices = mid_price + (levels + half_spread_ticks) * tick_size

    # Cumulative sum for sweep
    cumsum = jnp.cumsum(ask_volumes)
    # cumsum_prev[i] = cumsum up to but not including level i
    cumsum_prev = jnp.concatenate([jnp.zeros(1), cumsum[:-1]])

    # Consumed per level: min(vol[i], max(0, qty - cumsum_prev[i]))
    remaining = jnp.maximum(0.0, qty - cumsum_prev)
    consumed = jnp.minimum(ask_volumes, remaining)

    filled_qty = jnp.sum(consumed)

    # VWAP fill price
    total_cost = jnp.sum(consumed * prices)
    avg_fill_price = jnp.where(filled_qty > 0, total_cost / filled_qty, mid_price)

    # New volumes after fill
    new_volumes = ask_volumes - consumed

    # Find how many levels were newly consumed (not pre-existing gaps)
    post_has_vol = new_volumes > 0
    post_first = jnp.where(jnp.any(post_has_vol), jnp.argmax(post_has_vol).astype(jnp.int32), jnp.int32(0))
    any_volume = jnp.any(post_has_vol)

    # Only shift by newly consumed levels, preserving pre-existing gaps
    shift = jnp.where(any_volume, jnp.maximum(post_first - pre_first, jnp.int32(0)), jnp.int32(0))

    # Roll and zero-fill the deep end
    shifted = jnp.roll(new_volumes, -shift)
    mask = jnp.where(levels >= n - shift, 0.0, 1.0)
    new_ask_volumes = shifted * mask

    # Update mid-price: each consumed level shifts mid up by tick_size
    shift_f = shift.astype(jnp.float32)
    new_mid_price = mid_price + shift_f * tick_size

    # half_spread stays the same (spread widens by the shift on the ask side,
    # but mid moved to compensate)
    new_half_spread = half_spread_ticks

    return new_ask_volumes, filled_qty, avg_fill_price, new_mid_price, new_half_spread


def fill_market_sell(bid_volumes, qty, half_spread_ticks, mid_price, tick_size):
    """Fill a market sell order against the bid side of the book.

    Mirror of fill_market_buy. Returns (new_bid_volumes, filled_qty, avg_fill_price, new_mid_price, new_half_spread).
    """
    n = bid_volumes.shape[0]
    levels = jnp.arange(n)

    # Pre-existing gap: first level with volume
    pre_has_vol = bid_volumes > 0
    pre_first = jnp.where(jnp.any(pre_has_vol), jnp.argmax(pre_has_vol).astype(jnp.int32), jnp.int32(0))

    # Bid prices: mid_price - (i + half_spread_ticks) * tick_size
    prices = mid_price - (levels + half_spread_ticks) * tick_size

    cumsum = jnp.cumsum(bid_volumes)
    cumsum_prev = jnp.concatenate([jnp.zeros(1), cumsum[:-1]])

    remaining = jnp.maximum(0.0, qty - cumsum_prev)
    consumed = jnp.minimum(bid_volumes, remaining)

    filled_qty = jnp.sum(consumed)

    total_cost = jnp.sum(consumed * prices)
    avg_fill_price = jnp.where(filled_qty > 0, total_cost / filled_qty, mid_price)

    new_volumes = bid_volumes - consumed

    post_has_vol = new_volumes > 0
    post_first = jnp.where(jnp.any(post_has_vol), jnp.argmax(post_has_vol).astype(jnp.int32), jnp.int32(0))
    any_volume = jnp.any(post_has_vol)

    # Only shift by newly consumed levels, preserving pre-existing gaps
    shift = jnp.where(any_volume, jnp.maximum(post_first - pre_first, jnp.int32(0)), jnp.int32(0))

    shifted = jnp.roll(new_volumes, -shift)
    mask = jnp.where(levels >= n - shift, 0.0, 1.0)
    new_bid_volumes = shifted * mask

    # Mid-price decreases when bid levels are depleted
    shift_f = shift.astype(jnp.float32)
    new_mid_price = mid_price - shift_f * tick_size

    new_half_spread = half_spread_ticks

    return new_bid_volumes, filled_qty, avg_fill_price, new_mid_price, new_half_spread
