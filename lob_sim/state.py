"""OrderBookState NamedTuple and init_state()."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig


class OrderBookState(NamedTuple):
    bid_volumes: jnp.ndarray
    ask_volumes: jnp.ndarray
    mid_price: jnp.ndarray
    half_spread_ticks: jnp.ndarray
    regime: jnp.ndarray
    inventory: jnp.ndarray
    cash: jnp.ndarray
    step_count: jnp.ndarray
    done: jnp.ndarray
    rng_key: jnp.ndarray


def init_state(config: SimConfig, rng_key: jnp.ndarray) -> OrderBookState:
    levels = jnp.arange(config.n_levels, dtype=jnp.float32)
    volumes = config.initial_volume_per_level * jnp.exp(-config.depth_decay * levels)
    return OrderBookState(
        bid_volumes=volumes,
        ask_volumes=volumes,
        mid_price=jnp.float32(100.0),
        half_spread_ticks=jnp.int32(config.initial_spread_ticks // 2),
        regime=jnp.int32(0),
        inventory=jnp.float32(0.0),
        cash=jnp.float32(0.0),
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
        rng_key=rng_key,
    )
