"""HMM regime system — transition matrix, per-regime parameters.

All tunable simulation parameters live here, indexed by regime.
Structural/fixed parameters remain in SimConfig.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

NOISE = 0
BULL = 1
BEAR = 2
N_REGIMES = 3

TRANSITION_MATRIX = jnp.array([
    [0.9, 0.05, 0.05],  # NOISE
    [0.15, 0.7, 0.15],  # BULL
    [0.15, 0.15, 0.7],  # BEAR
])


class RegimeStepParams(NamedTuple):
    # Background flow
    market_buy_prob: jnp.ndarray
    market_sell_prob: jnp.ndarray
    market_order_size_min: jnp.ndarray
    market_order_size_max: jnp.ndarray
    cancel_prob: jnp.ndarray
    limit_order_rate: jnp.ndarray
    limit_order_size: jnp.ndarray
    # Price dynamics
    price_drift: jnp.ndarray
    volatility_scale: jnp.ndarray
    # Reward shaping
    inventory_penalty: jnp.ndarray
    inventory_penalty_threshold: jnp.ndarray
    spread_capture_bonus: jnp.ndarray


#                                        Noise   Bull    Bear
_MARKET_BUY_PROB          = jnp.array([  0.30,   0.15,   0.35])
_MARKET_SELL_PROB         = jnp.array([  0.30,   0.35,   0.15])
_MARKET_ORDER_SIZE_MIN    = jnp.array([  1.0,    1.0,    1.0 ])
_MARKET_ORDER_SIZE_MAX    = jnp.array([  2.0,    2.0,    2.0 ])
_CANCEL_PROB              = jnp.array([  0.10,   0.12,   0.12])
_LIMIT_ORDER_RATE         = jnp.array([  0.25,   0.20,   0.20])
_LIMIT_ORDER_SIZE         = jnp.array([  1.0,    1.0,    1.0 ])
_PRICE_DRIFT              = jnp.array([  0.0,    0.25,  -0.25])
_VOLATILITY_SCALE         = jnp.array([  1.0,    1.0,    1.0 ])
_INVENTORY_PENALTY        = jnp.array([  0.35,   0.0,    0.0 ])
_INVENTORY_PENALTY_THRESH = jnp.array([  3,      6,      6   ], dtype=jnp.int32)
_SPREAD_CAPTURE_BONUS     = jnp.array([  50.0,   0.0,    0.0 ])


def transition_regime(current, rng_key):
    """Transition to a new regime via the HMM transition matrix."""
    probs = TRANSITION_MATRIX[current]
    return jax.random.choice(rng_key, N_REGIMES, p=probs)


def get_regime_params(regime):
    """Get per-regime parameters by indexing into the parameter arrays."""
    return RegimeStepParams(
        market_buy_prob=_MARKET_BUY_PROB[regime],
        market_sell_prob=_MARKET_SELL_PROB[regime],
        market_order_size_min=_MARKET_ORDER_SIZE_MIN[regime],
        market_order_size_max=_MARKET_ORDER_SIZE_MAX[regime],
        cancel_prob=_CANCEL_PROB[regime],
        limit_order_rate=_LIMIT_ORDER_RATE[regime],
        limit_order_size=_LIMIT_ORDER_SIZE[regime],
        price_drift=_PRICE_DRIFT[regime],
        volatility_scale=_VOLATILITY_SCALE[regime],
        inventory_penalty=_INVENTORY_PENALTY[regime],
        inventory_penalty_threshold=_INVENTORY_PENALTY_THRESH[regime],
        spread_capture_bonus=_SPREAD_CAPTURE_BONUS[regime],
    )
