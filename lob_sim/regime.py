"""HMM regime system — transition matrix, per-regime parameters."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

NOISE = 0
BULL = 1
BEAR = 2
N_REGIMES = 3

TRANSITION_MATRIX = jnp.array([
    [0.98, 0.01, 0.01],  # NOISE
    [0.02, 0.97, 0.01],  # BULL
    [0.02, 0.01, 0.97],  # BEAR
])


class RegimeStepParams(NamedTuple):
    market_buy_prob: jnp.ndarray
    market_sell_prob: jnp.ndarray
    price_drift: jnp.ndarray
    volatility_scale: jnp.ndarray
    cancel_prob: jnp.ndarray
    limit_order_rate: jnp.ndarray


_MARKET_BUY_PROB = jnp.array([0.15, 0.30, 0.05])
_MARKET_SELL_PROB = jnp.array([0.15, 0.05, 0.30])
_PRICE_DRIFT = jnp.array([0.0, 0.003, -0.003])
_VOLATILITY_SCALE = jnp.array([1.0, 1.3, 1.3])
_CANCEL_PROB = jnp.array([0.10, 0.12, 0.12])
_LIMIT_ORDER_RATE = jnp.array([0.25, 0.20, 0.20])


def transition_regime(current, rng_key):
    """Transition to a new regime via the HMM transition matrix."""
    probs = TRANSITION_MATRIX[current]
    return jax.random.choice(rng_key, N_REGIMES, p=probs)


def get_regime_params(regime):
    """Get per-regime parameters by indexing into the parameter arrays."""
    return RegimeStepParams(
        market_buy_prob=_MARKET_BUY_PROB[regime],
        market_sell_prob=_MARKET_SELL_PROB[regime],
        price_drift=_PRICE_DRIFT[regime],
        volatility_scale=_VOLATILITY_SCALE[regime],
        cancel_prob=_CANCEL_PROB[regime],
        limit_order_rate=_LIMIT_ORDER_RATE[regime],
    )
