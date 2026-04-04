"""SimConfig NamedTuple — all simulator hyperparameters."""
from typing import NamedTuple


class SimConfig(NamedTuple):
    n_levels: int = 100
    tick_size: float = 0.02
    limit_order_rate: float = 0.25
    limit_order_size: float = 1.0
    market_buy_prob: float = 0.20
    market_sell_prob: float = 0.20
    market_order_size_min: float = 1.0
    market_order_size_max: float = 8.0
    cancel_prob: float = 0.10
    depth_decay: float = 0.05
    price_drift: float = 0.0
    volatility_scale: float = 1.0
    agent_order_size: float = 1.0
    max_inventory: int = 20
    inventory_penalty: float = 0.0
    max_steps: int = 1000
    initial_volume_per_level: float = 1.5
    initial_spread_ticks: int = 4
