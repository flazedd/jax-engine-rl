"""SimConfig NamedTuple — structural simulator parameters (regime-independent)."""
from typing import NamedTuple


class SimConfig(NamedTuple):
    n_levels: int = 100
    tick_size: float = 0.05
    depth_decay: float = 0.15
    agent_order_size: float = 1.0
    max_inventory: int = 16
    max_steps: int = 1000
    initial_volume_per_level: float = 0.5
    initial_spread_ticks: int = 4
