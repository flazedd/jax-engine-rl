"""Plot regime comparison — generates plots/regimes.png."""
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from lob_sim.config import SimConfig
from lob_sim.regime import NOISE, BULL, BEAR
from lob_sim.step import run_episode

T = 2000
NAMES = {NOISE: "Noise", BULL: "Bull", BEAR: "Bear"}
COLORS = {NOISE: "gray", BULL: "green", BEAR: "red"}


def main():
    config = SimConfig()
    actions = jnp.full((T,), 12, dtype=jnp.int32)
    key = jax.random.PRNGKey(0)

    results = {}
    for regime in [NOISE, BULL, BEAR]:
        _, outputs = run_episode(config, key, actions, locked_regime=regime)
        results[regime] = outputs

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Mid-price trajectories
    ax = axes[0, 0]
    for regime in [NOISE, BULL, BEAR]:
        mid = results[regime]['mid_price']
        ax.plot(mid, color=COLORS[regime], label=NAMES[regime], alpha=0.8)
    ax.set_title("Mid-Price Trajectories")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mid Price")
    ax.legend()

    # 2. Cumulative bid fills vs ask fills
    ax = axes[0, 1]
    width = 0.35
    x = jnp.arange(3)
    bid_fills = [float(jnp.sum(results[r]['bid_fill'])) for r in [NOISE, BULL, BEAR]]
    ask_fills = [float(jnp.sum(results[r]['ask_fill'])) for r in [NOISE, BULL, BEAR]]
    ax.bar(x - width / 2, bid_fills, width, label="Bid Fills", color="blue", alpha=0.7)
    ax.bar(x + width / 2, ask_fills, width, label="Ask Fills", color="orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[r] for r in [NOISE, BULL, BEAR]])
    ax.set_title("Cumulative Fills by Regime")
    ax.legend()

    # 3. Inventory paths
    ax = axes[1, 0]
    for regime in [NOISE, BULL, BEAR]:
        inv = results[regime]['inventory']
        ax.plot(inv, color=COLORS[regime], label=NAMES[regime], alpha=0.8)
    ax.set_title("Inventory Paths")
    ax.set_xlabel("Step")
    ax.set_ylabel("Inventory")
    ax.legend()

    # 4. Average LOB depth
    ax = axes[1, 1]
    for regime in [NOISE, BULL, BEAR]:
        bid_depth = jnp.mean(results[regime]['bid_volumes'], axis=0)
        ask_depth = jnp.mean(results[regime]['ask_volumes'], axis=0)
        levels = jnp.arange(bid_depth.shape[0])
        ax.plot(levels, bid_depth, color=COLORS[regime], linestyle="-", alpha=0.7,
                label=f"{NAMES[regime]} bid")
        ax.plot(levels, ask_depth, color=COLORS[regime], linestyle="--", alpha=0.7,
                label=f"{NAMES[regime]} ask")
    ax.set_title("Average LOB Depth (top 20 levels)")
    ax.set_xlabel("Level")
    ax.set_ylabel("Avg Volume")
    ax.legend(fontsize=7)

    plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, "regimes.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
