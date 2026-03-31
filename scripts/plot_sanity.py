"""Sanity plot — visual inspection of Phase 2 sim loop.

Generates plots/sanity.png with a 3x2 figure:
1. Mid-price trajectory
2. Spread over time
3. LOB depth snapshot at T/2
4. LOB depth heatmap
5. Agent inventory
6. Cumulative reward
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lob_sim.config import SimConfig
from lob_sim.step import run_episode


def main():
    config = SimConfig()
    key = jax.random.PRNGKey(0)
    T = config.max_steps
    actions = jnp.broadcast_to(jnp.array([3.0, 3.0]), (T, 2))

    jit_run = jax.jit(run_episode, static_argnums=(0,))

    # Compile
    t0 = time.time()
    final, outputs = jit_run(config, key, actions)
    outputs["mid_price"].block_until_ready()
    compile_time = time.time() - t0

    # Warm run
    t0 = time.time()
    final, outputs = jit_run(config, key, actions)
    outputs["mid_price"].block_until_ready()
    warm_time = time.time() - t0

    mid = outputs["mid_price"]
    spread = outputs["spread"]
    inv = outputs["inventory"]
    rew = outputs["reward"]
    bid_v = outputs["bid_volumes"]
    ask_v = outputs["ask_volumes"]

    plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 1. Mid-price
    axes[0, 0].plot(mid, linewidth=0.5)
    axes[0, 0].set_title("Mid-Price")
    axes[0, 0].set_xlabel("Step")

    # 2. Spread
    axes[0, 1].plot(spread, linewidth=0.5)
    axes[0, 1].set_title("Spread")
    axes[0, 1].set_xlabel("Step")

    # 3. LOB depth snapshot at T/2
    t_half = T // 2
    n_show = bid_v.shape[1]
    levels = jnp.arange(n_show)
    axes[1, 0].barh(levels, -bid_v[t_half, ::-1], color="green", alpha=0.7, label="Bids")
    axes[1, 0].barh(levels, ask_v[t_half], color="red", alpha=0.7, label="Asks")
    axes[1, 0].set_title(f"LOB Depth Snapshot (t={t_half})")
    axes[1, 0].set_ylabel("Level")
    axes[1, 0].legend()

    # 4. LOB depth heatmap
    combined = jnp.concatenate([bid_v[:, ::-1], ask_v], axis=1)
    axes[1, 1].imshow(combined.T, aspect="auto", cmap="hot", origin="lower")
    axes[1, 1].set_title("LOB Depth Heatmap")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Level (bid ← | → ask)")

    # 5. Inventory
    axes[2, 0].plot(inv, linewidth=0.5)
    axes[2, 0].set_title("Agent Inventory")
    axes[2, 0].set_xlabel("Step")

    # 6. Cumulative reward
    axes[2, 1].plot(jnp.cumsum(rew), linewidth=0.5)
    axes[2, 1].set_title("Cumulative Reward")
    axes[2, 1].set_xlabel("Step")

    plt.tight_layout()
    out_path = os.path.join(plot_dir, "sanity.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # Summary stats
    print(f"Price range: {float(jnp.min(mid)):.4f} – {float(jnp.max(mid)):.4f}")
    print(f"Price std:   {float(jnp.std(mid)):.4f}")
    print(f"Mean spread: {float(jnp.mean(spread)):.6f}")
    print(f"Final inv:   {float(final.inventory):.2f}")
    print(f"Total reward:{float(jnp.sum(rew)):.4f}")
    print(f"Compile time:{compile_time:.2f}s")
    print(f"Warm steps/s:{T / warm_time:.0f}")


if __name__ == "__main__":
    main()
