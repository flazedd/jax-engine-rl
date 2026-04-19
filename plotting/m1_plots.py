"""M1-specific figures: learning curve with analytical ceiling, and policy
skew vs the analytical AS optimal."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.mm_reduced import (
    ACTION_FAVOR_ASK,
    ACTION_FAVOR_BID,
    ACTION_SYM,
    MMReducedEnv,
)
from oracles.analytical_as import solve_analytical_as
from plotting.load_results import load_config, load_metrics
from plotting.style import COLORS, FIGSIZE_STANDARD, apply_style


def _env_from_config(cfg: dict) -> MMReducedEnv:
    env_params = cfg["env"]["params"]
    return MMReducedEnv(**env_params)


def plot_learning_curve_with_ceiling(experiment_dir: Path, output_path: Path) -> dict:
    apply_style()
    metrics = load_metrics(experiment_dir)
    cfg = load_config(experiment_dir)
    env = _env_from_config(cfg)
    sol = solve_analytical_as(env)
    ceiling = float(sol.expected_episode_return)

    iters = np.arange(metrics["iterations"])
    per_seed = np.asarray(metrics["per_seed_mean_return_per_iter"])  # [seeds, T]
    mean_curve = per_seed.mean(axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    color = COLORS.get("ppo", "#1f77b4")
    ax.plot(iters, mean_curve, color=color, label="PPO (mean over seeds)")

    if per_seed.shape[0] > 1:
        lo = np.percentile(per_seed, 2.5, axis=0)
        hi = np.percentile(per_seed, 97.5, axis=0)
        ax.fill_between(iters, lo, hi, color=color, alpha=0.2, label="seed 95% range")

    ax.axhline(ceiling, color="black", linestyle="--", linewidth=1.0, label=f"AS analytical = {ceiling:.1f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode return")
    ax.set_title("M1 — PPO on E0 (vanilla AS)")
    ax.legend(loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    return {"final_mean": float(mean_curve[-1]), "ceiling": ceiling}


def plot_policy_vs_as(experiment_dir: Path, output_path: Path) -> dict:
    apply_style()
    metrics = load_metrics(experiment_dir)
    cfg = load_config(experiment_dir)
    env = _env_from_config(cfg)
    sol = solve_analytical_as(env)

    per_seed_probs = np.asarray(metrics["per_seed_final_action_probs"])  # [seeds, inv, 3]
    probs_mean = per_seed_probs.mean(axis=0)
    ppo_skew = probs_mean[:, ACTION_FAVOR_ASK] - probs_mean[:, ACTION_FAVOR_BID]
    as_skew = sol.skew

    inventory_levels = np.arange(-env.inventory_max, env.inventory_max + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot(inventory_levels, as_skew, marker="o", color="black", label="AS analytical skew")
    ax.plot(inventory_levels, ppo_skew, marker="s", color=COLORS.get("ppo", "#1f77b4"), label="PPO skew (mean over seeds)")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("inventory q")
    ax.set_ylabel("P(favor_ask) − P(favor_bid)")
    ax.set_title("M1 — PPO policy vs analytical AS")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    return {"ppo_skew": ppo_skew.tolist(), "as_skew": as_skew.tolist()}
