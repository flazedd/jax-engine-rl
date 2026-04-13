#!/usr/bin/env python3
"""Agent Ladder — Comparison Plotting.

Reads results/ppo_baseline.json, results/rl2.json (optional),
results/rl2_hn.json (optional), results/varibad.json (optional),
and results/analytical_foundation.json.
Overlays all agents on the same axes.

Produces:
  plots/figure8_comparison_curves.png  — learning curves, all agents per regime

Usage:
    uv run python scripts/plot_comparison.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")
PPO_JSON = os.path.join(RESULTS_DIR, "ppo_baseline.json")
RL2_JSON = os.path.join(RESULTS_DIR, "rl2.json")
RL2HN_JSON = os.path.join(RESULTS_DIR, "rl2_hn.json")
VARIBAD_JSON = os.path.join(RESULTS_DIR, "varibad.json")
FOUNDATION_JSON = os.path.join(RESULTS_DIR, "analytical_foundation.json")

REGIME_TITLES = {"noise": "Noise", "bull": "Bull", "bear": "Bear"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_oracle_bounds():
    with open(FOUNDATION_JSON) as f:
        return json.load(f).get("oracle_bounds", {})


def _curve_stats(runs, regime_name):
    """Extract (iters, mean, std) from a list of run dicts."""
    all_iters = []
    all_returns = []
    for run in runs:
        all_iters.append(np.array(run["iters"]))
        prr = run.get("per_regime_returns", {})
        if regime_name in prr:
            all_returns.append(np.array(prr[regime_name]))
        else:
            all_returns.append(np.array(run["mean_returns"]))
    iters = all_iters[0]
    mat = np.array(all_returns)
    return iters, np.mean(mat, axis=0), np.std(mat, axis=0), len(runs)


def _draw_agent(ax, runs, regime_name, color, label):
    """Draw one agent's curve on the axes."""
    iters, mean, std, n_seeds = _curve_stats(runs, regime_name)
    ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)
    ax.plot(iters, mean, color=color, linewidth=1.5,
            label=f"{label} ({n_seeds} seeds)")


def _draw_oracle(ax, regime_name, oracle_bounds, x_max):
    if regime_name not in oracle_bounds:
        return
    ob = oracle_bounds[regime_name]
    m, s = ob["mean"], ob["std"]
    ax.axhline(m, color="red", linestyle="--", linewidth=1.5, label="Oracle")
    ax.axhspan(m - s, m + s, color="red", alpha=0.06)
    ax.text(x_max, m, f" {m:.1f}", va="bottom", ha="right",
            fontsize=7, color="red", fontweight="bold")


# ---------------------------------------------------------------------------
# Figure — Comparison curves (2 rows × 3 cols)
# ---------------------------------------------------------------------------

def plot_comparison(ppo_data, rl2_data, rl2hn_data, varibad_data, oracle_bounds):
    """2×3 grid: top = isolated, bottom = mixed. All agents overlaid."""
    regime_cols = ["noise", "bull", "bear"]
    all_data = [ppo_data, rl2_data, rl2hn_data, varibad_data]

    has_iso = any(r in d for d in all_data for r in regime_cols)
    has_mix = any("mixed" in d for d in all_data)
    n_rows = int(has_iso) + int(has_mix)

    if n_rows == 0:
        print("  (no data to plot)")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.0 * n_rows), squeeze=False)
    row = 0

    # --- Top row: isolated ---
    if has_iso:
        for i, name in enumerate(regime_cols):
            ax = axes[row, i]
            x_max = 0
            if name in ppo_data:
                _draw_agent(ax, ppo_data[name]["runs"], name, "C0", "PPO MLP")
                x_max = max(x_max, ppo_data[name]["runs"][0]["iters"][-1])
            if name in rl2_data:
                _draw_agent(ax, rl2_data[name]["runs"], name, "C1", "RL²")
                x_max = max(x_max, rl2_data[name]["runs"][0]["iters"][-1])
            if name in rl2hn_data:
                _draw_agent(ax, rl2hn_data[name]["runs"], name, "C2", "RL²+HN")
                x_max = max(x_max, rl2hn_data[name]["runs"][0]["iters"][-1])
            if name in varibad_data:
                _draw_agent(ax, varibad_data[name]["runs"], name, "C3", "VariBAD")
                x_max = max(x_max, varibad_data[name]["runs"][0]["iters"][-1])
            _draw_oracle(ax, name, oracle_bounds, x_max)
            ax.set_title(REGIME_TITLES[name], fontsize=12, fontweight="bold")
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
            if i == 0:
                ax.set_ylabel("Mean episode return")
        axes[row, 0].annotate(
            "Isolated", xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="right", va="center", rotation=90)
        row += 1

    # --- Bottom row: mixed ---
    if has_mix:
        for i, name in enumerate(regime_cols):
            ax = axes[row, i]
            x_max = 0
            if "mixed" in ppo_data:
                _draw_agent(ax, ppo_data["mixed"]["runs"], name, "C0", "PPO MLP")
                x_max = max(x_max, ppo_data["mixed"]["runs"][0]["iters"][-1])
            if "mixed" in rl2_data:
                _draw_agent(ax, rl2_data["mixed"]["runs"], name, "C1", "RL²")
                x_max = max(x_max, rl2_data["mixed"]["runs"][0]["iters"][-1])
            if "mixed" in rl2hn_data:
                _draw_agent(ax, rl2hn_data["mixed"]["runs"], name, "C2", "RL²+HN")
                x_max = max(x_max, rl2hn_data["mixed"]["runs"][0]["iters"][-1])
            if "mixed" in varibad_data:
                _draw_agent(ax, varibad_data["mixed"]["runs"], name, "C3", "VariBAD")
                x_max = max(x_max, varibad_data["mixed"]["runs"][0]["iters"][-1])
            _draw_oracle(ax, name, oracle_bounds, x_max)
            ax.set_title(REGIME_TITLES[name], fontsize=12, fontweight="bold")
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
            if i == 0:
                ax.set_ylabel("Mean episode return")
        axes[row, 0].annotate(
            "Mixed", xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="right", va="center", rotation=90)

    fig.suptitle("Agent Ladder — Learning Curves (per regime)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure8_comparison_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Agent Ladder — Comparison Plotting")
    print("=" * 60)

    missing = []
    for path, label in [(PPO_JSON, "ppo_baseline.json"),
                        (FOUNDATION_JSON, "analytical_foundation.json")]:
        if not os.path.exists(path):
            missing.append(label)
    if missing:
        print(f"\n  ERROR: missing {', '.join(missing)}")
        print("  Run the corresponding training scripts first.")
        sys.exit(1)

    with open(PPO_JSON) as f:
        ppo_data = json.load(f)

    rl2_data = {}
    if os.path.exists(RL2_JSON):
        with open(RL2_JSON) as f:
            rl2_data = json.load(f)
        print("  Loaded rl2.json")
    else:
        print("  (rl2.json not found — plotting without RL²)")

    rl2hn_data = {}
    if os.path.exists(RL2HN_JSON):
        with open(RL2HN_JSON) as f:
            rl2hn_data = json.load(f)
        print("  Loaded rl2_hn.json")
    else:
        print("  (rl2_hn.json not found — plotting without RL²+HN)")

    varibad_data = {}
    if os.path.exists(VARIBAD_JSON):
        with open(VARIBAD_JSON) as f:
            varibad_data = json.load(f)
        print("  Loaded varibad.json")
    else:
        print("  (varibad.json not found — plotting without VariBAD)")

    oracle_bounds = load_oracle_bounds()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n  Plotting comparison curves ...")
    plot_comparison(ppo_data, rl2_data, rl2hn_data, varibad_data, oracle_bounds)

    print("\n  Done.")


if __name__ == "__main__":
    main()
