#!/usr/bin/env python3
"""Phase 3 — Analytical Foundation Figures & Gate Checks.

Computes preconditions 1–3, generates Figure 1 (locked-regime oracle)
and Figure 2 (mixed-regime oracles), and runs the gate check.

Outputs:
  results/analytical_foundation.json  — precondition results + gate status
  plots/figure1_locked_regime.png     — Figure 1
  plots/figure2_mixed_regime.png      — Figure 2

Usage:
    uv run python scripts/analytical_foundation.py [--fast]
"""
import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lob_sim.jax_env import EnvParams
from lob_sim.analytical_mdp import (
    N_REGIMES, N_ACTIONS, REGIME_NAMES, ACTION_NAMES,
    build_mdp_tables, solve_all_locked, solve_oracle_a, solve_oracle_b,
    compute_q_max, precondition_1,
    simulate_oracle_a, simulate_oracle_b,
    precondition_2, precondition_3,
)

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")


# ---------------------------------------------------------------------------
# Shared plot constants
# ---------------------------------------------------------------------------

ACTION_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
ACTION_LABELS = ["sym(1,1)", "ask(1,3)", "bid(3,1)"]
REGIME_COLORS = ["#8C8C8C", "#55A868", "#C44E52"]  # noise=grey, bull=green, bear=red
CMAP3 = mcolors.ListedColormap(ACTION_COLORS)


def _action_heatstrip(ax, policy_arr, inv_grid, title):
    """Draw a single-row heatmap strip of discrete actions."""
    data = np.array(policy_arr, dtype=int).reshape(1, -1)
    ax.imshow(data, aspect="auto", cmap=CMAP3, vmin=0, vmax=2,
              interpolation="nearest",
              extent=[inv_grid[0] - 0.5, inv_grid[-1] + 0.5, -0.5, 0.5])
    ax.set_yticks([])
    ax.set_xticks(inv_grid)
    ax.set_xlabel("Inventory q")
    ax.set_title(title, fontsize=11)
    # Annotate action index inside each cell
    for i, q in enumerate(inv_grid):
        a = int(data[0, i])
        ax.text(q, 0, str(a), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")


def _action_dist_matrix(ax, action_dists, title):
    """Draw a 3x3 heatmap (rows=true regime, cols=action chosen).

    action_dists: (3, 3) array of proportions.
    """
    im = ax.imshow(action_dists, cmap="Blues", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(N_ACTIONS))
    ax.set_xticklabels(ACTION_LABELS, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(N_REGIMES))
    ax.set_yticklabels(REGIME_NAMES, fontsize=9)
    ax.set_xlabel("Action chosen")
    ax.set_ylabel("True regime")
    ax.set_title(title, fontsize=10)
    # Annotate cells
    for r in range(N_REGIMES):
        for a in range(N_ACTIONS):
            val = action_dists[r, a]
            color = "white" if val > 0.5 else "black"
            ax.text(a, r, f"{val:.0%}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)


# ---------------------------------------------------------------------------
# Figure 1 — Locked-Regime Oracle
# ---------------------------------------------------------------------------

def plot_figure1(locked_solutions, q_max, params, pc1):
    """Figure 1: heatmap strips + clamped gap charts + disagreement summary."""
    inv_max = int(params.inventory_max)
    n_inv = 2 * inv_max + 1
    inv_grid = np.arange(n_inv) - inv_max

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.4, 1, 1],
                          hspace=0.5, wspace=0.35)

    # --- Row 1: action heatmap strips ---
    ax_first = None
    for r in range(N_REGIMES):
        ax = fig.add_subplot(gs[0, r])
        policy = np.array(locked_solutions[r].policy)
        _action_heatstrip(ax, policy, inv_grid,
                          f"{REGIME_NAMES[r]} — Optimal Action")
        if r == 0:
            ax_first = ax
    # Legend for action colors (on existing first axis)
    for a in range(N_ACTIONS):
        ax_first.plot([], [], 's', color=ACTION_COLORS[a], ms=10,
                      label=ACTION_LABELS[a])
    ax_first.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # --- Row 2: relative gap bar charts (clamped y-axis) ---
    # Find a sensible y-max: show detail, not outliers
    all_gaps = [np.array(pc1["gap_per_regime"][r]["values"])
                for r in range(N_REGIMES)]
    p90 = max(np.percentile(g, 90) for g in all_gaps)
    y_max = min(max(p90 * 1.3, 15), 80)  # at least 15%, cap at 80%

    for r in range(N_REGIMES):
        ax = fig.add_subplot(gs[1, r])
        gap_vals = all_gaps[r]
        # Clip display but show indicator for clipped bars
        clipped = gap_vals > y_max
        display_vals = np.minimum(gap_vals, y_max)
        bars = ax.bar(inv_grid, display_vals, color=ACTION_COLORS[r],
                       alpha=0.85, width=0.75, edgecolor="white", linewidth=0.5)
        # Mark clipped bars with a triangle
        for qi in np.where(clipped)[0]:
            ax.annotate(f"{gap_vals[qi]:.0f}%",
                        xy=(inv_grid[qi], y_max), fontsize=6,
                        ha="center", va="bottom", color=ACTION_COLORS[r],
                        fontweight="bold")
        ax.axhline(5.0, color="#C44E52", linestyle="--", linewidth=1.5,
                   alpha=0.8, label="5% gate", zorder=5)
        ax.set_xlim(inv_grid[0] - 0.6, inv_grid[-1] + 0.6)
        ax.set_ylim(0, y_max * 1.08)
        ax.set_xlabel("Inventory q")
        if r == 0:
            ax.set_ylabel("Relative gap (% of Q_max)")
        mean_gap = pc1["gap_per_regime"][r]["mean"]
        ax.set_title(f"{REGIME_NAMES[r]} — Decisiveness", fontsize=11)
        ax.text(0.97, 0.95, f"mean = {mean_gap:.1f}%",
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))
        if r == 0:
            ax.legend(fontsize=8, loc="upper left")

    # --- Row 3: pairwise disagreements (left) + summary table (right) ---

    # Disagreement bars
    ax_bars = fig.add_subplot(gs[2, :2])
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = [f"{REGIME_NAMES[r1]} vs {REGIME_NAMES[r2]}"
                   for r1, r2 in pairs]
    pair_vals = [pc1["pairwise_disagreement"][(r1, r2)] for r1, r2 in pairs]
    bar_colors = ["#6A5ACD", "#CD853F", "#2E8B57"]
    bars = ax_bars.barh(pair_labels, pair_vals, color=bar_colors,
                         height=0.5, edgecolor="white")
    ax_bars.axvline(20.0, color="#C44E52", linestyle="--", linewidth=1.5,
                     alpha=0.8, label="20% gate")
    ax_bars.set_xlabel("Policy disagreement (%)")
    ax_bars.set_title("Pairwise Disagreement", fontsize=11)
    ax_bars.set_xlim(0, max(pair_vals) * 1.25)
    for bar, val in zip(bars, pair_vals):
        ax_bars.text(bar.get_width() + 0.8,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=10,
                     fontweight="bold")
    ax_bars.legend(fontsize=8)
    ax_bars.grid(axis="x", alpha=0.2)

    # Summary table
    ax_tab = fig.add_subplot(gs[2, 2])
    ax_tab.axis("off")
    table_data = []
    for r in range(N_REGIMES):
        g = pc1["gap_per_regime"][r]
        pct_above_10 = float(np.mean(np.array(g["values"]) > 10.0)) * 100
        table_data.append([
            REGIME_NAMES[r],
            f"{g['mean']:.1f}%",
            f"{g['min']:.1f}%",
            f"{pct_above_10:.0f}%",
        ])
    table = ax_tab.table(
        cellText=table_data,
        colLabels=["Regime", "Mean", "Min", ">10%"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # Color header
    for j in range(4):
        table[0, j].set_facecolor("#E8E8E8")
        table[0, j].set_text_props(fontweight="bold")

    # Gate status
    status = "PASS" if pc1["passed"] else "FAIL"
    scolor = "green" if pc1["passed"] else "red"
    fig.suptitle(f"Figure 1 — Locked-Regime Oracle   [{status}]",
                 fontsize=14, fontweight="bold", color=scolor, y=1.0)

    path = os.path.join(PLOTS_DIR, "figure1_locked_regime.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 — Mixed-Regime Oracles
# ---------------------------------------------------------------------------

def _compute_action_fracs(sim, inv_grid, inv_max):
    """Compute per-regime, per-inventory action fractions from simulation."""
    n_inv = len(inv_grid)
    fracs = np.zeros((N_REGIMES, n_inv, N_ACTIONS))
    for r in range(N_REGIMES):
        mask = sim.regimes == r
        actions = sim.actions[mask]
        invs = sim.inventories[mask]
        for qi in range(n_inv):
            inv_mask = invs == inv_grid[qi]
            n = inv_mask.sum()
            if n > 0:
                for a in range(N_ACTIONS):
                    fracs[r, qi, a] = (actions[inv_mask] == a).sum() / n
    return fracs


def plot_figure2(sim_a, sim_b, oracle_a_sol, locked_solutions, q_max, params, pc2, pc3):
    """Figure 2: 4 rows × 3 cols + disagreement summary.

    Row 1: Oracle A action heatstrips by true regime.
    Row 2: Oracle A relative gain by true regime (clamped).
    Row 3: Oracle B action heatstrips by true regime.
    Row 4: 3×3 action-distribution matrices (Locked, Oracle A, Oracle B).
    Below: side-by-side disagreement comparison bars.
    """
    inv_max = int(params.inventory_max)
    n_inv = 2 * inv_max + 1
    inv_grid = np.arange(n_inv) - inv_max

    fig = plt.figure(figsize=(15, 19))
    gs = fig.add_gridspec(5, 3, height_ratios=[0.6, 1, 0.6, 1.1, 1.0],
                          hspace=0.55, wspace=0.35)

    # Pre-compute per-inventory action fractions
    fracs_a = _compute_action_fracs(sim_a, inv_grid, inv_max)
    fracs_b = _compute_action_fracs(sim_b, inv_grid, inv_max)

    # --- Row 0: Oracle A action heatstrips (most frequent action per inv) ---
    ax_first = None
    for r in range(N_REGIMES):
        ax = fig.add_subplot(gs[0, r])
        dominant = np.argmax(fracs_a[r], axis=-1)
        _action_heatstrip(ax, dominant, inv_grid,
                          f"Oracle A — {REGIME_NAMES[r]}")
        if r == 0:
            ax_first = ax
    # Legend on existing first axis
    for a in range(N_ACTIONS):
        ax_first.plot([], [], 's', color=ACTION_COLORS[a], ms=10,
                      label=ACTION_LABELS[a])
    ax_first.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # --- Row 1: Oracle A relative gap (clamped) ---
    Q_a = np.array(oracle_a_sol.Q)
    Q_sorted = np.sort(Q_a, axis=-1)
    gap_a = (Q_sorted[:, :, -1] - Q_sorted[:, :, -2]) / q_max * 100.0
    p90 = max(np.percentile(gap_a[r], 90) for r in range(N_REGIMES))
    y_max = min(max(p90 * 1.3, 15), 80)

    for r in range(N_REGIMES):
        ax = fig.add_subplot(gs[1, r])
        gap = gap_a[r]
        clipped = gap > y_max
        display = np.minimum(gap, y_max)
        ax.bar(inv_grid, display, color=ACTION_COLORS[r], alpha=0.85,
               width=0.75, edgecolor="white", linewidth=0.5)
        for qi in np.where(clipped)[0]:
            ax.annotate(f"{gap[qi]:.0f}%", xy=(inv_grid[qi], y_max),
                        fontsize=6, ha="center", va="bottom",
                        color=ACTION_COLORS[r], fontweight="bold")
        ax.axhline(5.0, color="#C44E52", linestyle="--", linewidth=1.5,
                   alpha=0.8, zorder=5)
        ax.set_xlim(inv_grid[0] - 0.6, inv_grid[-1] + 0.6)
        ax.set_ylim(0, y_max * 1.08)
        ax.set_xlabel("Inventory q")
        if r == 0:
            ax.set_ylabel("Relative gap (% Q_max)")
        mean_g = float(np.mean(gap))
        ax.set_title(f"Oracle A gap — {REGIME_NAMES[r]}", fontsize=10)
        ax.text(0.97, 0.95, f"mean = {mean_g:.1f}%",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    # --- Row 2: Oracle B action heatstrips ---
    for r in range(N_REGIMES):
        ax = fig.add_subplot(gs[2, r])
        dominant = np.argmax(fracs_b[r], axis=-1)
        _action_heatstrip(ax, dominant, inv_grid,
                          f"Oracle B — {REGIME_NAMES[r]}")

    # --- Row 3: 3×3 action-distribution matrices (Locked / Oracle A / Oracle B) ---
    # Locked matrix: deterministic policy → one-hot per regime
    locked_dist = np.zeros((N_REGIMES, N_ACTIONS))
    for r in range(N_REGIMES):
        pol = np.array(locked_solutions[r].policy)
        for a in range(N_ACTIONS):
            locked_dist[r, a] = np.mean(pol == a)

    _action_dist_matrix(
        fig.add_subplot(gs[3, 0]),
        locked_dist,
        "Locked-Regime Policies")

    _action_dist_matrix(
        fig.add_subplot(gs[3, 1]),
        pc2["action_distributions"],
        "Oracle A (switching)")

    _action_dist_matrix(
        fig.add_subplot(gs[3, 2]),
        pc3["action_distributions"],
        "Oracle B (belief uncertainty)")

    # --- Row 4: Disagreement comparison bars (side-by-side) ---
    ax_d = fig.add_subplot(gs[4, :2])

    # Three groups: A-vs-Locked per regime, B-vs-A per regime
    labels = []
    a_vals = []
    b_vals = []
    for r in range(N_REGIMES):
        labels.append(REGIME_NAMES[r])
        a_vals.append(pc2["disagree_vs_locked"][r])
        b_vals.append(pc3["disagree_vs_oracle_a"][r])

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax_d.bar(x - w/2, a_vals, w, label="A vs Locked",
                     color="#6A5ACD", edgecolor="white")
    bars2 = ax_d.bar(x + w/2, b_vals, w, label="B vs A",
                     color="#CD853F", edgecolor="white")
    ax_d.axhline(20.0, color="#C44E52", linestyle="--", linewidth=1.5,
                  alpha=0.8, label="20% gate")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=10)
    ax_d.set_ylabel("Disagreement (%)")
    ax_d.set_title("Pairwise Disagreement Comparison", fontsize=11)
    ax_d.legend(fontsize=9)
    ax_d.grid(axis="y", alpha=0.2)
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax_d.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                      f"{h:.1f}%", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")

    # Summary text panel
    ax_txt = fig.add_subplot(gs[4, 2])
    ax_txt.axis("off")
    lines = [
        "Oracle A dominant actions:",
        f"  Noise→a{pc2['dominant_actions'][0]}  "
        f"Bull→a{pc2['dominant_actions'][1]}  "
        f"Bear→a{pc2['dominant_actions'][2]}",
        "",
        f"Overall mean gap: {pc2['overall_mean_gap']:.1f}%",
        "",
        "Oracle B vs A disagree:",
    ]
    for r in range(N_REGIMES):
        lines.append(
            f"  {REGIME_NAMES[r]}: {pc3['disagree_vs_oracle_a'][r]:.1f}%")
    ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                          edgecolor="gray"))

    passed = pc2["passed"] and pc3["passed"]
    status = "PASS" if passed else "FAIL"
    scolor = "green" if passed else "red"
    fig.suptitle(f"Figure 2 — Mixed-Regime Oracles   [{status}]",
                 fontsize=14, fontweight="bold", color=scolor, y=1.0)

    path = os.path.join(PLOTS_DIR, "figure2_mixed_regime.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(pc1, pc2, pc3, q_max, locked_solutions, oracle_a_sol):
    """Save analytical foundation results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert numpy/JAX arrays to JSON-safe types
    def jsonify(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    def to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    result = {
        "q_max": jsonify(q_max),
        "precondition_1": {
            "gap_per_regime": {
                REGIME_NAMES[r]: {
                    "mean": jsonify(pc1["gap_per_regime"][r]["mean"]),
                    "min": jsonify(pc1["gap_per_regime"][r]["min"]),
                    "values": to_list(pc1["gap_per_regime"][r]["values"]),
                }
                for r in range(N_REGIMES)
            },
            "pairwise_disagreement": {
                f"{REGIME_NAMES[r1]}_vs_{REGIME_NAMES[r2]}": jsonify(
                    pc1["pairwise_disagreement"][(r1, r2)])
                for r1, r2 in pc1["pairwise_disagreement"]
            },
            "pass_gap": bool(pc1["pass_gap"]),
            "pass_disagreement": bool(pc1["pass_disagreement"]),
            "passed": bool(pc1["passed"]),
        },
        "precondition_2": {
            "action_distributions": {
                REGIME_NAMES[r]: to_list(pc2["action_distributions"][r])
                for r in range(N_REGIMES)
            },
            "gap_per_regime": {
                REGIME_NAMES[r]: {
                    "mean": jsonify(pc2["gap_per_regime"][r]["mean"]),
                    "min": jsonify(pc2["gap_per_regime"][r]["min"]),
                    "values": to_list(pc2["gap_per_regime"][r]["values"]),
                }
                for r in range(N_REGIMES)
            },
            "disagree_vs_locked": {
                REGIME_NAMES[r]: jsonify(pc2["disagree_vs_locked"][r])
                for r in range(N_REGIMES)
            },
            "overall_mean_gap": jsonify(pc2["overall_mean_gap"]),
            "dominant_actions": pc2["dominant_actions"],
            "pass_gap": bool(pc2["pass_gap"]),
            "pass_separation": bool(pc2["pass_separation"]),
            "passed": bool(pc2["passed"]),
        },
        "precondition_3": {
            "action_distributions": {
                REGIME_NAMES[r]: to_list(pc3["action_distributions"][r])
                for r in range(N_REGIMES)
            },
            "disagree_vs_oracle_a": {
                REGIME_NAMES[r]: jsonify(pc3["disagree_vs_oracle_a"][r])
                for r in range(N_REGIMES)
            },
            "pass_separation": bool(pc3["pass_separation"]),
            "passed": bool(pc3["passed"]),
        },
        "gate": {
            "all_passed": bool(pc1["passed"] and pc2["passed"] and pc3["passed"]),
            "precondition_1": bool(pc1["passed"]),
            "precondition_2": bool(pc2["passed"]),
            "precondition_3": bool(pc3["passed"]),
        },
        "locked_policies": {
            REGIME_NAMES[r]: to_list(np.array(locked_solutions[r].policy))
            for r in range(N_REGIMES)
        },
        "oracle_a_policy": to_list(np.array(oracle_a_sol.policy)),
    }

    path = os.path.join(RESULTS_DIR, "analytical_foundation.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> {os.path.relpath(path, ROOT)}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Reduce episodes and grid size for quick validation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 3 — Analytical Foundation")
    print("=" * 60)

    params = EnvParams.default()
    grid_size = 10 if args.fast else 20
    n_episodes = 500 if args.fast else 2000

    # --- Solve ---
    print("\n  Building MDP tables ...")
    t0 = time.time()
    tables = build_mdp_tables(params)
    print(f"  {(time.time()-t0)*1000:.0f} ms")

    print("  Solving locked-regime VI ...")
    t0 = time.time()
    locked = solve_all_locked(tables, params)
    q_max = compute_q_max(locked)
    print(f"  {(time.time()-t0)*1000:.0f} ms  Q_max={q_max:.4f}")

    print("  Solving Oracle A ...")
    t0 = time.time()
    oracle_a = solve_oracle_a(tables, params)
    print(f"  {(time.time()-t0)*1000:.0f} ms  ({oracle_a.n_iters} iters)")

    print(f"  Solving Oracle B (grid={grid_size}) ...")
    t0 = time.time()
    oracle_b = solve_oracle_b(tables, params, grid_size=grid_size, verbose=True)
    print(f"  {time.time()-t0:.1f} s  ({oracle_b.n_iters} iters)")

    # --- Precondition 1 ---
    print("\n  Precondition 1 — Locked-regime policies ...")
    pc1 = precondition_1(locked, q_max)
    for r in range(N_REGIMES):
        g = pc1["gap_per_regime"][r]
        print(f"    {REGIME_NAMES[r]:>5s}: mean gap={g['mean']:.1f}%, min={g['min']:.1f}%")
    for (r1, r2), d in pc1["pairwise_disagreement"].items():
        print(f"    {REGIME_NAMES[r1]} vs {REGIME_NAMES[r2]}: {d:.1f}% disagree")
    print(f"    -> {'PASS' if pc1['passed'] else 'FAIL'}")

    # --- Precondition 2 ---
    print(f"\n  Precondition 2 — Oracle A rollout ({n_episodes} episodes) ...")
    t0 = time.time()
    sim_a = simulate_oracle_a(oracle_a, params, n_episodes=n_episodes)
    print(f"  {time.time()-t0:.1f} s")
    pc2 = precondition_2(sim_a, oracle_a, locked, q_max, params)
    for r in range(N_REGIMES):
        dist = pc2["action_distributions"][r]
        dist_str = " / ".join(f"{v:.1%}" for v in dist)
        print(f"    {REGIME_NAMES[r]:>5s}: actions=[{dist_str}]  "
              f"gap={pc2['gap_per_regime'][r]['mean']:.1f}%")
    print(f"    Overall mean gap: {pc2['overall_mean_gap']:.1f}%")
    print(f"    Dominant actions: {pc2['dominant_actions']}")
    print(f"    -> {'PASS' if pc2['passed'] else 'FAIL'}")

    # --- Precondition 3 ---
    print(f"\n  Precondition 3 — Oracle B rollout ({n_episodes} episodes) ...")
    t0 = time.time()
    sim_b = simulate_oracle_b(oracle_b, tables, params,
                               grid_size=grid_size, n_episodes=n_episodes)
    print(f"  {time.time()-t0:.1f} s")
    pc3 = precondition_3(sim_b, oracle_b, oracle_a, q_max, params)
    for r in range(N_REGIMES):
        dist = pc3["action_distributions"][r]
        dist_str = " / ".join(f"{v:.1%}" for v in dist)
        print(f"    {REGIME_NAMES[r]:>5s}: actions=[{dist_str}]  "
              f"B-vs-A disagree={pc3['disagree_vs_oracle_a'][r]:.1f}%")
    print(f"    -> {'PASS' if pc3['passed'] else 'FAIL'}")

    # --- Gate ---
    gate = pc1["passed"] and pc2["passed"] and pc3["passed"]
    print("\n" + "=" * 60)
    print(f"  GATE: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  WARNING: Gate failed — retune kappa/sigma_sq before proceeding!")
    print("=" * 60)

    # --- Save & Plot ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n  Saving results ...")
    save_results(pc1, pc2, pc3, q_max, locked, oracle_a)

    print("  Plotting Figure 1 ...")
    plot_figure1(locked, q_max, params, pc1)

    print("  Plotting Figure 2 ...")
    plot_figure2(sim_a, sim_b, oracle_a, locked, q_max, params, pc2, pc3)

    print("\n  Done.")


if __name__ == "__main__":
    main()
