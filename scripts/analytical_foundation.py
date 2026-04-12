#!/usr/bin/env python3
"""Analytical Foundation — Per-Regime Optimal Policies + Oracle Bounds.

Solves the locked-regime MDP for each regime via value iteration,
verifies that optimal policies are distinct across regimes,
runs the oracle policy to compute episode return statistics,
and plots per-regime optimal policy heatmaps + oracle bounds.

Outputs:
  results/analytical_foundation.json  — policy data + gate + oracle bounds
  plots/figure1_optimal_policies.png  — per-regime policy heatmaps
  plots/figure2_optimal_bounds.png    — oracle episode return distributions

Usage:
    uv run python scripts/analytical_foundation.py [--fast]
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lob_sim.jax_env import EnvParams, rollout_episode
from lob_sim.analytical_mdp import (
    N_REGIMES, N_ACTIONS, REGIME_NAMES,
    build_mdp_tables, solve_all_locked,
    compute_q_max, precondition_1,
)

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")

ACTION_LABELS = ["sym(1,1)", "ask(1,3)", "bid(3,1)"]


# ---------------------------------------------------------------------------
# Oracle rollout
# ---------------------------------------------------------------------------

def run_oracle_episodes(locked_solutions, params, n_batches, batch_size):
    """Run oracle policy per isolated regime in K batches of N episodes.

    Each batch produces a mean episode return. We report the mean and std
    of those batch means — the std reflects how stable the estimate is,
    not per-episode variance.

    All evaluations use init_inventory=0.
    """
    inv_max = params.inventory_max
    eval_params = params._replace(init_inventory=0)
    results = {}

    for r in range(N_REGIMES):
        oracle_pol = jnp.array(locked_solutions[r].policy, dtype=jnp.int32)
        ep = eval_params._replace(locked_regime=r)

        def policy_fn(k, obs, _pol=oracle_pol):
            return _pol[jnp.int32(obs[3] + inv_max)]

        batch_means = []
        for b in range(n_batches):
            keys = jax.random.split(jax.random.PRNGKey(b), batch_size)
            traj = jax.vmap(rollout_episode, in_axes=(0, None, None))(
                keys, policy_fn, ep)
            rets = jnp.sum(traj["rewards"], axis=1)
            batch_means.append(float(jnp.mean(rets)))

        batch_means = np.array(batch_means)
        results[REGIME_NAMES[r].lower()] = {
            "mean": float(batch_means.mean()),
            "std": float(batch_means.std()),
            "n_batches": n_batches,
            "batch_size": batch_size,
        }

    return results


# ---------------------------------------------------------------------------
# Figure 1 — Per-Regime Optimal Policy Heatmaps
# ---------------------------------------------------------------------------

def plot_optimal_policies(locked_solutions, q_max, params, pc1):
    """3 heatmaps (one per regime): x=inventory, y=action, cell=optimal %."""
    inv_max = int(params.inventory_max)
    n_inv = 2 * inv_max + 1
    inv_grid = np.arange(n_inv) - inv_max

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5), sharey=True)

    for r in range(N_REGIMES):
        ax = axes[r]
        policy = np.array(locked_solutions[r].policy)

        grid = np.zeros((N_ACTIONS, n_inv))
        for qi in range(n_inv):
            grid[int(policy[qi]), qi] = 1.0

        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_inv))
        ax.set_xticklabels(inv_grid, fontsize=8)
        ax.set_xlabel("Inventory q")
        ax.set_title(REGIME_NAMES[r], fontsize=12, fontweight="bold")

        if r == 0:
            ax.set_yticks(range(N_ACTIONS))
            ax.set_yticklabels(ACTION_LABELS, fontsize=10)
            ax.set_ylabel("Action")
        else:
            ax.set_yticks(range(N_ACTIONS))

        for a in range(N_ACTIONS):
            for qi in range(n_inv):
                val = grid[a, qi]
                color = "white" if val > 0.5 else "lightgray"
                ax.text(qi, a, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    pairs = [(0, 1), (0, 2), (1, 2)]
    disag = [f"{REGIME_NAMES[r1]}–{REGIME_NAMES[r2]}: "
             f"{pc1['pairwise_disagreement'][(r1, r2)]:.0f}%"
             for r1, r2 in pairs]

    status = "PASS" if pc1["passed"] else "FAIL"
    scolor = "green" if pc1["passed"] else "red"
    fig.suptitle(
        f"Per-Regime Optimal Policies   [{status}]"
        f"      disagreement: {' | '.join(disag)}",
        fontsize=11, fontweight="bold", color=scolor, y=1.04)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure1_optimal_policies.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 — Oracle Bounds (bar chart with error bars)
# ---------------------------------------------------------------------------

def plot_oracle_bounds(oracle_stats):
    """Bar chart: mean episode return ± std per isolated regime."""
    names = ["noise", "bull", "bear"]
    labels = ["Noise", "Bull", "Bear"]
    means = [oracle_stats[n]["mean"] for n in names]
    stds = [oracle_stats[n]["std"] for n in names]
    n_bat = oracle_stats[names[0]]["n_batches"]
    bat_sz = oracle_stats[names[0]]["batch_size"]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=["C0", "C1", "C2"],
           edgecolor="black", linewidth=0.5, alpha=0.85)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1, f"{m:.1f} ± {s:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean episode return", fontsize=11)
    ax.set_title(f"Oracle Bounds ({n_bat} batches x {bat_sz} episodes)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure2_optimal_bounds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(pc1, q_max, locked_solutions, oracle_stats):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def jsonify(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    def to_list(x):
        return x.tolist() if hasattr(x, "tolist") else x

    result = {
        "q_max": jsonify(q_max),
        "precondition": {
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
            "pass_disagreement": bool(pc1["pass_disagreement"]),
            "pass_gap": bool(pc1["pass_gap"]),
            "passed": bool(pc1["passed"]),
        },
        "locked_policies": {
            REGIME_NAMES[r]: to_list(np.array(locked_solutions[r].policy))
            for r in range(N_REGIMES)
        },
        "oracle_bounds": oracle_stats,
        "gate": {"passed": bool(pc1["passed"])},
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
                        help="Fewer oracle episodes for quick validation")
    args = parser.parse_args()

    n_batches = 16 if args.fast else 64
    batch_size = 64 if args.fast else 128

    print("=" * 60)
    print("  Analytical Foundation — Per-Regime Optimal Policies")
    print("=" * 60)

    params = EnvParams.default()

    print("\n  Building MDP tables ...")
    t0 = time.time()
    tables = build_mdp_tables(params)
    print(f"  {(time.time()-t0)*1000:.0f} ms")

    print("  Solving locked-regime VI ...")
    t0 = time.time()
    locked = solve_all_locked(tables, params)
    q_max = compute_q_max(locked)
    print(f"  {(time.time()-t0)*1000:.0f} ms  Q_max={q_max:.4f}")

    print("\n  Gate check — policies distinct and decisive ...")
    pc1 = precondition_1(locked, q_max)
    for r in range(N_REGIMES):
        g = pc1["gap_per_regime"][r]
        print(f"    {REGIME_NAMES[r]:>5s}: mean gap={g['mean']:.1f}%, min={g['min']:.1f}%")
    for (r1, r2), d in pc1["pairwise_disagreement"].items():
        print(f"    {REGIME_NAMES[r1]} vs {REGIME_NAMES[r2]}: {d:.1f}% disagree")

    status = "PASS" if pc1["passed"] else "FAIL"
    print(f"\n  GATE: {status}")
    if not pc1["passed"]:
        print("  WARNING: Gate failed — retune parameters before proceeding!")

    print(f"\n  Oracle rollout ({n_batches} batches x {batch_size} episodes) ...")
    t0 = time.time()
    oracle_stats = run_oracle_episodes(locked, params, n_batches, batch_size)
    print(f"  {time.time()-t0:.1f}s")
    for name in ["noise", "bull", "bear"]:
        s = oracle_stats[name]
        print(f"    {name:>5s}: mean={s['mean']:.1f}  std={s['std']:.1f}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n  Saving results ...")
    save_results(pc1, q_max, locked, oracle_stats)

    print("  Plotting ...")
    plot_optimal_policies(locked, q_max, params, pc1)
    plot_oracle_bounds(oracle_stats)

    print("\n  Done.")


if __name__ == "__main__":
    main()
