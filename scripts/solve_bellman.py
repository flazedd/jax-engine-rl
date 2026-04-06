#!/usr/bin/env python3
"""Solve the analytical market making MDP via Bellman equations.

Outputs:
  results/bellman_solution.json  — full solution (policy, Q, V, config, diagnostics)
  plots/bellman_solution.png     — 6-panel visualization

Usage:
    uv run python scripts/solve_bellman.py [--fast]
"""
import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Resolve paths relative to project root (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lob_sim.analytical_mdp import (
    MDPConfig, build_mdp_tables, solve_full_info, solve_pomdp_belief,
    print_policy, print_fill_probs, value_of_info, stationary_distribution,
    simulate_episodes,
    _build_belief_grid, _nearest_belief, _get_action_table,
    N_REGIMES,
)

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")


def save_results(cfg, tables, sol, sol_p=None):
    """Save full solution to results/bellman_solution.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    at = _get_action_table(cfg)
    n_actions = len(at)
    mi = cfg.max_inv
    n_inv = 2 * mi + 1
    inv_grid = list(range(-mi, mi + 1))
    regime_names = ["noise", "bull", "bear"]

    # --- Bellman residual diagnostics ---
    diagnostics = {
        "full_info": {
            "n_iterations": sol.n_iters,
            "final_residual": sol.residuals[-1],
            "converged": sol.residuals[-1] < 1e-8,
            "max_bellman_residual": float(np.max(np.abs(
                sol.values - np.max(sol.Q, axis=-1)))),
        },
    }
    if sol_p is not None:
        diagnostics["pomdp"] = {
            "n_iterations": sol_p.n_iters,
            "final_residual": sol_p.residuals[-1],
            "converged": sol_p.residuals[-1] < 1e-6,
            "n_belief_points": cfg.n_belief_points,
            "n_belief_states": len(_build_belief_grid(cfg.n_belief_points)),
        }

    # --- Per-state Q-value detail ---
    q_values = {}
    for r in range(N_REGIMES):
        for qi in range(n_inv):
            q = inv_grid[qi]
            Q_vec = sol.Q[r, qi].tolist()
            best = int(np.argmax(sol.Q[r, qi]))
            sorted_Q = sorted(Q_vec, reverse=True)
            gap = sorted_Q[0] - sorted_Q[1]
            q_values[f"{regime_names[r]}_q{q:+d}"] = {
                "Q": [round(v, 8) for v in Q_vec],
                "optimal_action": best,
                "optimal_offsets": [int(at[best, 0]), int(at[best, 1])],
                "advantage": round(gap, 8),
                "value": round(float(sol.values[r, qi]), 8),
            }

    # --- Fill probability table ---
    fills = tables.fill_probs
    fill_table = {}
    for r in range(N_REGIMES):
        for a in range(n_actions):
            fill_table[f"{regime_names[r]}_a{a}"] = {
                "offsets": [int(at[a, 0]), int(at[a, 1])],
                "p_bid": round(float(fills.bid[r, a]), 6),
                "p_ask": round(float(fills.ask[r, a]), 6),
                "net_inventory_flow": round(
                    float(fills.bid[r, a] - fills.ask[r, a]), 6),
            }

    # --- Optimality certificate ---
    # Verify Bellman equation: V(s) == max_a Q(s,a) everywhere
    bellman_err = float(np.max(np.abs(sol.values - np.max(sol.Q, axis=-1))))

    # Policy stability: is the argmax unique (no ties within tolerance)?
    sorted_Q = np.sort(sol.Q, axis=-1)
    min_advantage = float(np.min(sorted_Q[:, :, -1] - sorted_Q[:, :, -2]))

    data = {
        "description": (
            "Optimal policy for the analytical market making MDP with "
            "HMM regime switching. Solved via exact value iteration over "
            "the Bellman equation (no simulation or Monte Carlo)."
        ),
        "config": {
            "max_inventory": mi,
            "tick_size": cfg.tick_size,
            "half_spread_ticks": cfg.half_spread_ticks,
            "gamma": cfg.gamma,
            "fill_decay_kappa": cfg.fill_decay,
            "sell_arrival_rates": list(cfg.sell_arrival),
            "buy_arrival_rates": list(cfg.buy_arrival),
            "drift_per_step": list(cfg.drift),
            "inventory_penalty_phi": cfg.inv_penalty,
        },
        "actions": [
            {"index": a,
             "bid_offset": int(at[a, 0]),
             "ask_offset": int(at[a, 1]),
             "label": f"({int(at[a,0])},{int(at[a,1])})"}
            for a in range(n_actions)
        ],
        "regime_transition_matrix": np.array(
            tables.trans_regime).tolist(),
        "fill_probabilities": fill_table,
        "optimality_certificate": {
            "bellman_residual": bellman_err,
            "min_action_advantage": min_advantage,
            "bellman_satisfied": bellman_err < 1e-8,
            "policy_unique": min_advantage > 1e-10,
        },
        "diagnostics": diagnostics,
        "full_info_solution": {
            "inventory_grid": inv_grid,
            "regime_names": regime_names,
            "policy": sol.policy.tolist(),
            "values": [[round(v, 8) for v in row]
                       for row in sol.values.tolist()],
            "Q_values": [[[round(v, 8) for v in q_vec]
                          for q_vec in regime_q]
                         for regime_q in sol.Q.tolist()],
        },
        "per_state_detail": q_values,
    }

    if sol_p is not None:
        voi = value_of_info(tables, sol, sol_p)
        pi = stationary_distribution(tables)
        grid = _build_belief_grid(cfg.n_belief_points)
        bi_s = _nearest_belief(pi, grid)
        data["pomdp_solution"] = {
            "value_of_information_at_inv0": round(voi, 8),
            "pomdp_value_stationary_belief_inv0": round(
                float(sol_p.values[bi_s, mi]), 8),
            "full_info_stationary_value_inv0": round(
                float(pi @ sol.values[:, mi]), 8),
            "stationary_distribution": [round(float(p), 6) for p in pi],
            "policy_at_inv0": sol_p.policy[:, mi].tolist(),
        }

    if sol_p is not None:
        sim_f = simulate_episodes(tables, "full_info", full_sol=sol, seed=42)
        sim_p = simulate_episodes(tables, "pomdp", pomdp_sol=sol_p, seed=42)
        sim_b = simulate_episodes(tables, "blind", seed=42)
        n_steps = len(sim_f.mean_reward)
        data["simulation"] = {
            "n_episodes": 500,
            "n_steps": n_steps,
            "seed": 42,
            "per_step_reward": {
                "full_info": round(float(sim_f.mean_reward[-1] / n_steps), 6),
                "pomdp": round(float(sim_p.mean_reward[-1] / n_steps), 6),
                "regime_blind": round(float(sim_b.mean_reward[-1] / n_steps), 6),
            },
            "cumulative_reward_200": {
                "full_info": round(float(sim_f.mean_reward[-1]), 2),
                "pomdp": round(float(sim_p.mean_reward[-1]), 2),
                "regime_blind": round(float(sim_b.mean_reward[-1]), 2),
            },
        }

    # Per-regime locked simulations (full-info optimal vs blind)
    data["per_regime"] = {}
    for r, rname in enumerate(regime_names):
        sim_opt = simulate_episodes(
            tables, "full_info", full_sol=sol, seed=42, locked_regime=r)
        sim_bld = simulate_episodes(
            tables, "blind", seed=42, locked_regime=r)
        ns = len(sim_opt.mean_reward)
        data["per_regime"][rname] = {
            "optimal_per_step": round(float(sim_opt.mean_reward[-1] / ns), 6),
            "blind_per_step": round(float(sim_bld.mean_reward[-1] / ns), 6),
            "optimal_policy_at_inv0": int(sol.policy[r, mi]),
        }

    path = os.path.join(RESULTS_DIR, "bellman_solution.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {os.path.relpath(path, ROOT)}")
    return data


def plot_results(cfg, tables, sol, sol_p=None):
    """Generate plots/bellman_solution.png."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    at = _get_action_table(cfg)
    n_actions = len(at)
    n_inv = 2 * cfg.max_inv + 1
    inv = np.arange(n_inv) - cfg.max_inv
    names = ["Noise", "Bull", "Bear"]
    labels = [f"({int(at[a,0])},{int(at[a,1])})"
              for a in range(n_actions)]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    cmap3 = mcolors.ListedColormap(colors)

    ncols = 3 if sol_p else 2
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
    axes = axes.flatten()

    # 1 — Policy heatmap
    ax = axes[0]
    ax.imshow(sol.policy, aspect="auto", cmap=cmap3, vmin=0, vmax=2,
              interpolation="nearest",
              extent=[inv[0]-0.5, inv[-1]+0.5, N_REGIMES-0.5, -0.5])
    ax.set_yticks(range(N_REGIMES)); ax.set_yticklabels(names)
    ax.set_xlabel("Inventory"); ax.set_title("Optimal Policy (full info)")
    for a in range(n_actions):
        ax.plot([], [], 's', color=colors[a], ms=8, label=f"a{a}={labels[a]}")
    ax.legend(loc="lower right", fontsize=7)

    # 2 — Value function
    ax = axes[1]
    for r in range(N_REGIMES):
        ax.plot(inv, sol.values[r], label=names[r], lw=2)
    ax.set_xlabel("Inventory"); ax.set_ylabel("V(r, q)")
    ax.set_title("Value Function"); ax.legend(); ax.grid(alpha=0.3)

    # 3 — Q-gap
    ax = axes[2]
    sQ = np.sort(sol.Q, axis=-1)
    gap = sQ[:, :, -1] - sQ[:, :, -2]
    for r in range(N_REGIMES):
        ax.plot(inv, gap[r], label=names[r], lw=2)
    ax.set_xlabel("Inventory"); ax.set_ylabel("Q(best) - Q(2nd)")
    ax.set_title("Action Advantage"); ax.legend(); ax.grid(alpha=0.3)

    # 4 — Convergence
    ax = axes[ncols]
    ax.semilogy(sol.residuals, label="Full-info", lw=2)
    if sol_p:
        ax.semilogy(sol_p.residuals, label="POMDP", lw=2, alpha=0.7)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Bellman Residual")
    ax.set_title("Convergence"); ax.legend(); ax.grid(alpha=0.3)

    if sol_p:
        # 5 — POMDP simplex
        ax = axes[ncols + 1]
        grid = _build_belief_grid(cfg.n_belief_points)
        q0 = cfg.max_inv
        verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        xy = grid @ verts
        ax.scatter(xy[:, 0], xy[:, 1], c=sol_p.policy[:, q0], cmap=cmap3,
                   vmin=0, vmax=2, s=20, edgecolors='none')
        for i, n in enumerate(names):
            ax.annotate(n, verts[i], fontsize=9, ha='center', fontweight='bold',
                        xytext=(0, -12 if i < 2 else 10),
                        textcoords='offset points')
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.15, 1.05)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("POMDP Policy at inv=0")

        # 6 — Value of info
        ax = axes[ncols + 2]
        pi = stationary_distribution(tables)
        v_full = pi @ sol.values  # stationary-weighted full-info value
        bi_s = _nearest_belief(pi, grid)
        voi = v_full - sol_p.values[bi_s]
        ax.plot(inv, voi, lw=2, color='#C44E52')
        ax.fill_between(inv, 0, voi, alpha=0.2, color='#C44E52')
        ax.set_xlabel("Inventory"); ax.set_ylabel("V_full - V_pomdp")
        ax.set_title("Value of Information"); ax.grid(alpha=0.3)
    else:
        ax = axes[ncols + 1]
        R = tables.reward; q0 = cfg.max_inv
        x = np.arange(n_actions); w = 0.25
        for ri in range(N_REGIMES):
            ax.bar(x + ri*w, R[ri, q0, :], w, label=names[ri], color=colors[ri])
        ax.set_xticks(x + w); ax.set_xticklabels(labels)
        ax.set_ylabel("R(r, q=0, a)"); ax.set_title("Reward at inv=0")
        ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "bellman_solution.png")
    plt.savefig(path, dpi=150)
    print(f"  → {os.path.relpath(path, ROOT)}")


def plot_trajectories(tables, sol, sol_p):
    """Generate plots/reward_trajectories.png — cumulative reward comparison."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n  Simulating episodes ...")
    t0 = time.time()
    sim_full = simulate_episodes(tables, "full_info", full_sol=sol, seed=42)
    sim_pomdp = simulate_episodes(tables, "pomdp", pomdp_sol=sol_p, seed=42)
    sim_blind = simulate_episodes(tables, "blind", seed=42)
    print(f"  {time.time()-t0:.1f} s  (3 × 500 episodes × 200 steps)")

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(1, len(sim_full.mean_reward) + 1)

    for sim, label, color in [
        (sim_full, "Full-info (regime observed)", "#55A868"),
        (sim_pomdp, "POMDP (regime hidden)", "#4C72B0"),
        (sim_blind, "Regime-blind (always a0)", "#C44E52"),
    ]:
        ax.plot(steps, sim.mean_reward, label=label, lw=2, color=color)
        ax.fill_between(steps,
                        sim.mean_reward - sim.std_reward,
                        sim.mean_reward + sim.std_reward,
                        alpha=0.15, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Trajectories: Full-info vs POMDP vs Regime-blind")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "reward_trajectories.png")
    plt.savefig(path, dpi=150)
    print(f"  → {os.path.relpath(path, ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Analytical Market Making MDP — Bellman Solver")
    print("=" * 60)

    cfg = MDPConfig()
    if args.fast:
        cfg = cfg._replace(n_belief_points=11)

    t0 = time.time()
    tables = build_mdp_tables(cfg)
    at = _get_action_table(cfg)
    n_actions = len(at)
    n_inv = 2 * cfg.max_inv + 1
    print(f"\n  Built in {(time.time()-t0)*1000:.1f} ms")
    print(f"  States: {N_REGIMES}×{n_inv} = {N_REGIMES * n_inv}   "
          f"Actions: {n_actions}   γ={cfg.gamma}")
    print(f"  κ={cfg.fill_decay}   φ={cfg.inv_penalty}   "
          f"drift={cfg.drift}")
    print()
    print_fill_probs(tables)

    # Full info
    print("\n  Solving full-information MDP ...")
    t0 = time.time()
    sol = solve_full_info(tables, verbose=True)
    print(f"  {(time.time()-t0)*1000:.1f} ms  ({sol.n_iters} iters)")
    print("\n  Optimal policy:")
    print_policy(sol, tables)

    # POMDP
    sol_p = None
    if not args.fast:
        print("\n  Solving POMDP (regime hidden) ...")
        t0 = time.time()
        sol_p = solve_pomdp_belief(tables, verbose=True)
        print(f"  {time.time()-t0:.1f} s  ({sol_p.n_iters} iters)")
        print(f"  Value of information: {value_of_info(tables, sol, sol_p):.6f}")

    # Save results
    print()
    data = save_results(cfg, tables, sol, sol_p)

    # Verify optimality
    cert = data["optimality_certificate"]
    print(f"\n  Optimality certificate:")
    print(f"    Bellman residual:       {cert['bellman_residual']:.2e}  "
          f"({'PASS' if cert['bellman_satisfied'] else 'FAIL'})")
    print(f"    Min action advantage:   {cert['min_action_advantage']:.2e}  "
          f"({'unique' if cert['policy_unique'] else 'ties exist'})")

    # Plot
    plot_results(cfg, tables, sol, sol_p)
    if sol_p:
        plot_trajectories(tables, sol, sol_p)

    # Summary
    names = ["Noise", "Bull", "Bear"]
    print("\n" + "=" * 60)
    q0 = cfg.max_inv
    for r in range(N_REGIMES):
        a = sol.policy[r, q0]
        print(f"  {names[r]:>5s} inv=0: a{a}=({int(at[a,0])},{int(at[a,1])})  "
              f"V={sol.values[r,q0]:.4f}")
    print()


if __name__ == "__main__":
    main()
