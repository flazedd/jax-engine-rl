"""Regenerate every figure for a milestone from existing JSONs.

Called by scripts/make_milestone.py, also usable standalone:
    uv run python -m plotting.regenerate_figures M0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json

import numpy as np

from envs.mm_reduced import MMReducedEnv
from oracles.value_iteration import (
    compromise_policy_expected_returns,
    policy_disagreement,
    solve_value_iteration,
    wrong_regime_value_loss,
)
from plotting.learning_curves import plot_learning_curve
from plotting.m1_plots import plot_learning_curve_with_ceiling, plot_policy_vs_as
from plotting.m2_plots import (
    plot_belief_ppo_gap,
    plot_mixed_gap,
    plot_per_regime_ppo,
    plot_policy_heatmap,
    plot_posterior_entropy,
    plot_value_loss_distribution,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"


def regenerate_m0(run: ScriptRun) -> dict:
    experiment_dir = RESULTS_ROOT / "m0_dummy"
    output = FIGURES_ROOT / "milestones" / "M0" / "fig_M0_dummy_learning_curve.png"
    stats = plot_learning_curve(
        experiment_dir, output, title="M0 — dummy agent / dummy env (pipeline smoke test)"
    )
    run.add_output(str(output))
    return {"m0_dummy": stats}


def regenerate_m1(run: ScriptRun) -> dict:
    experiment_dir = RESULTS_ROOT / "m1_ppo_as"
    fig_dir = FIGURES_ROOT / "milestones" / "M1"
    curve_out = fig_dir / "fig_M1_ppo_learning_curve.png"
    policy_out = fig_dir / "fig_M1_ppo_policy_vs_as.png"

    curve_stats = plot_learning_curve_with_ceiling(experiment_dir, curve_out)
    run.add_output(str(curve_out))

    policy_stats = plot_policy_vs_as(experiment_dir, policy_out)
    run.add_output(str(policy_out))

    return {"m1_learning_curve": curve_stats, "m1_policy_vs_as": policy_stats}


def regenerate_m2(run: ScriptRun) -> dict:
    stats_path = RESULTS_ROOT / "milestones" / "M2" / "stats_M2_requirements.json"
    with open(stats_path) as f:
        stats = json.load(f)

    env_version = stats["env_version"]
    env = MMReducedEnv(**stats["parameters"])
    vi = solve_value_iteration(env)

    fig_dir = FIGURES_ROOT / "milestones" / "M2"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # R1 figures.
    p_heatmap = fig_dir / "fig_M2_R1_policy_heatmap.png"
    plot_policy_heatmap(vi, env, p_heatmap)
    run.add_output(str(p_heatmap))

    _, _, per_state_loss = wrong_regime_value_loss(env, vi)
    p_loss = fig_dir / "fig_M2_R1_value_loss_distribution.png"
    plot_value_loss_distribution(per_state_loss, p_loss)
    run.add_output(str(p_loss))

    # R2 figure — per-regime learning curves from per-experiment metrics.
    per_regime_metrics = []
    for r in range(env.n_regimes):
        mpath = RESULTS_ROOT / f"m2_verify_{env_version}_per_regime_{r}" / "metrics.json"
        with open(mpath) as f:
            per_regime_metrics.append(json.load(f))
    per_regime_optima = np.asarray(
        stats["vi"]["per_regime_expected_episode_return"], dtype=float
    )
    p_curves = fig_dir / "fig_M2_R2_per_regime_ppo.png"
    plot_per_regime_ppo(per_regime_metrics, per_regime_optima, p_curves)
    run.add_output(str(p_curves))

    # R3 figure — mixed-gap bars (regime-agnostic, oracle, per-regime mean).
    r3 = stats["R3_mixed_gap"]
    per_regime_means = [float(m["final_return_mean"]) for m in per_regime_metrics]
    per_regime_ci_lo = [float(m["final_return_ci95"][0]) for m in per_regime_metrics]
    per_regime_ci_hi = [float(m["final_return_ci95"][1]) for m in per_regime_metrics]
    p_mixed = fig_dir / "fig_M2_R3_mixed_gap.png"
    plot_mixed_gap(
        {
            "regime_agnostic": (r3["regime_agnostic_return_mean"], r3["regime_agnostic_return_ci"]),
            "oracle": (r3["oracle_ppo_return_mean"], r3["oracle_ppo_return_ci"]),
            "per_regime_mean": (
                float(np.mean(per_regime_means)),
                [float(np.mean(per_regime_ci_lo)), float(np.mean(per_regime_ci_hi))],
            ),
        },
        p_mixed,
    )
    run.add_output(str(p_mixed))

    # R4 figures.
    r4 = stats["R4_inferability"]
    # Re-simulate entropy curve — this is cheap (random-policy scan).
    from oracles.verify_requirements import (
        _entropy_over_time,
        _simulate_belief_trajectories,
    )
    beliefs = _simulate_belief_trajectories(env, n_envs=256, seed=0)
    ent_curve = _entropy_over_time(beliefs)
    p_ent = fig_dir / "fig_M2_R4_posterior_entropy.png"
    plot_posterior_entropy(ent_curve, p_ent)
    run.add_output(str(p_ent))

    p_belief = fig_dir / "fig_M2_R4_belief_ppo_gap.png"
    plot_belief_ppo_gap(
        {
            "regime_agnostic": (
                r3["regime_agnostic_return_mean"],
                r3["regime_agnostic_return_ci"],
            ),
            "belief": (r4["belief_ppo_return_mean"], r4["belief_ppo_return_ci"]),
            "oracle": (r3["oracle_ppo_return_mean"], r3["oracle_ppo_return_ci"]),
        },
        p_belief,
    )
    run.add_output(str(p_belief))

    return {
        "env_version": env_version,
        "figures": [str(p) for p in run.outputs[-6:]],
    }


_HANDLERS = {
    "M0": regenerate_m0,
    "M1": regenerate_m1,
    "M2": regenerate_m2,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="plotting.regenerate_figures")
    parser.add_argument("milestone", help="milestone id, e.g. M0")
    args = parser.parse_args()

    run = ScriptRun(script="regenerate_figures")
    handler = _HANDLERS.get(args.milestone)
    summary_dir = RESULTS_ROOT / "milestones" / args.milestone
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"plot_summary_{args.milestone}.json"
    if handler is None:
        run.fail(
            reason=f"no figure handler for milestone {args.milestone}",
            summary_path=summary_path,
        )
        raise SystemExit(1)

    stats = handler(run)
    run.ok(
        key_stats={"milestone": args.milestone, "figures": len(run.outputs)},
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()
