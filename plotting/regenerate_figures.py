"""Regenerate every figure for a milestone from existing JSONs.

Called by scripts/make_milestone.py, also usable standalone:
    uv run python -m plotting.regenerate_figures M0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plotting.learning_curves import plot_learning_curve
from plotting.m1_plots import plot_learning_curve_with_ceiling, plot_policy_vs_as
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


_HANDLERS = {
    "M0": regenerate_m0,
    "M1": regenerate_m1,
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
