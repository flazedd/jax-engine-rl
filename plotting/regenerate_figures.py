"""Regenerate every figure for a milestone from existing JSONs.

Called by scripts/make_milestone.py, also usable standalone:
    uv run python -m plotting.regenerate_figures M0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plotting.learning_curves import plot_learning_curve
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


_HANDLERS = {
    "M0": regenerate_m0,
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
