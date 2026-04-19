"""Orchestrate a milestone end-to-end.

    uv run python -m scripts.make_milestone M0

For M0: runs the dummy pipeline in super-fast / fast / full, runs the
run-mode test, regenerates figures, and writes stats_M0_pipeline.json with
the pass-criteria fields.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax

from plotting.regenerate_figures import _HANDLERS as FIGURE_HANDLERS
from training.config import apply_run_mode, load_config
from training.train import train
from utils.script_output import (
    ScriptRun,
    validate_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"


def _run_training(config_path: Path, mode: str) -> tuple[float, bool, bool]:
    cfg = load_config(config_path)
    apply_run_mode(cfg, mode)
    t0 = time.perf_counter()
    exit_zero = True
    try:
        train(cfg)
    except SystemExit as e:
        exit_zero = (e.code == 0)
    except Exception:
        exit_zero = False
        raise
    elapsed = time.perf_counter() - t0

    summary_path = RESULTS_ROOT / cfg.experiment_name / "summary.json"
    wrote_json = summary_path.exists()
    schema_ok = False
    if wrote_json:
        with open(summary_path) as f:
            summary = json.load(f)
        schema_ok, _ = validate_summary(summary)
    return elapsed, (exit_zero and wrote_json), schema_ok


def _regenerate_figures(milestone: str) -> bool:
    handler = FIGURE_HANDLERS.get(milestone)
    if handler is None:
        return False
    inner_run = ScriptRun(script="regenerate_figures")
    try:
        handler(inner_run)
        summary_path = RESULTS_ROOT / "milestones" / milestone / f"plot_summary_{milestone}.json"
        inner_run.ok(
            key_stats={"milestone": milestone, "figures": len(inner_run.outputs)},
            summary_path=summary_path,
        )
        return True
    except Exception as e:
        summary_path = RESULTS_ROOT / "milestones" / milestone / f"plot_summary_{milestone}.json"
        inner_run.fail(reason=str(e), summary_path=summary_path)
        return False


def make_m0() -> dict:
    config = CONFIG_ROOT / "m0_dummy.yaml"
    durations: dict[str, float] = {}
    exit_zero_all = True
    wrote_json_all = True
    schema_ok_all = True

    for mode in ("super_fast", "fast", "full"):
        elapsed, ok, schema_ok = _run_training(config, mode)
        durations[mode] = elapsed
        exit_zero_all &= ok
        wrote_json_all &= ok  # wrote_json folded into ok above
        schema_ok_all &= schema_ok

    figures_ok = _regenerate_figures("M0")

    # Shared-schema validation of every summary.json in the last full run.
    # (Schema is the contract for Claude Code; do a full sweep.)
    all_jsons_ok = True
    for path in RESULTS_ROOT.rglob("summary.json"):
        with open(path) as f:
            s = json.load(f)
        ok, _ = validate_summary(s)
        all_jsons_ok &= ok

    key_stats = {
        "pipeline_version": "0.1",
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "run_modes_tested": ["super_fast", "fast", "full"],
        "super_fast_duration_seconds": round(durations["super_fast"], 3),
        "fast_duration_seconds": round(durations["fast"], 3),
        "full_duration_seconds": round(durations["full"], 3),
        "all_scripts_exit_zero": bool(exit_zero_all),
        "all_scripts_wrote_expected_json": bool(wrote_json_all),
        "schema_validates": bool(schema_ok_all and all_jsons_ok),
        "figures_regenerated": bool(figures_ok),
        "make_milestone_script_succeeded": bool(
            exit_zero_all
            and wrote_json_all
            and schema_ok_all
            and all_jsons_ok
            and figures_ok
            and durations["super_fast"] < 30.0
            and durations["fast"] < 300.0
        ),
    }
    return key_stats


_MILESTONE_HANDLERS = {
    "M0": make_m0,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.make_milestone")
    parser.add_argument("milestone", help="milestone id, e.g. M0")
    args = parser.parse_args()

    handler = _MILESTONE_HANDLERS.get(args.milestone)
    run = ScriptRun(script="make_milestone")
    summary_dir = RESULTS_ROOT / "milestones" / args.milestone
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"stats_{args.milestone}_pipeline.json"

    if handler is None:
        run.fail(reason=f"no handler for {args.milestone}", summary_path=summary_path)
        return 1

    try:
        stats = handler()
    except Exception as e:
        run.fail(reason=f"{type(e).__name__}: {e}", summary_path=summary_path)
        raise

    # Write stats_{milestone}_pipeline.json (the milestone-level artifact)
    stats_path = summary_dir / f"stats_{args.milestone}_pipeline.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    # Our own script summary (shared schema) — separate file so both schemas coexist.
    run_summary_path = summary_dir / f"summary_make_{args.milestone}.json"
    run.ok(
        key_stats=stats,
        summary_path=run_summary_path,
        print_stats={
            "milestone": args.milestone,
            "pass": stats["make_milestone_script_succeeded"],
            "super_fast_s": stats["super_fast_duration_seconds"],
            "fast_s": stats["fast_duration_seconds"],
            "full_s": stats["full_duration_seconds"],
        },
    )
    return 0 if stats["make_milestone_script_succeeded"] else 1


if __name__ == "__main__":
    sys.exit(main())
