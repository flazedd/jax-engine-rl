"""Run-mode timing and schema contract tests.

Asserts super-fast completes in <30s and fast in <300s on the dummy pipeline,
and that both produce summary JSONs that validate against the shared schema.
Writes its own script-output JSON.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from training.config import apply_run_mode, load_config
from training.train import train
from utils.script_output import ScriptRun, validate_summary
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "experiments" / "configs" / "m0_dummy.yaml"
RESULTS_ROOT = REPO_ROOT / "results"


def _run_mode(mode: str) -> dict:
    cfg = load_config(CONFIG)
    apply_run_mode(cfg, mode)
    t0 = time.perf_counter()
    train(cfg)
    elapsed = time.perf_counter() - t0
    summary_path = RESULTS_ROOT / cfg.experiment_name / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    ok, reason = validate_summary(summary)
    return {"mode": mode, "elapsed": elapsed, "schema_ok": ok, "schema_reason": reason}


def main() -> int:
    run = ScriptRun(script="test_run_modes")
    tests_run = 0
    tests_passed = 0

    # super-fast
    result_sf = _run_mode("super_fast")
    tests_run += 2
    if result_sf["schema_ok"]:
        tests_passed += 1
    if result_sf["elapsed"] < 30.0:
        tests_passed += 1

    # fast
    result_f = _run_mode("fast")
    tests_run += 2
    if result_f["schema_ok"]:
        tests_passed += 1
    if result_f["elapsed"] < 300.0:
        tests_passed += 1

    summary_path = RESULTS_ROOT / "tests" / "run_modes.json"
    status_ok = tests_passed == tests_run
    if status_ok:
        run.ok(
            key_stats={
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "super_fast_seconds": round(result_sf["elapsed"], 3),
                "fast_seconds": round(result_f["elapsed"], 3),
            },
            summary_path=summary_path,
        )
        return 0
    run.fail(
        reason=(
            f"super_fast={result_sf['elapsed']:.2f}s schema={result_sf['schema_ok']} | "
            f"fast={result_f['elapsed']:.2f}s schema={result_f['schema_ok']}"
        ),
        summary_path=summary_path,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
