"""Reproduce the thesis environment-validation table (R1-R4).

Runs `oracles.verify_requirements` on the four RSMM environments the thesis
reports: the medium-difficulty reference environment and the three
difficulty-study instances. Per-env stats are preserved instead of being
overwritten by the next env, and the aggregated table is compared against the
values printed in the thesis so a drift shows up as a FAIL rather than as a
silently different number.

Supersedes `m6r_env_validation.py`, which targeted the pre-redesign sweep
environments (`m6_persistence_*`, `m6_distinguishability_*`) — two of whose
configs no longer exist. Those results are kept for history in
`results/M6R/env_validation/`.

    uv run python -m scripts.env_validation_final
    uv run python -m scripts.env_validation_final --fast        # smoke test
    uv run python -m scripts.env_validation_final --super-fast  # smoke test

Outputs:
  results/env_validation_final/{label}_stats.json     # per-env raw stats
  results/env_validation_final/validation_table.json  # aggregated R1-R4 table
  results/env_validation_final/summary.json           # shared-schema run summary
  results/env_validation_final/figures/{label}/     # per-env M2 diagnostics
  figures/milestones/M2/                            # reference-env diagnostics

Only `--full` (the default) reproduces the thesis numbers; the reduced modes
cut the PPO budget far below what R2 and R3 need and will not clear the
thresholds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from oracles.verify_requirements import (
    FIGURES_ROOT,
    THRESHOLDS,
    verify,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs" / "envs"
RESULTS_ROOT = REPO_ROOT / "results"
FULL_OUT_DIR = RESULTS_ROOT / "env_validation_final"
M2_FIG_DIR = FIGURES_ROOT / "milestones" / "M2"

SCRIPT = "env_validation_final"

# Set by main() once the run mode is known. Reduced modes never write into the
# full-mode directory: their PPO budget is a fraction of what R2 and R3 need,
# so a smoke test that wrote there would replace the artifacts behind the
# thesis table with numbers that fail their own thresholds.
OUT_DIR = FULL_OUT_DIR


def _out_dir_for(run_mode: str) -> Path:
    if run_mode == "full":
        return FULL_OUT_DIR
    return FULL_OUT_DIR / f"_smoke_{run_mode}"

# The four environments of the thesis validation table, in table row order.
# `thesis_row` is the row label as it appears there; `expected` is the measured
# value the thesis prints, rounded to 2dp, checked against on every re-run.
ENVS: list[dict[str, Any]] = [
    {
        "label": "e_final",
        "config": "e_final.yaml",
        "thesis_row": "Medium-difficulty, reference",
        "expected": {"R1": 1.00, "R2": 0.91, "R3": 0.75, "R4": 0.50},
        "reference_env": True,
    },
    {
        "label": "sweep_dist_easy",
        "config": "sweep_dist_easy.yaml",
        "thesis_row": "Distinguishability easy",
        "expected": {"R1": 1.00, "R2": 0.93, "R3": 0.64, "R4": 0.59},
        "reference_env": False,
    },
    {
        "label": "sweep_dist_hard",
        "config": "sweep_dist_hard.yaml",
        "thesis_row": "Distinguishability hard",
        "expected": {"R1": 1.00, "R2": 0.89, "R3": 0.88, "R4": 0.45},
        "reference_env": False,
    },
    {
        "label": "sweep_coupled_fast",
        "config": "sweep_coupled_fast.yaml",
        "thesis_row": "Coupled fast persistence",
        "expected": {"R1": 1.00, "R2": 0.93, "R3": 0.63, "R4": 0.37},
        "reference_env": False,
    },
]


def _measured(stats: dict[str, Any]) -> dict[str, float]:
    """Pull the four table statistics out of a verify() stats dict."""
    return {
        "R1": float(stats["R1_policy_disagreement"]["fraction_disagreeing_states"]),
        "R2": float(stats["R2_per_regime_ppo_vs_vi"]["min_ratio"]),
        "R3": float(stats["R3_mixed_gap"]["recovery_ratio"]),
        "R4": float(stats["R4_inferability"]["entropy_decay_fraction"]),
    }


def _passes(stats: dict[str, Any]) -> dict[str, bool]:
    return {
        "R1": bool(stats["R1_policy_disagreement"]["pass"]),
        "R2": bool(stats["R2_per_regime_ppo_vs_vi"]["pass"]),
        "R3": bool(stats["R3_mixed_gap"]["pass"]),
        "R4": bool(stats["R4_inferability"]["pass"]),
    }


def _snapshot_figures(label: str, src: Path) -> None:
    """Copy this env's diagnostic figures out before the next env overwrites them.

    verify() writes every env's figures to the same directory, so without this
    the surviving set belongs to whichever env happened to run last. The thesis
    appendix shows the reference env's figures, so provenance cannot be left to
    iteration order.
    """
    dest = OUT_DIR / "figures" / label
    dest.mkdir(parents=True, exist_ok=True)
    for png in sorted(src.glob("fig_M2_*.png")):
        shutil.copy(png, dest / png.name)


def _restore_reference_figures(label: str) -> list[Path]:
    """Put the reference env's figures back in figures/milestones/M2/.

    These are the ones the thesis appendix includes, so they must be the
    reference env's regardless of which env ran last.
    """
    src = OUT_DIR / "figures" / label
    restored = []
    for png in sorted(src.glob("fig_M2_*.png")):
        target = M2_FIG_DIR / png.name
        shutil.copy(png, target)
        restored.append(target)
    return restored


def _run_one(spec: dict[str, Any], run_mode: str) -> dict[str, Any]:
    label = spec["label"]
    cfg_path = CONFIG_ROOT / spec["config"]
    if not cfg_path.exists():
        raise FileNotFoundError(f"env config not found: {cfg_path}")

    print(f"\n[{SCRIPT}] === {label} ({spec['thesis_row']}) ===", flush=True)
    # Only a full-mode run may touch the committed figure directory.
    fig_dir = M2_FIG_DIR if run_mode == "full" else OUT_DIR / "_figures_scratch"
    t0 = time.perf_counter()
    stats = verify(cfg_path, run_mode=run_mode, fig_dir=fig_dir)
    elapsed = time.perf_counter() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUT_DIR / f"{label}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    _snapshot_figures(label, fig_dir)

    return _build_row(spec, stats, stats_path, round(elapsed, 1))


def _build_row(
    spec: dict[str, Any],
    stats: dict[str, Any],
    stats_path: Path,
    elapsed_sec: float | None,
) -> dict[str, Any]:
    measured = _measured(stats)
    passes = _passes(stats)
    expected = spec["expected"]
    # The thesis prints these to 2dp, so that is the resolution at which a
    # re-run can be said to reproduce it.
    rounded = {k: round(v, 2) for k, v in measured.items()}
    mismatches = {
        k: {"thesis": expected[k], "rerun": rounded[k]}
        for k in ("R1", "R2", "R3", "R4")
        if abs(rounded[k] - expected[k]) > 1e-9
    }

    return {
        "label": spec["label"],
        "thesis_row": spec["thesis_row"],
        "env_config": spec["config"],
        "stats_path": str(stats_path.relative_to(REPO_ROOT)),
        "elapsed_sec": elapsed_sec,
        "measured": measured,
        "rounded_2dp": rounded,
        "thesis_table": expected,
        "table_mismatches": mismatches,
        "pass": passes,
        "all_pass": bool(stats["all_pass"]),
        # Supporting diagnostics, not printed in the table itself.
        "R1_rel_value_loss": stats["R1_policy_disagreement"][
            "mean_relative_value_loss_at_disagreeing_states"
        ],
        "R2_per_regime_ratios": [
            stats["R2_per_regime_ppo_vs_vi"][f"regime_{r}_ratio"]
            for r in range(len(stats["vi"]["per_regime_expected_episode_return"]))
        ],
        "R3_regime_agnostic_return": stats["R3_mixed_gap"]["regime_agnostic_return_mean"],
        "R3_oracle_return": stats["R3_mixed_gap"]["oracle_ppo_return_mean"],
        "R3_gap_absolute": stats["R3_mixed_gap"]["gap_absolute"],
        "R3_gap_to_ci_ratio": stats["R3_mixed_gap"]["gap_to_ci_ratio"],
        "R4_belief_gap_closure": stats["R4_inferability"]["belief_ppo_gap_closure_fraction"],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print the table in thesis row/column order so it can be eyeballed."""
    head = (
        f"{'Environment':30s}  "
        f"{'R1>=' + str(THRESHOLDS['R1']['value']):>10s}  "
        f"{'R2>=' + str(THRESHOLDS['R2']['value']):>10s}  "
        f"{'R3<=' + str(THRESHOLDS['R3']['value']):>10s}  "
        f"{'R4>=' + str(THRESHOLDS['R4']['value']):>10s}   all"
    )
    print(f"\n[{SCRIPT}] === validation table ===", flush=True)
    print(head, flush=True)
    for r in rows:
        cells = []
        for k in ("R1", "R2", "R3", "R4"):
            flag = "" if r["pass"][k] else " FAIL"
            cells.append(f"{r['rounded_2dp'][k]:.2f}{flag:>5s}")
        print(
            f"{r['thesis_row']:30s}  "
            + "  ".join(f"{c:>10s}" for c in cells)
            + f"   {'Y' if r['all_pass'] else 'N'}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog=f"scripts.{SCRIPT}")
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="rebuild the aggregated table from the per-env stats already on "
             "disk, without re-running any training",
    )
    args = parser.parse_args()
    if args.super_fast and args.fast:
        raise SystemExit("--super-fast and --fast are mutually exclusive")
    run_mode = "super_fast" if args.super_fast else "fast" if args.fast else "full"

    global OUT_DIR
    OUT_DIR = _out_dir_for(run_mode)
    if run_mode != "full":
        print(
            f"[{SCRIPT}] run_mode={run_mode}: smoke test only. Writing to "
            f"{OUT_DIR.relative_to(REPO_ROOT)} and leaving the full-mode "
            f"artifacts and appendix figures untouched. The reduced PPO budget "
            f"will not clear R2/R3.",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Two distinct files. summary.json is the shared-schema ScriptRun summary
    # that every script in this repo writes; the aggregated table goes beside
    # it under its own name. Writing both to summary.json makes ScriptRun.ok()
    # overwrite the table.
    table_path = OUT_DIR / "validation_table.json"
    summary_path = OUT_DIR / "summary.json"
    run = ScriptRun(script=SCRIPT, run_mode=run_mode)

    rows: list[dict[str, Any]] = []
    errored: list[dict[str, str]] = []
    t_start = time.perf_counter()
    for spec in ENVS:
        try:
            if args.table_only:
                stats_path = OUT_DIR / f"{spec['label']}_stats.json"
                with open(stats_path) as f:
                    rows.append(_build_row(spec, json.load(f), stats_path, None))
            else:
                rows.append(_run_one(spec, run_mode))
        except Exception as e:  # noqa: BLE001 — one env failing must not lose the rest
            print(f"[{SCRIPT}] {spec['label']} ERRORED: {type(e).__name__}: {e}", flush=True)
            errored.append({"label": spec["label"], "error": f"{type(e).__name__}: {e}"})
    total_min = (time.perf_counter() - t_start) / 60

    reference = next((s["label"] for s in ENVS if s["reference_env"]), None)
    restored: list[Path] = []
    if (
        run_mode == "full"
        and not args.table_only
        and reference
        and any(r["label"] == reference for r in rows)
    ):
        restored = _restore_reference_figures(reference)

    failed_reqs = [r["label"] for r in rows if not r["all_pass"]]
    drifted = [r["label"] for r in rows if r["table_mismatches"]]

    summary = {
        "n_envs": len(ENVS),
        "n_errored": len(errored),
        "errored": errored,
        "run_mode": run_mode,
        "rebuilt_from_existing_stats": bool(args.table_only),
        "thresholds": THRESHOLDS,
        "reference_env": reference,
        "appendix_figures_from": reference,
        "envs_failing_requirements": failed_reqs,
        "envs_differing_from_thesis_table": drifted,
        "reproduces_thesis_table": bool(
            not errored and not failed_reqs and not drifted and len(rows) == len(ENVS)
        ),
        "total_min": round(total_min, 2),
        "rows": rows,
    }
    with open(table_path, "w") as f:
        json.dump(summary, f, indent=2)
    run.add_output(table_path)
    for p in restored:
        run.add_output(p)
    for r in rows:
        run.add_output(REPO_ROOT / r["stats_path"])

    _print_table(rows)
    for r in rows:
        for req, d in r["table_mismatches"].items():
            print(
                f"[{SCRIPT}] DRIFT | {r['label']} {req}: "
                f"thesis={d['thesis']:.2f} rerun={d['rerun']:.2f}",
                flush=True,
            )

    # In reduced modes only a crash is a failure: the cut budget is expected to
    # miss R2/R3, so requirement and table checks would fail by construction.
    fatal = list(errored) if run_mode != "full" else errored
    if fatal or (run_mode == "full" and failed_reqs):
        reason = "; ".join(
            filter(
                None,
                [
                    f"{len(errored)} envs errored: {[e['label'] for e in errored]}" if errored else "",
                    f"requirements failed: {failed_reqs}" if failed_reqs and run_mode == "full" else "",
                ],
            )
        )
        run.fail(reason=reason, summary_path=summary_path)
        return 1

    if drifted and run_mode == "full":
        run.fail(
            reason=f"all requirements pass but the table drifted from the thesis: {drifted}",
            summary_path=summary_path,
        )
        return 1

    run.ok(
        key_stats={
            "n_envs": len(rows),
            "all_pass_count": sum(1 for r in rows if r["all_pass"]),
            "reproduces_thesis_table": summary["reproduces_thesis_table"],
            "run_mode": run_mode,
            "total_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
