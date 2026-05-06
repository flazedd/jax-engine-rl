"""Run R1-R4 validation on the 5 difficulty-sweep environments.

Wraps oracles.verify_requirements so that per-env stats are preserved instead
of being overwritten. Outputs a single aggregated table.

Outputs:
  results/M6R/env_validation/{env}_stats.json    # per-env raw stats
  results/M6R/env_validation/summary.json        # aggregated R1-R4 table
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from oracles.verify_requirements import verify
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs" / "envs"
RESULTS_ROOT = REPO_ROOT / "results"
M2_STATS = RESULTS_ROOT / "milestones" / "M2" / "stats_M2_requirements.json"
OUT_DIR = RESULTS_ROOT / "M6R" / "env_validation"

ENVS = [
    ("e_final",                  "e_final.yaml"),
    ("persistence_easy",         "m6_persistence_easy.yaml"),
    ("persistence_hard",         "m6_persistence_hard_v2.yaml"),
    ("distinguishability_easy",  "m6_distinguishability_easy.yaml"),
    ("distinguishability_hard",  "m6_distinguishability_hard_v2.yaml"),
]


def _run_one(label: str, env_yaml: str) -> dict:
    cfg_path = CONFIG_ROOT / env_yaml
    print(f"\n[m6r_env_val] === {label} ===", flush=True)
    t0 = time.perf_counter()
    stats = verify(cfg_path, run_mode="full")
    elapsed = time.perf_counter() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_env_path = OUT_DIR / f"{label}_stats.json"
    with open(per_env_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Also copy the side-effect stats_M2_requirements.json out before the
    # next iteration overwrites it.
    if M2_STATS.exists():
        shutil.copy(M2_STATS, OUT_DIR / f"{label}_stats_M2_full.json")

    return {
        "label": label,
        "env_yaml": env_yaml,
        "elapsed_sec": round(elapsed, 1),
        "R1": stats["R1_policy_disagreement"]["pass"],
        "R2": stats["R2_per_regime_ppo_vs_vi"]["pass"],
        "R3": stats["R3_mixed_gap"]["pass"],
        "R4": stats["R4_inferability"]["pass"],
        "all_pass": stats["all_pass"],
        # Diagnostics for the table.
        "R1_disagree_frac": stats["R1_policy_disagreement"]["fraction_disagreeing_states"],
        "R1_rel_value_loss": stats["R1_policy_disagreement"]["mean_relative_value_loss_at_disagreeing_states"],
        "R2_min_ratio": stats["R2_per_regime_ppo_vs_vi"]["min_ratio"],
        "R2_per_regime_ratios": [
            stats["R2_per_regime_ppo_vs_vi"]["regime_0_ratio"],
            stats["R2_per_regime_ppo_vs_vi"]["regime_1_ratio"],
            stats["R2_per_regime_ppo_vs_vi"]["regime_2_ratio"],
        ],
        "R3_gap": stats["R3_mixed_gap"]["gap_absolute"],
        "R3_gap_to_ci": stats["R3_mixed_gap"]["gap_to_ci_ratio"],
        "R4_entropy_decay": stats["R4_inferability"]["entropy_decay_fraction"],
        "R4_belief_gap_closure": stats["R4_inferability"]["belief_ppo_gap_closure_fraction"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    run = ScriptRun(script="m6r_env_validation", run_mode="full")

    rows = []
    failed = []
    t_start = time.perf_counter()
    for label, env_yaml in ENVS:
        try:
            r = _run_one(label, env_yaml)
        except Exception as e:
            print(f"[m6r_env_val] {label} FAILED: {e}", flush=True)
            failed.append(label)
            continue
        rows.append(r)

    total_min = (time.perf_counter() - t_start) / 60

    summary = {
        "n_envs": len(ENVS),
        "n_failed": len(failed),
        "failed": failed,
        "total_min": round(total_min, 2),
        "rows": rows,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    run.add_output(summary_path)

    print("\n[m6r_env_val] === summary ===", flush=True)
    print(f"{'env':28s}  R1   R2   R3   R4   all", flush=True)
    for r in rows:
        print(
            f"{r['label']:28s}  "
            f"{('Y' if r['R1'] else 'N'):3s}  "
            f"{('Y' if r['R2'] else 'N'):3s}  "
            f"{('Y' if r['R3'] else 'N'):3s}  "
            f"{('Y' if r['R4'] else 'N'):3s}  "
            f"{('Y' if r['all_pass'] else 'N')}",
            flush=True,
        )

    if failed:
        run.fail(reason=f"{len(failed)} envs failed: {failed}",
                 summary_path=summary_path)
        return 1
    run.ok(
        key_stats={
            "n_envs": len(rows),
            "all_pass_count": sum(1 for r in rows if r["all_pass"]),
            "total_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
