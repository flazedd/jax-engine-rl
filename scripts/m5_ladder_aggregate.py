"""Post-hoc aggregator for M5 ladder.

The orchestrator (`scripts/m5_ladder.py`) wrote per-experiment metrics.json
but its first run had a bug: `ScriptRun.ok(summary_path=...)` clobbered the
rich stats JSON because they shared a path. This script rebuilds
`stats_M5_ladder.json` from the on-disk per-experiment metrics, without
retraining. The orchestrator has since been fixed (separate stats_path /
summary_path), so future runs land cleanly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import (
    gap_closed,
    leave_one_out_sensitivity,
    primary_hypothesis_test,
    ranking_stable_across_seeds,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

CONFIGS: list[tuple[str, str]] = [
    ("ppo",         "m5_ladder_ppo"),
    ("stacked_ppo", "m5_ladder_stacked_ppo"),
    ("rl2",         "m5_ladder_rl2"),
    ("varibad",     "m5_ladder_varibad"),
    ("belief_ppo",  "m5_ladder_belief_ppo"),
    ("oracle_ppo",  "m5_ladder_oracle_ppo"),
]

PRIMARY_HYPOTHESES: list[tuple[str, str, str]] = [
    ("varibad_beats_agnostic",    "varibad", "ppo"),
    ("rl2_beats_agnostic",        "rl2",     "ppo"),
    ("varibad_beats_stacked_ppo", "varibad", "stacked_ppo"),
    ("rl2_beats_stacked_ppo",     "rl2",     "stacked_ppo"),
]
FAMILY_SIZE = 6
ALPHA = 0.05

M3_REFERENCE_PATH = RESULTS_ROOT / "milestones" / "M3" / "stats_M3_reference_levels.json"


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _read_metrics(experiment_name: str) -> dict[str, Any]:
    with open(RESULTS_ROOT / experiment_name / "metrics.json") as f:
        return json.load(f)


def _load_m3_reference() -> dict[str, Any]:
    if not M3_REFERENCE_PATH.exists():
        return {}
    with open(M3_REFERENCE_PATH) as f:
        return json.load(f)


def main() -> int:
    run = ScriptRun(script="m5_ladder_aggregate")
    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M5_ladder.json"
    summary_path = out_dir / "stats_M5_ladder_run.json"

    # Read per-experiment metrics from disk.
    method_finals: dict[str, list[float]] = {}
    method_curves: dict[str, list[list[float]]] = {}
    method_meta: dict[str, dict[str, Any]] = {}
    for method, exp_name in CONFIGS:
        m = _read_metrics(exp_name)
        method_finals[method] = list(m["per_seed_final_return"])
        method_curves[method] = list(m["per_seed_mean_return_per_iter"])
        method_meta[method] = {
            "experiment_name": exp_name,
            "env": m["env"],
            "agent": m["agent"],
            "iterations": m["iterations"],
            "num_seeds": m["num_seeds"],
            "parallel_envs": m["parallel_envs"],
            "rollout_length": m["rollout_length"],
        }

    # Reference levels (from M3).
    m3 = _load_m3_reference()
    m3_ref = m3.get("reference_levels", {})
    floor_mean = float(m3_ref.get("regime_agnostic_ppo", {}).get("mean", float("nan")))
    belief_mean = float(m3_ref.get("belief_ppo", {}).get("mean", float("nan")))
    oracle_mean = float(m3_ref.get("oracle_ppo", {}).get("mean", float("nan")))

    # Per-method summary.
    methods_block: dict[str, Any] = {}
    for method, finals in method_finals.items():
        mean = float(np.mean(finals))
        ci_lo, ci_hi = _bootstrap_ci(finals)
        methods_block[method] = {
            "final_return": {
                "mean": mean,
                "ci": [ci_lo, ci_hi],
                "seed_returns": list(map(float, finals)),
            },
            "gap_closed_vs_oracle": gap_closed(mean, floor_mean, oracle_mean),
            "gap_closed_vs_belief_ppo": gap_closed(mean, floor_mean, belief_mean),
            "experiment_name": method_meta[method]["experiment_name"],
        }
        if method in ("rl2", "varibad", "belief_ppo"):
            methods_block[method]["posterior_mse_final"] = float("nan")
            methods_block[method]["regime_classification_accuracy"] = float("nan")

    ranking_by_return = sorted(
        method_finals.keys(),
        key=lambda m: methods_block[m]["final_return"]["mean"],
        reverse=True,
    )

    # Primary hypotheses (Holm correction across family of FAMILY_SIZE).
    primary: dict[str, dict[str, Any]] = {}
    for hyp_name, method, baseline in PRIMARY_HYPOTHESES:
        primary[hyp_name] = primary_hypothesis_test(
            method=method_finals[method],
            baseline=method_finals[baseline],
            family_size=FAMILY_SIZE,
            alpha=ALPHA,
            n_boot=10_000,
            alternative="greater",
        )

    # LOO sensitivity.
    loo: dict[str, dict[str, Any]] = {}
    for hyp_name, method, baseline in PRIMARY_HYPOTHESES:
        loo[hyp_name] = leave_one_out_sensitivity(
            method=method_finals[method],
            baseline=method_finals[baseline],
            alpha=ALPHA,
            n_corrections=FAMILY_SIZE,
            alternative="greater",
        )

    ranking_stability = ranking_stable_across_seeds(method_finals)

    posterior_quality_discrimination = {
        "posterior_mse_range": [float("nan"), float("nan")],
        "posterior_mse_range_spans_threshold": None,
        "note": "filled by follow-on probe pass (evaluation/posterior_compare).",
    }

    stats: dict[str, Any] = {
        "env_version": m3.get("env_version", "e_final"),
        "alpha": ALPHA,
        "family_size": FAMILY_SIZE,
        "reference_levels": {
            "regime_agnostic_ppo_mean": floor_mean,
            "belief_ppo_mean": belief_mean,
            "oracle_ppo_mean": oracle_mean,
            "source": "results/milestones/M3/stats_M3_reference_levels.json",
        },
        "methods": methods_block,
        "method_meta": method_meta,
        "ranking_by_return": ranking_by_return,
        "ranking_by_posterior_quality": None,
        "primary_hypotheses": primary,
        "leave_one_out_sensitivity": loo,
        "ranking_stable_across_seeds": ranking_stability,
        "posterior_quality_discrimination": posterior_quality_discrimination,
    }

    all_supported = all(primary[h].get("supported", False) for h, _, _ in PRIMARY_HYPOTHESES)
    all_loo_robust = all(loo[h].get("robust_to_loo", False) for h, _, _ in PRIMARY_HYPOTHESES)
    stats["pass_summary"] = {
        "varibad_beats_agnostic_supported": primary.get("varibad_beats_agnostic", {}).get("supported"),
        "rl2_beats_agnostic_supported": primary.get("rl2_beats_agnostic", {}).get("supported"),
        "varibad_beats_stacked_ppo_supported": primary.get("varibad_beats_stacked_ppo", {}).get("supported"),
        "rl2_beats_stacked_ppo_supported": primary.get("rl2_beats_stacked_ppo", {}).get("supported"),
        "all_primary_supported": all_supported,
        "all_loo_robust": all_loo_robust,
        "ranking_stable": ranking_stability["stable"],
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    run.ok(
        key_stats={
            "n_methods": len(method_finals),
            "ranking": ranking_by_return,
            "all_primary_supported": bool(all_supported),
            "all_loo_robust": bool(all_loo_robust),
            "ranking_stable": bool(ranking_stability["stable"]),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print("[m5_ladder_aggregate] === ladder summary ===", flush=True)
    for m in ranking_by_return:
        info = methods_block[m]
        gap_o = info["gap_closed_vs_oracle"]
        gap_b = info["gap_closed_vs_belief_ppo"]
        print(
            f"[m5_ladder_aggregate] {m:>14s} | mean={info['final_return']['mean']:7.2f} "
            f"| gap_closed_oracle={gap_o:.3f} | gap_closed_belief={gap_b:.3f}",
            flush=True,
        )
    print(f"[m5_ladder_aggregate] === primary hypotheses (Holm m={FAMILY_SIZE}) ===", flush=True)
    for hyp_name, _, _ in PRIMARY_HYPOTHESES:
        r = primary[hyp_name]
        sup = r.get("supported")
        p = r.get("holm_corrected_p")
        d = r.get("median_paired_delta")
        ci = r.get("delta_ci")
        ci_str = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "n/a"
        print(
            f"[m5_ladder_aggregate] {hyp_name:>30s} | supported={sup} | p_corr={p} "
            f"| delta_median={d:.2f} | ci={ci_str}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
