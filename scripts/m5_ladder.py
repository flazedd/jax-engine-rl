"""M5 ladder orchestrator — runs the 6 ladder methods on MM E_final and
computes the headline statistics for `stats_M5_ladder.json`.

Pipeline:
  1. Sequentially train 6 method configs at full budget (200 iter × 512 envs
     × 5 seeds each).
  2. For each method, read `per_seed_final_return` and the per-seed learning
     curve from `metrics.json`.
  3. Compute gap-closed fractions (vs M3 oracle ceiling and M3 belief ceiling).
  4. Run the 4 primary paired-hypothesis tests:
       varibad_beats_agnostic, rl2_beats_agnostic,
       varibad_beats_stacked_ppo, rl2_beats_stacked_ppo
     with Holm correction across `family_size = 4`.
  5. Compute leave-one-out seed sensitivity for each.
  6. Compute ranking stability across seeds.
  7. Write `results/milestones/M5/stats_M5_ladder.json`.

`posterior_quality_discrimination` is left as `null` here: it requires the
linear-probe machinery in `evaluation/posterior_compare.py` to run on
features collected from RL² / VariBAD / Belief-PPO, which is a separate
follow-on step. The other six pass criteria are computed end-to-end.

Usage:
  uv run python -m scripts.m5_ladder
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import (
    cliffs_delta,
    gap_closed,
    leave_one_out_sensitivity,
    primary_hypothesis_test,
    ranking_stable_across_seeds,
)
from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

# (method_label, config_name). Order doubles as ladder order.
CONFIGS: list[tuple[str, str]] = [
    ("ppo",         "m5_ladder_ppo.yaml"),
    ("stacked_ppo", "m5_ladder_stacked_ppo.yaml"),
    ("rl2",         "m5_ladder_rl2.yaml"),
    ("varibad",     "m5_ladder_varibad.yaml"),
    ("belief_ppo",  "m5_ladder_belief_ppo.yaml"),
    ("oracle_ppo",  "m5_ladder_oracle_ppo.yaml"),
]

# (hypothesis_name, method, baseline). All pre-registered, one-sided "greater".
PRIMARY_HYPOTHESES: list[tuple[str, str, str]] = [
    ("varibad_beats_agnostic",    "varibad", "ppo"),
    ("rl2_beats_agnostic",        "rl2",     "ppo"),
    ("varibad_beats_stacked_ppo", "varibad", "stacked_ppo"),
    ("rl2_beats_stacked_ppo",     "rl2",     "stacked_ppo"),
]
# Holm correction is across the full M5 family of 6 primaries (4 ladder + 2
# factorial main effects, see methodology.md → "Pre-registered primary
# hypotheses"). The factorial mains are tested in stats_M5_factorial.json
# but share the same FWER family, so the ladder runs use family_size = 6.
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
    path = RESULTS_ROOT / experiment_name / "metrics.json"
    with open(path) as f:
        return json.load(f)


def _load_m3_reference() -> dict[str, Any]:
    """Read M3 reference levels (used as the canonical floor/ceilings)."""
    if not M3_REFERENCE_PATH.exists():
        print(f"[m5_ladder] WARNING: M3 reference not found at {M3_REFERENCE_PATH}",
              flush=True)
        return {}
    with open(M3_REFERENCE_PATH) as f:
        return json.load(f)


def main() -> int:
    run = ScriptRun(script="m5_ladder")
    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "stats_M5_ladder.json"

    n_configs = len(CONFIGS)
    print(f"[m5_ladder] start: {n_configs} ladder configs at full budget",
          flush=True)
    t_start = time.perf_counter()

    # ----- Phase 1: train ----------------------------------------------------
    method_finals: dict[str, list[float]] = {}
    method_curves: dict[str, list[list[float]]] = {}
    method_meta: dict[str, dict[str, Any]] = {}
    failed: list[str] = []

    for i, (method, cfg_name) in enumerate(CONFIGS, start=1):
        cfg = load_config(CONFIG_ROOT / cfg_name)
        apply_run_mode(cfg, "full")
        print(
            f"[m5_ladder] ({i}/{n_configs}) launching method={method} "
            f"config={cfg_name} iters={cfg.iterations} envs={cfg.parallel_envs} "
            f"seeds={cfg.num_seeds}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            train_or_sweep(cfg)
        except SystemExit as e:
            if e.code != 0:
                print(
                    f"[m5_ladder] ({i}/{n_configs}) {method} FAILED (exit {e.code})",
                    flush=True,
                )
                failed.append(method)
                continue
        elapsed = time.perf_counter() - t0
        total_min = (time.perf_counter() - t_start) / 60
        print(
            f"[m5_ladder] ({i}/{n_configs}) {method} done in {elapsed/60:.1f} min "
            f"| total {total_min:.1f} min",
            flush=True,
        )

        m = _read_metrics(cfg.experiment_name)
        method_finals[method] = list(m["per_seed_final_return"])
        method_curves[method] = list(m["per_seed_mean_return_per_iter"])
        method_meta[method] = {
            "experiment_name": cfg.experiment_name,
            "env": m["env"],
            "agent": m["agent"],
            "iterations": m["iterations"],
            "num_seeds": m["num_seeds"],
            "parallel_envs": m["parallel_envs"],
            "rollout_length": m["rollout_length"],
        }

    if failed:
        run.fail(
            reason=f"{len(failed)}/{n_configs} runs failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1

    # ----- Phase 2: reference levels (from M3) -------------------------------
    m3 = _load_m3_reference()
    m3_ref = m3.get("reference_levels", {})
    floor_mean = float(m3_ref.get("regime_agnostic_ppo", {}).get("mean", float("nan")))
    belief_mean = float(m3_ref.get("belief_ppo", {}).get("mean", float("nan")))
    oracle_mean = float(m3_ref.get("oracle_ppo", {}).get("mean", float("nan")))

    # ----- Phase 3: per-method summary ---------------------------------------
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
        # Posterior-quality fields for methods that maintain a belief representation.
        # Filled by a follow-on probe pass; placeholder NaN for now.
        if method in ("rl2", "varibad", "belief_ppo"):
            methods_block[method]["posterior_mse_final"] = float("nan")
            methods_block[method]["regime_classification_accuracy"] = float("nan")

    # Ranking by mean return (descending).
    ranking_by_return = sorted(
        method_finals.keys(),
        key=lambda m: methods_block[m]["final_return"]["mean"],
        reverse=True,
    )

    # ----- Phase 4: primary hypotheses --------------------------------------
    primary: dict[str, dict[str, Any]] = {}
    for hyp_name, method, baseline in PRIMARY_HYPOTHESES:
        if method not in method_finals or baseline not in method_finals:
            primary[hyp_name] = {"error": "missing method or baseline runs"}
            continue
        result = primary_hypothesis_test(
            method=method_finals[method],
            baseline=method_finals[baseline],
            family_size=FAMILY_SIZE,
            alpha=ALPHA,
            n_boot=10_000,
            alternative="greater",
        )
        primary[hyp_name] = result

    # ----- Phase 5: leave-one-out sensitivity --------------------------------
    loo: dict[str, dict[str, Any]] = {}
    for hyp_name, method, baseline in PRIMARY_HYPOTHESES:
        if method not in method_finals or baseline not in method_finals:
            loo[hyp_name] = {"error": "missing method or baseline runs"}
            continue
        loo[hyp_name] = leave_one_out_sensitivity(
            method=method_finals[method],
            baseline=method_finals[baseline],
            alpha=ALPHA,
            n_corrections=FAMILY_SIZE,
            alternative="greater",
        )

    # ----- Phase 6: ranking stability across seeds --------------------------
    ranking_stability = ranking_stable_across_seeds(method_finals)

    # ----- Phase 7: posterior-quality discrimination (placeholder) ----------
    posterior_quality_discrimination = {
        "posterior_mse_range": [float("nan"), float("nan")],
        "posterior_mse_range_spans_threshold": None,
        "note": "filled by follow-on probe pass (evaluation/posterior_compare).",
    }

    # ----- Assemble final stats JSON ----------------------------------------
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

    # ----- Pass-criterion summary -------------------------------------------
    all_supported = all(
        primary.get(h, {}).get("supported", False) for h, _, _ in PRIMARY_HYPOTHESES
    )
    all_loo_robust = all(
        loo.get(h, {}).get("robust_to_loo", False) for h, _, _ in PRIMARY_HYPOTHESES
    )
    pass_summary = {
        "varibad_beats_agnostic_supported": primary.get("varibad_beats_agnostic", {}).get("supported"),
        "rl2_beats_agnostic_supported": primary.get("rl2_beats_agnostic", {}).get("supported"),
        "varibad_beats_stacked_ppo_supported": primary.get("varibad_beats_stacked_ppo", {}).get("supported"),
        "rl2_beats_stacked_ppo_supported": primary.get("rl2_beats_stacked_ppo", {}).get("supported"),
        "all_primary_supported": all_supported,
        "all_loo_robust": all_loo_robust,
        "ranking_stable": ranking_stability["stable"],
    }
    stats["pass_summary"] = pass_summary

    # Write file.
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    total_elapsed = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "n_methods": len(method_finals),
            "elapsed_min": round(total_elapsed, 2),
            "ranking": ranking_by_return,
            "all_primary_supported": bool(all_supported),
            "all_loo_robust": bool(all_loo_robust),
            "ranking_stable": bool(ranking_stability["stable"]),
        },
        summary_path=summary_path,
    )

    # Human-readable final report.
    print("[m5_ladder] === ladder summary ===", flush=True)
    for m in ranking_by_return:
        info = methods_block[m]
        gap_o = info["gap_closed_vs_oracle"]
        gap_b = info["gap_closed_vs_belief_ppo"]
        print(
            f"[m5_ladder] {m:>14s} | mean={info['final_return']['mean']:7.2f} "
            f"| gap_closed_oracle={gap_o:.3f} | gap_closed_belief={gap_b:.3f}",
            flush=True,
        )
    print("[m5_ladder] === primary hypotheses (Holm m=4) ===", flush=True)
    for hyp_name, _, _ in PRIMARY_HYPOTHESES:
        r = primary[hyp_name]
        sup = r.get("supported")
        p = r.get("holm_corrected_p")
        d = r.get("median_paired_delta")
        ci = r.get("delta_ci")
        ci_str = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "n/a"
        print(
            f"[m5_ladder] {hyp_name:>30s} | supported={sup} | p_corr={p} "
            f"| delta_median={d:.2f} | ci={ci_str}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
