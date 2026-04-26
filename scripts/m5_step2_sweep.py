"""M5 Step-2 sweep orchestrator — targeted re-tune on E_final.

Pipeline:
  1. Sequentially train 9 configs (3 VariBAD kl_coef + 6 RL² hidden_dim ×
     ent_coef) at half-budget (100 iter × 512 envs × 128 rollout × 3 seeds).
  2. Read each `metrics.json` for `final_return_mean` and the per-seed list.
  3. Compute pass-vs-floor flag (M3 PPO floor 137.81 from
     stats_M3_reference_levels.json) per config.
  4. Write `results/milestones/M5/stats_M5_step2_sweep.json`.

Pass criterion (recovery plan): at least one config per method clears the
PPO floor (mean across 3 seeds ≥ floor).

Usage:
  uv run python -m scripts.m5_step2_sweep
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

# (method_family, config_name, label).
CONFIGS: list[tuple[str, str, str]] = [
    # VariBAD kl_coef sweep — diagnostic-prioritized.
    ("varibad", "m5_step2_varibad_kl001.yaml", "varibad_kl0.001"),
    ("varibad", "m5_step2_varibad_kl01.yaml",  "varibad_kl0.01"),
    ("varibad", "m5_step2_varibad_kl10.yaml",  "varibad_kl0.1"),
    # RL² hidden_dim × ent_coef grid — full per recovery plan.
    ("rl2",     "m5_step2_rl2_h64_e001.yaml",  "rl2_h64_e0.001"),
    ("rl2",     "m5_step2_rl2_h64_e01.yaml",   "rl2_h64_e0.01"),
    ("rl2",     "m5_step2_rl2_h128_e001.yaml", "rl2_h128_e0.001"),
    ("rl2",     "m5_step2_rl2_h128_e01.yaml",  "rl2_h128_e0.01"),
    ("rl2",     "m5_step2_rl2_h256_e001.yaml", "rl2_h256_e0.001"),
    ("rl2",     "m5_step2_rl2_h256_e01.yaml",  "rl2_h256_e0.01"),
]

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


def _load_floor() -> float:
    if not M3_REFERENCE_PATH.exists():
        print(
            f"[m5_step2] WARNING: M3 reference not found at {M3_REFERENCE_PATH}; "
            "using hardcoded 137.81",
            flush=True,
        )
        return 137.81
    with open(M3_REFERENCE_PATH) as f:
        m3 = json.load(f)
    return float(m3["reference_levels"]["regime_agnostic_ppo"]["mean"])


def main() -> int:
    run = ScriptRun(script="m5_step2_sweep")
    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M5_step2_sweep.json"
    summary_path = out_dir / "stats_M5_step2_sweep_run.json"

    floor_mean = _load_floor()
    n_configs = len(CONFIGS)
    print(
        f"[m5_step2] start: {n_configs} configs at half-budget "
        f"(100 iter, 512 envs, 3 seeds) | floor={floor_mean:.2f}",
        flush=True,
    )
    t_start = time.perf_counter()

    config_results: list[dict[str, Any]] = []
    failed: list[str] = []

    for i, (family, cfg_name, label) in enumerate(CONFIGS, start=1):
        cfg = load_config(CONFIG_ROOT / cfg_name)
        apply_run_mode(cfg, "full")
        print(
            f"[m5_step2] ({i}/{n_configs}) launching {label} "
            f"family={family} config={cfg_name} iters={cfg.iterations} "
            f"envs={cfg.parallel_envs} seeds={cfg.num_seeds}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            train_or_sweep(cfg)
        except SystemExit as e:
            if e.code != 0:
                print(
                    f"[m5_step2] ({i}/{n_configs}) {label} FAILED (exit {e.code})",
                    flush=True,
                )
                failed.append(label)
                continue
        elapsed = time.perf_counter() - t0
        total_min = (time.perf_counter() - t_start) / 60

        m = _read_metrics(cfg.experiment_name)
        finals = list(map(float, m["per_seed_final_return"]))
        mean = float(np.mean(finals))
        ci_lo, ci_hi = _bootstrap_ci(finals)
        clears = mean >= floor_mean
        config_results.append({
            "label": label,
            "family": family,
            "config": cfg_name,
            "experiment_name": cfg.experiment_name,
            "final_return_mean": mean,
            "final_return_ci95": [ci_lo, ci_hi],
            "per_seed_final_return": finals,
            "clears_floor": clears,
            "delta_vs_floor": mean - floor_mean,
            "iterations": int(m["iterations"]),
            "num_seeds": int(m["num_seeds"]),
        })
        print(
            f"[m5_step2] ({i}/{n_configs}) {label} done in {elapsed/60:.1f} min "
            f"| total {total_min:.1f} min | mean={mean:.2f} "
            f"(Δfloor={mean-floor_mean:+.2f}) clears={clears}",
            flush=True,
        )

    if failed:
        run.fail(
            reason=f"{len(failed)}/{n_configs} runs failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1

    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in config_results:
        by_family.setdefault(r["family"], []).append(r)

    family_summary: dict[str, dict[str, Any]] = {}
    for family, rs in by_family.items():
        clearing = [r for r in rs if r["clears_floor"]]
        best = max(rs, key=lambda x: x["final_return_mean"]) if rs else None
        family_summary[family] = {
            "n_configs": len(rs),
            "n_clearing_floor": len(clearing),
            "any_clears_floor": len(clearing) > 0,
            "best_label": best["label"] if best else None,
            "best_mean": best["final_return_mean"] if best else None,
            "best_delta_vs_floor": (best["final_return_mean"] - floor_mean) if best else None,
        }

    all_pass = all(s["any_clears_floor"] for s in family_summary.values())

    stats = {
        "floor_mean_used": floor_mean,
        "floor_source": str(M3_REFERENCE_PATH),
        "n_configs": n_configs,
        "configs": config_results,
        "family_summary": family_summary,
        "step2_pass": all_pass,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_elapsed = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "n_configs": n_configs,
            "elapsed_min": round(total_elapsed, 2),
            "step2_pass": bool(all_pass),
            "varibad_any_clears": family_summary.get("varibad", {}).get("any_clears_floor"),
            "rl2_any_clears": family_summary.get("rl2", {}).get("any_clears_floor"),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    # Human-readable final report.
    print("[m5_step2] === sweep summary ===", flush=True)
    for family, rs in by_family.items():
        rs_sorted = sorted(rs, key=lambda x: x["final_return_mean"], reverse=True)
        print(f"[m5_step2] -- {family} --", flush=True)
        for r in rs_sorted:
            mark = "PASS" if r["clears_floor"] else "    "
            print(
                f"[m5_step2]   {mark} {r['label']:<22s} | mean={r['final_return_mean']:7.2f} "
                f"| Δfloor={r['delta_vs_floor']:+7.2f} | seeds={r['per_seed_final_return']}",
                flush=True,
            )
    print(
        f"[m5_step2] step2_pass={all_pass} (varibad_pass="
        f"{family_summary.get('varibad', {}).get('any_clears_floor')}, "
        f"rl2_pass={family_summary.get('rl2', {}).get('any_clears_floor')})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
