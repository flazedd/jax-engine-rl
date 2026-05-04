"""M5R — stacked-observation PPO across the 4 sweep environments.

Runs stacked-obs PPO at full budget × n=8 seeds on each of the four
difficulty-sweep environments (persistence_easy/hard,
distinguishability_easy/hard). Reuses the existing E_med stacked-obs
result from m5_ladder_stacked_ppo. No tuning is performed: stacked-obs
has no integration axis so the M5R fairness retune does not apply.

Outputs:
  - experiments/configs/m5r_stacked/{env_label}.yaml   (materialised configs)
  - results/m5r_stacked_obs_{env_label}/                (per-cell train output)
  - results/M5R/final/m5r_stacked_obs_sweep.json        (aggregated stats)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
RESULTS_ROOT = REPO_ROOT / "results"

SWEEP_ENVS = [
    ("e_final",                 "envs/e_final.yaml"),
    ("persistence_easy",        "envs/m6_persistence_easy.yaml"),
    ("persistence_hard",        "envs/m6_persistence_hard.yaml"),
    ("distinguishability_easy", "envs/m6_distinguishability_easy.yaml"),
    ("distinguishability_hard", "envs/m6_distinguishability_hard.yaml"),
]


def _materialise(env_label: str, env_yaml: str) -> Path:
    out_dir = CONFIG_ROOT / "m5r_stacked"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{env_label}.yaml"
    doc: dict[str, Any] = {
        "extends": ["base/base_ppo.yaml", env_yaml],
        "experiment_name": f"m5r_stacked_obs_{env_label}",
        "iterations": 200,
        "parallel_envs": 512,
        "rollout_length": 128,
        "num_seeds": 8,
        "seed_base": 0,
        "env": {
            "name": "market_making_v1_stacked",
            "params": {"stack_k": 4},
        },
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


def _read_metrics(experiment_name: str) -> dict[str, Any] | None:
    p = RESULTS_ROOT / experiment_name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        m = float(arr.mean()) if arr.size == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    out_dir = RESULTS_ROOT / "M5R" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "m5r_stacked_obs_sweep_run.json"
    stats_path = out_dir / "m5r_stacked_obs_sweep.json"

    run = ScriptRun(script="m5r_stacked_obs_sweep", run_mode="full")
    print(f"[m5r_stacked] start: {len(SWEEP_ENVS)} sweep envs", flush=True)
    t_start = time.perf_counter()

    by_env: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for i, (env_label, env_yaml) in enumerate(SWEEP_ENVS, start=1):
        yaml_path = _materialise(env_label, env_yaml)
        cfg = load_config(yaml_path)
        apply_run_mode(cfg, "full")
        existing = _read_metrics(cfg.experiment_name)
        skipped = existing is not None
        if skipped:
            metrics = existing
        else:
            t0 = time.perf_counter()
            print(
                f"[m5r_stacked] ({i}/{len(SWEEP_ENVS)}) {env_label} "
                f"starting (full budget, n=8)",
                flush=True,
            )
            try:
                train_or_sweep(cfg)
            except Exception as e:
                print(f"[m5r_stacked] {env_label} FAILED: {e}", flush=True)
                failed.append(env_label)
                continue
            metrics = _read_metrics(cfg.experiment_name)
            print(
                f"[m5r_stacked] ({i}/{len(SWEEP_ENVS)}) {env_label} "
                f"done in {(time.perf_counter()-t0)/60:.1f} min",
                flush=True,
            )
        finals = list(map(float, metrics.get("per_seed_final_return", [])))
        mean = float(np.mean(finals)) if finals else float("nan")
        ci = _bootstrap_ci(finals) if finals else (float("nan"), float("nan"))
        by_env[env_label] = {
            "experiment_name": cfg.experiment_name,
            "skipped": skipped,
            "per_seed_final_return": finals,
            "final_return_mean": mean,
            "final_return_ci95": list(ci),
        }
        print(
            f"[m5r_stacked]   {env_label}: mean={mean:.2f} "
            f"ci=[{ci[0]:.2f}, {ci[1]:.2f}]",
            flush=True,
        )

    total_min = (time.perf_counter() - t_start) / 60
    payload = {
        "envs": list(by_env.keys()),
        "n_seeds": 8,
        "iterations": 200,
        "parallel_envs": 512,
        "by_env": by_env,
        "total_min": total_min,
        "n_failed": len(failed),
        "failed": failed,
    }
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))

    if failed:
        run.fail(reason=f"{len(failed)} cells failed: {failed}",
                 summary_path=summary_path)
        return 1
    run.ok(
        key_stats={
            "n_envs": len(SWEEP_ENVS),
            "total_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
