"""M5 factorial toys: 8 variants × 3 toy envs = 24 runs.

Tests the full 2×2×2 ablation factorial — (rl2, varibad) × (concat, hypernet)
× (no_bonus, bonus) — on the M4 toy environments (bandit, gridworld,
regime_bandit). Templates from `experiments/configs/m4_<method>_<env>.yaml`
provide budgets and env settings; orchestrator applies the variant overrides
to `agent.params` and renames `experiment_name`.

Pass criterion: variants should perform at parity-or-better with the M4
baseline (concat, no_bonus). A regression vs M4 baseline indicates a wiring
bug; gains indicate the addon is genuinely helpful on toys.

Usage:
  uv run python -m scripts.m5_factorial_toys
"""
from __future__ import annotations

import copy
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

METHODS = ["rl2", "varibad"]
ENVS = ["bandit", "gridworld", "regime_bandit"]
VARIANTS = [
    # (integration, exploration_bonus, label)
    ("concat",   False, "concat_nobonus"),
    ("concat",   True,  "concat_bonus"),
    ("hypernet", False, "hypernet_nobonus"),
    ("hypernet", True,  "hypernet_bonus"),
]


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _build_factorial_config(method: str, env: str, integration: str, bonus: bool):
    """Load m4_<method>_<env>.yaml as template, override variant fields."""
    template_path = CONFIG_ROOT / f"m4_{method}_{env}.yaml"
    cfg = load_config(template_path)
    apply_run_mode(cfg, "full")
    label = f"{integration}_{'bonus' if bonus else 'nobonus'}"
    cfg.experiment_name = f"m5_factorial_{method}_{env}_{label}"
    cfg.agent.params = copy.deepcopy(cfg.agent.params)
    cfg.agent.params["integration"] = integration
    cfg.agent.params["exploration_bonus"] = bonus
    return cfg


def _read_metrics(experiment_name: str) -> dict[str, Any]:
    with open(RESULTS_ROOT / experiment_name / "metrics.json") as f:
        return json.load(f)


def _read_m4_baseline(method: str, env: str) -> float | None:
    """Return the M4 baseline final_return_mean for (method, env), if available."""
    path = RESULTS_ROOT / f"m4_{method}_{env}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return float(json.load(f)["final_return_mean"])


def main() -> int:
    run = ScriptRun(script="m5_factorial_toys")
    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M5_factorial_toys.json"
    summary_path = out_dir / "stats_M5_factorial_toys_run.json"

    runs: list[tuple[str, str, str, bool, str]] = [
        (method, env, integ, bonus, f"m5_factorial_{method}_{env}_{integ}_{'bonus' if bonus else 'nobonus'}")
        for method in METHODS
        for env in ENVS
        for integ, bonus, _ in VARIANTS
    ]
    n = len(runs)
    print(f"[m5_factorial_toys] start: {n} runs at full M4 budget", flush=True)
    t_start = time.perf_counter()

    config_results: list[dict[str, Any]] = []
    failed: list[str] = []

    for i, (method, env, integration, bonus, label) in enumerate(runs, start=1):
        print(
            f"[m5_factorial_toys] ({i}/{n}) {method} on {env}: integration={integration} "
            f"exploration_bonus={bonus}",
            flush=True,
        )
        try:
            cfg = _build_factorial_config(method, env, integration, bonus)
        except Exception as e:
            print(f"[m5_factorial_toys] ({i}/{n}) {label} CONFIG FAILED: {e}", flush=True)
            failed.append(label)
            continue

        t0 = time.perf_counter()
        try:
            train_or_sweep(cfg)
        except SystemExit as e:
            if e.code != 0:
                print(f"[m5_factorial_toys] ({i}/{n}) {label} FAILED (exit {e.code})", flush=True)
                failed.append(label)
                continue
        elapsed = time.perf_counter() - t0
        total_min = (time.perf_counter() - t_start) / 60

        m = _read_metrics(cfg.experiment_name)
        finals = list(map(float, m["per_seed_final_return"]))
        mean = float(np.mean(finals))
        ci_lo, ci_hi = _bootstrap_ci(finals)
        baseline = _read_m4_baseline(method, env)
        config_results.append({
            "label": label,
            "method": method,
            "env": env,
            "integration": integration,
            "exploration_bonus": bonus,
            "experiment_name": cfg.experiment_name,
            "final_return_mean": mean,
            "final_return_ci95": [ci_lo, ci_hi],
            "per_seed_final_return": finals,
            "m4_baseline_mean": baseline,
            "delta_vs_m4": (mean - baseline) if baseline is not None else None,
            "iterations": int(m["iterations"]),
            "num_seeds": int(m["num_seeds"]),
        })
        delta_str = f" Δm4={mean-baseline:+.2f}" if baseline is not None else ""
        print(
            f"[m5_factorial_toys] ({i}/{n}) {label} done in {elapsed/60:.1f} min "
            f"| total {total_min:.1f} min | mean={mean:.2f}{delta_str}",
            flush=True,
        )

    if failed:
        run.fail(
            reason=f"{len(failed)}/{n} runs failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1

    by_method_env: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in config_results:
        by_method_env.setdefault((r["method"], r["env"]), []).append(r)

    summary_blocks: list[dict[str, Any]] = []
    for (method, env), rs in sorted(by_method_env.items()):
        baseline = rs[0]["m4_baseline_mean"]
        rs_sorted = sorted(rs, key=lambda x: x["final_return_mean"], reverse=True)
        summary_blocks.append({
            "method": method,
            "env": env,
            "m4_baseline_mean": baseline,
            "best_label": rs_sorted[0]["label"],
            "best_mean": rs_sorted[0]["final_return_mean"],
            "best_delta_vs_m4": rs_sorted[0]["delta_vs_m4"],
            "ranked_variants": [
                {
                    "label": r["label"],
                    "mean": r["final_return_mean"],
                    "delta_vs_m4": r["delta_vs_m4"],
                }
                for r in rs_sorted
            ],
        })

    stats = {
        "n_runs": n,
        "methods": METHODS,
        "envs": ENVS,
        "variants": [
            {"integration": iv, "exploration_bonus": bv, "label": lv}
            for iv, bv, lv in VARIANTS
        ],
        "configs": config_results,
        "summary_per_method_env": summary_blocks,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_elapsed = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "n_runs": n,
            "elapsed_min": round(total_elapsed, 2),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print("[m5_factorial_toys] === factorial summary ===", flush=True)
    for blk in summary_blocks:
        print(
            f"[m5_factorial_toys] {blk['method']:>8s} on {blk['env']:<14s} | "
            f"m4_baseline={blk['m4_baseline_mean']:.2f} | best={blk['best_label']:<18s} "
            f"mean={blk['best_mean']:.2f} (Δm4={blk['best_delta_vs_m4']:+.2f})",
            flush=True,
        )
        for r in blk["ranked_variants"]:
            d = f"Δm4={r['delta_vs_m4']:+6.2f}" if r["delta_vs_m4"] is not None else "Δm4= n/a"
            print(
                f"[m5_factorial_toys]   {r['label']:<18s} mean={r['mean']:7.2f} | {d}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
