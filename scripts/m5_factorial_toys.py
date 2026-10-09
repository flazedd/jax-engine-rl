"""M5 factorial toys: 4 variants × 3 toy envs = 12 runs.

Tests the integration ablation — (rl2, varibad) × (concat, hypernet) — on
the M4 toy environments (bandit, gridworld, regime_bandit). Templates from
`experiments/configs/m4_<method>_<env>.yaml` provide budgets and env
settings; orchestrator applies the integration override to `agent.params`
and renames `experiment_name`.

Pass criterion: variants should perform at parity-or-better with the M4
baseline (concat). A regression indicates a wiring bug; gains indicate
the integration addon is genuinely helpful on toys.

Usage:
  uv run python -m scripts.m5_factorial_toys
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

from utils.paths import experiment_dir, foundations_dir
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
    # (integration, label)
    ("concat",   "concat_nobonus"),
    ("hypernet", "hypernet_nobonus"),
]

# Matched-compute architecture knobs at ~5k parameters per cell, matching
# the rsmm_legacy configurations used on MarketMakingV1. Injected into the
# m4_<method>_<env>.yaml template before training.
MATCHED_COMPUTE_KNOBS = {
    ("rl2", "concat"): {
        "hidden_dim": 27,
        "policy_trunk_layers": 0,
        "policy_trunk_hidden": 0,
    },
    ("rl2", "hypernet"): {
        "hidden_dim": 24,
        "hypernet_target_hidden": 4,
        "hypernet_hidden": 8,
        "hypernet_init_scale": 0.01,
        "policy_trunk_layers": 0,
        "policy_trunk_hidden": 0,
    },
    ("varibad", "concat"): {
        "hidden_dim": 20,
        "latent_dim": 2,
        "policy_trunk_layers": 2,
        "policy_trunk_hidden": 20,
    },
    ("varibad", "hypernet"): {
        "hidden_dim": 18,
        "latent_dim": 2,
        "hypernet_target_hidden": 4,
        "hypernet_hidden": 8,
        "hypernet_init_scale": 0.01,
        "policy_trunk_layers": 2,
        "policy_trunk_hidden": 18,
    },
}
TOYS_NUM_SEEDS = 8


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _build_factorial_config(method: str, env: str, integration: str):
    """Load m4_<method>_<env>.yaml as template, override integration and
    matched-compute architecture knobs."""
    template_path = CONFIG_ROOT / f"m4_{method}_{env}.yaml"
    cfg = load_config(template_path)
    apply_run_mode(cfg, "full")
    cfg.experiment_name = f"m5_factorial_{method}_{env}_{integration}_nobonus"
    cfg.agent.params = copy.deepcopy(cfg.agent.params)
    cfg.agent.params["integration"] = integration
    # Inject matched-compute architecture knobs.
    knobs = MATCHED_COMPUTE_KNOBS.get((method, integration), {})
    for k, v in knobs.items():
        cfg.agent.params[k] = v
    cfg.num_seeds = TOYS_NUM_SEEDS
    return cfg


def _read_metrics(experiment_name: str) -> dict[str, Any]:
    with open(experiment_dir(experiment_name) / "metrics.json") as f:
        return json.load(f)


def _read_m4_baseline(method: str, env: str) -> float | None:
    """Return the M4 baseline final_return_mean for (method, env), if available."""
    # Historical baseline is an explicit, versioned input on a clean checkout.
    from scripts.restore_validation_baselines import SOURCE, EXPECTED
    import hashlib
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("Historical toy baseline checksum mismatch")
    rows = json.loads(raw)["key_stats"]["rows"]
    row = next(r for r in rows if r["method"] == method and r["env"] == env)
    return float(row["final_return_mean"])


def _num(v, sign: bool = False) -> str:
    """Format a value that may be absent.

    The M4 baseline is looked up from an earlier milestone's results, which a
    clean tree does not have. Reporting used to assume it was always present,
    so a missing baseline destroyed an hour of finished training at the print
    statement.
    """
    if v is None:
        return "n/a"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def main() -> int:
    run = ScriptRun(script="m5_factorial_toys")
    out_dir = foundations_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M5_factorial_toys.json"
    summary_path = out_dir / "stats_M5_factorial_toys_run.json"

    runs: list[tuple[str, str, str, str]] = [
        (method, env, integ, f"m5_factorial_{method}_{env}_{integ}_nobonus")
        for method in METHODS
        for env in ENVS
        for integ, _ in VARIANTS
    ]
    n = len(runs)
    print(f"[m5_factorial_toys] start: {n} runs at full M4 budget", flush=True)
    t_start = time.perf_counter()

    config_results: list[dict[str, Any]] = []
    failed: list[str] = []

    for i, (method, env, integration, label) in enumerate(runs, start=1):
        print(
            f"[m5_factorial_toys] ({i}/{n}) {method} on {env}: integration={integration}",
            flush=True,
        )
        try:
            cfg = _build_factorial_config(method, env, integration)
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
            {"integration": iv, "label": lv}
            for iv, lv in VARIANTS
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
            f"m4_baseline={_num(blk['m4_baseline_mean'])} | "
            f"best={blk['best_label']:<18s} "
            f"mean={_num(blk['best_mean'])} (Δm4={_num(blk['best_delta_vs_m4'], sign=True)})",
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
