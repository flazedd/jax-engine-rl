"""Post-hoc aggregator for M4 full-budget eval.

The orchestrator (`scripts/m4_full_eval.py`) ran all 9 (method × env) configs
to completion and wrote each one's metrics.json, but its in-process table
build hit two bugs (wrong key path; unsupported `extra_summary` kwarg). This
script reads the on-disk results and produces the M4 method-ranking JSON +
human-readable table without retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from plotting.m4_plots import plot_m4_method_ranking
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"

CONFIGS: list[tuple[str, str, str]] = [
    ("ppo",     "bandit",        "m4_ppo_bandit"),
    ("ppo",     "gridworld",     "m4_ppo_gridworld"),
    ("ppo",     "regime_bandit", "m4_ppo_regime_bandit"),
    ("rl2",     "bandit",        "m4_rl2_bandit"),
    ("rl2",     "gridworld",     "m4_rl2_gridworld"),
    ("rl2",     "regime_bandit", "m4_rl2_regime_bandit"),
    ("varibad", "bandit",        "m4_varibad_bandit"),
    ("varibad", "gridworld",     "m4_varibad_gridworld"),
    ("varibad", "regime_bandit", "m4_varibad_regime_bandit"),
]


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        m = float(arr.mean()) if arr.size == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _per_seed_finals(experiment_name: str) -> list[float]:
    path = RESULTS_ROOT / experiment_name / "metrics.json"
    with open(path) as f:
        m = json.load(f)
    return list(m.get("per_seed_final_return", []))


def main() -> int:
    run = ScriptRun(script="m4_full_eval_aggregate")
    out_dir = RESULTS_ROOT / "milestones" / "M4"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "method_ranking.json"

    rows: list[dict[str, Any]] = []
    for method, env, experiment_name in CONFIGS:
        finals = _per_seed_finals(experiment_name)
        mean = float(np.mean(finals)) if finals else float("nan")
        ci_lo, ci_hi = _bootstrap_ci(finals)
        rows.append({
            "method": method,
            "env": env,
            "experiment_name": experiment_name,
            "n_seeds": len(finals),
            "per_seed_final_returns": finals,
            "final_return_mean": mean,
            "final_return_ci_lo": ci_lo,
            "final_return_ci_hi": ci_hi,
        })

    by_env: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_env.setdefault(r["env"], {})[r["method"]] = r

    table = []
    for env in ("bandit", "gridworld", "regime_bandit"):
        if env not in by_env:
            continue
        envrow: dict[str, Any] = {"env": env}
        ppo = by_env[env].get("ppo")
        rl2 = by_env[env].get("rl2")
        vb = by_env[env].get("varibad")
        if ppo:
            envrow["ppo_floor"] = ppo["final_return_mean"]
            envrow["ppo_ci"] = [ppo["final_return_ci_lo"], ppo["final_return_ci_hi"]]
        if rl2:
            envrow["rl2"] = rl2["final_return_mean"]
            envrow["rl2_ci"] = [rl2["final_return_ci_lo"], rl2["final_return_ci_hi"]]
            if ppo:
                envrow["rl2_clears_floor"] = rl2["final_return_mean"] > ppo["final_return_mean"]
        if vb:
            envrow["varibad"] = vb["final_return_mean"]
            envrow["varibad_ci"] = [vb["final_return_ci_lo"], vb["final_return_ci_hi"]]
            if ppo:
                envrow["varibad_clears_floor"] = vb["final_return_mean"] > ppo["final_return_mean"]
        table.append(envrow)

    fig_path = FIGURES_ROOT / "milestones" / "M4" / "m4_method_ranking.png"
    # Read the compute budget from one of the underlying training runs so
    # the chart footer matches reality (all M4 cells share the same budget).
    sample_meta_path = RESULTS_ROOT / CONFIGS[0][2] / "metrics.json"
    bar_budget: dict[str, Any] | None = None
    if sample_meta_path.exists():
        with open(sample_meta_path) as f:
            mm = json.load(f)
        bar_budget = {
            "iterations": int(mm.get("iterations", 0)) or None,
            "parallel_envs": int(mm.get("parallel_envs", 0)) or None,
            "rollout_length": int(mm.get("rollout_length", 0)) or None,
            "num_seeds": int(mm.get("num_seeds", 0)) or None,
        }
    plot_m4_method_ranking(table, fig_path, budget=bar_budget)

    run.ok(
        key_stats={
            "n_runs": len(rows),
            "table": table,
            "rows": rows,
            "figure_path": str(fig_path.relative_to(REPO_ROOT)),
        },
        summary_path=summary_path,
    )

    print("[m4_full_eval_aggregate] === final method ranking ===", flush=True)
    print(f"[m4_full_eval_aggregate] {'env':>14s} | {'ppo':>20s} | {'rl2':>20s} | {'varibad':>20s}",
          flush=True)
    for envrow in table:
        env = envrow["env"]
        def _fmt(method_key: str) -> str:
            m = envrow.get(method_key)
            ci = envrow.get(f"{method_key}_ci")
            if m is None:
                return f"{'-':>20s}"
            if ci and not np.isnan(ci[0]):
                return f"{m:6.2f} [{ci[0]:5.1f},{ci[1]:5.1f}]"
            return f"{m:6.2f}            "
        ppo_s = _fmt("ppo_floor") if envrow.get("ppo_floor") is not None else f"{'-':>20s}"
        # ppo uses "ppo_floor" key, others use direct method name
        ppo_m = envrow.get("ppo_floor")
        ppo_ci = envrow.get("ppo_ci")
        if ppo_m is not None and ppo_ci and not np.isnan(ppo_ci[0]):
            ppo_s = f"{ppo_m:6.2f} [{ppo_ci[0]:5.1f},{ppo_ci[1]:5.1f}]"
        else:
            ppo_s = f"{'-':>20s}"
        rl2_s = _fmt("rl2")
        vb_s = _fmt("varibad")
        clears = []
        if envrow.get("rl2_clears_floor"):
            clears.append("rl2>floor")
        if envrow.get("varibad_clears_floor"):
            clears.append("vb>floor")
        clear_s = " ".join(clears) if clears else ""
        print(
            f"[m4_full_eval_aggregate] {env:>14s} | {ppo_s} | {rl2_s} | {vb_s} {clear_s}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
