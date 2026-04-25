"""M4 full-budget evaluation across (method × validation env).

Runs PPO / RL² / VariBAD on bandit / gridworld / regime_bandit at the configs'
default full-mode budgets (200 iter × 512 envs × 3 seeds each). Aggregates
final per-seed returns into a single comparison table written to
`results/milestones/M4/method_ranking.json`.

This is the M4 method-validation deliverable: it confirms that the recurrent
(RL²) and variational (VariBAD) implementations clear the regime-agnostic
PPO floor on each validation env, which is the prerequisite for using the
same machinery on MM in M5.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.config import apply_run_mode, load_config
from training.train import train
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

CONFIGS: list[tuple[str, str, str]] = [
    # (method, env, config_name)
    ("ppo",     "bandit",        "m4_ppo_bandit.yaml"),
    ("ppo",     "gridworld",     "m4_ppo_gridworld.yaml"),
    ("ppo",     "regime_bandit", "m4_ppo_regime_bandit.yaml"),
    ("rl2",     "bandit",        "m4_rl2_bandit.yaml"),
    ("rl2",     "gridworld",     "m4_rl2_gridworld.yaml"),
    ("rl2",     "regime_bandit", "m4_rl2_regime_bandit.yaml"),
    ("varibad", "bandit",        "m4_varibad_bandit.yaml"),
    ("varibad", "gridworld",     "m4_varibad_gridworld.yaml"),
    ("varibad", "regime_bandit", "m4_varibad_regime_bandit.yaml"),
]


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _read_summary(experiment_name: str) -> dict[str, Any]:
    path = RESULTS_ROOT / experiment_name / "summary.json"
    with open(path) as f:
        return json.load(f)


def _per_seed_finals(experiment_name: str) -> list[float]:
    path = RESULTS_ROOT / experiment_name / "metrics.json"
    with open(path) as f:
        m = json.load(f)
    return list(m.get("per_seed_final_return", []))


def main() -> int:
    run = ScriptRun(script="m4_full_eval")

    out_dir = RESULTS_ROOT / "milestones" / "M4"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "method_ranking.json"

    n_total = len(CONFIGS)
    print(f"[m4_full_eval] start: running {n_total} (method × env) configs at full budget", flush=True)
    t_start = time.perf_counter()

    rows: list[dict[str, Any]] = []
    for i, (method, env, cfg_name) in enumerate(CONFIGS, start=1):
        cfg_path = CONFIG_ROOT / cfg_name
        cfg = load_config(cfg_path)
        apply_run_mode(cfg, "full")

        print(
            f"[m4_full_eval] ({i}/{n_total}) launching method={method} env={env} "
            f"config={cfg_name} iters={cfg.iterations} envs={cfg.parallel_envs} "
            f"seeds={cfg.num_seeds}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            train(cfg)
        except SystemExit as e:
            if e.code != 0:
                print(f"[m4_full_eval] ({i}/{n_total}) {method}/{env} FAILED (exit {e.code})", flush=True)
                rows.append({
                    "method": method, "env": env, "experiment_name": cfg.experiment_name,
                    "status": "fail", "reason": f"SystemExit {e.code}",
                })
                continue
        elapsed = time.perf_counter() - t0
        print(
            f"[m4_full_eval] ({i}/{n_total}) {method}/{env} done in {elapsed/60:.1f} min "
            f"| total elapsed {(time.perf_counter()-t_start)/60:.1f} min",
            flush=True,
        )

        finals = _per_seed_finals(cfg.experiment_name)
        mean = float(np.mean(finals)) if finals else float("nan")
        ci_lo, ci_hi = _bootstrap_ci(finals)
        rows.append({
            "method": method,
            "env": env,
            "experiment_name": cfg.experiment_name,
            "status": "ok",
            "n_seeds": len(finals),
            "per_seed_final_returns": finals,
            "final_return_mean": mean,
            "final_return_ci_lo": ci_lo,
            "final_return_ci_hi": ci_hi,
            "iterations": cfg.iterations,
            "parallel_envs": cfg.parallel_envs,
            "rollout_length": cfg.rollout_length,
        })

    # Build the comparison table: organize by env with PPO floor, RL², VariBAD.
    by_env: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_env.setdefault(row["env"], {})[row["method"]] = row

    table = []
    for env in ("bandit", "gridworld", "regime_bandit"):
        if env not in by_env:
            continue
        envrow = {"env": env}
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

    total_elapsed = time.perf_counter() - t_start
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        run.fail(
            reason=f"{len(failed)}/{len(rows)} runs failed: " + ", ".join(
                f"{r['method']}/{r['env']}" for r in failed
            ),
            summary_path=summary_path,
        )
        return 1

    run.ok(
        key_stats={
            "n_runs": len(rows),
            "elapsed_min": round(total_elapsed / 60, 2),
            "table": table,
            "rows": rows,
        },
        summary_path=summary_path,
    )

    # Final human-readable report.
    print("[m4_full_eval] === final method ranking ===", flush=True)
    for envrow in table:
        env = envrow["env"]
        ppo_floor = envrow.get("ppo_floor", float("nan"))
        rl2_v = envrow.get("rl2", float("nan"))
        vb_v = envrow.get("varibad", float("nan"))
        print(
            f"[m4_full_eval] {env:>14s} | ppo={ppo_floor:7.2f} | "
            f"rl2={rl2_v:7.2f} | varibad={vb_v:7.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
