"""Overnight v2 — extension experiments for the matched-compute thesis.

Three independent groups, each resumable, each writing its own stats JSON.

  Group A — Cartpole at matched protocol (n=12 meta-RL, n=8 refs, ~5k params).
            Re-runs cartpole_difficulty_sweep at the new protocol, then the
            posterior probe and hypothesis tests. Uses existing infrastructure;
            this script just orchestrates the call sequence.

  Group B — Kappa sensitivity sweep on MarketMakingV1.
            Three new envs at κ ∈ {0.02, 0.10, 0.20}; trains 3 refs + 4 meta-RL
            cells on each. The current paper's κ=0.05 results are reused.

  Group C — Persistence-very-hard on MarketMakingV1.
            One new env at diag=0.92 (was 0.96 at hard); 3 refs + 4 meta-RL
            cells. Maps the inference-dominated frontier flagged in the
            discussion's future-work paragraph.

Stages run sequentially. Each cell is resumable: if its metrics.json exists
with iterations and num_seeds at or above the configured target, it is
skipped. Kill and restart freely.

Usage:
  uv run python -m scripts.overnight_v2                   # full pipeline
  uv run python -m scripts.overnight_v2 --group cartpole  # only group A
  uv run python -m scripts.overnight_v2 --group kappa     # only group B
  uv run python -m scripts.overnight_v2 --group persist   # only group C
  uv run python -m scripts.overnight_v2 --skip cartpole   # skip group A
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
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
EXT_RESULTS = RESULTS_ROOT / "milestones" / "M5R_extension"

# -- Group A: cartpole ---------------------------------------------------------

CARTPOLE_SCRIPTS = [
    ["uv", "run", "python", "-m", "scripts.cartpole_difficulty_sweep", "--axis", "asymmetry"],
    ["uv", "run", "python", "-m", "scripts.cartpole_difficulty_sweep", "--axis", "persistence"],
    ["uv", "run", "python", "-m", "scripts.cartpole_posterior_probe"],
    ["uv", "run", "python", "-m", "scripts.cartpole_hypothesis_tests"],
]


# -- Groups B & C: new MM envs -------------------------------------------------

REFS: list[tuple[str, str]] = [
    ("regime_agnostic", "m3_regime_agnostic.yaml"),
    ("belief",          "m3_belief.yaml"),
    ("oracle",          "m3_oracle.yaml"),
]
CELLS: list[tuple[str, str]] = [
    ("rl2_concat",       "m5r_locked/rl2_concat.yaml"),
    ("rl2_hypernet",     "m5r_locked/rl2_hypernet.yaml"),
    ("varibad_concat",   "m5r_locked/varibad_concat.yaml"),
    ("varibad_hypernet", "m5r_locked/varibad_hypernet.yaml"),
]


def _load_env_params(env_yaml_rel: str) -> dict[str, Any]:
    """Read the `env.params` block from an env YAML."""
    env_yaml = CONFIG_ROOT / env_yaml_rel
    if not env_yaml.exists():
        raise FileNotFoundError(f"missing env config: {env_yaml}")
    with open(env_yaml) as f:
        d = yaml.safe_load(f)
    return d["env"]["params"]


def _build_cfg(base_yaml: str, env_yaml_rel: str, experiment_name: str):
    """Load base config, override ONLY env.params with the new env's params,
    and set experiment_name. Mirrors m6_difficulty_sweep._build_cell_cfg.
    Preserves env.name, agent.name, and other base settings."""
    cfg = load_config(CONFIG_ROOT / base_yaml)
    apply_run_mode(cfg, "full")
    cfg.env.params = copy.deepcopy(_load_env_params(env_yaml_rel))
    cfg.experiment_name = experiment_name
    return cfg


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
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _train_or_skip(cfg) -> tuple[dict[str, Any], bool, float]:
    """Train if not already done; return (metrics, skipped, elapsed_sec)."""
    existing = _read_metrics(cfg.experiment_name)
    if (
        existing
        and existing.get("iterations", 0) >= cfg.iterations
        and existing.get("num_seeds", 0) >= cfg.num_seeds
    ):
        return existing, True, float("nan")

    t0 = time.perf_counter()
    train_or_sweep(cfg)
    elapsed = time.perf_counter() - t0
    metrics = _read_metrics(cfg.experiment_name)
    if metrics is None:
        raise RuntimeError(f"{cfg.experiment_name}: metrics.json missing after train")
    return metrics, False, elapsed


def _summarise_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    finals = list(map(float, metrics.get("per_seed_final_return", [])))
    mean = float(np.mean(finals)) if finals else float("nan")
    ci_lo, ci_hi = _bootstrap_ci(finals) if finals else (float("nan"), float("nan"))
    return {
        "iterations": int(metrics.get("iterations", 0)),
        "num_seeds": int(metrics.get("num_seeds", 0)),
        "per_seed_final_return": finals,
        "final_return_mean": mean,
        "final_return_ci95": [ci_lo, ci_hi],
    }


def _train_one(label: str, base_yaml: str, env_yaml_rel: str, experiment_name: str) -> dict[str, Any]:
    cfg = _build_cfg(base_yaml, env_yaml_rel, experiment_name)
    print(f"[ovr]   {label}: training as {experiment_name}", flush=True)
    t0 = time.perf_counter()
    try:
        metrics, skipped, _ = _train_or_skip(cfg)
    except Exception as e:
        print(f"[ovr]   {label} FAILED: {e}", flush=True)
        return {"error": str(e), "experiment_name": experiment_name}
    elapsed = (time.perf_counter() - t0) / 60
    summary = _summarise_metrics(metrics)
    summary["skipped"] = skipped
    summary["experiment_name"] = experiment_name
    print(
        f"[ovr]   {label}: mean={summary['final_return_mean']:.2f} "
        f"({'skipped' if skipped else f'{elapsed:.1f}min'})",
        flush=True,
    )
    return summary


def _train_env(env_label: str, env_yaml_rel: str) -> dict[str, Any]:
    """Train all 3 refs + 4 cells on one new env. Refs first."""
    print(f"\n[ovr] === env: {env_label} ({env_yaml_rel}) ===", flush=True)
    out: dict[str, Any] = {"env_yaml": env_yaml_rel, "refs": {}, "cells": {}}

    for method, base_yaml in REFS:
        out["refs"][method] = _train_one(
            f"ref/{method}", base_yaml, env_yaml_rel,
            f"m5r_ext_ref_{method}_{env_label}",
        )

    for cell, base_yaml in CELLS:
        out["cells"][cell] = _train_one(
            f"cell/{cell}", base_yaml, env_yaml_rel,
            f"m5r_ext_cell_{cell}_{env_label}",
        )

    return out


def _aggregate_env(env_block: dict[str, Any]) -> dict[str, Any]:
    """Compute gap-decomposition + gap-closed fractions for one env."""
    refs = env_block.get("refs", {})
    floor = refs.get("regime_agnostic", {}).get("final_return_mean")
    belief = refs.get("belief", {}).get("final_return_mean")
    oracle = refs.get("oracle", {}).get("final_return_mean")

    total_gap = (oracle - floor) if (floor is not None and oracle is not None) else None
    compromise = (belief - floor) if (floor is not None and belief is not None) else None
    inference = (oracle - belief) if (belief is not None and oracle is not None) else None

    cells_summary: dict[str, Any] = {}
    for cell, data in env_block.get("cells", {}).items():
        cm = data.get("final_return_mean")
        if cm is None:
            continue
        gc_oracle = (
            (cm - floor) / (oracle - floor)
            if (floor is not None and oracle is not None and oracle != floor)
            else None
        )
        gc_belief = (
            (cm - floor) / (belief - floor)
            if (floor is not None and belief is not None and belief != floor)
            else None
        )
        cells_summary[cell] = {
            "final_return_mean": cm,
            "delta_vs_floor": (cm - floor) if floor is not None else None,
            "gap_closed_vs_oracle": gc_oracle,
            "gap_closed_vs_belief": gc_belief,
        }
    return {
        "refs": {
            "regime_agnostic": floor,
            "belief": belief,
            "oracle": oracle,
        },
        "gap_decomposition": {
            "total_gap": total_gap,
            "compromise_policy_cost": compromise,
            "inference_cost": inference,
        },
        "cells": cells_summary,
    }


# -- Cartpole orchestration ----------------------------------------------------


def _run_cartpole() -> dict[str, Any]:
    """Run the cartpole sweep + probe + tests as subprocesses. Each underlying
    script is itself resumable, so re-running this stage is safe."""
    print("\n[ovr] === GROUP A: cartpole at matched protocol ===", flush=True)
    out: dict[str, Any] = {"steps": []}
    for cmd in CARTPOLE_SCRIPTS:
        cmd_str = " ".join(cmd)
        print(f"[ovr]   $ {cmd_str}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        elapsed = (time.perf_counter() - t0) / 60
        ok = proc.returncode == 0
        print(
            f"[ovr]   {'OK' if ok else 'FAIL'} {cmd_str} ({elapsed:.1f}min)",
            flush=True,
        )
        out["steps"].append({"cmd": cmd_str, "rc": proc.returncode, "elapsed_min": elapsed})
        if not ok:
            out["error"] = f"step failed: {cmd_str}"
            break
    return out


# -- Main ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.overnight_v2")
    parser.add_argument("--group", choices=["cartpole", "kappa", "persist", "all"],
                        default="all")
    parser.add_argument("--skip", choices=["cartpole", "kappa", "persist", "none"],
                        default="none")
    args = parser.parse_args()

    EXT_RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = EXT_RESULTS / "overnight_v2_run.json"
    stats_path = EXT_RESULTS / "stats_overnight_v2.json"
    run = ScriptRun(script="overnight_v2", run_mode="full")

    do_cartpole = args.group in ("cartpole", "all") and args.skip != "cartpole"
    do_kappa    = args.group in ("kappa",    "all") and args.skip != "kappa"
    do_persist  = args.group in ("persist",  "all") and args.skip != "persist"

    t_start = time.perf_counter()
    stats: dict[str, Any] = {
        "groups_run": [
            *(("cartpole",) if do_cartpole else ()),
            *(("kappa",)    if do_kappa    else ()),
            *(("persist",)  if do_persist  else ()),
        ],
    }

    # Ordering: cheapest validation first (persist-very-hard, ~1.5h), then
    # medium (kappa, ~4h), then the largest (cartpole, ~7h). If the night
    # runs short we still get the highest-value validations done.
    kappa_envs = [
        ("kappa02", "envs/e6e_symmetric_kappa02.yaml"),
        ("kappa10", "envs/e6e_symmetric_kappa10.yaml"),
        ("kappa20", "envs/e6e_symmetric_kappa20.yaml"),
    ]
    persist_envs = [
        ("persistence_very_hard", "envs/m6_persistence_very_hard.yaml"),
    ]

    def _checkpoint() -> None:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    if do_persist:
        print("\n[ovr] === GROUP C: persistence-very-hard ===", flush=True)
        persist_block: dict[str, Any] = {}
        for env_label, env_yaml_rel in persist_envs:
            persist_block[env_label] = _train_env(env_label, env_yaml_rel)
            stats["persist_raw"] = persist_block
            stats["persist"] = {
                label: _aggregate_env(block)
                for label, block in persist_block.items()
            }
            _checkpoint()

    if do_kappa:
        print("\n[ovr] === GROUP B: kappa sensitivity sweep ===", flush=True)
        kappa_block: dict[str, Any] = {}
        for env_label, env_yaml_rel in kappa_envs:
            kappa_block[env_label] = _train_env(env_label, env_yaml_rel)
            stats["kappa_raw"] = kappa_block
            stats["kappa"] = {
                label: _aggregate_env(block)
                for label, block in kappa_block.items()
            }
            _checkpoint()

    if do_cartpole:
        stats["cartpole"] = _run_cartpole()
        _checkpoint()

    total_min = (time.perf_counter() - t_start) / 60
    stats["total_min"] = total_min

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))
    run.ok(
        key_stats={
            "total_min": round(total_min, 2),
            "groups_run": stats["groups_run"],
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print(f"\n[ovr] === DONE in {total_min:.1f} min ===", flush=True)
    print(f"[ovr] stats: {stats_path}", flush=True)
    print(f"[ovr] summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
