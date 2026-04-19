"""Training entrypoint.

Loads a config, applies the run-mode flag, runs the outer loop for every seed,
writes per-experiment JSONs (config, summary, metrics, eval), and prints the
final OK / FAIL line. JSON is the contract — the summary file is what downstream
tools consume.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agents.dummy import DummyAgent
from envs.validation.dummy import DummyEnv
from training.config import (
    CONFIG_ROOT,
    ExperimentConfig,
    apply_run_mode,
    dump_config_json,
    load_config,
)
from training.rollout import rollout
from utils.script_output import ScriptRun, iso_now

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Registry (M0: only dummy env + dummy agent; later milestones register more)
# ---------------------------------------------------------------------------


def _build_env(cfg: ExperimentConfig):
    if cfg.env.name == "dummy":
        return DummyEnv(**cfg.env.params)
    raise ValueError(f"unknown env: {cfg.env.name!r}")


def _build_agent(cfg: ExperimentConfig):
    if cfg.agent.name == "dummy":
        return DummyAgent()
    raise ValueError(f"unknown agent: {cfg.agent.name!r}")


# ---------------------------------------------------------------------------
# Per-seed training loop
# ---------------------------------------------------------------------------


def _commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _train_one_seed(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    env = _build_env(cfg)
    agent = _build_agent(cfg)

    key = jax.random.PRNGKey(seed)
    init_key, key = jax.random.split(key)
    agent_state = agent.init(
        init_key, obs_size=env.obs_size, n_actions=env.n_actions, config=cfg.agent.params
    )

    per_iter_times: list[float] = []
    mean_returns: list[float] = []
    var_returns: list[float] = []

    for it in range(cfg.iterations):
        rollout_key, key = jax.random.split(key)
        t0 = time.perf_counter()
        traj = rollout(
            env,
            agent,
            agent_state,
            rollout_key,
            parallel_envs=cfg.parallel_envs,
            rollout_length=cfg.rollout_length,
        )
        # dummy update — exercises the pytree path only
        agent_state, _metrics = agent.update(agent_state, traj)
        # block to measure wall time
        jax.block_until_ready(traj["reward"])
        per_iter_times.append(time.perf_counter() - t0)

        rewards = np.asarray(traj["reward"])  # [T, parallel_envs]
        per_env_sum = rewards.sum(axis=0)  # [parallel_envs]
        mean_returns.append(float(per_env_sum.mean()))
        var_returns.append(float(per_env_sum.var()))

    return {
        "mean_return_per_iter": mean_returns,
        "var_return_per_iter": var_returns,
        "per_iter_times": per_iter_times,
    }


def _ci_across_seeds(per_seed_values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    """95% bootstrap CI (percentile method)."""
    arr = np.asarray(per_seed_values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def train(cfg: ExperimentConfig) -> dict[str, Any]:
    exp_dir = RESULTS_ROOT / cfg.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility record
    config_json_path = exp_dir / "config.json"
    cfg_dict = cfg.to_dict()
    cfg_dict["commit_hash"] = _commit_hash()
    with open(config_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    run = ScriptRun(script="train", run_mode=cfg.run_mode, config_used=cfg_dict)
    run.add_output(str(config_json_path))

    per_seed: list[dict[str, Any]] = []
    for s in range(cfg.num_seeds):
        seed = cfg.seed_base + s
        per_seed.append(_train_one_seed(cfg, seed))

    # Aggregate for metrics.json
    num_iters = cfg.iterations
    mean_curve = np.zeros(num_iters)
    var_curve = np.zeros(num_iters)
    for s in per_seed:
        mean_curve += np.asarray(s["mean_return_per_iter"])
        var_curve += np.asarray(s["var_return_per_iter"])
    mean_curve /= max(cfg.num_seeds, 1)
    var_curve /= max(cfg.num_seeds, 1)

    final_returns_per_seed = [s["mean_return_per_iter"][-1] for s in per_seed]
    ci_lo, ci_hi = _ci_across_seeds(final_returns_per_seed)

    # Timing summary (compile = first iter of first seed; per_iter = median of iters 2..N across seeds)
    first_iter_times = [s["per_iter_times"][0] for s in per_seed]
    later_iter_times: list[float] = []
    for s in per_seed:
        later_iter_times.extend(s["per_iter_times"][1:])
    compile_time = float(max(first_iter_times))
    per_iter_time = float(np.median(later_iter_times)) if later_iter_times else float("nan")
    total_time = float(sum(sum(s["per_iter_times"]) for s in per_seed))
    compile_ratio = compile_time / total_time if total_time > 0 else 0.0

    metrics = {
        "experiment_name": cfg.experiment_name,
        "run_mode": cfg.run_mode,
        "iterations": cfg.iterations,
        "num_seeds": cfg.num_seeds,
        "parallel_envs": cfg.parallel_envs,
        "rollout_length": cfg.rollout_length,
        "mean_return_per_iter": mean_curve.tolist(),
        "var_return_per_iter": var_curve.tolist(),
        "per_seed_final_return": final_returns_per_seed,
        "final_return_mean": float(np.mean(final_returns_per_seed)),
        "final_return_ci95": [ci_lo, ci_hi],
        "timing": {
            "compile_time_seconds": compile_time,
            "per_iter_time_seconds": per_iter_time,
            "iterations_run": cfg.iterations,
            "compile_ratio": compile_ratio,
        },
    }
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    run.add_output(str(metrics_path))

    eval_result = {
        "final_return_mean": metrics["final_return_mean"],
        "final_return_ci95": metrics["final_return_ci95"],
    }
    eval_path = exp_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    run.add_output(str(eval_path))

    summary_path = exp_dir / "summary.json"
    run.ok(
        key_stats={
            "method": cfg.agent.name,
            "env": cfg.env.name,
            "final_return": round(metrics["final_return_mean"], 4),
            "iterations": cfg.iterations,
            "num_seeds": cfg.num_seeds,
        },
        summary_path=summary_path,
        print_stats={
            "method": cfg.agent.name,
            "env": cfg.env.name,
            "final_return": round(metrics["final_return_mean"], 4),
        },
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="training.train")
    parser.add_argument("--config", required=True, help="path to experiment YAML")
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if args.super_fast and args.fast:
        raise SystemExit("--super-fast and --fast are mutually exclusive")
    mode = "super_fast" if args.super_fast else "fast" if args.fast else "full"

    cfg = load_config(args.config)
    apply_run_mode(cfg, mode)
    train(cfg)


if __name__ == "__main__":
    main()
