"""Training entrypoint.

Loads a config, applies the run-mode flag, runs the outer loop for every seed,
writes per-experiment JSONs (config, summary, metrics, eval), and prints the
final OK / FAIL line. JSON is the contract — the summary file is what
downstream tools consume.

The per-iteration step is a single JIT-compiled function: collect a rollout,
compute GAE, run PPO epochs. Compiles once per seed; subsequent iterations
reuse the cache.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from agents.dummy import DummyAgent
from agents.ppo import PPOAgent
from envs.mm_reduced import MMReducedEnv
from envs.validation.dummy import DummyEnv
from training.config import (
    CONFIG_ROOT,
    ExperimentConfig,
    apply_run_mode,
    load_config,
)
from training.rollout import rollout
from utils.script_output import ScriptRun, iso_now

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def _build_env(cfg: ExperimentConfig):
    if cfg.env.name == "dummy":
        return DummyEnv(**cfg.env.params)
    if cfg.env.name == "mm_reduced":
        return MMReducedEnv(**cfg.env.params)
    raise ValueError(f"unknown env: {cfg.env.name!r}")


def _build_agent(cfg: ExperimentConfig, env):
    if cfg.agent.name == "dummy":
        return DummyAgent(obs_size=env.obs_size, n_actions=env.n_actions)
    if cfg.agent.name == "ppo":
        return PPOAgent(
            obs_size=env.obs_size,
            n_actions=env.n_actions,
            **cfg.agent.params,
        )
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


def _make_iter_step(env, agent, parallel_envs: int, rollout_length: int) -> Callable:
    """Return a JIT'd (rollout + update) step closure."""

    @jax.jit
    def step(agent_state, key):
        rollout_key, key = jax.random.split(key)
        traj, final_obs = rollout(
            env, agent, agent_state, rollout_key,
            parallel_envs=parallel_envs, rollout_length=rollout_length,
        )
        new_state, update_metrics = agent.update(agent_state, traj, final_obs)
        # Summary stats captured inside JIT to avoid host round-trips.
        per_env_return = traj["reward"].sum(axis=0)  # [N]
        metrics = {
            **update_metrics,
            "mean_return": per_env_return.mean(),
            "var_return": per_env_return.var(),
        }
        return new_state, key, metrics

    return step


def _policy_action_probs(agent, agent_state) -> np.ndarray | None:
    """For PPO agents, extract π(a | inv) at every inventory one-hot."""
    if not hasattr(agent, "action_probs"):
        return None
    # Build identity matrix of inventory one-hots and query the policy.
    identity = jnp.eye(agent.obs_size, dtype=jnp.float32)

    @jax.jit
    def batched(state):
        return jax.vmap(lambda o: agent.action_probs(state, o))(identity)

    return np.asarray(batched(agent_state))


def _train_one_seed(cfg: ExperimentConfig, seed: int, seed_idx: int, num_seeds: int) -> dict[str, Any]:
    env = _build_env(cfg)
    agent = _build_agent(cfg, env)

    key = jax.random.PRNGKey(seed)
    init_key, key = jax.random.split(key)
    agent_state = agent.init(init_key)

    step_fn = _make_iter_step(env, agent, cfg.parallel_envs, cfg.rollout_length)

    per_iter_times: list[float] = []
    mean_returns: list[float] = []
    var_returns: list[float] = []

    # Progress cadence: ~10 prints per seed, never silent >30s on slow hosts.
    log_every = max(1, cfg.iterations // 10)
    prefix = f"[train] seed {seed_idx+1}/{num_seeds} (seed={seed})"

    for it in range(cfg.iterations):
        t0 = time.perf_counter()
        agent_state, key, metrics = step_fn(agent_state, key)
        # block once per iter so per_iter_times reflect real compute time
        jax.block_until_ready(metrics["mean_return"])
        dt = time.perf_counter() - t0
        per_iter_times.append(dt)
        mean_returns.append(float(metrics["mean_return"]))
        var_returns.append(float(metrics["var_return"]))

        is_first = it == 0
        is_last = it == cfg.iterations - 1
        if is_first or is_last or (it + 1) % log_every == 0:
            # Use iterations after compile for ETA (first iter includes compile time)
            if len(per_iter_times) > 1:
                median_step = float(np.median(per_iter_times[1:]))
                iters_left = cfg.iterations - (it + 1)
                eta_s = iters_left * median_step
            else:
                eta_s = float("nan")
            tag = " [compiled]" if is_first else ""
            print(
                f"{prefix} iter {it+1}/{cfg.iterations} "
                f"| return={mean_returns[-1]:.3f} "
                f"| step={dt:.2f}s "
                f"| eta_seed={eta_s:.1f}s{tag}",
                flush=True,
            )

    out: dict[str, Any] = {
        "seed": seed,
        "mean_return_per_iter": mean_returns,
        "var_return_per_iter": var_returns,
        "per_iter_times": per_iter_times,
    }
    probs = _policy_action_probs(agent, agent_state)
    if probs is not None:
        out["final_action_probs"] = probs.tolist()
    return out


def _ci_across_seeds(per_seed_values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
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

    config_json_path = exp_dir / "config.json"
    cfg_dict = cfg.to_dict()
    cfg_dict["commit_hash"] = _commit_hash()
    with open(config_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    run = ScriptRun(script="train", run_mode=cfg.run_mode, config_used=cfg_dict)
    run.add_output(str(config_json_path))

    print(
        f"[train] start: experiment={cfg.experiment_name} mode={cfg.run_mode} "
        f"seeds={cfg.num_seeds} iterations={cfg.iterations} "
        f"parallel_envs={cfg.parallel_envs} rollout_length={cfg.rollout_length}",
        flush=True,
    )
    t_start_all = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    for s in range(cfg.num_seeds):
        seed = cfg.seed_base + s
        t_seed = time.perf_counter()
        per_seed.append(_train_one_seed(cfg, seed, s, cfg.num_seeds))
        seed_time = time.perf_counter() - t_seed
        done = s + 1
        total_so_far = time.perf_counter() - t_start_all
        # Estimate remaining: later seeds reuse the JIT cache so they're faster
        # than the first. Use per-seed mean after seed 1.
        if done == 1 and cfg.num_seeds > 1:
            # Separate compile vs steady-state: first iter is compile-dominated.
            first_iter = per_seed[0]["per_iter_times"][0]
            rest = per_seed[0]["per_iter_times"][1:]
            steady_per_seed = float(np.median(rest)) * cfg.iterations if rest else seed_time
            remaining_s = steady_per_seed * (cfg.num_seeds - done)
        else:
            avg_per_seed = total_so_far / done
            remaining_s = avg_per_seed * (cfg.num_seeds - done)
        print(
            f"[train] seed {done}/{cfg.num_seeds} done in {seed_time:.1f}s "
            f"| final_return={per_seed[-1]['mean_return_per_iter'][-1]:.3f} "
            f"| eta_total={remaining_s:.1f}s",
            flush=True,
        )

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
        "agent": cfg.agent.name,
        "env": cfg.env.name,
        "iterations": cfg.iterations,
        "num_seeds": cfg.num_seeds,
        "parallel_envs": cfg.parallel_envs,
        "rollout_length": cfg.rollout_length,
        "mean_return_per_iter": mean_curve.tolist(),
        "var_return_per_iter": var_curve.tolist(),
        "per_seed_mean_return_per_iter": [s["mean_return_per_iter"] for s in per_seed],
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
    if "final_action_probs" in per_seed[0]:
        metrics["per_seed_final_action_probs"] = [s["final_action_probs"] for s in per_seed]

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
