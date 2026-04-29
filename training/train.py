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
import copy
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from agents.dummy import DummyAgent
from agents.ppo import PPOAgent
from agents.ppo_belief import PPOBeliefAgent
from agents.ppo_oracle import PPOOracleAgent
from agents.ppo_per_regime import PPOPerRegimeAgent
from agents.rl2 import RL2Agent
from agents.varibad import VariBADAgent
from envs.cartpole_regime_v1 import CartPoleRegimeV1
from envs.market_making_v1 import MarketMakingV1
from envs.validation.bandit import BanditEnv
from envs.validation.dummy import DummyEnv
from envs.validation.gridworld import GridworldEnv
from envs.validation.regime_bandit import RegimeBanditEnv
from envs.wrappers.belief_obs import BeliefObsEnv
from envs.wrappers.oracle_obs import OracleObsEnv
from envs.wrappers.rl2_obs import RL2ObsEnv
from envs.wrappers.stack_obs import StackObsEnv
from training.config import (
    CONFIG_ROOT,
    ExperimentConfig,
    apply_run_mode,
    load_config,
)
from training.recurrent_rollout import recurrent_rollout
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
    if cfg.env.name == "bandit":
        return BanditEnv(**cfg.env.params)
    if cfg.env.name == "gridworld":
        return GridworldEnv(**cfg.env.params)
    if cfg.env.name == "regime_bandit":
        return RegimeBanditEnv(**cfg.env.params)
    if cfg.env.name == "market_making_v1":
        return MarketMakingV1(**cfg.env.params)
    if cfg.env.name == "market_making_v1_oracle":
        return OracleObsEnv(inner=MarketMakingV1(**cfg.env.params))
    if cfg.env.name == "market_making_v1_belief":
        return BeliefObsEnv(inner=MarketMakingV1(**cfg.env.params))
    if cfg.env.name == "market_making_v1_stacked":
        params = dict(cfg.env.params)
        k = int(params.pop("stack_k", 4))
        return StackObsEnv(inner=MarketMakingV1(**params), k=k)
    if cfg.env.name == "market_making_v1_belief_constant":
        return BeliefObsEnv(
            inner=MarketMakingV1(**cfg.env.params), constant_belief=True
        )
    if cfg.env.name == "cartpole_regime_v1":
        return CartPoleRegimeV1(**cfg.env.params)
    if cfg.env.name == "cartpole_regime_v1_oracle":
        return OracleObsEnv(inner=CartPoleRegimeV1(**cfg.env.params))
    if cfg.env.name == "cartpole_regime_v1_belief":
        return BeliefObsEnv(inner=CartPoleRegimeV1(**cfg.env.params))
    if cfg.env.name == "cartpole_regime_v1_stacked":
        params = dict(cfg.env.params)
        k = int(params.pop("stack_k", 4))
        return StackObsEnv(inner=CartPoleRegimeV1(**params), k=k)
    raise ValueError(f"unknown env: {cfg.env.name!r}")


_PPO_AGENT_CLASSES = {
    "ppo": PPOAgent,
    "ppo_oracle": PPOOracleAgent,
    "ppo_belief": PPOBeliefAgent,
    "ppo_per_regime": PPOPerRegimeAgent,
}


def _build_agent(cfg: ExperimentConfig, env):
    if cfg.agent.name == "dummy":
        return DummyAgent(obs_size=env.obs_size, n_actions=env.n_actions)
    if cfg.agent.name in _PPO_AGENT_CLASSES:
        cls = _PPO_AGENT_CLASSES[cfg.agent.name]
        return cls(
            obs_size=env.obs_size,
            n_actions=env.n_actions,
            **cfg.agent.params,
        )
    if cfg.agent.name == "rl2":
        return RL2Agent(
            obs_size=env.obs_size,
            n_actions=env.n_actions,
            **cfg.agent.params,
        )
    if cfg.agent.name == "varibad":
        return VariBADAgent(
            obs_size=env.obs_size,
            n_actions=env.n_actions,
            **cfg.agent.params,
        )
    raise ValueError(f"unknown agent: {cfg.agent.name!r}")


def _maybe_wrap_env_for_agent(cfg: ExperimentConfig, env):
    """RL² and VariBAD both consume the augmented
    (obs, prev_action, prev_reward, prev_done) observation.
    """
    if cfg.agent.name in ("rl2", "varibad"):
        return RL2ObsEnv(inner=env)
    return env


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
    """Return a JIT'd (rollout + update) step closure.

    Dispatches on `agent.is_recurrent`: recurrent agents use `recurrent_rollout`
    which threads a per-env carry through the scan and resets it on episode
    boundaries; their `update` consumes `init_carry` / `final_carry` for BPTT.
    """
    if getattr(agent, "is_recurrent", False):
        @jax.jit
        def step(agent_state, key):
            rollout_key, key = jax.random.split(key)
            initial_carry = agent.init_carry(parallel_envs)
            traj, final_obs, init_carry, final_carry = recurrent_rollout(
                env, agent, agent_state, initial_carry, rollout_key,
                parallel_envs=parallel_envs, rollout_length=rollout_length,
            )
            new_state, update_metrics = agent.update(
                agent_state, traj, final_obs, init_carry, final_carry
            )
            per_env_return = traj["reward"].sum(axis=0)
            metrics = {
                **update_metrics,
                "mean_return": per_env_return.mean(),
                "var_return": per_env_return.var(),
            }
            return new_state, key, metrics
        return step

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
    env = _maybe_wrap_env_for_agent(cfg, env)
    agent = _build_agent(cfg, env)

    key = jax.random.PRNGKey(seed)
    init_key, key = jax.random.split(key)
    agent_state = agent.init(init_key)

    step_fn = _make_iter_step(env, agent, cfg.parallel_envs, cfg.rollout_length)

    per_iter_times: list[float] = []
    mean_returns: list[float] = []
    var_returns: list[float] = []
    # Capture every scalar metric returned by agent.update() per iteration.
    # Used by the M5 diagnostic pass to compare loss-component scales between
    # MM E_final and M4 bandit/gridworld where methods cleared the floor.
    loss_components: dict[str, list[float]] = {}

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
        for k, v in metrics.items():
            if k in ("mean_return", "var_return"):
                continue
            arr = np.asarray(v)
            if arr.shape != ():
                continue
            loss_components.setdefault(k, []).append(float(arr))

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
        "loss_components_per_iter": loss_components,
        "agent_state": agent_state,
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
        seed_out = _train_one_seed(cfg, seed, s, cfg.num_seeds)
        # Save trained agent_state to disk before discarding it from the
        # per-seed dict (the rest of train() doesn't need the params, only
        # learning curves; carrying the state forward bloats memory).
        agent_state = seed_out.pop("agent_state")
        ckpt_path = exp_dir / f"checkpoint_seed_{seed}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(agent_state, f)
        del agent_state  # free memory before next seed
        per_seed.append(seed_out)
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

    # Per-iteration loss components (not in metrics.json to keep that file
    # focused on what downstream stats consumers actually read).
    loss_keys = sorted(per_seed[0].get("loss_components_per_iter", {}).keys())
    if loss_keys:
        loss_components_payload = {
            "experiment_name": cfg.experiment_name,
            "agent": cfg.agent.name,
            "env": cfg.env.name,
            "iterations": cfg.iterations,
            "num_seeds": cfg.num_seeds,
            "per_seed": [
                {
                    "seed": s["seed"],
                    "components": s["loss_components_per_iter"],
                }
                for s in per_seed
            ],
        }
        loss_components_path = exp_dir / "loss_components_per_iter.json"
        with open(loss_components_path, "w") as f:
            json.dump(loss_components_payload, f)
        run.add_output(str(loss_components_path))

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


def _stationary_distribution_np(T: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix T (shape [n, n])."""
    n = T.shape[0]
    b = np.ones(n) / n
    for _ in range(5000):
        b = b @ T
    return b


def train_per_regime_sweep(cfg: ExperimentConfig) -> dict[str, Any]:
    """Train per-regime PPO: N separate instances, one per locked regime.

    Each locked-regime run gets the full iteration budget (matched compute,
    not divided). Seeds are held fixed across regimes so per-seed combined
    returns can be computed by stationary-weighted sum across regimes.

    Writes:
      - results/<cfg.experiment_name>_r{r}/    — per-regime metrics (one dir each)
      - results/<cfg.experiment_name>/metrics.json — aggregate metrics with a
        combined (stationary-weighted) learning curve and per-seed final return.

    The aggregate metrics JSON follows the same schema as `train()`'s output
    so downstream consumers (orchestrator, plotting) can treat per-regime PPO
    as a single method.
    """
    base_env_params = dict(cfg.env.params)
    n_regimes = int(base_env_params.get("n_regimes", 1))
    if n_regimes < 1:
        raise ValueError("per-regime sweep requires n_regimes >= 1")

    T = np.asarray(
        base_env_params.get("transition_matrix", [1.0]), dtype=np.float64
    ).reshape(n_regimes, n_regimes)
    stationary = _stationary_distribution_np(T)

    per_regime_metrics: list[dict[str, Any]] = []
    for r in range(n_regimes):
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.experiment_name = f"{cfg.experiment_name}_r{r}"
        sub_cfg.env.name = "market_making_v1"  # locked env, not a wrapper
        sub_cfg.env.params = {**base_env_params, "lock_regime": r}
        # Underlying trainer accepts only vanilla ppo agent — the _PerRegime_
        # label is our convention, not a loss change. Dispatch as plain PPO.
        sub_cfg.agent.name = "ppo"
        print(
            f"\n[train_per_regime] --- regime {r+1}/{n_regimes} "
            f"(lock_regime={r}) — experiment={sub_cfg.experiment_name} ---",
            flush=True,
        )
        m = train(sub_cfg)
        per_regime_metrics.append(m)

    num_iters = int(cfg.iterations)
    num_seeds = int(cfg.num_seeds)

    per_seed_curves = np.zeros((num_seeds, num_iters), dtype=np.float64)
    for r, m in enumerate(per_regime_metrics):
        arr = np.asarray(m["per_seed_mean_return_per_iter"], dtype=np.float64)
        per_seed_curves += stationary[r] * arr
    mean_curve = per_seed_curves.mean(axis=0)
    var_curve = per_seed_curves.var(axis=0)

    final_per_seed = [float(per_seed_curves[s, -1]) for s in range(num_seeds)]
    ci_lo, ci_hi = _ci_across_seeds(final_per_seed)

    combined = {
        "experiment_name": cfg.experiment_name,
        "run_mode": cfg.run_mode,
        "agent": "ppo_per_regime",
        "env": cfg.env.name,
        "iterations": num_iters,
        "num_seeds": num_seeds,
        "parallel_envs": int(cfg.parallel_envs),
        "rollout_length": int(cfg.rollout_length),
        "stationary_weights": stationary.tolist(),
        "mean_return_per_iter": mean_curve.tolist(),
        "var_return_per_iter": var_curve.tolist(),
        "per_seed_mean_return_per_iter": per_seed_curves.tolist(),
        "per_seed_final_return": final_per_seed,
        "final_return_mean": float(np.mean(final_per_seed)),
        "final_return_ci95": [ci_lo, ci_hi],
        "per_regime_final_return_means": [
            float(m["final_return_mean"]) for m in per_regime_metrics
        ],
        "per_regime_experiment_names": [
            m["experiment_name"] for m in per_regime_metrics
        ],
    }

    exp_dir = RESULTS_ROOT / cfg.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Same shared-schema wrapper as train(): config + summary + eval.
    cfg_dict = cfg.to_dict()
    cfg_dict["commit_hash"] = _commit_hash()
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    with open(exp_dir / "eval.json", "w") as f:
        json.dump(
            {
                "final_return_mean": combined["final_return_mean"],
                "final_return_ci95": combined["final_return_ci95"],
            },
            f,
            indent=2,
        )

    run = ScriptRun(script="train", run_mode=cfg.run_mode, config_used=cfg_dict)
    run.add_output(str(metrics_path))
    run.ok(
        key_stats={
            "method": "ppo_per_regime",
            "env": cfg.env.name,
            "final_return": round(combined["final_return_mean"], 4),
            "iterations": num_iters,
            "num_seeds": num_seeds,
        },
        summary_path=exp_dir / "summary.json",
        print_stats={
            "method": "ppo_per_regime",
            "env": cfg.env.name,
            "final_return": round(combined["final_return_mean"], 4),
        },
    )
    return combined


def train_or_sweep(cfg: ExperimentConfig) -> dict[str, Any]:
    if cfg.agent.name == "ppo_per_regime":
        return train_per_regime_sweep(cfg)
    return train(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="training.train")
    parser.add_argument("--config", required=True, help="path to experiment YAML")
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mid", action="store_true")
    args = parser.parse_args()

    if sum([args.super_fast, args.fast, args.mid]) > 1:
        raise SystemExit("--super-fast / --fast / --mid are mutually exclusive")
    mode = (
        "super_fast" if args.super_fast
        else "fast" if args.fast
        else "mid" if args.mid
        else "full"
    )

    cfg = load_config(args.config)
    apply_run_mode(cfg, mode)
    train_or_sweep(cfg)


if __name__ == "__main__":
    main()
