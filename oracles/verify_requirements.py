"""Verify R1–R4 for a given MM env config.

This is the M2 main tool. CLI:

    uv run python -m oracles.verify_requirements --env-config experiments/configs/envs/e2_fill_switched.yaml
    uv run python -m oracles.verify_requirements --env-config ... --fast
    uv run python -m oracles.verify_requirements --env-config ... --super-fast

Outputs:
  results/foundations/stats_M2_requirements.json — pass/fail per R.
  figures/appendix/fig_M2_*.png                     — the M2 figures.

Budget per PPO run is deliberately short. M2 is a diagnostic; tight CIs and
final convergence are what M1 / M3 enforce with full budgets. The threshold
for R2 is therefore 0.85 (diagnostic), not M1's 0.95.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from beliefs.hmm_posterior import entropy as belief_entropy
from envs.market_making_v1 import MarketMakingV1
from envs.wrappers.belief_obs import BeliefObsEnv
from oracles.value_iteration import (
    compromise_policy_expected_returns,
    policy_disagreement,
    solve_value_iteration,
    wrong_regime_value_loss,
)
from plotting.m2_plots import (
    plot_belief_ppo_gap,
    plot_per_regime_ppo,
    plot_policy_heatmap,
    plot_posterior_entropy,
    plot_value_loss_distribution,
)
from training.config import (
    AgentConfig,
    EnvConfig,
    ExperimentConfig,
    _load_yaml_with_extends,
    apply_run_mode,
)
from training.train import train
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"

from utils.paths import foundations_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Pass thresholds — the single source of truth
# ---------------------------------------------------------------------------
# These are the values reported in the thesis (Table "Environment Validation
# Requirements"). Import them rather than restating the numbers: they were
# duplicated in prose once already and drifted (R4 was cited as 0.30 in one
# place while the check ran at 0.35).
R1_MIN_DISAGREE_FRAC = 0.80  # fraction of inventory levels where policies differ
R2_MIN_RATIO = 0.85  # worst-regime PPO / VI return ratio
R3_MAX_RECOVERY = 0.90  # regime-agnostic / Belief-PPO return ratio
R4_MIN_ENTROPY_DECAY = 0.35  # posterior entropy decay by episode midpoint

THRESHOLDS = {
    "R1": {"statistic": "fraction_disagreeing_states", "op": ">=", "value": R1_MIN_DISAGREE_FRAC},
    "R2": {"statistic": "min_ratio", "op": ">=", "value": R2_MIN_RATIO},
    "R3": {"statistic": "recovery_ratio", "op": "<=", "value": R3_MAX_RECOVERY},
    "R4": {"statistic": "entropy_decay_fraction", "op": ">=", "value": R4_MIN_ENTROPY_DECAY},
}


# ---------------------------------------------------------------------------
# Short-budget PPO runner
# ---------------------------------------------------------------------------


def _budget(run_mode: str) -> dict[str, int]:
    """Return (iterations, parallel_envs, rollout_length, num_seeds) budget."""
    if run_mode == "super_fast":
        return dict(iterations=2, parallel_envs=16, rollout_length=32, num_seeds=1)
    if run_mode == "fast":
        return dict(iterations=20, parallel_envs=128, rollout_length=128, num_seeds=1)
    # The M2 diagnostic full-mode budget. Bumped to 80 iter so that R2
    # (per-regime PPO vs VI) clears its 0.85-ratio threshold with
    # headroom rather than landing on the razor edge — at 40 iter, even
    # the medium-difficulty env only just cleared. R3 and R4 are
    # bottlenecked by inference / envelope size, not PPO budget, so the
    # extra iterations are essentially free for them. num_seeds=5
    # tightens the bootstrap CI on R3's gap-to-ci ratio enough that
    # narrow-envelope cells clear the 3.0 threshold reliably.
    return dict(iterations=80, parallel_envs=256, rollout_length=128, num_seeds=5)


def _run_short_ppo(
    experiment_name: str,
    env_name: str,
    env_params: dict[str, Any],
    agent_name: str,
    run_mode: str,
) -> dict[str, Any]:
    """Run a short PPO training job and return the training metrics.

    Bypasses the CLI — constructs an ExperimentConfig and calls train().
    """
    budget = _budget(run_mode)
    cfg = ExperimentConfig(
        experiment_name=experiment_name,
        env=EnvConfig(name=env_name, params=dict(env_params)),
        agent=AgentConfig(name=agent_name, params={}),
        iterations=budget["iterations"],
        parallel_envs=budget["parallel_envs"],
        rollout_length=budget["rollout_length"],
        num_seeds=budget["num_seeds"],
        seed_base=0,
        run_mode=run_mode if run_mode != "full" else "full",
    )
    # Force full-mode to avoid apply_run_mode overriding our chosen budget.
    # train() just reads cfg as-is.
    return train(cfg)


# ---------------------------------------------------------------------------
# Posterior entropy simulation (R4)
# ---------------------------------------------------------------------------


def _simulate_belief_trajectories(
    env: MarketMakingV1,
    n_envs: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Simulate n_envs random-policy trajectories, record belief at each step.

    Returns an [episode_length, n_envs, n_regimes] numpy array of beliefs
    *after* each step (what the agent would see as input at that step).
    """
    import jax
    import jax.numpy as jnp

    wrapped = BeliefObsEnv(inner=env)
    T = env.episode_length
    n_reg = env.n_regimes

    def one_traj(key):
        reset_key, key = jax.random.split(key)
        state, _ = wrapped.reset(reset_key)

        def step_body(carry, _):
            state, key = carry
            a_key, s_key, key = jax.random.split(key, 3)
            action = jax.random.randint(a_key, (), 0, env.n_actions)
            new_state, _obs, _r, _d, _info = wrapped.step(state, action, s_key)
            return (new_state, key), new_state["belief"]

        _, beliefs = jax.lax.scan(step_body, (state, key), xs=None, length=T)
        return beliefs  # [T, n_reg]

    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    beliefs = jax.vmap(one_traj)(keys)  # [n_envs, T, n_reg]
    return np.asarray(jnp.transpose(beliefs, (1, 0, 2)))


def _entropy_over_time(beliefs: np.ndarray) -> np.ndarray:
    """Return mean entropy [T] across envs, given beliefs [T, N, n_regimes]."""
    b = np.clip(beliefs, 1e-12, 1.0)
    ent = -np.sum(b * np.log(b), axis=-1)  # [T, N]
    return ent.mean(axis=1)


# ---------------------------------------------------------------------------
# CI helpers
# ---------------------------------------------------------------------------


def _ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def verify(
    env_config_path: Path,
    run_mode: str,
    fig_dir: Path | None = None,
) -> dict[str, Any]:
    """Check R1-R4 for one env config.

    fig_dir overrides where the diagnostic figures are written. Callers that
    must not disturb the committed figures (smoke tests) pass a scratch
    directory; the default is the M2 figure directory the thesis appendix
    draws from.
    """
    env_cfg = _load_yaml_with_extends(env_config_path)["env"]
    env_params = env_cfg["params"]
    env = MarketMakingV1(**env_params)

    env_version = env_config_path.stem
    # Reduced-mode runs get their own experiment namespace. train() writes to
    # results/{experiment_name}/, and plotting.regenerate_figures rebuilds the
    # R2 appendix figure from the per-regime metrics.json there — so a smoke
    # test sharing the namespace would silently replace full-budget learning
    # curves with 2-iteration ones.
    exp_prefix = f"m2_verify_{env_version}" + ("" if run_mode == "full" else f"_{run_mode}")
    n_ppo_runs = env.n_regimes + 3  # per-regime × n_regimes + agnostic + oracle + belief

    print(
        f"[verify] env={env_version} run_mode={run_mode} "
        f"n_regimes={env.n_regimes} "
        f"(will run: VI → {n_ppo_runs} PPO sub-runs → posterior sim → figures)",
        flush=True,
    )
    t_verify_start = time.perf_counter()

    # ------- R1: VI-based policy disagreement + value loss --------------
    print("[verify] [step 1] running VI on full-info MDP...", flush=True)
    t0 = time.perf_counter()
    vi = solve_value_iteration(env)
    print(f"[verify] [step 1] VI done in {time.perf_counter() - t0:.2f}s", flush=True)
    disagree_frac, disagree_mask = policy_disagreement(vi)
    mean_loss, rel_loss, per_state_loss = wrong_regime_value_loss(env, vi)
    per_regime_optima = vi.per_regime_expected_episode_return.astype(float)
    r1_pass = bool(disagree_frac >= R1_MIN_DISAGREE_FRAC)
    print(
        f"[verify] R1: disagree_frac={disagree_frac:.3f} "
        f"rel_loss={rel_loss:.3f} pass={r1_pass}",
        flush=True,
    )

    # ------- R2: short per-regime PPO vs VI per regime ------------------
    per_regime_metrics: list[dict[str, Any]] = []
    per_regime_ratios: list[float] = []
    ppo_runs_done = 0
    for r in range(env.n_regimes):
        ppo_runs_done += 1
        print(
            f"[verify] [step 2] PPO {ppo_runs_done}/{n_ppo_runs}: "
            f"per-regime (r={r})",
            flush=True,
        )
        t0 = time.perf_counter()
        params = {**env_params, "lock_regime": r}
        m = _run_short_ppo(
            experiment_name=f"{exp_prefix}_per_regime_{r}",
            env_name="market_making_v1",
            env_params=params,
            agent_name="ppo_per_regime",
            run_mode=run_mode,
        )
        per_regime_metrics.append(m)
        ratio = float(m["final_return_mean"] / per_regime_optima[r])
        per_regime_ratios.append(ratio)
        print(
            f"[verify] [step 2] PPO {ppo_runs_done}/{n_ppo_runs} done in "
            f"{time.perf_counter() - t0:.1f}s | final={m['final_return_mean']:.2f} "
            f"vi={per_regime_optima[r]:.2f} ratio={ratio:.3f}",
            flush=True,
        )
    r2_min = float(min(per_regime_ratios))
    r2_pass = bool(r2_min >= R2_MIN_RATIO)

    # ------- R3: regime-agnostic PPO vs Belief-PPO ----------------------
    ppo_runs_done += 1
    print(
        f"[verify] [step 3] PPO {ppo_runs_done}/{n_ppo_runs}: regime-agnostic",
        flush=True,
    )
    t0 = time.perf_counter()
    m_agn = _run_short_ppo(
        experiment_name=f"{exp_prefix}_regime_agnostic",
        # R3 compares the agents the thesis compares, so the references are
        # measured on the matched tuple u_t rather than the base observation.
        # On the base observation the agent cannot read regime evidence from the
        # previous reward, which understates what it recovers and passes
        # instances whose attainable gap is too narrow to measure against.
        env_name="market_making_v1_augmented",
        env_params=env_params,
        agent_name="ppo",
        run_mode=run_mode,
    )
    print(
        f"[verify] [step 3] PPO {ppo_runs_done}/{n_ppo_runs} done in "
        f"{time.perf_counter() - t0:.1f}s | final={m_agn['final_return_mean']:.2f}",
        flush=True,
    )

    ppo_runs_done += 1
    print(
        f"[verify] [step 3] PPO {ppo_runs_done}/{n_ppo_runs}: oracle",
        flush=True,
    )
    t0 = time.perf_counter()
    m_oracle = _run_short_ppo(
        experiment_name=f"{exp_prefix}_oracle",
        env_name="market_making_v1_oracle_augmented",
        env_params=env_params,
        agent_name="ppo_oracle",
        run_mode=run_mode,
    )
    print(
        f"[verify] [step 3] PPO {ppo_runs_done}/{n_ppo_runs} done in "
        f"{time.perf_counter() - t0:.1f}s | final={m_oracle['final_return_mean']:.2f}",
        flush=True,
    )
    agn_mean = float(m_agn["final_return_mean"])
    agn_ci = [float(x) for x in m_agn["final_return_ci95"]]
    oracle_mean = float(m_oracle["final_return_mean"])
    oracle_ci = [float(x) for x in m_oracle["final_return_ci95"]]

    # Paired bootstrap on the seed-level difference. The experiment uses fixed
    # seeds {0..N-1} across methods for paired tests — summing independent
    # per-method CI widths would vastly overstate uncertainty on the gap.
    agn_per_seed = np.asarray(m_agn["per_seed_final_return"], dtype=float)
    oracle_per_seed = np.asarray(m_oracle["per_seed_final_return"], dtype=float)
    diffs = oracle_per_seed - agn_per_seed
    gap_absolute = float(diffs.mean())
    gap_ci_lo, gap_ci_hi = _ci(diffs.tolist())
    gap_ci_width = float(gap_ci_hi - gap_ci_lo)
    gap_ci_width_safe = max(gap_ci_width, 1e-6)
    gap_to_ci_ratio = float(gap_absolute / gap_ci_width_safe)
    # R3 is stated against Belief-PPO, which the analytical posterior makes the
    # level a method can reach, and which is the denominator of the gap-closed
    # fraction. It is therefore computed after the Belief-PPO run below.
    # gap_to_ci_ratio is reported for reference but does not gate the check.

    # ------- R4: posterior entropy + Belief-PPO ------------------------
    print(
        "[verify] [step 4] simulating posterior entropy under random policy (256 trajectories)",
        flush=True,
    )
    t0 = time.perf_counter()
    beliefs = _simulate_belief_trajectories(env, n_envs=256, seed=0)
    ent_curve = _entropy_over_time(beliefs)
    initial_ent = float(np.log(env.n_regimes))
    ent_t1 = float(ent_curve[0])  # belief after 1 observation
    ent_mid = float(ent_curve[env.episode_length // 2])
    decay_frac = float(1.0 - ent_mid / max(ent_t1, 1e-12))
    print(
        f"[verify] [step 4] posterior sim done in {time.perf_counter() - t0:.2f}s | "
        f"ent(t=1)={ent_t1:.3f} ent(t=mid)={ent_mid:.3f} decay={decay_frac:.3f}",
        flush=True,
    )

    ppo_runs_done += 1
    print(
        f"[verify] [step 4] PPO {ppo_runs_done}/{n_ppo_runs}: belief",
        flush=True,
    )
    t0 = time.perf_counter()
    m_belief = _run_short_ppo(
        experiment_name=f"{exp_prefix}_belief",
        env_name="market_making_v1_belief_augmented",
        env_params=env_params,
        agent_name="ppo_belief",
        run_mode=run_mode,
    )
    print(
        f"[verify] [step 4] PPO {ppo_runs_done}/{n_ppo_runs} done in "
        f"{time.perf_counter() - t0:.1f}s | final={m_belief['final_return_mean']:.2f}",
        flush=True,
    )
    belief_mean = float(m_belief["final_return_mean"])

    recovery_ratio = float(agn_mean / belief_mean) if abs(belief_mean) > 1e-9 else 1.0
    r3_pass = bool(recovery_ratio <= R3_MAX_RECOVERY)
    print(
        f"[verify] R3: recovery_ratio={recovery_ratio:.3f} (<={R3_MAX_RECOVERY}) "
        f"agnostic={agn_mean:.2f} belief={belief_mean:.2f} "
        f"gap_to_ci_ratio={gap_to_ci_ratio:.2f} pass={r3_pass}",
        flush=True,
    )
    belief_ci = [float(x) for x in m_belief["final_return_ci95"]]
    oracle_gap = float(oracle_mean - agn_mean)
    gap_closure = (
        float((belief_mean - agn_mean) / oracle_gap) if abs(oracle_gap) > 1e-9 else 0.0
    )
    r4_pass = bool(decay_frac >= R4_MIN_ENTROPY_DECAY)

    # ------- Compromise-policy sanity check ----------------------------
    compromise_per_regime, compromise_mixed, compromise_policy = (
        compromise_policy_expected_returns(env, vi)
    )

    all_pass = bool(r1_pass and r2_pass and r3_pass and r4_pass)

    stats = {
        "env_version": env_version,
        "env_config_path": str(env_config_path),
        "run_mode": run_mode,
        "ppo_budget": _budget(run_mode),
        "thresholds": THRESHOLDS,
        "parameters": env_params,
        "R1_policy_disagreement": {
            "fraction_disagreeing_states": disagree_frac,
            "mean_wrong_regime_value_loss": mean_loss,
            "mean_relative_value_loss_at_disagreeing_states": rel_loss,
            "pass": r1_pass,
        },
        "R2_per_regime_ppo_vs_vi": {
            **{f"regime_{r}_ratio": per_regime_ratios[r] for r in range(env.n_regimes)},
            "min_ratio": r2_min,
            "pass": r2_pass,
        },
        "R3_mixed_gap": {
            "regime_agnostic_return_mean": agn_mean,
            "regime_agnostic_return_ci": agn_ci,
            "oracle_ppo_return_mean": oracle_mean,
            "oracle_ppo_return_ci": oracle_ci,
            "paired_gap_per_seed": diffs.tolist(),
            "gap_absolute": gap_absolute,
            "gap_ci": [gap_ci_lo, gap_ci_hi],
            "gap_ci_width": gap_ci_width,
            "gap_to_ci_ratio": gap_to_ci_ratio,
            "recovery_ratio": recovery_ratio,
            "pass": r3_pass,
        },
        "R4_inferability": {
            "mean_posterior_entropy_at_t1": ent_t1,
            "mean_posterior_entropy_at_mid_episode": ent_mid,
            "entropy_decay_fraction": decay_frac,
            "belief_ppo_return_mean": belief_mean,
            "belief_ppo_return_ci": belief_ci,
            "belief_ppo_gap_closure_fraction": gap_closure,
            "pass": r4_pass,
        },
        "all_pass": all_pass,
        "vi": {
            "per_regime_expected_episode_return": per_regime_optima.tolist(),
            "mixed_expected_episode_return": float(vi.mixed_expected_episode_return),
            "policy": vi.policy.tolist(),
        },
        "compromise_policy": {
            "policy_per_inventory": compromise_policy.tolist(),
            "per_regime_expected_episode_return": compromise_per_regime.tolist(),
            "mixed_expected_episode_return": compromise_mixed,
        },
    }

    # ------- Figures ---------------------------------------------------
    fig_dir = fig_dir if fig_dir is not None else FIGURES_ROOT / "appendix"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_policy_heatmap(vi, env, fig_dir / "fig_M2_R1_policy_heatmap.png")
    plot_value_loss_distribution(
        per_state_loss, fig_dir / "fig_M2_R1_value_loss_distribution.png"
    )
    plot_per_regime_ppo(
        per_regime_metrics, per_regime_optima, fig_dir / "fig_M2_R2_per_regime_ppo.png"
    )
    plot_posterior_entropy(ent_curve, fig_dir / "fig_M2_R4_posterior_entropy.png")
    plot_belief_ppo_gap(
        {
            "regime_agnostic": (agn_mean, agn_ci),
            "belief": (belief_mean, belief_ci),
            "oracle": (oracle_mean, oracle_ci),
        },
        fig_dir / "fig_M2_R4_belief_ppo_gap.png",
    )

    print(
        f"[verify] all steps complete in {time.perf_counter() - t_verify_start:.1f}s "
        f"| R1={r1_pass} R2={r2_pass} R3={r3_pass} R4={r4_pass} "
        f"all_pass={all_pass}",
        flush=True,
    )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(prog="oracles.verify_requirements")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if args.super_fast and args.fast:
        raise SystemExit("--super-fast and --fast are mutually exclusive")
    run_mode = "super_fast" if args.super_fast else "fast" if args.fast else "full"

    env_config_path = Path(args.env_config)
    if not env_config_path.exists():
        env_config_path = REPO_ROOT / env_config_path

    results_dir = foundations_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    run = ScriptRun(script="verify_requirements", run_mode=run_mode)
    summary_path = results_dir / "summary_verify.json"
    stats_path = results_dir / "stats_M2_requirements.json"

    t0 = time.perf_counter()
    try:
        stats = verify(env_config_path, run_mode)
    except Exception as e:
        run.fail(reason=f"{type(e).__name__}: {e}", summary_path=summary_path)
        raise
    elapsed = time.perf_counter() - t0

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    # Add figures to outputs so the summary reflects them. `verify` renders them
    # into the appendix tree when no explicit directory is passed, which is the
    # case for this CLI path.
    for p in (FIGURES_ROOT / "appendix").glob("fig_M2_*.png"):
        run.add_output(str(p))

    key_stats = {
        "env_version": stats["env_version"],
        "R1": stats["R1_policy_disagreement"]["pass"],
        "R2": stats["R2_per_regime_ppo_vs_vi"]["pass"],
        "R3": stats["R3_mixed_gap"]["pass"],
        "R4": stats["R4_inferability"]["pass"],
        "all_pass": stats["all_pass"],
        "wall_time_seconds": round(elapsed, 2),
    }
    run.ok(key_stats=key_stats, summary_path=summary_path, print_stats=key_stats)
    return 0 if stats["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
