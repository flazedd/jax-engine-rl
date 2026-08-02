"""M5R — counterfactual belief-swap diagnostic on E_final.

Companion to `scripts.m5r_action_distributions`. That script measures what a
policy did across locked regimes; this one holds the observation fixed and
swaps only the belief, so a policy that responds to the regime by steering
inventory rather than by changing its action at a given inventory is still
credited, and every method is scored on the same observation grid.

Outputs:
  - results/M5R/final/m5r_belief_swap.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from evaluation.belief_swap import belief_swap_separation
from evaluation.posterior_probe import collect_probe_rollouts, load_experiment
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

N_REGIMES = 3
INV_MAX = 5
N_INV_DIMS = 2 * INV_MAX + 1  # inventory one-hot block of the observation

# (label, experiment dir, belief-swap family). References come from the
# matched-fairness family, so every row of the output table sits on the same
# per-step tuple, optimiser settings, budget and capacity. The pre-matched
# `m3_*` runs train on the unaugmented observation and must not be used here.
METHODS: list[tuple[str, str, str]] = [
    ("regime_agnostic_ppo", "m5r_matched_regime_agnostic",     "none"),
    ("belief_ppo",          "m5r_matched_belief",              "obs_belief"),
    ("oracle_ppo",          "m5r_matched_oracle",              "obs_belief"),
    ("stacked_obs_ppo",     "m5r_matched_stacked_obs",         "none"),
    ("rl2_concat",          "m5r_final_rl2_concat_e_final",    "rl2"),
    ("rl2_hypernet",        "m5r_final_rl2_hypernet_e_final",  "rl2"),
    ("varibad_concat",      "m5r_final_varibad_concat_e_final",    "varibad"),
    ("varibad_hypernet",    "m5r_final_varibad_hypernet_e_final",  "varibad"),
]


def _collect_feedforward(
    env: Any, agent: Any, agent_state: Any,
    n_rollouts: int, rollout_length: int, key: Any,
) -> dict[str, np.ndarray]:
    """Observations and inventory from a feedforward agent's rollouts.

    `collect_probe_rollouts` assumes a recurrent carry and an internal belief
    vector. Belief-PPO and Oracle-PPO have neither: their belief arrives in the
    observation, so only obs and inventory are needed here.
    """
    from evaluation.posterior_probe import get_inner_mm  # noqa: F401

    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)
    keys = jax.random.split(key, n_rollouts + 1)
    env_states, obses = reset_v(keys[:n_rollouts])
    key = keys[-1]

    def get_q(state):
        s = state
        while "q" not in s and "inner" in s:
            s = s["inner"]
        return s["q"]

    out_obs, out_q = [], []
    for _ in range(rollout_length):
        act_key, key = jax.random.split(key)
        act_keys = jax.random.split(act_key, n_rollouts)
        action, _extras, _ = jax.vmap(
            lambda o, k: agent.act(agent_state, o, k)
        )(obses, act_keys)
        out_obs.append(np.asarray(obses))
        out_q.append(np.asarray(jax.vmap(get_q)(env_states)))
        step_keys = jax.random.split(key, n_rollouts + 1)
        env_states, obses, _r, _d, _info = step_v(
            env_states, action, step_keys[:n_rollouts],
        )
        key = step_keys[-1]

    return {"obs": np.stack(out_obs), "q": np.stack(out_q)}


def _probe_one(
    experiment_name: str, family: str, n_rollouts: int, rollout_length: int,
    swap_history: bool = False, hold_belief_fixed: bool = False,
) -> dict[str, Any]:
    exp_dir = RESULTS_ROOT / experiment_name
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")

    extra_keys = ("log_var",) if family == "varibad" else ()
    per_seed = []
    per_seed_pools = []
    for seed in seeds:
        if family == "none":
            per_seed.append(0.0)
            continue

        n_bins = 2 * INV_MAX + 1
        beliefs_by_regime: list[list[np.ndarray]] = []
        log_vars_by_regime: list[list[np.ndarray]] = []
        obs_pool: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
        hist_by_regime: list[list[np.ndarray]] = []
        bundle = None
        for regime in range(N_REGIMES):
            bundle = load_experiment(
                exp_dir, seed, env_overrides={"lock_regime": regime},
            )
            rollout_key = jax.random.PRNGKey(seed * N_REGIMES + regime)
            if family == "obs_belief":
                data = _collect_feedforward(
                    bundle.env, bundle.agent, bundle.agent_state,
                    n_rollouts, rollout_length, rollout_key,
                )
            else:
                data = collect_probe_rollouts(
                    bundle.env, bundle.agent, bundle.agent_state,
                    n_rollouts=n_rollouts, rollout_length=rollout_length,
                    key=rollout_key, extra_keys=extra_keys,
                )
            if family == "obs_belief":
                # The regime block sits after the inventory one-hot, not at the
                # end: the augmented tuple wraps the belief wrapper, so the
                # previous action, reward and done flag trail it.
                from evaluation.posterior_probe import get_inner_mm
                lo = get_inner_mm(bundle.env).obs_size
                belief_slice = (lo, lo + N_REGIMES)
                bel = data["obs"][:, :, lo:lo + N_REGIMES].reshape(-1, N_REGIMES)
            else:
                belief_slice = None
                bel = data["belief"].reshape(-1, data["belief"].shape[-1])
            lv = (
                data["log_var"].reshape(-1, bel.shape[-1])
                if family == "varibad" else None
            )

            obs_flat = data["obs"].reshape(-1, data["obs"].shape[-1])
            q_flat = data["q"].reshape(-1).astype(int)
            q_bin_flat = q_flat + INV_MAX

            # Beliefs are bucketed by the inventory they were recorded at, so a
            # swapped belief is always one this method actually held at that
            # inventory. Observations are pooled across regimes at each
            # inventory, so the situation carries no regime identity.
            beliefs_by_regime.append(
                [bel[q_bin_flat == b] for b in range(n_bins)]
            )
            if family == "varibad":
                log_vars_by_regime.append(
                    [lv[q_bin_flat == b] for b in range(n_bins)]
                )
            if swap_history:
                hist_by_regime.append(
                    [obs_flat[q_bin_flat == b][:, N_INV_DIMS:]
                     for b in range(n_bins)]
                )
            for b in range(n_bins):
                sel = obs_flat[q_bin_flat == b]
                if len(sel):
                    obs_pool[b].append(sel)

        obs_by_inventory = [
            np.concatenate(p) if p else np.zeros((0, bundle.env.obs_size), np.float32)
            for p in obs_pool
        ]
        res = belief_swap_separation(
            bundle.agent, bundle.agent_state, family, obs_by_inventory,
            beliefs_by_regime,
            log_vars_by_regime if family == "varibad" else None,
            rng=np.random.default_rng(seed),
            hist_by_regime=hist_by_regime if swap_history else None,
            n_inventory_dims=N_INV_DIMS if swap_history else None,
            hold_belief_fixed=hold_belief_fixed,
            belief_slice=belief_slice,
        )
        per_seed.append(res["separation"])
        per_seed_pools.append(res["inventory_pool_sizes"])

    arr = np.asarray(per_seed, dtype=np.float64)
    boot = np.array([
        np.random.default_rng(s).choice(arr, size=len(arr), replace=True).mean()
        for s in range(10_000)
    ])
    return {
        "experiment_name": experiment_name,
        "family": family,
        "seeds": seeds,
        "per_seed_separation": arr.tolist(),
        "mean_separation": float(arr.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_belief_swap")
    parser.add_argument("--n-rollouts", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--hold-belief-fixed", action="store_true",
                        help="score the observation-history channel alone by "
                             "giving every regime the same pooled beliefs")
    parser.add_argument("--swap-history", action="store_true",
                        help="also swap the observation history block, so the "
                             "measure covers every regime-carrying input")
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "M5R" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_history_only" if args.hold_belief_fixed else (
        "_with_history" if args.swap_history else "")
    stats_path = out_dir / f"m5r_belief_swap{suffix}.json"
    summary_path = out_dir / f"m5r_belief_swap{suffix}_run.json"

    run = ScriptRun(script="m5r_belief_swap")
    t0 = time.perf_counter()
    by_method: dict[str, Any] = {}
    failed = []
    for label, experiment_name, family in METHODS:
        try:
            print(f"[m5r_swap] {label} ({family})", flush=True)
            r = _probe_one(
                experiment_name, family, args.n_rollouts, args.rollout_length,
                swap_history=args.swap_history,
                hold_belief_fixed=args.hold_belief_fixed,
            )
            by_method[label] = r
            print(
                f"[m5r_swap] {label:>22s} | separation "
                f"{r['mean_separation']:.3f} "
                f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]",
                flush=True,
            )
        except Exception as e:
            print(f"[m5r_swap] {experiment_name} FAILED: {e}", flush=True)
            failed.append(experiment_name)

    if failed:
        run.fail(
            reason=f"{len(failed)} methods failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1

    payload = {
        "env": "market_making_v1 (E_final), matched-tuning",
        "diagnostic": "counterfactual belief swap at fixed observation",
        "swap_history": bool(args.swap_history),
        "hold_belief_fixed": bool(args.hold_belief_fixed),
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "n_regimes": N_REGIMES,
        "inv_max": INV_MAX,
        "by_method": by_method,
    }
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))
    run.ok(
        key_stats={
            "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
            "n_methods": len(METHODS),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
