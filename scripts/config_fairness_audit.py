"""Verify that every method in a comparison is treated identically.

A comparison between methods is only interpretable if the methods differ in the
thing under study and nothing else. This audits the four ways that can silently
break, for each evaluated family:

  1. Inputs      — every method reads the per-step tuple
                   u_t = [o_t, a_{t-1}, r_{t-1}, d_{t-1}]. Belief-PPO and
                   Oracle-PPO additionally read their regime signal, and the
                   stacked-observation baseline reads K such tuples; both are
                   the point of those methods, so their observation size is
                   checked against the composition expected for their role
                   rather than against the references'.
  2. Optimiser   — epochs, minibatching, learning rate, clipping, discount and
                   GAE-lambda are common. `ent_coef` is exempt: the meta-RL
                   methods take the small in-trial exploration bonus that is
                   standard in their implementations.
  3. Budget      — iterations, parallel envs, rollout length, seeds and the
                   seed base are common, so every method sees the same number
                   of environment steps under the same seed set.
  4. Capacity    — every method is within `--tol` of the parameter budget, and
                   the two conditioning variants of a method share an identical
                   belief encoder so only the conditioning architecture differs.

Usage:
  uv run python -m scripts.config_fairness_audit
  uv run python -m scripts.config_fairness_audit --family cartpole
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

from training.config import load_config
from training.train import _build_agent, _build_env, _maybe_wrap_env_for_agent
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
RESULTS_ROOT = REPO_ROOT / "results"

# role ∈ {reference, stacked, regime, meta}: determines the observation
# composition expected for that method.
FAMILIES: dict[str, dict[str, Any]] = {
    "market_making": {
        "budget": 5000,
        "base_obs": 11,
        "methods": [
            ("regime_agnostic", "m5r_matched/regime_agnostic.yaml", "reference"),
            ("stacked_obs", "m5r_matched/stacked_obs.yaml", "stacked"),
            ("belief_ppo", "m5r_matched/belief_ppo.yaml", "regime"),
            ("oracle_ppo", "m5r_matched/oracle_ppo.yaml", "regime"),
            ("rl2_concat", "m5r_matched/rl2_concat.yaml", "meta"),
            ("rl2_hypernet", "m5r_matched/rl2_hypernet.yaml", "meta"),
            ("varibad_concat", "m5r_matched/varibad_concat.yaml", "meta"),
            ("varibad_hypernet", "m5r_matched/varibad_hypernet.yaml", "meta"),
        ],
    },
    "cartpole": {
        "budget": 5000,
        "base_obs": 4,
        "methods": [
            ("regime_agnostic", "m_cartpole_matched/regime_agnostic.yaml", "reference"),
            ("stacked_obs", "m_cartpole_matched/stacked_obs.yaml", "stacked"),
            ("belief_ppo", "m_cartpole_matched/belief_ppo.yaml", "regime"),
            ("oracle_ppo", "m_cartpole_matched/oracle_ppo.yaml", "regime"),
            ("rl2_concat", "m_cartpole_matched/rl2_concat.yaml", "meta"),
            ("rl2_hypernet", "m_cartpole_matched/rl2_hypernet.yaml", "meta"),
            ("varibad_concat", "m_cartpole_matched/varibad_concat.yaml", "meta"),
            ("varibad_hypernet", "m_cartpole_matched/varibad_hypernet.yaml", "meta"),
        ],
    },
}

# Conditioning pairs whose belief encoder must be identical.
PAIRS = [("rl2", "rl2_concat", "rl2_hypernet"),
         ("varibad", "varibad_concat", "varibad_hypernet")]

SHARED_OPTIMISER_KEYS = ["epochs", "minibatch_envs", "learning_rate", "clip_eps",
                         "max_grad_norm", "gamma", "lam", "vf_coef"]
SHARED_BUDGET_KEYS = ["iterations", "parallel_envs", "rollout_length",
                      "num_seeds", "seed_base"]
ENCODER_KEYS = ["hidden_dim", "latent_dim", "reward_decoder"]


def _count(tree) -> int:
    return int(sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(tree)))


def inspect(cfg_rel: str) -> dict[str, Any]:
    cfg = load_config(CONFIG_ROOT / cfg_rel)
    env = _maybe_wrap_env_for_agent(cfg, _build_env(cfg))
    agent = _build_agent(cfg, env)
    params = agent.init(jax.random.PRNGKey(0))["params"]
    named = {k: _count(v) for k, v in params.items()
             if k in ("encoder", "decoder", "policy")}
    return {
        "config": cfg_rel,
        "env": cfg.env.name,
        "obs_size": int(env.obs_size),
        "n_actions": int(env.n_actions),
        "n_regimes": int(cfg.env.params.get("n_regimes", 0)),
        "stack_k": int(cfg.env.params.get("stack_k", 0)),
        "param_count": _count(params),
        "named_subtrees": named,
        "optimiser": {k: cfg.agent.params.get(k) for k in SHARED_OPTIMISER_KEYS},
        "budget": {k: getattr(cfg, k) for k in SHARED_BUDGET_KEYS},
        "encoder_spec": {k: cfg.agent.params.get(k) for k in ENCODER_KEYS},
        "ent_coef": cfg.agent.params.get("ent_coef"),
    }


def _expected_obs(role: str, base_obs: int, r: dict[str, Any]) -> int:
    tuple_obs = base_obs + r["n_actions"] + 2
    if role == "reference" or role == "meta":
        return tuple_obs
    if role == "regime":
        return tuple_obs + r["n_regimes"]
    if role == "stacked":
        return (r["stack_k"] or 1) * tuple_obs
    raise ValueError(role)


def audit_family(name: str, spec: dict[str, Any], tol: float) -> dict[str, Any]:
    base_obs, budget = spec["base_obs"], spec["budget"]
    rows = {m: inspect(rel) for m, rel, _ in spec["methods"]}
    roles = {m: role for m, _, role in spec["methods"]}
    violations: list[str] = []

    # 1. inputs
    for m, r in rows.items():
        want = _expected_obs(roles[m], base_obs, r)
        if r["obs_size"] != want:
            violations.append(
                f"{name}/{m}: obs_size {r['obs_size']} != {want} expected for "
                f"role '{roles[m]}' on env '{r['env']}' (missing the u_t tuple?)"
            )

    # 2. optimiser, 3. budget
    for group, key in (("optimiser", "optimiser"), ("budget", "budget")):
        reference = rows[spec["methods"][0][0]][key]
        for m, r in rows.items():
            for k, v in reference.items():
                if r[key][k] != v:
                    violations.append(
                        f"{name}/{m}: {group}.{k} = {r[key][k]}, "
                        f"expected {v} (shared across the family)"
                    )

    # 4. capacity
    counts = {m: r["param_count"] for m, r in rows.items()}
    for m, n in counts.items():
        if abs(n - budget) / budget > tol:
            violations.append(
                f"{name}/{m}: {n} parameters is "
                f"{100 * abs(n - budget) / budget:.1f}% from the {budget} budget "
                f"(tolerance {100 * tol:.1f}%)"
            )
    for method, a, b in PAIRS:
        if a not in rows or b not in rows:
            continue
        if rows[a]["encoder_spec"] != rows[b]["encoder_spec"]:
            violations.append(
                f"{name}/{method}: encoder differs across the conditioning pair, "
                f"{rows[a]['encoder_spec']} vs {rows[b]['encoder_spec']}"
            )
        for key in ("encoder", "decoder"):
            ca, cb = rows[a]["named_subtrees"].get(key), rows[b]["named_subtrees"].get(key)
            if ca is not None and cb is not None and ca != cb:
                violations.append(
                    f"{name}/{method}: {key} parameter count differs across the "
                    f"conditioning pair, {ca} vs {cb}"
                )

    lo, hi = min(counts.values()), max(counts.values())
    return {
        "family": name,
        "rows": rows,
        "param_spread": {"min": lo, "max": hi,
                         "pct_of_budget": 100.0 * (hi - lo) / budget},
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.config_fairness_audit")
    ap.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    ap.add_argument("--tol", type=float, default=0.025,
                    help="allowed deviation from the parameter budget")
    args = ap.parse_args()

    run = ScriptRun(script="config_fairness_audit")
    out_dir = RESULTS_ROOT / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "config_fairness.json"
    summary_path = out_dir / "config_fairness_run.json"

    names = list(FAMILIES) if args.family == "all" else [args.family]
    reports, violations = [], []
    for n in names:
        rep = audit_family(n, FAMILIES[n], args.tol)
        reports.append(rep)
        violations += rep["violations"]

        print(f"\n{'='*94}\n{n}\n{'='*94}")
        print(f"{'method':<20}{'env':<40}{'obs':>5}{'params':>9}{'ent':>7}")
        for m, r in rep["rows"].items():
            print(f"{m:<20}{r['env']:<40}{r['obs_size']:>5}"
                  f"{r['param_count']:>9}{r['ent_coef']:>7}")
        s = rep["param_spread"]
        print(f"\nparameter spread: {s['min']} to {s['max']} "
              f"({s['pct_of_budget']:.1f}% of budget)")
        for v in rep["violations"]:
            print(f"  VIOLATION  {v}")

    with open(stats_path, "w") as f:
        json.dump({"reports": reports, "n_violations": len(violations)}, f, indent=2)
    run.add_output(str(stats_path))

    if violations:
        run.fail(reason=f"{len(violations)} fairness violations", summary_path=summary_path)
        return 1
    run.ok(
        key_stats={"families": len(reports), "violations": 0,
                   "max_spread_pct": max(r["param_spread"]["pct_of_budget"]
                                         for r in reports)},
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
