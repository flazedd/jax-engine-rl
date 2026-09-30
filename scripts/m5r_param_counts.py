"""M5R parameter-count audit.

Instantiates each method at its locked / reference config on the medium-difficulty
environment, initialises the agent's parameter tree, and reports the total
parameter count. Used to back the thesis claim that total parameter counts are
matched within 1% across (method, integration) cells of each meta-RL row.

Outputs:
  results/M5R/final/param_counts.json
  results/M5R/final/m5r_param_counts_run.json

Usage:
  uv run python -m scripts.m5r_param_counts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from utils.paths import analysis_dir
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from training.config import load_config
from training.train import _build_agent, _build_env, _maybe_wrap_env_for_agent
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
from evaluation.protocol import MEDIUM_ENV

CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = analysis_dir()

# (display_label, config_path_relative_to_CONFIG_ROOT). Every target is the
# config it was actually trained under, so the counts this reports are the
# counts the parameter-budget claim rests on. Composed from the current config
# set rather than written out: the paths used to name the superseded set, so
# the audit reported the budget of configs nothing had been trained from.
_SET = f"m5r_{MEDIUM_ENV}"
TARGETS: list[tuple[str, str]] = [
    ("rl2_concat",          f"{_SET}/rl2_concat.yaml"),
    ("rl2_hypernet",        f"{_SET}/rl2_hypernet.yaml"),
    ("varibad_concat",      f"{_SET}/varibad_concat.yaml"),
    ("varibad_hypernet",    f"{_SET}/varibad_hypernet.yaml"),
    ("regime_agnostic_ppo", f"{_SET}/regime_agnostic.yaml"),
    ("stacked_obs_ppo",     f"{_SET}/stacked_obs.yaml"),
    ("belief_ppo",          f"{_SET}/belief_ppo.yaml"),
    ("oracle_ppo",          f"{_SET}/oracle_ppo.yaml"),
]


def _count_params(params_tree) -> int:
    leaves = jax.tree_util.tree_leaves(params_tree)
    return int(sum(int(np.prod(leaf.shape)) for leaf in leaves))


def _audit_one(label: str, cfg_rel: str) -> dict[str, Any]:
    cfg_path = CONFIG_ROOT / cfg_rel
    cfg = load_config(cfg_path)
    env = _build_env(cfg)
    env = _maybe_wrap_env_for_agent(cfg, env)
    agent = _build_agent(cfg, env)
    state = agent.init(jax.random.PRNGKey(0))
    params = state["params"]
    n_params = _count_params(params)
    if label.startswith("rl2_"):
        modules = params["params"]
        encoder = _count_params(modules["Dense_0"]) + _count_params(modules["GRUCell_0"])
        components = {"encoder": encoder, "actor_critic": n_params - encoder, "decoder": 0}
    elif label.startswith("varibad_"):
        components = {"encoder": _count_params(params["encoder"]),
                      "actor_critic": _count_params(params["policy"]),
                      "decoder": _count_params(params["decoder"])}
    else:
        components = {"encoder": 0, "actor_critic": n_params, "decoder": 0}
    assert sum(components.values()) == n_params
    return {
        "label": label,
        "config_path": str(cfg_path.relative_to(REPO_ROOT)),
        "agent_name": cfg.agent.name,
        "integration": cfg.agent.params.get("integration", "n/a"),
        "obs_size": int(env.obs_size),
        "n_actions": int(env.n_actions),
        "hidden_dim": int(cfg.agent.params.get("hidden_dim", -1)),
        "latent_dim": int(cfg.agent.params.get("latent_dim", -1))
            if "latent_dim" in cfg.agent.params else None,
        "param_count": n_params,
        "component_counts": components,
    }


def _matched_pair_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify that {concat, hypernet} variants of each meta-RL method are
    matched to within 1%. The thesis claim depends on this."""
    by_label = {r["label"]: r for r in rows}
    out: dict[str, Any] = {}
    for method in ("rl2", "varibad"):
        c = by_label.get(f"{method}_concat", {}).get("param_count")
        h = by_label.get(f"{method}_hypernet", {}).get("param_count")
        if c is None or h is None:
            out[method] = {"concat": c, "hypernet": h, "rel_diff": None,
                           "within_1pct": False}
            continue
        rel = abs(h - c) / max(c, h)
        out[method] = {
            "concat": c,
            "hypernet": h,
            "abs_diff": abs(h - c),
            "rel_diff": rel,
            "within_1pct": rel <= 0.01,
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "m5r_param_counts_run.json"
    run = ScriptRun(script="m5r_param_counts", run_mode="full")

    print(f"[m5r_param_counts] auditing {len(TARGETS)} configs", flush=True)
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for label, cfg_rel in TARGETS:
        try:
            r = _audit_one(label, cfg_rel)
        except Exception as e:
            print(f"[m5r_param_counts] {label} FAILED: {e}", flush=True)
            failed.append(label)
            continue
        rows.append(r)
        print(f"[m5r_param_counts] {label:22s} "
              f"params={r['param_count']:>10,d} "
              f"hidden={r['hidden_dim']} obs={r['obs_size']}",
              flush=True)

    matched = _matched_pair_check(rows)
    payload = {
        "rows": rows,
        "matched_pair_check": matched,
        "n_failed": len(failed),
        "failed": failed,
    }
    out_path = OUT_DIR / "param_counts.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(out_path)

    print("\n[m5r_param_counts] matched-pair check:", flush=True)
    for method, d in matched.items():
        if d.get("rel_diff") is None:
            print(f"  {method}: missing data", flush=True)
            continue
        print(f"  {method}: concat={d['concat']:,} hypernet={d['hypernet']:,} "
              f"|Δ|={d['abs_diff']:,} ({100*d['rel_diff']:.2f}%) "
              f"within_1pct={d['within_1pct']}", flush=True)

    if failed:
        run.fail(reason=f"{len(failed)} configs failed: {failed}",
                 summary_path=summary_path)
        return 1
    run.ok(
        key_stats={
            "n_audited": len(rows),
            "rl2_within_1pct": matched.get("rl2", {}).get("within_1pct"),
            "varibad_within_1pct": matched.get("varibad", {}).get("within_1pct"),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
