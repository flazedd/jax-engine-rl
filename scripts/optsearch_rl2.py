"""Optimisation-ease search: are hypernets easier to optimise than concat?

Symmetric design: the SAME set of random PPO-optimiser hyperparameter configs is
applied to RL2 concat and RL2 hypernet, each at its locked matched-compute
architecture, on the medium environment. We then compare the resulting return
distributions. The architecture (and hence parameter count) is held fixed in each
locked config; only the optimiser knobs are searched, so the comparison isolates
how forgiving each integration mechanism's optimisation landscape is.

Search space (5 PPO knobs): learning_rate, clip_eps, ent_coef, lam (GAE-lambda), epochs.
Per (arch, config): seeds x iters. Default 16 configs x 2 archs x 3 seeds = 96 runs.

Resumable (budget-aware skip). Writes results/M5R/final/optsearch_rl2.json,
updated after every config so partial progress is never lost.

Usage:
  uv run python -m scripts.optsearch_rl2 --smoke   # 2 cfgs / 2 seeds / 6 iters
  uv run python -m scripts.optsearch_rl2           # full 16 / 3 / 200
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "experiments" / "configs"
RESULTS = REPO / "results"
OUT = RESULTS / "M5R" / "final" / "optsearch_rl2.json"

# Medium-environment reference levels (established at 20s/300it).
FLOOR, BELIEF = 138.0, 168.8

ARCHS = [("concat", "m5r_locked/rl2_concat.yaml"),
         ("hypernet", "m5r_locked/rl2_hypernet.yaml")]
ENV_YAML = "envs/e_final.yaml"


def _env_params():
    with open(CONFIG / ENV_YAML) as f:
        return yaml.safe_load(f)["env"]["params"]


def _sample_configs(n, rng):
    """The SAME n configs are used for both architectures (symmetric search)."""
    out = []
    for _ in range(n):
        out.append({
            "learning_rate": float(10 ** rng.uniform(-4.0, -2.0)),   # 1e-4 .. 1e-2
            "clip_eps": float(rng.uniform(0.10, 0.30)),
            "ent_coef": float(10 ** rng.uniform(-4.0, -1.5)),        # ~1e-4 .. 3e-2
            "lam": float(rng.uniform(0.90, 0.99)),
            "epochs": int(rng.choice([2, 4, 8])),
        })
    return out


def _build_cfg(base_yaml, hp, name, iters, seeds, optimiser="ppo"):
    cfg = load_config(CONFIG / base_yaml)
    apply_run_mode(cfg, "full")
    cfg.env.params = copy.deepcopy(_env_params())
    for k, v in hp.items():
        cfg.agent.params[k] = v
    if optimiser == "a2c":
        # Vanilla advantage actor-critic: no clipping, single on-policy epoch.
        cfg.agent.params["policy_objective"] = "a2c"
        cfg.agent.params["epochs"] = 1
    cfg.experiment_name = name
    cfg.iterations = iters
    cfg.num_seeds = seeds
    return cfg


def _metrics(name):
    p = RESULTS / name / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def _train_or_skip(cfg):
    m = _metrics(cfg.experiment_name)
    if m and int(m.get("iterations", 0)) >= cfg.iterations and int(m.get("num_seeds", 0)) >= cfg.num_seeds:
        print(f"[optsearch]   {cfg.experiment_name}: reuse", flush=True)
        return list(map(float, m["per_seed_final_return"]))
    print(f"[optsearch]   {cfg.experiment_name}: train {cfg.num_seeds}s x {cfg.iterations}it", flush=True)
    train_or_sweep(cfg)
    return list(map(float, _metrics(cfg.experiment_name)["per_seed_final_return"]))


def _dump(results):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.optsearch_rl2")
    ap.add_argument("--method", default="rl2", choices=["rl2", "varibad"])
    ap.add_argument("--optimiser", default="ppo", choices=["ppo", "a2c"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-configs", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()
    n_cfg = 2 if args.smoke else args.n_configs
    seeds = 2 if args.smoke else args.seeds
    iters = 6 if args.smoke else args.iterations
    tag = "_smoke" if args.smoke else ""

    osuffix = "" if args.optimiser == "ppo" else f"_{args.optimiser}"
    global OUT
    OUT = RESULTS / "M5R" / "final" / f"optsearch_{args.method}{osuffix}.json"
    archs = [("concat", f"m5r_locked/{args.method}_concat.yaml"),
             ("hypernet", f"m5r_locked/{args.method}_hypernet.yaml")]

    rng = np.random.default_rng(0)
    hp_configs = _sample_configs(n_cfg, rng)

    results = {
        "floor": FLOOR, "belief": BELIEF, "optimiser": args.optimiser,
        "search_space": "learning_rate, clip_eps, ent_coef, lam, epochs",
        "n_configs": n_cfg, "seeds": seeds, "iterations": iters,
        "configs": hp_configs, "by_arch": {},
    }
    for arch, base in archs:
        per_cfg = []
        for i, hp in enumerate(hp_configs):
            name = f"optsearch_{args.method}{osuffix}_{arch}_cfg{i:02d}{tag}"
            try:
                sr = _train_or_skip(_build_cfg(base, hp, name, iters, seeds, args.optimiser))
            except Exception as e:
                print(f"[optsearch] !!! {name} FAILED: {e}; continuing", flush=True)
                sr = [float("nan")]
            mean = float(np.nanmean(sr))
            per_cfg.append({"cfg": hp, "mean_return": mean, "per_seed": sr})
            print(f"[optsearch] {arch} cfg{i:02d}: mean={mean:.1f}", flush=True)
            results["by_arch"][arch] = {"per_config": per_cfg}
            _dump(results)
        means = [c["mean_return"] for c in per_cfg if not np.isnan(c["mean_return"])]
        results["by_arch"][arch] = {
            "per_config": per_cfg,
            "mean_over_search": float(np.mean(means)) if means else float("nan"),
            "best": float(np.max(means)) if means else float("nan"),
            "std_over_search": float(np.std(means)) if means else float("nan"),
            "frac_above_floor": float(np.mean([m > FLOOR for m in means])) if means else 0.0,
            "frac_above_belief": float(np.mean([m >= BELIEF for m in means])) if means else 0.0,
        }
        _dump(results)

    print("\n[optsearch] === SUMMARY (mean / best / %>floor / %>ceiling over the search) ===", flush=True)
    for arch in ("concat", "hypernet"):
        a = results["by_arch"].get(arch, {})
        print(f"  {arch:>9}: mean={a.get('mean_over_search', float('nan')):.1f}  "
              f"best={a.get('best', float('nan')):.1f}  "
              f">floor={a.get('frac_above_floor', 0) * 100:.0f}%  "
              f">ceiling={a.get('frac_above_belief', 0) * 100:.0f}%", flush=True)
    print(f"[optsearch] output={OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
