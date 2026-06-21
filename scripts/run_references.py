"""Re-train the three reference levels (floor / belief / oracle) at a chosen
budget and write results/milestones/M3/stats_M3_reference_levels.json, the file
that the gap decomposition and m5r_final_eval read for the medium environment.

Budget-aware resumable: a reference already trained at the target budget is reused.

Usage:
  uv run python -m scripts.run_references --iterations 300 --num-seeds 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.config import load_config, apply_run_mode
from training.train import train_or_sweep

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CONFIG = REPO / "experiments" / "configs"
OUT = RESULTS / "milestones" / "M3" / "stats_M3_reference_levels.json"

REFS = {  # key in stats file -> (config name, role label)
    "regime_agnostic_ppo": ("m3_regime_agnostic", "Floor"),
    "belief_ppo": ("m3_belief", "Inferred-belief ceiling"),
    "oracle_ppo": ("m3_oracle", "Ground-truth ceiling"),
}


def _ci(x, nb=10_000):
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        m = float(x.mean()) if x.size else float("nan")
        return [m, m]
    r = np.random.default_rng(0)
    idx = r.integers(0, x.size, (nb, x.size))
    mm = x[idx].mean(1)
    return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]


def _train_ref(cfg_name: str, iters: int, seeds: int) -> list[float]:
    cfg = load_config(CONFIG / f"{cfg_name}.yaml")
    apply_run_mode(cfg, "full")
    cfg.iterations = iters
    cfg.num_seeds = seeds
    mp = RESULTS / cfg.experiment_name / "metrics.json"
    if mp.exists():
        m = json.load(open(mp))
        if int(m.get("num_seeds", 0)) == seeds and int(m.get("iterations", 0)) == iters:
            print(f"[refs] {cfg_name}: reuse existing ({seeds} seeds, {iters} iter)", flush=True)
            return list(map(float, m["per_seed_final_return"]))
    print(f"[refs] {cfg_name}: training {seeds} seeds x {iters} iter", flush=True)
    train_or_sweep(cfg)
    return list(map(float, json.load(open(mp))["per_seed_final_return"]))


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.run_references")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--num-seeds", type=int, default=20)
    args = ap.parse_args()

    env_version = "e6e_symmetric_kappa05"
    if OUT.exists():
        env_version = json.load(open(OUT)).get("env_version", env_version)

    levels = {}
    for key, (cfg_name, role) in REFS.items():
        sr = _train_ref(cfg_name, args.iterations, args.num_seeds)
        levels[key] = {
            "role": role, "mean": float(np.mean(sr)), "ci": _ci(sr),
            "seed_returns": sr, "experiment_dir": cfg_name,
        }

    ag = np.asarray(levels["regime_agnostic_ppo"]["seed_returns"])
    be = np.asarray(levels["belief_ppo"]["seed_returns"])
    orc = np.asarray(levels["oracle_ppo"]["seed_returns"])
    n = min(ag.size, be.size, orc.size)
    total, comp, inf = orc[:n] - ag[:n], be[:n] - ag[:n], orc[:n] - be[:n]
    gap = {
        "total": {"mean": float(total.mean()), "ci": _ci(total)},
        "compromise": {"mean": float(comp.mean()), "ci": _ci(comp)},
        "inference": {"mean": float(inf.mean()), "ci": _ci(inf)},
    }

    out = {
        "env_version": env_version, "reference_levels": levels, "gap_components": gap,
        "iterations": args.iterations, "num_seeds": args.num_seeds,
        "ordering_valid": bool(levels["regime_agnostic_ppo"]["mean"]
                               < levels["belief_ppo"]["mean"]
                               <= levels["oracle_ppo"]["mean"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("[refs] wrote", OUT.name,
          {k: round(v["mean"], 1) for k, v in levels.items()},
          "| gap", {k: round(v["mean"], 1) for k, v in gap.items()}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
