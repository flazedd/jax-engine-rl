"""Redesigned difficulty sweep at 20 seeds / 300 iters.

Replaces the old distinguishability axis with R1-R4-validated levels and adds a
coupled fast-regime cell that the verification showed is the only way to push
switching fast while keeping the regime inferable:

  distinguishability_easy : s = 2.0  (regimes far apart, ~one fill identifies)
  distinguishability_hard : s = 0.7  (compressed but gap intact)
  coupled_fast            : P_ii = 0.85 (dwell 7) with s = 2.0 distinguishability

The medium env (e_final, s=1.0, P_ii=0.98) is the shared centre and is already
done at 20/300, so it is not retrained here.

Per env: 3 references (floor/belief/oracle) + 4 meta-RL cells, trained in-process
(overnight_v2 style) with a 20/300 budget, then merged into per_cell_env.json in
the structure the probe and sweep plots expect, then probed (logistic + MLP).

Resumable (budget-aware skip), backed up before any write, continues past
failures. Cells use the standard m5r_final_{cell}_{label} experiment names so the
posterior probe finds their checkpoints.

Usage:
  uv run python -m scripts.sweep_redesign_n20 --smoke   # 2 seeds / 6 iters
  uv run python -m scripts.sweep_redesign_n20           # full 20 / 300
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.paths import analysis_dir

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "experiments" / "configs"
RESULTS = REPO / "results"
PCE = analysis_dir() / "per_cell_env.json"
BACKUP = RESULTS / "_backup_pre_sweep_redesign"

# Every method runs from the matched-fairness configs, so a difficulty level
# compares methods on identical inputs, optimiser settings, budget and
# capacity. `scripts.config_fairness_audit` enforces that; changing a config
# here without re-running the audit breaks the comparison.
REFS = [("regime_agnostic", "rsmm_matched/regime_agnostic.yaml"),
        ("belief", "rsmm_matched/belief_ppo.yaml"),
        ("oracle", "rsmm_matched/oracle_ppo.yaml"),
        ("stacked_obs", "rsmm_matched/stacked_obs.yaml")]
CELLS = [("rl2_concat", "rsmm_matched/rl2_concat.yaml"),
         ("rl2_hypernet", "rsmm_matched/rl2_hypernet.yaml"),
         ("varibad_concat", "rsmm_matched/varibad_concat.yaml"),
         ("varibad_hypernet", "rsmm_matched/varibad_hypernet.yaml")]

# (env_label, env_yaml_rel). Medium (e_final) is the shared centre, already done.
ENVS = [
    ("distinguishability_easy", "envs/sweep_dist_easy.yaml"),
    ("distinguishability_hard", "envs/sweep_dist_hard.yaml"),
    ("coupled_fast", "envs/sweep_coupled_fast.yaml"),
]


def _env_params(env_yaml_rel: str) -> dict[str, Any]:
    with open(CONFIG / env_yaml_rel) as f:
        return yaml.safe_load(f)["env"]["params"]


def _build_cfg(base_yaml: str, env_yaml_rel: str, name: str, iters: int, seeds: int):
    cfg = load_config(CONFIG / base_yaml)
    apply_run_mode(cfg, "full")
    # Merge rather than replace: the level YAML carries the regime parameters,
    # while wrapper settings that live in the method config (stacked-obs `K`)
    # must survive the override.
    cfg.env.params = {**cfg.env.params, **copy.deepcopy(_env_params(env_yaml_rel))}
    cfg.experiment_name = name
    cfg.iterations = iters
    cfg.num_seeds = seeds
    return cfg


def _ci(values, n_boot=10_000):
    a = np.asarray(values, dtype=float)
    if a.size <= 1:
        m = float(a.mean()) if a.size else float("nan")
        return [m, m]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, a.size, (n_boot, a.size))
    b = a[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _metrics(name: str):
    p = RESULTS / name / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def _train_or_skip(cfg) -> list[float]:
    m = _metrics(cfg.experiment_name)
    if m and int(m.get("iterations", 0)) >= cfg.iterations and int(m.get("num_seeds", 0)) >= cfg.num_seeds:
        print(f"[sweep]   {cfg.experiment_name}: reuse", flush=True)
        return list(map(float, m["per_seed_final_return"]))
    print(f"[sweep]   {cfg.experiment_name}: train {cfg.num_seeds}s x {cfg.iterations}it", flush=True)
    train_or_sweep(cfg)
    return list(map(float, _metrics(cfg.experiment_name)["per_seed_final_return"]))


def _train_env(label: str, env_yaml: str, iters: int, seeds: int) -> dict[str, Any]:
    print(f"\n[sweep] === env {label} ({env_yaml}) ===", flush=True)
    ref_means: dict[str, float] = {}
    for method, base in REFS:
        sr = _train_or_skip(_build_cfg(base, env_yaml, f"m5r_ref_{method}_{label}", iters, seeds))
        ref_means[method] = float(np.mean(sr))
    floor, belief, oracle = ref_means["regime_agnostic"], ref_means["belief"], ref_means["oracle"]

    cells: dict[str, Any] = {}
    for cell, base in CELLS:
        sr = _train_or_skip(_build_cfg(base, env_yaml, f"m5r_final_{cell}_{label}", iters, seeds))
        mean = float(np.mean(sr))
        cells[cell] = {
            "final_return_mean": mean,
            "final_return_ci95": _ci(sr),
            "delta_vs_floor": mean - floor,
            "gap_closed_vs_oracle": (mean - floor) / (oracle - floor) if oracle != floor else None,
            "gap_closed_vs_belief": (mean - floor) / (belief - floor) if belief != floor else None,
            "per_seed_final_return": sr,
        }
    return {
        "refs": {"regime_agnostic_ppo": floor, "belief_ppo": belief,
                 "oracle_ppo": oracle,
                 "stacked_obs_ppo": ref_means.get("stacked_obs")},
        "cells": cells,
    }


def _merge_pce(label: str, entry: dict[str, Any]) -> None:
    pce = json.load(open(PCE))
    pce.setdefault("per_env", {})[label] = entry
    json.dump(pce, open(PCE, "w"), indent=2)
    print(f"[sweep]   merged {label} into per_cell_env.json", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.sweep_redesign_n20")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=600)
    args = ap.parse_args()
    seeds = 2 if args.smoke else args.seeds
    iters = 6 if args.smoke else args.iterations
    rollouts = 50 if args.smoke else 500

    if not BACKUP.exists() and PCE.exists():
        BACKUP.mkdir(parents=True)
        shutil.copy2(PCE, BACKUP / "per_cell_env.json")
        print(f"[sweep] backed up per_cell_env.json to {BACKUP}", flush=True)

    # smoke runs under suffixed labels so it never overwrites real experiment
    # dirs or per_cell_env entries; delete the *_smoke artefacts afterwards.
    envs = ENVS if not args.smoke else [(f"{l}_smoke", y) for l, y in ENVS]

    t0 = time.perf_counter()
    results = []
    for label, env_yaml in envs:
        try:
            entry = _train_env(label, env_yaml, iters, seeds)
            _merge_pce(label, entry)
            results.append((label, "ok"))
        except Exception as e:
            print(f"[sweep] !!! {label} FAILED: {e}; continuing", flush=True)
            results.append((label, f"FAILED: {e}"))

    # probe each new env (logistic + MLP), tagged PER ENV so sequential
    # --env-filter runs don't overwrite each other's single-env output file.
    for label, _ in envs:
        for clf in (["--classifier", "mlp"], []):
            cmd = ["uv", "run", "python", "-m", "scripts.posterior_probe",
                   "--env-filter", label, "--n-rollouts", str(rollouts),
                   "--tag", f"sweep_{label}", *clf]
            print(f"[sweep] probe: {' '.join(cmd)}", flush=True)
            try:
                subprocess.run(cmd, cwd=str(REPO), check=False)
            except Exception as e:
                print(f"[sweep] probe {label} {clf} failed: {e}", flush=True)

    dt = (time.perf_counter() - t0) / 60
    print(f"\n[sweep] === DONE in {dt:.1f} min ===", flush=True)
    for label, status in results:
        print(f"  {label:>26}: {status}", flush=True)
    return 1 if any(s.startswith("FAILED") for _, s in results) else 0


if __name__ == "__main__":
    sys.exit(main())
