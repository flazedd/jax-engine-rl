"""Cartpole — train the matched family on the reported instance.

`scripts.cartpole_difficulty_sweep` trains the family across the levels of one
difficulty axis, overriding the env params per level. That is the RQ3 sweep, and
it cannot train the family on a single named instance: every level it knows
resolves to an axis YAML, so the instance the configs themselves point at is the
one thing it will not run. This script runs exactly that, the configs as
written, which is what the reported numbers must come from.

Arms are named by the config's own `experiment_name`, so a run lands where the
gate and the analyses already look for it.

Usage:
  uv run python -m scripts.cartpole_matched_train
  uv run python -m scripts.cartpole_matched_train --arms rl2_concat rl2_hypernet
  uv run python -m scripts.cartpole_matched_train --seeds 5 --mid
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import bootstrap_paired_mean_ci
from evaluation import protocol as P
from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.paths import cartpole_dir, experiment_dir
from utils.script_output import ScriptRun, print_table

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

ARMS: dict[str, str] = {
    "regime_agnostic": "m_cartpole_matched/regime_agnostic.yaml",
    "stacked_obs": "m_cartpole_matched/stacked_obs.yaml",
    "belief_ppo": "m_cartpole_matched/belief_ppo.yaml",
    "oracle_ppo": "m_cartpole_matched/oracle_ppo.yaml",
    "rl2_concat": "m_cartpole_matched/rl2_concat.yaml",
    "rl2_hypernet": "m_cartpole_matched/rl2_hypernet.yaml",
    "varibad_concat": "m_cartpole_matched/varibad_concat.yaml",
    "varibad_hypernet": "m_cartpole_matched/varibad_hypernet.yaml",
}

# The reference triple plus the memory baseline: the arms that have to land
# before any variant is interpretable.
DEFAULT_ARMS = ["regime_agnostic", "stacked_obs", "belief_ppo", "oracle_ppo"]


def _read_metrics(experiment_name: str) -> dict[str, Any] | None:
    p = experiment_dir(experiment_name) / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _train(
    arm: str, mode: str, iterations: int | None, seeds: int | None,
    skip_existing: bool,
) -> dict[str, Any] | None:
    cfg = load_config(CONFIG_ROOT / ARMS[arm])
    apply_run_mode(cfg, mode)
    if iterations is not None:
        cfg.iterations = iterations
    if seeds is not None:
        cfg.num_seeds = seeds

    if skip_existing:
        existing = _read_metrics(cfg.experiment_name)
        if (
            existing
            and existing.get("iterations", 0) >= cfg.iterations
            and existing.get("num_seeds", 0) >= cfg.num_seeds
        ):
            print(f"[cp_train] skip {cfg.experiment_name}: already at budget",
                  flush=True)
            return {
                "experiment_name": cfg.experiment_name,
                "final_return_mean": float(existing["final_return_mean"]),
                "per_seed_final_return": list(
                    map(float, existing["per_seed_final_return"])),
                "iterations": int(existing["iterations"]),
                "num_seeds": int(existing["num_seeds"]),
                "skipped_existing": True,
            }

    print(
        f"[cp_train] launching {cfg.experiment_name} | iters={cfg.iterations} "
        f"envs={cfg.parallel_envs} seeds={cfg.num_seeds}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        train_or_sweep(cfg)
    except SystemExit as e:
        if e.code != 0:
            print(f"[cp_train] {cfg.experiment_name} FAILED (exit {e.code})",
                  flush=True)
            return None
    m = _read_metrics(cfg.experiment_name)
    if m is None:
        return None
    print(
        f"[cp_train] {cfg.experiment_name} done in "
        f"{(time.perf_counter() - t0) / 60:.1f} min | "
        f"mean={float(m['final_return_mean']):.2f}",
        flush=True,
    )
    return {
        "experiment_name": cfg.experiment_name,
        "final_return_mean": float(m["final_return_mean"]),
        "per_seed_final_return": list(map(float, m["per_seed_final_return"])),
        "iterations": int(m["iterations"]),
        "num_seeds": int(m["num_seeds"]),
        "skipped_existing": False,
    }


def _instance_of(cfg_rel: str) -> str:
    """The env YAML the arm inherits, for the log header."""
    with open(CONFIG_ROOT / cfg_rel) as f:
        for line in f:
            if "envs/e_cartpole" in line:
                return line.strip().lstrip("- ")
    return "unknown"


# Adjacent steps of the ladder, printed once every arm they need is present.
LADDER = [
    ("stacked over agnostic", "stacked_obs", "regime_agnostic"),
    ("belief over stacked", "belief_ppo", "stacked_obs"),
    ("oracle over belief", "oracle_ppo", "belief_ppo"),
]


def _print_arms(results: dict[str, Any], *, title: str, steps: bool = False) -> None:
    rows = []
    for arm, cell in results.items():
        per_seed = np.asarray(cell["per_seed_final_return"], dtype=float)
        sd = per_seed.std(ddof=1) if per_seed.size > 1 else 0.0
        rows.append([
            arm, f"{cell['final_return_mean']:.2f}", f"{sd:.2f}",
            f"{per_seed.min():.2f}", f"{per_seed.max():.2f}",
            cell["num_seeds"], cell["iterations"],
            "reused" if cell["skipped_existing"] else "trained",
        ])
    print_table(
        ["arm", "mean", "seed SD", "min", "max", "seeds", "iters", "source"],
        rows, title=title,
    )
    if not steps:
        return
    step_rows = []
    for label, hi, lo in LADDER:
        if hi not in results or lo not in results:
            continue
        a = np.asarray(results[hi]["per_seed_final_return"], dtype=float)
        b = np.asarray(results[lo]["per_seed_final_return"], dtype=float)
        if a.size != b.size or a.size < 2:
            continue
        mean, ci_lo, ci_hi = bootstrap_paired_mean_ci(
            a, b, n_boot=P.BOOTSTRAP_RESAMPLES, alpha=1 - P.INTERVAL_LEVEL)
        step_rows.append([
            label, f"{mean:+.2f}", f"[{ci_lo:+.2f}, {ci_hi:+.2f}]",
            "yes" if ci_lo > 0 else "NO",
        ])
    if step_rows:
        print_table(
            ["step", "paired mean", "95% CI", "separated"], step_rows,
            title="Steps of the ladder",
            note="separated = lower bootstrap bound above zero, the criterion "
                 "the reference-ordering gate applies.",
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.cartpole_matched_train")
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mid", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS),
                        default=DEFAULT_ARMS)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    if sum([args.super_fast, args.fast, args.mid]) > 1:
        raise SystemExit("--super-fast / --fast / --mid are mutually exclusive")
    mode = (
        "super_fast" if args.super_fast
        else "fast" if args.fast
        else "mid" if args.mid
        else "full"
    )

    run = ScriptRun(script="cartpole_matched_train", run_mode=mode)
    out_dir = cartpole_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_cartpole_matched_train.json"
    summary_path = out_dir / "stats_cartpole_matched_train_run.json"

    # The instance every arm inherits, named once from the first config so the
    # log says what was trained rather than leaving it to be inferred.
    first = load_config(CONFIG_ROOT / ARMS[args.arms[0]])
    apply_run_mode(first, mode)
    instance_name = _instance_of(ARMS[args.arms[0]])
    print_table(
        ["setting", "value"],
        [["instance", instance_name],
         ["arms", ", ".join(args.arms)],
         ["iterations", args.iterations or first.iterations],
         ["seeds", args.seeds or first.num_seeds],
         ["parallel envs", first.parallel_envs],
         ["run mode", mode]],
        title="Cartpole matched family, plan",
    )

    t_start = time.perf_counter()
    results: dict[str, Any] = {}
    failed: list[str] = []
    for i, arm in enumerate(args.arms):
        print(f"\n=== arm {i + 1} of {len(args.arms)}: {arm} ===", flush=True)
        cell = _train(arm, mode, args.iterations, args.seeds,
                      not args.no_skip_existing)
        if cell is None:
            failed.append(arm)
            continue
        results[arm] = cell
        # A running table after every arm, so the log reads as a result at any
        # point rather than only once the last arm lands.
        elapsed = (time.perf_counter() - t_start) / 60
        remaining = (elapsed / (i + 1)) * (len(args.arms) - i - 1)
        _print_arms(results, title=f"After {i + 1} of {len(args.arms)} arms "
                                   f"({elapsed:.0f} min elapsed, "
                                   f"{remaining:.0f} min remaining)")

    if failed:
        run.fail(reason="arms failed: " + ", ".join(failed),
                 summary_path=summary_path)
        return 1

    stats = {
        "instance": instance_name,
        "run_mode": mode,
        "arms": args.arms,
        "results": results,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    _print_arms(results, title=f"Cartpole matched family on {instance_name}",
                steps=True)

    run.ok(
        key_stats={
            "n_arms": len(results),
            "elapsed_min": round((time.perf_counter() - t_start) / 60, 2),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
