"""Cartpole — reference-level screen over candidate env instances.

Instance choice for the appendix replication, mirroring how the market-making
instance was chosen: an instance earns the appendix only if its three reference
levels separate, since the gap-closed fraction divides by
`oracle - regime_agnostic` and reads as noise when that denominator is small.

Trains the three reference arms and the window baseline of the matched cartpole
family on each candidate env YAML and reports every adjacent component:

    stacked_obs - regime_agnostic   what a fixed K-step window recovers
    belief - stacked_obs            what full-history inference adds over it
    oracle - belief                 the posterior-information component

The middle one decides whether the instance is worth running the meta-RL arms
on. If a four-tuple window already reaches Belief-PPO, per-step evidence is
strong enough that little is left for a learned belief to recover, whatever the
total gap looks like.

`scripts.cartpole_difficulty_sweep` answers a different question, how the
methods move along one difficulty axis, and its levels each hold the other axis
at medium. This screen compares whole instances, including ones that move both
axes at once, and ranks them by reference separation.

Usage:
  uv run python -m scripts.cartpole_instance_screen --mid
  uv run python -m scripts.cartpole_instance_screen --instances wide extreme \
      --iterations 600 --seeds 5
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.paths import cartpole_dir, experiment_dir
from utils.script_output import ScriptRun, print_table

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

# The three reference arms of the matched family. The four method arms are not
# screened: an instance that fails to separate its references is rejected
# before any method is trained on it.
REFERENCES: list[tuple[str, str]] = [
    ("regime_agnostic", "m_cartpole_matched/regime_agnostic.yaml"),
    ("stacked_obs", "m_cartpole_matched/stacked_obs.yaml"),
    ("belief", "m_cartpole_matched/belief_ppo.yaml"),
    ("oracle", "m_cartpole_matched/oracle_ppo.yaml"),
]

# Candidate instances, each an env YAML under experiments/configs/envs.
INSTANCES: dict[str, str] = {
    "medium": "e_cartpole_v1.yaml",
    "asym_easy": "e_cartpole_v1_easy.yaml",
    "pers_easy": "e_cartpole_v1_persistence_easy.yaml",
    "wide": "e_cartpole_v1_wide.yaml",
    "extreme": "e_cartpole_v1_extreme.yaml",
    "slow_noise04": "e_cartpole_v1_slow_noise04.yaml",
    "slow_noise06": "e_cartpole_v1_slow_noise06.yaml",
    "slow_mild": "e_cartpole_v1_slow_mild.yaml",
}


def _load_env_params(instance: str) -> dict[str, Any]:
    env_yaml = CONFIG_ROOT / "envs" / INSTANCES[instance]
    if not env_yaml.exists():
        raise FileNotFoundError(f"missing cartpole env config: {env_yaml}")
    with open(env_yaml) as f:
        return yaml.safe_load(f)["env"]["params"]


def _experiment_name(instance: str, method: str) -> str:
    return f"m_cartpole_screen_{instance}_{method}"


def _bootstrap_diff_ci(
    a: list[float], b: list[float], n_boot: int = 10_000,
) -> tuple[float, float]:
    """Bootstrap CI of mean(a) - mean(b), paired over seeds when the two arms
    ran the same seed count, since every arm shares the seed list."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size or x.size <= 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = (x[idx] - y[idx]).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _train_reference(
    instance: str, method: str, base_yaml: str, mode: str,
    iterations: int | None, seeds: int | None,
) -> dict[str, Any] | None:
    cfg = load_config(CONFIG_ROOT / base_yaml)
    apply_run_mode(cfg, mode)
    # Merge so the wrapper settings the method config carries survive the
    # per-instance env override.
    cfg.env.params = {**cfg.env.params, **copy.deepcopy(_load_env_params(instance))}
    cfg.experiment_name = _experiment_name(instance, method)
    if iterations is not None:
        cfg.iterations = iterations
    if seeds is not None:
        cfg.num_seeds = seeds

    print(
        f"[cp_screen] launching {cfg.experiment_name} | iters={cfg.iterations} "
        f"envs={cfg.parallel_envs} seeds={cfg.num_seeds}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        train_or_sweep(cfg)
    except SystemExit as e:
        if e.code != 0:
            print(f"[cp_screen] {cfg.experiment_name} FAILED (exit {e.code})", flush=True)
            return None
    metrics_path = experiment_dir(cfg.experiment_name) / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    print(
        f"[cp_screen] {cfg.experiment_name} done in "
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
    }


def _report(
    ranked: list[dict[str, Any]], mode: str,
    iterations: int | None, seeds: int | None,
) -> None:
    """Print the screen for a person: the four levels of the ladder, then the
    three steps between them, best instance first."""
    budget = f"run mode {mode}"
    if iterations is not None:
        budget += f", {iterations} iterations"
    if seeds is not None:
        budget += f", {seeds} seeds"

    print_table(
        ["instance", "agnostic", "stacked", "belief", "oracle"],
        [
            [e["instance"], f"{e['regime_agnostic']:.2f}", f"{e['stacked_obs']:.2f}",
             f"{e['belief']:.2f}", f"{e['oracle']:.2f}"]
            for e in ranked
        ],
        title=f"Reference levels ({budget})",
    )

    def _with_ci(entry: dict[str, Any], key: str) -> str:
        lo, hi = entry[f"{key}_ci"]
        if lo != lo or hi != hi:  # NaN: one seed, no interval to report
            return f"{entry[key]:+.2f}"
        return f"{entry[key]:+.2f} [{lo:+.2f},{hi:+.2f}]"

    print_table(
        ["instance", "window", "headroom", "posterior", "weakest", "ordered"],
        [
            [
                e["instance"],
                _with_ci(e, "window"),
                _with_ci(e, "inference_headroom"),
                _with_ci(e, "posterior_information"),
                f"{e['min_component']:+.2f}",
                "yes" if e["ordering_holds"] else "NO",
            ]
            for e in ranked
        ],
        title="Steps of the ladder, with 95% bootstrap intervals",
        note="window = stacked - agnostic, headroom = belief - stacked, "
             "posterior = oracle - belief. Ranked by the weakest step.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.cartpole_instance_screen")
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mid", action="store_true")
    parser.add_argument("--instances", nargs="+", choices=sorted(INSTANCES),
                        default=sorted(INSTANCES))
    parser.add_argument("--iterations", type=int, default=None,
                        help="override the run mode's iteration count")
    parser.add_argument("--seeds", type=int, default=None,
                        help="override the run mode's seed count")
    parser.add_argument("--report-only", action="store_true",
                        help="re-print the last screen from its stats file "
                             "without training anything")
    args = parser.parse_args()
    if sum([args.super_fast, args.fast, args.mid]) > 1:
        raise SystemExit("--super-fast / --fast / --mid are mutually exclusive")
    mode = (
        "super_fast" if args.super_fast
        else "fast" if args.fast
        else "mid" if args.mid
        else "full"
    )

    run = ScriptRun(script="cartpole_instance_screen", run_mode=mode)
    out_dir = cartpole_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_cartpole_instance_screen.json"
    summary_path = out_dir / "stats_cartpole_instance_screen_run.json"

    if args.report_only:
        if not stats_path.exists():
            print(f"[cp_screen] no screen to report: {stats_path} missing",
                  flush=True)
            return 1
        with open(stats_path) as f:
            prior = json.load(f)
        _report(prior["ranked"], prior["run_mode"],
                prior.get("iterations_override"), prior.get("seeds_override"))
        return 0

    results: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for instance in args.instances:
        cells: dict[str, Any] = {}
        for method, base_yaml in REFERENCES:
            cell = _train_reference(
                instance, method, base_yaml, mode, args.iterations, args.seeds,
            )
            if cell is None:
                failed.append(f"{instance}/{method}")
                continue
            cells[method] = cell
        results[instance] = cells

    if failed:
        run.fail(
            reason=f"{len(failed)} cells failed: " + ", ".join(failed[:5]),
            summary_path=summary_path,
        )
        return 1

    # Adjacent steps of the ladder, in the order the thesis asserts them.
    STEPS = [
        ("window", "stacked_obs", "regime_agnostic"),
        ("inference_headroom", "belief", "stacked_obs"),
        ("posterior_information", "oracle", "belief"),
    ]

    ranked: list[dict[str, Any]] = []
    for instance, cells in results.items():
        entry: dict[str, Any] = {
            "instance": instance,
            "env_yaml": INSTANCES[instance],
        }
        for arm in cells:
            entry[arm] = cells[arm]["final_return_mean"]
        components = {}
        for label, hi_arm, lo_arm in STEPS:
            d = cells[hi_arm]["final_return_mean"] - cells[lo_arm]["final_return_mean"]
            components[label] = d
            entry[label] = d
            lo, hi = _bootstrap_diff_ci(
                cells[hi_arm]["per_seed_final_return"],
                cells[lo_arm]["per_seed_final_return"],
            )
            entry[f"{label}_ci"] = [lo, hi]
        entry["reference_gap"] = (
            cells["oracle"]["final_return_mean"]
            - cells["regime_agnostic"]["final_return_mean"]
        )
        entry["ordering_holds"] = all(d > 0 for d in components.values())
        entry["min_component"] = min(components.values())
        ranked.append(entry)

    # Ranked by the weakest adjacent step. An instance with one large component
    # and one near zero has two lines of the ladder on top of each other, which
    # is what the screen exists to reject, and the meta-RL arms need room in
    # every step rather than a large total.
    ranked.sort(key=lambda e: e["min_component"], reverse=True)

    stats = {
        "run_mode": mode,
        "iterations_override": args.iterations,
        "seeds_override": args.seeds,
        "instances": args.instances,
        "results": results,
        "ranked": ranked,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    _report(ranked, mode, args.iterations, args.seeds)

    best = ranked[0]
    run.ok(
        key_stats={
            "best_instance": best["instance"],
            "best_min_component": round(best["min_component"], 3),
            "best_reference_gap": round(best["reference_gap"], 3),
            "n_instances": len(ranked),
            "n_ordered": sum(1 for e in ranked if e["ordering_holds"]),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
