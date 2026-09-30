"""M5R Stage D — posterior-vs-performance probe across the M5R final cells.

Mirrors `scripts/m6_posterior_probe.py` but reads cells from M5R's
`results/M5R/final/per_cell_env.json` instead of the M6 axis-level structure.

For each of the four meta-RL variants on the selected RSMM environment, load
per-seed checkpoints, collect rollouts, and fit logistic or MLP regime probes.
Fresh frozen-policy evaluation returns are joined by experiment and seed to
produce the reference-gap fraction used in the return association analysis.

References (regime-agnostic / belief / oracle PPO) are NOT probed for the
same reasons as in M6: floor has no regime representation, belief ingests
the analytical posterior directly, and oracle sees the true regime.

Outputs:
  - results/analysis/m5r_posterior_vs_performance{_classifier}.json

Usage:
  uv run python -m scripts.m5r_posterior_probe                    # logistic
  uv run python -m scripts.m5r_posterior_probe --classifier mlp   # MLP probe
  uv run python -m scripts.m5r_posterior_probe --n-rollouts 50    # quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from utils.paths import analysis_dir, experiment_dir
from typing import Any

import numpy as np

from evaluation import protocol as P
from evaluation.posterior_probe import load_experiment, probe_one_seed
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

PROBED_METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
PER_CELL_ENV_PATH = analysis_dir() / "per_cell_env.json"


def _gap_closed_per_seed(
    per_seed_return: list[float], floor_mean: float | None, ceiling_mean: float | None,
) -> list[float]:
    """The gap-closed fraction of Section 3.9.1, one value per seed.

    The ceiling is Belief-PPO, not Oracle-PPO. Dividing by the oracle gap
    instead puts this figure on a different scale from every other gap-closed
    number the thesis reports, under the same axis label.
    """
    if floor_mean is None or ceiling_mean is None:
        return [float("nan")] * len(per_seed_return)
    if np.isnan(floor_mean) or np.isnan(ceiling_mean) or ceiling_mean == floor_mean:
        return [float("nan")] * len(per_seed_return)
    arr = np.asarray(per_seed_return, dtype=float)
    return ((arr - floor_mean) / (ceiling_mean - floor_mean)).tolist()


def _probe_cell(
    experiment_name: str, n_rollouts: int, rollout_length: int, classifier: str,
) -> dict[str, Any]:
    exp_dir = experiment_dir(experiment_name)
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment dir missing: {exp_dir}")
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")

    method_accs: list[float] = []
    analytical_accs: list[float] = []
    method_kls: list[float] = []
    analytical_kls: list[float] = []
    method_lls: list[float] = []
    method_briers: list[float] = []
    method_per_t: list[list[float]] = []
    analytical_per_t: list[list[float]] = []
    method_per_t_kl: list[list[float]] = []
    analytical_per_t_kl: list[list[float]] = []
    method_since_change: list[list[float]] = []
    analytical_since_change: list[list[float]] = []
    since_change_counts: list[list[int]] = []
    since_change_labels: list[str] | None = None
    for seed in seeds:
        bundle = load_experiment(exp_dir, seed)
        result = probe_one_seed(
            bundle, n_rollouts=n_rollouts, rollout_length=rollout_length,
            classifier=classifier, rng_key=seed,
        )
        method_accs.append(float(result["method"]["test_acc"]))
        analytical_accs.append(float(result["analytical"]["test_acc"]))
        method_kls.append(float(result["method"]["test_kl_to_omega"]))
        analytical_kls.append(float(result["analytical"]["test_kl_to_omega"]))
        method_lls.append(float(result["method"]["test_log_loss"]))
        method_briers.append(float(result["method"]["test_brier"]))
        method_per_t.append(list(map(float, result["method"]["per_t_test_acc"])))
        analytical_per_t.append(
            list(map(float, result["analytical"]["per_t_test_acc"]))
        )
        method_per_t_kl.append(
            list(map(float, result["method"]["per_t_test_kl_to_omega"]))
        )
        analytical_per_t_kl.append(
            list(map(float, result["analytical"]["per_t_test_kl_to_omega"]))
        )
        method_since_change.append(
            list(map(float, result["method"]["test_acc_since_change"]))
        )
        analytical_since_change.append(
            list(map(float, result["analytical"]["test_acc_since_change"]))
        )
        since_change_counts.append(
            list(map(int, result["method"]["test_count_since_change"]))
        )
        since_change_labels = result["method"]["steps_since_change_labels"]
    return {
        "seeds": seeds,
        "method_test_acc_per_seed": method_accs,
        "analytical_test_acc_per_seed": analytical_accs,
        "method_kl_per_seed": method_kls,
        "analytical_kl_per_seed": analytical_kls,
        "method_log_loss_per_seed": method_lls,
        "method_brier_per_seed": method_briers,
        "method_per_t_test_acc_per_seed": method_per_t,
        "analytical_per_t_test_acc_per_seed": analytical_per_t,
        "method_per_t_kl_per_seed": method_per_t_kl,
        "analytical_per_t_kl_per_seed": analytical_per_t_kl,
        "steps_since_change_labels": since_change_labels,
        "method_test_acc_since_change_per_seed": method_since_change,
        "analytical_test_acc_since_change_per_seed": analytical_since_change,
        "test_count_since_change_per_seed": since_change_counts,
    }


def _decoupling_diagnostic(
    scatter_points: list[dict[str, Any]], error_key: str = "posterior_error",
) -> dict[str, Any]:
    """Flag cells where belief-quality rank and performance rank disagree.

    ``error_key`` selects the belief-error measure (higher = worse belief);
    quality is its negation. Default ``posterior_error`` is accuracy-based;
    pass ``belief_error_kl`` for the KL-based (primary) diagnostic.
    """
    by_cell: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for p in scatter_points:
        by_cell.setdefault(p["env_label"], {}).setdefault(p["method"], []).append(p)
    flagged: dict[str, list[str]] = {}
    for env_label, methods in by_cell.items():
        if len(methods) < 2:
            continue
        post_means = {
            m: float(np.mean([-p[error_key] for p in pts]))
            for m, pts in methods.items()
        }
        gc_means = {
            m: float(np.mean([p["gap_closed"] for p in pts]))
            for m, pts in methods.items()
        }
        post_order = sorted(post_means, key=post_means.get, reverse=True)
        gc_order = sorted(gc_means, key=gc_means.get, reverse=True)
        n = len(methods)
        half = n // 2 if n % 2 == 0 else n // 2 + 1
        for method in methods:
            in_top_post = post_order.index(method) < half
            in_top_gc = gc_order.index(method) < half
            if in_top_post != in_top_gc:
                flagged.setdefault(method, []).append(env_label)
    return {
        "methods_with_decoupling": sorted(flagged),
        "per_method_envs": flagged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_posterior_probe")
    # The protocol fixes the probe buffer at PROBE_TIMESTEPS, so the default
    # is derived rather than typed: a bare run and the programme run then
    # produce the same numbers, which a hard-coded 200 did not.
    parser.add_argument("--n-rollouts", type=int,
                        default=P.PROBE_TIMESTEPS // P.ROLLOUT_LENGTH)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument(
        "--classifier", choices=["logistic", "mlp"], default="logistic",
    )
    parser.add_argument(
        "--env-filter", default=None,
        help="if set, probe only this env_label (e.g. e_final)",
    )
    parser.add_argument(
        "--tag", default="",
        help="suffix appended to the output filename so a targeted run does "
             "not overwrite the canonical probe output",
    )
    args = parser.parse_args()

    out_dir = analysis_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.classifier == "logistic" else f"_{args.classifier}"
    tag = f"_{args.tag}" if args.tag else ""
    stats_path = out_dir / f"m5r_posterior_vs_performance{suffix}{tag}.json"
    summary_path = out_dir / f"m5r_posterior_vs_performance{suffix}{tag}_run.json"

    run = ScriptRun(script="m5r_posterior_probe")
    if not PER_CELL_ENV_PATH.exists():
        run.fail(
            reason=f"missing {PER_CELL_ENV_PATH}; run scripts.m5r_final_eval first",
            summary_path=summary_path,
        )
        return 1
    with open(PER_CELL_ENV_PATH) as f:
        per_cell_env = json.load(f)

    with open(out_dir / "m5r_post_training_evaluation.json") as f:
        evaluation_methods = json.load(f)["methods"]

    t_start = time.perf_counter()
    scatter_points: list[dict[str, Any]] = []
    per_method_per_env: dict[str, dict[str, dict[str, Any]]] = {}
    failed: list[str] = []
    n_total = 0
    for env_label, env_block in per_cell_env.get("per_env", {}).items():
        if args.env_filter and env_label != args.env_filter:
            continue
        refs = env_block.get("refs", {})
        floor = evaluation_methods["regime_agnostic_ppo"]["evaluation_return_mean"]
        ceiling = evaluation_methods["belief_ppo"]["evaluation_return_mean"]
        cells = env_block.get("cells", {})
        for method in PROBED_METHODS:
            cell = cells.get(method)
            if cell is None:
                continue
            n_total += 1
            experiment_name = f"m5r_final_{method}_{env_label}"
            try:
                probe = _probe_cell(
                    experiment_name,
                    n_rollouts=args.n_rollouts,
                    rollout_length=args.rollout_length,
                    classifier=args.classifier,
                )
            except Exception as e:
                print(f"[probe] {experiment_name} FAILED: {e}", flush=True)
                failed.append(experiment_name)
                continue
            from scripts.m5r_refresh_probe_returns import evaluation_returns_for_seeds
            evaluation_returns = evaluation_returns_for_seeds(
                evaluation_methods, method, experiment_name, probe["seeds"])
            gc_per_seed = _gap_closed_per_seed(evaluation_returns, floor, ceiling)
            for seed, m_acc, a_acc, m_kl, a_kl, m_ll, m_br, gc in zip(
                probe["seeds"], probe["method_test_acc_per_seed"],
                probe["analytical_test_acc_per_seed"],
                probe["method_kl_per_seed"], probe["analytical_kl_per_seed"],
                probe["method_log_loss_per_seed"], probe["method_brier_per_seed"],
                gc_per_seed,
            ):
                scatter_points.append({
                    "method": method,
                    "env_label": env_label,
                    "experiment_name": experiment_name,
                    "seed": seed,
                    # Decodability (kept as a secondary metric).
                    "method_test_acc": m_acc,
                    "analytical_test_acc": a_acc,
                    "posterior_error": a_acc - m_acc,
                    # Belief quality (primary): KL of the analytical ceiling
                    # belief from the probe-recovered belief, plus proper-score
                    # cross-checks. belief_error_kl subtracts the analytical
                    # probe's small reconstruction residual.
                    "method_kl_to_omega": m_kl,
                    "analytical_kl_to_omega": a_kl,
                    "method_log_loss": m_ll,
                    "method_brier": m_br,
                    "belief_error_kl": m_kl - a_kl,
                    "gap_closed": gc,
                })
            method_per_t = np.asarray(probe["method_per_t_test_acc_per_seed"])
            analytical_per_t = np.asarray(probe["analytical_per_t_test_acc_per_seed"])
            method_per_t_kl = np.asarray(probe["method_per_t_kl_per_seed"])
            analytical_per_t_kl = np.asarray(probe["analytical_per_t_kl_per_seed"])
            method_since_change = np.asarray(
                probe["method_test_acc_since_change_per_seed"]
            )
            analytical_since_change = np.asarray(
                probe["analytical_test_acc_since_change_per_seed"]
            )
            per_method_per_env.setdefault(env_label, {})[method] = {
                "n_seeds": int(method_per_t.shape[0]),
                "method_per_t_test_acc_mean": method_per_t.mean(axis=0).tolist(),
                "method_per_t_test_acc_per_seed": method_per_t.tolist(),
                "analytical_per_t_test_acc_mean": analytical_per_t.mean(axis=0).tolist(),
                "method_per_t_kl_mean": method_per_t_kl.mean(axis=0).tolist(),
                "method_per_t_kl_per_seed": method_per_t_kl.tolist(),
                "analytical_per_t_kl_mean": analytical_per_t_kl.mean(axis=0).tolist(),
                "steps_since_change_labels": probe["steps_since_change_labels"],
                "method_test_acc_since_change_mean": np.nanmean(
                    method_since_change, axis=0
                ).tolist(),
                "method_test_acc_since_change_per_seed": method_since_change.tolist(),
                "analytical_test_acc_since_change_mean": np.nanmean(
                    analytical_since_change, axis=0
                ).tolist(),
                "analytical_test_acc_since_change_per_seed": analytical_since_change.tolist(),
                "test_count_since_change_per_seed": probe["test_count_since_change_per_seed"],
                "method_test_acc_mean": float(np.mean(probe["method_test_acc_per_seed"])),
                "method_test_acc_per_seed": probe["method_test_acc_per_seed"],
                "method_test_acc_ci95": [
                    float(np.percentile(np.asarray(probe["method_test_acc_per_seed"]), 2.5)),
                    float(np.percentile(np.asarray(probe["method_test_acc_per_seed"]), 97.5)),
                ],
                "analytical_test_acc_per_seed": probe["analytical_test_acc_per_seed"],
                # Belief-quality metrics (primary).
                "method_kl_to_omega_mean": float(np.mean(probe["method_kl_per_seed"])),
                "method_kl_to_omega_per_seed": probe["method_kl_per_seed"],
                "method_kl_to_omega_ci95": [
                    float(np.percentile(np.asarray(probe["method_kl_per_seed"]), 2.5)),
                    float(np.percentile(np.asarray(probe["method_kl_per_seed"]), 97.5)),
                ],
                "analytical_kl_to_omega_mean": float(np.mean(probe["analytical_kl_per_seed"])),
                "method_log_loss_mean": float(np.mean(probe["method_log_loss_per_seed"])),
                "method_log_loss_per_seed": probe["method_log_loss_per_seed"],
                "method_brier_mean": float(np.mean(probe["method_brier_per_seed"])),
                # Per-seed, so the proper-score tables carry an interval across
                # seeds without reading the scatter points back.
                "method_brier_per_seed": probe["method_brier_per_seed"],
            }
            m_mean = float(np.mean(probe["method_test_acc_per_seed"]))
            a_mean = float(np.mean(probe["analytical_test_acc_per_seed"]))
            m_kl_mean = float(np.mean(probe["method_kl_per_seed"]))
            m_ll_mean = float(np.mean(probe["method_log_loss_per_seed"]))
            print(
                f"[probe] {experiment_name:>50s} | acc={m_mean:.3f} "
                f"(anal={a_mean:.3f}) KL={m_kl_mean:.3f} logloss={m_ll_mean:.3f}",
                flush=True,
            )

    if not scatter_points:
        run.fail(
            reason="no scatter points collected",
            summary_path=summary_path,
        )
        return 1

    def _corr(points: list[dict[str, Any]], xkey: str, ykey: str = "gap_closed") -> float:
        x = np.asarray([p[xkey] for p in points], dtype=float)
        y = np.asarray([p[ykey] for p in points], dtype=float)
        v = ~np.isnan(x) & ~np.isnan(y)
        if v.sum() >= 2 and np.std(x[v]) > 0 and np.std(y[v]) > 0:
            return float(np.corrcoef(x[v], y[v])[0, 1])
        return float("nan")

    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in scatter_points:
        by_method.setdefault(p["method"], []).append(p)

    # Accuracy-based correlation kept for continuity; KL-based is primary.
    correlation_overall = _corr(scatter_points, "posterior_error")
    correlation_overall_kl = _corr(scatter_points, "belief_error_kl")
    correlation_per_method = {
        m: _corr(pts, "posterior_error") for m, pts in by_method.items()
    }
    correlation_per_method_kl = {
        m: _corr(pts, "belief_error_kl") for m, pts in by_method.items()
    }

    decoupling = _decoupling_diagnostic(scatter_points, error_key="posterior_error")
    decoupling_kl = _decoupling_diagnostic(scatter_points, error_key="belief_error_kl")

    stats = {
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "classifier": args.classifier,
        "n_scatter_points": len(scatter_points),
        "n_cells_attempted": n_total,
        "n_cells_failed": len(failed),
        "scatter_points": scatter_points,
        "per_method_per_env": per_method_per_env,
        "correlation_overall": correlation_overall,
        "correlation_per_method": correlation_per_method,
        "correlation_overall_kl": correlation_overall_kl,
        "correlation_per_method_kl": correlation_per_method_kl,
        "decoupling_detected": decoupling,
        "decoupling_detected_kl": decoupling_kl,
    }
    from scripts.m5r_refresh_probe_returns import refresh_probe
    stats = refresh_probe(stats, evaluation_methods)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    if failed:
        run.fail(
            reason=f"{len(failed)}/{n_total} cells failed: " + ", ".join(failed[:3]),
            summary_path=summary_path,
        )
        return 1

    run.ok(
        key_stats={
            "n_scatter_points": len(scatter_points),
            "correlation_overall": correlation_overall,
            "elapsed_min": round(total_min, 2),
            "classifier": args.classifier,
        },
        summary_path=summary_path,
    )

    print("[probe] === summary ===", flush=True)
    print(
        f"[probe] r_overall(acc)={correlation_overall:+.3f} "
        f"r_overall(KL)={correlation_overall_kl:+.3f}",
        flush=True,
    )
    for method in correlation_per_method:
        print(
            f"[probe]   {method:>22s} | r_acc={correlation_per_method[method]:+.3f} "
            f"r_kl={correlation_per_method_kl[method]:+.3f}",
            flush=True,
        )
    if decoupling["methods_with_decoupling"]:
        print(
            f"[probe] decoupling: {decoupling['methods_with_decoupling']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
