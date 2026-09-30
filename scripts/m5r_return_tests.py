"""The two return comparison sets that had no producing script.

`m5r_hypothesis_tests` covers the conditioning contrast within each method, and
`m5r_diagnostic_tests` and `m5r_belief_quality_tests` cover their own sets. Two
of the five sets declared in `evaluation.protocol` had no script and no
artifact, so the six corrected p-values the thesis prints for them could not be
regenerated or audited:

  * ``returns_method`` — RL2 against VariBAD at a fixed conditioning
    architecture, from fresh post-training evaluation episodes.
  * ``time_to_threshold`` — zero-based start index of the first forward
    25-iteration window whose mean reaches the regime-agnostic reference
    return, one comparison per method, from the per-seed learning curves.

Both are computed here in the same schema the other set scripts write, so
`scripts.thesis_contract` can check them alongside the rest.

Reads:
  results/analysis/m5r_post_training_evaluation.json (fresh evaluation returns)
  results/analysis/per_cell_env.json          (training reference level)
  results/<experiment>/metrics.json           (per-seed learning curves)
Writes:
  results/analysis/m5r_method_return_tests.json
  results/analysis/m5r_time_to_threshold_tests.json

    uv run python -m scripts.m5r_return_tests
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from evaluation import protocol as P
from evaluation.metrics import (
    bootstrap_independent_mean_ci,
    holm_bonferroni,
    leave_one_run_out_independent_sensitivity,
    permutation_mean_test,
)
from utils.paths import analysis_dir, experiment_dir, resolve_data
from utils.script_output import ScriptRun

ALPHA = 0.05
N_BOOT = 10_000
SMOOTH_WINDOW = 25

# One target, shared by both methods: the return the regime-agnostic agent
# reaches at the end of training. A defined reference level rather than a chosen
# number removes the target as a degree of freedom, and the same target for both
# methods makes the two comparisons comparable. A run that never reaches it is
# assigned the end of the training budget, so every run remains in the analysis.
THRESHOLD_TARGET_REF = "regime_agnostic_ppo"
METHODS = ("rl2", "varibad")
ARCHITECTURES = ("concat", "hypernet")


def _per_seed_evaluation(methods: dict, variant: str) -> np.ndarray:
    return np.asarray(
        methods[variant]["per_seed_evaluation_return"], dtype=float
    )


def _per_seed_curves(experiment: str) -> np.ndarray | None:
    path = resolve_data(experiment_dir(experiment) / "metrics.json")
    if not path.exists():
        return None
    with open(path) as f:
        metrics = json.load(f)
    curves = metrics.get("per_seed_mean_return_per_iter")
    return None if curves is None else np.asarray(curves, dtype=float)


def _first_reach(curve: np.ndarray, target: float) -> int | None:
    """Start index of the first forward window whose mean reaches `target`.

    Smoothing first, because a single noisy iteration touching the target is
    not the run having learned to reach it.
    """
    smoothed = np.convolve(curve, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                           mode="valid")
    idx = int(np.argmax(smoothed >= target))
    return idx if smoothed[idx] >= target else None


def _compare(better: np.ndarray, worse: np.ndarray, name: str,
             family_size: int) -> dict:
    """One comparison of two independently trained groups."""
    perm = permutation_mean_test(better, worse)
    mean, lo, hi = bootstrap_independent_mean_ci(better, worse, n_boot=N_BOOT,
                                                  alpha=ALPHA)
    omission = leave_one_run_out_independent_sensitivity(
        better, worse, alpha=ALPHA, n_corrections=family_size,
    )
    return {
        "name": name,
        "n_method": int(better.size),
        "n_baseline": int(worse.size),
        "mean_difference": mean,
        "delta_ci": [lo, hi],
        "permutation_p": perm["p"],
        "permutation_null_distribution": perm["null_distribution"],
        "n_permutations": perm["n_permutations"],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "stable_under_run_omission": omission["stable_under_run_omission"],
    }


def _finalise(comparisons: list[dict], key: str) -> dict:
    expected = P.COMPARISON_SETS[key]
    for comp, p_holm in zip(comparisons,
                            holm_bonferroni([c["permutation_p"] for c in comparisons])):
        comp["holm_corrected_p"] = p_holm
        comp["supported"] = bool(p_holm == p_holm and p_holm < ALPHA)
    return {
        "comparison_set": key,
        "alternative": P.ALTERNATIVE,
        "alpha": ALPHA,
        "family_size": expected.size,
        "n_supported": sum(1 for c in comparisons if c["supported"]),
        "comparisons": comparisons,
    }


def method_return_tests(methods: dict) -> dict:
    """RL2 minus VariBAD at each conditioning architecture."""
    size = P.COMPARISON_SETS["returns_method"].size
    comparisons = []
    for arch in ARCHITECTURES:
        rl2 = _per_seed_evaluation(methods, f"rl2_{arch}")
        varibad = _per_seed_evaluation(methods, f"varibad_{arch}")
        comp = _compare(rl2, varibad, f"rl2_minus_varibad_{arch}", size)
        comp["architecture"] = arch
        comp["seeds_favouring_rl2"] = int((rl2 > varibad).sum())
        comp["rl2_mean"] = float(rl2.mean())
        comp["varibad_mean"] = float(varibad.mean())
        comparisons.append(comp)
    return _finalise(comparisons, "returns_method")


def time_to_threshold_tests(env_label: str, target: float) -> dict | None:
    """Threshold-window start index, hypernetwork minus concatenation."""
    size = P.COMPARISON_SETS["time_to_threshold"].size
    comparisons = []
    for method in METHODS:
        concat = _per_seed_curves(f"m5r_final_{method}_concat_{env_label}")
        hyper = _per_seed_curves(f"m5r_final_{method}_hypernet_{env_label}")
        if concat is None or hyper is None:
            return None
        reach_c = [_first_reach(c, target) for c in concat]
        reach_h = [_first_reach(c, target) for c in hyper]
        budget = max(len(curve) for curve in np.vstack([concat, hyper]))
        c_it = np.asarray([budget if value is None else value for value in reach_c], dtype=float)
        h_it = np.asarray([budget if value is None else value for value in reach_h], dtype=float)
        comp = _compare(h_it, c_it, f"{method}_hypernet_vs_concat", size)
        comp["method"] = method
        comp["target_return"] = float(target)
        comp["target_source"] = THRESHOLD_TARGET_REF
        comp["smoothing_window"] = SMOOTH_WINDOW
        comp["index_definition"] = "zero_based_start_of_forward_window"
        comp["concat_mean_iterations"] = float(c_it.mean())
        comp["hypernet_mean_iterations"] = float(h_it.mean())
        comp["runs_faster_under_hypernet"] = int((h_it < c_it).sum())
        comp["n_censored_runs"] = int(sum(r is None for r in reach_c) + sum(r is None for r in reach_h))
        comp["censored_concat"] = int(sum(1 for r in reach_c if r is None))
        comp["censored_hypernet"] = int(sum(1 for r in reach_h if r is None))
        comparisons.append(comp)
    return _finalise(comparisons, "time_to_threshold")


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_return_tests")
    parser.add_argument("--env-label", default=P.MEDIUM_ENV)
    args = parser.parse_args()

    run = ScriptRun(script="m5r_return_tests")
    t0 = time.perf_counter()
    out_dir = analysis_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "m5r_return_tests_run.json"

    per_cell = resolve_data(out_dir / "per_cell_env.json")
    if not per_cell.exists():
        run.fail(reason=f"missing {per_cell}", summary_path=summary_path)
        return 1
    with open(per_cell) as f:
        env_block = json.load(f)["per_env"].get(args.env_label)
    if env_block is None:
        run.fail(reason=f"no cells for env {args.env_label}",
                 summary_path=summary_path)
        return 1

    evaluation_path = resolve_data(out_dir / "m5r_post_training_evaluation.json")
    if not evaluation_path.exists():
        run.fail(reason=f"missing {evaluation_path}", summary_path=summary_path)
        return 1
    with open(evaluation_path) as f:
        evaluation_methods = json.load(f).get("methods", {})

    required = {f"{method}_{arch}" for method in METHODS for arch in ARCHITECTURES}
    missing = sorted(required - set(evaluation_methods))
    if missing:
        run.fail(reason=f"evaluation methods missing: {missing}",
                 summary_path=summary_path)
        return 1

    written = []
    method_payload = method_return_tests(evaluation_methods)
    method_path = out_dir / "m5r_method_return_tests.json"
    with open(method_path, "w") as f:
        json.dump(method_payload, f, indent=2)
    written.append(method_path)

    target = env_block["refs"].get(THRESHOLD_TARGET_REF)
    if target is None:
        run.fail(reason=f"no {THRESHOLD_TARGET_REF} reference return",
                 summary_path=summary_path)
        return 1
    speed_payload = time_to_threshold_tests(args.env_label, float(target))
    if speed_payload is None:
        run.fail(reason="per-seed learning curves missing for one variant",
                 summary_path=summary_path)
        return 1
    speed_path = out_dir / "m5r_time_to_threshold_tests.json"
    with open(speed_path, "w") as f:
        json.dump(speed_payload, f, indent=2)
    written.append(speed_path)

    for payload in (method_payload, speed_payload):
        for c in payload["comparisons"]:
            lo, hi = c["delta_ci"]
            print(f"[{payload['comparison_set']}] {c['name']}: "
                  f"mean difference={c['mean_difference']:+.2f} CI[{lo:+.2f},{hi:+.2f}] "
                  f"p_holm={c['holm_corrected_p']:.4g} "
                  f"supported={c['supported']}", flush=True)

    for path in written:
        run.add_output(str(path))
    run.ok(key_stats={
        "n_comparisons": sum(len(p["comparisons"])
                             for p in (method_payload, speed_payload)),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
    }, summary_path=summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
