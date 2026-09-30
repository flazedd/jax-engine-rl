"""Belief-quality comparison set — the third Holm family of the protocol.

The statistical protocol corrects three sets separately: the return
comparisons, the behavioural diagnostics, and the belief-quality comparisons
tested here. Belief quality concerns the belief a variant *forms*, which is a
different question from what it does with that belief or what it earns, so
pooling it with either of the other sets would make return claims pay a
correction for tests that carry no claim about returns.

The set spans two metrics over two methods, so four hypernet-versus-concat
comparisons:

  * ``method_kl_to_omega`` — forward KL to the analytical posterior, the
    primary belief metric. Lower is better, so a negative mean difference
    (hypernet minus concat) favours the hypernetwork.
  * ``method_test_acc`` — decodability, top-1 accuracy. Higher is better.

Both are read from the *linear* probe, which is the main instrument. The MLP
probe is corrected as its own four-test family. The two proper scores
(log-loss, Brier) are additional, uncorrected robustness checks.

Reads:
  results/analysis/m5r_posterior_vs_performance.json
Writes:
  results/analysis/m5r_belief_quality_tests.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    bootstrap_independent_mean_ci,
    holm_bonferroni,
    leave_one_run_out_independent_sensitivity,
    permutation_mean_test,
)
from utils.script_output import ScriptRun

from evaluation import protocol as P
from evaluation.protocol import MEDIUM_ENV
from utils.paths import analysis_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = analysis_dir()

ALPHA = 0.05
N_BOOT = 10_000

# (metric key in scatter_points, direction that favours the hypernetwork)
METRICS = [
    ("method_kl_to_omega", "lower"),
    ("method_test_acc", "higher"),
]
# The proper scores of the protocol. They are computed on both probe families
# and left uncorrected; the primary MLP metrics have their own corrected set.
ROBUSTNESS_METRICS = [
    ("method_log_loss", "lower"),
    ("method_brier", "lower"),
]
ALL_METRICS = METRICS + ROBUSTNESS_METRICS
METHODS = ["rl2", "varibad"]


def _per_seed(
    scatter: list[dict], env_label: str
) -> dict[tuple[str, str], dict[int, float]]:
    """(cell, metric) -> {seed: value}, so pairs can be aligned by seed."""
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in scatter:
        if row.get("env_label") != env_label:
            continue
        for metric, _ in ALL_METRICS:
            if metric in row:
                out[(row["method"], metric)][int(row["seed"])] = float(row[metric])
    return out


def _aligned(
    table: dict[tuple[str, str], dict[int, float]], hyp: str, con: str, metric: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    a = table.get((hyp, metric), {})
    b = table.get((con, metric), {})
    seeds = sorted(set(a) & set(b))
    return (
        np.array([a[s] for s in seeds], dtype=float),
        np.array([b[s] for s in seeds], dtype=float),
        seeds,
    )


def _compare(
    hyp_vals: np.ndarray, con_vals: np.ndarray, seeds: list[int],
    method: str, metric: str, direction: str, probe: str,
) -> dict:
    """One independent hypernetwork-versus-concatenation comparison."""
    perm = permutation_mean_test(hyp_vals, con_vals)
    mean_delta, lo, hi = bootstrap_independent_mean_ci(
        hyp_vals, con_vals, n_boot=N_BOOT, alpha=ALPHA
    )
    # "Favours the hypernetwork" depends on the metric's direction, per
    # the metric-direction convention of the protocol.
    favours = mean_delta < 0 if direction == "lower" else mean_delta > 0
    # Seeds favouring the hypernetwork, in the metric's own direction. The
    # protocol reports this beside the interval instead of an effect size.
    per_seed = (hyp_vals < con_vals) if direction == "lower" else (hyp_vals > con_vals)
    return {
        "seeds_favouring_hypernet": int(per_seed.sum()),
        "name": f"{method}_hypernet_vs_concat_{metric}",
        "method": method,
        "metric": metric,
        "probe": probe,
        "direction_favouring_hypernet": direction,
        "n_pairs": int(hyp_vals.size),
        "n_hypernet": int(hyp_vals.size),
        "n_concat": int(con_vals.size),
        "seeds": seeds,
        "hypernet_mean": float(hyp_vals.mean()),
        "concat_mean": float(con_vals.mean()),
        "mean_difference": mean_delta,
        "permutation_p": perm["p"],
        "mean_paired_delta": mean_delta,
        "delta_ci": [lo, hi],
        "rank_biserial": float("nan"),
        "wilcoxon_p": perm["p"],
        "wilcoxon_null_distribution": perm["null_distribution"],
        "n_zero_dropped": 0,
        "favours_hypernet": bool(favours),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_belief_quality_tests")
    parser.add_argument("--env-label", default=MEDIUM_ENV)
    parser.add_argument(
        "--probe",
        default=str(RESULTS_ROOT / "m5r_posterior_vs_performance.json"),
        help="linear-probe output; the MLP probe is a robustness check and is "
             "not corrected here",
    )
    parser.add_argument(
        "--probe-mlp",
        default=str(RESULTS_ROOT / "m5r_posterior_vs_performance_mlp.json"),
        help="MLP-probe output, whose comparisons are robustness checks",
    )
    args = parser.parse_args()

    run = ScriptRun(script="m5r_belief_quality_tests")
    t0 = time.perf_counter()

    probe_path = Path(args.probe)
    if not probe_path.exists():
        run.fail(reason=f"probe output missing: {probe_path}")
        return 1
    with open(probe_path) as f:
        probe = json.load(f)

    scatter = probe.get("scatter_points", [])
    table = _per_seed(scatter, args.env_label)

    missing = [
        f"{m}_{arch}/{metric}"
        for m in METHODS
        for arch in ("hypernet", "concat")
        for metric, _ in METRICS
        if (f"{m}_{arch}", metric) not in table
    ]
    if missing:
        run.fail(reason=f"probe output lacks per-seed metrics: {missing[:4]}")
        return 1

    comparisons = []
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        for metric, direction in METRICS:
            hyp_vals, con_vals, seeds = _aligned(
                table, f"{method}_hypernet", f"{method}_concat", metric
            )
            if hyp_vals.size < 2:
                run.fail(reason=f"too few runs for {method}/{metric}")
                return 1
            comp = _compare(hyp_vals, con_vals, seeds, method, metric,
                            direction, "linear")
            pairs[comp["name"]] = (hyp_vals, con_vals)
            comparisons.append(comp)

    family_size = len(comparisons)
    holm = holm_bonferroni([c["wilcoxon_p"] for c in comparisons])
    for comp, p_holm in zip(comparisons, holm):
        comp["holm_corrected_p"] = p_holm
        comp["supported"] = bool(
            p_holm == p_holm and p_holm < ALPHA
        )  # NaN-safe: NaN != NaN
        hyp_vals, con_vals = pairs[comp["name"]]
        comp["stable_under_seed_omission"] = bool(leave_one_run_out_independent_sensitivity(
            hyp_vals, con_vals, alpha=ALPHA, n_corrections=family_size,
        )["stable_under_run_omission"])
        comp["stable_under_run_omission"] = comp["stable_under_seed_omission"]

    # The robustness family: the proper scores under the linear probe, and every
    # metric under the MLP probe. Reported beside the corrected set and never
    # corrected with it, so no claim can rest on them alone.
    robustness: list[dict] = []
    mlp_pairs: dict[str, tuple] = {}
    for probe_label, probe_path_str, metric_list in (
        ("linear", args.probe, ROBUSTNESS_METRICS),
        ("mlp", args.probe_mlp, ALL_METRICS),
    ):
        probe_file = Path(probe_path_str)
        if not probe_file.exists():
            print(f"[belief_quality] skip {probe_label} robustness: "
                  f"{probe_file} missing", flush=True)
            continue
        with open(probe_file) as f:
            other = json.load(f)
        other_table = _per_seed(other.get("scatter_points", []), args.env_label)
        for method in METHODS:
            for metric, direction in metric_list:
                hyp_vals, con_vals, seeds = _aligned(
                    other_table, f"{method}_hypernet", f"{method}_concat", metric
                )
                if hyp_vals.size < 2:
                    print(f"[belief_quality] skip {probe_label}/{method}/{metric}: "
                          "too few runs", flush=True)
                    continue
                comp = _compare(hyp_vals, con_vals, seeds, method, metric,
                                direction, probe_label)
                if probe_label == "mlp":
                    mlp_pairs[comp["name"]] = (hyp_vals, con_vals)
                robustness.append(comp)

    # The MLP probe reads out the same two metrics over the same two methods, so
    # it is corrected the same way, within its own family. Every other
    # robustness entry stays uncorrected.
    mlp_family = [c for c in robustness
                  if c["probe"] == "mlp" and c["metric"] in dict(METRICS)]
    mlp_size = P.COMPARISON_SETS["belief_quality_mlp"].size
    if mlp_family:
        for comp, p_holm in zip(mlp_family,
                                holm_bonferroni([c["wilcoxon_p"] for c in mlp_family])):
            comp["holm_corrected_p"] = p_holm
            comp["comparison_set"] = "belief_quality_mlp"
            comp["supported"] = bool(p_holm == p_holm and p_holm < ALPHA)
            hyp_vals, con_vals = mlp_pairs[comp["name"]]
            comp["stable_under_seed_omission"] = bool(leave_one_run_out_independent_sensitivity(
                hyp_vals, con_vals, alpha=ALPHA, n_corrections=mlp_size,
            )["stable_under_run_omission"])
            comp["stable_under_run_omission"] = comp["stable_under_seed_omission"]

    n_supported = sum(1 for c in comparisons if c["supported"])
    payload = {
        "comparison_set": "belief_quality",
        "probe_source": str(probe_path),
        "classifier": probe.get("classifier"),
        "env_label": args.env_label,
        "alternative": "two-sided",
        "alpha": ALPHA,
        "family_size": family_size,
        "uncorrected_robustness_checks": [
            "method_log_loss", "method_brier", "mlp probe (all metrics)",
        ],
        "n_supported": n_supported,
        "comparisons": comparisons,
        "robustness_comparisons": robustness,
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_ROOT / "m5r_belief_quality_tests.json"
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))

    # The MLP set gets its own artifact, like every other comparison set, so the
    # contract checks it as a set rather than re-reading the linear one.
    mlp_path = RESULTS_ROOT / "m5r_belief_quality_mlp_tests.json"
    with open(mlp_path, "w") as f:
        json.dump({
            "comparison_set": "belief_quality_mlp",
            "probe_source": args.probe_mlp,
            "classifier": P.ROBUSTNESS_PROBE,
            "env_label": args.env_label,
            "alternative": "two-sided",
            "alpha": ALPHA,
            "family_size": mlp_size,
            "n_supported": sum(1 for c in mlp_family if c["supported"]),
            "comparisons": mlp_family,
        }, f, indent=2)
    run.add_output(str(mlp_path))

    for c in comparisons:
        print(
            f"[belief_quality] {c['name']}: d_bar={c['mean_paired_delta']:+.4f} "
            f"CI[{c['delta_ci'][0]:+.4f},{c['delta_ci'][1]:+.4f}] "
            f"r_rb={c['rank_biserial']:+.3f} p_holm={c['holm_corrected_p']:.4g} "
            f"supported={c['supported']}",
            flush=True,
        )

    run.ok(
        key_stats={
            "family_size": family_size,
            "n_robustness": len(robustness),
            "n_supported": n_supported,
            "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        },
        summary_path=RESULTS_ROOT / "m5r_belief_quality_tests_run.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
