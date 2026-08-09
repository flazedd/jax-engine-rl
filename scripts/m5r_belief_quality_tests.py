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
    primary belief metric. Lower is better, so a negative paired difference
    (hypernet minus concat) favours the hypernetwork.
  * ``method_test_acc`` — decodability, top-1 accuracy. Higher is better.

Both are read from the *linear* probe, which is the main instrument. The MLP
probe and the two proper scores (log-loss, Brier) are robustness checks and are
deliberately left uncorrected: they are reported next to these results but no
claim rests on them alone.

Reads:
  results/M5R/final/m5r_posterior_vs_performance.json
Writes:
  results/M5R/final/m5r_belief_quality_tests.json
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
    holm_bonferroni,
    leave_one_out_sensitivity,
    paired_wilcoxon,
    rank_biserial,
    bootstrap_paired_mean_ci,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "M5R" / "final"

ALPHA = 0.05
N_BOOT = 10_000

# (metric key in scatter_points, direction that favours the hypernetwork)
METRICS = [
    ("method_kl_to_omega", "lower"),
    ("method_test_acc", "higher"),
]
METHODS = ["rl2", "varibad"]


def _per_seed(
    scatter: list[dict], env_label: str
) -> dict[tuple[str, str], dict[int, float]]:
    """(cell, metric) -> {seed: value}, so pairs can be aligned by seed."""
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in scatter:
        if row.get("env_label") != env_label:
            continue
        for metric, _ in METRICS:
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_belief_quality_tests")
    parser.add_argument("--env-label", default="e_final")
    parser.add_argument(
        "--probe",
        default=str(RESULTS_ROOT / "m5r_posterior_vs_performance.json"),
        help="linear-probe output; the MLP probe is a robustness check and is "
             "not corrected here",
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
    for method in METHODS:
        for metric, direction in METRICS:
            hyp_vals, con_vals, seeds = _aligned(
                table, f"{method}_hypernet", f"{method}_concat", metric
            )
            if hyp_vals.size < 2:
                run.fail(reason=f"too few paired seeds for {method}/{metric}")
                return 1
            wil = paired_wilcoxon(hyp_vals, con_vals, alternative="two-sided")
            mean_delta, lo, hi = bootstrap_paired_mean_ci(
                hyp_vals, con_vals, n_boot=N_BOOT, alpha=ALPHA
            )
            # "Favours the hypernetwork" depends on the metric's direction, per
            # the metric-direction convention of the protocol.
            favours_hypernet = mean_delta < 0 if direction == "lower" else mean_delta > 0
            comparisons.append({
                "name": f"{method}_hypernet_vs_concat_{metric}",
                "method": method,
                "metric": metric,
                "direction_favouring_hypernet": direction,
                "n_pairs": int(hyp_vals.size),
                "seeds": seeds,
                "hypernet_mean": float(hyp_vals.mean()),
                "concat_mean": float(con_vals.mean()),
                "mean_paired_delta": mean_delta,
                "delta_ci": [lo, hi],
                "rank_biserial": rank_biserial(hyp_vals, con_vals),
                "wilcoxon_p": wil["p"],
                "wilcoxon_null_distribution": wil["null_distribution"],
                "n_zero_dropped": wil["n_zero_dropped"],
                "favours_hypernet": bool(favours_hypernet),
                "_pair": (hyp_vals, con_vals),
            })

    family_size = len(comparisons)
    holm = holm_bonferroni([c["wilcoxon_p"] for c in comparisons])
    for comp, p_holm in zip(comparisons, holm):
        comp["holm_corrected_p"] = p_holm
        comp["supported"] = bool(
            p_holm == p_holm and p_holm < ALPHA
        )  # NaN-safe: NaN != NaN
        hyp_vals, con_vals = comp.pop("_pair")
        comp["stable_under_seed_omission"] = bool(leave_one_out_sensitivity(
            hyp_vals, con_vals, alpha=ALPHA, n_corrections=family_size,
            alternative="two-sided",
        )["stable_under_seed_omission"])
        comp["ci_excludes_zero"] = bool(
            comp["delta_ci"][0] > 0 or comp["delta_ci"][1] < 0
        )

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
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_ROOT / "m5r_belief_quality_tests.json"
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))

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
            "n_supported": n_supported,
            "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        },
        summary_path=RESULTS_ROOT / "m5r_belief_quality_tests_run.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
