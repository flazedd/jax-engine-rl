"""M6 — pre-registered hypothesis tests on the difficulty-sweep results.

Three families of hypotheses, each Holm-corrected separately:

Family A (12 hypotheses): hypernet > concat at every (axis, level).
  For RL² and VariBAD independently, at each of the 6 (axis, level) cells.
  Paired Wilcoxon (one-sided, alternative="greater") on n=8 seeds — both
  hypernet and concat cells use the same seed protocol from the M5 Step-4
  base configs. Family size = 12 for Holm correction.

Family B (4 hypotheses): hypernet > Belief-PPO at the two cells where
  the M6 sweep showed visible separation — persistence_hard (Belief
  collapses, hypernet retains) and distinguishability_easy (hypernet
  pushes past Belief). Paired Wilcoxon at n=5 (using hypernet seeds 0-4
  to match Belief's n=5). Family size = 4.

Family C (1 confidence interval): posterior_error ↔ gap_closed
  correlation. Bootstrap a 95% CI on the overall Pearson r across the
  192 (cell, seed) probe points. Decision: "support decoupling" iff the
  CI is contained within (-0.3, +0.3) — i.e., the correlation is
  meaningfully close to zero.

Outputs:
  results/milestones/M6/stats_M6_hypothesis_tests.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import (
    leave_one_out_sensitivity,
    primary_hypothesis_test,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

ALPHA = 0.05
N_BOOT = 10_000

# Family A: hypernet > concat at every (axis, level). 12 hypotheses.
FAMILY_A: list[tuple[str, str, str, str, str]] = []
for method in ("rl2", "varibad"):
    for axis in ("persistence", "distinguishability"):
        for level in ("easy", "medium", "hard"):
            hyp_name = f"{method}_hypernet_beats_concat_{axis}_{level}"
            FAMILY_A.append((
                hyp_name,
                f"{method}_hypernet",
                f"{method}_concat",
                axis,
                level,
            ))
FAMILY_A_SIZE = len(FAMILY_A)

# Family B: hypernet > Belief-PPO at the two "hypernet-pulls-ahead" cells
# identified during the sweep. Belief has n=5 seeds (M3 reference config),
# hypernet has n=8; we pair using hypernet seeds 0-4.
FAMILY_B: list[tuple[str, str, str, str, str]] = []
for method in ("rl2", "varibad"):
    for axis, level in (("persistence", "hard"), ("distinguishability", "easy")):
        hyp_name = f"{method}_hypernet_beats_belief_{axis}_{level}"
        FAMILY_B.append((
            hyp_name,
            f"{method}_hypernet",
            "belief_ppo",
            axis,
            level,
        ))
FAMILY_B_SIZE = len(FAMILY_B)


def _bootstrap_correlation_ci(
    xs: np.ndarray, ys: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA,
) -> dict[str, Any]:
    """Pearson correlation point estimate + percentile bootstrap CI."""
    valid = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[valid], ys[valid]
    n = xs.size
    if n < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return {
            "n": int(n),
            "correlation": float("nan"),
            "ci": [float("nan"), float("nan")],
        }
    r_hat = float(np.corrcoef(xs, ys)[0, 1])
    rng = np.random.default_rng(0)
    boot_rs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_b, y_b = xs[idx], ys[idx]
        if np.std(x_b) == 0 or np.std(y_b) == 0:
            boot_rs[b] = 0.0
        else:
            boot_rs[b] = np.corrcoef(x_b, y_b)[0, 1]
    lo = float(np.percentile(boot_rs, 100 * alpha / 2))
    hi = float(np.percentile(boot_rs, 100 * (1 - alpha / 2)))
    return {
        "n": int(n),
        "correlation": r_hat,
        "ci": [lo, hi],
    }


def _per_seed(cell: dict[str, Any], n_keep: int | None = None) -> list[float]:
    seeds = list(map(float, cell["per_seed_final_return"]))
    if n_keep is not None:
        seeds = seeds[:n_keep]
    return seeds


def _run_family(
    family: list[tuple[str, str, str, str, str]],
    family_size: int,
    sweep_results: dict[str, dict[str, dict[str, dict]]],
    *,
    pair_n_seeds: int | None,
) -> dict[str, dict[str, Any]]:
    """Run paired Wilcoxon + LOO for every hypothesis in `family`.

    `pair_n_seeds=None` → use all seeds present in both cells (must match).
    `pair_n_seeds=k` → truncate both cells to first k seeds (for cross-
    config pairings where the baseline has fewer seeds).
    """
    out: dict[str, dict[str, Any]] = {}
    for hyp_name, method, baseline, axis, level in family:
        cells = sweep_results.get(axis, {}).get(level, {})
        if method not in cells or baseline not in cells:
            out[hyp_name] = {"error": "missing cell"}
            continue
        method_seeds = _per_seed(cells[method], n_keep=pair_n_seeds)
        baseline_seeds = _per_seed(cells[baseline], n_keep=pair_n_seeds)
        if len(method_seeds) != len(baseline_seeds):
            # Truncate to the shorter list — paired Wilcoxon needs equal length.
            n_pair = min(len(method_seeds), len(baseline_seeds))
            method_seeds = method_seeds[:n_pair]
            baseline_seeds = baseline_seeds[:n_pair]
        test = primary_hypothesis_test(
            method=method_seeds,
            baseline=baseline_seeds,
            family_size=family_size,
            alpha=ALPHA,
            n_boot=N_BOOT,
            alternative="greater",
        )
        loo = leave_one_out_sensitivity(
            method=method_seeds,
            baseline=baseline_seeds,
            alpha=ALPHA,
            n_corrections=family_size,
            alternative="greater",
        )
        out[hyp_name] = {
            **test,
            "loo": loo,
            "axis": axis,
            "level": level,
            "method": method,
            "baseline": baseline,
            "n_pairs": len(method_seeds),
        }
    return out


def main() -> int:
    run = ScriptRun(script="m6_hypothesis_tests")
    out_dir = RESULTS_ROOT / "milestones" / "M6"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M6_hypothesis_tests.json"
    summary_path = out_dir / "stats_M6_hypothesis_tests_run.json"

    sweep_path = out_dir / "stats_M6_sweep.json"
    probe_path = out_dir / "stats_M6_posterior_vs_performance.json"
    if not sweep_path.exists():
        run.fail(reason=f"missing {sweep_path}", summary_path=summary_path)
        return 1
    if not probe_path.exists():
        run.fail(reason=f"missing {probe_path}", summary_path=summary_path)
        return 1
    with open(sweep_path) as f:
        sweep = json.load(f)
    with open(probe_path) as f:
        probe = json.load(f)

    t_start = time.perf_counter()

    # ---- Family A: hypernet > concat ---------------------------------------
    print(f"[m6_tests] Family A: {FAMILY_A_SIZE} hypotheses (hypernet > concat)", flush=True)
    family_a = _run_family(
        FAMILY_A, FAMILY_A_SIZE, sweep["results"], pair_n_seeds=None,
    )
    a_supported = sum(1 for h in family_a.values() if h.get("supported"))
    a_loo_robust = sum(
        1 for h in family_a.values()
        if h.get("supported") and h.get("loo", {}).get("robust_to_loo")
    )
    print(
        f"[m6_tests]   {a_supported}/{FAMILY_A_SIZE} supported "
        f"(Holm-corrected p < {ALPHA}, CI excludes 0); "
        f"{a_loo_robust} robust to LOO",
        flush=True,
    )

    # ---- Family B: hypernet > belief (selective) ---------------------------
    print(f"[m6_tests] Family B: {FAMILY_B_SIZE} hypotheses (hypernet > Belief-PPO)", flush=True)
    family_b = _run_family(
        FAMILY_B, FAMILY_B_SIZE, sweep["results"], pair_n_seeds=5,
    )
    b_supported = sum(1 for h in family_b.values() if h.get("supported"))
    b_loo_robust = sum(
        1 for h in family_b.values()
        if h.get("supported") and h.get("loo", {}).get("robust_to_loo")
    )
    print(
        f"[m6_tests]   {b_supported}/{FAMILY_B_SIZE} supported; "
        f"{b_loo_robust} robust to LOO",
        flush=True,
    )

    # ---- Family C: correlation CI ------------------------------------------
    points = probe.get("scatter_points", [])
    arr_pe = np.asarray([p["posterior_error"] for p in points], dtype=float)
    arr_gc = np.asarray([p["gap_closed"] for p in points], dtype=float)
    overall = _bootstrap_correlation_ci(arr_pe, arr_gc)
    # Decoupling supported iff the CI is contained in (-0.3, +0.3).
    decoupling_threshold = 0.30
    if any(np.isnan(overall["ci"])):
        decoupling_supported = False
    else:
        decoupling_supported = (
            overall["ci"][0] > -decoupling_threshold
            and overall["ci"][1] < decoupling_threshold
        )
    per_method_corr: dict[str, dict[str, Any]] = {}
    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        by_method.setdefault(p["method"], []).append(p)
    for method, pts in by_method.items():
        pe = np.asarray([p["posterior_error"] for p in pts], dtype=float)
        gc = np.asarray([p["gap_closed"] for p in pts], dtype=float)
        per_method_corr[method] = _bootstrap_correlation_ci(pe, gc)
    print(
        f"[m6_tests] Family C: overall correlation r = {overall['correlation']:+.3f} "
        f"(CI [{overall['ci'][0]:+.3f}, {overall['ci'][1]:+.3f}]) on n={overall['n']} points; "
        f"decoupling_supported={decoupling_supported} (|r| < {decoupling_threshold})",
        flush=True,
    )

    # ---- Pass summary ------------------------------------------------------
    pass_summary = {
        "family_a_all_supported": a_supported == FAMILY_A_SIZE,
        "family_a_supported_count": a_supported,
        "family_a_total": FAMILY_A_SIZE,
        "family_a_all_loo_robust": (
            a_supported > 0 and a_loo_robust == a_supported
        ),
        "family_b_all_supported": b_supported == FAMILY_B_SIZE,
        "family_b_supported_count": b_supported,
        "family_b_total": FAMILY_B_SIZE,
        "family_b_all_loo_robust": (
            b_supported > 0 and b_loo_robust == b_supported
        ),
        "decoupling_supported": decoupling_supported,
    }

    stats = {
        "alpha": ALPHA,
        "n_boot": N_BOOT,
        "family_a": {
            "name": "hypernet_beats_concat",
            "size": FAMILY_A_SIZE,
            "alternative": "greater",
            "n_seeds": 8,
            "results": family_a,
        },
        "family_b": {
            "name": "hypernet_beats_belief",
            "size": FAMILY_B_SIZE,
            "alternative": "greater",
            "n_seeds": 5,
            "results": family_b,
        },
        "family_c": {
            "name": "posterior_performance_decoupling",
            "decoupling_threshold": decoupling_threshold,
            "overall": overall,
            "per_method": per_method_corr,
            "decoupling_supported": decoupling_supported,
        },
        "pass_summary": pass_summary,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "elapsed_min": round(total_min, 2),
            "family_a_supported": f"{a_supported}/{FAMILY_A_SIZE}",
            "family_b_supported": f"{b_supported}/{FAMILY_B_SIZE}",
            "decoupling_supported": decoupling_supported,
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print("[m6_tests] === Family A: hypernet > concat ===", flush=True)
    for hyp_name, hyp in family_a.items():
        if "error" in hyp:
            print(f"[m6_tests]   {hyp_name}: {hyp['error']}", flush=True)
            continue
        loo = hyp.get("loo", {})
        print(
            f"[m6_tests]   {hyp_name:>50s} | "
            f"Δmedian={hyp['median_paired_delta']:+7.2f} "
            f"CI=[{hyp['delta_ci'][0]:+6.2f},{hyp['delta_ci'][1]:+6.2f}] | "
            f"p_corr={hyp['holm_corrected_p']:.4f} | "
            f"sup={hyp['supported']} loo_robust={loo.get('robust_to_loo')}",
            flush=True,
        )
    print("[m6_tests] === Family B: hypernet > Belief-PPO ===", flush=True)
    for hyp_name, hyp in family_b.items():
        if "error" in hyp:
            print(f"[m6_tests]   {hyp_name}: {hyp['error']}", flush=True)
            continue
        loo = hyp.get("loo", {})
        print(
            f"[m6_tests]   {hyp_name:>50s} | "
            f"Δmedian={hyp['median_paired_delta']:+7.2f} "
            f"CI=[{hyp['delta_ci'][0]:+6.2f},{hyp['delta_ci'][1]:+6.2f}] | "
            f"p_corr={hyp['holm_corrected_p']:.4f} | "
            f"sup={hyp['supported']} loo_robust={loo.get('robust_to_loo')}",
            flush=True,
        )
    print("[m6_tests] === Family C: correlation CI ===", flush=True)
    print(
        f"[m6_tests]   overall: r={overall['correlation']:+.3f} "
        f"CI=[{overall['ci'][0]:+.3f},{overall['ci'][1]:+.3f}] "
        f"n={overall['n']}",
        flush=True,
    )
    for method, c in per_method_corr.items():
        print(
            f"[m6_tests]   {method:>22s}: r={c['correlation']:+.3f} "
            f"CI=[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] n={c['n']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
