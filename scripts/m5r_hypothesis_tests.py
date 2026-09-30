"""Return and belief analyses for the final RSMM experiment.

Three families, mapped to the three research questions:

Family A records the descriptive ordering of the three trained references on
  fresh evaluation episodes in the selected environment.

Family B compares hypernetwork and concatenation conditioning for RL² and
  VariBAD. It uses the frozen policies' returns on fresh evaluation episodes,
  an independent bootstrap interval for the mean difference, and a two-sided
  random-label permutation test. The two p-values are adjusted together with
  the Holm procedure.

Family C correlates excess forward KL with the Belief-PPO reference-gap
fraction from fresh evaluation returns joined by seed. It reports pooled and
within-variant relationships, with independent stratified run bootstraps.

Outputs:
  results/M5R/final/m5r_hypothesis_tests.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from utils.paths import analysis_dir
from typing import Any

import numpy as np

from evaluation.metrics import (
    bootstrap_independent_mean_ci,
    holm_bonferroni,
    leave_one_run_out_independent_sensitivity,
    permutation_mean_test,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FINAL_DIR = analysis_dir()
PER_CELL_ENV_PATH = FINAL_DIR / "per_cell_env.json"
PROBE_LOGISTIC_PATH = FINAL_DIR / "m5r_posterior_vs_performance.json"
PROBE_MLP_PATH = FINAL_DIR / "m5r_posterior_vs_performance_mlp.json"
POST_TRAINING_EVALUATION_PATH = FINAL_DIR / "m5r_post_training_evaluation.json"

ALPHA = 0.05
N_BOOT = 10_000
DECOUPLING_THRESHOLD = 0.30

# The medium env plus the three levels `scripts.sweep_redesign_n20` produces.
# Keep in sync with that script's ENVS; see the Family B note in the module
# docstring for why a mismatch silently corrupts the corrected p-values.
# RQ3's difficulty sweep is deferred: its instances were defined as
# perturbations of the superseded environment and need redefining. Only the
# medium instance is run, so Family B is the two conditioning comparisons
# rather than two methods over four instances.
from evaluation.protocol import MEDIUM_ENV, SEEDS

ENVS = (
    MEDIUM_ENV,
)
METHODS = ("rl2", "varibad")
FAMILY_B: list[tuple[str, str, str, str]] = []
for method in METHODS:
    for env in ENVS:
        hyp_name = f"{method}_hypernet_beats_concat_{env}"
        FAMILY_B.append((
            hyp_name,
            f"{method}_hypernet",
            f"{method}_concat",
            env,
        ))
FAMILY_B_SIZE = len(FAMILY_B)


def _bootstrap_correlation_ci(
    xs: np.ndarray, ys: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA,
) -> dict[str, Any]:
    valid = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[valid], ys[valid]
    n = xs.size
    if n < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return {"n": int(n), "correlation": float("nan"),
                "ci": [float("nan"), float("nan")]}
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
    return {
        "n": int(n),
        "correlation": r_hat,
        "ci": [
            float(np.percentile(boot_rs, 100 * alpha / 2)),
            float(np.percentile(boot_rs, 100 * (1 - alpha / 2))),
        ],
    }


def _bootstrap_ci_mean(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(N_BOOT, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return (
        float(np.percentile(boot_means, 100 * ALPHA / 2)),
        float(np.percentile(boot_means, 100 * (1 - ALPHA / 2))),
    )


def _check_family_a(methods: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reference ordering from the same fresh evaluation used for final returns."""
    floor = methods["regime_agnostic_ppo"]["evaluation_return_mean"]
    belief = methods["belief_ppo"]["evaluation_return_mean"]
    oracle = methods["oracle_ppo"]["evaluation_return_mean"]
    return {MEDIUM_ENV: {"floor": floor, "belief": belief, "oracle": oracle,
            "ordering_holds": bool(floor < belief <= oracle),
            "compromise_cost": belief-floor, "inference_cost": oracle-belief,
            "supported": bool(floor < belief <= oracle),
            "source": "fresh post-training evaluation; descriptive ordering"}}


def _per_seed_evaluation(methods: dict[str, Any], name: str) -> list[float]:
    return list(map(float, methods[name].get("per_seed_evaluation_return", [])))


def _run_family_b(methods: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for hyp_name, method_cell, baseline_cell, env in FAMILY_B:
        if method_cell not in methods or baseline_cell not in methods:
            out[hyp_name] = {"error": "missing cell"}
            continue
        method_seeds = _per_seed_evaluation(methods, method_cell)
        baseline_seeds = _per_seed_evaluation(methods, baseline_cell)
        if not method_seeds or not baseline_seeds:
            out[hyp_name] = {"error": "no evaluation returns"}
            continue
        mean, lo, hi = bootstrap_independent_mean_ci(
            method_seeds, baseline_seeds, n_boot=N_BOOT, alpha=ALPHA,
        )
        permutation = permutation_mean_test(method_seeds, baseline_seeds)
        loo = leave_one_run_out_independent_sensitivity(
            method_seeds, baseline_seeds, alpha=ALPHA,
            n_corrections=FAMILY_B_SIZE,
        )
        result = {
            "mean_difference": mean,
            "mean_paired_delta": mean,
            "delta_ci": [lo, hi],
            "permutation_p": permutation["p"],
            "wilcoxon_p": permutation["p"],
            "null_distribution": permutation["null_distribution"],
            "n_permutations": permutation["n_permutations"],
            "loo": loo,
            "stable_under_seed_omission": loo["stable_under_run_omission"],
            "env": env,
            "method": method_cell, "baseline": baseline_cell,
            "n_method": len(method_seeds),
            "n_baseline": len(baseline_seeds),
        }
        out[hyp_name] = result
        ordered.append(result)

    corrected = holm_bonferroni([result["permutation_p"] for result in ordered])
    for result, p_holm in zip(ordered, corrected):
        result["holm_corrected_p"] = p_holm
        lo, hi = result["delta_ci"]
        result["supported"] = bool(p_holm < ALPHA and (lo > 0 or hi < 0))
    return out


def _stratified_correlation(points: list[dict[str, Any]], error_key: str,
                            centre: bool = False, n_boot: int = N_BOOT,
                            alpha: float = ALPHA) -> dict[str, Any]:
    """Resample independent runs within each variant, keeping each x/y pair.

    Numeric seed labels do not pair independently trained architectures.
    Re-estimate variant means in every centred bootstrap sample.
    """
    groups = [np.asarray([(p[error_key], p["gap_closed"]) for p in points
                          if p["method"] == method], dtype=float)
              for method in sorted({p["method"] for p in points})]
    def correlation(samples):
        if centre:
            samples = [a - a.mean(axis=0) for a in samples]
        a = np.concatenate(samples)
        if np.any(a.std(axis=0) == 0):
            return float("nan")
        return float(np.corrcoef(a.T)[0, 1])
    rng = np.random.default_rng(0)
    boot = [correlation([g[rng.integers(len(g), size=len(g))] for g in groups])
            for _ in range(n_boot)]
    r = correlation(groups)
    lo, hi = np.nanpercentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return {"n": sum(map(len, groups)), "n_variants": len(groups),
            "n_seeds": min(map(len, groups)), "correlation": r,
            "r_squared": r*r, "ci": [float(lo), float(hi)],
            "bootstrap": ("independent runs within variant; paired x/y"
                          + ("; recentered per resample" if centre else "")),
            "decoupling_supported": bool(lo > -DECOUPLING_THRESHOLD and hi < DECOUPLING_THRESHOLD)}


def _within_variant_correlation(points: list[dict[str, Any]], error_key: str,
                                n_boot: int = N_BOOT, alpha: float = ALPHA) -> dict[str, Any]:
    return _stratified_correlation(points, error_key, centre=True, n_boot=n_boot, alpha=alpha)


def _family_c_from_probe(probe_path: Path) -> dict[str, Any] | None:
    if not probe_path.exists():
        return None
    with open(probe_path) as f:
        probe = json.load(f)
    points = probe.get("scatter_points", [])
    # Belief error is the excess forward KL over the probe's own residual on the
    # analytical posterior, the primary belief metric of Section 3.9.3. This
    # test read `posterior_error`, the accuracy-based measure, which the probe
    # demoted to a secondary metric.
    error_key = ("belief_error_kl" if points and "belief_error_kl" in points[0]
                 else "posterior_error")
    arr_pe = np.asarray([p[error_key] for p in points], dtype=float)
    arr_gc = np.asarray([p["gap_closed"] for p in points], dtype=float)

    def _within(ci) -> bool:
        return (not any(np.isnan(ci))
                and ci[0] > -DECOUPLING_THRESHOLD and ci[1] < DECOUPLING_THRESHOLD)

    overall = _stratified_correlation(points, error_key)
    decoupling_supported = _within(overall["ci"])
    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        by_method.setdefault(p["method"], []).append(p)
    per_method = {}
    for method, pts in by_method.items():
        pe = np.asarray([p[error_key] for p in pts], dtype=float)
        gc = np.asarray([p["gap_closed"] for p in pts], dtype=float)
        ci = _bootstrap_correlation_ci(pe, gc)
        # An equivalence test needs an interval inside the threshold, which at
        # this seed count it cannot reach even when the estimate is near zero.
        # Recording both keeps "not shown to be decoupled" distinct from
        # "shown to be coupled".
        ci["decoupling_supported"] = _within(ci["ci"])
        ci["estimate_within_threshold"] = abs(ci["correlation"]) < DECOUPLING_THRESHOLD
        per_method[method] = ci
    return {
        "classifier": probe.get("classifier", "logistic"),
        "n_scatter_points": probe.get("n_scatter_points"),
        "overall": overall,
        "within_variant": _within_variant_correlation(points, error_key),
        "per_method": per_method,
        "belief_error_metric": error_key,
        "decoupling_supported": decoupling_supported,
        "decoupling_threshold": DECOUPLING_THRESHOLD,
    }


def main() -> int:
    run = ScriptRun(script="m5r_hypothesis_tests")
    out_dir = FINAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "m5r_hypothesis_tests.json"
    summary_path = out_dir / "m5r_hypothesis_tests_run.json"

    if not PER_CELL_ENV_PATH.exists():
        run.fail(reason=f"missing {PER_CELL_ENV_PATH}", summary_path=summary_path)
        return 1
    with open(PER_CELL_ENV_PATH) as f:
        per_cell_env = json.load(f)
    if not POST_TRAINING_EVALUATION_PATH.exists():
        run.fail(reason=f"missing {POST_TRAINING_EVALUATION_PATH}",
                 summary_path=summary_path)
        return 1
    with open(POST_TRAINING_EVALUATION_PATH) as f:
        evaluation_methods = json.load(f).get("methods", {})

    t_start = time.perf_counter()

    # Family A
    print("[m5r_tests] Family A: gap decomposition non-degenerate per env",
          flush=True)
    family_a = _check_family_a(evaluation_methods)
    a_supported = sum(1 for r in family_a.values() if r.get("supported"))
    a_total = len(family_a)
    print(f"[m5r_tests]   {a_supported}/{a_total} supported", flush=True)

    # Family B
    print(f"[m5r_tests] Family B: {FAMILY_B_SIZE} hypotheses (hypernet > concat)",
          flush=True)
    family_b = _run_family_b(evaluation_methods)
    b_supported = sum(1 for h in family_b.values() if h.get("supported"))
    b_loo_robust = sum(
        1 for h in family_b.values()
        if h.get("supported")
        and h.get("loo", {}).get("stable_under_run_omission")
    )
    print(
        f"[m5r_tests]   {b_supported}/{FAMILY_B_SIZE} supported "
        f"(Holm-corrected p < {ALPHA}, CI excludes 0); "
        f"{b_loo_robust} robust to LOO",
        flush=True,
    )

    # Family C
    fam_c_logistic = _family_c_from_probe(PROBE_LOGISTIC_PATH)
    fam_c_mlp = _family_c_from_probe(PROBE_MLP_PATH)
    if fam_c_logistic is not None:
        oc = fam_c_logistic["overall"]
        print(
            f"[m5r_tests] Family C (logistic): r={oc['correlation']:+.3f} "
            f"CI=[{oc['ci'][0]:+.3f},{oc['ci'][1]:+.3f}] n={oc['n']} "
            f"supported={fam_c_logistic['decoupling_supported']}",
            flush=True,
        )
    if fam_c_mlp is not None:
        oc = fam_c_mlp["overall"]
        print(
            f"[m5r_tests] Family C (mlp): r={oc['correlation']:+.3f} "
            f"CI=[{oc['ci'][0]:+.3f},{oc['ci'][1]:+.3f}] n={oc['n']} "
            f"supported={fam_c_mlp['decoupling_supported']}",
            flush=True,
        )

    pass_summary = {
        "family_a_all_supported": a_supported == a_total,
        "family_a_supported_count": a_supported,
        "family_a_total": a_total,
        "family_b_all_supported": b_supported == FAMILY_B_SIZE,
        "family_b_supported_count": b_supported,
        "family_b_total": FAMILY_B_SIZE,
        "family_b_all_loo_robust": (
            b_supported > 0 and b_loo_robust == b_supported
        ),
        "family_c_logistic_supported": (
            fam_c_logistic["decoupling_supported"]
            if fam_c_logistic is not None else None
        ),
        "family_c_mlp_supported": (
            fam_c_mlp["decoupling_supported"]
            if fam_c_mlp is not None else None
        ),
    }

    stats = {
        "alpha": ALPHA,
        "n_boot": N_BOOT,
        "decoupling_threshold": DECOUPLING_THRESHOLD,
        "family_a": {
            "name": "gap_decomposition_non_degenerate",
            "size": a_total,
            "results": family_a,
        },
        "family_b": {
            "name": "hypernet_beats_concat",
            "size": FAMILY_B_SIZE,
            "alternative": "two-sided",
            "n_seeds": SEEDS,
            "results": family_b,
        },
        "family_c": {
            "name": "posterior_performance_decoupling",
            "logistic": fam_c_logistic,
            "mlp": fam_c_mlp,
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
            "family_a_supported": f"{a_supported}/{a_total}",
            "family_b_supported": f"{b_supported}/{FAMILY_B_SIZE}",
            "family_c_logistic": (
                fam_c_logistic["decoupling_supported"]
                if fam_c_logistic is not None else "missing"
            ),
            "family_c_mlp": (
                fam_c_mlp["decoupling_supported"]
                if fam_c_mlp is not None else "missing"
            ),
        },
        summary_path=summary_path,
    )

    print("[m5r_tests] === Family A: gap decomposition per env ===", flush=True)
    for env, info in family_a.items():
        if "error" in info:
            print(f"[m5r_tests]   {env}: {info['error']}", flush=True)
            continue
        print(
            f"[m5r_tests]   {env:>26s} | floor={info['floor']:.1f} "
            f"belief={info['belief']:.1f} oracle={info['oracle']:.1f} "
            f"compromise={info['compromise_cost']:.1f} "
            f"inference={info['inference_cost']:.1f} "
            f"sup={info['supported']}",
            flush=True,
        )
    print("[m5r_tests] === Family B: hypernet > concat ===", flush=True)
    for hyp_name, hyp in family_b.items():
        if "error" in hyp:
            print(f"[m5r_tests]   {hyp_name}: {hyp['error']}", flush=True)
            continue
        loo = hyp.get("loo", {})
        print(
            f"[m5r_tests]   {hyp_name:>50s} | "
            f"Δmean={hyp['mean_paired_delta']:+7.2f} "
            f"CI=[{hyp['delta_ci'][0]:+6.2f},{hyp['delta_ci'][1]:+6.2f}] | "
            f"p_corr={hyp['holm_corrected_p']:.4f} | "
            f"sup={hyp['supported']} "
            f"loo={loo.get('stable_under_run_omission')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
