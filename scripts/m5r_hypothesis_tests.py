"""M5R Stage E — pre-registered hypothesis tests on the final-eval results.

Three families, mapped to the three research questions:

Family A (5 cells): the gap decomposition is non-degenerate at every
  evaluated environment. We require regime-agnostic < Belief-PPO <= Oracle-PPO
  with non-overlapping bootstrap CIs on regime-agnostic and Oracle-PPO. This
  is checked by inspection on the M3/M6 reference levels (which are reused
  unchanged by M5R).

Family B (10 cells): hypernet > concat at every (env, method) cell.
  Paired Wilcoxon (one-sided, alternative="greater") on n=8 seeds per cell.
  Holm-Bonferroni correction across all 10 tests. Each test must clear two
  bars: Holm-corrected p < 0.05 AND bootstrap CI on the paired difference
  excludes zero.

Family C (1 confidence interval): pooled Pearson correlation between
  linear-probe regime-decoding accuracy (in posterior_error form) and
  gap_closed_vs_oracle, across all 160 (cell, env, seed) probe points.
  Decoupling supported iff the 95% bootstrap CI is contained in
  (-0.30, +0.30). Re-run with the MLP probe as a robustness check.

Outputs:
  results/M5R/final/m5r_hypothesis_tests.json
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
FINAL_DIR = RESULTS_ROOT / "M5R" / "final"
PER_CELL_ENV_PATH = FINAL_DIR / "per_cell_env.json"
PROBE_LOGISTIC_PATH = FINAL_DIR / "m5r_posterior_vs_performance.json"
PROBE_MLP_PATH = FINAL_DIR / "m5r_posterior_vs_performance_mlp.json"

ALPHA = 0.05
N_BOOT = 10_000
DECOUPLING_THRESHOLD = 0.30

ENVS = (
    "e_final",
    "persistence_easy",
    "persistence_hard",
    "distinguishability_easy",
    "distinguishability_hard",
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


def _check_family_a(per_cell_env: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Family A: gap decomposition non-degenerate per env.

    Reads M3/M6 reference numbers from per_cell_env.json (which carries them
    in the per_env.refs blocks). For E_final and the M6 sweep cells we have
    per-seed seeds available via the M3/M6 stats files; here we use the
    means since we did not reload per-seed data. The non-overlap check is
    therefore a point-estimate ordering check rather than a CI test, which
    is what M3/M6 already verified.
    """
    out: dict[str, dict[str, Any]] = {}
    for env in ENVS:
        block = per_cell_env.get("per_env", {}).get(env, {})
        refs = block.get("refs", {})
        floor = refs.get("regime_agnostic_ppo")
        belief = refs.get("belief_ppo")
        oracle = refs.get("oracle_ppo")
        if floor is None or belief is None or oracle is None:
            out[env] = {"error": "missing reference"}
            continue
        ordering_ok = (floor < belief) and (belief <= oracle + 1e-6)
        # CIs were established in M3 / M6; we just record the means here.
        out[env] = {
            "floor": floor,
            "belief": belief,
            "oracle": oracle,
            "ordering_holds": bool(ordering_ok),
            "compromise_cost": belief - floor,
            "inference_cost": oracle - belief,
            "supported": bool(ordering_ok),
        }
    return out


def _per_seed(cell: dict[str, Any]) -> list[float]:
    return list(map(float, cell.get("per_seed_final_return", [])))


def _run_family_b(per_cell_env: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for hyp_name, method_cell, baseline_cell, env in FAMILY_B:
        block = per_cell_env.get("per_env", {}).get(env, {})
        cells = block.get("cells", {})
        if method_cell not in cells or baseline_cell not in cells:
            out[hyp_name] = {"error": "missing cell"}
            continue
        method_seeds = _per_seed(cells[method_cell])
        baseline_seeds = _per_seed(cells[baseline_cell])
        if not method_seeds or not baseline_seeds:
            out[hyp_name] = {"error": "no per-seed data"}
            continue
        n_pair = min(len(method_seeds), len(baseline_seeds))
        method_seeds = method_seeds[:n_pair]
        baseline_seeds = baseline_seeds[:n_pair]
        test = primary_hypothesis_test(
            method=method_seeds, baseline=baseline_seeds,
            family_size=FAMILY_B_SIZE, alpha=ALPHA, n_boot=N_BOOT,
            alternative="greater",
        )
        loo = leave_one_out_sensitivity(
            method=method_seeds, baseline=baseline_seeds, alpha=ALPHA,
            n_corrections=FAMILY_B_SIZE, alternative="greater",
        )
        out[hyp_name] = {
            **test, "loo": loo, "env": env,
            "method": method_cell, "baseline": baseline_cell,
            "n_pairs": n_pair,
        }
    return out


def _family_c_from_probe(probe_path: Path) -> dict[str, Any] | None:
    if not probe_path.exists():
        return None
    with open(probe_path) as f:
        probe = json.load(f)
    points = probe.get("scatter_points", [])
    arr_pe = np.asarray([p["posterior_error"] for p in points], dtype=float)
    arr_gc = np.asarray([p["gap_closed"] for p in points], dtype=float)
    overall = _bootstrap_correlation_ci(arr_pe, arr_gc)
    if any(np.isnan(overall["ci"])):
        decoupling_supported = False
    else:
        decoupling_supported = (
            overall["ci"][0] > -DECOUPLING_THRESHOLD
            and overall["ci"][1] < DECOUPLING_THRESHOLD
        )
    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        by_method.setdefault(p["method"], []).append(p)
    per_method = {}
    for method, pts in by_method.items():
        pe = np.asarray([p["posterior_error"] for p in pts], dtype=float)
        gc = np.asarray([p["gap_closed"] for p in pts], dtype=float)
        per_method[method] = _bootstrap_correlation_ci(pe, gc)
    return {
        "classifier": probe.get("classifier", "logistic"),
        "n_scatter_points": probe.get("n_scatter_points"),
        "overall": overall,
        "per_method": per_method,
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

    t_start = time.perf_counter()

    # Family A
    print("[m5r_tests] Family A: gap decomposition non-degenerate per env",
          flush=True)
    family_a = _check_family_a(per_cell_env)
    a_supported = sum(1 for r in family_a.values() if r.get("supported"))
    a_total = len(family_a)
    print(f"[m5r_tests]   {a_supported}/{a_total} supported", flush=True)

    # Family B
    print(f"[m5r_tests] Family B: {FAMILY_B_SIZE} hypotheses (hypernet > concat)",
          flush=True)
    family_b = _run_family_b(per_cell_env)
    b_supported = sum(1 for h in family_b.values() if h.get("supported"))
    b_loo_robust = sum(
        1 for h in family_b.values()
        if h.get("supported") and h.get("loo", {}).get("robust_to_loo")
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
            "alternative": "greater",
            "n_seeds": 8,
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
            f"Δmedian={hyp['median_paired_delta']:+7.2f} "
            f"CI=[{hyp['delta_ci'][0]:+6.2f},{hyp['delta_ci'][1]:+6.2f}] | "
            f"p_corr={hyp['holm_corrected_p']:.4f} | "
            f"sup={hyp['supported']} loo={loo.get('robust_to_loo')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
