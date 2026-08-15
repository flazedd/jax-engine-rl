"""Cartpole external-validity probe — Family A hypothesis test.

Family A: hypernet versus concat at each difficulty cell, two-sided paired
Wilcoxon, Holm-corrected within the level's comparison set. Runs through
`evaluation.metrics` so the second domain uses the same statistical protocol as
the RSMM programme rather than a private copy of it.

Writes:
  results/milestones/cartpole/stats_cartpole_hypothesis_tests.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from utils.paths import cartpole_dir, experiment_dir

import numpy as np

from evaluation.metrics import (
    bootstrap_paired_mean_ci,
    holm_bonferroni,
    leave_one_out_sensitivity,
    paired_wilcoxon,
    rank_biserial,
)
from scripts.cartpole_names import cartpole_experiment_name
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _load_seeds(method: str, level: str = "medium", axis: str = "asymmetry") -> np.ndarray:
    name = cartpole_experiment_name(method, axis, level)
    p = experiment_dir(name) / "metrics.json"
    with open(p) as f:
        m = json.load(f)
    return np.asarray(m["per_seed_final_return"], dtype=float)


def _bootstrap_median_ci(diffs: np.ndarray, n_boot: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    boot = np.array([
        np.median(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _loo(hyp: np.ndarray, conc: np.ndarray, family_size: int) -> bool:
    """Stability under seed omission, via the shared protocol helper."""
    return bool(leave_one_out_sensitivity(
        hyp, conc, n_corrections=family_size, alternative="two-sided",
    )["stable_under_seed_omission"])


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="scripts.cartpole_hypothesis_tests")
    parser.add_argument(
        "--levels", nargs="+", choices=("easy", "medium", "hard"),
        default=["medium"],
        help="Levels to test. Default: medium only (single-cell Family A). "
             "Pass `--levels easy medium hard` for the difficulty sweep "
             "(family-of-6 Holm correction).",
    )
    parser.add_argument(
        "--axis", choices=("asymmetry", "persistence"), default="asymmetry",
        help="Which difficulty axis the easy / hard levels refer to. "
             "Default 'asymmetry'. 'persistence' looks up the persistence-"
             "axis cells.",
    )
    args = parser.parse_args()

    run = ScriptRun(script="cartpole_hypothesis_tests")
    out_dir = cartpole_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    is_sweep = set(args.levels) != {"medium"}
    sweep_suffix = "_sweep" if is_sweep else ""
    axis_suffix = "" if args.axis == "asymmetry" else f"_{args.axis}"
    stats_path = out_dir / f"stats_cartpole_hypothesis_tests{sweep_suffix}{axis_suffix}.json"
    summary_path = out_dir / f"stats_cartpole_hypothesis_tests{sweep_suffix}{axis_suffix}_run.json"

    t0 = time.perf_counter()

    # Per-level reference + method per-seed returns.
    references_by_level: dict[str, dict[str, np.ndarray]] = {}
    methods_by_level: dict[str, dict[str, np.ndarray]] = {}
    for level in args.levels:
        references_by_level[level] = {
            m: _load_seeds(m, level, args.axis)
            for m in ("regime_agnostic", "belief", "oracle")
        }
        methods_by_level[level] = {
            m: _load_seeds(m, level, args.axis)
            for m in ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
        }

    # Family A: hypernet > concat at each (method × level) cell.
    hypotheses: list[tuple[str, str, str, str]] = []
    for level in args.levels:
        for hyp, conc in [
            ("rl2_hypernet", "rl2_concat"),
            ("varibad_hypernet", "varibad_concat"),
        ]:
            hypotheses.append((f"{hyp}_beats_concat_{level}", hyp, conc, level))

    raw = []
    for name, hyp, conc, level in hypotheses:
        h_vals = methods_by_level[level][hyp]
        c_vals = methods_by_level[level][conc]
        diffs = h_vals - c_vals
        wil = paired_wilcoxon(h_vals, c_vals, alternative="two-sided")
        mean_delta, ci_low, ci_high = bootstrap_paired_mean_ci(h_vals, c_vals)
        raw.append({
            "name": name,
            "level": level,
            "p_raw": wil["p"],
            "wilcoxon_null_distribution": wil["null_distribution"],
            "delta_median": float(np.median(diffs)),
            "delta_mean": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "rank_biserial": rank_biserial(h_vals, c_vals),
            "n_pos": int((diffs > 0).sum()),
            "n_total": int(len(diffs)),
            "diffs": diffs.tolist(),
            "_pair": (h_vals, c_vals),
        })

    m = len(raw)
    for idx, p_holm in enumerate(holm_bonferroni([r["p_raw"] for r in raw])):
        raw[idx]["p_holm"] = p_holm
        raw[idx]["supported"] = bool(p_holm < 0.05)
    for r in raw:
        h_vals, c_vals = r.pop("_pair")
        r["loo_robust"] = _loo(h_vals, c_vals, m)
        r["stable_under_seed_omission"] = r["loo_robust"]

    n_supported = sum(1 for r in raw if r["supported"])
    n_loo_robust = sum(1 for r in raw if r["loo_robust"])

    payload = {
        "env": "cartpole_regime_v1",
        "levels": list(args.levels),
        "n_seeds": int(next(iter(methods_by_level[args.levels[0]].values())).shape[0]),
        "reference_means": {
            level: {k: float(v.mean()) for k, v in refs.items()}
            for level, refs in references_by_level.items()
        },
        "method_means": {
            level: {k: float(v.mean()) for k, v in mths.items()}
            for level, mths in methods_by_level.items()
        },
        "family_a": {
            "hypotheses": raw,
            "n_supported": n_supported,
            "n_loo_robust": n_loo_robust,
            "n_total": len(raw),
        },
    }
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))

    elapsed = (time.perf_counter() - t0) / 60
    run.ok(
        key_stats={
            "elapsed_min": round(elapsed, 3),
            "family_a_supported": f"{n_supported}/{len(raw)}",
            "family_a_loo_robust": f"{n_loo_robust}/{len(raw)}",
        },
        summary_path=summary_path,
    )

    print(
        f"[cartpole_tests] Family A: {n_supported}/{len(raw)} supported, "
        f"{n_loo_robust}/{len(raw)} LOO-robust",
        flush=True,
    )
    for r in raw:
        print(
            f"[cartpole_tests]   {r['name']:45s} | "
            f"Δmean={r['delta_mean']:+.2f} "
            f"CI=[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | "
            f"n_pos={r['n_pos']}/{r['n_total']} | "
            f"p_raw={r['p_raw']:.4f} p_holm={r['p_holm']:.4f} | "
            f"sup={r['supported']} loo={r['loo_robust']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
