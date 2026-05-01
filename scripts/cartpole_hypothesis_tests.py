"""Cartpole external-validity probe — Family A hypothesis test.

Family A: hypernet > concat at the single difficulty cell, paired
Wilcoxon n=8, Holm-corrected over 2 hypotheses. Same protocol as
M5 Stage A and M6 Family A but on the second-POMDP env.

Writes:
  results/milestones/cartpole/stats_cartpole_hypothesis_tests.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _load_seeds(method: str, level: str = "medium") -> np.ndarray:
    """Medium reuses the historical m_cartpole_<method> dirs; easy /
    hard get level-suffixed names from the difficulty sweep."""
    name = f"m_cartpole_{method}" if level == "medium" else f"m_cartpole_{method}_{level}"
    p = RESULTS_ROOT / name / "metrics.json"
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


def _loo(diffs: np.ndarray) -> bool:
    """Leave-one-out: every n-1 subset must still be Wilcoxon-significant
    at the Holm-corrected level. Mirrors M5/M6 protocol."""
    for i in range(len(diffs)):
        d = np.delete(diffs, i)
        try:
            res = stats.wilcoxon(d, alternative="greater")
        except ValueError:
            return False
        # min Holm-corrected p over family-of-2 at n=7 paired Wilcoxon
        # is 2 / 128 = 0.0156 × 2 = 0.0313 (still passes), but if any LOO
        # subset has p_raw > 0.025 we cannot mark it robust to LOO at α.
        if res.pvalue > 0.025:
            return False
    return True


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
    args = parser.parse_args()

    run = ScriptRun(script="cartpole_hypothesis_tests")
    out_dir = RESULTS_ROOT / "milestones" / "cartpole"
    out_dir.mkdir(parents=True, exist_ok=True)
    is_sweep = set(args.levels) != {"medium"}
    suffix = "_sweep" if is_sweep else ""
    stats_path = out_dir / f"stats_cartpole_hypothesis_tests{suffix}.json"
    summary_path = out_dir / f"stats_cartpole_hypothesis_tests{suffix}_run.json"

    t0 = time.perf_counter()

    # Per-level reference + method per-seed returns.
    references_by_level: dict[str, dict[str, np.ndarray]] = {}
    methods_by_level: dict[str, dict[str, np.ndarray]] = {}
    for level in args.levels:
        references_by_level[level] = {
            m: _load_seeds(m, level)
            for m in ("regime_agnostic", "belief", "oracle")
        }
        methods_by_level[level] = {
            m: _load_seeds(m, level)
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
        diffs = methods_by_level[level][hyp] - methods_by_level[level][conc]
        res = stats.wilcoxon(diffs, alternative="greater")
        ci_low, ci_high = _bootstrap_median_ci(diffs)
        raw.append({
            "name": name,
            "level": level,
            "p_raw": float(res.pvalue),
            "delta_median": float(np.median(diffs)),
            "delta_mean": float(diffs.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_pos": int((diffs > 0).sum()),
            "n_total": int(len(diffs)),
            "loo_robust": bool(_loo(diffs)),
            "diffs": diffs.tolist(),
        })

    sorted_idx = sorted(range(len(raw)), key=lambda i: raw[i]["p_raw"])
    m = len(raw)
    for rank, idx in enumerate(sorted_idx):
        raw[idx]["p_holm"] = min(1.0, raw[idx]["p_raw"] * (m - rank))
        raw[idx]["supported"] = raw[idx]["p_holm"] < 0.05

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
            f"Δmedian={r['delta_median']:+.2f} "
            f"CI=[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | "
            f"n_pos={r['n_pos']}/{r['n_total']} | "
            f"p_raw={r['p_raw']:.4f} p_holm={r['p_holm']:.4f} | "
            f"sup={r['supported']} loo={r['loo_robust']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
