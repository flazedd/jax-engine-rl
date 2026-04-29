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


def _load_seeds(method: str) -> np.ndarray:
    p = RESULTS_ROOT / f"m_cartpole_{method}" / "metrics.json"
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
    run = ScriptRun(script="cartpole_hypothesis_tests")
    out_dir = RESULTS_ROOT / "milestones" / "cartpole"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_cartpole_hypothesis_tests.json"
    summary_path = out_dir / "stats_cartpole_hypothesis_tests_run.json"

    t0 = time.perf_counter()

    references = {m: _load_seeds(m) for m in ("regime_agnostic", "belief", "oracle")}
    methods = {
        m: _load_seeds(m) for m in (
            "rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet",
        )
    }

    hypotheses = [
        ("rl2_hypernet_beats_concat", "rl2_hypernet", "rl2_concat"),
        ("varibad_hypernet_beats_concat", "varibad_hypernet", "varibad_concat"),
    ]

    raw = []
    for name, hyp, conc in hypotheses:
        diffs = methods[hyp] - methods[conc]
        res = stats.wilcoxon(diffs, alternative="greater")
        ci_low, ci_high = _bootstrap_median_ci(diffs)
        raw.append({
            "name": name,
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

    # Holm correction over the family.
    sorted_idx = sorted(range(len(raw)), key=lambda i: raw[i]["p_raw"])
    m = len(raw)
    for rank, idx in enumerate(sorted_idx):
        raw[idx]["p_holm"] = min(1.0, raw[idx]["p_raw"] * (m - rank))
        raw[idx]["supported"] = raw[idx]["p_holm"] < 0.05

    n_supported = sum(1 for r in raw if r["supported"])
    n_loo_robust = sum(1 for r in raw if r["loo_robust"])

    payload = {
        "env": "cartpole_regime_v1",
        "n_seeds": int(next(iter(methods.values())).shape[0]),
        "reference_means": {k: float(v.mean()) for k, v in references.items()},
        "method_means": {k: float(v.mean()) for k, v in methods.items()},
        "family_a": {
            "hypotheses": raw,
            "n_supported": n_supported,
            "n_loo_robust": n_loo_robust,
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
            f"[cartpole_tests]   {r['name']:35s} | "
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
