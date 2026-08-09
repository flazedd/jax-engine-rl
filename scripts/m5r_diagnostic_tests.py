"""M5R — paired tests on the two regime-conditioning diagnostics.

Applies the thesis statistical protocol to the hypernet-versus-concat
comparison on each diagnostic:

  - locked-regime action distribution: separation between regimes at a fixed
    inventory, weighted by the minimum per-regime visitation of that inventory
    so that near-empty cells cannot dominate;
  - counterfactual belief swap, belief only and belief plus observation
    history.

The test is TWO-SIDED. The RQ1 protocol pre-registers a one-sided hypernet >
concat alternative for return, but these diagnostics are not return, and the
two methods point in opposite directions here, so a fixed-direction test would
foreclose the question it is meant to answer.

Outputs:
  - results/M5R/final/m5r_diagnostic_tests.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.action_distribution import regime_separation_per_seed
from evaluation.metrics import (
    holm_bonferroni,
    leave_one_out_sensitivity,
    primary_hypothesis_test,
)
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "M5R" / "final"

PAIRS = [
    ("RL2", "rl2_hypernet", "rl2_concat"),
    ("VariBAD", "varibad_hypernet", "varibad_concat"),
]


def _per_seed_locked_separation(method: dict[str, Any]) -> np.ndarray:
    """Per-seed regime separation from the locked-regime tables.

    Definition lives in `evaluation.action_distribution` so the tested value
    and the value the heatmap figure prints are the same statistic.
    """
    return regime_separation_per_seed(
        np.asarray(method["per_seed_action_given_regime_inventory"]),
        np.asarray(method["per_seed_inventory_counts"], dtype=float),
    )


def main() -> int:
    run = ScriptRun(script="m5r_diagnostic_tests")
    t0 = time.perf_counter()

    sources: dict[str, dict[str, np.ndarray]] = {}

    locked = json.loads((RESULTS_ROOT / "m5r_action_distributions.json").read_text())
    sources["locked_regime_action_distribution"] = {
        k: _per_seed_locked_separation(v)
        for k, v in locked["by_method"].items()
        if "per_seed_action_given_regime_inventory" in v
    }

    for tag, fname in [
        ("belief_swap_belief_only", "m5r_belief_swap.json"),
        ("belief_swap_history_only", "m5r_belief_swap_history_only.json"),
        ("belief_swap_both_channels", "m5r_belief_swap_with_history.json"),
    ]:
        path = RESULTS_ROOT / fname
        if not path.exists():
            print(f"[m5r_diag_tests] missing {path}, skipping {tag}", flush=True)
            continue
        payload = json.loads(path.read_text())
        # Guard against consuming a smoke-scale file: every diagnostic entering
        # the family must have been produced at the same rollout budget.
        if payload["n_rollouts"] < 64 or payload["rollout_length"] < 128:
            raise RuntimeError(
                f"{fname} is at reduced scale "
                f"({payload['n_rollouts']}x{payload['rollout_length']}); "
                "rerun it at the full budget before testing"
            )
        sources[tag] = {
            k: np.asarray(v["per_seed_separation"], dtype=float)
            for k, v in payload["by_method"].items()
        }

    # The protocol's diagnostic comparison set is the action separation and the
    # belief-swap diagnostic over both methods, which is four comparisons. The
    # two further belief-swap channels are robustness checks on the same
    # question and are reported without correction, exactly as the extra probe
    # metrics are: correcting them would enlarge the set and cost the primary
    # diagnostics power for tests no claim rests on.
    PRIMARY_TAGS = ("locked_regime_action_distribution", "belief_swap_belief_only")
    corrected = {k: v for k, v in sources.items() if k in PRIMARY_TAGS}
    uncorrected = {k: v for k, v in sources.items() if k not in PRIMARY_TAGS}
    family_size = sum(
        1 for per_seed in corrected.values() for _n, h, c in PAIRS
        if h in per_seed and c in per_seed
    )

    results: dict[str, Any] = {}
    raw_p: list[float] = []
    index: list[tuple[str, str]] = []

    for tag, per_seed in {**corrected, **uncorrected}.items():
        results[tag] = {}
        in_set = tag in corrected
        for name, hyper, concat in PAIRS:
            if hyper not in per_seed or concat not in per_seed:
                continue
            h, c = per_seed[hyper], per_seed[concat]
            if h.shape != c.shape:
                raise ValueError(f"{tag}/{name}: seed counts differ")
            test = primary_hypothesis_test(
                h, c, family_size=family_size, alternative="two-sided",
            )
            loo = leave_one_out_sensitivity(
                h, c, n_corrections=family_size, alternative="two-sided",
            )
            test["leave_one_out"] = loo
            # Promoted to the top level: the protocol reports this property for
            # every claim, and the contract audit looks for it there.
            test["stable_under_seed_omission"] = bool(
                loo["stable_under_seed_omission"]
            )
            test["mean_hypernet"] = float(h.mean())
            test["mean_concat"] = float(c.mean())
            test["direction"] = "hypernet > concat" if h.mean() > c.mean() \
                else "concat > hypernet"
            test["in_comparison_set"] = in_set
            results[tag][name] = test
            if in_set:
                raw_p.append(test["wilcoxon_p"])
                index.append((tag, name))

    # Recompute Holm properly across the whole family rather than the
    # per-test Bonferroni scaling primary_hypothesis_test applies.
    for (tag, name), p_corr in zip(index, holm_bonferroni(raw_p)):
        r = results[tag][name]
        r["holm_corrected_p"] = p_corr
        # The protocol makes the corrected p the inferential decision; the
        # interval states magnitude and precision and is reported beside it.
        lo, hi = r["delta_ci"]
        r["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
        r["supported"] = bool((not np.isnan(p_corr)) and p_corr < 0.05)

    for tag, block in results.items():
        for name, r in block.items():
            print(
                f"[m5r_diag_tests] {tag:34s} {name:8s} "
                f"hyper={r['mean_hypernet']:.3f} concat={r['mean_concat']:.3f} "
                f"delta={r['mean_paired_delta']:+.3f} "
                f"CI[{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}] "
                f"p_holm={r['holm_corrected_p']:.4g} "
                f"supported={r['supported']} ({r['direction']})",
                flush=True,
            )

    stats_path = RESULTS_ROOT / "m5r_diagnostic_tests.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "alternative": "two-sided",
                "comparison_set": "diagnostics",
                "family_size": family_size,
                "uncorrected_robustness_checks": sorted(uncorrected),
                "results": results,
            },
            f, indent=2,
        )
    run.add_output(str(stats_path))
    run.ok(
        key_stats={
            "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
            "family_size": family_size,
        },
        summary_path=RESULTS_ROOT / "m5r_diagnostic_tests_run.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
