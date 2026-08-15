"""The prerequisite check Chapter 4 runs before any variant is interpreted.

Every claim in the thesis is read against the reference scale, and the
gap-closed fraction divides by `oracle - agnostic`. If that ordering does not
hold on an instance, the denominator is at or below zero and the normalised
metric is undefined or misleading. Chapter 4 states the check as: the lower
bootstrap confidence bound on each adjacent paired difference is positive,
which is stronger than eyeballing two separate intervals.

It runs early and it is allowed to fail. A failure here means the instance is
excluded from normalised comparisons, not that the programme is broken, so the
gate reports per-instance rather than aborting everything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from evaluation.metrics import bootstrap_paired_mean_ci
from evaluation import protocol as P
from utils.paths import analysis_dir, experiment_dir, resolve_data, results_root
from utils.script_output import ScriptRun

# Adjacent pairs of the ordering, in the direction the thesis asserts.
ADJACENT = [
    ("belief_over_agnostic", "m5r_ref_belief_e9", "m5r_ref_regime_agnostic_e9"),
    ("oracle_over_belief", "m5r_ref_oracle_e9", "m5r_ref_belief_e9"),
]


def _per_seed(experiment: str) -> np.ndarray | None:
    p = resolve_data(experiment_dir(experiment) / "metrics.json")
    if not p.exists():
        return None
    with open(p) as f:
        return np.asarray(json.load(f)["per_seed_final_return"], dtype=float)


def main() -> int:
    run = ScriptRun(script="reference_ordering_gate")
    checks = []
    for name, hi_exp, lo_exp in ADJACENT:
        hi, lo = _per_seed(hi_exp), _per_seed(lo_exp)
        if hi is None or lo is None:
            run.fail(reason=f"missing reference metrics for {name}")
            return 1
        if hi.shape != lo.shape:
            run.fail(reason=f"{name}: seed counts differ ({hi.size} vs {lo.size})")
            return 1
        mean, ci_lo, ci_hi = bootstrap_paired_mean_ci(
            hi, lo, n_boot=P.BOOTSTRAP_RESAMPLES, alpha=1 - P.INTERVAL_LEVEL
        )
        passed = bool(ci_lo > 0)
        checks.append({
            "pair": name, "higher": hi_exp, "lower": lo_exp,
            "mean_paired_delta": mean, "delta_ci": [ci_lo, ci_hi],
            "n_pairs": int(hi.size), "lower_bound_positive": passed,
        })
        print(f"[ref_gate] {name}: d_bar={mean:+.2f} "
              f"CI[{ci_lo:+.2f},{ci_hi:+.2f}] pass={passed}", flush=True)

    all_pass = all(c["lower_bound_positive"] for c in checks)
    payload = {
        "gate": "reference_ordering",
        "ordering": "regime-agnostic < Belief-PPO < Oracle-PPO",
        "criterion": "lower bootstrap bound on each adjacent paired difference "
                     "is positive",
        "checks": checks,
        "ordering_holds": all_pass,
        "consequence_if_failed": "instance excluded from normalised "
                                 "(gap-closed) comparisons; raw returns remain "
                                 "reportable",
    }
    out_dir = analysis_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reference_ordering_gate.json"
    out_path.write_text(json.dumps(payload, indent=2))
    run.add_output(str(out_path))

    if not all_pass:
        run.fail(reason="reference ordering does not hold; gap-closed fraction "
                        "is not interpretable on this instance",
                 summary_path=out_dir / "reference_ordering_gate_run.json")
        return 1
    run.ok(key_stats={"ordering_holds": True, "n_checks": len(checks)},
           summary_path=out_dir / "reference_ordering_gate_run.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
