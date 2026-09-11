"""Check produced artifacts against the protocol the thesis states.

Two failure modes this exists to catch, both of which have already happened
here once:

  * a constant drifts — the bootstrap was on the median while the thesis
    defined the mean, the return tests were one-sided while the thesis said
    two-sided;
  * a required quantity is absent — for example, a comparison omits its mean
    difference, permutation p-value, or run-omission check.

Run it as a gate before the programme starts and again after it finishes:

  uv run python -m scripts.thesis_contract            # audit what exists
  uv run python -m scripts.thesis_contract --strict   # non-zero exit on any gap

It never trains anything, so it costs seconds and can run as often as wanted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation import protocol as P
from utils.paths import results_root
from utils.script_output import ScriptRun


def _load(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _check_comparisons(payload, source: str, findings: list[dict]) -> int:
    """Every corrected comparison carries the fields the thesis reports."""
    checked = 0
    raw = payload.get("comparisons") or payload.get("results") or []
    comparisons = []
    stack = [raw]
    while stack:  # results may be a list, or nested {tag: {name: test}}
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            if "permutation_p" in node or "holm_corrected_p" in node:
                comparisons.append(node)
            else:
                stack.extend(node.values())
    for comp in comparisons:
        # Only comparisons inside a corrected set carry the reported fields.
        if comp.get("in_comparison_set") is False:
            continue
        checked += 1
        missing = [f for f in P.REQUIRED_COMPARISON_FIELDS if f not in comp]
        if missing:
            findings.append({
                "kind": "missing_fields", "source": source,
                "comparison": comp.get("name", "?"), "missing": missing,
            })
    return checked


def audit(root: Path) -> dict:
    findings: list[dict] = []
    # Analyses moved out of the retired M5R/final layout; the audit was
    # looking for its inputs where nothing has been written since.
    final = root / "analysis"
    checked_comparisons = 0

    # --- the corrected sets ------------------------------------------------
    # Every set declared in the protocol, not a subset of them. Two sets had
    # no artifact and were therefore not audited at all, which is exactly the
    # state a contract check exists to make visible.
    set_files = {
        "returns_rsmm": final / "m5r_hypothesis_tests.json",
        "returns_method": final / "m5r_method_return_tests.json",
        "time_to_threshold": final / "m5r_time_to_threshold_tests.json",
        "diagnostics": final / "m5r_diagnostic_tests.json",
        "belief_quality": final / "m5r_belief_quality_tests.json",
        "belief_quality_mlp": final / "m5r_belief_quality_mlp_tests.json",
    }
    undeclared = set(P.COMPARISON_SETS) - set(set_files)
    if undeclared:
        findings.append({"kind": "set_not_audited", "sets": sorted(undeclared),
                         "note": "declared in the protocol but not checked here"})
    for key, path in set_files.items():
        expected = P.COMPARISON_SETS[key]
        payload = _load(path)
        if payload is None:
            findings.append({"kind": "missing_artifact", "set": key,
                             "path": str(path)})
            continue
        size = payload.get("family_size")
        if size is not None and size != expected.size:
            findings.append({
                "kind": "family_size", "set": key,
                "expected": expected.size, "found": size,
                "note": "Appendix D records the expected size",
            })
        alt = payload.get("alternative")
        if alt is not None and alt != P.ALTERNATIVE:
            findings.append({"kind": "alternative", "set": key,
                             "expected": P.ALTERNATIVE, "found": alt})
        checked_comparisons += _check_comparisons(payload, key, findings)

    # --- the probe ---------------------------------------------------------
    probe = _load(final / "m5r_posterior_vs_performance.json")
    if probe is None:
        findings.append({"kind": "missing_artifact", "set": "probe",
                         "path": str(final / "m5r_posterior_vs_performance.json")})
    else:
        # The probe buffer is a protocol constant, and a probe run at a smaller
        # buffer still writes a well-formed artifact. That is exactly how the
        # linear and non-linear probes came to be reported at different sample
        # sizes, so the size is checked rather than assumed.
        n_roll = probe.get("n_rollouts")
        roll_len = probe.get("rollout_length")
        if n_roll is not None and roll_len is not None:
            timesteps = int(n_roll) * int(roll_len)
            if timesteps != P.PROBE_TIMESTEPS:
                findings.append({
                    "kind": "probe_buffer_size",
                    "expected_timesteps": P.PROBE_TIMESTEPS,
                    "found_timesteps": timesteps,
                    "n_rollouts": n_roll, "rollout_length": roll_len,
                    "note": "the probe ran at a buffer the protocol does not specify",
                })
        if probe.get("classifier") != P.PRIMARY_PROBE:
            findings.append({"kind": "probe_classifier",
                             "expected": P.PRIMARY_PROBE,
                             "found": probe.get("classifier")})
        scatter = probe.get("scatter_points") or []
        needed = (P.PRIMARY_BELIEF_METRIC, P.DECODABILITY_METRIC,
                  *P.UNCORRECTED_PROBE_METRICS)
        if scatter:
            absent = [m for m in needed if m not in scatter[0]]
            if absent:
                findings.append({
                    "kind": "probe_metrics_absent", "missing": absent,
                    "note": "the belief-quality set cannot be tested without these",
                })

    # --- the budget --------------------------------------------------------
    for experiment in ("m5r_ref_regime_agnostic_e9", "m5r_final_rl2_hypernet_e9"):
        m = _load(root / experiment / "metrics.json")
        if m is None:
            continue
        for field, expected in (("num_seeds", P.SEEDS),
                                ("iterations", P.ITERATIONS),
                                ("parallel_envs", P.PARALLEL_ENVS),
                                ("rollout_length", P.ROLLOUT_LENGTH)):
            found = m.get(field)
            if found is not None and found != expected:
                findings.append({"kind": "budget", "experiment": experiment,
                                 "field": field, "expected": expected,
                                 "found": found})

    return {
        "protocol": {
            "seeds": P.SEEDS, "bootstrap_resamples": P.BOOTSTRAP_RESAMPLES,
            "alpha": P.ALPHA, "alternative": P.ALTERNATIVE,
            "hypothesis_test": "random-label permutation",
            "correction": P.CORRECTION,
        },
        "comparisons_checked": checked_comparisons,
        "n_findings": len(findings),
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.thesis_contract")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when anything fails the contract")
    args = ap.parse_args()

    run = ScriptRun(script="thesis_contract")
    root = results_root()
    report = audit(root)

    out_dir = root / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "thesis_contract.json"
    out_path.write_text(json.dumps(report, indent=2))
    run.add_output(str(out_path))

    for f in report["findings"]:
        print(f"[contract] {f['kind']}: "
              + ", ".join(f"{k}={v}" for k, v in f.items() if k != "kind"),
              flush=True)

    if args.strict and report["n_findings"]:
        run.fail(reason=f"{report['n_findings']} contract violations",
                 summary_path=out_dir / "thesis_contract_run.json")
        return 1
    run.ok(key_stats={"n_findings": report["n_findings"],
                      "comparisons_checked": report["comparisons_checked"]},
           summary_path=out_dir / "thesis_contract_run.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
