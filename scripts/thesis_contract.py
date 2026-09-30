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

import numpy as np

from evaluation import protocol as P
from evaluation.metrics import bootstrap_independent_mean_ci
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


def _signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _check_thesis_result_tables(thesis_root: Path, root: Path, findings: list[dict]) -> None:
    """Check manually typeset result rows against their analysis artifacts."""
    results_path = thesis_root / "sections" / "results.tex"
    try:
        text = results_path.read_text()
    except OSError:
        findings.append({"kind": "missing_thesis_source", "path": str(results_path)})
        return
    normalised = " ".join(text.split())

    post = _load(root / "analysis" / "m5r_post_training_evaluation.json")
    if post is not None:
        methods = post["methods"]
        stacked = np.asarray(methods["stacked_obs_ppo"]["per_seed_evaluation_return"], dtype=float)
        floor = float(methods["regime_agnostic_ppo"]["evaluation_return_mean"])
        belief = float(methods["belief_ppo"]["evaluation_return_mean"])
        gap = belief - floor
        labels = {
            "Regime-agnostic PPO": "regime_agnostic_ppo",
            "VariBAD concat": "varibad_concat",
            "VariBAD hypernet": "varibad_hypernet",
            "RL\\textsuperscript{2} concat": "rl2_concat",
            "RL\\textsuperscript{2} hypernet": "rl2_hypernet",
        }
        for label, key in labels.items():
            other = np.asarray(methods[key]["per_seed_evaluation_return"], dtype=float)
            mean, lo, hi = bootstrap_independent_mean_ci(stacked, other)
            row = (f"{label} & ${_signed(mean)}$ & $[{_signed(lo)},\\ {_signed(hi)}]$ "
                   f"& ${_signed(mean / gap)}$")
            if row not in normalised:
                findings.append({"kind": "thesis_table_drift", "table": "stacked_baseline",
                                 "row": label, "expected_fragment": row})

    expected_fragments = {"learning_speed": [], "diagnostics": [], "decoupling": []}
    speed = _load(root / "analysis" / "m5r_time_to_threshold_tests.json")
    if speed:
        from decimal import Decimal, ROUND_HALF_UP
        def rounded(value, digits):
            return format(Decimal(str(round(value, 10))).quantize(Decimal(10)**-digits,
                          rounding=ROUND_HALF_UP), f".{digits}f")
        for r in speed["comparisons"]:
            label = "RL\\textsuperscript{2}" if r["method"] == "rl2" else "VariBAD"
            lo,hi = r["delta_ci"]
            expected_fragments["learning_speed"].append(
                f"{label} & ${rounded(r['concat_mean_iterations'],1)}$ & "
                f"${rounded(r['hypernet_mean_iterations'],1)}$ & "
                f"${rounded(r['mean_difference'],1)}$ & $[{lo:.0f},\\ {hi:.0f}]$")
    diagnostics = _load(root / "analysis" / "m5r_diagnostic_tests.json")
    if diagnostics:
        for tag,label in [("locked_regime_action_distribution", "Policy response"),
                          ("belief_swap_belief_only", "Belief substitution")]:
            for method, r in diagnostics["results"][tag].items():
                method_label = "RL\\textsuperscript{2}" if method == "RL2" else method
                lo,hi = r["delta_ci"]
                pvalue = r["holm_corrected_p"]
                ptext = "<0.001" if pvalue < 0.001 else f"{pvalue:.3f}"
                expected_fragments["diagnostics"].append(
                    f"{label} & {method_label} & ${r['mean_difference']:+.3f}$ & "
                    f"$[{lo:+.3f},\\ {hi:+.3f}]$ & ${ptext}$")
    hypotheses = _load(root / "analysis" / "m5r_hypothesis_tests.json")
    if hypotheses:
        for classifier in ("logistic", "mlp"):
            for key in ("overall", "within_variant"):
                r = hypotheses["family_c"][classifier][key]
                lo,hi = r["ci"]
                expected_fragments["decoupling"].append(
                    f"${r['correlation']:.3f}$ & $[{lo:.3f},\\ {hi:.3f}]$")
    for table, fragments in expected_fragments.items():
        for fragment in fragments:
            if " ".join(fragment.split()) not in normalised:
                findings.append({"kind": "thesis_table_drift", "table": table,
                                 "expected_fragment": fragment})

    counts = _load(root / "analysis" / "param_counts.json")
    design_path = thesis_root / "sections" / "experimental_design.tex"
    design = " ".join(design_path.read_text().split()) if design_path.exists() else ""
    if counts is None:
        findings.append({"kind": "missing_artifact", "set": "parameter_counts"})
    else:
        labels = {"rl2_concat": "RL\\textsuperscript{2} concat",
                  "rl2_hypernet": "RL\\textsuperscript{2} hypernet",
                  "varibad_concat": "VariBAD concat", "varibad_hypernet": "VariBAD hypernet",
                  "regime_agnostic_ppo": "Regime-agnostic PPO",
                  "stacked_obs_ppo": "Stacked-observation PPO",
                  "belief_ppo": "Belief-PPO", "oracle_ppo": "Oracle-PPO"}
        for label, display in labels.items():
            row = next((r for r in counts["rows"] if r["label"] == label), {})
            components = row.get("component_counts")
            if components is None:
                findings.append({"kind": "missing_parameter_components", "row": label})
                continue
            def tex_count(n):
                return "$" + f"{n:,}".replace(",", "{,}") + "$" if n else "--"
            values = [components[k] for k in ("encoder", "actor_critic", "decoder")]
            fragment = display + " & " + " & ".join(map(tex_count, values + [row["param_count"]]))
            if sum(values) != row["param_count"] or fragment not in design:
                findings.append({"kind": "thesis_table_drift", "table": "parameter_counts",
                                 "row": label, "expected_fragment": fragment})

    appendix_path = thesis_root / "sections" / "appendix.tex"
    appendix = " ".join(appendix_path.read_text().split()) if appendix_path.exists() else ""
    for fragment in ("& $\\geq 0.85$ & $0.964$ & Pass", "$96.4\\%$ of the exact finite-horizon optimum"):
        if fragment not in appendix:
            findings.append({"kind": "thesis_table_drift", "table": "environment_validation",
                             "expected_fragment": fragment})


def audit(root: Path, thesis_root: Path | None = None) -> dict:
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

    # Explicitly verify the seed join, rather than trusting positional arrays.
    evaluation = _load(final / "m5r_post_training_evaluation.json")
    if evaluation:
        methods = evaluation["methods"]
        floor = methods["regime_agnostic_ppo"]["evaluation_return_mean"]
        gap = methods["belief_ppo"]["evaluation_return_mean"] - floor
        for suffix in ("", "_mlp"):
            data = _load(final / f"m5r_posterior_vs_performance{suffix}.json")
            if not data:
                continue
            for point in data["scatter_points"]:
                result = methods[point["method"]]
                ids = result.get("seeds", [])
                values = result["per_seed_evaluation_return"]
                if len(ids) != len(values) or len(set(ids)) != len(ids):
                    findings.append({"kind":"missing_or_duplicate_evaluation_seeds", "method":point["method"]})
                    break
                by_seed = dict(zip(ids, values))
                expected = (by_seed[point["seed"]]-floor)/gap
                if (point["experiment_name"] != result["experiment"] or
                        not np.isclose(point["gap_closed"],expected,rtol=0,atol=1e-12)):
                    findings.append({"kind":"probe_return_seed_join", "method":point["method"],
                                     "seed":point["seed"], "classifier":suffix or "linear"})
    substitution = _load(final / "m5r_belief_swap.json")
    if substitution and any(v.get("diagnostic_version") != 2
                            for v in substitution["by_method"].values()):
        findings.append({"kind":"stale_belief_substitution", "required_version":2})

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

    if thesis_root is not None:
        _check_thesis_result_tables(thesis_root, root, findings)

    env_stats = _load(root / "foundations" / "env_validation" / "e9_stats.json")
    if env_stats is not None:
        r2 = env_stats.get("R2_per_regime_ppo_vs_vi", {})
        expected_reference = "exact_undiscounted_finite_horizon_optimum"
        if r2.get("reference") != expected_reference:
            findings.append({"kind": "environment_validation_reference",
                             "expected": expected_reference, "found": r2.get("reference")})

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
    ap.add_argument("--thesis-root", type=Path,
                    help="also check manually typeset result tables in this thesis checkout")
    args = ap.parse_args()

    run = ScriptRun(script="thesis_contract")
    root = results_root()
    report = audit(root, args.thesis_root)

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
