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
    if source == "returns_rsmm":
        payload = payload.get("family_b", {})
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
        for field in ("mean_difference", "delta_ci", "permutation_p", "holm_corrected_p"):
            try:
                value = np.asarray(comp[field], dtype=float)
                valid = np.all(np.isfinite(value))
                if field == "delta_ci":
                    valid = valid and value.shape == (2,) and value[0] <= value[1]
                elif field.endswith("_p"):
                    valid = valid and value.shape == () and 0 <= value <= 1
                else:
                    valid = valid and value.shape == ()
                if not valid:
                    raise ValueError(field)
            except (KeyError, TypeError, ValueError):
                findings.append({"kind": "invalid_comparison_value", "source": source, "field": field})
    expected = P.COMPARISON_SETS[source].size
    if checked != expected:
        findings.append({"kind": "comparison_count", "source": source,
                         "expected": expected, "found": checked})
    return checked


def _signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _thesis_table_text(thesis_root: Path, label: str) -> str:
    """Locate a labelled table independently of its chapter or appendix."""
    import re
    matches = []
    for source in sorted((thesis_root / "sections").glob("*.tex")):
        for block in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source.read_text(), re.S):
            if r"\label{" + label + "}" in block:
                matches.append(" ".join(block.split()))
    # Ambiguous or absent tables must fail the row check.
    return matches[0] if len(matches) == 1 else ""


def _check_thesis_result_tables(thesis_root: Path, root: Path, findings: list[dict]) -> None:
    """Check manually typeset result rows against their analysis artifacts."""
    results_path = thesis_root / "sections" / "results.tex"
    try:
        text = results_path.read_text()
    except OSError:
        findings.append({"kind": "missing_thesis_source", "path": str(results_path)})
        return
    normalised = _thesis_table_text(thesis_root, "tab:stacked_baseline")

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
    if post:
        expected_fragments["reference_gaps"] = []
        for high, low, label in (("belief_ppo", "regime_agnostic_ppo", "Belief-PPO $-$ regime-agnostic PPO"),
                                 ("oracle_ppo", "belief_ppo", "Oracle-PPO $-$ Belief-PPO")):
            difference, lower, upper = bootstrap_independent_mean_ci(
                post["methods"][high]["per_seed_evaluation_return"],
                post["methods"][low]["per_seed_evaluation_return"])
            expected_fragments["reference_gaps"].append(
                f"{label} & ${difference:+.2f}$ & $[{lower:+.2f},\\ {upper:+.2f}]$")
    # Main architecture/method effect tables, not only the supplementary rows.
    if hypotheses and post:
        expected_fragments["architecture_returns"] = []
        for r in hypotheses["family_b"]["results"].values():
            label = "RL\\textsuperscript{2}" if r["method"].startswith("rl2") else "VariBAD"
            lo, hi = r["delta_ci"]
            pv = r["holm_corrected_p"]
            ptext = "<0.001" if pv < .001 else f"{pv:.3f}"
            expected_fragments["architecture_returns"].append(
                f"{label} & ${r['mean_difference']:+.2f}$ & $[{lo:+.2f},\\ {hi:+.2f}]$ & "
                f"${ptext}$")
    method_tests = _load(root / "analysis/m5r_method_return_tests.json")
    if method_tests and post:
        expected_fragments["method_returns"] = []
        for r in method_tests["comparisons"]:
            label = "Concatenation" if r["architecture"] == "concat" else "Hypernetwork"
            lo, hi = r["delta_ci"]
            pv = r["holm_corrected_p"]
            ptext = "<0.001" if pv < .001 else f"{pv:.3f}"
            expected_fragments["method_returns"].append(
                f"{label} & ${r['mean_difference']:+.2f}$ & $[{lo:+.2f},\\ {hi:+.2f}]$ & "
                f"${ptext}$")
    table_labels = {"learning_speed": "tab:learning_speed", "diagnostics": "tab:diagnostics",
                    "decoupling": "tab:decoupling", "reference_gaps": "tab:reference_gaps",
                    "architecture_returns": "tab:integration_gap", "method_returns": "tab:method_effect"}
    for table, fragments in expected_fragments.items():
        table_text = _thesis_table_text(thesis_root, table_labels[table])
        for fragment in fragments:
            if " ".join(fragment.split()) not in table_text:
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
    if post:
        for key, display in labels.items():
            result = post["methods"][key]
            mean = result["evaluation_return_mean"]
            sd = np.std(result["per_seed_evaluation_return"], ddof=1)
            fragment = f"{display} & ${mean:.2f}$ & ${sd:.2f}$ & ${(mean-floor)/gap:.2f}$"
            if fragment not in appendix:
                findings.append({"kind":"thesis_table_drift", "table":"variant_ladder", "row":key})
            if key in ("regime_agnostic_ppo", "stacked_obs_ppo", "belief_ppo", "oracle_ppo"):
                lo,hi = result["evaluation_return_ci95"]
                fragment = f"{display} & ${mean:.2f}$ & ${sd:.2f}$ & $[{lo:.2f},\\ {hi:.2f}]$"
                if fragment not in appendix:
                    findings.append({"kind":"thesis_table_drift", "table":"reference_levels", "row":key})
    for fragment in ("& $\\geq 0.85$ & $0.964$ & Pass",):
        if fragment not in _thesis_table_text(thesis_root, "tab:validation_requirements"):
            findings.append({"kind": "thesis_table_drift", "table": "environment_validation",
                             "expected_fragment": fragment})


def _audit(root: Path, thesis_root: Path | None = None) -> dict:
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
        metadata = payload.get("family_b", {}) if key == "returns_rsmm" else payload
        size = metadata.get("family_size", metadata.get("size"))
        if size != expected.size:
            findings.append({
                "kind": "family_size", "set": key,
                "expected": expected.size, "found": size,
                "note": "Appendix D records the expected size",
            })
        alt = metadata.get("alternative")
        if alt != P.ALTERNATIVE:
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

    # Every published condition, using the actual routed medium directory.
    for experiment in EXPERIMENTS.values():
        m = _load(root / "medium" / experiment / "metrics.json")
        if not isinstance(m, dict):
            findings.append({"kind": "missing_artifact", "experiment": experiment})
            continue
        for field, expected in (("num_seeds", P.SEEDS), ("iterations", P.ITERATIONS),
                                ("parallel_envs", P.PARALLEL_ENVS),
                                ("rollout_length", P.ROLLOUT_LENGTH)):
            if m.get(field) != expected:
                findings.append({"kind": "budget", "experiment": experiment,
                                 "field": field, "expected": expected, "found": m.get(field)})
        try:
            curves = np.asarray(m["per_seed_mean_return_per_iter"], dtype=float)
            assert curves.shape == (P.SEEDS, P.ITERATIONS) and np.isfinite(curves).all()
            ids = m.get("seeds", list(range(P.SEEDS)))  # historical train() appended in numeric order
            assert ids == list(range(P.SEEDS))
        except (KeyError, TypeError, ValueError, AssertionError):
            findings.append({"kind": "training_curves", "experiment": experiment})

    if thesis_root is not None:
        _check_thesis_result_tables(thesis_root, root, findings)
        from scripts import make_tables
        previous = make_tables.FINAL
        make_tables.FINAL = lambda: root / "analysis"
        try:
            for name, builder in make_tables.TABLES.items():
                expected = builder()
                path = thesis_root / "tables" / f"{name}.tex"
                if expected is None or not path.exists() or path.read_text() != expected:
                    findings.append({"kind": "generated_table_drift", "table": name})
        finally:
            make_tables.FINAL = previous

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



EXPERIMENTS = {
    "regime_agnostic_ppo": "m5r_ref_regime_agnostic_e9",
    "belief_ppo": "m5r_ref_belief_e9", "oracle_ppo": "m5r_ref_oracle_e9",
    "stacked_obs_ppo": "m5r_ref_stacked_obs_e9",
    **{m: f"m5r_final_{m}_e9" for m in P.METHOD_ARMS},
}


def _required_inputs(root: Path) -> list[dict]:
    findings = []
    def require(relative):
        value = _load(root / relative)
        if not isinstance(value, dict) or not value:
            findings.append({"kind": "missing_or_corrupt_artifact", "path": relative})
            return {}
        return value
    def check(label, function):
        try:
            assert function()
        except (KeyError, TypeError, ValueError, IndexError, AssertionError):
            findings.append({"kind": "invalid_schema", "source": label})
    def finite(values, shape):
        array = np.asarray(values, dtype=float)
        return array.shape == shape and np.isfinite(array).all()
    ev = require("analysis/m5r_post_training_evaluation.json")
    check("evaluation methods", lambda: set(ev["methods"]) == set(EXPERIMENTS))
    for method, experiment in EXPERIMENTS.items():
        def validate(method=method, experiment=experiment):
            b = ev["methods"][method]
            return (b["experiment"] == experiment and sorted(b["seeds"]) == list(range(P.SEEDS))
                    and b["n_trained_runs"] == P.SEEDS
                    and b["evaluation_episodes_per_run"] == 512 and b["episode_length"] == 128
                    and finite(b["per_seed_evaluation_return"], (P.SEEDS,))
                    and np.isclose(np.mean(b["per_seed_evaluation_return"]), b["evaluation_return_mean"]))
        check("evaluation:" + method, validate)
    for suffix, classifier in (("", P.PRIMARY_PROBE), ("_mlp", P.ROBUSTNESS_PROBE)):
        data = require(f"analysis/m5r_posterior_vs_performance{suffix}.json")
        def validate_probe(data=data, classifier=classifier):
            points = data["scatter_points"]
            expected = {(m, seed) for m in P.METHOD_ARMS for seed in range(P.SEEDS)}
            fields = (P.PRIMARY_BELIEF_METRIC, P.DECODABILITY_METRIC,
                      *P.UNCORRECTED_PROBE_METRICS, "belief_error_kl", "gap_closed")
            return (data["classifier"] == classifier and data["n_rollouts"] * data["rollout_length"] == P.PROBE_TIMESTEPS
                    and len(points) == len(expected)
                    and {(p["method"], p["seed"]) for p in points} == expected
                    and all(finite([p[k] for k in fields], (len(fields),)) for p in points))
        check("probe" + suffix, validate_probe)
    for name in ("m5r_belief_swap", "m5r_belief_swap_with_history", "m5r_belief_swap_history_only", "m5r_action_distributions"):
        if name in ("m5r_belief_swap_with_history", "m5r_belief_swap_history_only") and not (root / f"analysis/{name}.json").exists():
            continue  # optional, unreported robustness interventions
        data = require(f"analysis/{name}.json")
        for method in P.METHOD_ARMS:
            def validate(data=data, method=method, name=name):
                b = data["by_method"][method]
                if sorted(b["seeds"]) != list(range(P.SEEDS)):
                    return False
                if "swap" in name:
                    return b["diagnostic_version"] == 2 and finite(b["per_seed_separation"], (P.SEEDS,))
                return (finite(b["per_seed_action_given_regime_inventory"], (P.SEEDS, 3, 11, 3))
                        and finite(b["per_seed_inventory_counts"], (P.SEEDS, 3, 11)))
            check(name + ":" + method, validate)
    sensitivity = require("analysis/m5r_seed_block_sensitivity.json")
    check("seed-block sensitivity", lambda: len(sensitivity["comparisons"]) == 18
          and sensitivity["seed_ids"] == list(range(P.SEEDS))
          and all(finite(r["paired_bootstrap_ci"], (2,)) and 0 <= r["paired_p_holm"] <= 1
                  for r in sensitivity["comparisons"]))
    confusion = require("analysis/m5r_probe_confusion.json")
    for method in P.METHOD_ARMS:
        check("confusion:" + method, lambda m=method: finite(confusion["by_method"][m]["row_normalized"], (3,3))
              and np.allclose(np.sum(confusion["by_method"][m]["row_normalized"], axis=1), 1))
    for relative in ("analysis/per_cell_env.json", "analysis/param_counts.json",
                     "foundations/env_validation/e9_stats.json", "foundations/env_validation/validation_table.json",
                     "foundations/stats_M5_factorial_toys.json", "foundations/method_ranking.json"):
        require(relative)
    return findings


def audit(root: Path, thesis_root: Path | None = None) -> dict:
    findings = _required_inputs(root)
    try:
        report = _audit(root, thesis_root)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        report = {"comparisons_checked": 0, "findings": [{"kind": "invalid_artifact_schema", "error": str(exc)}]}
    report["findings"] = findings + report["findings"]
    report["n_findings"] = len(report["findings"])
    return report

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
