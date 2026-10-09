"""Verify and summarise exploratory linear/MLP full-belief probes.

The MLP replay must reproduce the published mean-only scores and the direct
posterior predictions from the linear replay on the same test episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plotting.trading_results import MEDIUM_ENV, _load_probe_for_per_t
from utils.paths import analysis_dir

METHODS = ("varibad_concat", "varibad_hypernet")
INPUTS = ("posterior_mean", "full_policy_input")
METRICS = ("accuracy", "kl_to_analytical")


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    indices = np.random.default_rng(0).integers(len(values), size=(10_000, len(values)))
    return {"mean": float(values.mean()),
            "ci95": np.percentile(values[indices].mean(axis=1), [2.5, 97.5]).tolist()}


def indexed_rows(payload):
    if payload["n_rollouts_per_seed"] != 500:
        raise ValueError("Final comparison requires 500 episodes per run")
    rows = {}
    for row in payload["rows"]:
        if row["method"] not in METHODS:
            continue
        key = row["method"], row["seed"]
        if key in rows or row["n_train_episodes"] != 400 or row["n_test_episodes"] != 100:
            raise ValueError(f"Duplicate run or incompatible split: {key}")
        rows[key] = row
    if set(rows) != {(method, seed) for method in METHODS for seed in range(20)}:
        raise ValueError("Expected all 20 seeds for both VariBAD variants")
    return rows


def summarise(linear, mlp):
    indexed = {"logistic": indexed_rows(linear), "mlp": indexed_rows(mlp)}
    for key, row in indexed["mlp"].items():
        np.testing.assert_allclose(row["direct_posterior_accuracy_by_step"],
                                   indexed["logistic"][key]["direct_posterior_accuracy_by_step"],
                                   rtol=0, atol=1e-12)
    summary = {"status": "exploratory; paired seed bootstrap; no multiplicity correction",
               "bootstrap_resamples": 10_000, "bootstrap_seed": 0, "classifiers": {}}
    for classifier, rows in indexed.items():
        source = _load_probe_for_per_t(classifier)
        points = {(p["method"], p["seed"]): p for p in source["scatter_points"]
                  if p["env_label"] == MEDIUM_ENV and p["method"] in METHODS}
        for key, row in rows.items():
            for metric, source_key in (("accuracy", "method_test_acc"),
                                       ("kl_to_analytical", "method_kl_to_omega")):
                np.testing.assert_allclose(row["posterior_mean"][metric], points[key][source_key],
                                           rtol=0, atol=1e-10)
        values = {m: {inp: {metric: np.array([rows[m, s][inp][metric] for s in range(20)])
                            for metric in METRICS} for inp in INPUTS} for m in METHODS}
        summary["classifiers"][classifier] = {
            "means": {m: {inp: {metric: float(v.mean()) for metric, v in scores.items()}
                           for inp, scores in by_input.items()} for m, by_input in values.items()},
            "full_minus_mean": {m: {metric: paired_interval(values[m][INPUTS[1]][metric]
                                                            - values[m][INPUTS[0]][metric])
                                    for metric in METRICS} for m in METHODS},
            "full_hypernet_minus_concat": {
                metric: paired_interval(values[METHODS[1]][INPUTS[1]][metric]
                                         - values[METHODS[0]][INPUTS[1]][metric])
                for metric in METRICS},
        }
    summary["verification"] = "All 80 mean-only fits reproduce published accuracy and KL; direct posterior curves match across classifier replays"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=Path, default=analysis_dir() / "m5r_supplemental_belief_checks.json")
    parser.add_argument("--mlp", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=analysis_dir() / "m5r_full_belief_summary.json")
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    linear = json.loads(args.linear.read_text())
    mlp = None
    for path in args.mlp:
        part = json.loads(path.read_text())
        if part.get("classifier") != "mlp" or part["n_rollouts_per_seed"] != 500:
            raise ValueError("Expected MLP results from 500-episode replays")
        if mlp is None:
            mlp = dict(part, rows=[])
        mlp["rows"].extend(part["rows"])
    summary = summarise(linear, mlp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.table:
        lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
                 r"& & \multicolumn{2}{c}{Mean only} & \multicolumn{2}{c}{Mean and std. dev.} \\",
                 r"Probe & Variant & Accuracy & KL & Accuracy & KL \\", r"\midrule"]
        for i, classifier in enumerate(("logistic", "mlp")):
            if i:
                lines.append(r"\midrule")
            for method in METHODS:
                cells = ["Linear" if classifier == "logistic" else "MLP",
                         "Concatenation" if method == METHODS[0] else "Hypernetwork"]
                cells += [f"${summary['classifiers'][classifier]['means'][method][inp][metric]:.3f}$"
                          for inp in INPUTS for metric in METRICS]
                lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
