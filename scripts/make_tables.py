"""Generate the thesis tables whose numbers come from experiments.

Every table in the thesis used to be typed by hand, which has two costs: the
numbers have to be retyped when a run lands, and a table cannot be marked as
synthetic the way a figure is watermarked. This writes each experiment-derived
table as a LaTeX fragment the thesis `\\input`s, so the same source switch that
governs figures governs tables too.

Configuration tables are deliberately not generated. The optimiser settings,
the input widths and the budget are specifications, not results: they are the
same whether or not anything has been trained.

Writes into <thesis>/tables/:
  probe_quality.tex        decodability and belief quality, both probe families
  integration_gap.tex      within-method conditioning gap on the medium instance
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from evaluation.protocol import MEDIUM_ENV
from utils.paths import analysis_dir, is_dummy, resolve_data, results_root, thesis_fig_dir
from utils.script_output import ScriptRun

FINAL = lambda: analysis_dir()          # noqa: E731

VARIANTS = [
    ("rl2_concat", "RL\\textsuperscript{2} concat"),
    ("rl2_hypernet", "RL\\textsuperscript{2} hypernet"),
    ("varibad_concat", "VariBAD concat"),
    ("varibad_hypernet", "VariBAD hypernet"),
]


def _load(path: Path):
    p = resolve_data(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _banner(n_cols: int) -> str:
    """The row that marks a table as synthetic, mirroring the figure watermark."""
    if not is_dummy():
        return ""
    return (
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\textcolor{{red}}"
        f"{{\\footnotesize DUMMY DATA, synthetic values, not a result}}}} \\\\\n"
        "\\hline\n"
    )


def _fmt(x, nd=3, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return f"${s}$"


# ---------------------------------------------------------------------------


# The four per-measurement probe tables of the posterior-quality probe, in the
# order the methodology defines the metrics. Each is self-contained: the level
# each architecture reaches, the independent difference in means between them, and the test
# the protocol requires of a reported comparison.
_PROBE_METRIC_LABEL = {
    "method_test_acc": ("probe_decodability", 3),
    "method_kl_to_omega": ("probe_kl", 3),
    "method_log_loss": ("probe_log_loss", 3),
    "method_brier": ("probe_brier", 3),
}
_METHOD_LABEL = {"rl2": "RL\\textsuperscript{2}", "varibad": "VariBAD"}
_PROBE_LABEL = {"linear": "Linear", "mlp": "MLP"}


def _fmt_p(p) -> str:
    """A p-value, with a floor rather than a rounded zero."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def _belief_rows(metric: str, nd: int, with_holm: bool) -> list[str] | None:
    tests = _load(FINAL() / "m5r_belief_quality_tests.json")
    if tests is None:
        return None
    pool = list(tests.get("comparisons", []))
    pool += list(tests.get("robustness_comparisons", []))
    rows = []
    for method in ("rl2", "varibad"):
        for probe in ("linear", "mlp"):
            # The corrected family predates the `probe` field, so an entry
            # without one is the linear probe by construction.
            hit = next(
                (c for c in pool
                 if c["method"] == method and c["metric"] == metric
                 and c.get("probe", "linear") == probe),
                None,
            )
            if hit is None:
                continue
            lo, hi = hit["delta_ci"]
            # Section 3.10 fixes what a results table carries: the level, the
            # paired difference, its interval, the seed count and the corrected
            # p. The raw p and the effect size stay in the Appendix D artefacts.
            cells = [
                _METHOD_LABEL[method] if probe == "linear" else "",
                _PROBE_LABEL[probe],
                _fmt(hit["concat_mean"], nd),
                _fmt(hit["hypernet_mean"], nd),
                _fmt(hit["mean_paired_delta"], nd, plus=True),
                f"$[{lo:+.{nd}f},\\ {hi:+.{nd}f}]$",
                f"${hit['seeds_favouring_hypernet']}/{hit['n_pairs']}$",
            ]
            if with_holm:
                cells.append(_fmt_p(hit.get("holm_corrected_p")))
            rows.append(" & ".join(cells) + r" \\")
        if method == "rl2" and rows:
            rows.append(r"\cmidrule(l){2-%d}" % (8 if with_holm else 7))
    return rows or None


def _belief_table(metric: str, with_holm: bool = True) -> str | None:
    """One metric's table. The Holm column is dropped for the proper scores,
    which the protocol never corrects, so it would be a column of dashes."""
    _, nd = _PROBE_METRIC_LABEL[metric]
    rows = _belief_rows(metric, nd, with_holm)
    if rows is None:
        return None
    n_cols = 8 if with_holm else 7
    head = (r"Method & Probe & Concatenation & Hypernetwork & $\bar{d}$ & "
            r"$95\%$ CI & Seeds favouring")
    if with_holm:
        head += r" & $p_{\text{Holm}}$"
    body = "\n".join(rows)
    spec = "ll" + "r" * (n_cols - 2)
    return (
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        f"\\begin{{tabular}}{{{spec}}}\n"
        "\\toprule\n"
        f"{_banner(n_cols)}{head} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def probe_decodability() -> str | None:
    return _belief_table("method_test_acc")


def probe_kl() -> str | None:
    return _belief_table("method_kl_to_omega")


def probe_log_loss() -> str | None:
    return _belief_table("method_log_loss", with_holm=False)


def probe_brier() -> str | None:
    return _belief_table("method_brier", with_holm=False)


def integration_gap() -> str | None:
    tests = _load(FINAL() / "m5r_hypothesis_tests.json")
    evals = _load(FINAL() / "m5r_post_training_evaluation.json")
    if tests is None:
        return None
    results = tests.get("family_b", {}).get("results", {})

    methods = evals.get("methods", {}) if evals else {}
    floor = methods.get("regime_agnostic_ppo", {}).get("evaluation_return_mean")
    belief = methods.get("belief_ppo", {}).get("evaluation_return_mean")
    reference_gap = belief - floor if floor is not None and belief is not None else None

    rows = []
    for method, label in (("rl2", "RL\\textsuperscript{2}"), ("varibad", "VariBAD")):
        r = results.get(f"{method}_hypernet_beats_concat_{MEDIUM_ENV}")
        if r is None:
            continue
        lo, hi = r.get("delta_ci", [None, None])
        difference = r.get("mean_difference", r.get("mean_paired_delta"))
        fraction = difference / reference_gap if reference_gap else None
        rows.append(
            f"{label} & {_fmt(difference, 2, plus=True)} "
            f"& $[{lo:+.2f},\\, {hi:+.2f}]$ "
            f"& {_fmt(fraction, 2, plus=True)} "
            f"& {_fmt_p(r.get('holm_corrected_p'))} \\\\"
        )
    if not rows:
        return None
    body = "\n".join(rows)
    return f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
{_banner(5)}Method & $\\bar{{d}}$ & $95\\%$ CI & Reference gap fraction & $p_{{\\mathrm{{Holm}}}}$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
"""


TABLES = {
    "probe_decodability": probe_decodability,
    "probe_kl": probe_kl,
    "probe_log_loss": probe_log_loss,
    "probe_brier": probe_brier,
    "integration_gap": integration_gap,
}


def probe_confusion() -> str | None:
    """Held-out, row-normalized regime confusion from the saved probe rerun."""
    data = _load(FINAL() / "m5r_probe_confusion.json")
    if data is None:
        return None
    rows = []
    for key, label in VARIANTS:
        matrix = data["by_method"][key]["row_normalized"]
        for regime, probabilities in enumerate(matrix):
            shown_label = label if regime == 0 else ""
            cells = " & ".join(f"${value:.3f}$" for value in probabilities)
            rows.append(f"{shown_label} & ${regime}$ & {cells} \\\\")
        rows.append("\\addlinespace")
    return ("\\begin{tabular}{llrrr}\n"
            "\\toprule\n"
            f"{_banner(5)}Method & True regime & Pred. $0$ & Pred. $1$ & Pred. $2$ \\\\\n"
            "\\midrule\n"
            + "\n".join(rows[:-1]) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n")


TABLES["probe_confusion"] = probe_confusion



def seed_block_sensitivity() -> str | None:
    data = _load(FINAL() / "m5r_seed_block_sensitivity.json")
    if data is None or len(data.get("comparisons", [])) != 18:
        return None
    families = {"architecture_returns": "Architecture return", "method_returns": "Method return",
                "speed": "Time to reference", "probe": "Linear probe", "probe_mlp": "MLP probe",
                "diagnostics": "Behaviour"}
    rows = []
    for item in data["comparisons"]:
        name = item["comparison"]
        method = "RL2" if name.startswith(("rl2", "locked_regime_action_distribution_rl2", "belief_swap_belief_only_rl2")) else "VariBAD"
        if item["family"] == "method_returns":
            label = "RL2--VariBAD, " + ("hypernet" if name.endswith("hypernet") else "concat")
        elif item["family"] in ("probe", "probe_mlp"):
            label = method + (", KL" if item["metric"] == "method_kl_to_omega" else ", accuracy")
        elif item["family"] == "diagnostics":
            label = method + (", response" if name.startswith("locked") else ", substitution")
        else:
            label = method
        lo, hi = item["paired_bootstrap_ci"]
        digits = 1 if item["family"] == "speed" else 3
        rows.append(f"{families[item['family']]} & {label} & {_fmt(item['delta'], digits, plus=True)} "
                    f"& $[{lo:+.{digits}f},\\ {hi:+.{digits}f}]$ & {_fmt_p(item['paired_p_holm'])} " + r"\\")
    return ("\\begin{tabular}{llrrr}\n\\toprule\n" + _banner(5)
            + r"Family & Comparison & $\bar d$ & Seed-block $95\%$ CI & $p_{\mathrm{Holm}}$ " + r"\\" + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


TABLES["seed_block_sensitivity"] = seed_block_sensitivity


def main() -> int:
    run = ScriptRun(script="make_tables")
    out_dir = thesis_fig_dir().parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for name, builder in TABLES.items():
        try:
            body = builder()
        except Exception as exc:  # one bad table must not lose the rest
            print(f"[make_tables] {name}: FAILED ({exc})", flush=True)
            skipped.append(name)
            continue
        if body is None:
            print(f"[make_tables] {name}: skipped, source data absent", flush=True)
            skipped.append(name)
            continue
        (out_dir / f"{name}.tex").write_text(body)
        written.append(name)
        print(f"[make_tables] wrote {out_dir / (name + '.tex')}", flush=True)

    marker = out_dir / "PROVENANCE.txt"
    marker.write_text(
        ("dummy\n" if is_dummy() else "real\n")
        + "tables generated by scripts.make_tables\n"
    )
    stats_path = analysis_dir() / "make_tables_run.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    run.add_output(str(out_dir))
    if skipped:
        run.fail(reason=f"Required tables missing or invalid: {skipped}", summary_path=stats_path)
        return 1
    run.ok(key_stats={"written": len(written), "skipped": len(skipped),
                      "dummy": is_dummy()},
           summary_path=stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
