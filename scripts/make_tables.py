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
# each architecture reaches, the paired difference between them, and the test
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
        "\\setlength{\\tabcolsep}{4pt}\n"
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
    evals = _load(FINAL() / "m5r_final_run.json") or _load(FINAL() / "per_cell_env.json")
    if tests is None:
        return None
    results = tests.get("family_b", {}).get("results", {})

    def cell_mean(name):
        if not evals:
            return None
        per = evals.get("per_env", {}).get("e_final", {})
        block = per.get(name) or {}
        for k in ("final_return_mean", "mean", "return_mean"):
            if k in block:
                return block[k]
        return None

    rows = []
    for method, label in (("rl2", "RL\\textsuperscript{2}"), ("varibad", "VariBAD")):
        r = results.get(f"{method}_hypernet_beats_concat_e_final")
        if r is None:
            continue
        lo, hi = r.get("delta_ci", [None, None])
        rows.append(
            f"{label} & {_fmt(cell_mean(f'{method}_concat'), 1)} "
            f"& {_fmt(cell_mean(f'{method}_hypernet'), 1)} "
            f"& ${r['mean_paired_delta']:+.2f}\\;[{lo:+.2f},\\, {hi:+.2f}]$ "
            f"& ${r['holm_corrected_p']:.1e}$ "
            f"& {_fmt(r.get('rank_biserial'), 2, plus=True)} \\\\"
        )
    if not rows:
        return None
    body = "\n".join(rows)
    return f"""\\begin{{tabular}}{{lccccc}}
\\hline
{_banner(6)}\\textbf{{Method}} & \\textbf{{concat}} & \\textbf{{hypernet}}
 & \\textbf{{$\\Delta$ (95\\% CI)}} & \\textbf{{Holm $p$}} & \\textbf{{$r_{{\\mathrm{{rb}}}}$}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
"""


TABLES = {
    "probe_decodability": probe_decodability,
    "probe_kl": probe_kl,
    "probe_log_loss": probe_log_loss,
    "probe_brier": probe_brier,
    "integration_gap": integration_gap,
}


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
    run.ok(key_stats={"written": len(written), "skipped": len(skipped),
                      "dummy": is_dummy()},
           summary_path=stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
