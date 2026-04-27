"""M5 plot module — figures for every M5 stage that has data on disk.

Produces:
  - step3_mm_hypernet.png   — Step-3 MM hypernet probe (4 bars + reference lines)
  - probe_per_t.png         — Step-5 per-timestep regime classification accuracy
  - step4_ladder.png        — Step-4 4-cell ladder bar chart (when ready)
  - step4_learning_curves.png — Step-4 per-iter learning curves with seed CIs

The Step-3 toy factorial chart was retitled and moved under M4 (its
content is implementation validation on toy envs). It now lives in
`plotting.m4_plots.plot_factorial_toys` and writes to
`figures/milestones/M4/factorial_toys.png`.

Each plot function loads a stats JSON, produces one PNG, and tolerates
missing inputs (skips gracefully). The CLI entry generates everything it
can find data for.

Output directory: `figures/milestones/M5/`. Dpi=150, tight bounding box,
applied via `plotting.style.apply_style()`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import COLORS, FIGSIZE_STANDARD, FIGSIZE_WIDE, apply_style

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures" / "milestones" / "M5"


def _budget_annotation(
    fig,
    iterations: int | None = None,
    parallel_envs: int | None = None,
    rollout_length: int | None = None,
    num_seeds: int | None = None,
    extra: str = "",
    y: float = 0.005,
) -> None:
    """Add a small caption at the bottom of the figure documenting compute
    budget. Reader can immediately tell whether a chart is from a smoke
    run or a full-budget experiment.
    """
    parts = []
    if iterations is not None:
        parts.append(f"{iterations} iter")
    if parallel_envs is not None:
        parts.append(f"{parallel_envs} envs")
    if rollout_length is not None:
        parts.append(f"rollout {rollout_length}")
    if num_seeds is not None:
        parts.append(f"n={num_seeds} seeds")
    if extra:
        parts.append(extra)
    if not parts:
        return
    text = "Compute: " + " × ".join(parts)
    fig.text(0.5, y, text, ha="center", fontsize=7, style="italic", color="#555555")


# ---------------------------------------------------------------------------
# References (M3 floor / belief / oracle)
# ---------------------------------------------------------------------------


def _load_m3_refs() -> dict[str, float]:
    path = RESULTS_ROOT / "milestones" / "M3" / "stats_M3_reference_levels.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    refs = d.get("reference_levels", {})
    return {
        "agnostic": float(refs.get("regime_agnostic_ppo", {}).get("mean", float("nan"))),
        "belief": float(refs.get("belief_ppo", {}).get("mean", float("nan"))),
        "oracle": float(refs.get("oracle_ppo", {}).get("mean", float("nan"))),
    }


def _draw_reference_lines(ax, refs: dict[str, float], xmin=None, xmax=None) -> None:
    """Horizontal dashed lines for floor / belief / oracle references."""
    if not refs:
        return
    style = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.6}
    if not np.isnan(refs.get("agnostic", float("nan"))):
        ax.axhline(refs["agnostic"], color=COLORS["ppo"], **style,
                   label=f"Regime-agnostic PPO floor = {refs['agnostic']:.1f}")
    if not np.isnan(refs.get("belief", float("nan"))):
        ax.axhline(refs["belief"], color=COLORS["belief_ppo"], **style,
                   label=f"Belief-PPO ceiling = {refs['belief']:.1f}")
    if not np.isnan(refs.get("oracle", float("nan"))):
        ax.axhline(refs["oracle"], color=COLORS["oracle_ppo"], **style,
                   label=f"Oracle-PPO ceiling = {refs['oracle']:.1f}")


# ---------------------------------------------------------------------------
# Step-3 MM hypernet — 4 bars from existing concat (Step 2) + hypernet (Step 3) data
# ---------------------------------------------------------------------------


def plot_step3_mm_hypernet(out_path: Path) -> bool:
    """4-bar chart of MM hypernet probe results.

    Reuses existing experiment metrics (no aggregate stats JSON exists for
    this step):
      - rl2_concat:  m5_step2_rl2_h64_e01    (h64, n=3, 100 iter)
      - rl2_hypernet: m5_step3_rl2_hypernet  (h64, n=3, 100 iter, hypernet)
      - varibad_concat: m5_step2_varibad_kl10 (kl=0.1, n=3, 100 iter)
      - varibad_hypernet: m5_step3_varibad_hypernet (n=3, 100 iter, hypernet)
    """
    cells = [
        # (xtick_label, legend_label, experiment_name, color, alpha)
        ("RL²\nConcat",          "RL² Concat",          "m5_step2_rl2_h64_e01",        COLORS["rl2"],     0.6),
        ("RL²\nHyper-\nnetwork", "RL² Hypernetwork",    "m5_step3_rl2_hypernet",       COLORS["rl2"],     1.0),
        ("VariBAD\nConcat",      "VariBAD Concat",      "m5_step2_varibad_kl10",       COLORS["varibad"], 0.6),
        ("VariBAD\nHyper-\nnetwork", "VariBAD Hypernetwork", "m5_step3_varibad_hypernet", COLORS["varibad"], 1.0),
    ]
    means: list[float] = []
    cis: list[tuple[float, float]] = []
    xtick_labels: list[str] = []
    legend_labels: list[str] = []
    colors: list[Any] = []
    alphas: list[float] = []
    for xtick_label, legend_label, exp_name, color, alpha in cells:
        m_path = RESULTS_ROOT / exp_name / "metrics.json"
        if not m_path.exists():
            print(f"[m5_plots] skip step3_mm_hypernet: missing {m_path}")
            return False
        with open(m_path) as f:
            m = json.load(f)
        means.append(float(m["final_return_mean"]))
        cis.append(tuple(m["final_return_ci95"]))
        xtick_labels.append(xtick_label)
        legend_labels.append(legend_label)
        colors.append(color)
        alphas.append(alpha)

    refs = _load_m3_refs()
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    x = np.arange(len(xtick_labels))
    err = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T
    bars = ax.bar(
        x, means, yerr=err, capsize=3, edgecolor="black", linewidth=0.4,
    )
    for bar, color, alpha in zip(bars, colors, alphas):
        bar.set_facecolor(color)
        bar.set_alpha(alpha)
    # Place value labels inside the bar, just below the lower CI cap, so
    # they sit clearly below the vertical CI line (offset in points keeps
    # the gap consistent regardless of data scale).
    for xi, mean, (lo, hi) in zip(x, means, cis):
        ax.annotate(f"{mean:.1f}", xy=(xi, lo),
                    xytext=(0, -3), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_title(
        "MarketMakingV1 — Concat vs. Hypernetwork (half-budget probe)\n"
        "Final return (mean across 3 seeds, 100 iter)"
    )
    _draw_reference_lines(ax, refs)
    # Per the repo legend convention, list every chart element. Add proxy
    # patches for the bars (one per cell) and combine with the reference-
    # line handles already on the axes.
    from matplotlib.patches import Patch
    bar_handles = [
        Patch(facecolor=color, alpha=alpha, edgecolor="black",
              linewidth=0.4, label=legend_label)
        for legend_label, color, alpha in zip(legend_labels, colors, alphas)
    ]
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=bar_handles + ref_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    # Cap at oracle + 5 so the Oracle-PPO ceiling line is clearly visible
    # above the bars rather than pinned to the chart's top edge.
    ax.set_ylim(80, max(refs.get("oracle", 200) + 5, max(means) + 10))
    # Read budget from the first cell's metrics (all 4 cells share budget
    # in this comparison: 100 iter × 3 seeds × 512 envs).
    first_meta = json.load(open(RESULTS_ROOT / cells[0][2] / "metrics.json"))
    _budget_annotation(
        fig,
        iterations=int(first_meta["iterations"]),
        parallel_envs=int(first_meta["parallel_envs"]),
        rollout_length=int(first_meta["rollout_length"]),
        num_seeds=int(first_meta["num_seeds"]),
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.62, bottom=0.20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Step-5 posterior probe per-timestep accuracy curves
# ---------------------------------------------------------------------------


def _method_label_and_color(exp_name: str) -> tuple[str, str]:
    """Map experiment_name to (display label, color).

    Uses the per-cell COLORS keys (`rl2_concat`, `rl2_hypernet`,
    `varibad_concat`, `varibad_hypernet`) so the four cells are visually
    distinct in every chart that includes them.
    """
    family = "rl2" if "rl2" in exp_name else "varibad" if "varibad" in exp_name else None
    if family is None:
        return exp_name, "#666666"
    pretty = "RL²" if family == "rl2" else "VariBAD"
    if "hypernet" in exp_name:
        return f"{pretty} Hypernetwork", COLORS[f"{family}_hypernet"]
    if "concat" in exp_name:
        return f"{pretty} Concat", COLORS[f"{family}_concat"]
    return pretty, COLORS[family]


def _smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean (window must be odd or +1 of even). Edges
    use the available samples (no padding artifacts)."""
    if window <= 1:
        return y.copy()
    half = window // 2
    out = np.zeros_like(y)
    for i in range(y.size):
        lo = max(0, i - half)
        hi = min(y.size, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def plot_probe_per_t(out_path: Path) -> bool:
    """Two-panel probe figure.

    Left:  headline bar chart of mean test accuracy (across seeds, with
           bootstrap 95% CI) for each method + analytical-posterior
           reference. Chance line marked.
    Right: per-timestep accuracy curves with shaded CI bands and a 9-step
           rolling-mean smoother to suppress high-frequency sampling
           noise. Same color palette as left panel.
    """
    stats_path = RESULTS_ROOT / "milestones" / "M5" / "stats_M5_posterior_probe.json"
    if not stats_path.exists():
        print(f"[m5_plots] skip probe_per_t: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)

    apply_style()
    fig, (ax_bar, ax_curve) = plt.subplots(
        1, 2, figsize=(11.0, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.7]},
    )

    method_items = list(stats["methods"].items())
    n_classes = method_items[0][1].get("n_classes_observed", 3)
    chance = 1.0 / n_classes
    smoothing_window = 9

    # ----- Left panel: headline bar chart -----------------------------------
    bar_labels: list[str] = []
    bar_means: list[float] = []
    bar_lo: list[float] = []
    bar_hi: list[float] = []
    bar_colors: list[str] = []
    for exp_name, m in method_items:
        label, color = _method_label_and_color(exp_name)
        # Single-line label; rotation below prevents adjacent-bar overlap
        # without splitting onto two lines (which crowded the panel).
        bar_labels.append(label)
        bar_means.append(m["method_test_acc_mean"])
        lo, hi = m["method_test_acc_ci95"]
        bar_lo.append(m["method_test_acc_mean"] - lo)
        bar_hi.append(hi - m["method_test_acc_mean"])
        bar_colors.append(color)

    # Add analytical posterior as the rightmost reference bar (mean across
    # methods — they all see the same analytical posterior up to seed
    # variation, so averaging is reasonable).
    ana_means = [m["analytical_test_acc_mean"] for _, m in method_items]
    ana_mean = float(np.mean(ana_means))
    ana_per_seed = np.concatenate([
        np.asarray(m["analytical_test_acc_per_seed"]) for _, m in method_items
    ])
    rng = np.random.default_rng(0)
    boot = rng.integers(0, ana_per_seed.size, size=(10_000, ana_per_seed.size))
    ana_lo = float(np.percentile(ana_per_seed[boot].mean(axis=1), 2.5))
    ana_hi = float(np.percentile(ana_per_seed[boot].mean(axis=1), 97.5))
    bar_labels.append("Analytical posterior")
    bar_means.append(ana_mean)
    bar_lo.append(ana_mean - ana_lo)
    bar_hi.append(ana_hi - ana_mean)
    bar_colors.append(COLORS["analytical"])

    x_bar = np.arange(len(bar_labels))
    yerr = np.array([bar_lo, bar_hi])
    bars = ax_bar.bar(
        x_bar, bar_means, yerr=yerr, capsize=3, edgecolor="black",
        linewidth=0.4,
    )
    for bar, color in zip(bars, bar_colors):
        bar.set_facecolor(color)
    # Place value labels inside the bar, just below the lower CI cap, so
    # they sit clearly below the vertical CI line (offset in points keeps
    # the gap consistent regardless of data scale).
    for xi, mean, lo_err in zip(x_bar, bar_means, bar_lo):
        ax_bar.annotate(f"{mean:.2f}", xy=(xi, mean - lo_err),
                        xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=9, color="black")
    # Random-guess line — drawn but not legended on this panel. The right
    # panel's outside legend already documents the same dotted line, so
    # adding a second legend here would overlap the vertical bar labels.
    ax_bar.axhline(chance, color="#999999", linestyle=":", linewidth=1.0)
    ax_bar.set_xticks(x_bar)
    # Vertical labels avoid the adjacent-bar overlap that rotation=25-35
    # still leaves on this narrow panel; reads cleanly with `ha="center"`.
    ax_bar.set_xticklabels(bar_labels, fontsize=8, rotation=90, ha="center")
    ax_bar.set_ylim(max(0.0, chance - 0.05), 1.05)
    ax_bar.set_title("Headline test accuracy\n(regime classification, test set)")

    # ----- Right panel: per-timestep curves (smoothed) ---------------------
    analytical_drawn = False
    for exp_name, m in method_items:
        label, color = _method_label_and_color(exp_name)
        per_t = np.array(m["method_per_t_test_acc_mean"])
        per_t_per_seed = np.array(m["method_per_t_test_acc_per_seed"])
        n_seeds = per_t_per_seed.shape[0]
        T = per_t.shape[0]

        # Bootstrap 95% CI per t across seeds, then smooth the band.
        if n_seeds > 1:
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n_seeds, size=(1000, n_seeds))
            lo_band = np.zeros(T); hi_band = np.zeros(T)
            for t in range(T):
                vals = per_t_per_seed[idx, t].mean(axis=1)
                lo_band[t] = np.percentile(vals, 2.5)
                hi_band[t] = np.percentile(vals, 97.5)
        else:
            lo_band, hi_band = per_t.copy(), per_t.copy()

        per_t_s = _smooth_curve(per_t, smoothing_window)
        lo_s = _smooth_curve(lo_band, smoothing_window)
        hi_s = _smooth_curve(hi_band, smoothing_window)
        ts = np.arange(T)
        ax_curve.plot(ts, per_t_s, color=color, label=f"{label} (n={n_seeds})")
        ax_curve.fill_between(ts, lo_s, hi_s, color=color, alpha=0.2)

        if not analytical_drawn:
            ana = np.array(m["analytical_per_t_test_acc_mean"])
            ax_curve.plot(ts, _smooth_curve(ana, smoothing_window),
                          color=COLORS["analytical"], linestyle="--",
                          linewidth=1.2,
                          label="Analytical posterior (reference)")
            analytical_drawn = True

    ax_curve.axhline(chance, color="#999999", linestyle=":", linewidth=1.0,
                     label=f"Random guess ({100.0/n_classes:.1f}%)")
    ax_curve.set_xlabel("Timestep within episode")
    ax_curve.set_title(
        f"Per-timestep accuracy (rolling-mean window {smoothing_window})\n"
        "Regime classification, test set"
    )
    ax_curve.set_ylim(max(0.0, chance - 0.05), 1.05)
    ax_curve.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )

    env_label = stats.get("env_label", "MarketMakingV1")
    fig.suptitle(f"Posterior-quality probe on {env_label}", y=1.02)
    n_seeds = max(m["n_seeds"] for _, m in method_items)
    _budget_annotation(
        fig,
        rollout_length=int(stats.get("rollout_length", 0)) or None,
        num_seeds=n_seeds,
        extra=f"{stats.get('n_rollouts')} rollouts/seed, classifier={stats.get('classifier')}",
    )
    fig.tight_layout()
    # Generous bottom margin so the vertical bar tick labels (longest:
    # "VariBAD Hypernetwork") fit fully without clipping.
    fig.subplots_adjust(right=0.82, bottom=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Step-4 4-cell ladder (bar chart) and learning curves — empty stubs until data
# ---------------------------------------------------------------------------


def plot_step4_ladder(out_path: Path) -> bool:
    stats_path = RESULTS_ROOT / "milestones" / "M5" / "stats_M5_step4_ladder.json"
    if not stats_path.exists():
        print(f"[m5_plots] skip step4_ladder: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    refs = _load_m3_refs()

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    methods = ["rl2", "varibad"]
    integrations = ["concat", "hypernet"]
    bar_w = 0.38

    cells = stats.get("cells", {})
    method_labels_full = {"rl2": "RL²", "varibad": "VariBAD"}
    for i_int, integ in enumerate(integrations):
        means = []
        errs = []
        for method in methods:
            cell_key = f"{method}_{integ}"
            if cell_key not in cells:
                means.append(np.nan)
                errs.append([0.0, 0.0])
                continue
            c = cells[cell_key]
            means.append(c["final_return"]["mean"])
            lo, hi = c["final_return"]["ci"]
            errs.append([c["final_return"]["mean"] - lo, hi - c["final_return"]["mean"]])
        offset = (i_int - 0.5) * bar_w
        x = np.arange(len(methods)) + offset
        yerr = np.array(errs).T
        bars = ax.bar(
            x, means, bar_w, yerr=yerr, capsize=3, edgecolor="black",
            linewidth=0.4,
        )
        # Per-cell colour from the dedicated COLORS keys (no alpha tricks)
        # so each of the 4 cells is independently distinguishable.
        for bar, m in zip(bars, methods):
            bar.set_facecolor(COLORS[f"{m}_{integ}"])
        # Value labels inside each bar, just below the lower CI cap so they
        # don't overlap the vertical CI line. Skip NaN bars.
        for xi, mean, e in zip(x, means, errs):
            if np.isnan(mean):
                continue
            lo_abs = mean - e[0]
            ax.annotate(f"{mean:.1f}", xy=(xi, lo_abs),
                        xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color="black")
    # Build legend manually with proxy artists so every coloured bar
    # combination is documented (method × integration), not just one axis.
    from matplotlib.patches import Patch
    legend_handles = []
    integ_pretty = {"concat": "Concat", "hypernet": "Hypernetwork"}
    for method in methods:
        for integ in integrations:
            legend_handles.append(Patch(
                facecolor=COLORS[f"{method}_{integ}"], edgecolor="black",
                linewidth=0.4,
                label=f"{method_labels_full[method]} {integ_pretty[integ]}",
            ))

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(["RL²", "VariBAD"])
    ax.set_title(
        "MarketMakingV1 — RL²/VariBAD × Concat/Hypernetwork\n"
        "Final return (mean across n=8 seeds, 200 iter)"
    )
    _draw_reference_lines(ax, refs)
    # Y-axis convention: start at floor − 5 so bars are comparable to the
    # floor reference (full-budget run). For smoke runs whose means sit
    # below the floor, extend the bottom so the bars (and their lower CI
    # caps) are still visible.
    floor = refs.get("agnostic", float("nan"))
    if not np.isnan(floor):
        finite_means = [c["final_return"]["mean"] for c in cells.values() if not np.isnan(c["final_return"]["mean"])]
        ci_los = [c["final_return"]["ci"][0] for c in cells.values() if not np.isnan(c["final_return"]["mean"])]
        ymin = min([floor, *ci_los]) - 5
        ymax = max([refs.get("oracle", 200), *finite_means, floor]) + 5
        ax.set_ylim(ymin, ymax)
    # Combine method×integration proxy patches with reference-line handles.
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=legend_handles + ref_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    # Compute budget — read iterations / num_seeds / parallel_envs from
    # the first cell's saved metrics.json (all cells share budget here).
    first_cell = next(iter(cells.values()), None)
    if first_cell is not None:
        first_meta_path = RESULTS_ROOT / first_cell["experiment_name"] / "metrics.json"
        if first_meta_path.exists():
            mm = json.load(open(first_meta_path))
            _budget_annotation(
                fig,
                iterations=int(mm["iterations"]),
                parallel_envs=int(mm["parallel_envs"]),
                rollout_length=int(mm["rollout_length"]),
                num_seeds=int(mm["num_seeds"]),
            )
    fig.tight_layout()
    fig.subplots_adjust(right=0.65, bottom=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


def plot_step4_learning_curves(out_path: Path) -> bool:
    stats_path = RESULTS_ROOT / "milestones" / "M5" / "stats_M5_step4_ladder.json"
    if not stats_path.exists():
        print(f"[m5_plots] skip step4_learning_curves: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    refs = _load_m3_refs()

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    cells = stats.get("cells", {})
    method_pretty = {"rl2": "RL²", "varibad": "VariBAD"}
    integ_pretty = {"concat": "Concat", "hypernet": "Hypernetwork"}
    for cell_key, c in cells.items():
        # cell_key e.g. "rl2_hypernet"
        method = cell_key.split("_")[0]
        integ = "_".join(cell_key.split("_")[1:])
        # Per-cell colour from the dedicated COLORS keys; solid line for
        # all four so each cell is identified by its colour, not a line
        # style + alpha combo (which is harder to distinguish).
        color = COLORS.get(f"{method}_{integ}", "#666666")
        linestyle = "-"
        cell_label = f"{method_pretty.get(method, method)} {integ_pretty.get(integ, integ)}"
        exp_name = c["experiment_name"]
        m_path = RESULTS_ROOT / exp_name / "metrics.json"
        if not m_path.exists():
            continue
        with open(m_path) as f:
            m = json.load(f)
        per_seed = np.array(m["per_seed_mean_return_per_iter"])  # [n_seeds, T]
        mean = per_seed.mean(axis=0)
        n_seeds = per_seed.shape[0]
        rng = np.random.default_rng(0)
        if n_seeds > 1:
            idx = rng.integers(0, n_seeds, size=(1000, n_seeds))
            T = per_seed.shape[1]
            lo = np.zeros(T); hi = np.zeros(T)
            for t in range(T):
                vals = per_seed[idx, t].mean(axis=1)
                lo[t] = np.percentile(vals, 2.5)
                hi[t] = np.percentile(vals, 97.5)
        else:
            lo, hi = mean.copy(), mean.copy()
        iters = np.arange(len(mean))
        ax.plot(iters, mean, color=color, linestyle=linestyle, label=cell_label)
        ax.fill_between(iters, lo, hi, color=color, alpha=0.15)

    _draw_reference_lines(ax, refs)
    ax.set_xlabel("Iteration")
    ax.set_title(
        "MarketMakingV1 — learning curves by method × integration\n"
        "Mean return across seeds, shaded = 95% CI"
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    # Compute budget from the first cell's metrics.
    first_cell = next(iter(cells.values()), None)
    if first_cell is not None:
        first_meta_path = RESULTS_ROOT / first_cell["experiment_name"] / "metrics.json"
        if first_meta_path.exists():
            mm = json.load(open(first_meta_path))
            _budget_annotation(
                fig,
                iterations=int(mm["iterations"]),
                parallel_envs=int(mm["parallel_envs"]),
                rollout_length=int(mm["rollout_length"]),
                num_seeds=int(mm["num_seeds"]),
            )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78, bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# CLI: generate all available plots
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="plotting.m5_plots")
    parser.add_argument(
        "--out-dir", default=str(FIGURES_ROOT),
        help="output directory for PNGs (default: figures/m5/)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `step3_mm_hypernet` was the half-budget (n=3 × 100 iter) preview that
    # motivated the full Step-4 run. Now superseded by step4_ladder.png at
    # n=8 × 200 iter — same 4 cells, same env. Function kept in this module
    # for archival regen if needed; not generated by default.
    written = 0
    for name, fn in [
        ("probe_per_t.png", plot_probe_per_t),
        ("step4_ladder.png", plot_step4_ladder),
        ("step4_learning_curves.png", plot_step4_learning_curves),
    ]:
        if fn(out_dir / name):
            written += 1
    print(f"[m5_plots] wrote {written} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
