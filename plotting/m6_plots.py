"""M6 plot module — difficulty sweep (two-panel) and posterior-vs-performance scatter.

Produces:
  - rq3_persistence_sweep.png        — two-panel chart on persistence axis
  - rq3_distinguishability_sweep.png — two-panel chart on distinguishability axis
  - rq3_posterior_vs_performance.png — scatter (posterior_error vs gap_closed)

Each sweep figure has two panels side by side:
  Left  — absolute mean return per method per level (all 7 methods, with
          regime-agnostic / oracle drawn dashed as reference brackets).
          Shows how the absolute floor / belief / oracle / meta-RL means
          drift across difficulty in raw return units.
  Right — gap_closed = (mean − floor) / (oracle − floor), the normalized
          comparison RQ3 uses. Drops regime-agnostic and oracle from the
          line set since they sit at 0 and 1 by construction; the
          reference dashes encode them.

Each plot function loads a stats JSON, produces one PNG, and tolerates
missing inputs (skips gracefully). Output: `figures/milestones/M6/`.

CLI:
    uv run python -m plotting.m6_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.style import COLORS, apply_style, budget_annotation

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures" / "milestones" / "M6"


_LEVELS = ("easy", "medium", "hard")
_LEVEL_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}

# All methods plotted on the left (absolute return) panel, in vertical
# stacking order top-to-bottom matching the typical ranking.
_ABS_METHOD_ORDER = (
    "oracle_ppo",
    "belief_ppo",
    "rl2_hypernet",
    "varibad_hypernet",
    "rl2_concat",
    "varibad_concat",
    "regime_agnostic_ppo",
)
# Meta-RL methods plotted on the right (gap_closed) panel. regime_agnostic
# and oracle are excluded — by construction they sit at 0 and 1 every
# level, identical to the dashed reference lines on that panel.
_GC_METHOD_ORDER = (
    "belief_ppo",
    "rl2_concat",
    "rl2_hypernet",
    "varibad_concat",
    "varibad_hypernet",
)
_METHOD_LABELS = {
    "regime_agnostic_ppo": "Regime-agnostic PPO",
    "belief_ppo":          "Belief-PPO",
    "oracle_ppo":          "Oracle-PPO",
    "rl2_concat":          "RL² Concat",
    "rl2_hypernet":        "RL² Hypernetwork",
    "varibad_concat":      "VariBAD Concat",
    "varibad_hypernet":    "VariBAD Hypernetwork",
}
# M6 colors: each of the 7 methods gets a distinct hue so two lines on
# the same panel never share a colour. The three reference methods
# (regime-agnostic, Belief, Oracle) keep their canonical project colours
# from M3/M5 and are drawn *dashed* — they are upper/lower bounds, not
# methods being benchmarked. The four meta-RL methods are drawn solid;
# rl2_concat (was blue, collided with regime-agnostic) is bumped to
# cyan, and varibad_concat (was purple, collided with Belief-PPO) is
# bumped to pink. RL² and VariBAD hypernet keep their M5 orange/green.
_METHOD_COLORS = {
    "regime_agnostic_ppo": COLORS["ppo"],          # blue, dashed (reference)
    "belief_ppo":          COLORS["belief_ppo"],   # purple, dashed (reference)
    "oracle_ppo":          COLORS["oracle_ppo"],   # brown, dashed (reference)
    "rl2_concat":          "#17becf",              # cyan (overridden vs M5 blue)
    "rl2_hypernet":        COLORS["rl2_hypernet"], # orange (M5 standard)
    "varibad_concat":      "#e377c2",              # pink (overridden vs M5 purple)
    "varibad_hypernet":    COLORS["varibad_hypernet"],  # green (M5 standard)
}
# All three references render dashed on both panels. Belief-PPO joins
# regime-agnostic and Oracle here because it's a reference benchmark
# (analytical-posterior upper bound on regime-only methods), not one of
# the meta-RL methods being characterised.
_REFERENCE_METHODS = {"regime_agnostic_ppo", "belief_ppo", "oracle_ppo"}

_AXIS_LABELS = {
    "persistence":        "Persistence (mean regime duration)",
    "distinguishability": "Distinguishability (regime fill separation)",
}


def _gap_closed(mean: float, floor: float, ceiling: float) -> float:
    if np.isnan(floor) or np.isnan(ceiling) or ceiling == floor:
        return float("nan")
    return (mean - floor) / (ceiling - floor)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2_000) -> tuple[float, float]:
    if values.size <= 1:
        m = float(values[0]) if values.size == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(0)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _per_method_curves(
    axis_results: dict[str, dict[str, dict]],
) -> dict[str, dict[str, Any]]:
    """For each method, collect per-level absolute return + CI and gap_closed + CI."""
    out: dict[str, dict[str, Any]] = {}
    for level in _LEVELS:
        cells = axis_results.get(level, {})
        floor = cells.get("regime_agnostic_ppo", {}).get("final_return_mean", float("nan"))
        oracle = cells.get("oracle_ppo", {}).get("final_return_mean", float("nan"))
        for method, cell in cells.items():
            per_seed = np.asarray(cell.get("per_seed_final_return", []), dtype=float)
            mean = float(cell["final_return_mean"])
            abs_lo, abs_hi = _bootstrap_ci(per_seed) if per_seed.size >= 2 else (mean, mean)

            gc_mean = _gap_closed(mean, floor, oracle)
            if (per_seed.size >= 2 and not np.isnan(floor) and not np.isnan(oracle)
                    and oracle != floor):
                gc_seeds = (per_seed - floor) / (oracle - floor)
                gc_lo, gc_hi = _bootstrap_ci(gc_seeds)
            else:
                gc_lo, gc_hi = gc_mean, gc_mean

            entry = out.setdefault(method, {
                "levels": [], "abs_mean": [], "abs_lo": [], "abs_hi": [],
                "gc_mean": [], "gc_lo": [], "gc_hi": [],
            })
            entry["levels"].append(level)
            entry["abs_mean"].append(mean)
            entry["abs_lo"].append(abs_lo)
            entry["abs_hi"].append(abs_hi)
            entry["gc_mean"].append(gc_mean)
            entry["gc_lo"].append(gc_lo)
            entry["gc_hi"].append(gc_hi)
    return out


def _budget_from_first_cell(axis_results: dict[str, dict[str, dict]]) -> dict[str, Any] | None:
    for level_cells in axis_results.values():
        for cell in level_cells.values():
            mp = RESULTS_ROOT / cell["experiment_name"] / "metrics.json"
            if mp.exists():
                with open(mp) as f:
                    m = json.load(f)
                return {
                    "iterations": int(m.get("iterations", 0)) or None,
                    "parallel_envs": int(m.get("parallel_envs", 0)) or None,
                    "rollout_length": int(m.get("rollout_length", 0)) or None,
                    "num_seeds": int(m.get("num_seeds", 0)) or None,
                }
    return None


def _aligned_curve(entry: dict[str, Any], key_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a method's per-level curve with NaN on missing levels so plots
    render correctly even if a cell is missing for some axis level."""
    means = [np.nan] * len(_LEVELS)
    los = [np.nan] * len(_LEVELS)
    his = [np.nan] * len(_LEVELS)
    for level, m, lo, hi in zip(
        entry["levels"], entry[f"{key_prefix}_mean"],
        entry[f"{key_prefix}_lo"], entry[f"{key_prefix}_hi"],
    ):
        i = _LEVELS.index(level)
        means[i] = m
        los[i] = lo
        his[i] = hi
    return np.asarray(means), np.asarray(los), np.asarray(his)


def plot_difficulty_sweep(axis: str, out_path: Path) -> bool:
    """Single-panel sweep figure: gap_closed across difficulty levels.

    The original two-panel layout (absolute returns + gap_closed) ran into
    visual redundancy on the distinguishability axis where both panels
    showed similar monotonic declines. gap_closed is the canonical RQ3
    metric (it normalises method return by the optimality gap and ties
    directly to the pre-registered hypotheses), so the single-panel form
    keeps that and folds the absolute floor / oracle values into the
    legend so context isn't lost.
    """
    stats_path = RESULTS_ROOT / "milestones" / "M6" / "stats_M6_sweep.json"
    if not stats_path.exists():
        print(f"[m6_plots] skip {axis}_sweep: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    axis_results = stats.get("results", {}).get(axis)
    if not axis_results:
        print(f"[m6_plots] skip {axis}_sweep: no '{axis}' axis data in stats")
        return False

    by_method = _per_method_curves(axis_results)

    apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(_LEVELS))

    # Plot every method (refs dashed, meta-RL solid). Belief-PPO is a
    # reference benchmark on this panel — dashed like regime-agnostic
    # and Oracle.
    for method in _ABS_METHOD_ORDER:
        entry = by_method.get(method)
        if entry is None:
            continue
        means, los, his = _aligned_curve(entry, "gc")
        color = _METHOD_COLORS[method]
        is_ref = method in _REFERENCE_METHODS
        linestyle = "--" if is_ref else "-"
        ax.plot(x, means, marker="o", color=color, linestyle=linestyle,
                linewidth=1.6 if not is_ref else 1.2,
                alpha=1.0 if not is_ref else 0.8,
                label=_METHOD_LABELS[method])
        valid = ~np.isnan(los) & ~np.isnan(his) & (los != his)
        if valid.any():
            ax.fill_between(x, los, his, color=color,
                            alpha=0.15 if not is_ref else 0.10)

    # Constant-by-construction reference dashes for the 0-1 envelope.
    # regime_agnostic and oracle method lines already sit on these,
    # but the dashes give the chart a clear "[floor — oracle]" frame.
    ax.axhline(0.0, color=_METHOD_COLORS["regime_agnostic_ppo"],
               linestyle=":", linewidth=0.9, alpha=0.5)
    ax.axhline(1.0, color=_METHOD_COLORS["oracle_ppo"],
               linestyle=":", linewidth=0.9, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([_LEVEL_LABELS[lv] for lv in _LEVELS])
    ax.set_xlabel(_AXIS_LABELS.get(axis, axis))
    ax.set_ylabel(
        "Gap closed  =  (method return − floor) / (oracle − floor)"
    )

    axis_pretty = axis.capitalize()
    ax.set_title(
        f"MarketMakingV1 — {axis_pretty} difficulty sweep\n"
        "Gap closed across difficulty (mean across seeds, shaded = 95% CI)"
    )

    # Build legend with absolute floor / oracle values per level so the
    # absolute scale isn't lost when we drop the absolute panel.
    legend_handles: list[Any] = []
    for method in _ABS_METHOD_ORDER:
        if method not in by_method:
            continue
        is_ref = method in _REFERENCE_METHODS
        legend_handles.append(Line2D(
            [0], [0],
            color=_METHOD_COLORS[method],
            linestyle="--" if is_ref else "-",
            linewidth=1.6, marker="o",
            label=_METHOD_LABELS[method],
        ))
    floor_per_level = []
    oracle_per_level = []
    for lv in _LEVELS:
        cells = axis_results.get(lv, {})
        f = cells.get("regime_agnostic_ppo", {}).get("final_return_mean")
        o = cells.get("oracle_ppo", {}).get("final_return_mean")
        floor_per_level.append(f"{f:.0f}" if f is not None else "?")
        oracle_per_level.append(f"{o:.0f}" if o is not None else "?")
    legend_handles.append(Line2D(
        [0], [0], color="white",
        label=f"Floor:  {' / '.join(floor_per_level)}",
    ))
    legend_handles.append(Line2D(
        [0], [0], color="white",
        label=f"Oracle: {' / '.join(oracle_per_level)}",
    ))

    ax.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True, facecolor="white",
        edgecolor="#cccccc", framealpha=1.0,
    )

    budget = _budget_from_first_cell(axis_results)
    if budget:
        budget_annotation(fig, **budget,
                          extra=f"7 methods × 3 {axis} levels")
    fig.tight_layout()
    fig.subplots_adjust(right=0.72, bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m6_plots] wrote {out_path}")
    return True


def plot_posterior_vs_performance(out_path: Path) -> bool:
    """Scatter of posterior_error vs gap_closed across (method, axis, level, seed) cells."""
    stats_path = RESULTS_ROOT / "milestones" / "M6" / "stats_M6_posterior_vs_performance.json"
    if not stats_path.exists():
        print(f"[m6_plots] skip posterior_vs_performance: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    points = stats.get("scatter_points", [])
    if not points:
        print("[m6_plots] skip posterior_vs_performance: no scatter_points")
        return False

    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    by_method: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        by_method.setdefault(p["method"], []).append(
            (float(p["posterior_error"]), float(p["gap_closed"])),
        )
    # Plot meta-RL methods, in order so that hypernet (upper band) draws
    # on top of concat (lower band) for any near-overlapping points.
    for method in _GC_METHOD_ORDER:
        pts = by_method.get(method)
        if pts is None:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, color=_METHOD_COLORS.get(method, "#666666"),
                   label=_METHOD_LABELS.get(method, method), s=32,
                   edgecolor="black", linewidth=0.4, alpha=0.85)

    # Reference horizontal lines: gap_closed = 0 (floor) and = 1 (Oracle).
    ax.axhline(0.0, color=_METHOD_COLORS["regime_agnostic_ppo"],
               linestyle="--", linewidth=1.0, alpha=0.6,
               label="Regime-agnostic PPO floor (gap_closed = 0)")
    ax.axhline(1.0, color=_METHOD_COLORS["oracle_ppo"],
               linestyle="--", linewidth=1.0, alpha=0.6,
               label="Oracle-PPO ceiling (gap_closed = 1)")

    ax.set_xlabel(
        "Posterior error  =  analytical HMM probe accuracy  −  method probe accuracy\n"
        "(low → method's belief decodes regime almost as well as analytical)"
    )
    ax.set_ylabel(
        "Gap closed  =  (method return − floor) / (oracle − floor)\n"
        "(0 = floor, 1 = oracle)"
    )

    # Pull headline correlation from the same stats JSON for an inline
    # annotation. Keeps the chart self-explanatory.
    corr_overall = stats.get("correlation_overall", float("nan"))
    n_points = stats.get("n_scatter_points", len(points))
    ax.set_title(
        "MarketMakingV1 — posterior quality vs task performance\n"
        f"Pearson r = {corr_overall:+.3f} across n = {n_points} (cell × seed) points"
        " · the policy interface dominates"
    )

    # Region annotations — describe what the two horizontal bands mean.
    # Place text inside the data area in a corner that's empty.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_text = xmin + 0.62 * (xmax - xmin)
    ax.annotate(
        "Hypernetwork integration\nclears or exceeds the floor",
        xy=(x_text, 0.85), xycoords="data",
        ha="left", va="center", fontsize=8.5,
        color="#444444", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )
    ax.annotate(
        "Concat integration\nstays below the floor\n(M5 decoupling, replicated)",
        xy=(x_text, -0.6), xycoords="data",
        ha="left", va="center", fontsize=8.5,
        color="#444444", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )

    # Compute footer — tells the reader the data backing this chart.
    budget_annotation(
        fig,
        rollout_length=int(stats.get("rollout_length", 0)) or None,
        extra=(
            f"4 meta-RL cells × 6 (axis × level) × n=8 seeds = {n_points} points · "
            f"classifier={stats.get('classifier', 'logistic')}"
        ),
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.72, bottom=0.22, top=0.86)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m6_plots] wrote {out_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(prog="plotting.m6_plots")
    parser.add_argument(
        "--out-dir", default=str(FIGURES_ROOT),
        help="output directory for PNGs (default: figures/milestones/M6/)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, fn in [
        ("rq3_persistence_sweep.png",        lambda p: plot_difficulty_sweep("persistence", p)),
        ("rq3_distinguishability_sweep.png", lambda p: plot_difficulty_sweep("distinguishability", p)),
        ("rq3_posterior_vs_performance.png", plot_posterior_vs_performance),
    ]:
        if fn(out_dir / name):
            written += 1
    print(f"[m6_plots] wrote {written} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
