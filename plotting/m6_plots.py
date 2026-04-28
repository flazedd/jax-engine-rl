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
    """Two-panel sweep figure for one axis (persistence or distinguishability)."""
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
    fig, (ax_abs, ax_gc) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), sharex=True,
    )
    x = np.arange(len(_LEVELS))

    # ----- Left panel: absolute returns ------------------------------------
    for method in _ABS_METHOD_ORDER:
        entry = by_method.get(method)
        if entry is None:
            continue
        means, los, his = _aligned_curve(entry, "abs")
        color = _METHOD_COLORS[method]
        is_ref = method in _REFERENCE_METHODS
        linestyle = "--" if is_ref else "-"
        ax_abs.plot(x, means, marker="o", color=color, linestyle=linestyle,
                    linewidth=1.6 if not is_ref else 1.2,
                    alpha=1.0 if not is_ref else 0.8,
                    label=_METHOD_LABELS[method])
        valid = ~np.isnan(los) & ~np.isnan(his) & (los != his)
        if valid.any():
            ax_abs.fill_between(x, los, his, color=color, alpha=0.12)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels([_LEVEL_LABELS[lv] for lv in _LEVELS])
    ax_abs.set_xlabel(_AXIS_LABELS.get(axis, axis))
    ax_abs.set_title(
        "Absolute returns by difficulty\n"
        "Mean episode return (shaded = 95% CI)"
    )

    # ----- Right panel: gap_closed -----------------------------------------
    for method in _GC_METHOD_ORDER:
        entry = by_method.get(method)
        if entry is None:
            continue
        means, los, his = _aligned_curve(entry, "gc")
        color = _METHOD_COLORS[method]
        is_ref = method in _REFERENCE_METHODS
        linestyle = "--" if is_ref else "-"
        ax_gc.plot(x, means, marker="o", color=color, linestyle=linestyle,
                   linewidth=1.6 if not is_ref else 1.2,
                   alpha=1.0 if not is_ref else 0.8,
                   label=_METHOD_LABELS[method])
        valid = ~np.isnan(los) & ~np.isnan(his) & (los != his)
        if valid.any():
            ax_gc.fill_between(x, los, his, color=color,
                               alpha=0.15 if not is_ref else 0.10)
    # Constant-by-construction references: regime_agnostic at 0 and
    # oracle at 1. Dashed in their canonical project colours so the
    # legend reads consistently with the absolute panel.
    ax_gc.axhline(0.0, color=_METHOD_COLORS["regime_agnostic_ppo"],
                  linestyle="--", linewidth=1.0, alpha=0.6,
                  label="Regime-agnostic PPO floor (= 0)")
    ax_gc.axhline(1.0, color=_METHOD_COLORS["oracle_ppo"],
                  linestyle="--", linewidth=1.0, alpha=0.6,
                  label="Oracle-PPO ceiling (= 1)")

    ax_gc.set_xticks(x)
    ax_gc.set_xticklabels([_LEVEL_LABELS[lv] for lv in _LEVELS])
    ax_gc.set_xlabel(_AXIS_LABELS.get(axis, axis))
    ax_gc.set_title(
        "Gap closed by difficulty\n"
        "(mean − floor) / (oracle − floor)"
    )

    # ----- Shared figure-level title + outside legend ----------------------
    axis_pretty = axis.capitalize()
    fig.suptitle(
        f"MarketMakingV1 — {axis_pretty} difficulty sweep",
        y=1.02, fontsize=12,
    )

    # Build a single shared legend covering every method that appears
    # in either panel. Reference methods (regime-agnostic, Belief,
    # Oracle) are dashed in the legend too — matches both panels.
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
    fig.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0.84, 0.95),
        fontsize=8, frameon=True, facecolor="white",
        edgecolor="#cccccc", framealpha=1.0,
    )

    budget = _budget_from_first_cell(axis_results)
    if budget:
        budget_annotation(fig, **budget,
                          extra=f"7 methods × 3 {axis} levels")
    fig.tight_layout()
    fig.subplots_adjust(right=0.83, bottom=0.18, top=0.85)
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
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    by_method: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        by_method.setdefault(p["method"], []).append(
            (float(p["posterior_error"]), float(p["gap_closed"])),
        )
    for method, pts in by_method.items():
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, color=_METHOD_COLORS.get(method, "#666666"),
                   label=_METHOD_LABELS.get(method, method), s=30,
                   edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.set_xlabel("Posterior error (vs analytical HMM)")
    ax.set_title(
        "MarketMakingV1 — posterior quality vs task performance\n"
        "Gap closed (per cell, per seed)"
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.65, bottom=0.18)
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
