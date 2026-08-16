"""M5R figure regeneration.

Produces four figures from the matched-tuning final-eval results:
  - m5r_method_ladder.png         — 4-cell ladder on E_final
  - m5r_persistence_sweep.png     — 3-panel persistence sweep
  - m5r_distinguishability_sweep.png — 3-panel distinguishability sweep
  - m5r_posterior_vs_performance.png — scatter from logistic probe

The figures are written directly into the thesis figure directory so the
PDF picks them up on the next build.

Usage:
  uv run python -m plotting.m5r_plots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plotting.style import (
    COLORS, FIGSIZE_STANDARD, FIGSIZE_WIDE, PALETTE, apply_style, polish, ref_line,
)

# Stacked-obs PPO is not in the COLORS dict; pick a distinct gray.
STACKED_OBS_COLOR = "#8a949c"

# Brand cell colours for the sweep ladders: concat in the slate family,
# hypernet in the teal family (matching the rest of the thesis figures).
_CELL_BRAND = {
    "rl2_concat": "#b4bcc2", "varibad_concat": "#8a949c",
    "rl2_hypernet": "#2a9d8f", "varibad_hypernet": "#73b8ad",
}

# The canonical variant palette (docs/plotting.md → Colour): hue for the
# conditioning architecture, shade for the method, the darker tone being RL².
# `_CELL_BRAND` predates it and orders the two slates the other way round; the
# bar ladders still read from it.
_VARIANT_COLOR = {
    "rl2_hypernet": "#1d7870", "varibad_hypernet": "#7ec8bd",
    "rl2_concat": "#6b757d", "varibad_concat": "#c3cad0",
}

from evaluation.action_distribution import regime_separation_per_seed
from evaluation.protocol import MEDIUM_ENV
from utils.paths import experiment_dir, fig_targets, final_dir, project_fig_dir, resolve_data, results_root

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = results_root()
FINAL_DIR = final_dir()
def _both_targets(name: str) -> list[Path]:
    """Both destinations for a chart: the repo tree and the thesis tree.

    The relative path comes from utils.paths.FIGURE_HOME, so a chart lands
    under the research question it answers in both trees.
    """
    return fig_targets(name)

CELLS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
CELL_LABEL = {
    "rl2_concat": "RL² Concat",
    "rl2_hypernet": "RL² Hypernet",
    "varibad_concat": "VariBAD Concat",
    "varibad_hypernet": "VariBAD Hypernet",
}
PERSISTENCE_LEVELS = (
    ("persistence_easy", "Easy ($P_{ii}=0.99$)"),
    (MEDIUM_ENV, "Medium ($P_{ii}=0.995$)"),
    ("persistence_hard", "Hard ($P_{ii}=0.96$)"),
    ("persistence_very_hard", "Very-hard ($P_{ii}=0.92$)"),
)
DISTINGUISHABILITY_LEVELS = (
    ("distinguishability_easy", "Easy"),
    (MEDIUM_ENV, "Medium"),
    ("distinguishability_hard", "Hard"),
)
# The figure the thesis includes as m5r_sweep_n20.png: the three
# distinguishability levels plus the coupled fast-persistence instance, which
# is what its caption describes.
THESIS_SWEEP_LEVELS = (
    ("distinguishability_easy", "Easy"),
    (MEDIUM_ENV, "Medium"),
    ("distinguishability_hard", "Hard"),
    ("coupled_fast", "Coupled fast persistence ($P_{ii}=0.95$)"),
)
KAPPA_LEVELS = (
    ("kappa02", "$\\kappa = 0.02$"),
    (MEDIUM_ENV, "$\\kappa = 0.05$"),
    ("kappa10", "$\\kappa = 0.10$"),
    ("kappa20", "$\\kappa = 0.20$"),
)


def _load() -> dict:
    with open(resolve_data(FINAL_DIR / "per_cell_env.json")) as f:
        return json.load(f)


def _load_probe(classifier: str = "logistic") -> dict | None:
    suffix = "" if classifier == "logistic" else f"_{classifier}"
    p = FINAL_DIR / f"m5r_posterior_vs_performance{suffix}.json"
    if not resolve_data(p).exists():
        return None
    with open(resolve_data(p)) as f:
        return json.load(f)


def _load_stacked_obs() -> dict | None:
    """Stacked-obs level, read from the matched reference run.

    This used to read `m5r_stacked_obs_sweep.json`, which trains its own
    stacked-obs agents on the unaugmented environment from a separately written
    config, at a smaller budget and seed count. Drawing that beside arms trained
    under the matched protocol compares agents that differ in inputs, budget and
    seeds as though they differed only in method. The matched reference arm is
    the one the protocol defines, so the figure reads it directly.
    """
    p = experiment_dir(f"m5r_ref_stacked_obs_{MEDIUM_ENV}") / "metrics.json"
    if not resolve_data(p).exists():
        return None
    with open(resolve_data(p)) as f:
        vals = np.asarray(json.load(f)["per_seed_final_return"], dtype=float)
    if vals.size == 0:
        return None
    rng = np.random.default_rng(0)
    boot = vals[rng.integers(0, vals.size, (10_000, vals.size))].mean(axis=1)
    return {"by_env": {MEDIUM_ENV: {
        "final_return_mean": float(vals.mean()),
        "final_return_ci95": [float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5))],
        "n_seeds": int(vals.size),
    }}}


def _draw_refs(ax, refs: dict, label_x: float | None = None) -> None:
    """Add horizontal reference lines for floor / belief / oracle. If `label_x`
    is given (a data x-coordinate to the right of the bars), inline labels are
    placed there, left-aligned, as in the method-ladder figure."""
    # Shaded zones: floor -> belief (recoverable gap) and belief -> oracle (inference
    # remainder), matching the ladder figure.
    _fl, _be, _or = (refs.get("regime_agnostic_ppo"), refs.get("belief_ppo"),
                     refs.get("oracle_ppo"))
    if _fl is not None and _be is not None:
        ax.axhspan(_fl, _be, color=PALETTE["hyper"], alpha=0.08, zorder=0)
    if _be is not None and _or is not None:
        ax.axhspan(_be, _or, color=PALETTE["hyper"], alpha=0.04, zorder=0)
    label_map = {
        "regime_agnostic_ppo": "Regime-agnostic floor",
        "belief_ppo": "Belief-PPO",
        "oracle_ppo": "Oracle-PPO",
    }
    # Brand convention: reference lines in neutral grey, distinguished by linestyle.
    color_map = {
        "regime_agnostic_ppo": "#555555",
        "belief_ppo": "#555555",
        "oracle_ppo": "#555555",
    }
    style_map = {
        "regime_agnostic_ppo": "-",
        "belief_ppo": (0, (1, 1.5)),
        "oracle_ppo": (0, (6, 2)),
    }
    for key, label in label_map.items():
        v = refs.get(key)
        if v is None:
            continue
        ax.axhline(v, linestyle=style_map[key], color=color_map[key], linewidth=1.2,
                   zorder=2)
        if label_x is not None:
            ax.text(label_x, v, f" {label}", va="center", ha="left", fontsize=7.5,
                    color="#444444", zorder=5, clip_on=False,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))


def _ladder_bars(
    ax, env_block: dict, env_label: str | None = None,
    stacked_data: dict | None = None,
    with_ylim_floor: bool = True,
) -> None:
    """Plot the ladder with stacked-obs as a 5th group plus CIs.

    The x-axis has three positions: stacked-obs, RL², VariBAD. Stacked-obs
    is one bar at position 0; the meta-RL groups have two offset bars each.
    `env_label` and `stacked_data` are required to plot the stacked-obs bar;
    if either is None or no data exists for that env, the stacked group is
    skipped silently (so legacy callers still get the 4-cell view).
    """
    cells = env_block["cells"]
    refs = env_block["refs"]
    methods = ("rl2", "varibad")
    integrations = ("concat", "hypernet")
    bar_w = 0.38

    # Decide whether to include the stacked-obs group.
    stacked_cell = None
    if stacked_data is not None and env_label is not None:
        stacked_cell = stacked_data.get("by_env", {}).get(env_label)

    show_stacked = stacked_cell is not None
    method_x = np.arange(len(methods)) + (1.0 if show_stacked else 0.0)

    if show_stacked:
        m = stacked_cell["final_return_mean"]
        lo, hi = stacked_cell["final_return_ci95"]
        ax.bar(
            [0.0], [m], bar_w, yerr=[[m - lo], [hi - m]], capsize=3,
            edgecolor="black", linewidth=0.4, color=STACKED_OBS_COLOR,
        )
        if not np.isnan(m):
            # Place label above the upper CI cap, not above the bar top, so
            # the annotation cannot collide with the error-bar whisker.
            ax.annotate(
                f"{m:.1f}", xy=(0.0, hi), xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    for i_int, integ in enumerate(integrations):
        means, lo_err, hi_err = [], [], []
        for method in methods:
            key = f"{method}_{integ}"
            c = cells.get(key)
            if c is None:
                means.append(np.nan); lo_err.append(0); hi_err.append(0)
                continue
            m = c["final_return_mean"]
            lo, hi = c["final_return_ci95"]
            means.append(m); lo_err.append(m - lo); hi_err.append(hi - m)
        offset = (i_int - 0.5) * bar_w
        x = method_x + offset
        bars = ax.bar(
            x, means, bar_w, yerr=[lo_err, hi_err], capsize=3,
            edgecolor="white", linewidth=1.0,
        )
        for bar, method in zip(bars, methods):
            bar.set_facecolor(_CELL_BRAND[f"{method}_{integ}"])
        for xi, m, he in zip(x, means, hi_err):
            if not np.isnan(m):
                # Label sits above the upper CI cap, not the bar top.
                ax.annotate(
                    f"{m:.1f}", xy=(xi, m + he), xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    if show_stacked:
        ax.set_xticks([0.0, *method_x])
        ax.set_xticklabels(["Stacked-obs", "RL²", "VariBAD"])
    else:
        ax.set_xticks(method_x)
        ax.set_xticklabels(["RL²", "VariBAD"])
    # Place reference-line labels in a right-hand margin, clear of the bars
    # (matching the method-ladder figure).
    label_x = method_x[-1] + 0.5
    ax.set_xlim(-0.55, label_x + 1.15)
    _draw_refs(ax, refs, label_x=label_x)
    if with_ylim_floor:
        floor = refs.get("regime_agnostic_ppo", float("nan"))
        oracle = refs.get("oracle_ppo", float("nan"))
        if not np.isnan(floor):
            finite = [c["final_return_mean"]
                      for c in cells.values()
                      if not np.isnan(c.get("final_return_mean", np.nan))]
            if show_stacked and not np.isnan(stacked_cell["final_return_mean"]):
                finite.append(stacked_cell["final_return_mean"])
            ymin = min([floor, *finite]) - 6
            ymax = max([oracle if not np.isnan(oracle) else 200, *finite, floor]) + 6
            ax.set_ylim(ymin, ymax)


def _legend_handles(include_stacked: bool = True) -> list[Patch]:
    methods = ("rl2", "varibad")
    integrations = ("concat", "hypernet")
    method_full = {"rl2": "RL²", "varibad": "VariBAD"}
    integ_pretty = {"concat": "Concat", "hypernet": "Hypernet"}
    handles = []
    if include_stacked:
        handles.append(Patch(
            facecolor=STACKED_OBS_COLOR, edgecolor="black", linewidth=0.4,
            label="Stacked-obs PPO",
        ))
    for m in methods:
        for i in integrations:
            handles.append(Patch(
                facecolor=_CELL_BRAND[f"{m}_{i}"], edgecolor="white", linewidth=1.0,
                label=f"{method_full[m]} {integ_pretty[i]}",
            ))
    return handles


def plot_method_ladder(out_path: Path) -> None:
    """Hero ladder figure: spotlights the hypernet 'winners' in a vivid accent
    against muted baselines, with the floor-to-ceiling gap shaded as the
    recoverable region so the story reads at a glance."""
    apply_style()
    data = _load()
    stacked = _load_stacked_obs()
    env_block = data["per_env"][MEDIUM_ENV]
    cells = env_block["cells"]
    refs = env_block["refs"]
    floor = refs["regime_agnostic_ppo"]
    belief = refs["belief_ppo"]
    oracle = refs["oracle_ppo"]

    C_HYPER = "#2a9d8f"   # vivid teal — the heroes
    C_CONCAT = "#b4bcc2"  # muted slate — falls short
    C_STACK = "#8a949c"   # neutral baseline

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    # Shaded reference zones: the recoverable gap (floor -> belief) and the
    # smaller irreducible inference remainder (belief -> oracle).
    ax.axhspan(floor, belief, color=C_HYPER, alpha=0.09, zorder=0)
    ax.axhspan(belief, oracle, color=C_HYPER, alpha=0.04, zorder=0)

    bw = 0.34

    def _bar(x, cell, color):
        m = cell["final_return_mean"]
        lo, hi = cell["final_return_ci95"]
        ax.bar(x, m, bw, yerr=[[m - lo], [hi - m]], capsize=4, color=color,
               edgecolor="white", linewidth=1.2, zorder=3,
               error_kw={"ecolor": "#3a3a3a", "elinewidth": 1.2})
        ax.annotate(f"{m:.1f}", xy=(x, hi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#333333")

    stacked_cell = (stacked or {}).get("by_env", {}).get(MEDIUM_ENV)
    if stacked_cell is not None:
        _bar(0.0, stacked_cell, C_STACK)
    for i_m, method in enumerate(["rl2", "varibad"]):
        xc = i_m + 1
        _bar(xc - bw * 0.62, cells[f"{method}_concat"], C_CONCAT)
        _bar(xc + bw * 0.62, cells[f"{method}_hypernet"], C_HYPER)

    # Reference lines with inline labels at the right edge (no legend clutter).
    xr = 2.62
    for v, lab, ls in [(oracle, "Oracle-PPO", (0, (6, 2))),
                       (belief, "Belief-PPO", (0, (1, 1.5))),
                       (floor, "Regime-agnostic floor", "solid")]:
        ax.axhline(v, color="#555555", linewidth=1.1, linestyle=ls, zorder=2)
        ax.text(xr, v, f" {lab}", va="center", ha="left", fontsize=8.5,
                color="#444444", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Stacked-obs", "RL²", "VariBAD"], fontsize=11)
    ax.set_ylabel("Return at the end of training", fontsize=11)
    ax.set_ylim(floor - 36, oracle + 7)
    ax.set_xlim(-0.55, 3.7)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [Patch(facecolor=C_HYPER, edgecolor="white", label="Hypernet integration"),
               Patch(facecolor=C_CONCAT, edgecolor="white", label="Concat integration"),
               Patch(facecolor=C_STACK, edgecolor="white", label="Stacked-obs PPO")]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_sweep(axis_levels: tuple, suptitle: str, out_path: Path) -> None:
    apply_style()
    data = _load()
    stacked = _load_stacked_obs()
    # Only plot levels that exist. Figure stages now run as soon as their
    # dependencies land, so a sweep figure can be attempted before the sweep
    # has produced its environments; dying on the first missing key would take
    # every other figure in this module down with it.
    available = set(data.get("per_env", {}))
    axis_levels = tuple(lv for lv in axis_levels if lv[0] in available)
    if not axis_levels:
        print(f"[m5r_plots] skip {out_path.name}: none of its levels exist yet")
        return
    n_panels = len(axis_levels)
    if n_panels == 4:
        nrows, ncols = 2, 2
        figsize = (12, 8)
    else:
        nrows, ncols = n_panels, 1
        figsize = (10, 4 * n_panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=False, sharey=False)
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (env_label, level_label) in zip(flat_axes, axis_levels):
        env_block = data["per_env"][env_label]
        _ladder_bars(ax, env_block, env_label=env_label, stacked_data=stacked)
        ax.set_title(level_label, fontsize=11)
        ax.set_ylabel("Final return", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
    ref_handles, ref_labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles=_legend_handles() + ref_handles,
        loc="lower center", ncol=4, fontsize=10,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_posterior_vs_performance(
    out_path: Path, classifier: str = "logistic", metric: str = "kl",
) -> None:
    """Scatter of belief quality against gap-closed, one point per run.

    `classifier="logistic"` plots the linear probe; `classifier="mlp"` plots
    the two-layer MLP probe. The two figures share axes ranges and styling
    so they can be displayed side by side.

    `metric="kl"` puts the primary belief-quality measure on the x-axis, the
    excess of the forward KL to the analytical posterior over that posterior's
    own residual, so zero is a belief indistinguishable from it. `metric="acc"`
    puts the secondary decodability measure there, as the shortfall in probe
    accuracy against the same posterior, subtracted from one so that better
    beliefs sit to the right in both cases.
    """
    if metric not in ("kl", "acc"):
        raise ValueError(f"metric must be 'kl' or 'acc', got {metric!r}")
    apply_style()
    probe = _load_probe(classifier)
    if probe is None:
        print(f"[m5r_plots] skip posterior_vs_performance ({classifier}): "
              "probe data missing")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    by_method = {m: [] for m in CELLS}
    for p in probe["scatter_points"]:
        by_method.setdefault(p["method"], []).append(p)
    # Hue for the conditioning architecture, shade for the method, as in the
    # learning-curve figure. Marker shape repeats the method, which a scatter
    # needs because shade alone is hard to judge on isolated points.
    marker = {"rl2_concat": "o", "rl2_hypernet": "o",
              "varibad_concat": "^", "varibad_hypernet": "^"}
    for m in CELLS:
        pts = by_method[m]
        if not pts:
            continue
        if metric == "kl":
            xs = [p["belief_error_kl"] for p in pts]
        else:
            xs = [1.0 - p["posterior_error"] for p in pts]
        ys = [p["gap_closed"] for p in pts]
        ax.scatter(
            xs, ys, color=_VARIANT_COLOR[m], s=30, alpha=0.85, marker=marker[m],
            edgecolor="#33403f", linewidth=0.4, label=CELL_LABEL[m],
        )
    probe_label = "linear probe" if classifier == "logistic" else "MLP probe"
    # Shared x-range across the linear and MLP variants so the two figures are
    # directly comparable when shown side by side, padded so that points at the
    # end of the range are not clipped by the spine. Ranges across both probes:
    # excess KL is roughly [0.06, 0.64], decodability [0.42, 0.95].
    if metric == "kl":
        ax.set_xlabel(f"Excess KL to the analytical posterior, {probe_label}")
        ax.set_xlim(-0.02, 0.70)
        ax.axvline(0.0, color=PALETTE["analytical"], linewidth=1.2,
                   linestyle="--", label="Analytical posterior")
    else:
        ax.set_xlabel(f"Regime decodability against the analytical posterior, "
                      f"{probe_label}")
        ax.set_xlim(0.40, 1.02)
        ax.axvline(1.0, color=PALETTE["analytical"], linewidth=1.2,
                   linestyle="--", label="Analytical posterior")
    ax.set_ylabel("Gap-closed fraction")
    ax.axhline(0.0, color="#555555", linewidth=1.0,
               label="Regime-agnostic floor")
    # The pooled correlation is annotated rather than written into the thesis
    # prose, which stays qualitative. It is a between-method relation: the
    # within-method clouds carry no comparable trend.
    r = probe["correlation_overall_kl" if metric == "kl" else "correlation_overall"]
    ax.text(0.97, 0.96, f"Across all runs, r = {r:+.2f}".replace("-", "−"),
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#444444")
    polish(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_m5r_learning_curves(out_path: Path) -> None:
    """Per-iteration learning curves for the four meta-RL cells on the
    reference instance, drawn from the matched-compute Stage C
    metrics. Reference horizontal lines for the regime-agnostic floor and
    the Belief-PPO and Oracle-PPO ceilings."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    data = _load()
    refs = data["per_env"][MEDIUM_ENV]["refs"]
    floor = refs["regime_agnostic_ppo"]
    belief = refs["belief_ppo"]
    oracle = refs["oracle_ppo"]

    # Colour by integration (concat = muted slate, hypernet = teal), method by
    # linestyle (RL² solid, VariBAD dashed), so "hypernet climbs, concat stays
    # low" reads at a glance.
    # One colour per curve, all solid. Dash patterns were used to encode the
    # method, but at the line width these curves need they are hard to tell
    # apart, and the reader has to decode two channels at once. Hue keeps the
    # ladder figure's convention, teal for hypernetwork and slate for
    # concatenation, and shade separates the two methods within each family.
    cell_style = {
        "rl2_hypernet":     ("#1d7870", "-", "RL² Hypernet"),
        "varibad_hypernet": ("#7ec8bd", "-", "VariBAD Hypernet"),
        "rl2_concat":       ("#6b757d", "-", "RL² Concat"),
        "varibad_concat":   ("#c3cad0", "-", "VariBAD Concat"),
    }
    cell_specs = [
        (cell, label, color, ls,
         experiment_dir(f"m5r_final_{cell}_{MEDIUM_ENV}") / "metrics.json")
        for cell, (color, ls, label) in cell_style.items()
    ]
    # Matched-family stacked-obs run, so its curve sits on the same per-step
    # tuple, optimiser settings, budget and capacity as the variant curves.
    cell_specs.append((
        "stacked_obs", "Stacked-obs PPO", "#e09f3e", "-",
        experiment_dir(f"m5r_ref_stacked_obs_{MEDIUM_ENV}") / "metrics.json",
    ))
    last_iter = 0
    end_labels: list[tuple[int, float, str, str]] = []
    for cell_key, label, color, ls, m_path in cell_specs:
        if not resolve_data(m_path).exists():
            continue
        with open(resolve_data(m_path)) as f:
            m = json.load(f)
        per_seed = np.asarray(m["per_seed_mean_return_per_iter"])  # [seeds, T]
        mean = per_seed.mean(axis=0)
        n_seeds = per_seed.shape[0]
        rng = np.random.default_rng(0)
        if n_seeds > 1:
            T = per_seed.shape[1]
            idx = rng.integers(0, n_seeds, size=(1000, n_seeds))
            lo = np.zeros(T); hi = np.zeros(T)
            for t in range(T):
                vals = per_seed[idx, t].mean(axis=1)
                lo[t] = np.percentile(vals, 2.5)
                hi[t] = np.percentile(vals, 97.5)
        else:
            lo, hi = mean.copy(), mean.copy()
        iters = np.arange(len(mean))
        last_iter = max(last_iter, len(mean) - 1)
        # Thin lines and no confidence band: with six curves the bands overlap
        # into a single wash that obscures the ordering the figure exists to
        # show. Seed uncertainty is reported in the tables instead.
        ax.plot(iters, mean, color=color, label=label, linewidth=1.3, linestyle=ls)
        end_labels.append((len(mean) - 1, float(mean[-1]), label, color))

    # Curves that finish close together would print their labels on top of one
    # another, so the labels are pushed apart vertically and leadered back to the
    # curve end. The reference lines label further right, clear of these.
    span = max(oracle - floor, 1.0)
    min_gap = 0.055 * span
    placed: list[float] = []
    for x_end, y_end, label, color in sorted(end_labels, key=lambda r: -r[1]):
        y_lab = y_end
        for prev in placed:
            if abs(y_lab - prev) < min_gap:
                y_lab = prev - min_gap
        placed.append(y_lab)
        ax.annotate(label, xy=(x_end, y_end), xytext=(x_end + 0.035 * last_iter, y_lab),
                    textcoords="data", va="center", ha="left", fontsize=9,
                    color=color, fontweight="bold", clip_on=False, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                                    alpha=0.5, shrinkA=0, shrinkB=0))

    ax.axhspan(floor, belief, color=PALETTE["hyper"], alpha=0.06, zorder=0)
    xr = last_iter * 1.20
    ref_line(ax, floor, "Regime-agnostic floor", x=xr, linestyle="-")
    ref_line(ax, belief, "Belief-PPO", x=xr, linestyle=(0, (1, 1.5)))
    ref_line(ax, oracle, "Oracle-PPO", x=xr, linestyle=(0, (6, 2)))

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean return across seeds", fontsize=12)
    ax.set_xlim(0, last_iter * 1.30)
    ax.tick_params(axis="both", labelsize=11)
    polish(ax)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def _smooth_curve(arr, w):
    """Centered moving average over available samples (no edge zero-padding)."""
    if w <= 1:
        return arr
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    half = w // 2
    n = len(arr)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def _draw_probe_per_t(ax, classifier: str, env_label: str = MEDIUM_ENV) -> bool:
    """Draw the per-timestep regime-decoding curves for one probe class onto
    `ax` (4 cells + analytical posterior + random-guess line). Returns False if
    the data is missing. Prefers the high-rollout file for stable curves."""
    probe = _load_probe_for_per_t(classifier)
    if probe is None:
        return False
    env_block = probe.get("per_method_per_env", {}).get(env_label, {})
    if not env_block:
        return False

    smoothing_window = 9
    cell_style = {
        "rl2_concat":       (PALETTE["concat"], "-"),
        "rl2_hypernet":     (PALETTE["hyper"],  "-"),
        "varibad_concat":   (PALETTE["concat"], "--"),
        "varibad_hypernet": (PALETTE["hyper"],  "--"),
    }
    analytical_drawn = False
    for cell in CELLS:
        m = env_block.get(cell)
        if m is None:
            continue
        per_t = np.asarray(m["method_per_t_test_acc_mean"])
        per_t_per_seed = np.asarray(m["method_per_t_test_acc_per_seed"])
        n_seeds = per_t_per_seed.shape[0]
        if n_seeds > 1:
            T = per_t.shape[0]
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n_seeds, size=(1000, n_seeds))
            lo = np.zeros(T); hi = np.zeros(T)
            for t in range(T):
                vals = per_t_per_seed[idx, t].mean(axis=1)
                lo[t] = np.percentile(vals, 2.5)
                hi[t] = np.percentile(vals, 97.5)
        else:
            lo, hi = per_t.copy(), per_t.copy()
        per_t_s = _smooth_curve(per_t, smoothing_window)
        lo_s = _smooth_curve(lo, smoothing_window)
        hi_s = _smooth_curve(hi, smoothing_window)
        ts = np.arange(per_t.shape[0])
        color, ls = cell_style.get(cell, ("#666666", "-"))
        ax.plot(ts, per_t_s, color=color, label=CELL_LABEL[cell],
                linewidth=2.2, linestyle=ls)
        ax.fill_between(ts, lo_s, hi_s, color=color, alpha=0.12)
        if not analytical_drawn:
            ana = np.asarray(m["analytical_per_t_test_acc_mean"])
            # Align analytical to the same information horizon as the method
            # curves: shift right one step and anchor t=0 at chance so every
            # curve starts at the random-guess baseline with zero observations.
            ana_aligned = np.concatenate(([1.0 / 3.0], ana[:-1]))
            ax.plot(ts, ana_aligned, color=PALETTE["analytical"],
                    linestyle="-", linewidth=2.8, label="Analytical posterior")
            analytical_drawn = True

    ax.axhline(1.0 / 3, color="#999999", linestyle=":", linewidth=1.0,
               label="Random guess (33.3%)")
    ax.set_xlabel("Timestep within episode", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0.20, 1.05)
    polish(ax)
    return True


def plot_m5r_probe_per_t(
    out_path: Path, env_label: str = MEDIUM_ENV, classifier: str = "logistic",
) -> None:
    """Single-panel per-timestep regime-decoding curves for one probe class."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    if not _draw_probe_per_t(ax, classifier, env_label):
        print("[m5r_plots] skip probe_per_t: no probe data")
        plt.close(fig)
        return
    ax.set_ylabel("Probe test accuracy", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#dddddd", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_m5r_probe_per_t_combined(out_path: Path, env_label: str = MEDIUM_ENV) -> None:
    """Linear and MLP probe per-timestep curves side by side, shared y-axis and a
    single shared legend below."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharey=True)
    drew_any = False
    for ax, (clf, title) in zip(axes, [("logistic", "Linear probe"),
                                       ("mlp", "MLP probe")]):
        if _draw_probe_per_t(ax, clf, env_label):
            drew_any = True
        ax.set_title(title, fontsize=13, loc="left")
    if not drew_any:
        print("[m5r_plots] skip probe_per_t_combined: no probe data")
        plt.close(fig)
        return
    axes[0].set_ylabel("Probe test accuracy", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=6, fontsize=10,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#dddddd", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def _load_probe_for_per_t(classifier: str) -> dict | None:
    """Probe results for the per-timestep figures, preferring the high-rollout
    file when it exists so the curves are less sample-noisy."""
    suffix = "" if classifier == "logistic" else f"_{classifier}"
    hires = FINAL_DIR / f"m5r_posterior_vs_performance{suffix}_hires.json"
    if resolve_data(hires).exists():
        with open(resolve_data(hires)) as f:
            return json.load(f)
    return _load_probe(classifier)


def _paired_delta_ci(delta_ST: np.ndarray, n_boot: int = 10_000) -> tuple:
    """Mean paired difference per timestep and its bootstrap CI.

    Resamples seeds, not timesteps: one seed contributes its whole curve to a
    resample, which is what pairing across seeds means here.
    """
    n_seeds = delta_ST.shape[0]
    rng = np.random.default_rng(0)
    means = np.empty((n_boot, delta_ST.shape[1]))
    step = 2000
    for lo_i in range(0, n_boot, step):
        idx = rng.integers(0, n_seeds, size=(min(step, n_boot - lo_i), n_seeds))
        means[lo_i:lo_i + idx.shape[0]] = delta_ST[idx].mean(axis=1)
    return (delta_ST.mean(axis=0),
            np.percentile(means, 2.5, axis=0),
            np.percentile(means, 97.5, axis=0))


# The architecture contrast is the plotted quantity here, so a line identifies a
# method. It keeps the colour and dash of that method's hypernetwork arm in the
# levels figures, since the difference is taken in that arm's favour.
# Shade separates the two methods, as in the learning-curve figure.
_METHOD_DELTA_STYLE = {
    "rl2": (_VARIANT_COLOR["rl2_hypernet"], "-", "RL²"),
    "varibad": (_VARIANT_COLOR["varibad_hypernet"], "--", "VariBAD"),
}


# Per-timestep belief-quality series, primary metric first.
_PER_T_METRIC = {
    "kl": ("method_per_t_kl_per_seed",
           "Difference in KL to the analytical posterior"),
    "acc": ("method_per_t_test_acc_per_seed",
            "Difference in probe test accuracy"),
}


def _draw_probe_delta_per_t(
    ax, classifier: str, env_label: str = MEDIUM_ENV, metric: str = "acc",
) -> bool:
    """Draw hypernetwork-minus-concatenation belief quality at each
    within-episode timestep, paired across seeds, for both methods."""
    key, _ = _PER_T_METRIC[metric]
    probe = _load_probe_for_per_t(classifier)
    if probe is None:
        return False
    env_block = probe.get("per_method_per_env", {}).get(env_label, {})
    if not env_block:
        return False

    smoothing_window = 9
    drew = False
    for method, (color, ls, label) in _METHOD_DELTA_STYLE.items():
        hyper = env_block.get(f"{method}_hypernet")
        concat = env_block.get(f"{method}_concat")
        if hyper is None or concat is None:
            continue
        if key not in hyper or key not in concat:
            print(f"[m5r_plots] skip {method} ({metric}): {key} missing, "
                  "rerun scripts.m5r_posterior_probe")
            continue
        h = np.asarray(hyper[key])
        c = np.asarray(concat[key])
        if h.shape != c.shape:
            print(f"[m5r_plots] skip {method}: unpaired seed counts "
                  f"{h.shape[0]} vs {c.shape[0]}")
            continue
        mean, lo, hi = _paired_delta_ci(h - c)
        ts = np.arange(mean.shape[0])
        ax.plot(ts, _smooth_curve(mean, smoothing_window), color=color,
                label=label, linewidth=2.2, linestyle=ls)
        # Two bands overlap over much of the episode, so each carries a thin
        # edge in its own colour and the fill stays light enough to see through.
        ax.fill_between(ts, _smooth_curve(lo, smoothing_window),
                        _smooth_curve(hi, smoothing_window),
                        facecolor=color, alpha=0.13, edgecolor=color,
                        linewidth=0.8)
        drew = True

    if not drew:
        return False
    ax.axhline(0.0, color="#555555", linestyle="-", linewidth=1.0,
               label="No difference between architectures")
    ax.set_xlabel("Timestep within episode", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    polish(ax)
    return True


def plot_m5r_probe_delta_per_t(out_path: Path, env_label: str = MEDIUM_ENV) -> None:
    """Paired architecture contrast in belief quality across the episode: both
    metrics by row, both probe families by column.

    The levels figure plots four variants against each other, where the
    method-to-method separation dominates the architecture contrast the
    comparison actually tests. This plots that contrast directly, with the
    across-seed pairing the statistical protocol uses. The divergence row comes
    first because it is the primary metric, and it runs the other way round:
    a negative difference favours the hypernetwork there.
    """
    apply_style()
    rows = [m for m in ("kl", "acc")]
    cols = [("logistic", "Linear probe"), ("mlp", "MLP probe")]
    fig, axes = plt.subplots(
        len(rows), len(cols), figsize=(14.0, 8.4), sharex=True, sharey="row",
    )
    drew_any = False
    for r, metric in enumerate(rows):
        for c, (clf, title) in enumerate(cols):
            ax = axes[r][c]
            if _draw_probe_delta_per_t(ax, clf, env_label, metric=metric):
                drew_any = True
            if r == 0:
                ax.set_title(title, fontsize=13, loc="left")
            if r < len(rows) - 1:
                ax.set_xlabel("")
        axes[r][0].set_ylabel(_PER_T_METRIC[metric][1], fontsize=12)
    if not drew_any:
        print("[m5r_plots] skip probe_delta_per_t: no probe data")
        plt.close(fig)
        return
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#dddddd", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def _diagnostic_separation_per_seed(diagnostic: str) -> dict[str, np.ndarray] | None:
    """Per-seed separation score for one behavioural diagnostic, by method.

    Both diagnostics score a policy on the same scale, zero for behaviour that
    ignores the regime and one for behaviour that changes completely with it,
    so the two figures share axes and reference lines.
    """
    if diagnostic == "action":
        path = FINAL_DIR / "m5r_action_distributions.json"
        if not resolve_data(path).exists():
            return None
        with open(resolve_data(path)) as f:
            by_method = json.load(f)["by_method"]
        # The separation statistic has one definition, in
        # evaluation.action_distribution, so the figure and the hypothesis test
        # cannot drift apart.
        return {
            m: regime_separation_per_seed(
                np.asarray(b["per_seed_action_given_regime_inventory"]),
                np.asarray(b["per_seed_inventory_counts"]),
            )
            for m, b in by_method.items()
        }
    if diagnostic == "swap":
        path = FINAL_DIR / "m5r_belief_swap.json"
        if not resolve_data(path).exists():
            return None
        with open(resolve_data(path)) as f:
            by_method = json.load(f)["by_method"]
        return {m: np.asarray(b["per_seed_separation"]) for m, b in by_method.items()}
    raise ValueError(f"unknown diagnostic: {diagnostic!r}")


def plot_m5r_diagnostic_separation(
    out_path: Path, diagnostic: str = "action",
) -> None:
    """Per-seed separation score of each variant, paired within seed.

    One point per run, a segment joining the two architectures of a seed, and
    the reference levels behind them. The table reports the paired difference;
    this shows the level each variant reaches and how far it sits from the
    Belief-PPO reference, which a difference alone cannot say.
    """
    per_seed = _diagnostic_separation_per_seed(diagnostic)
    if per_seed is None:
        print(f"[m5r_plots] skip diagnostic_separation ({diagnostic}): data missing")
        return
    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    # Two method groups, the two architectures side by side inside each.
    positions = {"rl2_concat": 0.0, "rl2_hypernet": 1.0,
                 "varibad_concat": 2.4, "varibad_hypernet": 3.4}
    rng = np.random.default_rng(0)
    for method in ("rl2", "varibad"):
        c = per_seed.get(f"{method}_concat")
        h = per_seed.get(f"{method}_hypernet")
        if c is None or h is None or c.shape != h.shape:
            continue
        x_c, x_h = positions[f"{method}_concat"], positions[f"{method}_hypernet"]
        jitter = rng.uniform(-0.09, 0.09, size=c.shape[0])
        # A segment is one seed's paired difference, which is what the Wilcoxon
        # test consumes. Colouring it by direction makes the majority countable:
        # in uniform grey the segments cross into a wash and the pairing, the
        # reason overlapping clouds can still separate, is lost.
        rising = h > c
        for i in range(c.shape[0]):
            ax.plot([x_c + jitter[i], x_h + jitter[i]], [c[i], h[i]],
                    color="#2a9d8f" if rising[i] else PALETTE["accent2"],
                    linewidth=0.9, alpha=0.5, zorder=3)
        ax.text((x_c + x_h) / 2, 1.005,
                f"{int(rising.sum())} of {c.shape[0]} seeds rise",
                ha="center", va="bottom", fontsize=9, color="#444444")
        for key, vals in ((f"{method}_concat", c), (f"{method}_hypernet", h)):
            x = positions[key]
            ax.scatter(x + jitter, vals, s=26, color=_VARIANT_COLOR[key],
                       edgecolor="#33403f", linewidth=0.4, zorder=4)
            ax.plot([x - 0.28, x + 0.28], [np.nanmean(vals)] * 2,
                    color="#33403f", linewidth=2.0, zorder=5)

    refs = {k: float(np.nanmean(per_seed[k]))
            for k in ("regime_agnostic_ppo", "belief_ppo", "oracle_ppo")
            if k in per_seed}
    ax.set_xlim(-0.6, 5.1)
    _draw_refs(ax, refs, label_x=4.05)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["RL²\nConcat", "RL²\nHypernet",
                        "VariBAD\nConcat", "VariBAD\nHypernet"], fontsize=11)
    ylabel = ("Regime separation of the action distribution"
              if diagnostic == "action"
              else "Behaviour change under a swapped belief")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(-0.05, 1.12)
    polish(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def main() -> int:
    for target in _both_targets("m5r_method_ladder.png"):
        plot_method_ladder(target)
    for target in _both_targets("m5r_distinguishability_sweep.png"):
        plot_sweep(
            DISTINGUISHABILITY_LEVELS,
            "MarketMakingV1, distinguishability-axis difficulty sweep",
            target,
        )
    for target in _both_targets("m5r_sweep_n20.png"):
        plot_sweep(
            THESIS_SWEEP_LEVELS,
            "MarketMakingV1, in-domain difficulty study at $n=20$",
            target,
        )
    for target in _both_targets("m5r_posterior_vs_performance.png"):
        plot_posterior_vs_performance(target, classifier="logistic")
    for target in _both_targets("m5r_posterior_vs_performance_mlp.png"):
        plot_posterior_vs_performance(target, classifier="mlp")
    for target in _both_targets("m5r_learning_curves.png"):
        plot_m5r_learning_curves(target)
    # The thesis includes the two-panel version; it was defined but never
    # called, so the figure it references had no producer.
    for target in _both_targets("m5r_probe_per_t_combined.png"):
        plot_m5r_probe_per_t_combined(target)
    for target in _both_targets("m5r_probe_delta_per_t.png"):
        plot_m5r_probe_delta_per_t(target)
    for target in _both_targets("m5r_action_separation.png"):
        plot_m5r_diagnostic_separation(target, diagnostic="action")
    for target in _both_targets("m5r_belief_swap_separation.png"):
        plot_m5r_diagnostic_separation(target, diagnostic="swap")
    for target in _both_targets("m5r_probe_per_t.png"):
        plot_m5r_probe_per_t(target, classifier="logistic")
    for target in _both_targets("m5r_probe_per_t_mlp.png"):
        plot_m5r_probe_per_t(target, classifier="mlp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
