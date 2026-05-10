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

from plotting.style import COLORS, FIGSIZE_STANDARD, FIGSIZE_WIDE, apply_style

# Stacked-obs PPO is not in the COLORS dict; pick a distinct gray.
STACKED_OBS_COLOR = "#7f7f7f"

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FINAL_DIR = RESULTS_ROOT / "M5R" / "final"
PROJECT_FIG_DIR = REPO_ROOT / "figures" / "milestones" / "M5R"
THESIS_FIG_DIR = (
    REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"
)


def _both_targets(name: str) -> list[Path]:
    """Write figure to both the project milestones dir and the thesis figures dir."""
    return [PROJECT_FIG_DIR / name, THESIS_FIG_DIR / name]

CELLS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
CELL_LABEL = {
    "rl2_concat": "RL² Concat",
    "rl2_hypernet": "RL² Hypernet",
    "varibad_concat": "VariBAD Concat",
    "varibad_hypernet": "VariBAD Hypernet",
}
PERSISTENCE_LEVELS = (
    ("persistence_easy", "Easy ($P_{ii}=0.99$)"),
    ("e_final", "Medium ($P_{ii}=0.98$)"),
    ("persistence_hard", "Hard ($P_{ii}=0.96$)"),
    ("persistence_very_hard", "Very-hard ($P_{ii}=0.92$)"),
)
DISTINGUISHABILITY_LEVELS = (
    ("distinguishability_easy", "Easy"),
    ("e_final", "Medium"),
    ("distinguishability_hard", "Hard"),
)
KAPPA_LEVELS = (
    ("kappa02", "$\\kappa = 0.02$"),
    ("e_final", "$\\kappa = 0.05$"),
    ("kappa10", "$\\kappa = 0.10$"),
    ("kappa20", "$\\kappa = 0.20$"),
)


def _load() -> dict:
    with open(FINAL_DIR / "per_cell_env.json") as f:
        return json.load(f)


def _load_probe(classifier: str = "logistic") -> dict | None:
    suffix = "" if classifier == "logistic" else f"_{classifier}"
    p = FINAL_DIR / f"m5r_posterior_vs_performance{suffix}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_stacked_obs() -> dict | None:
    """Load stacked-obs sweep stats (mean / CI per env). None if missing."""
    p = FINAL_DIR / "m5r_stacked_obs_sweep.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _draw_refs(ax, refs: dict, label_each: bool = True) -> None:
    """Add horizontal reference lines for floor / belief / oracle."""
    label_map = {
        "regime_agnostic_ppo": "Floor",
        "belief_ppo": "Belief-PPO",
        "oracle_ppo": "Oracle-PPO",
    }
    color_map = {
        "regime_agnostic_ppo": COLORS["ppo"],
        "belief_ppo": COLORS["belief_ppo"],
        "oracle_ppo": COLORS["oracle_ppo"],
    }
    style_map = {
        "regime_agnostic_ppo": "--",
        "belief_ppo": ":",
        "oracle_ppo": "-.",
    }
    for key, label in label_map.items():
        v = refs.get(key)
        if v is None:
            continue
        ax.axhline(
            v, linestyle=style_map[key], color=color_map[key], linewidth=1.2,
            label=label if label_each else None,
        )


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
            edgecolor="black", linewidth=0.4,
        )
        for bar, method in zip(bars, methods):
            bar.set_facecolor(COLORS[f"{method}_{integ}"])
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
    _draw_refs(ax, refs)
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
                facecolor=COLORS[f"{m}_{i}"], edgecolor="black", linewidth=0.4,
                label=f"{method_full[m]} {integ_pretty[i]}",
            ))
    return handles


def plot_method_ladder(out_path: Path) -> None:
    apply_style()
    data = _load()
    stacked = _load_stacked_obs()
    env_block = data["per_env"]["e_final"]
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    _ladder_bars(ax, env_block, env_label="e_final", stacked_data=stacked)
    ax.set_title("Method ladder on MarketMakingV1, medium difficulty")
    ax.set_ylabel("Final-episode return")
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=_legend_handles() + ref_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
        frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.65)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_sweep(axis_levels: tuple, suptitle: str, out_path: Path) -> None:
    apply_style()
    data = _load()
    stacked = _load_stacked_obs()
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
    fig.suptitle(suptitle, fontsize=13)
    ref_handles, ref_labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles=_legend_handles() + ref_handles,
        loc="lower center", ncol=4, fontsize=10,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_posterior_vs_performance(
    out_path: Path, classifier: str = "logistic"
) -> None:
    """Scatter of probe accuracy vs gap-closed.

    `classifier="logistic"` plots the linear probe; `classifier="mlp"` plots
    the two-layer MLP probe. The two figures share axes ranges and styling
    so they can be displayed side by side.
    """
    apply_style()
    probe = _load_probe(classifier)
    if probe is None:
        print(f"[m5r_plots] skip posterior_vs_performance ({classifier}): "
              "probe data missing")
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    methods = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
    by_method = {m: [] for m in methods}
    for p in probe["scatter_points"]:
        by_method.setdefault(p["method"], []).append(p)
    for m in methods:
        pts = by_method[m]
        if not pts:
            continue
        xs = [1.0 - p["posterior_error"] for p in pts]
        ys = [p["gap_closed"] for p in pts]
        ax.scatter(
            xs, ys, color=COLORS[m], s=22, alpha=0.7,
            edgecolor="black", linewidth=0.3, label=CELL_LABEL[m],
        )
    probe_label = "Linear-probe" if classifier == "logistic" else "MLP-probe"
    ax.set_xlabel(f"{probe_label} regime-decoding accuracy")
    ax.set_ylabel("Gap-closed vs.\\ Oracle")
    ax.set_title(
        f"Posterior accuracy vs.\\ final return, {probe_label.lower()}\n"
        "MarketMakingV1, all five difficulty levels pooled"
    )
    # Shared x-range across the linear and MLP variants so the two figures
    # are directly comparable when shown side by side. Data range is
    # roughly [0.42, 0.92] across both probes; pad to [0.40, 0.95].
    ax.set_xlim(0.40, 0.95)
    ax.axhspan(-0.30, 0.30, color="#eeeeee", alpha=0.0)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_m5r_learning_curves(out_path: Path) -> None:
    """Per-iteration learning curves for the four meta-RL cells on the
    medium-difficulty environment, drawn from the matched-compute Stage C
    metrics. Reference horizontal lines for the regime-agnostic floor and
    the Belief-PPO and Oracle-PPO ceilings."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    data = _load()
    refs = data["per_env"]["e_final"]["refs"]
    floor = refs["regime_agnostic_ppo"]
    belief = refs["belief_ppo"]
    oracle = refs["oracle_ppo"]

    cell_specs = [
        (cell, CELL_LABEL[cell], COLORS.get(cell, "#666666"),
         RESULTS_ROOT / f"m5r_final_{cell}_e_final" / "metrics.json")
        for cell in CELLS
    ]
    cell_specs.append((
        "stacked_obs", "Stacked-obs PPO", STACKED_OBS_COLOR,
        RESULTS_ROOT / "m5r_stacked_obs_e_final" / "metrics.json",
    ))
    for cell_key, label, color, m_path in cell_specs:
        if not m_path.exists():
            continue
        with open(m_path) as f:
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
        ax.plot(iters, mean, color=color, label=label, linewidth=1.6)
        ax.fill_between(iters, lo, hi, color=color, alpha=0.15)

    ax.axhline(floor,  color=COLORS["ppo"],         linestyle="--",
               linewidth=1.2, alpha=0.7,
               label=f"Regime-agnostic floor = {floor:.1f}")
    ax.axhline(belief, color=COLORS["belief_ppo"],  linestyle="--",
               linewidth=1.2, alpha=0.7,
               label=f"Belief-PPO ceiling = {belief:.1f}")
    ax.axhline(oracle, color=COLORS["oracle_ppo"],  linestyle="--",
               linewidth=1.2, alpha=0.7,
               label=f"Oracle-PPO ceiling = {oracle:.1f}")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean return across seeds", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        "MarketMakingV1 — meta-RL learning curves at matched compute "
        "($\\sim 5$k parameters)",
        fontsize=14,
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4, fontsize=12,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def plot_m5r_probe_per_t(out_path: Path, env_label: str = "e_final") -> None:
    """Linear-probe regime-decoding accuracy as a function of within-episode
    timestep on the medium-difficulty environment, drawn from the matched-compute
    posterior probe."""
    apply_style()
    probe = _load_probe("logistic")
    if probe is None:
        print("[m5r_plots] skip probe_per_t: no logistic probe data")
        return
    per_method_per_env = probe.get("per_method_per_env", {})
    env_block = per_method_per_env.get(env_label, {})
    if not env_block:
        print(f"[m5r_plots] skip probe_per_t: no env data for {env_label}")
        return

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    smoothing_window = 9

    def _smooth(arr, w):
        """Centered moving average that averages only the available samples
        at the boundaries. Avoids the edge-attenuation artifact of
        np.convolve(..., mode='same') which zero-pads the edges and pulls
        them spuriously toward zero."""
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
        per_t_s = _smooth(per_t, smoothing_window)
        lo_s = _smooth(lo, smoothing_window)
        hi_s = _smooth(hi, smoothing_window)
        ts = np.arange(per_t.shape[0])
        color = COLORS.get(cell, "#666666")
        ax.plot(ts, per_t_s, color=color, label=CELL_LABEL[cell], linewidth=1.6)
        ax.fill_between(ts, lo_s, hi_s, color=color, alpha=0.15)
        if not analytical_drawn:
            ana = np.asarray(m["analytical_per_t_test_acc_mean"])
            ax.plot(ts, _smooth(ana, smoothing_window),
                    color=COLORS.get("analytical", "#9467bd"),
                    linestyle="--", linewidth=1.4,
                    label="Analytical posterior")
            analytical_drawn = True

    n_classes = 3
    ax.axhline(1.0 / n_classes, color="#999999", linestyle=":", linewidth=1.0,
               label=f"Random guess ({100.0 / n_classes:.1f}\\%)")

    ax.set_xlabel("Timestep within episode", fontsize=12)
    ax.set_ylabel("Probe test accuracy", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        "Linear-probe regime decoding per timestep on the medium-difficulty "
        "environment ($\\sim 5$k parameters)",
        fontsize=14,
    )
    ax.set_ylim(0.20, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, fontsize=12,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[m5r_plots] wrote {out_path}")


def main() -> int:
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for target in _both_targets("m5r_method_ladder.png"):
        plot_method_ladder(target)
    for target in _both_targets("m5r_persistence_sweep.png"):
        plot_sweep(
            PERSISTENCE_LEVELS,
            "MarketMakingV1, persistence-axis difficulty sweep",
            target,
        )
    for target in _both_targets("m5r_kappa_sweep.png"):
        plot_sweep(
            KAPPA_LEVELS,
            "MarketMakingV1, inventory-penalty $\\kappa$ sweep",
            target,
        )
    for target in _both_targets("m5r_distinguishability_sweep.png"):
        plot_sweep(
            DISTINGUISHABILITY_LEVELS,
            "MarketMakingV1, distinguishability-axis difficulty sweep",
            target,
        )
    for target in _both_targets("m5r_posterior_vs_performance.png"):
        plot_posterior_vs_performance(target, classifier="logistic")
    for target in _both_targets("m5r_posterior_vs_performance_mlp.png"):
        plot_posterior_vs_performance(target, classifier="mlp")
    for target in _both_targets("m5r_learning_curves.png"):
        plot_m5r_learning_curves(target)
    for target in _both_targets("m5r_probe_per_t.png"):
        plot_m5r_probe_per_t(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
