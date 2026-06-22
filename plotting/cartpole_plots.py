"""Cartpole external-validity figures.

Produces:
  - cartpole_method_ladder.png — 7-method bar chart with 95% bootstrap
    CIs across seeds. References (regime-agnostic / Belief / Oracle PPO)
    drawn as horizontal dashed lines for visual reference.

The plot mirrors the M5 method-ladder figure conventions (apply_style,
COLORS, budget_annotation, LEGEND_OUTSIDE_RIGHT) so the cross-env
comparison is visually consistent.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.style import (
    COLORS, LEGEND_OUTSIDE_RIGHT, PALETTE, apply_style, budget_annotation, polish,
)

# Brand cell colours: concat in the slate family, hypernet in the teal family.
_CELL_COLORS = {
    "rl2_concat":       "#b4bcc2",
    "varibad_concat":   "#8a949c",
    "rl2_hypernet":     "#2a9d8f",
    "varibad_hypernet": "#73b8ad",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures" / "milestones" / "cartpole"


def _load(method: str) -> np.ndarray:
    p = RESULTS_ROOT / f"m_cartpole_{method}" / "metrics.json"
    with open(p) as f:
        m = json.load(f)
    return np.asarray(m["per_seed_final_return"], dtype=float)


def _bootstrap_ci(arr: np.ndarray, n_boot: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def plot_method_ladder(out_path: Path) -> None:
    refs = ("regime_agnostic", "belief", "oracle")
    metas = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
    ref_data = {m: _load(m) for m in refs}
    meta_data = {m: _load(m) for m in metas}

    apply_style()
    floor_mean = float(ref_data["regime_agnostic"].mean())
    belief_mean = float(ref_data["belief"].mean())
    oracle_mean = float(ref_data["oracle"].mean())

    labels = {
        "rl2_concat": "RL² Concat",
        "rl2_hypernet": "RL² Hypernet",
        "varibad_concat": "VariBAD Concat",
        "varibad_hypernet": "VariBAD Hypernet",
    }
    xs = np.arange(len(metas))
    means = np.array([meta_data[m].mean() for m in metas])
    cis = np.array([_bootstrap_ci(meta_data[m]) for m in metas])
    err_low = means - cis[:, 0]
    err_high = cis[:, 1] - means

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    # Shaded gap zones: recoverable (floor->belief) + inference remainder.
    ax.axhspan(floor_mean, belief_mean, color=PALETTE["hyper"], alpha=0.09, zorder=0)
    ax.axhspan(belief_mean, oracle_mean, color=PALETTE["hyper"], alpha=0.04, zorder=0)

    bar_colors = [_CELL_COLORS[m] for m in metas]
    ax.bar(xs, means, width=0.62, color=bar_colors, edgecolor="white", linewidth=1.2,
           yerr=[err_low, err_high], capsize=4, zorder=3,
           error_kw=dict(ecolor="#3a3a3a", elinewidth=1.2))
    for x, mu, hi in zip(xs, means, err_high):
        ax.annotate(f"{mu:.1f}", xy=(x, mu + hi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

    # Reference lines with inline white-backed labels at the right edge.
    xr = len(metas) - 0.55
    for v, lab, ls in [(oracle_mean, "Oracle-PPO", (0, (6, 2))),
                       (belief_mean, "Belief-PPO ceiling", (0, (1, 1.5))),
                       (floor_mean, "Regime-agnostic floor", "solid")]:
        ax.axhline(v, color="#555555", linewidth=1.1, linestyle=ls, zorder=2)
        ax.text(xr, v, f"  {lab}", va="center", ha="left", fontsize=8.5,
                color="#444444", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels([labels[m] for m in metas], fontsize=10)
    ax.set_ylabel("Mean episode return", fontsize=11)
    ax.set_ylim(0, oracle_mean + 9)
    ax.set_xlim(-0.6, len(metas) - 0.55 + 2.2)
    polish(ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)


_METHOD_LABELS = {
    "rl2_concat":       "RL² Concat",
    "rl2_hypernet":     "RL² Hypernet",
    "varibad_concat":   "VariBAD Concat",
    "varibad_hypernet": "VariBAD Hypernet",
}


def plot_posterior_vs_performance(out_path: Path) -> bool:
    """Per-(method, seed) scatter of posterior_error vs gap_closed.

    Mirrors the `figures/milestones/M6/rq3_posterior_vs_performance.png`
    layout: x = analytical_acc − method_acc (low → method belief
    decodes regime almost as well as analytical), y = gap_closed
    (0 = floor, 1 = oracle). Title carries the headline correlation
    + 95% bootstrap CI."""
    stats_path = (
        RESULTS_ROOT / "milestones" / "cartpole"
        / "stats_cartpole_posterior_vs_performance.json"
    )
    if not stats_path.exists():
        print(f"[cartpole_plots] skip scatter: missing {stats_path}", flush=True)
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    points = stats.get("scatter_points", [])
    if not points:
        return False

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    by_method: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        by_method.setdefault(p["method"], []).append(
            (float(p["posterior_error"]), float(p["gap_closed"])),
        )

    method_order = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
    for method in method_order:
        pts = by_method.get(method)
        if pts is None:
            continue
        xs, ys = zip(*pts)
        ax.scatter(
            xs, ys, color=COLORS.get(method, "#666666"),
            label=_METHOD_LABELS.get(method, method), s=42,
            edgecolor="black", linewidth=0.5, alpha=0.85,
        )

    ax.axhline(0.0, color=COLORS["ppo"], linestyle="--", linewidth=1.0,
               alpha=0.6, label="Floor (gap_closed = 0)")
    ax.axhline(1.0, color=COLORS["oracle_ppo"], linestyle="--", linewidth=1.0,
               alpha=0.6, label="Oracle (gap_closed = 1)")

    ax.set_xlabel(
        "Posterior error  =  analytical HMM probe accuracy  −  method probe accuracy\n"
        "(low → method belief decodes regime almost as well as analytical)"
    )
    ax.set_ylabel(
        "Gap closed  =  (method return − floor) / (oracle − floor)\n"
        "(0 = floor, 1 = oracle)"
    )

    r = stats.get("correlation_overall", float("nan"))
    ci = stats.get("correlation_overall_ci95", [float("nan"), float("nan")])
    n = stats.get("n_scatter_points", len(points))
    ax.set_title(
        "CartPoleRegimeV1 — posterior quality vs task performance\n"
        f"Pearson r = {r:+.3f}  CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]  across n = {n} (cell × seed)"
    )

    ax.legend(**LEGEND_OUTSIDE_RIGHT)
    budget_annotation(
        fig,
        iterations=200, parallel_envs=512, rollout_length=128, num_seeds=8,
        extra="| 4 meta-RL cells × n=8 seeds = 32 points | classifier=logistic, n_rollouts=200",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def plot_logistic_vs_mlp_side_by_side(out_path: Path) -> bool:
    """Two-panel scatter: logistic-probe (left) vs MLP-probe (right).

    Demonstrates that the inversion (concat decodes better, performs
    worse) survives moving from a linear classifier to a non-linear one
    — falsifying the "linear-probe-blind" interpretation of the
    inverted decoupling. Headline stats live in each panel's title.
    """
    paths = {
        "logistic": (
            RESULTS_ROOT / "milestones" / "cartpole"
            / "stats_cartpole_posterior_vs_performance.json"
        ),
        "mlp": (
            RESULTS_ROOT / "milestones" / "cartpole"
            / "stats_cartpole_posterior_vs_performance_mlp.json"
        ),
    }
    stats_by_clf = {}
    for clf, p in paths.items():
        if not p.exists():
            print(f"[cartpole_plots] skip side-by-side: missing {p}", flush=True)
            return False
        with open(p) as f:
            stats_by_clf[clf] = json.load(f)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=True)
    method_order = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")

    for ax, (clf, stats) in zip(axes, stats_by_clf.items()):
        by_method: dict[str, list[tuple[float, float]]] = {}
        for p in stats["scatter_points"]:
            by_method.setdefault(p["method"], []).append(
                (float(p["posterior_error"]), float(p["gap_closed"])),
            )
        for method in method_order:
            pts = by_method.get(method)
            if pts is None:
                continue
            xs, ys = zip(*pts)
            ax.scatter(
                xs, ys, color=COLORS.get(method, "#666666"),
                label=_METHOD_LABELS.get(method, method), s=42,
                edgecolor="black", linewidth=0.5, alpha=0.85,
            )
        ax.axhline(0.0, color=COLORS["ppo"], linestyle="--", linewidth=1.0,
                   alpha=0.6, label="Floor (gap_closed = 0)")
        ax.axhline(1.0, color=COLORS["oracle_ppo"], linestyle="--", linewidth=1.0,
                   alpha=0.6, label="Oracle (gap_closed = 1)")
        r = stats["correlation_overall"]
        ci = stats["correlation_overall_ci95"]
        n = stats["n_scatter_points"]
        clf_label = "logistic regression" if clf == "logistic" else "MLP (1×64)"
        ax.set_title(
            f"{clf_label} probe\n"
            f"r = {r:+.3f}  CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]  n = {n}"
        )
        ax.set_xlabel(
            "Posterior error  =  analytical acc − method acc"
        )
    axes[0].set_ylabel(
        "Gap closed  =  (return − floor) / (oracle − floor)\n(0 = floor, 1 = oracle)"
    )
    axes[1].legend(**LEGEND_OUTSIDE_RIGHT)
    fig.suptitle(
        "CartPoleRegimeV1 — inverted decoupling survives non-linear probe\n"
        "(concat decodes better than hypernet under both classifiers; performs worse under both)",
        y=1.02,
    )
    budget_annotation(
        fig,
        iterations=200, parallel_envs=512, rollout_length=128, num_seeds=8,
        extra="| 4 cells × n=8 = 32 points per panel | n_rollouts=200",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def plot_cross_env_2x2(out_path: Path) -> bool:
    """2×2 grid (env × probe) showing decoupling-vs-inversion across both
    envs and both probe classifiers. Top row = MM (decoupling), bottom
    row = cartpole (inversion). Left column = logistic, right = MLP.
    The four panels share x and y conventions so the cross-env contrast
    is visible at a glance."""
    M5R = REPO_ROOT / "results" / "M5R" / "final"
    CP = RESULTS_ROOT / "milestones" / "cartpole"
    paths = {
        ("MM", "logistic"): M5R / "m5r_posterior_vs_performance.json",
        ("MM", "mlp"): M5R / "m5r_posterior_vs_performance_mlp.json",
        ("Cartpole", "logistic"): CP / "stats_cartpole_posterior_vs_performance.json",
        ("Cartpole", "mlp"): CP / "stats_cartpole_posterior_vs_performance_mlp.json",
    }
    stats = {}
    for k, p in paths.items():
        if not p.exists():
            print(f"[cartpole_plots] skip 2x2: missing {p}", flush=True)
            return False
        with open(p) as f:
            stats[k] = json.load(f)

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=False)
    method_order = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")

    for row, env in enumerate(("MM", "Cartpole")):
        for col, clf in enumerate(("logistic", "mlp")):
            ax = axes[row, col]
            s = stats[(env, clf)]
            by_method: dict[str, list[tuple[float, float]]] = {}
            for p in s["scatter_points"]:
                by_method.setdefault(p["method"], []).append(
                    (float(p["posterior_error"]), float(p["gap_closed"])),
                )
            for method in method_order:
                pts = by_method.get(method)
                if pts is None:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(
                    xs, ys, color=_CELL_COLORS.get(method, "#666666"),
                    label=_METHOD_LABELS.get(method, method) if (row == 0 and col == 1) else None,
                    s=26, edgecolor="white", linewidth=0.4, alpha=0.85,
                )
            ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
            r = s["correlation_overall"]
            ci = s.get("correlation_overall_ci95", None)
            n = s["n_scatter_points"]
            ci_str = f"  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""
            clf_label = "logistic" if clf == "logistic" else "MLP (1×64)"
            interp = "decoupling" if abs(r) < 0.30 else "inversion"
            ax.set_title(
                f"{env} · {clf_label}  ·  r = {r:+.3f}{ci_str}  ·  n = {n}\n"
                f"({interp})",
                fontsize=10,
            )
            if row == 1:
                ax.set_xlabel("Posterior error  =  analytical acc − method acc")
            if col == 0:
                ax.set_ylabel("Gap closed (0 = floor, 1 = oracle)")

    axes[0, 1].legend(**LEGEND_OUTSIDE_RIGHT)
    budget_annotation(
        fig,
        extra=(
            "MM n=240 (20 cells x 12 seeds)  ·  "
            "Cartpole n=48 (4 cells x 12 seeds)  ·  matched-compute protocol"
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def plot_difficulty_sweep_returns(out_path: Path) -> bool:
    """7-method × 3-level line plot of mean returns. Mirrors M6's
    rq3_persistence_sweep / rq3_distinguishability_sweep figure style."""
    sweep_path = RESULTS_ROOT / "milestones" / "cartpole" / "stats_cartpole_sweep.json"
    if not sweep_path.exists():
        print(f"[cartpole_plots] skip sweep_returns: missing {sweep_path}", flush=True)
        return False

    # Load per-method per-level means and CIs across all 3 levels
    # (medium values are not in stats_cartpole_sweep.json — we read
    # them directly from the historical experiment dirs).
    levels = ("easy", "medium", "hard")
    methods = (
        "regime_agnostic", "belief", "oracle",
        "rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet",
    )

    def get_seeds(method: str, level: str) -> np.ndarray:
        name = f"m_cartpole_{method}" if level == "medium" else f"m_cartpole_{method}_{level}"
        with open(RESULTS_ROOT / name / "metrics.json") as f:
            return np.asarray(json.load(f)["per_seed_final_return"], dtype=float)

    means = {m: [float(get_seeds(m, l).mean()) for l in levels] for m in methods}
    cis = {}
    for m in methods:
        cis[m] = []
        for l in levels:
            arr = get_seeds(m, l)
            rng = np.random.default_rng(0)
            boot = rng.choice(arr, size=(10_000, arr.size), replace=True).mean(axis=1)
            cis[m].append((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xs = np.arange(len(levels))

    plot_order = (
        ("oracle",           "Oracle-PPO",        COLORS["oracle_ppo"], "--"),
        ("belief",           "Belief-PPO",        COLORS["belief_ppo"], "--"),
        ("regime_agnostic",  "Regime-agnostic",   COLORS["ppo"],        "--"),
        ("rl2_hypernet",     "RL² Hypernet",      COLORS["rl2_hypernet"], "-"),
        ("varibad_hypernet", "VariBAD Hypernet",  COLORS["varibad_hypernet"], "-"),
        ("rl2_concat",       "RL² Concat",        COLORS["rl2_concat"], "-"),
        ("varibad_concat",   "VariBAD Concat",    COLORS["varibad_concat"], "-"),
    )
    for key, label, color, linestyle in plot_order:
        ys = means[key]
        lo = np.array([c[0] for c in cis[key]])
        hi = np.array([c[1] for c in cis[key]])
        ax.plot(xs, ys, marker="o", color=color, linewidth=1.6,
                linestyle=linestyle, label=f"{label} (mean={ys[1]:.1f} medium)")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_xticks(xs)
    ax.set_xticklabels([l.capitalize() for l in levels])
    ax.set_xlabel("Difficulty (within-regime asymmetry strength: easy=0.78, medium=0.65, hard=0.40)")
    ax.set_ylabel("Mean episode return")
    ax.set_title(
        "CartPoleRegimeV1 — difficulty sweep across the action-success-asymmetry axis\n"
        "Hypernet beats Concat at every level (Family A 6/6 Holm-supported, LOO-robust)"
    )
    ax.legend(**LEGEND_OUTSIDE_RIGHT)
    budget_annotation(
        fig,
        iterations=200, parallel_envs=512, rollout_length=128,
        extra="| 7 methods × 3 levels | refs n=5, meta-RL n=8 | shaded = 95% bootstrap CI",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def plot_difficulty_sweep_scatter(out_path: Path) -> bool:
    """3 panels (one per level) × 2 rows (logistic / MLP) = 6 panels showing
    posterior_error vs gap_closed across the cartpole difficulty sweep.
    Demonstrates the inversion is robust at easy/medium and attenuates
    at hard (where the envelope collapses)."""
    paths = {
        "logistic": (
            RESULTS_ROOT / "milestones" / "cartpole"
            / "stats_cartpole_posterior_vs_performance_sweep.json"
        ),
        "mlp": (
            RESULTS_ROOT / "milestones" / "cartpole"
            / "stats_cartpole_posterior_vs_performance_sweep_mlp.json"
        ),
    }
    stats_by = {}
    for clf, p in paths.items():
        if not p.exists():
            print(f"[cartpole_plots] skip sweep_scatter: missing {p}", flush=True)
            return False
        with open(p) as f:
            stats_by[clf] = json.load(f)

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharey=True)
    method_order = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
    levels = ("easy", "medium", "hard")

    for row, clf in enumerate(("logistic", "mlp")):
        s = stats_by[clf]
        by_level: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for p in s["scatter_points"]:
            by_level.setdefault(p["level"], {}).setdefault(p["method"], []).append(
                (float(p["posterior_error"]), float(p["gap_closed"])),
            )
        for col, level in enumerate(levels):
            ax = axes[row, col]
            for method in method_order:
                pts = by_level.get(level, {}).get(method)
                if pts is None:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(
                    xs, ys, color=COLORS.get(method, "#666666"),
                    label=_METHOD_LABELS.get(method, method) if (row == 0 and col == 2) else None,
                    s=32, edgecolor="black", linewidth=0.4, alpha=0.85,
                )
            ax.axhline(0.0, color=COLORS["ppo"], linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axhline(1.0, color=COLORS["oracle_ppo"], linestyle="--", linewidth=0.8, alpha=0.5)
            # Per-cell correlation from the scatter points.
            xs_all = [x for m in method_order for x, _ in by_level.get(level, {}).get(m, [])]
            ys_all = [y for m in method_order for _, y in by_level.get(level, {}).get(m, [])]
            if len(xs_all) >= 3 and np.std(xs_all) > 0 and np.std(ys_all) > 0:
                r_cell = float(np.corrcoef(xs_all, ys_all)[0, 1])
            else:
                r_cell = float("nan")
            clf_label = "logistic" if clf == "logistic" else "MLP"
            ax.set_title(f"{level.capitalize()} · {clf_label}  ·  r = {r_cell:+.3f}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Posterior error")

    axes[0, 0].set_ylabel("Gap closed (logistic)\n(0 = floor, 1 = oracle)")
    axes[1, 0].set_ylabel("Gap closed (MLP)\n(0 = floor, 1 = oracle)")
    axes[0, 2].legend(**LEGEND_OUTSIDE_RIGHT)
    fig.suptitle(
        "CartPoleRegimeV1 — posterior decoding vs task performance across difficulty\n"
        "Inversion present at easy + medium · attenuates at hard (small envelope, noisy gap_closed)",
        fontsize=12,
    )
    budget_annotation(
        fig,
        extra="6 panels (3 levels × 2 probes) · n=32 per panel · n_rollouts=200",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def plot_axis_returns(out_path: Path, axis: str) -> bool:
    """7-method × 3-level line plot for a given difficulty axis."""
    levels = ("easy", "medium", "hard")
    methods = (
        "regime_agnostic", "belief", "oracle",
        "rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet",
    )

    def get_seeds(method: str, level: str) -> np.ndarray:
        if level == "medium":
            name = f"m_cartpole_{method}"
        elif axis == "asymmetry":
            name = f"m_cartpole_{method}_{level}"
        else:
            name = f"m_cartpole_{method}_{axis}_{level}"
        with open(RESULTS_ROOT / name / "metrics.json") as f:
            return np.asarray(json.load(f)["per_seed_final_return"], dtype=float)

    means = {m: [float(get_seeds(m, l).mean()) for l in levels] for m in methods}
    cis = {}
    for m in methods:
        cis[m] = []
        for l in levels:
            arr = get_seeds(m, l)
            rng = np.random.default_rng(0)
            boot = rng.choice(arr, size=(10_000, arr.size), replace=True).mean(axis=1)
            cis[m].append((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xs = np.arange(len(levels))

    plot_order = (
        ("oracle",           "Oracle-PPO",        COLORS["oracle_ppo"], "--"),
        ("belief",           "Belief-PPO",        COLORS["belief_ppo"], "--"),
        ("regime_agnostic",  "Regime-agnostic",   COLORS["ppo"],        "--"),
        ("rl2_hypernet",     "RL² Hypernet",      COLORS["rl2_hypernet"], "-"),
        ("varibad_hypernet", "VariBAD Hypernet",  COLORS["varibad_hypernet"], "-"),
        ("rl2_concat",       "RL² Concat",        COLORS["rl2_concat"], "-"),
        ("varibad_concat",   "VariBAD Concat",    COLORS["varibad_concat"], "-"),
    )
    for key, label, color, linestyle in plot_order:
        ys = means[key]
        lo = np.array([c[0] for c in cis[key]])
        hi = np.array([c[1] for c in cis[key]])
        ax.plot(xs, ys, marker="o", color=color, linewidth=1.6,
                linestyle=linestyle, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)

    axis_label = (
        "Within-regime asymmetry strength (easy=0.78, medium=0.65, hard=0.40)"
        if axis == "asymmetry"
        else "HMM persistence — diagonal (easy=0.99, medium=0.98, hard=0.92)"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([l.capitalize() for l in levels])
    ax.set_xlabel(f"Difficulty: {axis_label}")
    ax.set_ylabel("Mean episode return")
    ax.set_title(
        f"CartPoleRegimeV1 — {axis} sweep\n"
        f"Hypernet beats Concat at every level (Family A 6/6 Holm-supported, LOO-robust)"
    )
    ax.legend(**LEGEND_OUTSIDE_RIGHT)
    budget_annotation(
        fig,
        iterations=200, parallel_envs=512, rollout_length=128,
        extra="| 7 methods × 3 levels | refs n=5, meta-RL n=8 | shaded = 95% bootstrap CI",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def plot_two_axis_scatter_grid(out_path: Path) -> bool:
    """4×3 grid: rows = (asymmetry-logistic, asymmetry-mlp,
    persistence-logistic, persistence-mlp); columns = (easy, medium,
    hard). 12 panels total. Headline figure for the cartpole
    cross-axis robustness claim."""
    cp = RESULTS_ROOT / "milestones" / "cartpole"
    sources = (
        ("Asymmetry · logistic",   cp / "stats_cartpole_posterior_vs_performance_sweep.json"),
        ("Asymmetry · MLP",        cp / "stats_cartpole_posterior_vs_performance_sweep_mlp.json"),
        ("Persistence · logistic", cp / "stats_cartpole_posterior_vs_performance_sweep_persistence.json"),
        ("Persistence · MLP",      cp / "stats_cartpole_posterior_vs_performance_sweep_persistence_mlp.json"),
    )
    stats_by = {}
    for label, p in sources:
        if not p.exists():
            print(f"[cartpole_plots] skip 4x3 grid: missing {p}", flush=True)
            return False
        with open(p) as f:
            stats_by[label] = json.load(f)

    apply_style()
    fig, axes = plt.subplots(4, 3, figsize=(14.0, 16.0), sharey=True)
    method_order = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
    levels = ("easy", "medium", "hard")

    for row, (label, _) in enumerate(sources):
        s = stats_by[label]
        by_level: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for p in s["scatter_points"]:
            by_level.setdefault(p["level"], {}).setdefault(p["method"], []).append(
                (float(p["posterior_error"]), float(p["gap_closed"])),
            )
        for col, level in enumerate(levels):
            ax = axes[row, col]
            for method in method_order:
                pts = by_level.get(level, {}).get(method)
                if pts is None:
                    continue
                xs, ys = zip(*pts)
                show_legend = (row == 0 and col == 2)
                ax.scatter(
                    xs, ys, color=COLORS.get(method, "#666666"),
                    label=_METHOD_LABELS.get(method, method) if show_legend else None,
                    s=24, edgecolor="black", linewidth=0.3, alpha=0.8,
                )
            ax.axhline(0.0, color=COLORS["ppo"], linestyle="--", linewidth=0.7, alpha=0.5)
            ax.axhline(1.0, color=COLORS["oracle_ppo"], linestyle="--", linewidth=0.7, alpha=0.5)
            xs_all = [x for m in method_order for x, _ in by_level.get(level, {}).get(m, [])]
            ys_all = [y for m in method_order for _, y in by_level.get(level, {}).get(m, [])]
            if len(xs_all) >= 3 and np.std(xs_all) > 0 and np.std(ys_all) > 0:
                r_cell = float(np.corrcoef(xs_all, ys_all)[0, 1])
            else:
                r_cell = float("nan")
            if col == 0:
                ax.set_ylabel(f"{label}\nGap closed")
            ax.set_title(f"{level.capitalize()}  ·  r = {r_cell:+.3f}", fontsize=10)
            if row == 3:
                ax.set_xlabel("Posterior error")

    axes[0, 2].legend(**LEGEND_OUTSIDE_RIGHT)
    fig.suptitle(
        "CartPoleRegimeV1 — posterior decoding vs task performance across both difficulty axes\n"
        "Inversion robust at every level on both axes  ·  attenuates only where envelope is smallest (asymmetry-hard)",
        fontsize=12,
    )
    budget_annotation(
        fig,
        extra="12 panels (2 axes × 2 probes × 3 levels) · n=32 per panel · n_rollouts=200",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)
    return True


def main() -> None:
    plot_method_ladder(FIGURES_ROOT / "cartpole_method_ladder.png")
    plot_posterior_vs_performance(FIGURES_ROOT / "cartpole_posterior_vs_performance.png")
    plot_logistic_vs_mlp_side_by_side(
        FIGURES_ROOT / "cartpole_posterior_vs_performance_logistic_vs_mlp.png",
    )
    plot_cross_env_2x2(FIGURES_ROOT / "cross_env_decoupling_vs_inversion_2x2.png")
    plot_difficulty_sweep_returns(FIGURES_ROOT / "cartpole_difficulty_sweep_returns.png")
    plot_difficulty_sweep_scatter(FIGURES_ROOT / "cartpole_difficulty_sweep_scatter.png")
    plot_axis_returns(
        FIGURES_ROOT / "cartpole_persistence_sweep_returns.png", "persistence",
    )
    plot_two_axis_scatter_grid(
        FIGURES_ROOT / "cartpole_two_axis_scatter_grid.png",
    )


if __name__ == "__main__":
    main()
