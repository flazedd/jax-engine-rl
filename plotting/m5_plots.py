"""M5 plot module — figures for every M5 stage that has data on disk.

Produces:
  - step2_sweep.png         — Step-2 hidden_dim/kl_coef sweep on MM E_final
  - factorial_toys.png      — Step-3 24-config factorial across 3 toy envs
  - step3_mm_hypernet.png   — Step-3 MM hypernet probe (4 bars + reference lines)
  - probe_per_t.png         — Step-5 per-timestep regime classification accuracy
  - step4_ladder.png        — Step-4 4-cell ladder bar chart (when ready)
  - step4_learning_curves.png — Step-4 per-iter learning curves with seed CIs

Each plot function loads a stats JSON, produces one PNG, and tolerates
missing inputs (skips gracefully). The CLI entry generates everything it
can find data for.

Output directory: `figures/m5/`. Dpi=150, tight bounding box, applied via
`plotting.style.apply_style()`.
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
                   label=f"floor (agnostic-PPO M3) = {refs['agnostic']:.1f}")
    if not np.isnan(refs.get("belief", float("nan"))):
        ax.axhline(refs["belief"], color=COLORS["belief_ppo"], **style,
                   label=f"belief-PPO M3 = {refs['belief']:.1f}")
    if not np.isnan(refs.get("oracle", float("nan"))):
        ax.axhline(refs["oracle"], color=COLORS["oracle_ppo"], **style,
                   label=f"oracle-PPO M3 = {refs['oracle']:.1f}")


# ---------------------------------------------------------------------------
# Step-2 hyperparameter sweep on MM E_final
# ---------------------------------------------------------------------------


def plot_step2_sweep(out_path: Path) -> bool:
    stats_path = RESULTS_ROOT / "milestones" / "M5" / "stats_M5_step2_sweep.json"
    if not stats_path.exists():
        print(f"[m5_plots] skip step2_sweep: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    refs = _load_m3_refs()

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    rows = stats["configs"]
    labels = [r["label"] for r in rows]
    means = [r["final_return_mean"] for r in rows]
    cis = [r["final_return_ci95"] for r in rows]
    families = [r["family"] for r in rows]
    err = np.array([
        [m - lo, hi - m] for m, (lo, hi) in zip(means, cis)
    ]).T

    bar_colors = [COLORS["rl2"] if fam == "rl2" else COLORS["varibad"] for fam in families]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=err, color=bar_colors, capsize=2,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("final return (mean across 3 seeds)")
    ax.set_title("M5 Step-2 sweep on MM E_final (100 iter × n=3)")
    _draw_reference_lines(ax, refs)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Step-3 toy factorial — 24 configs × 3 envs
# ---------------------------------------------------------------------------


_TOY_ENVS = ["bandit", "gridworld", "regime_bandit"]
_TOY_ENV_LABELS = {
    "bandit": "Bandit (5-arm Bernoulli)",
    "gridworld": "Gridworld (random goal)",
    "regime_bandit": "Regime-switching bandit",
}
_VARIANT_ORDER = ("concat_nobonus", "hypernet_nobonus")
_VARIANT_LABELS = {
    "concat_nobonus": "concat",
    "hypernet_nobonus": "hypernet",
}


def plot_factorial_toys(out_path: Path) -> bool:
    stats_path = RESULTS_ROOT / "milestones" / "M5" / "stats_M5_factorial_toys.json"
    if not stats_path.exists():
        print(f"[m5_plots] skip factorial_toys: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)

    # Re-index configs by (method, env, variant_label). Tolerate older
    # JSONs that carry an `exploration_bonus` field; bonus configs are
    # ignored by the plot regardless (only `*_nobonus` variants are read).
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in stats["configs"]:
        bonus = r.get("exploration_bonus", False)
        variant = f"{r['integration']}_{'bonus' if bonus else 'nobonus'}"
        by_key[(r["method"], r["env"], variant)] = r

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)

    methods = ["rl2", "varibad"]
    method_labels = {"rl2": "RL²", "varibad": "VariBAD"}
    n_v = len(_VARIANT_ORDER)
    bar_w = 0.38

    for ax, env in zip(axes, _TOY_ENVS):
        for i_method, method in enumerate(methods):
            means = []
            errs_lo = []
            errs_hi = []
            for v in _VARIANT_ORDER:
                key = (method, env, v)
                if key in by_key:
                    r = by_key[key]
                    means.append(r["final_return_mean"])
                    lo, hi = r["final_return_ci95"]
                    errs_lo.append(r["final_return_mean"] - lo)
                    errs_hi.append(hi - r["final_return_mean"])
                else:
                    means.append(np.nan)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
            offset = (i_method - 0.5) * bar_w
            x = np.arange(n_v) + offset
            yerr = np.array([errs_lo, errs_hi])
            ax.bar(
                x, means, bar_w, yerr=yerr, color=COLORS[method],
                edgecolor="black", linewidth=0.4, capsize=2,
                label=method_labels[method],
            )
        # M4 baseline reference line per env (concat_nobonus = M4 default).
        m4_baselines = [
            by_key.get((m, env, "concat_nobonus"), {}).get("m4_baseline_mean")
            for m in methods
        ]
        for m, mv in zip(methods, m4_baselines):
            if mv is not None:
                ax.axhline(
                    mv, color=COLORS[m], linestyle="--", linewidth=0.8,
                    alpha=0.5,
                )

        ax.set_xticks(np.arange(n_v))
        ax.set_xticklabels([_VARIANT_LABELS[v] for v in _VARIANT_ORDER], fontsize=8)
        ax.set_title(_TOY_ENV_LABELS[env])
        ax.set_ylabel("final return (mean across 3 seeds)")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "M5 Step-3 factorial on toy envs — RL²/VariBAD × concat/hypernet",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


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
        ("RL²\nconcat", "m5_step2_rl2_h64_e01", COLORS["rl2"], 0.6),
        ("RL²\nhypernet", "m5_step3_rl2_hypernet", COLORS["rl2"], 1.0),
        ("VariBAD\nconcat", "m5_step2_varibad_kl10", COLORS["varibad"], 0.6),
        ("VariBAD\nhypernet", "m5_step3_varibad_hypernet", COLORS["varibad"], 1.0),
    ]
    means: list[float] = []
    cis: list[tuple[float, float]] = []
    labels: list[str] = []
    colors: list[Any] = []
    alphas: list[float] = []
    for label, exp_name, color, alpha in cells:
        m_path = RESULTS_ROOT / exp_name / "metrics.json"
        if not m_path.exists():
            print(f"[m5_plots] skip step3_mm_hypernet: missing {m_path}")
            return False
        with open(m_path) as f:
            m = json.load(f)
        means.append(float(m["final_return_mean"]))
        cis.append(tuple(m["final_return_ci95"]))
        labels.append(label)
        colors.append(color)
        alphas.append(alpha)

    refs = _load_m3_refs()
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    x = np.arange(len(labels))
    err = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T
    bars = ax.bar(
        x, means, yerr=err, capsize=3, edgecolor="black", linewidth=0.4,
    )
    for bar, color, alpha in zip(bars, colors, alphas):
        bar.set_facecolor(color)
        bar.set_alpha(alpha)
    for xi, mean in zip(x, means):
        ax.text(xi, mean + 1.5, f"{mean:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("final return (mean across 3 seeds, 100 iter)")
    ax.set_title("M5 Step-3 MM E_final — concat vs hypernet (half-budget probe)")
    _draw_reference_lines(ax, refs)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(80, max(refs.get("oracle", 200), max(means) + 10))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m5_plots] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Step-5 posterior probe per-timestep accuracy curves
# ---------------------------------------------------------------------------


def _method_label_and_color(exp_name: str) -> tuple[str, str]:
    """Map experiment_name to (display label, color)."""
    palette = {"rl2": COLORS["rl2"], "varibad": COLORS["varibad"]}
    for prefix, color in palette.items():
        if prefix in exp_name:
            label = (
                exp_name
                .replace("m5_step5_smoke_", "")
                .replace("m5_step4_", "")
                .replace("m5_step3_", "")
                .replace("_hypernet", " hypernet")
                .replace("_concat", " concat")
            )
            return label, color
    return exp_name, "#666666"


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
    bar_labels.append("analytical\nposterior")
    bar_means.append(ana_mean)
    bar_lo.append(ana_mean - ana_lo)
    bar_hi.append(ana_hi - ana_mean)
    bar_colors.append("#222222")

    x_bar = np.arange(len(bar_labels))
    yerr = np.array([bar_lo, bar_hi])
    bars = ax_bar.bar(
        x_bar, bar_means, yerr=yerr, capsize=3, edgecolor="black",
        linewidth=0.4,
    )
    for bar, color in zip(bars, bar_colors):
        bar.set_facecolor(color)
    for xi, mean in zip(x_bar, bar_means):
        ax_bar.text(xi, mean + 0.02, f"{mean:.2f}", ha="center", fontsize=9)
    ax_bar.axhline(chance, color="#999999", linestyle=":", linewidth=1.0,
                   label=f"chance (1/{n_classes})")
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(bar_labels, fontsize=8)
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_ylabel("regime classification accuracy (test set)")
    ax_bar.set_title("Headline test accuracy")
    ax_bar.legend(loc="lower right", fontsize=8)

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
                          color="#222222", linestyle="--", linewidth=1.2,
                          label="analytical posterior (reference)")
            analytical_drawn = True

    ax_curve.axhline(chance, color="#999999", linestyle=":", linewidth=1.0,
                     label=f"chance (1/{n_classes})")
    ax_curve.set_xlabel("timestep within episode")
    ax_curve.set_ylabel("regime classification accuracy")
    ax_curve.set_title(f"Per-timestep accuracy (rolling-mean window {smoothing_window})")
    ax_curve.set_ylim(0.0, 1.05)
    ax_curve.legend(loc="lower right", fontsize=8)

    fig.suptitle("M5 Step-5 — posterior-quality probe", y=1.02)
    fig.tight_layout()
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
        alpha = 0.6 if integ == "concat" else 1.0
        bars = ax.bar(
            x, means, bar_w, yerr=yerr, capsize=3, edgecolor="black",
            linewidth=0.4, label=integ,
        )
        for bar, m in zip(bars, methods):
            bar.set_facecolor(COLORS[m])
            bar.set_alpha(alpha)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(["RL²", "VariBAD"])
    ax.set_ylabel("final return (mean across n=8 seeds, 200 iter)")
    ax.set_title("M5 Step-4 ladder — 4-cell factorial on MM E_final")
    _draw_reference_lines(ax, refs)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
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
    for cell_key, c in cells.items():
        # cell_key e.g. "rl2_hypernet"
        method = cell_key.split("_")[0]
        integ = "_".join(cell_key.split("_")[1:])
        color = COLORS.get(method, "#666666")
        linestyle = "--" if integ == "concat" else "-"
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
        ax.plot(iters, mean, color=color, linestyle=linestyle, label=cell_key)
        ax.fill_between(iters, lo, hi, color=color, alpha=0.15)

    _draw_reference_lines(ax, refs)
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean return (across seeds, shaded = 95% CI)")
    ax.set_title("M5 Step-4 learning curves — concat vs hypernet × method")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
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

    written = 0
    for name, fn in [
        ("step2_sweep.png", plot_step2_sweep),
        ("factorial_toys.png", plot_factorial_toys),
        ("step3_mm_hypernet.png", plot_step3_mm_hypernet),
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
