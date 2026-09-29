"""M2 figures: R1 policy heatmap + value-loss distribution, R2 per-regime
learning curves, R4 posterior-entropy curve + Belief-PPO bars.

R3 has no figure of its own — the R4 Belief-PPO bar chart is a strict
superset (regime_agnostic → belief → oracle), and R3's pass criterion is
the JSON field `gap_to_ci_ratio` rather than anything visual.

Every plot takes plain numpy / dict inputs and writes a PNG. Claude Code
reads the stats JSON produced by verify_requirements.py; these plots exist
for the human to sanity-check.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import VIResult
from plotting.style import (
    COLORS,
    FIGSIZE_STANDARD,
    FIGSIZE_WIDE,
    LEGEND_OUTSIDE_RIGHT,
    PALETTE,
    apply_style,
    budget_annotation,
    polish,
)


_REGIME_NAMES = ("Regime 0", "Regime 1", "Regime 2")
_ACTION_NAMES = ("sym", "favor_ask", "favor_bid")
_ACTION_LABELS = ("Bid 1 / ask 1", "Bid 3 / ask 1", "Bid 1 / ask 3")


def _regime_label(r: int) -> str:
    if r < len(_REGIME_NAMES):
        return _REGIME_NAMES[r]
    return f"regime_{r}"


def _budget_from_metrics(metrics: dict) -> dict:
    return {
        "iterations": int(metrics.get("iterations", 0)) or None,
        "parallel_envs": int(metrics.get("parallel_envs", 0)) or None,
        "rollout_length": int(metrics.get("rollout_length", 0)) or None,
        "num_seeds": int(metrics.get("num_seeds", 0)) or None,
    }


def plot_policy_heatmap(vi: VIResult, env: MarketMakingV1, output_path: Path) -> None:
    """Heatmap of VI-optimal action per (regime, inventory) state."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    apply_style()
    policy = vi.policy  # [n_inv, n_reg]
    n_inv, n_reg = policy.shape
    inv_levels = np.arange(-env.inventory_max, env.inventory_max + 1)
    grid = policy.T  # rows = regimes, cols = inventory.

    # Semantic palette: the symmetric quote is neutral, while the two
    # directional quotes get a warm/cool pair so the bid/ask lean reads at a
    # glance. Muted tones (slate / amber / steel-blue), colourblind-safe.
    # Brand palette as three distinct categorical action colours: neutral slate
    # for the symmetric quote, the warm/cool brand pair for the two directional
    # quotes (amber = favor ask, teal = favor bid).
    action_colors = [PALETTE["concat"], PALETTE["analytical"], PALETTE["hyper"]]
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    xedges = np.arange(n_inv + 1) - 0.5
    yedges = np.arange(n_reg + 1) - 0.5
    ax.pcolormesh(xedges, yedges, grid, cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=1.5)
    ax.invert_yaxis()  # regime 0 on top, matching the row order
    ax.set_yticks(range(n_reg))
    ax.set_yticklabels([_regime_label(r) for r in range(n_reg)])
    ax.set_xticks(range(n_inv))
    ax.set_xticklabels(inv_levels)
    ax.set_xlabel("Inventory $q$")
    ax.set_aspect("auto")
    ax.tick_params(length=0)
    ax.grid(False)  # white cell borders carry the structure; drop the style grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Replace the colorbar with a proper legend so the chart self-documents
    # the action encoding (per the every-element-in-legend convention).
    legend_handles = [
        Patch(facecolor=action_colors[i], edgecolor="#888888", linewidth=0.5,
              label=_ACTION_LABELS[i])
        for i in range(3)
    ]
    ax.legend(handles=legend_handles, **LEGEND_OUTSIDE_RIGHT)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_value_loss_distribution(
    per_state_rel_loss: np.ndarray, output_path: Path
) -> None:
    """Histogram of forward-rolled relative value loss from committing to
    the wrong regime's policy.

    Each datapoint is a (true regime, other regime, inventory) bucket:
    (V^π_true(s) − V^π_other(s)) / V^π_true(s), evaluated on the locked
    true-regime chain via exact policy evaluation. A long right tail means
    there exist states where committing to the wrong regime's policy costs
    a substantial fraction of attainable value over the full horizon.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    data = np.asarray(per_state_rel_loss, dtype=float)
    weights = (
        np.full(data.size, 100.0 / data.size) if data.size else np.zeros(0)
    )
    ax.hist(
        data,
        bins=20,
        weights=weights,
        color=COLORS.get("ppo", "#1f77b4"),
        edgecolor="black",
        label="(r_true, r_other, q) buckets",
    )
    mean = float(data.mean()) if data.size else 0.0
    ax.axvline(
        mean,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean:.3f}",
    )
    ax.set_xlabel("Relative value loss (V^π_true − V^π_other) / V^π_true")
    ax.set_title(
        "MarketMakingV1 — wrong-regime policy-commitment loss distribution\n"
        "% of (r_true, r_other, inventory) buckets"
    )
    ax.legend(**LEGEND_OUTSIDE_RIGHT)
    fig.tight_layout()
    fig.subplots_adjust(right=0.65, bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_regime_ppo(
    per_regime_metrics: Iterable[dict],
    vi_per_regime_returns: np.ndarray,
    output_path: Path,
) -> None:
    """One subplot per regime, stacked vertically: PPO learning curve with VI line."""
    apply_style()
    metrics = list(per_regime_metrics)
    n_reg = len(metrics)
    fig, axes = plt.subplots(n_reg, 1, figsize=(11.0, 3.5 * n_reg), sharex=True)
    if n_reg == 1:
        axes = [axes]
    color = PALETTE["accent"]
    for r, (ax, m) in enumerate(zip(axes, metrics)):
        curve = np.asarray(m["mean_return_per_iter"])
        per_seed = np.asarray(m["per_seed_mean_return_per_iter"])  # [seeds, T]
        iters = np.arange(curve.size)
        ax.plot(iters, curve, color=color, linewidth=2.0, label="PPO (mean over seeds)")
        if per_seed.shape[0] > 1:
            n_seeds = per_seed.shape[0]
            lo = np.percentile(per_seed, 2.5, axis=0)
            hi = np.percentile(per_seed, 97.5, axis=0)
            ax.fill_between(iters, lo, hi, color=color, alpha=0.18,
                            label=f"Per-seed spread: 2.5–97.5th percentile (n={n_seeds})")
        ax.axhline(
            vi_per_regime_returns[r],
            color="#555555", linestyle="--", linewidth=1.2,
        )
        ax.text(
            0.99, vi_per_regime_returns[r],
            f"  VI optimum = {vi_per_regime_returns[r]:.1f}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=11, color="#444444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
        )
        ax.set_title(f"Regime {r} ({_regime_label(r)})", fontsize=13, loc="left")
        ax.set_ylabel("Episode return", fontsize=11)
        ax.tick_params(axis="both", labelsize=11)
        polish(ax)
    axes[-1].set_xlabel("Iteration", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, fontsize=12,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_posterior_entropy(ent_curve: np.ndarray, output_path: Path) -> None:
    """Entropy of analytical HMM posterior, averaged over simulated trajectories."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    # Drop the terminal step: the BeliefObsEnv wrapper resets the posterior
    # to the flat prior on done=True, so ent_curve[-1] is log(n_regimes)
    # by construction — a recording artifact, not a real re-entropification.
    curve = ent_curve[:-1] if ent_curve.size > 1 else ent_curve
    t = np.arange(curve.size)
    ax.plot(t, curve, color=PALETTE["accent"],
            linewidth=2.2, label="Mean posterior entropy")
    ax.axhline(
        np.log(3), color="#555555", linestyle="--", linewidth=1.2,
        label="log(3) = 1.099 (flat prior)",
    )
    ax.set_xlabel("Timestep within episode", fontsize=12)
    ax.set_ylabel("Posterior entropy (nats)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    polish(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, fontsize=12,
        bbox_to_anchor=(0.5, -0.005), frameon=True,
        facecolor="white", edgecolor="#cccccc", framealpha=1.0,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_belief_ppo_gap(
    bars: dict[str, tuple[float, list[float]]], output_path: Path,
    budget: dict | None = None,
) -> None:
    """Bar chart: regime-agnostic, Belief-PPO, Oracle-PPO, with CIs."""
    apply_style()
    names = list(bars.keys())
    means = [bars[k][0] for k in names]
    cis = [bars[k][1] for k in names]
    errs_lo = [m - ci[0] for m, ci in zip(means, cis)]
    errs_hi = [ci[1] - m for m, ci in zip(means, cis)]
    palette = {
        "regime_agnostic": PALETTE["floor"],
        "belief": PALETTE["belief"],
        "oracle": PALETTE["oracle"],
    }
    label_map = {
        "regime_agnostic": "Regime-agnostic PPO",
        "belief": "Belief-PPO",
        "oracle": "Oracle-PPO",
    }
    colors = [palette.get(n, "#888") for n in names]
    display_labels = [label_map.get(n, n) for n in names]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xs = np.arange(len(names))
    ax.bar(
        xs, means, yerr=[errs_lo, errs_hi], capsize=4,
        color=colors, edgecolor="white", linewidth=1.2, zorder=3,
        error_kw={"ecolor": "#3a3a3a", "elinewidth": 1.2},
    )
    # Bold value labels above the upper CI cap.
    for x, mean, ci in zip(xs, means, cis):
        ax.annotate(f"{mean:.1f}", xy=(x, ci[1]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels(display_labels)
    polish(ax)
    if budget:
        budget_annotation(fig, **budget)
    fig.tight_layout()
    fig.subplots_adjust(right=0.65, bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
