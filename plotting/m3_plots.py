"""M3 figures: ceilings bar, learning curves, and gap-fraction stack.

These give the human a visual on the reference levels (regime-agnostic
PPO floor, Belief-PPO and Oracle-PPO ceilings) and the decomposition of
the optimality gap into its inference-cost and compromise-policy-cost
components. The stats JSON produced by `scripts.make_milestone M3` is the
machine-readable contract; these plots exist for the human.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plotting.style import (
    COLORS,
    FIGSIZE_STANDARD,
    FIGSIZE_WIDE,
    LEGEND_OUTSIDE_RIGHT,
    apply_style,
    budget_annotation,
)


_METHOD_ORDER = ("regime_agnostic_ppo", "belief_ppo", "oracle_ppo")
_METHOD_LABELS = {
    "regime_agnostic_ppo": "Regime-agnostic PPO",
    "belief_ppo": "Belief-PPO",
    "oracle_ppo": "Oracle-PPO",
}
_METHOD_COLORS = {
    "regime_agnostic_ppo": COLORS.get("ppo", "#1f77b4"),
    "belief_ppo": COLORS.get("belief_ppo", "#9467bd"),
    "oracle_ppo": COLORS.get("oracle_ppo", "#8c564b"),
}

_GAP_COLORS = {
    "inference_cost": "#8c564b",
    "compromise_policy_cost": "#1f77b4",
}
_GAP_LABELS = {
    "inference_cost": "Inference cost",
    "compromise_policy_cost": "Compromise-policy cost",
}
# Compact labels for in-chart bracket annotations where horizontal space
# is tight; the legend / longer prose can use the full names above.
_GAP_LABELS_SHORT = {
    "inference_cost": "Inference",
    "compromise_policy_cost": "Compromise-policy",
}


def _budget_from_metrics(metrics: dict) -> dict:
    return {
        "iterations": int(metrics.get("iterations", 0)) or None,
        "parallel_envs": int(metrics.get("parallel_envs", 0)) or None,
        "rollout_length": int(metrics.get("rollout_length", 0)) or None,
        "num_seeds": int(metrics.get("num_seeds", 0)) or None,
    }


def plot_rq1_ceilings_bar(
    reference_levels: dict[str, dict],
    gap_components: dict[str, dict],
    output_path: Path,
    budget: dict | None = None,
) -> None:
    """Three bars (one per reference level) with brackets labeling gap components.

    reference_levels: {method: {"mean": float, "ci": [lo, hi], ...}, ...}
    gap_components: {name: {"absolute": float, "fraction_of_total": float}, ...}
    """
    apply_style()
    names = list(_METHOD_ORDER)
    means = [reference_levels[n]["mean"] for n in names]
    cis = [reference_levels[n]["ci"] for n in names]
    errs_lo = [m - ci[0] for m, ci in zip(means, cis)]
    errs_hi = [ci[1] - m for m, ci in zip(means, cis)]
    colors = [_METHOD_COLORS[n] for n in names]
    display_labels = [_METHOD_LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    xs = np.arange(len(names))
    ax.bar(
        xs, means, yerr=[errs_lo, errs_hi], capsize=5,
        color=colors, edgecolor="black",
    )
    # Value labels inside each bar, just below the lower CI cap so they
    # never overlap the vertical CI line.
    for x, mean, ci in zip(xs, means, cis):
        ax.annotate(f"{mean:.1f}", xy=(x, ci[0]),
                    xytext=(0, -3), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Episode return (mean over seeds)")
    ax.set_title(
        "MarketMakingV1 — reference levels and gap decomposition"
    )

    # Brackets labeling the two gap components between consecutive bars.
    # Order of bars is [agnostic, belief, oracle], so:
    #   agnostic→belief: compromise_policy_cost
    #   belief→oracle:   inference_cost
    bracket_pairs = [
        (0, 1, "compromise_policy_cost"),
        (1, 2, "inference_cost"),
    ]
    top = max(ci[1] for ci in cis)
    span = top - min(m - e for m, e in zip(means, errs_lo))
    bracket_y = top + 0.05 * span
    step = 0.08 * span
    for i, (a, b, name) in enumerate(bracket_pairs):
        y = bracket_y + i * step
        ax.plot([xs[a], xs[a], xs[b], xs[b]],
                [y - 0.01 * span, y, y, y - 0.01 * span],
                color="black", linewidth=0.9)
        gap = gap_components.get(name, {})
        absolute = gap.get("absolute", means[b] - means[a])
        frac = gap.get("fraction_of_total")
        label = f"{_GAP_LABELS_SHORT[name]}: Δ{absolute:.1f}"
        if frac is not None:
            label += f" ({frac*100:.0f}%)"
        ax.text(
            0.5 * (xs[a] + xs[b]), y + 0.01 * span, label,
            ha="center", va="bottom", fontsize=7,
        )
    # Leave extra headroom so the topmost bracket label clears the chart top.
    ax.set_ylim(top=bracket_y + (len(bracket_pairs) + 2) * step)

    # Legend lists every bar plus a textual note for the bracket markings.
    legend_handles = [
        Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=lbl)
        for c, lbl in zip(colors, display_labels)
    ]
    ax.legend(handles=legend_handles, **LEGEND_OUTSIDE_RIGHT)
    if budget:
        budget_annotation(fig, **budget)
    fig.tight_layout()
    fig.subplots_adjust(right=0.72, bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rq1_learning_curves(
    per_method_curves: dict[str, dict],
    output_path: Path,
) -> None:
    """Three learning curves on shared axes.

    per_method_curves: {method: {
        "mean_return_per_iter": [float, ...],
        "per_seed_mean_return_per_iter": [[float, ...], ...],
        "iterations": int, "num_seeds": int, "parallel_envs": int, "rollout_length": int
    }}
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sample_metrics: dict | None = None
    for name in _METHOD_ORDER:
        m = per_method_curves.get(name)
        if m is None:
            continue
        sample_metrics = sample_metrics or m
        mean = np.asarray(m["mean_return_per_iter"], dtype=float)
        per_seed = np.asarray(m.get("per_seed_mean_return_per_iter", []), dtype=float)
        iters = np.arange(mean.size)
        color = _METHOD_COLORS[name]
        ax.plot(iters, mean, color=color, label=_METHOD_LABELS[name])
        if per_seed.ndim == 2 and per_seed.shape[0] > 1:
            lo = np.percentile(per_seed, 2.5, axis=0)
            hi = np.percentile(per_seed, 97.5, axis=0)
            ax.fill_between(iters, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Episode return (mean over seeds, shaded = 95% CI)")
    ax.set_title("MarketMakingV1 — reference-level learning curves")
    ax.legend(**LEGEND_OUTSIDE_RIGHT)
    if sample_metrics is not None:
        budget_annotation(fig, **_budget_from_metrics(sample_metrics))
    fig.tight_layout()
    fig.subplots_adjust(right=0.65, bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rq1_gap_fractions(
    gap_components: dict[str, dict],
    output_path: Path,
) -> None:
    """Single stacked bar showing fractional contribution of each gap component."""
    apply_style()
    order = ("compromise_policy_cost", "inference_cost")
    fractions = [max(0.0, float(gap_components[n]["fraction_of_total"])) for n in order]
    absolutes = [float(gap_components[n]["absolute"]) for n in order]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    bottom = 0.0
    legend_handles = []
    for name, frac, absolute in zip(order, fractions, absolutes):
        label = f"{_GAP_LABELS[name]} ({frac*100:.0f}%, Δ={absolute:.2f})"
        ax.bar(
            [0], [frac], bottom=bottom,
            color=_GAP_COLORS[name], edgecolor="black",
            label=label,
        )
        legend_handles.append(
            Patch(facecolor=_GAP_COLORS[name], edgecolor="black",
                  linewidth=0.4, label=label)
        )
        bottom += frac
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("Fraction of total optimality gap")
    ax.set_title("MarketMakingV1 — gap-component fractions")
    ax.legend(handles=legend_handles, **LEGEND_OUTSIDE_RIGHT)
    fig.tight_layout()
    fig.subplots_adjust(right=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
