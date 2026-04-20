"""M3 / RQ1 figures: ceilings bar, learning curves, and gap-fraction stack.

Every plot takes plain numpy / dict inputs and writes a PNG. The stats JSON
produced by `scripts.make_milestone M3` is the machine-readable contract;
these plots exist for the human.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import COLORS, FIGSIZE_STANDARD, FIGSIZE_WIDE, apply_style


_METHOD_ORDER = ("regime_agnostic_ppo", "belief_ppo", "oracle_ppo", "per_regime_ppo")
_METHOD_LABELS = {
    "regime_agnostic_ppo": "regime-agnostic",
    "belief_ppo": "Belief-PPO",
    "oracle_ppo": "Oracle-PPO",
    "per_regime_ppo": "per-regime",
}
_METHOD_COLORS = {
    "regime_agnostic_ppo": COLORS.get("ppo", "#1f77b4"),
    "belief_ppo": COLORS.get("belief_ppo", "#9467bd"),
    "oracle_ppo": COLORS.get("oracle_ppo", "#8c564b"),
    "per_regime_ppo": COLORS.get("per_regime_ppo", "#17becf"),
}

_GAP_COLORS = {
    "shared_network_cost": "#17becf",
    "inference_cost": "#8c564b",
    "compromise_policy_cost": "#1f77b4",
}
_GAP_LABELS = {
    "shared_network_cost": "shared-network",
    "inference_cost": "inference",
    "compromise_policy_cost": "compromise-policy",
}


def plot_rq1_ceilings_bar(
    reference_levels: dict[str, dict],
    gap_components: dict[str, dict],
    output_path: Path,
) -> None:
    """Four bars (one per reference level) with brackets labeling gap components.

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

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    xs = np.arange(len(names))
    ax.bar(
        xs, means, yerr=[errs_lo, errs_hi], capsize=5,
        color=colors, edgecolor="black",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([_METHOD_LABELS[n] for n in names])
    ax.set_ylabel("episode return (mean over seeds)")
    ax.set_title("RQ1 — reference levels and gap decomposition")

    # Brackets labeling the three gap components between consecutive bars.
    # Order of bars is [agnostic, belief, oracle, per_regime], so:
    #   agnostic→belief:    compromise_policy_cost
    #   belief→oracle:      inference_cost
    #   oracle→per_regime:  shared_network_cost
    bracket_pairs = [
        (0, 1, "compromise_policy_cost"),
        (1, 2, "inference_cost"),
        (2, 3, "shared_network_cost"),
    ]
    top = max(ci[1] for ci in cis)
    span = top - min(m - e for m, e in zip(means, errs_lo))
    bracket_y = top + 0.05 * span
    step = 0.08 * span
    for i, (a, b, name) in enumerate(bracket_pairs):
        y = bracket_y + i * step
        ax.plot([xs[a], xs[a], xs[b], xs[b]], [y - 0.01 * span, y, y, y - 0.01 * span],
                color="black", linewidth=0.9)
        gap = gap_components.get(name, {})
        absolute = gap.get("absolute", means[b] - means[a])
        frac = gap.get("fraction_of_total")
        label = f"{_GAP_LABELS[name]}: {absolute:.2f}"
        if frac is not None:
            label += f" ({frac*100:.0f}%)"
        ax.text(
            0.5 * (xs[a] + xs[b]), y + 0.01 * span, label,
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(top=bracket_y + (len(bracket_pairs) + 1) * step)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rq1_learning_curves(
    per_method_curves: dict[str, dict],
    output_path: Path,
) -> None:
    """Four learning curves on shared axes.

    per_method_curves: {method: {
        "mean_return_per_iter": [float, ...],
        "per_seed_mean_return_per_iter": [[float, ...], ...],
    }}
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    for name in _METHOD_ORDER:
        m = per_method_curves.get(name)
        if m is None:
            continue
        mean = np.asarray(m["mean_return_per_iter"], dtype=float)
        per_seed = np.asarray(m.get("per_seed_mean_return_per_iter", []), dtype=float)
        iters = np.arange(mean.size)
        color = _METHOD_COLORS[name]
        ax.plot(iters, mean, color=color, label=_METHOD_LABELS[name])
        if per_seed.ndim == 2 and per_seed.shape[0] > 1:
            lo = np.percentile(per_seed, 2.5, axis=0)
            hi = np.percentile(per_seed, 97.5, axis=0)
            ax.fill_between(iters, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode return")
    ax.set_title("RQ1 — reference-level learning curves")
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rq1_gap_fractions(
    gap_components: dict[str, dict],
    output_path: Path,
) -> None:
    """Single stacked bar showing fractional contribution of each gap component."""
    apply_style()
    order = ("compromise_policy_cost", "inference_cost", "shared_network_cost")
    fractions = [max(0.0, float(gap_components[n]["fraction_of_total"])) for n in order]
    absolutes = [float(gap_components[n]["absolute"]) for n in order]

    fig, ax = plt.subplots(figsize=(3.2, 4.0))
    bottom = 0.0
    for name, frac, absolute in zip(order, fractions, absolutes):
        ax.bar(
            [0], [frac], bottom=bottom,
            color=_GAP_COLORS[name], edgecolor="black",
            label=f"{_GAP_LABELS[name]} ({frac*100:.0f}%, Δ={absolute:.2f})",
        )
        bottom += frac
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("fraction of total gap")
    ax.set_title("RQ1 — gap fractions")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
