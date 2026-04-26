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

from envs.market_making_v1 import (
    ACTION_FAVOR_ASK,
    ACTION_FAVOR_BID,
    ACTION_SYM,
    MarketMakingV1,
)
from oracles.value_iteration import VIResult
from plotting.style import COLORS, FIGSIZE_STANDARD, FIGSIZE_WIDE, apply_style


_REGIME_NAMES = ("noise", "bull", "bear")
_ACTION_NAMES = ("sym", "favor_ask", "favor_bid")
_ACTION_COLORS = ("#cccccc", "#d62728", "#2ca02c")


def _regime_label(r: int) -> str:
    if r < len(_REGIME_NAMES):
        return _REGIME_NAMES[r]
    return f"regime_{r}"


def plot_policy_heatmap(vi: VIResult, env: MarketMakingV1, output_path: Path) -> None:
    """Heatmap of VI-optimal action per (regime, inventory) state."""
    apply_style()
    policy = vi.policy  # [n_inv, n_reg]
    n_inv, n_reg = policy.shape
    inv_levels = np.arange(-env.inventory_max, env.inventory_max + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    # policy is [n_inv, n_reg] → heatmap rows = regimes, cols = inventory.
    grid = policy.T
    cmap = plt.get_cmap("Set1", 3)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(range(n_reg))
    ax.set_yticklabels([_regime_label(r) for r in range(n_reg)])
    ax.set_xticks(range(n_inv))
    ax.set_xticklabels(inv_levels)
    ax.set_xlabel("inventory q")
    ax.set_ylabel("regime")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(list(_ACTION_NAMES))
    ax.set_title("M2 R1 — VI-optimal action per (regime, inventory)")

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
    )
    mean = float(data.mean()) if data.size else 0.0
    ax.axvline(
        mean,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"mean = {mean:.3f}",
    )
    ax.set_xlabel("relative value loss (V^π_true − V^π_other) / V^π_true")
    ax.set_ylabel("% of (r_true, r_other, inventory) buckets")
    ax.set_title("M2 R1 — policy-commitment loss distribution")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_regime_ppo(
    per_regime_metrics: Iterable[dict],
    vi_per_regime_returns: np.ndarray,
    output_path: Path,
) -> None:
    """One subplot per regime: PPO learning curve with VI horizontal line."""
    apply_style()
    metrics = list(per_regime_metrics)
    n_reg = len(metrics)
    fig, axes = plt.subplots(1, n_reg, figsize=(4.5 * n_reg, 3.2), sharey=True)
    if n_reg == 1:
        axes = [axes]
    for r, (ax, m) in enumerate(zip(axes, metrics)):
        curve = np.asarray(m["mean_return_per_iter"])
        per_seed = np.asarray(m["per_seed_mean_return_per_iter"])  # [seeds, T]
        iters = np.arange(curve.size)
        color = COLORS.get("per_regime_ppo", "#17becf")
        ax.plot(iters, curve, color=color, label="PPO mean")
        if per_seed.shape[0] > 1:
            lo = np.percentile(per_seed, 2.5, axis=0)
            hi = np.percentile(per_seed, 97.5, axis=0)
            ax.fill_between(iters, lo, hi, color=color, alpha=0.2)
        ax.axhline(
            vi_per_regime_returns[r],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"VI = {vi_per_regime_returns[r]:.1f}",
        )
        ax.set_title(f"regime {r} ({_regime_label(r)})")
        ax.set_xlabel("iteration")
        if r == 0:
            ax.set_ylabel("episode return")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("M2 R2 — per-regime PPO vs VI-optimal")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_posterior_entropy(ent_curve: np.ndarray, output_path: Path) -> None:
    """Entropy of analytical HMM posterior, averaged over simulated trajectories."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    # Drop the terminal step: the BeliefObsEnv wrapper resets the posterior
    # to the flat prior on done=True, so ent_curve[-1] is log(n_regimes)
    # by construction — a recording artifact, not a real re-entropification.
    curve = ent_curve[:-1] if ent_curve.size > 1 else ent_curve
    t = np.arange(curve.size)
    ax.plot(t, curve, color=COLORS.get("belief_ppo", "#9467bd"))
    ax.axhline(
        np.log(3),
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="log(3) = 1.099 (flat prior)",
    )
    ax.set_xlabel("timestep within episode")
    ax.set_ylabel("mean posterior entropy (nats)")
    ax.set_title("M2 R4 — posterior entropy over time (random-policy rollouts)")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_belief_ppo_gap(
    bars: dict[str, tuple[float, list[float]]], output_path: Path
) -> None:
    """Bar chart: regime-agnostic, Belief-PPO, Oracle-PPO, with CIs."""
    apply_style()
    names = list(bars.keys())
    means = [bars[k][0] for k in names]
    cis = [bars[k][1] for k in names]
    errs_lo = [m - ci[0] for m, ci in zip(means, cis)]
    errs_hi = [ci[1] - m for m, ci in zip(means, cis)]
    palette = {
        "regime_agnostic": COLORS.get("ppo", "#1f77b4"),
        "belief": COLORS.get("belief_ppo", "#9467bd"),
        "oracle": COLORS.get("oracle_ppo", "#8c564b"),
    }
    label_map = {
        "regime_agnostic": "Regime-agnostic PPO",
        "belief": "Belief-PPO",
        "oracle": "Oracle-PPO",
    }
    colors = [palette.get(n, "#888") for n in names]
    display_labels = [label_map.get(n, n) for n in names]
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=[errs_lo, errs_hi], capsize=5, color=colors, edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(display_labels, rotation=0)
    ax.set_ylabel("episode return")
    ax.set_title("M2 R4 — Regime-agnostic PPO vs Belief-PPO vs Oracle-PPO")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
