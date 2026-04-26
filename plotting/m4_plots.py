"""Method-validation figure: per-env method ranking with 95% bootstrap CIs.

Reads the structured `table` from `results/milestones/M4/method_ranking.json`
(written by `scripts.m4_full_eval_aggregate`). Produces one grouped bar chart:
3 validation envs × 3 methods (PPO floor, RL², VariBAD).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plotting.style import (
    COLORS,
    FIGSIZE_WIDE,
    LEGEND_OUTSIDE_RIGHT,
    apply_style,
    budget_annotation,
)


_ENV_ORDER = ("bandit", "gridworld", "regime_bandit")
_ENV_LABELS = {
    "bandit": "Bernoulli bandit",
    "gridworld": "Gridworld (random goal)",
    "regime_bandit": "Regime-switching bandit",
}

_METHOD_ORDER = ("ppo", "rl2", "varibad")
_METHOD_LABELS = {
    "ppo": "PPO floor",
    "rl2": "RL²",
    "varibad": "VariBAD",
}
_METHOD_COLORS = {
    "ppo": COLORS["ppo"],
    "rl2": COLORS["rl2"],
    "varibad": COLORS["varibad"],
}
_METHOD_KEYS = {
    "ppo": ("ppo_floor", "ppo_ci"),
    "rl2": ("rl2", "rl2_ci"),
    "varibad": ("varibad", "varibad_ci"),
}


def plot_m4_method_ranking(
    table: list[dict[str, Any]],
    output_path: Path,
    budget: dict | None = None,
) -> None:
    """Grouped bar chart: 3 envs × 3 methods, error bars = 95% bootstrap CI across seeds."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    by_env = {row["env"]: row for row in table}
    envs = [e for e in _ENV_ORDER if e in by_env]
    n_envs = len(envs)
    n_methods = len(_METHOD_ORDER)
    bar_w = 0.8 / n_methods
    x = np.arange(n_envs)

    # Track everything we draw so the value labels can use the right CIs.
    drawn: dict[str, list[tuple[float, float, float, float]]] = {}  # method -> [(x_pos, mean, lo, hi)]

    for j, method in enumerate(_METHOD_ORDER):
        mean_key, ci_key = _METHOD_KEYS[method]
        means: list[float] = []
        errs_lo: list[float] = []
        errs_hi: list[float] = []
        positions: list[float] = []
        for i_env, env in enumerate(envs):
            row = by_env[env]
            m = row.get(mean_key, float("nan"))
            ci = row.get(ci_key, [m, m])
            means.append(m)
            errs_lo.append(max(0.0, m - ci[0]))
            errs_hi.append(max(0.0, ci[1] - m))
            positions.append(x[i_env] + (j - (n_methods - 1) / 2) * bar_w)
            drawn.setdefault(method, []).append((positions[-1], m, ci[0], ci[1]))
        ax.bar(
            positions, means, bar_w,
            yerr=[errs_lo, errs_hi], capsize=3,
            color=_METHOD_COLORS[method], edgecolor="black",
        )

    # Value labels inside each bar, just below the lower CI cap.
    for method, items in drawn.items():
        for x_pos, mean, lo, hi in items:
            if np.isnan(mean):
                continue
            ax.annotate(f"{mean:.1f}", xy=(x_pos, lo),
                        xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=7, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([_ENV_LABELS[e] for e in envs])
    ax.set_ylabel("Final episode return (mean over seeds)")
    ax.set_title(
        "Toy environments — meta-RL methods clear the PPO floor"
    )

    legend_handles = [
        Patch(facecolor=_METHOD_COLORS[m], edgecolor="black",
              linewidth=0.4, label=_METHOD_LABELS[m])
        for m in _METHOD_ORDER
    ]
    ax.legend(handles=legend_handles, **LEGEND_OUTSIDE_RIGHT)
    if budget:
        budget_annotation(fig, **budget)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78, bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
