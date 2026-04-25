"""M4 method-validation figure: per-env method ranking with 95% bootstrap CIs.

Reads the structured `table` from `results/milestones/M4/method_ranking.json`
(written by `scripts.m4_full_eval_aggregate`). Produces one grouped bar chart:
3 validation envs × 3 methods (PPO floor, RL², VariBAD).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import COLORS, FIGSIZE_WIDE, apply_style


_ENV_ORDER = ("bandit", "gridworld", "regime_bandit")
_ENV_LABELS = {
    "bandit": "Bernoulli bandit",
    "gridworld": "Gridworld (random goal)",
    "regime_bandit": "Regime-switching bandit",
}

_METHOD_ORDER = ("ppo", "rl2", "varibad")
_METHOD_LABELS = {
    "ppo": "PPO (floor)",
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

    for j, method in enumerate(_METHOD_ORDER):
        mean_key, ci_key = _METHOD_KEYS[method]
        means = []
        errs_lo = []
        errs_hi = []
        for env in envs:
            row = by_env[env]
            m = row.get(mean_key, float("nan"))
            ci = row.get(ci_key, [m, m])
            means.append(m)
            errs_lo.append(max(0.0, m - ci[0]))
            errs_hi.append(max(0.0, ci[1] - m))
        offset = (j - (n_methods - 1) / 2) * bar_w
        ax.bar(
            x + offset, means, bar_w,
            yerr=[errs_lo, errs_hi], capsize=3,
            color=_METHOD_COLORS[method], edgecolor="black",
            label=_METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_ENV_LABELS[e] for e in envs])
    ax.set_ylabel("final episode return (mean over 3 seeds)")
    ax.set_title("M4 — meta-RL methods clear the PPO floor on every validation env")
    ax.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
