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
    COLORS, LEGEND_OUTSIDE_RIGHT, apply_style, budget_annotation,
)

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
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    # Reference horizontal dashed lines.
    floor_mean = float(ref_data["regime_agnostic"].mean())
    belief_mean = float(ref_data["belief"].mean())
    oracle_mean = float(ref_data["oracle"].mean())
    ax.axhline(floor_mean, color=COLORS["ppo"], linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Regime-agnostic PPO (floor) = {floor_mean:.2f}")
    ax.axhline(belief_mean, color=COLORS["belief_ppo"], linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Belief-PPO = {belief_mean:.2f}")
    ax.axhline(oracle_mean, color=COLORS["oracle_ppo"], linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Oracle-PPO (ceiling) = {oracle_mean:.2f}")

    # Bars for the 4 meta-RL cells.
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

    bar_colors = [COLORS[m] for m in metas]
    ax.bar(xs, means, color=bar_colors, alpha=0.85, edgecolor="black",
           linewidth=0.7, yerr=[err_low, err_high], capsize=4,
           error_kw=dict(ecolor="black", lw=1.0))

    # In-bar labels with the mean.
    for x, m, mu in zip(xs, metas, means):
        ax.text(x, mu - 1.5, f"{mu:.1f}", ha="center", va="top",
                color="white", fontsize=9, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([labels[m] for m in metas], rotation=10)
    ax.set_ylabel("Mean episode return")
    ax.set_title(
        "CartPoleRegimeV1 — second-POMDP external-validity probe\n"
        "Hypernet ≫ concat decoupling reproduces (Family A: 2/2 Holm-supported, LOO-robust)"
    )

    # Legend with reference dashes only — bars carry their own xtick labels.
    ax.legend(**LEGEND_OUTSIDE_RIGHT)

    budget_annotation(
        fig,
        iterations=200, parallel_envs=512, rollout_length=128, num_seeds=8,
        extra="| 7 methods × 1 cell | references at n=5",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[cartpole_plots] wrote {out_path}", flush=True)


def main() -> None:
    plot_method_ladder(FIGURES_ROOT / "cartpole_method_ladder.png")


if __name__ == "__main__":
    main()
