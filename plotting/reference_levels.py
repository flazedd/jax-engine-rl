"""The two reference-level figures the results chapter opens with.

  - fig_rq1_ceilings_bar.png     mean final-episode return per reference, with
                                 95% bootstrap CIs across seeds
  - fig_rq1_learning_curves.png  the same three references over training

These used to come from `plotting.regenerate_figures M3`, which reads a
`stats_M3_reference_levels.json` that no longer exists anywhere in the results
tree — so the two figures the thesis includes had no working producer. They are
rebuilt here from the reference runs of the matched programme, which is the
data the thesis actually reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import COLORS, apply_style, polish
from utils.paths import experiment_dir, fig_targets, project_fig_dir, resolve_data, results_root

# Experiment directory -> label, in the order the ordering claim is made.
REFERENCES = [
    ("m5r_ref_regime_agnostic_e9", "Regime-agnostic PPO"),
    ("m5r_ref_belief_e9", "Belief-PPO"),
    ("m5r_ref_oracle_e9", "Oracle-PPO"),
]
N_BOOT = 10_000


def _load(experiment: str) -> dict | None:
    # Resolve the source switch *before* the existence check: in a dummy run
    # only the .dummy.json sibling exists, so checking the real path first
    # would report the stage as missing.
    p = resolve_data(experiment_dir(experiment) / "metrics.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _boot_ci(vals: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(N_BOOT, vals.size))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _boot_mean_ci_per_iteration(
    per_seed_curve: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% bootstrap CIs for a mean learning curve.

    Resampling whole seed-curves preserves the within-run temporal dependence.
    The resulting bands quantify uncertainty in the mean at each iteration;
    they are not seed-to-seed percentile ranges and are not simultaneous bands
    over the full curve.
    """
    rng = np.random.default_rng(seed)
    n_seeds = per_seed_curve.shape[0]
    indices = rng.integers(0, n_seeds, size=(N_BOOT, n_seeds))
    boot_means = per_seed_curve[indices].mean(axis=1)
    return (
        np.percentile(boot_means, 2.5, axis=0),
        np.percentile(boot_means, 97.5, axis=0),
    )


def _targets(name: str) -> list[Path]:
    return fig_targets(name)


def plot_ceilings_bar(blocks: list[tuple[str, dict]]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    palette = [COLORS.get("floor", "#b4bcc2"), COLORS.get("belief", "#2a9d8f"),
               COLORS.get("oracle", "#264653")]
    for i, ((label, m), colour) in enumerate(zip(blocks, palette)):
        vals = np.asarray(m["per_seed_final_return"], dtype=float)
        mean = float(vals.mean())
        lo, hi = _boot_ci(vals, seed=i)
        ax.bar(i, mean, width=0.6, color=colour, edgecolor="white", linewidth=1.2,
               yerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]], capsize=4,
               error_kw={"ecolor": "#3a3a3a", "elinewidth": 1.2, "capthick": 1.2},
               zorder=3)
        ax.annotate(f"{mean:.1f}", xy=(i, hi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([lbl for lbl, _ in blocks])
    ax.set_ylabel("Return at the end of training")
    n_seeds = len(blocks[0][1]["per_seed_final_return"])
    polish(ax)
    fig.tight_layout()
    # Repo tree only. The thesis dropped this chart: the learning curves carry
    # the same three levels at their right edge and Table 5.1 carries the paired
    # separations, so including it would print one claim three times. It is kept
    # here because it remains the fastest read of the level ordering.
    t = project_fig_dir("appendix") / "fig_rq1_ceilings_bar.png"
    t.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(t, dpi=200)
    print(f"[reference_levels] wrote {t}")
    plt.close(fig)


def plot_learning_curves(blocks: list[tuple[str, dict]]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    palette = [COLORS.get("floor", "#b4bcc2"), COLORS.get("belief", "#2a9d8f"),
               COLORS.get("oracle", "#264653")]
    ends: list[tuple[float, float, str, str]] = []
    last_iter = 0
    for seed, ((label, m), colour) in enumerate(zip(blocks, palette)):
        curve = np.asarray(m["mean_return_per_iter"], dtype=float)
        ax.plot(np.arange(curve.size), curve, color=colour, linewidth=1.8)
        per_seed = m.get("per_seed_mean_return_per_iter")
        if per_seed:
            arr = np.asarray(per_seed, dtype=float)
            lo, hi = _boot_mean_ci_per_iteration(arr, seed=seed)
            ax.fill_between(np.arange(curve.size), lo, hi, color=colour, alpha=0.15,
                            linewidth=0)
        last_iter = max(last_iter, int(curve.size) - 1)
        ends.append((float(curve.size - 1), float(curve[-1]), label, colour))

    # Labels at the right end of each curve rather than a legend box, matching
    # the variant learning curves: with three curves that never cross, naming
    # each where it ends is read without a colour lookup.
    span = max((e[1] for e in ends), default=1.0) - min((e[1] for e in ends), default=0.0)
    min_gap = 0.06 * (span or 1.0)
    placed: list[float] = []
    for x_end, y_end, label, colour in sorted(ends, key=lambda e: -e[1]):
        y_lab = y_end
        if placed and abs(y_lab - placed[-1]) < min_gap:
            y_lab = placed[-1] - min_gap
        placed.append(y_lab)
        ax.annotate(label, xy=(x_end, y_end),
                    xytext=(x_end + 0.035 * last_iter, y_lab),
                    textcoords="data", va="center", ha="left", fontsize=9,
                    color=colour, fontweight="bold", clip_on=False, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.6,
                                    alpha=0.5, shrinkA=0, shrinkB=0))

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean return")
    ax.set_xlim(0, last_iter * 1.30)
    polish(ax)
    fig.tight_layout()
    for t in _targets("fig_rq1_learning_curves.png"):
        t.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(t, dpi=600)
        print(f"[reference_levels] wrote {t}")
    plt.close(fig)


def main() -> int:
    blocks: list[tuple[str, dict]] = []
    for experiment, label in REFERENCES:
        m = _load(experiment)
        if m is None:
            print(f"[reference_levels] FAIL | missing {experiment}/metrics.json")
            return 1
        blocks.append((label, m))
    plot_ceilings_bar(blocks)
    plot_learning_curves(blocks)
    print("[reference_levels] OK | figures=2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
