"""M4 plot module — implementation-validation figure on the toy envs.

Produces:
  - factorial_toys.png — RL²/VariBAD × concat/hypernetwork across the three
    validation environments. The solid Regime-agnostic-PPO line and its light
    95% interval band use the original M4 baseline runs.

CLI:
    uv run python -m plotting.m4_plots
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from plotting.style import COLORS, PALETTE, apply_style, polish

# Cell colours: concat in the slate family, hypernet in the teal family, with
# RL² vs VariBAD distinguished by shade (matches the hero-ladder aesthetic).
_CELL_COLORS = {
    "rl2_concat":       "#b4bcc2",  # light slate
    "varibad_concat":   "#8a949c",  # dark slate
    "rl2_hypernet":     "#2a9d8f",  # vivid teal
    "varibad_hypernet": "#73b8ad",  # light teal
}

from utils.paths import fig_targets, fig_appendix_dir, foundations_dir, project_fig_dir, resolve_data, results_root

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = results_root()
FIGURES_ROOT = fig_appendix_dir()


def _budget_annotation(
    fig,
    iterations: int | None = None,
    parallel_envs: int | None = None,
    rollout_length: int | None = None,
    num_seeds: int | None = None,
    extra: str = "",
    y: float = 0.005,
) -> None:
    # No-op: compute budget is now reported in the thesis figure captions
    # rather than rendered onto the figure. Kept as a stub for call sites.
    return


_TOY_ENVS = ["bandit", "gridworld", "regime_bandit"]
_TOY_ENV_LABELS = {
    "bandit": "Five armed Bernoulli bandit",
    "gridworld": "Random goal gridworld",
    "regime_bandit": "Regime switching bandit",
}
def _load_m4_validation() -> dict[str, dict[str, object]]:
    candidates = (
        foundations_dir() / "method_ranking.json",
        RESULTS_ROOT / "_archive" / "milestones" / "M4" / "method_ranking.json",
    )
    for path in candidates:
        resolved = resolve_data(path)
        if resolved.exists():
            with open(resolved) as f:
                data = json.load(f)
            return {row["env"]: row for row in data["key_stats"]["table"]}
    return {}


def plot_factorial_toys(out_path: Path) -> bool:
    validation = _load_m4_validation()
    if not validation:
        print("[m4_plots] skip factorial_toys: missing M4 validation results")
        return False
    stats_path = foundations_dir() / "stats_M5_factorial_toys.json"
    if not resolve_data(stats_path).exists():
        print(f"[m4_plots] skip factorial_toys: missing {stats_path}")
        return False
    with open(resolve_data(stats_path)) as f:
        stats = json.load(f)

    variants = ("concat_nobonus", "hypernet_nobonus")
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in stats["configs"]:
        variant = f"{row['integration']}_{'bonus' if row.get('exploration_bonus', False) else 'nobonus'}"
        by_key[(row["method"], row["env"], variant)] = row

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 12.0), sharey=False)
    methods = ("rl2", "varibad")
    method_labels = {"rl2": "RL²", "varibad": "VariBAD"}
    bar_width = 0.34
    integration_colors = {
        "concat_nobonus": "#b4bcc2",
        "hypernet_nobonus": "#2a9d8f",
    }

    from matplotlib.patches import Patch
    legend_handles: list[Any] = [
        Patch(facecolor=integration_colors["hypernet_nobonus"], edgecolor="white",
              label="Hypernetwork"),
        Patch(facecolor=integration_colors["concat_nobonus"], edgecolor="white",
              label="Concatenation"),
    ]

    for ax, env in zip(axes, _TOY_ENVS):
        for method_index, method in enumerate(methods):
            means, lows, highs = [], [], []
            for variant in variants:
                item = by_key[(method, env, variant)]
                mean = float(item["final_return_mean"])
                low, high = item["final_return_ci95"]
                means.append(mean)
                lows.append(mean - low)
                highs.append(high - mean)
            x = method_index + np.array([-bar_width * 0.62, bar_width * 0.62])
            ax.bar(x, means, bar_width, yerr=np.array([lows, highs]),
                   color=[integration_colors[variant] for variant in variants],
                   edgecolor="white", linewidth=1.2, capsize=4, zorder=3,
                   error_kw={"ecolor": "#3a3a3a", "elinewidth": 1.1})

        ppo_mean = float(validation[env]["ppo_floor"])
        ppo_low, ppo_high = validation[env]["ppo_ci"]
        ax.axhspan(ppo_low, ppo_high, color="#555555", alpha=0.08, zorder=0)
        ax.axhline(ppo_mean, color="#555555", linewidth=1.2, zorder=2)
        ax.text(2.02, ppo_mean, " Regime-agnostic-PPO", va="center", ha="left", fontsize=8.5,
                color="#444444", zorder=4, clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([method_labels[method] for method in methods], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel("Final return", fontsize=11)
        ax.set_title(_TOY_ENV_LABELS[env], fontsize=13, loc="left")
        polish(ax)

    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.005), frameon=True, facecolor="white",
               edgecolor="#dddddd", framealpha=0.95)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[m4_plots] wrote {out_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(prog="plotting.m4_plots")
    parser.add_argument(
        "--out-dir", default=str(FIGURES_ROOT),
        help="output directory for PNGs (default: figures/milestones/M4/)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    if plot_factorial_toys(out_dir / "factorial_toys.png"):
        written += 1
        if out_dir == FIGURES_ROOT:
            source = out_dir / "factorial_toys.png"
            for target in fig_targets("factorial_toys.png"):
                if target.resolve() != source.resolve():
                    shutil.copy2(source, target)
    print(f"[m4_plots] wrote {written} figures to {out_dir}")
    return 0 if written == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
