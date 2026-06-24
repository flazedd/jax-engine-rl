"""Optimisation-ease search figure: concat vs hypernet under a symmetric search.

Strip plot of the per-config mean returns for concat vs hypernet over the SAME
random hyperparameter configs, against the regime-agnostic floor and the
Belief-PPO / Oracle-PPO ceilings. One panel per method that has data (RL2,
VariBAD). Produces the PPO figure (m5r_optsearch.png) and, when A2C data exists,
the A2C figure (m5r_optsearch_a2c.png) the same way.

Usage: uv run python -m plotting.optsearch_plot
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plotting.style import PALETTE, apply_style, polish, ref_line

REPO = Path(__file__).resolve().parent.parent
FINAL = REPO / "results" / "M5R" / "final"
PROJECT_FIG = REPO / "figures" / "milestones" / "M5R"
THESIS_FIG = REPO.parent / "master_thesis_reinier_schep_final" / "figures"

ORACLE = 182.0  # medium-env oracle ceiling, for context
METHODS = [("rl2", "RL²"), ("varibad", "VariBAD")]


def _panel(ax, data, title, ylim, show_ylabel):
    floor, belief = float(data["floor"]), float(data["belief"])
    ymin, ymax = ylim
    ax.axhspan(floor, belief, color=PALETTE["hyper"], alpha=0.08, zorder=0)
    rng = np.random.default_rng(0)
    for i, (key, color) in enumerate([("concat", PALETTE["concat"]),
                                      ("hypernet", PALETTE["hyper"])]):
        means = np.array([c["mean_return"]
                          for c in data["by_arch"][key]["per_config"]])
        x = i + rng.uniform(-0.13, 0.13, means.size)
        ax.scatter(x, means, s=70, color=color, edgecolor="white", linewidth=1.1,
                   alpha=0.9, zorder=3)
        m = float(means.mean())
        ax.plot([i - 0.23, i + 0.23], [m, m], color=color, lw=3.0, zorder=4)
        frac = float((means > floor).mean()) * 100
        ax.text(i, ymin + 4, f"{frac:.0f}% clear\nthe floor", ha="center",
                va="bottom", fontsize=9, color=color, fontweight="bold")
    # Inline reference-line labels in a right-hand margin, as in the ladder,
    # transplant, and sweep figures (no separate legend).
    xr = 1.58
    ref_line(ax, floor, "Regime-agnostic floor", x=xr, linestyle="-")
    ref_line(ax, belief, "Belief-PPO", x=xr, linestyle=(0, (1, 1.5)))
    ref_line(ax, ORACLE, "Oracle-PPO", x=xr, linestyle=(0, (6, 2)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Concatenation", "Hypernetwork"], fontsize=12)
    if show_ylabel:
        ax.set_ylabel("Final-episode return (mean over seeds)")
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=12)
    polish(ax)


def _build(suffix, out_name):
    panels = []
    for method, label in METHODS:
        p = FINAL / f"optsearch_{method}{suffix}.json"
        if p.exists():
            d = json.load(open(p))
            # require both arches present and non-empty
            if all(d["by_arch"].get(a, {}).get("per_config") for a in ("concat", "hypernet")):
                panels.append((d, label))
    if not panels:
        print(f"[optsearch_plot] no data for suffix '{suffix}', skip {out_name}")
        return
    all_means = [c["mean_return"]
                 for d, _ in panels for a in ("concat", "hypernet")
                 for c in d["by_arch"][a]["per_config"]]
    floor = min(float(d["floor"]) for d, _ in panels)
    ymin = min([floor, *all_means]) - 14
    ymax = ORACLE + 8
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0 * len(panels), 5.8),
                             squeeze=False)
    for j, (ax, (data, label)) in enumerate(zip(axes[0], panels)):
        _panel(ax, data, label, (ymin, ymax), show_ylabel=(j == 0))
    fig.tight_layout()
    for dd in (PROJECT_FIG, THESIS_FIG):
        dd.mkdir(parents=True, exist_ok=True)
        fig.savefig(dd / out_name)
    plt.close(fig)
    print(f"[optsearch_plot] wrote {out_name} ({len(panels)} panel(s))")


def main() -> int:
    _build("", "m5r_optsearch.png")        # PPO (RL2 [+ VariBAD if present])
    _build("_a2c", "m5r_optsearch_a2c.png")  # A2C (when those runs land)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
