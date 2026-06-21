"""Transplant figure: frozen-belief return by head form, per method.

For RL² and VariBAD, plots the deployed pure methods against the frozen-encoder
transplant (same belief, fresh concat vs hypernet head trained by PPO), with the
regime-agnostic floor and the Belief-PPO ceiling as reference lines. Reads the
transplant JSONs written by the transplant scripts.

Output: figures/milestones/M5R/m5r_transplant.png (+ thesis copy)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plotting.style import apply_style

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL = REPO_ROOT / "results" / "M5R" / "final"
PROJECT_FIG = REPO_ROOT / "figures" / "milestones" / "M5R"
THESIS_FIG = REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"

FLOOR, BELIEF = 137.8, 169.9

# (method label, transplant json prefix, deployed concat, deployed hypernet)
METHODS = [
    ("RL²", "m5r_transplant_e_final", 108.6, 158.9),
    ("VariBAD", "m5r_transplant_vb", 103.8, 160.2),
]


def _load(prefix, head):
    # RL² medium files are m5r_transplant_e_final_{head}; VB are m5r_transplant_vb_{head}
    p = FINAL / f"{prefix}_{head}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["final_return_mean"], d["final_return_std"]


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    group_w = 0.8
    bw = group_w / 4
    xs = np.arange(len(METHODS))

    C_DEP_C, C_TR_C = "#9ecae1", "#1f77b4"   # concat: light=deployed, dark=transplant
    C_TR_H, C_DEP_H = "#ff7f0e", "#fdd0a2"   # hypernet: dark=transplant, light=deployed

    for i, (lab, prefix, dep_c, dep_h) in enumerate(METHODS):
        tc = _load(prefix, "concat")
        th = _load(prefix, "hypernet")
        x0 = xs[i] - 1.5 * bw
        # Order: deployed then transplant, for concat then hypernet.
        ax.bar(x0 + 0 * bw, dep_c, bw, color=C_DEP_C, edgecolor="black", linewidth=0.4)
        if tc:
            ax.bar(x0 + 1 * bw, tc[0], bw, yerr=tc[1], color=C_TR_C, edgecolor="black",
                   linewidth=0.4, error_kw={"ecolor": "black", "capsize": 3})
        ax.bar(x0 + 2 * bw, dep_h, bw, color=C_DEP_H, edgecolor="black", linewidth=0.4)
        if th:
            ax.bar(x0 + 3 * bw, th[0], bw, yerr=th[1], color=C_TR_H, edgecolor="black",
                   linewidth=0.4, error_kw={"ecolor": "black", "capsize": 3})

    ax.axhline(FLOOR, color="#555555", linestyle="--", linewidth=1.2,
               label=f"Regime-agnostic floor = {FLOOR:.0f}")
    ax.axhline(BELIEF, color="#9467bd", linestyle="--", linewidth=1.2,
               label=f"Belief-PPO ceiling = {BELIEF:.0f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in METHODS])
    ax.set_ylabel("Final-episode return")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_DEP_C, edgecolor="black", label="Deployed concat"),
        Patch(facecolor=C_TR_C, edgecolor="black", label="Transplant: frozen belief + concat head"),
        Patch(facecolor=C_DEP_H, edgecolor="black", label="Deployed hypernet"),
        Patch(facecolor=C_TR_H, edgecolor="black", label="Transplant: frozen belief + hypernet head"),
    ]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True,
              facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)
    for d in (PROJECT_FIG, THESIS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "m5r_transplant.png")
    plt.close(fig)
    print("[plot] wrote m5r_transplant.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
