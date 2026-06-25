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

from plotting.style import PALETTE, apply_style, polish, ref_line

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL = REPO_ROOT / "results" / "M5R" / "final"
PROJECT_FIG = REPO_ROOT / "figures" / "milestones" / "M5R"
THESIS_FIG = REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"

FLOOR, BELIEF, ORACLE = 138.0, 168.8, 182.0

# Deployed pure-method returns (hardcoded references), per method.
DEPLOYED = {"rl2": (118.7, 163.1), "vb": (108.9, 163.5)}  # (concat, hypernet)


def _load(prefix, head):
    # RL² medium files are m5r_transplant_e_final_{head}; VB are m5r_transplant_vb_{head}
    p = FINAL / f"{prefix}_{head}.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    return d["final_return_mean"], d["final_return_std"]


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    bw = 0.16

    C_DEP_C, C_TR_C = "#d3d8dc", "#9aa5ad"          # concat: deployed / transplant
    C_DEP_H, C_TR_H = "#a8d8d0", PALETTE["hyper"]   # hypernet: deployed / transplant

    # Shaded zones: recoverable gap (floor -> belief) + inference remainder (belief -> oracle).
    ax.axhspan(FLOOR, BELIEF, color=PALETTE["hyper"], alpha=0.08, zorder=0)
    ax.axhspan(BELIEF, ORACLE, color=PALETTE["hyper"], alpha=0.04, zorder=0)

    ekw = {"ecolor": "#3a3a3a", "capsize": 3, "elinewidth": 1.1}

    def _group(cx, bars):
        """bars: list of (value, std_or_None, color), drawn centered at cx."""
        n = len(bars)
        offs = (np.arange(n) - (n - 1) / 2.0) * bw
        for off, (val, std, color) in zip(offs, bars):
            if val is None:
                continue
            ax.bar(cx + off, val, bw, yerr=std, color=color, edgecolor="white",
                   linewidth=1.2, zorder=3, error_kw=ekw if std else None)

    rl2_cm, rl2_cs = _load("m5r_transplant_e_final", "concat")
    rl2_hm, rl2_hs = _load("m5r_transplant_e_final", "hypernet")
    vb_cm, vb_cs = _load("m5r_transplant_vb", "concat")
    vb_hm, vb_hs = _load("m5r_transplant_vb", "hypernet")

    centers = [0.0, 1.0]
    # RL²: deployed concat, transplant concat, deployed hypernet, transplant hypernet.
    _group(centers[0], [
        (DEPLOYED["rl2"][0], None, C_DEP_C), (rl2_cm, rl2_cs, C_TR_C),
        (DEPLOYED["rl2"][1], None, C_DEP_H), (rl2_hm, rl2_hs, C_TR_H),
    ])
    _group(centers[1], [
        (DEPLOYED["vb"][0], None, C_DEP_C), (vb_cm, vb_cs, C_TR_C),
        (DEPLOYED["vb"][1], None, C_DEP_H), (vb_hm, vb_hs, C_TR_H),
    ])

    xr = centers[-1] + 2.2 * bw
    ref_line(ax, FLOOR, "Regime-agnostic floor", x=xr, linestyle="-")
    ref_line(ax, BELIEF, "Belief-PPO", x=xr, linestyle=(0, (1, 1.5)))
    ref_line(ax, ORACLE, "Oracle-PPO", x=xr, linestyle=(0, (6, 2)))

    ax.set_xticks(centers)
    ax.set_xticklabels(["RL²", "VariBAD"])
    ax.set_ylabel("Final-episode return")
    ax.set_xlim(-0.45, centers[-1] + 1.0)
    ax.set_ylim(88, 192)
    polish(ax)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_DEP_C, edgecolor="white", label="Deployed concat"),
        Patch(facecolor=C_TR_C, edgecolor="white", label="Transplant: frozen belief + concat head"),
        Patch(facecolor=C_DEP_H, edgecolor="white", label="Deployed hypernet"),
        Patch(facecolor=C_TR_H, edgecolor="white", label="Transplant: frozen belief + hypernet head"),
    ]
    ax.legend(handles=handles,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True,
              facecolor="white", edgecolor="#dddddd")
    fig.tight_layout()
    fig.subplots_adjust(right=0.60)
    for d in (PROJECT_FIG, THESIS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "m5r_transplant.png")
    plt.close(fig)
    print("[plot] wrote m5r_transplant.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
