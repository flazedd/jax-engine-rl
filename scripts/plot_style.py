"""Shared plotting style for paper-ready figures.

All plotting scripts should import from here to ensure visual coherence.

Usage:
    from plot_style import *
    apply_style()
    fig, ax = plt.subplots(...)
    action_heatmap(ax, matrix, "Title", kind="frequency")
    mark_optimal(ax, flat_idx)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np

from lob_sim.actions import BID_TICKS, ASK_TICKS, N_ACTIONS

# ── Palette ──────────────────────────────────────────────────────
NOISE_COLOR = "#5C6BC0"   # indigo
BULL_COLOR  = "#43A047"   # green
BEAR_COLOR  = "#E53935"   # red
MIXED_COLOR = "#263238"   # dark slate

REGIME_NAMES  = ["Noise", "Bull", "Bear"]
REGIME_COLORS = [NOISE_COLOR, BULL_COLOR, BEAR_COLOR]

OPTIMAL_EDGE  = "#1565C0"  # blue — used for optimal-action markers everywhere
OPTIMAL_LW    = 3.0

# ── Heatmap settings ────────────────────────────────────────────
REWARD_CMAP = "RdYlGn"     # diverging — green=good, red=bad
FREQ_CMAP   = "YlOrRd"     # sequential — for action-frequency plots

N_BID = len(BID_TICKS)
N_ASK = len(ASK_TICKS)


def apply_style():
    """Set global rcParams for paper-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Lines
        "lines.linewidth": 1.8,
        "lines.antialiased": True,
        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid (when explicitly enabled)
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def action_heatmap(ax, matrix, title, *, kind="reward",
                   vmin=None, vmax=None, se_matrix=None):
    """Draw a 3×3 action heatmap on *ax*.

    Parameters
    ----------
    matrix : (N_BID, N_ASK) array
        Values to display.  Either mean rewards or frequency fractions.
    kind : "reward" | "frequency"
        Controls colormap and annotation format.
    se_matrix : optional (N_BID, N_ASK) array
        If provided (reward kind), show ±SE in each cell.
    """
    cmap = REWARD_CMAP if kind == "reward" else FREQ_CMAP
    origin = "lower" if kind == "reward" else "upper"

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin=origin, aspect="equal",
                   interpolation="nearest")

    ax.set_xticks(range(N_ASK))
    ax.set_xticklabels(ASK_TICKS)
    ax.set_yticks(range(N_BID))
    if origin == "lower":
        ax.set_yticklabels(BID_TICKS)
    else:
        ax.set_yticklabels(BID_TICKS)
    ax.set_xlabel("Ask offset (ticks)")
    ax.set_ylabel("Bid offset (ticks)")
    ax.set_title(title)

    # Re-enable all four spines for the heatmap frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if kind == "frequency":
                txt = f"{val * 100:.1f}%"
                thresh = (vmax if vmax else matrix.max()) * 0.55
                color = "white" if val > thresh else "black"
            else:
                if se_matrix is not None:
                    txt = f"{val:.0f}±{se_matrix[i, j]:.0f}"
                else:
                    txt = f"{val:.1f}"
                mid = ((vmax or matrix.max()) + (vmin or matrix.min())) / 2
                color = "white" if abs(val - mid) > 0.6 * abs((vmax or matrix.max()) - mid) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
    return im


def mark_optimal(ax, flat_idx, *, origin="lower"):
    """Highlight the optimal cell with a blue rectangle.

    Parameters
    ----------
    flat_idx : int
        Flat index into the (N_BID, N_ASK) matrix.
    origin : "lower" | "upper"
        Must match the origin used in action_heatmap.
    """
    row, col = divmod(int(flat_idx), N_ASK)
    rect = Rectangle((col - 0.5, row - 0.5), 1, 1,
                      linewidth=OPTIMAL_LW, edgecolor=OPTIMAL_EDGE,
                      facecolor="none", zorder=5)
    ax.add_patch(rect)


def regime_color(idx):
    """Return colour for regime index 0/1/2."""
    return REGIME_COLORS[idx]


def save_fig(fig, filename, *, script_file=None, subdir="plots"):
    """Save figure using an absolute path anchored to the project root.

    If *script_file* is given (pass ``__file__``), the project root is
    derived as its parent's parent.  Otherwise falls back to cwd.
    """
    import os
    if script_file:
        base = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(script_file)), ".."))
    else:
        base = os.getcwd()
    out_dir = os.path.join(base, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    print(f">>> Saved {path}")
    return path
