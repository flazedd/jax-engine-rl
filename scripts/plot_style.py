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

from lob_sim.actions import ACTION_TABLE, N_ACTIONS

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

import numpy as np
_at = np.array(ACTION_TABLE)
ACTION_LABELS = [f"({int(_at[a][0])},{int(_at[a][1])})" for a in range(N_ACTIONS)]


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


def action_bar(ax, values, title, *, kind="reward", se_values=None):
    """Draw a horizontal bar chart for per-action values.

    Parameters
    ----------
    values : (N_ACTIONS,) array
        Values to display — either mean rewards or frequency fractions.
    kind : "reward" | "frequency"
        Controls colormap and annotation format.
    se_values : optional (N_ACTIONS,) array
        If provided (reward kind), show ±SE as error bars.
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(REWARD_CMAP if kind == "reward" else FREQ_CMAP)
    values = np.asarray(values)
    vmin, vmax = values.min(), values.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = [cmap(norm(v)) for v in values]

    y = np.arange(N_ACTIONS)
    xerr = np.asarray(se_values) if se_values is not None else None
    bars = ax.barh(y, values, xerr=xerr, color=colors, edgecolor="grey",
                   linewidth=0.5, capsize=3)

    ax.set_yticks(y)
    ax.set_yticklabels(ACTION_LABELS)
    ax.set_ylabel("Action (bid, ask)")
    ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # Annotate bars
    for i, v in enumerate(values):
        if kind == "frequency":
            txt = f"{v * 100:.1f}%"
        else:
            if se_values is not None:
                txt = f"{v:.1f}±{se_values[i]:.1f}"
            else:
                txt = f"{v:.1f}"
        ax.text(v, i, f" {txt}", ha="left", va="center", fontsize=8)

    return bars


def mark_optimal_bar(ax, action_idx):
    """Highlight the optimal action bar with a blue rectangle."""
    idx = int(action_idx)
    rect = Rectangle(
        (ax.get_xlim()[0], idx - 0.5),
        ax.get_xlim()[1] - ax.get_xlim()[0], 1,
        linewidth=OPTIMAL_LW, edgecolor=OPTIMAL_EDGE,
        facecolor="none", zorder=5,
    )
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
