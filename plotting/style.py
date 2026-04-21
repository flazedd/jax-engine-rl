"""Centralized plot style. `apply_style()` is called by every plotting script."""
from __future__ import annotations

import matplotlib as mpl


# Method colors (stable across thesis figures)
COLORS: dict[str, str] = {
    "dummy": "#888888",
    "ppo": "#1f77b4",
    "stacked_ppo": "#2ca02c",
    "rl2": "#ff7f0e",
    "varibad": "#d62728",
    "belief_ppo": "#9467bd",
    "oracle_ppo": "#8c564b",
}


FIGSIZE_SMALL = (4.0, 3.0)
FIGSIZE_STANDARD = (6.0, 4.0)
FIGSIZE_WIDE = (8.0, 3.5)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })
