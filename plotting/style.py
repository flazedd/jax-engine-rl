"""Centralized plot style. `apply_style()` is called by every plotting script.

Also exposes the shared thesis-figure conventions enforced across all
milestone plots — see `feedback_plot_principles` in user memory for the
full list. The helpers below are the canonical implementation; every plot
script should use them rather than re-implementing.
"""
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
    # Per-cell colours for the M5 4-cell factorial. Each cell gets a
    # distinct hue so the four series are independently distinguishable
    # in every chart that includes them. The analytical posterior
    # reference line uses a fifth distinct hue (red).
    "rl2_concat":        "#1f77b4",  # blue
    "rl2_hypernet":      "#ff7f0e",  # orange
    "varibad_concat":    "#9467bd",  # purple
    "varibad_hypernet":  "#2ca02c",  # green
    "analytical":        "#d62728",  # red — for the analytical-posterior reference
}


FIGSIZE_SMALL = (4.0, 3.0)
FIGSIZE_STANDARD = (6.0, 4.0)
FIGSIZE_WIDE = (8.0, 3.5)


# Polished thesis palette (the "hero ladder" look). Winners get a vivid teal
# accent, baselines a muted slate, references neutral greys. Use these in place
# of the legacy per-cell rainbow in COLORS for any restyled figure.
PALETTE = {
    "hyper":      "#2a9d8f",  # vivid teal — hypernet / the winners
    "concat":     "#b4bcc2",  # muted slate — concat / falls short
    "stacked":    "#8a949c",  # neutral baseline
    "floor":      "#8a949c",  # regime-agnostic floor (neutral grey)
    "belief":     "#2a9d8f",  # belief ceiling (teal = realistic target)
    "oracle":     "#264653",  # oracle ceiling (deep slate-teal)
    "analytical": "#e09f3e",  # analytical-posterior reference (amber)
    "accent":     "#2a9d8f",
    "accent2":    "#e07a5f",  # secondary accent (warm) when two series needed
    "ref":        "#555555",  # generic reference line
}


def polish(ax, grid_axis: str = "y") -> None:
    """Apply the shared figure look to an axis: drop the top/right spines and
    tick marks, and use a soft dotted grid on one axis only."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.25, linestyle=":", zorder=0)


def ref_line(ax, y, label=None, *, x=None, color="#555555", linestyle="--") -> None:
    """A neutral horizontal reference line with an optional inline label given a
    white backing, so it reads cleanly without crowding the legend."""
    ax.axhline(y, color=color, linewidth=1.1, linestyle=linestyle, zorder=2)
    if x is not None and label:
        ax.text(x, y, f" {label}", va="center", ha="left", fontsize=8.5,
                color="#444444", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))


# Shared kwargs for an outside-the-axes legend with an opaque white
# background — every thesis figure uses these. `loc="upper left"` +
# bbox_to_anchor=(1.02, 1.0) places it to the right of the axes; flip to
# "upper center" + (0.5, -0.10) for a below-axes legend.
LEGEND_OUTSIDE_RIGHT = dict(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=8,
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=1.0,
)
LEGEND_BELOW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    fontsize=8,
    ncol=2,
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=1.0,
)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        # Shared "hero" look applied to every figure: soft dotted y-grid,
        # no top/right spines, no tick marks.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def budget_annotation(
    fig,
    iterations: int | None = None,
    parallel_envs: int | None = None,
    rollout_length: int | None = None,
    num_seeds: int | None = None,
    extra: str = "",
    y: float = 0.005,
) -> None:
    """Compute-budget annotation.

    Intentionally a no-op: the compute budget is now reported in the thesis
    figure captions rather than rendered onto the figure. Kept as a stub so the
    existing call sites remain valid.
    """
    return


def draw_reference_lines(
    ax,
    *,
    floor: float | None = None,
    belief: float | None = None,
    oracle: float | None = None,
    floor_label: str = "Regime-agnostic PPO floor",
    belief_label: str = "Belief-PPO ceiling",
    oracle_label: str = "Oracle-PPO ceiling",
) -> None:
    """Draw the canonical floor / belief / oracle dashed reference lines.

    Each label includes its numeric value so the legend is self-documenting.
    Pass `None` to skip a line.
    """
    style = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.6}
    if floor is not None:
        ax.axhline(floor, color=COLORS["ppo"], **style,
                   label=f"{floor_label} = {floor:.1f}")
    if belief is not None:
        ax.axhline(belief, color=COLORS["belief_ppo"], **style,
                   label=f"{belief_label} = {belief:.1f}")
    if oracle is not None:
        ax.axhline(oracle, color=COLORS["oracle_ppo"], **style,
                   label=f"{oracle_label} = {oracle:.1f}")
