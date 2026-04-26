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
}


FIGSIZE_SMALL = (4.0, 3.0)
FIGSIZE_STANDARD = (6.0, 4.0)
FIGSIZE_WIDE = (8.0, 3.5)


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
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
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
    """Italic compute-budget caption at the bottom of the figure.

    Lets the reader tell smoke runs from full-budget runs at a glance.
    Every thesis figure should include one — call this after `tight_layout`.
    """
    parts: list[str] = []
    if iterations is not None:
        parts.append(f"{iterations} iter")
    if parallel_envs is not None:
        parts.append(f"{parallel_envs} envs")
    if rollout_length is not None:
        parts.append(f"rollout {rollout_length}")
    if num_seeds is not None:
        parts.append(f"n={num_seeds} seeds")
    if extra:
        parts.append(extra)
    if not parts:
        return
    text = "Compute: " + " × ".join(parts)
    fig.text(0.5, y, text, ha="center", fontsize=7, style="italic", color="#555555")


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
