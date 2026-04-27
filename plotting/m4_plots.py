"""M4 plot module — implementation-validation figure on the toy envs.

Produces:
  - factorial_toys.png — RL²/VariBAD × Concat/Hypernetwork across the
    three M4 validation envs (bandit, gridworld, regime_bandit), with a
    per-env PPO floor reference line. This single chart shows both the
    M4 pass criterion (meta-RL clears floor) and the M5 integration
    ablation (concat vs hypernet) on the toy envs.

The factorial cells were collected during M5 Step-3 toy sweep, so the
underlying stats JSON lives at `results/milestones/M5/stats_M5_factorial_toys.json`;
the PPO floor numbers come from `results/milestones/M4/method_ranking.json`.
The chart's *role* is M4 (implementation validation), so the PNG output
is written under `figures/milestones/M4/`.

CLI:
    uv run python -m plotting.m4_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import COLORS, apply_style

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures" / "milestones" / "M4"


def _budget_annotation(
    fig,
    iterations: int | None = None,
    parallel_envs: int | None = None,
    rollout_length: int | None = None,
    num_seeds: int | None = None,
    extra: str = "",
    y: float = 0.005,
) -> None:
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


_TOY_ENVS = ["bandit", "gridworld", "regime_bandit"]
_TOY_ENV_LABELS = {
    "bandit": "Bandit (5-arm Bernoulli)",
    "gridworld": "Gridworld (random goal)",
    "regime_bandit": "Regime-switching bandit",
}
_VARIANT_ORDER = ("concat_nobonus", "hypernet_nobonus")
_VARIANT_LABELS = {
    "concat_nobonus": "Concat",
    "hypernet_nobonus": "Hypernetwork",
}


def _load_m4_floors() -> dict[str, float]:
    path = RESULTS_ROOT / "milestones" / "M4" / "method_ranking.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    out: dict[str, float] = {}
    for row in d.get("key_stats", {}).get("table", []):
        if "ppo_floor" in row:
            out[row["env"]] = float(row["ppo_floor"])
    return out


def plot_factorial_toys(out_path: Path) -> bool:
    stats_path = RESULTS_ROOT / "milestones" / "M5" / "stats_M5_factorial_toys.json"
    if not stats_path.exists():
        print(f"[m4_plots] skip factorial_toys: missing {stats_path}")
        return False
    with open(stats_path) as f:
        stats = json.load(f)

    floors = _load_m4_floors()

    # Re-index configs by (method, env, variant_label). Tolerate older
    # JSONs that carry an `exploration_bonus` field; bonus configs are
    # ignored by the plot regardless (only `*_nobonus` variants are read).
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in stats["configs"]:
        bonus = r.get("exploration_bonus", False)
        variant = f"{r['integration']}_{'bonus' if bonus else 'nobonus'}"
        by_key[(r["method"], r["env"], variant)] = r

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)

    methods = ["rl2", "varibad"]
    method_labels = {"rl2": "RL²", "varibad": "VariBAD"}
    n_v = len(_VARIANT_ORDER)
    bar_w = 0.38

    def _cell_color(method: str, variant: str) -> str:
        integ = "hypernet" if "hypernet" in variant else "concat"
        return COLORS[f"{method}_{integ}"]

    for ax, env in zip(axes, _TOY_ENVS):
        for i_method, method in enumerate(methods):
            means = []
            errs_lo = []
            errs_hi = []
            for v in _VARIANT_ORDER:
                key = (method, env, v)
                if key in by_key:
                    r = by_key[key]
                    means.append(r["final_return_mean"])
                    lo, hi = r["final_return_ci95"]
                    errs_lo.append(r["final_return_mean"] - lo)
                    errs_hi.append(hi - r["final_return_mean"])
                else:
                    means.append(np.nan)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
            offset = (i_method - 0.5) * bar_w
            x = np.arange(n_v) + offset
            yerr = np.array([errs_lo, errs_hi])
            colors = [_cell_color(method, v) for v in _VARIANT_ORDER]
            ax.bar(
                x, means, bar_w, yerr=yerr, color=colors,
                edgecolor="black", linewidth=0.4, capsize=2,
            )
        # PPO floor for this env (loaded from M4 ranking JSON). Same
        # floor/belief/oracle convention as the M5 ladder — dashed,
        # PPO blue, alpha 0.6. The floor value matches the
        # `concat_nobonus` bar by construction (M4's PPO baseline and
        # the factorial concat cell are the same training run); the
        # line makes the "did meta-RL clear the floor?" check visually
        # explicit and lets this single chart serve M4's purpose.
        floor_line = None
        floor_val = floors.get(env)
        if floor_val is not None:
            floor_line = ax.axhline(
                floor_val, color=COLORS["ppo"],
                linestyle="--", linewidth=1.0, alpha=0.6,
            )
        ax.set_xticks(np.arange(n_v))
        ax.set_xticklabels([_VARIANT_LABELS[v] for v in _VARIANT_ORDER], fontsize=8)
        ax.set_title(_TOY_ENV_LABELS[env])
        from matplotlib.lines import Line2D as _Line2D
        from matplotlib.patches import Patch as _Patch
        legend_handles: list[Any] = [
            _Patch(facecolor=_cell_color(m, v), edgecolor="black",
                   linewidth=0.4,
                   label=f"{method_labels[m]} {_VARIANT_LABELS[v]}")
            for m in methods for v in _VARIANT_ORDER
        ]
        if floor_line is not None:
            legend_handles.append(_Line2D(
                [0], [0], color=COLORS["ppo"], linestyle="--",
                linewidth=1.0, alpha=0.6,
                label=f"PPO floor = {floor_val:.1f}",
            ))
        ax.legend(
            handles=legend_handles,
            loc="upper center", bbox_to_anchor=(0.5, -0.10),
            fontsize=7, ncol=2,
            frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1.0,
        )

    fig.suptitle(
        "Toy environments — meta-RL implementation validation (RL²/VariBAD × Concat/Hypernetwork)\n"
        "Final return (mean across 3 seeds)",
        y=1.02,
        fontsize=11,
    )
    sample_cfg = stats["configs"][0] if stats["configs"] else {}
    _budget_annotation(
        fig,
        iterations=sample_cfg.get("iterations"),
        num_seeds=sample_cfg.get("num_seeds"),
        extra="across 12 method×env×integration cells",
    )
    fig.tight_layout()
    # Bump bottom margin to fit 5 legend entries (4 cells + floor) at
    # ncol=2 — three rows instead of two.
    fig.subplots_adjust(bottom=0.36)
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
    print(f"[m4_plots] wrote {written} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
