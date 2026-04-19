"""Minimal learning-curve plotter.

Single-seed runs (fast modes) render a solid line with no CI band; multi-seed
runs render the mean with a shaded CI95 band derived from per-iteration
per-seed values (simple percentile band for M0; full bootstrap lives in
evaluation/comparisons.py in later milestones).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.load_results import load_metrics
from plotting.style import COLORS, FIGSIZE_STANDARD, apply_style
from utils.script_output import ScriptRun


def plot_learning_curve(experiment_dir: Path, output_path: Path, *, title: str | None = None) -> dict:
    apply_style()
    metrics = load_metrics(experiment_dir)
    iters = np.arange(metrics["iterations"])
    mean_curve = np.asarray(metrics["mean_return_per_iter"])

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    color = COLORS.get("dummy", "#1f77b4")
    ax.plot(iters, mean_curve, color=color, label="mean return")

    if metrics["num_seeds"] > 1:
        var_curve = np.asarray(metrics["var_return_per_iter"])
        std = np.sqrt(np.maximum(var_curve, 0.0))
        ax.fill_between(
            iters,
            mean_curve - 1.96 * std / np.sqrt(metrics["num_seeds"]),
            mean_curve + 1.96 * std / np.sqrt(metrics["num_seeds"]),
            color=color,
            alpha=0.2,
            label="±1.96 SE",
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("return")
    ax.set_title(title or metrics["experiment_name"])
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "iterations": metrics["iterations"],
        "num_seeds": metrics["num_seeds"],
        "final_return": metrics["final_return_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="plotting.learning_curves")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()

    run = ScriptRun(script="plot_learning_curves")
    experiment_dir = Path(args.experiment_dir)
    output = Path(args.output)
    run.add_output(str(output))

    stats = plot_learning_curve(experiment_dir, output, title=args.title)

    summary_path = Path(args.summary_path) if args.summary_path else output.with_suffix(".summary.json")
    run.ok(key_stats=stats, summary_path=summary_path)


if __name__ == "__main__":
    main()
