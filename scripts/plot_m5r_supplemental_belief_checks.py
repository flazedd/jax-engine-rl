"""Plot direct analytical-posterior accuracy from exploratory own-trajectory checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.paths import analysis_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=analysis_dir() / "m5r_supplemental_belief_checks.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if data["n_rollouts_per_seed"] != 500:
        raise ValueError("final thesis figure requires the 500-rollout diagnostic")
    names = {
        "rl2_concat": ("RL² concatenation", "#55616b", "-"),
        "rl2_hypernet": ("RL² hypernetwork", "#177a70", "-"),
        "varibad_concat": ("VariBAD concatenation", "#a9b4bc", "--"),
        "varibad_hypernet": ("VariBAD hypernetwork", "#6cc7ba", "--"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for method, (label, color, ls) in names.items():
        rows = [row for row in data["rows"] if row["method"] == method]
        if len(rows) != 20:
            raise ValueError(f"expected 20 runs for {method}")
        curves = np.array([row["direct_posterior_accuracy_by_step"] for row in rows])
        mean = curves.mean(axis=0)
        ix = np.random.default_rng(0).integers(20, size=(10000, 20))
        lo, hi = np.percentile(curves[ix].mean(axis=1), [2.5, 97.5], axis=0)
        ax.plot(np.arange(128), mean, color=color, linestyle=ls, label=label, lw=2)
        ax.fill_between(np.arange(128), lo, hi, color=color, alpha=0.11)
    ax.set(xlabel="Step within episode", ylabel="Direct posterior accuracy", xlim=(0, 127), ylim=(0.3, 1.0))
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9, loc="lower center")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
