"""Plot the final fresh-episode evaluation of the three reference agents."""
from __future__ import annotations

import json

from plotting.m2_plots import plot_belief_ppo_gap
from utils.paths import analysis_dir, project_fig_dir, resolve_data


def main() -> int:
    source = resolve_data(analysis_dir() / "m5r_post_training_evaluation.json")
    with source.open() as handle:
        methods = json.load(handle)["methods"]

    keys = {
        "regime_agnostic": "regime_agnostic_ppo",
        "belief": "belief_ppo",
        "oracle": "oracle_ppo",
    }
    bars = {
        label: (
            methods[key]["evaluation_return_mean"],
            methods[key]["evaluation_return_ci95"],
        )
        for label, key in keys.items()
    }
    output = project_fig_dir("appendix") / "fig_reference_agents_final_evaluation.png"
    plot_belief_ppo_gap(bars, output, value_decimals=2)
    print(f"[final_reference_evaluation] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
