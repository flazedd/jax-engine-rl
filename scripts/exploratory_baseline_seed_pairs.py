"""Exploratory paired intervals for reference and short-history comparisons.

These comparisons were not part of the 18 planned tests. The output is descriptive
and does not alter the original confidence intervals or Holm families.
"""
import json

import numpy as np

from scripts.seed_block_sensitivity import paired_comparison
from utils.paths import analysis_dir


PAIRS = (
    ("stacked_obs_ppo", "regime_agnostic_ppo"),
    ("stacked_obs_ppo", "varibad_concat"),
    ("stacked_obs_ppo", "varibad_hypernet"),
    ("stacked_obs_ppo", "rl2_concat"),
    ("stacked_obs_ppo", "rl2_hypernet"),
    ("belief_ppo", "regime_agnostic_ppo"),
    ("oracle_ppo", "belief_ppo"),
)


def main():
    evaluation = json.loads((analysis_dir() / "m5r_post_training_evaluation.json").read_text())["methods"]
    values = {}
    for method in {m for pair in PAIRS for m in pair}:
        row = evaluation[method]
        lookup = dict(zip(row["seeds"], row["per_seed_evaluation_return"], strict=True))
        if set(lookup) != set(range(20)):
            raise ValueError(f"incomplete seed set for {method}")
        values[method] = np.array([lookup[s] for s in range(20)], dtype=float)
    rows = []
    for high, low in PAIRS:
        mean, interval, p = paired_comparison(values[high], values[low])
        rows.append({"high": high, "low": low, "mean_difference": mean,
                     "paired_bootstrap_ci": interval, "signflip_p_uncorrected": p})
    output = {"status": "exploratory post-review analysis; no planned test or multiplicity adjustment",
              "seeds": list(range(20)), "bootstrap_draws": 10000, "sign_flips": 100000,
              "rows": rows}
    path = analysis_dir() / "m5r_exploratory_baseline_seed_pairs.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
