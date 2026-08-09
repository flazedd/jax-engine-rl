"""The statistical protocol, as the thesis states it, in one place.

Every constant here has a counterpart in the thesis: Table 3.5 in the
methodology chapter and Table D.8 in the executed-configuration appendix. Code
and thesis drifted apart once already — the bootstrap was on the median while
the protocol defined the mean, the return tests were one-sided while the
protocol said two-sided — so the values live here and `scripts.thesis_contract`
checks the produced artifacts against them.

Change a number here only when the thesis changes, and rerun the contract
audit afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Table 3.5 / Table D.8 --------------------------------------------------
SEEDS = 20
BOOTSTRAP_RESAMPLES = 10_000
INTERVAL_LEVEL = 0.95
ALPHA = 0.05                      # error rate across a comparison set
ALTERNATIVE = "two-sided"         # no direction committed to before the runs
ZERO_METHOD = "wilcox"            # zero differences dropped, n reduced
EFFECT_SIZE = "rank_biserial"     # matched-pairs rank-biserial correlation
CORRECTION = "holm-bonferroni"

# --- Chapter 4 budget -------------------------------------------------------
ITERATIONS = 300
PARALLEL_ENVS = 512
ROLLOUT_LENGTH = 128              # equals the episode length H
EPISODE_LENGTH = 128
PARAM_BUDGET_BAND = (4_900, 5_100)

# The transplant is explicitly outside the protocol: too few seeds for the
# intervals and tests, reported descriptively and in no comparison set.
TRANSPLANT_SEEDS = 3
TRANSPLANT_ITERATIONS = 200

# --- Probe ------------------------------------------------------------------
PROBE_TIMESTEPS = 64_000
PROBE_TRAIN_TIMESTEPS = 51_200
PROBE_TEST_TIMESTEPS = 12_800
PRIMARY_PROBE = "logistic"        # linear probe is the main instrument
ROBUSTNESS_PROBE = "mlp"
PRIMARY_BELIEF_METRIC = "method_kl_to_omega"
DECODABILITY_METRIC = "method_test_acc"
UNCORRECTED_PROBE_METRICS = ("method_log_loss", "method_brier")


@dataclass(frozen=True)
class ComparisonSet:
    """One Holm family. `size` is the number of comparisons corrected together."""

    key: str
    description: str
    size: int


# The three sets of the protocol, with the sizes Appendix D records.
COMPARISON_SETS = {
    "returns_rsmm": ComparisonSet(
        "returns_rsmm",
        "conditioning-architecture return comparisons, two methods over four "
        "RSMM environment instances",
        8,
    ),
    "returns_cartpole": ComparisonSet(
        "returns_cartpole",
        "conditioning-architecture return comparisons, two methods over three "
        "levels, per difficulty axis",
        6,
    ),
    "diagnostics": ComparisonSet(
        "diagnostics",
        "behavioural diagnostics: action separation and belief swap, over both "
        "methods",
        4,
    ),
    "belief_quality": ComparisonSet(
        "belief_quality",
        "belief quality: forward KL and decodability under the linear probe, "
        "over both methods",
        4,
    ),
}

# Fields every corrected comparison must carry, so a figure or table can always
# report what the protocol says is reported.
REQUIRED_COMPARISON_FIELDS = (
    "mean_paired_delta",
    "delta_ci",
    "wilcoxon_p",
    "holm_corrected_p",
    "rank_biserial",
    "stable_under_seed_omission",
)
