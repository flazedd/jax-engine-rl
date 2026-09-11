"""The statistical protocol, as the thesis states it, in one place.

Every constant here has a counterpart in the thesis: the statistical protocol in the
methodology chapter and Tables C.6--C.7 in the reproduction appendix. Code
and thesis drifted apart once already — the bootstrap was on the median while
the protocol defined the mean, the return tests were one-sided while the
protocol said two-sided — so the values live here and `scripts.thesis_contract`
checks the produced artifacts against them.

Change a number here only when the thesis changes, and rerun the contract
audit afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Statistical protocol / Appendix C --------------------------------------
SEEDS = 20
BOOTSTRAP_RESAMPLES = 10_000
INTERVAL_LEVEL = 0.95
ALPHA = 0.05                      # error rate across a comparison set
ALTERNATIVE = "two-sided"         # no direction committed to before the runs
CORRECTION = "holm-bonferroni"

# --- The current experiment set ---------------------------------------------
# Which environment the medium experiment runs on, as a suffix on every
# experiment name. This exists because the suffix was written out by hand in
# nine modules: when the environment changed, the analyses silently kept reading
# the previous set and reported its numbers as current.
#
# e9  rare fills, persistence 0.995. Replaced e_final, whose matched tuple let a
#     memoryless agent identify the regime from one observation, so the floor sat
#     two thirds of the way to the oracle and the attainable gap was 9.41 with a
#     confidence interval 78% as wide as the estimate.
MEDIUM_ENV = "e9"

REFERENCE_ARMS = ("regime_agnostic", "belief", "oracle", "stacked_obs")
METHOD_ARMS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")


def ref_experiment(arm: str, env: str = MEDIUM_ENV) -> str:
    """Experiment name of a reference arm, e.g. m5r_ref_belief_e9."""
    if arm not in REFERENCE_ARMS:
        raise ValueError(f"unknown reference arm {arm!r}; valid: {REFERENCE_ARMS}")
    return f"m5r_ref_{arm}_{env}"


def method_experiment(arm: str, env: str = MEDIUM_ENV) -> str:
    """Experiment name of a method variant, e.g. m5r_final_rl2_concat_e9."""
    if arm not in METHOD_ARMS:
        raise ValueError(f"unknown method arm {arm!r}; valid: {METHOD_ARMS}")
    return f"m5r_final_{arm}_{env}"


# --- Chapter 4 budget -------------------------------------------------------
ITERATIONS = 1500
PARALLEL_ENVS = 512
ROLLOUT_LENGTH = 128              # equals the episode length H
EPISODE_LENGTH = 128
PARAM_BUDGET_BAND = (4_900, 5_100)

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


# The six sets of the protocol, with the sizes Appendix C records.
COMPARISON_SETS = {
    # RQ3's difficulty sweep is deferred: its instances were all defined as
    # perturbations of e_final and need redefining against the current
    # environment. Only the medium instance is run, so the family is the two
    # conditioning comparisons rather than two methods over four instances.
    # Correcting over eight when two were made would misstate the procedure.
    "returns_rsmm": ComparisonSet(
        "returns_rsmm",
        "conditioning-architecture return comparisons, two methods on the "
        "selected RSMM instance",
        2,
    ),
    # Declared after the fact and recorded as such: these two families were
    # reported uncorrected in an earlier draft, which left the thesis correcting
    # some multiplicities and not others. Every member survives correction, so
    # declaring them changes no conclusion, only the consistency of the
    # procedure.
    # Revised after the first analysis: the target is now the regime-agnostic
    # reference return rather than two chosen values per method, so the set is
    # two comparisons rather than four. Appendix C records the change.
    "time_to_threshold": ComparisonSet(
        "time_to_threshold",
        "iterations to reach the regime-agnostic reference return, one per method",
        2,
    ),
    "returns_method": ComparisonSet(
        "returns_method",
        "method return comparisons, RL2 against VariBAD at each conditioning "
        "architecture",
        2,
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
    # The same two metrics read out by the non-linear probe. Corrected within
    # itself rather than pooled with the linear set: pooling would apply an
    # eight-way correction to four claimed hypotheses and weaken them for no
    # gain. Declared after the first analysis; Appendix C records that.
    "belief_quality_mlp": ComparisonSet(
        "belief_quality_mlp",
        "belief quality under the MLP probe: forward KL and decodability, over "
        "both methods",
        4,
    ),
}

# Fields every corrected comparison must carry, so a figure or table can always
# report what the protocol says is reported.
REQUIRED_COMPARISON_FIELDS = (
    "mean_difference",
    "delta_ci",
    "permutation_p",
    "holm_corrected_p",
    "stable_under_run_omission",
)
