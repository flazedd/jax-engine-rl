import numpy as np

from scripts.recovery_audit import ages_since_change, summarize


def test_recovery_age_excludes_initial_regime_and_resets_at_each_change():
    truth = np.array([[0, 2], [0, 2], [1, 2], [1, 2], [2, 2], [2, 2]])
    np.testing.assert_array_equal(ages_since_change(truth),
                                 [[-1, -1], [-1, -1], [0, -1], [1, -1], [0, -1], [1, -1]])


def test_extended_bins_partition_post_change_steps_and_preserve_paired_gap():
    truth = np.zeros((128, 2), dtype=int)
    truth[1:, 0] = 1
    age = ages_since_change(truth)
    method = age % 2 == 0
    reference = age % 3 == 0
    method_scores, counts = summarize(method, age)
    ref_scores, _ = summarize(reference, age)
    assert sum(counts) == 127
    assert counts[-1] == 27
    weighted_gap = np.dot(np.subtract(method_scores, ref_scores), counts) / sum(counts)
    assert np.isclose(weighted_gap, (method[age >= 0].astype(float) - reference[age >= 0]).mean())


def test_empty_bins_are_explicitly_missing():
    scores, counts = summarize(np.ones((2, 1), bool), np.full((2, 1), -1))
    assert all(score is None for score in scores)
    assert not any(counts)
