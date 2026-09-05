"""Statistical primitives used by M5 ladder + factorial analysis.

All functions take plain numpy / list inputs and return plain dicts so the
results serialize directly into the M5 stats JSON. No JAX, no plotting.

Conventions:
- "method_seeds" / "baseline_seeds" are length-N arrays of per-seed final
  returns, paired by seed index across methods (seed i in method A pairs
  with seed i in method B). All M3/M4/M5 runs use seed_base=0 with seeds
  {0..N-1}, so pairing is just same-index.
- "supported" means BOTH the Holm-corrected p < α AND the bootstrap CI of
  the median paired delta excludes zero. Both are required (not either).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Gap-closed fractions
# ---------------------------------------------------------------------------


def gap_closed(
    method_mean: float,
    floor_mean: float,
    ceiling_mean: float,
) -> float:
    """Fraction of (ceiling − floor) that the method closes.

    Returns NaN if the gap is degenerate (ceiling ≤ floor).
    """
    gap = ceiling_mean - floor_mean
    if gap <= 0:
        return float("nan")
    return float((method_mean - floor_mean) / gap)


# ---------------------------------------------------------------------------
# Bootstrap CI on the (paired) median delta across seeds
# ---------------------------------------------------------------------------


def bootstrap_paired_delta_ci(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Median paired delta + bootstrap CI.

    Returns (median_delta, ci_lo, ci_hi).
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    deltas = a - b
    median = float(np.median(deltas))
    if deltas.size <= 1:
        return median, median, median
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    boot_meds = np.median(deltas[idx], axis=1)
    lo = float(np.percentile(boot_meds, 100 * alpha / 2))
    hi = float(np.percentile(boot_meds, 100 * (1 - alpha / 2)))
    return median, lo, hi


# ---------------------------------------------------------------------------
# Paired Wilcoxon
# ---------------------------------------------------------------------------


def bootstrap_paired_mean_ci(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean paired delta + percentile bootstrap CI.

    This is the estimand the thesis reports: `d_bar`, the mean of the per-seed
    differences, with a percentile interval from resampling the *pairs* as
    units so the dependence induced by the shared seed is preserved.

    Returns (mean_delta, ci_lo, ci_hi).
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    deltas = a - b
    mean = float(np.mean(deltas))
    if deltas.size <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    boot_means = np.mean(deltas[idx], axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def bootstrap_independent_mean_ci(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Difference in sample means with a percentile bootstrap interval.

    The two groups are resampled separately.  Reusing a seed value in two
    training configurations makes a run reproducible; it does not make the
    two trained policies a matched statistical pair.
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    mean = float(a.mean() - b.mean())
    if a.size <= 1 or b.size <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    a_idx = rng.integers(0, a.size, size=(n_boot, a.size))
    b_idx = rng.integers(0, b.size, size=(n_boot, b.size))
    boot = a[a_idx].mean(axis=1) - b[b_idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def permutation_mean_test(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    n_permutations: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided random-label permutation test for a difference in means."""
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        diff = shuffled[:n_a].mean() - shuffled[n_a:].mean()
        count += abs(diff) >= abs(observed)
    return {
        "p": float((count + 1) / (n_permutations + 1)),
        "statistic": observed,
        "n_method": int(a.size),
        "n_baseline": int(b.size),
        "n_permutations": n_permutations,
        "null_distribution": "random-label permutation",
        "alternative": "two-sided",
    }


# ---------------------------------------------------------------------------
# Paired Wilcoxon
# ---------------------------------------------------------------------------


def paired_wilcoxon(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test, per the thesis statistical protocol.

    Two-sided by default: no direction is committed to before the runs. Zero
    differences are dropped and the pair count reduced accordingly, tied
    absolute differences take average ranks, and the exact null distribution is
    used whenever it is valid. Exactness is impossible once ties or dropped
    zeros are present, so the normal approximation with continuity correction
    is used there instead and the choice is recorded in `null_distribution`.

    Falls back to NaN p-value if all pairs are zero (Wilcoxon is undefined).
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    deltas = a - b
    n_all = int(deltas.size)
    if np.all(deltas == 0):
        return {
            "p": float("nan"), "statistic": 0.0, "n_pairs": n_all,
            "n_nonzero_pairs": 0, "n_zero_dropped": n_all,
            "null_distribution": "undefined", "alternative": alternative,
        }
    nonzero = deltas[deltas != 0]
    n_zero = n_all - int(nonzero.size)
    absd = np.abs(nonzero)
    has_ties = bool(np.unique(absd).size != absd.size)
    exact_ok = (n_zero == 0) and (not has_ties)
    method_name = "exact" if exact_ok else "asymptotic"
    res = sps.wilcoxon(
        a, b, alternative=alternative, zero_method="wilcox",
        method=method_name, correction=not exact_ok,
    )
    return {
        "p": float(res.pvalue),
        "statistic": float(res.statistic),
        "n_pairs": n_all,
        "n_nonzero_pairs": int(nonzero.size),
        "n_zero_dropped": n_zero,
        "null_distribution": method_name,
        "alternative": alternative,
    }


# ---------------------------------------------------------------------------
# Matched-pairs rank-biserial correlation (the reported effect size)
# ---------------------------------------------------------------------------


def rank_biserial(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
) -> float:
    """Matched-pairs rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-).

    W+ and W- are the sums of the positive and the negative signed ranks over
    the non-zero differences, with average ranks for ties. Runs from -1 to +1
    on the same rank scale the signed-rank test uses. NaN when every pair is
    zero, where no ranking exists.
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    deltas = a - b
    nonzero = deltas[deltas != 0]
    if nonzero.size == 0:
        return float("nan")
    ranks = sps.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total = w_pos + w_neg
    if total == 0:
        return float("nan")
    return float((w_pos - w_neg) / total)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm step-down correction. NaN inputs pass through as NaN.

    Returns the same number of corrected p-values as inputs, in input order.
    """
    arr = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(arr)
    pv = arr[valid]
    n = pv.size
    out = np.full_like(arr, np.nan, dtype=float)
    if n == 0:
        return out.tolist()
    order = np.argsort(pv, kind="mergesort")
    corrected_sorted = np.empty(n, dtype=float)
    running_max = 0.0
    for i, src in enumerate(order):
        adj = pv[src] * (n - i)
        running_max = max(running_max, adj)
        corrected_sorted[src] = min(1.0, running_max)
    out[valid] = corrected_sorted
    return out.tolist()


# ---------------------------------------------------------------------------
# Cliff's δ (paired)
# ---------------------------------------------------------------------------


def cliffs_delta(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
) -> float:
    """Paired Cliff's δ on the (method − baseline) deltas — fraction of
    positive minus fraction of negative deltas, ties counted as zero.
    Range: [−1, +1]. Convention: δ > 0 means method tends to exceed baseline.
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    deltas = a - b
    n = deltas.size
    if n == 0:
        return float("nan")
    pos = int(np.sum(deltas > 0))
    neg = int(np.sum(deltas < 0))
    return float((pos - neg) / n)


# ---------------------------------------------------------------------------
# Leave-one-seed-out robustness
# ---------------------------------------------------------------------------


def leave_one_out_sensitivity(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    alpha: float = 0.05,
    n_corrections: int = 1,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """For each seed i, drop it and re-test. Flag if removing any single
    seed flips the "supported" decision.

    Supported test = Holm-corrected p < alpha for this single comparison
    (n_corrections=1 means no correction); for primary-hypothesis use,
    pass n_corrections = total family size.
    """
    a = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    n = a.size
    if n <= 1:
        return {"robust_to_loo": False, "flipping_seed": None, "n_seeds": int(n)}
    flipping = None
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        a_i, b_i = a[keep], b[keep]
        res = paired_wilcoxon(a_i, b_i, alternative=alternative)
        p_i = res["p"]
        # Same correction multiplier as the family-wide test.
        p_corr = min(1.0, p_i * n_corrections) if not np.isnan(p_i) else np.nan
        supported_i = (not np.isnan(p_corr)) and p_corr < alpha
        if not supported_i:
            flipping = int(i)
            break
    return {
        # Thesis name for the property; `robust_to_loo` is kept as an alias so
        # existing readers of this dict keep working.
        "stable_under_seed_omission": flipping is None,
        "robust_to_loo": flipping is None,
        "flipping_seed": flipping,
        "n_seeds": int(n),
    }


# ---------------------------------------------------------------------------
# Per-seed method ranking — stable across seeds?
# ---------------------------------------------------------------------------


def ranking_stable_across_seeds(
    seed_returns_per_method: dict[str, list[float]],
    min_agreement: int | None = None,
) -> dict[str, Any]:
    """For each seed, compute the descending-by-return ranking of methods.
    Pick the modal ranking and report the count of seeds that match it.

    seed_returns_per_method must have all methods aligned by seed index,
    same length. Returns:
      {"reference_ranking": [...], "agreeing_seeds": k, "n_seeds": n,
       "stable": bool}
    "stable" defaults to "agreeing_seeds >= ceil(0.8 * n_seeds)".
    """
    methods = list(seed_returns_per_method.keys())
    arrs = np.stack([np.asarray(seed_returns_per_method[m]) for m in methods], axis=0)
    n_methods, n_seeds = arrs.shape
    rankings: list[tuple[str, ...]] = []
    for s in range(n_seeds):
        order = np.argsort(-arrs[:, s], kind="mergesort")
        rankings.append(tuple(methods[i] for i in order))
    counts: dict[tuple[str, ...], int] = {}
    for r in rankings:
        counts[r] = counts.get(r, 0) + 1
    ref_ranking, ref_count = max(counts.items(), key=lambda kv: kv[1])
    if min_agreement is None:
        min_agreement = int(np.ceil(0.8 * n_seeds))
    return {
        "reference_ranking": list(ref_ranking),
        "agreeing_seeds": int(ref_count),
        "n_seeds": int(n_seeds),
        "min_agreement": int(min_agreement),
        "stable": bool(ref_count >= min_agreement),
    }


# ---------------------------------------------------------------------------
# Convenience: full primary-hypothesis test
# ---------------------------------------------------------------------------


def primary_hypothesis_test(
    method: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    family_size: int,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Run the full primary-hypothesis decision of the thesis protocol.

    A two-sided random-label permutation test, Holm correction across
    `family_size`, and the difference between sample means with its percentile
    bootstrap interval.

    `supported` is the Holm-corrected p alone: the protocol treats the test as
    the inferential decision and the interval as the statement of magnitude and
    precision, not as a second gate. `ci_excludes_zero` is reported alongside so
    a reader can see both.
    """
    perm = permutation_mean_test(method, baseline)
    p_corr = (
        float("nan") if np.isnan(perm["p"]) else min(1.0, perm["p"] * family_size)
    )
    mean, lo, hi = bootstrap_independent_mean_ci(
        method, baseline, n_boot=n_boot, alpha=alpha
    )

    if alternative == "greater":
        ci_excludes_zero = lo > 0
    elif alternative == "less":
        ci_excludes_zero = hi < 0
    else:
        ci_excludes_zero = (lo > 0) or (hi < 0)
    p_passes = (not np.isnan(p_corr)) and p_corr < alpha
    return {
        "mean_paired_delta": mean,
        "delta_ci": [lo, hi],
        "wilcoxon_p": perm["p"],
        "wilcoxon_statistic": perm["statistic"],
        "wilcoxon_null_distribution": perm["null_distribution"],
        "alternative": alternative,
        "n_pairs": int(np.asarray(method).size),
        "n_nonzero_pairs": int(np.asarray(method).size),
        "n_zero_dropped": 0,
        "holm_corrected_p": p_corr,
        "ci_excludes_zero": bool(ci_excludes_zero),
        "supported": bool(p_passes),
    }
