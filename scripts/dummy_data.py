"""Synthetic stand-ins for every artifact the programme produces.

Purpose: let ``run_matched_programme --dummy`` produce a complete set of
figures and tables in a couple of minutes, so the results chapter can be laid
out while the real programme is still training. The numbers are fake; the
*shapes* are real, because every figure is then rendered by the same plotting
code the real run uses.

How the values are produced, in order of preference:

1. **Clone the real artifact.** Most stages already have output on disk from a
   previous programme run. That file is read as a schema template, its
   structure kept exactly, and its numeric leaves jittered. Figures then come
   out looking like the finished thesis, which is what makes the layout
   decisions transfer.
2. **Synthesise from the registry below**, for artifacts that have never been
   produced — the belief-quality comparison set, for instance, which the
   protocol added after the last real run.

Every file written carries ``"dummy": true`` at the top level, and every run
writes under a separate results root, so a dummy artifact can never be mistaken
for a real one or overwrite one.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Fields whose value must survive jitter untouched: they are structural, and
# perturbing them either breaks a downstream reader or silently changes what
# the figure claims.
_STRUCTURAL = {
    "seed", "seeds", "n_seeds", "n_pairs", "n_total", "n_nonzero_pairs",
    "n_zero_dropped", "family_size", "iterations", "num_seeds", "parallel_envs",
    "rollout_length", "n_rollouts", "n_boot", "alpha", "n_scatter_points",
    "n_cells_attempted", "n_cells_failed", "schema_version", "episode_length",
}


def _jitter_scalar(x: float, rng: np.random.Generator, rel: float) -> float:
    """Relative jitter that preserves sign and magnitude.

    Kept small on purpose. The orderings the figures exist to show — hypernet
    above concat, references bracketing the variants — must survive, or the
    dummy figures stop being useful for planning the narrative.
    """
    if not math.isfinite(x) or x == 0.0:
        return x
    return float(x * (1.0 + rng.normal(0.0, rel)))


def _jitter(obj: Any, rng: np.random.Generator, rel: float, key: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: _jitter(v, rng, rel, key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jitter(v, rng, rel, key=key) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if key in _STRUCTURAL:
            return obj
        # p-values stay in [0, 1] and stay on the same side of alpha, so
        # "supported" flags in the same file remain self-consistent.
        if "p" == key or key.endswith("_p") or key.startswith("p_"):
            return float(min(1.0, max(0.0, _jitter_scalar(obj, rng, rel))))
        return _jitter_scalar(obj, rng, rel)
    return obj


_CI_HINTS = ("_ci95", "_ci", "ci95", "ci_", "_interval")


def _is_ci_key(key: str) -> bool:
    return key == "ci" or key.endswith(("_ci95", "_ci")) or key in {"ci95", "delta_ci"}


def _repair_intervals(obj: Any) -> Any:
    """Restore the invariant lo <= centre <= hi after jitter.

    Jitter moves a mean and its interval bounds independently, which can leave
    a bound on the wrong side of its centre. Matplotlib rejects that outright
    with "yerr must not contain negative values", so every confidence interval
    is re-widened to contain the sibling means it belongs to.
    """
    if isinstance(obj, list):
        return [_repair_intervals(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out = {k: _repair_intervals(v) for k, v in obj.items()}

    centres = [
        v for k, v in out.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        and ("mean" in k or "median" in k or k in {"value", "delta"})
    ]
    for key, val in list(out.items()):
        pair = None
        if _is_ci_key(key) and isinstance(val, list) and len(val) == 2:
            pair = [float(val[0]), float(val[1])]
        if pair is None:
            continue
        lo, hi = sorted(pair)
        for c in centres:
            if math.isfinite(c):
                lo, hi = min(lo, c), max(hi, c)
        out[key] = [lo, hi]

    if "ci_low" in out and "ci_high" in out:
        lo, hi = sorted((float(out["ci_low"]), float(out["ci_high"])))
        for c in centres:
            if math.isfinite(c):
                lo, hi = min(lo, c), max(hi, c)
        out["ci_low"], out["ci_high"] = lo, hi
    return out


# Belief quality by variant, for a probe artifact produced before the KL
# metrics existed. Ordered as the thesis reports it: within each method the
# concat variant sits at the lower divergence, which is the reversal the
# decoupling argument rests on. Keyed by probe family, since the MLP probe
# reads every representation a little closer.
_DUMMY_KL = {
    "logistic": {"rl2_concat": 0.32, "rl2_hypernet": 0.42,
                 "varibad_concat": 0.61, "varibad_hypernet": 0.70},
    "mlp": {"rl2_concat": 0.27, "rl2_hypernet": 0.33,
            "varibad_concat": 0.50, "varibad_hypernet": 0.55},
}


def _enrich_probe(payload: dict, rng: np.random.Generator) -> dict:
    """Fill in probe metrics the stored artifact predates.

    A dummy table exists to show what the real one will look like. Leaving the
    belief-quality column empty because the last real probe run did not emit KL
    defeats that, so the dummy layer supplies the metrics the current probe code
    produces.
    """
    if "per_method_per_env" not in payload:
        return payload
    fam = "mlp" if payload.get("classifier") == "mlp" else "logistic"
    table = _DUMMY_KL[fam]
    for env_block in payload.get("per_method_per_env", {}).values():
        for method, block in env_block.items():
            if not isinstance(block, dict) or "method_kl_to_omega_mean" in block:
                continue
            base = table.get(method)
            if base is None:
                continue
            n = int(block.get("n_seeds") or SEEDS)
            per_seed = [float(x) for x in rng.normal(base, base * 0.08, n)]
            block["method_kl_to_omega_mean"] = float(np.mean(per_seed))
            block["method_kl_to_omega_per_seed"] = per_seed
            block["method_kl_to_omega_ci95"] = [
                float(np.percentile(per_seed, 2.5)),
                float(np.percentile(per_seed, 97.5)),
            ]
            block["analytical_kl_to_omega_mean"] = float(abs(rng.normal(0.02, 0.005)))
    # the per-seed scatter feeds the belief-quality comparison set
    for pt in payload.get("scatter_points", []):
        base = table.get(pt.get("method"))
        if base is None or "method_kl_to_omega" in pt:
            continue
        pt["method_kl_to_omega"] = float(rng.normal(base, base * 0.08))
        pt["analytical_kl_to_omega"] = float(abs(rng.normal(0.02, 0.005)))
        pt["method_log_loss"] = float(abs(rng.normal(base * 2.0, base * 0.15)))
        pt["method_brier"] = float(abs(rng.normal(base * 0.9, base * 0.1)))
        pt["belief_error_kl"] = pt["method_kl_to_omega"] - pt["analytical_kl_to_omega"]
    return payload


def clone_with_jitter(
    template: Path, rel: float = 0.03, seed: int = 0
) -> dict[str, Any] | None:
    """Read a real artifact and return a jittered copy of it, or None."""
    try:
        payload = json.loads(template.read_text())
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    out = _repair_intervals(_jitter(payload, rng, rel))
    if isinstance(out, dict) and "posterior_vs_performance" in template.name:
        out = _enrich_probe(out, rng)
    if isinstance(out, dict):
        out["dummy"] = True
        out["dummy_source"] = str(template)
    return out


# ---------------------------------------------------------------------------
# Synthesised artifacts, for stages with no prior output on disk
# ---------------------------------------------------------------------------

# Anchored on the values the thesis currently reports, so a synthesised figure
# sits on the same scale as a cloned one.
# Keyed by the experiment-directory suffix, which is what the driver passes in:
# results/m5r_ref_belief_e9 -> "belief", not "belief_ppo".
_REFS = {"regime_agnostic": 138.0, "belief": 168.8, "belief_ppo": 168.8,
         "oracle": 182.0, "oracle_ppo": 182.0, "stacked_obs": 141.5}
_VARIANTS = {"rl2_concat": 118.7, "rl2_hypernet": 163.1,
             "varibad_concat": 108.9, "varibad_hypernet": 163.5}
SEEDS = 20


def _per_seed(mean: float, sd: float, rng: np.random.Generator) -> list[float]:
    return [float(x) for x in rng.normal(mean, sd, SEEDS)]


def synth_training_summary(experiment: str, seed: int = 0) -> dict[str, Any]:
    return {
        "script": "train", "status": "OK",
        "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:01:00",
        "run_mode": "full", "config_used": {"experiment_name": experiment},
        "key_stats": {"method": "ppo", "num_seeds": SEEDS, "iterations": 300},
        "outputs_written": [], "error": None, "schema_version": "0.1",
        "dummy": True,
    }


def synth_training_metrics(experiment: str, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(abs(hash(experiment)) % (2**32))
    # m5r_ref_<name>_e_final / m5r_final_<cell>_e_final -> <name> / <cell>
    key = experiment
    for pre in ("m5r_ref_", "m5r_final_"):
        if key.startswith(pre):
            key = key[len(pre):]
    for suf in ("_e_final",):
        if key.endswith(suf):
            key = key[: -len(suf)]
    mean = {**_REFS, **_VARIANTS}.get(key, 140.0)
    per_seed = _per_seed(mean, 6.0, rng)
    curve = list(np.linspace(mean - 45, mean, 300) + rng.normal(0, 1.5, 300))
    return {
        "experiment_name": experiment, "run_mode": "full", "agent": "ppo",
        "env": "market_making_v1", "iterations": 300, "num_seeds": SEEDS,
        "parallel_envs": 512, "rollout_length": 128,
        "mean_return_per_iter": [float(x) for x in curve],
        "var_return_per_iter": [float(abs(x)) for x in rng.normal(20, 3, 300)],
        "per_seed_mean_return_per_iter": [
            [float(v) for v in curve + rng.normal(0, 4, 300)] for _ in range(SEEDS)
        ],
        "per_seed_final_return": per_seed,
        "final_return_mean": float(np.mean(per_seed)),
        "final_return_ci95": [
            float(np.percentile(per_seed, 2.5)), float(np.percentile(per_seed, 97.5))
        ],
        "timing": {"total_s": 1.0},
        "dummy": True,
    }


def synth_belief_quality_tests() -> dict[str, Any]:
    """The third comparison set, which no previous run produced."""
    rng = np.random.default_rng(7)
    comps = []
    for method, kl_gap, acc_gap in (("rl2", 0.095, -0.039), ("varibad", 0.091, -0.059)):
        for metric, delta, direction in (
            ("method_kl_to_omega", kl_gap, "lower"),
            ("method_test_acc", acc_gap, "higher"),
        ):
            lo, hi = sorted((delta * 0.75, delta * 1.25))
            comps.append({
                "name": f"{method}_hypernet_vs_concat_{metric}",
                "method": method, "metric": metric,
                "direction_favouring_hypernet": direction,
                "n_pairs": SEEDS, "seeds": list(range(SEEDS)),
                "hypernet_mean": float(rng.uniform(0.2, 0.8)),
                "concat_mean": float(rng.uniform(0.2, 0.8)),
                "mean_paired_delta": float(delta),
                "delta_ci": [float(lo), float(hi)],
                "rank_biserial": float(np.sign(delta) * rng.uniform(0.65, 0.99)),
                "wilcoxon_p": float(rng.uniform(1e-5, 1e-3)),
                "wilcoxon_null_distribution": "exact",
                "n_zero_dropped": 0,
                "favours_hypernet": bool(
                    delta < 0 if direction == "lower" else delta > 0
                ),
                "holm_corrected_p": float(rng.uniform(1e-4, 6e-3)),
                "supported": True,
                "stable_under_seed_omission": True,
                "ci_excludes_zero": True,
            })
    return {
        "comparison_set": "belief_quality", "classifier": "logistic",
        "env_label": "e_final", "alternative": "two-sided", "alpha": 0.05,
        "family_size": len(comps),
        "uncorrected_robustness_checks": [
            "method_log_loss", "method_brier", "mlp probe (all metrics)",
        ],
        "n_supported": len(comps), "comparisons": comps, "dummy": True,
    }


# Stage name -> zero-argument synthesiser, for artifacts with no template.
def synth_reference_ordering_gate() -> dict[str, Any]:
    """The Chapter 4 prerequisite, which no previous run produced."""
    return {
        "gate": "reference_ordering",
        "ordering": "regime-agnostic < Belief-PPO < Oracle-PPO",
        "criterion": "lower bootstrap bound on each adjacent paired difference "
                     "is positive",
        "checks": [
            {"pair": "belief_over_agnostic", "higher": "m5r_ref_belief_e9",
             "lower": "m5r_ref_regime_agnostic_e9",
             "mean_paired_delta": 30.8, "delta_ci": [27.4, 34.1],
             "n_pairs": SEEDS, "lower_bound_positive": True},
            {"pair": "oracle_over_belief", "higher": "m5r_ref_oracle_e9",
             "lower": "m5r_ref_belief_e9",
             "mean_paired_delta": 13.2, "delta_ci": [10.1, 16.4],
             "n_pairs": SEEDS, "lower_bound_positive": True},
        ],
        "ordering_holds": True,
        "consequence_if_failed": "instance excluded from normalised "
                                 "(gap-closed) comparisons; raw returns remain "
                                 "reportable",
        "dummy": True,
    }


SYNTHESISERS = {
    "analysis:belief_quality_tests": synth_belief_quality_tests,
    "gate:reference_ordering": synth_reference_ordering_gate,
}


# ---------------------------------------------------------------------------
# Seeding the dummy tree with the inputs the figure scripts read
# ---------------------------------------------------------------------------

# The figure stages run for real in a dummy programme, so every artifact they
# read has to exist under the dummy results root. These are the globs those
# scripts touch; anything outside them is never opened by a figure.
INPUT_GLOBS = (
    "foundations/**/*.json",
    "medium/*/metrics.json",
    "sweep/*/metrics.json",
    "cartpole/**/*.json",
    "analysis/*.json",
    "audits/*.json",
    # The archive is the only source of templates until the new layout has
    # been populated by a real run; cloning from it keeps the dummy chain
    # useful in the meantime.
    "_archive/**/metrics.json",
    "_archive/milestones/**/*.json",   # M2/M4/M5 stats the figures read
    "_archive/audits/*.json",
    "_archive/M5R/final/*.json",
)


def _relocate(src: Path, real_root: Path) -> Path:
    """Map an archived template onto the layout its reader now expects.

    Templates come from `_archive`, which mirrors the old milestone tree, but
    every reader has moved to the thesis-aligned layout. Writing the sibling
    beside the template would put it where nothing looks for it.
    """
    from utils.paths import analysis_dir, cartpole_dir, experiment_dir

    rel_path = src.relative_to(real_root)
    parts = rel_path.parts
    if parts[0] != "_archive":
        return src
    inner = Path(*parts[1:])
    # Experiment directories first: a per-run artifact keeps its directory,
    # or every run's metrics collapse onto one path.
    if len(inner.parts) == 2 and inner.name in ("metrics.json", "summary.json"):
        return experiment_dir(inner.parts[0]) / inner.name
    if inner.parts[:2] == ("M5R", "final"):
        return analysis_dir() / inner.name
    if "cartpole" in inner.parts[0] or (
        inner.parts[0] == "milestones" and "cartpole" in inner.parts
    ):
        return cartpole_dir() / inner.name
    if inner.parts[0] == "milestones":
        from utils.paths import foundations_dir
        return foundations_dir() / inner.name
    return real_root / inner


def mirror_inputs(real_root: Path, rel: float = 0.03) -> int:
    """Write a jittered .dummy.json beside every figure input.

    Jittered on the way through, so a dummy figure is visibly not the real one
    under inspection, while keeping every ordering the layout depends on.
    """
    from utils.paths import dummy_sibling

    written = 0
    for pattern in INPUT_GLOBS:
        for src in real_root.glob(pattern):
            if not src.is_file() or src.name.endswith(".dummy.json"):
                continue
            dst = dummy_sibling(_relocate(src, real_root))
            # Refresh a stale sibling: when the real artifact gains a field the
            # thesis now reports, a sibling cloned before that change silently
            # produces tables missing the new column.
            if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                continue
            payload = clone_with_jitter(src, rel=rel, seed=abs(hash(str(src))) % 2**31)
            if payload is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(payload, indent=2))
            written += 1
    return written
