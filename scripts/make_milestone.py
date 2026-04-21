"""Orchestrate a milestone end-to-end.

    uv run python -m scripts.make_milestone M0

For M0: runs the dummy pipeline in super-fast / fast / full, runs the
run-mode test, regenerates figures, and writes stats_M0_pipeline.json with
the pass-criteria fields.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from envs.mm_reduced import ACTION_FAVOR_ASK, ACTION_FAVOR_BID, MMReducedEnv
from oracles.analytical_as import solve_analytical_as
from plotting.regenerate_figures import _HANDLERS as FIGURE_HANDLERS
from training.config import apply_run_mode, load_config
from training.train import train, train_or_sweep
from utils.script_output import (
    ScriptRun,
    validate_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"


def _run_training(config_path: Path, mode: str) -> tuple[float, bool, bool]:
    cfg = load_config(config_path)
    apply_run_mode(cfg, mode)
    t0 = time.perf_counter()
    exit_zero = True
    try:
        train(cfg)
    except SystemExit as e:
        exit_zero = (e.code == 0)
    except Exception:
        exit_zero = False
        raise
    elapsed = time.perf_counter() - t0

    summary_path = RESULTS_ROOT / cfg.experiment_name / "summary.json"
    wrote_json = summary_path.exists()
    schema_ok = False
    if wrote_json:
        with open(summary_path) as f:
            summary = json.load(f)
        schema_ok, _ = validate_summary(summary)
    return elapsed, (exit_zero and wrote_json), schema_ok


def _regenerate_figures(milestone: str) -> bool:
    handler = FIGURE_HANDLERS.get(milestone)
    if handler is None:
        return False
    inner_run = ScriptRun(script="regenerate_figures")
    try:
        handler(inner_run)
        summary_path = RESULTS_ROOT / "milestones" / milestone / f"plot_summary_{milestone}.json"
        inner_run.ok(
            key_stats={"milestone": milestone, "figures": len(inner_run.outputs)},
            summary_path=summary_path,
        )
        return True
    except Exception as e:
        summary_path = RESULTS_ROOT / "milestones" / milestone / f"plot_summary_{milestone}.json"
        inner_run.fail(reason=str(e), summary_path=summary_path)
        return False


def make_m0() -> dict:
    config = CONFIG_ROOT / "m0_dummy.yaml"
    durations: dict[str, float] = {}
    exit_zero_all = True
    wrote_json_all = True
    schema_ok_all = True

    for mode in ("super_fast", "fast", "full"):
        elapsed, ok, schema_ok = _run_training(config, mode)
        durations[mode] = elapsed
        exit_zero_all &= ok
        wrote_json_all &= ok  # wrote_json folded into ok above
        schema_ok_all &= schema_ok

    figures_ok = _regenerate_figures("M0")

    # Shared-schema validation of every summary.json in the last full run.
    # (Schema is the contract for Claude Code; do a full sweep.)
    all_jsons_ok = True
    for path in RESULTS_ROOT.rglob("summary.json"):
        with open(path) as f:
            s = json.load(f)
        ok, _ = validate_summary(s)
        all_jsons_ok &= ok

    key_stats = {
        "pipeline_version": "0.1",
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "run_modes_tested": ["super_fast", "fast", "full"],
        "super_fast_duration_seconds": round(durations["super_fast"], 3),
        "fast_duration_seconds": round(durations["fast"], 3),
        "full_duration_seconds": round(durations["full"], 3),
        "all_scripts_exit_zero": bool(exit_zero_all),
        "all_scripts_wrote_expected_json": bool(wrote_json_all),
        "schema_validates": bool(schema_ok_all and all_jsons_ok),
        "figures_regenerated": bool(figures_ok),
        "make_milestone_script_succeeded": bool(
            exit_zero_all
            and wrote_json_all
            and schema_ok_all
            and all_jsons_ok
            and figures_ok
            and durations["super_fast"] < 30.0
            and durations["fast"] < 300.0
        ),
    }
    return key_stats


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0.0:
        return 0.0
    return float((xc * yc).sum() / denom)


def make_m1() -> dict:
    config = CONFIG_ROOT / "m1_ppo_as.yaml"

    print(
        "[make_milestone] M1 plan: super_fast → fast → full, then figures + diagnostics.\n"
        "[make_milestone] rough wall-clock: super_fast ~10s, fast ~30s, full ~4-6min (depends on epochs/seeds).",
        flush=True,
    )

    durations: dict[str, float] = {}
    exit_zero_all = True
    schema_ok_all = True
    for mode in ("super_fast", "fast", "full"):
        print(f"[make_milestone] running mode={mode}", flush=True)
        elapsed, ok, schema_ok = _run_training(config, mode)
        print(f"[make_milestone] mode={mode} done in {elapsed:.1f}s", flush=True)
        durations[mode] = elapsed
        exit_zero_all &= ok
        schema_ok_all &= schema_ok

    cfg_for_env = load_config(config)
    env = MMReducedEnv(**cfg_for_env.env.params)
    sol = solve_analytical_as(env)
    as_return = float(sol.expected_episode_return)

    metrics_path = RESULTS_ROOT / "m1_ppo_as" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    final_per_seed = [float(x) for x in metrics["per_seed_final_return"]]
    final_mean = float(metrics["final_return_mean"])
    final_ci = [float(x) for x in metrics["final_return_ci95"]]
    return_ratio = final_mean / as_return if as_return > 0 else 0.0

    mean_curve = np.asarray(metrics["mean_return_per_iter"], dtype=float)
    iters = np.arange(mean_curve.size)
    last20 = mean_curve[-20:]
    if last20.size >= 2:
        slope = float(np.polyfit(np.arange(last20.size), last20, 1)[0])
    else:
        slope = 0.0
    plateau_threshold = 0.99 * final_mean
    above = np.where(mean_curve >= plateau_threshold)[0]
    plateau_reached_at = int(above[0]) if above.size > 0 else int(mean_curve.size)
    diffs = np.diff(mean_curve)
    monotonic_increase_fraction = float((diffs > 0).mean()) if diffs.size > 0 else 0.0
    converged = bool(abs(slope) < 0.5 and plateau_reached_at < 90)

    per_seed_probs = np.asarray(metrics["per_seed_final_action_probs"], dtype=float)
    probs_mean = per_seed_probs.mean(axis=0)
    ppo_skew = probs_mean[:, ACTION_FAVOR_ASK] - probs_mean[:, ACTION_FAVOR_BID]
    as_skew = np.asarray(sol.skew, dtype=float)
    q_margin = np.asarray(sol.q_margin, dtype=float)

    # Margin filter: states where AS has a clear preference. At small margin
    # (incl. boundary ties and near-tied states where skew ≈ sym in Q-value)
    # the AS argmax is arbitrary and shouldn't dominate the comparison.
    # Threshold 0.05 chosen so the filter keeps states with >~5% per-step Q gap.
    margin_threshold = 0.05
    meaningful = q_margin > margin_threshold

    corr_full = _pearson(ppo_skew, as_skew)
    if meaningful.sum() >= 2:
        corr_meaningful = _pearson(ppo_skew[meaningful], as_skew[meaningful])
    else:
        corr_meaningful = float("nan")

    nonzero = as_skew != 0
    if nonzero.any():
        agreement_full = float((np.sign(ppo_skew[nonzero]) == np.sign(as_skew[nonzero])).mean())
    else:
        agreement_full = 1.0
    if meaningful.any():
        agreement_meaningful = float(
            (np.sign(ppo_skew[meaningful]) == np.sign(as_skew[meaningful])).mean()
        )
    else:
        agreement_meaningful = 1.0

    # Primary pass criterion: shape matches on states where AS has a real
    # preference. Full-vector stats kept as diagnostics.
    policy_shape_matches = bool(
        (not np.isnan(corr_meaningful)) and corr_meaningful > 0.9 and agreement_meaningful > 0.95
    )

    figures_ok = _regenerate_figures("M1")

    wall_time = float(sum(durations.values()))

    pass_criteria = bool(
        return_ratio >= 0.95 and converged and policy_shape_matches
    )

    stats = {
        "method": "ppo",
        "env": "as_e0",
        "seeds": list(range(cfg_for_env.num_seeds)),
        "final_return_per_seed": final_per_seed,
        "final_return_mean": final_mean,
        "final_return_ci": final_ci,
        "as_analytical_return": as_return,
        "return_ratio": return_ratio,
        "iterations": int(metrics["iterations"]),
        "parallel_envs": int(metrics["parallel_envs"]),
        "wall_time_seconds": wall_time,
        "slope_last_20_iterations": slope,
        "plateau_reached_at_iteration": plateau_reached_at,
        "monotonic_increase_fraction": monotonic_increase_fraction,
        "converged": converged,
        "q_margin_threshold": margin_threshold,
        "n_meaningful_states": int(meaningful.sum()),
        "policy_skew_correlation": corr_meaningful,
        "policy_skew_direction_agreement": agreement_meaningful,
        "policy_skew_correlation_full": corr_full,
        "policy_skew_direction_agreement_full": agreement_full,
        "policy_shape_matches": policy_shape_matches,
        "figures_regenerated": bool(figures_ok),
        "all_scripts_exit_zero": bool(exit_zero_all),
        "schema_validates": bool(schema_ok_all),
        "make_milestone_script_succeeded": bool(
            pass_criteria and figures_ok and exit_zero_all and schema_ok_all
        ),
    }
    return stats


def _stream_subprocess(cmd: list[str], prefix: str) -> tuple[int, float]:
    """Run `cmd`, streaming stdout/stderr line-by-line with a prefix.

    Returns (returncode, elapsed_seconds). Output is not captured — it goes
    straight to this process's stdout so the user sees progress live.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        # Strip trailing newline, we'll add our own via print.
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    proc.wait()
    elapsed = time.perf_counter() - t0
    return proc.returncode, elapsed


def make_m2() -> dict:
    """Run R1–R4 verification on the current E_final env in all three modes.

    Regen figures from the last (full-mode) run's JSON. Full mode is the one
    that actually carries statistical power; super_fast and fast are sanity
    checks that the pipeline is wiring-compatible.
    """
    env_cfg_path = CONFIG_ROOT / "envs" / "e_final.yaml"

    # Per-mode budget, used for the ETA breakdown.
    budgets = {
        "super_fast": dict(iterations=2, parallel_envs=16, num_seeds=1),
        "fast":       dict(iterations=20, parallel_envs=128, num_seeds=1),
        "full":       dict(iterations=40, parallel_envs=256, num_seeds=3),
    }
    rough_estimates = {"super_fast": 5, "fast": 20, "full": 240}
    n_ppo_runs = 6  # 3 per-regime + regime-agnostic + oracle + belief

    print(
        "[make_milestone] M2 orchestration plan:\n"
        "[make_milestone]   3 modes × (VI + 6 PPO runs + posterior sim + figures)\n"
        f"[make_milestone]   super_fast: {budgets['super_fast']['iterations']} iter × "
        f"{budgets['super_fast']['parallel_envs']} envs × "
        f"{budgets['super_fast']['num_seeds']} seeds → ~{rough_estimates['super_fast']}s\n"
        f"[make_milestone]   fast:       {budgets['fast']['iterations']} iter × "
        f"{budgets['fast']['parallel_envs']} envs × "
        f"{budgets['fast']['num_seeds']} seeds → ~{rough_estimates['fast']}s\n"
        f"[make_milestone]   full:       {budgets['full']['iterations']} iter × "
        f"{budgets['full']['parallel_envs']} envs × "
        f"{budgets['full']['num_seeds']} seeds → ~{rough_estimates['full']}s (~4 min)\n"
        f"[make_milestone]   total rough: ~{sum(rough_estimates.values())}s",
        flush=True,
    )

    durations: dict[str, float] = {}
    pass_per_mode: dict[str, bool] = {}
    stats_per_mode: dict[str, dict] = {}
    exit_zero_all = True
    t_start = time.perf_counter()

    for i, mode in enumerate(("super_fast", "fast", "full"), start=1):
        elapsed_so_far = time.perf_counter() - t_start
        eta_remaining = sum(
            rough_estimates[m] for m in ("super_fast", "fast", "full") if m not in durations
        )
        print(
            f"\n[make_milestone] ===== mode {i}/3: {mode} "
            f"(elapsed={elapsed_so_far:.1f}s, eta_remaining≈{eta_remaining}s) =====",
            flush=True,
        )
        print(
            f"[make_milestone]   expected: ~{n_ppo_runs} PPO runs "
            f"(~{rough_estimates[mode] / n_ppo_runs:.1f}s each + VI + posterior sim)",
            flush=True,
        )

        flag = "--super-fast" if mode == "super_fast" else ("--fast" if mode == "fast" else "")
        cmd = [
            sys.executable,
            "-m",
            "oracles.verify_requirements",
            "--env-config",
            str(env_cfg_path),
        ]
        if flag:
            cmd.append(flag)

        rc, elapsed = _stream_subprocess(cmd, prefix=f"[{mode}] ")
        durations[mode] = elapsed
        # verify_requirements returns 1 when all_pass is False — still a valid
        # orchestrator run; we record the per-R pass from the stats JSON.
        exit_zero_all &= (rc in (0, 1))
        delta_vs_estimate = elapsed - rough_estimates[mode]
        print(
            f"[make_milestone]   mode={mode} done in {elapsed:.1f}s "
            f"(Δ vs rough estimate: {delta_vs_estimate:+.1f}s, rc={rc})",
            flush=True,
        )

        stats_path = (
            RESULTS_ROOT / "milestones" / "M2" / "stats_M2_requirements.json"
        )
        if stats_path.exists():
            with open(stats_path) as f:
                s = json.load(f)
            pass_per_mode[mode] = bool(s.get("all_pass", False))
            stats_per_mode[mode] = s
            print(
                f"[make_milestone]   {mode} R1={s['R1_policy_disagreement']['pass']} "
                f"R2={s['R2_per_regime_ppo_vs_vi']['pass']} "
                f"R3={s['R3_mixed_gap']['pass']} "
                f"R4={s['R4_inferability']['pass']} "
                f"→ all_pass={s.get('all_pass', False)}",
                flush=True,
            )
        else:
            pass_per_mode[mode] = False
            stats_per_mode[mode] = {}
            print(f"[make_milestone]   {mode} stats JSON missing — treating as fail", flush=True)

    total_elapsed = time.perf_counter() - t_start
    print(
        f"\n[make_milestone] all modes done in {total_elapsed:.1f}s "
        f"(super_fast={durations['super_fast']:.1f}s, "
        f"fast={durations['fast']:.1f}s, full={durations['full']:.1f}s) → regenerating figures",
        flush=True,
    )

    figures_ok = _regenerate_figures("M2")

    final = stats_per_mode.get("full", {})
    R1 = bool(final.get("R1_policy_disagreement", {}).get("pass", False))
    R2 = bool(final.get("R2_per_regime_ppo_vs_vi", {}).get("pass", False))
    R3 = bool(final.get("R3_mixed_gap", {}).get("pass", False))
    R4 = bool(final.get("R4_inferability", {}).get("pass", False))
    all_pass_full = bool(final.get("all_pass", False))

    stats = {
        "env_version": final.get("env_version", "unknown"),
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "R4": R4,
        "all_pass_full_mode": all_pass_full,
        "super_fast_duration_seconds": round(durations["super_fast"], 3),
        "fast_duration_seconds": round(durations["fast"], 3),
        "full_duration_seconds": round(durations["full"], 3),
        "pass_per_mode": pass_per_mode,
        "figures_regenerated": bool(figures_ok),
        "all_scripts_exit_zero": bool(exit_zero_all),
        "make_milestone_script_succeeded": bool(
            all_pass_full and figures_ok and exit_zero_all
        ),
        # Diagnostic details from the full-mode run.
        "full_mode_stats": {
            "R1_policy_disagreement": final.get("R1_policy_disagreement", {}),
            "R2_per_regime_ppo_vs_vi": final.get("R2_per_regime_ppo_vs_vi", {}),
            "R3_mixed_gap": final.get("R3_mixed_gap", {}),
            "R4_inferability": final.get("R4_inferability", {}),
        },
    }
    return stats


def _convergence_diagnostics(
    mean_curve: np.ndarray, final_mean: float
) -> tuple[float, int, bool]:
    """Slope over last 20 iters, plateau index, convergence.

    Converged iff |slope_last_20| < 0.5 (same threshold as M1 — a curve
    still climbing at > 0.5 return units per iteration is not plateaued)
    AND the mean curve first reaches 99% of its final value before 90%
    of the iteration budget.
    """
    iters = mean_curve.size
    if iters >= 20:
        last20 = mean_curve[-20:]
        slope = float(np.polyfit(np.arange(last20.size), last20, 1)[0])
    else:
        slope = 0.0
    if final_mean > 0:
        threshold = 0.99 * final_mean
    elif final_mean < 0:
        threshold = 1.01 * final_mean
    else:
        threshold = 0.0
    if final_mean >= 0:
        above = np.where(mean_curve >= threshold)[0]
    else:
        above = np.where(mean_curve <= threshold)[0]
    plateau_reached_at = int(above[0]) if above.size > 0 else int(iters)
    converged = bool(
        abs(slope) < 0.5 and plateau_reached_at < 0.9 * iters
    )
    return slope, plateau_reached_at, converged


def _run_m3_training(config_name: str, mode: str) -> tuple[float, int]:
    """Run one M3 training config via the full train_or_sweep path.

    Returns (elapsed_seconds, returncode). rc is 0 on success, 1 on failure.
    We shell out so stdout streams live to the user (consistent with M2).
    """
    cfg_path = CONFIG_ROOT / config_name
    cmd = [
        sys.executable,
        "-m",
        "training.train",
        "--config",
        str(cfg_path),
    ]
    if mode == "super_fast":
        cmd.append("--super-fast")
    elif mode == "fast":
        cmd.append("--fast")
    rc, elapsed = _stream_subprocess(cmd, prefix=f"[{mode}:{config_name}] ")
    return elapsed, rc


def _ci_paired(diffs: np.ndarray, n_boot: int = 10_000) -> tuple[float, float]:
    """Bootstrap CI of the mean of `diffs` (paired across seeds)."""
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size <= 1:
        return float(diffs.mean()), float(diffs.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _gap_component(
    hi_per_seed: np.ndarray, lo_per_seed: np.ndarray
) -> dict[str, float]:
    diffs = hi_per_seed - lo_per_seed
    absolute = float(diffs.mean())
    ci_lo, ci_hi = _ci_paired(diffs)
    return {
        "absolute": absolute,
        "per_seed_diff": diffs.tolist(),
        "ci": [ci_lo, ci_hi],
        "ci_width": float(ci_hi - ci_lo),
    }


def make_m3() -> dict:
    """Run the three reference-level methods on E_final, compute the gap
    decomposition, and regenerate RQ1 figures.

    Sequence:
      1. super_fast smoke for all three configs (wiring + shape).
      2. fast smoke to confirm each method shows directional signal.
      3. full run — 200 iter × 512 envs × 5 seeds.
      4. Build stats_M3_reference_levels.json from the full-mode metrics.
      5. Regenerate RQ1 figures.

    Only the full-mode results count as the M3 answer. super_fast/fast are
    wiring checks (no pass criteria).
    """
    configs = {
        "regime_agnostic_ppo": "m3_regime_agnostic.yaml",
        "oracle_ppo":          "m3_oracle.yaml",
        "belief_ppo":          "m3_belief.yaml",
    }
    experiment_dirs = {
        "regime_agnostic_ppo": "m3_regime_agnostic",
        "oracle_ppo":          "m3_oracle",
        "belief_ppo":          "m3_belief",
    }

    rough_per_mode_seconds = {
        "super_fast": 25,
        "fast":       90,
        "full":       1200,
    }

    print(
        "[make_milestone] M3 orchestration plan:\n"
        "[make_milestone]   super_fast → fast → full, 3 method configs × mode\n"
        f"[make_milestone]   rough per-mode: sf~{rough_per_mode_seconds['super_fast']}s, "
        f"fast~{rough_per_mode_seconds['fast']}s, full~{rough_per_mode_seconds['full']}s",
        flush=True,
    )

    durations: dict[str, float] = {}
    rcs_per_mode: dict[str, dict[str, int]] = {}
    t_start = time.perf_counter()

    for mi, mode in enumerate(("super_fast", "fast", "full"), start=1):
        elapsed_so_far = time.perf_counter() - t_start
        eta = sum(
            rough_per_mode_seconds[m]
            for m in ("super_fast", "fast", "full") if m not in durations
        )
        print(
            f"\n[make_milestone] ===== mode {mi}/3: {mode} "
            f"(elapsed={elapsed_so_far:.1f}s, eta_remaining≈{eta}s) =====",
            flush=True,
        )
        t_mode = time.perf_counter()
        rcs: dict[str, int] = {}
        for method, cfg_name in configs.items():
            _, rc = _run_m3_training(cfg_name, mode)
            rcs[method] = rc
        durations[mode] = time.perf_counter() - t_mode
        rcs_per_mode[mode] = rcs
        print(
            f"[make_milestone]   mode={mode} done in {durations[mode]:.1f}s "
            f"(Δ={durations[mode] - rough_per_mode_seconds[mode]:+.1f}s) "
            f"rcs={rcs}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Aggregate from full-mode metrics.
    # ------------------------------------------------------------------
    method_metrics: dict[str, dict] = {}
    for method, exp_name in experiment_dirs.items():
        mpath = RESULTS_ROOT / exp_name / "metrics.json"
        with open(mpath) as f:
            method_metrics[method] = json.load(f)

    reference_levels: dict[str, dict] = {}
    for method, m in method_metrics.items():
        mean_curve = np.asarray(m["mean_return_per_iter"], dtype=float)
        final_mean = float(m["final_return_mean"])
        slope, plateau_at, converged = _convergence_diagnostics(mean_curve, final_mean)
        reference_levels[method] = {
            "role": "floor" if method == "regime_agnostic_ppo" else "ceiling",
            "mean": final_mean,
            "ci": [float(x) for x in m["final_return_ci95"]],
            "seed_returns": [float(x) for x in m["per_seed_final_return"]],
            "converged": converged,
            "plateau_reached_at_iteration": plateau_at,
            "slope_last_20_iterations": slope,
            "experiment_dir": experiment_dirs[method],
        }

    # Per-seed alignment: every method ran with seed_base=0 and num_seeds=5.
    agn_per_seed = np.asarray(
        method_metrics["regime_agnostic_ppo"]["per_seed_final_return"], dtype=float
    )
    oracle_per_seed = np.asarray(
        method_metrics["oracle_ppo"]["per_seed_final_return"], dtype=float
    )
    belief_per_seed = np.asarray(
        method_metrics["belief_ppo"]["per_seed_final_return"], dtype=float
    )

    inference = _gap_component(oracle_per_seed, belief_per_seed)
    compromise_policy = _gap_component(belief_per_seed, agn_per_seed)
    total_gap = _gap_component(oracle_per_seed, agn_per_seed)

    total_abs = total_gap["absolute"]
    for g in (inference, compromise_policy):
        g["fraction_of_total"] = (
            g["absolute"] / total_abs if abs(total_abs) > 1e-9 else 0.0
        )

    # Ordering check: regime_agnostic ≤ belief ≤ oracle, CI-tolerant.
    m_agn = reference_levels["regime_agnostic_ppo"]["mean"]
    m_bel = reference_levels["belief_ppo"]["mean"]
    m_ora = reference_levels["oracle_ppo"]["mean"]
    strict_ordering = (m_agn <= m_bel <= m_ora)

    def _ordered_or_overlap(lo_method: dict, hi_method: dict) -> bool:
        if hi_method["mean"] >= lo_method["mean"]:
            return True
        return hi_method["ci"][1] >= lo_method["ci"][0]

    ci_tolerant_ordering = all(
        _ordered_or_overlap(reference_levels[lo], reference_levels[hi])
        for lo, hi in (
            ("regime_agnostic_ppo", "belief_ppo"),
            ("belief_ppo", "oracle_ppo"),
        )
    )
    ordering_valid = bool(strict_ordering or ci_tolerant_ordering)
    ordering_details = (
        f"agnostic={m_agn:.2f} <= belief={m_bel:.2f} <= oracle={m_ora:.2f}"
    )

    all_converged = all(reference_levels[m]["converged"] for m in reference_levels)

    stats = {
        "env_version": "e_final",
        "reference_levels": reference_levels,
        "gap_components": {
            "inference_cost": inference,
            "compromise_policy_cost": compromise_policy,
            "total_gap": total_gap["absolute"],
            "total_gap_ci": total_gap["ci"],
        },
        "ordering_valid": ordering_valid,
        "ordering_details": ordering_details,
        "all_converged": all_converged,
    }

    stats_path = RESULTS_ROOT / "milestones" / "M3" / "stats_M3_reference_levels.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    figures_ok = _regenerate_figures("M3")

    full_exit_zero = all(rc == 0 for rc in rcs_per_mode.get("full", {}).values())

    pipeline_stats = {
        "env_version": "e_final",
        "ordering_valid": ordering_valid,
        "all_converged": all_converged,
        "shared_network_cost_is_measurable": shared_network_measurable,
        "total_gap_absolute": total_gap["absolute"],
        "shared_network_cost_absolute": shared_network["absolute"],
        "inference_cost_absolute": inference["absolute"],
        "compromise_policy_cost_absolute": compromise_policy["absolute"],
        "shared_network_cost_fraction": shared_network["fraction_of_total"],
        "inference_cost_fraction": inference["fraction_of_total"],
        "compromise_policy_cost_fraction": compromise_policy["fraction_of_total"],
        "ordering_details": ordering_details,
        "super_fast_duration_seconds": round(durations["super_fast"], 3),
        "fast_duration_seconds": round(durations["fast"], 3),
        "full_duration_seconds": round(durations["full"], 3),
        "full_mode_rcs": rcs_per_mode.get("full", {}),
        "figures_regenerated": bool(figures_ok),
        "all_scripts_exit_zero": bool(full_exit_zero),
        "make_milestone_script_succeeded": bool(
            ordering_valid and all_converged and figures_ok and full_exit_zero
        ),
        "reference_level_means": {
            m: reference_levels[m]["mean"] for m in reference_levels
        },
    }
    return pipeline_stats


_MILESTONE_HANDLERS = {
    "M0": make_m0,
    "M1": make_m1,
    "M2": make_m2,
    "M3": make_m3,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.make_milestone")
    parser.add_argument("milestone", help="milestone id, e.g. M0")
    args = parser.parse_args()

    handler = _MILESTONE_HANDLERS.get(args.milestone)
    run = ScriptRun(script="make_milestone")
    summary_dir = RESULTS_ROOT / "milestones" / args.milestone
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"stats_{args.milestone}_pipeline.json"

    if handler is None:
        run.fail(reason=f"no handler for {args.milestone}", summary_path=summary_path)
        return 1

    try:
        stats = handler()
    except Exception as e:
        run.fail(reason=f"{type(e).__name__}: {e}", summary_path=summary_path)
        raise

    # Write stats_{milestone}_pipeline.json (the milestone-level artifact)
    stats_path = summary_dir / f"stats_{args.milestone}_pipeline.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    # Our own script summary (shared schema) — separate file so both schemas coexist.
    run_summary_path = summary_dir / f"summary_make_{args.milestone}.json"
    print_stats = {
        "milestone": args.milestone,
        "pass": stats["make_milestone_script_succeeded"],
    }
    for k in (
        "super_fast_duration_seconds",
        "fast_duration_seconds",
        "full_duration_seconds",
        "return_ratio",
        "converged",
        "policy_shape_matches",
        "R1",
        "R2",
        "R3",
        "R4",
        "all_pass_full_mode",
        "env_version",
        "ordering_valid",
        "all_converged",
        "shared_network_cost_is_measurable",
        "total_gap_absolute",
    ):
        if k in stats:
            print_stats[k] = stats[k]
    run.ok(
        key_stats=stats,
        summary_path=run_summary_path,
        print_stats=print_stats,
    )
    return 0 if stats["make_milestone_script_succeeded"] else 1


if __name__ == "__main__":
    sys.exit(main())
