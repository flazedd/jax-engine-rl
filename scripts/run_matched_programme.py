"""Run the whole matched-input experimental programme in one pass.

Every method now receives the same per-step tuple and the two conditioning
variants of each method are matched on encoder capacity, so every trained
artifact downstream of the old wiring has to be regenerated. This driver runs
that programme end to end, in dependency order, over days if needed.

Design notes:
  - **Resumable.** Each stage declares the file it produces. A stage whose
    output already exists is skipped, so an interrupted run picks up where it
    stopped and a stage can be forced by deleting its output.
  - **Incremental.** Results land as each stage finishes and the status file is
    rewritten after every stage, so the programme can be reviewed while it is
    still running.
  - **Fault tolerant.** A failing stage is recorded and the programme continues
    with stages that do not depend on it; dependants are skipped rather than run
    against missing inputs.
  - **Ordered by value.** The medium environment answers RQ1 and RQ2 and comes
    first, so the central claim can be checked before the replication phases
    consume the bulk of the compute.

Usage:
  uv run python -m scripts.run_matched_programme            # full programme
  uv run python -m scripts.run_matched_programme --dry-run  # list the plan
  uv run python -m scripts.run_matched_programme --phase medium
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
CONFIGS = REPO_ROOT / "experiments" / "configs" / "m5r_matched"
STATUS = RESULTS / "matched_programme_status.json"

REFERENCES = ["regime_agnostic", "belief_ppo", "oracle_ppo", "stacked_obs"]
VARIANTS = ["rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet"]

# Experiment name written by each medium config, used for the skip check.
MEDIUM_EXPERIMENT = {
    "regime_agnostic": "m5r_matched_regime_agnostic",
    "belief_ppo": "m5r_matched_belief",
    "oracle_ppo": "m5r_matched_oracle",
    "stacked_obs": "m5r_matched_stacked_obs",
    **{v: f"m5r_matched_{v}" for v in VARIANTS},
}


SEEDS = 20
ITERATIONS = 300


@dataclass
class Stage:
    name: str
    phase: str
    cmd: list[str]
    produces: Path
    depends_on: list[str] = field(default_factory=list)
    # Rough wall-clock in minutes, used only for the remaining-time estimate.
    est_min: float = 40.0
    # Training stages verify the budget actually used; analysis stages verify
    # their output is newer than everything they depend on.
    kind: str = "analysis"
    # Cheap checks that gate the rest of the programme and are never skipped,
    # so an edited config cannot be waved through by a stale pass on disk.
    always: bool = False


def _training_is_complete(produces: Path) -> bool:
    """A training output counts only if it ran the full budget.

    Existence is not enough: a `--super-fast` smoke run writes the same
    `summary.json`, and treating that as done would silently skip the real run.
    """
    if not produces.exists():
        return False
    try:
        summary = json.loads(produces.read_text())
    except Exception:
        return False
    if summary.get("status") != "OK":
        return False
    # Count checkpoints rather than trusting config.json: `--super-fast`
    # overrides the seed and iteration counts at runtime while the saved config
    # still shows the full budget, so a smoke artifact would otherwise pass.
    n_ckpt = len(list(produces.parent.glob("checkpoint_seed_*.pkl")))
    return n_ckpt >= SEEDS


def _is_complete(stage: "Stage", by_name: dict[str, "Stage"]) -> bool:
    """Whether a stage can be skipped.

    Training: the full budget must already be on disk. Analyses: the output must
    exist *and* be newer than every input it derives from, so artifacts left by
    the previous architecture are regenerated rather than mistaken for current.
    """
    if stage.always:
        return False
    if stage.kind == "train":
        return _training_is_complete(stage.produces)
    if not stage.produces.exists():
        return False
    own = stage.produces.stat().st_mtime
    for dep_name in stage.depends_on:
        dep = by_name.get(dep_name)
        if dep is None or not dep.produces.exists():
            return False
        if dep.produces.stat().st_mtime > own:
            return False
    return True


def _train(key: str, phase: str, cfg: Path, experiment: str, est: float,
           depends_on: list[str] | None = None) -> Stage:
    return Stage(
        name=f"train:{key}",
        phase=phase,
        cmd=[sys.executable, "-u", "-m", "training.train", "--config", str(cfg)],
        produces=RESULTS / experiment / "summary.json",
        depends_on=depends_on or [],
        est_min=est,
        kind="train",
    )


def _analysis(
    name: str, phase: str, module: str, produces: Path,
    args: list[str] | None = None, depends_on: list[str] | None = None,
    est: float = 10.0, always: bool = False,
) -> Stage:
    return Stage(
        name=name, phase=phase,
        cmd=[sys.executable, "-u", "-m", module, *(args or [])],
        produces=produces,
        depends_on=depends_on or [],
        est_min=est,
        always=always,
    )


def build_plan() -> list[Stage]:
    stages: list[Stage] = []
    final = RESULTS / "M5R" / "final"

    # ---- Phase 0: fairness gate ----------------------------------------
    # Every downstream comparison assumes the methods differ only in the thing
    # under study. Checking that before spending days of compute is cheap, and
    # a failure here invalidates the whole programme rather than one stage.
    stages.append(_analysis(
        "audit:config_fairness", "audit", "scripts.config_fairness_audit",
        RESULTS / "audits" / "config_fairness.json", est=2, always=True,
    ))
    gate = ["audit:config_fairness"]

    # ---- Phase 1: medium environment, training -------------------------
    for key in REFERENCES:
        stages.append(_train(
            key, "medium", CONFIGS / f"{key}.yaml", MEDIUM_EXPERIMENT[key], est=80,
            depends_on=gate,
        ))
    for key in VARIANTS:
        stages.append(_train(
            key, "medium", CONFIGS / f"{key}.yaml", MEDIUM_EXPERIMENT[key], est=40,
            depends_on=gate,
        ))
    medium_training = [s.name for s in stages if s.kind == "train"]

    # ---- Phase 2: medium environment, analyses -------------------------
    stages += [
        _analysis("analysis:param_counts", "medium", "scripts.m5r_param_counts",
                  final / "m5r_param_counts_run.json", depends_on=medium_training, est=2),
        _analysis("analysis:final_eval", "medium", "scripts.m5r_final_eval",
                  final / "m5r_final_run.json", depends_on=medium_training, est=15),
        # 500 rollouts x 128 steps = the 64,000-timestep probe buffer the
        # evaluation protocol specifies; the script default is smaller.
        _analysis("analysis:posterior_probe", "medium", "scripts.m5r_posterior_probe",
                  final / "m5r_posterior_vs_performance.json",
                  args=["--n-rollouts", "500"],
                  depends_on=medium_training, est=30),
        _analysis("analysis:posterior_probe_mlp", "medium", "scripts.m5r_posterior_probe",
                  final / "m5r_posterior_vs_performance_mlp.json",
                  args=["--n-rollouts", "500", "--classifier", "mlp"],
                  depends_on=medium_training, est=30),
        _analysis("analysis:action_distributions", "medium",
                  "scripts.m5r_action_distributions",
                  final / "m5r_action_distributions.json",
                  depends_on=medium_training, est=10),
        _analysis("analysis:belief_swap", "medium", "scripts.m5r_belief_swap",
                  final / "m5r_belief_swap.json", depends_on=medium_training, est=10),
        _analysis("analysis:belief_swap_history", "medium", "scripts.m5r_belief_swap",
                  final / "m5r_belief_swap_with_history.json",
                  args=["--swap-history"], depends_on=medium_training, est=10),
        _analysis("analysis:belief_swap_history_only", "medium", "scripts.m5r_belief_swap",
                  final / "m5r_belief_swap_history_only.json",
                  args=["--swap-history", "--hold-belief-fixed"],
                  depends_on=medium_training, est=10),
        _analysis("analysis:diagnostic_tests", "medium", "scripts.m5r_diagnostic_tests",
                  final / "m5r_diagnostic_tests.json",
                  depends_on=["analysis:action_distributions", "analysis:belief_swap",
                              "analysis:belief_swap_history",
                              "analysis:belief_swap_history_only"], est=2),
        _analysis("analysis:hypothesis_tests", "medium", "scripts.m5r_hypothesis_tests",
                  final / "m5r_hypothesis_tests.json",
                  depends_on=["analysis:final_eval"], est=5),
        # Both heads, explicitly. The comparison this stage exists to make is
        # concat-head against hypernet-head on one frozen encoder, so running
        # the script once leaves the claim with a single arm. `--head` also
        # defaults to concat, which is why omitting it produced no hypernet
        # run at all. The declared outputs are the files the script actually
        # writes, and the ones `scripts.plot_transplant_figure` reads.
        _analysis("analysis:frozen_encoder_rl2_concat", "medium",
                  "scripts.m5r_frozen_encoder_transplant",
                  final / "m5r_transplant_e_final_concat.json",
                  args=["--head", "concat"],
                  depends_on=medium_training, est=60),
        _analysis("analysis:frozen_encoder_rl2_hypernet", "medium",
                  "scripts.m5r_frozen_encoder_transplant",
                  final / "m5r_transplant_e_final_hypernet.json",
                  args=["--head", "hypernet"],
                  depends_on=medium_training, est=60),
        # The transplant result is reported for both methods and both head
        # forms, so VariBAD needs one run per head.
        _analysis("analysis:frozen_encoder_vb_concat", "medium",
                  "scripts.m5r_frozen_encoder_transplant_vb",
                  final / "m5r_transplant_vb_concat.json",
                  args=["--head", "concat"],
                  depends_on=medium_training, est=60),
        _analysis("analysis:frozen_encoder_vb_hypernet", "medium",
                  "scripts.m5r_frozen_encoder_transplant_vb",
                  final / "m5r_transplant_vb_hypernet.json",
                  args=["--head", "hypernet"],
                  depends_on=medium_training, est=60),
        _analysis("analysis:bc_belief_usage", "medium", "scripts.m5r_bc_belief_usage",
                  final / "m5r_bc_belief_usage.json",
                  depends_on=medium_training, est=30),
    ]

    # ---- Phase 3: difficulty sweep -------------------------------------
    # The existing driver retrains every cell per level and merges into the
    # structure the probe and sweep plots expect.
    stages.append(_analysis(
        "sweep:redesign_n20", "sweep", "scripts.sweep_redesign_n20",
        final / "per_cell_env.json",
        depends_on=medium_training, est=13 * 60,
    ))

    # ---- Phase 4: second domain ----------------------------------------
    # Both difficulty axes, three levels each. The axis is not a default the
    # driver can leave implicit: omitting it ran the asymmetry axis alone and
    # the persistence half of the external-validity claim went unproduced.
    cartpole_root = RESULTS / "milestones" / "cartpole"
    # Asymmetry runs first and trains all 21 of its cells; persistence reuses
    # the shared medium level and trains only 14, so its estimate is lower.
    # Both budgets are 300 iterations x 20 seeds, matching the RSMM programme;
    # at the measured ~0.4 s per iteration a cell costs roughly 45 minutes.
    cartpole_est = {"asymmetry": 21 * 45, "persistence": 14 * 45}
    for axis in ("asymmetry", "persistence"):
        suffix = "" if axis == "asymmetry" else f"_{axis}"
        stages.append(_analysis(
            f"cartpole:sweep:{axis}", "cartpole", "scripts.cartpole_difficulty_sweep",
            cartpole_root / f"stats_cartpole_sweep{suffix}.json",
            args=["--axis", axis],
            depends_on=medium_training, est=cartpole_est[axis],
        ))
        # `--levels` also has to be explicit: both scripts default to the
        # medium level alone, which would analyse one of the three levels the
        # sweep just trained.
        levels = ["--levels", "easy", "medium", "hard"]
        stages.append(_analysis(
            f"cartpole:probe:{axis}", "cartpole", "scripts.cartpole_posterior_probe",
            cartpole_root
            / f"stats_cartpole_posterior_vs_performance_sweep{suffix}.json",
            args=["--axis", axis, *levels, "--n-rollouts", "500"],
            depends_on=[f"cartpole:sweep:{axis}"], est=40,
        ))
        stages.append(_analysis(
            f"cartpole:tests:{axis}", "cartpole", "scripts.cartpole_hypothesis_tests",
            cartpole_root / f"stats_cartpole_hypothesis_tests_sweep{suffix}.json",
            args=["--axis", axis, *levels],
            depends_on=[f"cartpole:sweep:{axis}"], est=5,
        ))

    # ---- Phase 5: figures ----------------------------------------------
    stages += [
        _analysis("figures:main", "figures", "plotting.m5r_plots",
                  REPO_ROOT / "figures" / "milestones" / "M5R" / "m5r_method_ladder.png",
                  depends_on=["analysis:final_eval"], est=5),
        _analysis("figures:action_heatmap", "figures",
                  "plotting.m5r_action_inventory_heatmap",
                  REPO_ROOT / "figures" / "milestones" / "M5R"
                  / "m5r_action_given_regime_inventory.png",
                  depends_on=["analysis:action_distributions"], est=2),
        _analysis("figures:transplant", "figures", "scripts.plot_transplant_figure",
                  REPO_ROOT / "figures" / "milestones" / "M5R" / "m5r_transplant.png",
                  depends_on=["analysis:frozen_encoder_rl2_concat",
                              "analysis:frozen_encoder_rl2_hypernet",
                              "analysis:frozen_encoder_vb_concat",
                              "analysis:frozen_encoder_vb_hypernet"], est=2),
    ]
    return stages


def _fmt(minutes: float) -> str:
    minutes = max(0.0, minutes)
    h, m = divmod(int(minutes), 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h:02d}h"
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.run_matched_programme")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and what would be skipped, run nothing")
    ap.add_argument("--phase", action="append",
                    help="restrict to these phases; repeatable")
    ap.add_argument("--force", action="append", default=[],
                    help="stage names to rerun even if their output exists")
    args = ap.parse_args()

    plan = build_plan()
    if args.phase:
        plan = [s for s in plan if s.phase in args.phase]

    done: set[str] = set()
    failed: set[str] = set()
    records: list[dict] = []
    t0 = time.time()

    by_name = {st.name: st for st in plan}

    if args.dry_run:
        total = 0.0
        for s in plan:
            skip = _is_complete(s, by_name) and s.name not in args.force
            total += 0.0 if skip else s.est_min
            print(f"  [{'skip' if skip else 'run '}] {s.phase:9s} {s.name:38s} "
                  f"~{_fmt(s.est_min):>7s}  -> {s.produces.relative_to(REPO_ROOT)}")
        print(f"\nestimated remaining wall-clock: {_fmt(total)}")
        return 0

    def _write_status(current: str | None) -> None:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        remaining = sum(
            s.est_min for s in plan
            if s.name not in done and s.name not in failed and s.name != current
        )
        STATUS.write_text(json.dumps({
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "estimated_remaining_min": round(remaining, 1),
            "completed": sorted(done),
            "failed": sorted(failed),
            "running": current,
            "stages": records,
        }, indent=2))

    for i, stage in enumerate(plan, start=1):
        blocked = [d for d in stage.depends_on if d in failed]
        if blocked:
            print(f"\n[programme] SKIP {stage.name}: dependency failed ({blocked[0]})",
                  flush=True)
            failed.add(stage.name)
            records.append({"stage": stage.name, "status": "skipped_dependency"})
            _write_status(None)
            continue

        if _is_complete(stage, by_name) and stage.name not in args.force:
            print(f"[programme] have {stage.name} already, skipping", flush=True)
            done.add(stage.name)
            records.append({"stage": stage.name, "status": "cached"})
            _write_status(None)
            continue

        remaining = sum(
            s.est_min for s in plan[i - 1:]
            if not (_is_complete(s, by_name) and s.name not in args.force)
        )
        print(
            f"\n[programme] === {i}/{len(plan)} {stage.name} "
            f"({stage.phase}) | elapsed {_fmt((time.time() - t0) / 60)} "
            f"| remaining ~{_fmt(remaining)} ===",
            flush=True,
        )
        _write_status(stage.name)

        started = time.time()
        proc = subprocess.Popen(
            stage.cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"  [{stage.name}] {line.rstrip()}", flush=True)
        code = proc.wait()
        took = (time.time() - started) / 60

        if code == 0 and stage.produces.exists():
            done.add(stage.name)
            status = "ok"
        else:
            failed.add(stage.name)
            status = f"failed(exit={code}, output_written={stage.produces.exists()})"
            print(f"[programme] {stage.name} {status}", flush=True)
        records.append({
            "stage": stage.name, "phase": stage.phase, "status": status,
            "minutes": round(took, 1),
            "produces": str(stage.produces.relative_to(REPO_ROOT)),
        })
        _write_status(None)
        print(f"[programme] {stage.name} {status} in {_fmt(took)}", flush=True)

    _write_status(None)
    print(
        f"\n[programme] finished | {len(done)} ok, {len(failed)} failed "
        f"| total {_fmt((time.time() - t0) / 60)} | status: "
        f"{STATUS.relative_to(REPO_ROOT)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
