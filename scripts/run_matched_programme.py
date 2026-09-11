"""Run the thesis experimental programme in one pass.

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
  - **Ordered by dependency.** Validation precedes training, and training
    precedes evaluation, analysis, and figure generation.

Usage:
  uv run python -m scripts.run_matched_programme            # full programme
  uv run python -m scripts.run_matched_programme --dry-run  # list the plan
  uv run python -m scripts.run_matched_programme --phase medium
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --dummy has to be seen before any path constant is built, because the whole
# point is that a dummy run writes nowhere near the real results or the
# thesis figures. Subprocesses inherit these, so the plotting stages resolve
# the same roots without being passed anything.
if "--dummy" in sys.argv:
    os.environ["THESIS_DUMMY"] = "1"
    # The results root is deliberately NOT redirected: dummy artifacts are
    # written beside their real counterparts as <stem>.dummy.json, so one
    # listing shows which stages have landed. Figures still go to their own
    # tree, because a figure filename carries no such marker.
    os.environ.setdefault("THESIS_PROJECT_FIGS", str(REPO_ROOT / "figures_dummy"))
    os.environ.setdefault(
        "THESIS_FIG_ROOT", str(REPO_ROOT / "figures_dummy" / "thesis")
    )

from utils.paths import (  # noqa: E402
    dummy_sibling, is_dummy, project_fig_dir as project_figs, results_root,
    analysis_dir, experiment_dir, foundations_dir,
    fig_appendix_dir, project_fig_dir as _pfd, fig_home, thesis_fig_dir,
)

RESULTS = results_root()
CONFIGS = REPO_ROOT / "experiments" / "configs" / "m5r_e9"
STATUS = RESULTS / "matched_programme_status.json"
if is_dummy():
    STATUS = dummy_sibling(STATUS)

REFERENCES = ["regime_agnostic", "belief_ppo", "oracle_ppo", "stacked_obs"]
VARIANTS = ["rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet"]

# Experiment name written by each medium config, used for the skip check.
MEDIUM_EXPERIMENT = {
    "regime_agnostic": "m5r_ref_regime_agnostic_e9",
    "belief_ppo": "m5r_ref_belief_e9",
    "oracle_ppo": "m5r_ref_oracle_e9",
    "stacked_obs": "m5r_ref_stacked_obs_e9",
    **{v: f"m5r_final_{v}_e9" for v in VARIANTS},
}


SEEDS = 20
ITERATIONS = 1500


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
    if n_ckpt < SEEDS:
        return False
    # Provenance, not just budget. A run at the right seed and iteration counts
    # but from a superseded architecture passes every numeric check; requiring
    # a provenance stamp means an un-stamped directory is treated as needing a
    # rerun rather than silently adopted.
    prov = produces.parent / "provenance.json"
    if not prov.exists():
        print(f"[programme] {produces.parent.name}: no provenance stamp, "
              f"treating as stale", flush=True)
        return False
    return True


def _is_complete(stage: "Stage", by_name: dict[str, "Stage"]) -> bool:
    if is_dummy() and stage.phase != "figures":
        return dummy_sibling(stage.produces).exists()
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
        produces=experiment_dir(experiment) / "summary.json",
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
    final = analysis_dir()

    # ---- Phase 0: fairness gate ----------------------------------------
    # Every downstream comparison assumes the methods differ only in the thing
    # under study. Checking that before spending days of compute is cheap, and
    # a failure here invalidates the whole programme rather than one stage.
    stages.append(_analysis(
        "audit:config_fairness", "audit", "scripts.config_fairness_audit",
        RESULTS / "audits" / "config_fairness.json", est=2, always=True,
    ))
    gate = ["audit:config_fairness"]

    # ---- Phase 0b: foundations ------------------------------------------
    # These were treated as fixed inputs produced by earlier milestones, which
    # stopped being defensible once the conditioning redesign changed the very
    # agents they characterise. R2 and R3 are measured on trained agents, so
    # the environment validation moves with the redesign; the optimiser search
    # produces two thesis figures from training runs. Both are experiments and
    # belong in the plan.
    #
    # The implementation validation is different in kind: the thesis holds the
    # implementations fixed after validating them once, so this stage exists to
    # *check* that claim rather than to retune anything.
    stages += [
        _analysis("foundations:env_validation", "foundations",
                  "scripts.env_validation_final",
                  foundations_dir() / "env_validation" / "validation_table.json",
                  depends_on=gate, est=90),
        _analysis("foundations:impl_validation", "foundations",
                  "scripts.m5_factorial_toys",
                  foundations_dir() / "stats_M5_factorial_toys.json",
                  depends_on=gate, est=45),
    ]
    # ---- Phase 1: training ---------------------------------------------
    for key in REFERENCES:
        stages.append(_train(
            key, "medium", CONFIGS / f"{key}.yaml", MEDIUM_EXPERIMENT[key], est=80,
            depends_on=gate,
        ))
    # The reference ordering must hold before any variant is trained. It is
    # cheap, it can fail, and every normalised comparison depends on it.
    reference_training = [f"train:{k}" for k in REFERENCES]
    stages.append(_analysis(
        "gate:reference_ordering", "medium", "scripts.reference_ordering_gate",
        final / "reference_ordering_gate.json",
        depends_on=reference_training, est=1,
    ))
    for key in VARIANTS:
        stages.append(_train(
            key, "medium", CONFIGS / f"{key}.yaml", MEDIUM_EXPERIMENT[key], est=40,
            depends_on=["gate:reference_ordering"],
        ))
    medium_training = [s.name for s in stages if s.kind == "train"]
    medium_gate = medium_training + ["gate:reference_ordering"]

    # ---- Phase 2: evaluation and analyses ------------------------------
    stages += [
        # The ladder figure reads this file; without the stage a clean-room run
        # trains everything and then fails at the figure.
        _analysis("analysis:stacked_obs_sweep", "medium",
                  "scripts.m5r_stacked_obs_sweep",
                  final / "m5r_stacked_obs_sweep.json",
                  depends_on=medium_training, est=10),
        _analysis("analysis:param_counts", "medium", "scripts.m5r_param_counts",
                  final / "m5r_param_counts_run.json", depends_on=medium_training, est=2),
        _analysis("analysis:final_eval", "medium", "scripts.m5r_final_eval",
                  final / "m5r_final_run.json", depends_on=medium_gate, est=15),
        _analysis("analysis:post_training_evaluation", "medium",
                  "scripts.m5r_post_training_evaluation",
                  final / "m5r_post_training_evaluation.json",
                  depends_on=medium_training, est=30),
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
        # The third comparison set of the protocol: the probe metrics are
        # formally tested, corrected apart from returns and diagnostics.
        _analysis("analysis:belief_quality_tests", "medium",
                  "scripts.m5r_belief_quality_tests",
                  final / "m5r_belief_quality_tests.json",
                  depends_on=["analysis:posterior_probe"], est=2),
        # The method-return and time-to-threshold sets. They had no stage here,
        # so their corrected p-values could not be regenerated by this command.
        _analysis("analysis:return_tests", "medium",
                  "scripts.m5r_return_tests",
                  final / "m5r_time_to_threshold_tests.json",
                  depends_on=["analysis:final_eval",
                              "analysis:post_training_evaluation"], est=1),
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
                  depends_on=["analysis:final_eval",
                              "analysis:post_training_evaluation"], est=5),
    ]

    # ---- Phase 3: tables and figures -----------------------------------
    stages += [
        # Tables are published on the same trigger as figures: they are
        # experiment output too, and were the last thing still typed by hand.
        # Keyed on a table make_tables still writes: probe_quality.tex was
        # replaced by one table per probe metric and no longer exists.
        _analysis("figures:tables", "figures", "scripts.make_tables",
                  thesis_fig_dir().parent / "tables" / "probe_kl.tex",
                  depends_on=["analysis:post_training_evaluation",
                              "analysis:posterior_probe"],
                  est=1, always=True),
        _analysis("figures:main", "figures", "plotting.m5r_plots",
                  _repo_fig("m5r_method_ladder.png"),
                  depends_on=["analysis:post_training_evaluation"], est=5),
        _analysis("figures:action_heatmap", "figures",
                  "plotting.m5r_action_inventory_heatmap",
                  _repo_fig("m5r_action_given_regime_inventory.png"),
                  depends_on=["analysis:action_distributions"], est=2),
        # Figures the thesis includes that had no stage: without these a run
        # produces results the thesis cannot render.
        # No figures:m2 stage. Environment validation renders the five M2
        # charts itself, for every env it checks, and copies the reference
        # env's set into the figure tree. A second stage re-rendering them from
        # a separate stats file duplicated the work and read a path nothing
        # writes any more.
        # Keyed on the learning curves: the bar chart the thesis dropped is now
        # written to the repo tree only, so it has no entry in the registry.
        _analysis("figures:reference_levels", "figures", "plotting.reference_levels",
                  _repo_fig("fig_rq1_learning_curves.png"),
                  depends_on=medium_training, est=2),
        _analysis("figures:factorial_toys", "figures", "plotting.m4_plots",
                  _repo_fig("factorial_toys.png"),
                  depends_on=medium_training, est=2),
    ]
    return stages



def _real_counterpart(produces: Path) -> Path:
    """The same artifact under the real results root, used as a schema template."""
    real_root = REPO_ROOT / "results"
    try:
        return real_root / produces.relative_to(RESULTS)
    except ValueError:
        return produces


def _write_dummy(stage: "Stage") -> bool:
    """Write a plausible synthetic artifact for one stage.

    Cloning a real artifact is preferred, because then the figure rendered from
    it has the shape the finished thesis will have. Stages that never ran get a
    synthesiser instead.
    """
    from scripts import dummy_data as dd

    produces = dummy_sibling(stage.produces)
    produces.parent.mkdir(parents=True, exist_ok=True)

    if stage.kind == "train":
        experiment = produces.parent.name
        produces.write_text(json.dumps(
            dd.synth_training_summary(experiment), indent=2))
        (produces.parent / "metrics.dummy.json").write_text(json.dumps(
            dd.synth_training_metrics(experiment), indent=2))
        return True

    synth = dd.SYNTHESISERS.get(stage.name)
    if synth is not None:
        produces.write_text(json.dumps(synth(), indent=2))
        return True

    template = stage.produces
    if template.exists():
        payload = dd.clone_with_jitter(template, seed=abs(hash(stage.name)) % 2**31)
        if payload is not None:
            produces.write_text(json.dumps(payload, indent=2))
            return True

    # No template and no synthesiser: say so rather than writing something the
    # plotting code will silently misread.
    print(f"  [{stage.name}] no dummy template at {template}", flush=True)
    produces.write_text(json.dumps(
        {"dummy": True, "stage": stage.name,
         "note": "no template available; structure unknown"}, indent=2))
    return True



def _repo_fig(name: str) -> Path:
    """A figure's path in the repo tree, from the one registry."""
    return _pfd(*fig_home(name).split("/")) / name


def _thesis_referenced_figures(thesis_root: Path) -> dict[str, str]:
    """Figure basename -> path relative to the thesis figure root.

    The include paths in the .tex are what define the thesis layout, so
    publication reads them rather than assuming a flat directory.

    Publication is driven by this rather than by whatever a plotter happens to
    emit: several plotters write charts the thesis deliberately leaves out, and
    copying those in would re-add files that were removed on purpose.
    """
    names: dict[str, str] = {}
    # The .tex lives with the document, not with the figure output root: a
    # dummy run redirects the figure root into figures_dummy/, and deriving
    # sections from it found nothing, so publication silently copied nothing.
    sections = thesis_root.parent / "sections"
    if not sections.exists():
        return names
    for tex in sections.glob("*.tex"):
        for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}",
                             tex.read_text()):
            rel = m.group(1)
            names[rel.rsplit("/", 1)[-1]] = rel
    return names


def _publish_figures(since: float = 0.0) -> int:
    """Copy figures produced by *this run* into the thesis.

    Some plotters write to the thesis directory themselves and some only to the
    repo's figure tree, so relying on the plotters alone leaves half the
    chapter stale.

    `since` is the run's start time, and it matters: without it, publishing
    picks up figures left on disk by earlier programmes. That is how a set of
    pre-rerun charts once replaced the dummy placeholders and quietly presented
    itself as current. Only files this run wrote are published.
    """
    configured = os.environ.get("THESIS_FIG_ROOT")
    if not configured:
        return 0
    thesis = Path(configured)
    if not thesis.exists():
        return 0
    wanted = _thesis_referenced_figures(thesis)
    produced: dict[str, Path] = {}
    # Through the override, not REPO_ROOT/figures: a dummy run must publish
    # from the dummy figure tree, which is what lets it prove this step at all.
    for src in project_figs().rglob("*.png"):
        if src.name not in wanted:
            continue
        if src.stat().st_mtime < since:
            continue  # left by an earlier run, not produced by this one
        best = produced.get(src.name)
        if best is None or src.stat().st_mtime > best.stat().st_mtime:
            produced[src.name] = src
    published = 0
    for name, src in produced.items():
        dst = thesis / wanted[name]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        dst.write_bytes(src.read_bytes())
        published += 1
    return published


def _figure_source(png: Path) -> str:
    """Whether a published figure came from the real programme or a dummy run.

    Read from the PNG's own text chunk, written at savefig. The previous test
    compared the file against its twin under figures_dummy/, which reported
    every dummy chart as real as soon as that tree was regenerated. A figure
    written before the stamp existed reports "unknown" rather than guessing.
    """
    try:
        from PIL import Image
        with Image.open(png) as im:
            return im.text.get("thesis-source", "unknown")
    except Exception:
        return "unknown"


def _write_provenance() -> None:
    """Record which thesis figures are real yet and which are still dummy.

    A real run publishes figures into the thesis as their inputs land, so for
    most of the run the chapter is a mix. The watermark shows that on the page;
    this file makes it checkable without opening every image.
    """
    configured = os.environ.get("THESIS_FIG_ROOT")
    if not configured:
        return
    thesis = Path(configured)
    if not thesis.exists():
        return
    entries = {}
    for fig in sorted(thesis.rglob("*.png")):
        entries[str(fig.relative_to(thesis))] = {
            "source": _figure_source(fig),
            "published_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(fig.stat().st_mtime)),
        }
    n_real = sum(1 for e in entries.values() if e["source"] == "real")
    (thesis / "PROVENANCE.json").write_text(json.dumps({
        "note": "which figures in this directory come from the real programme "
                "and which are still synthetic placeholders",
        "n_real": n_real,
        "n_dummy": sum(1 for e in entries.values() if e["source"] == "dummy"),
        "n_unknown": sum(1 for e in entries.values() if e["source"] == "unknown"),
        "figures": entries,
    }, indent=2))


def _run_ready_figures(plan, by_name, done, failed, args, run_stage,
                       since: float) -> None:
    """Run any figure stage whose inputs have just landed.

    Figures sit last in the plan, so waiting for their turn would mean the
    thesis gets nothing until the whole programme finishes. Running them as
    soon as their dependencies complete publishes each chart the moment its
    data exists, and leaves the dummy placeholder in place for everything not
    yet produced.
    """
    for stage in plan:
        if stage.phase != "figures" or stage.name in done or stage.name in failed:
            continue
        if any(d in failed for d in stage.depends_on):
            continue
        if not all(d in done or _is_complete(by_name[d], by_name)
                   for d in stage.depends_on if d in by_name):
            continue
        if _is_complete(stage, by_name) and stage.name not in args.force:
            continue
        print(f"[programme] publishing {stage.name} (inputs ready)", flush=True)
        run_stage(stage)
        n = _publish_figures(since=since)
        if n:
            print(f"[programme] published {n} figure(s) to the thesis", flush=True)
        _write_provenance()


def _validate_output(stage: "Stage") -> str | None:
    """Cheap sanity check on a stage's artifact, run the moment it lands.

    A malformed artifact that is only noticed at the figure stage costs the
    whole run, because by then the training that produced it is hours behind.
    Catching it here means one stage is rerun, not the programme.
    """
    produces = stage.produces
    if not produces.exists():
        return "no output written"
    if produces.suffix != ".json":
        return None if produces.stat().st_size > 0 else "empty file"
    try:
        payload = json.loads(produces.read_text())
    except Exception as exc:
        return f"unparseable JSON ({exc})"
    if payload in ({}, [], None):
        return "empty JSON"
    if isinstance(payload, dict) and payload.get("status") == "FAIL":
        return f"stage reported FAIL: {payload.get('error')}"
    return None


def _rel(p: Path) -> str:
    """Repo-relative when possible; a redirected root may sit outside it."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


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
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run without first proving the chain on dummy data "
                         "(not recommended: the point of preflight is that a "
                         "downstream bug is found in seconds, not after days "
                         "of training)")
    ap.add_argument("--dummy", action="store_true",
                    help="write plausible synthetic results for every stage and "
                         "render the real figures from them, under results_dummy/ "
                         "and figures_dummy/. Produces the full artifact set in "
                         "minutes so the results chapter can be laid out while "
                         "the real programme trains.")
    args = ap.parse_args()

    if is_dummy():
        from scripts import dummy_data as dd
        n = dd.mirror_inputs(REPO_ROOT / "results")
        print(f"[programme] dummy mode: wrote {n} .dummy.json siblings under "
              f"results/, figures under "
              f"{os.environ['THESIS_PROJECT_FIGS'].split('/')[-1]}/", flush=True)

    if not is_dummy() and not args.dry_run and not args.skip_preflight:
        print("[programme] preflight: proving the chain on dummy data", flush=True)
        pre = subprocess.run(
            [sys.executable, "-u", "-m", "scripts.run_matched_programme", "--dummy"],
            cwd=REPO_ROOT,
        )
        if pre.returncode != 0:
            print("[programme] ABORT: the dummy chain fails, so the real run "
                  "would train for days and then fail the same way. Fix it, or "
                  "pass --skip-preflight to override.", flush=True)
            return 1
        contract = subprocess.run(
            [sys.executable, "-u", "-m", "scripts.thesis_contract"], cwd=REPO_ROOT,
        )
        print(f"[programme] preflight passed (contract exit={contract.returncode})",
              flush=True)

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
                  f"~{_fmt(s.est_min):>7s}  -> {_rel(s.produces)}")
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

    def run_stage(stage: "Stage") -> None:
        started = time.time()
        if is_dummy() and stage.phase != "figures":
            ok = _write_dummy(stage)
            took = (time.time() - started) / 60
            status = "dummy" if ok else "dummy_failed"
            (done if ok else failed).add(stage.name)
            records.append({
                "stage": stage.name, "phase": stage.phase, "status": status,
                "minutes": round(took, 2), "produces": str(stage.produces),
            })
            _write_status(None)
            print(f"[programme] {stage.name} {status}", flush=True)
            return

        proc = subprocess.Popen(
            stage.cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"  [{stage.name}] {line.rstrip()}", flush=True)
        code = proc.wait()
        took = (time.time() - started) / 60

        problem = _validate_output(stage) if code == 0 else "non-zero exit"
        if code == 0 and problem is None:
            done.add(stage.name)
            status = "ok"
        elif code == 0:
            failed.add(stage.name)
            status = f"invalid_output({problem})"
            print(f"[programme] {stage.name} {status}", flush=True)
        else:
            failed.add(stage.name)
            status = f"failed(exit={code}, output_written={stage.produces.exists()})"
            print(f"[programme] {stage.name} {status}", flush=True)
        records.append({
            "stage": stage.name, "phase": stage.phase, "status": status,
            "minutes": round(took, 1), "produces": str(stage.produces),
        })
        _write_status(None)
        print(f"[programme] {stage.name} {status} in {_fmt(took)}", flush=True)

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

        run_stage(stage)
        if not is_dummy():
            # Publish whatever the stage just unlocked, so the thesis fills in
            # while the rest of the programme is still running.
            _run_ready_figures(plan, by_name, done, failed, args, run_stage,
                               since=t0)

    # Both modes: in a dummy run every root points into figures_dummy/, so this
    # exercises publication rather than skipping it. Stages that finish last
    # otherwise never reach the thesis tree at all.
    _publish_figures(since=t0)
    _write_provenance()
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
