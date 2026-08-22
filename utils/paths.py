"""Output roots, redirectable by environment variable.

Every module that reads results or writes figures resolves its roots through
here rather than hardcoding ``REPO_ROOT / "results"``. That exists for one
reason: the dummy programme run has to produce a complete set of figures and
tables without touching real results, and without overwriting the figures
already committed to the thesis.

Set by ``scripts.run_matched_programme --dummy``:

  THESIS_RESULTS_ROOT   where result JSON is read from and written to
  THESIS_PROJECT_FIGS   where the repo's own figure copies go
  THESIS_FIG_ROOT       where the thesis figure copies go

Unset, every path resolves exactly as before, so a normal run is unaffected.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def results_root() -> Path:
    return Path(os.environ.get("THESIS_RESULTS_ROOT", REPO_ROOT / "results"))


def final_dir() -> Path:
    """Backwards-compatible alias for the analysis directory.

    Kept so callers written against the old M5R/final layout keep working;
    new code should call analysis_dir() directly.
    """
    return results_root() / "analysis"


def project_fig_dir(*parts: str) -> Path:
    root = Path(os.environ.get("THESIS_PROJECT_FIGS", REPO_ROOT / "figures"))
    return root.joinpath(*parts)


def thesis_fig_dir() -> Path:
    return Path(os.environ.get(
        "THESIS_FIG_ROOT",
        REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures",
    ))


def is_dummy() -> bool:
    """True when the current process is part of a dummy programme run."""
    return os.environ.get("THESIS_DUMMY", "") == "1"


def resolve_data(path):
    """Pick the real artifact or its dummy sibling, per the run's source switch.

    Dummy artifacts live beside the real ones as ``<stem>.dummy.json`` rather
    than in a parallel tree, so one directory listing shows which stages have
    landed. A dummy run reads the ``.dummy.json`` where one exists and the real
    file where it does not; a normal run never looks at a dummy file at all.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    if not is_dummy() or p.suffix != ".json" or p.name.endswith(".dummy.json"):
        return p
    sibling = p.with_name(p.name[: -len(".json")] + ".dummy.json")
    return sibling if sibling.exists() else p


def dummy_sibling(path):
    """Where the dummy counterpart of a real artifact is written."""
    from pathlib import Path as _Path

    p = _Path(path)
    if p.suffix != ".json":
        return p.with_suffix(p.suffix + ".dummy")
    return p.with_name(p.name[: -len(".json")] + ".dummy.json")


# ---------------------------------------------------------------------------
# Thesis-aligned layout
# ---------------------------------------------------------------------------
#
# Directories are named for the part of the thesis they serve, not for the
# milestone that happened to produce them. The old M0-M7 names encoded project
# history, which is why `M5` sat beside `M5R` and `milestones/cartpole` beside
# `M5R/final`: nothing in a name told you which claim an artifact supported.
#
#   foundations/  environment validation R1-R4 (3.5), implementation
#                 validation (3.8)
#   medium/       the eight runs on e_final; feeds RQ1, RQ2 and RQ3's medium
#                 level, which is why it is named for the experiment rather
#                 than for one research question
#   sweep/        RSMM difficulty instances (RQ3)
#   cartpole/     second domain (RQ3, Appendix B)
#   analysis/     everything derived: comparison sets, probes, diagnostics,
#                 generated tables
#   _archive/     superseded data, kept for provenance comparison

def foundations_dir() -> Path:
    return results_root() / "foundations"


def medium_dir() -> Path:
    return results_root() / "medium"


def sweep_dir() -> Path:
    return results_root() / "sweep"


def cartpole_dir() -> Path:
    return results_root() / "cartpole"


def analysis_dir() -> Path:
    return results_root() / "analysis"


def archive_dir() -> Path:
    return results_root() / "_archive"


def experiment_dir(name: str) -> Path:
    """Where one training run lives, routed by its experiment name.

    Training writes through here so a run lands in the section of the tree its
    results belong to, without every caller needing to know the rule.
    """
    if name.startswith("m_cartpole") or "cartpole" in name:
        return cartpole_dir() / name
    # `_e9` is the reference environment; `_e_final` was its predecessor and is
    # kept so the superseded runs still resolve. Anything else under the m5r_
    # prefixes is a difficulty-sweep instance.
    MEDIUM_SUFFIXES = ("_e9", "_e_final")
    if name.endswith(MEDIUM_SUFFIXES) or name.startswith(("m5r_ref_", "m5r_final_")):
        if name.endswith(MEDIUM_SUFFIXES):
            return medium_dir() / name
        return sweep_dir() / name
    if name.startswith(("m2_", "m4_", "toys")):
        return foundations_dir() / name
    return results_root() / name


# --- Figure homes -----------------------------------------------------------
# Two destinations exist and only two: ``results/<RQ>`` for the charts Chapter 5
# includes, ``appendix/`` for everything the appendices carry. The repo's figure
# tree and the thesis's figure tree are mirror images, so a chart is written to
# the same relative path in both and the thesis include path names the research
# question it answers.
#
# This mapping is the single definition of that layout. A chart with no entry
# has no home, which is the point: an unregistered figure cannot appear in the
# thesis tree by accident, and `scripts.thesis_contract` can check the two trees
# against one list.
FIGURE_HOME: dict[str, str] = {
    # Chapter 5 — RQ1: performance under each conditioning architecture
    "fig_rq1_ceilings_bar.png": "results/RQ1",
    "m5r_method_ladder.png": "results/RQ1",
    # Chapter 5 — RQ2: belief formed versus belief used
    "m5r_probe_kl_per_t.png": "results/RQ2",
    "m5r_probe_acc_per_t.png": "results/RQ2",
    "m5r_action_separation.png": "results/RQ2",
    "m5r_belief_swap_separation.png": "results/RQ2",
    "m5r_posterior_vs_performance.png": "results/RQ2",
    "m5r_posterior_vs_performance_mlp.png": "results/RQ2",
    # Chapter 5 — RQ3: replication in-domain and in a second domain
    "m5r_sweep_n20.png": "results/RQ3",
    # learning curves support the time-to-threshold result of RQ1
    "m5r_learning_curves.png": "results/RQ1",
    # Appendix A — environment validity (R1–R4) and implementation validation
    "fig_M2_R1_policy_heatmap.png": "appendix",
    "fig_M2_R1_value_loss_distribution.png": "appendix",
    "fig_M2_R2_per_regime_ppo.png": "appendix",
    "fig_M2_R4_belief_ppo_gap.png": "appendix",
    "fig_M2_R4_posterior_entropy.png": "appendix",
    "factorial_toys.png": "appendix",
    # the paired per-timestep contrast supports the levels figure of RQ2
    "m5r_probe_delta_per_t.png": "appendix",
    # Appendix B — second domain
    "cartpole_difficulty_sweep_returns.png": "appendix",
    "cartpole_difficulty_sweep_scatter.png": "appendix",
    "cartpole_method_ladder.png": "appendix",
    "cartpole_persistence_sweep_returns.png": "appendix",
    "cartpole_posterior_vs_performance.png": "appendix",
    "cartpole_posterior_vs_performance_logistic_vs_mlp.png": "appendix",
    "cartpole_two_axis_scatter_grid.png": "appendix",
    "cross_env_decoupling_vs_inversion_2x2.png": "appendix",
    # Appendix C — supporting RSMM charts
    "fig_rq1_learning_curves.png": "appendix",
    "m5r_action_given_regime_inventory.png": "appendix",
    "m5r_probe_per_t.png": "appendix",
    "m5r_probe_per_t_mlp.png": "appendix",
    # Produced but not included: kept in the appendix tree so the two-folder
    # rule holds, and deliberately absent from every .tex.
    "fig_M0_dummy_learning_curve.png": "appendix",
    "fig_M1_ppo_learning_curve.png": "appendix",
    "fig_M1_ppo_policy_vs_as.png": "appendix",
    "fig_rq1_gap_fractions.png": "appendix",
    "m5r_distinguishability_sweep.png": "appendix",
    # The two-panel levels figure: superseded in the main text by the paired
    # architecture contrast, and covered per probe family by the two
    # full-width appendix figures.
    "m5r_probe_per_t_combined.png": "appendix",
}


def fig_home(name: str) -> str:
    """The registered relative directory for a figure, e.g. ``results/RQ2``."""
    try:
        return FIGURE_HOME[name]
    except KeyError:
        raise KeyError(
            f"figure {name!r} has no home; add it to utils.paths.FIGURE_HOME "
            f"under results/RQ1, results/RQ2, results/RQ3 or appendix"
        ) from None


def fig_targets(name: str) -> list[Path]:
    """Both destinations for a figure: the repo tree and the thesis tree.

    Plotters write to both, so a chart is current in the repo even when the
    thesis directory is unavailable, and the two trees never disagree about
    where a chart belongs.
    """
    home = fig_home(name)
    out = [project_fig_dir(*home.split("/")) / name,
           thesis_fig_dir().joinpath(*home.split("/")) / name]
    for p in out:
        p.parent.mkdir(parents=True, exist_ok=True)
    return out


def fig_results_dir(rq: str) -> Path:
    """Chapter 5 charts, under the research question they answer."""
    if rq not in ("RQ1", "RQ2", "RQ3"):
        raise ValueError(f"rq must be RQ1, RQ2 or RQ3, got {rq!r}")
    return project_fig_dir("results", rq)


def fig_appendix_dir() -> Path:
    return project_fig_dir("appendix")
