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
    """The M5R/final directory most analysis output lands in."""
    return results_root() / "M5R" / "final"


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
