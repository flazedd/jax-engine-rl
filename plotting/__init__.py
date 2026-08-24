"""Plotting package.

Installs one global behaviour: every figure saved anywhere in the codebase is
stamped, at ``Figure.savefig`` rather than per plotting module, because a new
module can forget a convention but cannot forget the save call.

Provenance travels with every figure, but not on its face. The generation time
is written into the PNG metadata as ``thesis-written`` and appended to a
``PROVENANCE.tsv`` at the root of each figure tree, so a stale chart is still
detectable without printing a date onto a thesis figure. This project has
already had seven-week-old charts pass for current ones, which is why the
record is kept at all.

Two overlays remain available:

* **A timestamp**, off by default. Set ``THESIS_FIG_STAMP=1`` while iterating
  to see at a glance which charts a rerun actually refreshed.
* **A DUMMY overlay**, whenever the process is part of a dummy programme run,
  and not suppressible. A synthetic figure reaching the thesis unlabelled is
  the one failure mode of a dummy run that would actually cost something.
"""
from __future__ import annotations

import os
import time

from matplotlib.figure import Figure

_original_savefig = Figure.savefig


def _is_dummy() -> bool:
    return os.environ.get("THESIS_DUMMY") == "1"


def _stamp_enabled() -> bool:
    # Off by default: the date belongs in the manifest and in the file's own
    # metadata, not rendered onto a figure that goes into the document.
    return os.environ.get("THESIS_FIG_STAMP", "0") == "1"


def record_figure_provenance(target, written: str, source: str) -> None:
    """Append this figure to the PROVENANCE.tsv of the tree it was written to.

    Keyed by the path relative to that tree's `figures/` root, so the same
    chart written to the repo tree and the thesis tree is one row in each and
    a rerun replaces its own row rather than accumulating history.
    """
    from pathlib import Path

    if not isinstance(target, (str, os.PathLike)):
        return
    path = Path(target)
    root = next((p for p in path.parents if p.name == "figures"), None)
    if root is None:
        return
    try:
        key = str(path.relative_to(root))
        manifest = root / "PROVENANCE.tsv"
        rows = {}
        if manifest.exists():
            for line in manifest.read_text().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) == 3:
                    rows[parts[0]] = parts
        rows[key] = [key, written, source]
        body = "\n".join("\t".join(r) for r in sorted(rows.values()))
        manifest.write_text("figure\twritten\tsource\n" + body + "\n")
    except Exception:
        # Provenance must never be the reason a figure fails to save.
        pass


def _savefig_stamped(self, *args, **kwargs):
    # Provenance travels inside the file. It used to be recovered by comparing
    # a published figure byte-for-byte against its twin under figures_dummy/,
    # which silently reported every dummy chart as real the moment that tree
    # was regenerated: the twins changed, the comparison stopped matching, and
    # nothing said so. A PNG text chunk survives copying and regeneration.
    meta = dict(kwargs.pop("metadata", None) or {})
    source = "dummy" if _is_dummy() else "real"
    written = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta.setdefault("thesis-source", source)
    meta.setdefault("thesis-written", written)
    kwargs["metadata"] = meta
    if args:
        record_figure_provenance(args[0], written, source)

    if not getattr(self, "_thesis_stamped", False):
        if _stamp_enabled():
            self.text(
                0.005, 0.005, time.strftime("%Y-%m-%d %H:%M"),
                transform=self.transFigure, fontsize=6, color="#9a9a9a",
                ha="left", va="bottom", zorder=1000,
            )
        if _is_dummy():
            self.text(
                0.5, 0.5, "DUMMY DATA", transform=self.transFigure,
                fontsize=64, color="#d00000", alpha=0.16, ha="center",
                va="center", rotation=30, zorder=1000,
            )
            self.text(
                0.995, 0.005, "synthetic values, not a result",
                transform=self.transFigure, fontsize=7, color="#d00000",
                alpha=0.75, ha="right", va="bottom", zorder=1000,
            )
        self._thesis_stamped = True
    return _original_savefig(self, *args, **kwargs)


Figure.savefig = _savefig_stamped
