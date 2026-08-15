"""Plotting package.

Installs one global behaviour: every figure saved anywhere in the codebase is
stamped, at ``Figure.savefig`` rather than per plotting module, because a new
module can forget a convention but cannot forget the save call.

Two stamps:

* **A timestamp**, always. A figure sitting in the thesis carries no other
  indication of when it was produced, and this project has already had
  seven-week-old charts pass for current ones. Small and grey, bottom left.
* **A DUMMY overlay**, when the process is part of a dummy programme run. A
  synthetic figure reaching the thesis unlabelled is the one failure mode of a
  dummy run that would actually cost something.

Set ``THESIS_FIG_STAMP=0`` to suppress the timestamp for a final build, where
the date belongs in the document rather than on every chart.
"""
from __future__ import annotations

import os
import time

from matplotlib.figure import Figure

_original_savefig = Figure.savefig


def _is_dummy() -> bool:
    return os.environ.get("THESIS_DUMMY") == "1"


def _stamp_enabled() -> bool:
    return os.environ.get("THESIS_FIG_STAMP", "1") != "0"


def _savefig_stamped(self, *args, **kwargs):
    # Provenance travels inside the file. It used to be recovered by comparing
    # a published figure byte-for-byte against its twin under figures_dummy/,
    # which silently reported every dummy chart as real the moment that tree
    # was regenerated: the twins changed, the comparison stopped matching, and
    # nothing said so. A PNG text chunk survives copying and regeneration.
    meta = dict(kwargs.pop("metadata", None) or {})
    meta.setdefault("thesis-source", "dummy" if _is_dummy() else "real")
    meta.setdefault("thesis-written", time.strftime("%Y-%m-%dT%H:%M:%S"))
    kwargs["metadata"] = meta

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
