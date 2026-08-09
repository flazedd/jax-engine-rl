"""Plotting package.

Installs one global behaviour: when the process is part of a dummy programme
run, every figure saved anywhere in the codebase is stamped. A synthetic figure
that reaches the thesis unlabelled is the one failure mode of a dummy run that
would actually cost something, so the stamp is applied at ``Figure.savefig``
rather than per plotting module, where a new module could forget it.
"""
from __future__ import annotations

import os

if os.environ.get("THESIS_DUMMY") == "1":  # pragma: no cover - display only
    from matplotlib.figure import Figure

    _original_savefig = Figure.savefig

    def _savefig_stamped(self, *args, **kwargs):
        if not getattr(self, "_dummy_stamped", False):
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
            self._dummy_stamped = True
        return _original_savefig(self, *args, **kwargs)

    Figure.savefig = _savefig_stamped
