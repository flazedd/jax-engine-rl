"""M6 overnight wrapper — chains training sweep, posterior probe, plot regen.

Designed for unattended overnight execution: each step runs only if the
previous step exited cleanly. On failure, prints a clear marker and
exits with the failed step's return code so the user can diagnose in
the morning by tailing the log.

Usage:
  uv run python -m scripts.m6_overnight                # full chain (~14h)

The wrapper does not commit, tag, or modify docs. Manual review is
expected before tagging `m6-passed` (mirrors the M5 protocol).
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime

STEPS: list[tuple[str, list[str]]] = [
    (
        "training sweep (full budget, both axes)",
        ["uv", "run", "python", "-m", "scripts.m6_difficulty_sweep", "--axis", "both"],
    ),
    (
        "posterior probe (24 cells × n_seeds)",
        ["uv", "run", "python", "-m", "scripts.m6_posterior_probe"],
    ),
    (
        "hypothesis tests (Holm-corrected)",
        ["uv", "run", "python", "-m", "scripts.m6_hypothesis_tests"],
    ),
    (
        "plot regeneration (3 RQ3 figures)",
        ["uv", "run", "python", "-m", "plotting.m6_plots"],
    ),
]


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> int:
    print(f"[m6_overnight] start {_stamp()}", flush=True)
    t_total = time.perf_counter()
    for i, (label, cmd) in enumerate(STEPS, start=1):
        print(
            f"\n[m6_overnight] === step {i}/{len(STEPS)}: {label} ===\n"
            f"[m6_overnight] cmd: {' '.join(cmd)}",
            flush=True,
        )
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        dt = (time.perf_counter() - t0) / 60
        if rc != 0:
            print(
                f"\n[m6_overnight] step {i} FAILED (rc={rc}) after {dt:.1f} min "
                f"at {_stamp()}; aborting chain",
                flush=True,
            )
            return rc
        print(
            f"[m6_overnight] step {i} OK in {dt:.1f} min at {_stamp()}",
            flush=True,
        )
    total_min = (time.perf_counter() - t_total) / 60
    print(
        f"\n[m6_overnight] === all steps complete in {total_min:.1f} min "
        f"at {_stamp()} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
