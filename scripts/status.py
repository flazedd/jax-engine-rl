"""One-screen view of a running programme.

  uv run python -m scripts.status

Reads the status file the driver rewrites after every stage, the tail of the
current log for in-stage progress, and the thesis provenance file, so a single
command answers: what is running, how far in, what has landed, what failed, and
which parts of the thesis are real yet.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUS = REPO / "results" / "matched_programme_status.json"
THESIS_FIGS = (
    Path(os.environ["THESIS_FIG_ROOT"])
    if os.environ.get("THESIS_FIG_ROOT") else None
)


def _fmt_min(m: float) -> str:
    m = max(0.0, float(m))
    if m < 60:
        return f"{m:.0f}m"
    if m < 60 * 24:
        return f"{m / 60:.1f}h"
    return f"{m / 1440:.1f}d"


def _current_log() -> Path | None:
    marker = REPO / "logs" / ".current"
    if not marker.exists():
        return None
    p = REPO / marker.read_text().strip()
    return p if p.exists() else None


def _in_stage_progress(log: Path | None) -> list[str]:
    """The most recent per-seed and per-iteration lines, for live progress."""
    if log is None:
        return []
    try:
        tail = subprocess.run(["tail", "-400", str(log)], capture_output=True,
                              text=True, timeout=10).stdout.splitlines()
    except Exception:
        return []
    # Only look at lines emitted since the current stage began. Stages that
    # print no per-seed progress would otherwise show the previous stage's
    # last line, which reads as live progress that is not happening.
    for i in range(len(tail) - 1, -1, -1):
        if "[programme] === " in tail[i]:
            tail = tail[i:]
            break
    out = []
    for pat, label in ((r"seed (\d+)/(\d+) .*iter (\d+)/(\d+).*?return=([-\d.]+)", "iter"),
                       (r"seed (\d+)/(\d+) done", "seed")):
        for line in reversed(tail):
            m = re.search(pat, line)
            if m:
                out.append(("  " + line.split("] ")[-1]).rstrip())
                break
    return out


def main() -> int:
    if not STATUS.exists():
        print("no programme status file; nothing has run yet")
        return 1
    s = json.loads(STATUS.read_text())

    alive = subprocess.run(["pgrep", "-f", "scripts.run_matched_programme"],
                           capture_output=True, text=True).stdout.strip()
    # Elapsed comes from the process, not the status file: the driver only
    # rewrites that file at stage boundaries, so during a long stage the header
    # would otherwise sit unchanged for hours and read as though nothing moved.
    live_elapsed = None
    if alive:
        pid = alive.split()[0]
        et = subprocess.run(["ps", "-o", "etime=", "-p", pid],
                            capture_output=True, text=True).stdout.strip()
        if et:
            try:
                days, _, rest = et.partition("-")
                if not rest:
                    days, rest = "0", days
                parts = [float(x) for x in rest.split(":")]
                secs = 0.0
                for v, m in zip(reversed(parts), (1, 60, 3600)):
                    secs += v * m
                live_elapsed = (secs + float(days) * 86400) / 60.0
            except Exception:
                live_elapsed = None
    total = len(s.get("completed", [])) + len(s.get("failed", []))

    print("=" * 66)
    print(f"  {'RUNNING' if alive else 'NOT RUNNING'}"
          f"{'  pid ' + alive.split()[0] if alive else ''}"
          f"   elapsed {_fmt_min(live_elapsed if live_elapsed is not None else s.get('elapsed_min', 0))}"
          f"   remaining ~{_fmt_min(s.get('estimated_remaining_min', 0))}")
    print("=" * 66)
    print(f"  stage        {s.get('running') or '-'}")
    for line in _in_stage_progress(_current_log()):
        print(line)
    print(f"  done         {len(s.get('completed', []))} stages"
          f"   failed {len(s.get('failed', []))}   seen {total}")

    if s.get("failed"):
        print("\n  FAILED:")
        for name in s["failed"]:
            rec = next((r for r in s.get("stages", []) if r.get("stage") == name), {})
            print(f"    {name}: {rec.get('status', '?')}")

    recent = [r for r in s.get("stages", []) if r.get("status") == "ok"][-5:]
    if recent:
        print("\n  last completed:")
        for r in recent:
            print(f"    {r['stage']:38s} {_fmt_min(r.get('minutes', 0)):>6s}")

    prov = THESIS_FIGS / "PROVENANCE.json" if THESIS_FIGS else None
    if prov and prov.exists():
        p = json.loads(prov.read_text())
        print(f"\n  thesis figures  real {p.get('n_real', 0)}"
              f"   still dummy {p.get('n_dummy', 0)}")
    tprov = THESIS_FIGS.parent / "tables" / "PROVENANCE.txt" if THESIS_FIGS else None
    if tprov and tprov.exists():
        print(f"  thesis tables   {tprov.read_text().splitlines()[0]}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
