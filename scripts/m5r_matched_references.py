"""Train the four reference methods on the matched per-step tuple.

Runs the `m5r_matched/` configs in sequence and streams each training line
through with an overall progress prefix, so the wall-clock remaining across
the whole sequence is visible rather than only the remaining time within the
current method.

Usage:
  uv run python -m scripts.m5r_matched_references
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "experiments" / "configs" / "m5r_matched"

RUNS = [
    ("regime-agnostic", "regime_agnostic.yaml"),
    ("Belief-PPO", "belief_ppo.yaml"),
    ("Oracle-PPO", "oracle_ppo.yaml"),
    ("stacked-obs", "stacked_obs.yaml"),
]
SEEDS_PER_RUN = 20
TOTAL_SEEDS = SEEDS_PER_RUN * len(RUNS)

_SEED_DONE = re.compile(r"seed (\d+)/(\d+) done")


def _fmt(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> int:
    t0 = time.time()
    seeds_done = 0

    for idx, (label, cfg_name) in enumerate(RUNS, start=1):
        cfg = CONFIG_DIR / cfg_name
        if not cfg.exists():
            print(f"[matched-refs] missing config {cfg}", flush=True)
            return 1

        print(
            f"\n[matched-refs] === run {idx}/{len(RUNS)}: {label} "
            f"({cfg_name}) ===",
            flush=True,
        )
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "training.train", "--config", str(cfg)],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if m := _SEED_DONE.search(line):
                seeds_done += 1
                elapsed = time.time() - t0
                # Overall pace from every seed finished so far, across runs.
                per_seed = elapsed / seeds_done
                remaining = per_seed * (TOTAL_SEEDS - seeds_done)
                pct = 100.0 * seeds_done / TOTAL_SEEDS
                print(
                    f"[matched-refs] {label:15s} | seed {m.group(1)}/{m.group(2)}"
                    f" | overall {seeds_done:2d}/{TOTAL_SEEDS} ({pct:4.1f}%)"
                    f" | elapsed {_fmt(elapsed)}"
                    f" | remaining ~{_fmt(remaining)}"
                    f" | {per_seed / 60:.1f} min/seed",
                    flush=True,
                )
            else:
                print(f"[{label}] {line}", flush=True)

        if proc.wait() != 0:
            print(f"[matched-refs] {label} FAILED (exit {proc.returncode})", flush=True)
            return 1

    print(
        f"\n[matched-refs] OK | {len(RUNS)} runs, {TOTAL_SEEDS} seeds "
        f"| total {_fmt(time.time() - t0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
