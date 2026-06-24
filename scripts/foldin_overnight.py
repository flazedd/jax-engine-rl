"""One-command overnight fold-in.

Waits for the single-stage VariBAD, RL2 A2C, and VariBAD A2C runs to finish, then
regenerates the affected figures, rebuilds the thesis + presentation, and writes a
report. Run once in the background; needs no further input or permissions.

The prose and figure blocks are ALREADY in the thesis. This script only swaps in
the real figures and reports the numbers, with a check that they match what the
prose claims (so any surprise is flagged for review rather than silently wrong).

Usage: uv run python -m scripts.foldin_overnight
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/Volumes/samsungssd990/github/workspace/jax-engine-rl")
THESIS = Path("/Volumes/samsungssd990/github/workspace/master_thesis_reinier_schep_final")
FINAL = REPO / "results" / "M5R" / "final"
REPORT = Path("/tmp/foldin_report.txt")

FLOOR, BELIEF = 138.0, 168.8


def _mean(xs):
    return sum(xs) / len(xs)


def _ss():
    p = FINAL / "m5r_single_stage_auxdecode_vb_detach_n20.json"
    if not p.exists():
        return None
    c = json.load(open(p)).get("by_coef", {}).get("1.0")
    if c and len(c.get("returns", [])) >= 20:
        return c
    return None


def _a2c(method):
    p = FINAL / f"optsearch_{method}_a2c.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    ba = d.get("by_arch", {})
    if all("mean_over_search" in ba.get(a, {})
           and len(ba.get(a, {}).get("per_config", [])) >= 16
           for a in ("concat", "hypernet")):
        return d
    return None


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def main() -> int:
    log = []

    def emit(s):
        log.append(s)
        print(s, flush=True)
        REPORT.write_text("\n".join(log))

    emit("[foldin] waiting for single-stage VariBAD + RL2 A2C + VariBAD A2C ...")
    deadline = time.time() + 16 * 3600
    while True:
        ss, a2c_rl2, a2c_vb = _ss(), _a2c("rl2"), _a2c("varibad")
        if ss and a2c_rl2 and a2c_vb:
            break
        if time.time() > deadline:
            emit("[foldin] TIMEOUT after 16h; results still incomplete. Aborting.")
            return 1
        time.sleep(120)

    emit("[foldin] all three results complete. Regenerating figures ...")
    r1 = _run(["uv", "run", "python", "-m", "plotting.optsearch_plot"], REPO)
    emit("  optsearch_plot: " + (r1.stdout.strip().splitlines()[-1]
                                 if r1.stdout.strip() else r1.stderr[-300:]))
    r2 = _run(["uv", "run", "python", "-m", "scripts.plot_transplant_figure"], REPO)
    emit("  transplant: " + (r2.stdout.strip().splitlines()[-1]
                             if r2.stdout.strip() else r2.stderr[-300:]))

    emit("[foldin] rebuilding thesis + presentation ...")
    for _ in range(2):
        tb = _run(["latexmk", "-pdf", "-interaction=nonstopmode",
                   "-halt-on-error", "main.tex"], THESIS)
    pb = _run(["latexmk", "-pdf", "-interaction=nonstopmode",
               "-halt-on-error", "presentation.tex"], THESIS)
    emit(f"  thesis build: {'OK' if tb.returncode == 0 else 'FAILED'}; "
         f"presentation build: {'OK' if pb.returncode == 0 else 'FAILED'}")

    emit("\n===== RESULTS (review against the written prose) =====")
    emit(f"Single-stage VariBAD: return {ss['return_mean']:.1f} "
         f"+/- {ss['return_std']:.1f}, belief-probe {ss['probe_acc_mean']:.3f}")
    if ss["return_mean"] < FLOOR:
        emit("  !! WARNING: single-stage BELOW FLOOR. Prose says 'reaches the "
             "transplant level' -- REVISE the single-stage paragraph.")
    elif ss["return_mean"] < BELIEF - 10:
        emit("  ~ NOTE: single-stage below the belief ceiling; 'reaches the "
             "transplant level' may need softening.")
    else:
        emit("  OK: single-stage reaches transplant/ceiling level -- prose holds.")

    for name, d in [("RL2", a2c_rl2), ("VariBAD", a2c_vb)]:
        emit(f"A2C {name}:")
        stats = {}
        for a in ("concat", "hypernet"):
            ms = [c["mean_return"] for c in d["by_arch"][a]["per_config"]]
            stats[a] = ms
            emit(f"  {a:8s}: mean {_mean(ms):6.1f}  best {max(ms):6.1f}  "
                 f"%>floor {100 * _mean([1.0 if x > FLOOR else 0.0 for x in ms]):3.0f}%")
        if _mean(stats["hypernet"]) > _mean(stats["concat"]) + 5:
            emit(f"  OK: hypernet > concat under A2C ({name}) -- prose holds.")
        else:
            emit(f"  !! WARNING: hypernet NOT clearly > concat under A2C ({name}) "
                 "-- REVISE the optimisation-scope limitation.")

    emit("\nFigures refreshed: m5r_optsearch.png (PPO, both methods), "
         "m5r_optsearch_a2c.png (A2C, both methods), m5r_transplant.png "
         "(VariBAD single-stage bar).")
    emit("Prose already in the thesis. Revise only if a WARNING fired above.")
    emit("[foldin] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
