"""Single-command overnight regeneration of all medium-environment (e_final)
experiments at higher confidence: 20 seeds and 300 iterations. The persistence
and distinguishability sweep axes are left untouched.

Robustness for an unattended run:
  * Backs up every results path it will overwrite to results/_backup_pre_n20/
    (once), since this workspace has no git safety net. Restore from there if a
    run goes wrong.
  * Each component runs as an isolated subprocess; a failure is logged and the
    batch continues.
  * Resumable: a component that already wrote its success marker is skipped, and
    the training scripts themselves skip cells already trained at the target
    budget.

Order (dependencies respected): references -> meta-RL cells -> stacked-obs ->
transplant -> single-stage -> probe / action-distribution eval passes.

Usage:
  uv run python -m scripts.overnight_n20 --smoke   # validate end-to-end (~minutes)
  uv run python -m scripts.overnight_n20           # full 20-seed / 300-iter run
  uv run python -m scripts.overnight_n20 --force    # ignore done-markers, rerun all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
BACKUP = RESULTS / "_backup_pre_n20"

CELLS = ["rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet"]

# results paths overwritten by this run, backed up once before anything else
BACKUP_PATHS = (
    [f"m5r_final_{c}_e_final" for c in CELLS]
    + ["m3_regime_agnostic", "m3_belief", "m3_oracle", "m5r_stacked_obs_e_final",
       "M5R/final", "milestones/M3/stats_M3_reference_levels.json"]
)


def _backup():
    if BACKUP.exists():
        print(f"[overnight] backup already present at {BACKUP}, leaving it", flush=True)
        return
    BACKUP.mkdir(parents=True)
    for rel in BACKUP_PATHS:
        src = RESULTS / rel
        if not src.exists():
            continue
        dst = BACKUP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"[overnight] backed up {len(BACKUP_PATHS)} paths to {BACKUP}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.overnight_n20")
    ap.add_argument("--smoke", action="store_true",
                    help="validate the whole pipeline at tiny scale (2 seeds, 6 iters)")
    ap.add_argument("--force", action="store_true", help="ignore done-markers, rerun all")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=300)
    args = ap.parse_args()

    seeds = 2 if args.smoke else args.seeds
    iters = 6 if args.smoke else args.iterations
    rollouts = 50 if args.smoke else 500
    seed_list = [str(s) for s in range(seeds)]
    mode = "smoke" if args.smoke else "full"

    markers = RESULTS / "M5R" / "final" / f"_markers_{mode}"
    markers.mkdir(parents=True, exist_ok=True)

    def py(mod, *a):
        return ["uv", "run", "python", "-m", mod, *map(str, a)]

    # (name, command). Order encodes dependencies.
    steps = [
        ("references", py("scripts.run_references", "--iterations", iters, "--num-seeds", seeds)),
        ("cells", py("scripts.m5r_final_eval", "--env", "e_final", "--iterations", iters, "--num-seeds", seeds)),
        ("stacked_obs", py("scripts.m5r_stacked_obs_sweep", "--env", "e_final", "--iterations", iters, "--num-seeds", seeds)),
        ("transplant_rl2_concat", py("scripts.m5r_frozen_encoder_transplant", "--head", "concat", "--iterations", iters, "--seeds", *seed_list)),
        ("transplant_rl2_hypernet", py("scripts.m5r_frozen_encoder_transplant", "--head", "hypernet", "--iterations", iters, "--seeds", *seed_list)),
        ("transplant_vb_concat", py("scripts.m5r_frozen_encoder_transplant_vb", "--head", "concat", "--iterations", iters, "--seeds", *seed_list)),
        ("transplant_vb_hypernet", py("scripts.m5r_frozen_encoder_transplant_vb", "--head", "hypernet", "--iterations", iters, "--seeds", *seed_list)),
        ("single_stage_rl2", py("scripts.m5r_single_stage_auxdecode", "--detach", "--aux-coefs", "1.0", "--iterations", iters, "--seeds", *seed_list, "--tag", "detach_n20")),
        ("probe_logistic", py("scripts.m5r_posterior_probe", "--env-filter", "e_final", "--n-rollouts", rollouts, "--tag", "n20")),
        ("probe_mlp", py("scripts.m5r_posterior_probe", "--classifier", "mlp", "--env-filter", "e_final", "--n-rollouts", rollouts, "--tag", "n20")),
        ("action_distributions", py("scripts.m5r_action_distributions")),
    ]

    print(f"[overnight] mode={mode} seeds={seeds} iters={iters} | {len(steps)} steps", flush=True)
    _backup()

    t_start = time.perf_counter()
    results = []
    for name, cmd in steps:
        marker = markers / f"{name}.done"
        if marker.exists() and not args.force:
            print(f"[overnight] SKIP {name} (marker present)", flush=True)
            results.append((name, "skipped", 0.0))
            continue
        print(f"\n[overnight] >>> {name}\n    {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, cwd=str(REPO), check=True)
            dt = time.perf_counter() - t0
            marker.write_text(f"ok {dt:.0f}s\n")
            results.append((name, "ok", dt))
            print(f"[overnight] <<< {name} OK ({dt/60:.1f} min)", flush=True)
        except subprocess.CalledProcessError as e:
            dt = time.perf_counter() - t0
            results.append((name, f"FAILED (exit {e.returncode})", dt))
            print(f"[overnight] !!! {name} FAILED (exit {e.returncode}) after {dt/60:.1f} min; continuing", flush=True)

    total = (time.perf_counter() - t_start) / 60
    print(f"\n[overnight] === DONE in {total:.1f} min ===", flush=True)
    for name, status, dt in results:
        print(f"  {name:>26}: {status:>20}  {dt/60:6.1f} min", flush=True)
    n_failed = sum(1 for _, s, _ in results if s.startswith("FAILED"))
    print(f"[overnight] {n_failed} step(s) failed. Backup at {BACKUP}", flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
