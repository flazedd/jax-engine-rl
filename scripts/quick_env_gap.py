"""Fast iteration loop for env design — measures belief vs agnostic PPO gap.

Per env candidate (~60s on a workstation, 1 seed):
  1. Probe (R1 + R3-analytical + R4-entropy, no training, ~5s).
  2. regime_agnostic_PPO at --fast (20 iter × 128 envs × 1 seed, ~30s).
  3. belief_PPO at --fast (20 iter × 128 envs × 1 seed, ~30s).
  4. Print measured belief - agnostic gap and analytical compromise_cost
     side-by-side.

Operates by temporarily pointing `experiments/configs/envs/e_final.yaml`
at the candidate, since all M3 configs extend `envs/e_final.yaml`. The
symlink is restored to its original target at exit (also on Ctrl-C).

Usage:
  uv run python -m scripts.quick_env_gap \
      experiments/configs/envs/e3_asymmetric.yaml \
      experiments/configs/envs/e5a_mirror_balanced.yaml \
      experiments/configs/envs/e5b_directional_default.yaml

Each candidate's training overwrites results/m3_regime_agnostic and
results/m3_belief — that's intentional, we only need the headline
final_return per run. The summary table at the end is the artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = REPO_ROOT / "experiments" / "configs" / "envs"
E_FINAL = ENV_DIR / "e_final.yaml"
RESULTS_ROOT = REPO_ROOT / "results"

# M3 configs that extend envs/e_final.yaml.
AGNOSTIC_CFG = REPO_ROOT / "experiments" / "configs" / "m3_regime_agnostic.yaml"
BELIEF_CFG = REPO_ROOT / "experiments" / "configs" / "m3_belief.yaml"


def _swap_symlink(target: Path) -> None:
    """Repoint e_final.yaml → target. Target must be in the same dir."""
    if E_FINAL.exists() or E_FINAL.is_symlink():
        E_FINAL.unlink()
    # Symlink as a relative path (just filename), matching the existing convention.
    os.symlink(target.name, E_FINAL)


def _read_symlink_target() -> str:
    return os.readlink(E_FINAL)


def _run(cmd: list[str], prefix: str) -> tuple[int, float]:
    """Stream subprocess output with a prefix. Return (rc, elapsed_seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    rc = proc.wait()
    return rc, time.perf_counter() - t0


def _final_return(experiment_name: str) -> float | None:
    summary_path = RESULTS_ROOT / experiment_name / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        s = json.load(f)
    fr = s.get("key_stats", {}).get("final_return")
    return float(fr) if fr is not None else None


def _run_probe(env_path: Path) -> dict:
    """Run env_requirements_probe; capture output and parse R1/R4 pass."""
    print(f"\n[quick] === probe: {env_path.name} ===", flush=True)
    cmd = [
        "uv", "run", "python", "-m", "scripts.env_requirements_probe",
        "--env-config", str(env_path),
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        print(proc.stderr, end="", flush=True)
        return {"rc": proc.returncode, "r1_pass": False, "r4_pass": False, "raw": proc.stdout}

    # Parse from probe output lines.
    r1_pass = "R1 analytical  : PASS" in proc.stdout
    r4_pass = "R4 entropy     : PASS" in proc.stdout
    return {
        "rc": proc.returncode,
        "r1_pass": r1_pass,
        "r4_pass": r4_pass,
        "raw": proc.stdout,
    }


def _eval_candidate(env_path: Path, mode: str = "fast") -> dict:
    """Run probe + agnostic-fast + belief-fast for one env."""
    out = {"env": env_path.name, "ok": False}

    # 1. Probe — analytical sanity, no training.
    probe = _run_probe(env_path)
    out["probe_r1_pass"] = probe["r1_pass"]
    out["probe_r4_pass"] = probe["r4_pass"]
    if probe["rc"] != 0:
        out["error"] = "probe failed"
        return out
    if not (probe["r1_pass"] and probe["r4_pass"]):
        print(
            f"[quick] probe FAILED for {env_path.name} "
            f"(R1={probe['r1_pass']}, R4={probe['r4_pass']}) — skipping training.",
            flush=True,
        )
        out["error"] = f"probe gate fail (R1={probe['r1_pass']} R4={probe['r4_pass']})"
        return out

    flag = f"--{mode}" if mode in ("fast", "super-fast", "mid") else None

    # 2. regime_agnostic_PPO.
    print(f"\n[quick] === train regime_agnostic_PPO ({mode}): {env_path.name} ===", flush=True)
    cmd = ["uv", "run", "python", "-m", "training.train", "--config", str(AGNOSTIC_CFG)]
    if flag:
        cmd.append(flag)
    rc_a, t_a = _run(cmd, prefix="[agnostic] ")
    if rc_a != 0:
        out["error"] = f"agnostic training rc={rc_a}"
        return out
    fr_a = _final_return("m3_regime_agnostic")

    # 3. belief_PPO.
    print(f"\n[quick] === train belief_PPO ({mode}): {env_path.name} ===", flush=True)
    cmd = ["uv", "run", "python", "-m", "training.train", "--config", str(BELIEF_CFG)]
    if flag:
        cmd.append(flag)
    rc_b, t_b = _run(cmd, prefix="[belief] ")
    if rc_b != 0:
        out["error"] = f"belief training rc={rc_b}"
        return out
    fr_b = _final_return("m3_belief")

    if fr_a is None or fr_b is None:
        out["error"] = f"could not parse final_return (agn={fr_a}, bel={fr_b})"
        return out
    out.update({
        "ok": True,
        "agnostic_final_return": fr_a,
        "belief_final_return": fr_b,
        "measured_gap": fr_b - fr_a,
        "agnostic_seconds": t_a,
        "belief_seconds": t_b,
    })
    return out


def _print_summary(rows: list[dict], baseline_name: str | None = None) -> None:
    print("\n" + "=" * 92, flush=True)
    print(
        f"{'env':<32} {'R1':>3} {'R4':>3} {'agnostic':>10} {'belief':>10} "
        f"{'gap':>8} {'%agn':>6}",
        flush=True,
    )
    print("-" * 92, flush=True)
    baseline_gap = None
    for r in rows:
        if not r.get("ok"):
            print(f"{r['env']:<32} ! {r.get('error', 'fail')}", flush=True)
            continue
        agn = r["agnostic_final_return"]
        bel = r["belief_final_return"]
        gap = r["measured_gap"]
        gap_pct = (gap / agn * 100) if agn else 0.0
        if r["env"] == baseline_name:
            baseline_gap = gap
        print(
            f"{r['env']:<32} "
            f"{'Y' if r['probe_r1_pass'] else 'n':>3} "
            f"{'Y' if r['probe_r4_pass'] else 'n':>3} "
            f"{agn:>10.2f} {bel:>10.2f} "
            f"{gap:>8.2f} {gap_pct:>5.1f}%",
            flush=True,
        )

    # Highlight winners over baseline.
    if baseline_gap is not None:
        winners = [
            r for r in rows
            if r.get("ok") and r["env"] != baseline_name
            and r["measured_gap"] is not None
            and r["measured_gap"] > baseline_gap
        ]
        if winners:
            print(f"\n[quick] candidates beating baseline ({baseline_name}, gap={baseline_gap:.2f}):", flush=True)
            for r in sorted(winners, key=lambda x: x["measured_gap"], reverse=True):
                print(
                    f"[quick]   {r['env']}: gap={r['measured_gap']:.2f} "
                    f"(+{r['measured_gap'] - baseline_gap:.2f})",
                    flush=True,
                )
        else:
            print(f"\n[quick] no candidate beats baseline ({baseline_name}, gap={baseline_gap:.2f}).", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.quick_env_gap")
    parser.add_argument(
        "envs", nargs="+",
        help="paths to env yaml configs to evaluate (relative to repo or absolute)"
    )
    parser.add_argument("--mode", default="fast", choices=["fast", "super-fast", "mid"])
    parser.add_argument(
        "--baseline-name", default=None,
        help="env filename to treat as baseline for the winners-over-baseline summary"
    )
    args = parser.parse_args()

    # Resolve env paths.
    env_paths: list[Path] = []
    for s in args.envs:
        p = Path(s)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            print(f"[quick] env not found: {s}", flush=True)
            return 1
        env_paths.append(p)

    # Save current symlink target so we can restore it at exit.
    if E_FINAL.is_symlink():
        original_target = _read_symlink_target()
    else:
        original_target = None
    print(f"[quick] e_final currently → {original_target}", flush=True)

    rows: list[dict] = []
    t_total = time.perf_counter()
    try:
        for ep in env_paths:
            print(f"\n[quick] ##### candidate: {ep.name} #####", flush=True)
            _swap_symlink(ep)
            rows.append(_eval_candidate(ep, mode=args.mode))
    finally:
        # Restore.
        if original_target is not None:
            _swap_symlink(ENV_DIR / original_target)
            print(f"\n[quick] e_final restored → {original_target}", flush=True)

    elapsed_total = time.perf_counter() - t_total
    _print_summary(rows, baseline_name=args.baseline_name)
    print(f"\n[quick] OK | candidates={len(rows)} | total={elapsed_total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
