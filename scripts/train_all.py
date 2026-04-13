#!/usr/bin/env python3
"""Train all agents and generate comparison plot.

Runs each training script in sequence (PPO MLP → RL² → RL²+HN → VariBAD),
forwarding common flags, then produces the comparison figure.

Usage:
    uv run python scripts/train_all.py --fast
    uv run python scripts/train_all.py --regime mixed
    uv run python scripts/train_all.py --n-iters 300 --lr 1e-4
"""
import argparse
import builtins
import subprocess
import sys
import time
import os

_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    ("PPO MLP",   os.path.join(ROOT, "scripts", "train_ppo.py")),
    ("RL²",       os.path.join(ROOT, "scripts", "train_rl2.py")),
    ("RL²+HN",    os.path.join(ROOT, "scripts", "train_rl2_hn.py")),
    ("VariBAD",   os.path.join(ROOT, "scripts", "train_varibad.py")),
]

PLOT_SCRIPT = os.path.join(ROOT, "scripts", "plot_comparison.py")


def _fmt_time(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--regime", type=str, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
    args = parser.parse_args()

    # Build shared flags to forward
    shared = []
    if args.fast:
        shared.append("--fast")
    if args.regime is not None:
        shared.extend(["--regime", args.regime])
    if args.n_seeds is not None:
        shared.extend(["--n-seeds", str(args.n_seeds)])
    if args.n_iters is not None:
        shared.extend(["--n-iters", str(args.n_iters)])
    if args.lr is not None:
        shared.extend(["--lr", str(args.lr)])
    if args.master_seed is not None:
        shared.extend(["--master-seed", str(args.master_seed)])

    print("=" * 60)
    print("  Train All Agents")
    print("=" * 60)
    print(f"  Agents: {', '.join(name for name, _ in SCRIPTS)}")
    if shared:
        print(f"  Flags:  {' '.join(shared)}")
    print()

    t0 = time.time()
    results = []

    for i, (name, script) in enumerate(SCRIPTS):
        print(f"  [{i+1}/{len(SCRIPTS)}] {name}")
        print("-" * 60)
        agent_t0 = time.time()

        cmd = [sys.executable, script] + shared
        ret = subprocess.run(cmd, cwd=ROOT)

        elapsed = time.time() - agent_t0
        if ret.returncode != 0:
            print(f"\n  ERROR: {name} failed (exit code {ret.returncode})")
            results.append((name, "FAILED", elapsed))
        else:
            results.append((name, "OK", elapsed))
        print()

    # Generate comparison plot
    print(f"  [{len(SCRIPTS)+1}/{len(SCRIPTS)+1}] Comparison plot")
    print("-" * 60)
    subprocess.run([sys.executable, PLOT_SCRIPT], cwd=ROOT)
    print()

    # Summary
    total = time.time() - t0
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, status, elapsed in results:
        print(f"    {name:<12s}  {status:<8s}  {_fmt_time(elapsed)}")
    print(f"\n  Total: {_fmt_time(total)}")
    print("  Done.")


if __name__ == "__main__":
    main()
