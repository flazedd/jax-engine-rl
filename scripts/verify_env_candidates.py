"""Screen candidate persistence / distinguishability sweep levels against the
R1-R4 structural requirements, to find how far each axis can be pushed while the
regime stays decision-relevant (R1), per-regime solvable (R2), the gap stays
non-degenerate (R3), and the regime stays inferable (R4).

Persistence: vary the HMM transition diagonal P_ii (off-diagonals split equally).
Distinguishability: vary a separation scalar s that interpolates each regime's
fill probabilities between the cross-regime mean (s=0, identical) and the medium
fills (s=1); s>1 spreads them apart, s<1 compresses them.

Usage:
  uv run python -m scripts.verify_env_candidates
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
MEDIUM = REPO / "experiments" / "configs" / "envs" / "e6e_symmetric_kappa05.yaml"
TMP = REPO / "experiments" / "configs" / "envs" / "_cand"
STATS = REPO / "results" / "milestones" / "M2" / "stats_M2_requirements.json"

MED_TIGHT = np.array([0.30, 0.80, 0.50])
MED_WIDE = np.array([0.65, 0.05, 0.02])


def _persistence_matrix(pii: float) -> list[float]:
    off = (1.0 - pii) / 2.0
    return [pii, off, off, off, pii, off, off, off, pii]


def _scaled_fills(s: float):
    mt, mw = MED_TIGHT.mean(), MED_WIDE.mean()
    t = np.clip(mt + s * (MED_TIGHT - mt), 0.02, 0.98)
    w = np.clip(mw + s * (MED_WIDE - mw), 0.02, 0.98)
    return [round(float(x), 3) for x in t], [round(float(x), 3) for x in w]


def _make_config(name: str, *, pii: float | None = None, s: float | None = None) -> Path:
    base = yaml.safe_load(open(MEDIUM))
    p = base["env"]["params"]
    if pii is not None:
        p["transition_matrix"] = _persistence_matrix(pii)
    if s is not None:
        t, w = _scaled_fills(s)
        p["regime_p_tight_bid"] = t
        p["regime_p_tight_ask"] = t
        p["regime_p_wide_bid"] = w
        p["regime_p_wide_ask"] = w
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{name}.yaml"
    yaml.safe_dump(base, open(out, "w"), sort_keys=False)
    return out


def _verify(cfg: Path, budget: str = "fast") -> dict:
    # verify returns exit 1 when not all R pass (e.g. R2 at a low budget); that
    # is expected, so we ignore the exit code and read the stats.
    if STATS.exists():
        STATS.unlink()
    flag = {"fast": ["--fast"], "mid": ["--mid"], "full": []}[budget]
    subprocess.run(
        ["uv", "run", "python", "-m", "oracles.verify_requirements",
         "--env-config", str(cfg), *flag],
        cwd=str(REPO), check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not STATS.exists():
        raise RuntimeError("verify wrote no stats")
    d = json.load(open(STATS))
    return {
        "R1_disagree": d["R1_policy_disagreement"]["fraction_disagreeing_states"],
        "R1": d["R1_policy_disagreement"]["pass"],
        "R2_ratio": d["R2_per_regime_ppo_vs_vi"]["min_ratio"],
        "R2": d["R2_per_regime_ppo_vs_vi"]["pass"],
        "R3_gap_ci": d["R3_mixed_gap"]["gap_to_ci_ratio"],
        "R3": d["R3_mixed_gap"]["pass"],
        "R4_decay": d["R4_inferability"]["entropy_decay_fraction"],
        "R4_close": d["R4_inferability"]["belief_ppo_gap_closure_fraction"],
        "R4": d["R4_inferability"]["pass"],
    }


SETS = {
    "screen": [
        ("medium_ref", dict(pii=0.98)),
        ("pers_0p90", dict(pii=0.90)),
        ("pers_0p85", dict(pii=0.85)),
        ("pers_0p80", dict(pii=0.80)),
        ("pers_0p75", dict(pii=0.75)),
        ("dist_s2p0", dict(s=2.0)),
        ("dist_s0p6", dict(s=0.6)),
        ("dist_s0p4", dict(s=0.4)),
    ],
    # full-budget confirmation of the distinguishability finalists, plus coupled
    # cells testing whether high distinguishability restores inferability under
    # fast switching.
    "finalists": [
        ("medium_ref", dict(pii=0.98)),
        ("dist_s2p0", dict(s=2.0)),
        ("dist_s0p55", dict(s=0.55)),
        ("coupled_p88_s2", dict(pii=0.88, s=2.0)),
        ("coupled_p85_s2", dict(pii=0.85, s=2.0)),
        ("coupled_p80_s2", dict(pii=0.80, s=2.0)),
    ],
    # re-tune the distinguishability-hard level: s=0.55 collapsed the gap at
    # full budget, so find the most-compressed level that still clears R3/R4.
    "disthard": [
        ("dist_s0p70", dict(s=0.70)),
        ("dist_s0p75", dict(s=0.75)),
        ("dist_s0p80", dict(s=0.80)),
    ],
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="scripts.verify_env_candidates")
    ap.add_argument("--set", choices=list(SETS), default="screen")
    ap.add_argument("--budget", choices=["fast", "mid", "full"], default="fast")
    args = ap.parse_args()
    candidates = SETS[args.set]
    rows = []
    for name, kw in candidates:
        cfg = _make_config(name, **kw)
        print(f"[verify-sweep] running {name} ({kw}) at {args.budget} budget ...", flush=True)
        try:
            r = _verify(cfg, args.budget)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}", flush=True)
            continue
        dwell = 1.0 / (1.0 - kw["pii"]) if "pii" in kw else None
        rows.append((name, kw, dwell, r))
        flags = "".join(k for k, v in [("1", r["R1"]), ("2", r["R2"]), ("3", r["R3"]), ("4", r["R4"])] if v)
        print(f"  R1d={r['R1_disagree']:.2f} R2={r['R2_ratio']:.2f} R3gap/ci={r['R3_gap_ci']:.1f} "
              f"R4decay={r['R4_decay']:.2f} R4close={r['R4_close']:.2f} | PASS:[{flags}]", flush=True)

    print("\n=== SUMMARY (thresholds: R1d>=0.80  R2>=0.85  R3ratio<=0.90  R4decay>=0.35) ===")
    print(f"{'candidate':>12} {'dwell':>6} | {'R1d':>5} {'R2':>5} {'R3':>6} {'R4dec':>6} {'R4cls':>6} | pass")
    for name, kw, dwell, r in rows:
        dw = f"{dwell:.0f}" if dwell else "-"
        flags = "".join(k for k, v in [("R1", r["R1"]), ("R2", r["R2"]), ("R3", r["R3"]), ("R4", r["R4"])] if v) or "none"
        print(f"{name:>12} {dw:>6} | {r['R1_disagree']:>5.2f} {r['R2_ratio']:>5.2f} "
              f"{r['R3_gap_ci']:>6.1f} {r['R4_decay']:>6.2f} {r['R4_close']:>6.2f} | {flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
