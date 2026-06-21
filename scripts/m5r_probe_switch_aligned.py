"""Verify the switch-lag explanation of the non-monotonic per-timestep probe curve.

Re-aligns regime-decoding accuracy on time-since-last-regime-switch instead of
absolute within-episode timestep, on the medium env, for the four meta-RL cells
and the analytical posterior. If the non-monotonic absolute-time curve is driven
by regime switching, accuracy should reset low right after each switch (tss small)
and recover as evidence re-accumulates (tss large), independent of absolute time.

As a second check it splits each curve into early (t < H/2) and late (t >= H/2)
halves of the episode: if only time-since-switch matters, the two halves overlap.

Outputs:
  - results/M5R/final/m5r_probe_switch_aligned.json
  - figures/milestones/M5R/m5r_probe_switch_aligned.png  (+ thesis copy)

Usage:
  uv run python -m scripts.m5r_probe_switch_aligned --n-rollouts 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from evaluation.posterior_probe import collect_probe_rollouts, load_experiment
from plotting.style import COLORS, apply_style

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FINAL = RESULTS_ROOT / "M5R" / "final"
PROJECT_FIG = REPO_ROOT / "figures" / "milestones" / "M5R"
THESIS_FIG = REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"

METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
CELL_LABEL = {
    "rl2_concat": "RL² Concat", "rl2_hypernet": "RL² Hypernet",
    "varibad_concat": "VariBAD Concat", "varibad_hypernet": "VariBAD Hypernet",
}
ENV = "e_final"
TEST_FRAC = 0.2


def _tss(regime_TN: np.ndarray) -> np.ndarray:
    """Steps since the last regime switch (or episode start), per (t, rollout)."""
    T, N = regime_TN.shape
    sw = np.zeros((T, N), dtype=bool)
    sw[1:] = regime_TN[1:] != regime_TN[:-1]
    sw[0, :] = True  # episode start counts as a reset
    t_idx = np.arange(T)[:, None] * np.ones((1, N), dtype=int)
    last = np.where(sw, t_idx, 0)
    last = np.maximum.accumulate(last, axis=0)
    return t_idx - last


def _fit_predict(feat_TND, regime_TN, train_idx, test_idx, seed):
    T, N, D = feat_TND.shape
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=500, C=0.1, random_state=seed),
    )
    clf.fit(feat_TND[:, train_idx, :].reshape(-1, D), regime_TN[:, train_idx].reshape(-1))
    test = feat_TND[:, test_idx, :]
    preds = np.stack([clf.predict(test[t]) for t in range(T)])  # [T, n_test]
    return preds == regime_TN[:, test_idx]  # correct [T, n_test]


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.m5r_probe_switch_aligned")
    ap.add_argument("--n-rollouts", type=int, default=1000)
    ap.add_argument("--rollout-length", type=int, default=128)
    ap.add_argument("--max-tss", type=int, default=60)
    args = ap.parse_args()

    keys = list(METHODS) + ["analytical"]
    # numerator / denominator for accuracy-vs-tss, plus early/late split
    num = {k: np.zeros(args.max_tss + 1) for k in keys}
    den = {k: np.zeros(args.max_tss + 1) for k in keys}
    num_e = {k: np.zeros(args.max_tss + 1) for k in keys}
    den_e = {k: np.zeros(args.max_tss + 1) for k in keys}
    num_l = {k: np.zeros(args.max_tss + 1) for k in keys}
    den_l = {k: np.zeros(args.max_tss + 1) for k in keys}
    half = args.rollout_length // 2

    for method in METHODS:
        exp_dir = RESULTS_ROOT / f"m5r_final_{method}_{ENV}"
        seeds = sorted(int(p.stem.split("_")[-1]) for p in exp_dir.glob("checkpoint_seed_*.pkl"))
        for seed in seeds:
            bundle = load_experiment(exp_dir, seed)
            data = collect_probe_rollouts(
                bundle.env, bundle.agent, bundle.agent_state,
                n_rollouts=args.n_rollouts, rollout_length=args.rollout_length,
                key=jax.random.PRNGKey(seed),
            )
            reg = data["regime"]
            T, N = reg.shape
            rng = np.random.default_rng(seed)
            perm = rng.permutation(N)
            n_test = max(1, int(round(N * TEST_FRAC)))
            test_idx, train_idx = perm[:n_test], perm[n_test:]
            rt = reg[:, test_idx]
            tss = _tss(rt)
            t_grid = (np.arange(T)[:, None] * np.ones((1, n_test), dtype=int))

            correct_m = _fit_predict(data["belief"], reg, train_idx, test_idx, seed)
            correct_a = _fit_predict(data["analytical_belief"], reg, train_idx, test_idx, seed)
            for k, correct in ((method, correct_m), ("analytical", correct_a)):
                for tv in range(args.max_tss + 1):
                    m = tss == tv
                    num[k][tv] += correct[m].sum(); den[k][tv] += m.sum()
                    me = m & (t_grid < half)
                    num_e[k][tv] += correct[me].sum(); den_e[k][tv] += me.sum()
                    ml = m & (t_grid >= half)
                    num_l[k][tv] += correct[ml].sum(); den_l[k][tv] += ml.sum()
            print(f"[switch] {method} seed {seed} done", flush=True)

    def curve(n, d):
        d = np.where(d == 0, np.nan, d)
        return n / d

    acc = {k: curve(num[k], den[k]) for k in keys}
    acc_e = {k: curve(num_e[k], den_e[k]) for k in keys}
    acc_l = {k: curve(num_l[k], den_l[k]) for k in keys}

    out = {
        "n_rollouts": args.n_rollouts, "max_tss": args.max_tss,
        "accuracy_vs_tss": {k: acc[k].tolist() for k in keys},
        "accuracy_vs_tss_early": {k: acc_e[k].tolist() for k in keys},
        "accuracy_vs_tss_late": {k: acc_l[k].tolist() for k in keys},
    }
    FINAL.mkdir(parents=True, exist_ok=True)
    with open(FINAL / "m5r_probe_switch_aligned.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    apply_style()
    xs = np.arange(args.max_tss + 1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for method in METHODS:
        ax.plot(xs, acc[method], color=COLORS.get(method, "#666"),
                label=CELL_LABEL[method], linewidth=1.8)
    ax.plot(xs, acc["analytical"], color=COLORS.get("analytical", "#9467bd"),
            linestyle="--", linewidth=1.6, label="Analytical posterior")
    ax.axhline(1.0 / 3.0, color="#999", linestyle=":", linewidth=1.0,
               label="Random guess (33.3\\%)")
    ax.set_xlabel("Timesteps since last regime switch")
    ax.set_ylabel("Probe test accuracy")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="lower right")

    # early vs late overlap for the four cells (verification of time-independence)
    for method in METHODS:
        c = COLORS.get(method, "#666")
        ax2.plot(xs, acc_e[method], color=c, linewidth=1.5)
        ax2.plot(xs, acc_l[method], color=c, linewidth=1.5, linestyle="--")
    ax2.plot([], [], color="#444", linewidth=1.5, label="early half (t < H/2)")
    ax2.plot([], [], color="#444", linewidth=1.5, linestyle="--", label="late half (t ≥ H/2)")
    ax2.set_xlabel("Timesteps since last regime switch")
    ax2.set_ylabel("Probe test accuracy")
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    for d in (PROJECT_FIG, THESIS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "m5r_probe_switch_aligned.png")
    plt.close(fig)
    print("[switch] OK | wrote m5r_probe_switch_aligned.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
