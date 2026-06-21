"""Behavior-cloning screen: is a frozen belief action-sufficient, and does the
integration *form* (concat vs hypernet) gate how well the optimal action can be
recovered from it?

For each frozen meta-RL encoder (medium env), we collect rollouts, take the
belief at each step, and train small heads to imitate the VALUE-ITERATION
optimal action at the true (regime, inventory). No RL — pure supervised BC.

Heads (all map (obs, belief) -> action logits):
  - obs_only        : MLP(obs)                      (floor: no belief)
  - concat_matched  : MLP([obs, belief])            (~agent-scale capacity)
  - hypernet_matched: belief -> weights of (obs->a) (~agent-scale capacity)
  - concat_big      : MLP([obs, belief]) big        (belief-sufficiency ceiling)

Belief sources per rollout set: the method's own belief, the analytical
posterior, and the true-regime one-hot (oracle). The decisive reads:
  * method belief: hypernet_matched >> concat_matched  => belief carries the
    action info but the concat form can't use it ("good belief, bad usage").
  * concat_big high on method belief                  => belief IS action-sufficient.
  * oracle one-hot: concat_matched high               => concat CAN use a clean
    belief, so the method's issue is belief *format*, not concat expressivity.

Output:
  - results/M5R/final/m5r_bc_belief_usage.json
  - figures/milestones/M5R/m5r_bc_belief_usage.png (+ thesis copy)

Usage:
  uv run python -m scripts.m5r_bc_belief_usage --n-rollouts 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from agents.modules.hypernet import Hypernet
from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import solve_value_iteration
from evaluation.posterior_probe import collect_probe_rollouts, load_experiment
from plotting.style import apply_style

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
N_ACTIONS = 3


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------
class MLPHead(nn.Module):
    hidden: tuple
    n_actions: int

    @nn.compact
    def __call__(self, obs, belief):
        x = obs if belief is None else jnp.concatenate([obs, belief], axis=-1)
        for h in self.hidden:
            x = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x))
        return nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)


class HypernetHead(nn.Module):
    obs_dim: int
    target_hidden: int
    n_actions: int
    hypernet_hidden: int

    @nn.compact
    def __call__(self, obs, belief):
        hn = Hypernet(
            target_obs_dim=self.obs_dim, target_hidden=self.target_hidden,
            target_output_dim=self.n_actions, hypernet_hidden=self.hypernet_hidden,
            init_scale=0.01,
        )
        flat = hn(belief)
        return jax.vmap(hn.apply_target)(flat, obs)


def _train_head(model, obs_tr, bel_tr, y_tr, obs_te, bel_te, y_te, key,
                steps=600, lr=1e-2):
    params = model.init(key, obs_tr[:2], None if bel_tr is None else bel_tr[:2])
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(p, ob, be, y):
        logits = model.apply(p, ob, be)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(p, os_, ob, be, y):
        loss, g = jax.value_and_grad(loss_fn)(p, ob, be, y)
        upd, os_ = opt.update(g, os_, p)
        return optax.apply_updates(p, upd), os_, loss

    for _ in range(steps):
        params, opt_state, _ = step(params, opt_state, obs_tr, bel_tr, y_tr)

    logits = model.apply(params, obs_te, bel_te)
    acc = float((jnp.argmax(logits, axis=-1) == y_te).mean())
    return acc, n_params


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.m5r_bc_belief_usage")
    ap.add_argument("--n-rollouts", type=int, default=500)
    ap.add_argument("--rollout-length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    # VI-optimal action map for the medium env.
    stats = json.load(open(RESULTS_ROOT / "milestones" / "M2" / "stats_M2_requirements.json"))
    mm = MarketMakingV1(**stats["parameters"])
    vi_policy = np.asarray(solve_value_iteration(mm).policy)  # [n_inv, n_reg]
    inv_max = int(mm.inventory_max)

    # accumulators: source -> head -> list of per-seed accuracies
    results: dict[str, dict[str, list[float]]] = {}
    param_counts: dict[str, int] = {}

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
            T, N, _ = data["belief"].shape
            obs = data["obs"].reshape(T * N, -1).astype(np.float32)
            reg = data["regime"].reshape(T * N)
            q = data["q"].reshape(T * N)
            inv_idx = np.clip(np.rint(q).astype(int) + inv_max, 0, 2 * inv_max)
            y = vi_policy[inv_idx, reg].astype(np.int32)

            method_bel = data["belief"].reshape(T * N, -1).astype(np.float32)
            ana_bel = data["analytical_belief"].reshape(T * N, -1).astype(np.float32)
            oracle_bel = np.eye(mm.n_regimes, dtype=np.float32)[reg]

            # trajectory-level split
            rng = np.random.default_rng(seed)
            perm = rng.permutation(N)
            n_test = max(1, int(round(N * TEST_FRAC)))
            test_cols, train_cols = perm[:n_test], perm[n_test:]
            row = np.arange(T)[:, None]
            tr = (row * N + train_cols[None, :]).reshape(-1)
            te = (row * N + test_cols[None, :]).reshape(-1)

            obs_dim = obs.shape[1]
            jobs = jnp.asarray(obs)
            jy = jnp.asarray(y)

            def fit(model, bel, tag):
                bel_tr = None if bel is None else jnp.asarray(bel[tr])
                bel_te = None if bel is None else jnp.asarray(bel[te])
                acc, npar = _train_head(
                    model, jobs[tr], bel_tr, jy[tr], jobs[te], bel_te, jy[te],
                    jax.random.PRNGKey(seed), steps=args.steps,
                )
                results.setdefault(tag[0], {}).setdefault(tag[1], []).append(acc)
                param_counts[f"{tag[0]}::{tag[1]}"] = npar

            # floor: obs only
            fit(MLPHead(hidden=(48,), n_actions=N_ACTIONS), None, (method, "obs_only"))

            for src_name, bel in (("method", method_bel), ("analytical", ana_bel), ("oracle", oracle_bel)):
                bdim = bel.shape[1]
                # matched-capacity concat vs hypernet
                fit(MLPHead(hidden=(48,), n_actions=N_ACTIONS), bel, (f"{method}|{src_name}", "concat_matched"))
                fit(HypernetHead(obs_dim=obs_dim, target_hidden=16, n_actions=N_ACTIONS, hypernet_hidden=16),
                    bel, (f"{method}|{src_name}", "hypernet_matched"))
            # belief-sufficiency ceiling: high-capacity concat on the method belief
            fit(MLPHead(hidden=(128, 128), n_actions=N_ACTIONS), method_bel, (f"{method}|method", "concat_big"))
            print(f"[bc] {method} seed {seed} done", flush=True)

    def summ(xs):
        a = np.asarray(xs)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    out = {
        "n_rollouts": args.n_rollouts,
        "param_counts": param_counts,
        "results": {k: {h: summ(v) for h, v in hd.items()} for k, hd in results.items()},
    }
    FINAL.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(FINAL / "m5r_bc_belief_usage.json", "w"), indent=2)

    # ---- figure: per method, matched concat vs hypernet on the method's belief,
    # plus obs-only floor and oracle reference ----
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(METHODS))
    w = 0.2

    def get(key, head):
        return out["results"].get(key, {}).get(head, {"mean": np.nan})["mean"]

    floor = [get(m, "obs_only") for m in METHODS]
    cm = [get(f"{m}|method", "concat_matched") for m in METHODS]
    hm = [get(f"{m}|method", "hypernet_matched") for m in METHODS]
    big = [get(f"{m}|method", "concat_big") for m in METHODS]

    ax.bar(x - 1.5 * w, floor, w, label="obs only (floor)", color="#bbbbbb")
    ax.bar(x - 0.5 * w, cm, w, label="concat head (matched)", color="#1f77b4")
    ax.bar(x + 0.5 * w, hm, w, label="hypernet head (matched)", color="#ff7f0e")
    ax.bar(x + 1.5 * w, big, w, label="concat head (high-capacity)", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels([CELL_LABEL[m] for m in METHODS])
    ax.set_ylabel("BC accuracy: match VI-optimal action (held-out)")
    ax.axhline(1.0 / N_ACTIONS, color="#999", linestyle=":", linewidth=1.0)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for d in (PROJECT_FIG, THESIS_FIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "m5r_bc_belief_usage.png")
    plt.close(fig)
    print("[bc] OK | wrote m5r_bc_belief_usage.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
