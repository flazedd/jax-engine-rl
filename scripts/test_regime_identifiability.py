#!/usr/bin/env python3
"""Regime Identifiability Test.

Runs episodes under each locked regime and applies an exact Bayesian filter
to the observation stream.  Measures how quickly P(true regime | history)
concentrates — the theoretical ceiling for any learning agent.

Three action policies are tested:
  - Symmetric (action 0): only drift is distinctive
  - Random uniform: all fill-rate channels active
  - Cycling (0→1→2→…): deterministic full coverage

Produces:
  plots/regime_identifiability.png

Usage:
    uv run python scripts/test_regime_identifiability.py
    uv run python scripts/test_regime_identifiability.py --n-episodes 1024
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lob_sim.jax_env import EnvParams, env_reset, env_step

PLOTS_DIR = os.path.join(ROOT, "plots")
REGIME_NAMES = ["noise", "bull", "bear"]
REGIME_TITLES = {"noise": "Noise", "bull": "Bull", "bear": "Bear"}
POLICY_NAMES = ["Symmetric (a=0)", "Random", "Cycling (0→1→2)"]
POLICY_COLORS = ["C0", "C1", "C2"]
POLICY_STYLES = ["-", "--", ":"]

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Bayesian filter over full episodes
# ---------------------------------------------------------------------------

def run_filter(key, params, true_regime, policy_type, n_episodes):
    """Run exact Bayesian filter on n_episodes under a locked regime.

    policy_type: 0=symmetric, 1=random, 2=cycling
    Returns: posteriors (n_episodes, T, 3)
    """
    t_episode = int(params.t_episode)
    env_p = params._replace(locked_regime=true_regime, init_inventory=0)

    # Pre-compute fill probabilities: (3 regimes, 3 actions, 2 sides)
    # P(fill) = exp(-kappa[r,s] * delta[a,s])
    kappa = np.array(params.kappa)   # (3,2)
    delta = np.array(params.delta)   # (3,2)
    drift = np.array(params.drift_probs)  # (3,3)
    p_fill = np.exp(-kappa[:, None, :] * delta[None, :, :])  # (3,3,2) = (regime,action,side)

    def single_episode(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, env_p)
        belief = jnp.ones(3) / 3.0
        step_keys = jax.random.split(k_steps, t_episode)

        def scan_fn(carry, key_t):
            state, belief, step_idx = carry
            k_act, k_env = jax.random.split(key_t)

            # Action selection
            action = jnp.where(
                policy_type == 0, 0,
                jnp.where(policy_type == 1,
                           jax.random.randint(k_act, (), 0, 3),
                           step_idx % 3))

            new_state, new_obs, reward, done, _ = env_step(
                k_env, state, action, env_p)

            # Observation likelihoods per regime
            fb = new_obs[0]  # fill_bid  ∈ {0,1}
            fa = new_obs[1]  # fill_ask  ∈ {0,1}
            mc = new_obs[2]  # mid_change ∈ {-1,0,+1}

            pf_bid = jnp.exp(-params.kappa[:, 0] * params.delta[action, 0])
            pf_ask = jnp.exp(-params.kappa[:, 1] * params.delta[action, 1])
            l_bid = fb * pf_bid + (1 - fb) * (1 - pf_bid)
            l_ask = fa * pf_ask + (1 - fa) * (1 - pf_ask)
            mid_idx = (mc + 1).astype(jnp.int32)
            l_mid = params.drift_probs[:, mid_idx]

            likelihood = l_bid * l_ask * l_mid           # (3,)
            new_belief = belief * likelihood
            new_belief = new_belief / (new_belief.sum() + 1e-30)

            return (new_state, new_belief, step_idx + 1), new_belief

        _, posteriors = jax.lax.scan(
            scan_fn, (state, belief, jnp.int32(0)), step_keys)
        return posteriors   # (T, 3)

    keys = jax.random.split(key, n_episodes)
    return jax.vmap(single_episode)(keys)   # (n_episodes, T, 3)


# ---------------------------------------------------------------------------
# Analytical per-step KL divergence
# ---------------------------------------------------------------------------

def per_step_kl(params, action):
    """KL(p_true || p_other) averaged over all regime pairs for a given action.

    Returns a scalar: mean pairwise KL across regimes — higher = more
    informative action.
    """
    kappa = np.array(params.kappa)
    delta = np.array(params.delta)
    drift = np.array(params.drift_probs)

    def obs_dist(regime):
        """Return the probability table over all possible observations
        (fill_bid × fill_ask × mid_change) = 2×2×3 = 12 outcomes."""
        pb = np.exp(-kappa[regime, 0] * delta[action, 0])
        pa = np.exp(-kappa[regime, 1] * delta[action, 1])
        probs = []
        for fb in [0, 1]:
            for fa in [0, 1]:
                for mi in range(3):
                    p = ((pb if fb else 1-pb) *
                         (pa if fa else 1-pa) *
                         drift[regime, mi])
                    probs.append(p)
        return np.array(probs)

    kls = []
    for r1 in range(3):
        for r2 in range(3):
            if r1 == r2:
                continue
            p = obs_dist(r1)
            q = obs_dist(r2)
            kl = np.sum(p * np.log(p / (q + 1e-30) + 1e-30))
            kls.append(kl)
    return float(np.mean(kls))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_identifiability(results, params):
    """
    results: dict[(regime_idx, policy_type)] -> posteriors (n_episodes, T, 3)
    """
    t_episode = int(params.t_episode)
    steps = np.arange(1, t_episode + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    summary_lines = []

    for col, (rid, rname) in enumerate(zip(range(3), REGIME_NAMES)):
        ax = axes[col]

        for pid, (pname, pcol, pstyle) in enumerate(
                zip(POLICY_NAMES, POLICY_COLORS, POLICY_STYLES)):
            posts = np.array(results[(rid, pid)])   # (N, T, 3)
            p_true = posts[:, :, rid]               # (N, T)
            mean = p_true.mean(axis=0)
            std = p_true.std(axis=0)

            ax.plot(steps, mean, color=pcol, linestyle=pstyle,
                    linewidth=1.5, label=pname)
            ax.fill_between(steps, mean - std, mean + std,
                            color=pcol, alpha=0.10)

            # Steps to 90% (median episode)
            med_cross = np.median(
                np.argmax(p_true > 0.9, axis=1).astype(float))
            frac_reach = float((p_true[:, -1] > 0.9).mean())
            summary_lines.append(
                f"  {REGIME_TITLES[rname]:5s} | {pname:20s} | "
                f"median step→90%: {med_cross:5.0f} | "
                f"reach 90% by end: {frac_reach:5.1%}")

        ax.axhline(1/3, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(t_episode, 1/3, " chance", va="bottom", ha="right",
                fontsize=7, color="gray")
        ax.text(t_episode, 0.9, " 90%", va="bottom", ha="right",
                fontsize=7, color="gray")

        ax.set_title(REGIME_TITLES[rname], fontsize=12, fontweight="bold")
        ax.set_xlabel("Step within episode")
        ax.set_xlim(0, t_episode)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("P(true regime | observations)")
        if col == 2:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Regime Identifiability — Exact Bayesian Filter",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "regime_identifiability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")

    # Print summary table
    print("\n  Steps to 90% confidence (median over episodes):")
    print("  " + "-" * 68)
    for line in summary_lines:
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print("  Regime Identifiability Test")
    print("=" * 60)

    params = EnvParams.default()

    # Per-step KL divergences
    print("\n  Per-step KL divergence (mean pairwise, nats):")
    for a, aname in enumerate(["symmetric(1,1)", "lean-ask(1,3)", "lean-bid(3,1)"]):
        kl = per_step_kl(params, a)
        print(f"    action {a} ({aname}): {kl:.4f}")

    # Run Bayesian filter
    results = {}
    key = jax.random.PRNGKey(0)
    n_eps = args.n_episodes
    for rid, rname in enumerate(REGIME_NAMES):
        for pid, pname in enumerate(POLICY_NAMES):
            k, key = jax.random.split(key)
            print(f"\n  Running: regime={rname}, policy={pname}, "
                  f"n={n_eps} episodes ...")
            posts = run_filter(k, params, rid, pid, n_eps)
            results[(rid, pid)] = posts

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n  Plotting ...")
    plot_identifiability(results, params)

    print("\n  Done.")


if __name__ == "__main__":
    main()
