"""Cartpole — training-free search for instances that separate all three arms.

The ladder's first three levels differ in one thing, how much of the regime each
one can know:

    regime-agnostic   the prior, and nothing else
    stacked-obs       whatever a K-step window of evidence supports
    Belief-PPO        the full-history analytical posterior

So the spacing between them is a property of the environment's inference
problem, not of any policy, and it can be measured by running those three
estimators over rollouts instead of training four agents per candidate. A
training run costs about forty minutes per instance; this costs seconds, which
is the difference between screening three hand-picked configs and screening a
grid.

What is measured, per candidate:

    prior_acc    accuracy of the initial distribution, the agnostic level
    window_acc   accuracy of a filter given only the last K likelihoods
    full_acc     accuracy of the full-history filter, Belief-PPO's own input
    headroom     full_acc - window_acc, the room a learned belief can win

The window filter is an *upper* bound on the stacked-obs arm, which sees raw
tuples rather than likelihoods and has to learn the extraction; the full filter
is exactly Belief-PPO's input. Headroom is therefore a lower bound on the return
gap it stands for, which is the safe direction for a screen.

Reads the same `info["regime_likelihood"]` the BeliefObsEnv consumes, so the
filter here and the one the agent is given cannot drift apart.

Usage:
  uv run python -m scripts.cartpole_evidence_screen
  uv run python -m scripts.cartpole_evidence_screen --behaviour random --top 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from envs.cartpole_regime_v1 import CartPoleRegimeV1
from utils.paths import cartpole_dir
from utils.script_output import ScriptRun, print_table

# The window the stacked-observation baseline carries, fixed by the matched
# protocol at K=4 and not a free parameter of this search.
STACK_K = 4

# Search axes. Asymmetry sets what the regime is worth for control; the noise
# sets how many observations it takes to find it; persistence sets how long the
# regime holds still to be accumulated over.
ASYMMETRY = {
    "0.99/0.10": (0.99, 0.10),
    "0.98/0.20": (0.98, 0.20),
    "0.95/0.30": (0.95, 0.30),
    "0.90/0.35": (0.90, 0.35),
}
NOISE = (0.15, 0.2, 0.3, 0.4, 0.6, 0.8)
PERSISTENCE = (0.98, 0.99, 0.995)


def _env_params(asym: tuple[float, float], noise: float, diag: float) -> dict[str, Any]:
    hi, lo = asym
    off = (1.0 - diag) / 2.0
    return {
        "episode_length": 128,
        "gamma": 0.99,
        "angular_velocity_noise_std": noise,
        "n_regimes": 3,
        "regime_action_success": (hi, lo, 0.80, 0.80, lo, hi),
        "transition_matrix": (diag, off, off, off, diag, off, off, off, diag),
        "initial_distribution": (0.333, 0.333, 0.334),
    }


def _rollout(env: CartPoleRegimeV1, n_envs: int, behaviour: str, key):
    """Roll out and record the per-step regime likelihood and true regime.

    The behaviour policy matters because the evidence depends on which action
    was tried: success probabilities are per direction. A balancer keeps the
    pole in the state region a trained agent actually occupies; random is the
    neutral alternative and is offered for contrast.
    """
    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)
    keys = jax.random.split(key, n_envs + 1)
    states, _ = reset_v(keys[:n_envs])
    key = keys[-1]

    def act(states_, key_):
        if behaviour == "random":
            return jax.random.randint(key_, (n_envs,), 0, env.n_actions)
        # Push toward the side the pole is falling to, the textbook heuristic.
        return (states_["theta"] + 0.5 * states_["theta_dot"] > 0).astype(jnp.int32)

    liks, regimes = [], []
    for _ in range(env.episode_length):
        k_act, k_step, key = jax.random.split(key, 3)
        actions = act(states, k_act)
        step_keys = jax.random.split(k_step, n_envs)
        states, _, _, _, info = step_v(states, actions, step_keys)
        liks.append(np.asarray(info["regime_likelihood"]))
        regimes.append(np.asarray(info["regime"]))
    return np.stack(liks), np.stack(regimes)  # [T, N, R], [T, N]


def _filters(liks: np.ndarray, T_mat: np.ndarray, prior: np.ndarray, k: int):
    """Full-history and K-window posteriors at every step.

    Both run the same forward recursion. The window filter restarts from the
    prior k steps back, which is exactly the evidence a K-tuple observation
    carries and no more.
    """
    n_t, n_env, n_r = liks.shape
    full = np.zeros((n_t, n_env, n_r))
    b = np.tile(prior, (n_env, 1))
    for t in range(n_t):
        b = b * liks[t]
        b /= np.maximum(b.sum(axis=1, keepdims=True), 1e-30)
        full[t] = b
        b = b @ T_mat

    window = np.zeros((n_t, n_env, n_r))
    for t in range(n_t):
        b = np.tile(prior, (n_env, 1))
        for s in range(max(0, t - k + 1), t + 1):
            b = b * liks[s]
            b /= np.maximum(b.sum(axis=1, keepdims=True), 1e-30)
            if s < t:
                b = b @ T_mat
        window[t] = b
    return full, window


def _accuracy(post: np.ndarray, regimes: np.ndarray) -> float:
    return float((post.argmax(axis=-1) == regimes).mean())


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.cartpole_evidence_screen")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--behaviour", choices=("balancer", "random"),
                        default="balancer")
    parser.add_argument("--top", type=int, default=15,
                        help="how many candidates to print")
    parser.add_argument("--noise", type=float, nargs="+", default=list(NOISE))
    parser.add_argument("--persistence", type=float, nargs="+",
                        default=list(PERSISTENCE))
    parser.add_argument("--asymmetry", nargs="+", default=None,
                        help="pairs as hi/lo, e.g. 0.99/0.10")
    parser.add_argument("--tag", default="",
                        help="suffix for the stats file, so a follow-up search "
                             "does not overwrite the one it follows up on")
    args = parser.parse_args()

    asymmetry = dict(ASYMMETRY)
    if args.asymmetry:
        asymmetry = {}
        for spec in args.asymmetry:
            hi, lo = spec.split("/")
            asymmetry[spec] = (float(hi), float(lo))

    run = ScriptRun(script="cartpole_evidence_screen", run_mode="n/a")
    out_dir = cartpole_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    stats_path = out_dir / f"stats_cartpole_evidence_screen{tag}.json"
    summary_path = out_dir / f"stats_cartpole_evidence_screen{tag}_run.json"

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    grid = list(itertools.product(asymmetry.items(), args.noise, args.persistence))
    for i, ((asym_name, asym), noise, diag) in enumerate(grid):
        params = _env_params(asym, noise, diag)
        env = CartPoleRegimeV1(**params)
        liks, regimes = _rollout(env, args.n_envs, args.behaviour,
                                 jax.random.PRNGKey(0))
        T_mat = np.asarray(env.transition_matrix, dtype=float).reshape(3, 3)
        prior = np.asarray(env.initial_distribution, dtype=float)
        full, window = _filters(liks, T_mat, prior, STACK_K)
        prior_acc = _accuracy(np.tile(prior, (*regimes.shape, 1)), regimes)
        window_acc = _accuracy(window, regimes)
        full_acc = _accuracy(full, regimes)
        rows.append({
            "asymmetry": asym_name,
            "noise": noise,
            "persistence": diag,
            "prior_acc": prior_acc,
            "window_acc": window_acc,
            "full_acc": full_acc,
            # What a K-step window buys over the prior, and what full-history
            # inference buys over that window.
            "window_gain": window_acc - prior_acc,
            "headroom": full_acc - window_acc,
        })
        print(f"[cp_evidence] {i + 1}/{len(grid)} "
              f"asym={asym_name} noise={noise} diag={diag} "
              f"window={window_acc:.3f} full={full_acc:.3f}", flush=True)

    # Both steps have to be open, so rank on the weaker of the two. An instance
    # where the window already matches the full filter is the case that made
    # `wide` unsatisfying; one where neither beats the prior is worse.
    for r in rows:
        r["weakest_step"] = min(r["window_gain"], r["headroom"])
    rows.sort(key=lambda r: r["weakest_step"], reverse=True)

    stats = {
        "stack_k": STACK_K,
        "behaviour": args.behaviour,
        "n_envs": args.n_envs,
        "grid": {"asymmetry": list(asymmetry), "noise": list(args.noise),
                 "persistence": list(args.persistence)},
        "candidates": rows,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    print_table(
        ["asymmetry", "noise", "persist", "prior", "window", "full",
         "window gain", "headroom", "weakest"],
        [
            [r["asymmetry"], f"{r['noise']:.2f}", f"{r['persistence']:.3f}",
             f"{r['prior_acc']:.3f}", f"{r['window_acc']:.3f}",
             f"{r['full_acc']:.3f}", f"{r['window_gain']:+.3f}",
             f"{r['headroom']:+.3f}", f"{r['weakest_step']:+.3f}"]
            for r in rows[:args.top]
        ],
        title=f"Regime-identification accuracy by estimator "
              f"({args.behaviour} behaviour, K={STACK_K}), best first",
        note="window gain = window - prior, the room for stacked-obs above "
             "regime-agnostic. headroom = full - window, the room for a "
             "learned belief above stacked-obs. Ranked by the weaker of the two.",
    )

    best = rows[0]
    run.ok(
        key_stats={
            "best_asymmetry": best["asymmetry"],
            "best_noise": best["noise"],
            "best_persistence": best["persistence"],
            "best_weakest_step": round(best["weakest_step"], 4),
            "n_candidates": len(rows),
            "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
