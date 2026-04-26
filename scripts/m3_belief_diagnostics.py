"""Diagnose why Belief-PPO ≈ Regime-agnostic PPO on E_final (M3 finding).

Three tests, cheapest first:

  D1 — structural sanity: log obs/posterior shape and variability across
       timesteps of a single rollout. Catch shape / concatenation bugs.

  D3 — posterior informativeness: run many rollouts under a uniform
       random policy, compute posterior entropy statistics and regime
       classification accuracy (argmax posterior vs true regime). If the
       posterior rarely sharpens below uniform, no method can exploit it.

  D2 — training ablation: train Belief-PPO with belief replaced by the
       constant initial prior at every step. If the ablated return
       matches unablated return, the network is ignoring the belief
       (bug). Only run if D3 says the posterior is actually informative.

Written to run on the current E_final (`experiments/configs/envs/e_final.yaml`)
with no retraining for D1+D3. D2 retrains briefly.

Usage:
  uv run python -m scripts.m3_belief_diagnostics
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from beliefs.hmm_posterior import entropy as belief_entropy
from envs.market_making_v1 import MarketMakingV1
from envs.wrappers.belief_obs import BeliefObsEnv
from training.config import _load_yaml_with_extends

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_CFG = REPO_ROOT / "experiments" / "configs" / "envs" / "e_final.yaml"
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "belief_diagnostics.json"


def _build_env() -> BeliefObsEnv:
    raw = _load_yaml_with_extends(ENV_CFG)
    params = raw["env"]["params"]
    inner = MarketMakingV1(**params)
    return BeliefObsEnv(inner=inner)


def _random_rollout(env: BeliefObsEnv, seed: int, n_envs: int, n_steps: int):
    """Uniform random policy rollout. Returns belief [T, N, R] and true regime [T, N]."""
    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(key, n_envs)
    states, _ = jax.vmap(env.reset)(reset_keys)

    n_actions = env.n_actions

    def step_one(carry, _):
        states, key = carry
        key, sub = jax.random.split(key)
        act_key, step_key = jax.random.split(sub)
        actions = jax.random.randint(
            act_key, shape=(n_envs,), minval=0, maxval=n_actions
        )
        step_keys = jax.random.split(step_key, n_envs)
        new_states, _obs, _r, _d, info = jax.vmap(env.step)(states, actions, step_keys)
        belief = info["belief"]  # [N, R]
        regime = new_states["regime"]  # [N]
        return (new_states, key), (belief, regime)

    _, (beliefs, regimes) = jax.lax.scan(
        step_one, (states, key), None, length=n_steps
    )
    return np.asarray(beliefs), np.asarray(regimes)  # [T, N, R], [T, N]


def d1_structural(env: BeliefObsEnv) -> dict:
    """Log obs/posterior shape and variability of one rollout."""
    inner = env.inner
    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)
    belief_dim = inner.n_regimes
    inv_dim = inner.obs_size

    obs_seq = [np.asarray(obs)]
    belief_seq = [np.asarray(state["belief"])]
    for t in range(16):
        key, sub = jax.random.split(key)
        act_key, step_key = jax.random.split(sub)
        a = jax.random.randint(act_key, (), 0, env.n_actions)
        state, obs, _r, _d, info = env.step(state, a, step_key)
        obs_seq.append(np.asarray(obs))
        belief_seq.append(np.asarray(info["belief"]))

    obs_seq = np.stack(obs_seq)  # [17, obs_size]
    belief_seq = np.stack(belief_seq)  # [17, R]

    # Does the belief portion of obs match the tracked belief?
    obs_belief_portion = obs_seq[:, inv_dim:]
    match = bool(np.allclose(obs_belief_portion, belief_seq, atol=1e-6))

    # Is the belief actually varying?
    belief_var_per_dim = belief_seq.var(axis=0)
    max_var = float(belief_var_per_dim.max())

    return {
        "obs_size": int(env.obs_size),
        "inventory_dim": int(inv_dim),
        "belief_dim": int(belief_dim),
        "obs_layout_ok": match,
        "belief_variance_max_across_17_steps": max_var,
        "first_3_beliefs": belief_seq[:3].tolist(),
        "initial_distribution": list(inner.initial_distribution),
    }


def d3_informativeness(env: BeliefObsEnv, n_envs: int = 512, n_steps: int = 128) -> dict:
    """Posterior entropy + argmax-vs-truth accuracy under random policy."""
    t0 = time.perf_counter()
    beliefs, regimes = _random_rollout(env, seed=0, n_envs=n_envs, n_steps=n_steps)
    elapsed = time.perf_counter() - t0

    ent = np.asarray(belief_entropy(jnp.asarray(beliefs)))  # [T, N]
    uniform_ent = float(np.log(env.inner.n_regimes))

    # Per-step entropy (averaged across envs), and overall distribution.
    entropy_per_step = ent.mean(axis=1)  # [T]
    entropy_flat = ent.reshape(-1)

    # Argmax accuracy (posterior's most-likely regime vs true regime).
    argmax_regime = beliefs.argmax(axis=-1)  # [T, N]
    accuracy_per_step = (argmax_regime == regimes).mean(axis=1)  # [T]
    accuracy_overall = float((argmax_regime == regimes).mean())

    # Fraction of timesteps where posterior is "confident" (entropy < 0.5 * uniform).
    confident_mask = ent < 0.5 * uniform_ent
    frac_confident = float(confident_mask.mean())

    # Per-regime mass under the posterior on timesteps matching each true regime.
    n_reg = env.inner.n_regimes
    per_regime_mass = []
    for r in range(n_reg):
        mask = regimes == r  # [T, N]
        if mask.sum() == 0:
            per_regime_mass.append(None)
            continue
        # Average posterior conditional on true regime being r.
        b_given_r = beliefs[mask]  # [M, R]
        per_regime_mass.append(b_given_r.mean(axis=0).tolist())

    # Stationary fraction (how often each regime actually occurs in rollouts).
    regime_fracs = [float((regimes == r).mean()) for r in range(n_reg)]

    # Posterior precision: E[b_true_regime | true_regime]
    # = average posterior probability the model assigns to the correct regime.
    true_regime_prob_per_step = np.take_along_axis(
        beliefs, regimes[..., None], axis=-1
    ).squeeze(-1)  # [T, N]
    avg_correct_prob = float(true_regime_prob_per_step.mean())

    return {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "uniform_entropy": uniform_ent,
        "entropy_overall_mean": float(entropy_flat.mean()),
        "entropy_overall_min": float(entropy_flat.min()),
        "entropy_overall_p25": float(np.percentile(entropy_flat, 25)),
        "entropy_overall_p75": float(np.percentile(entropy_flat, 75)),
        "entropy_decay_fraction_from_uniform": float(
            1.0 - entropy_flat.mean() / uniform_ent
        ),
        "entropy_by_step_first5": entropy_per_step[:5].tolist(),
        "entropy_by_step_last5": entropy_per_step[-5:].tolist(),
        "frac_confident_timesteps": frac_confident,
        "argmax_accuracy_overall": accuracy_overall,
        "argmax_accuracy_first5": accuracy_per_step[:5].tolist(),
        "argmax_accuracy_last5": accuracy_per_step[-5:].tolist(),
        "avg_prob_assigned_to_true_regime": avg_correct_prob,
        "regime_fractions_under_random_policy": regime_fracs,
        "posterior_conditional_on_true_regime": per_regime_mass,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    print("[diag] building env ...", flush=True)
    env = _build_env()

    print("[diag] D1 — structural sanity ...", flush=True)
    d1 = d1_structural(env)
    print(
        f"[diag]   obs_size={d1['obs_size']} "
        f"(inv={d1['inventory_dim']} + belief={d1['belief_dim']})  "
        f"obs_layout_ok={d1['obs_layout_ok']}  "
        f"max_belief_variance_17steps={d1['belief_variance_max_across_17_steps']:.4f}",
        flush=True,
    )

    print("[diag] D3 — posterior informativeness (random policy, 512 × 128) ...", flush=True)
    d3 = d3_informativeness(env)
    print(
        f"[diag]   uniform_entropy={d3['uniform_entropy']:.3f}  "
        f"mean_entropy={d3['entropy_overall_mean']:.3f}  "
        f"(decay={d3['entropy_decay_fraction_from_uniform']*100:.1f}% from uniform)",
        flush=True,
    )
    print(
        f"[diag]   argmax_accuracy={d3['argmax_accuracy_overall']:.3f}  "
        f"avg_prob_assigned_to_true_regime={d3['avg_prob_assigned_to_true_regime']:.3f}  "
        f"frac_confident(entropy<0.5·uniform)={d3['frac_confident_timesteps']:.3f}",
        flush=True,
    )
    print(
        f"[diag]   regime_fracs={['%.2f' % x for x in d3['regime_fractions_under_random_policy']]}  "
        f"entropy_first5={['%.2f' % x for x in d3['entropy_by_step_first5']]}  "
        f"entropy_last5={['%.2f' % x for x in d3['entropy_by_step_last5']]}",
        flush=True,
    )

    out = {"D1_structural": d1, "D3_informativeness": d3}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[diag] OK | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
