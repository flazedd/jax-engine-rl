"""Option 3 probe — does Belief-PPO actually use its belief input?

Trains Belief-PPO in-process (fast, single seed) on E_final, then evaluates
the trained policy on a grid of (inventory, belief) inputs to measure how
sensitive the action distribution is to the belief vector.

Three diagnostics:
1. ||d logits / d belief_input||_2 — gradient of logits w.r.t. the three
   belief channels of the obs vector. If near-zero, the network has
   effectively zero weight on the belief channels.
2. max_a,inv  KL( π(·|inv, one_hot_r=0) || π(·|inv, one_hot_r=r) ) for
   r ∈ {1, 2}. If near-zero, the policy responds identically to every belief
   — it is ignoring the input even if gradients exist (dead neuron branch).
3. max action-prob swing across beliefs: sup over beliefs of
   |P(a|inv, belief_1) - P(a|inv, belief_2)|. Intuitive magnitude.

Verdict:
  - If grad norm near-zero AND KL near-zero: belief-PPO is ignoring belief
    (architectural bug or dead feature). Needs fix before M4/M5.
  - If grad / KL meaningful: belief-PPO responds to belief, but on E_final
    the compromise policy is already near the per-regime optimum (VI shows
    this), so extra regime info can't improve expected return. Real finding,
    not a bug — document and keep belief as a ceiling.

Writes:  results/milestones/M3/belief_sensitivity_probe.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from training.config import apply_run_mode, load_config
from training.rollout import rollout
from training.train import _build_agent, _build_env

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "belief_sensitivity_probe.json"


def _train_inplace(cfg, iterations: int, parallel_envs: int, rollout_length: int):
    """Run the training loop in-process and return the final agent + params."""
    env = _build_env(cfg)
    agent = _build_agent(cfg, env)

    key = jax.random.PRNGKey(cfg.seed_base)
    init_key, key = jax.random.split(key)
    agent_state = agent.init(init_key)

    @jax.jit
    def step(agent_state, key):
        rollout_key, key = jax.random.split(key)
        traj, final_obs = rollout(
            env, agent, agent_state, rollout_key,
            parallel_envs=parallel_envs, rollout_length=rollout_length,
        )
        new_state, _ = agent.update(agent_state, traj, final_obs)
        per_env_return = traj["reward"].sum(axis=0)
        return new_state, key, per_env_return.mean()

    returns = []
    for it in range(iterations):
        agent_state, key, r = step(agent_state, key)
        jax.block_until_ready(r)
        returns.append(float(r))
        if it == 0 or (it + 1) % max(1, iterations // 5) == 0 or it == iterations - 1:
            print(
                f"[probe] belief train iter {it+1}/{iterations} | return={returns[-1]:.2f}",
                flush=True,
            )
    return agent, agent_state, returns


def _belief_sensitivity(agent, agent_state) -> dict:
    """Evaluate the trained policy's sensitivity to the belief input."""
    env_obs_size = agent.obs_size  # 11 inventory + 3 belief = 14
    n_regimes = 3
    n_inv = env_obs_size - n_regimes

    # Build grid: every inventory state crossed with each pure regime belief.
    inv_identities = jnp.eye(n_inv, dtype=jnp.float32)
    belief_one_hots = jnp.eye(n_regimes, dtype=jnp.float32)

    def _policy(params, obs):
        logits, _ = agent._model().apply(params, obs)
        return jax.nn.softmax(logits), logits

    # Gradient of logits w.r.t. the belief input channels.
    # Compute for each (inv, base-belief) point; aggregate mean/max magnitude.
    def _logits_wrt_belief(inv_vec, belief_vec):
        def f(bvec):
            obs = jnp.concatenate([inv_vec, bvec])
            logits, _ = agent._model().apply(agent_state["params"], obs)
            return logits  # [n_actions]

        jac = jax.jacfwd(f)(belief_vec)  # [n_actions, n_regimes]
        return jac

    # Flat grid of points: for each inv, for each belief_regime center.
    inv_grid, bel_grid = [], []
    for i in range(n_inv):
        for r in range(n_regimes):
            inv_grid.append(inv_identities[i])
            bel_grid.append(belief_one_hots[r])
    inv_grid = jnp.stack(inv_grid)   # [n_inv * n_regimes, n_inv]
    bel_grid = jnp.stack(bel_grid)   # [n_inv * n_regimes, n_regimes]

    jacs = jax.vmap(_logits_wrt_belief)(inv_grid, bel_grid)
    jac_norms = jnp.linalg.norm(jacs.reshape(jacs.shape[0], -1), axis=-1)

    grad_norm_mean = float(jnp.mean(jac_norms))
    grad_norm_max = float(jnp.max(jac_norms))
    grad_norm_per_inv_regime = jac_norms.reshape(n_inv, n_regimes).tolist()

    # Policy comparison across beliefs, per inventory state.
    # For each inv, compute π(·|inv, belief=δ_r) for r=0,1,2. Measure KL and
    # max total-variation.
    def _probs_for(inv_vec, belief_vec):
        obs = jnp.concatenate([inv_vec, belief_vec])
        probs, _ = _policy(agent_state["params"], obs)
        return probs

    # Shape: [n_inv, n_regimes, n_actions]
    probs_grid = jax.vmap(
        lambda inv: jax.vmap(lambda b: _probs_for(inv, b))(belief_one_hots)
    )(inv_identities)

    # KL divergence between pairs of regimes.
    def _kl(p, q):
        return jnp.sum(p * (jnp.log(p + 1e-12) - jnp.log(q + 1e-12)), axis=-1)

    # kl(r=0 || r=1), kl(r=0 || r=2), kl(r=1 || r=2)
    kls = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        kl_per_inv = _kl(probs_grid[:, a], probs_grid[:, b])
        kls.append(kl_per_inv)
    kls = jnp.stack(kls)   # [3, n_inv]
    kl_mean = float(jnp.mean(kls))
    kl_max = float(jnp.max(kls))

    # Max action-prob swing across the three regime beliefs.
    probs_min = jnp.min(probs_grid, axis=1)
    probs_max = jnp.max(probs_grid, axis=1)
    swing = float(jnp.max(probs_max - probs_min))

    return {
        "grad_logits_wrt_belief_norm_mean": grad_norm_mean,
        "grad_logits_wrt_belief_norm_max": grad_norm_max,
        "grad_norm_per_inv_regime": grad_norm_per_inv_regime,
        "kl_across_beliefs_mean": kl_mean,
        "kl_across_beliefs_max": kl_max,
        "max_action_prob_swing": swing,
        "policy_probs_per_inv_regime": probs_grid.tolist(),
    }


def main() -> int:
    cfg = load_config(REPO_ROOT / "experiments" / "configs" / "m3_belief.yaml")
    apply_run_mode(cfg, "full")
    cfg.iterations = 100
    cfg.num_seeds = 1

    print(
        f"[probe] training belief-PPO in-process: "
        f"iters={cfg.iterations} envs={cfg.parallel_envs} seed={cfg.seed_base}",
        flush=True,
    )
    t0 = time.perf_counter()
    agent, agent_state, returns = _train_inplace(
        cfg,
        iterations=cfg.iterations,
        parallel_envs=cfg.parallel_envs,
        rollout_length=cfg.rollout_length,
    )
    train_secs = time.perf_counter() - t0
    print(f"[probe] trained in {train_secs:.1f}s final_return={returns[-1]:.3f}", flush=True)

    t0 = time.perf_counter()
    diag = _belief_sensitivity(agent, agent_state)
    probe_secs = time.perf_counter() - t0

    # Verdict classification.
    is_effectively_dead = (
        diag["grad_logits_wrt_belief_norm_max"] < 1e-3
        and diag["kl_across_beliefs_max"] < 1e-3
    )
    is_responsive = (
        diag["kl_across_beliefs_max"] > 1e-2
        or diag["max_action_prob_swing"] > 1e-2
    )
    if is_effectively_dead:
        verdict = "belief_input_ignored"
    elif is_responsive:
        verdict = "belief_responsive_but_near_compromise_optimum"
    else:
        verdict = "weak_response"

    out = {
        "env_version": "e_final",
        "seed": cfg.seed_base,
        "iterations": cfg.iterations,
        "final_return": returns[-1],
        "return_curve": returns,
        "train_seconds": train_secs,
        "probe_seconds": probe_secs,
        "diagnostics": diag,
        "verdict": verdict,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[probe] OK | verdict={verdict} | "
        f"grad_norm_max={diag['grad_logits_wrt_belief_norm_max']:.5f} | "
        f"kl_max={diag['kl_across_beliefs_max']:.5f} | "
        f"prob_swing={diag['max_action_prob_swing']:.5f} | "
        f"output={OUT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
