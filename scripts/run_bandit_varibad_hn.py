#!/usr/bin/env python3
"""Train VariBAD+HN on Bernoulli Bandit, merge into existing JSON.

Usage:
    uv run python scripts/run_bandit_varibad_hn.py
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.bandit import BanditParams, env_reset, env_step
from agents.varibad_hn import (VariBADHNActorCritic,
                                varibad_hn_elbo_loss_fn, varibad_hn_ppo_loss_fn)
from agents.common import batch_gae

# -- Reuse constants from the long run --
OBS_SIZE = 1
N_ACTIONS = 3
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1
HIDDEN_SIZE = 32
EPISODES_PER_TRIAL = 4
N_ITERS = 750
N_EVAL = 5000
N_TRIALS = 128

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "bandit_validation.json")

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Trajectory (same as run_bandit_long)
# ---------------------------------------------------------------------------

class Trajectory(eqx.Module):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    prev_actions_oh: jnp.ndarray
    prev_rewards: jnp.ndarray
    gru_h: jnp.ndarray
    next_obs: jnp.ndarray


def collect_recurrent_rollout(key, model, params, n_trials):
    n_steps = EPISODES_PER_TRIAL * params.t_episode
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_trials)
    states, obs = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    prev_act = jnp.zeros((n_trials, N_ACTIONS))
    prev_rew = jnp.zeros((n_trials, 1))
    gru_st = jnp.zeros((n_trials, model.hidden_size))
    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs, prev_act, prev_rew, gru_st = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]
        aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)
        stored_h = gru_st
        logits, values, new_gru_st = jax.vmap(
            lambda x, o, h: model.forward_step(x, o, h))(aug, obs, gru_st)
        act_keys = jax.random.split(k_act, n_trials)
        actions = jax.vmap(
            jax.random.categorical, in_axes=(0, 0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
        env_keys = jax.random.split(k_env, n_trials)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(env_keys, states, actions, params)
        next_obs = new_o
        rst_keys = jax.random.split(k_rst, n_trials)
        def maybe_reset(done, ns, no, rk):
            rs, ro = env_reset(rk, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset)(dones, new_s, new_o, rst_keys)
        carry = (final_s, final_o, jax.nn.one_hot(actions, N_ACTIONS),
                 rews[:, None], new_gru_st)
        return carry, Trajectory(
            obs=obs, actions=actions, rewards=rews, dones=dones,
            log_probs=lp, values=values,
            prev_actions_oh=prev_act, prev_rewards=prev_rew,
            gru_h=stored_h, next_obs=next_obs)

    _, traj = jax.lax.scan(
        scan_step, (states, obs, prev_act, prev_rew, gru_st), all_keys)
    return traj


def train_recurrent(key, model, loss_fn, params, n_iters,
                    elbo_fn=None, label="recurrent"):
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    opt_st = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def ppo_update(model, opt_st, batch):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: loss_fn(m, batch))(model)
        updates, new_opt = optimizer.update(
            grads, opt_st, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt

    if elbo_fn is not None:
        @eqx.filter_jit
        def elbo_update(model, opt_st, elbo_data, rng_key):
            loss, grads = eqx.filter_value_and_grad(
                lambda m: elbo_fn(m, elbo_data, rng_key))(model)
            updates, new_opt = optimizer.update(
                grads, opt_st, eqx.filter(model, eqx.is_array))
            return eqx.apply_updates(model, updates), new_opt

    history = []
    t0 = time.time()
    for it in range(n_iters):
        k_coll, k_elbo, key = jax.random.split(key, 3)
        traj = collect_recurrent_rollout(k_coll, model, params, N_TRIALS)
        history.append(float(jnp.mean(traj.rewards)))
        bootstrap = traj.values[-1]
        adv, ret = batch_gae(traj.rewards, traj.values, traj.dones,
                             bootstrap, 0.99, 0.95)
        if elbo_fn is not None:
            elbo_data = (traj.obs, traj.actions, traj.rewards, traj.next_obs,
                         traj.prev_actions_oh, traj.prev_rewards)
            model, opt_st = elbo_update(model, opt_st, elbo_data, k_elbo)
        flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj)
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)
        total = flat.obs.shape[0]
        k_shuf, key = jax.random.split(key)
        for _ in range(3):
            k_p, k_shuf = jax.random.split(k_shuf)
            perm = jax.random.permutation(k_p, total)
            for start in range(0, total, 512):
                idx = perm[start:start + 512]
                mb_adv = adv_f[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb = (flat.obs[idx], flat.prev_actions_oh[idx],
                      flat.prev_rewards[idx], flat.gru_h[idx],
                      flat.actions[idx], flat.log_probs[idx],
                      mb_adv, ret_f[idx])
                model, opt_st = ppo_update(model, opt_st, mb)
        if it % max(1, n_iters // 20) == 0:
            _print(f"\r    {label:<12s} iter {it:4d}/{n_iters} "
                   f"[{time.time() - t0:.0f}s]", end="", flush=True)
    _print(f"\r    {label:<12s} iter {n_iters:4d}/{n_iters} "
           f"[{time.time() - t0:.0f}s]    ")
    return model, history


def eval_recurrent_per_step(model, params, key, n_episodes):
    t = params.t_episode
    def single_episode(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)
        h = jnp.zeros(model.hidden_size)
        prev_act = jnp.zeros(N_ACTIONS)
        prev_rew = jnp.zeros(1)
        step_keys = jax.random.split(k_steps, t)
        def scan_fn(carry, k):
            state, obs, h, prev_act, prev_rew = carry
            aug = jnp.concatenate([obs, prev_act, prev_rew])
            logits, _, new_h = model.forward_step(aug, obs, h)
            action = jnp.argmax(logits)
            new_s, new_o, rew, _, _ = env_step(k, state, action, params)
            carry = (new_s, new_o, new_h,
                     jax.nn.one_hot(action, N_ACTIONS), rew[None])
            return carry, rew
        _, rewards = jax.lax.scan(
            scan_fn, (state, obs, h, prev_act, prev_rew), step_keys)
        return rewards
    keys = jax.random.split(key, n_episodes)
    return jnp.mean(jax.vmap(single_episode)(keys), axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params = BanditParams()
    key = jax.random.PRNGKey(99)

    print("=" * 60)
    print("  VariBAD+HN — Training on Bernoulli Bandit")
    print("=" * 60)
    print(f"  K={params.n_arms} arms, T={params.t_episode} steps/episode")
    print(f"  {N_ITERS} iters, eval: {N_EVAL} episodes")
    print()

    k_m, k_t = jax.random.split(key)
    model = VariBADHNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=2, policy_hidden=16, key=k_m)

    print("  Training ...")
    model, history = train_recurrent(
        k_t, model, varibad_hn_ppo_loss_fn, params, N_ITERS,
        elbo_fn=varibad_hn_elbo_loss_fn, label="VariBAD+HN")

    print("\n  Evaluating ...")
    k_eval = jax.random.PRNGKey(0)
    curve = np.array(eval_recurrent_per_step(
        model, params, k_eval, N_EVAL)).tolist()
    rps = float(np.mean(curve))

    # --- Merge into existing JSON ---
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            data = json.load(f)
    else:
        data = {}

    data["varibad_hn"] = {
        "n_iters": N_ITERS,
        "train_history": history,
        "reward_per_step": rps,
        "per_step_curve": curve,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    # --- Summary ---
    random_rps = data.get("random", {}).get("reward_per_step", 0.5)
    ts_rps = data.get("thompson", {}).get("reward_per_step", 0.0)

    print(f"\n  VariBAD+HN:  {rps:.4f} reward/step "
          f"(vs random: {rps - random_rps:+.4f}, "
          f"vs Thompson: {rps - ts_rps:+.4f})")
    print(f"\n  Merged into: {os.path.relpath(OUT_JSON, ROOT)}")
    print(f"  Plot with: uv run python scripts/plot_bandit.py\n")


if __name__ == "__main__":
    main()
