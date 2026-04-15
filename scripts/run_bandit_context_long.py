#!/usr/bin/env python3
"""Train all agents on Contextual Bandit with Side Information.

Side signals: 2D Gaussian per task, σ=2.0 so single obs ≈ chance level.
VariBAD's decoder can predict side signals → dense ELBO gradients for task ID.
RL² agents must discover side signal value purely through RL.

  PPO MLP:  300 iters
  Recurrent: 750 iters

Saves to results/bandit_context_validation.json.
Plot with: uv run python scripts/plot_bandit_context.py

Usage:
    uv run python scripts/run_bandit_context_long.py
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

from envs.bandit_context import (BanditContextParams, SIDE_MEANS,
                                  env_reset, env_step)
from agents.ppo import ActorCritic, actor_loss_fn, critic_loss_fn
from agents.rl2 import GRUActorCritic, rl2_loss_fn
from agents.rl2_hn import HNActorCritic, rl2_hn_loss_fn
from agents.varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
from agents.varibad_hn import (VariBADHNActorCritic, varibad_hn_elbo_loss_fn,
                                varibad_hn_ppo_loss_fn)
from agents.common import batch_gae, NumpyEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_SIZE = 5          # (time, side_0..3)
N_ACTIONS = 5
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1    # 11
HIDDEN_SIZE = 64
EPISODES_PER_TRIAL = 4

N_ITERS_PPO = 300
N_ITERS_REC = 750
N_EVAL = 5000
N_ENVS = 512
N_TRIALS = 128

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "bandit_context_validation.json")

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def thompson_per_step(key, params, n_episodes):
    """Thompson sampling — uses rewards only, ignores side signals."""
    def single_episode(key):
        k_task, k_steps = jax.random.split(key)
        task_id = jax.random.randint(k_task, (), 0, params.n_arms)
        arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
        arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
        alphas = jnp.ones(params.n_arms)
        betas_post = jnp.ones(params.n_arms)
        step_keys = jax.random.split(k_steps, params.t_episode)

        def scan_fn(carry, k):
            alphas, betas_post = carry
            k_ts, k_pull = jax.random.split(k)
            samples = jax.random.beta(k_ts, alphas, betas_post)
            action = jnp.argmax(samples)
            reward = jax.random.bernoulli(
                k_pull, arm_probs[action]).astype(jnp.float32)
            alphas = alphas.at[action].add(reward)
            betas_post = betas_post.at[action].add(1.0 - reward)
            return (alphas, betas_post), reward

        _, rewards = jax.lax.scan(scan_fn, (alphas, betas_post), step_keys)
        return rewards

    keys = jax.random.split(key, n_episodes)
    return jnp.mean(jax.vmap(single_episode)(keys), axis=0)


def bayes_side_per_step(key, params, n_episodes):
    """Bayes-Side oracle — exact Bayesian inference on side signals only."""
    sigma2 = params.side_sigma ** 2

    def single_episode(key):
        k_task, k_steps = jax.random.split(key)
        task_id = jax.random.randint(k_task, (), 0, params.n_arms)
        arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
        arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
        step_keys = jax.random.split(k_steps, params.t_episode)
        log_prior = jnp.log(jnp.ones(params.n_arms) / params.n_arms)

        def scan_fn(log_post, k):
            k_side, k_pull = jax.random.split(k)
            side = SIDE_MEANS[task_id] + params.side_sigma * jax.random.normal(
                k_side, (params.side_dim,))
            log_liks = -0.5 * jnp.sum(
                (side[None, :] - SIDE_MEANS) ** 2, axis=-1) / sigma2
            log_post = log_post + log_liks
            log_post = log_post - jax.nn.logsumexp(log_post)
            action = jnp.argmax(log_post)
            reward = jax.random.bernoulli(
                k_pull, arm_probs[action]).astype(jnp.float32)
            return log_post, reward

        _, rewards = jax.lax.scan(scan_fn, log_prior, step_keys)
        return rewards

    keys = jax.random.split(key, n_episodes)
    return jnp.mean(jax.vmap(single_episode)(keys), axis=0)


# ---------------------------------------------------------------------------
# Trajectory
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


# ---------------------------------------------------------------------------
# PPO MLP
# ---------------------------------------------------------------------------

def collect_ppo_rollout(key, model, params, n_envs):
    t = params.t_episode
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_envs)
    states, obs = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    step_keys = jax.random.split(k_roll, t * 2).reshape(t, 2, -1)

    def scan_step(carry, keys):
        states, obs = carry
        k_act, k_env = keys[0], keys[1]
        logits, values = jax.vmap(model)(obs)
        act_keys = jax.random.split(k_act, n_envs)
        actions = jax.vmap(
            jax.random.categorical, in_axes=(0, 0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
        env_keys = jax.random.split(k_env, n_envs)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(env_keys, states, actions, params)
        return (new_s, new_o), (obs, actions, rews, dones, lp, values)

    _, (all_obs, all_act, all_rew, all_done, all_lp, all_val) = \
        jax.lax.scan(scan_step, (states, obs), step_keys)
    return all_obs, all_act, all_rew, all_done, all_lp, all_val


def train_ppo(key, params, n_iters):
    k_model, key = jax.random.split(key)
    model = ActorCritic(OBS_SIZE, N_ACTIONS, hidden=32, key=k_model)
    actor_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    critic_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    a_opt_st = actor_opt.init(eqx.filter(model.actor, eqx.is_array))
    c_opt_st = critic_opt.init(eqx.filter(model.critic, eqx.is_array))

    @eqx.filter_jit
    def update(model, a_opt_st, c_opt_st, batch):
        al, ag = eqx.filter_value_and_grad(actor_loss_fn)(
            model.actor, model.critic, batch)
        au, a_opt_st2 = actor_opt.update(
            ag, a_opt_st, eqx.filter(model.actor, eqx.is_array))
        cl, cg = eqx.filter_value_and_grad(critic_loss_fn)(model.critic, batch)
        cu, c_opt_st2 = critic_opt.update(
            cg, c_opt_st, eqx.filter(model.critic, eqx.is_array))
        m2 = eqx.tree_at(lambda m: m.actor, model,
                          eqx.apply_updates(model.actor, au))
        m2 = eqx.tree_at(lambda m: m.critic, m2,
                          eqx.apply_updates(model.critic, cu))
        return m2, a_opt_st2, c_opt_st2

    history = []
    t0 = time.time()
    for it in range(n_iters):
        k_coll, key = jax.random.split(key)
        obs, actions, rewards, dones, lp, values = collect_ppo_rollout(
            k_coll, model, params, N_ENVS)
        history.append(float(jnp.mean(rewards)))
        bootstrap = values[-1]
        adv, ret = batch_gae(rewards, values, dones, bootstrap, 0.99, 0.95)
        obs_f = obs.reshape(-1, OBS_SIZE)
        adv_f = adv.reshape(-1)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        batch = (obs_f, actions.reshape(-1), lp.reshape(-1),
                 adv_f, ret.reshape(-1))
        model, a_opt_st, c_opt_st = update(model, a_opt_st, c_opt_st, batch)
        if it % max(1, n_iters // 10) == 0:
            _print(f"\r    PPO MLP  iter {it:4d}/{n_iters} "
                   f"[{time.time() - t0:.0f}s]", end="", flush=True)
    _print(f"\r    PPO MLP  iter {n_iters:4d}/{n_iters} "
           f"[{time.time() - t0:.0f}s]    ")
    return model, history


def eval_ppo_per_step(model, params, key, n_episodes):
    def single_episode(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)
        step_keys = jax.random.split(k_steps, params.t_episode)
        def scan_fn(carry, k):
            state, obs = carry
            logits, _ = model(obs)
            action = jnp.argmax(logits)
            new_s, new_o, rew, _, _ = env_step(k, state, action, params)
            return (new_s, new_o), rew
        _, rewards = jax.lax.scan(scan_fn, (state, obs), step_keys)
        return rewards
    keys = jax.random.split(key, n_episodes)
    return jnp.mean(jax.vmap(single_episode)(keys), axis=0)


# ---------------------------------------------------------------------------
# Recurrent agents (RL², RL²+HN, VariBAD, VariBAD+HN)
# ---------------------------------------------------------------------------

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
    params = BanditContextParams()
    key = jax.random.PRNGKey(42)

    print("=" * 60)
    print("  Contextual Bandit with Side Information")
    print("=" * 60)
    print(f"  K={params.n_arms} arms, T={params.t_episode} steps/episode, "
          f"side_σ={params.side_sigma}")
    print(f"  arm_probs: best={params.arm_prob_high}, "
          f"others={params.arm_prob_low}")
    print(f"  PPO MLP: {N_ITERS_PPO} iters | "
          f"Recurrent: {N_ITERS_REC} iters")
    print(f"  Eval: {N_EVAL} episodes")
    print()

    out = {}

    # --- Random ---
    random_rps = float(
        (params.arm_prob_high + (params.n_arms - 1) * params.arm_prob_low)
        / params.n_arms)
    out["random"] = {"reward_per_step": random_rps,
                     "per_step_curve": [random_rps] * params.t_episode}

    # --- Thompson (reward-only) ---
    print("  Computing baselines ...")
    k_ts, key = jax.random.split(key)
    ts_curve = np.array(thompson_per_step(k_ts, params, N_EVAL)).tolist()
    out["thompson"] = {"reward_per_step": float(np.mean(ts_curve)),
                       "per_step_curve": ts_curve}
    print(f"    Random:     {random_rps:.4f} reward/step")
    print(f"    Thompson:   {out['thompson']['reward_per_step']:.4f} "
          f"reward/step (reward only)")

    # --- Bayes-Side (side-signal only) ---
    k_bs, key = jax.random.split(key)
    bs_curve = np.array(bayes_side_per_step(k_bs, params, N_EVAL)).tolist()
    out["bayes_side"] = {"reward_per_step": float(np.mean(bs_curve)),
                         "per_step_curve": bs_curve}
    print(f"    Bayes-Side: {out['bayes_side']['reward_per_step']:.4f} "
          f"reward/step (side signal only)")

    # --- PPO MLP ---
    print("\n  Training agents ...")
    k_ppo, key = jax.random.split(key)
    ppo_model, ppo_hist = train_ppo(k_ppo, params, N_ITERS_PPO)
    k_eval, key = jax.random.split(key)
    ppo_curve = np.array(eval_ppo_per_step(
        ppo_model, params, k_eval, N_EVAL)).tolist()
    out["ppo_mlp"] = {"n_iters": N_ITERS_PPO,
                      "train_history": ppo_hist,
                      "reward_per_step": float(np.mean(ppo_curve)),
                      "per_step_curve": ppo_curve}

    # --- RL² ---
    k_rl2, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_rl2)
    rl2_model = GRUActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS, hidden_size=HIDDEN_SIZE, key=k_m)
    rl2_model, rl2_hist = train_recurrent(
        k_t, rl2_model, rl2_loss_fn, params, N_ITERS_REC, label="RL²")
    k_eval, key = jax.random.split(key)
    rl2_curve = np.array(eval_recurrent_per_step(
        rl2_model, params, k_eval, N_EVAL)).tolist()
    out["rl2"] = {"n_iters": N_ITERS_REC,
                  "train_history": rl2_hist,
                  "reward_per_step": float(np.mean(rl2_curve)),
                  "per_step_curve": rl2_curve}

    # --- RL²+HN ---
    k_hn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_hn)
    hn_model = HNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, policy_hidden=16, key=k_m)
    hn_model, hn_hist = train_recurrent(
        k_t, hn_model, rl2_hn_loss_fn, params, N_ITERS_REC, label="RL²+HN")
    k_eval, key = jax.random.split(key)
    hn_curve = np.array(eval_recurrent_per_step(
        hn_model, params, k_eval, N_EVAL)).tolist()
    out["rl2_hn"] = {"n_iters": N_ITERS_REC,
                     "train_history": hn_hist,
                     "reward_per_step": float(np.mean(hn_curve)),
                     "per_step_curve": hn_curve}

    # --- VariBAD ---
    k_vb, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vb)
    vb_model = VariBADActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=4, key=k_m)
    vb_model, vb_hist = train_recurrent(
        k_t, vb_model, varibad_ppo_loss_fn, params, N_ITERS_REC,
        elbo_fn=elbo_loss_fn, label="VariBAD")
    k_eval, key = jax.random.split(key)
    vb_curve = np.array(eval_recurrent_per_step(
        vb_model, params, k_eval, N_EVAL)).tolist()
    out["varibad"] = {"n_iters": N_ITERS_REC,
                      "train_history": vb_hist,
                      "reward_per_step": float(np.mean(vb_curve)),
                      "per_step_curve": vb_curve}

    # --- VariBAD+HN ---
    k_vbhn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vbhn)
    vbhn_model = VariBADHNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=4, policy_hidden=16, key=k_m)
    vbhn_model, vbhn_hist = train_recurrent(
        k_t, vbhn_model, varibad_hn_ppo_loss_fn, params, N_ITERS_REC,
        elbo_fn=varibad_hn_elbo_loss_fn, label="VariBAD+HN")
    k_eval, key = jax.random.split(key)
    vbhn_curve = np.array(eval_recurrent_per_step(
        vbhn_model, params, k_eval, N_EVAL)).tolist()
    out["varibad_hn"] = {"n_iters": N_ITERS_REC,
                         "train_history": vbhn_hist,
                         "reward_per_step": float(np.mean(vbhn_curve)),
                         "per_step_curve": vbhn_curve}

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: {os.path.relpath(OUT_JSON, ROOT)}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  {'Agent':<14} {'reward/step':>12} {'vs Random':>10} "
          f"{'vs BayesSide':>13}")
    print("  " + "-" * 56)
    bs_rps = out["bayes_side"]["reward_per_step"]
    for label, k in [("Random", "random"), ("PPO MLP", "ppo_mlp"),
                     ("RL²", "rl2"), ("RL²+HN", "rl2_hn"),
                     ("VariBAD", "varibad"), ("VariBAD+HN", "varibad_hn"),
                     ("Thompson", "thompson"),
                     ("Bayes-Side", "bayes_side")]:
        rps = out[k]["reward_per_step"]
        print(f"  {label:<14} {rps:>12.4f} {rps - random_rps:>+10.4f} "
              f"{rps - bs_rps:>+13.4f}")
    print("=" * 60)
    print(f"\n  Plot with: uv run python scripts/plot_bandit_context.py\n")


if __name__ == "__main__":
    main()
