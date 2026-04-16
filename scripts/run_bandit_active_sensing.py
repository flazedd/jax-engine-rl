#!/usr/bin/env python3
"""Train agents on Contextual Bandit with Active Sensing + Abstain.

Explore-exploit under uncertainty: pulling the best arm degrades future
observations (σ_exploit=4.0), while pulling wrong arms yields clearer
signals (σ_explore=1.5). Abstaining gives moderate noise (σ_default=2.5).

This tests whether agents learn strategic exploration — sacrificing
short-term reward for information that improves long-term decisions.

Saves to results/bandit_active_sensing.json.
Plot with: uv run python scripts/plot_bandit_active_sensing.py

Usage:
    uv run python scripts/run_bandit_active_sensing.py --agent ppo_mlp
    uv run python scripts/run_bandit_active_sensing.py --agent all
    uv run python scripts/run_bandit_active_sensing.py --agent all --fast
"""
import argparse
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

from envs.bandit_context import SIDE_MEANS
from envs.bandit_active_sensing import (BanditActiveSensingParams,
                                         ABSTAIN_PAYOFF, SIGMA_EXPLOIT,
                                         SIGMA_EXPLORE, SIGMA_DEFAULT,
                                         env_reset, env_step, _get_sigma)
from agents.ppo import ActorCritic, actor_loss_fn, critic_loss_fn
from agents.rl2 import GRUActorCritic, rl2_loss_fn
from agents.rl2_hn import HNActorCritic, rl2_hn_loss_fn
from agents.varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
from agents.varibad_hn_v2_mu import (V2MuOnlyActorCritic, v2_mu_ppo_loss_fn,
                                      v2_mu_elbo_loss_fn)
from agents.varibad_hn_v10_musigma import (V10MuSigmaActorCritic,
                                            v10_ppo_loss_fn, v10_elbo_loss_fn)
from agents.varibad_hn_v11_curiosity import (V11CuriosityActorCritic,
                                              v11_ppo_loss_fn, v11_elbo_loss_fn,
                                              posterior_entropy, BETA_CURIOSITY)
from agents.varibad_hn_v12_dualmu import (V12DualMuActorCritic,
                                           v12_ppo_loss_fn, v12_elbo_loss_fn)
from agents.varibad_hn_v13_prederr import (V13PredErrActorCritic,
                                            v13_ppo_loss_fn, v13_elbo_loss_fn)
from agents.v14_fastweight import (V14FastWeightActorCritic, v14_fw_loss_fn)
from agents.v15_plasticity import (V15PlasticityActorCritic,
                                    v15_plasticity_loss_fn)
from agents.v16_attention import (V16AttentionActorCritic,
                                   v16_attention_loss_fn)
from agents.v17_lottery import (V17LotteryActorCritic, v17_lottery_loss_fn)
from agents.v18_predcoding import (V18PredCodingActorCritic,
                                    v18_predcoding_ppo_loss_fn,
                                    v18_predcoding_elbo_loss_fn)
from agents.v19_worldmodel import (V19WorldModelActorCritic,
                                    v19_wm_ppo_loss_fn, v19_wm_loss_fn)
from agents.v20_neuromod import (V20NeuromodActorCritic, v20_neuromod_loss_fn)
from agents.v21_attn_hn import (V21AttnHNActorCritic, v21_attn_hn_loss_fn)
from agents.v22_posterior_vel import (V22PosteriorVelActorCritic,
                                      v22_vel_ppo_loss_fn, v22_vel_elbo_loss_fn)
from agents.v23_contrastive import (V23ContrastiveActorCritic,
                                     v23_contrastive_ppo_loss_fn,
                                     v23_contrastive_elbo_loss_fn)
from agents.v24_aux_classifier import (V24AuxClassifierActorCritic,
                                        v24_aux_ppo_loss_fn,
                                        v24_aux_elbo_loss_fn)
from agents.v25_moe_hn import (V25MoEHNActorCritic, v25_moe_ppo_loss_fn,
                                v25_moe_elbo_loss_fn)
from agents.v26_pred_attn import (V26PredAttnActorCritic,
                                   v26_pred_attn_ppo_loss_fn,
                                   v26_pred_attn_elbo_loss_fn)
from agents.v27_hindsight import (V27HindsightActorCritic,
                                   v27_hindsight_ppo_loss_fn,
                                   v27_hindsight_elbo_loss_fn)
from agents.common import batch_gae, NumpyEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_SIZE = 5          # (time, side_0..3)
N_ARMS = 5
N_ACTIONS = 6         # 5 arms + 1 abstain
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1    # 12
HIDDEN_SIZE = 64
LATENT_DIM = 4
POLICY_HIDDEN = 16
EPISODES_PER_TRIAL = 4

N_ITERS_PPO = 300
N_ITERS_REC = 300
N_EVAL = 5000
N_ENVS = 512
N_TRIALS = 128

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "bandit_active_sensing.json")

ALL_AGENTS = ["ppo_mlp", "rl2", "rl2_hn", "varibad", "v2_mu",
              "v10_musigma", "v11_curiosity", "v12_dualmu", "v13_prederr",
              "v14_fastweight", "v15_plasticity", "v16_attention",
              "v17_lottery", "v18_predcoding", "v19_worldmodel",
              "v20_neuromod", "v21_attn_hn", "v22_posterior_vel",
              "v23_contrastive", "v24_aux_classifier", "v25_moe_hn",
              "v26_pred_attn", "v27_hindsight"]

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def bayes_optimal_per_step(key, params, n_episodes):
    """Bayes-optimal: exact posterior over tasks with action-dependent σ.

    The likelihood for each observation accounts for the noise level that
    was determined by the previous action:
      - prev_action == task_t → σ_exploit
      - prev_action != task_t and prev_action < K → σ_explore
      - prev_action == K (abstain) → σ_default
    """

    def single_episode(key):
        k_task, k_steps = jax.random.split(key)
        task_id = jax.random.randint(k_task, (), 0, params.n_arms)
        arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
        arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
        step_keys = jax.random.split(k_steps, params.t_episode)
        log_prior = jnp.log(jnp.ones(params.n_arms) / params.n_arms)
        prev_action = jnp.int32(params.n_arms)  # initial = abstain

        def scan_fn(carry, k):
            log_post, arm_probs, prev_action = carry
            k_side, k_pull = jax.random.split(k)

            # Generate side signal with action-dependent noise
            sigma = _get_sigma(prev_action, task_id, params)
            side = SIDE_MEANS[task_id] + sigma * jax.random.normal(
                k_side, (params.side_dim,))

            # Update posterior: P(task=t | side, prev_action)
            # Likelihood depends on what sigma would have been if task=t
            def log_lik_for_task(t):
                s = _get_sigma(prev_action, t, params)
                return -0.5 * jnp.sum((side - SIDE_MEANS[t]) ** 2) / (s ** 2) \
                       - params.side_dim * jnp.log(s)
            log_liks = jax.vmap(log_lik_for_task)(jnp.arange(params.n_arms))
            log_post = log_post + log_liks
            log_post = log_post - jax.nn.logsumexp(log_post)
            post = jnp.exp(log_post)

            # Decision: pull MAP arm or abstain
            map_arm = jnp.argmax(post)
            e_pull = post[map_arm] * params.arm_prob_high + \
                     (1.0 - post[map_arm]) * params.arm_prob_low
            should_abstain = e_pull < params.abstain_payoff

            reward_pull = jax.random.bernoulli(
                k_pull, arm_probs[map_arm]).astype(jnp.float32)
            reward = jnp.where(should_abstain, params.abstain_payoff,
                               reward_pull)
            action = jnp.where(should_abstain,
                               jnp.int32(params.n_arms), map_arm)
            return (log_post, arm_probs, action), (reward, action)

        _, (rewards, actions) = jax.lax.scan(
            scan_fn, (log_prior, arm_probs, prev_action), step_keys)
        abstain_mask = (actions == params.n_arms).astype(jnp.float32)
        return rewards, abstain_mask

    keys = jax.random.split(key, n_episodes)
    all_rewards, all_abstains = jax.vmap(single_episode)(keys)
    return (jnp.mean(all_rewards, axis=0),
            jnp.mean(all_abstains, axis=0))


def always_abstain_rps(params):
    """Reward/step from always abstaining."""
    return params.abstain_payoff


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
    task_ids: jnp.ndarray


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
            is_abstain = (action == params.n_arms).astype(jnp.float32)
            return (new_s, new_o), (rew, is_abstain)
        _, (rewards, abstains) = jax.lax.scan(scan_fn, (state, obs), step_keys)
        return rewards, abstains
    keys = jax.random.split(key, n_episodes)
    all_r, all_a = jax.vmap(single_episode)(keys)
    return jnp.mean(all_r, axis=0), jnp.mean(all_a, axis=0)


# ---------------------------------------------------------------------------
# Recurrent agents
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
            gru_h=stored_h, next_obs=next_obs,
            task_ids=states.task_id)

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


def train_recurrent_taskids(key, model, loss_fn, params, n_iters,
                            elbo_fn=None, label="recurrent"):
    """Like train_recurrent but passes task_ids as 7th ELBO element."""
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
                         traj.prev_actions_oh, traj.prev_rewards,
                         traj.task_ids)
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
            is_abstain = (action == params.n_arms).astype(jnp.float32)
            carry = (new_s, new_o, new_h,
                     jax.nn.one_hot(action, N_ACTIONS), rew[None])
            return carry, (rew, is_abstain)
        _, (rewards, abstains) = jax.lax.scan(
            scan_fn, (state, obs, h, prev_act, prev_rew), step_keys)
        return rewards, abstains
    keys = jax.random.split(key, n_episodes)
    all_r, all_a = jax.vmap(single_episode)(keys)
    return jnp.mean(all_r, axis=0), jnp.mean(all_a, axis=0)


# ---------------------------------------------------------------------------
# V11: Curiosity bonus — custom rollout with intrinsic reward
# ---------------------------------------------------------------------------

def collect_curiosity_rollout(key, model, params, n_trials):
    """Rollout with curiosity bonus: r_total = r_ext + β · ΔH(posterior)."""
    n_steps = EPISODES_PER_TRIAL * params.t_episode
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_trials)
    states, obs = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    prev_act = jnp.zeros((n_trials, N_ACTIONS))
    prev_rew = jnp.zeros((n_trials, 1))
    gru_st = jnp.zeros((n_trials, model.hidden_size))
    # Initial log_sigma for entropy tracking
    prev_log_sig = jnp.zeros((n_trials, model.latent_dim))
    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs, prev_act, prev_rew, gru_st, prev_log_sig = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]
        aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)
        stored_h = gru_st
        logits, values, new_gru_st = jax.vmap(
            lambda x, o, h: model.forward_step(x, o, h))(aug, obs, gru_st)
        # Compute current log_sigma for entropy bonus
        def get_log_sig(h):
            post = model.posterior_net(model.gru_cell(
                jnp.zeros(aug.shape[-1]), h))  # dummy — we need new_h
            return jnp.clip(post[model.latent_dim:], -5.0, 2.0)
        curr_log_sig = jax.vmap(model.get_log_sigma)(new_gru_st)
        # Entropy reduction: H(prev) - H(curr) = sum(prev_log_sig - curr_log_sig)
        entropy_reduction = jnp.sum(prev_log_sig - curr_log_sig, axis=-1)
        curiosity_bonus = BETA_CURIOSITY * entropy_reduction

        act_keys = jax.random.split(k_act, n_trials)
        actions = jax.vmap(
            jax.random.categorical, in_axes=(0, 0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
        env_keys = jax.random.split(k_env, n_trials)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(env_keys, states, actions, params)
        # Add curiosity bonus to reward
        augmented_rews = rews + curiosity_bonus
        next_obs = new_o
        rst_keys = jax.random.split(k_rst, n_trials)
        def maybe_reset(done, ns, no, rk):
            rs, ro = env_reset(rk, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset)(dones, new_s, new_o, rst_keys)
        # Reset log_sig on done
        reset_log_sig = jnp.where(
            dones[:, None], jnp.zeros_like(curr_log_sig), curr_log_sig)
        carry = (final_s, final_o, jax.nn.one_hot(actions, N_ACTIONS),
                 rews[:, None], new_gru_st, reset_log_sig)
        return carry, Trajectory(
            obs=obs, actions=actions, rewards=augmented_rews, dones=dones,
            log_probs=lp, values=values,
            prev_actions_oh=prev_act, prev_rewards=prev_rew,
            gru_h=stored_h, next_obs=next_obs,
            task_ids=states.task_id)

    _, traj = jax.lax.scan(
        scan_step,
        (states, obs, prev_act, prev_rew, gru_st, prev_log_sig),
        all_keys)
    return traj


def train_curiosity(key, model, loss_fn, params, n_iters,
                    elbo_fn=None, label="V11:Curiosity"):
    """Train with curiosity-augmented rollouts."""
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
        traj = collect_curiosity_rollout(k_coll, model, params, N_TRIALS)
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


# ---------------------------------------------------------------------------
# V13: Prediction-error input — custom rollout with pred_err tracking
# ---------------------------------------------------------------------------

def collect_prederr_rollout(key, model, params, n_trials):
    """Rollout for V13: tracks prediction error and feeds it into policy."""
    n_steps = EPISODES_PER_TRIAL * params.t_episode
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_trials)
    states, obs = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    prev_act = jnp.zeros((n_trials, N_ACTIONS))
    prev_rew = jnp.zeros((n_trials, 1))
    gru_st = jnp.zeros((n_trials, model.hidden_size))
    pred_err = jnp.zeros(n_trials)
    prev_action_idx = jnp.full(n_trials, N_ACTIONS - 1, dtype=jnp.int32)
    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs, prev_act, prev_rew, gru_st, pred_err, prev_action_idx = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]
        aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)
        stored_h = gru_st
        # Forward with prediction error
        logits, values, new_gru_st = jax.vmap(
            lambda x, o, h, pe: model.forward_step(x, o, h, pe))(
            aug, obs, gru_st, pred_err)
        act_keys = jax.random.split(k_act, n_trials)
        actions = jax.vmap(
            jax.random.categorical, in_axes=(0, 0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
        env_keys = jax.random.split(k_env, n_trials)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(env_keys, states, actions, params)
        # Compute prediction error for next step
        new_pred_err = jax.vmap(model.compute_pred_error)(
            new_gru_st, new_o, actions)
        next_obs = new_o
        rst_keys = jax.random.split(k_rst, n_trials)
        def maybe_reset(done, ns, no, rk):
            rs, ro = env_reset(rk, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset)(dones, new_s, new_o, rst_keys)
        reset_pred_err = jnp.where(dones, 0.0, new_pred_err)
        carry = (final_s, final_o, jax.nn.one_hot(actions, N_ACTIONS),
                 rews[:, None], new_gru_st, reset_pred_err, actions)
        return carry, Trajectory(
            obs=obs, actions=actions, rewards=rews, dones=dones,
            log_probs=lp, values=values,
            prev_actions_oh=prev_act, prev_rewards=prev_rew,
            gru_h=stored_h, next_obs=next_obs,
            task_ids=states.task_id)

    _, traj = jax.lax.scan(
        scan_step,
        (states, obs, prev_act, prev_rew, gru_st, pred_err, prev_action_idx),
        all_keys)
    return traj


def train_prederr(key, model, loss_fn, params, n_iters,
                  elbo_fn=None, label="V13:PredErr"):
    """Train V13 with prediction-error-aware rollouts."""
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
        traj = collect_prederr_rollout(k_coll, model, params, N_TRIALS)
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


def eval_prederr_per_step(model, params, key, n_episodes):
    """Eval V13 with prediction error tracking."""
    t = params.t_episode
    def single_episode(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)
        h = jnp.zeros(model.hidden_size)
        prev_act = jnp.zeros(N_ACTIONS)
        prev_rew = jnp.zeros(1)
        pred_err = jnp.float32(0.0)
        step_keys = jax.random.split(k_steps, t)
        def scan_fn(carry, k):
            state, obs, h, prev_act, prev_rew, pred_err = carry
            aug = jnp.concatenate([obs, prev_act, prev_rew])
            logits, _, new_h = model.forward_step(aug, obs, h, pred_err)
            action = jnp.argmax(logits)
            new_s, new_o, rew, _, _ = env_step(k, state, action, params)
            new_pred_err = model.compute_pred_error(new_h, new_o, action)
            is_abstain = (action == params.n_arms).astype(jnp.float32)
            carry = (new_s, new_o, new_h,
                     jax.nn.one_hot(action, N_ACTIONS), rew[None], new_pred_err)
            return carry, (rew, is_abstain)
        _, (rewards, abstains) = jax.lax.scan(
            scan_fn, (state, obs, h, prev_act, prev_rew, pred_err), step_keys)
        return rewards, abstains
    keys = jax.random.split(key, n_episodes)
    all_r, all_a = jax.vmap(single_episode)(keys)
    return jnp.mean(all_r, axis=0), jnp.mean(all_a, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global N_ITERS_PPO, N_ITERS_REC, N_EVAL, N_ENVS, N_TRIALS

    parser = argparse.ArgumentParser(
        description="Contextual Bandit with Active Sensing + Abstain")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke test")
    parser.add_argument("--agent", type=str, default="all",
                        choices=ALL_AGENTS + ["all"],
                        help="Train a single agent (merges into existing JSON)")
    args = parser.parse_args()

    if args.fast:
        N_ITERS_PPO = 100
        N_ITERS_REC = 100
        N_EVAL = 1000
        N_ENVS = 128
        N_TRIALS = 64

    run_agent = args.agent
    agents_to_run = ALL_AGENTS if run_agent == "all" else [run_agent]
    params = BanditActiveSensingParams()

    # Load existing results to merge into
    if os.path.exists(OUT_JSON) and run_agent != "all":
        with open(OUT_JSON) as f:
            out = json.load(f)
    else:
        out = {}

    print("=" * 60)
    suffix = f" (FAST)" if args.fast else ""
    if run_agent != "all":
        suffix += f" agent={run_agent}"
    print(f"  Active Sensing Bandit + Abstain{suffix}")
    print("=" * 60)
    print(f"  K={params.n_arms} arms + 1 abstain, T={params.t_episode} steps")
    print(f"  σ_exploit={params.sigma_exploit}, σ_explore={params.sigma_explore}, "
          f"σ_default={params.sigma_default}")
    print(f"  abstain_payoff={params.abstain_payoff}, "
          f"arm_probs: best={params.arm_prob_high}, "
          f"others={params.arm_prob_low}")
    print(f"  PPO MLP: {N_ITERS_PPO} iters | "
          f"Recurrent: {N_ITERS_REC} iters")
    print(f"  Eval: {N_EVAL} episodes")
    print()

    # --- Baselines (always computed) ---
    key = jax.random.PRNGKey(42)

    random_rps = float(
        (params.arm_prob_high + (params.n_arms - 1) * params.arm_prob_low
         + params.abstain_payoff)
        / (params.n_arms + 1))
    out["random"] = {
        "reward_per_step": random_rps,
        "per_step_curve": [random_rps] * params.t_episode,
        "abstain_curve": [1.0 / (params.n_arms + 1)] * params.t_episode}

    abstain_rps = float(params.abstain_payoff)
    out["always_abstain"] = {
        "reward_per_step": abstain_rps,
        "per_step_curve": [abstain_rps] * params.t_episode,
        "abstain_curve": [1.0] * params.t_episode}

    print("  Computing baselines ...")
    k_bo, key = jax.random.split(key)
    bo_reward_curve, bo_abstain_curve = bayes_optimal_per_step(
        k_bo, params, N_EVAL)
    bo_reward_curve = np.array(bo_reward_curve).tolist()
    bo_abstain_curve = np.array(bo_abstain_curve).tolist()
    out["bayes_optimal"] = {
        "reward_per_step": float(np.mean(bo_reward_curve)),
        "per_step_curve": bo_reward_curve,
        "abstain_curve": bo_abstain_curve}
    print(f"    Random:         {random_rps:.4f} reward/step")
    print(f"    Always Abstain: {abstain_rps:.4f} reward/step")
    print(f"    Bayes-Optimal:  {out['bayes_optimal']['reward_per_step']:.4f} "
          f"reward/step")

    def should_run(name):
        return name in agents_to_run

    print("\n  Training agents ...")

    # Deterministic key schedule: always advance regardless of which agent runs
    # PPO MLP
    k_ppo, key = jax.random.split(key)
    k_eval_ppo, key = jax.random.split(key)
    if should_run("ppo_mlp"):
        ppo_model, ppo_hist = train_ppo(k_ppo, params, N_ITERS_PPO)
        ppo_r, ppo_a = eval_ppo_per_step(ppo_model, params, k_eval_ppo, N_EVAL)
        out["ppo_mlp"] = {"n_iters": N_ITERS_PPO,
                          "train_history": ppo_hist,
                          "reward_per_step": float(np.mean(np.array(ppo_r))),
                          "per_step_curve": np.array(ppo_r).tolist(),
                          "abstain_curve": np.array(ppo_a).tolist()}

    # RL²
    k_rl2, key = jax.random.split(key)
    k_eval_rl2, key = jax.random.split(key)
    if should_run("rl2"):
        k_m, k_t = jax.random.split(k_rl2)
        rl2_model = GRUActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS, hidden_size=HIDDEN_SIZE, key=k_m)
        rl2_model, rl2_hist = train_recurrent(
            k_t, rl2_model, rl2_loss_fn, params, N_ITERS_REC, label="RL²")
        rl2_r, rl2_a = eval_recurrent_per_step(
            rl2_model, params, k_eval_rl2, N_EVAL)
        out["rl2"] = {"n_iters": N_ITERS_REC,
                      "train_history": rl2_hist,
                      "reward_per_step": float(np.mean(np.array(rl2_r))),
                      "per_step_curve": np.array(rl2_r).tolist(),
                      "abstain_curve": np.array(rl2_a).tolist()}

    # RL²+HN
    k_hn, key = jax.random.split(key)
    k_eval_hn, key = jax.random.split(key)
    if should_run("rl2_hn"):
        k_m, k_t = jax.random.split(k_hn)
        hn_model = HNActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, policy_hidden=POLICY_HIDDEN, key=k_m)
        hn_model, hn_hist = train_recurrent(
            k_t, hn_model, rl2_hn_loss_fn, params, N_ITERS_REC, label="RL²+HN")
        hn_r, hn_a = eval_recurrent_per_step(
            hn_model, params, k_eval_hn, N_EVAL)
        out["rl2_hn"] = {"n_iters": N_ITERS_REC,
                         "train_history": hn_hist,
                         "reward_per_step": float(np.mean(np.array(hn_r))),
                         "per_step_curve": np.array(hn_r).tolist(),
                         "abstain_curve": np.array(hn_a).tolist()}

    # VariBAD
    k_vb, key = jax.random.split(key)
    k_eval_vb, key = jax.random.split(key)
    if should_run("varibad"):
        k_m, k_t = jax.random.split(k_vb)
        vb_model = VariBADActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM, key=k_m)
        vb_model, vb_hist = train_recurrent(
            k_t, vb_model, varibad_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=elbo_loss_fn, label="VariBAD")
        vb_r, vb_a = eval_recurrent_per_step(
            vb_model, params, k_eval_vb, N_EVAL)
        out["varibad"] = {"n_iters": N_ITERS_REC,
                          "train_history": vb_hist,
                          "reward_per_step": float(np.mean(np.array(vb_r))),
                          "per_step_curve": np.array(vb_r).tolist(),
                          "abstain_curve": np.array(vb_a).tolist()}

    # V2: μ-Only
    k_v2, key = jax.random.split(key)
    k_eval_v2, key = jax.random.split(key)
    if should_run("v2_mu"):
        k_m, k_t = jax.random.split(k_v2)
        v2_model = V2MuOnlyActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v2_model, v2_hist = train_recurrent(
            k_t, v2_model, v2_mu_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v2_mu_elbo_loss_fn, label="V2:MuOnly")
        v2_r, v2_a = eval_recurrent_per_step(
            v2_model, params, k_eval_v2, N_EVAL)
        out["v2_mu"] = {"n_iters": N_ITERS_REC,
                        "train_history": v2_hist,
                        "reward_per_step": float(np.mean(np.array(v2_r))),
                        "per_step_curve": np.array(v2_r).tolist(),
                        "abstain_curve": np.array(v2_a).tolist()}

    # V10: μ+σ HN
    k_v10, key = jax.random.split(key)
    k_eval_v10, key = jax.random.split(key)
    if should_run("v10_musigma"):
        k_m, k_t = jax.random.split(k_v10)
        v10_model = V10MuSigmaActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v10_model, v10_hist = train_recurrent(
            k_t, v10_model, v10_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v10_elbo_loss_fn, label="V10:μσ")
        v10_r, v10_a = eval_recurrent_per_step(
            v10_model, params, k_eval_v10, N_EVAL)
        out["v10_musigma"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v10_hist,
            "reward_per_step": float(np.mean(np.array(v10_r))),
            "per_step_curve": np.array(v10_r).tolist(),
            "abstain_curve": np.array(v10_a).tolist()}

    # V11: Curiosity bonus
    k_v11, key = jax.random.split(key)
    k_eval_v11, key = jax.random.split(key)
    if should_run("v11_curiosity"):
        k_m, k_t = jax.random.split(k_v11)
        v11_model = V11CuriosityActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v11_model, v11_hist = train_curiosity(
            k_t, v11_model, v11_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v11_elbo_loss_fn, label="V11:Curiosity")
        v11_r, v11_a = eval_recurrent_per_step(
            v11_model, params, k_eval_v11, N_EVAL)
        out["v11_curiosity"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v11_hist,
            "reward_per_step": float(np.mean(np.array(v11_r))),
            "per_step_curve": np.array(v11_r).tolist(),
            "abstain_curve": np.array(v11_a).tolist()}

    # V12: Dual μ input
    k_v12, key = jax.random.split(key)
    k_eval_v12, key = jax.random.split(key)
    if should_run("v12_dualmu"):
        k_m, k_t = jax.random.split(k_v12)
        v12_model = V12DualMuActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v12_model, v12_hist = train_recurrent(
            k_t, v12_model, v12_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v12_elbo_loss_fn, label="V12:DualMu")
        v12_r, v12_a = eval_recurrent_per_step(
            v12_model, params, k_eval_v12, N_EVAL)
        out["v12_dualmu"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v12_hist,
            "reward_per_step": float(np.mean(np.array(v12_r))),
            "per_step_curve": np.array(v12_r).tolist(),
            "abstain_curve": np.array(v12_a).tolist()}

    # V13: Prediction-error input
    k_v13, key = jax.random.split(key)
    k_eval_v13, key = jax.random.split(key)
    if should_run("v13_prederr"):
        k_m, k_t = jax.random.split(k_v13)
        v13_model = V13PredErrActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v13_model, v13_hist = train_prederr(
            k_t, v13_model, v13_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v13_elbo_loss_fn, label="V13:PredErr")
        v13_r, v13_a = eval_prederr_per_step(
            v13_model, params, k_eval_v13, N_EVAL)
        out["v13_prederr"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v13_hist,
            "reward_per_step": float(np.mean(np.array(v13_r))),
            "per_step_curve": np.array(v13_r).tolist(),
            "abstain_curve": np.array(v13_a).tolist()}

    # V14: Fast Weights
    k_v14, key = jax.random.split(key)
    k_eval_v14, key = jax.random.split(key)
    if should_run("v14_fastweight"):
        k_m, k_t = jax.random.split(k_v14)
        v14_model = V14FastWeightActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS, fw_dim=8, key=k_m)
        v14_model, v14_hist = train_recurrent(
            k_t, v14_model, v14_fw_loss_fn, params, N_ITERS_REC,
            label="V14:FW")
        v14_r, v14_a = eval_recurrent_per_step(
            v14_model, params, k_eval_v14, N_EVAL)
        out["v14_fastweight"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v14_hist,
            "reward_per_step": float(np.mean(np.array(v14_r))),
            "per_step_curve": np.array(v14_r).tolist(),
            "abstain_curve": np.array(v14_a).tolist()}

    # V15: Plasticity
    k_v15, key = jax.random.split(key)
    k_eval_v15, key = jax.random.split(key)
    if should_run("v15_plasticity"):
        k_m, k_t = jax.random.split(k_v15)
        v15_model = V15PlasticityActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, policy_hidden=POLICY_HIDDEN, key=k_m)
        v15_model, v15_hist = train_recurrent(
            k_t, v15_model, v15_plasticity_loss_fn, params, N_ITERS_REC,
            label="V15:Plast")
        v15_r, v15_a = eval_recurrent_per_step(
            v15_model, params, k_eval_v15, N_EVAL)
        out["v15_plasticity"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v15_hist,
            "reward_per_step": float(np.mean(np.array(v15_r))),
            "per_step_curve": np.array(v15_r).tolist(),
            "abstain_curve": np.array(v15_a).tolist()}

    # V16: Attention
    k_v16, key = jax.random.split(key)
    k_eval_v16, key = jax.random.split(key)
    if should_run("v16_attention"):
        k_m, k_t = jax.random.split(k_v16)
        v16_model = V16AttentionActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            buffer_size=10, attn_dim=16, key=k_m)
        v16_model, v16_hist = train_recurrent(
            k_t, v16_model, v16_attention_loss_fn, params, N_ITERS_REC,
            label="V16:Attn")
        v16_r, v16_a = eval_recurrent_per_step(
            v16_model, params, k_eval_v16, N_EVAL)
        out["v16_attention"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v16_hist,
            "reward_per_step": float(np.mean(np.array(v16_r))),
            "per_step_curve": np.array(v16_r).tolist(),
            "abstain_curve": np.array(v16_a).tolist()}

    # V17: Lottery Ticket
    k_v17, key = jax.random.split(key)
    k_eval_v17, key = jax.random.split(key)
    if should_run("v17_lottery"):
        k_m, k_t = jax.random.split(k_v17)
        v17_model = V17LotteryActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, lottery_hidden=64, key=k_m)
        v17_model, v17_hist = train_recurrent(
            k_t, v17_model, v17_lottery_loss_fn, params, N_ITERS_REC,
            label="V17:Lottery")
        v17_r, v17_a = eval_recurrent_per_step(
            v17_model, params, k_eval_v17, N_EVAL)
        out["v17_lottery"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v17_hist,
            "reward_per_step": float(np.mean(np.array(v17_r))),
            "per_step_curve": np.array(v17_r).tolist(),
            "abstain_curve": np.array(v17_a).tolist()}

    # V18: Predictive Coding
    k_v18, key = jax.random.split(key)
    k_eval_v18, key = jax.random.split(key)
    if should_run("v18_predcoding"):
        k_m, k_t = jax.random.split(k_v18)
        v18_model = V18PredCodingActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v18_model, v18_hist = train_recurrent(
            k_t, v18_model, v18_predcoding_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v18_predcoding_elbo_loss_fn, label="V18:PredCod")
        v18_r, v18_a = eval_recurrent_per_step(
            v18_model, params, k_eval_v18, N_EVAL)
        out["v18_predcoding"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v18_hist,
            "reward_per_step": float(np.mean(np.array(v18_r))),
            "per_step_curve": np.array(v18_r).tolist(),
            "abstain_curve": np.array(v18_a).tolist()}

    # V19: World Model
    k_v19, key = jax.random.split(key)
    k_eval_v19, key = jax.random.split(key)
    if should_run("v19_worldmodel"):
        k_m, k_t = jax.random.split(k_v19)
        v19_model = V19WorldModelActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, key=k_m)
        v19_model, v19_hist = train_recurrent(
            k_t, v19_model, v19_wm_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v19_wm_loss_fn, label="V19:WM")
        v19_r, v19_a = eval_recurrent_per_step(
            v19_model, params, k_eval_v19, N_EVAL)
        out["v19_worldmodel"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v19_hist,
            "reward_per_step": float(np.mean(np.array(v19_r))),
            "per_step_curve": np.array(v19_r).tolist(),
            "abstain_curve": np.array(v19_a).tolist()}

    # V20: Neuromodulation
    k_v20, key = jax.random.split(key)
    k_eval_v20, key = jax.random.split(key)
    if should_run("v20_neuromod"):
        k_m, k_t = jax.random.split(k_v20)
        v20_model = V20NeuromodActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, policy_hidden=POLICY_HIDDEN, key=k_m)
        v20_model, v20_hist = train_recurrent(
            k_t, v20_model, v20_neuromod_loss_fn, params, N_ITERS_REC,
            label="V20:Neuro")
        v20_r, v20_a = eval_recurrent_per_step(
            v20_model, params, k_eval_v20, N_EVAL)
        out["v20_neuromod"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v20_hist,
            "reward_per_step": float(np.mean(np.array(v20_r))),
            "per_step_curve": np.array(v20_r).tolist(),
            "abstain_curve": np.array(v20_a).tolist()}

    # V21: Attention-conditioned HN
    k_v21, key = jax.random.split(key)
    k_eval_v21, key = jax.random.split(key)
    if should_run("v21_attn_hn"):
        k_m, k_t = jax.random.split(k_v21)
        v21_model = V21AttnHNActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            gru_hidden=HIDDEN_SIZE, buffer_size=10, attn_dim=16,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v21_model, v21_hist = train_recurrent(
            k_t, v21_model, v21_attn_hn_loss_fn, params, N_ITERS_REC,
            label="V21:AttnHN")
        v21_r, v21_a = eval_recurrent_per_step(
            v21_model, params, k_eval_v21, N_EVAL)
        out["v21_attn_hn"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v21_hist,
            "reward_per_step": float(np.mean(np.array(v21_r))),
            "per_step_curve": np.array(v21_r).tolist(),
            "abstain_curve": np.array(v21_a).tolist()}

    # V22: Posterior Velocity
    k_v22, key = jax.random.split(key)
    k_eval_v22, key = jax.random.split(key)
    if should_run("v22_posterior_vel"):
        k_m, k_t = jax.random.split(k_v22)
        v22_model = V22PosteriorVelActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            gru_hidden=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v22_model, v22_hist = train_recurrent(
            k_t, v22_model, v22_vel_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v22_vel_elbo_loss_fn, label="V22:Vel")
        v22_r, v22_a = eval_recurrent_per_step(
            v22_model, params, k_eval_v22, N_EVAL)
        out["v22_posterior_vel"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v22_hist,
            "reward_per_step": float(np.mean(np.array(v22_r))),
            "per_step_curve": np.array(v22_r).tolist(),
            "abstain_curve": np.array(v22_a).tolist()}

    # V23: Contrastive
    k_v23, key = jax.random.split(key)
    k_eval_v23, key = jax.random.split(key)
    if should_run("v23_contrastive"):
        k_m, k_t = jax.random.split(k_v23)
        v23_model = V23ContrastiveActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v23_model, v23_hist = train_recurrent(
            k_t, v23_model, v23_contrastive_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v23_contrastive_elbo_loss_fn, label="V23:Contr")
        v23_r, v23_a = eval_recurrent_per_step(
            v23_model, params, k_eval_v23, N_EVAL)
        out["v23_contrastive"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v23_hist,
            "reward_per_step": float(np.mean(np.array(v23_r))),
            "per_step_curve": np.array(v23_r).tolist(),
            "abstain_curve": np.array(v23_a).tolist()}

    # V24: Auxiliary Classifier (needs task_ids)
    k_v24, key = jax.random.split(key)
    k_eval_v24, key = jax.random.split(key)
    if should_run("v24_aux_classifier"):
        k_m, k_t = jax.random.split(k_v24)
        v24_model = V24AuxClassifierActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS, n_tasks=N_ARMS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v24_model, v24_hist = train_recurrent_taskids(
            k_t, v24_model, v24_aux_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v24_aux_elbo_loss_fn, label="V24:AuxCls")
        v24_r, v24_a = eval_recurrent_per_step(
            v24_model, params, k_eval_v24, N_EVAL)
        out["v24_aux_classifier"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v24_hist,
            "reward_per_step": float(np.mean(np.array(v24_r))),
            "per_step_curve": np.array(v24_r).tolist(),
            "abstain_curve": np.array(v24_a).tolist()}

    # V25: Mixture-of-Experts HN
    k_v25, key = jax.random.split(key)
    k_eval_v25, key = jax.random.split(key)
    if should_run("v25_moe_hn"):
        k_m, k_t = jax.random.split(k_v25)
        v25_model = V25MoEHNActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v25_model, v25_hist = train_recurrent(
            k_t, v25_model, v25_moe_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v25_moe_elbo_loss_fn, label="V25:MoE")
        v25_r, v25_a = eval_recurrent_per_step(
            v25_model, params, k_eval_v25, N_EVAL)
        out["v25_moe_hn"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v25_hist,
            "reward_per_step": float(np.mean(np.array(v25_r))),
            "per_step_curve": np.array(v25_r).tolist(),
            "abstain_curve": np.array(v25_a).tolist()}

    # V26: Predictive Attention (triple fusion)
    k_v26, key = jax.random.split(key)
    k_eval_v26, key = jax.random.split(key)
    if should_run("v26_pred_attn"):
        k_m, k_t = jax.random.split(k_v26)
        v26_model = V26PredAttnActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS,
            gru_hidden=HIDDEN_SIZE, buffer_size=10, attn_dim=16,
            latent_dim=LATENT_DIM, policy_hidden=POLICY_HIDDEN, key=k_m)
        v26_model, v26_hist = train_recurrent(
            k_t, v26_model, v26_pred_attn_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v26_pred_attn_elbo_loss_fn, label="V26:PredAttn")
        v26_r, v26_a = eval_recurrent_per_step(
            v26_model, params, k_eval_v26, N_EVAL)
        out["v26_pred_attn"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v26_hist,
            "reward_per_step": float(np.mean(np.array(v26_r))),
            "per_step_curve": np.array(v26_r).tolist(),
            "abstain_curve": np.array(v26_a).tolist()}

    # V27: Hindsight Belief Distillation (needs task_ids)
    k_v27, key = jax.random.split(key)
    k_eval_v27, key = jax.random.split(key)
    if should_run("v27_hindsight"):
        k_m, k_t = jax.random.split(k_v27)
        v27_model = V27HindsightActorCritic(
            INPUT_SIZE, OBS_SIZE, N_ACTIONS, n_tasks=N_ARMS,
            hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
            policy_hidden=POLICY_HIDDEN, key=k_m)
        v27_model, v27_hist = train_recurrent_taskids(
            k_t, v27_model, v27_hindsight_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v27_hindsight_elbo_loss_fn, label="V27:Hindsight")
        v27_r, v27_a = eval_recurrent_per_step(
            v27_model, params, k_eval_v27, N_EVAL)
        out["v27_hindsight"] = {
            "n_iters": N_ITERS_REC,
            "train_history": v27_hist,
            "reward_per_step": float(np.mean(np.array(v27_r))),
            "per_step_curve": np.array(v27_r).tolist(),
            "abstain_curve": np.array(v27_a).tolist()}

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: {os.path.relpath(OUT_JSON, ROOT)}")

    # --- Summary ---
    print("\n" + "=" * 68)
    print(f"  {'Agent':<16} {'reward/step':>12} {'vs Abstain':>11} "
          f"{'vs BayesOpt':>12}")
    print("  " + "-" * 62)
    bo_rps = out["bayes_optimal"]["reward_per_step"]
    for label, k in [("Random", "random"),
                     ("Always Abstain", "always_abstain"),
                     ("PPO MLP", "ppo_mlp"),
                     ("RL²", "rl2"), ("RL²+HN", "rl2_hn"),
                     ("VariBAD", "varibad"),
                     ("V2:MuOnly", "v2_mu"),
                     ("V10:μ+σ", "v10_musigma"),
                     ("V14:FW", "v14_fastweight"),
                     ("V15:Plast", "v15_plasticity"),
                     ("V16:Attn", "v16_attention"),
                     ("V17:Lottery", "v17_lottery"),
                     ("V18:PredCod", "v18_predcoding"),
                     ("V19:WM", "v19_worldmodel"),
                     ("V20:Neuro", "v20_neuromod"),
                     ("V21:AttnHN", "v21_attn_hn"),
                     ("V22:Vel", "v22_posterior_vel"),
                     ("V23:Contr", "v23_contrastive"),
                     ("V24:AuxCls", "v24_aux_classifier"),
                     ("V25:MoE", "v25_moe_hn"),
                     ("V26:PredAttn", "v26_pred_attn"),
                     ("V27:Hindsight", "v27_hindsight"),
                     ("Bayes-Optimal", "bayes_optimal")]:
        if k not in out:
            continue
        rps = out[k]["reward_per_step"]
        print(f"  {label:<16} {rps:>12.4f} {rps - abstain_rps:>+11.4f} "
              f"{rps - bo_rps:>+12.4f}")
    print("=" * 68)
    print(f"\n  Plot: uv run python scripts/plot_bandit_active_sensing.py\n")


if __name__ == "__main__":
    main()
