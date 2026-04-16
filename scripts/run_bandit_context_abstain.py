#!/usr/bin/env python3
"""Train agents on Contextual Bandit with Abstain action (calibration stress test).

Adds a fixed-payoff abstain action. Optimal policy: abstain while uncertain,
commit once posterior confidence exceeds threshold. Tests whether agents
develop calibrated uncertainty — overconfident agents commit too early,
underconfident agents abstain too long.

Saves to results/bandit_context_abstain.json.
Plot with: uv run python scripts/plot_bandit_context_abstain.py

Usage:
    uv run python scripts/run_bandit_context_abstain.py --agent ppo_mlp
    uv run python scripts/run_bandit_context_abstain.py --agent all
    uv run python scripts/run_bandit_context_abstain.py --agent all --fast
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
from envs.bandit_context_abstain import (BanditAbstainParams, ABSTAIN_PAYOFF,
                                          env_reset, env_step)
from agents.ppo import ActorCritic, actor_loss_fn, critic_loss_fn
from agents.rl2 import GRUActorCritic, rl2_loss_fn
from agents.rl2_hn import HNActorCritic, rl2_hn_loss_fn
from agents.varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
from agents.varibad_hn_v2_mu import (V2MuOnlyActorCritic, v2_mu_ppo_loss_fn,
                                      v2_mu_elbo_loss_fn)
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
OUT_JSON = os.path.join(RESULTS_DIR, "bandit_context_abstain.json")

ALL_AGENTS = ["ppo_mlp", "rl2", "rl2_hn", "varibad", "v2_mu"]

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def bayes_optimal_per_step(key, params, n_episodes):
    """Bayes-optimal: exact posterior over tasks, abstain vs pull decision."""
    sigma2 = params.side_sigma ** 2

    def single_episode(key):
        k_task, k_steps = jax.random.split(key)
        task_id = jax.random.randint(k_task, (), 0, params.n_arms)
        arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
        arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
        step_keys = jax.random.split(k_steps, params.t_episode)
        log_prior = jnp.log(jnp.ones(params.n_arms) / params.n_arms)

        def scan_fn(carry, k):
            log_post, arm_probs = carry
            k_side, k_pull = jax.random.split(k)
            side = SIDE_MEANS[task_id] + params.side_sigma * jax.random.normal(
                k_side, (params.side_dim,))
            log_liks = -0.5 * jnp.sum(
                (side[None, :] - SIDE_MEANS) ** 2, axis=-1) / sigma2
            log_post = log_post + log_liks
            log_post = log_post - jax.nn.logsumexp(log_post)
            post = jnp.exp(log_post)
            # Expected reward from pulling MAP arm
            map_arm = jnp.argmax(post)
            e_pull = post[map_arm] * params.arm_prob_high + \
                     (1.0 - post[map_arm]) * params.arm_prob_low
            # Abstain if expected pull reward < abstain payoff
            should_abstain = e_pull < params.abstain_payoff
            reward_pull = jax.random.bernoulli(
                k_pull, arm_probs[map_arm]).astype(jnp.float32)
            reward = jnp.where(should_abstain, params.abstain_payoff, reward_pull)
            action = jnp.where(should_abstain, params.n_arms, map_arm)
            return (log_post, arm_probs), (reward, action)

        _, (rewards, actions) = jax.lax.scan(
            scan_fn, (log_prior, arm_probs), step_keys)
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
# Main
# ---------------------------------------------------------------------------

def main():
    global N_ITERS_PPO, N_ITERS_REC, N_EVAL, N_ENVS, N_TRIALS

    parser = argparse.ArgumentParser(
        description="Contextual Bandit with Abstain (calibration stress test)")
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
    params = BanditAbstainParams()

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
    print(f"  Calibration Stress: Bandit + Abstain{suffix}")
    print("=" * 60)
    print(f"  K={params.n_arms} arms + 1 abstain, T={params.t_episode} steps")
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
                     ("Bayes-Optimal", "bayes_optimal")]:
        if k not in out:
            continue
        rps = out[k]["reward_per_step"]
        print(f"  {label:<16} {rps:>12.4f} {rps - abstain_rps:>+11.4f} "
              f"{rps - bo_rps:>+12.4f}")
    print("=" * 68)
    print(f"\n  Plot: uv run python scripts/plot_bandit_context_abstain.py\n")


if __name__ == "__main__":
    main()
