#!/usr/bin/env python3
"""Train agents on 2D Point Navigation with hidden goal.

Classic VariBAD dummy task (discretized, 4-compass + stay). Validates that
the ladder (PPO MLP vs RL²+HN vs VariBAD vs V2:μOnly) travels across task
families. Saves results to results/point_nav.json.

Usage:
    uv run python scripts/run_point_nav.py --agent ppo_mlp
    uv run python scripts/run_point_nav.py --agent all
    uv run python scripts/run_point_nav.py --agent all --fast
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

from envs.point_nav import (PointNavParams, env_reset, env_step,
                             N_ACTIONS, OBS_SIZE)
from agents.ppo import ActorCritic, actor_loss_fn, critic_loss_fn
from agents.rl2_hn import HNActorCritic, rl2_hn_loss_fn
from agents.varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
from agents.varibad_hn_v2_mu import (V2MuOnlyActorCritic, v2_mu_ppo_loss_fn,
                                      v2_mu_elbo_loss_fn)
from agents.amago_hn import AMAGOHNActorCritic, amago_hn_loss_fn
from agents.common import batch_gae, NumpyEncoder

INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1    # 2 + 5 + 1 = 8
HIDDEN_SIZE = 64
LATENT_DIM = 4
POLICY_HIDDEN = 16
EPISODES_PER_TRIAL = 4

N_ITERS_PPO = 300
N_ITERS_REC = 300
N_EVAL = 2000
N_ENVS = 512
N_TRIALS = 128

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "point_nav.json")

ALL_AGENTS = ["ppo_mlp", "rl2_hn", "varibad", "v2_mu", "amago"]

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


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
        if it % max(1, n_iters // 20) == 0 or it == n_iters - 1:
            print(f"    PPO MLP  iter {it+1:4d}/{n_iters} "
                  f"rew/step={history[-1]:+.4f}  [{time.time() - t0:.0f}s]")
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
    all_r = jax.vmap(single_episode)(keys)
    return jnp.mean(all_r, axis=0)


# ---------------------------------------------------------------------------
# Recurrent agents (GRU / HN / VariBAD / V2)
# ---------------------------------------------------------------------------

def collect_recurrent_rollout(key, model, params, n_trials):
    n_steps = EPISODES_PER_TRIAL * params.t_episode
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_trials)
    states, obs = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    prev_act = jnp.zeros((n_trials, N_ACTIONS))
    prev_rew = jnp.zeros((n_trials, 1))
    h0 = model.init_state()
    gru_st = jnp.broadcast_to(h0, (n_trials,) + h0.shape)
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
        if it % max(1, n_iters // 20) == 0 or it == n_iters - 1:
            print(f"    {label:<12s} iter {it+1:4d}/{n_iters} "
                  f"rew/step={history[-1]:+.4f}  [{time.time() - t0:.0f}s]")
    return model, history


def eval_recurrent_per_step(model, params, key, n_episodes):
    t = params.t_episode
    def single_episode(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)
        h = model.init_state()
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
    all_r = jax.vmap(single_episode)(keys)
    return jnp.mean(all_r, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global N_ITERS_PPO, N_ITERS_REC, N_EVAL, N_ENVS, N_TRIALS

    parser = argparse.ArgumentParser(description="2D Point Navigation")
    parser.add_argument("--fast", action="store_true", help="Quick smoke test")
    parser.add_argument("--agent", type=str, default="all",
                        choices=ALL_AGENTS + ["all"])
    args = parser.parse_args()

    if args.fast:
        N_ITERS_PPO = 60
        N_ITERS_REC = 60
        N_EVAL = 500
        N_ENVS = 128
        N_TRIALS = 64

    run_agent = args.agent
    agents_to_run = ALL_AGENTS if run_agent == "all" else [run_agent]
    params = PointNavParams()

    # Always merge-load any existing results, so agents accumulate over runs.
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            out = json.load(f)
    else:
        out = {}

    os.makedirs(RESULTS_DIR, exist_ok=True)

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2, cls=NumpyEncoder)
        print(f"    -> saved to {os.path.relpath(OUT_JSON, ROOT)}")

    print("=" * 60)
    suffix = " (FAST)" if args.fast else ""
    if run_agent != "all":
        suffix += f" agent={run_agent}"
    print(f"  Point Navigation{suffix}")
    print("=" * 60)
    print(f"  actions={params.n_actions}, T={params.t_episode}, "
          f"step={params.step_size}, goal_radius={params.goal_radius}, "
          f"tol={params.goal_tolerance}")
    print(f"  PPO MLP: {N_ITERS_PPO} iters | Recurrent: {N_ITERS_REC} iters")
    print(f"  Eval: {N_EVAL} episodes")
    print()

    key = jax.random.PRNGKey(42)

    # Random-policy baseline (Monte Carlo)
    k_rand, key = jax.random.split(key)
    def random_episode(key):
        k_reset, k_steps, k_act = jax.random.split(key, 3)
        state, obs = env_reset(k_reset, params)
        step_keys = jax.random.split(k_steps, params.t_episode)
        act_keys = jax.random.split(k_act, params.t_episode)
        def scan_fn(carry, ks):
            state, obs = carry
            k_s, k_a = ks
            action = jax.random.randint(k_a, (), 0, params.n_actions)
            new_s, new_o, rew, _, _ = env_step(k_s, state, action, params)
            return (new_s, new_o), rew
        _, rews = jax.lax.scan(scan_fn, (state, obs), (step_keys, act_keys))
        return rews
    rand_keys = jax.random.split(k_rand, N_EVAL)
    rand_rew = jax.vmap(random_episode)(rand_keys)
    random_rps = float(jnp.mean(rand_rew))
    rand_curve = jnp.mean(rand_rew, axis=0)
    out["random"] = {"reward_per_step": random_rps,
                     "per_step_curve": np.array(rand_curve).tolist()}
    print(f"  Random policy: {random_rps:.4f} reward/step")
    save()

    def should_run(name):
        return name in agents_to_run

    print("\n  Training agents ...")

    # PPO MLP
    k_ppo, key = jax.random.split(key)
    k_eval_ppo, key = jax.random.split(key)
    if should_run("ppo_mlp"):
        print("\n  [ppo_mlp] training ...")
        ppo_model, ppo_hist = train_ppo(k_ppo, params, N_ITERS_PPO)
        ppo_r = eval_ppo_per_step(ppo_model, params, k_eval_ppo, N_EVAL)
        out["ppo_mlp"] = {"n_iters": N_ITERS_PPO, "train_history": ppo_hist,
                          "reward_per_step": float(jnp.mean(ppo_r)),
                          "per_step_curve": np.array(ppo_r).tolist()}
        print(f"    eval rew/step = {out['ppo_mlp']['reward_per_step']:.4f}")
        save()

    # RL²+HN
    k_hn, key = jax.random.split(key)
    k_eval_hn, key = jax.random.split(key)
    if should_run("rl2_hn"):
        print("\n  [rl2_hn] training ...  (ent_coef=0.05)")
        k_m, k_t = jax.random.split(k_hn)
        hn_model = HNActorCritic(INPUT_SIZE, OBS_SIZE, N_ACTIONS,
                                  hidden_size=HIDDEN_SIZE,
                                  policy_hidden=POLICY_HIDDEN, key=k_m)
        rl2_hn_loss_hot = lambda m, b: rl2_hn_loss_fn(m, b, ent_coef=0.05)
        hn_model, hn_hist = train_recurrent(
            k_t, hn_model, rl2_hn_loss_hot, params, N_ITERS_REC, label="RL²+HN")
        hn_r = eval_recurrent_per_step(hn_model, params, k_eval_hn, N_EVAL)
        out["rl2_hn"] = {"n_iters": N_ITERS_REC, "train_history": hn_hist,
                         "reward_per_step": float(jnp.mean(hn_r)),
                         "per_step_curve": np.array(hn_r).tolist()}
        print(f"    eval rew/step = {out['rl2_hn']['reward_per_step']:.4f}")
        save()

    # VariBAD
    k_vb, key = jax.random.split(key)
    k_eval_vb, key = jax.random.split(key)
    if should_run("varibad"):
        print("\n  [varibad] training ...")
        k_m, k_t = jax.random.split(k_vb)
        vb_model = VariBADActorCritic(INPUT_SIZE, OBS_SIZE, N_ACTIONS,
                                       hidden_size=HIDDEN_SIZE,
                                       latent_dim=LATENT_DIM, key=k_m)
        vb_model, vb_hist = train_recurrent(
            k_t, vb_model, varibad_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=elbo_loss_fn, label="VariBAD")
        vb_r = eval_recurrent_per_step(vb_model, params, k_eval_vb, N_EVAL)
        out["varibad"] = {"n_iters": N_ITERS_REC, "train_history": vb_hist,
                          "reward_per_step": float(jnp.mean(vb_r)),
                          "per_step_curve": np.array(vb_r).tolist()}
        print(f"    eval rew/step = {out['varibad']['reward_per_step']:.4f}")
        save()

    # V2:μOnly
    k_v2, key = jax.random.split(key)
    k_eval_v2, key = jax.random.split(key)
    if should_run("v2_mu"):
        print("\n  [v2_mu] training ...")
        k_m, k_t = jax.random.split(k_v2)
        v2_model = V2MuOnlyActorCritic(INPUT_SIZE, OBS_SIZE, N_ACTIONS,
                                        hidden_size=HIDDEN_SIZE,
                                        latent_dim=LATENT_DIM,
                                        policy_hidden=POLICY_HIDDEN, key=k_m)
        v2_model, v2_hist = train_recurrent(
            k_t, v2_model, v2_mu_ppo_loss_fn, params, N_ITERS_REC,
            elbo_fn=v2_mu_elbo_loss_fn, label="V2:MuOnly")
        v2_r = eval_recurrent_per_step(v2_model, params, k_eval_v2, N_EVAL)
        out["v2_mu"] = {"n_iters": N_ITERS_REC, "train_history": v2_hist,
                        "reward_per_step": float(jnp.mean(v2_r)),
                        "per_step_curve": np.array(v2_r).tolist()}
        print(f"    eval rew/step = {out['v2_mu']['reward_per_step']:.4f}")
        save()

    # AMAGO (causal transformer over sliding context)
    k_am, key = jax.random.split(key)
    k_eval_am, key = jax.random.split(key)
    if should_run("amago"):
        print("\n  [amago] training ...  (ctx=64, d_model=64, 1 layer)")
        k_m, k_t = jax.random.split(k_am)
        am_model = AMAGOHNActorCritic(INPUT_SIZE, OBS_SIZE, N_ACTIONS,
                                       context_len=64, d_model=64,
                                       n_heads=4, d_ff=128,
                                       policy_hidden=POLICY_HIDDEN, key=k_m)
        am_model, am_hist = train_recurrent(
            k_t, am_model, amago_hn_loss_fn, params, N_ITERS_REC, label="AMAGO")
        am_r = eval_recurrent_per_step(am_model, params, k_eval_am, N_EVAL)
        out["amago"] = {"n_iters": N_ITERS_REC, "train_history": am_hist,
                        "reward_per_step": float(jnp.mean(am_r)),
                        "per_step_curve": np.array(am_r).tolist()}
        print(f"    eval rew/step = {out['amago']['reward_per_step']:.4f}")
        save()

    print(f"\n  Results stored at: {os.path.relpath(OUT_JSON, ROOT)}")

    print("\n" + "=" * 60)
    print(f"  {'Agent':<14} {'reward/step':>12} {'vs Random':>11}")
    print("  " + "-" * 54)
    for label, k in [("Random", "random"),
                     ("PPO MLP", "ppo_mlp"),
                     ("RL²+HN", "rl2_hn"),
                     ("VariBAD", "varibad"),
                     ("V2:MuOnly", "v2_mu"),
                     ("AMAGO", "amago")]:
        if k not in out:
            continue
        rps = out[k]["reward_per_step"]
        print(f"  {label:<14} {rps:>12.4f} {rps - random_rps:>+11.4f}")
    print("=" * 60)
    print(f"\n  Plot: uv run python scripts/plot_point_nav.py\n")


if __name__ == "__main__":
    main()
