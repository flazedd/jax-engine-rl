#!/usr/bin/env python3
"""Validate meta-RL agents on Windy Chain MDP.

7-state chain with 3 hidden regimes.  Each regime determines:
  - a goal position (0, 3, or 6)
  - a stochastic wind pattern (strong rightward / calm / strong leftward)
  - 4D side-signal means (regime-dependent Gaussian noise)

Three mechanisms give VariBAD(+HN) a learning advantage over RL²(+HN):
  1. Transition dynamics (wind) carry dense regime info at every step.
  2. Side signals (4D Gaussian) provide additional regime-dependent obs.
  3. Reward is Bernoulli(0.4) at goal — noisy and sparse RL signal.
VariBAD's ELBO decoder gets supervised gradients on transitions AND
side signals at every step.  RL²(+HN) must discover all of this
purely through RL credit assignment (slower).

Expected hierarchy:
  Random / PPO MLP  << RL² ≤ RL²+HN < VariBAD ≤ VariBAD+HN

Usage:
    uv run python scripts/validate_windy_chain.py --fast
    uv run python scripts/validate_windy_chain.py
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

from envs.windy_chain import (WindyChainParams, GOALS, WIND_PROBS,
                               SIDE_DIM, env_reset, env_step)
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

N_STATES = 7
OBS_SIZE = N_STATES + SIDE_DIM  # one_hot(pos, 7) + 4D side signals = 11
N_ACTIONS = 3                   # left=0, stay=1, right=2
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1  # aug = (obs, prev_act_oh, prev_rew)
HIDDEN_SIZE = 64
EPISODES_PER_TRIAL = 4

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "windy_chain_validation.json")

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Oracle baseline (knows regime → knows goal + optimal action)
# ---------------------------------------------------------------------------

def oracle_action(pos, regime):
    """Optimal oracle action given known regime.

    Regime 0: always left  — clip at pos=0 neutralises rightward wind.
    Regime 1: greedy toward centre (goal=3), stay at goal.
    Regime 2: always right — clip at pos=6 neutralises leftward wind.
    """
    goal = GOALS[regime]
    return jnp.where(
        regime == 0, jnp.int32(0),    # always left
        jnp.where(
            regime == 2, jnp.int32(2),  # always right
            jnp.where(pos < goal, jnp.int32(2),
                       jnp.where(pos > goal, jnp.int32(0),
                                  jnp.int32(1)))))


def oracle_per_step(key, params, n_episodes):
    """Returns (t_episode,) mean reward per step for oracle policy."""
    def single_episode(key):
        k_reset, k_steps = jax.random.split(key)
        state, _ = env_reset(k_reset, params)
        step_keys = jax.random.split(k_steps, params.t_episode)

        def scan_fn(state, k):
            action = oracle_action(state.pos, state.regime)
            new_state, _, reward, _, _ = env_step(k, state, action, params)
            return new_state, reward

        _, rewards = jax.lax.scan(scan_fn, state, step_keys)
        return rewards

    keys = jax.random.split(key, n_episodes)
    return jnp.mean(jax.vmap(single_episode)(keys), axis=0)


# ---------------------------------------------------------------------------
# Trajectory storage
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
# PPO MLP — rollout, training, evaluation
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


def train_ppo(key, params, n_iters, n_envs=512):
    k_model, key = jax.random.split(key)
    model = ActorCritic(OBS_SIZE, N_ACTIONS, hidden=64, key=k_model)
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
            k_coll, model, params, n_envs)
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
# Recurrent agents — generic rollout, training, evaluation
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


def train_recurrent(key, model, loss_fn, params, n_iters, n_trials=128,
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
        traj = collect_recurrent_rollout(k_coll, model, params, n_trials)
        history.append(float(jnp.mean(traj.rewards)))

        bootstrap = traj.values[-1]
        adv, ret = batch_gae(traj.rewards, traj.values, traj.dones,
                             bootstrap, 0.99, 0.95)

        # ELBO update (VariBAD / VariBAD+HN only)
        if elbo_fn is not None:
            elbo_data = (traj.obs, traj.actions, traj.rewards, traj.next_obs,
                         traj.prev_actions_oh, traj.prev_rewards)
            model, opt_st = elbo_update(model, opt_st, elbo_data, k_elbo)

        # PPO updates
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
    parser = argparse.ArgumentParser(
        description="Validate meta-RL agents on Windy Chain MDP")
    parser.add_argument("--fast", action="store_true",
                        help="Quick validation (fewer iters/episodes)")
    args = parser.parse_args()

    if args.fast:
        n_iters_ppo = 150
        n_iters_rec = 500
        n_trials = 128
        n_eval = 3000
    else:
        n_iters_ppo = 300
        n_iters_rec = 1000
        n_trials = 128
        n_eval = 5000

    params = WindyChainParams()
    key = jax.random.PRNGKey(42)

    print("=" * 65)
    print("  Windy Chain MDP — Transition-Informative Validation")
    print("=" * 65)
    print(f"  N={params.n_states} states, T={params.t_episode} steps/episode,"
          f" {params.n_regimes} regimes")
    print(f"  Goals: {list(np.array(GOALS))}, "
          f"reward=Bernoulli({params.reward_prob})")
    print(f"  Wind:  regime 0 → right w.p.0.5, regime 1 → calm,"
          f" regime 2 → left w.p.0.5")
    print(f"  Side signals: 4D Gaussian, σ={params.side_sigma}")
    print(f"  Obs dim: {OBS_SIZE} (7 pos + 4 side)")
    print(f"  Recurrent: {EPISODES_PER_TRIAL} episodes/trial,"
          f" hidden={HIDDEN_SIZE}")
    print()

    out = {}

    # --- Random baseline ---
    # Random gets reward when randomly landing on goal: ~1/7 * 1/3 * adjusted
    # Simplest: just simulate
    print("  Computing baselines ...")
    k_rand, key = jax.random.split(key)

    def random_per_step(key, n_episodes):
        def single(key):
            k_r, k_s = jax.random.split(key)
            state, _ = env_reset(k_r, params)
            skeys = jax.random.split(k_s, params.t_episode)
            def step(state, ks):
                ka, ke = jax.random.split(ks)
                action = jax.random.randint(ka, (), 0, N_ACTIONS)
                ns, _, rew, _, _ = env_step(ke, state, action, params)
                return ns, rew
            _, rews = jax.lax.scan(step, state, skeys)
            return rews
        keys = jax.random.split(key, n_episodes)
        return jnp.mean(jax.vmap(single)(keys), axis=0)

    rand_curve = np.array(random_per_step(k_rand, n_eval))
    rand_rps = float(rand_curve.mean())
    out["random"] = {"reward_per_step": rand_rps,
                     "per_step_curve": rand_curve.tolist()}

    # --- Oracle ---
    k_or, key = jax.random.split(key)
    oracle_curve = np.array(oracle_per_step(k_or, params, n_eval))
    oracle_rps = float(oracle_curve.mean())
    out["oracle"] = {"reward_per_step": oracle_rps,
                     "per_step_curve": oracle_curve.tolist()}
    print(f"    Random:  {rand_rps:.4f} reward/step")
    print(f"    Oracle:  {oracle_rps:.4f} reward/step")

    # --- PPO MLP ---
    print("\n  Training agents ...")
    k_ppo, key = jax.random.split(key)
    ppo_model, ppo_hist = train_ppo(k_ppo, params, n_iters_ppo)
    k_eval, key = jax.random.split(key)
    ppo_curve = np.array(eval_ppo_per_step(ppo_model, params, k_eval, n_eval))
    out["ppo_mlp"] = {"n_iters": n_iters_ppo,
                      "train_history": ppo_hist,
                      "reward_per_step": float(ppo_curve.mean()),
                      "per_step_curve": ppo_curve.tolist()}

    # --- RL² ---
    k_rl2, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_rl2)
    rl2_model = GRUActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS, hidden_size=HIDDEN_SIZE, key=k_m)
    rl2_model, rl2_hist = train_recurrent(
        k_t, rl2_model, rl2_loss_fn, params, n_iters_rec,
        n_trials=n_trials, label="RL²")
    k_eval, key = jax.random.split(key)
    rl2_curve = np.array(eval_recurrent_per_step(
        rl2_model, params, k_eval, n_eval))
    out["rl2"] = {"n_iters": n_iters_rec,
                  "train_history": rl2_hist,
                  "reward_per_step": float(rl2_curve.mean()),
                  "per_step_curve": rl2_curve.tolist()}

    # --- RL²+HN ---
    k_hn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_hn)
    hn_model = HNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, policy_hidden=16, key=k_m)
    hn_model, hn_hist = train_recurrent(
        k_t, hn_model, rl2_hn_loss_fn, params, n_iters_rec,
        n_trials=n_trials, label="RL²+HN")
    k_eval, key = jax.random.split(key)
    hn_curve = np.array(eval_recurrent_per_step(
        hn_model, params, k_eval, n_eval))
    out["rl2_hn"] = {"n_iters": n_iters_rec,
                     "train_history": hn_hist,
                     "reward_per_step": float(hn_curve.mean()),
                     "per_step_curve": hn_curve.tolist()}

    # --- VariBAD ---
    k_vb, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vb)
    vb_model = VariBADActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=4, key=k_m)
    vb_model, vb_hist = train_recurrent(
        k_t, vb_model, varibad_ppo_loss_fn, params, n_iters_rec,
        n_trials=n_trials, elbo_fn=elbo_loss_fn, label="VariBAD")
    k_eval, key = jax.random.split(key)
    vb_curve = np.array(eval_recurrent_per_step(
        vb_model, params, k_eval, n_eval))
    out["varibad"] = {"n_iters": n_iters_rec,
                      "train_history": vb_hist,
                      "reward_per_step": float(vb_curve.mean()),
                      "per_step_curve": vb_curve.tolist()}

    # --- VariBAD+HN ---
    k_vbhn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vbhn)
    vbhn_model = VariBADHNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=4, policy_hidden=16, key=k_m)
    vbhn_model, vbhn_hist = train_recurrent(
        k_t, vbhn_model, varibad_hn_ppo_loss_fn, params, n_iters_rec,
        n_trials=n_trials, elbo_fn=varibad_hn_elbo_loss_fn, label="VariBAD+HN")
    k_eval, key = jax.random.split(key)
    vbhn_curve = np.array(eval_recurrent_per_step(
        vbhn_model, params, k_eval, n_eval))
    out["varibad_hn"] = {"n_iters": n_iters_rec,
                         "train_history": vbhn_hist,
                         "reward_per_step": float(vbhn_curve.mean()),
                         "per_step_curve": vbhn_curve.tolist()}

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: {os.path.relpath(OUT_JSON, ROOT)}")

    # --- Summary table ---
    print("\n" + "=" * 65)
    print(f"  {'Agent':<14} {'reward/step':>12} {'vs Random':>10} "
          f"{'vs Oracle':>10}")
    print("  " + "-" * 61)
    for label, k in [("Random", "random"), ("PPO MLP", "ppo_mlp"),
                     ("RL²", "rl2"), ("RL²+HN", "rl2_hn"),
                     ("VariBAD", "varibad"), ("VariBAD+HN", "varibad_hn"),
                     ("Oracle", "oracle")]:
        rps = out[k]["reward_per_step"]
        print(f"  {label:<14} {rps:>12.4f} {rps - rand_rps:>+10.4f} "
              f"{rps - oracle_rps:>+10.4f}")
    print("=" * 65)

    # --- Key comparison ---
    hn_rps = out["rl2_hn"]["reward_per_step"]
    vbhn_rps = out["varibad_hn"]["reward_per_step"]
    gap = vbhn_rps - hn_rps
    print(f"\n  RL²+HN vs VariBAD+HN gap: {gap:+.4f} reward/step")
    if gap > 0.01:
        print("  --> VariBAD+HN outperforms RL²+HN"
              " (ELBO exploits transition dynamics)")
    elif gap < -0.01:
        print("  --> RL²+HN outperforms VariBAD+HN (unexpected)")
    else:
        print("  --> No clear gap (consider increasing n_iters or wind_prob)")

    # --- Per-step curves (abbreviated) ---
    agents = ["Random", "Oracle", "PPO MLP", "RL²", "RL²+HN",
              "VariBAD", "VariBAD+HN"]
    keys_map = ["random", "oracle", "ppo_mlp", "rl2", "rl2_hn",
                "varibad", "varibad_hn"]
    print(f"\n  Per-step reward (T={params.t_episode}, every 5 steps):")
    header = f"  {'step':>4}"
    for name in agents:
        header += f"  {name:>10}"
    print(header)
    print("  " + "-" * (4 + len(agents) * 12))
    for t in range(0, params.t_episode, 5):
        row = f"  {t + 1:>4}"
        for k in keys_map:
            row += f"  {out[k]['per_step_curve'][t]:>10.3f}"
        print(row)
    # Final step
    t = params.t_episode - 1
    row = f"  {t + 1:>4}"
    for k in keys_map:
        row += f"  {out[k]['per_step_curve'][t]:>10.3f}"
    print(row)

    print()


if __name__ == "__main__":
    main()
