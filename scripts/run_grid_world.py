#!/usr/bin/env python3
"""RL²+HN, VariBAD & VariBAD+HN on Grid World (GridNavi from Zintgraf et al., 2020).

5×5 grid, 8 possible goal positions (corners + edge midpoints).
Goal is hidden and fixed within a 4-episode trial. Agent starts at
center (2,2) each episode and must learn to find the goal.

Key result: all agents explore in episode 1 and exploit in episodes 2-4,
demonstrating the explore→exploit meta-learning pattern.

Produces:
    results/grid_world.json

Usage:
    uv run python scripts/run_grid_world.py --fast
    uv run python scripts/run_grid_world.py
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

from envs.grid_world import (GridWorldParams, GridWorldState,
                              GOAL_POSITIONS, _make_obs,
                              env_reset, env_step, env_episode_reset)
from agents.varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
from agents.varibad_hn import (VariBADHNActorCritic, varibad_hn_elbo_loss_fn,
                                varibad_hn_ppo_loss_fn)
from agents.rl2_hn import HNActorCritic, rl2_hn_loss_fn
from agents.common import batch_gae, NumpyEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_SIZE = 2                    # (x, y) normalized
N_ACTIONS = 5                   # up, down, left, right, stay
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1   # aug = (obs, prev_act_oh, prev_rew)
HIDDEN_SIZE = 128
LATENT_DIM = 4
POLICY_HIDDEN = 16
EPISODES_PER_TRIAL = 4

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "grid_world.json")

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


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
# Rollout collection (goal fixed within trial)
# ---------------------------------------------------------------------------

def collect_rollout(key, model, params, n_trials):
    """Collect multi-episode trial rollouts with goal persistence."""
    n_steps = EPISODES_PER_TRIAL * params.t_episode
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_trials)
    states, obs = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    prev_act = jnp.zeros((n_trials, N_ACTIONS))
    prev_rew = jnp.zeros((n_trials, 1))
    gru_st = jnp.zeros((n_trials, model.hidden_size))
    all_keys = jax.random.split(k_roll, n_steps * 2).reshape(n_steps, 2, -1)

    def scan_step(carry, keys_t):
        states, obs, prev_act, prev_rew, gru_st = carry
        k_act, k_env = keys_t[0], keys_t[1]
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
        next_obs = new_o  # pre-reset, for ELBO

        # Episode reset: keep goal, reset position to center
        def maybe_reset(done, ns, no):
            rs, ro = env_episode_reset(ns, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset)(dones, new_s, new_o)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_recurrent(key, model, ppo_loss_fn, elbo_fn, params, n_iters,
                    n_trials=128, label="recurrent"):
    """Train a recurrent agent on grid world. Returns (model, history)."""
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    opt_st = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def ppo_update(model, opt_st, batch):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: ppo_loss_fn(m, batch))(model)
        updates, new_opt = optimizer.update(
            grads, opt_st, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt

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
        traj = collect_rollout(k_coll, model, params, n_trials)
        history.append(float(jnp.mean(traj.rewards)))

        bootstrap = traj.values[-1]
        adv, ret = batch_gae(traj.rewards, traj.values, traj.dones,
                             bootstrap, 0.99, 0.95)

        # ELBO update (only for models with world model, e.g. VariBAD)
        if elbo_fn is not None:
            elbo_data = (traj.obs, traj.actions, traj.rewards, traj.next_obs,
                         traj.prev_actions_oh, traj.prev_rewards)
            model, opt_st = elbo_update(model, opt_st, elbo_data, k_elbo)

        # PPO minibatch updates
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
# Evaluation (full multi-episode trials)
# ---------------------------------------------------------------------------

def eval_random_trial(params, key, n_trials):
    """Random policy over full trial. Returns (n_steps_total,) rewards."""
    n_steps = EPISODES_PER_TRIAL * params.t_episode

    def single_trial(key):
        k_reset, k_steps = jax.random.split(key)
        state, _ = env_reset(k_reset, params)
        step_keys = jax.random.split(k_steps, n_steps)

        def scan_fn(state, ks):
            ka, ke = jax.random.split(ks)
            action = jax.random.randint(ka, (), 0, params.n_actions)
            new_s, _, rew, done, _ = env_step(ke, state, action, params)
            rs, _ = env_episode_reset(new_s, params)
            final_s = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), rs, new_s)
            return final_s, rew

        _, rewards = jax.lax.scan(scan_fn, state, step_keys)
        return rewards

    keys = jax.random.split(key, n_trials)
    return jnp.mean(jax.vmap(single_trial)(keys), axis=0)


def eval_oracle_trial(params, key, n_trials):
    """Oracle (knows goal) over full trial. Returns (n_steps_total,) rewards."""
    n_steps = EPISODES_PER_TRIAL * params.t_episode

    def single_trial(key):
        k_reset, k_steps = jax.random.split(key)
        state, _ = env_reset(k_reset, params)
        step_keys = jax.random.split(k_steps, n_steps)

        def scan_fn(state, k):
            goal = GOAL_POSITIONS[state.goal_idx]
            # Greedy: move toward goal (x first, then y)
            action = jnp.where(
                state.pos_x < goal[0], 3,      # right
                jnp.where(state.pos_x > goal[0], 2,   # left
                    jnp.where(state.pos_y < goal[1], 1,   # down
                        jnp.where(state.pos_y > goal[1], 0,   # up
                            4))))               # stay
            new_s, _, rew, done, _ = env_step(
                k, state, jnp.int32(action), params)
            rs, _ = env_episode_reset(new_s, params)
            final_s = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), rs, new_s)
            return final_s, rew

        _, rewards = jax.lax.scan(scan_fn, state, step_keys)
        return rewards

    keys = jax.random.split(key, n_trials)
    return jnp.mean(jax.vmap(single_trial)(keys), axis=0)


def eval_varibad_trial(model, params, key, n_trials):
    """Greedy VariBAD over full trial. Returns (n_steps_total,) rewards."""
    n_steps = EPISODES_PER_TRIAL * params.t_episode

    def single_trial(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)
        h = jnp.zeros(model.hidden_size)
        prev_act = jnp.zeros(N_ACTIONS)
        prev_rew = jnp.zeros(1)
        step_keys = jax.random.split(k_steps, n_steps)

        def scan_fn(carry, k):
            state, obs, h, prev_act, prev_rew = carry
            aug = jnp.concatenate([obs, prev_act, prev_rew])
            logits, _, new_h = model.forward_step(aug, obs, h)
            action = jnp.argmax(logits)
            new_s, new_o, rew, done, _ = env_step(k, state, action, params)
            # Episode reset: keep goal
            rs, ro = env_episode_reset(new_s, params)
            final_s = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), rs, new_s)
            final_o = jnp.where(done, ro, new_o)
            carry = (final_s, final_o, new_h,
                     jax.nn.one_hot(action, N_ACTIONS), rew[None])
            return carry, rew

        _, rewards = jax.lax.scan(
            scan_fn, (state, obs, h, prev_act, prev_rew), step_keys)
        return rewards

    keys = jax.random.split(key, n_trials)
    return jnp.mean(jax.vmap(single_trial)(keys), axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RL²+HN, VariBAD & VariBAD+HN on Grid World (GridNavi)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick run (fewer iters/eval episodes)")
    args = parser.parse_args()

    if args.fast:
        n_iters = 300
        n_trials_train = 64
        n_eval = 1000
    else:
        n_iters = 250
        n_trials_train = 128
        n_eval = 5000

    params = GridWorldParams()
    key = jax.random.PRNGKey(42)
    t_ep = params.t_episode
    n_steps = EPISODES_PER_TRIAL * t_ep

    print("=" * 65)
    print("  Grid World (GridNavi) — RL²+HN, VariBAD & VariBAD+HN")
    print("=" * 65)
    print(f"  {params.grid_size}×{params.grid_size} grid, "
          f"{params.n_goals} goal positions, T={t_ep} steps/episode")
    print(f"  {EPISODES_PER_TRIAL} episodes/trial, "
          f"goal fixed within trial")
    print(f"  Actions: up(0), down(1), left(2), right(3), stay(4)")
    print(f"  Obs: (x, y) normalized — {OBS_SIZE}D")
    print(f"  hidden={HIDDEN_SIZE}, latent_dim={LATENT_DIM}, "
          f"policy_hidden={POLICY_HIDDEN}")
    print()

    out = {}

    # --- Random baseline ---
    print("  Computing baselines ...")
    k_rand, key = jax.random.split(key)
    rand_curve = np.array(eval_random_trial(params, k_rand, n_eval))
    rand_rps = float(rand_curve.mean())
    rand_ep = rand_curve.reshape(EPISODES_PER_TRIAL, t_ep).mean(axis=1)
    out["random"] = {
        "reward_per_step": rand_rps,
        "per_step_curve": rand_curve.tolist(),
        "per_episode_curve": rand_ep.tolist(),
    }
    print(f"    Random:  {rand_rps:.4f} reward/step")

    # --- Oracle baseline ---
    k_or, key = jax.random.split(key)
    oracle_curve = np.array(eval_oracle_trial(params, k_or, n_eval))
    oracle_rps = float(oracle_curve.mean())
    oracle_ep = oracle_curve.reshape(EPISODES_PER_TRIAL, t_ep).mean(axis=1)
    out["oracle"] = {
        "reward_per_step": oracle_rps,
        "per_step_curve": oracle_curve.tolist(),
        "per_episode_curve": oracle_ep.tolist(),
    }
    print(f"    Oracle:  {oracle_rps:.4f} reward/step")

    # --- RL²+HN ---
    print("\n  Training agents ...")
    k_rl2hn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_rl2hn)
    rl2hn_model = HNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, policy_hidden=POLICY_HIDDEN, key=k_m)
    rl2hn_model, rl2hn_hist = train_recurrent(
        k_t, rl2hn_model, rl2_hn_loss_fn, None, params, n_iters,
        n_trials=n_trials_train, label="RL²+HN")

    k_eval, key = jax.random.split(key)
    rl2hn_curve = np.array(eval_varibad_trial(
        rl2hn_model, params, k_eval, n_eval))
    rl2hn_rps = float(rl2hn_curve.mean())
    rl2hn_ep = rl2hn_curve.reshape(EPISODES_PER_TRIAL, t_ep).mean(axis=1)
    out["rl2_hn"] = {
        "n_iters": n_iters,
        "train_history": rl2hn_hist,
        "reward_per_step": rl2hn_rps,
        "per_step_curve": rl2hn_curve.tolist(),
        "per_episode_curve": rl2hn_ep.tolist(),
    }

    # --- VariBAD ---
    k_vb, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vb)
    vb_model = VariBADActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM, key=k_m)
    vb_model, vb_hist = train_recurrent(
        k_t, vb_model, varibad_ppo_loss_fn, elbo_loss_fn, params, n_iters,
        n_trials=n_trials_train, label="VariBAD")

    k_eval, key = jax.random.split(key)
    vb_curve = np.array(eval_varibad_trial(vb_model, params, k_eval, n_eval))
    vb_rps = float(vb_curve.mean())
    vb_ep = vb_curve.reshape(EPISODES_PER_TRIAL, t_ep).mean(axis=1)
    out["varibad"] = {
        "n_iters": n_iters,
        "train_history": vb_hist,
        "reward_per_step": vb_rps,
        "per_step_curve": vb_curve.tolist(),
        "per_episode_curve": vb_ep.tolist(),
    }

    # --- VariBAD+HN ---
    k_vbhn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vbhn)
    vbhn_model = VariBADHNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM,
        policy_hidden=POLICY_HIDDEN, key=k_m)
    vbhn_model, vbhn_hist = train_recurrent(
        k_t, vbhn_model, varibad_hn_ppo_loss_fn, varibad_hn_elbo_loss_fn,
        params, n_iters, n_trials=n_trials_train, label="VariBAD+HN")

    k_eval, key = jax.random.split(key)
    vbhn_curve = np.array(eval_varibad_trial(
        vbhn_model, params, k_eval, n_eval))
    vbhn_rps = float(vbhn_curve.mean())
    vbhn_ep = vbhn_curve.reshape(EPISODES_PER_TRIAL, t_ep).mean(axis=1)
    out["varibad_hn"] = {
        "n_iters": n_iters,
        "train_history": vbhn_hist,
        "reward_per_step": vbhn_rps,
        "per_step_curve": vbhn_curve.tolist(),
        "per_episode_curve": vbhn_ep.tolist(),
    }

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved: {os.path.relpath(OUT_JSON, ROOT)}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print(f"  {'Agent':<14} {'reward/step':>12} {'vs Random':>10} "
          f"{'vs Oracle':>10}")
    print("  " + "-" * 58)
    for label, k in [("Random", "random"), ("RL²+HN", "rl2_hn"),
                     ("VariBAD", "varibad"), ("VariBAD+HN", "varibad_hn"),
                     ("Oracle", "oracle")]:
        rps = out[k]["reward_per_step"]
        print(f"  {label:<14} {rps:>12.4f} {rps - rand_rps:>+10.4f} "
              f"{rps - oracle_rps:>+10.4f}")
    print("=" * 72)

    # --- Per-episode reward ---
    print(f"\n  Per-episode reward within trial "
          f"({EPISODES_PER_TRIAL} episodes × {t_ep} steps):")
    print(f"  {'episode':>8}  {'Random':>8}  {'RL²+HN':>8}  {'VariBAD':>10}  "
          f"{'VariBAD+HN':>10}  {'Oracle':>8}")
    print("  " + "-" * 62)
    for ep in range(EPISODES_PER_TRIAL):
        print(f"  {ep + 1:>8}  {rand_ep[ep]:>8.4f}  "
              f"{rl2hn_ep[ep]:>8.4f}  "
              f"{vb_ep[ep]:>10.4f}  {vbhn_ep[ep]:>10.4f}  "
              f"{oracle_ep[ep]:>8.4f}")

    # --- Explore→exploit check ---
    for name, ep_arr in [("RL²+HN", rl2hn_ep),
                         ("VariBAD", vb_ep), ("VariBAD+HN", vbhn_ep)]:
        imp = ep_arr[-1] - ep_arr[0]
        status = "detected" if imp > 0.05 else "not detected"
        print(f"\n  {name} episode 1 → {EPISODES_PER_TRIAL}: "
              f"{ep_arr[0]:.4f} → {ep_arr[-1]:.4f} ({imp:+.4f}) "
              f"— explore→exploit {status}")

    # --- Gaps ---
    print(f"\n  Gaps:")
    print(f"    VariBAD+HN vs VariBAD:  {vbhn_rps - vb_rps:+.4f} reward/step")
    print(f"    VariBAD+HN vs RL²+HN:  {vbhn_rps - rl2hn_rps:+.4f} reward/step")
    print(f"    RL²+HN     vs VariBAD: {rl2hn_rps - vb_rps:+.4f} reward/step")

    # --- Per-step curves (abbreviated) ---
    print(f"\n  Per-step reward across trial "
          f"(every {t_ep} steps = episode boundary):")
    print(f"  {'step':>4}  {'Random':>8}  {'RL²+HN':>8}  {'VariBAD':>10}  "
          f"{'VariBAD+HN':>10}  {'Oracle':>8}")
    print("  " + "-" * 52)
    for t in range(0, n_steps, t_ep):
        print(f"  {t + 1:>4}  {rand_curve[t]:>8.4f}  "
              f"{rl2hn_curve[t]:>8.4f}  "
              f"{vb_curve[t]:>10.4f}  {vbhn_curve[t]:>10.4f}  "
              f"{oracle_curve[t]:>8.4f}")
    t = n_steps - 1
    print(f"  {t + 1:>4}  {rand_curve[t]:>8.4f}  "
          f"{rl2hn_curve[t]:>8.4f}  "
          f"{vb_curve[t]:>10.4f}  {vbhn_curve[t]:>10.4f}  "
          f"{oracle_curve[t]:>8.4f}")

    print()


if __name__ == "__main__":
    main()
