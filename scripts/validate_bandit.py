#!/usr/bin/env python3
"""Validate meta-RL agent architectures on K-armed Bernoulli Bandit.

Each episode: K=3 arms with hidden probs ~ Beta(1,1), T=10 steps.
Each env_reset samples new arm probs (new task).
Multi-episode trials (4 episodes) for recurrent agents.

Expected performance hierarchy:
  Random ≈ PPO MLP   ≈ 0.50 reward/step  (no memory of past pulls)
  RL²/RL²+HN/VariBAD ≈ 0.55-0.65 reward/step (learn to explore)
  Thompson sampling   ≈ 0.65 reward/step  (Bayes-optimal)

Also reports per-step reward curves: trained explorers should show
increasing reward over the episode (explore early → exploit late),
while memoryless agents stay flat at 0.50.

Usage:
    uv run python scripts/validate_bandit.py --fast
    uv run python scripts/validate_bandit.py
"""
import argparse
import time
import os
import sys

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import optax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.bandit import BanditParams, env_reset, env_step
from agents.ppo import ActorCritic, actor_loss_fn, critic_loss_fn
from agents.rl2 import GRUActorCritic, rl2_loss_fn
from agents.rl2_hn import HNActorCritic, rl2_hn_loss_fn
from agents.varibad import VariBADActorCritic, elbo_loss_fn, varibad_ppo_loss_fn
from agents.common import batch_gae

# ---------------------------------------------------------------------------
# Constants (bandit-specific)
# ---------------------------------------------------------------------------

OBS_SIZE = 1           # obs = (step / t_episode,)
N_ACTIONS = 3          # K arms
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1  # aug = (obs, prev_act_oh, prev_rew)
HIDDEN_SIZE = 32       # small GRU for simple problem
EPISODES_PER_TRIAL = 4

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Thompson Sampling baseline (Bayes-optimal for Bernoulli bandits)
# ---------------------------------------------------------------------------

def thompson_per_step(key, params, n_episodes):
    """Returns (t_episode,) mean reward per step position for Thompson."""
    def single_episode(key):
        k_arms, k_steps = jax.random.split(key)
        arm_probs = jax.random.beta(
            k_arms, params.prior_alpha, params.prior_beta, (params.n_arms,))
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
        return rewards  # (t_episode,)

    keys = jax.random.split(key, n_episodes)
    all_rewards = jax.vmap(single_episode)(keys)  # (n_episodes, t_episode)
    return jnp.mean(all_rewards, axis=0)


# ---------------------------------------------------------------------------
# Trajectory storage (shared by recurrent agents)
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
    """Collect one full episode per env. Returns tuple of arrays."""
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


def _converged(history, patience, min_delta, min_iters):
    """Check if the last `patience` iters show no improvement over the
    previous `patience` iters.  Returns True once plateaued.
    Never triggers before min_iters to avoid killing slow-warmup agents."""
    if len(history) < max(2 * patience, min_iters):
        return False
    recent = np.mean(history[-patience:])
    previous = np.mean(history[-2 * patience:-patience])
    return recent - previous < min_delta


def train_ppo(key, params, max_iters, n_envs=512, lr=3e-4,
              patience=30, min_delta=0.002, min_iters=60, verbose=True):
    """Train PPO MLP on bandit. Returns (trained model, reward_history)."""
    k_model, key = jax.random.split(key)
    model = ActorCritic(OBS_SIZE, N_ACTIONS, hidden=32, key=k_model)
    actor_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    critic_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    a_opt_st = actor_opt.init(eqx.filter(model.actor, eqx.is_array))
    c_opt_st = critic_opt.init(eqx.filter(model.critic, eqx.is_array))

    @eqx.filter_jit
    def update(model, a_opt_st, c_opt_st, batch):
        al, ag = eqx.filter_value_and_grad(actor_loss_fn)(
            model.actor, model.critic, batch)
        au, a_opt_st2 = actor_opt.update(
            ag, a_opt_st, eqx.filter(model.actor, eqx.is_array))
        cl, cg = eqx.filter_value_and_grad(critic_loss_fn)(
            model.critic, batch)
        cu, c_opt_st2 = critic_opt.update(
            cg, c_opt_st, eqx.filter(model.critic, eqx.is_array))
        m2 = eqx.tree_at(lambda m: m.actor, model,
                          eqx.apply_updates(model.actor, au))
        m2 = eqx.tree_at(lambda m: m.critic, m2,
                          eqx.apply_updates(model.critic, cu))
        return m2, a_opt_st2, c_opt_st2

    reward_history = []
    stopped_early = False
    t0 = time.time()
    for it in range(max_iters):
        k_coll, key = jax.random.split(key)
        obs, actions, rewards, dones, lp, values = collect_ppo_rollout(
            k_coll, model, params, n_envs)

        reward_history.append(float(jnp.mean(rewards)))

        if _converged(reward_history, patience, min_delta, min_iters):
            stopped_early = True
            if verbose:
                _print(f"\r    PPO MLP  converged at iter {it + 1} "
                       f"[{time.time() - t0:.0f}s]    ")
            break

        bootstrap = values[-1]
        adv, ret = batch_gae(rewards, values, dones, bootstrap, 0.99, 0.95)

        obs_f = obs.reshape(-1, OBS_SIZE)
        adv_f = adv.reshape(-1)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        batch = (obs_f, actions.reshape(-1), lp.reshape(-1),
                 adv_f, ret.reshape(-1))
        model, a_opt_st, c_opt_st = update(model, a_opt_st, c_opt_st, batch)

        if verbose and it % max(1, max_iters // 10) == 0:
            elapsed = time.time() - t0
            _print(f"\r    PPO MLP  iter {it:4d}/{max_iters} "
                   f"[{elapsed:.0f}s]", end="", flush=True)
    if not stopped_early and verbose:
        _print(f"\r    PPO MLP  iter {max_iters:4d}/{max_iters} "
               f"[{time.time() - t0:.0f}s]  (max iters)")
    return model, reward_history


def eval_ppo_per_step(model, params, key, n_episodes):
    """Returns (t_episode,) mean reward per step position for PPO MLP."""
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

def collect_recurrent_rollout(key, model, params, n_trials,
                              episodes_per_trial=EPISODES_PER_TRIAL):
    """Collect multi-episode trial rollouts. Uses model.forward_step."""
    n_steps = episodes_per_trial * params.t_episode
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

        def fwd(x, o, h):
            return model.forward_step(x, o, h)

        logits, values, new_gru_st = jax.vmap(fwd)(aug, obs, gru_st)

        act_keys = jax.random.split(k_act, n_trials)
        actions = jax.vmap(
            jax.random.categorical, in_axes=(0, 0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)

        env_keys = jax.random.split(k_env, n_trials)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(env_keys, states, actions, params)

        next_obs = new_o  # pre-reset, for VariBAD ELBO

        rst_keys = jax.random.split(k_rst, n_trials)
        def maybe_reset(done, ns, no, rk):
            rs, ro = env_reset(rk, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset)(
            dones, new_s, new_o, rst_keys)

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


def train_recurrent(key, model, loss_fn, params, max_iters, n_trials=64,
                    n_epochs=3, minibatch_size=512, lr=3e-4,
                    elbo_fn=None, label="recurrent",
                    patience=50, min_delta=0.002, min_iters=200,
                    verbose=True):
    """Train a recurrent agent on bandit. Returns (trained model, reward_history)."""
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
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

    reward_history = []
    stopped_early = False
    t0 = time.time()
    for it in range(max_iters):
        k_coll, k_elbo, key = jax.random.split(key, 3)
        traj = collect_recurrent_rollout(k_coll, model, params, n_trials)

        reward_history.append(float(jnp.mean(traj.rewards)))

        if _converged(reward_history, patience, min_delta, min_iters):
            stopped_early = True
            if verbose:
                _print(f"\r    {label:<9s} converged at iter {it + 1} "
                       f"[{time.time() - t0:.0f}s]    ")
            break

        bootstrap = traj.values[-1]
        adv, ret = batch_gae(traj.rewards, traj.values, traj.dones,
                             bootstrap, 0.99, 0.95)

        # ELBO update (VariBAD only)
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
        for _ in range(n_epochs):
            k_p, k_shuf = jax.random.split(k_shuf)
            perm = jax.random.permutation(k_p, total)
            for start in range(0, total, minibatch_size):
                idx = perm[start:start + minibatch_size]
                mb_adv = adv_f[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb = (flat.obs[idx], flat.prev_actions_oh[idx],
                      flat.prev_rewards[idx], flat.gru_h[idx],
                      flat.actions[idx], flat.log_probs[idx],
                      mb_adv, ret_f[idx])
                model, opt_st = ppo_update(model, opt_st, mb)

        if verbose and it % max(1, max_iters // 10) == 0:
            elapsed = time.time() - t0
            _print(f"\r    {label:<9s} iter {it:4d}/{max_iters} "
                   f"[{elapsed:.0f}s]", end="", flush=True)
    if not stopped_early and verbose:
        _print(f"\r    {label:<9s} iter {max_iters:4d}/{max_iters} "
               f"[{time.time() - t0:.0f}s]  (max iters)")
    return model, reward_history


def eval_recurrent_per_step(model, params, key, n_episodes):
    """Returns (t_episode,) mean reward per step position for a single
    fresh episode (no cross-episode GRU state — tests within-episode
    exploration)."""
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
# Plotting
# ---------------------------------------------------------------------------

PLOTS_DIR = os.path.join(ROOT, "plots")

AGENT_COLORS = {
    "PPO MLP": "C0",
    "RL²": "C1",
    "RL²+HN": "C2",
    "VariBAD": "C3",
}


def plot_bandit_learning_curves(train_histories, per_step, results):
    """Two-panel chart: training curves + per-step reward curves.

    Args:
        train_histories: dict name -> list of mean reward/step per iter
        per_step: dict name -> (t_episode,) array of per-step rewards
        results: dict name -> scalar mean reward/step (final eval)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: training curves ---
    ax1.set_title("Training Curves (rollout reward/step)", fontweight="bold")
    for name in ["PPO MLP", "RL²", "RL²+HN", "VariBAD"]:
        if name not in train_histories:
            continue
        hist = train_histories[name]
        iters = np.arange(len(hist))
        color = AGENT_COLORS[name]
        # smooth with rolling mean for readability
        window = max(1, len(hist) // 20)
        smoothed = np.convolve(hist, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(window - 1, len(hist))
        ax1.plot(x_smooth, smoothed, color=color, linewidth=1.5, label=name)
        ax1.plot(iters, hist, color=color, alpha=0.2, linewidth=0.5)

    ax1.axhline(results["Random"], color="gray", linestyle=":",
                linewidth=1, label="Random (0.50)")
    ax1.axhline(results["Thompson"], color="red", linestyle="--",
                linewidth=1, label=f"Thompson ({results['Thompson']:.3f})")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Mean reward / step")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Right panel: per-step reward curves ---
    ax2.set_title("Within-Episode Reward (eval, greedy)", fontweight="bold")
    steps = np.arange(1, len(per_step["Random"]) + 1)

    ax2.fill_between(steps, 0.5, 0.5, alpha=0)  # dummy for axis
    ax2.plot(steps, per_step["Random"], color="gray", linestyle=":",
             linewidth=1, label="Random")
    ax2.plot(steps, per_step["Thompson"], color="red", linestyle="--",
             linewidth=1.5, label="Thompson")
    for name in ["PPO MLP", "RL²", "RL²+HN", "VariBAD"]:
        if name not in per_step:
            continue
        ax2.plot(steps, per_step[name], color=AGENT_COLORS[name],
                 linewidth=1.5, label=name)

    ax2.set_xlabel("Step within episode")
    ax2.set_ylabel("Mean reward")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(steps)

    fig.suptitle("Bernoulli Bandit — Meta-RL Validation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "bandit_learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate meta-RL agents on Bernoulli Bandit")
    parser.add_argument("--fast", action="store_true",
                        help="Quick validation (fewer iters/episodes)")
    args = parser.parse_args()

    if args.fast:
        max_iters_ppo = 200
        max_iters_rec = 1000
        n_trials = 128
        n_eval = 2000
    else:
        max_iters_ppo = 500
        max_iters_rec = 2000
        n_trials = 128
        n_eval = 5000

    params = BanditParams()
    key = jax.random.PRNGKey(42)

    print("=" * 60)
    print("  Bernoulli Bandit — Meta-RL Validation")
    print("=" * 60)
    print(f"  K={params.n_arms} arms, Beta({params.prior_alpha},{params.prior_beta})"
          f" prior, T={params.t_episode} steps/episode")
    print(f"  Recurrent: {EPISODES_PER_TRIAL} episodes/trial, "
          f"hidden={HIDDEN_SIZE}")
    print(f"  Max iters: PPO={max_iters_ppo}, recurrent={max_iters_rec} "
          f"(early stop: patience=50, min_delta=0.002, "
          f"min_iters=60/200)")
    print()

    results = {}       # name -> mean reward/step
    per_step = {}      # name -> (t_episode,) array
    train_histories = {}  # name -> list of mean reward/step per training iter

    # --- Random baseline ---
    print("  Computing baselines ...")
    results["Random"] = 0.5  # E[p] for Beta(1,1)
    per_step["Random"] = np.full(params.t_episode, 0.5)

    # --- Thompson sampling (Bayes-optimal) ---
    k_ts, key = jax.random.split(key)
    ts_curve = thompson_per_step(k_ts, params, n_eval)
    ts_curve = np.array(ts_curve)
    results["Thompson"] = float(ts_curve.mean())
    per_step["Thompson"] = ts_curve
    print(f"    Thompson sampling: {results['Thompson']:.4f} reward/step")

    # --- PPO MLP ---
    print("\n  Training agents ...")
    k_ppo, key = jax.random.split(key)
    ppo_model, ppo_hist = train_ppo(k_ppo, params, max_iters_ppo)
    train_histories["PPO MLP"] = ppo_hist
    k_eval, key = jax.random.split(key)
    ppo_curve = np.array(eval_ppo_per_step(ppo_model, params, k_eval, n_eval))
    results["PPO MLP"] = float(ppo_curve.mean())
    per_step["PPO MLP"] = ppo_curve

    # --- RL² ---
    k_rl2, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_rl2)
    rl2_model = GRUActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS, hidden_size=HIDDEN_SIZE, key=k_m)
    rl2_model, rl2_hist = train_recurrent(
        k_t, rl2_model, rl2_loss_fn, params, max_iters_rec,
        n_trials=n_trials, label="RL²")
    train_histories["RL²"] = rl2_hist
    k_eval, key = jax.random.split(key)
    rl2_curve = np.array(eval_recurrent_per_step(
        rl2_model, params, k_eval, n_eval))
    results["RL²"] = float(rl2_curve.mean())
    per_step["RL²"] = rl2_curve

    # --- RL²+HN ---
    k_hn, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_hn)
    hn_model = HNActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, policy_hidden=16, key=k_m)
    hn_model, hn_hist = train_recurrent(
        k_t, hn_model, rl2_hn_loss_fn, params, max_iters_rec,
        n_trials=n_trials, label="RL²+HN")
    train_histories["RL²+HN"] = hn_hist
    k_eval, key = jax.random.split(key)
    hn_curve = np.array(eval_recurrent_per_step(
        hn_model, params, k_eval, n_eval))
    results["RL²+HN"] = float(hn_curve.mean())
    per_step["RL²+HN"] = hn_curve

    # --- VariBAD ---
    k_vb, key = jax.random.split(key)
    k_m, k_t = jax.random.split(k_vb)
    vb_model = VariBADActorCritic(
        INPUT_SIZE, OBS_SIZE, N_ACTIONS,
        hidden_size=HIDDEN_SIZE, latent_dim=2, key=k_m)
    vb_model, vb_hist = train_recurrent(
        k_t, vb_model, varibad_ppo_loss_fn, params, max_iters_rec,
        n_trials=n_trials, elbo_fn=elbo_loss_fn, label="VariBAD")
    train_histories["VariBAD"] = vb_hist
    k_eval, key = jax.random.split(key)
    vb_curve = np.array(eval_recurrent_per_step(
        vb_model, params, k_eval, n_eval))
    results["VariBAD"] = float(vb_curve.mean())
    per_step["VariBAD"] = vb_curve

    # --- Results table ---
    random_rps = results["Random"]
    print("\n" + "=" * 60)
    print(f"  {'Agent':<14} {'reward/step':>12} {'vs Random':>10} "
          f"{'vs Thompson':>12}")
    print("  " + "-" * 56)
    for name in ["Random", "PPO MLP", "RL²", "RL²+HN", "VariBAD", "Thompson"]:
        rps = results[name]
        diff_rand = rps - random_rps
        diff_ts = rps - results["Thompson"]
        print(f"  {name:<14} {rps:>12.4f} {diff_rand:>+10.4f} {diff_ts:>+12.4f}")
    print("=" * 60)

    # --- Per-step reward curve ---
    print(f"\n  Per-step reward (within single episode, T={params.t_episode}):")
    header = f"  {'step':>4}"
    for name in ["Random", "Thompson", "PPO MLP", "RL²", "RL²+HN", "VariBAD"]:
        header += f"  {name:>8}"
    print(header)
    print("  " + "-" * (4 + 6 * 10))
    for t in range(params.t_episode):
        row = f"  {t + 1:>4}"
        for name in ["Random", "Thompson", "PPO MLP",
                      "RL²", "RL²+HN", "VariBAD"]:
            row += f"  {per_step[name][t]:>8.3f}"
        print(row)

    # --- Verdict ---
    meta_agents = ["RL²", "RL²+HN", "VariBAD"]
    meta_rps = [results[n] for n in meta_agents]
    best_meta = meta_agents[np.argmax(meta_rps)]
    best_meta_rps = max(meta_rps)

    print(f"\n  Verdict:")
    all_pass = True
    for name in meta_agents:
        gap = results[name] - random_rps
        status = "PASS" if gap > 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {name}: {status} "
              f"(+{gap:.3f} vs random, "
              f"{gap / (results['Thompson'] - random_rps) * 100:.0f}% "
              f"of Thompson gap)")

    if all_pass:
        print("\n  All meta-RL agents learn to explore on the bandit.")
    else:
        print("\n  WARNING: Some agents failed to outperform random.")
        print("  Check architecture/training — may indicate a bug.")

    # --- Learning curve chart ---
    print("\n  Plotting learning curves ...")
    plot_bandit_learning_curves(train_histories, per_step, results)

    print()


if __name__ == "__main__":
    main()
