#!/usr/bin/env python3
"""RL²+HN (Recurrent HyperNetwork) — Training.

Trains RL²+HN on multi-episode trials (4 episodes × 200 steps).
Same GRU backbone as RL², but the policy MLP is *generated* by a HyperNetwork
from the GRU hidden state. The observation o_t is re-passed as input to the
generated policy (dual conditioning: h_t via weights, o_t via input).

Architecture (Beck et al. 2023):
  GRU: (o_t, a_{t-1}, r_{t-1}) → h_t  (size 128)
  HyperNet: h_t → all weights & biases of policy MLP  (515 params)
  Generated policy: o_t → 64 → action logits  (uses o_t, NOT aug)
  Critic: MLP(h_t) → value  (standard, not generated)
  Init: Bias-HyperInit — HN last layer zero weights, random bias
        → all trajectories produce identical base-network at init

Outputs:
  results/rl2_hn.json

Usage:
    uv run python scripts/train_rl2_hn.py --fast
    uv run python scripts/train_rl2_hn.py --regime noise
    uv run python scripts/train_rl2_hn.py --regime mixed --n-seeds 8
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

from lob_sim.jax_env import EnvParams, env_reset, env_step
from agents.rl2_hn import (HNActorCritic, apply_generated_policy,
                           rl2_hn_loss_fn as combined_loss_fn,
                           compute_n_policy_params)
from agents.common import batch_gae, NumpyEncoder

RESULTS_DIR = os.path.join(ROOT, "results")
JSON_PATH = os.path.join(RESULTS_DIR, "rl2_hn.json")

OBS_SIZE = 4
N_ACTIONS = 3
INPUT_SIZE = OBS_SIZE + N_ACTIONS + 1  # obs + prev_action_onehot + prev_reward
HIDDEN_SIZE = 128
POLICY_HIDDEN = 64
EPISODES_PER_TRIAL = 4
REGIME_MAP = {"noise": 0, "bull": 1, "bear": 2, "mixed": -1}

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

def _fmt_time(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


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


# ---------------------------------------------------------------------------
# Trial rollout collection
# ---------------------------------------------------------------------------

def collect_trial_rollout(key, model, params, n_trials,
                          episodes_per_trial=EPISODES_PER_TRIAL):
    """Collect multi-episode trial rollouts with HyperNet-generated policy."""
    t_episode = int(params.t_episode)
    n_steps = episodes_per_trial * t_episode

    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_trials)
    states, obs_init = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)

    prev_act = jnp.zeros((n_trials, N_ACTIONS))
    prev_rew = jnp.zeros((n_trials, 1))
    H = model.hidden_size
    gru_st = jnp.zeros((n_trials, H))

    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs, prev_act, prev_rew, gru_st = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]

        aug = jnp.concatenate([obs, prev_act, prev_rew], axis=-1)
        stored_h = gru_st

        def fwd(x, o, h):
            new_h = model.gru_cell(x, h)
            policy_params = model.hyper_net(new_h)
            logits = apply_generated_policy(
                o, policy_params, model.obs_size, model.policy_hidden,
                model.n_actions)
            critic_in = jnp.concatenate([new_h, o])
            value = model.critic_head(critic_in).squeeze(-1)
            return logits, value, new_h

        logits, values, new_gru_st = jax.vmap(fwd)(aug, obs, gru_st)

        act_keys = jax.random.split(k_act, n_trials)
        actions = jax.vmap(jax.random.categorical, in_axes=(0,0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)

        env_keys = jax.random.split(k_env, n_trials)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0,0,0,None))(env_keys, states, actions, params)

        rst_keys = jax.random.split(k_rst, n_trials)
        def maybe_reset(done, ns, no, rk):
            rs, ro = env_reset(rk, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset)(dones, new_s, new_o, rst_keys)

        new_prev_act = jax.nn.one_hot(actions, N_ACTIONS)
        new_prev_rew = rews[:, None]

        carry = (final_s, final_o, new_prev_act, new_prev_rew, new_gru_st)
        return carry, Trajectory(
            obs=obs, actions=actions, rewards=rews, dones=dones,
            log_probs=lp, values=values,
            prev_actions_oh=prev_act, prev_rewards=prev_rew,
            gru_h=stored_h)

    _, traj = jax.lax.scan(
        scan_step, (states, obs_init, prev_act, prev_rew, gru_st), all_keys)
    return traj


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_trial_returns(key, model, params, n_trials,
                           episodes_per_trial=EPISODES_PER_TRIAL):
    """Run full trials with greedy HN-generated policy."""
    t_episode = int(params.t_episode)

    def single_trial(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)

        H = model.hidden_size
        gru_st = jnp.zeros(H)
        prev_act = jnp.zeros(N_ACTIONS)
        prev_rew = jnp.zeros(1)

        n_steps = episodes_per_trial * t_episode
        step_keys = jax.random.split(k_steps, n_steps)

        def scan_fn(carry, key_t):
            state, obs, gru_st, prev_act, prev_rew = carry
            k_env, k_rst = jax.random.split(key_t)

            aug = jnp.concatenate([obs, prev_act, prev_rew])
            new_h = model.gru_cell(aug, gru_st)
            policy_params = model.hyper_net(new_h)
            logits = apply_generated_policy(
                obs, policy_params, model.obs_size, model.policy_hidden,
                model.n_actions)
            action = jnp.argmax(logits)

            cur_regime = state.regime
            new_state, new_obs, reward, done, _ = env_step(
                k_env, state, action, params)

            rst_state, rst_obs = env_reset(k_rst, params)
            final_state = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), rst_state, new_state)
            final_obs = jnp.where(done, rst_obs, new_obs)

            carry = (final_state, final_obs, new_h,
                     jax.nn.one_hot(action, N_ACTIONS), reward[None])
            return carry, (reward, cur_regime)

        _, (rewards, regimes) = jax.lax.scan(
            scan_fn, (state, obs, gru_st, prev_act, prev_rew), step_keys)

        ep_rewards = rewards.reshape(episodes_per_trial, t_episode)
        ep_regimes = regimes.reshape(episodes_per_trial, t_episode)[:, 0]
        ep_returns = jnp.sum(ep_rewards, axis=1)
        return ep_returns, ep_regimes

    keys = jax.random.split(key, n_trials)
    returns, regimes = jax.vmap(single_trial)(keys)
    return returns, regimes


def collect_action_fracs(model, params, n_trials=64,
                         episodes_per_trial=EPISODES_PER_TRIAL):
    """Returns (N_REGIMES, n_inv, N_ACTIONS) action fractions."""
    key = jax.random.PRNGKey(42)
    t_episode = int(params.t_episode)

    def single_trial(key):
        k_reset, k_steps = jax.random.split(key)
        state, obs = env_reset(k_reset, params)

        H = model.hidden_size
        gru_st = jnp.zeros(H)
        prev_act = jnp.zeros(N_ACTIONS)
        prev_rew = jnp.zeros(1)

        n_steps = episodes_per_trial * t_episode
        step_keys = jax.random.split(k_steps, n_steps)

        def scan_fn(carry, key_t):
            state, obs, gru_st, prev_act, prev_rew = carry
            k_act, k_env, k_rst = jax.random.split(key_t, 3)

            aug = jnp.concatenate([obs, prev_act, prev_rew])
            new_h = model.gru_cell(aug, gru_st)
            policy_params = model.hyper_net(new_h)
            logits = apply_generated_policy(
                obs, policy_params, model.obs_size, model.policy_hidden,
                model.n_actions)
            action = jax.random.categorical(k_act, logits)

            cur_regime = state.regime
            new_state, new_obs, reward, done, _ = env_step(
                k_env, state, action, params)

            rst_state, rst_obs = env_reset(k_rst, params)
            final_state = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), rst_state, new_state)
            final_obs = jnp.where(done, rst_obs, new_obs)

            carry = (final_state, final_obs, new_h,
                     jax.nn.one_hot(action, N_ACTIONS), reward[None])
            return carry, (obs, action, cur_regime)

        _, (obs_seq, act_seq, reg_seq) = jax.lax.scan(
            scan_fn, (state, obs, gru_st, prev_act, prev_rew), step_keys)
        return obs_seq, act_seq, reg_seq

    keys = jax.random.split(key, n_trials)
    obs_all, actions_all, regimes_all = jax.vmap(single_trial)(keys)

    actions = np.array(actions_all).reshape(-1)
    inv = np.array(obs_all[:, :, 3]).reshape(-1).astype(int)
    regimes = np.array(regimes_all).reshape(-1).astype(int)

    inv_max = int(params.inventory_max)
    n_inv = 2 * inv_max + 1
    inv_grid = np.arange(n_inv) - inv_max

    fracs = np.zeros((3, n_inv, N_ACTIONS))
    for r in range(3):
        for qi, q in enumerate(inv_grid):
            mask = (regimes == r) & (inv == q)
            n = mask.sum()
            if n > 0:
                for a in range(N_ACTIONS):
                    fracs[r, qi, a] = float((actions[mask] == a).sum()) / n
    return fracs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single(seed, regime, seed_label=0, n_iters=400,
                 n_trials=64, n_epochs=3, minibatch_size=2048,
                 lr=3e-4, eval_every=1, n_eval_trials=16, patience=100,
                 verbose=True,
                 global_t0=None, global_done_iters=0, global_total_iters=0):
    """Train RL²+HN. Returns (metrics_dict, final_model)."""
    key = jax.random.PRNGKey(seed)
    env_params = EnvParams.default()._replace(
        locked_regime=regime, init_inventory=0)
    eval_params = env_params

    k_model, key = jax.random.split(key)
    model = HNActorCritic(INPUT_SIZE, OBS_SIZE, N_ACTIONS,
                          hidden_size=HIDDEN_SIZE, policy_hidden=POLICY_HIDDEN,
                          key=k_model)

    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    opt_st = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def update(model, opt_st, batch):
        loss, grads = eqx.filter_value_and_grad(combined_loss_fn)(model, batch)
        updates, new_opt_st = optimizer.update(
            grads, opt_st, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_st, loss

    is_mixed = (regime == -1)
    regime_names = ["noise", "bull", "bear"]
    t_episode = int(env_params.t_episode)

    iters = []
    agent_mean_returns = []
    per_regime_returns = {n: [] for n in regime_names}
    best_mean = -np.inf
    iters_since_best = 0
    t0 = time.time()

    for it in range(n_iters):
        k_collect, k_eval, key = jax.random.split(key, 3)

        traj = collect_trial_rollout(k_collect, model, env_params, n_trials)

        # GAE: use actual episode dones (stable per-episode advantage estimates)
        # GRU state still persists across episodes — policy sees cross-episode info
        bootstrap = traj.values[-1]
        adv, ret = batch_gae(traj.rewards, traj.values, traj.dones,
                             bootstrap, 0.99, 0.95)

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
                      flat.prev_rewards[idx],
                      flat.gru_h[idx],
                      flat.actions[idx], flat.log_probs[idx],
                      mb_adv, ret_f[idx])
                model, opt_st, _ = update(model, opt_st, mb)

        if it % eval_every == 0:
            if is_mixed:
                n_eval_per = max(n_eval_trials // 3, 1)
                regime_means = {}
                for rid in range(3):
                    k_e, k_eval = jax.random.split(k_eval)
                    ep = eval_params._replace(locked_regime=rid)
                    ret_r, _ = evaluate_trial_returns(
                        k_e, model, ep, n_eval_per)
                    regime_means[regime_names[rid]] = float(jnp.mean(ret_r))
                agent_mean = float(np.mean(list(regime_means.values())))
            else:
                ret_r, _ = evaluate_trial_returns(
                    k_eval, model, eval_params, n_eval_trials)
                agent_mean = float(jnp.mean(ret_r))
                regime_means = {regime_names[regime]: agent_mean}

            iters.append(it)
            agent_mean_returns.append(round(agent_mean, 4))
            for n in regime_names:
                per_regime_returns[n].append(
                    round(regime_means.get(n, float("nan")), 4))

            if verbose:
                rps = agent_mean / t_episode
                if global_t0 is not None:
                    elapsed = time.time() - global_t0
                    done = global_done_iters + it + 1
                    avg = elapsed / done
                    eta = avg * (global_total_iters - done)
                else:
                    elapsed = time.time() - t0
                    avg = elapsed / (it + 1)
                    eta = avg * (n_iters - it - 1)
                print(f"    seed {seed_label} iter {it:4d} | "
                      f"reward/step {rps:.4f} | "
                      f"{_fmt_time(elapsed)} elapsed | "
                      f"~{_fmt_time(eta)} remaining | "
                      f"~{_fmt_time(elapsed + eta)} total")

            # Early stopping: break if no new best for `patience` iters
            if agent_mean > best_mean:
                best_mean = agent_mean
                iters_since_best = 0
            else:
                iters_since_best += eval_every
            if patience and iters_since_best >= patience:
                if verbose:
                    print(f"    seed {seed_label} — early stop at iter {it} "
                          f"(no improvement for {patience} iters)")
                break

    result = {
        "seed": int(seed),
        "iters": iters,
        "mean_returns": agent_mean_returns,
        "per_regime_returns": per_regime_returns,
        "n_iters": n_iters,
        "train_time": time.time() - t0,
    }
    return result, model


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return {}


def save_results(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  -> {os.path.relpath(JSON_PATH, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--regime", type=str, default="all",
                        choices=["noise", "bull", "bear", "mixed", "all"])
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--master-seed", type=int, default=0)
    args = parser.parse_args()

    if args.fast:
        n_seeds = args.n_seeds or 1
        n_iters = args.n_iters or 30
        n_trials = args.n_trials or 64
        n_eval_trials = 6
        eval_every = 1
        n_heatmap_trials = 16
    else:
        n_seeds = args.n_seeds or 1
        n_iters = args.n_iters or 400
        n_trials = args.n_trials or 64
        n_eval_trials = 24
        eval_every = 5
        n_heatmap_trials = 64

    if args.regime == "all":
        regimes = [("noise", 0), ("bull", 1), ("bear", 2), ("mixed", -1)]
    else:
        regimes = [(args.regime, REGIME_MAP[args.regime])]

    print("=" * 60)
    print("  RL²+HN (Recurrent HyperNetwork) — Training")
    print("=" * 60)
    t_ep = int(EnvParams.default().t_episode)
    print(f"  Trial structure: {EPISODES_PER_TRIAL} episodes × "
          f"{t_ep} steps = {EPISODES_PER_TRIAL * t_ep} steps/trial")
    n_policy_params = compute_n_policy_params(OBS_SIZE, POLICY_HIDDEN, N_ACTIONS)
    print(f"  Generated policy: {OBS_SIZE}→{POLICY_HIDDEN}→{N_ACTIONS} "
          f"({n_policy_params} params)")

    params = EnvParams.default()
    data = load_results()
    data["t_episode"] = t_ep
    data["episodes_per_trial"] = EPISODES_PER_TRIAL
    master_rng = np.random.default_rng(args.master_seed)

    global_t0 = time.time()
    global_total = len(regimes) * n_seeds * n_iters
    global_done = 0

    for name, rid in regimes:
        print(f"\n  --- {name.upper()} ---")
        print(f"  Training ({n_seeds} seeds × {n_iters} iters, "
              f"{n_trials} trials/iter)")
        run_seeds = master_rng.integers(0, 2**31, size=n_seeds)
        runs = []
        best_ret, best_model = -np.inf, None
        for si, seed in enumerate(run_seeds):
            result, model = train_single(
                seed=int(seed), regime=rid,
                seed_label=si,
                n_iters=n_iters, n_trials=n_trials, lr=args.lr,
                eval_every=eval_every, n_eval_trials=n_eval_trials,
                verbose=True,
                global_t0=global_t0, global_done_iters=global_done,
                global_total_iters=global_total)
            global_done += n_iters
            runs.append(result)
            final_ret = result["mean_returns"][-1]
            if final_ret > best_ret:
                best_ret, best_model = final_ret, model

        print("  Collecting action distributions ...")
        ep = params._replace(locked_regime=rid, init_inventory=0)
        fracs = collect_action_fracs(best_model, ep, n_trials=n_heatmap_trials)

        data[name] = {
            "locked_regime": rid,
            "runs": runs,
            "action_fracs": fracs.tolist(),
        }

        final_rps = best_ret / t_ep
        print(f"  Best seed: reward/step={final_rps:.4f}")

    save_results(data)
    print("\n  Done. Run `uv run python scripts/plot_rl2_hn.py` to generate figures.")


if __name__ == "__main__":
    main()
