#!/usr/bin/env python3
"""PPO MLP Baseline — Training.

Trains PPO on locked-regime (noise/bull/bear) or mixed (HMM switching).
Saves learning curves and action distributions to JSON.
Oracle bounds are computed separately in analytical_foundation.py.

Supports incremental updates: --regime noise only retrains noise.

Outputs:
  results/ppo_baseline.json  — all data needed for plotting

Usage:
    uv run python scripts/train_ppo.py --fast                    # all regimes, quick
    uv run python scripts/train_ppo.py --regime noise            # only noise
    uv run python scripts/train_ppo.py --regime mixed --n-seeds 8
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

from lob_sim.jax_env import EnvParams, env_reset, env_step, rollout_episode, batch_rollout

RESULTS_DIR = os.path.join(ROOT, "results")
JSON_PATH = os.path.join(RESULTS_DIR, "ppo_baseline.json")

OBS_SIZE = 4
N_ACTIONS = 3
REGIME_MAP = {"noise": 0, "bull": 1, "bear": 2, "mixed": -1}

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    layers: list
    def __init__(self, sizes, *, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(a, b, key=k)
                       for a, b, k in zip(sizes[:-1], sizes[1:], keys)]
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class ActorCritic(eqx.Module):
    actor: MLP
    critic: MLP
    def __init__(self, *, key):
        k1, k2 = jax.random.split(key)
        self.actor = MLP([OBS_SIZE, 64, 64, N_ACTIONS], key=k1)
        self.critic = MLP([OBS_SIZE, 64, 64, 1], key=k2)
    def __call__(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, bootstrap_value, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    def scan_fn(gae, t):
        idx = T - 1 - t
        next_val = jnp.where(idx < T - 1, values[idx + 1], bootstrap_value)
        delta = rewards[idx] + gamma * next_val * (1 - dones[idx]) - values[idx]
        gae = delta + gamma * lam * (1 - dones[idx]) * gae
        return gae, gae
    _, adv_rev = jax.lax.scan(scan_fn, jnp.float32(0.0), jnp.arange(T))
    adv = adv_rev[::-1]
    return adv, adv + values

batch_gae = jax.vmap(compute_gae, in_axes=(1,1,1,0,None,None), out_axes=(1,1))


# ---------------------------------------------------------------------------
# PPO losses
# ---------------------------------------------------------------------------

def actor_loss_fn(actor, critic, batch, clip_eps=0.2, ent_coef=0.01):
    obs, actions, old_lp, advantages, _ = batch
    logits = jax.vmap(actor)(obs)
    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(lp - old_lp)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1))
    return loss - ent_coef * entropy

def critic_loss_fn(critic, batch):
    obs, _, _, _, returns = batch
    vals = jax.vmap(critic)(obs).squeeze(-1)
    return jnp.mean((vals - returns) ** 2)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

class Trajectory(eqx.Module):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray

def collect_rollout(key, model, params, n_envs, n_steps):
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_envs)
    states, obs_init = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)
    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]
        logits, values = jax.vmap(model)(obs)
        act_keys = jax.random.split(k_act, n_envs)
        actions = jax.vmap(jax.random.categorical, in_axes=(0,0))(act_keys, logits)
        lp_all = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp_all, actions[:, None], axis=1).squeeze(1)
        env_keys = jax.random.split(k_env, n_envs)
        new_s, new_o, rews, dones, _ = jax.vmap(
            env_step, in_axes=(0,0,0,None))(env_keys, states, actions, params)
        rst_keys = jax.random.split(k_rst, n_envs)
        def maybe_reset(done, ns, no, rk):
            rs, ro = env_reset(rk, params)
            return (jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, ns),
                    jnp.where(done, ro, no))
        final_s, final_o = jax.vmap(maybe_reset, in_axes=(0,0,0,0))(
            dones, new_s, new_o, rst_keys)
        return (final_s, final_o), Trajectory(
            obs=obs, actions=actions, rewards=rews, dones=dones,
            log_probs=lp, values=values)

    (_, _), traj = jax.lax.scan(scan_step, (states, obs_init), all_keys)
    return traj


# ---------------------------------------------------------------------------
# Evaluation + action distribution collection
# ---------------------------------------------------------------------------

def evaluate_returns(key, model, params, n_episodes):
    """Returns (per_episode_returns, per_episode_regimes)."""
    def policy(k, obs):
        logits, _ = model(obs)
        return jnp.argmax(logits)
    keys = jax.random.split(key, n_episodes)
    traj = jax.vmap(rollout_episode, in_axes=(0, None, None))(keys, policy, params)
    returns = jnp.sum(traj["rewards"], axis=1)
    regimes = traj["true_regimes"][:, 0]
    return returns, regimes


def collect_action_fracs(model, params, n_episodes=256):
    """Returns (N_REGIMES, n_inv, N_ACTIONS) action fractions from simulation."""
    key = jax.random.PRNGKey(42)
    def policy(k, obs):
        logits, _ = model(obs)
        return jax.random.categorical(k, logits)
    traj = batch_rollout(key, policy, params, n_envs=n_episodes)

    actions = np.array(traj["actions"]).reshape(-1)
    inv = np.array(traj["obs"][:, :, 3]).reshape(-1).astype(int)
    regimes = np.array(traj["true_regimes"]).reshape(-1).astype(int)

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

def train_single(seed, regime, seed_label=0, n_iters=200,
                 n_envs=128, n_steps=64, n_epochs=3, minibatch_size=512,
                 lr=3e-4, eval_every=1, n_eval=64, verbose=True):
    """Train PPO MLP. Returns (metrics_dict, final_model)."""
    key = jax.random.PRNGKey(seed)
    env_params = EnvParams.default()._replace(locked_regime=regime)
    eval_params = env_params._replace(init_inventory=0)

    k_model, key = jax.random.split(key)
    model = ActorCritic(key=k_model)

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
        cl, cg = eqx.filter_value_and_grad(critic_loss_fn)(model.critic, batch)
        cu, c_opt_st2 = critic_opt.update(
            cg, c_opt_st, eqx.filter(model.critic, eqx.is_array))
        m2 = eqx.tree_at(lambda m: m.actor, model, eqx.apply_updates(model.actor, au))
        m2 = eqx.tree_at(lambda m: m.critic, m2, eqx.apply_updates(model.critic, cu))
        return m2, a_opt_st2, c_opt_st2, al, cl

    is_mixed = (regime == -1)
    regime_names = ["noise", "bull", "bear"]
    iters = []
    agent_mean_returns = []
    # For mixed: per-regime returns; for isolated: single-regime returns
    per_regime_returns = {n: [] for n in regime_names}
    t0 = time.time()

    for it in range(n_iters):
        k_collect, k_eval, key = jax.random.split(key, 3)

        # Collect training rollout — stratified for mixed
        if is_mixed:
            n_per = n_envs // 3
            trajs = []
            for rid in range(3):
                k_c, k_collect = jax.random.split(k_collect)
                ep = env_params._replace(locked_regime=rid)
                trajs.append(collect_rollout(k_c, model, ep, n_per, n_steps))
            traj = jax.tree.map(
                lambda *xs: jnp.concatenate(xs, axis=1), *trajs)
        else:
            traj = collect_rollout(k_collect, model, env_params, n_envs, n_steps)

        bootstrap = traj.values[-1]
        adv, ret = batch_gae(traj.rewards, traj.values, traj.dones, bootstrap, 0.99, 0.95)

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
                mb = (flat.obs[idx], flat.actions[idx], flat.log_probs[idx],
                      mb_adv, ret_f[idx])
                model, a_opt_st, c_opt_st, _, _ = update(model, a_opt_st, c_opt_st, mb)

        # Evaluate agent — always per-regime, inv fixed at 0
        n_eval_per = n_eval if not is_mixed else n_eval // 3
        regime_means = {}
        for rid in range(3):
            k_e, k_eval = jax.random.split(k_eval)
            ep = eval_params._replace(locked_regime=rid)
            ret_r, _ = evaluate_returns(k_e, model, ep, n_eval_per)
            regime_means[regime_names[rid]] = float(jnp.mean(ret_r))

        # For isolated: only the matching regime matters for the overall mean
        if is_mixed:
            agent_mean = float(np.mean(list(regime_means.values())))
        else:
            agent_mean = regime_means[regime_names[regime]]

        iters.append(it)
        agent_mean_returns.append(round(agent_mean, 4))
        for n in regime_names:
            per_regime_returns[n].append(round(regime_means[n], 4))

        if verbose:
            rps = agent_mean / env_params.t_episode
            print(f"    seed {seed_label} iter {it:4d} | "
                  f"reward/step {rps:.4f} | {time.time()-t0:.0f}s")

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
    parser.add_argument("--n-envs", type=int, default=128)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--master-seed", type=int, default=0)
    args = parser.parse_args()

    if args.fast:
        n_seeds = args.n_seeds or 1
        n_iters = args.n_iters or 30
        n_eval = 64
        n_heatmap_eps = 64
    else:
        n_seeds = args.n_seeds or 3
        n_iters = args.n_iters or 200
        n_eval = 128
        n_heatmap_eps = 256

    if args.regime == "all":
        regimes = [("noise", 0), ("bull", 1), ("bear", 2), ("mixed", -1)]
    else:
        regimes = [(args.regime, REGIME_MAP[args.regime])]

    print("=" * 60)
    print("  PPO MLP Baseline — Training")
    print("=" * 60)

    params = EnvParams.default()
    t_ep = int(params.t_episode)
    data = load_results()
    data["t_episode"] = t_ep
    master_rng = np.random.default_rng(args.master_seed)

    for name, rid in regimes:
        print(f"\n  --- {name.upper()} ---")
        print(f"  Training ({n_seeds} seeds x {n_iters} iters)")
        run_seeds = master_rng.integers(0, 2**31, size=n_seeds)
        runs = []
        best_ret, best_model = -np.inf, None
        for si, seed in enumerate(run_seeds):
            result, model = train_single(
                seed=int(seed), regime=rid,
                seed_label=si,
                n_iters=n_iters, n_envs=args.n_envs, lr=args.lr,
                n_eval=n_eval, verbose=True)
            runs.append(result)
            final_ret = result["mean_returns"][-1]
            if final_ret > best_ret:
                best_ret, best_model = final_ret, model

        # Collect action distribution from best model
        print("  Collecting action distributions ...")
        ep = params._replace(locked_regime=rid)
        fracs = collect_action_fracs(best_model, ep, n_episodes=n_heatmap_eps)

        # Store
        data[name] = {
            "locked_regime": rid,
            "runs": runs,
            "action_fracs": fracs.tolist(),
        }

        final_rps = best_ret / t_ep
        print(f"  Best seed: reward/step={final_rps:.4f}")

    save_results(data)
    print("\n  Done. Run `uv run python scripts/plot_ppo.py` to generate figures.")


if __name__ == "__main__":
    main()
