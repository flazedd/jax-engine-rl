#!/usr/bin/env python3
"""Phase 4 — PPO MLP Baseline.

Trains PPO with a feedforward actor-critic on:
  - 3 locked-regime environments (noise, bull, bear)
  - 1 mixed-regime (HMM switching) environment

Multiple seeds per regime. Computes normalised returns, generates Figure 3,
and runs the gate check (locked curves must reach ~1.0).

Uses Equinox for actor-critic, Optax for optimization, JAX-native env
for fully jit-compiled rollouts + updates.

Outputs:
  results/ppo_phase4.json           — all metrics, baselines, normalised returns
  plots/figure3_ppo_validation.png  — Figure 3

Usage:
    uv run python scripts/train_ppo.py [--fast]
    uv run python scripts/train_ppo.py --regime noise --n-seeds 1
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

from lob_sim.jax_env import EnvParams, env_reset, env_step, rollout_episode

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")

OBS_SIZE = 4   # (fill_bid, fill_ask, mid_change, inventory)
N_ACTIONS = 3

REGIME_MAP = {"noise": 0, "bull": 1, "bear": 2, "mixed": -1}
REGIME_NAMES = {0: "noise", 1: "bull", 2: "bear", -1: "mixed"}

import builtins
_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes: list[int], *, key):
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

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 64, *, key):
        k1, k2 = jax.random.split(key)
        self.actor = MLP([obs_size, hidden, hidden, n_actions], key=k1)
        self.critic = MLP([obs_size, hidden, hidden, 1], key=k2)

    def __call__(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value


# ---------------------------------------------------------------------------
# GAE via lax.scan
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, bootstrap_value, gamma=0.99, lam=0.95):
    """GAE over a single env trajectory (T,)."""
    T = rewards.shape[0]

    def scan_fn(gae, t):
        idx = T - 1 - t
        next_val = jnp.where(idx < T - 1, values[idx + 1], bootstrap_value)
        non_terminal = 1.0 - dones[idx]
        delta = rewards[idx] + gamma * next_val * non_terminal - values[idx]
        gae = delta + gamma * lam * non_terminal * gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(scan_fn, jnp.float32(0.0), jnp.arange(T))
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


batch_gae = jax.vmap(
    compute_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=(1, 1))


# ---------------------------------------------------------------------------
# PPO loss (separate actor/critic)
# ---------------------------------------------------------------------------

def actor_loss_fn(actor, critic, batch, clip_eps=0.2, ent_coef=0.01):
    obs, actions, old_log_probs, advantages, _ = batch
    logits = jax.vmap(actor)(obs)
    log_probs_all = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(
        log_probs_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))
    return policy_loss - ent_coef * entropy


def critic_loss_fn(critic, batch):
    obs, _, _, _, returns = batch
    values = jax.vmap(critic)(obs).squeeze(-1)
    return jnp.mean((values - returns) ** 2)


# ---------------------------------------------------------------------------
# Rollout collection (jit-compiled)
# ---------------------------------------------------------------------------

class Trajectory(eqx.Module):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray


def collect_rollout(key, model, params, n_envs, n_steps):
    """Collect (n_steps, n_envs, ...) trajectory with auto-reset."""
    k_reset, k_roll = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, n_envs)
    states, obs_init = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)

    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]

        logits, values = jax.vmap(model)(obs)
        act_keys = jax.random.split(k_act, n_envs)
        actions = jax.vmap(jax.random.categorical, in_axes=(0, 0))(
            act_keys, logits)
        log_probs_all = jax.nn.log_softmax(logits)
        log_probs = jnp.take_along_axis(
            log_probs_all, actions[:, None], axis=1).squeeze(1)

        env_keys = jax.random.split(k_env, n_envs)
        new_states, new_obs, rewards, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(
            env_keys, states, actions, params)

        # Auto-reset done envs
        rst_keys = jax.random.split(k_rst, n_envs)
        def maybe_reset(done, new_s, new_o, rk):
            rs, ro = env_reset(rk, params)
            return (
                jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, new_s),
                jnp.where(done, ro, new_o),
            )
        final_states, final_obs = jax.vmap(
            maybe_reset, in_axes=(0, 0, 0, 0))(
            dones, new_states, new_obs, rst_keys)

        return (final_states, final_obs), Trajectory(
            obs=obs, actions=actions, rewards=rewards, dones=dones,
            log_probs=log_probs, values=values)

    (final_states, final_obs), trajectory = jax.lax.scan(
        scan_step, (states, obs_init), all_keys)
    return trajectory, final_obs


# ---------------------------------------------------------------------------
# Evaluation: full-episode rollouts
# ---------------------------------------------------------------------------

def evaluate_returns(key, model, params, n_episodes):
    """Run full episodes with argmax policy. Returns per-episode returns."""
    def policy(k, obs):
        logits, _ = model(obs)
        return jnp.argmax(logits)

    keys = jax.random.split(key, n_episodes)
    traj = jax.vmap(rollout_episode, in_axes=(0, None, None))(
        keys, policy, params)
    # traj["rewards"]: (n_episodes, T)
    episode_returns = jnp.sum(traj["rewards"], axis=1)
    return episode_returns


def compute_baseline_returns(key, params, n_episodes):
    """Compute random (action=0) baseline episode returns."""
    def random_policy(k, obs):
        return jnp.int32(0)

    keys = jax.random.split(key, n_episodes)
    traj = jax.vmap(rollout_episode, in_axes=(0, None, None))(
        keys, random_policy, params)
    return jnp.sum(traj["rewards"], axis=1)


def compute_oracle_a_returns(key, params, oracle_a_policy, n_episodes):
    """Simulate Oracle A using full-episode rollouts.

    Oracle A sees true regime, but rollout_episode's policy_fn only gets obs.
    We use a custom rollout that passes regime via env state.
    """
    # Oracle A needs regime — use numpy simulation for simplicity
    from lob_sim.analytical_mdp import (
        build_mdp_tables, solve_all_locked, solve_oracle_a, compute_q_max,
        simulate_oracle_a,
    )
    tables = build_mdp_tables(params)
    oracle_a = solve_oracle_a(tables, params)
    sim = simulate_oracle_a(oracle_a, params, n_episodes=n_episodes,
                             seed=int(key[0]) % 2**31)
    # Sum rewards per episode
    return np.sum(sim.rewards, axis=1)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single(
    seed: int,
    regime: int,
    seed_label: int = 0,
    n_iters: int = 200,
    n_envs: int = 128,
    n_steps: int = 64,
    n_epochs: int = 3,
    minibatch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lam: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    eval_every: int = 10,
    n_eval_episodes: int = 64,
    verbose: bool = True,
):
    """Train PPO MLP on a single regime with a single seed.

    Returns dict with training metrics and eval returns at checkpoints.
    """
    regime_name = REGIME_NAMES[regime]
    key = jax.random.PRNGKey(seed)
    env_params = EnvParams.default()._replace(locked_regime=regime)

    k_model, key = jax.random.split(key)
    model = ActorCritic(
        obs_size=OBS_SIZE, n_actions=N_ACTIONS, hidden=64, key=k_model)

    actor_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    critic_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    actor_opt_state = actor_opt.init(eqx.filter(model.actor, eqx.is_array))
    critic_opt_state = critic_opt.init(eqx.filter(model.critic, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, actor_opt_state, critic_opt_state, batch):
        a_loss, a_grads = eqx.filter_value_and_grad(actor_loss_fn)(
            model.actor, model.critic, batch)
        a_updates, new_a_opt = actor_opt.update(
            a_grads, actor_opt_state, eqx.filter(model.actor, eqx.is_array))
        new_actor = eqx.apply_updates(model.actor, a_updates)

        c_loss, c_grads = eqx.filter_value_and_grad(critic_loss_fn)(
            model.critic, batch)
        c_updates, new_c_opt = critic_opt.update(
            c_grads, critic_opt_state, eqx.filter(model.critic, eqx.is_array))
        new_critic = eqx.apply_updates(model.critic, c_updates)

        new_model = eqx.tree_at(lambda m: m.actor, model, new_actor)
        new_model = eqx.tree_at(lambda m: m.critic, new_model, new_critic)
        return new_model, new_a_opt, new_c_opt, a_loss, c_loss

    checkpoints = []  # (iter, mean_episode_return)
    t0 = time.time()
    converge_window = max(50 // eval_every, 3)  # checkpoints to check
    converge_rtol = 0.02  # stop if std/|mean| < this over window
    converged = False

    for it in range(n_iters):
        k_collect, k_eval, key = jax.random.split(key, 3)

        traj, last_obs = collect_rollout(
            k_collect, model, env_params, n_envs, n_steps)

        _, bootstrap_values = jax.vmap(model)(last_obs)
        advantages, returns = batch_gae(
            traj.rewards, traj.values, traj.dones,
            bootstrap_values, gamma, gae_lam)

        flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        total = flat.obs.shape[0]

        k_shuffle, key = jax.random.split(key)
        for _ in range(n_epochs):
            k_perm, k_shuffle = jax.random.split(k_shuffle)
            perm = jax.random.permutation(k_perm, total)
            for start in range(0, total, minibatch_size):
                idx = perm[start:start + minibatch_size]
                mb_adv = adv_flat[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb = (flat.obs[idx], flat.actions[idx], flat.log_probs[idx],
                      mb_adv, ret_flat[idx])
                model, actor_opt_state, critic_opt_state, a_loss, c_loss = (
                    update_step(model, actor_opt_state, critic_opt_state, mb))

        if it % eval_every == 0 or it == n_iters - 1:
            ep_returns = evaluate_returns(k_eval, model, env_params,
                                          n_eval_episodes)
            mean_ret = float(jnp.mean(ep_returns))
            std_ret = float(jnp.std(ep_returns))
            checkpoints.append((it, mean_ret, std_ret))
            if verbose:
                elapsed = time.time() - t0
                rps = mean_ret / env_params.t_episode
                print(f"    seed {seed_label} iter {it:4d} | "
                      f"reward/step {rps:.4f} | {elapsed:.0f}s")

            # Early stopping: check if reward/step is stable
            if len(checkpoints) >= converge_window:
                recent = [c[1] for c in checkpoints[-converge_window:]]
                mu = abs(np.mean(recent))
                if mu > 1e-6 and np.std(recent) / mu < converge_rtol:
                    if verbose:
                        print(f"    seed {seed_label} converged at iter {it} "
                              f"(last {converge_window} evals stable)")
                    converged = True
                    break

    return {
        "seed": seed,
        "regime": regime_name,
        "locked_regime": regime,
        "iters": [c[0] for c in checkpoints],
        "mean_returns": [c[1] for c in checkpoints],
        "std_returns": [c[2] for c in checkpoints],
        "n_iters": it + 1,
        "converged": converged,
        "train_time": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def compute_oracle_b_episode_returns(params, n_episodes):
    """Compute Oracle B per-episode returns for mixed regime."""
    from lob_sim.analytical_mdp import (
        build_mdp_tables, solve_oracle_b, simulate_oracle_b,
    )
    tables = build_mdp_tables(params)
    oracle_b = solve_oracle_b(tables, params, grid_size=20)
    sim = simulate_oracle_b(oracle_b, tables, params,
                            grid_size=20, n_episodes=n_episodes, seed=42)
    return np.sum(sim.rewards, axis=1)


def compute_baselines(params, regime, n_episodes=256):
    """Compute random and Oracle A returns for a given regime setting."""
    env_params = params._replace(locked_regime=regime)
    regime_name = REGIME_NAMES[regime]

    key = jax.random.PRNGKey(9999)
    k_rand, k_oracle = jax.random.split(key)

    # Random baseline
    rand_returns = compute_baseline_returns(k_rand, env_params, n_episodes)
    random_mean = float(jnp.mean(rand_returns))

    # Oracle A
    oracle_a_returns = compute_oracle_a_returns(
        k_oracle, env_params, None, n_episodes)
    oracle_a_mean = float(np.mean(oracle_a_returns))

    result = {
        "regime": regime_name,
        "random_mean_return": random_mean,
        "oracle_a_mean_return": oracle_a_mean,
        "oracle_a_std_return": float(np.std(oracle_a_returns)),
    }

    # Oracle B for mixed regime
    if regime == -1:
        oracle_b_returns = compute_oracle_b_episode_returns(params, n_episodes)
        result["oracle_b_mean_return"] = float(np.mean(oracle_b_returns))
        result["oracle_b_std_return"] = float(np.std(oracle_b_returns))

    return result


# ---------------------------------------------------------------------------
# Figure 3 — PPO Validation
# ---------------------------------------------------------------------------

def plot_figure3(all_results, baselines, t_episode=200):
    """Figure 3: Learning curves for PPO locked + mixed.

    Locked panels: normalised return (Oracle A = 1.0, random = 0).
    Mixed panel: raw per-step reward with Oracle B / Oracle A reference lines.
    Shading: 25th-75th percentile across seeds.
    """
    import matplotlib.pyplot as plt

    regime_order = ["noise", "bull", "bear", "mixed"]
    regime_colors = {
        "noise": "#3b82f6", "bull": "#22c55e",
        "bear": "#ef4444", "mixed": "#8b5cf6",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for idx, regime_name in enumerate(regime_order):
        ax = axes[idx]
        runs = [r for r in all_results if r["regime"] == regime_name]
        if not runs:
            ax.set_title(f"{regime_name.capitalize()} (no data)")
            continue

        bl = baselines.get(regime_name, {})
        iters = np.array(runs[0]["iters"])
        color = regime_colors[regime_name]

        if regime_name == "mixed":
            # Episode return with ±std bands across eval episodes
            # PPO: mean ± std at each checkpoint, averaged across seeds
            means_all = np.array([r["mean_returns"] for r in runs])
            stds_all = np.array([r["std_returns"] for r in runs])
            # Average across seeds
            ppo_mean = np.mean(means_all, axis=0)
            # Pool std: sqrt(mean of variances + variance of means)
            ppo_std = np.sqrt(np.mean(stds_all**2, axis=0)
                              + np.var(means_all, axis=0))

            ax.plot(iters, ppo_mean, color=color, linewidth=2,
                    label=f"PPO MLP ({len(runs)} seeds)")
            ax.fill_between(iters, ppo_mean - ppo_std, ppo_mean + ppo_std,
                            color=color, alpha=0.15)

            # Oracle B band
            ob_mean = bl.get("oracle_b_mean_return")
            ob_std = bl.get("oracle_b_std_return")
            if ob_mean is not None:
                ax.axhline(ob_mean, color="black", linestyle="--",
                           linewidth=1.5, alpha=0.8, label="Oracle B")
                if ob_std is not None:
                    ax.axhspan(ob_mean - ob_std, ob_mean + ob_std,
                               color="black", alpha=0.08)

            # Oracle A band
            oa_mean = bl.get("oracle_a_mean_return", 0)
            oa_std = bl.get("oracle_a_std_return", 0)
            ax.axhline(oa_mean, color="gray", linestyle=":",
                       linewidth=1.5, alpha=0.8, label="Oracle A")
            if oa_std:
                ax.axhspan(oa_mean - oa_std, oa_mean + oa_std,
                           color="gray", alpha=0.08)

            # Zoom y-axis to converged range
            y_top = oa_mean + oa_std * 2
            y_bot = min(0, ppo_mean[-1] - ppo_std[-1] * 2)
            ax.set_ylim(y_bot, y_top)

            ax.set_ylabel("Episode return")
            ax.set_xlabel("Training iteration")
            ax.set_title("Mixed", fontsize=12)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.2)
        else:
            # Locked panels: episode return with ±std bands
            means_all = np.array([r["mean_returns"] for r in runs])
            stds_all = np.array([r["std_returns"] for r in runs])
            ppo_mean = np.mean(means_all, axis=0)
            ppo_std = np.sqrt(np.mean(stds_all**2, axis=0)
                              + np.var(means_all, axis=0))

            ax.plot(iters, ppo_mean, color=color, linewidth=2,
                    label=f"PPO MLP ({len(runs)} seeds)")
            ax.fill_between(iters, ppo_mean - ppo_std, ppo_mean + ppo_std,
                            color=color, alpha=0.15)

            # Oracle A band
            oa_mean = bl.get("oracle_a_mean_return", 0)
            oa_std = bl.get("oracle_a_std_return", 0)
            ax.axhline(oa_mean, color="black", linestyle="--",
                       linewidth=1.5, alpha=0.8, label="Oracle A")
            if oa_std:
                ax.axhspan(oa_mean - oa_std, oa_mean + oa_std,
                           color="black", alpha=0.08)

            # Zoom to converged range
            y_top = oa_mean + oa_std * 2
            y_bot = min(0, ppo_mean[-1] - ppo_std[-1] * 2)
            ax.set_ylim(y_bot, y_top)

            if idx == 0:
                ax.set_ylabel("Episode return")
            ax.set_xlabel("Training iteration")
            ax.set_title(f"{regime_name.capitalize()}", fontsize=12)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.2)

    fig.suptitle("Figure 3 — PPO MLP Validation", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "figure3_ppo_validation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke test (2 seeds, few iters)")
    parser.add_argument("--regime", type=str, default="all",
                        choices=["noise", "bull", "bear", "mixed", "all"])
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Seeds per regime (default: 8, fast: 2)")
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--master-seed", type=int, default=0,
                        help="Master seed for deriving per-run seeds")
    args = parser.parse_args()

    # Defaults based on --fast
    if args.fast:
        n_seeds = args.n_seeds or 2
        n_iters = args.n_iters or 60
        n_eval = 16
        eval_every = 5
    else:
        n_seeds = args.n_seeds or 3
        n_iters = args.n_iters or 100
        n_eval = 64
        eval_every = 5

    if args.regime == "all":
        regimes = [("noise", 0), ("bull", 1), ("bear", 2), ("mixed", -1)]
    else:
        regimes = [(args.regime, REGIME_MAP[args.regime])]

    print("=" * 60)
    print("  Phase 4 — PPO MLP Baseline")
    print("=" * 60)

    params = EnvParams.default()

    # --- Compute baselines ---
    print("\n  Computing baselines ...")
    baselines = {}
    n_bl_episodes = 64 if args.fast else 256
    for name, rid in regimes:
        t0 = time.time()
        bl = compute_baselines(params, rid, n_episodes=n_bl_episodes)
        baselines[name] = bl
        ob_str = ""
        if "oracle_b_mean_return" in bl:
            ob_str = f"  oracle_b={bl['oracle_b_mean_return']:.1f}"
        print(f"    {name:>6s}: random={bl['random_mean_return']:.1f}  "
              f"oracle_a={bl['oracle_a_mean_return']:.1f}{ob_str}  "
              f"({time.time()-t0:.1f}s)")

    # --- Train ---
    all_results = []
    t_ep = int(params.t_episode)
    master_rng = np.random.default_rng(args.master_seed)
    for name, rid in regimes:
        bl = baselines[name]
        oa_rps = bl["oracle_a_mean_return"] / t_ep
        ref = f"oracle_a={oa_rps:.4f}"
        if "oracle_b_mean_return" in bl:
            ob_rps = bl["oracle_b_mean_return"] / t_ep
            ref += f"  oracle_b={ob_rps:.4f}"
        print(f"\n  Training {name} ({n_seeds} seeds x {n_iters} iters) "
              f"[target reward/step: {ref}]")
        run_seeds = master_rng.integers(0, 2**31, size=n_seeds)
        for seed_idx, seed in enumerate(run_seeds):
            result = train_single(
                seed=int(seed), regime=rid, seed_label=seed_idx,
                n_iters=n_iters,
                n_envs=args.n_envs, lr=args.lr,
                eval_every=eval_every, n_eval_episodes=n_eval,
                verbose=True,
            )
            all_results.append(result)
            final = result["mean_returns"][-1]
            print(f"    seed {seed_idx} done: reward/step={final/t_ep:.4f}  "
                  f"({result['train_time']:.0f}s)")

    # --- Gate check ---
    print("\n" + "=" * 60)
    gate_pass = True
    for name, rid in regimes:
        if rid == -1:
            continue  # mixed doesn't need to reach 1.0
        runs = [r for r in all_results if r["regime"] == name]
        bl = baselines[name]
        denom = bl["oracle_a_mean_return"] - bl["random_mean_return"]
        finals = []
        for r in runs:
            final = r["mean_returns"][-1]
            norm = (final - bl["random_mean_return"]) / denom if denom > 0 else 0
            finals.append(norm)
        mean_norm = np.mean(finals)
        passed = mean_norm > 0.8
        gate_pass = gate_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name:>6s}: mean normalised = {mean_norm:.3f}  [{status}]")

    print(f"\n  GATE: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("  WARNING: locked curves did not reach ~1.0!")
    print("=" * 60)

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # JSON-safe encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.bool_)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    save_data = {
        "master_seed": args.master_seed,
        "baselines": baselines,
        "runs": all_results,
        "gate_passed": bool(gate_pass),
    }
    path = os.path.join(RESULTS_DIR, "ppo_phase4.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n  -> {os.path.relpath(path, ROOT)}")

    # --- Plot Figure 3 ---
    print("  Plotting Figure 3 ...")
    plot_figure3(all_results, baselines)

    print("\n  Done.")


if __name__ == "__main__":
    main()
