"""PPO feedforward training on the JAX market making environment.

Uses Equinox for the actor-critic network, Optax for optimization,
and the JAX-native env for fully jit-compiled rollouts + updates.

Usage:
    uv run python scripts/train_ppo.py
    uv run python scripts/train_ppo.py --fast          # quick smoke test
    uv run python scripts/train_ppo.py --n-iters 500   # longer run
"""
import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from lob_sim.jax_env import EnvParams, env_reset, env_step, rollout_episode

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    """Simple MLP."""
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
    """Actor-critic with separate networks."""
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
# GAE (per-environment, respects episode boundaries)
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, bootstrap_value, gamma=0.99, lam=0.95):
    """GAE over a single env trajectory.

    rewards/values/dones: (T,), bootstrap_value: scalar V(s_T) for the state
    after the last collected step.
    """
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


# Vectorize GAE across envs: (T, n_envs) -> (T, n_envs)
batch_gae = jax.vmap(
    compute_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=(1, 1))


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def actor_loss_fn(actor, critic, batch, clip_eps=0.2, ent_coef=0.01):
    """PPO clipped actor loss + entropy bonus."""
    obs, actions, old_log_probs, advantages, _ = batch
    logits = jax.vmap(actor)(obs)

    log_probs_all = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))

    return policy_loss - ent_coef * entropy


def critic_loss_fn(critic, batch):
    """Value function MSE loss."""
    obs, _, _, _, returns = batch
    values = jax.vmap(critic)(obs).squeeze(-1)
    return jnp.mean((values - returns) ** 2)


# ---------------------------------------------------------------------------
# Rollout collection (fully jit-compiled)
# ---------------------------------------------------------------------------

class Trajectory(NamedTuple):
    """Rollout data with shape (T, n_envs, ...)."""
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray


def collect_rollout(key, model, params, n_envs, n_steps):
    """Collect n_steps from n_envs parallel envs. Returns Trajectory (T, n_envs, ...)."""
    k_reset, k_roll = jax.random.split(key)

    reset_keys = jax.random.split(k_reset, n_envs)
    states, obs_init = jax.vmap(env_reset, in_axes=(0, None))(reset_keys, params)

    # Pre-split keys: (n_steps, 3) for action/env/reset
    all_keys = jax.random.split(k_roll, n_steps * 3).reshape(n_steps, 3, -1)

    def scan_step(carry, keys_t):
        states, obs = carry
        k_act, k_env, k_rst = keys_t[0], keys_t[1], keys_t[2]

        logits, values = jax.vmap(model)(obs)

        act_keys = jax.random.split(k_act, n_envs)
        actions = jax.vmap(jax.random.categorical, in_axes=(0, 0))(act_keys, logits)

        log_probs_all = jax.nn.log_softmax(logits)
        log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1).squeeze(1)

        env_keys = jax.random.split(k_env, n_envs)
        new_states, new_obs, rewards, dones, _ = jax.vmap(env_step, in_axes=(0, 0, 0, None))(
            env_keys, states, actions, params)

        # Auto-reset done envs
        rst_keys = jax.random.split(k_rst, n_envs)
        def maybe_reset(done, new_s, new_o, rk):
            rs, ro = env_reset(rk, params)
            return (
                jax.tree.map(lambda a, b: jnp.where(done, a, b), rs, new_s),
                jnp.where(done, ro, new_o),
            )
        final_states, final_obs = jax.vmap(maybe_reset, in_axes=(0, 0, 0, 0))(
            dones, new_states, new_obs, rst_keys)

        return (final_states, final_obs), Trajectory(
            obs=obs, actions=actions, rewards=rewards, dones=dones,
            log_probs=log_probs, values=values)

    (final_states, final_obs), trajectory = jax.lax.scan(
        scan_step, (states, obs_init), all_keys)
    return trajectory, final_obs  # trajectory: (n_steps, n_envs, ...)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

REGIME_NAMES = {-1: "mixed", 0: "noise", 1: "bull", 2: "bear"}


def train(
    seed: int = 0,
    regime: int = -1,
    n_iters: int = 300,
    n_envs: int = 256,
    n_steps: int = 64,
    n_epochs: int = 3,
    minibatch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.95,
    gae_lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    eval_every: int = 10,
    n_eval_episodes: int = 32,
):
    regime_name = REGIME_NAMES.get(regime, f"regime_{regime}")
    print(f"Training on regime: {regime_name}")

    key = jax.random.PRNGKey(seed)
    env_params = EnvParams.default()._replace(locked_regime=regime)

    k_model, key = jax.random.split(key)
    model = ActorCritic(
        obs_size=3, n_actions=env_params.n_actions, hidden=64, key=k_model)

    # Separate optimizers — prevents critic gradient from destabilizing actor
    actor_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    critic_opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))
    actor_opt_state = actor_opt.init(eqx.filter(model.actor, eqx.is_array))
    critic_opt_state = critic_opt.init(eqx.filter(model.critic, eqx.is_array))

    @eqx.filter_jit
    def update_step(model, actor_opt_state, critic_opt_state, batch):
        # Actor update
        a_loss, a_grads = eqx.filter_value_and_grad(actor_loss_fn)(
            model.actor, model.critic, batch)
        a_updates, new_a_opt = actor_opt.update(
            a_grads, actor_opt_state, eqx.filter(model.actor, eqx.is_array))
        new_actor = eqx.apply_updates(model.actor, a_updates)

        # Critic update
        c_loss, c_grads = eqx.filter_value_and_grad(critic_loss_fn)(
            model.critic, batch)
        c_updates, new_c_opt = critic_opt.update(
            c_grads, critic_opt_state, eqx.filter(model.critic, eqx.is_array))
        new_critic = eqx.apply_updates(model.critic, c_updates)

        new_model = eqx.tree_at(lambda m: m.actor, model, new_actor)
        new_model = eqx.tree_at(lambda m: m.critic, new_model, new_critic)
        return new_model, new_a_opt, new_c_opt, a_loss, c_loss

    # Regime-blind baseline (always 200-step episodes for eval)
    eval_ep_len = env_params.episode_length
    def blind_policy(k, obs):
        return jnp.int32(0)
    blind_keys = jax.random.split(jax.random.PRNGKey(999), 100)
    blind_traj = jax.vmap(rollout_episode, in_axes=(0, None, None, None))(
        blind_keys, blind_policy, env_params, eval_ep_len)
    blind_baseline = float(jnp.mean(blind_traj["rewards"]))
    print(f"Regime-blind baseline: {blind_baseline:.3f}/step")

    # Eval: deterministic (argmax) policy, full 200-step episodes
    def evaluate(key, model):
        def policy(k, obs):
            logits, _ = model(obs)
            return jnp.argmax(logits)
        keys = jax.random.split(key, n_eval_episodes)
        traj = jax.vmap(rollout_episode, in_axes=(0, None, None, None))(
            keys, policy, env_params, eval_ep_len)
        return float(jnp.mean(traj["rewards"]))

    # Load analytical baselines from bellman solution
    baselines_path = RESULTS_DIR / "bellman_solution.json"
    if baselines_path.exists():
        with open(baselines_path) as f:
            bellman = json.load(f)
        analytical = bellman["simulation"]["per_step_reward"]
    else:
        analytical = {"full_info": None, "pomdp": None, "regime_blind": None}

    metrics = {
        "regime": regime_name,
        "locked_regime": regime,
        "blind_baseline": blind_baseline,
        "analytical_baselines": analytical,
        "eval_rewards": [], "train_losses": [], "iters": [],
    }

    t0 = time.time()
    for it in range(n_iters):
        k_collect, k_eval, key = jax.random.split(key, 3)

        # Collect (n_steps, n_envs, ...) trajectory
        traj, last_obs = collect_rollout(k_collect, model, env_params, n_envs, n_steps)

        # Bootstrap value: V(s_T) for each env at end of rollout
        _, bootstrap_values = jax.vmap(model)(last_obs)  # (n_envs,)

        # GAE per-env with proper bootstrap, then flatten
        advantages, returns = batch_gae(
            traj.rewards, traj.values, traj.dones,
            bootstrap_values, gamma, gae_lam)

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        total = flat.obs.shape[0]

        # PPO update epochs
        k_shuffle, key = jax.random.split(key)
        actor_losses, critic_losses = [], []
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
                actor_losses.append(float(a_loss))
                critic_losses.append(float(c_loss))

        mean_a_loss = sum(actor_losses) / len(actor_losses)
        mean_c_loss = sum(critic_losses) / len(critic_losses)

        if it % eval_every == 0 or it == n_iters - 1:
            eval_reward = evaluate(k_eval, model)
            elapsed = time.time() - t0
            gap = eval_reward - blind_baseline
            print(f"iter {it:4d} | a_loss {mean_a_loss:.4f} c_loss {mean_c_loss:.1f} | "
                  f"reward {eval_reward:.3f}/step | "
                  f"gap vs blind {gap:+.3f} | {elapsed:.1f}s")
            metrics["eval_rewards"].append(eval_reward)
            metrics["train_losses"].append(mean_a_loss)
            metrics["iters"].append(it)

    total_time = time.time() - t0
    final = metrics["eval_rewards"][-1]
    print(f"\nDone in {total_time:.1f}s — "
          f"final {final:.3f}/step, blind {blind_baseline:.3f}/step, "
          f"gap {final - blind_baseline:+.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"ppo_metrics_{regime_name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {path.relative_to(ROOT)}")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Quick smoke test")
    parser.add_argument("--regime", type=str, default="noise",
                        choices=["mixed", "noise", "bull", "bear", "all"],
                        help="Regime to train on (default: mixed)")
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    if args.fast:
        args.n_iters = 10
        args.n_envs = 16

    name_to_id = {"mixed": -1, "noise": 0, "bull": 1, "bear": 2}

    if args.regime == "all":
        regimes = [("noise", 0), ("bull", 1), ("bear", 2), ("mixed", -1)]
    else:
        regimes = [(args.regime, name_to_id[args.regime])]

    for name, rid in regimes:
        train(
            seed=args.seed,
            regime=rid,
            n_iters=args.n_iters,
            n_envs=args.n_envs,
            lr=args.lr,
            eval_every=5 if args.fast else 10,
            n_eval_episodes=8 if args.fast else 32,
        )
