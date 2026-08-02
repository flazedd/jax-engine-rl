"""Shared PPO update step.

Used by every PPO-based agent (vanilla PPO, Oracle-PPO, Belief-PPO, and RL² —
whose loss is ordinary PPO loss with recurrent architecture). VariBAD has a
different joint PPO+ELBO update, so it owns its own file.

Functions in this module are pure and JIT-compatible: they take params and
optimizer state, return updated params / opt_state / metrics. The agent wires
them together behind the Agent.update interface.
"""
from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp
import optax


def compute_gae(
    rewards: chex.Array,      # [T, N]
    values: chex.Array,       # [T, N]
    dones: chex.Array,        # [T, N]  (bool)
    last_value: chex.Array,   # [N]
    gamma: float,
    lam: float,
) -> tuple[chex.Array, chex.Array]:
    """Generalized advantage estimation via lax.scan (backward pass).

    Returns (advantages[T,N], returns[T,N]) where returns = advantages + values.
    Episode boundaries mask bootstrap through the done flag.
    """
    not_done = (~dones).astype(rewards.dtype)

    def step(carry, t_inputs):
        next_value, next_adv = carry
        reward_t, value_t, not_done_t = t_inputs
        delta = reward_t + gamma * next_value * not_done_t - value_t
        adv_t = delta + gamma * lam * next_adv * not_done_t
        # Carry next_value = value_t (for the prior timestep, reading backward).
        return (value_t, adv_t), adv_t

    xs = (rewards, values, not_done)
    init_carry = (last_value, jnp.zeros_like(last_value))
    _, advantages = jax.lax.scan(step, init_carry, xs, reverse=True)
    returns = advantages + values
    return advantages, returns


def _policy_value_forward(apply_fn, params, obs):
    """apply_fn(params, obs) → (logits, value) for a batch of obs."""
    return apply_fn(params, obs)


def _ppo_loss(
    params,
    apply_fn,
    batch: dict[str, chex.Array],
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    logits, values = apply_fn(params, batch["obs"])
    log_probs_all = jax.nn.log_softmax(logits)
    lp_new = jnp.take_along_axis(log_probs_all, batch["action"][:, None], axis=-1).squeeze(-1)

    ratio = jnp.exp(lp_new - batch["log_prob"])
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -jnp.minimum(unclipped, clipped).mean()

    value_loss = 0.5 * jnp.mean((values - batch["return"]) ** 2)

    probs = jnp.exp(log_probs_all)
    entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    approx_kl = (batch["log_prob"] - lp_new).mean()
    clipped_frac = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32).mean()

    metrics = {
        "ppo/policy_loss": policy_loss,
        "ppo/value_loss": value_loss,
        "ppo/entropy": entropy,
        "ppo/approx_kl": approx_kl,
        "ppo/clipped_frac": clipped_frac,
        "ppo/total_loss": loss,
    }
    return loss, metrics


def make_ppo_step(apply_fn, optimizer: optax.GradientTransformation, *, clip_eps, ent_coef, vf_coef):
    """Return a JIT-compatible function that runs one gradient step on a minibatch."""

    grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True, argnums=0)

    def step(params, opt_state, batch):
        (loss, metrics), grads = grad_fn(params, apply_fn, batch, clip_eps, ent_coef, vf_coef)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["ppo/total_loss"] = loss
        return params, opt_state, metrics

    return step


def run_ppo_epochs(
    apply_fn,
    optimizer,
    params,
    opt_state,
    batch: dict[str, chex.Array],  # each leaf shape [N_samples, ...]
    *,
    key: chex.PRNGKey,
    epochs: int,
    minibatch_size: int,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
    shuffle_block: int = 0,
) -> tuple[dict, dict, dict[str, chex.Array]]:
    """Run `epochs` passes of PPO updates, each with fresh minibatch shuffling.

    Produces mean metrics across all minibatches. Uses fixed-shape minibatches
    (N_samples must be divisible by minibatch_size) for JIT stability.
    """
    step = make_ppo_step(apply_fn, optimizer, clip_eps=clip_eps, ent_coef=ent_coef, vf_coef=vf_coef)

    n_samples = batch["obs"].shape[0]
    n_minibatches = n_samples // minibatch_size

    def epoch_body(carry, epoch_key):
        params, opt_state = carry
        if shuffle_block > 0:
            # Permute whole blocks so rows that belong together, one
            # environment's trajectory, stay together across epochs.
            n_blocks = n_samples // shuffle_block
            block_perm = jax.random.permutation(epoch_key, n_blocks)
            perm = (
                block_perm[:, None] * shuffle_block
                + jnp.arange(shuffle_block)[None, :]
            ).reshape(-1)
        else:
            perm = jax.random.permutation(epoch_key, n_samples)
        # Reshape so each row is a minibatch.
        shuffled = jax.tree_util.tree_map(lambda x: x[perm].reshape((n_minibatches, minibatch_size) + x.shape[1:]), batch)

        def minibatch_body(carry, mb_idx):
            params, opt_state = carry
            mb = jax.tree_util.tree_map(lambda x: x[mb_idx], shuffled)
            params, opt_state, metrics = step(params, opt_state, mb)
            return (params, opt_state), metrics

        (params, opt_state), mb_metrics = jax.lax.scan(
            minibatch_body, (params, opt_state), jnp.arange(n_minibatches)
        )
        return (params, opt_state), mb_metrics

    epoch_keys = jax.random.split(key, epochs)
    (params, opt_state), all_metrics = jax.lax.scan(
        epoch_body, (params, opt_state), epoch_keys
    )
    # Mean across all (epochs, minibatches) steps.
    mean_metrics = jax.tree_util.tree_map(lambda x: x.mean(), all_metrics)
    return params, opt_state, mean_metrics
